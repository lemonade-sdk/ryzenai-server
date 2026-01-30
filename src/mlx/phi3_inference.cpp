/*
 * phi3_inference.cpp
 * 
 * Phi-3 transformer inference implementation.
 * Supports both quantized and full-precision weights.
 */

#include "ryzenai/mlx/phi3_inference.h"
#include "ryzenai/mlx/quantization.h"
#include <iostream>
#include <string>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>

using namespace mlx::core;


Phi3Inference::Phi3Inference(const OgaModel& model) : model_(model) {
    actual_hidden_size_ = model_.hidden_size;
    head_dim_ = actual_hidden_size_ / model_.num_attention_heads;
    
    std::cout << "[Phi3Inference] hidden_size=" << actual_hidden_size_
              << ", head_dim=" << head_dim_
              << ", num_heads=" << model_.num_attention_heads
              << ", num_kv_heads=" << model_.num_key_value_heads
              << ", quantized=" << (model_.is_quantized() ? "yes" : "no")
              << std::endl;
    
    cache_weights();
}


/*
 * linear
 * 
 * Computes a linear projection. Uses quantized_matmul when scale
 * parameters are available, otherwise falls back to standard matmul.
 */
array Phi3Inference::linear(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        throw std::runtime_error("Weight not found: " + weight_name);
    }
    
    auto scales_it = cached_weights_.find(weight_name + ".scales");
    if (scales_it != cached_weights_.end()) {
        auto biases_it = cached_weights_.find(weight_name + ".biases");
        array biases = (biases_it != cached_weights_.end()) 
            ? biases_it->second : zeros_like(scales_it->second);
        
        static bool logged_quantized = true;
        if (logged_quantized) {
            std::cout << "[Phi3Inference] Using quantized_matmul for " << weight_name
                      << " (bits=" << model_.quantization.bits 
                      << ", group_size=" << model_.quantization.group_size << ")"
                      << " x:" << x.shape() << " w:" << weight_it->second.shape()
                      << " scales:" << scales_it->second.shape() << std::endl;
            logged_quantized = false;
        }
        
        return quantized_matmul(
            x, weight_it->second, scales_it->second, biases,
            true,
            model_.quantization.group_size,
            model_.quantization.bits,
            "affine", {}
        );
    }
    
    static bool logged_fp = true;
    if (logged_fp) {
        std::cout << "[Phi3Inference] Using regular matmul for " << weight_name << std::endl;
        logged_fp = false;
    }
    return matmul(x, transpose(weight_it->second, {1, 0}));
}


/*
 * rms_norm
 * 
 * Root mean square layer normalization.
 * Normalizes by RMS and scales by learned weights.
 */
array Phi3Inference::rms_norm(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        return x;
    }
    
    array variance = mean(square(x), -1, true);
    return x * weight_it->second / sqrt(variance + model_.rms_norm_eps);
}


/*
 * forward
 * 
 * Full forward pass through the Phi-3 model.
 * Embeds tokens, processes through transformer layers,
 * applies final normalization, and projects to vocabulary.
 */
array Phi3Inference::forward(const std::vector<int32_t>& input_tokens,
                            const OgaGeneratorParams& params) {
    int seq_len = static_cast<int>(input_tokens.size());

    auto embed_it = cached_weights_.find("embed_tokens.weight");
    if (embed_it == cached_weights_.end()) {
        throw std::runtime_error("embed_tokens.weight not found");
    }
    array h = take(embed_it->second, array(input_tokens.data(), {seq_len}, int32), 0);

    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        std::string prefix = "layers." + std::to_string(i) + ".";
        
        array attn_out = self_attention_no_residual(rms_norm(h, prefix + "input_layernorm"), prefix, seq_len);
        h = h + attn_out;
        
        array mlp_out = mlp_block_no_residual(rms_norm(h, prefix + "post_attention_layernorm"), prefix);
        h = h + mlp_out;
    }

    h = rms_norm(h, "norm");
    
    array last_h = reshape(take(h, array({seq_len - 1}, {1}, int32), 0), {actual_hidden_size_});
    
    array logits = linear(reshape(last_h, {1, actual_hidden_size_}), "lm_head");
    logits = reshape(logits, {model_.vocab_size});
    
    eval(logits);
    return logits;
}


array Phi3Inference::process_layer(const array& h, int layer_idx, int seq_len) {
    return h;
}


/*
 * self_attention_no_residual
 * 
 * Multi-head self attention using combined QKV projection.
 * Applies rotary position embeddings and causal masking.
 */
array Phi3Inference::self_attention_no_residual(const array& x, const std::string& prefix, int seq_len) {
    array qkv = linear(x, prefix + "self_attn.qkv_proj");
    
    int q_size = model_.num_attention_heads * head_dim_;
    int kv_size = model_.num_key_value_heads * head_dim_;
    
    array q = slice(qkv, {0, 0}, {seq_len, q_size});
    array k = slice(qkv, {0, q_size}, {seq_len, q_size + kv_size});
    array v = slice(qkv, {0, q_size + kv_size}, {seq_len, q_size + 2 * kv_size});
    
    q = transpose(reshape(q, {seq_len, model_.num_attention_heads, head_dim_}), {1, 0, 2});
    k = transpose(reshape(k, {seq_len, model_.num_key_value_heads, head_dim_}), {1, 0, 2});
    v = transpose(reshape(v, {seq_len, model_.num_key_value_heads, head_dim_}), {1, 0, 2});
    
    float rope_base = model_.rope_theta > 0 ? model_.rope_theta : 10000.0f;
    q = fast::rope(q, head_dim_, false, rope_base, 1.0f, 0);
    k = fast::rope(k, head_dim_, false, rope_base, 1.0f, 0);
    
    float scale = 1.0f / sqrt(static_cast<float>(head_dim_));
    array scores = matmul(q, transpose(k, {0, 2, 1})) * scale;
    scores = scores + triu(full({seq_len, seq_len}, -INFINITY), 1);
    array attn = softmax(scores, -1);
    array out = matmul(attn, v);
    
    out = reshape(transpose(out, {1, 0, 2}), {seq_len, actual_hidden_size_});
    
    return linear(out, prefix + "self_attn.o_proj");
}


array Phi3Inference::self_attention(const array& x, const std::string& prefix, int seq_len) {
    return x + self_attention_no_residual(x, prefix, seq_len);
}


/*
 * mlp_block_no_residual
 * 
 * SwiGLU MLP block with combined gate/up projection.
 * Computes: down_proj(silu(gate) * up)
 */
array Phi3Inference::mlp_block_no_residual(const array& x, const std::string& prefix) {
    int seq_len = x.shape()[0];
    
    array gate_up = linear(x, prefix + "mlp.gate_up_proj");
    int mid = gate_up.shape().back() / 2;
    
    array gate = slice(gate_up, {0, 0}, {seq_len, mid});
    array up = slice(gate_up, {0, mid}, {seq_len, 2 * mid});
    
    array activated = gate * sigmoid(gate) * up;
    
    return linear(activated, prefix + "mlp.down_proj");
}


array Phi3Inference::mlp_block(const array& x, const std::string& prefix) {
    return x + mlp_block_no_residual(x, prefix);
}


/*
 * sample_token
 * 
 * Greedy decoding with optional temperature scaling.
 */
int Phi3Inference::sample_token(const array& logits, const OgaGeneratorParams& params) {
    array l = params.do_sample && params.temperature > 0.0f 
        ? logits / params.temperature : logits;
    return static_cast<int>(argmax(l).item<int32_t>());
}


/*
 * cache_weights
 * 
 * Pre-loads and organizes model weights for inference.
 * Dequantizes embeddings since take() requires float weights.
 * Keeps linear layer weights in packed form for quantized_matmul.
 */
void Phi3Inference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    const auto& q = model_.quantization;
    
    std::cout << "[Phi3Inference] Caching weights (" 
              << (q.is_quantized() ? std::to_string(q.bits) + "-bit" : "fp32") << ")" << std::endl;

    auto cache_weight = [&](const std::string& name) {
        auto it = w.find(name + ".weight");
        if (it != w.end()) {
            cached_weights_.emplace(name + ".weight", it->second);
            
            auto scales_it = w.find(name + ".scales");
            if (scales_it != w.end()) {
                cached_weights_.emplace(name + ".scales", scales_it->second);
            }
            auto biases_it = w.find(name + ".biases");
            if (biases_it != w.end()) {
                cached_weights_.emplace(name + ".biases", biases_it->second);
            }
        }
    };

    auto embed_it = w.find("embed_tokens.weight");
    if (embed_it != w.end()) {
        cached_weights_.emplace("embed_tokens.weight", 
            apply_quantization(embed_it->second, "embed_tokens", w, q));
    }

    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        std::string p = "layers." + std::to_string(i) + ".";
        
        auto in_ln = w.find(p + "input_layernorm.weight");
        if (in_ln != w.end()) {
            cached_weights_.emplace(p + "input_layernorm.weight", in_ln->second);
        }
        
        auto post_ln = w.find(p + "post_attention_layernorm.weight");
        if (post_ln != w.end()) {
            cached_weights_.emplace(p + "post_attention_layernorm.weight", post_ln->second);
        }
        
        cache_weight(p + "self_attn.qkv_proj");
        cache_weight(p + "self_attn.o_proj");
        cache_weight(p + "mlp.gate_up_proj");
        cache_weight(p + "mlp.down_proj");
    }

    auto norm_it = w.find("norm.weight");
    if (norm_it != w.end()) {
        cached_weights_.emplace("norm.weight", norm_it->second);
    }

    cache_weight("lm_head");

    std::cout << "[Phi3Inference] Cached " << cached_weights_.size() << " tensors" << std::endl;
}