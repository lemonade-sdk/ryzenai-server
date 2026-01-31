/*
 * llama_inference.cpp
 * 
 * Llama transformer inference implementation.
 * Base for Llama-style architectures with RoPE and SwiGLU.
 */

#include "ryzenai/mlx/models/llama_inference.h"
#include "ryzenai/mlx/quantization.h"
#include <iostream>
#include <fstream>
#include <string>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>

using namespace mlx::core;


LlamaInference::LlamaInference(const MlxOgaModel& model) : model_(model) {
    actual_hidden_size_ = model_.hidden_size;

    // 1. Priority: Explicit config value
    if (model_.head_dim > 0) {
        head_dim_ = model_.head_dim;
    }
    // 2. Fallback: Derive from weight matrix shape (the most "automatic" way)
    else {
        auto q_proj_it = model_.weights.find("layers.0.self_attn.q_proj.weight");
        if (q_proj_it != model_.weights.end()) {
            // Q weight shape is usually [Total_Query_Dim, Hidden_Size]
            int q_output_size = static_cast<int>(q_proj_it->second.shape(0));
            head_dim_ = q_output_size / model_.num_attention_heads;
        }
        // 3. Last Resort: Standard math
        else {
            head_dim_ = actual_hidden_size_ / model_.num_attention_heads;
        }
    }

    tie_word_embeddings_ = model_.tie_word_embeddings;
    attention_bias_ = model_.attention_bias;
    mlp_bias_ = model_.mlp_bias;
    
    std::cout << "[LlamaInference] hidden_size=" << actual_hidden_size_
              << ", head_dim=" << head_dim_
              << ", num_heads=" << model_.num_attention_heads
              << ", num_kv_heads=" << model_.num_key_value_heads
              << ", tie_word_embeddings=" << (tie_word_embeddings_ ? "yes" : "no")
              << ", quantized=" << (model_.is_quantized() ? "yes" : "no")
              << std::endl;
    
    cache_weights();
}


array LlamaInference::linear(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        throw std::runtime_error("Weight not found: " + weight_name);
    }
    
    auto scales_it = cached_weights_.find(weight_name + ".scales");
    if (scales_it != cached_weights_.end()) {
        auto biases_it = cached_weights_.find(weight_name + ".biases");
        array biases = (biases_it != cached_weights_.end()) 
            ? biases_it->second : zeros_like(scales_it->second);
        
        return quantized_matmul(
            x, weight_it->second, scales_it->second, biases,
            true,
            model_.quantization.group_size,
            model_.quantization.bits,
            "affine", {}
        );
    }
    
    return matmul(x, transpose(weight_it->second, {1, 0}));
}


array LlamaInference::linear_with_bias(const array& x, const std::string& weight_name) {
    array out = linear(x, weight_name);
    
    auto bias_it = cached_weights_.find(weight_name + ".bias");
    if (bias_it != cached_weights_.end()) {
        out = out + bias_it->second;
    }
    
    return out;
}


array LlamaInference::rms_norm(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        return x;
    }
    
    array variance = mean(square(x), -1, true);
    return x * weight_it->second / sqrt(variance + model_.rms_norm_eps);
}


array LlamaInference::forward(const std::vector<int32_t>& input_tokens,
                              const MlxOgaGeneratorParams& params) {
    int seq_len = static_cast<int>(input_tokens.size());

    auto embed_it = cached_weights_.find("embed_tokens.weight");
    if (embed_it == cached_weights_.end()) {
        throw std::runtime_error("embed_tokens.weight not found");
    }
    array h = take(embed_it->second, array(input_tokens.data(), {seq_len}, int32), 0);

    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        h = process_layer(h, i, seq_len);
    }

    h = rms_norm(h, "norm");
    
    array last_h = reshape(take(h, array({seq_len - 1}, {1}, int32), 0), {actual_hidden_size_});

    array logits = [&]() {
        if (tie_word_embeddings_) {
            return matmul(reshape(last_h, {1, actual_hidden_size_}), 
                        transpose(embed_it->second, {1, 0}));
        } else {
            return linear(reshape(last_h, {1, actual_hidden_size_}), "lm_head");
        }
    }();

    int vocab = model_.vocab_size;
    logits = reshape(logits, {vocab});
    
    eval(logits);
    return logits;
}


array LlamaInference::process_layer(const array& hidden_states, int layer_idx, int seq_len) {
    std::string prefix = "layers." + std::to_string(layer_idx) + ".";
    
    // Self attention with residual
    array normed = rms_norm(hidden_states, prefix + "input_layernorm");
    array attn_out = self_attention(normed, prefix, seq_len);
    array h = hidden_states + attn_out;
    
    // MLP with residual
    array normed2 = rms_norm(h, prefix + "post_attention_layernorm");
    array mlp_out = mlp_block(normed2, prefix);
    
    return h + mlp_out;
}


array LlamaInference::self_attention(const array& x, const std::string& prefix, int seq_len) {
    // Separate Q, K, V projections
    auto q = attention_bias_ ? linear_with_bias(x, prefix + "self_attn.q_proj") 
                         : linear(x, prefix + "self_attn.q_proj");

    auto k = attention_bias_ ? linear_with_bias(x, prefix + "self_attn.k_proj") 
                            : linear(x, prefix + "self_attn.k_proj");

    auto v = attention_bias_ ? linear_with_bias(x, prefix + "self_attn.v_proj") 
                            : linear(x, prefix + "self_attn.v_proj");

    // Reshape for multi-head attention
    q = reshape(q, {seq_len, model_.num_attention_heads, head_dim_});
    k = reshape(k, {seq_len, model_.num_key_value_heads, head_dim_});
    v = reshape(v, {seq_len, model_.num_key_value_heads, head_dim_});
    
    // Transpose to [num_heads, seq_len, head_dim]
    q = transpose(q, {1, 0, 2});
    k = transpose(k, {1, 0, 2});
    v = transpose(v, {1, 0, 2});
    
    // Apply RoPE positional embeddings
    float rope_base = model_.rope_theta > 0 ? model_.rope_theta : 10000.0f;
    q = fast::rope(q, head_dim_, false, rope_base, 1.0f, 0);
    k = fast::rope(k, head_dim_, false, rope_base, 1.0f, 0);
    
    // Repeat KV heads if needed (GQA)
    if (model_.num_key_value_heads < model_.num_attention_heads) {
        int n_rep = model_.num_attention_heads / model_.num_key_value_heads;
        k = repeat(k, n_rep, 0);
        v = repeat(v, n_rep, 0);
    }
    
    // Scaled dot-product attention
    float scale = 1.0f / sqrt(static_cast<float>(head_dim_));
    array scores = matmul(q, transpose(k, {0, 2, 1})) * scale;
    scores = scores + triu(full({seq_len, seq_len}, -INFINITY), 1);
    array attn = softmax(scores, -1);
    array out = matmul(attn, v);
    
    // Reshape back to [seq_len, hidden_size]
    out = reshape(transpose(out, {1, 0, 2}), {seq_len, model_.num_attention_heads * head_dim_});
    
    if (attention_bias_) {
        return linear_with_bias(out, prefix + "self_attn.o_proj");
    }
    return linear(out, prefix + "self_attn.o_proj");
}


array LlamaInference::mlp_block(const array& x, const std::string& prefix) {
    auto gate = mlp_bias_ ? linear_with_bias(x, prefix + "mlp.gate_proj") 
                      : linear(x, prefix + "mlp.gate_proj");

    auto up = mlp_bias_ ? linear_with_bias(x, prefix + "mlp.up_proj") 
                        : linear(x, prefix + "mlp.up_proj");
    
    // SwiGLU activation
    array activated = gate * sigmoid(gate) * up;
    
    if (mlp_bias_) {
        return linear_with_bias(activated, prefix + "mlp.down_proj");
    }
    return linear(activated, prefix + "mlp.down_proj");
}


int LlamaInference::sample_token(const array& logits, const MlxOgaGeneratorParams& params) {
    array l = params.do_sample && params.temperature > 0.0f 
        ? logits / params.temperature : logits;
    return static_cast<int>(argmax(l).item<int32_t>());
}


void LlamaInference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    const auto& q = model_.quantization;
    
    std::cout << "[LlamaInference] Caching weights (" 
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
        
        // Also cache bias separately
        auto bias_it = w.find(name + ".bias");
        if (bias_it != w.end()) {
            cached_weights_.emplace(name + ".bias", bias_it->second);
        }
    };

    // Embedding weights
    auto embed_it = w.find("embed_tokens.weight");
    if (embed_it != w.end()) {
        cached_weights_.emplace("embed_tokens.weight", 
            apply_quantization(embed_it->second, "embed_tokens", w, q));
    }

    // Layer weights
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        std::string p = "layers." + std::to_string(i) + ".";
        
        // Layer norms
        auto in_ln = w.find(p + "input_layernorm.weight");
        if (in_ln != w.end()) {
            cached_weights_.emplace(p + "input_layernorm.weight", in_ln->second);
        }
        
        auto post_ln = w.find(p + "post_attention_layernorm.weight");
        if (post_ln != w.end()) {
            cached_weights_.emplace(p + "post_attention_layernorm.weight", post_ln->second);
        }
        
        // Attention projections
        cache_weight(p + "self_attn.q_proj");
        cache_weight(p + "self_attn.k_proj");
        cache_weight(p + "self_attn.v_proj");
        cache_weight(p + "self_attn.o_proj");
        
        // MLP projections
        cache_weight(p + "mlp.gate_proj");
        cache_weight(p + "mlp.up_proj");
        cache_weight(p + "mlp.down_proj");
    }

    // Final norm
    auto norm_it = w.find("norm.weight");
    if (norm_it != w.end()) {
        cached_weights_.emplace("norm.weight", norm_it->second);
    }

    // LM head
    if (!tie_word_embeddings_) {
        cache_weight("lm_head");
    }

    std::cout << "[LlamaInference] Cached " << cached_weights_.size() << " tensors" << std::endl;
}
