/*
 * phi_inference.cpp
 * 
 * Original Phi transformer inference implementation.
 * Features parallel attention and MLP, LayerNorm, GELU, and partial RoPE.
 */

#include "ryzenai/mlx/models/phi_inference.h"
#include "ryzenai/mlx/quantization.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>

using namespace mlx::core;


PhiInference::PhiInference(const OgaModel& model) : model_(model) {
    actual_hidden_size_ = model_.hidden_size;
    head_dim_ = model_.head_dim > 0 ? model_.head_dim : (actual_hidden_size_ / model_.num_attention_heads);
    
    // Phi-specific config
    partial_rotary_factor_ = 0.4f;
    layer_norm_eps_ = 1e-5f;
    
    std::string config_path = model_.model_path + "/config.json";
    std::ifstream f(config_path);
    if (f.is_open()) {
        nlohmann::json config;
        f >> config;
        if (config.contains("partial_rotary_factor")) {
            partial_rotary_factor_ = config["partial_rotary_factor"];
        }
        if (config.contains("layer_norm_eps")) {
            layer_norm_eps_ = config["layer_norm_eps"];
        }
    }
    
    std::cout << "[PhiInference] hidden_size=" << actual_hidden_size_
              << ", head_dim=" << head_dim_
              << ", num_heads=" << model_.num_attention_heads
              << ", partial_rotary_factor=" << partial_rotary_factor_
              << ", quantized=" << (model_.is_quantized() ? "yes" : "no")
              << std::endl;
    
    cache_weights();
}


array PhiInference::linear(const array& x, const std::string& weight_name) {
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


array PhiInference::linear_with_bias(const array& x, const std::string& weight_name) {
    array out = linear(x, weight_name);
    
    auto bias_it = cached_weights_.find(weight_name + ".bias");
    if (bias_it != cached_weights_.end()) {
        out = out + bias_it->second;
    }
    
    return out;
}


/*
 * layer_norm
 * 
 * Standard layer normalization with learned weight and bias.
 * Normalizes to mean=0, variance=1, then applies affine transform.
 */
array PhiInference::layer_norm(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        return x;
    }
    
    // Compute mean and variance
    array mean_val = mean(x, -1, true);
    array variance = mean(square(x - mean_val), -1, true);
    array normalized = (x - mean_val) / sqrt(variance + layer_norm_eps_);
    
    // Apply weight
    array result = normalized * weight_it->second;
    
    // Apply bias if present
    auto bias_it = cached_weights_.find(weight_name + ".bias");
    if (bias_it != cached_weights_.end()) {
        result = result + bias_it->second;
    }
    
    return result;
}


array PhiInference::forward(const std::vector<int32_t>& input_tokens,
                            const OgaGeneratorParams& params) {
    int seq_len = static_cast<int>(input_tokens.size());

    auto embed_it = cached_weights_.find("embed_tokens.weight");
    if (embed_it == cached_weights_.end()) {
        throw std::runtime_error("embed_tokens.weight not found");
    }
    array h = take(embed_it->second, array(input_tokens.data(), {seq_len}, int32), 0);

    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        h = process_layer(h, i, seq_len);
    }

    h = layer_norm(h, "final_layernorm");
    
    array last_h = reshape(take(h, array({seq_len - 1}, {1}, int32), 0), {actual_hidden_size_});
    
    array logits = linear_with_bias(reshape(last_h, {1, actual_hidden_size_}), "lm_head");
    int vocab = model_.vocab_size;
    logits = reshape(logits, {vocab});
    
    eval(logits);
    return logits;
}


/*
 * process_layer
 * 
 * Phi uses parallel attention and MLP:
 * h = x + attention(layernorm(x)) + mlp(layernorm(x))
 */
array PhiInference::process_layer(const array& hidden_states, int layer_idx, int seq_len) {
    std::string prefix = "layers." + std::to_string(layer_idx) + ".";
    
    // Apply input layernorm once (shared for attention and MLP)
    array normed = layer_norm(hidden_states, prefix + "input_layernorm");
    
    // Parallel attention and MLP
    array attn_out = self_attention(normed, prefix, seq_len);
    array mlp_out = mlp_block(normed, prefix);
    
    // Combine: residual + attention + mlp
    return hidden_states + attn_out + mlp_out;
}


array PhiInference::self_attention(const array& x, const std::string& prefix, int seq_len) {
    // Separate Q, K, V projections with bias
    array q = linear_with_bias(x, prefix + "self_attn.q_proj");
    array k = linear_with_bias(x, prefix + "self_attn.k_proj");
    array v = linear_with_bias(x, prefix + "self_attn.v_proj");
    
    // Reshape for multi-head attention
    q = reshape(q, {seq_len, model_.num_attention_heads, head_dim_});
    k = reshape(k, {seq_len, model_.num_key_value_heads, head_dim_});
    v = reshape(v, {seq_len, model_.num_key_value_heads, head_dim_});
    
    // Transpose to [num_heads, seq_len, head_dim]
    q = transpose(q, {1, 0, 2});
    k = transpose(k, {1, 0, 2});
    v = transpose(v, {1, 0, 2});
    
    // Apply partial RoPE (only on portion of head_dim)
    int rope_dim = static_cast<int>(partial_rotary_factor_ * head_dim_);
    float rope_base = model_.rope_theta > 0 ? model_.rope_theta : 10000.0f;
    q = fast::rope(q, rope_dim, false, rope_base, 1.0f, 0);
    k = fast::rope(k, rope_dim, false, rope_base, 1.0f, 0);
    
    // Repeat KV heads if needed (GQA)
    if (model_.num_key_value_heads < model_.num_attention_heads) {
        int n_rep = model_.num_attention_heads / model_.num_key_value_heads;
        k = repeat(k, n_rep, 0);
        v = repeat(v, n_rep, 0);
    }
    
    // Scaled dot-product attention
    float scale = 1.0f / sqrt(static_cast<float>(head_dim_));
    array scores = matmul(astype(q, float32), transpose(k, {0, 2, 1})) * scale;
    scores = scores + triu(full({seq_len, seq_len}, -INFINITY), 1);
    array attn = softmax(scores, -1);
    array out = matmul(attn, v);
    
    // Reshape back to [seq_len, hidden_size]
    out = reshape(transpose(out, {1, 0, 2}), {seq_len, model_.num_attention_heads * head_dim_});
    
    // Output projection with bias (called "dense" in Phi)
    return linear_with_bias(out, prefix + "self_attn.dense");
}


/*
 * mlp_block
 * 
 * Phi uses simple GELU MLP with two linear layers.
 * fc1 -> GELU -> fc2
 */
array PhiInference::mlp_block(const array& x, const std::string& prefix) {
    // fc1 with bias
    array h = linear_with_bias(x, prefix + "mlp.fc1");
    
    // GELU activation (approximate)
    // gelu_approx(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float sqrt_2_over_pi = 0.7978845608f;
    array x3 = h * h * h;
    array inner = sqrt_2_over_pi * (h + 0.044715f * x3);
    h = 0.5f * h * (1.0f + tanh(inner));
    
    // fc2 with bias
    return linear_with_bias(h, prefix + "mlp.fc2");
}


int PhiInference::sample_token(const array& logits, const OgaGeneratorParams& params) {
    array l = params.do_sample && params.temperature > 0.0f 
        ? logits / params.temperature : logits;
    return static_cast<int>(argmax(l).item<int32_t>());
}


void PhiInference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    const auto& q = model_.quantization;
    
    std::cout << "[PhiInference] Caching weights (" 
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
        
        // Input LayerNorm
        auto ln_w = w.find(p + "input_layernorm.weight");
        if (ln_w != w.end()) {
            cached_weights_.emplace(p + "input_layernorm.weight", ln_w->second);
        }
        auto ln_b = w.find(p + "input_layernorm.bias");
        if (ln_b != w.end()) {
            cached_weights_.emplace(p + "input_layernorm.bias", ln_b->second);
        }
        
        // Attention projections with bias
        cache_weight(p + "self_attn.q_proj");
        cache_weight(p + "self_attn.k_proj");
        cache_weight(p + "self_attn.v_proj");
        cache_weight(p + "self_attn.dense");  // output projection
        
        // MLP with bias
        cache_weight(p + "mlp.fc1");
        cache_weight(p + "mlp.fc2");
    }

    // Final LayerNorm
    auto final_ln_w = w.find("final_layernorm.weight");
    if (final_ln_w != w.end()) {
        cached_weights_.emplace("final_layernorm.weight", final_ln_w->second);
    }
    auto final_ln_b = w.find("final_layernorm.bias");
    if (final_ln_b != w.end()) {
        cached_weights_.emplace("final_layernorm.bias", final_ln_b->second);
    }

    // LM head with bias
    cache_weight("lm_head");

    std::cout << "[PhiInference] Cached " << cached_weights_.size() << " tensors" << std::endl;
}
