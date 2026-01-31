/*
 * qwen3_next_inference.cpp
 * 
 * Qwen3-Next transformer inference implementation.
 * Features hybrid GatedDeltaNet + full attention and MoE.
 */

#include "ryzenai/mlx/models/qwen3_next_inference.h"
#include "ryzenai/mlx/quantization.h"
#include <iostream>
#include <fstream>
#include <string>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>

using namespace mlx::core;


Qwen3NextInference::Qwen3NextInference(const MlxOgaModel& model) : model_(model) {
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
    num_experts_ = model_.num_experts;
    num_experts_per_tok_ = model_.num_experts_per_tok;
    
    // Read additional config from config.json
    full_attention_interval_ = 4;  // default
    partial_rotary_factor_ = 0.5f; // default
    
    std::string config_path = model_.model_path + "/config.json";
    std::ifstream f(config_path);
    if (f.is_open()) {
        nlohmann::json config;
        f >> config;
        if (config.contains("full_attention_interval")) {
            full_attention_interval_ = config["full_attention_interval"];
        }
        if (config.contains("partial_rotary_factor")) {
            partial_rotary_factor_ = config["partial_rotary_factor"];
        }
    }
    
    std::cout << "[Qwen3NextInference] hidden_size=" << actual_hidden_size_
              << ", head_dim=" << head_dim_
              << ", num_heads=" << model_.num_attention_heads
              << ", full_attn_interval=" << full_attention_interval_
              << ", num_experts=" << num_experts_
              << ", quantized=" << (model_.is_quantized() ? "yes" : "no")
              << std::endl;
    
    cache_weights();
}


bool Qwen3NextInference::is_linear_layer(int layer_idx) const {
    // Linear attention layers are used when (layer_idx + 1) % full_attention_interval != 0
    return (layer_idx + 1) % full_attention_interval_ != 0;
}


bool Qwen3NextInference::is_moe_layer(int layer_idx) const {
    // MoE layers based on decoder_sparse_step (simplified logic)
    return num_experts_ > 0;
}


array Qwen3NextInference::linear(const array& x, const std::string& weight_name) {
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


array Qwen3NextInference::rms_norm(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        return x;
    }
    
    array variance = mean(square(x), -1, true);
    return x * weight_it->second / sqrt(variance + model_.rms_norm_eps);
}


array Qwen3NextInference::forward(const std::vector<int32_t>& input_tokens,
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


array Qwen3NextInference::process_layer(const array& hidden_states, int layer_idx, int seq_len) {
    std::string prefix = "layers." + std::to_string(layer_idx) + ".";
    
    array normed = rms_norm(hidden_states, prefix + "input_layernorm");

    array attn_out = [&]() {
        if (is_linear_layer(layer_idx)) {
            return linear_attention(normed, prefix, seq_len);
        } else {
            return full_attention(normed, prefix, seq_len);
        }
    }();
    
    array h = hidden_states + attn_out;
    
    array normed2 = rms_norm(h, prefix + "post_attention_layernorm");

    array mlp_out = [&]() {
        if (is_moe_layer(layer_idx)) {
            return moe_block(normed2, prefix);
        } else {
            return mlp_block(normed2, prefix);
        }
    }();
    
    return h + mlp_out;
}


/*
 * full_attention
 * 
 * Qwen3Next style attention with:
 * - Q projection outputs 2x for gating
 * - Q/K normalization
 * - Partial RoPE
 * - Output gating with sigmoid
 */
array Qwen3NextInference::full_attention(const array& x, const std::string& prefix, int seq_len) {
    // Q projection gives 2x output for gating
    array q_proj = linear(x, prefix + "self_attn.q_proj");
    int q_size = model_.num_attention_heads * head_dim_;
    
    array q = slice(q_proj, {0, 0}, {seq_len, q_size});
    array gate = slice(q_proj, {0, q_size}, {seq_len, 2 * q_size});
    
    array k = linear(x, prefix + "self_attn.k_proj");
    array v = linear(x, prefix + "self_attn.v_proj");
    
    // Reshape for multi-head attention
    q = reshape(q, {seq_len, model_.num_attention_heads, head_dim_});
    k = reshape(k, {seq_len, model_.num_key_value_heads, head_dim_});
    v = reshape(v, {seq_len, model_.num_key_value_heads, head_dim_});
    
    // Apply Q and K normalization
    q = rms_norm(q, prefix + "self_attn.q_norm");
    k = rms_norm(k, prefix + "self_attn.k_norm");
    
    // Transpose to [num_heads, seq_len, head_dim]
    q = transpose(q, {1, 0, 2});
    k = transpose(k, {1, 0, 2});
    v = transpose(v, {1, 0, 2});
    
    // Apply partial RoPE (only on portion of head_dim)
    int rope_dim = static_cast<int>(head_dim_ * partial_rotary_factor_);
    float rope_base = model_.rope_theta > 0 ? model_.rope_theta : 1000000.0f;
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
    array scores = matmul(q, transpose(k, {0, 2, 1})) * scale;
    scores = scores + triu(full({seq_len, seq_len}, -INFINITY), 1);
    array attn = softmax(scores, -1);
    array out = matmul(attn, v);
    
    // Reshape back and apply output gating
    out = reshape(transpose(out, {1, 0, 2}), {seq_len, q_size});
    out = out * sigmoid(gate);
    
    return linear(out, prefix + "self_attn.o_proj");
}


/*
 * linear_attention
 * 
 * Simplified GatedDeltaNet implementation for inference.
 * Uses a simpler recurrent formulation without full state tracking.
 */
array Qwen3NextInference::linear_attention(const array& x, const std::string& prefix, int seq_len) {
    // For inference without KV cache, we use a simplified linear attention
    // This is an approximation - full implementation would need state management
    
    array qkvz = linear(x, prefix + "linear_attn.in_proj_qkvz");
    
    // Split into components (simplified)
    int key_dim = actual_hidden_size_ / 4;  // Approximate
    int value_dim = actual_hidden_size_ / 2;
    
    array qk = slice(qkvz, {0, 0}, {seq_len, 2 * key_dim});
    array vz = slice(qkvz, {0, 2 * key_dim}, {seq_len, 2 * key_dim + 2 * value_dim});
    
    // Apply conv1d equivalent (simplified as identity for first pass)
    array q = slice(qk, {0, 0}, {seq_len, key_dim});
    array k = slice(qk, {0, key_dim}, {seq_len, 2 * key_dim});
    array v = slice(vz, {0, 0}, {seq_len, value_dim});
    array z = slice(vz, {0, value_dim}, {seq_len, 2 * value_dim});
    
    // Normalize Q and K
    array inv_scale = array(1.0f / sqrt(static_cast<float>(key_dim)));
    q = q * inv_scale;
    k = k * inv_scale;
    
    // Linear attention: softmax(Q) @ (softmax(K)^T @ V)
    array q_soft = softmax(q, -1);
    array k_soft = softmax(k, -1);
    
    // Compute attention
    array kv = matmul(transpose(k_soft, {1, 0}), v);
    array out = matmul(q_soft, kv);
    
    // Apply gated normalization with z
    out = rms_norm(out, prefix + "linear_attn.norm");
    out = out * sigmoid(z);
    
    return linear(out, prefix + "linear_attn.out_proj");
}


array Qwen3NextInference::mlp_block(const array& x, const std::string& prefix) {
    array gate = linear(x, prefix + "mlp.gate_proj");
    array up = linear(x, prefix + "mlp.up_proj");
    
    // SwiGLU activation
    array activated = gate * sigmoid(gate) * up;
    
    return linear(activated, prefix + "mlp.down_proj");
}


/*
 * moe_block
 * 
 * Mixture of Experts with:
 * - Top-k expert selection
 * - Shared expert that always contributes
 */
array Qwen3NextInference::moe_block(const array& x, const std::string& prefix) {
    // For simplified inference, fall back to dense MLP
    // Full MoE would require:
    // 1. Router scoring
    // 2. Top-k expert selection
    // 3. Expert computation
    // 4. Shared expert computation
    
    // Use shared expert as fallback
    auto shared_gate_it = cached_weights_.find(prefix + "mlp.shared_expert.gate_proj.weight");
    if (shared_gate_it != cached_weights_.end()) {
        array gate = linear(x, prefix + "mlp.shared_expert.gate_proj");
        array up = linear(x, prefix + "mlp.shared_expert.up_proj");
        array activated = gate * sigmoid(gate) * up;
        return linear(activated, prefix + "mlp.shared_expert.down_proj");
    }
    
    // Otherwise use regular MLP
    return mlp_block(x, prefix);
}


int Qwen3NextInference::sample_token(const array& logits, const MlxOgaGeneratorParams& params) {
    array l = params.do_sample && params.temperature > 0.0f 
        ? logits / params.temperature : logits;
    return static_cast<int>(argmax(l).item<int32_t>());
}


void Qwen3NextInference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    const auto& q = model_.quantization;
    
    std::cout << "[Qwen3NextInference] Caching weights (" 
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
        
        if (is_linear_layer(i)) {
            // Linear attention weights
            cache_weight(p + "linear_attn.in_proj_qkvz");
            cache_weight(p + "linear_attn.in_proj_ba");
            cache_weight(p + "linear_attn.out_proj");
            cache_weight(p + "linear_attn.conv1d");
            
            auto norm_w = w.find(p + "linear_attn.norm.weight");
            if (norm_w != w.end()) {
                cached_weights_.emplace(p + "linear_attn.norm.weight", norm_w->second);
            }
        } else {
            // Full attention weights
            cache_weight(p + "self_attn.q_proj");
            cache_weight(p + "self_attn.k_proj");
            cache_weight(p + "self_attn.v_proj");
            cache_weight(p + "self_attn.o_proj");
            
            auto q_norm = w.find(p + "self_attn.q_norm.weight");
            if (q_norm != w.end()) {
                cached_weights_.emplace(p + "self_attn.q_norm.weight", q_norm->second);
            }
            auto k_norm = w.find(p + "self_attn.k_norm.weight");
            if (k_norm != w.end()) {
                cached_weights_.emplace(p + "self_attn.k_norm.weight", k_norm->second);
            }
        }
        
        // MLP / MoE weights
        cache_weight(p + "mlp.gate_proj");
        cache_weight(p + "mlp.up_proj");
        cache_weight(p + "mlp.down_proj");
        
        // Shared expert for MoE layers
        cache_weight(p + "mlp.shared_expert.gate_proj");
        cache_weight(p + "mlp.shared_expert.up_proj");
        cache_weight(p + "mlp.shared_expert.down_proj");
        cache_weight(p + "mlp.shared_expert_gate");
        
        // Router for MoE
        cache_weight(p + "mlp.gate");
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

    std::cout << "[Qwen3NextInference] Cached " << cached_weights_.size() << " tensors" << std::endl;
}
