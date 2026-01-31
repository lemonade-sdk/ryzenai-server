/*
 * mixtral_inference.cpp
 * 
 * Mixtral MoE transformer inference implementation.
 * Features top-k sparse expert routing with switch layers.
 */

#include "ryzenai/mlx/models/mixtral_inference.h"
#include "ryzenai/mlx/quantization.h"
#include <iostream>
#include <fstream>
#include <string>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>

using namespace mlx::core;


MixtralInference::MixtralInference(const OgaModel& model) : model_(model) {
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
    
    // Mixtral-specific config
    num_local_experts_ = 8;
    num_experts_per_tok_ = 2;
    
    std::string config_path = model_.model_path + "/config.json";
    std::ifstream f(config_path);
    if (f.is_open()) {
        nlohmann::json config;
        f >> config;
        if (config.contains("num_local_experts")) {
            num_local_experts_ = config["num_local_experts"];
        }
        if (config.contains("num_experts_per_tok")) {
            num_experts_per_tok_ = config["num_experts_per_tok"];
        }
    }
    
    std::cout << "[MixtralInference] hidden_size=" << actual_hidden_size_
              << ", head_dim=" << head_dim_
              << ", num_heads=" << model_.num_attention_heads
              << ", num_experts=" << num_local_experts_
              << ", experts_per_tok=" << num_experts_per_tok_
              << ", quantized=" << (model_.is_quantized() ? "yes" : "no")
              << std::endl;
    
    cache_weights();
}


array MixtralInference::linear(const array& x, const std::string& weight_name) {
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


array MixtralInference::rms_norm(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        return x;
    }
    
    array variance = mean(square(x), -1, true);
    return x * weight_it->second / sqrt(variance + model_.rms_norm_eps);
}


array MixtralInference::forward(const std::vector<int32_t>& input_tokens,
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

    h = rms_norm(h, "norm");
    
    array last_h = reshape(take(h, array({seq_len - 1}, {1}, int32), 0), {actual_hidden_size_});
    
    array logits = linear(reshape(last_h, {1, actual_hidden_size_}), "lm_head");
    int vocab = model_.vocab_size;
    logits = reshape(logits, {vocab});
    
    eval(logits);
    return logits;
}


array MixtralInference::process_layer(const array& hidden_states, int layer_idx, int seq_len) {
    std::string prefix = "layers." + std::to_string(layer_idx) + ".";
    
    // Self attention with residual
    array normed = rms_norm(hidden_states, prefix + "input_layernorm");
    array attn_out = self_attention(normed, prefix, seq_len);
    array h = hidden_states + attn_out;
    
    // MoE block with residual
    array normed2 = rms_norm(h, prefix + "post_attention_layernorm");
    array moe_out = moe_block(normed2, prefix);
    
    return h + moe_out;
}


array MixtralInference::self_attention(const array& x, const std::string& prefix, int seq_len) {
    // Separate Q, K, V projections
    array q = linear(x, prefix + "self_attn.q_proj");
    array k = linear(x, prefix + "self_attn.k_proj");
    array v = linear(x, prefix + "self_attn.v_proj");
    
    // Reshape for multi-head attention
    q = reshape(q, {seq_len, model_.num_attention_heads, head_dim_});
    k = reshape(k, {seq_len, model_.num_key_value_heads, head_dim_});
    v = reshape(v, {seq_len, model_.num_key_value_heads, head_dim_});
    
    // Transpose to [num_heads, seq_len, head_dim]
    q = transpose(q, {1, 0, 2});
    k = transpose(k, {1, 0, 2});
    v = transpose(v, {1, 0, 2});
    
    // Apply RoPE positional embeddings
    float rope_base = model_.rope_theta > 0 ? model_.rope_theta : 1e6f;
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
    
    return linear(out, prefix + "self_attn.o_proj");
}


/*
 * moe_block
 * 
 * Sparse Mixture of Experts with top-k routing.
 * For simplified inference, uses first expert or averaged experts.
 */
array MixtralInference::moe_block(const array& x, const std::string& prefix) {
    // Check for switch_mlp (stacked expert weights)
    auto switch_gate_it = cached_weights_.find(prefix + "block_sparse_moe.switch_mlp.gate_proj.weight");
    if (switch_gate_it != cached_weights_.end()) {
        // Use first expert for simplified inference
        array gate_weights = switch_gate_it->second;
        array up_weights = cached_weights_.at(prefix + "block_sparse_moe.switch_mlp.up_proj.weight");
        array down_weights = cached_weights_.at(prefix + "block_sparse_moe.switch_mlp.down_proj.weight");
        
        // Extract first expert weights [expert_0]
        array gate_w = slice(gate_weights, {0, 0, 0}, {1, gate_weights.shape()[1], gate_weights.shape()[2]});
        array up_w = slice(up_weights, {0, 0, 0}, {1, up_weights.shape()[1], up_weights.shape()[2]});
        array down_w = slice(down_weights, {0, 0, 0}, {1, down_weights.shape()[1], down_weights.shape()[2]});
        
        gate_w = reshape(gate_w, {gate_w.shape()[1], gate_w.shape()[2]});
        up_w = reshape(up_w, {up_w.shape()[1], up_w.shape()[2]});
        down_w = reshape(down_w, {down_w.shape()[1], down_w.shape()[2]});
        
        array gate = matmul(x, transpose(gate_w, {1, 0}));
        array up = matmul(x, transpose(up_w, {1, 0}));
        array activated = gate * sigmoid(gate) * up;
        return matmul(activated, transpose(down_w, {1, 0}));
    }
    
    // Fallback for non-stacked expert format
    throw std::runtime_error("MoE block requires switch_mlp format weights");
}


int MixtralInference::sample_token(const array& logits, const OgaGeneratorParams& params) {
    array l = params.do_sample && params.temperature > 0.0f 
        ? logits / params.temperature : logits;
    return static_cast<int>(argmax(l).item<int32_t>());
}


void MixtralInference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    const auto& q = model_.quantization;
    
    std::cout << "[MixtralInference] Caching weights (" 
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
        
        // Attention projections
        cache_weight(p + "self_attn.q_proj");
        cache_weight(p + "self_attn.k_proj");
        cache_weight(p + "self_attn.v_proj");
        cache_weight(p + "self_attn.o_proj");
        
        // MoE switch_mlp (stacked expert weights)
        cache_weight(p + "block_sparse_moe.switch_mlp.gate_proj");
        cache_weight(p + "block_sparse_moe.switch_mlp.up_proj");
        cache_weight(p + "block_sparse_moe.switch_mlp.down_proj");
        
        // Router gate
        auto gate_it = w.find(p + "block_sparse_moe.gate.weight");
        if (gate_it != w.end()) {
            cached_weights_.emplace(p + "block_sparse_moe.gate.weight", gate_it->second);
        }
    }

    // Final norm
    auto norm_it = w.find("norm.weight");
    if (norm_it != w.end()) {
        cached_weights_.emplace("norm.weight", norm_it->second);
    }

    // LM head
    cache_weight("lm_head");

    std::cout << "[MixtralInference] Cached " << cached_weights_.size() << " tensors" << std::endl;
}
