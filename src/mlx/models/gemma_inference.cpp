/*
 * gemma_inference.cpp
 * 
 * Gemma transformer inference implementation.
 * Features sliding window attention and separate Q/K/V projections.
 */

#include "ryzenai/mlx/models/gemma_inference.h"
#include "ryzenai/mlx/quantization.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <json.hpp>

using namespace mlx::core;


GemmaInference::GemmaInference(const OgaModel& model) : model_(model) {
    sliding_window_ = 512;

    // Determine actual hidden size from weights or config
    auto embed_it = model_.weights.find("embed_tokens.weight");
    if (embed_it != model_.weights.end()) {
        actual_hidden_size_ = embed_it->second.shape().back();
    } else {
        actual_hidden_size_ = model.hidden_size;
    }

    // 1. Priority: Explicit config value
    if (model.head_dim > 0) {
        head_dim_ = model.head_dim;
    }
    // 2. Fallback: Derive from weight matrix shape (the most "automatic" way)
    else {
        auto q_proj_it = model.weights.find("model.layers.0.self_attn.q_proj.weight");
        if (q_proj_it != model.weights.end()) {
            // Q weight shape is usually [Hidden_Size, Total_Query_Dim]
            int q_output_size = static_cast<int>(q_proj_it->second.shape(1));
            head_dim_ = q_output_size / model.num_attention_heads;
        }
        // 3. Last Resort: Standard math
        else {
            head_dim_ = actual_hidden_size_ / model.num_attention_heads;
        }
    }

    std::cout << "[GemmaInference] hidden_size: " << actual_hidden_size_
              << ", head_dim: " << head_dim_ << std::endl;

    std::string config_path = model.model_path + "/config.json";
    std::ifstream f(config_path);
    if (f.is_open()) {
        nlohmann::json config;
        f >> config;
        if (config.contains("sliding_window")) {
            sliding_window_ = config["sliding_window"];
        }
    }
}


/*
 * forward
 * 
 * Full forward pass through the Gemma model.
 * Embeds input tokens, processes through layers, and projects to vocabulary.
 */
array GemmaInference::forward(const std::vector<int32_t>& input_tokens,
                             const OgaGeneratorParams& params) {
    int seq_len = static_cast<int>(input_tokens.size());

    auto embed_it = model_.weights.find("embed_tokens.weight");
    if (embed_it == model_.weights.end()) {
        throw std::runtime_error("embed_tokens.weight not found");
    }

    array embeddings = take(embed_it->second, array(input_tokens.data(), {seq_len}, int32), 0);
    embeddings = apply_quantization(embeddings, "embed_tokens", model_.weights);

    array hidden_states = embeddings;
    for (int layer = 0; layer < model_.num_hidden_layers; ++layer) {
        hidden_states = process_layer(hidden_states, layer, seq_len);
    }

    auto norm_it = model_.weights.find("model.norm.weight");
    if (norm_it != model_.weights.end()) {
        array variance = mean(square(hidden_states), -1, true);
        hidden_states = hidden_states * norm_it->second / sqrt(variance + model_.rms_norm_eps);
    }

    auto lm_head_it = model_.weights.find("lm_head.weight");
    if (lm_head_it == model_.weights.end()) {
        throw std::runtime_error("lm_head.weight not found");
    }

    array lm_weight = apply_quantization(lm_head_it->second, "lm_head", model_.weights);

    array last_hidden = take(hidden_states, array({seq_len-1}, {1}, int32), 0);
    auto actual_shape = last_hidden.shape();
    last_hidden = reshape(last_hidden, {actual_shape[1]});
    array lm_weight_t = transpose(lm_weight, {1, 0});
    array logits = matmul(last_hidden, lm_weight_t);

    array logits_mean = mean(logits);
    logits = logits - logits_mean;

    return logits;
}


/*
 * process_layer
 * 
 * Processes a single transformer layer.
 * Applies pre-norm, attention, post-norm, and MLP blocks.
 */
array GemmaInference::process_layer(const array& hidden_states, int layer_idx, int seq_len) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    array current_states = hidden_states;

    auto norm_it = model_.weights.find(prefix + "input_layernorm.weight");
    if (norm_it != model_.weights.end()) {
        array variance = mean(square(current_states), -1, true);
        current_states = current_states * norm_it->second / sqrt(variance + model_.rms_norm_eps);
    }

    current_states = self_attention(current_states, prefix, seq_len);

    auto post_norm_it = model_.weights.find(prefix + "post_attention_layernorm.weight");
    if (post_norm_it != model_.weights.end()) {
        array variance = mean(square(current_states), -1, true);
        current_states = current_states * post_norm_it->second / sqrt(variance + model_.rms_norm_eps);
    }

    current_states = mlp_block(current_states, prefix);

    return current_states;
}


/*
 * self_attention
 * 
 * Multi-head self attention with separate Q/K/V projections.
 * Uses sliding window mask for efficient attention over long sequences.
 */
array GemmaInference::self_attention(const array& hidden_states, const std::string& prefix, int seq_len) {
    auto q_proj_it = model_.weights.find(prefix + "self_attn.q_proj.weight");
    auto k_proj_it = model_.weights.find(prefix + "self_attn.k_proj.weight");
    auto v_proj_it = model_.weights.find(prefix + "self_attn.v_proj.weight");
    auto o_proj_it = model_.weights.find(prefix + "self_attn.o_proj.weight");

    if (q_proj_it == model_.weights.end() || k_proj_it == model_.weights.end() ||
        v_proj_it == model_.weights.end() || o_proj_it == model_.weights.end()) {
        return hidden_states;
    }

    array q_weight = apply_quantization(q_proj_it->second, prefix + "self_attn.q_proj", model_.weights);
    array k_weight = apply_quantization(k_proj_it->second, prefix + "self_attn.k_proj", model_.weights);
    array v_weight = apply_quantization(v_proj_it->second, prefix + "self_attn.v_proj", model_.weights);
    array o_weight = apply_quantization(o_proj_it->second, prefix + "self_attn.o_proj", model_.weights);

    array q = matmul(hidden_states, q_weight);
    array k = matmul(hidden_states, k_weight);
    array v = matmul(hidden_states, v_weight);

    q = reshape(q, {seq_len, model_.num_attention_heads, head_dim_});
    k = reshape(k, {seq_len, model_.num_attention_heads, head_dim_});
    v = reshape(v, {seq_len, model_.num_attention_heads, head_dim_});

    q = transpose(q, {1, 0, 2});
    k = transpose(k, {1, 0, 2});
    v = transpose(v, {1, 0, 2});

    array attn_scores = matmul(q, transpose(k, {0, 2, 1})) / sqrt(static_cast<float>(head_dim_));
    attn_scores = apply_sliding_window_mask(attn_scores, seq_len);
    array attn_weights = softmax(attn_scores, -1);
    array attn_output = matmul(attn_weights, v);

    attn_output = transpose(attn_output, {1, 0, 2});
    auto hidden_shape = hidden_states.shape();
    attn_output = reshape(attn_output, {seq_len, hidden_shape.back()});

    array final_output = matmul(attn_output, o_weight);

    return hidden_states + final_output;
}


/*
 * apply_sliding_window_mask
 * 
 * Creates a causal attention mask limited to the sliding window size.
 * Positions outside the window receive -inf to zero out attention.
 */
array GemmaInference::apply_sliding_window_mask(const array& attn_scores, int seq_len) {
    if (seq_len <= 1) {
        return tril(attn_scores, 0);
    }

    array i_indices = arange(seq_len);
    array j_indices = arange(seq_len);

    array i_expanded = expand_dims(i_indices, 1);
    array j_expanded = expand_dims(j_indices, 0);

    array window_mask = (j_expanded >= (i_expanded - sliding_window_ + 1)) & (j_expanded <= i_expanded);
    array float_mask = astype(window_mask, float32) * 0.0f + astype(~window_mask, float32) * (-INFINITY);
    array head_mask = expand_dims(float_mask, 0);
    array final_mask = broadcast_to(head_mask, {model_.num_attention_heads, seq_len, seq_len});

    return attn_scores + final_mask;
}


/*
 * mlp_block
 * 
 * SwiGLU MLP block with gate, up, and down projections.
 * Computes: down_proj(silu(gate_proj(x)) * up_proj(x))
 */
array GemmaInference::mlp_block(const array& hidden_states, const std::string& prefix) {
    auto gate_proj_it = model_.weights.find(prefix + "mlp.gate_proj.weight");
    auto up_proj_it = model_.weights.find(prefix + "mlp.up_proj.weight");
    auto down_proj_it = model_.weights.find(prefix + "mlp.down_proj.weight");

    if (gate_proj_it == model_.weights.end() || up_proj_it == model_.weights.end() ||
        down_proj_it == model_.weights.end()) {
        return hidden_states;
    }

    array gate_weight = apply_quantization(gate_proj_it->second, prefix + "mlp.gate_proj", model_.weights);
    array up_weight = apply_quantization(up_proj_it->second, prefix + "mlp.up_proj", model_.weights);
    array down_weight = apply_quantization(down_proj_it->second, prefix + "mlp.down_proj", model_.weights);

    array gate = matmul(hidden_states, gate_weight);
    array up = matmul(hidden_states, up_weight);
    array silu_gate = gate * sigmoid(gate);
    array mlp_output = matmul(silu_gate * up, down_weight);

    return hidden_states + mlp_output;
}


/*
 * sample_token
 * 
 * Samples next token from logits.
 * Uses temperature scaling when sampling is enabled.
 */
int GemmaInference::sample_token(const array& logits, const OgaGeneratorParams& params) {
    if (params.do_sample) {
        if (params.temperature > 0.0f) {
            array scaled_logits = logits / params.temperature;
            array probs = softmax(scaled_logits);
            auto max_idx = argmax(probs);
            return static_cast<int>(max_idx.item<int32_t>());
        } else {
            auto max_idx = argmax(logits);
            return static_cast<int>(max_idx.item<int32_t>());
        }
    } else {
        auto max_idx = argmax(logits);
        return static_cast<int>(max_idx.item<int32_t>());
    }
}
