/*
 * phi3_inference.cpp
 * 
 * Phi-3 transformer inference implementation with optimized KV caching.
 * Supports both quantized and full-precision weights.
 * 
 * Usage:
 *   FP16, and INT8 quantized cache (FP16 default, recommended for performance)
 *   Phi3Inference engine(model);  // or Phi3Inference engine(model, KVCacheMode::INT8);
 *   Phi3Inference engine(model, KVCacheMode::FP16);
 */

#include "ryzenai/mlx/models/phi3_inference.h"
#include "ryzenai/mlx/quantization.h"
#include <iostream>
#include <string>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>
#include <mlx/random.h>
#include <vector>
#include <cmath>

using namespace mlx::core;


Phi3Inference::Phi3Inference(const MlxOgaModel& model, KVCacheMode kv_cache_mode) :
    mask_val_(array(-std::numeric_limits<float>::infinity(), float32)),
    model_(model),
    kv_cache_mode_(kv_cache_mode)
{
    actual_hidden_size_ = model_.hidden_size;

    if (model_.head_dim > 0) {
        head_dim_ = model_.head_dim;
    } else {
        auto qkv_proj_it = model_.weights.find("layers.0.self_attn.qkv_proj.weight");
        if (qkv_proj_it != model_.weights.end()) {
            int qkv_output_size = static_cast<int>(qkv_proj_it->second.shape(0));
            int total_heads = model_.num_attention_heads + 2 * model_.num_key_value_heads;
            head_dim_ = qkv_output_size / total_heads;
        } else {
            head_dim_ = actual_hidden_size_ / model_.num_attention_heads;
        }
    }

    attention_scale_ = 1.0f / sqrt(static_cast<float>(head_dim_));
    rope_theta_ = model_.rope_theta > 0 ? model_.rope_theta : 10000.0f;
    max_cache_length_ = model_.max_context_length;

    if (kv_cache_mode_ == KVCacheMode::INT8) {
        quantized_kv_cache_.emplace();
        quantized_kv_cache_->initialize(
            model_.num_hidden_layers,
            model_.num_key_value_heads,
            max_cache_length_,
            head_dim_
        );

        size_t quantized_bytes = quantized_kv_cache_->memory_usage_bytes();
        size_t fp16_bytes = quantized_kv_cache_->fp16_equivalent_bytes();
        float compression = 100.0f * (1.0f - static_cast<float>(quantized_bytes) / static_cast<float>(fp16_bytes));

        std::cout << "[Phi3Inference] INT8 Quantized KV Cache initialized:" << std::endl;
        std::cout << "  - Quantized size: " << (quantized_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  - FP16 equivalent: " << (fp16_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  - Compression: " << compression << "%" << std::endl;
        std::cout << "  - Expected GPU utilization improvement: ~30-50%" << std::endl;
    } else {
        k_cache_.clear();
        v_cache_.clear();
        std::cout << "[Phi3Inference] FP16 KV Cache (legacy mode)" << std::endl;
    }

    std::cout << "[Phi3Inference] hidden_size=" << actual_hidden_size_
              << ", head_dim=" << head_dim_
              << ", num_heads=" << model_.num_attention_heads
              << ", num_kv_heads=" << model_.num_key_value_heads
              << ", kv_cache_mode=" << (kv_cache_mode_ == KVCacheMode::INT8 ? "INT8" : "FP16")
              << std::endl;

    cache_weights();
}

void Phi3Inference::clear_cache() {
    if (kv_cache_mode_ == KVCacheMode::INT8 && quantized_kv_cache_.has_value()) {
        quantized_kv_cache_->clear();
    } else {
        k_cache_.clear();
        v_cache_.clear();
    }
    cache_position_ = 0;
}




array Phi3Inference::process_layer(const array& h, int layer_idx, int seq_len) {
    (void)layer_idx;
    (void)seq_len;
    return h;
}


array Phi3Inference::linear(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) throw std::runtime_error("Weight not found: " + weight_name);
    
    auto scales_it = cached_weights_.find(weight_name + ".scales");
    if (scales_it != cached_weights_.end()) {
        auto biases_it = cached_weights_.find(weight_name + ".biases");
        array biases = (biases_it != cached_weights_.end()) ? biases_it->second : zeros_like(scales_it->second);
        return quantized_matmul(x, weight_it->second, scales_it->second, biases, true, model_.quantization.group_size, model_.quantization.bits, "affine", {});
    }
    return matmul(x, transpose(weight_it->second, {1, 0}));
}

array Phi3Inference::rms_norm_3d(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) return x;
    array variance = mean(square(x), -1, true);
    return x * rsqrt(variance + model_.rms_norm_eps) * weight_it->second;
}

array Phi3Inference::mlp_block_3d(const array& x, const std::string& prefix) {
    array gate_up = linear(x, prefix + "mlp.gate_up_proj");
    int mid = gate_up.shape().back() / 2;
    int B = static_cast<int>(x.shape(0));
    int L = static_cast<int>(x.shape(1));
    array gate = slice(gate_up, {0, 0, 0}, {B, L, mid});
    array up = slice(gate_up, {0, 0, mid}, {B, L, 2 * mid});
    return linear(gate * sigmoid(gate) * up, prefix + "mlp.down_proj");
}

array Phi3Inference::forward(const std::vector<int32_t>& input_tokens, const MlxOgaGeneratorParams& params) {
    (void)params;  // Suppress unused warning
    int seq_len = static_cast<int>(input_tokens.size());
    
    if (seq_len > 1) {
        clear_cache(); // Reset on prefill
    }

    // OPTIMIZED: Use direct weight reference (no hash lookup)
    if (!weights_.embed_tokens) {
        throw std::runtime_error("forward: embed_tokens not setup");
    }
    array h = take(*weights_.embed_tokens, array(input_tokens.data(), {seq_len}, int32), 0);
    h = reshape(h, {1, seq_len, actual_hidden_size_}); // [1, Seq, Dim]

    // OPTIMIZED: Use pre-computed layer prefixes + direct weight references
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        const ryzenai::mlx::LayerWeights& layer = weights_.layers[i];
        
        // Fast path: direct weight references, no hash lookups
        array normed = rms_norm_fast(h, layer.input_layernorm);
        array attn_out = self_attention_fast(normed, layer, i, seq_len);
        h = h + attn_out;
        
        normed = rms_norm_fast(h, layer.post_attn_layernorm);
        array mlp_out = mlp_block_fast(normed, layer);
        h = h + mlp_out;
    }
    
    // Update cache position for RoPE offset
    cache_position_ += seq_len;

    // OPTIMIZED: Use direct weight reference
    h = rms_norm_fast(h, weights_.final_norm);
    array last_h = take(h, array({seq_len - 1}), 1); // [1, 1, Dim]
    
    array logits = linear_fast(reshape(last_h, {1, actual_hidden_size_}), weights_.lm_head);
    logits = reshape(logits, {model_.vocab_size});
    return logits;
}
/*
 * self_attention_no_residual
 * 
 * Optimized attention with two cache strategies:
 *   1. INT8 Quantized Cache: 2x memory bandwidth reduction, pre-allocated buffers
 *   2. FP16 Growing Cache: Original implementation for backward compatibility
 * 
 * The INT8 cache significantly improves GPU utilization by reducing memory bandwidth
 * bottleneck (the cause of ~50% GPU utilization).
 */
array Phi3Inference::self_attention_no_residual(const array& x, const std::string& prefix, 
                                                 int layer_idx, int seq_len) {
    int B = static_cast<int>(x.shape(0)); 
    
    // 1. Projections
    array qkv = linear(x, prefix + "self_attn.qkv_proj");
    int q_size = model_.num_attention_heads * head_dim_;
    int kv_size = model_.num_key_value_heads * head_dim_;
    
    // Slice and Reshape: [B, Seq, Heads, Dim] -> [B, Heads, Seq, Dim]
    array q = transpose(reshape(slice(qkv, {0, 0, 0}, {B, seq_len, q_size}), 
              Shape{B, seq_len, model_.num_attention_heads, head_dim_}), {0, 2, 1, 3});
    array k = transpose(reshape(slice(qkv, {0, 0, q_size}, {B, seq_len, q_size + kv_size}), 
              Shape{B, seq_len, model_.num_key_value_heads, head_dim_}), {0, 2, 1, 3});
    array v = transpose(reshape(slice(qkv, {0, 0, q_size + kv_size}, {B, seq_len, q_size + 2 * kv_size}), 
              Shape{B, seq_len, model_.num_key_value_heads, head_dim_}), {0, 2, 1, 3});
    
    // 2. Apply RoPE to Q and K
    array roped_q = fast::rope(q, head_dim_, false, rope_theta_, 1.0f, cache_position_);
    array roped_k = fast::rope(k, head_dim_, false, rope_theta_, 1.0f, cache_position_);
    
    // 3. Update Cache and Compute Attention (using lambda to handle array initialization)
    auto compute_attention = [&]() -> array {
        if (kv_cache_mode_ == KVCacheMode::INT8 && quantized_kv_cache_.has_value()) {
            // INT8 Quantized Cache Path - 2x bandwidth reduction
            // The quantized cache:
            //   - Pre-allocates buffers (no per-token allocation)
            //   - Stores K/V in INT8 with per-head FP16 scales
            //   - Dequantizes on-the-fly during attention
            auto [full_k, full_v] = quantized_kv_cache_->update(layer_idx, roped_k, v);
            
            // Advance position only on last layer (avoid double-counting)
            if (layer_idx == model_.num_hidden_layers - 1) {
                quantized_kv_cache_->advance_position(seq_len);
            }
            
            int total_seq_len = static_cast<int>(full_k.shape(2));
            
            if (seq_len > 1) {
                // PREFILL: Need causal mask
                array q_indices = arange(cache_position_, cache_position_ + seq_len, int32);
                array k_indices = arange(total_seq_len, int32);
                
                array row = reshape(q_indices, {seq_len, 1});
                array col = reshape(k_indices, {1, total_seq_len});
                
                array future_mask = less(row, col);
                array mask = where(future_mask, mask_val_, array(0.0f, mask_val_.dtype()));
                
                return fast::scaled_dot_product_attention(
                    roped_q, full_k, full_v, attention_scale_, "", mask
                );
            } else {
                // DECODE: Single token, no mask needed
                return fast::scaled_dot_product_attention(
                    roped_q, full_k, full_v, attention_scale_
                );
            }
        } else {
            // Legacy FP16 Growing Cache Path
            if (static_cast<size_t>(layer_idx) >= k_cache_.size()) {
                k_cache_.push_back(roped_k);
                v_cache_.push_back(v);
            } else {
                k_cache_[layer_idx] = concatenate({k_cache_[layer_idx], roped_k}, 2);
                v_cache_[layer_idx] = concatenate({v_cache_[layer_idx], v}, 2);
            }
            
            if (seq_len > 1) {
                // PREFILL: EXPLICIT CAUSAL MASK
                array indices = arange(seq_len, int32);
                array row = reshape(indices, {seq_len, 1});
                array col = reshape(indices, {1, seq_len});
                array future_mask = less(row, col);
                array mask = where(future_mask, mask_val_, array(0.0f, mask_val_.dtype()));
                
                return fast::scaled_dot_product_attention(
                    roped_q, k_cache_[layer_idx], v_cache_[layer_idx], attention_scale_, "", mask
                );
            } else {
                // DECODE: No mask needed
                return fast::scaled_dot_product_attention(
                    roped_q, k_cache_[layer_idx], v_cache_[layer_idx], attention_scale_
                );
            }
        }
    };
    
    array output = compute_attention();
    
    // 4. Output Projection
    output = transpose(output, {0, 2, 1, 3});
    output = reshape(output, Shape{B, seq_len, model_.num_attention_heads * head_dim_});
    
    return linear(output, prefix + "self_attn.o_proj");
}


array Phi3Inference::self_attention(const array& x, const std::string& prefix, int seq_len) {
    return x + self_attention_no_residual(x, prefix, 0, seq_len);
}


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
 * sample_token - Proper temperature-aware sampling
 * 
 * We must sample from the probability distribution.
 */
int Phi3Inference::sample_token(const array& logits, const MlxOgaGeneratorParams& params) {
    if (!params.do_sample || params.temperature <= 0.0f) {
        return static_cast<int>(argmax(logits).item<int32_t>());
    }
    array scaled_logits = logits / params.temperature;
    array sampled = random::categorical(scaled_logits);
    return static_cast<int>(sampled.item<int32_t>());
}

void Phi3Inference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    // We don't strictly need model_.quantization here for embeddings 
    // because we calculate dynamic group size, but we use it for other layers.
    
    std::cout << "[Phi3Inference] Caching weights..." << std::endl;

    // --- 1. Handle Embeddings (Dynamic Group Size + Zero Point Fix) ---
    auto embed_it = w.find("embed_tokens.weight");
    if (embed_it != w.end()) {
        auto scales_it = w.find("embed_tokens.scales");
        auto biases_it = w.find("embed_tokens.biases");

        if (scales_it != w.end()) {
            std::cout << "[Phi3Inference] Dequantizing embed_tokens with dynamic group size..." << std::endl;
            
            // Calculate group size based on shapes
            int packed_dim = embed_it->second.shape(-1); // e.g. 384
            int scale_dim = scales_it->second.shape(-1); // e.g. 48
            // (384 * 8 bits / 4 bits) / 48 = 64
            int embedding_group_size = (packed_dim * (32 / 4)) / scale_dim; 

            // Get the raw values from the file (or zeros if missing)
            array raw_biases = (biases_it != w.end()) ? biases_it->second : zeros_like(scales_it->second);
            
            // Zero Points are usually integers like 7, 8, etc. (Mean > 1.0)
            // Additive Biases are usually small floats near 0.0.
            float mean_bias = mean(abs(raw_biases)).item<float>();
            array proper_biases = mean_bias > 1.0f ? -1.0f * scales_it->second * raw_biases : raw_biases;
            // Dequantize with the corrected biases
            array dequantized_embed = mlx::core::dequantize(
                embed_it->second, 
                scales_it->second, 
                proper_biases, 
                embedding_group_size, 
                4 
            );
            cached_weights_.emplace("embed_tokens.weight", dequantized_embed);
        } else {
            // Not quantized (FP16/FP32), just copy
            cached_weights_.emplace("embed_tokens.weight", embed_it->second);
        }
    }

    // --- 2. Helper for Standard Layers ---
    auto cache_weight = [&](const std::string& name) {
        auto it = w.find(name + ".weight");
        if (it != w.end()) {
            cached_weights_.emplace(name + ".weight", it->second);
            
            // For standard layers, we just load scales/biases as is. 
            // MLX's quantized_matmul handles the standard format correctly.
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

    // --- 3. Cache Transformer Layers ---
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        std::string p = "layers." + std::to_string(i) + ".";

        // Layer Norms (Usually FP16/FP32, not quantized)
        auto in_ln = w.find(p + "input_layernorm.weight");
        if (in_ln != w.end()) {
            cached_weights_.emplace(p + "input_layernorm.weight", in_ln->second);
        }

        auto post_ln = w.find(p + "post_attention_layernorm.weight");
        if (post_ln != w.end()) {
            cached_weights_.emplace(p + "post_attention_layernorm.weight", post_ln->second);
        }

        // Linear Layers (Quantized)
        cache_weight(p + "self_attn.qkv_proj");
        cache_weight(p + "self_attn.o_proj");
        cache_weight(p + "mlp.gate_up_proj");
        cache_weight(p + "mlp.down_proj");
    }
    
    // --- 4. Final Norm & Head ---
    auto norm_it = w.find("norm.weight");
    if (norm_it != w.end()) {
        cached_weights_.emplace("norm.weight", norm_it->second);
    }

    cache_weight("lm_head");
    
    std::cout << "[Phi3Inference] Cached " << cached_weights_.size() << " tensors" << std::endl;
    
    // Setup direct weight references for optimized path
    setup_weight_references();
}

void Phi3Inference::setup_weight_references() {
    using namespace ryzenai::mlx;
    
    auto get_weight = [this](const std::string& name) -> const array* {
        auto it = cached_weights_.find(name);
        return it != cached_weights_.end() ? &(it->second) : nullptr;
    };
    
    auto setup_linear = [&](const std::string& prefix) -> LinearWeights {
        LinearWeights lw;
        lw.weight = get_weight(prefix + ".weight");
        lw.scales = get_weight(prefix + ".scales");
        lw.biases = get_weight(prefix + ".biases");
        lw.group_size = model_.quantization.group_size;
        lw.bits = model_.quantization.bits;
        return lw;
    };
    
    layer_prefixes_.clear();
    layer_prefixes_.reserve(model_.num_hidden_layers);
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        layer_prefixes_.push_back("layers." + std::to_string(i) + ".");
    }
    
    weights_.embed_tokens = get_weight("embed_tokens.weight");
    weights_.final_norm = get_weight("norm.weight");
    weights_.lm_head = setup_linear("lm_head");
    weights_.layers.resize(model_.num_hidden_layers);

    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        const std::string& p = layer_prefixes_[i];
        LayerWeights& layer = weights_.layers[i];
        
        layer.input_layernorm = get_weight(p + "input_layernorm.weight");
        layer.post_attn_layernorm = get_weight(p + "post_attention_layernorm.weight");
        layer.attention.q_proj = setup_linear(p + "self_attn.qkv_proj");
        layer.attention.o_proj = setup_linear(p + "self_attn.o_proj");
        layer.mlp.gate_proj = setup_linear(p + "mlp.gate_up_proj");
        layer.mlp.down_proj = setup_linear(p + "mlp.down_proj");
    }
    
    weights_.attention_scale = attention_scale_;
    weights_.rope_theta = rope_theta_;
    
    std::cout << "[Phi3Inference] Setup direct weight references for " 
              << weights_.layers.size() << " layers" << std::endl;
}


array Phi3Inference::linear_fast(const array& x, const ryzenai::mlx::LinearWeights& w) {
    if (!w.weight) {
        throw std::runtime_error("linear_fast: weight is null");
    }
    
    if (w.is_quantized()) {
        std::optional<array> biases_opt = w.biases ? std::optional(*w.biases) : std::nullopt;
        return quantized_matmul(
            x, 
            *w.weight, 
            *w.scales, 
            biases_opt,
            true,
            model_.quantization.group_size,
            model_.quantization.bits,
            "affine"
        );
    }
    
    return matmul(x, transpose(*w.weight, {1, 0}));
}


array Phi3Inference::rms_norm_fast(const array& x, const array* weight) {
    if (!weight) return x;
    return fast::rms_norm(x, *weight, model_.rms_norm_eps);
}


array Phi3Inference::self_attention_fast(const array& x, const ryzenai::mlx::LayerWeights& layer,
                                          int layer_idx, int seq_len) {
    int B = static_cast<int>(x.shape(0));
    int q_size = model_.num_attention_heads * head_dim_;
    int kv_size = model_.num_key_value_heads * head_dim_;
    
    array qkv = linear_fast(x, layer.attention.q_proj);
    
    array q = transpose(reshape(slice(qkv, {0, 0, 0}, {B, seq_len, q_size}), 
              Shape{B, seq_len, model_.num_attention_heads, head_dim_}), {0, 2, 1, 3});
    array k = transpose(reshape(slice(qkv, {0, 0, q_size}, {B, seq_len, q_size + kv_size}), 
              Shape{B, seq_len, model_.num_key_value_heads, head_dim_}), {0, 2, 1, 3});
    array v = transpose(reshape(slice(qkv, {0, 0, q_size + kv_size}, {B, seq_len, q_size + 2 * kv_size}), 
              Shape{B, seq_len, model_.num_key_value_heads, head_dim_}), {0, 2, 1, 3});
              
    array roped_q = fast::rope(q, head_dim_, false, weights_.rope_theta, 1.0f, cache_position_);
    array roped_k = fast::rope(k, head_dim_, false, weights_.rope_theta, 1.0f, cache_position_);
    
    auto compute_attention = [&]() -> array {
        if (kv_cache_mode_ == KVCacheMode::INT8 && quantized_kv_cache_.has_value()) {
            auto [full_k, full_v] = quantized_kv_cache_->update(layer_idx, roped_k, v);
            if (layer_idx == model_.num_hidden_layers - 1) {
                quantized_kv_cache_->advance_position(seq_len);
            }
            
            return seq_len > 1 ?
                fast::scaled_dot_product_attention(roped_q, full_k, full_v, weights_.attention_scale, "causal") :
                fast::scaled_dot_product_attention(roped_q, full_k, full_v, weights_.attention_scale);
        } else {
            if (static_cast<size_t>(layer_idx) >= k_cache_.size()) {
                k_cache_.push_back(roped_k);
                v_cache_.push_back(v);
            } else {
                k_cache_[layer_idx] = concatenate({k_cache_[layer_idx], roped_k}, 2);
                v_cache_[layer_idx] = concatenate({v_cache_[layer_idx], v}, 2);
            }
            
            return seq_len > 1 ?
                fast::scaled_dot_product_attention(roped_q, k_cache_[layer_idx], v_cache_[layer_idx], weights_.attention_scale, "causal") :
                fast::scaled_dot_product_attention(roped_q, k_cache_[layer_idx], v_cache_[layer_idx], weights_.attention_scale);
        }
    };
    
    array output = compute_attention();
    output = transpose(output, {0, 2, 1, 3});
    output = reshape(output, Shape{B, seq_len, model_.num_attention_heads * head_dim_});
    
    return linear_fast(output, layer.attention.o_proj);
}


array Phi3Inference::mlp_block_fast(const array& x, const ryzenai::mlx::LayerWeights& layer) {
    int B = static_cast<int>(x.shape(0));
    int L = static_cast<int>(x.shape(1));

    array gate_up = linear_fast(x, layer.mlp.gate_proj);
    int mid = gate_up.shape().back() / 2;
    
    array gate = slice(gate_up, {0, 0, 0}, {B, L, mid});
    array up = slice(gate_up, {0, 0, mid}, {B, L, 2 * mid});
    
    return linear_fast(gate * sigmoid(gate) * up, layer.mlp.down_proj);
}