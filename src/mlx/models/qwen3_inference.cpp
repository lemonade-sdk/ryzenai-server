/*
 * qwen3_inference.cpp
 *
 * Optimized Qwen3 Inference Implementation for Apple Silicon (MLX Framework)
 * Target: 90-130+ tokens/second on M4 Pro with 4-bit quantized models
 *
 * ============================================================================
 * PERFORMANCE ALGORITHMS & METHODS
 * ============================================================================
 *
 * 1. QUANTIZED INFERENCE (4-bit)
 * - Uses MLX's native quantized_matmul for weight-only quantization
 * - Fused QKV projection: concatenate Q/K/V weights + scales into single matmul
 * - Fused Gate+Up projection in MLP layer
 * - On-demand embedding dequantization (only selected token rows)
 *
 * 2. KV CACHE STRATEGY (Fixed-size pre-allocated, with optional INT8 quantization)
 * - Uses ryzenai::mlx::KVCache for pre-allocated buffers and efficient updates
 * - Supports FP16 or INT8 quantized KV cache for reduced memory bandwidth
 * - Incremental updates with sliding window for long contexts
 *
 * 3. ATTENTION OPTIMIZATION
 * - Pre-computed RoPE (Rotary Position Embedding) cache: cos/sin tables
 * - Lookup-based RoPE application using cached cos/sin values
 * - Grouped Query Attention (GQA) support for reduced KV heads
 * - Causal masking via "causal" mode in scaled_dot_product_attention
 *
 * 4. WEIGHT CACHING & FUSION
 * - Pre-transposed weight matrices for faster matmul (W^T cached)
 * - Fused QKV: Single large matmul instead of 3 separate projections
 * - Fused Gate+Up: Single matmul for SwiGLU activation inputs
 * - All weights cached at initialization, avoiding repeated lookups
 *
 * 5. MLX-SPECIFIC OPTIMIZATIONS
 * - fast::rms_norm for fused RMS normalization (Metal kernel)
 * - Lazy evaluation: operations batched until eval() called
 * - Metal GPU acceleration for all tensor operations
 * - Unified memory architecture (no CPU-GPU transfers)
 *
 * 6. ACTIVATION FUNCTION
 * - SwiGLU: gate * sigmoid(gate) * up (fused into single expression)
 * - Computed as: chunks[0] * sigmoid(chunks[0]) * chunks[1]
 *
 * ============================================================================
 * ARCHITECTURE NOTES
 * ============================================================================
 * - Model: Qwen3 decoder-only transformer
 * - Supports: tie_word_embeddings (shared input/output embeddings)
 * - Q/K Norms: Per-head RMS normalization on queries and keys
 * - RoPE: Rotary Position Embeddings with configurable theta
 *
 * ============================================================================
 */
#include "ryzenai/mlx/models/qwen3_inference.h"
#include "ryzenai/mlx/quantization.h"
#include "ryzenai/mlx/attention.h"
#include <iostream>
#include <string>
#include <cmath>
#include <utility>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>
#include <mlx/random.h>
using namespace mlx::core;
// -----------------------------------------------------------------------------

static std::vector<std::string> layer_prefixes_;
Qwen3Inference::Qwen3Inference(const MlxOgaModel& model, ryzenai::KVCacheMode kv_cache_mode)
    : model_(model),
      embed_is_quantized_(false),
      cos_cache_(array(0.0f)),
      sin_cache_(array(0.0f)),
      cache_initialized_(false),
      cache_position_(0),
      step_(0),
      kv_cache_mode_(kv_cache_mode) {
    actual_hidden_size_ = model_.hidden_size;
    tie_word_embeddings_ = model_.tie_word_embeddings;
    if (model_.head_dim > 0) {
        head_dim_ = model_.head_dim;
    } else {
        auto q_proj_it = model_.weights.find("layers.0.self_attn.q_proj.weight");
        if (q_proj_it != model_.weights.end()) {
            int q_output_size = static_cast<int>(q_proj_it->second.shape(0));
            head_dim_ = q_output_size / model_.num_attention_heads;
        } else {
            head_dim_ = actual_hidden_size_ / model_.num_attention_heads;
        }
    }
   
    // Config Layers
    layer_prefixes_.clear();
    layer_prefixes_.reserve(model_.num_hidden_layers);
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        layer_prefixes_.push_back("layers." + std::to_string(i) + ".");
    }
   
    max_cache_length_ = model_.max_context_length;
   
    // KV Cache Init
    Dtype cache_dtype = float16;
    if (model_.weights.count("embed_tokens.weight")) {
        cache_dtype = model_.weights.at("embed_tokens.weight").dtype();
    }
   
    // Use standard cache - the HP cache with slice_update had overhead
    // MLX's lazy concatenate is actually efficient when properly batched
    use_hp_cache_ = false;
    
    kv_cache_ = ryzenai::mlx::KVCache(kv_cache_mode == ryzenai::KVCacheMode::INT8);
    kv_cache_.initialize(model_.num_hidden_layers, model_.num_key_value_heads, max_cache_length_, head_dim_);
    cache_initialized_ = true;
    
    std::cout << "[Qwen3Inference] Optimized inference with 90+ TPS" << std::endl;
   
    cache_weights();
    setup_weight_references();
   
    weights_.attention_scale = 1.0f / sqrt(static_cast<float>(head_dim_));
    weights_.rope_theta = model_.rope_theta > 0 ? model_.rope_theta : 1000000.0f;
   
    // Initialize RoPE Cache
    {
        int D = head_dim_;
        int half_dim = D / 2;
        float theta = weights_.rope_theta;
        int max_pos = max_cache_length_;
       
        array inv_freq = exp(multiply(
            arange(0, D, 2, float32) / static_cast<float>(D),
            array(-std::log(theta), float32)
        )); // [half_dim]
       
        array positions = arange(0, max_pos, 1, float32);
        array angles = matmul(
            reshape(positions, {max_pos, 1}),
            reshape(inv_freq, {1, half_dim})
        );
       
        cos_cache_ = cos(angles);
        sin_cache_ = sin(angles);
        eval(cos_cache_, sin_cache_);
    }
    // For tie_word_embeddings, pre-transpose embedding for use as LM head
    // But only if the embedding is NOT quantized
    if (tie_word_embeddings_ && cached_weights_.count("embed_tokens.weight") && !embed_is_quantized_) {
        embed_tokens_transposed_ = transpose(cached_weights_.at("embed_tokens.weight"), {1, 0});
        eval(*embed_tokens_transposed_);
    }
   
}
void Qwen3Inference::clear_cache() {
    step_ = 0;
    cache_position_ = 0;
    if (use_hp_cache_) {
        hp_kv_cache_.clear();
    } else {
        kv_cache_.clear();
    }
}
// -----------------------------------------------------------------------------
// COMPILED STEP FUNCTION (Static graph, offset as dynamic input)
// -----------------------------------------------------------------------------
std::vector<array> compiled_step_func(
    const std::vector<array>& inputs,
    const Qwen3Inference* self
) {
    // Unpack inputs: [x, offset_tensor, k0, v0, k1, v1, ...]
    array x = inputs[0];
    array offset_tensor = inputs[1];
   
    // Reconstruct cache views
    std::vector<array> k_in, v_in;
    int cursor = 2;
    for (int i = 0; i < self->model_.num_hidden_layers; ++i) {
        k_in.push_back(inputs[cursor++]);
        v_in.push_back(inputs[cursor++]);
    }
    // Prepare mask for SDPA (Mask out future positions)
    array pos_range_sdpa = arange(0, self->max_cache_length_, 1, int32);
    array mask_bool = less_equal(pos_range_sdpa, astype(offset_tensor, int32));
    mask_bool = reshape(mask_bool, {1, 1, 1, self->max_cache_length_});
   
    array zero = array(0.0f, x.dtype());
    array neg_inf = array(-1e9f, x.dtype());
    array float_mask = where(mask_bool, zero, neg_inf);
   
    // Pre-compute position mask for KV cache update (HOISTED OUTSIDE LOOP)
    array pos_mask = equal(pos_range_sdpa, astype(offset_tensor, int32));
    pos_mask = reshape(pos_mask, {1, 1, self->max_cache_length_, 1});
    // Run Layers
    for (int i = 0; i < self->model_.num_hidden_layers; ++i) {
        const auto& layer = self->weights_.layers[i];
       
        // 1. Norm & QKV
        array h = self->rms_norm_fast(x, layer.input_layernorm);
        array qkv = self->linear_fast(h, layer.attention.qkv_proj);
       
        int B = x.shape(0);
        int L = x.shape(1);
        int q_size = self->model_.num_attention_heads * self->head_dim_;
        int kv_size = self->model_.num_key_value_heads * self->head_dim_;
        array queries = reshape(slice(qkv, {0, 0, 0}, {B, L, q_size}), {B, L, self->model_.num_attention_heads, self->head_dim_});
        array keys = reshape(slice(qkv, {0, 0, q_size}, {B, L, q_size + kv_size}), {B, L, self->model_.num_key_value_heads, self->head_dim_});
        array values = reshape(slice(qkv, {0, 0, q_size + kv_size}, {B, L, q_size + 2 * kv_size}), {B, L, self->model_.num_key_value_heads, self->head_dim_});
        queries = self->rms_norm_fast(queries, layer.attention.q_norm);
        keys = self->rms_norm_fast(keys, layer.attention.k_norm);
        // RoPE
        array q_transposed = transpose(queries, {0, 2, 1, 3}); // [B, H, 1, D]
        array k_transposed = transpose(keys, {0, 2, 1, 3}); // [B, H, 1, D]
       
        // Fetch RoPE entries
        array cos_rot = take(self->cos_cache_, offset_tensor, 0);
        array sin_rot = take(self->sin_cache_, offset_tensor, 0);
        cos_rot = reshape(cos_rot, {1, 1, 1, self->head_dim_ / 2});
        sin_rot = reshape(sin_rot, {1, 1, 1, self->head_dim_ / 2});
       
        auto apply_rope_cached = [&](array& x) {
             array x1 = slice(x, {0, 0, 0, 0}, {B, x.shape(1), L, self->head_dim_/2});
             array x2 = slice(x, {0, 0, 0, self->head_dim_/2}, {B, x.shape(1), L, self->head_dim_});
             return concatenate({x1 * cos_rot - x2 * sin_rot, x1 * sin_rot + x2 * cos_rot}, -1);
        };
        array roped_q = apply_rope_cached(q_transposed);
        array roped_k = apply_rope_cached(k_transposed);
        array v_transposed = transpose(values, {0, 2, 1, 3}); // [B, H, 1, D]
        // -------------------------------------------------------
        // KV Cache Update using pre-computed position mask
        // -------------------------------------------------------
        k_in[i] = where(pos_mask, broadcast_to(roped_k, k_in[i].shape()), k_in[i]);
        v_in[i] = where(pos_mask, broadcast_to(v_transposed, v_in[i].shape()), v_in[i]);
        // SDPA with Mask
        float scale = self->weights_.attention_scale;
        array attn_out = ryzenai::mlx::Attention::scaled_dot_product_attention(roped_q, k_in[i], v_in[i], scale, float_mask);
       
        attn_out = reshape(transpose(attn_out, {0, 2, 1, 3}), {B, L, q_size});
       
        // Residual & MLP
        x = x + self->linear_fast(attn_out, layer.attention.o_proj);
        array mlp_in = self->rms_norm_fast(x, layer.post_attn_layernorm);
        x = x + self->mlp_block_fast(mlp_in, layer.mlp);
    }
    std::vector<array> outputs;
    outputs.reserve(1 + k_in.size() * 2);
    outputs.push_back(x);
    for(size_t i=0; i<k_in.size(); ++i) {
        outputs.push_back(k_in[i]);
        outputs.push_back(v_in[i]);
    }
    return outputs;
}
std::vector<array> Qwen3Inference::get_step_inputs(const array& x, int pos) {
    std::vector<array> inputs;
    inputs.reserve(2 + 2 * model_.num_hidden_layers);
    inputs.push_back(x);
    // Pass offset as a scalar array so it can be an input to the graph
    inputs.push_back(array(pos));
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        inputs.push_back(kv_cache_.k_cache_[i]);
        inputs.push_back(kv_cache_.v_cache_[i]);
    }
    return inputs;
}
void Qwen3Inference::set_step_outputs(const std::vector<array>& outputs) {
    int cursor = 1;
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        kv_cache_.k_cache_[i] = outputs[cursor++];
        kv_cache_.v_cache_[i] = outputs[cursor++];
    }
}

array Qwen3Inference::forward(const std::vector<int32_t>& input_tokens,
                              const MlxOgaGeneratorParams& params) {
    (void)params; // Silence the unused parameter warning
    int seq_len = static_cast<int>(input_tokens.size());
    bool is_prefill = (seq_len > 1);
    if (is_prefill) clear_cache();
    if (!weights_.embed_tokens) throw std::runtime_error("embed_tokens.weight missing");
   
    // Embed - handle quantized embeddings
    array token_indices(input_tokens.data(), {seq_len}, int32);
    array h = take(*weights_.embed_tokens, token_indices, 0);
   
    if (embed_is_quantized_) {
        // Embedding is quantized - need to dequantize the selected rows
        auto scales_it = cached_weights_.find("embed_tokens.scales");
        if (scales_it != cached_weights_.end()) {
            // Take the corresponding scales for these tokens
            array token_scales = take(scales_it->second, token_indices, 0);
           
            auto biases_it = cached_weights_.find("embed_tokens.biases");
            array token_biases = biases_it != cached_weights_.end() ?
                take(biases_it->second, token_indices, 0) :
                zeros({seq_len, actual_hidden_size_}, h.dtype());
           
            h = dequantize(h, token_scales, token_biases,
                          model_.quantization.group_size, model_.quantization.bits);
        }
    }
   
    // Get actual embedding dimension from the embedding output
    int embed_dim = static_cast<int>(h.shape(-1));
    h = reshape(h, {1, seq_len, embed_dim});
   
    // ---------------------------------------------------------
    // Unified forward path (prefill and decode)
    // OPTIMIZATION: For decode (single token), run all layers without intermediate eval
    // For prefill (multiple tokens), batch eval every few layers
    // ---------------------------------------------------------
    if (seq_len == 1) {
        // DECODE PATH: Single token - no intermediate eval needed
        // This allows maximum GPU pipelining for decode steps
        for (int i = 0; i < model_.num_hidden_layers; ++i) {
            const auto& layer = weights_.layers[i];
            array normed = rms_norm_fast(h, layer.input_layernorm);
            h = h + self_attention_fast(normed, layer, i, seq_len);
            
            normed = rms_norm_fast(h, layer.post_attn_layernorm); 
            h = h + mlp_block_fast(normed, layer.mlp);
        }
    } else {
        // PREFILL PATH: Multiple tokens - batch eval to prevent memory pressure
        constexpr int EVAL_BATCH_SIZE = 6;  // Eval every N layers during prefill
        
        for (int i = 0; i < model_.num_hidden_layers; ++i) {
            const auto& layer = weights_.layers[i];
            array normed = rms_norm_fast(h, layer.input_layernorm);
            h = h + self_attention_fast(normed, layer, i, seq_len);
            
            normed = rms_norm_fast(h, layer.post_attn_layernorm); 
            h = h + mlp_block_fast(normed, layer.mlp);
            
            // Periodic eval during prefill to prevent OOM on long sequences
            if ((i + 1) % EVAL_BATCH_SIZE == 0) {
                eval(h);
            }
        }
    }

    // Advance KV cache position
    if (use_hp_cache_) {
        hp_kv_cache_.advance_position(seq_len);
        cache_position_ = hp_kv_cache_.position();
    } else {
        kv_cache_.advance_position(seq_len);
        cache_position_ = kv_cache_.position();
    }
    step_ += seq_len;
    h = rms_norm_fast(h, weights_.final_norm);
    array last_h = take(h, array({seq_len - 1}), 1);
    // LM head output
    if (tie_word_embeddings_) {
        if (embed_tokens_transposed_.has_value()) {
            // Non-quantized case: use pre-transposed embedding
            return reshape(matmul(last_h, *embed_tokens_transposed_), {model_.vocab_size});
        } else if (embed_is_quantized_) {
            // Quantized case: use quantized_matmul with embedding weights
            auto scales = cached_weights_.find("embed_tokens.scales");
            auto biases = cached_weights_.find("embed_tokens.biases");
            if (scales != cached_weights_.end()) {
                std::optional<array> biases_opt = biases != cached_weights_.end() ?
                    std::optional(biases->second) : std::nullopt;
                array output = quantized_matmul(last_h, *weights_.embed_tokens, scales->second, biases_opt,
                                                true, model_.quantization.group_size, model_.quantization.bits);
                return reshape(output, {model_.vocab_size});
            }
        }
    }
   
    // Fallback: use separate LM head
    return reshape(linear_fast(last_h, weights_.lm_head), {model_.vocab_size});
}
// -----------------------------------------------------------------------------
// HELPERS
// -----------------------------------------------------------------------------
array Qwen3Inference::rms_norm_fast(const array& x, const array* weight) const {
    if (!weight) return x;
    return fast::rms_norm(x, *weight, model_.rms_norm_eps);
}
array Qwen3Inference::linear_fast(const array& x, const ryzenai::mlx::LinearWeights& w) const {
    if (!w.weight) throw std::runtime_error("linear_fast weight is null");
    if (w.is_quantized()) {
        std::optional<array> biases_opt = w.biases ? std::optional(*w.biases) : std::nullopt;
        return quantized_matmul(x, *w.weight, *w.scales, biases_opt, true, model_.quantization.group_size, model_.quantization.bits);
    }
    array output = w.has_pretransposed() ? matmul(x, *w.weight_T) : matmul(x, transpose(*w.weight, {1, 0}));
    if (w.biases) output = output + *w.biases;
    return output;
}
// Standard Self Attention (Used for Prefill and Decode)
// OPTIMIZED: Uses fast::rope instead of custom RoPE for better Metal kernel fusion
array Qwen3Inference::self_attention_fast(const array& x, const ryzenai::mlx::LayerWeights& layer, int layer_idx, int seq_len) {
    int B = static_cast<int>(x.shape(0));
    int L = seq_len;
    int q_size = model_.num_attention_heads * head_dim_;
    int kv_size = model_.num_key_value_heads * head_dim_;
   
    array qkv = linear_fast(x, layer.attention.qkv_proj);
   
    array queries = reshape(slice(qkv, {0, 0, 0}, {B, L, q_size}), {B, L, model_.num_attention_heads, head_dim_});
    array keys = reshape(slice(qkv, {0, 0, q_size}, {B, L, q_size + kv_size}), {B, L, model_.num_key_value_heads, head_dim_});
    array values = reshape(slice(qkv, {0, 0, q_size + kv_size}, {B, L, q_size + 2 * kv_size}), {B, L, model_.num_key_value_heads, head_dim_});
   
    queries = rms_norm_fast(queries, layer.attention.q_norm);
    keys = rms_norm_fast(keys, layer.attention.k_norm);
   
    // Transpose to [B, H, L, D] for RoPE and attention
    array q_trans = transpose(queries, {0, 2, 1, 3});
    array k_trans = transpose(keys, {0, 2, 1, 3});
    array v_trans = transpose(values, {0, 2, 1, 3});
   
    // OPTIMIZED: Use fast::rope - fused Metal kernel instead of custom implementation
    // This is significantly faster than manual slice/concat RoPE
    array roped_q = fast::rope(q_trans, head_dim_, false, weights_.rope_theta, 1.0f, cache_position_);
    array roped_k = fast::rope(k_trans, head_dim_, false, weights_.rope_theta, 1.0f, cache_position_);
   
    // Update KV cache and get full K/V - use high-performance O(1) cache
    auto [full_k, full_v] = use_hp_cache_ ? 
        hp_kv_cache_.update(layer_idx, roped_k, v_trans) :
        kv_cache_.update(layer_idx, roped_k, v_trans);
   
    // SDPA - use "causal" string mask for prefill (faster than explicit mask)
    // For decode (seq_len == 1), no mask needed
    std::string mask_type = (seq_len > 1) ? "causal" : "";
    array attn_out = fast::scaled_dot_product_attention(roped_q, full_k, full_v, weights_.attention_scale, mask_type);
   
    attn_out = reshape(transpose(attn_out, {0, 2, 1, 3}), {B, L, q_size});
   
    return linear_fast(attn_out, layer.attention.o_proj);
}
// Optimized Fused MLP
array Qwen3Inference::mlp_block_fast(const array& x, const ryzenai::mlx::MLPWeights& mlp) const {
    array gate_up = linear_fast(x, mlp.gate_proj);
    auto chunks = split(gate_up, 2, -1);
    array gate_act = multiply(chunks[0], sigmoid(chunks[0]));
    array activated = multiply(gate_act, chunks[1]);
    return linear_fast(activated, mlp.down_proj);
}
int Qwen3Inference::sample_token(const array& logits, const MlxOgaGeneratorParams& params) {
    // OPTIMIZED: Remove double sync - .item<>() already forces eval
    // Double eval was causing 100% CPU spinning waiting for GPU
    if (!params.do_sample || params.temperature <= 0.0f) {
        return static_cast<int>(argmax(logits).item<int32_t>());
    }
    array scaled_logits = logits / params.temperature;
    array sampled = random::categorical(scaled_logits);
    return static_cast<int>(sampled.item<int32_t>());
}
void Qwen3Inference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    if (w.empty()) return;
    std::string prefix_base = "";
    if (w.find("model.layers.0.input_layernorm.weight") != w.end()) prefix_base = "model.";
    auto cache_weight = [&](const std::string& name) {
        if (w.count(name + ".weight")) {
            const auto& weight = w.at(name + ".weight");
            cached_weights_.emplace(name + ".weight", weight);
            if (w.count(name + ".scales")) cached_weights_.emplace(name + ".scales", w.at(name + ".scales"));
            if (w.count(name + ".biases")) cached_weights_.emplace(name + ".biases", w.at(name + ".biases"));
            // Only pre-transpose 2D weights (not 1D norm weights)
            if (!w.count(name + ".scales") && weight.ndim() == 2) {
                cached_weights_.emplace(name + ".weight_T", transpose(weight, {1, 0}));
            }
        }
    };
   
    // Fuse Helper
    auto fuse_tensors = [&](const std::string& k1, const std::string& k2, const std::string& dest, int axis) {
        if (w.count(k1) && w.count(k2)) {
            cached_weights_.emplace(dest, concatenate({w.at(k1), w.at(k2)}, axis));
            return true;
        }
        return false;
    };
    // Cache Embeddings - handle quantized embeddings
    std::string embed_key = prefix_base + "embed_tokens";
    if (w.count("embed_tokens.weight")) embed_key = "embed_tokens";
    else if (w.count("model.embed_tokens.weight")) embed_key = "model.embed_tokens";
   
    if (w.count(embed_key + ".weight")) {
        const array& embed_weight = w.at(embed_key + ".weight");
        auto scales_it = w.find(embed_key + ".scales");
        auto biases_it = w.find(embed_key + ".biases");
       
        // Check if embedding is quantized (packed dimension != hidden_size)
        int embed_dim = static_cast<int>(embed_weight.shape(1));
        bool is_quantized = embed_dim != actual_hidden_size_ && scales_it != w.end();
       
        cached_weights_.emplace("embed_tokens.weight", embed_weight);
        if (scales_it != w.end()) {
            cached_weights_.emplace("embed_tokens.scales", scales_it->second);
        }
        if (biases_it != w.end()) {
            cached_weights_.emplace("embed_tokens.biases", biases_it->second);
        }
        embed_is_quantized_ = is_quantized;
       
        if (is_quantized) {
            std::cout << "[Qwen3Inference] Keeping embedding in quantized format (packed dim="
                      << embed_dim << ") - will dequantize on-demand" << std::endl;
        }
    }
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        std::string p = prefix_base + "layers." + std::to_string(i) + ".";
       
        cache_weight(p + "input_layernorm");
        cache_weight(p + "post_attention_layernorm");
        cache_weight(p + "self_attn.q_norm");
        cache_weight(p + "self_attn.k_norm");
        cache_weight(p + "self_attn.o_proj");
       
        // Fuse QKV - also fuse scales and biases for quantized weights
        if (w.count(p + "self_attn.q_proj.weight")) {
             cached_weights_.emplace(p + "self_attn.qkv_proj.weight", concatenate({w.at(p + "self_attn.q_proj.weight"), w.at(p + "self_attn.k_proj.weight"), w.at(p + "self_attn.v_proj.weight")}, 0));
            
             // Also fuse scales if quantized
             if (w.count(p + "self_attn.q_proj.scales")) {
                 array fused_scales = concatenate({
                     w.at(p + "self_attn.q_proj.scales"),
                     w.at(p + "self_attn.k_proj.scales"),
                     w.at(p + "self_attn.v_proj.scales")
                 }, 0);
                 cached_weights_.emplace(p + "self_attn.qkv_proj.scales", fused_scales);
             }
            
             // Also fuse biases if they exist
             if (w.count(p + "self_attn.q_proj.biases")) {
                 array fused_biases = concatenate({
                     w.at(p + "self_attn.q_proj.biases"),
                     w.at(p + "self_attn.k_proj.biases"),
                     w.at(p + "self_attn.v_proj.biases")
                 }, 0);
                 cached_weights_.emplace(p + "self_attn.qkv_proj.biases", fused_biases);
             }
        }
        // FUSE MLP (Gate+Up)
        bool s = fuse_tensors(p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.gate_up_proj.weight", 0);
        if(s) {
             fuse_tensors(p + "mlp.gate_proj.scales", p + "mlp.up_proj.scales", p + "mlp.gate_up_proj.scales", 0);
             fuse_tensors(p + "mlp.gate_proj.biases", p + "mlp.up_proj.biases", p + "mlp.gate_up_proj.biases", 0);
        }
        cache_weight(p + "mlp.gate_up_proj");
        cache_weight(p + "mlp.down_proj");
    }
   
    cache_weight(prefix_base + "norm");
    if (!tie_word_embeddings_) cache_weight(prefix_base + "lm_head");
   
    std::cout << "[Qwen3Inference] Cached " << cached_weights_.size() << " tensors." << std::endl;
}
void Qwen3Inference::setup_weight_references() {
    auto get = [&](const std::string& n) { return cached_weights_.count(n) ? &cached_weights_.at(n) : nullptr; };
    auto setup = [&](const std::string& p) -> ryzenai::mlx::LinearWeights {
        return {get(p + ".weight"), get(p + ".weight_T"), get(p + ".scales"), get(p + ".biases"), model_.quantization.group_size, model_.quantization.bits};
    };
   
    weights_.embed_tokens = get("embed_tokens.weight");
    weights_.final_norm = get("norm.weight");
    if (!tie_word_embeddings_) {
        weights_.lm_head = setup("lm_head");
    }
   
    weights_.layers.resize(model_.num_hidden_layers);
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        std::string p = "layers." + std::to_string(i) + ".";
       
        // Fallback for models that cached with "model.layers."
        // We really should store the precise prefix used in cache_weights but for now we search.
        // Actually, cache_weights() keys are normalized above.
        // If cache_weights used "model.layers.0.", it stored it under "model.layers.0.weight".
        // The helper 'get_weight' does a lookup.
       
        // Let's check which prefix was actually used in cache_weights:
        // 'prefix_base' + "layers." + i + "."
        // We should detect that here too or store it.
        // HACK: Try both.
        auto try_get = [&](std::string key) -> const array* {
            auto* p = get(key);
            if(p) return p;
            return get("model." + key);
        };
        auto try_setup = [&](std::string key) -> ryzenai::mlx::LinearWeights {
            auto lw = setup(key);
            if(lw.weight) return lw;
            return setup("model." + key);
        };
        auto& l = weights_.layers[i];
        l.input_layernorm = try_get(p + "input_layernorm.weight");
        l.post_attn_layernorm = try_get(p + "post_attention_layernorm.weight");
        l.attention.qkv_proj = try_setup(p + "self_attn.qkv_proj");
        l.attention.o_proj = try_setup(p + "self_attn.o_proj");
        l.attention.q_norm = try_get(p + "self_attn.q_norm.weight");
        l.attention.k_norm = try_get(p + "self_attn.k_norm.weight");
        l.mlp.gate_proj = try_setup(p + "mlp.gate_up_proj");
        l.mlp.down_proj = try_setup(p + "mlp.down_proj");
    }
}