/*
 * qwen3_inference.cpp
 * 
 * Qwen3 transformer inference implementation with KV caching.
 * Closely follows the Python mlx-lm/models/qwen3.py implementation.
 * 
 * Features:
 *   - Q/K normalization before RoPE (applied on [B, L, n_heads, head_dim] shape)
 *   - Separate Q/K/V projections
 *   - KV cache for efficient autoregressive generation
 *   - Uses mx.fast.scaled_dot_product_attention with proper causal masking
 *   - Proper RoPE position tracking with offset
 */

#include "ryzenai/mlx/models/qwen3_inference.h"
#include "ryzenai/mlx/quantization.h"
#include "ryzenai/mlx/attention.h"
#include <iostream>
#include <string>
#include <utility>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>

using namespace mlx::core;

// Pre-computed layer prefixes to avoid string allocation in hot path
static std::vector<std::string> layer_prefixes_;


Qwen3Inference::Qwen3Inference(const OgaModel& model) 
    : model_(model), cache_initialized_(false), cache_position_(0), step_(0) {
    actual_hidden_size_ = model_.hidden_size;
    tie_word_embeddings_ = model_.tie_word_embeddings;

    // 1. Priority: Explicit config value (e.g., 128 for Qwen3-14B/32B)
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
    
    std::cout << "[Qwen3Inference] hidden_size=" << actual_hidden_size_
              << ", head_dim=" << head_dim_
              << ", num_heads=" << model_.num_attention_heads
              << ", num_kv_heads=" << model_.num_key_value_heads
              << ", tie_word_embeddings=" << (tie_word_embeddings_ ? "yes" : "no")
              << ", quantized=" << (model_.is_quantized() ? "yes" : "no")
              << std::endl;
    
    // Pre-compute layer prefixes to avoid string allocation in hot path
    layer_prefixes_.clear();
    layer_prefixes_.reserve(model_.num_hidden_layers);
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        layer_prefixes_.push_back("layers." + std::to_string(i) + ".");
    }
    
    // Use context length from model config (set from command line --ctx-size)
    max_cache_length_ = model_.max_context_length;
    
    // Determine dtype from first weight (typically bfloat16 for MLX models)
    Dtype cache_dtype = float16;
    auto first_weight = model_.weights.find("embed_tokens.weight");
    if (first_weight != model_.weights.end()) {
        cache_dtype = first_weight->second.dtype();
    }
    
    // Pre-allocate KV cache buffers (optimization #2)
    // Use reserve + push_back since MLX array has no default constructor
    // Shape: [1, num_kv_heads, max_cache_length, head_dim]
    std::cout << "[Qwen3Inference] Pre-allocating KV cache: " 
              << model_.num_hidden_layers << " layers x " 
              << max_cache_length_ << " tokens" << std::endl;
    
    k_cache_.clear();
    v_cache_.clear();
    k_cache_.reserve(model_.num_hidden_layers);
    v_cache_.reserve(model_.num_hidden_layers);
    
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        k_cache_.push_back(zeros({1, model_.num_key_value_heads, max_cache_length_, head_dim_}, cache_dtype));
        v_cache_.push_back(zeros({1, model_.num_key_value_heads, max_cache_length_, head_dim_}, cache_dtype));
    }
    eval(k_cache_);
    eval(v_cache_);
    cache_initialized_ = true;
    
    cache_weights();
    
    // Setup direct weight references (optimization #3 & #4)
    setup_weight_references();
    
    // Pre-compute attention scale (optimization #1)
    weights_.attention_scale = 1.0f / sqrt(static_cast<float>(head_dim_));
    weights_.rope_theta = model_.rope_theta > 0 ? model_.rope_theta : 1000000.0f;
    
    // Pre-transpose embed tokens for tie_word_embeddings (optimization #2)
    if (tie_word_embeddings_) {
        auto embed_it = cached_weights_.find("embed_tokens.weight");
        if (embed_it != cached_weights_.end()) {
            embed_tokens_transposed_ = transpose(embed_it->second, {1, 0});
            eval(*embed_tokens_transposed_);  // Force evaluation
            std::cout << "[Qwen3Inference] Pre-transposed embed_tokens for LM head" << std::endl;
        }
    }
    
    std::cout << "[Qwen3Inference] Optimizations enabled: scale=" << weights_.attention_scale 
              << ", rope_theta=" << weights_.rope_theta << std::endl;
}


void Qwen3Inference::clear_cache() {
    // Reset position counters only - don't deallocate pre-allocated buffers
    // This avoids memory allocation overhead between conversations
    step_ = 0;
    cache_position_ = 0;
}


array Qwen3Inference::linear(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        throw std::runtime_error("Weight not found: " + weight_name);
    }
    
    auto scales_it = cached_weights_.find(weight_name + ".scales");
    if (scales_it != cached_weights_.end()) {
        // Quantized linear: use mlx::core::quantized_matmul
        auto biases_it = cached_weights_.find(weight_name + ".biases");
        std::optional<array> biases_opt = std::nullopt;
        if (biases_it != cached_weights_.end()) {
            biases_opt = biases_it->second;
        }
        
        return quantized_matmul(
            x, 
            weight_it->second, 
            scales_it->second, 
            biases_opt,
            true,  // transpose
            model_.quantization.group_size,
            model_.quantization.bits,
            "affine"
        );
    }
    
    // Non-quantized linear
    return matmul(x, transpose(weight_it->second, {1, 0}));
}


/*
 * RMSNorm - applies RMS normalization over the last axis
 */
array Qwen3Inference::rms_norm(const array& x, const std::string& weight_name) {
    auto weight_it = cached_weights_.find(weight_name + ".weight");
    if (weight_it == cached_weights_.end()) {
        return x;
    }
    
    array variance = mean(square(x), -1, true);
    array inv_std = rsqrt(variance + model_.rms_norm_eps);
    return x * inv_std * weight_it->second;
}


/*
 * create_causal_mask - creates a causal attention mask
 * 
 * Returns boolean mask where true = attend (lind >= rind)
 * Shape: [seq_len, total_seq_len]
 */
array Qwen3Inference::create_causal_mask(int seq_len, int offset) {
    int total_seq_len = offset + seq_len;
    
    array linds = arange(offset, offset + seq_len, int32);
    array rinds = arange(total_seq_len, int32);
    
    linds = reshape(linds, {seq_len, 1});
    rinds = reshape(rinds, {1, total_seq_len});
    
    return greater_equal(linds, rinds);
}


/*
 * forward
 * 
 * Matches Python mlx-lm pattern:
 *   mask = create_attention_mask(h, cache)
 *   # where create_attention_mask returns:
 *   #   None if N==1 (single token generation)
 *   #   "causal" for efficient causal masking during prefill
 */
array Qwen3Inference::forward(const std::vector<int32_t>& input_tokens,
                              const OgaGeneratorParams& params) {
    int seq_len = static_cast<int>(input_tokens.size());
    bool is_prefill = (seq_len > 1);

    if (is_prefill) {
        clear_cache();
    }

    // Use direct embedding reference (optimization)
    if (!weights_.embed_tokens) {
        throw std::runtime_error("embed_tokens.weight not found");
    }
    
    // Token indices as 1D array
    array token_indices(input_tokens.data(), {seq_len}, int32);
    
    // Embedding lookup using direct reference → [seq_len, hidden]
    array h = take(*weights_.embed_tokens, token_indices, 0);
    
    // Add batch dim → [1, seq_len, hidden]
    h = reshape(h, {1, seq_len, actual_hidden_size_});
    
    // Determine mask type following Python create_attention_mask:
    // - seq_len == 1: no mask needed (None)
    // - seq_len > 1: use "causal" for efficient causal masking
    std::string mask_type = ryzenai::mlx::Attention::get_mask_type(seq_len);

    // Process layers using OPTIMIZED path with direct weight references
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        const ryzenai::mlx::LayerWeights& layer = weights_.layers[i];
        
        // Use fast norm/attention/mlp with direct references
        array normed = rms_norm_fast(h, layer.input_layernorm);
        array attn_out = self_attention_fast(normed, layer, i, mask_type);
        h = h + attn_out;
        
        normed = rms_norm_fast(h, layer.post_attn_layernorm);
        array mlp_out = mlp_block_fast(normed, layer.mlp);
        h = h + mlp_out;
    }

    // Update cache position after processing all layers
    cache_position_ += seq_len;
    step_ += seq_len;
    
    // Handle cache overflow with sliding window
    if (cache_position_ > max_cache_length_) {
        cache_position_ = max_cache_length_;
        step_ = max_cache_length_;
    }

    // Final norm using direct reference
    h = rms_norm_fast(h, weights_.final_norm);
    
    // Last token hidden state [1, 1, hidden]
    array last_h = take(h, array({seq_len - 1}), 1);

    // Logits with pre-transposed embed or LM head
    array logits = (tie_word_embeddings_ && embed_tokens_transposed_.has_value()) ? 
        reshape(matmul(last_h, *embed_tokens_transposed_), {model_.vocab_size}) :
        reshape(linear_fast(last_h, weights_.lm_head), {model_.vocab_size});
    return logits;
}


/*
 * self_attention
 * 
 * Matches Python mlx-lm attention pattern:
 *   mask = create_attention_mask(h, cache)  # returns None/"causal"/array
 *   output = scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
 */
array Qwen3Inference::self_attention(const array& x, const std::string& prefix, 
                                      int layer_idx, const std::string& mask_type) {
    int B = static_cast<int>(x.shape(0));
    int L = static_cast<int>(x.shape(1));
    
    // 1. Projections
    array queries = linear(x, prefix + "self_attn.q_proj");
    array keys = linear(x, prefix + "self_attn.k_proj");
    array values = linear(x, prefix + "self_attn.v_proj");
    
    queries = reshape(queries, {B, L, model_.num_attention_heads, head_dim_});
    keys = reshape(keys, {B, L, model_.num_key_value_heads, head_dim_});
    values = reshape(values, {B, L, model_.num_key_value_heads, head_dim_});
    
    // 2. Q/K Normalization (Specific to Qwen2.5/Qwen3)
    queries = rms_norm(queries, prefix + "self_attn.q_norm");
    keys = rms_norm(keys, prefix + "self_attn.k_norm");
    
    // 3. Transpose to [B, n_heads, L, head_dim] for cache and RoPE
    array q_transposed = transpose(queries, {0, 2, 1, 3});
    array k_transposed = transpose(keys, {0, 2, 1, 3});
    array v_transposed = transpose(values, {0, 2, 1, 3});
    
    // 4. Apply RoPE BEFORE caching (only to NEW tokens, not cached ones)
    // This avoids recomputing RoPE on the entire cache every step!
    int past_len = step_;
    float rope_base = model_.rope_theta > 0 ? model_.rope_theta : 1000000.0f;
    array roped_q = fast::rope(q_transposed, head_dim_, false, rope_base, 1.0f, past_len);
    array roped_k = fast::rope(k_transposed, head_dim_, false, rope_base, 1.0f, past_len);
    
    // 5. Concat RoPE'd keys/values to cache (keys already have correct positional encoding)
    array full_k = roped_k;
    array full_v = v_transposed;
    if (static_cast<size_t>(layer_idx) < k_cache_.size()) {
        full_k = concatenate({k_cache_[layer_idx], roped_k}, 2);
        full_v = concatenate({v_cache_[layer_idx], v_transposed}, 2);
        k_cache_[layer_idx] = full_k;
        v_cache_[layer_idx] = full_v;
    } else {
        k_cache_.push_back(full_k);
        v_cache_.push_back(full_v);
    }
    
    // 6. Attention using full cached K/V
    float scale = 1.0f / sqrt(static_cast<float>(head_dim_));
    
    array output = mask_type == "none" ? 
            fast::scaled_dot_product_attention(roped_q, full_k, full_v, scale) :
            fast::scaled_dot_product_attention(roped_q, full_k, full_v, scale, "causal");
    
    output = transpose(output, {0, 2, 1, 3});
    output = reshape(output, {B, L, model_.num_attention_heads * head_dim_});
    
    return linear(output, prefix + "self_attn.o_proj");
}


/*
 * mlp_block
 */
array Qwen3Inference::mlp_block(const array& x, const std::string& prefix) {
    array gate = linear(x, prefix + "mlp.gate_proj");
    array up = linear(x, prefix + "mlp.up_proj");
    
    array activated = gate * sigmoid(gate) * up;
    
    return linear(activated, prefix + "mlp.down_proj");
}


/*
 * sample_token
 */
int Qwen3Inference::sample_token(const array& logits, const OgaGeneratorParams& params) {
    if (!params.do_sample || params.temperature <= 0.0f) {
        return static_cast<int>(argmax(logits).item<int32_t>());
    }
    
    array scaled_logits = logits / params.temperature;
    array probs = softmax(scaled_logits, -1);
    return static_cast<int>(argmax(probs).item<int32_t>());
}


void Qwen3Inference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    const auto& q = model_.quantization;
    
    std::cout << "[Qwen3Inference] Caching weights (" 
              << (q.is_quantized() ? std::to_string(q.bits) + "-bit" : "fp32") << ")" << std::endl;

    // Helper to cache a weight and pre-transpose if non-quantized (optimization #1)
    auto cache_weight = [&](const std::string& name) {
        auto it = w.find(name + ".weight");
        if (it != w.end()) {
            cached_weights_.emplace(name + ".weight", it->second);
            
            auto scales_it = w.find(name + ".scales");
            if (scales_it != w.end()) {
                // Quantized: cache scales/biases
                cached_weights_.emplace(name + ".scales", scales_it->second);
                auto biases_it = w.find(name + ".biases");
                if (biases_it != w.end()) {
                    cached_weights_.emplace(name + ".biases", biases_it->second);
                }
            } else {
                // Non-quantized: pre-transpose weight to avoid per-token transpose
                array transposed = transpose(it->second, {1, 0});
                eval(transposed);  // Force evaluation once at load time
                cached_weights_.emplace(name + ".weight_T", std::move(transposed));
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
        if (in_ln != w.end()) cached_weights_.emplace(p + "input_layernorm.weight", in_ln->second);
        
        auto post_ln = w.find(p + "post_attention_layernorm.weight");
        if (post_ln != w.end()) cached_weights_.emplace(p + "post_attention_layernorm.weight", post_ln->second);
        
        auto q_norm = w.find(p + "self_attn.q_norm.weight");
        if (q_norm != w.end()) cached_weights_.emplace(p + "self_attn.q_norm.weight", q_norm->second);
        
        auto k_norm = w.find(p + "self_attn.k_norm.weight");
        if (k_norm != w.end()) cached_weights_.emplace(p + "self_attn.k_norm.weight", k_norm->second);
        
        cache_weight(p + "self_attn.q_proj");
        cache_weight(p + "self_attn.k_proj");
        cache_weight(p + "self_attn.v_proj");
        cache_weight(p + "self_attn.o_proj");
        
        cache_weight(p + "mlp.gate_proj");
        cache_weight(p + "mlp.up_proj");
        cache_weight(p + "mlp.down_proj");
    }

    auto norm_it = w.find("norm.weight");
    if (norm_it != w.end()) {
        cached_weights_.emplace("norm.weight", norm_it->second);
    }

    if (!tie_word_embeddings_) {
        cache_weight("lm_head");
    }

    std::cout << "[Qwen3Inference] Cached " << cached_weights_.size() << " tensors" << std::endl;
}


/*
 * setup_weight_references
 * Populates the weights_ structure with direct pointers to cached weights.
 * This enables the optimized inference path with zero hash map lookups.
 */
void Qwen3Inference::setup_weight_references() {
    using namespace ryzenai::mlx;
    
    // Helper to get pointer to cached weight (returns nullptr if not found)
    auto get_weight = [this](const std::string& name) -> const array* {
        auto it = cached_weights_.find(name);
        return it != cached_weights_.end() ? &(it->second) : nullptr;
    };
    
    // Helper to populate LinearWeights (includes pre-transposed for non-quantized)
    auto setup_linear = [&](const std::string& prefix) -> LinearWeights {
        LinearWeights lw;
        lw.weight = get_weight(prefix + ".weight");
        lw.weight_T = get_weight(prefix + ".weight_T");  // Pre-transposed (non-quantized only)
        lw.scales = get_weight(prefix + ".scales");
        lw.biases = get_weight(prefix + ".biases");
        return lw;
    };
    
    // Setup embedding
    weights_.embed_tokens = get_weight("embed_tokens.weight");
    
    // Setup final norm
    weights_.final_norm = get_weight("norm.weight");
    
    // Setup LM head (only if not tie_word_embeddings)
    if (!tie_word_embeddings_) {
        weights_.lm_head = setup_linear("lm_head");
    }
    
    // Setup per-layer weights
    weights_.layers.resize(model_.num_hidden_layers);
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        const std::string& p = layer_prefixes_[i];
        LayerWeights& layer = weights_.layers[i];
        
        // Normalization weights
        layer.input_layernorm = get_weight(p + "input_layernorm.weight");
        layer.post_attn_layernorm = get_weight(p + "post_attention_layernorm.weight");
        
        // Attention weights
        layer.attention.q_proj = setup_linear(p + "self_attn.q_proj");
        layer.attention.k_proj = setup_linear(p + "self_attn.k_proj");
        layer.attention.v_proj = setup_linear(p + "self_attn.v_proj");
        layer.attention.o_proj = setup_linear(p + "self_attn.o_proj");
        layer.attention.q_norm = get_weight(p + "self_attn.q_norm.weight");
        layer.attention.k_norm = get_weight(p + "self_attn.k_norm.weight");
        
        // MLP weights
        layer.mlp.gate_proj = setup_linear(p + "mlp.gate_proj");
        layer.mlp.up_proj = setup_linear(p + "mlp.up_proj");
        layer.mlp.down_proj = setup_linear(p + "mlp.down_proj");
    }
    
    std::cout << "[Qwen3Inference] Setup direct weight references for " 
              << weights_.layers.size() << " layers" << std::endl;
}


/*
 * linear_fast - Optimized linear using direct weight references
 * Uses pre-transposed weights for non-quantized models (eliminates per-token transpose)
 */
array Qwen3Inference::linear_fast(const array& x, const ryzenai::mlx::LinearWeights& w) {
    if (!w.weight) {
        throw std::runtime_error("linear_fast: weight is null");
    }
    
    // 1. Quantized Path
    if (w.is_quantized()) {
        std::optional<array> biases_opt = w.biases ? std::optional(*w.biases) : std::nullopt;
        return quantized_matmul(
            x, 
            *w.weight, 
            *w.scales, 
            biases_opt,
            true,  // transpose
            model_.quantization.group_size,
            model_.quantization.bits,
            "affine"
        );
    }
    
    // 2. Non-Quantized Path
    array output = w.has_pretransposed() ? matmul(x, *w.weight_T) : 
                    matmul(x, transpose(*w.weight, {1, 0}));
    if (w.biases) {
        output = output + *w.biases;
    }
    
    return output;
}

/*
 * rms_norm_fast - Optimized RMS norm using MLX fast::rms_norm
 * OPTIMIZED: Uses fused kernel instead of manual calculation
 */
array Qwen3Inference::rms_norm_fast(const array& x, const array* weight) {
    if (!weight) return x;
    // OPTIMIZED: Use MLX fast::rms_norm - fused kernel, no intermediates
    return fast::rms_norm(x, *weight, model_.rms_norm_eps);
}


/*
 * self_attention_fast - Optimized attention using direct weight references
 * Uses pre-allocated KV cache to reduce memory fragmentation
 */
array Qwen3Inference::self_attention_fast(const array& x, const ryzenai::mlx::LayerWeights& layer,
                                           int layer_idx, const std::string& mask_type) {
    int B = static_cast<int>(x.shape(0));
    int L = static_cast<int>(x.shape(1));
    
    // Direct Q/K/V projections (Qwen3 architecture)
    array queries = reshape(linear_fast(x, layer.attention.q_proj), 
                           {B, L, model_.num_attention_heads, head_dim_});
    array keys = reshape(linear_fast(x, layer.attention.k_proj), 
                        {B, L, model_.num_key_value_heads, head_dim_});
    array values = reshape(linear_fast(x, layer.attention.v_proj), 
                          {B, L, model_.num_key_value_heads, head_dim_});
    
    // 2. Q/K Normalization using direct references
    queries = rms_norm_fast(queries, layer.attention.q_norm);
    keys = rms_norm_fast(keys, layer.attention.k_norm);
    
    // 3. Transpose to [B, n_heads, L, head_dim]
    array q_transposed = transpose(queries, {0, 2, 1, 3});
    array k_transposed = transpose(keys, {0, 2, 1, 3});
    array v_transposed = transpose(values, {0, 2, 1, 3});
    
    // 4. Apply RoPE using pre-computed theta
    int past_len = cache_position_;  // Use cache_position_ for RoPE offset
    array roped_q = fast::rope(q_transposed, head_dim_, false, weights_.rope_theta, 1.0f, past_len);
    array roped_k = fast::rope(k_transposed, head_dim_, false, weights_.rope_theta, 1.0f, past_len);
    
    // 5. Update KV cache (optimization #2: pre-allocated buffers reduce fragmentation)
    int new_cache_pos = cache_position_ + L;
    
    // Use lambda to compute full K/V avoiding default-constructed array
    auto compute_kv = [&]() -> std::pair<array, array> {
        if (cache_position_ == 0) {
            // First tokens - just use the new K/V directly
            return {roped_k, v_transposed};
        } else if (new_cache_pos <= max_cache_length_) {
            // Within cache limit - get valid portion and concatenate
            int b = static_cast<int>(k_cache_[layer_idx].shape(0));
            int h = static_cast<int>(k_cache_[layer_idx].shape(1));
            int d = static_cast<int>(k_cache_[layer_idx].shape(3));
            
            array cached_k = slice(k_cache_[layer_idx], {0, 0, 0, 0}, {b, h, cache_position_, d});
            array cached_v = slice(v_cache_[layer_idx], {0, 0, 0, 0}, {b, h, cache_position_, d});
            
            return {concatenate({cached_k, roped_k}, 2), concatenate({cached_v, v_transposed}, 2)};
        } else {
            // Cache overflow - use sliding window (keep most recent tokens)
            int keep_from = cache_position_ - (max_cache_length_ - L);
            if (keep_from < 0) keep_from = 0;
            int keep_len = cache_position_ - keep_from;
            
            if (keep_len > 0) {
                int b = static_cast<int>(k_cache_[layer_idx].shape(0));
                int h = static_cast<int>(k_cache_[layer_idx].shape(1));
                int d = static_cast<int>(k_cache_[layer_idx].shape(3));
                
                array cached_k = slice(k_cache_[layer_idx], {0, 0, keep_from, 0}, {b, h, cache_position_, d});
                array cached_v = slice(v_cache_[layer_idx], {0, 0, keep_from, 0}, {b, h, cache_position_, d});
                
                return {concatenate({cached_k, roped_k}, 2), concatenate({cached_v, v_transposed}, 2)};
            } else {
                return {roped_k, v_transposed};
            }
        }
    };
    
    auto [full_k, full_v] = compute_kv();
    
    // Store in pre-allocated cache (may reallocate if exceeds, but benefits from pre-allocation)
    k_cache_[layer_idx] = full_k;
    v_cache_[layer_idx] = full_v;
    
    // 6. Attention using pre-computed scale
    array output = mask_type == "none" ? 
            fast::scaled_dot_product_attention(roped_q, full_k, full_v, weights_.attention_scale) :
            fast::scaled_dot_product_attention(roped_q, full_k, full_v, weights_.attention_scale, "causal");
    
    output = transpose(output, {0, 2, 1, 3});
    output = reshape(output, {B, L, model_.num_attention_heads * head_dim_});
    
    return linear_fast(output, layer.attention.o_proj);
}


/*
 * mlp_block_fast - Optimized MLP using direct weight references
 */
array Qwen3Inference::mlp_block_fast(const array& x, const ryzenai::mlx::MLPWeights& mlp) {
    array gate = linear_fast(x, mlp.gate_proj);
    array up = linear_fast(x, mlp.up_proj);
    
    // SwiGLU: gate * sigmoid(gate) * up
    array activated = gate * sigmoid(gate) * up;
    
    return linear_fast(activated, mlp.down_proj);
}
