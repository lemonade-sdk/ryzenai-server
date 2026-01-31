/*
 * qwen3_inference.h
 * 
 * Qwen3 model inference engine for MLX backend.
 * Closely follows the Python mlx-lm/models/qwen3.py implementation.
 * 
 * Implements the Qwen3 architecture with:
 *   - Separate Q/K/V projections
 *   - Q/K normalization before RoPE (Qwen3 specific)
 *   - Proper 4D tensor shapes [B, n_heads, L, head_dim]
 *   - mx.fast.scaled_dot_product_attention for efficient attention
 *   - SwiGLU MLP blocks
 * 
 * Features:
 *   - KV Cache for efficient autoregressive generation
 *   - Proper RoPE position tracking with offset
 *   - Support for prefill and decode modes
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include "ryzenai/mlx/layer_weights.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>


class Qwen3Inference : public BaseInferenceEngine {
public:
    /*
     * Qwen3Inference
     * Initializes the inference engine with model weights and configuration.
     * Caches weights for efficient repeated forward passes.
     */
    Qwen3Inference(const MlxOgaModel& model);

    /*
     * forward
     * Runs a forward pass through all transformer layers.
     * Handles both prefill (prompt) and decode (generation) modes.
     * Returns logits for the final token position.
     * 
     * If input_tokens.size() > 1: Prefill mode - clears cache, processes prompt
     * If input_tokens.size() == 1: Decode mode - appends to cache, generates
     */
    array forward(const std::vector<int32_t>& input_tokens,
                  const MlxOgaGeneratorParams& params) override;

    /*
     * sample_token
     * Selects the next token from output logits.
     * Supports temperature scaling for sampling.
     */
    int sample_token(const array& logits, const MlxOgaGeneratorParams& params) override;

    /*
     * clear_cache
     * Resets the KV cache and step counter for a new conversation.
     */
    void clear_cache();
    
    /*
     * supports_kv_cache
     * Returns true - Qwen3 implements KV caching for efficient incremental decoding.
     */
    bool supports_kv_cache() const override { return true; }

private:
    const MlxOgaModel& model_;
    int actual_hidden_size_;
    int head_dim_;
    bool tie_word_embeddings_;
    std::unordered_map<std::string, array> cached_weights_;
    
    // Optimized: Direct weight references (eliminates hash map lookups)
    ryzenai::mlx::ModelWeights weights_;
    
    // Optimized: Pre-transposed embedding for tie_word_embeddings
    std::optional<array> embed_tokens_transposed_;

    // KV Cache: Pre-allocated buffers for each layer (optimization #2)
    // Shape: [B, n_kv_heads, max_cache_length, head_dim]
    // Uses fixed-size buffers to avoid per-token memory allocation
    std::vector<array> k_cache_;
    std::vector<array> v_cache_;
    
    // Cache state tracking
    bool cache_initialized_;  // True if cache buffers have been pre-allocated
    int cache_position_;      // Current write position in pre-allocated cache
    
    // Step counter for RoPE position tracking
    // Represents the total number of tokens processed so far
    int step_;
    
    // Maximum KV cache length to limit memory usage
    int max_cache_length_;

    /*
     * cache_weights
     * Pre-processes and stores model weights for fast access.
     * Handles quantized and non-quantized weights appropriately.
     */
    void cache_weights();

    /*
     * linear
     * Performs linear projection with optional quantized matmul.
     * Legacy version using string lookup (for compatibility during transition)
     */
    array linear(const array& x, const std::string& weight_name);
    
    /*
     * linear_fast
     * Optimized: Performs linear projection using direct weight references.
     * Eliminates hash map lookup overhead.
     */
    array linear_fast(const array& x, const ryzenai::mlx::LinearWeights& w);

    /*
     * rms_norm
     * Applies RMS normalization with learned scale weights.
     * Normalizes over the last axis (following nn.RMSNorm behavior).
     * Legacy version using string lookup
     */
    array rms_norm(const array& x, const std::string& weight_name);
    
    /*
     * rms_norm_fast
     * Optimized: Applies RMS normalization using direct weight reference.
     */
    array rms_norm_fast(const array& x, const array* weight);

    /*
     * create_causal_mask
     * Creates a boolean causal attention mask.
     * 
     * @param seq_len Number of query positions
     * @param offset Position offset for cached keys (step_ value)
     * @return Boolean mask [seq_len, offset + seq_len] where True = attend
     */
    array create_causal_mask(int seq_len, int offset);

    /*
     * self_attention
     * Self-attention computation following Python Attention.__call__.
     * Legacy version using string lookup
     */
    array self_attention(const array& x, const std::string& prefix, 
                         int layer_idx, const std::string& mask_type);
    
    /*
     * self_attention_fast
     * Optimized: Self-attention using direct weight references.
     * Eliminates string construction and hash map lookups.
     */
    array self_attention_fast(const array& x, const ryzenai::mlx::LayerWeights& layer,
                              int layer_idx, const std::string& mask_type);

    /*
     * mlp_block
     * SwiGLU MLP computation following Python MLP.__call__.
     * Legacy version using string lookup
     */
    array mlp_block(const array& x, const std::string& prefix);
    
    /*
     * mlp_block_fast
     * Optimized: SwiGLU MLP using direct weight references.
     */
    array mlp_block_fast(const array& x, const ryzenai::mlx::MLPWeights& mlp);
    
    /*
     * setup_weight_references
     * Populates the weights_ structure with direct pointers to cached weights.
     * Called after cache_weights() to enable optimized inference path.
     */
    void setup_weight_references();
};
