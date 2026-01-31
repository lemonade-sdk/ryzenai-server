/*
 * phi3_inference.h
 * 
 * Phi-3 model inference engine for MLX backend.
 * Implements the Phi-3 architecture with combined QKV projections
 * and SwiGLU MLP blocks.
 * 
 * Optimizations:
 *   - Quantized KV Cache (INT8) for 2x memory bandwidth reduction
 *   - Pre-allocated buffers to eliminate per-token memory allocation
 *   - Pre-transposed weights for non-quantized models
 *   - Direct weight references to eliminate hash map lookups
 * 
 * Performance notes:
 *   - 50% GPU utilization typically indicates memory-bandwidth bound
 *   - INT8 KV cache reduces bandwidth by ~48%, improving GPU utilization
 *   - Pre-allocation eliminates O(N) concatenation overhead
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include "ryzenai/mlx/quantized_kv_cache.h"
#include "ryzenai/mlx/layer_weights.h"
#include "ryzenai/types.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>

// Use common KVCacheMode from types.h
using ryzenai::KVCacheMode;

class Phi3Inference : public BaseInferenceEngine {
public:
    /*
     * Phi3Inference
     * Initializes the inference engine with model weights and configuration.
     * Caches weights for efficient repeated forward passes.
     * 
     * @param model The loaded OGA model
     * @param kv_cache_mode KV cache quantization mode (default: INT8 for optimal performance)
     */
    Phi3Inference(const MlxOgaModel& model, KVCacheMode kv_cache_mode = KVCacheMode::INT8);

    /*
     * forward
     * Runs a forward pass through all transformer layers.
     * Returns logits for the final token position.
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
     * supports_kv_cache
     * Returns true - KV cache enabled for efficient incremental decoding.
     */
    bool supports_kv_cache() const override { return true; }
    
    /*
     * clear_cache
     * Resets the KV cache for new conversation.
     */
    void clear_cache();

private:
    mlx::core::array mask_val_;

    const MlxOgaModel& model_;
    int actual_hidden_size_;
    int head_dim_;
    int max_cache_length_;
    std::unordered_map<std::string, array> cached_weights_;
    
    // KV cache mode (INT8 quantized or FP16)
    KVCacheMode kv_cache_mode_;
    
    // Quantized KV cache (INT8) - used when kv_cache_mode_ == INT8
    // Provides 2x memory bandwidth reduction for improved GPU utilization
    std::optional<ryzenai::mlx::QuantizedKVCache> quantized_kv_cache_;
    
    // Legacy FP16 KV cache - used when kv_cache_mode_ == FP16
    std::vector<array> k_cache_;
    std::vector<array> v_cache_;
    int cache_position_ = 0;
    
    // Pre-computed constants
    float attention_scale_;
    float rope_theta_;
    
    // ============================================================
    // OPTIMIZATION: Direct weight references (eliminate hash lookups)
    // ============================================================
    ryzenai::mlx::ModelWeights weights_;
    std::vector<std::string> layer_prefixes_;  // Pre-computed to avoid string alloc
    
    // Setup direct weight references after caching
    void setup_weight_references();
    
    // Fast linear using direct references (no hash lookup)
    array linear_fast(const array& x, const ryzenai::mlx::LinearWeights& w);
    
    // Fast RMS norm using direct reference
    array rms_norm_fast(const array& x, const array* weight);
    
    // Fast attention using direct references
    array self_attention_fast(const array& x, const ryzenai::mlx::LayerWeights& layer, int layer_idx, int seq_len);
    
    // Fast MLP using direct references  
    array mlp_block_fast(const array& x, const ryzenai::mlx::LayerWeights& layer);

    /*
     * cache_weights
     * Pre-processes and stores model weights for fast access.
     * Handles quantized and non-quantized weights appropriately.
     * Now also pre-transposes non-quantized weights.
     */
    void cache_weights();

    /*
     * linear
     * Performs linear projection with optional quantized matmul.
     */
    array linear(const array& x, const std::string& weight_name);

    /*
     * rms_norm
     * Applies RMS normalization with learned scale weights.
     */
    array rms_norm(const array& x, const std::string& weight_name);

    /*
     * process_layer
     * Placeholder for layer-wise processing (logic moved to forward).
     */
    array process_layer(const array& hidden_states, int layer_idx, int seq_len);

    /*
     * self_attention
     * Full self-attention with residual connection.
     */
    array self_attention(const array& hidden_states, const std::string& prefix, int seq_len);

    /*
     * self_attention_no_residual
     * Self-attention computation without residual add.
     * Uses combined QKV projection and RoPE embeddings.
     * Now uses KV cache for incremental decoding.
     */
    array self_attention_no_residual(const array& x, const std::string& prefix, 
                                                 int layer_idx, int seq_len);
    
    /*
     * rms_norm_3d - RMS norm for 3D tensors [B, L, hidden]
     */
    array rms_norm_3d(const array& x, const std::string& weight_name);
    
    /*
     * mlp_block_3d - MLP for 3D tensors [B, L, hidden]
     */
    array mlp_block_3d(const array& x, const std::string& prefix);

    /*
     * mlp_block
     * Full MLP block with residual connection.
     */
    array mlp_block(const array& hidden_states, const std::string& prefix);

    /*
     * mlp_block_no_residual
     * SwiGLU MLP computation without residual add.
     * Uses combined gate/up projection.
     */
    array mlp_block_no_residual(const array& hidden_states, const std::string& prefix);
};
