/*
 * qwen3_inference.h
 *
 * Qwen3 model inference engine for MLX backend.
 * Optimized for Apple Silicon (Metal) using MLX.
 *
 * Features:
 * - Graph Compilation for Decoding Step (High TPS on M-series chips)
 * - Fused QKV Projections
 * - Pre-allocated KV Cache (FP16 or INT8 quantized)
 * - Direct Weight References (Zero-overhead lookups)
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include "ryzenai/mlx/layer_weights.h"
#include "ryzenai/mlx/kv_cache.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <functional>


class Qwen3Inference : public BaseInferenceEngine {
public:
    const MlxOgaModel& model_;

public:
    Qwen3Inference(const MlxOgaModel& model, ryzenai::KVCacheMode kv_cache_mode = ryzenai::KVCacheMode::FP16);

    array forward(const std::vector<int32_t>& input_tokens,
                  const MlxOgaGeneratorParams& params) override;

    int sample_token(const array& logits, const MlxOgaGeneratorParams& params) override;

    void clear_cache();
    
    bool supports_kv_cache() const override { return true; }

    /*
     * linear_fast
     * Performs linear projection using direct weight references.
     */
    array linear_fast(const array& x, const ryzenai::mlx::LinearWeights& w) const;

    /*
     * rms_norm_fast
     * Applies RMS normalization using fused Metal kernel.
     */
    array rms_norm_fast(const array& x, const array* weight) const;

    /*
     * mlp_block_fast
     * SwiGLU MLP Block: (SiLU(Gate) * Up) -> Down
     */
    array mlp_block_fast(const array& x, const ryzenai::mlx::MLPWeights& mlp) const;
    
    /*
     * self_attention_fast
     * Note: This is NOT const because it modifies k_cache_ / v_cache_ (via returning new arrays).
     */
    array self_attention_fast(const array& x, const ryzenai::mlx::LayerWeights& layer, int layer_idx, int seq_len);
    

private:
    int actual_hidden_size_;
    int head_dim_;
    bool tie_word_embeddings_;
    
    std::function<std::vector<array>(const std::vector<array>&)> compiled_step_;
    bool step_compiled_ = false;

    // Helpers to marshal data in/out of the compiled graph
    std::vector<array> get_step_inputs(const array& x, int pos);
    void set_step_outputs(const std::vector<array>& outputs);

    std::unordered_map<std::string, array> cached_weights_;
    ryzenai::mlx::ModelWeights weights_;
    std::optional<array> embed_tokens_transposed_;
    bool embed_is_quantized_;

    // RoPE Cache (Precomputed)
    array cos_cache_;
    array sin_cache_;

    std::vector<array> k_cache_;
    std::vector<array> v_cache_;
    
    bool cache_initialized_;
    int cache_position_;
    int step_;
    int max_cache_length_;

    ryzenai::KVCacheMode kv_cache_mode_;
    ryzenai::mlx::HighPerformanceKVCache hp_kv_cache_;  // High-performance O(1) cache
    ryzenai::mlx::KVCache kv_cache_;  // Legacy cache (fallback)
    bool use_hp_cache_ = true;  // Use high-performance cache by default

    void cache_weights();
    void setup_weight_references();
    void precompile_graphs();

    array create_causal_mask(int seq_len, int offset);
    array linear(const array& x, const std::string& weight_name);
    array rms_norm(const array& x, const std::string& weight_name);
};