/*
 * qwen3_moe_inference.h
 * 
 * Qwen3 Mixture of Experts (MoE) model inference engine for MLX backend.
 * Supports Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Coder-30B-A3B variants.
 * 
 * Architecture based on mlx-lm/models/qwen3_moe.py:
 *   - Same Qwen3 attention (Q/K normalization, RoPE)
 *   - Sparse MoE MLP with top-k expert routing via SwitchGLU
 *   - Hybrid layers: some layers use dense MLP, others use MoE
 * 
 * Key config parameters:
 *   - num_experts: Total experts (e.g., 64 for 30B-A3B)
 *   - num_experts_per_tok: Active experts per token (e.g., 8)
 *   - decoder_sparse_step: MoE layer frequency
 *   - mlp_only_layers: Indices of dense MLP layers
 *   - moe_intermediate_size: Expert hidden dimension
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include "ryzenai/mlx/layer_weights.h"
#include "ryzenai/mlx/moe.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <optional>


/*
 * Qwen3 MoE layer weights - can be either dense MLP or MoE
 */
struct Qwen3MoELayerWeights {
    // Normalization
    const ::mlx::core::array* input_layernorm = nullptr;
    const ::mlx::core::array* post_attn_layernorm = nullptr;
    
    // Attention (same structure as dense Qwen3)
    ryzenai::mlx::AttentionWeights attention;
    
    // Dense MLP (for mlp_only_layers)
    ryzenai::mlx::MLPWeights dense_mlp;
    
    // Whether this layer uses MoE or dense MLP
    bool is_moe_layer = true;
};


class Qwen3MoEInference : public BaseInferenceEngine {
public:
    /*
     * Qwen3MoEInference
     * Initializes the MoE inference engine with model weights and configuration.
     */
    Qwen3MoEInference(const MlxOgaModel& model);

    /*
     * forward
     * Runs a forward pass through all transformer layers.
     * Handles both prefill (prompt) and decode (generation) modes.
     */
    ::mlx::core::array forward(const std::vector<int32_t>& input_tokens,
                                const MlxOgaGeneratorParams& params) override;

    /*
     * sample_token
     * Selects the next token from output logits.
     */
    int sample_token(const ::mlx::core::array& logits, 
                     const MlxOgaGeneratorParams& params) override;

    /*
     * clear_cache
     * Resets the KV cache for a new conversation.
     */
    void clear_cache();
    
    bool supports_kv_cache() const override { return true; }

private:
    const MlxOgaModel& model_;
    
    // Model dimensions
    int actual_hidden_size_;
    int head_dim_;
    int intermediate_size_;
    int moe_intermediate_size_;
    bool tie_word_embeddings_;
    
    // MoE configuration
    ryzenai::moe::MoEConfig moe_config_;
    std::unordered_set<int> mlp_only_layers_;  // Layers using dense MLP
    std::unordered_set<int> moe_layers_initialized_;  // Layers with MoE already set up (not needing lazy init)
    
    // Quantized embedding flag
    bool embed_is_quantized_ = false;
    
    // Cached weights
    std::unordered_map<std::string, ::mlx::core::array> cached_weights_;
    
    // Direct weight references
    const ::mlx::core::array* embed_tokens_ = nullptr;
    const ::mlx::core::array* final_norm_ = nullptr;
    ryzenai::mlx::LinearWeights lm_head_;
    std::vector<Qwen3MoELayerWeights> layer_weights_;
    
    // MoE blocks per layer (only for MoE layers)
    std::vector<std::optional<ryzenai::moe::SparseMoEBlock>> moe_blocks_;
    
    // Pre-transposed embedding for tie_word_embeddings
    std::optional<::mlx::core::array> embed_tokens_transposed_;
    
    // Pre-computed values
    float attention_scale_ = 0.0f;
    float rope_theta_ = 1000000.0f;

    // KV Cache
    std::vector<::mlx::core::array> k_cache_;
    std::vector<::mlx::core::array> v_cache_;
    bool cache_initialized_ = false;
    int cache_position_ = 0;
    int step_ = 0;
    int max_cache_length_;

    /*
     * cache_weights
     * Pre-processes and stores model weights.
     */
    void cache_weights();
    
    /*
     * setup_weight_references
     * Sets up direct pointers to weights for fast access.
     */
    void setup_weight_references();
    
    /*
     * setup_moe_blocks
     * Creates SparseMoEBlock for each MoE layer.
     */
    void setup_moe_blocks();

    /*
     * linear_fast
     * Optimized linear projection.
     */
    ::mlx::core::array linear_fast(const ::mlx::core::array& x, 
                                    const ryzenai::mlx::LinearWeights& w);

    /*
     * rms_norm_fast
     * Optimized RMS normalization using MLX fast::rms_norm.
     */
    ::mlx::core::array rms_norm_fast(const ::mlx::core::array& x, 
                                      const ::mlx::core::array* weight);

    /*
     * self_attention_fast
     * Self-attention with Q/K normalization, RoPE, and KV caching.
     */
    ::mlx::core::array self_attention_fast(const ::mlx::core::array& x, 
                                            const Qwen3MoELayerWeights& layer,
                                            int layer_idx, 
                                            const std::string& mask_type);

    /*
     * dense_mlp_block
     * Dense SwiGLU MLP for mlp_only_layers.
     */
    ::mlx::core::array dense_mlp_block(const ::mlx::core::array& x,
                                        const ryzenai::mlx::MLPWeights& mlp);
    
    /*
     * is_moe_layer
     * Checks if a layer uses MoE (based on decoder_sparse_step and mlp_only_layers).
     */
    bool is_moe_layer(int layer_idx) const;
    
    /*
     * initialize_moe_layer_lazy
     * Lazily initializes MoE block for a layer on first use.
     * Stacks expert weights and creates SwitchGLU.
     */
    void initialize_moe_layer_lazy(int layer_idx);
};
