/*
 * qwen3_next_inference.h
 * 
 * Qwen3-Next model inference engine for MLX backend.
 * Implements hybrid architecture with GatedDeltaNet (linear attention),
 * full attention, and Mixture of Experts (MoE).
 * Only compiled when USE_MLX is defined (macOS with Apple Silicon).
 */

#pragma once

#ifdef USE_MLX

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include <vector>
#include <string>
#include <unordered_map>


class Qwen3NextInference : public BaseInferenceEngine {
public:
    /*
     * Qwen3NextInference
     * Initializes the inference engine with model weights and configuration.
     */
    Qwen3NextInference(const MlxOgaModel& model);

    /*
     * forward
     * Runs a forward pass through all layers.
     * Returns logits for the final token position.
     */
    array forward(const std::vector<int32_t>& input_tokens,
                  const MlxOgaGeneratorParams& params) override;

    /*
     * sample_token
     * Selects the next token from output logits.
     */
    int sample_token(const array& logits, const MlxOgaGeneratorParams& params) override;

private:
    const MlxOgaModel& model_;
    int actual_hidden_size_;
    int head_dim_;
    int full_attention_interval_;
    int num_experts_;
    int num_experts_per_tok_;
    float partial_rotary_factor_;
    bool tie_word_embeddings_;
    std::unordered_map<std::string, array> cached_weights_;

    void cache_weights();
    array linear(const array& x, const std::string& weight_name);
    array rms_norm(const array& x, const std::string& weight_name);
    
    /*
     * process_layer
     * Processes a single layer - either linear attention or full attention
     */
    array process_layer(const array& hidden_states, int layer_idx, int seq_len);
    
    /*
     * full_attention_layer
     * Standard multi-head attention with Q/K normalization and partial RoPE
     */
    array full_attention(const array& x, const std::string& prefix, int seq_len);
    
    /*
     * linear_attention_layer
     * GatedDeltaNet linear attention
     */
    array linear_attention(const array& x, const std::string& prefix, int seq_len);
    
    /*
     * mlp_block
     * Standard SwiGLU MLP
     */
    array mlp_block(const array& x, const std::string& prefix);
    
    /*
     * moe_block
     * Mixture of Experts block with shared expert
     */
    array moe_block(const array& x, const std::string& prefix);
    
    /*
     * is_linear_layer
     * Returns true if layer uses linear attention (GatedDeltaNet)
     */
    bool is_linear_layer(int layer_idx) const;
    
    /*
     * is_moe_layer
     * Returns true if layer uses MoE
     */
    bool is_moe_layer(int layer_idx) const;
};

#endif // USE_MLX
