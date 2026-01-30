/*
 * phi3_inference.h
 * 
 * Phi-3 model inference engine for MLX backend.
 * Implements the Phi-3 architecture with combined QKV projections
 * and SwiGLU MLP blocks.
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include <vector>
#include <string>
#include <unordered_map>


class Phi3Inference : public BaseInferenceEngine {
public:
    /*
     * Phi3Inference
     * Initializes the inference engine with model weights and configuration.
     * Caches weights for efficient repeated forward passes.
     */
    Phi3Inference(const OgaModel& model);

    /*
     * forward
     * Runs a forward pass through all transformer layers.
     * Returns logits for the final token position.
     */
    array forward(const std::vector<int32_t>& input_tokens,
                  const OgaGeneratorParams& params) override;

    /*
     * sample_token
     * Selects the next token from output logits.
     * Supports temperature scaling for sampling.
     */
    int sample_token(const array& logits, const OgaGeneratorParams& params) override;

private:
    const OgaModel& model_;
    int actual_hidden_size_;
    int head_dim_;
    std::unordered_map<std::string, array> cached_weights_;

    /*
     * cache_weights
     * Pre-processes and stores model weights for fast access.
     * Handles quantized and non-quantized weights appropriately.
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
     */
    array self_attention_no_residual(const array& hidden_states, const std::string& prefix, int seq_len);

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
