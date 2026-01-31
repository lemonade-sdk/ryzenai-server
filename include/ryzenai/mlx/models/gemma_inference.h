/*
 * gemma_inference.h
 * 
 * Gemma model inference engine for MLX backend.
 * Implements sliding window attention and separate Q/K/V projections.
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include <vector>


class GemmaInference : public BaseInferenceEngine {
public:
    /*
     * GemmaInference
     * Initializes the Gemma inference engine with model configuration.
     * Extracts sliding window size from config if available.
     */
    GemmaInference(const MlxOgaModel& model);

    /*
     * forward
     * Runs a forward pass through all transformer layers.
     * Returns vocabulary logits for the final token position.
     */
    array forward(const std::vector<int32_t>& input_tokens,
                  const MlxOgaGeneratorParams& params) override;

    /*
     * sample_token
     * Selects next token from output logits.
     * Supports temperature-based sampling.
     */
    int sample_token(const array& logits, const MlxOgaGeneratorParams& params) override;

private:
    const MlxOgaModel& model_;
    int sliding_window_;
    int head_dim_;
    int actual_hidden_size_;

    /*
     * process_layer
     * Processes a single transformer layer with attention and MLP.
     */
    array process_layer(const array& hidden_states, int layer_idx, int seq_len);

    /*
     * self_attention
     * Multi-head self attention with separate Q/K/V projections.
     * Applies sliding window mask and RoPE embeddings.
     */
    array self_attention(const array& hidden_states, const std::string& prefix, int seq_len);

    /*
     * apply_sliding_window_mask
     * Creates causal mask limited to sliding window size.
     */
    array apply_sliding_window_mask(const array& attn_scores, int seq_len);

    /*
     * mlp_block
     * SwiGLU MLP with gate, up, and down projections.
     */
    array mlp_block(const array& hidden_states, const std::string& prefix);
};
