/*
 * phi_inference.h
 * 
 * Original Phi model inference engine for MLX backend.
 * Implements the Phi architecture with parallel attention and MLP,
 * LayerNorm, GELU activation, and partial RoPE.
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include <vector>
#include <string>
#include <unordered_map>


class PhiInference : public BaseInferenceEngine {
public:
    PhiInference(const MlxOgaModel& model);

    array forward(const std::vector<int32_t>& input_tokens,
                  const MlxOgaGeneratorParams& params) override;

    int sample_token(const array& logits, const MlxOgaGeneratorParams& params) override;

private:
    const MlxOgaModel& model_;
    int actual_hidden_size_;
    int head_dim_;
    float partial_rotary_factor_;
    float layer_norm_eps_;
    std::unordered_map<std::string, array> cached_weights_;

    void cache_weights();
    array linear(const array& x, const std::string& weight_name);
    array linear_with_bias(const array& x, const std::string& weight_name);
    array layer_norm(const array& x, const std::string& weight_name);
    
    array process_layer(const array& hidden_states, int layer_idx, int seq_len);
    array self_attention(const array& x, const std::string& prefix, int seq_len);
    array mlp_block(const array& x, const std::string& prefix);
};
