/*
 * mixtral_inference.h
 * 
 * Mixtral model inference engine for MLX backend.
 * Implements the Mixtral MoE architecture with sparse expert routing.
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/generator.h"
#include <vector>
#include <string>
#include <unordered_map>


class MixtralInference : public BaseInferenceEngine {
public:
    MixtralInference(const OgaModel& model);

    array forward(const std::vector<int32_t>& input_tokens,
                  const OgaGeneratorParams& params) override;

    int sample_token(const array& logits, const OgaGeneratorParams& params) override;

private:
    const OgaModel& model_;
    int actual_hidden_size_;
    int head_dim_;
    int num_local_experts_;
    int num_experts_per_tok_;
    std::unordered_map<std::string, array> cached_weights_;

    void cache_weights();
    array linear(const array& x, const std::string& weight_name);
    array rms_norm(const array& x, const std::string& weight_name);
    
    array process_layer(const array& hidden_states, int layer_idx, int seq_len);
    array self_attention(const array& x, const std::string& prefix, int seq_len);
    array moe_block(const array& x, const std::string& prefix);
};
