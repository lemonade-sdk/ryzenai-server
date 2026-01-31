/*
 * model.h
 * 
 * Model configuration and weight storage for MLX inference.
 * Holds all model parameters and loaded weight tensors.
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/quantization.h"
#include "ryzenai/types.h"
#include <unordered_map>
#include <string>
#include <vector>

// Use common types from types.h
using ryzenai::SpecialTokenType;
using ryzenai::AdditionalToken;


/*
 * MlxOgaModel
 * 
 * Contains model configuration, quantization settings, and weight tensors.
 * Loaded from safetensors format with config.json for architecture params.
 * Named MlxOga* to distinguish from ONNX OGA types.
 */
struct MlxOgaModel {
    std::string model_path;
    std::vector<int32_t> eos_token_ids = {2};
    int vocab_size = 32000;
    int hidden_size = 4096;
    int num_attention_heads = 32;
    int num_key_value_heads = 32;
    int num_hidden_layers = 32;
    int max_position_embeddings = 4096;
    int intermediate_size = 11008;
    int head_dim = 0;  // 0 means compute from hidden_size/num_attention_heads
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool tie_word_embeddings = true;
    bool attention_bias = false;
    bool mlp_bias = false;
    int sliding_window = 0;  // 0 means no sliding window
    int num_experts = 0;  // For MoE models
    int num_experts_per_tok = 0;  // For MoE models
    int decoder_sparse_step = 1;  // For MoE models: frequency of MoE layers
    int moe_intermediate_size = 0;  // For MoE models: expert hidden dim (0 = use intermediate_size)
    std::vector<int> mlp_only_layers;  // For MoE models: indices of dense MLP layers
    std::string model_type = "";
    
    // Context length (passed from command line or config)
    // Used to limit KV cache memory usage
    int max_context_length = 2048;

    // Additional special tokens for streaming detection
    std::vector<AdditionalToken> additional_tags;

    QuantizationConfig quantization;
    std::unordered_map<std::string, array> weights;

    /*
     * Create
     * Factory function that loads model from directory.
     * Reads config.json and model.safetensors.
     */
    static std::unique_ptr<MlxOgaModel> Create(const char* model_path);

    /*
     * is_quantized
     * Returns true if model uses quantized weights.
     */
    bool is_quantized() const { return quantization.is_quantized(); }

    /*
     * get_actual_hidden_size
     * Returns the hidden dimension size.
     */
    int get_actual_hidden_size() const { return hidden_size; }

    /*
     * GetEosId
     * Returns the primary EOS token ID for this model (for backward compatibility).
     */
    int32_t GetEosId() const { return eos_token_ids.empty() ? 2 : eos_token_ids[0]; }

    /*
     * IsEos
     * Returns true if the given token ID is an EOS token for this model.
     */
    bool IsEos(int32_t token_id) const {
        return std::find(eos_token_ids.begin(), eos_token_ids.end(), token_id) != eos_token_ids.end();
    }

    /*
     * GetStopSequences
     * Returns a list of string stop sequences for this model.
     * Used to detect when generation should stop even if the EOS token ID is not generated.
     */
    std::vector<std::string> GetStopSequences() const;
};
