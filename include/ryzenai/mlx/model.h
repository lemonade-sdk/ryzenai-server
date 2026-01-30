/*
 * model.h
 * 
 * Model configuration and weight storage for MLX inference.
 * Holds all model parameters and loaded weight tensors.
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/quantization.h"
#include <unordered_map>
#include <string>


/*
 * OgaModel
 * 
 * Contains model configuration, quantization settings, and weight tensors.
 * Loaded from safetensors format with config.json for architecture params.
 */
struct OgaModel {
    std::string model_path;
    int eos_token_id = 2;
    int vocab_size = 32000;
    int hidden_size = 4096;
    int num_attention_heads = 32;
    int num_key_value_heads = 32;
    int num_hidden_layers = 32;
    int max_position_embeddings = 4096;
    int intermediate_size = 11008;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    std::string model_type = "";

    QuantizationConfig quantization;
    std::unordered_map<std::string, array> weights;

    /*
     * Create
     * Factory function that loads model from directory.
     * Reads config.json and model.safetensors.
     */
    static std::unique_ptr<OgaModel> Create(const char* model_path);

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
};
