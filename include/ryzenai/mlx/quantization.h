#pragma once

/*
 * quantization.h
 * 
 * Low-bit weight quantization support for MLX inference.
 * Handles 2, 4, and 8-bit affine quantization with group-wise scaling.
 * Only compiled when USE_MLX is defined (macOS with Apple Silicon).
 */

#include "ryzenai/mlx/common.h"
#include <unordered_map>
#include <string>
#include <optional>

struct QuantizationConfig {
    int bits = 0;
    int group_size = 64;
    bool symmetric = false;
    std::string scheme = "";

    bool is_quantized() const { return bits > 0; }
    int unpack_factor() const { return (bits == 0) ? 1 : 8 / bits; }
};

struct QuantizedWeightNames {
    std::string weight;
    std::string scales;
    std::string biases;
    std::string zero_point;
};

/*
 * detect_quantization_config
 * Parses model config.json to extract quantization parameters.
 * Supports GPTQ, AWQ, and standard MLX quantization formats.
 */
QuantizationConfig detect_quantization_config(const std::string& model_path);

/*
 * detect_quantization_from_weights
 * Infers quantization parameters by analyzing weight tensor shapes
 * when config metadata is unavailable.
 */
QuantizationConfig detect_quantization_from_weights(
    const std::unordered_map<std::string, array>& weights,
    const std::string& base_name);

/*
 * is_weight_quantized
 * Returns true if the specified weight has associated scale parameters.
 */
bool is_weight_quantized(const std::string& base_name,
                         const std::unordered_map<std::string, array>& weights);

/*
 * get_quantized_weight_names
 * Generates the full parameter names for a quantized weight group.
 */
QuantizedWeightNames get_quantized_weight_names(const std::string& base_name);

/*
 * dequantize_generic
 * Unpacks quantized weights using MLX's native dequantize operation.
 */
array dequantize_generic(const array& quantized_weight, const array& scales,
                         const array& biases, int bits, int group_size);

/*
 * quantized_linear
 * Performs matrix multiplication directly on packed quantized weights.
 * More memory efficient than explicit dequantization.
 */
array quantized_linear(const array& x, const array& weight,
                       const array& scales, const array& biases,
                       int bits, int group_size, bool transpose_weight = true);

/*
 * dequantize_weight
 * Main entry point for weight dequantization.
 */
array dequantize_weight(const array& quantized_weight, const array& scales,
                        const array& biases, int bits, int group_size);

/*
 * apply_quantization
 * Conditionally dequantizes a weight if quantization parameters exist.
 * Returns the original weight cast to float32 if not quantized.
 */
array apply_quantization(const array& weight, const std::string& base_name,
                         const std::unordered_map<std::string, array>& weights,
                         const QuantizationConfig& config = QuantizationConfig());