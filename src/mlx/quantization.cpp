/*
 * quantization.cpp
 * 
 * Weight quantization and dequantization routines for MLX inference.
 * Supports 2, 4, and 8-bit packed integer formats with affine scaling.
 */

#include "ryzenai/mlx/quantization.h"
#include <mlx/mlx.h>
#include <json.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace mlx::core;
namespace fs = std::filesystem;


/*
 * detect_quantization_config
 * 
 * Reads the model's config.json and extracts quantization settings.
 * Checks multiple locations where quantization config may appear:
 *   - quantization section (MLX native)
 *   - quantization_config section (HuggingFace style)
 *   - gptq section
 *   - awq section
 */
QuantizationConfig detect_quantization_config(const std::string& model_path) {
    QuantizationConfig config;
    std::string config_path = model_path + "/config.json";

    if (!fs::exists(config_path)) {
        std::cout << "[Quantization] No config.json found, using defaults" << std::endl;
        return config;
    }

    try {
        std::ifstream file(config_path);
        nlohmann::json json_config;
        file >> json_config;

        if (json_config.contains("quantization")) {
            auto& quant = json_config["quantization"];
            
            if (quant.contains("bits"))
                config.bits = quant["bits"].get<int>();
            if (quant.contains("group_size"))
                config.group_size = quant["group_size"].get<int>();
            if (quant.contains("quant_method"))
                config.scheme = quant["quant_method"].get<std::string>();
            if (quant.contains("symmetric"))
                config.symmetric = quant["symmetric"].get<bool>();
            
            std::cout << "[Quantization] Config: bits=" << config.bits
                      << ", group_size=" << config.group_size
                      << ", scheme=" << config.scheme
                      << ", symmetric=" << (config.symmetric ? "true" : "false")
                      << std::endl;
        }

        if (config.bits == 0 && json_config.contains("quantization_config")) {
            auto& quant = json_config["quantization_config"];
            
            if (quant.contains("bits"))
                config.bits = quant["bits"].get<int>();
            if (quant.contains("group_size"))
                config.group_size = quant["group_size"].get<int>();
            if (quant.contains("quant_method"))
                config.scheme = quant["quant_method"].get<std::string>();
        }

        if (json_config.contains("gptq")) {
            auto& gptq = json_config["gptq"];
            config.scheme = "gptq";
            
            if (gptq.contains("bits"))
                config.bits = gptq["bits"].get<int>();
            if (gptq.contains("group_size"))
                config.group_size = gptq["group_size"].get<int>();
        }

        if (json_config.contains("awq")) {
            auto& awq = json_config["awq"];
            config.scheme = "awq";
            
            if (awq.contains("bits"))
                config.bits = awq["bits"].get<int>();
            if (awq.contains("group_size"))
                config.group_size = awq["group_size"].get<int>();
        }

    } catch (const std::exception& e) {
        std::cerr << "[Quantization] Parse error in config.json: " << e.what() << std::endl;
    }

    return config;
}


/*
 * detect_quantization_from_weights
 * 
 * Analyzes weight and scale tensor shapes to infer quantization parameters.
 * Tries different bit widths (4, 2, 8) to find a consistent configuration.
 */
QuantizationConfig detect_quantization_from_weights(
    const std::unordered_map<std::string, array>& weights,
    const std::string& base_name) {

    QuantizationConfig config;

    std::string scales_name = base_name + ".scales";
    std::string biases_name = base_name + ".biases";
    std::string zero_point_name = base_name + ".zero_point";
    std::string weight_name = base_name + ".weight";

    auto scales_it = weights.find(scales_name);
    auto biases_it = weights.find(biases_name);
    auto zero_point_it = weights.find(zero_point_name);
    auto weight_it = weights.find(weight_name);

    if (scales_it == weights.end()) {
        return config;
    }

    config.symmetric = (zero_point_it != weights.end());

    if (!config.symmetric && biases_it == weights.end()) {
        std::cout << "[Quantization] Found scales without biases or zero_point for "
                  << base_name << std::endl;
    }

    if (weight_it != weights.end() && scales_it != weights.end()) {
        const array& weight_tensor = weight_it->second;
        const array& scales_tensor = scales_it->second;

        auto wshape = weight_tensor.shape();
        auto sshape = scales_tensor.shape();

        if (wshape.size() >= 2 && sshape.size() >= 2) {
            int64_t packed_cols = wshape[1];
            int64_t num_groups = sshape[1];

            int candidate_bits[] = {4, 2, 8};
            for (int bits : candidate_bits) {
                int unpack_factor = 8 / bits;
                int64_t in_features = packed_cols * unpack_factor;
                int64_t group_size = in_features / num_groups;

                if (group_size * num_groups == in_features && group_size > 0) {
                    config.bits = bits;
                    config.group_size = static_cast<int>(group_size);
                    
                    std::cout << "[Quantization] Detected " << base_name
                              << ": bits=" << config.bits
                              << ", group_size=" << config.group_size << std::endl;
                    return config;
                }
            }

            config.bits = 4;
            int64_t in_features = packed_cols * 2;
            config.group_size = static_cast<int>(in_features / num_groups);
            std::cout << "[Quantization] Defaulting to 4-bit for " << base_name << std::endl;
        }
    }

    return config;
}


/*
 * is_weight_quantized
 * 
 * Checks whether a weight tensor has associated quantization scales.
 */
bool is_weight_quantized(const std::string& base_name,
                         const std::unordered_map<std::string, array>& weights) {
    return weights.find(base_name + ".scales") != weights.end();
}


/*
 * get_quantized_weight_names
 * 
 * Returns the standard parameter names for a quantized weight group.
 */
QuantizedWeightNames get_quantized_weight_names(const std::string& base_name) {
    QuantizedWeightNames names;
    names.weight = base_name + ".weight";
    names.scales = base_name + ".scales";
    names.biases = base_name + ".biases";
    names.zero_point = base_name + ".zero_point";
    return names;
}


/*
 * dequantize_generic
 * 
 * Calls MLX's native dequantize function for unpacking quantized weights.
 * The packed weight tensor is unpacked according to the bit width,
 * then scaled and biased: output = scale * quantized + bias
 */
array dequantize_generic(const array& quantized_weight, const array& scales,
                         const array& biases, int bits, int group_size) {
    auto wshape = quantized_weight.shape();
    
    if (wshape.size() != 2) {
        throw std::runtime_error("Expected 2D packed weight tensor for dequantization");
    }

    return dequantize(
        quantized_weight,
        scales,
        biases,
        group_size,
        bits,
        "affine",
        float32,
        {}
    );
}


/*
 * quantized_linear
 * 
 * Performs matrix multiplication on quantized weights without unpacking.
 * Uses MLX's quantized_matmul for memory-efficient computation.
 */
array quantized_linear(const array& x, const array& weight,
                       const array& scales, const array& biases,
                       int bits, int group_size, bool transpose_weight) {
    return quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose_weight,
        group_size,
        bits,
        "affine",
        {}
    );
}


/*
 * dequantize_weight
 * 
 * Entry point for dequantizing a weight tensor.
 * Validates bit width and dispatches to dequantize_generic.
 */
array dequantize_weight(const array& quantized_weight, const array& scales,
                        const array& biases, int bits, int group_size) {
    if (bits != 2 && bits != 4 && bits != 8) {
        throw std::runtime_error("Unsupported quantization bit width: " + 
                                 std::to_string(bits) + ". Valid values: 2, 4, 8");
    }
    return dequantize_generic(quantized_weight, scales, biases, bits, group_size);
}


/*
 * apply_quantization
 * 
 * Conditionally dequantizes a weight tensor based on available parameters.
 * If scales exist, performs dequantization. Otherwise returns float32 cast.
 */
array apply_quantization(const array& weight, const std::string& base_name,
                         const std::unordered_map<std::string, array>& weights,
                         const QuantizationConfig& config) {
    
    std::string scales_name = base_name + ".scales";
    std::string biases_name = base_name + ".biases";
    std::string zero_point_name = base_name + ".zero_point";

    auto scales_it = weights.find(scales_name);
    auto biases_it = weights.find(biases_name);
    auto zero_point_it = weights.find(zero_point_name);

    if (scales_it == weights.end()) {
        std::cout << "[Quantization] Non-quantized weight: " << base_name << std::endl;
        return astype(weight, float32);
    }

    const array& scales = scales_it->second;
    
    array biases = [&]() {
        if (biases_it != weights.end()) {
            return biases_it->second;
        }
        if (zero_point_it != weights.end()) {
            return astype(zero_point_it->second, float32);
        }
        return zeros_like(scales);
    }();

    int bits = config.bits;
    if (bits == 0) {
        auto detected = detect_quantization_from_weights(weights, base_name);
        bits = detected.bits;
        if (bits == 0) {
            bits = 4;
        }
    }

    std::cout << "[Quantization] Dequantizing " << base_name
              << " (bits=" << bits << ")"
              << " weight shape: " << weight.shape()
              << " scales: " << scales.shape()
              << ", biases: " << biases.shape() << std::endl;

    return dequantize_weight(weight, scales, biases, bits, config.group_size);
}
