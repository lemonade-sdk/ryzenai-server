/*
 * inference_factory.cpp
 * 
 * Factory for creating model-specific inference engines.
 * Detects model architecture and instantiates the appropriate engine.
 */

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/gemma_inference.h"
#include "ryzenai/mlx/phi3_inference.h"
#include <mlx/io.h>
#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;


/*
 * LFMInference
 * 
 * Placeholder inference engine for LFM/Llama-style models.
 * To be implemented with full forward pass logic.
 */
class LFMInference : public BaseInferenceEngine {
public:
    LFMInference(const OgaModel& model) : model_(model) {
        std::cout << "[LFMInference] Initialized" << std::endl;
    }

    array forward(const std::vector<int32_t>& input_tokens,
                  const OgaGeneratorParams& params) override {
        std::cerr << "[LFMInference] Forward pass not implemented" << std::endl;
        return zeros({model_.vocab_size});
    }

    int sample_token(const array& logits, const OgaGeneratorParams& params) override {
        return argmax(logits).item<int32_t>();
    }

private:
    const OgaModel& model_;
};


/*
 * detect_model_type
 * 
 * Determines model architecture from config.json and weight names.
 * Checks for model_type field, architecture hints, and weight patterns.
 */
std::string detect_model_type(const OgaModel& model) {
    std::string config_path = model.model_path + "/config.json";
    
    if (fs::exists(config_path)) {
        try {
            std::ifstream f(config_path);
            nlohmann::json config;
            f >> config;

            if (config.contains("model_type")) {
                std::string model_type = config["model_type"];
                if (model_type == "phi3") return "phi3";
                if (model_type == "gemma") return "gemma";
                if (model_type == "llama") return "llama";
            }

            if (config.contains("architectures")) {
                auto architectures = config["architectures"];
                if (architectures.is_array() && !architectures.empty()) {
                    std::string arch = architectures[0];
                    if (arch.find("Phi3") != std::string::npos) return "phi3";
                    if (arch.find("Gemma") != std::string::npos) return "gemma";
                    if (arch.find("Llama") != std::string::npos) return "llama";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[InferenceFactory] Config parse error: " << e.what() << std::endl;
        }
    }

    bool has_qkv_proj = false;
    bool has_separate_qkv = false;
    bool has_feed_forward = false;

    for (const auto& [key, _] : model.weights) {
        if (key.find("qkv_proj") != std::string::npos) has_qkv_proj = true;
        if (key.find("q_proj") != std::string::npos ||
            key.find("k_proj") != std::string::npos ||
            key.find("v_proj") != std::string::npos) has_separate_qkv = true;
        if (key.find("feed_forward") != std::string::npos) has_feed_forward = true;
    }

    if (has_qkv_proj) return "phi3";
    if (has_feed_forward) return "lfm";
    if (has_separate_qkv) return "gemma";

    std::cout << "[InferenceFactory] Unknown model type, defaulting to Gemma" << std::endl;
    return "gemma";
}


/*
 * create_inference_engine
 * 
 * Factory function that creates the appropriate inference engine
 * based on detected model architecture.
 */
std::unique_ptr<BaseInferenceEngine> create_inference_engine(const OgaModel& model) {
    std::string model_type = detect_model_type(model);
    std::cout << "[InferenceFactory] Model type: " << model_type << std::endl;

    if (model_type == "phi3") {
        return std::make_unique<Phi3Inference>(model);
    } else if (model_type == "lfm" || model_type == "llama") {
        return std::make_unique<LFMInference>(model);
    } else {
        return std::make_unique<GemmaInference>(model);
    }
}
