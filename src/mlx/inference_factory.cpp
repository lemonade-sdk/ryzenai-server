/*
 * inference_factory.cpp
 * 
 * Factory for creating model-specific inference engines.
 * Detects model architecture and instantiates the appropriate engine.
 */

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"

// Include all model inference headers
//#include "ryzenai/mlx/models/gemma_inference.h"
#include "ryzenai/mlx/models/phi3_inference.h"
//#include "ryzenai/mlx/models/phi_inference.h"
#include "ryzenai/mlx/models/qwen3_inference.h"
#include "ryzenai/mlx/models/qwen3_moe_inference.h"
//#include "ryzenai/mlx/models/qwen3_next_inference.h"
//#include "ryzenai/mlx/models/deepseek_inference.h"
//#include "ryzenai/mlx/models/mixtral_inference.h"
//#include "ryzenai/mlx/models/llama_inference.h"

#include <mlx/io.h>
#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;


/*
 * detect_model_type
 * 
 * Determines model architecture from config.json and weight names.
 * Checks for model_type field, architecture hints, and weight patterns.
 */
std::string detect_model_type(const MlxOgaModel& model) {
    std::string config_path = model.model_path + "/config.json";
    
    if (fs::exists(config_path)) {
        try {
            std::ifstream f(config_path);
            nlohmann::json config;
            f >> config;

            // First check model_type field
            if (config.contains("model_type")) {
                std::string model_type = config["model_type"];
                
                // Direct model type matches
                if (model_type == "phi3") return "phi3";
                if (model_type == "phi") return "phi";
                if (model_type == "gemma") return "gemma";
                if (model_type == "gemma2") return "gemma2";
                if (model_type == "llama") return "llama";
                if (model_type == "qwen3_moe") return "qwen3_moe";
                if (model_type == "qwen3") {
                    // Check if it's MoE variant (has num_experts > 0)
                    if (config.contains("num_experts") && config["num_experts"].get<int>() > 0) {
                        return "qwen3_moe";
                    }
                    return "qwen3";
                }
                if (model_type == "qwen3_next") return "qwen3_next";
                if (model_type == "qwen2") return "qwen2";
                if (model_type == "qwen") return "qwen";
                if (model_type == "deepseek") return "deepseek";
                if (model_type == "deepseek_v2") return "deepseek_v2";
                if (model_type == "deepseek_v3") return "deepseek_v3";
                if (model_type == "mixtral") return "mixtral";
                if (model_type == "mistral") return "mistral";
                if (model_type == "internlm" || model_type == "internlm2" || model_type == "internlm3") return "llama";
                if (model_type == "olmo" || model_type == "olmo2" || model_type == "olmo3") return "llama";
                if (model_type == "granite") return "llama";
                if (model_type == "starcoder2") return "llama";
                if (model_type == "cohere" || model_type == "cohere2") return "llama";
            }

            // Check architectures array
            if (config.contains("architectures")) {
                auto architectures = config["architectures"];
                if (architectures.is_array() && !architectures.empty()) {
                    std::string arch = architectures[0];
                    
                    if (arch.find("Phi3") != std::string::npos) return "phi3";
                    if (arch.find("Phi") != std::string::npos) return "phi";
                    if (arch.find("Gemma") != std::string::npos) return "gemma";
                    if (arch.find("Llama") != std::string::npos) return "llama";
                    if (arch.find("Qwen3") != std::string::npos) return "qwen3";
                    if (arch.find("Qwen2") != std::string::npos) return "qwen2";
                    if (arch.find("Qwen") != std::string::npos) return "qwen";
                    if (arch.find("DeepSeek") != std::string::npos || arch.find("Deepseek") != std::string::npos) return "deepseek";
                    if (arch.find("Mixtral") != std::string::npos) return "mixtral";
                    if (arch.find("Mistral") != std::string::npos) return "mistral";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[InferenceFactory] Config parse error: " << e.what() << std::endl;
        }
    }

    // Fallback: detect from weight names
    bool has_qkv_proj = false;
    bool has_separate_qkv = false;
    bool has_feed_forward = false;
    bool has_q_norm = false;
    bool has_block_sparse_moe = false;
    bool has_switch_mlp = false;
    bool has_linear_attn = false;

    for (const auto& [key, _] : model.weights) {
        if (key.find("qkv_proj") != std::string::npos) has_qkv_proj = true;
        if (key.find("q_proj") != std::string::npos ||
            key.find("k_proj") != std::string::npos ||
            key.find("v_proj") != std::string::npos) has_separate_qkv = true;
        if (key.find("feed_forward") != std::string::npos) has_feed_forward = true;
        if (key.find("q_norm") != std::string::npos) has_q_norm = true;
        if (key.find("block_sparse_moe") != std::string::npos) has_block_sparse_moe = true;
        if (key.find("switch_mlp") != std::string::npos) has_switch_mlp = true;
        if (key.find("linear_attn") != std::string::npos) has_linear_attn = true;
    }

    // Decision tree based on weight patterns
    if (has_linear_attn) return "qwen3_next";
    if (has_block_sparse_moe || has_switch_mlp) {
        // Could be mixtral, deepseek, or other MoE
        return "mixtral";
    }
    if (has_qkv_proj) return "phi3";
    if (has_q_norm && has_separate_qkv) return "qwen3";
    if (has_separate_qkv) return "llama";  // Default for separate Q/K/V

    std::cout << "[InferenceFactory] Unknown model type, defaulting to Llama" << std::endl;
    return "llama";
}


/*
 * create_inference_engine
 * 
 * Factory function that creates the appropriate inference engine
 * based on detected model architecture.
 */
std::unique_ptr<BaseInferenceEngine> create_inference_engine(const MlxOgaModel& model) {
    std::string model_type = detect_model_type(model);
    std::cout << "[InferenceFactory] Detected model type: " << model_type << std::endl;

    // Phi family
    if (model_type == "phi3") {
        // Use KV cache format from model (set via --kv-format command line option)
        KVCacheMode kv_mode = model.kv_format;
        
        const char* format_name = (kv_mode == KVCacheMode::INT8) ? "INT8" : 
                                  (kv_mode == KVCacheMode::INT4) ? "INT4" : "FP16";
        std::cout << "[InferenceFactory] Using " << format_name << " KV cache" << std::endl;
        
        return std::make_unique<Phi3Inference>(model, kv_mode);
    }
    /*if (model_type == "phi") {
        return std::make_unique<PhiInference>(model);
    }*/
    
    // Qwen family
    if (model_type == "qwen3_moe") {
        std::cout << "[InferenceFactory] Using Qwen3 MoE inference" << std::endl;
        return std::make_unique<Qwen3MoEInference>(model);
    }
    if (model_type == "qwen3") {
        return std::make_unique<Qwen3Inference>(model);
    }
    /*if (model_type == "qwen3_next") {
        return std::make_unique<Qwen3NextInference>(model);
    }
    if (model_type == "qwen2" || model_type == "qwen") {
        // Qwen2 is similar to Llama architecture
        return std::make_unique<LlamaInference>(model);
    }
    
    // Deepseek family
    if (model_type == "deepseek" || model_type == "deepseek_v2" || model_type == "deepseek_v3") {
        return std::make_unique<DeepseekInference>(model);
    }
    
    // MoE models
    if (model_type == "mixtral") {
        return std::make_unique<MixtralInference>(model);
    }
    
    // Gemma family
    if (model_type == "gemma" || model_type == "gemma2") {
        return std::make_unique<GemmaInference>(model);
    }
    
    // Llama-based models (default for many architectures)
    if (model_type == "llama" || model_type == "mistral") {
        return std::make_unique<LlamaInference>(model);
    }*/

    // Default fallback
    std::cout << "[InferenceFactory] Using default Llama inference for unknown type: " << model_type << std::endl;
    return std::make_unique<Qwen3Inference>(model);
}
