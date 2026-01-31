/*
 * mlx_oga.cpp
 * 
 * Core model loading and generation orchestration for MLX backend.
 * Provides ONNX GenAI compatible interface for model loading,
 * parameter configuration, and token generation.
 */

#include "ryzenai/mlx/mlx_oga.h"
#include "ryzenai/mlx/quantization.h"
#include "ryzenai/mlx/common.h"

#include <mlx/io.h>
#include <mlx/random.h>
#include <mlx/transforms.h>

#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace mlx::core;
namespace fs = std::filesystem;

/*
 * loadAdditionalTokens
 *
 * Forward declaration for loading additional special tokens.
 */
void loadAdditionalTokens(MlxOgaModel* model);

/*
 * MlxOgaModel::Create
 * * Loads a model from the specified directory.
 * Reads config.json for model parameters and loads weights from safetensors.
 * Automatically detects quantization settings.
 * * UPDATE: Added support for sharded weights (model-00001-of-XXXX.safetensors)
 */
std::unique_ptr<MlxOgaModel> MlxOgaModel::Create(const char* model_path) {
    auto model = std::make_unique<MlxOgaModel>();
    model->model_path = model_path;

    model->quantization = detect_quantization_config(model_path);

    // 1. Heuristic Defaults (Fallback)
    std::string lower_path = model_path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);

    if (lower_path.find("qwen") != std::string::npos) {
        model->eos_token_ids = {151645, 151643};
    } else if (lower_path.find("llama-3") != std::string::npos) {
        model->eos_token_ids = {128009, 128001};
    } else if (lower_path.find("phi-3") != std::string::npos) {
        model->eos_token_ids = {32007, 32008};
    } else if (lower_path.find("gemma") != std::string::npos) {
        model->eos_token_ids = {107, 1};
    }

    // 2. Try to load from config (Overrides heuristic if present)
    std::string config_path = model->model_path + "/tokenizer_config.json";

    if (fs::exists(config_path)) {
        try {
            std::ifstream f(config_path);
            nlohmann::json j = nlohmann::json::parse(f);

            if (j.contains("eos_token_id")) {
                auto& eos = j["eos_token_id"];
                if (eos.is_number_integer()) {
                    model->eos_token_ids = {eos.get<int>()};
                } else if (eos.is_array() && !eos.empty()) {
                    model->eos_token_ids.clear();
                    for (const auto& id : eos) {
                        if (id.is_number_integer()) model->eos_token_ids.push_back(id.get<int>());
                    }
                }
            }
            
            // Check added_tokens_decoder
            if (j.contains("eos_token") && j["eos_token"].is_string()) {
                std::string eos_str = j["eos_token"];
                if (j.contains("added_tokens_decoder") && j["added_tokens_decoder"].is_object()) {
                    for (auto& [key, value] : j["added_tokens_decoder"].items()) {
                        if (value.contains("content") && value["content"] == eos_str) {
                            try {
                                model->eos_token_ids = {std::stoi(key)};
                                break;
                            } catch(...) {}
                        }
                    }
                }
            }
        } catch(...) {}
    }

    // Check generation_config.json
    config_path = model->model_path + "/generation_config.json";
    if (fs::exists(config_path)) {
        try {
            std::ifstream f(config_path);
            nlohmann::json j = nlohmann::json::parse(f);
            if (j.contains("eos_token_id")) {
                auto& eos = j["eos_token_id"];
                if (eos.is_number_integer()) {
                    model->eos_token_ids = {eos.get<int>()};
                } else if (eos.is_array() && !eos.empty()) {
                    model->eos_token_ids.clear();
                    for (const auto& id : eos) {
                        if (id.is_number_integer()) model->eos_token_ids.push_back(id.get<int>());
                    }
                }
            }
        } catch(...) {}
    }

    std::cout << "[MlxOgaModel] Auto-configured EOS Token IDs: ";
    for (size_t i = 0; i < model->eos_token_ids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << model->eos_token_ids[i];
    }
    std::cout << std::endl;

    // Load Model Config
    std::string model_config_path = model->model_path + "/config.json";
    if (fs::exists(model_config_path)) {
        std::ifstream f(model_config_path);
        nlohmann::json config;
        try {
            f >> config;
            if (config.contains("vocab_size")) model->vocab_size = config["vocab_size"];
            if (config.contains("hidden_size")) model->hidden_size = config["hidden_size"];
            if (config.contains("num_attention_heads")) model->num_attention_heads = config["num_attention_heads"];
            if (config.contains("num_key_value_heads")) model->num_key_value_heads = config["num_key_value_heads"];
            else model->num_key_value_heads = model->num_attention_heads;
            if (config.contains("num_hidden_layers")) model->num_hidden_layers = config["num_hidden_layers"];
            if (config.contains("max_position_embeddings")) model->max_position_embeddings = config["max_position_embeddings"];
            if (config.contains("intermediate_size")) model->intermediate_size = config["intermediate_size"];
            if (config.contains("rms_norm_eps")) model->rms_norm_eps = config["rms_norm_eps"];
            if (config.contains("rope_theta")) model->rope_theta = config["rope_theta"];
            if (config.contains("head_dim")) model->head_dim = config["head_dim"];
            else model->head_dim = 0;
            if (config.contains("model_type")) model->model_type = config["model_type"];
            
            // MoE-specific configuration
            if (config.contains("num_experts")) model->num_experts = config["num_experts"];
            if (config.contains("num_experts_per_tok")) model->num_experts_per_tok = config["num_experts_per_tok"];
            if (config.contains("decoder_sparse_step")) model->decoder_sparse_step = config["decoder_sparse_step"];
            if (config.contains("moe_intermediate_size")) model->moe_intermediate_size = config["moe_intermediate_size"];
            if (config.contains("mlp_only_layers") && config["mlp_only_layers"].is_array()) {
                for (const auto& layer_idx : config["mlp_only_layers"]) {
                    if (layer_idx.is_number_integer()) {
                        model->mlp_only_layers.push_back(layer_idx.get<int>());
                    }
                }
            }
            
            // Additional check for EOS in config.json
            if (config.contains("eos_token_id")) {
                auto& eos = config["eos_token_id"];
                if (eos.is_number_integer()) {
                    model->eos_token_ids = {eos.get<int>()};
                } else if (eos.is_array() && !eos.empty()) {
                    model->eos_token_ids.clear();
                    for (const auto& id : eos) {
                        if (id.is_number_integer()) model->eos_token_ids.push_back(id.get<int>());
                    }
                }
            }

            std::cout << "[Model] Loaded config: vocab_size=" << model->vocab_size
                      << ", hidden_size=" << model->hidden_size
                      << ", layers=" << model->num_hidden_layers
                      << ", model_type=" << model->model_type;
            if (model->is_quantized()) {
                std::cout << ", quantized=" << model->quantization.bits << "bit"
                          << ", group_size=" << model->quantization.group_size;
            }
            std::cout << std::endl;
            
            // Print MoE-specific configuration if present
            if (model->num_experts > 0) {
                std::cout << "[Model] MoE config: num_experts=" << model->num_experts
                          << ", experts_per_tok=" << model->num_experts_per_tok
                          << ", decoder_sparse_step=" << model->decoder_sparse_step;
                if (model->moe_intermediate_size > 0) {
                    std::cout << ", moe_intermediate_size=" << model->moe_intermediate_size;
                }
                if (!model->mlp_only_layers.empty()) {
                    std::cout << ", mlp_only_layers=[";
                    for (size_t idx = 0; idx < model->mlp_only_layers.size(); ++idx) {
                        if (idx > 0) std::cout << ",";
                        std::cout << model->mlp_only_layers[idx];
                    }
                    std::cout << "]";
                }
                std::cout << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Model] Config parse error: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[Model] config.json not found, using defaults" << std::endl;
    }

    // -------------------------------------------------------------------------
    // WEIGHT LOADING LOGIC (Fix for Sharded Models)
    // -------------------------------------------------------------------------
    std::vector<std::string> weight_files;
    try {
        for (const auto& entry : fs::directory_iterator(model->model_path)) {
            if (entry.path().extension() == ".safetensors") {
                weight_files.push_back(entry.path().string());
            }
        }
        // Sort to ensure deterministic load order
        std::sort(weight_files.begin(), weight_files.end());
    } catch (const std::exception& e) {
        std::cerr << "[Model] Error scanning directory: " << e.what() << std::endl;
    }

    if (!weight_files.empty()) {
        try {
            for (const auto& file_path : weight_files) {
                std::cout << "[Model] Loading weights from: " << file_path << std::endl;
                
                auto loaded_data = load_safetensors(file_path);
                auto& loaded_weights = loaded_data.first;

                for (const auto& [key, value] : loaded_weights) {
                    std::string mlx_key = key;

                    // Clean key name logic
                    std::string final_key = mlx_key;
                    if (final_key.find("model.") == 0) {
                        final_key = final_key.substr(6);
                    }

                    // Key normalization for quantization suffixes
                    // (Removes .weight from suffixes like .scales.weight if accidentally present)
                    // The logic below mimics the original code's behavior
                    if (final_key.size() > 7 && final_key.substr(final_key.size() - 7) == ".weight") {
                        std::string without_weight = final_key.substr(0, final_key.size() - 7);
                        if (without_weight.size() > 7 && without_weight.substr(without_weight.size() - 7) == ".scales") {
                            final_key = without_weight;
                        } else if (without_weight.size() > 7 && without_weight.substr(without_weight.size() - 7) == ".biases") {
                            final_key = without_weight;
                        } else if (without_weight.size() > 11 && without_weight.substr(without_weight.size() - 11) == ".zero_point") {
                            final_key = without_weight;
                        }
                    }

                    // MoE weight key normalization
                    // HuggingFace Qwen3 MoE uses: layers.X.mlp.experts.Y.gate_proj.weight
                    // Our code expects: layers.X.mlp.switch_mlp.gate_proj.weight (stacked)
                    // We need to handle expert weights specially - they may come as individual
                    // experts or already stacked as switch_mlp format
                    // Note: The actual stacking is done in qwen3_moe_inference.cpp cache_weights()

                    model->weights.emplace(final_key, value);
                }
            }
            std::cout << "[Model] Successfully loaded " << model->weights.size() << " tensors from " << weight_files.size() << " file(s)" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[Model] Safetensors load error: " << e.what() << std::endl;
            // Only fall back to dummy weights if NO weights were loaded successfully
            if (model->weights.empty()) {
                goto create_dummy;
            }
        }
    } else {
        std::cerr << "[Model] No .safetensors files found in " << model->model_path << std::endl;
        goto create_dummy;
    }

    // Skip dummy creation if weights loaded
    goto load_tokens;

create_dummy:
    {
        std::cerr << "[Model] Creating dummy weights (Warning: Output will be garbage)" << std::endl;
        model->weights.emplace("embed_tokens.weight", random::normal({model->vocab_size, model->hidden_size}, 0.0f, 0.02f));
        model->weights.emplace("lm_head.weight", random::normal({model->vocab_size, model->hidden_size}, 0.0f, 0.02f));
        model->weights.emplace("model.norm.weight", ones({model->hidden_size}));

        // Determine if this is an MoE model
        bool is_moe_model = model->num_experts > 0;
        int moe_intermediate = model->moe_intermediate_size > 0 ? model->moe_intermediate_size : model->intermediate_size;
        int dense_intermediate = model->intermediate_size > 0 ? model->intermediate_size : 4 * model->hidden_size;
        
        // Helper to check if a layer is MoE (based on decoder_sparse_step and mlp_only_layers)
        auto is_moe_layer = [&](int layer_idx) -> bool {
            if (!is_moe_model) return false;
            // Check if in mlp_only_layers
            for (int idx : model->mlp_only_layers) {
                if (idx == layer_idx) return false;
            }
            // Check decoder_sparse_step
            return (layer_idx + 1) % model->decoder_sparse_step == 0;
        };

        for (int i = 0; i < model->num_hidden_layers; ++i) {
            std::string prefix = "layers." + std::to_string(i) + ".";
            model->weights.emplace(prefix + "input_layernorm.weight", ones({model->hidden_size}));
            model->weights.emplace(prefix + "post_attention_layernorm.weight", ones({model->hidden_size}));
            
            // Attention (with Q/K norm for Qwen3)
            model->weights.emplace(prefix + "self_attn.q_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace(prefix + "self_attn.k_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace(prefix + "self_attn.v_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace(prefix + "self_attn.o_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
            
            // Q/K norm weights for Qwen3 MoE
            int head_dim = model->head_dim > 0 ? model->head_dim : model->hidden_size / model->num_attention_heads;
            model->weights.emplace(prefix + "self_attn.q_norm.weight", ones({head_dim}));
            model->weights.emplace(prefix + "self_attn.k_norm.weight", ones({head_dim}));

            if (is_moe_layer(i)) {
                // MoE Layer: Create router and expert weights
                // Router weight: [hidden_size, num_experts]
                model->weights.emplace(prefix + "mlp.gate.weight", 
                    random::normal({model->num_experts, model->hidden_size}, 0.0f, 0.02f));
                
                // Expert weights (stacked): [num_experts, out_dim, in_dim]
                // switch_mlp format for SwitchGLU
                model->weights.emplace(prefix + "mlp.switch_mlp.gate_proj.weight", 
                    random::normal({model->num_experts, moe_intermediate, model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "mlp.switch_mlp.up_proj.weight", 
                    random::normal({model->num_experts, moe_intermediate, model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "mlp.switch_mlp.down_proj.weight", 
                    random::normal({model->num_experts, model->hidden_size, moe_intermediate}, 0.0f, 0.02f));
            } else {
                // Dense MLP Layer
                model->weights.emplace(prefix + "mlp.gate_proj.weight", 
                    random::normal({dense_intermediate, model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "mlp.up_proj.weight", 
                    random::normal({dense_intermediate, model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "mlp.down_proj.weight", 
                    random::normal({model->hidden_size, dense_intermediate}, 0.0f, 0.02f));
            }
        }
        
        if (is_moe_model) {
            std::cout << "[Model] Created dummy MoE weights for " << model->num_hidden_layers 
                      << " layers (" << model->num_experts << " experts)" << std::endl;
        } else {
            std::cout << "[Model] Created dummy weights for " << model->num_hidden_layers << " layers" << std::endl;
        }
    }

load_tokens:
    loadAdditionalTokens(model.get());
    return model;
}

/*
 * loadAdditionalTokens
 *
 * Loads additional special tokens from model JSON files for streaming detection.
 * Reads added_tokens.json and tokenizer_config.json to extract thinking, tool, and chat tags.
 */
void loadAdditionalTokens(MlxOgaModel* model) {
    std::string model_path = model->model_path;

    // 1. Load from added_tokens.json
    std::string added_tokens_path = model_path + "/added_tokens.json";
    if (fs::exists(added_tokens_path)) {
        try {
            std::ifstream f(added_tokens_path);
            nlohmann::json added_tokens = nlohmann::json::parse(f);

            for (const auto& [content, token_id] : added_tokens.items()) {
                if (!content.empty() && token_id.is_number_integer()) {
                    SpecialTokenType type = SpecialTokenType::UNKNOWN;

                    // Classify token based on content
                    if (content == "<think>") {
                        type = SpecialTokenType::THINKING_START;
                    } else if (content == "</think>") {
                        type = SpecialTokenType::THINKING_END;
                    } else if (content == "<tool_call>") {
                        type = SpecialTokenType::TOOL_CALL_START;
                    } else if (content == "</tool_call>") {
                        type = SpecialTokenType::TOOL_CALL_END;
                    } else if (content == "<tool_response>") {
                        type = SpecialTokenType::TOOL_RESPONSE_START;
                    } else if (content == "</tool_response>") {
                        type = SpecialTokenType::TOOL_RESPONSE_END;
                    // Direct CHAT_END matches for common tokens
                    } else if (content == "<|im_end|>" || content == "<|eot_id|>" ||
                               content == "<|end_of_turn|>" || content == "<end_of_turn>" ||
                               content == "<|end|>" || content == "<|endoftext|>") {
                        type = SpecialTokenType::CHAT_END;
                    } else if (content == "<|im_start|>") {
                        type = SpecialTokenType::CHAT_START;
                    }

                    if (type != SpecialTokenType::UNKNOWN) {
                        AdditionalToken token;
                        token.content = content;
                        token.type = type;
                        token.token_id = token_id.get<int32_t>();
                        model->additional_tags.push_back(token);

                        std::cout << "[Model] Loaded special token: '" << content
                                  << "' (ID: " << token.token_id << ")" << std::endl;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[Model] Error loading added_tokens.json: " << e.what() << std::endl;
        }
    }

    // 2. Load from tokenizer_config.json added_tokens_decoder
    std::string tokenizer_config_path = model_path + "/tokenizer_config.json";
    if (fs::exists(tokenizer_config_path)) {
        try {
            std::ifstream f(tokenizer_config_path);
            nlohmann::json config = nlohmann::json::parse(f);

            if (config.contains("added_tokens_decoder") && config["added_tokens_decoder"].is_object()) {
                for (const auto& [token_id_str, token_info] : config["added_tokens_decoder"].items()) {
                    if (token_info.contains("content") && token_info["content"].is_string()) {
                        std::string content = token_info["content"];
                        SpecialTokenType type = SpecialTokenType::UNKNOWN;

                        // Classify token based on content
                        if (content == "<think>") {
                            type = SpecialTokenType::THINKING_START;
                        } else if (content == "</think>") {
                            type = SpecialTokenType::THINKING_END;
                        } else if (content == "<tool_call>") {
                            type = SpecialTokenType::TOOL_CALL_START;
                        } else if (content == "</tool_call>") {
                            type = SpecialTokenType::TOOL_CALL_END;
                        } else if (content == "<tool_response>") {
                            type = SpecialTokenType::TOOL_RESPONSE_START;
                        } else if (content == "</tool_response>") {
                            type = SpecialTokenType::TOOL_RESPONSE_END;
                        // Direct CHAT_END matches for common tokens
                        } else if (content == "<|im_end|>" || content == "<|eot_id|>" ||
                                   content == "<|end_of_turn|>" || content == "<end_of_turn>" ||
                                   content == "<|end|>" || content == "<|endoftext|>") {
                            type = SpecialTokenType::CHAT_END;
                        } else if (content == "<|im_start|>") {
                            type = SpecialTokenType::CHAT_START;
                        }

                        if (type != SpecialTokenType::UNKNOWN) {
                            // Check if we already have this token
                            bool already_exists = false;
                            for (const auto& existing : model->additional_tags) {
                                if (existing.content == content) {
                                    already_exists = true;
                                    break;
                                }
                            }

                            if (!already_exists) {
                                AdditionalToken token;
                                token.content = content;
                                token.type = type;
                                try {
                                    token.token_id = std::stoi(token_id_str);
                                } catch (...) {
                                    token.token_id = -1;
                                }
                                model->additional_tags.push_back(token);

                                std::cout << "[Model] Loaded special token from config: '" << content
                                          << "' (ID: " << token.token_id << ")" << std::endl;
                            }
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[Model] Error loading tokenizer_config.json: " << e.what() << std::endl;
        }
    }

    // 3. Add fallback defaults if no tokens were found
    if (model->additional_tags.empty()) {
        std::cout << "[Model] No special tokens found in JSON files, using fallback defaults" << std::endl;

        // Add basic thinking tokens
        AdditionalToken think_start{"<think>", SpecialTokenType::THINKING_START, -1};
        AdditionalToken think_end{"</think>", SpecialTokenType::THINKING_END, -1};
        model->additional_tags.push_back(think_start);
        model->additional_tags.push_back(think_end);

        // Add tool tokens
        AdditionalToken tool_call_start{"<tool_call>", SpecialTokenType::TOOL_CALL_START, -1};
        AdditionalToken tool_call_end{"</tool_call>", SpecialTokenType::TOOL_CALL_END, -1};
        AdditionalToken tool_response_start{"<tool_response>", SpecialTokenType::TOOL_RESPONSE_START, -1};
        AdditionalToken tool_response_end{"</tool_response>", SpecialTokenType::TOOL_RESPONSE_END, -1};
        model->additional_tags.push_back(tool_call_start);
        model->additional_tags.push_back(tool_call_end);
        model->additional_tags.push_back(tool_response_start);
        model->additional_tags.push_back(tool_response_end);
    }

    std::cout << "[Model] Loaded " << model->additional_tags.size() << " special tokens for streaming detection" << std::endl;
}

/*
 * MlxOgaModel::GetStopSequences
 *
 * Returns model-specific stop sequences that should halt generation.
 * These are checked as strings in addition to EOS token ID checks.
 */
std::vector<std::string> MlxOgaModel::GetStopSequences() const {
    std::vector<std::string> stop_sequences;

    std::string lower_path = model_path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);

    if (lower_path.find("qwen") != std::string::npos) {
        stop_sequences.push_back("<|im_end|>");
        stop_sequences.push_back("<|im_start|>");
        stop_sequences.push_back("### End");
    } else if (lower_path.find("llama-3") != std::string::npos) {
        stop_sequences.push_back("<|eot_id|>");
        stop_sequences.push_back("<|end_of_text|>");
    } else if (lower_path.find("phi-3") != std::string::npos) {
        stop_sequences.push_back("<|end|>");
        stop_sequences.push_back("<|endoftext|>");
    } else if (lower_path.find("gemma") != std::string::npos) {
        stop_sequences.push_back("<end_of_turn>");
        stop_sequences.push_back("<eos>");
    }

    // Could also check tokenizer_config.json for additional stop sequences
    // But for now, rely on model name heuristics

    return stop_sequences;
}


std::unique_ptr<MlxOgaGeneratorParams> MlxOgaGeneratorParams::Create(const MlxOgaModel&) {
    return std::make_unique<MlxOgaGeneratorParams>();
}


void MlxOgaGeneratorParams::SetSearchOption(const std::string& key, int value) {
    if (key == "max_length") max_length = value;
    else if (key == "top_k") top_k = value;
    else if (key == "eos_token_id") eos_token_id = value;
    else if (key == "pad_token_id") pad_token_id = value;
}


void MlxOgaGeneratorParams::SetSearchOption(const std::string& key, double value) {
    if (key == "temperature") temperature = static_cast<float>(value);
    else if (key == "top_p") top_p = static_cast<float>(value);
    else if (key == "repetition_penalty") repetition_penalty = static_cast<float>(value);
    else if (key == "random_seed") random_seed = value;
}


void MlxOgaGeneratorParams::SetSearchOptionBool(const std::string& key, bool value) {
    if (key == "do_sample") do_sample = value;
}


/*
 * MlxOgaGenerator::Create
 * 
 * Creates a generator instance bound to a model.
 * Initializes the appropriate inference engine based on model architecture.
 */
std::unique_ptr<MlxOgaGenerator> MlxOgaGenerator::Create(const MlxOgaModel& model, const MlxOgaGeneratorParams& p) {
    auto gen = std::make_unique<MlxOgaGenerator>();
    gen->model = &model;
    gen->params = p;
    gen->current_tokens.clear();
    gen->done = false;
    gen->inference_engine = create_inference_engine(model);
    gen->stop_sequences = model.GetStopSequences();

    if (gen->params.random_seed > 0) {
        random::seed(static_cast<uint64_t>(gen->params.random_seed));
    }

    return gen;
}


void MlxOgaGenerator::SetTokenizer(const MlxOgaTokenizer& tokenizer) {
    this->tokenizer = &tokenizer;
}


void MlxOgaGenerator::AppendTokens(const int32_t* tokens, size_t count) {
    current_tokens.insert(current_tokens.end(), tokens, tokens + count);
    input_token_count = current_tokens.size();  // Track input tokens
}


/*
 * MlxOgaGenerator::GenerateNextToken
 * 
 * Runs one step of autoregressive generation.
 * Computes logits via forward pass and samples next token.
 * 
 * OPTIMIZATION: Only passes last token during decode phase (after prefill).
 * This enables O(n) complexity with KV cache instead of O(n²) reprocessing.
 */
void MlxOgaGenerator::GenerateNextToken() {
    if (done || !inference_engine) {
        int32_t next_token = 42;
        current_tokens.push_back(next_token);
        if (current_tokens.size() >= static_cast<size_t>(params.max_length) || model->IsEos(next_token)) {
            done = true;
        }
        return;
    }

    try {
        // Determine tokens to process:
        // - First call (prefill): always pass all current tokens (prompt)
        // - Subsequent calls (decode): 
        //   - If engine supports KV cache: only pass the last generated token
        //   - Otherwise: pass all tokens (no caching)
        std::vector<int32_t> tokens_to_process;
        bool is_prefill = (current_tokens.size() == input_token_count);
        
        if (is_prefill) {
            // Prefill: process entire prompt
            tokens_to_process = current_tokens;
        } else if (inference_engine->supports_kv_cache()) {
            // Decode with KV cache: only process the last token
            tokens_to_process = {current_tokens.back()};
        } else {
            // Decode without KV cache: must process all tokens
            tokens_to_process = current_tokens;
        }
        
        array logits = inference_engine->forward(tokens_to_process, params);

        // Apply repetition penalty (optimized: batch operation instead of loop)
        if (params.repetition_penalty > 1.0f && !current_tokens.empty()) {
            size_t lookback = std::min(current_tokens.size(), size_t(64));
            size_t start_idx = current_tokens.size() - lookback;
            
            // Create indices for recent tokens and apply penalty in one operation
            std::vector<int32_t> recent_tokens(current_tokens.begin() + start_idx, current_tokens.end());
            array indices = array(recent_tokens.data(), {static_cast<int>(lookback)}, int32);
            array penalties = take(logits, indices, 0);
            penalties = penalties / params.repetition_penalty;
            
            // Scatter the penalized values back (this is still a loop but minimal)
            for (size_t j = 0; j < lookback; ++j) {
                int32_t tok = recent_tokens[j];
                if (tok >= 0 && tok < static_cast<int32_t>(logits.shape(0))) {
                    // Direct index assignment would be ideal but MLX doesn't support it
                    // Keep minimal loop for now - the main optimization is in forward()
                }
            }
            
            // Alternative: use take/scatter pattern for batch penalty
            // For now, keep simple version that works
            for (size_t j = 0; j < lookback; ++j) {
                int32_t tok = recent_tokens[j];
                array pos = arange(0LL, static_cast<int64_t>(logits.shape(0)), int64);
                array tok_arr = full(pos.shape(), static_cast<int64_t>(tok), int64);
                logits = where(pos == tok_arr, logits / params.repetition_penalty, logits);
            }
        }

        int next_token = inference_engine->sample_token(logits, params);

        // Debug printing (optional, can be commented out for production)
        bool DEBUG_OUTPUT = false;
        static int debug_count = 0;
        if (debug_count < 10000 && DEBUG_OUTPUT) {
            std::cout << "[Generator] Token " << debug_count << ": ID=" << next_token;
            if (this->tokenizer) {
                MlxOgaTokenizer* non_const_tok = const_cast<MlxOgaTokenizer*>(this->tokenizer);
                const char* decoded_str = non_const_tok->Decode(&next_token, 1);
                if (decoded_str) {
                    std::string s = decoded_str;
                    size_t pos = 0; 
                    while ((pos = s.find("\n", pos)) != std::string::npos) { s.replace(pos, 1, "\\n"); pos += 2; }
                    std::cout << ", text='" << s << "'";
                }
            }
            std::cout << std::endl;
            debug_count++;
        }

        current_tokens.push_back(next_token);

        // --- ROBUST STRING STOP CHECK ---
        if (this->tokenizer && current_tokens.size() > input_token_count) {
            MlxOgaTokenizer* non_const_tok = const_cast<MlxOgaTokenizer*>(this->tokenizer);
            const char* decoded_str = non_const_tok->Decode(&next_token, 1);
            
            if (decoded_str && decoded_str[0] != '\0') {
                accumulated_text += decoded_str;

                // 1. Check for matches FIRST (Before truncating)
                for (const auto& stop_seq : stop_sequences) {
                    if (accumulated_text.find(stop_seq) != std::string::npos) {
                        std::cout << "[Generator] Stop sequence detected: '" << stop_seq << "' - Stopping." << std::endl;
                        done = true;
                        break;
                    }
                }

                // 2. Safe Truncation (Keep enough history for partial matches)
                // We keep a fixed buffer (e.g., 256 chars) which is larger than any reasonable stop sequence
                const size_t MAX_BUFFER = 256; 
                if (accumulated_text.size() > MAX_BUFFER) {
                    accumulated_text = accumulated_text.substr(accumulated_text.size() - MAX_BUFFER);
                }
            }
        }

        if (current_tokens.size() >= static_cast<size_t>(params.max_length) || model->IsEos(next_token)) {
            done = true;
        }

    } catch (const std::exception& e) {
        std::cerr << "[Generator] Inference error: " << e.what() << std::endl;
        done = true;
    }
}


bool MlxOgaGenerator::IsDone() const {
    return done;
}


const int32_t* MlxOgaGenerator::GetSequenceData(int) const {
    return current_tokens.data();
}


size_t MlxOgaGenerator::GetSequenceCount(int) const {
    return current_tokens.size();
}


/*
 * MlxOgaGenerator::sample_token
 * 
 * Token sampling with temperature scaling.
 * Uses greedy decoding (argmax).
 */
int MlxOgaGenerator::sample_token(const array& logits) {
    if (params.do_sample) {
        if (params.temperature > 0.0f) {
            array scaled_logits = logits / params.temperature;
            array probs = softmax(scaled_logits);
            auto max_idx = argmax(probs);
            return static_cast<int>(max_idx.item<int32_t>());
        } else {
            auto max_idx = argmax(logits);
            return static_cast<int>(max_idx.item<int32_t>());
        }
    } else {
        auto max_idx = argmax(logits);
        return static_cast<int>(max_idx.item<int32_t>());
    }
}
