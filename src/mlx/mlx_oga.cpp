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
#include <thread>
#include <sstream>

using namespace mlx::core;
namespace fs = std::filesystem;


/*
 * OgaModel::Create
 * 
 * Loads a model from the specified directory.
 * Reads config.json for model parameters and loads weights from safetensors.
 * Automatically detects quantization settings.
 */
std::unique_ptr<OgaModel> OgaModel::Create(const char* model_path) {
    auto model = std::make_unique<OgaModel>();
    model->model_path = model_path;

    model->quantization = detect_quantization_config(model_path);

    std::string config_path = model->model_path + "/config.json";
    if (fs::exists(config_path)) {
        std::ifstream f(config_path);
        nlohmann::json config;
        try {
            f >> config;
            
            if (config.contains("vocab_size")) 
                model->vocab_size = config["vocab_size"];
            if (config.contains("hidden_size")) 
                model->hidden_size = config["hidden_size"];
            if (config.contains("num_attention_heads")) 
                model->num_attention_heads = config["num_attention_heads"];
            if (config.contains("num_key_value_heads")) 
                model->num_key_value_heads = config["num_key_value_heads"];
            else 
                model->num_key_value_heads = model->num_attention_heads;
            if (config.contains("num_hidden_layers")) 
                model->num_hidden_layers = config["num_hidden_layers"];
            if (config.contains("max_position_embeddings")) 
                model->max_position_embeddings = config["max_position_embeddings"];
            if (config.contains("intermediate_size")) 
                model->intermediate_size = config["intermediate_size"];
            if (config.contains("rms_norm_eps")) 
                model->rms_norm_eps = config["rms_norm_eps"];
            if (config.contains("eos_token_id")) 
                model->eos_token_id = config["eos_token_id"];
            if (config.contains("rope_theta")) 
                model->rope_theta = config["rope_theta"];
            if (config.contains("model_type")) 
                model->model_type = config["model_type"];

            std::cout << "[Model] Loaded config: vocab_size=" << model->vocab_size
                      << ", hidden_size=" << model->hidden_size
                      << ", layers=" << model->num_hidden_layers
                      << ", model_type=" << model->model_type;
            if (model->is_quantized()) {
                std::cout << ", quantized=" << model->quantization.bits << "bit"
                          << ", group_size=" << model->quantization.group_size;
            }
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Model] Config parse error: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[Model] config.json not found, using defaults" << std::endl;
    }

    std::string safetensors_path = model->model_path + "/model.safetensors";
    if (fs::exists(safetensors_path)) {
        try {
            std::cout << "[Model] Loading weights from: " << safetensors_path << std::endl;

            auto loaded_data = load_safetensors(safetensors_path);
            auto& loaded_weights = loaded_data.first;

            for (const auto& [key, value] : loaded_weights) {
                std::string mlx_key = key;

                std::string final_key = mlx_key;
                if (final_key.find("model.") == 0) {
                    final_key = final_key.substr(6);
                }

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

                model->weights.emplace(final_key, value);
                std::cout << "[Model] Loaded weight: " << final_key << " shape: " << value.shape() << std::endl;
            }

            std::cout << "[Model] Loaded " << loaded_weights.size() << " weight tensors" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[Model] Safetensors load error: " << e.what() << std::endl;
            std::cerr << "[Model] Using dummy weights" << std::endl;

            model->weights.emplace("embed_tokens.weight", random::normal({model->vocab_size, model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace("lm_head.weight", random::normal({model->vocab_size, model->hidden_size}, 0.0f, 0.02f));

            for (int i = 0; i < model->num_hidden_layers; ++i) {
                std::string prefix = "model.layers." + std::to_string(i) + ".";
                model->weights.emplace(prefix + "input_layernorm.weight", ones({model->hidden_size}));
                model->weights.emplace(prefix + "post_attention_layernorm.weight", ones({model->hidden_size}));

                model->weights.emplace(prefix + "self_attn.q_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "self_attn.k_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "self_attn.v_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "self_attn.o_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));

                model->weights.emplace(prefix + "mlp.gate_proj.weight", random::normal({model->hidden_size, 4 * model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "mlp.up_proj.weight", random::normal({model->hidden_size, 4 * model->hidden_size}, 0.0f, 0.02f));
                model->weights.emplace(prefix + "mlp.down_proj.weight", random::normal({4 * model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
            }

            model->weights.emplace("model.norm.weight", ones({model->hidden_size}));
            std::cout << "[Model] Created dummy weights for " << model->num_hidden_layers << " layers" << std::endl;
        }
    } else {
        std::cerr << "[Model] model.safetensors not found, creating dummy weights" << std::endl;

        model->weights.emplace("embed_tokens.weight", random::normal({model->vocab_size, model->hidden_size}, 0.0f, 0.02f));
        model->weights.emplace("lm_head.weight", random::normal({model->vocab_size, model->hidden_size}, 0.0f, 0.02f));

        for (int i = 0; i < model->num_hidden_layers; ++i) {
            std::string prefix = "model.layers." + std::to_string(i) + ".";
            model->weights.emplace(prefix + "input_layernorm.weight", ones({model->hidden_size}));
            model->weights.emplace(prefix + "post_attention_layernorm.weight", ones({model->hidden_size}));

            model->weights.emplace(prefix + "self_attn.q_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace(prefix + "self_attn.k_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace(prefix + "self_attn.v_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace(prefix + "self_attn.o_proj.weight", random::normal({model->hidden_size, model->hidden_size}, 0.0f, 0.02f));

            model->weights.emplace(prefix + "mlp.gate_proj.weight", random::normal({model->hidden_size, 4 * model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace(prefix + "mlp.up_proj.weight", random::normal({model->hidden_size, 4 * model->hidden_size}, 0.0f, 0.02f));
            model->weights.emplace(prefix + "mlp.down_proj.weight", random::normal({4 * model->hidden_size, model->hidden_size}, 0.0f, 0.02f));
        }

        model->weights.emplace("model.norm.weight", ones({model->hidden_size}));
        std::cout << "[Model] Created dummy weights for " << model->num_hidden_layers << " layers" << std::endl;
    }

    return model;
}


std::unique_ptr<OgaGeneratorParams> OgaGeneratorParams::Create(const OgaModel&) {
    return std::make_unique<OgaGeneratorParams>();
}


void OgaGeneratorParams::SetSearchOption(const std::string& key, int value) {
    if (key == "max_length") max_length = value;
    else if (key == "top_k") top_k = value;
}


void OgaGeneratorParams::SetSearchOption(const std::string& key, double value) {
    if (key == "temperature") temperature = static_cast<float>(value);
    else if (key == "top_p") top_p = static_cast<float>(value);
    else if (key == "repetition_penalty") repetition_penalty = static_cast<float>(value);
    else if (key == "random_seed") random_seed = value;
}


void OgaGeneratorParams::SetSearchOptionBool(const std::string& key, bool value) {
    if (key == "do_sample") do_sample = value;
}


/*
 * OgaGenerator::Create
 * 
 * Creates a generator instance bound to a model.
 * Initializes the appropriate inference engine based on model architecture.
 */
std::unique_ptr<OgaGenerator> OgaGenerator::Create(const OgaModel& model, const OgaGeneratorParams& p) {
    auto gen = std::make_unique<OgaGenerator>();
    gen->model = &model;
    gen->params = p;
    gen->current_tokens.clear();
    gen->done = false;
    gen->inference_engine = create_inference_engine(model);

    if (gen->params.random_seed > 0) {
        random::seed(static_cast<uint64_t>(gen->params.random_seed));
    }

    return gen;
}


void OgaGenerator::SetTokenizer(const OgaTokenizer& tokenizer) {
    this->tokenizer = &tokenizer;
}


void OgaGenerator::AppendTokens(const int32_t* tokens, size_t count) {
    current_tokens.insert(current_tokens.end(), tokens, tokens + count);
}


/*
 * OgaGenerator::GenerateNextToken
 * 
 * Runs one step of autoregressive generation.
 * Computes logits via forward pass and samples next token.
 */
void OgaGenerator::GenerateNextToken() {
    if (done || !inference_engine) {
        int32_t next_token = 42;
        current_tokens.push_back(next_token);
        if (current_tokens.size() >= static_cast<size_t>(params.max_length) || next_token == model->eos_token_id) {
            done = true;
        }
        return;
    }

    try {
        array logits = inference_engine->forward(current_tokens, params);
        int next_token = inference_engine->sample_token(logits, params);

        static int debug_count = 0;
        if (debug_count < 10) {
            std::cout << "[Generator] Token " << debug_count << ": ID=" << next_token;

            if (this->tokenizer) {
                if (this->tokenizer->use_sentencepiece && this->tokenizer->sp_processor) {
                    std::vector<int> token_vec = {next_token};
                    std::string decoded;
                    if (this->tokenizer->sp_processor->Decode(token_vec, &decoded).ok()) {
                        std::cout << ", text='" << decoded << "'";
                    }
                } else if (this->tokenizer->use_hf_tokenizer) {
                    auto it = this->tokenizer->reverse_vocab.find(next_token);
                    if (it != this->tokenizer->reverse_vocab.end()) {
                        std::cout << ", text='" << it->second << "'";
                    }
                }
            }

            array max_logits = max(logits);
            array argmax_logits = argmax(logits);
            float max_val = max_logits.item<float>();
            int max_idx = static_cast<int>(argmax_logits.item<int32_t>());
            std::cout << ", logits_max=" << max_val << " at " << max_idx;

            std::cout << std::endl;
            debug_count++;
        }

        current_tokens.push_back(next_token);

        if (current_tokens.size() >= static_cast<size_t>(params.max_length) || next_token == model->eos_token_id) {
            done = true;
        }

    } catch (const std::exception& e) {
        std::cerr << "[Generator] Inference error: " << e.what() << std::endl;
        int32_t next_token = 42;
        current_tokens.push_back(next_token);
        if (current_tokens.size() >= static_cast<size_t>(params.max_length) || next_token == model->eos_token_id) {
            done = true;
        }
    }
}


bool OgaGenerator::IsDone() const {
    return done;
}


const int32_t* OgaGenerator::GetSequenceData(int) const {
    return current_tokens.data();
}


size_t OgaGenerator::GetSequenceCount(int) const {
    return current_tokens.size();
}


/*
 * OgaGenerator::sample_token
 * 
 * Token sampling with temperature scaling.
 * Uses greedy decoding (argmax).
 */
int OgaGenerator::sample_token(const array& logits) {
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
