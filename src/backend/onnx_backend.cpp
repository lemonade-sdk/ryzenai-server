/*
 * onnx_backend.cpp
 * 
 * ONNX Runtime GenAI backend implementation for Ryzen AI.
 * Only compiled when RYZENAI_ON is defined (Windows with Ryzen AI).
 */

#ifdef RYZENAI_ON

#include <ryzenai/backend/onnx_backend.h>
#include <ort_genai.h>
#include <ort_genai_c.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <json.hpp>

namespace ryzenai {

namespace fs = std::filesystem;
using json = nlohmann::json;

OnnxBackend::OnnxBackend(BackendType type) 
    : type_(type)
    , model_(nullptr, OgaDestroyModel)
    , tokenizer_(nullptr, OgaDestroyTokenizer) {
    
    // Validate backend type
    if (type != BackendType::ONNX_RYZENAI && 
        type != BackendType::ONNX_DIRECTML &&
        type != BackendType::ONNX_CPU) {
        throw std::runtime_error("Invalid backend type for OnnxBackend");
    }
}

void OnnxBackend::loadModel(const std::string& model_path) {
    model_path_ = resolveModelPath(model_path);
    
    std::cout << "[OnnxBackend] Loading model from: " << model_path_ << std::endl;
    
    // Create model using ONNX Runtime GenAI C API
    OgaModel* raw_model = nullptr;
    OgaResult* result = OgaCreateModel(model_path_.c_str(), &raw_model);
    if (result) {
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to load model: " + error);
    }
    model_.reset(raw_model);
    
    // Create tokenizer
    OgaTokenizer* raw_tokenizer = nullptr;
    result = OgaCreateTokenizer(model_.get(), &raw_tokenizer);
    if (result) {
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to create tokenizer: " + error);
    }
    tokenizer_.reset(raw_tokenizer);
    
    // Extract model name from path
    model_name_ = fs::path(model_path_).filename().string();
    
    // Load configuration
    loadConfig();
    loadDefaultParams();
    loadSpecialTokens();
    
    std::cout << "[OnnxBackend] Model loaded: " << model_name_ << std::endl;
}

std::string OnnxBackend::getName() const {
    switch (type_) {
        case BackendType::ONNX_RYZENAI: return "ONNX-RyzenAI";
        case BackendType::ONNX_DIRECTML: return "ONNX-DirectML";
        case BackendType::ONNX_CPU: return "ONNX-CPU";
        default: return "ONNX-Unknown";
    }
}

BackendCapabilities OnnxBackend::getCapabilities() const {
    BackendCapabilities caps;
    caps.supports_streaming = true;
    caps.supports_quantization = true;
    caps.supports_kv_cache = true;
    caps.supports_tool_calls = true;
    caps.supports_thinking = true;
    caps.max_context_length = max_context_length_;
    caps.supported_model_types = {"phi3", "qwen3", "llama", "mistral"};
    return caps;
}

std::vector<int32_t> OnnxBackend::encode(const std::string& text) {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    OgaSequences* sequences = nullptr;
    OgaResult* result = OgaCreateSequences(&sequences);
    if (result) {
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to create sequences");
    }
    
    result = OgaTokenizerEncode(tokenizer_.get(), text.c_str(), sequences);
    if (result) {
        OgaDestroySequences(sequences);
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to encode: " + error);
    }
    
    size_t count = 0;
    const int32_t* ids = OgaSequencesGetSequenceData(sequences, 0);
    OgaSequencesGetSequenceCount(sequences, &count);
    
    std::vector<int32_t> result_ids(ids, ids + count);
    OgaDestroySequences(sequences);
    
    return result_ids;
}

std::string OnnxBackend::decode(const std::vector<int32_t>& tokens) {
    if (!tokenizer_ || tokens.empty()) {
        return "";
    }
    
    const char* decoded = nullptr;
    OgaResult* result = OgaTokenizerDecode(tokenizer_.get(), tokens.data(), tokens.size(), &decoded);
    if (result) {
        OgaDestroyResult(result);
        return "";
    }
    
    std::string text(decoded);
    // Note: decoded string is owned by tokenizer, don't free
    return text;
}

std::string OnnxBackend::applyChatTemplate(const std::string& messages_json, const std::string& tools_json) {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    const char* template_str = chat_template_.empty() ? nullptr : chat_template_.c_str();
    const char* tools_str = tools_json.empty() ? nullptr : tools_json.c_str();
    
    const char* result_str = nullptr;
    OgaResult* result = OgaTokenizerApplyChatTemplate(
        tokenizer_.get(), template_str, messages_json.c_str(), tools_str, true, &result_str);
    
    if (result) {
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to apply chat template: " + error);
    }
    
    return std::string(result_str);
}

int OnnxBackend::countTokens(const std::string& text) {
    auto tokens = encode(text);
    return static_cast<int>(tokens.size());
}

std::string OnnxBackend::complete(const std::string& prompt, const GenerationParams& params, 
                                   CompletionTimingData* out_timing) {
    if (!model_ || !tokenizer_) {
        throw std::runtime_error("Model not loaded");
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    auto tokenize_start = total_start;
    
    // Encode prompt
    std::vector<int32_t> input_ids = encode(prompt);
    input_ids = truncatePrompt(input_ids, max_context_length_ - params.max_length);
    
    auto tokenize_end = std::chrono::high_resolution_clock::now();
    
    // Create generator params
    OgaGeneratorParams* gen_params = nullptr;
    OgaResult* result = OgaCreateGeneratorParams(model_.get(), &gen_params);
    if (result) {
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to create generator params: " + error);
    }
    
    // Set search options
    OgaGeneratorParamsSetSearchNumber(gen_params, "max_length", input_ids.size() + params.max_length);
    OgaGeneratorParamsSetSearchNumber(gen_params, "temperature", params.temperature);
    OgaGeneratorParamsSetSearchNumber(gen_params, "top_p", params.top_p);
    OgaGeneratorParamsSetSearchNumber(gen_params, "top_k", params.top_k);
    OgaGeneratorParamsSetSearchNumber(gen_params, "repetition_penalty", params.repetition_penalty);
    OgaGeneratorParamsSetSearchBool(gen_params, "do_sample", params.do_sample);
    OgaGeneratorParamsSetSearchNumber(gen_params, "eos_token_id", eos_token_id_);
    OgaGeneratorParamsSetSearchNumber(gen_params, "pad_token_id", eos_token_id_);
    
    // Create generator
    OgaGenerator* generator = nullptr;
    result = OgaCreateGenerator(model_.get(), gen_params, &generator);
    if (result) {
        OgaDestroyGeneratorParams(gen_params);
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to create generator: " + error);
    }
    
    // Append input tokens
    result = OgaGenerator_AppendTokens(generator, input_ids.data(), input_ids.size());
    if (result) {
        OgaDestroyGenerator(generator);
        OgaDestroyGeneratorParams(gen_params);
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to append tokens: " + error);
    }
    
    auto prefill_start = std::chrono::high_resolution_clock::now();
    bool first_token = true;
    auto decode_start = prefill_start;
    
    // Generate tokens
    std::vector<int32_t> generated_tokens;
    generated_tokens.reserve(params.max_length);
    
    while (!OgaGenerator_IsDone(generator)) {
        result = OgaGenerator_GenerateNextToken(generator);
        if (result) {
            OgaDestroyGenerator(generator);
            OgaDestroyGeneratorParams(gen_params);
            std::string error = OgaResultGetError(result);
            OgaDestroyResult(result);
            throw std::runtime_error("Failed to generate token: " + error);
        }
        
        if (first_token) {
            decode_start = std::chrono::high_resolution_clock::now();
            first_token = false;
        }
        
        size_t seq_count = 0;
        OgaGenerator_GetSequenceCount(generator, &seq_count);
        const int32_t* seq = OgaGenerator_GetSequenceData(generator, 0);
        
        if (seq_count > 0) {
            int32_t new_token = seq[seq_count - 1];
            
            // Check EOS
            if (new_token == eos_token_id_) break;
            
            // Check special stop tokens
            bool is_stop = false;
            for (const auto& token : special_tokens_) {
                if (token.type == SpecialTokenType::CHAT_END && 
                    token.token_id == new_token) {
                    is_stop = true;
                    break;
                }
            }
            if (is_stop) break;
            
            generated_tokens.push_back(new_token);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Fill timing data
    if (out_timing) {
        out_timing->token_count = static_cast<int>(generated_tokens.size());
        out_timing->tokenize_ms = std::chrono::duration<double, std::milli>(tokenize_end - tokenize_start).count();
        out_timing->prefill_ms = std::chrono::duration<double, std::milli>(decode_start - prefill_start).count();
        out_timing->decode_ms = std::chrono::duration<double, std::milli>(end_time - decode_start).count();
        out_timing->total_time_ms = std::chrono::duration<double, std::milli>(end_time - total_start).count();
        
        if (out_timing->decode_ms > 0) {
            out_timing->tps = out_timing->token_count * 1000.0 / out_timing->decode_ms;
        }
    }
    
    // Decode result
    std::string result_text = decode(generated_tokens);
    
    // Cleanup
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(gen_params);
    
    return result_text;
}

void OnnxBackend::streamComplete(const std::string& prompt, const GenerationParams& params, 
                                  StreamCallback callback) {
    if (!model_ || !tokenizer_) {
        throw std::runtime_error("Model not loaded");
    }
    
    // Encode prompt
    std::vector<int32_t> input_ids = encode(prompt);
    input_ids = truncatePrompt(input_ids, max_context_length_ - params.max_length);
    
    // Create generator params
    OgaGeneratorParams* gen_params = nullptr;
    OgaResult* result = OgaCreateGeneratorParams(model_.get(), &gen_params);
    if (result) {
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to create generator params: " + error);
    }
    
    // Set search options
    OgaGeneratorParamsSetSearchNumber(gen_params, "max_length", input_ids.size() + params.max_length);
    OgaGeneratorParamsSetSearchNumber(gen_params, "temperature", params.temperature);
    OgaGeneratorParamsSetSearchNumber(gen_params, "top_p", params.top_p);
    OgaGeneratorParamsSetSearchNumber(gen_params, "top_k", params.top_k);
    OgaGeneratorParamsSetSearchNumber(gen_params, "repetition_penalty", params.repetition_penalty);
    OgaGeneratorParamsSetSearchBool(gen_params, "do_sample", params.do_sample);
    OgaGeneratorParamsSetSearchNumber(gen_params, "eos_token_id", eos_token_id_);
    OgaGeneratorParamsSetSearchNumber(gen_params, "pad_token_id", eos_token_id_);
    
    // Create generator
    OgaGenerator* generator = nullptr;
    result = OgaCreateGenerator(model_.get(), gen_params, &generator);
    if (result) {
        OgaDestroyGeneratorParams(gen_params);
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to create generator: " + error);
    }
    
    // Append input tokens
    result = OgaGenerator_AppendTokens(generator, input_ids.data(), input_ids.size());
    if (result) {
        OgaDestroyGenerator(generator);
        OgaDestroyGeneratorParams(gen_params);
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to append tokens: " + error);
    }
    
    // Create tokenizer stream for incremental decoding
    OgaTokenizerStream* stream = nullptr;
    result = OgaCreateTokenizerStream(tokenizer_.get(), &stream);
    if (result) {
        OgaDestroyGenerator(generator);
        OgaDestroyGeneratorParams(gen_params);
        std::string error = OgaResultGetError(result);
        OgaDestroyResult(result);
        throw std::runtime_error("Failed to create tokenizer stream: " + error);
    }
    
    std::string accumulated;
    bool should_stop = false;
    
    while (!OgaGenerator_IsDone(generator) && !should_stop) {
        result = OgaGenerator_GenerateNextToken(generator);
        if (result) {
            OgaDestroyTokenizerStream(stream);
            OgaDestroyGenerator(generator);
            OgaDestroyGeneratorParams(gen_params);
            std::string error = OgaResultGetError(result);
            OgaDestroyResult(result);
            throw std::runtime_error("Failed to generate token: " + error);
        }
        
        size_t seq_count = 0;
        OgaGenerator_GetSequenceCount(generator, &seq_count);
        const int32_t* seq = OgaGenerator_GetSequenceData(generator, 0);
        int32_t new_token = seq[seq_count - 1];
        
        // Check EOS
        if (new_token == eos_token_id_) break;
        
        // Check special stop tokens
        for (const auto& token : special_tokens_) {
            if (token.type == SpecialTokenType::CHAT_END && 
                token.token_id == new_token) {
                should_stop = true;
                break;
            }
        }
        if (should_stop) break;
        
        // Decode incrementally
        const char* decoded = nullptr;
        result = OgaTokenizerStreamDecode(stream, new_token, &decoded);
        if (result || !decoded) continue;
        
        std::string token_str(decoded);
        accumulated += token_str;
        
        // Check stop sequences
        for (const auto& stop_seq : params.stop_sequences) {
            if (accumulated.find(stop_seq) != std::string::npos) {
                should_stop = true;
                break;
            }
        }
        
        if (!should_stop) {
            if (!callback(token_str, OgaGenerator_IsDone(generator))) {
                break;  // Client disconnected
            }
        }
    }
    
    // Cleanup
    OgaDestroyTokenizerStream(stream);
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(gen_params);
}

std::string OnnxBackend::getModelName() const {
    return model_name_;
}

std::string OnnxBackend::getModelType() const {
    return model_type_;
}

int OnnxBackend::getMaxContextLength() const {
    return max_context_length_;
}

int OnnxBackend::getEosTokenId() const {
    return eos_token_id_;
}

bool OnnxBackend::isEos(int32_t token_id) const {
    return token_id == eos_token_id_;
}

const std::vector<AdditionalToken>& OnnxBackend::getSpecialTokens() const {
    return special_tokens_;
}

GenerationParams OnnxBackend::getDefaultParams() const {
    return default_params_;
}

void OnnxBackend::loadConfig() {
    std::string config_path = model_path_ + "/genai_config.json";
    if (!fs::exists(config_path)) return;
    
    try {
        std::ifstream file(config_path);
        json config = json::parse(file);
        
        if (config.contains("model") && config["model"].contains("context_length")) {
            max_context_length_ = config["model"]["context_length"];
        }
        
        if (config.contains("model") && config["model"].contains("eos_token_id")) {
            eos_token_id_ = config["model"]["eos_token_id"];
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[OnnxBackend] Failed to load config: " << e.what() << std::endl;
    }
    
    // Load chat template from tokenizer_config.json
    std::string tokenizer_config_path = model_path_ + "/tokenizer_config.json";
    if (fs::exists(tokenizer_config_path)) {
        try {
            std::ifstream file(tokenizer_config_path);
            json config = json::parse(file);
            
            if (config.contains("chat_template") && config["chat_template"].is_string()) {
                chat_template_ = config["chat_template"];
            }
            
            if (config.contains("model_type") && config["model_type"].is_string()) {
                model_type_ = config["model_type"];
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[OnnxBackend] Failed to load tokenizer config: " << e.what() << std::endl;
        }
    }
}

void OnnxBackend::loadDefaultParams() {
    std::string config_path = model_path_ + "/genai_config.json";
    if (!fs::exists(config_path)) return;
    
    try {
        std::ifstream file(config_path);
        json config = json::parse(file);
        
        if (config.contains("search")) {
            json search = config["search"];
            has_default_params_ = true;
            
            if (search.contains("temperature") && search["temperature"].is_number()) {
                default_params_.temperature = search["temperature"];
            }
            if (search.contains("top_p") && search["top_p"].is_number()) {
                default_params_.top_p = search["top_p"];
            }
            if (search.contains("top_k") && search["top_k"].is_number()) {
                default_params_.top_k = search["top_k"];
            }
            if (search.contains("repetition_penalty") && search["repetition_penalty"].is_number()) {
                default_params_.repetition_penalty = search["repetition_penalty"];
            }
            if (search.contains("do_sample") && search["do_sample"].is_boolean()) {
                default_params_.do_sample = search["do_sample"];
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[OnnxBackend] Failed to load default params: " << e.what() << std::endl;
    }
}

void OnnxBackend::loadSpecialTokens() {
    std::string tokenizer_config_path = model_path_ + "/tokenizer_config.json";
    if (!fs::exists(tokenizer_config_path)) return;
    
    try {
        std::ifstream file(tokenizer_config_path);
        json config = json::parse(file);
        
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
                    } else if (content == "<|im_end|>" || content == "<|eot_id|>" || 
                               content == "<|end_of_turn|>" || content == "<|end|>") {
                        type = SpecialTokenType::CHAT_END;
                    }
                    
                    if (type != SpecialTokenType::UNKNOWN) {
                        AdditionalToken token;
                        token.content = content;
                        token.type = type;
                        try {
                            token.token_id = std::stoi(token_id_str);
                        } catch (...) {
                            token.token_id = -1;
                        }
                        special_tokens_.push_back(token);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[OnnxBackend] Failed to load special tokens: " << e.what() << std::endl;
    }
    
    // Add fallback defaults if no tokens found
    if (special_tokens_.empty()) {
        special_tokens_.push_back({"<think>", SpecialTokenType::THINKING_START, -1});
        special_tokens_.push_back({"</think>", SpecialTokenType::THINKING_END, -1});
        special_tokens_.push_back({"<|im_end|>", SpecialTokenType::CHAT_END, -1});
    }
}

std::string OnnxBackend::resolveModelPath(const std::string& path) {
    // Handle Hugging Face cache structure
    std::string snapshots_dir = path + "/snapshots";
    if (fs::exists(snapshots_dir) && fs::is_directory(snapshots_dir)) {
        for (const auto& entry : fs::directory_iterator(snapshots_dir)) {
            if (entry.is_directory()) {
                return entry.path().string();
            }
        }
    }
    return path;
}

std::vector<int32_t> OnnxBackend::truncatePrompt(const std::vector<int32_t>& input_ids, int max_length) {
    if (static_cast<int>(input_ids.size()) <= max_length) {
        return input_ids;
    }
    
    // Truncate from the beginning to keep most recent context
    size_t truncate_amount = input_ids.size() - max_length;
    std::cout << "[OnnxBackend] Truncating " << truncate_amount << " tokens from beginning" << std::endl;
    
    return std::vector<int32_t>(input_ids.begin() + truncate_amount, input_ids.end());
}

// Factory registration
void registerOnnxBackend() {
    auto& registry = BackendRegistry::instance();
    
    registry.registerBackend(BackendType::ONNX_RYZENAI, [](const std::string& model_path) {
        auto backend = std::make_unique<OnnxBackend>(BackendType::ONNX_RYZENAI);
        backend->loadModel(model_path);
        return backend;
    });
    
    registry.registerBackend(BackendType::ONNX_DIRECTML, [](const std::string& model_path) {
        auto backend = std::make_unique<OnnxBackend>(BackendType::ONNX_DIRECTML);
        backend->loadModel(model_path);
        return backend;
    });
    
    registry.registerBackend(BackendType::ONNX_CPU, [](const std::string& model_path) {
        auto backend = std::make_unique<OnnxBackend>(BackendType::ONNX_CPU);
        backend->loadModel(model_path);
        return backend;
    });
}

} // namespace ryzenai

#endif // RYZENAI_ON
