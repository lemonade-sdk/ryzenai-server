/*
 * mlx_backend.cpp
 * 
 * MLX backend implementation - wraps the MLX inference pipeline.
 */

#include <ryzenai/backend/mlx_backend.h>
#include <ryzenai/inference_engine.h>
#include <ryzenai/mlx/mlx_oga.h>
#include <ryzenai/mlx/gpu_utils.h>
#include <mlx/device.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>

namespace ryzenai {

namespace fs = std::filesystem;

MlxBackend::MlxBackend(BackendType type) : type_(type) {
    // Validate backend type
    if (type != BackendType::MLX_METAL && 
        type != BackendType::MLX_ROCM && 
        type != BackendType::MLX_CUDA) {
        throw std::runtime_error("Invalid backend type for MlxBackend");
    }
}

void MlxBackend::loadModel(const std::string& model_path) {
    model_path_ = resolveModelPath(model_path);

    std::cout << "[MlxBackend] Loading model from: " << model_path_ << std::endl;

    // Set the appropriate MLX device based on backend type
    // Return value is GPU index on success, -1 on failure
    if (ryzenai::mlx::GpuUtils::setMlxDeviceForBackend(type_) < 0) {
        throw std::runtime_error("Failed to set MLX device for backend type: " + getName());
    }

    // Create model using factory method
    model_ = MlxOgaModel::Create(model_path_.c_str());
    if (!model_) {
        throw std::runtime_error("Failed to create model");
    }

    // Set the backend type for this model
    model_->backend_type = type_;

    // Note: Model weights are loaded on the current default device
    // In MLX, arrays are created on the default device when loaded
    
    // Create tokenizer
    tokenizer_ = MlxOgaTokenizer::Create(*model_);
    if (!tokenizer_) {
        throw std::runtime_error("Failed to create tokenizer");
    }
    
    // Extract model name from path
    model_name_ = fs::path(model_path_).filename().string();
    
    // Load default generation params
    loadDefaultParams();
    
    std::cout << "[MlxBackend] Model loaded: " << model_name_ << std::endl;
    std::cout << "[MlxBackend] Vocab size: " << model_->vocab_size << std::endl;
    std::cout << "[MlxBackend] Hidden size: " << model_->hidden_size << std::endl;
    std::cout << "[MlxBackend] Layers: " << model_->num_hidden_layers << std::endl;
}

void MlxBackend::setContextSize(int ctx_size) {
    if (model_) {
        model_->max_context_length = ctx_size;
        std::cout << "[MlxBackend] Context size set to: " << ctx_size << " tokens" << std::endl;
    }
}

void MlxBackend::setKVFormat(KVCacheMode format) {
    if (model_) {
        model_->kv_format = format;
        const char* format_name = (format == KVCacheMode::INT8) ? "int8" : 
                                  (format == KVCacheMode::INT4) ? "int4" : "fp16";
        std::cout << "[MlxBackend] KV cache format set to: " << format_name << std::endl;
    }
}

std::string MlxBackend::getName() const {
    switch (type_) {
        case BackendType::MLX_METAL: return "MLX-Metal";
        case BackendType::MLX_ROCM: return "MLX-ROCm";
        case BackendType::MLX_CUDA: return "MLX-CUDA";
        default: return "MLX-Unknown";
    }
}

BackendCapabilities MlxBackend::getCapabilities() const {
    BackendCapabilities caps;
    caps.supports_streaming = true;
    caps.supports_quantization = true;
    caps.supports_kv_cache = true;
    caps.supports_tool_calls = true;
    caps.supports_thinking = true;
    caps.max_context_length = model_ ? model_->max_context_length : 4096;
    caps.supported_model_types = {"phi3", "qwen3", "llama", "mistral", "gemma"};
    return caps;
}

std::vector<int32_t> MlxBackend::encode(const std::string& text) {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    auto sequences = MlxOgaSequences::Create();
    tokenizer_->Encode(text.c_str(), *sequences);
    
    const int32_t* ids = sequences->SequenceData(0);
    size_t count = sequences->SequenceCount(0);
    
    return std::vector<int32_t>(ids, ids + count);
}

std::string MlxBackend::decode(const std::vector<int32_t>& tokens) {
    if (!tokenizer_ || tokens.empty()) {
        return "";
    }
    
    auto decoded = tokenizer_->Decode(tokens.data(), tokens.size());
    return std::string(decoded);
}

std::string MlxBackend::applyChatTemplate(const std::string& messages_json, const std::string& tools_json) {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    // Parse messages to determine template style
    json messages = json::parse(messages_json);
    std::ostringstream prompt;
    
    // Check if model uses Qwen/ChatML style
    bool is_qwen_style = (model_->model_type.find("qwen") != std::string::npos);
    
    if (!tools_json.empty()) {
        // Use OGA's built-in template for tools
        auto result = tokenizer_->ApplyChatTemplate(
            nullptr, messages_json.c_str(), tools_json.c_str(), true);
        return std::string(result);
    } 
    
    if (is_qwen_style) {
        // ChatML format: <|im_start|>role\ncontent<|im_end|>
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            prompt << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        }
        prompt << "<|im_start|>assistant\n";
    } else {
        // Use tokenizer's apply template
        auto result = tokenizer_->ApplyChatTemplate(
            nullptr, messages_json.c_str(), nullptr, true);
        return std::string(result);
    }
    
    return prompt.str();
}

int MlxBackend::countTokens(const std::string& text) {
    auto tokens = encode(text);
    return static_cast<int>(tokens.size());
}

std::string MlxBackend::complete(const std::string& prompt, const GenerationParams& params, 
                                  CompletionTimingData* out_timing) {
    if (!model_ || !tokenizer_) {
        throw std::runtime_error("Model not loaded");
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    auto tokenize_start = total_start;
    
    // Encode prompt
    auto sequences = MlxOgaSequences::Create();
    tokenizer_->Encode(prompt.c_str(), *sequences);
    
    auto tokenize_end = std::chrono::high_resolution_clock::now();
    
    const int32_t* input_ids_ptr = sequences->SequenceData(0);
    size_t input_ids_count = sequences->SequenceCount(0);
    std::vector<int32_t> input_ids(input_ids_ptr, input_ids_ptr + input_ids_count);
    
    // Validate input length against both:
    // 1. User-configured --ctx-size
    // 2. Hardware limit based on GPU's max buffer size
    //
    // Memory calculation for attention:
    //   attention_memory = num_heads * seq_len^2 * bytes_per_element
    //   bytes_per_element depends on KV format: FP16=2, INT8=1, INT4=0.5
    //   max_seq = sqrt(max_buffer / (num_heads * bytes_per_element))
    //
    // We use 50% of max_buffer to leave room for KV cache and other allocations
    int user_max_tokens = model_->max_context_length;
    int hardware_max_tokens = user_max_tokens;  // Default to user setting
    
    try {
        // Query GPU device info for max buffer length
        auto device_info = ::mlx::core::device_info(::mlx::core::Device(::mlx::core::Device::gpu, 0));
        auto it = device_info.find("max_buffer_length");
        if (it != device_info.end()) {
            size_t max_buffer = std::get<size_t>(it->second);
            // Use 50% of max buffer for attention, leave rest for KV cache and weights
            size_t available_for_attention = max_buffer / 2;
            
            // Bytes per element depends on KV format
            // FP16: 2 bytes, INT8: 1 byte, INT4: 0.5 bytes
            double bytes_per_element = 2.0;  // Default FP16
            const char* format_name = "FP16";
            if (model_->kv_format == KVCacheMode::INT8) {
                bytes_per_element = 1.0;
                format_name = "INT8";
            } else if (model_->kv_format == KVCacheMode::INT4) {
                bytes_per_element = 0.5;
                format_name = "INT4";
            }
            
            int num_heads = model_->num_attention_heads;
            // max_seq = sqrt(available / (num_heads * bytes_per_element))
            hardware_max_tokens = static_cast<int>(std::sqrt(
                static_cast<double>(available_for_attention) / (num_heads * bytes_per_element)
            ));
            
            // Log only once on first request
            static bool logged_once = false;
            if (!logged_once) {
                std::cout << "[MlxBackend] GPU max_buffer_length: " << (max_buffer / 1024 / 1024) << " MB, "
                          << "KV format: " << format_name << ", "
                          << "calculated safe max context: " << hardware_max_tokens << " tokens" << std::endl;
                logged_once = true;
            }
        }
    } catch (...) {
        // Could not get device info, use user setting
    }
    
    int effective_max_tokens = std::min(user_max_tokens, hardware_max_tokens);
    
    if (static_cast<int>(input_ids.size()) > effective_max_tokens) {
        std::ostringstream error;
        error << "Input too large: " << input_ids.size() << " tokens exceeds safe maximum of " 
              << effective_max_tokens << " tokens. ";
        if (hardware_max_tokens < user_max_tokens) {
            error << "Your GPU's max buffer (" << (hardware_max_tokens) 
                  << " token limit) is smaller than configured --ctx-size (" << user_max_tokens << "). ";
        }
        error << "Please reduce your input.";
        throw std::runtime_error(error.str());
    }
    
    // Setup generator params
    auto gen_params = MlxOgaGeneratorParams::Create(*model_);
    gen_params->SetSearchOption("max_length", static_cast<int>(input_ids.size()) + params.max_length);
    gen_params->SetSearchOption("temperature", params.temperature);
    gen_params->SetSearchOption("top_p", params.top_p);
    gen_params->SetSearchOption("top_k", static_cast<double>(params.top_k));
    gen_params->SetSearchOption("repetition_penalty", params.repetition_penalty);
    gen_params->SetSearchOptionBool("do_sample", params.do_sample);
    
    int eos_id = model_->GetEosId();
    gen_params->SetSearchOption("eos_token_id", eos_id);
    gen_params->SetSearchOption("pad_token_id", eos_id);
    
    // Create generator
    auto generator = MlxOgaGenerator::Create(*model_, *gen_params);
    generator->SetTokenizer(*tokenizer_);
    generator->AppendTokens(input_ids.data(), input_ids.size());
    
    // Get stop token IDs (CHAT_END only, not THINKING_END)
    std::vector<int32_t> stop_token_ids;
    for (const auto& tag : model_->additional_tags) {
        if (tag.type == SpecialTokenType::CHAT_END && tag.token_id >= 0) {
            stop_token_ids.push_back(tag.token_id);
        }
    }
    
    std::vector<int32_t> generated_tokens;
    generated_tokens.reserve(params.max_length);
    
    auto prefill_start = std::chrono::high_resolution_clock::now();
    bool first_token = true;
    auto decode_start = prefill_start;
    
    // Generate tokens
    while (!generator->IsDone()) {
        generator->GenerateNextToken();
        
        if (first_token) {
            decode_start = std::chrono::high_resolution_clock::now();
            first_token = false;
        }
        
        const int32_t* seq = generator->GetSequenceData(0);
        size_t seq_count = generator->GetSequenceCount(0);
        
        if (seq_count > 0) {
            int32_t new_token = seq[seq_count - 1];
            
            // Check EOS
            if (model_->IsEos(new_token)) break;
            
            // Check stop tokens
            bool is_stop = false;
            for (int32_t stop_id : stop_token_ids) {
                if (new_token == stop_id) {
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
    const int32_t* output_ptr = generator->GetSequenceData(0);
    size_t output_count = generator->GetSequenceCount(0);
    
    std::string result;
    if (output_count > input_ids.size()) {
        size_t decode_count = output_count - input_ids.size();
        if (decode_count > 0 && output_ptr[output_count-1] == eos_id) {
            decode_count--;
        }
        auto decoded = tokenizer_->Decode(output_ptr + input_ids.size(), decode_count);
        result = std::string(decoded);
    }
    
    return result;
}

void MlxBackend::streamComplete(const std::string& prompt, const GenerationParams& params, 
                                 StreamCallback callback) {
    if (!model_ || !tokenizer_) {
        throw std::runtime_error("Model not loaded");
    }
    
    // Encode prompt
    auto sequences = MlxOgaSequences::Create();
    tokenizer_->Encode(prompt.c_str(), *sequences);
    
    const int32_t* input_ids_ptr = sequences->SequenceData(0);
    size_t input_ids_count = sequences->SequenceCount(0);
    std::vector<int32_t> input_ids(input_ids_ptr, input_ids_ptr + input_ids_count);
    
    // Validate input length against both:
    // 1. User-configured --ctx-size
    // 2. Hardware limit based on GPU's max buffer size
    int user_max_tokens = model_->max_context_length;
    int hardware_max_tokens = user_max_tokens;  // Default to user setting
    
    try {
        // Query GPU device info for max buffer length
        auto device_info = ::mlx::core::device_info(::mlx::core::Device(::mlx::core::Device::gpu, 0));
        auto it = device_info.find("max_buffer_length");
        if (it != device_info.end()) {
            size_t max_buffer = std::get<size_t>(it->second);
            // Use 50% of max buffer for attention, leave rest for KV cache and weights
            size_t available_for_attention = max_buffer / 2;
            
            // Bytes per element depends on KV format
            // FP16: 2 bytes, INT8: 1 byte, INT4: 0.5 bytes
            double bytes_per_element = 2.0;  // Default FP16
            if (model_->kv_format == KVCacheMode::INT8) {
                bytes_per_element = 1.0;
            } else if (model_->kv_format == KVCacheMode::INT4) {
                bytes_per_element = 0.5;
            }
            
            int num_heads = model_->num_attention_heads;
            // max_seq = sqrt(available / (num_heads * bytes_per_element))
            hardware_max_tokens = static_cast<int>(std::sqrt(
                static_cast<double>(available_for_attention) / (num_heads * bytes_per_element)
            ));
        }
    } catch (...) {
        // Could not get device info, use user setting
    }
    
    int effective_max_tokens = std::min(user_max_tokens, hardware_max_tokens);
    
    if (static_cast<int>(input_ids.size()) > effective_max_tokens) {
        std::ostringstream error;
        error << "Input too large: " << input_ids.size() << " tokens exceeds safe maximum of " 
              << effective_max_tokens << " tokens. ";
        if (hardware_max_tokens < user_max_tokens) {
            error << "Your GPU's max buffer (" << (hardware_max_tokens) 
                  << " token limit) is smaller than configured --ctx-size (" << user_max_tokens << "). ";
        }
        error << "Please reduce your input.";
        throw std::runtime_error(error.str());
    }
    
    // Setup generator params
    auto gen_params = MlxOgaGeneratorParams::Create(*model_);
    gen_params->SetSearchOption("max_length", static_cast<int>(input_ids.size()) + params.max_length);
    gen_params->SetSearchOption("temperature", params.temperature);
    gen_params->SetSearchOption("top_p", params.top_p);
    gen_params->SetSearchOption("top_k", static_cast<double>(params.top_k));
    gen_params->SetSearchOption("repetition_penalty", params.repetition_penalty);
    gen_params->SetSearchOptionBool("do_sample", params.do_sample);
    
    int eos_id = model_->GetEosId();
    gen_params->SetSearchOption("eos_token_id", eos_id);
    gen_params->SetSearchOption("pad_token_id", eos_id);
    
    // Create generator
    auto generator = MlxOgaGenerator::Create(*model_, *gen_params);
    generator->SetTokenizer(*tokenizer_);
    generator->AppendTokens(input_ids.data(), input_ids.size());
    
    // Get stop token IDs
    std::vector<int32_t> stop_token_ids;
    for (const auto& tag : model_->additional_tags) {
        if (tag.type == SpecialTokenType::CHAT_END && tag.token_id >= 0) {
            stop_token_ids.push_back(tag.token_id);
        }
    }
    
    auto tokenizer_stream = MlxOgaTokenizerStream::Create(*tokenizer_);
    std::string accumulated;
    bool should_stop = false;
    
    while (!generator->IsDone() && !should_stop) {
        generator->GenerateNextToken();
        
        const int32_t* seq = generator->GetSequenceData(0);
        size_t seq_count = generator->GetSequenceCount(0);
        int32_t new_token = seq[seq_count - 1];
        
        // Check EOS
        if (model_->IsEos(new_token)) break;
        
        // Check stop tokens
        for (int32_t stop_id : stop_token_ids) {
            if (new_token == stop_id) {
                should_stop = true;
                break;
            }
        }
        if (should_stop) break;
        
        // Decode and stream
        const char* decoded = tokenizer_stream->Decode(new_token);
        if (decoded && decoded[0] != '\0') {
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
                if (!callback(token_str, generator->IsDone())) {
                    break;  // Client disconnected
                }
            }
        }
    }
}

std::string MlxBackend::getModelName() const {
    return model_name_;
}

std::string MlxBackend::getModelType() const {
    return model_ ? model_->model_type : "";
}

int MlxBackend::getMaxContextLength() const {
    return model_ ? model_->max_context_length : 4096;
}

int MlxBackend::getEosTokenId() const {
    return model_ ? model_->GetEosId() : 2;
}

bool MlxBackend::isEos(int32_t token_id) const {
    return model_ ? model_->IsEos(token_id) : (token_id == 2);
}

const std::vector<AdditionalToken>& MlxBackend::getSpecialTokens() const {
    static std::vector<AdditionalToken> empty;
    return model_ ? model_->additional_tags : empty;
}

GenerationParams MlxBackend::getDefaultParams() const {
    return default_params_;
}

void MlxBackend::loadDefaultParams() {
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
        std::cerr << "[MlxBackend] Failed to load default params: " << e.what() << std::endl;
    }
}

std::string MlxBackend::resolveModelPath(const std::string& path) {
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

// Factory registration
void registerMlxBackend() {
    auto& registry = BackendRegistry::instance();
    
    registry.registerBackend(BackendType::MLX_METAL, [](const std::string& model_path) {
        auto backend = std::make_unique<MlxBackend>(BackendType::MLX_METAL);
        backend->loadModel(model_path);
        return backend;
    });
    
    // Note: ROCm and CUDA variants would use same implementation,
    // but MLX would initialize different compute backend internally
    registry.registerBackend(BackendType::MLX_ROCM, [](const std::string& model_path) {
        auto backend = std::make_unique<MlxBackend>(BackendType::MLX_ROCM);
        backend->loadModel(model_path);
        return backend;
    });
    
    registry.registerBackend(BackendType::MLX_CUDA, [](const std::string& model_path) {
        auto backend = std::make_unique<MlxBackend>(BackendType::MLX_CUDA);
        backend->loadModel(model_path);
        return backend;
    });
}

} // namespace ryzenai
