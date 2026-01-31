#include "ryzenai/inference_engine.h"
#ifdef RYZENAI_ON
#include <ort_genai.h>
#include <ort_genai_c.h>
#elif MLX_ON
//MacOS Specific Enablement
#include "ryzenai/mlx/mlx_oga.h"
#endif
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <chrono>

namespace ryzenai {

namespace fs = std::filesystem;

InferenceEngine::InferenceEngine(const std::string& model_path, const std::string& mode, 
                                   const OptimizationSettings& opt)
    : execution_mode_(mode), ctx_size_(opt.ctx_size) {

    std::cout << "[InferenceEngine] Initializing with model: " << model_path << std::endl;
    std::cout << "[InferenceEngine] Optimization settings:" << std::endl;
    std::cout << "  - Context size: " << opt.ctx_size << " tokens" << std::endl;
    std::cout << "  - Repetition lookback: " << opt.repetition_lookback << " tokens" << std::endl;
    std::cout << "  - KV cache: " << (opt.kv_cache ? "enabled" : "disabled") << std::endl;
    std::cout << "  - Prefill chunk: " << opt.prefill_chunk << " tokens" << std::endl;
    std::cout << "[InferenceEngine] Execution mode: " << mode << std::endl;
    
    // Resolve model path (handles Hugging Face cache structure)
    model_path_ = resolveModelPath(model_path);
    if (model_path_ != model_path) {
        std::cout << "[InferenceEngine] Resolved to: " << model_path_ << std::endl;
    }
    
    // Validate model directory
    if (!validateModelDirectory(model_path_)) {
        throw std::runtime_error("Invalid model directory: " + model_path_);
    }
    
    // Detect Ryzen AI version and load config
    loadRaiConfig();
    
    // Setup execution provider
    setupExecutionProvider();
    
    // Load the model
    loadModel();
    
    // Extract model name from path
    model_name_ = fs::path(model_path_).filename().string();
    
    // Load default generation params from genai_config.json
    std::string config_path = model_path_ + "/genai_config.json";
    if (fs::exists(config_path)) {
        try {
            std::ifstream file(config_path);
            json config = json::parse(file);
            
            if (config.contains("search")) {
                json search = config["search"];
                has_search_config_ = true;
                
                // Load defaults from search config (matching Python implementation)
                // NOTE: search.max_length from genai_config.json means TOTAL sequence length,
                // but default_params_.max_length is used as "max NEW tokens" in our code.
                // We intentionally DON'T load search.max_length here to avoid semantic confusion.
                // The user's max_tokens parameter will be used directly as max new tokens.
                
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
                
                std::cout << "[InferenceEngine] Loaded search config from genai_config.json" << std::endl;
                std::cout << "  - temperature: " << default_params_.temperature << std::endl;
                std::cout << "  - top_p: " << default_params_.top_p << std::endl;
                std::cout << "  - top_k: " << default_params_.top_k << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] Failed to load search config from genai_config.json: " << e.what() << std::endl;
        }
    }
    
    std::cout << "[InferenceEngine] Model loaded successfully: " << model_name_ << std::endl;
    std::cout << "[InferenceEngine] Max prompt length: " << max_prompt_length_ << " tokens" << std::endl;
}

InferenceEngine::~InferenceEngine() {
    std::cout << "[InferenceEngine] Shutting down" << std::endl;
}

const std::vector<AdditionalToken>& InferenceEngine::getAdditionalTags() const {
#ifdef MLX_ON
    // For MLX backend, use the model's additional_tags which were loaded from config files
    return model_->additional_tags;
#else
    // For Onyx/RYZENAI backend, use the fallback tags
    return fallback_additional_tags_;
#endif
}

GenerationParams InferenceEngine::getDefaultParams() const {
    return default_params_;
}

std::string InferenceEngine::applyChatTemplate(const std::string& messages_json, const std::string& tools_json) {
    // Parse messages
    json messages = json::parse(messages_json);
    std::ostringstream prompt;
    
    // Check if we have a Qwen-style chat template (contains <|im_start|>)
    bool is_qwen_style = !chat_template_.empty() && 
                         (chat_template_.find("<|im_start|>") != std::string::npos ||
                          chat_template_.find("\\u003c|im_start|\\u003e") != std::string::npos);
    
    // If tools are provided, always use OGA's built-in template (it handles tools properly)
    if (!tools_json.empty()) {
        try {
            const char* template_str = chat_template_.empty() ? nullptr : chat_template_.c_str();
            const char* tools_str = tools_json.c_str();
            
            auto result = tokenizer_->ApplyChatTemplate(
                template_str,
                messages_json.c_str(),
                tools_str,
                true
            );
            
            std::cout << "[InferenceEngine] Applied chat template with tools" << std::endl;
            return std::string(result);
            
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to apply chat template with tools: " << e.what() << std::endl;
            throw;
        }
    } else if (is_qwen_style) {
        // Use Qwen/ChatML format: <|im_start|>role\ncontent<|im_end|>\n
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            
            prompt << "<|im_start|>" << role << "\n"
                   << content << "<|im_end|>\n";
        }
        
        // Add generation prompt for assistant
        prompt << "<|im_start|>assistant\n";
        
        std::cout << "[InferenceEngine] Applied Qwen/ChatML template" << std::endl;
    } else {
        // Try using the OGA's built-in chat template
        try {
            const char* template_str = chat_template_.empty() ? nullptr : chat_template_.c_str();
            
            auto result = tokenizer_->ApplyChatTemplate(
                template_str,
                messages_json.c_str(),
                nullptr,
                true
            );
            
            return std::string(result);
            
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] OGA chat template failed: " << e.what() << std::endl;
            std::cerr << "[WARNING] Using simple fallback template" << std::endl;
            
            // Simple fallback template
            prompt.str("");  // Clear
            for (const auto& msg : messages) {
                std::string role = msg.value("role", "user");
                std::string content = msg.value("content", "");
                
                if (role == "system") {
                    prompt << "System: " << content << "\n\n";
                } else if (role == "user") {
                    prompt << "User: " << content << "\n\n";
                } else if (role == "assistant") {
                    prompt << "Assistant: " << content << "\n\n";
                }
            }
            
            prompt << "Assistant: ";
        }
    }
    
    return prompt.str();
}

std::string InferenceEngine::resolveModelPath(const std::string& path) {
    // If path has a "snapshots" subdirectory (Hugging Face cache structure),
    // automatically find the latest snapshot
    std::string snapshots_dir = path + "/snapshots";
    if (fs::exists(snapshots_dir) && fs::is_directory(snapshots_dir)) {
        std::cout << "[InferenceEngine] Detected Hugging Face cache structure, looking for snapshot..." << std::endl;
        
        // Find the first (and usually only) snapshot directory
        for (const auto& entry : fs::directory_iterator(snapshots_dir)) {
            if (entry.is_directory()) {
                std::string snapshot_path = entry.path().string();
                std::cout << "[InferenceEngine] Found snapshot: " << snapshot_path << std::endl;
                return snapshot_path;
            }
        }
        
        std::cerr << "[ERROR] No snapshot found in: " << snapshots_dir << std::endl;
        return path;
    }
    
    // Otherwise, use the path as-is
    return path;
}

bool InferenceEngine::validateModelDirectory(const std::string& path) {
    if (!fs::exists(path) || !fs::is_directory(path)) {
        std::cerr << "[ERROR] Model path does not exist or is not a directory: " << path << std::endl;
        return false;
    }

    // Check for required files
    std::string onnx_config_path = path + "/genai_config.json";
    std::string mlx_config_path = path + "/config.json";

    bool has_onnx_config = fs::exists(onnx_config_path);
    bool has_mlx_config = fs::exists(mlx_config_path);

#ifdef MLX_ON
    // MLX backend: accept either ONNX or MLX config files
    if (!has_onnx_config && !has_mlx_config) {
        std::cerr << "[ERROR] Required config file not found. Expected either:" << std::endl;
        std::cerr << "[ERROR]   - " << onnx_config_path << " (ONNX models)" << std::endl;
        std::cerr << "[ERROR]   - " << mlx_config_path << " (MLX models)" << std::endl;
        return false;
    }
#else
    // ONNX/RyzenAI backend: require ONNX config
    if (!has_onnx_config) {
        std::cerr << "[ERROR] Required file not found: " << onnx_config_path << std::endl;
        return false;
    }
#endif

    return true;
}

std::string InferenceEngine::detectRyzenAIVersion() {
    // Check for Ryzen AI 1.6.0 installation
    std::string ryzenai_path_16 = "C:/Program Files/RyzenAI/1.6.0";
    if (fs::exists(ryzenai_path_16)) {
        return "1.6.0";
    }
    
    // Check for 1.5.0
    std::string ryzenai_path_15 = "C:/Program Files/RyzenAI/1.5.0";
    if (fs::exists(ryzenai_path_15)) {
        return "1.5.0";
    }
    
    // Check environment variable
    const char* version_env = std::getenv("RYZENAI_VERSION");
    if (version_env) {
        return std::string(version_env);
    }
    
    // Default to 1.6.0
    return "1.6.0";
}

void InferenceEngine::loadRaiConfig() {
    // Detect Ryzen AI version
    ryzenai_version_ = detectRyzenAIVersion();
    std::cout << "[InferenceEngine] Ryzen AI version: " << ryzenai_version_ << std::endl;
    
    // Load rai_config.json if it exists
    std::string rai_config_path = model_path_ + "/rai_config.json";
    if (fs::exists(rai_config_path)) {
        try {
            std::ifstream file(rai_config_path);
            json config = json::parse(file);
            
            if (config.contains("max_prompt_length") && 
                config["max_prompt_length"].contains(ryzenai_version_)) {
                max_prompt_length_ = config["max_prompt_length"][ryzenai_version_];
                std::cout << "[InferenceEngine] Loaded max_prompt_length from rai_config.json: " 
                         << max_prompt_length_ << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] Failed to parse rai_config.json: " << e.what() << std::endl;
        }
    }
}

void InferenceEngine::setupExecutionProvider() {
    std::cout << "[InferenceEngine] Setting up execution provider for mode: " << execution_mode_ << std::endl;
    
    // Note: Actual execution provider configuration happens in ONNX Runtime GenAI
    // based on the genai_config.json file. This method is mainly for validation.
    
    if (execution_mode_ == "npu") {
        std::cout << "[InferenceEngine] Using NPU (VitisAI) execution provider" << std::endl;
    } else if (execution_mode_ == "hybrid") {
        std::cout << "[InferenceEngine] Using Hybrid (NPU + iGPU) execution provider" << std::endl;
    } else if (execution_mode_ == "cpu") {
        std::cout << "[InferenceEngine] Using CPU execution provider" << std::endl;
    }
}

void InferenceEngine::loadModel() {
    try {
        std::cout << "[InferenceEngine] Loading ONNX model from: " << model_path_ << std::endl;
        
        // Create model using factory method
        model_ = OgaModel::Create(model_path_.c_str());
        
#ifdef MLX_ON
        // Set context size from command line (flows to KV cache max length)
        model_->max_context_length = ctx_size_;
        std::cout << "[InferenceEngine] Set model max_context_length: " << ctx_size_ << std::endl;
#endif
        
        // Create tokenizer using factory method
        tokenizer_ = OgaTokenizer::Create(*model_);
        
        // Load chat template from tokenizer_config.json
        std::string tokenizer_config_path = model_path_ + "/tokenizer_config.json";
        if (fs::exists(tokenizer_config_path)) {
            try {
                std::ifstream file(tokenizer_config_path);
                json config = json::parse(file);
                if (config.contains("chat_template") && config["chat_template"].is_string()) {
                    chat_template_ = config["chat_template"];
                    std::cout << "[InferenceEngine] Loaded chat template from tokenizer_config.json" << std::endl;
                }
                
#ifndef MLX_ON
                // For non-MLX backends (Onyx/RYZENAI), load additional tokens into fallback_additional_tags_
                // MLX backend loads these in OgaModel::Create via loadAdditionalTokens()
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
                            } else if (content == "<|im_end|>") {
                                type = SpecialTokenType::CHAT_END;
                            } else if (content == "<|eot_id|>" || content == "<|end_of_turn|>" ||
                                       content == "<|end|>" || content == "<end_of_turn>") {
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
                                fallback_additional_tags_.push_back(token);
                                std::cout << "[InferenceEngine] Loaded special token: '" << content
                                          << "' (ID: " << token.token_id << ")" << std::endl;
                            }
                        }
                    }
                }
#endif
            } catch (const std::exception& e) {
                std::cerr << "[WARNING] Failed to load chat template: " << e.what() << std::endl;
            }
        }
        
#ifndef MLX_ON
        // Ensure we have fallback defaults if no tokens were found for non-MLX backends
        if (fallback_additional_tags_.empty()) {
            std::cout << "[InferenceEngine] No special tokens found, using fallback defaults" << std::endl;
            
            // Add basic thinking tokens with default values (token IDs unknown)
            fallback_additional_tags_.push_back({"<think>", SpecialTokenType::THINKING_START, -1});
            fallback_additional_tags_.push_back({"</think>", SpecialTokenType::THINKING_END, -1});
            
            // Add common chat end tokens
            fallback_additional_tags_.push_back({"<|im_end|>", SpecialTokenType::CHAT_END, -1});
            
            // Add tool tokens
            fallback_additional_tags_.push_back({"<tool_call>", SpecialTokenType::TOOL_CALL_START, -1});
            fallback_additional_tags_.push_back({"</tool_call>", SpecialTokenType::TOOL_CALL_END, -1});
            fallback_additional_tags_.push_back({"<tool_response>", SpecialTokenType::TOOL_RESPONSE_START, -1});
            fallback_additional_tags_.push_back({"</tool_response>", SpecialTokenType::TOOL_RESPONSE_END, -1});
        }
        
        std::cout << "[InferenceEngine] Loaded " << fallback_additional_tags_.size() 
                  << " special tokens for streaming detection" << std::endl;
#endif
        
        std::cout << "[InferenceEngine] Model and tokenizer loaded successfully" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

std::vector<int32_t> InferenceEngine::truncatePrompt(const std::vector<int32_t>& input_ids) {
    if (input_ids.size() <= static_cast<size_t>(max_prompt_length_)) {
        return input_ids;
    }
    
    // Truncate from the beginning to keep the most recent context
    size_t truncate_amount = input_ids.size() - max_prompt_length_;
    std::cout << "[WARNING] Prompt exceeds maximum length (" 
              << input_ids.size() << " > " << max_prompt_length_ 
              << "). Truncating " << truncate_amount << " tokens from the beginning."
              << std::endl;
    
    return std::vector<int32_t>(
        input_ids.begin() + truncate_amount, 
        input_ids.end()
    );
}

std::string InferenceEngine::complete(const std::string& prompt, const GenerationParams& params, CompletionTimingData* out_timing) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto sequences = OgaSequences::Create();
        tokenizer_->Encode(prompt.c_str(), *sequences);
        
        const int32_t* input_ids_ptr = sequences->SequenceData(0);
        size_t input_ids_count = sequences->SequenceCount(0);
        std::vector<int32_t> input_ids(input_ids_ptr, input_ids_ptr + input_ids_count);
        input_ids = truncatePrompt(input_ids);
        
        auto gen_params = OgaGeneratorParams::Create(*model_);
        gen_params->SetSearchOption("max_length", static_cast<int>(input_ids.size()) + params.max_length);
        gen_params->SetSearchOption("temperature", params.temperature);
        gen_params->SetSearchOption("top_p", params.top_p);
        gen_params->SetSearchOption("top_k", static_cast<double>(params.top_k));
        gen_params->SetSearchOption("repetition_penalty", params.repetition_penalty);
        gen_params->SetSearchOptionBool("do_sample", params.do_sample);
        gen_params->SetSearchOption("random_seed", 1.0);

        int eos_id = model_->GetEosId();
        gen_params->SetSearchOption("eos_token_id", eos_id);
        gen_params->SetSearchOption("pad_token_id", eos_id);
        
        auto generator = OgaGenerator::Create(*model_, *gen_params);
#ifdef MLX_ON
        generator->SetTokenizer(*tokenizer_);
#endif
        generator->AppendTokens(input_ids.data(), input_ids.size());
        
        std::cout << "[InferenceEngine] Generating..." << std::endl;

        // Get model-specific stop sequences
        std::vector<std::string> model_stop_sequences;
#ifdef MLX_ON
        model_stop_sequences = model_->GetStopSequences();
#endif

        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer_);
        std::string accumulated_output;

        while (!generator->IsDone()) {
            generator->GenerateNextToken();

            const int32_t* seq = generator->GetSequenceData(0);
            if (generator->GetSequenceCount(0) > 0) {
                int32_t new_token = seq[generator->GetSequenceCount(0) - 1];
                if (model_->IsEos(new_token)) break;

                // Check for string stop sequences
                const char* decoded = tokenizer_stream->Decode(new_token);
                if (decoded && decoded[0] != '\0') {
                    std::string token_str(decoded);
                    std::string temp_output = accumulated_output + token_str;

                    // Check model-specific stop sequences
                    bool should_stop = false;
                    for (const auto& stop_seq : model_stop_sequences) {
                        if (temp_output.find(stop_seq) != std::string::npos) {
                            std::cout << "[InferenceEngine] Stop sequence detected: '" << stop_seq << "' - Stopping generation." << std::endl;
                            should_stop = true;
                            break;
                        }
                    }
                    if (should_stop) break;

                    accumulated_output = temp_output;
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        const int32_t* output_ptr = generator->GetSequenceData(0);
        size_t output_count = generator->GetSequenceCount(0);
        
        if (out_timing) {
            int generated_count = (output_count > input_ids.size()) ? (output_count - input_ids.size()) : 0;
            auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            out_timing->total_time_ms = static_cast<double>(total.count());
            out_timing->token_count = generated_count;
        }
        
        std::string result;
        if (output_count > input_ids.size()) {
            // Trim EOS if present at the very end
            size_t decode_count = output_count - input_ids.size();
            if (decode_count > 0 && output_ptr[output_count-1] == eos_id) {
                decode_count--;
            }
            auto decoded = tokenizer_->Decode(output_ptr + input_ids.size(), decode_count);
            result = std::string(decoded);
        }
        
        // Strip custom stop sequences
        for (const auto& stop_seq : params.stop_sequences) {
            size_t pos = result.find(stop_seq);
            if (pos != std::string::npos) result = result.substr(0, pos);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Inference failed: " + std::string(e.what()));
    }
}

void InferenceEngine::streamComplete(const std::string& prompt, 
                                     const GenerationParams& params,
                                     StreamCallback callback) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    try {
        auto sequences = OgaSequences::Create();
        tokenizer_->Encode(prompt.c_str(), *sequences);
        
        const int32_t* input_ids_ptr = sequences->SequenceData(0);
        size_t input_ids_count = sequences->SequenceCount(0);
        std::vector<int32_t> input_ids(input_ids_ptr, input_ids_ptr + input_ids_count);
        input_ids = truncatePrompt(input_ids);
        
        auto gen_params = OgaGeneratorParams::Create(*model_);
        int total_max_length = static_cast<int>(input_ids.size()) + params.max_length;
        gen_params->SetSearchOption("max_length", total_max_length);
        gen_params->SetSearchOption("temperature", params.temperature);
        gen_params->SetSearchOption("top_p", params.top_p);
        gen_params->SetSearchOption("top_k", static_cast<double>(params.top_k));
        gen_params->SetSearchOption("repetition_penalty", params.repetition_penalty);
        gen_params->SetSearchOptionBool("do_sample", params.do_sample);
        gen_params->SetSearchOption("random_seed", 1.0);

        int eos_id = model_->GetEosId();
        gen_params->SetSearchOption("eos_token_id", eos_id);
        gen_params->SetSearchOption("pad_token_id", eos_id);
        
        auto generator = OgaGenerator::Create(*model_, *gen_params);
#ifdef MLX_ON
        generator->SetTokenizer(*tokenizer_);
#endif
        generator->AppendTokens(input_ids.data(), input_ids.size());
        
        std::cout << "[InferenceEngine] Streaming... (EOS Token ID: " << eos_id << ")" << std::endl;

        // Get model-specific stop sequences
        std::vector<std::string> model_stop_sequences;
#ifdef MLX_ON
        model_stop_sequences = model_->GetStopSequences();
#endif

        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer_);
        size_t token_count = 0;
        std::string accumulated_output;
        bool client_disconnected = false;

        // Get CHAT_END token IDs for additional stop checking
        std::vector<int32_t> chat_end_token_ids;
#ifdef MLX_ON
        for (const auto& tag : model_->additional_tags) {
            if (tag.type == SpecialTokenType::CHAT_END && tag.token_id >= 0) {
                chat_end_token_ids.push_back(tag.token_id);
                std::cout << "[InferenceEngine] Added CHAT_END token ID: " << tag.token_id << " ('" << tag.content << "')" << std::endl;
            }
        }
#else
        for (const auto& tag : fallback_additional_tags_) {
            if (tag.type == SpecialTokenType::CHAT_END && tag.token_id >= 0) {
                chat_end_token_ids.push_back(tag.token_id);
                std::cout << "[InferenceEngine] Added CHAT_END token ID: " << tag.token_id << " ('" << tag.content << "')" << std::endl;
            }
        }
#endif
        
        if (chat_end_token_ids.empty()) {
            std::cout << "[InferenceEngine] WARNING: No CHAT_END token IDs found!" << std::endl;
        }

        while (!generator->IsDone() && !client_disconnected) {
            generator->GenerateNextToken();

            const int32_t* all_tokens = generator->GetSequenceData(0);
            size_t num_tokens = generator->GetSequenceCount(0);
            int32_t new_token = all_tokens[num_tokens - 1];

            // Even if the backend logic misses it, we force break here.
            if (model_->IsEos(new_token)) {
                std::cout << "[InferenceEngine] Hit EOS token - Stopping." << std::endl;
                break;
            }
            
            // Also check for CHAT_END token IDs (e.g., <|im_end|> for Qwen)
            bool is_chat_end = false;
            for (int32_t chat_end_id : chat_end_token_ids) {
                if (new_token == chat_end_id) {
                    std::cout << "[InferenceEngine] Hit CHAT_END token (ID: " << new_token << ") - Stopping." << std::endl;
                    is_chat_end = true;
                    break;
                }
            }
            if (is_chat_end) break;

            const char* decoded = tokenizer_stream->Decode(new_token);
            if (decoded && decoded[0] != '\0') {
                std::string token_str(decoded);

                // Stop Sequence Check
                bool should_stop = false;

                // Check params stop sequences
                for (const auto& stop_seq : params.stop_sequences) {
                    std::string temp = accumulated_output + token_str;
                    if (temp.find(stop_seq) != std::string::npos) {
                        should_stop = true;
                        break;
                    }
                }

                // Check model-specific stop sequences
                if (!should_stop) {
                    for (const auto& stop_seq : model_stop_sequences) {
                        std::string temp = accumulated_output + token_str;
                        if (temp.find(stop_seq) != std::string::npos) {
                            std::cout << "[InferenceEngine] Stop sequence detected: '" << stop_seq << "' - Stopping streaming." << std::endl;
                            should_stop = true;
                            break;
                        }
                    }
                }

                if (should_stop) break;

                accumulated_output += token_str;
                bool is_final = generator->IsDone();

                if (!callback(token_str, is_final)) {
                    client_disconnected = true;
                    std::cout << "[InferenceEngine] Client disconnected" << std::endl;
                    break;
                }
            }
            token_count++;
        }
        
        std::cout << "[InferenceEngine] Streamed " << token_count << " tokens." << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Streaming inference failed: " + std::string(e.what()));
    }
}

int InferenceEngine::countTokens(const std::string& text) {
    try {
        auto sequences = OgaSequences::Create();
        tokenizer_->Encode(text.c_str(), *sequences);
        return static_cast<int>(sequences->SequenceCount(0));
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to count tokens: " << e.what() << std::endl;
        return 0;
    }
}

} // namespace ryzenai

