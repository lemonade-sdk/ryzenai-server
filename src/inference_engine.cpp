#include "ryzenai/inference_engine.h"
#include <ort_genai.h>
#include <ort_genai_c.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <thread>
#include <chrono>

namespace ryzenai {

namespace fs = std::filesystem;

InferenceEngine::InferenceEngine(const std::string& model_path) {
    
    std::cout << "[InferenceEngine] Initializing with model: " << model_path << std::endl;
    
    // Resolve model path (handles Hugging Face cache structure)
    model_path_ = resolveModelPath(model_path);
    if (model_path_ != model_path) {
        std::cout << "[InferenceEngine] Resolved to: " << model_path_ << std::endl;
    }
    
    // Validate model directory
    if (!validateModelDirectory(model_path_)) {
        throw std::runtime_error("Invalid model directory: " + model_path_);
    }
    
    // Auto-detect execution mode from genai_config.json
    execution_mode_ = detectExecutionMode();
    std::cout << "[InferenceEngine] Detected execution mode: " << execution_mode_ << std::endl;
    
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
    
    // Content-part arrays carry image/audio placeholders that only the model's own
    // template knows how to render into media tokens.
    bool has_content_parts = false;
    for (const auto& msg : messages) {
        if (msg.contains("content") && msg["content"].is_array()) {
            has_content_parts = true;
            break;
        }
    }

    // VideoChat-Flash ships a plain-text Qwen2.5 template with no vision token support.
    // The VideoChatFlashProcessor requires explicit <|vision_start|><|vision_end|> markers
    // per image, so the prompt must be built manually rather than via ApplyChatTemplate.
    if (has_content_parts && model_type_ == "videochat_flash_qwen") {
        std::string vision_blocks, text;
        for (const auto& msg : messages) {
            for (const auto& part : msg.value("content", json::array())) {
                if (part.value("type", "") == "image") vision_blocks += "<|vision_start|><|vision_end|>";
                if (part.value("type", "") == "text") {
                    if (!text.empty()) text += "\n";
                    text += part.value("text", "");
                }
            }
        }
        std::cout << "[InferenceEngine] Applied VideoChat-Flash prompt format" << std::endl;
        return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
               "<|im_start|>user\n" + vision_blocks + text + "<|im_end|>\n"
               "<|im_start|>assistant\n";
    }

    // Tools and content parts need the full template, so use OGA's renderer
    if (!tools_json.empty() || has_content_parts) {
        try {
            const char* template_str = chat_template_.empty() ? nullptr : chat_template_.c_str();
            const char* tools_str = tools_json.empty() ? nullptr : tools_json.c_str();

            auto result = tokenizer_->ApplyChatTemplate(
                template_str,
                messages_json.c_str(),
                tools_str,
                true
            );

            std::cout << "[InferenceEngine] Applied chat template"
                      << (tools_str ? " with tools" : "")
                      << (has_content_parts ? " with media parts" : "") << std::endl;
            return std::string(result);

        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to apply chat template: " << e.what() << std::endl;
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
    
    // Check for required files (at minimum genai_config.json)
    std::string config_path = path + "/genai_config.json";
    if (!fs::exists(config_path)) {
        std::cerr << "[ERROR] Required file not found: " << config_path << std::endl;
        return false;
    }
    
    return true;
}

std::string InferenceEngine::detectRyzenAIVersion() {
    // Priority 1: Check RYZENAI_VERSION environment variable directly
    const char* version_env = std::getenv("RYZENAI_VERSION");
    if (version_env) {
        return std::string(version_env);
    }

    // Priority 2: Check RYZENAI_INSTALL_PATH environment variable and extract version
    const char* install_path_env = std::getenv("RYZENAI_INSTALL_PATH");
    if (install_path_env) {
        std::string path_str(install_path_env);
        // Remove trailing slashes
        while (!path_str.empty() && (path_str.back() == '/' || path_str.back() == '\\')) {
            path_str.pop_back();
        }
        // Extract version from path (e.g., "/opt/ryzenai/1.7.0" -> "1.7.0")
        size_t last_slash = path_str.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            std::string version = path_str.substr(last_slash + 1);
            // Validate it looks like a version number: starts with digit and contains a dot
            if (!version.empty() && std::isdigit(version[0]) && version.find('.') != std::string::npos) {
                return version;
            }
        }
    }

    // Priority 3: Check platform-specific default paths
#ifdef _WIN32
    std::string ryzenai_path_19 = "C:/Program Files/RyzenAI/1.9.0";
#else
    std::string ryzenai_path_19 = "/opt/ryzenai/1.9.0";
#endif

    if (fs::exists(ryzenai_path_19)) {
        return "1.9.0";
    }

    // Default fallback
    return "1.9.0";
}

std::string InferenceEngine::detectExecutionMode() {
    // Auto-detect execution mode from genai_config.json by inspecting
    // session_options in model.decoder. The key indicators are:
    //
    // NPU:    "hybrid_opt_token_backend": "npu" in either config_entries
    //         or provider_options[].RyzenAI
    // Hybrid: provider_options[].RyzenAI exists (without token_backend=npu)
    // CPU:    empty provider_options, no config_entries
    
    std::string config_path = model_path_ + "/genai_config.json";
    if (!fs::exists(config_path)) {
        std::cout << "[InferenceEngine] No genai_config.json found, defaulting to cpu mode" << std::endl;
        return "cpu";
    }
    
    try {
        std::ifstream file(config_path);
        json config = json::parse(file);
        
        auto& session_opts = config["model"]["decoder"]["session_options"];
        
        // Check config_entries for hybrid_opt_token_backend == "npu"
        if (session_opts.contains("config_entries")) {
            auto& entries = session_opts["config_entries"];
            if (entries.contains("hybrid_opt_token_backend") &&
                entries["hybrid_opt_token_backend"] == "npu") {
                return "npu";
            }
        }
        
        // Check provider_options for RyzenAI configuration
        if (session_opts.contains("provider_options") && 
            session_opts["provider_options"].is_array()) {
            for (const auto& provider : session_opts["provider_options"]) {
                if (provider.contains("RyzenAI")) {
                    auto& ryzenai = provider["RyzenAI"];
                    // If RyzenAI has hybrid_opt_token_backend == "npu", it's NPU
                    if (ryzenai.contains("hybrid_opt_token_backend") &&
                        ryzenai["hybrid_opt_token_backend"] == "npu") {
                        return "npu";
                    }
                    // Otherwise RyzenAI provider present means hybrid
                    return "hybrid";
                }
            }
        }
        
        // No RyzenAI provider, no NPU config_entries → CPU
        return "cpu";
        
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to detect execution mode from genai_config.json: " 
                  << e.what() << std::endl;
        std::cerr << "[WARNING] Defaulting to cpu mode" << std::endl;
        return "cpu";
    }
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
            
            // Checkpoints don't always list the running SDK (the 1.9.0 ones stop at
            // 1.7.1), so take the newest entry that isn't newer than it.
            auto parse_version = [](const std::string& v) {
                std::vector<int> parts;
                std::stringstream ss(v);
                for (std::string p; std::getline(ss, p, '.');) {
                    parts.push_back(std::atoi(p.c_str()));
                }
                return parts;
            };
            const auto current = parse_version(ryzenai_version_);
            std::vector<int> best_version;
            std::string best_key;
            const json limits = config.value("max_prompt_length", json::object());
            for (const auto& [key, value] : limits.items()) {
                auto version = parse_version(key);
                if (value.is_number_integer() && version <= current && version >= best_version) {
                    best_version = version;
                    best_key = key;
                }
            }
            if (!best_key.empty()) {
                max_prompt_length_ = config["max_prompt_length"][best_key];
                std::cout << "[InferenceEngine] Loaded max_prompt_length from rai_config.json: "
                          << max_prompt_length_ << " (entry " << best_key << ")" << std::endl;
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
        
        model_ = OgaModel::Create(model_path_.c_str());

        // Same signal OGA uses: a vision/speech encoder section in genai_config.json.
        std::string genai_config_path = model_path_ + "/genai_config.json";
        if (fs::exists(genai_config_path)) {
            std::ifstream file(genai_config_path);
            json config = json::parse(file);
            const json& model_cfg = config.value("model", json::object());
            if (model_cfg.contains("type") && model_cfg["type"].is_string())
                model_type_ = model_cfg["type"].get<std::string>();
            if (model_cfg.contains("vision") || model_cfg.contains("speech")) {
                processor_ = OgaMultiModalProcessor::Create(*model_);
                std::cout << "[InferenceEngine] Multimodal model (type=" << model_type_
                          << "), created MultiModalProcessor" << std::endl;
            }
        }

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
            } catch (const std::exception& e) {
                std::cerr << "[WARNING] Failed to load chat template: " << e.what() << std::endl;
            }
        }

        // Newer HF exports (including the RAI 1.9 checkpoints) keep the template
        // only in chat_template.jinja; without it tools never reach the prompt.
        std::string jinja_path = model_path_ + "/chat_template.jinja";
        if (chat_template_.empty() && fs::exists(jinja_path)) {
            std::ifstream file(jinja_path, std::ios::binary);
            chat_template_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            std::cout << "[InferenceEngine] Loaded chat template from chat_template.jinja" << std::endl;
        }

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

InferenceEngine::Generation InferenceEngine::startGeneration(const std::string& prompt,
                                                             const GenerationParams& params,
                                                             const MediaInputs& media) {
    Generation gen;
    std::vector<int32_t> input_ids;

    if (media.empty()) {
        auto sequences = OgaSequences::Create();
        tokenizer_->Encode(prompt.c_str(), *sequences);
        const int32_t* input_ids_ptr = sequences->SequenceData(0);
        input_ids.assign(input_ids_ptr, input_ids_ptr + sequences->SequenceCount(0));
        input_ids = truncatePrompt(input_ids);
        gen.prompt_tokens = input_ids.size();
    } else {
        if (!processor_) {
            throw std::invalid_argument("Model '" + model_name_ + "' does not accept image or audio input");
        }
        std::vector<const void*> image_data, audio_data;
        std::vector<size_t> image_sizes, audio_sizes;
        for (const auto& image : media.images) {
            image_data.push_back(image.data());
            image_sizes.push_back(image.size());
        }
        for (const auto& audio : media.audios) {
            audio_data.push_back(audio.data());
            audio_sizes.push_back(audio.size());
        }
        auto images = media.images.empty() ? nullptr
            : OgaImages::Load(image_data.data(), image_sizes.data(), image_data.size());
        auto audios = media.audios.empty() ? nullptr
            : OgaAudios::Load(audio_data.data(), audio_sizes.data(), audio_data.size());
        gen.inputs = processor_->ProcessImagesAndAudios(prompt.c_str(), images.get(), audios.get());

        // input_ids shape is [1, seq_len]; index 1 is the token count.
        auto shape = gen.inputs->Get("input_ids")->Shape();
        gen.prompt_tokens = static_cast<size_t>(shape.at(1));
        if (gen.prompt_tokens > static_cast<size_t>(max_prompt_length_)) {
            throw std::invalid_argument("Prompt with media is " + std::to_string(gen.prompt_tokens) +
                                        " tokens, over this model's limit of " +
                                        std::to_string(max_prompt_length_));
        }
        gen.has_media = true;
        std::cout << "[InferenceEngine] Processed " << media.images.size() << " image(s), "
                  << media.audios.size() << " audio clip(s) -> " << gen.prompt_tokens
                  << " prompt tokens" << std::endl;
    }

    // params.max_length is max_new_tokens; OGA's max_length counts the prompt too
    auto gen_params = OgaGeneratorParams::Create(*model_);
    gen_params->SetSearchOption("max_length", static_cast<double>(gen.prompt_tokens + params.max_length));
    gen_params->SetSearchOption("temperature", params.temperature);
    gen_params->SetSearchOption("top_p", params.top_p);
    gen_params->SetSearchOption("top_k", static_cast<double>(params.top_k));
    gen_params->SetSearchOption("repetition_penalty", params.repetition_penalty);
    gen_params->SetSearchOptionBool("do_sample", params.do_sample);
    // Pin seed only when greedy/deterministic so temperature>0 requests vary as expected.
    if (!params.do_sample || params.temperature == 0.0f) {
        gen_params->SetSearchOption("random_seed", 1.0);
    }

    gen.generator = OgaGenerator::Create(*model_, *gen_params);
    if (gen.inputs) {
        gen.generator->SetInputs(*gen.inputs);
    } else {
        gen.generator->AppendTokens(input_ids.data(), input_ids.size());
    }
    return gen;
}

std::string InferenceEngine::complete(const std::string& prompt, const GenerationParams& params,
                                      CompletionTimingData* out_timing, const MediaInputs& media) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto first_token_time = start_time;
        bool first_token_received = false;

        auto gen = startGeneration(prompt, params, media);
        auto& generator = gen.generator;

        int generated_token_count = 0;
        while (!generator->IsDone()) {
            generator->GenerateNextToken();
            ++generated_token_count;
            if (!first_token_received) {
                first_token_time = std::chrono::high_resolution_clock::now();
                first_token_received = true;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        const int32_t* output_ptr = generator->GetSequenceData(0);
        size_t output_count = generator->GetSequenceCount(0);

        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto ttft_duration = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        double ttft_seconds = ttft_duration.count() / 1000.0;
        double total_time_ms = static_cast<double>(total_duration.count());
        // TPS excludes the first token (prefill-dominated); single-token falls back to total time.
        double decode_time_seconds = (total_duration.count() - ttft_duration.count()) / 1000.0;
        double tps = 0.0;
        if (generated_token_count > 1 && decode_time_seconds > 0) {
            tps = (generated_token_count - 1) / decode_time_seconds;
        } else if (generated_token_count == 1 && total_time_ms > 0) {
            tps = 1.0 / (total_time_ms / 1000.0);
        }
        if (out_timing != nullptr) {
            out_timing->token_count = generated_token_count;
            out_timing->ttft_seconds = ttft_seconds;
            out_timing->tps = tps;
            out_timing->total_time_ms = total_time_ms;
        }
        
        std::string result;
        size_t new_tokens = std::min(static_cast<size_t>(generated_token_count), output_count);
        if (new_tokens > 0) {
            auto decoded = tokenizer_->Decode(output_ptr + output_count - new_tokens, new_tokens);
            result = std::string(decoded);
        }
        
        for (const auto& stop_seq : params.stop_sequences) {
            size_t pos = result.find(stop_seq);
            if (pos != std::string::npos) {
                result = result.substr(0, pos);
                break;
            }
        }
        
        std::cout << "[InferenceEngine] Generated " << generated_token_count << " tokens in " 
                  << total_time_ms << "ms (TTFT: " << (ttft_seconds * 1000) << "ms, TPS: " << tps << ")" << std::endl;
        
        return result;

    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::exception& e) {
        throw std::runtime_error("Inference failed: " + std::string(e.what()));
    }
}

void InferenceEngine::streamComplete(const std::string& prompt,
                                     const GenerationParams& params,
                                     StreamCallback callback,
                                     const MediaInputs& media,
                                     size_t* out_prompt_tokens) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    try {
        auto gen = startGeneration(prompt, params, media);
        auto& generator = gen.generator;
        if (out_prompt_tokens) *out_prompt_tokens = gen.prompt_tokens;
        std::cout << "[InferenceEngine::streamComplete] prompt_length=" << gen.prompt_tokens
                  << ", max_new_tokens=" << params.max_length << std::endl;

        std::cout << "[InferenceEngine] Generating tokens (streaming)..." << std::endl;
        
        // Multimodal generators must use the processor stream — it suppresses image-pad
        // sentinel tokens that the plain tokenizer stream would decode as garbage.
        auto tokenizer_stream = gen.has_media
            ? OgaTokenizerStream::Create(*processor_)
            : OgaTokenizerStream::Create(*tokenizer_);
        
        size_t token_count = 0;
        std::string accumulated_output;  // Track full output for stop sequence detection
        bool client_disconnected = false;  // Track if client disconnected
        
        while (!generator->IsDone() && !client_disconnected) {
            generator->GenerateNextToken();
            
            // Get just the new token
            const int32_t* all_tokens = generator->GetSequenceData(0);
            size_t num_tokens = generator->GetSequenceCount(0);
            int32_t new_token = all_tokens[num_tokens - 1];
            
            // Decode incrementally using tokenizer stream (this works!)
            const char* decoded = tokenizer_stream->Decode(new_token);
            if (decoded && decoded[0] != '\0') {
                std::string token_str(decoded);
                
                // Check for stop sequences before accumulating
                bool should_stop = false;
                for (const auto& stop_seq : params.stop_sequences) {
                    // Check if adding this token would complete a stop sequence
                    std::string temp_output = accumulated_output + token_str;
                    if (temp_output.find(stop_seq) != std::string::npos) {
                        should_stop = true;
                        break;
                    }
                }
                
                if (should_stop) {
                    // Stop generation - don't send this token
                    break;
                }
                
                accumulated_output += token_str;
                bool is_final = generator->IsDone();
                
                // Call callback and check if client is still connected
                if (!callback(token_str, is_final)) {
                    client_disconnected = true;
                    std::cout << "[InferenceEngine] Client disconnected, stopping generation" << std::endl;
                    break;
                }
            }
            
            token_count++;
        }
        
        if (client_disconnected) {
            std::cout << "[InferenceEngine] Generation stopped early due to client disconnect (generated " 
                      << token_count << " tokens)" << std::endl;
        }
        
        std::cout << "[InferenceEngine] Generated " << token_count << " tokens (streaming)" << std::endl;

    } catch (const std::invalid_argument&) {
        throw;
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

