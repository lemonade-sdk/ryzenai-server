#pragma once

#include "types.h"
#include "mlx/model.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>

// Forward declarations for ONNX Runtime GenAI
struct OgaModel;
struct OgaTokenizer;
struct OgaGeneratorParams;
struct OgaGenerator;
struct OgaSequences;

namespace ryzenai {

// Timing data returned from completion
struct CompletionTimingData {
    int token_count = 0;           // Number of generated tokens
    double ttft_seconds = 0.0;     // Time to first token in seconds
    double tps = 0.0;              // Tokens per second (decode speed)
    double total_time_ms = 0.0;    // Total completion time in milliseconds
    
    // Detailed profiling (only filled when PROFILE_INFERENCE is defined)
    double tokenize_ms = 0.0;      // Input tokenization time
    double prefill_ms = 0.0;       // First token / prefill time
    double decode_ms = 0.0;        // Total decode time (excluding prefill)
    double detokenize_ms = 0.0;    // Output detokenization time
};

// Optimization settings passed from command line
struct OptimizationSettings {
    int ctx_size = 2048;              // --ctx-size
    int repetition_lookback = 64;     // --rep-lookback
    bool kv_cache = true;             // --kv-cache / --no-kv-cache
    int prefill_chunk = 512;          // --prefill-chunk
};

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path, const std::string& mode, 
                   const OptimizationSettings& opt = OptimizationSettings());
    ~InferenceEngine();
    
    // Synchronous completion
    // Returns generated text. If out_timing is provided, stores timing data.
    std::string complete(const std::string& prompt, const GenerationParams& params, CompletionTimingData* out_timing = nullptr);
    
    // Streaming completion
    void streamComplete(const std::string& prompt, 
                       const GenerationParams& params,
                       StreamCallback callback);
    
    // Apply chat template to messages
    std::string applyChatTemplate(const std::string& messages_json, const std::string& tools_json = "");
    
    // Getters
    std::string getModelName() const { return model_name_; }
    std::string getExecutionMode() const { return execution_mode_; }
    int getMaxPromptLength() const { return max_prompt_length_; }
    std::string getRyzenAIVersion() const { return ryzenai_version_; }
    
    // Get default generation params from genai_config.json (if available)
    GenerationParams getDefaultParams() const;
    
    // Token counting
    int countTokens(const std::string& text);

    // Get additional special tokens for streaming detection
    const std::vector<AdditionalToken>& getAdditionalTags() const;
    
private:
    void loadModel();
    void setupExecutionProvider();
    void loadRaiConfig();
    std::string detectRyzenAIVersion();
    std::string resolveModelPath(const std::string& path);
    std::vector<int32_t> truncatePrompt(const std::vector<int32_t>& input_ids);
    bool validateModelDirectory(const std::string& path);
    
    std::unique_ptr<OgaModel> model_;
    std::unique_ptr<OgaTokenizer> tokenizer_;
    
    std::string model_path_;
    std::string model_name_;
    std::string execution_mode_;  // "npu", "hybrid", or "cpu"
    std::string ryzenai_version_;
    std::string chat_template_;  // Chat template from tokenizer_config.json
    int max_prompt_length_ = 2048;  // Default, overridden by rai_config.json
    int ctx_size_ = 2048;  // Context size for KV cache (from --ctx-size)
    
    // Default generation params from genai_config.json search section
    GenerationParams default_params_;
    bool has_search_config_ = false;
    
    // Fallback additional tokens for non-MLX backends (Onyx)
    // For MLX, we use model_->additional_tags directly
    std::vector<AdditionalToken> fallback_additional_tags_;
    
    std::mutex inference_mutex_;  // Protect inference operations
};

} // namespace ryzenai

