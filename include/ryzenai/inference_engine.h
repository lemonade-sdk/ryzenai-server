/*
 * inference_engine.h
 * 
 * Multi-backend inference engine that supports loading multiple models
 * across different backends (MLX, ONNX/Ryzen AI) simultaneously.
 */

#pragma once

#include <ryzenai/types.h>
#include <ryzenai/backend/backend.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace ryzenai {

// Timing data returned from completion
struct CompletionTimingData {
    int token_count = 0;           // Number of generated tokens
    double ttft_seconds = 0.0;     // Time to first token in seconds
    double tps = 0.0;              // Tokens per second (decode speed)
    double total_time_ms = 0.0;    // Total completion time in milliseconds
    
    // Detailed profiling
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
    KVCacheMode kv_format = KVCacheMode::FP16;  // --kv-format (fp16|int8|int4)
    int prefill_chunk = 512;          // --prefill-chunk
};

// Information about a loaded model
struct LoadedModel {
    std::unique_ptr<IBackend> backend;
    std::string model_name;       // Short name: "phi3", "qwen3"
    std::string model_path;       // Full path to model
    BackendType backend_type;
    
    LoadedModel() = default;
    LoadedModel(LoadedModel&&) = default;
    LoadedModel& operator=(LoadedModel&&) = default;
};

/*
 * InferenceEngine
 * 
 * Manages multiple loaded models across different backends.
 * Routes API requests to the appropriate backend based on model name.
 */
class InferenceEngine {
public:
    explicit InferenceEngine(const OptimizationSettings& opt = OptimizationSettings());
    ~InferenceEngine();
    
    // ==================== Model Management ====================
    
    // Load a model on the specified backend
    // Returns the assigned model name (extracted from path or config)
    std::string loadModel(const std::string& model_path, BackendType backend_type);
    
    // Unload a model by name
    void unloadModel(const std::string& model_name);
    
    // Get list of loaded model names
    std::vector<std::string> getLoadedModels() const;
    
    // Get backend for a specific model (for direct access)
    IBackend* getBackendForModel(const std::string& model_name);
    const IBackend* getBackendForModel(const std::string& model_name) const;
    
    // Get the first/default loaded model (backward compatibility)
    IBackend* getDefaultBackend();
    
    // ==================== Inference (Model-Specific) ====================
    
    // Synchronous completion for a specific model
    std::string complete(const std::string& model_name, 
                        const std::string& prompt, 
                        const GenerationParams& params, 
                        CompletionTimingData* out_timing = nullptr);
    
    // Streaming completion for a specific model
    void streamComplete(const std::string& model_name,
                       const std::string& prompt, 
                       const GenerationParams& params,
                       StreamCallback callback);
    
    // Apply chat template for a specific model (requires 3 args to avoid ambiguity)
    std::string applyChatTemplate(const std::string& model_name,
                                  const std::string& messages_json, 
                                  const std::string& tools_json);
    
    // Token counting for a specific model
    int countTokens(const std::string& model_name, const std::string& text);
    
    // ==================== Model Info ====================
    
    // Get model info by name
    std::string getModelType(const std::string& model_name) const;
    int getMaxContextLength(const std::string& model_name) const;
    GenerationParams getDefaultParams(const std::string& model_name) const;
    const std::vector<AdditionalToken>& getSpecialTokens(const std::string& model_name) const;
    
    // ==================== Backward Compatibility ====================
    // These use the first loaded model (for single-model usage)
    
    std::string complete(const std::string& prompt, 
                        const GenerationParams& params, 
                        CompletionTimingData* out_timing = nullptr);
    
    void streamComplete(const std::string& prompt, 
                       const GenerationParams& params,
                       StreamCallback callback);
    
    std::string applyChatTemplate(const std::string& messages_json, 
                                  const std::string& tools_json = "");
    
    int countTokens(const std::string& text);
    
    // Legacy getters
    std::string getModelName() const;
    std::string getExecutionMode() const { return execution_mode_; }
    GenerationParams getDefaultParams() const;
    const std::vector<AdditionalToken>& getAdditionalTags() const;
    
private:
    // Normalize model name for lookup (lowercase, remove special chars)
    static std::string normalizeModelName(const std::string& name);
    
    // Extract model name from path or config
    std::string extractModelName(const std::string& model_path);
    
    // Find loaded model by name (case-insensitive)
    LoadedModel* findModel(const std::string& model_name);
    const LoadedModel* findModel(const std::string& model_name) const;
    
    // All loaded models
    std::vector<LoadedModel> loaded_models_;
    
    // Quick lookup by normalized name
    std::unordered_map<std::string, size_t> model_index_;
    
    // Settings
    OptimizationSettings opt_settings_;
    std::string execution_mode_ = "auto";  // For compatibility
    
    // Thread safety
    mutable std::mutex models_mutex_;
    
    // Empty fallbacks for const references
    static const std::vector<AdditionalToken> empty_tokens_;
    static const GenerationParams default_params_;
};

} // namespace ryzenai
