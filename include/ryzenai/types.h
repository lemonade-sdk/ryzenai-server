#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <json.hpp>

namespace ryzenai {

using json = nlohmann::json;

/*
 * KV Cache quantization mode
 * Controls memory usage vs precision trade-off for KV cache.
 * INT8 provides ~2x memory savings, enabling longer contexts.
 */
enum class KVCacheMode {
    FP16,       // Original FP16 cache (higher accuracy, faster on Apple Silicon)
    INT8,       // INT8 quantized cache (2x memory savings, slight accuracy loss)
    INT4        // INT4 quantized cache (4x memory savings, more accuracy loss) [experimental]
};

/*
 * Special token types for streaming detection
 */
enum class SpecialTokenType {
    THINKING_START,       // <think>
    THINKING_END,         // </think>
    TOOL_CALL_START,      // <tool_call>
    TOOL_CALL_END,        // </tool_call>
    TOOL_RESPONSE_START,  // <tool_response>
    TOOL_RESPONSE_END,    // </tool_response>
    CHAT_START,           // <|im_start|>
    CHAT_END,             // <|im_end|>, <|eot_id|>, etc.
    UNKNOWN
};

/*
 * Additional special token with type information
 */
struct AdditionalToken {
    std::string content;
    SpecialTokenType type;
    int32_t token_id = -1;  // Token ID if known
};

// Model configuration for multi-model support
struct ModelConfig {
    std::string path;                 // Path to model directory
    std::string backend = "auto";     // Backend type: auto|mlx|onnx|ryzenai|cpu
};

// Command line arguments
struct CommandLineArgs {
    std::string model_path;           // -m, --model (required, backward compat for single model)
    std::string host = "127.0.0.1";   // --host
    int port = 8080;                  // --port
    std::string mode = "hybrid";      // --mode (npu|hybrid|cpu) - also used as default backend
    int ctx_size = 2048;              // --ctx-size
    int threads = 4;                  // --threads
    bool verbose = false;             // --verbose
    
    // Multi-model support
    std::vector<ModelConfig> models;  // Multiple models with backends
    
    // Optimization parameters
    int repetition_lookback = 64;     // --rep-lookback (tokens to check for repetition)
    bool kv_cache = true;             // --kv-cache / --no-kv-cache
    KVCacheMode kv_format = KVCacheMode::FP16;  // --kv-format (fp16|int8|int4)
    int prefill_chunk = 512;          // --prefill-chunk (tokens per chunk for long prompts)
};

// Chat message structure
struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

// Completion request (OpenAI format)
struct CompletionRequest {
    std::string prompt;
    int max_tokens = 1500;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    bool stream = false;
    bool echo = false;
    std::vector<std::string> stop;
    
    // Parse from JSON
    static CompletionRequest fromJSON(const json& j);
};

// Chat completion request (OpenAI format)
struct ChatCompletionRequest {
    std::string model;  // Model to use for completion
    std::vector<ChatMessage> messages;
    int max_tokens = 1500;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    bool stream = false;
    std::vector<std::string> stop;
    json tools;  // Tool definitions (OpenAI format)
    
    // Parse from JSON
    static ChatCompletionRequest fromJSON(const json& j);
    
    // Convert messages to a single prompt string
    std::string toPrompt() const;
};

// Generation parameters for ONNX GenAI
struct GenerationParams {
    int max_length = 2048;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repetition_penalty = 1.1f;
    int min_length = 0;
    bool do_sample = true;
    std::vector<std::string> stop_sequences;
};

// Token generation callback for streaming
// Returns true to continue generation, false to stop (e.g., if client disconnected)
using StreamCallback = std::function<bool(const std::string& token, bool is_final)>;

} // namespace ryzenai

