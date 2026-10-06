#pragma once

#include <string>
#include <vector>
#include <json.hpp>

namespace ryzenai {

using json = nlohmann::json;

// Command line arguments
struct CommandLineArgs {
    std::string model_path;           // -m, --model (required)
    std::string host = "127.0.0.1";   // --host
    int port = 8080;                  // --port
    int ctx_size = 2048;              // --ctx-size
    int threads = 4;                  // --threads
    bool verbose = false;             // --verbose
};

// Image content part extracted from an OpenAI vision message
struct ImageContent {
    std::vector<uint8_t> data;  // raw decoded bytes
    std::string mime_type;      // e.g. "image/jpeg"
};

// Chat message — content is text; images holds any inline image_url parts
struct ChatMessage {
    std::string role;
    std::string content;
    std::vector<ImageContent> images;
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

