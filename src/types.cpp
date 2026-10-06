#include "ryzenai/types.h"
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace ryzenai {

// Base64 decode — handles data URIs like "data:image/jpeg;base64,<data>"
static std::vector<uint8_t> decode_base64(const std::string& input) {
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> out;
    int val = 0, bits = -8;
    for (unsigned char c : input) {
        if (c == '=') break;
        auto pos = chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

// Parse a single content part array from an OpenAI vision message.
// Fills message.content (text) and message.images (decoded image bytes).
static void parse_content_array(const json& content_arr, ChatMessage& message) {
    for (const auto& part : content_arr) {
        std::string type = part.value("type", "");
        if (type == "text") {
            if (!message.content.empty()) message.content += "\n";
            message.content += part.value("text", "");
        } else if (type == "image_url") {
            if (!part.contains("image_url")) continue;
            std::string url = part["image_url"].value("url", "");
            // Expect "data:<mime>;base64,<data>"
            const std::string prefix = "data:";
            const std::string b64marker = ";base64,";
            if (url.rfind(prefix, 0) == 0) {
                auto semi = url.find(b64marker);
                if (semi != std::string::npos) {
                    std::string mime = url.substr(prefix.size(), semi - prefix.size());
                    std::string b64data = url.substr(semi + b64marker.size());
                    ImageContent img;
                    img.mime_type = mime;
                    img.data = decode_base64(b64data);
                    message.images.push_back(std::move(img));
                }
            }
        }
    }
}

CompletionRequest CompletionRequest::fromJSON(const json& j) {
    CompletionRequest req;
    
    if (j.contains("prompt") && j["prompt"].is_string()) {
        req.prompt = j["prompt"];
    }
    
    // Support both max_tokens (deprecated) and max_completion_tokens (newer OpenAI API)
    // max_completion_tokens takes precedence if both are provided
    if (j.contains("max_completion_tokens")) {
        req.max_tokens = j["max_completion_tokens"];
    } else if (j.contains("max_tokens")) {
        req.max_tokens = j["max_tokens"];
    }
    
    if (j.contains("temperature")) {
        req.temperature = j["temperature"];
    }
    
    if (j.contains("top_p")) {
        req.top_p = j["top_p"];
    }
    
    if (j.contains("top_k")) {
        req.top_k = j["top_k"];
    }
    
    if (j.contains("repeat_penalty")) {
        req.repeat_penalty = j["repeat_penalty"];
    } else if (j.contains("repetition_penalty")) {
        req.repeat_penalty = j["repetition_penalty"];
    } else if (j.contains("frequency_penalty")) {
        // OpenAI uses frequency_penalty, we map it to repeat_penalty
        req.repeat_penalty = 1.0f + j["frequency_penalty"].get<float>();
    }
    
    if (j.contains("stream")) {
        req.stream = j["stream"];
    }
    
    if (j.contains("echo")) {
        req.echo = j["echo"];
    }
    
    if (j.contains("stop")) {
        if (j["stop"].is_string()) {
            req.stop.push_back(j["stop"]);
        } else if (j["stop"].is_array()) {
            for (const auto& s : j["stop"]) {
                req.stop.push_back(s);
            }
        }
    }
    
    return req;
}

ChatCompletionRequest ChatCompletionRequest::fromJSON(const json& j) {
    ChatCompletionRequest req;
    
    if (j.contains("messages") && j["messages"].is_array()) {
        for (const auto& msg : j["messages"]) {
            ChatMessage message;
            message.role = msg.value("role", "user");
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    message.content = msg["content"].get<std::string>();
                } else if (msg["content"].is_array()) {
                    // OpenAI vision format: content is an array of text/image_url parts
                    parse_content_array(msg["content"], message);
                }
            }
            req.messages.push_back(message);
        }
    }
    
    // Support both max_tokens (deprecated) and max_completion_tokens (newer OpenAI API)
    // max_completion_tokens takes precedence if both are provided
    if (j.contains("max_completion_tokens")) {
        req.max_tokens = j["max_completion_tokens"];
        std::cout << "[ChatCompletionRequest] Parsed max_completion_tokens=" << req.max_tokens << std::endl;
    } else if (j.contains("max_tokens")) {
        req.max_tokens = j["max_tokens"];
        std::cout << "[ChatCompletionRequest] Parsed max_tokens=" << req.max_tokens << std::endl;
    } else {
        std::cout << "[ChatCompletionRequest] No max_tokens specified, using default=" << req.max_tokens << std::endl;
    }
    
    if (j.contains("temperature")) {
        req.temperature = j["temperature"];
    }
    
    if (j.contains("top_p")) {
        req.top_p = j["top_p"];
    }
    
    if (j.contains("top_k")) {
        req.top_k = j["top_k"];
    }
    
    if (j.contains("repeat_penalty")) {
        req.repeat_penalty = j["repeat_penalty"];
    } else if (j.contains("repetition_penalty")) {
        req.repeat_penalty = j["repetition_penalty"];
    } else if (j.contains("frequency_penalty")) {
        req.repeat_penalty = 1.0f + j["frequency_penalty"].get<float>();
    }
    
    if (j.contains("stream")) {
        req.stream = j["stream"];
    }
    
    if (j.contains("stop")) {
        if (j["stop"].is_string()) {
            req.stop.push_back(j["stop"]);
        } else if (j["stop"].is_array()) {
            for (const auto& s : j["stop"]) {
                req.stop.push_back(s);
            }
        }
    }
    
    if (j.contains("tools")) {
        req.tools = j["tools"];
    }
    
    return req;
}

std::string ChatCompletionRequest::toPrompt() const {
    std::ostringstream prompt;
    
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        
        // Simple chat template formatting
        if (msg.role == "system") {
            prompt << "System: " << msg.content << "\n\n";
        } else if (msg.role == "user") {
            prompt << "User: " << msg.content << "\n\n";
        } else if (msg.role == "assistant") {
            prompt << "Assistant: " << msg.content << "\n\n";
        }
    }
    
    // Add final "Assistant: " prompt for the model to complete
    prompt << "Assistant: ";
    
    return prompt.str();
}

} // namespace ryzenai

