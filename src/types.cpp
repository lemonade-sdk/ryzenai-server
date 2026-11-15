#include "ryzenai/types.h"
#include <cctype>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace ryzenai {

static std::vector<uint8_t> decode_base64(const std::string& input) {
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> out;
    out.reserve(input.size() * 3 / 4);
    int val = 0, bits = -8;
    for (unsigned char c : input) {
        if (c == '=') break;
        if (std::isspace(c)) continue;
        auto pos = chars.find(static_cast<char>(c));
        if (pos == std::string::npos) {
            throw std::invalid_argument("Media payload is not valid base64");
        }
        val = (val << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("Media payload is empty");
    }
    return out;
}

// Strips a "data:<mime>;base64," prefix; returns an empty string if `url` is not a data URL.
static std::string data_url_payload(const std::string& url) {
    const std::string marker = ";base64,";
    if (url.rfind("data:", 0) != 0) return "";
    auto pos = url.find(marker);
    return pos == std::string::npos ? "" : url.substr(pos + marker.size());
}

static void parse_content_parts(const json& parts, ChatMessage& message) {
    for (const auto& part : parts) {
        std::string type = part.value("type", "");
        if (type == "text") {
            std::string text = part.value("text", "");
            if (!message.content.empty()) message.content += "\n";
            message.content += text;
            message.parts.push_back({{"type", "text"}, {"text", text}});
        } else if (type == "image_url") {
            std::string url = part.contains("image_url") && part["image_url"].is_object()
                ? part["image_url"].value("url", "") : "";
            std::string payload = data_url_payload(url);
            if (payload.empty()) {
                throw std::invalid_argument(
                    "image_url must be a base64 data URL (data:<mime>;base64,...)");
            }
            message.media.images.push_back(decode_base64(payload));
            message.parts.push_back({{"type", "image"}});
        } else if (type == "input_audio") {
            if (!part.contains("input_audio") || !part["input_audio"].is_object()) {
                throw std::invalid_argument("input_audio part is missing its input_audio object");
            }
            std::string data = part["input_audio"].value("data", "");
            std::string payload = data_url_payload(data);
            message.media.audios.push_back(decode_base64(payload.empty() ? data : payload));
            message.parts.push_back({{"type", "audio"}});
        } else {
            throw std::invalid_argument("Unsupported message content part type: '" + type + "'");
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
                const auto& content = msg["content"];
                if (content.is_string()) {
                    message.content = content.get<std::string>();
                } else if (content.is_array()) {
                    parse_content_parts(content, message);
                }
            }
            req.messages.push_back(std::move(message));
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

