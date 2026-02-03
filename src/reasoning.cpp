#include "ryzenai/reasoning.h"
#include <iostream>
#include <algorithm>

namespace ryzenai {

ReasoningParseResult parseReasoningContent(const std::string& text) {
    ReasoningParseResult result;
    result.has_reasoning = false;
    result.is_thinking = false;
    
    // Look for </think> tag
    size_t close_pos = text.find("</think>");
    
    if (close_pos == std::string::npos) {
        // No closing tag found
        // Check if there's an unclosed <think> tag
        size_t open_pos = text.find("<think>");
        if (open_pos != std::string::npos) {
            // Found opening tag without closing - still thinking
            result.regular_content = text.substr(0, open_pos);
            result.reasoning_content = text.substr(open_pos + 7); // Skip "<think>"
            result.has_reasoning = !result.reasoning_content.empty();
            result.is_thinking = true;
        } else {
            // No tags at all - all regular content
            result.regular_content = text;
        }
        return result;
    }
    
    // Found closing tag, look for opening tag
    size_t open_pos = text.rfind("<think>", close_pos);
    
    if (open_pos != std::string::npos) {
        // Both tags found
        result.regular_content = text.substr(0, open_pos);
        result.reasoning_content = text.substr(open_pos + 7, close_pos - (open_pos + 7));
        result.regular_content += text.substr(close_pos + 8); // Append content after </think>
        result.has_reasoning = true;
        result.is_thinking = false;
    } else {
        // Only closing tag found (Qwen3-Thinking style)
        // Treat everything before </think> as reasoning
        result.reasoning_content = text.substr(0, close_pos);
        result.regular_content = text.substr(close_pos + 8);
        result.has_reasoning = true;
        result.is_thinking = false;
    }
    
    return result;
}

// Parse reasoning content using model-specific tags
ReasoningParseResult parseReasoningContentWithModel(const std::string& text, const std::vector<AdditionalToken>& additional_tags) {
    ReasoningParseResult result;
    result.has_reasoning = false;
    result.is_thinking = false;

    // Find thinking start and end tags from model
    std::string think_start, think_end;
    for (const auto& token : additional_tags) {
        if (token.type == SpecialTokenType::THINKING_START) {
            think_start = token.content;
        } else if (token.type == SpecialTokenType::THINKING_END) {
            think_end = token.content;
        }
    }

    // Fallback to defaults if not found
    if (think_start.empty()) think_start = "<think>";
    if (think_end.empty()) think_end = "</think>";
    size_t close_pos = text.rfind(think_end);  // Use rfind to get LAST occurrence!
#if defined(DEBUG)
    std::cout << "[parseReasoning] text length=" << text.length() 
              << ", think_start='" << think_start << "'"
              << ", think_end='" << think_end << "'" << std::endl;
    
    // Debug: Show first 100 chars of text to see what we're parsing
    std::string preview = text.substr(0, std::min(text.length(), size_t(100)));
    std::cout << "[parseReasoning] text preview: '" << preview << "'" << std::endl;

    // For Qwen3-Thinking style (no opening tag), we need to find the LAST </think>
    // because the model might output multiple close tags or a quick close at the start

    std::cout << "[parseReasoning] close_pos (rfind)=" << close_pos << " (npos=" << std::string::npos << ")" << std::endl;
    
    if (close_pos != std::string::npos) {
        std::cout << "[parseReasoning] Found </think> at position " << close_pos 
                  << " out of " << text.length() << " total chars" << std::endl;
    }
#endif
    if (close_pos == std::string::npos) {
        // No closing tag found
        // Check if there's an unclosed opening tag
        size_t open_pos = text.find(think_start);
        if (open_pos != std::string::npos) {
            // Found opening tag without closing - still thinking
            result.regular_content = text.substr(0, open_pos);
            result.reasoning_content = text.substr(open_pos + think_start.length());
            result.has_reasoning = !result.reasoning_content.empty();
            result.is_thinking = true;
        } else {
            // No tags at all - all regular content
            result.regular_content = text;
        }
        return result;
    }

    // Found closing tag, look for opening tag
    size_t open_pos = text.rfind(think_start, close_pos);

    if (open_pos != std::string::npos) {
        // Both tags found
        result.regular_content = text.substr(0, open_pos);
        result.reasoning_content = text.substr(open_pos + think_start.length(), close_pos - (open_pos + think_start.length()));
        result.regular_content += text.substr(close_pos + think_end.length());
        result.has_reasoning = true;
        result.is_thinking = false;
    } else {
        // Only closing tag found (Qwen3-Thinking style)
        // Treat everything before closing tag as reasoning
        result.reasoning_content = text.substr(0, close_pos);
        result.regular_content = text.substr(close_pos + think_end.length());
        result.has_reasoning = true;
        result.is_thinking = false;
        
        // Clean up reasoning content - remove any early </think> tags and prefixes
        // Some models output ": ...\n</think>" at the very start before actual thinking
        size_t early_close = result.reasoning_content.find(think_end);
        if (early_close != std::string::npos && early_close < 50) {
            // Found an early close tag - remove everything up to and including it
            result.reasoning_content = result.reasoning_content.substr(early_close + think_end.length());
#if defined(DEBUG)
            std::cout << "[parseReasoning] Cleaned up early </think> prefix from reasoning" << std::endl;
#endif
        }
        
        // Trim leading whitespace from reasoning content
        size_t first_non_ws = result.reasoning_content.find_first_not_of(" \t\n\r");
        if (first_non_ws != std::string::npos && first_non_ws > 0) {
            result.reasoning_content = result.reasoning_content.substr(first_non_ws);
        }
    }

    return result;
}

// Initialize tag strings from additional_tags_ or use defaults
void ReasoningStreamParser::initializeTags() {
    // Set defaults first
    think_start_tag_ = "<think>";
    think_end_tag_ = "</think>";
    chat_end_tag_ = "<|im_end|>";  // Common chat end tag

    // Override with model-specific tags if available
    for (const auto& token : additional_tags_) {
        if (token.type == SpecialTokenType::THINKING_START && !token.content.empty()) {
            think_start_tag_ = token.content;
        } else if (token.type == SpecialTokenType::THINKING_END && !token.content.empty()) {
            think_end_tag_ = token.content;
        } else if (token.type == SpecialTokenType::CHAT_END && !token.content.empty()) {
            chat_end_tag_ = token.content;
        }
    }

    // Calculate max buffer size based on the longest tag we need to detect
    // Add some margin for safety
    max_buffer_size_ = std::max({
        think_start_tag_.length(),
        think_end_tag_.length(),
        chat_end_tag_.length(),
        size_t(16)  // Minimum buffer size
    }) + 8;  // Extra margin for partial matches across token boundaries
}

// Trim buffer to rolling window size (keeps the end of the buffer)
void ReasoningStreamParser::trimBuffer() {
    if (buffer_.length() > max_buffer_size_) {
        buffer_ = buffer_.substr(buffer_.length() - max_buffer_size_);
    }
}

// Default constructor
ReasoningStreamParser::ReasoningStreamParser()
    : state_(ReasoningState::NORMAL), buffer_(""), use_token_ids_(false), max_buffer_size_(24) {
    initializeTags();
}

// Constructor with model tags
ReasoningStreamParser::ReasoningStreamParser(const std::vector<AdditionalToken>& additional_tags)
    : state_(ReasoningState::NORMAL), buffer_(""), additional_tags_(additional_tags), use_token_ids_(true), max_buffer_size_(24) {
    initializeTags();
}

// Main parsing logic using model-specific tags
ReasoningResult ReasoningStreamParser::consume(const std::string& token) {
    ReasoningResult result;
    result.has_reasoning = false;
    result.thinking_ended = false;
    result.should_stop = false;

    // 1. Append new token to our internal buffer
    buffer_ += token;

    // 2. Check State Transitions
    if (state_ == ReasoningState::NORMAL) {
        // Check for CHAT_END first (should stop generation)
        size_t chat_end_pos = buffer_.find(chat_end_tag_);
        if (chat_end_pos != std::string::npos) {
            // Chat end detected - output content before it and signal stop
            if (chat_end_pos > 0) {
                result.regular_content = buffer_.substr(0, chat_end_pos);
            }
            result.should_stop = true;
            buffer_.clear();
            return result;
        }

        // Check if we are STARTING to think
        size_t think_start = buffer_.find(think_start_tag_);

        if (think_start != std::string::npos) {
            // Found thinking start tag! Switch state.
            state_ = ReasoningState::THINKING;

            // Everything BEFORE the start tag is normal content
            if (think_start > 0) {
                result.regular_content = buffer_.substr(0, think_start);
            }

            // Remove everything up to and including the start tag from buffer
            buffer_ = buffer_.substr(think_start + think_start_tag_.length());

            // If there is leftover in buffer, it is now reasoning content
            // But we need to check if thinking already ended in the same chunk
            size_t think_end = buffer_.find(think_end_tag_);
            if (think_end != std::string::npos) {
                // Thinking started AND ended in same token sequence
                state_ = ReasoningState::NORMAL;
                result.reasoning_content = buffer_.substr(0, think_end);
                result.has_reasoning = !result.reasoning_content.empty();
                result.thinking_ended = true;
                buffer_ = buffer_.substr(think_end + think_end_tag_.length());
            } else if (!buffer_.empty()) {
                result.reasoning_content = buffer_;
                result.has_reasoning = true;
                buffer_.clear();
            }
        }
        else {
            // Check for CLOSING tag without opening (Qwen3 style - implicit start)
            size_t think_end = buffer_.find(think_end_tag_);

            if (think_end != std::string::npos) {
                // Found end tag without preceding start tag - treat everything before as reasoning
                result.reasoning_content = buffer_.substr(0, think_end);
                result.has_reasoning = true;
                result.thinking_ended = true;

                // Everything after the end tag is regular content
                buffer_ = buffer_.substr(think_end + think_end_tag_.length());

                // Leftover is regular content
                if (!buffer_.empty()) {
                    result.regular_content = buffer_;
                    buffer_.clear();
                }
            }
            else {
                // Check if we have a PARTIAL tag at the end that we should wait for
                bool has_partial = is_potential_tag(buffer_) || 
                                   is_potential_closing_tag(buffer_) ||
                                   is_potential_chat_end_tag(buffer_);
                
                if (!has_partial) {
                    // No partial tags - safe to output buffer content
                    result.regular_content = buffer_;
                    buffer_.clear();
                } else {
                    // Has partial tag - check if only the END of the buffer is partial
                    // Output everything except the potential partial match
                    size_t safe_output_len = 0;
                    for (size_t i = 1; i <= buffer_.length() && i <= max_buffer_size_; ++i) {
                        std::string suffix = buffer_.substr(buffer_.length() - i);
                        if (is_potential_tag(suffix) || 
                            is_potential_closing_tag(suffix) ||
                            is_potential_chat_end_tag(suffix)) {
                            // Found potential partial tag starting at this position
                            safe_output_len = buffer_.length() - i;
                            break;
                        }
                    }
                    
                    if (safe_output_len > 0) {
                        result.regular_content = buffer_.substr(0, safe_output_len);
                        buffer_ = buffer_.substr(safe_output_len);
                    }
                    // If safe_output_len is 0, we output nothing and wait for next token
                }
            }
        }
    }
    else if (state_ == ReasoningState::THINKING) {
        // Check if we are FINISHING thinking
        size_t think_end = buffer_.find(think_end_tag_);
        
        if (think_end != std::string::npos) {
            // Found the end! Switch back to normal.
            state_ = ReasoningState::NORMAL;
            result.thinking_ended = true;

            // Everything BEFORE the end tag is reasoning
            if (think_end > 0) {
                result.reasoning_content = buffer_.substr(0, think_end);
                result.has_reasoning = true;
            }

            // Remove everything up to and including the end tag
            buffer_ = buffer_.substr(think_end + think_end_tag_.length());

            // Leftover is regular content
            if (!buffer_.empty()) {
                result.regular_content = buffer_;
                buffer_.clear();
            }
        }
        else {
            // Check for PARTIAL closing tag
            bool has_partial = is_potential_closing_tag(buffer_);
            
            if (!has_partial) {
                result.reasoning_content = buffer_;
                result.has_reasoning = true;
                buffer_.clear();
            } else {
                // Find safe output length (content before potential partial tag)
                size_t safe_output_len = 0;
                for (size_t i = 1; i <= buffer_.length() && i <= max_buffer_size_; ++i) {
                    std::string suffix = buffer_.substr(buffer_.length() - i);
                    if (is_potential_closing_tag(suffix)) {
                        safe_output_len = buffer_.length() - i;
                        break;
                    }
                }
                
                if (safe_output_len > 0) {
                    result.reasoning_content = buffer_.substr(0, safe_output_len);
                    result.has_reasoning = true;
                    buffer_ = buffer_.substr(safe_output_len);
                }
            }
        }
    }

    // Apply rolling buffer limit to prevent unbounded growth
    trimBuffer();

    return result;
}

// Helper to detect partials for thinking start tag
bool ReasoningStreamParser::is_potential_tag(const std::string& s) const {
    if (s.empty() || s.length() > think_start_tag_.length()) return false;
    return think_start_tag_.compare(0, s.length(), s) == 0;
}

// Helper to detect partials for thinking end tag
bool ReasoningStreamParser::is_potential_closing_tag(const std::string& s) const {
    if (s.empty() || s.length() > think_end_tag_.length()) return false;
    return think_end_tag_.compare(0, s.length(), s) == 0;
}

// Helper to detect partials for chat end tag
bool ReasoningStreamParser::is_potential_chat_end_tag(const std::string& s) const {
    if (s.empty() || s.length() > chat_end_tag_.length()) return false;
    return chat_end_tag_.compare(0, s.length(), s) == 0;
}

// Token ID-based consume method (more efficient)
ReasoningResult ReasoningStreamParser::consume(int32_t token_id, const std::string& decoded_text) {
    ReasoningResult result;
    result.has_reasoning = false;
    result.thinking_ended = false;
    result.should_stop = false;

    // If not using token IDs or no tags available, fall back to string-based parsing
    if (!use_token_ids_ || additional_tags_.empty()) {
        return consume(decoded_text);
    }

    // Token ID-based parsing
    if (state_ == ReasoningState::NORMAL) {
        // Check for chat end token first
        if (is_chat_end_token(token_id)) {
            result.should_stop = true;
            return result;
        }
        
        if (is_thinking_start_token(token_id)) {
            // Started thinking - switch to thinking state
            state_ = ReasoningState::THINKING;
            // The start token itself is not part of the reasoning content
        } else if (is_thinking_end_token(token_id)) {
            // Found end tag without start (Qwen3 style) - treat previous content as reasoning
            // This is rare in token ID mode since we should catch the start token first
            result.thinking_ended = true;
            // No content to output here - the end token marks completion
        } else {
            // Regular content token
            result.regular_content = decoded_text;
        }
    } else if (state_ == ReasoningState::THINKING) {
        if (is_thinking_end_token(token_id)) {
            // Finished thinking - switch back to normal state
            state_ = ReasoningState::NORMAL;
            result.thinking_ended = true;
            // The end token itself is not part of the content
        } else if (is_chat_end_token(token_id)) {
            // Chat ended while thinking - stop generation
            state_ = ReasoningState::NORMAL;
            result.thinking_ended = true;
            result.should_stop = true;
        } else {
            // Reasoning content token
            result.reasoning_content = decoded_text;
            result.has_reasoning = true;
        }
    }

    return result;
}

// Check if token ID matches a thinking start tag
bool ReasoningStreamParser::is_thinking_start_token(int32_t token_id) const {
    for (const auto& token : additional_tags_) {
        if (token.type == SpecialTokenType::THINKING_START && token.token_id == token_id) {
            return true;
        }
    }
    return false;
}

// Check if token ID matches a thinking end tag
bool ReasoningStreamParser::is_thinking_end_token(int32_t token_id) const {
    for (const auto& token : additional_tags_) {
        if (token.type == SpecialTokenType::THINKING_END && token.token_id == token_id) {
            return true;
        }
    }
    return false;
}

// Check if token ID matches a chat end tag
bool ReasoningStreamParser::is_chat_end_token(int32_t token_id) const {
    for (const auto& token : additional_tags_) {
        if (token.type == SpecialTokenType::CHAT_END && token.token_id == token_id) {
            return true;
        }
    }
    return false;
}

} // namespace ryzenai
