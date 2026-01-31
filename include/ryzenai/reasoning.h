#pragma once

#include <string>
#include <utility>
#include <vector>
#include "ryzenai/types.h"
#include "ryzenai/mlx/model.h"

namespace ryzenai {

// Result of parsing reasoning content from model output
struct ReasoningParseResult {
    std::string reasoning_content;  // Content inside <think> tags (without the tags)
    std::string regular_content;     // Content outside <think> tags
    bool has_reasoning;              // True if reasoning content was found
    bool is_thinking;                // True if still inside unclosed <think> tag
};

// Parse reasoning content from model output
// Extracts content between <think> and </think> tags
// If only </think> is found (no opening tag), treats everything before it as reasoning
ReasoningParseResult parseReasoningContent(const std::string& text);

// Parse reasoning content using model-specific tags
// Extracts content between model-defined thinking tags
ReasoningParseResult parseReasoningContentWithModel(const std::string& text, const std::vector<AdditionalToken>& additional_tags);

// Reasoning state for streaming parser
enum class ReasoningState {
    NORMAL,
    THINKING
};

// Result of consuming a token in streaming parser
struct ReasoningResult {
    std::string reasoning_content;
    std::string regular_content;
    bool has_reasoning;
    bool thinking_ended;
    bool should_stop;
};

// For streaming: tracks state across multiple token callbacks
class ReasoningStreamParser {
public:
    ReasoningStreamParser();
    ReasoningStreamParser(const std::vector<AdditionalToken>& additional_tags);

    // Process a single token
    // Returns ReasoningResult with content separated
    ReasoningResult consume(const std::string& token);

    // Process a single token by ID (more efficient)
    // Returns ReasoningResult with content separated
    ReasoningResult consume(int32_t token_id, const std::string& decoded_text);

private:
    ReasoningState state_;
    std::string buffer_;
    std::vector<AdditionalToken> additional_tags_;
    bool use_token_ids_;
    
    // Model-specific tag strings (resolved from additional_tags_ or defaults)
    std::string think_start_tag_;
    std::string think_end_tag_;
    std::string chat_end_tag_;
    size_t max_buffer_size_;  // Rolling buffer limit

    // Helper to detect partial opening tags like "<", "<t", "<thi"...
    bool is_potential_tag(const std::string& s) const;

    // Helper to detect partial closing tags like "</", "</t", "</th"...
    bool is_potential_closing_tag(const std::string& s) const;
    
    // Helper to detect partial chat end tags
    bool is_potential_chat_end_tag(const std::string& s) const;

    // Check if token ID matches a thinking start/end tag
    bool is_thinking_start_token(int32_t token_id) const;
    bool is_thinking_end_token(int32_t token_id) const;
    bool is_chat_end_token(int32_t token_id) const;
    
    // Initialize tag strings from additional_tags_ or use defaults
    void initializeTags();
    
    // Trim buffer to rolling window size
    void trimBuffer();
};

} // namespace ryzenai

