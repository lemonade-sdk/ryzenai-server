#pragma once

#include <string>
#include <vector>
#include <json.hpp>
#include <regex>

namespace ryzenai {

using json = nlohmann::json;

// Extracted tool call structure
struct ToolCall {
    std::string name;
    json arguments;
};

// Escape a raw string so it is safe to embed in a JSON string literal.
// Handles quotes, backslashes, the named control escapes, and any remaining
// character below 0x20 as \u00XX.
std::string escapeJsonString(const std::string& s);

// Extract tool calls from generated text (Qwen format: <tool_call>...</tool_call>)
// Returns: pair of (extracted_tool_calls, cleaned_text_without_tool_calls)
std::pair<std::vector<ToolCall>, std::string> extractToolCalls(const std::string& text);

// Format tool calls in OpenAI API format
json formatToolCallsForOpenAI(const std::vector<ToolCall>& tool_calls);

// Build chat.completion.chunk JSON payloads (without the "data: " prefix) for
// streaming tool_calls deltas. Uses the same OpenAI shape as formatToolCallsForOpenAI.
std::vector<std::string> buildToolCallStreamChunks(const json& openai_tool_calls,
                                                   const std::string& model_id);

} // namespace ryzenai

