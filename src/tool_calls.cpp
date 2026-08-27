#include "ryzenai/tool_calls.h"
#include <iostream>
#include <ctime>
#include <optional>
#include <cctype>

namespace ryzenai {

namespace {

// JSON forbids raw control characters inside string literals. Both the repair
// path (model output we are trying to salvage) and the emit path (strings we
// hand-build into SSE frames) need the same mapping, so it lives here once.
void appendEscapedControlChar(std::string& out, unsigned char c) {
    static constexpr char hex[] = "0123456789abcdef";
    switch (c) {
        case '\b': out += "\\b"; return;
        case '\f': out += "\\f"; return;
        case '\n': out += "\\n"; return;
        case '\r': out += "\\r"; return;
        case '\t': out += "\\t"; return;
        default:
            out += "\\u00";
            out += hex[(c >> 4) & 0x0f];
            out += hex[c & 0x0f];
            return;
    }
}

std::string trimWhitespace(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = input.find_last_not_of(" \t\n\r");
    return input.substr(start, end - start + 1);
}

// Only reached from the repair path, i.e. after strict json::parse has already
// rejected the text. In that situation a lone backslash is far more often an
// under-escaped Windows path separator ("C:\tmp", "C:\build") than a deliberate
// \t / \n / \r / \b / \f escape, so those are NOT treated as structural and get
// doubled into literal backslashes by the caller.
//
// Quote, backslash, solidus and \uXXXX are structural -- they decide where the
// string ends -- so they must always be preserved as escapes.
bool isStructuralJsonEscape(const std::string& input, size_t backslash_index) {
    if (backslash_index + 1 >= input.size()) {
        return false;
    }
    const char next = input[backslash_index + 1];
    switch (next) {
        case '"':
        case '\\':
        case '/':
            return true;
        case 'u':
            // \u must be followed by exactly 4 hex digits; bare \users is not valid JSON.
            if (backslash_index + 5 >= input.size()) {
                return false;
            }
            for (size_t j = backslash_index + 2; j < backslash_index + 6; ++j) {
                if (!std::isxdigit(static_cast<unsigned char>(input[j]))) {
                    return false;
                }
            }
            return true;
        default:
            return false;
    }
}

// LLMs often emit literal newlines or Windows paths with single backslashes inside
// JSON string values. Repair those before json::parse.
std::string sanitizeJsonStringLiterals(const std::string& input) {
    std::string result;
    result.reserve(input.size() + 32);

    bool in_string = false;
    bool escape_next = false;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (escape_next) {
            result += c;
            escape_next = false;
            continue;
        }

        if (!in_string) {
            if (c == '"') {
                in_string = true;
            }
            result += c;
            continue;
        }

        // Inside a JSON string value
        if (c == '\\') {
            if (isStructuralJsonEscape(input, i)) {
                result += c;
                escape_next = true;
            } else {
                result += "\\\\";
            }
        } else if (c == '"') {
            in_string = false;
            result += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            appendEscapedControlChar(result, static_cast<unsigned char>(c));
        } else {
            result += c;
        }
    }

    return result;
}

std::string balanceJsonBrackets(const std::string& input) {
    int brace_depth = 0;
    int bracket_depth = 0;
    bool in_string = false;
    bool escape_next = false;

    for (char c : input) {
        if (escape_next) {
            escape_next = false;
            continue;
        }

        if (in_string) {
            if (c == '\\') {
                escape_next = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++brace_depth;
        } else if (c == '}') {
            --brace_depth;
        } else if (c == '[') {
            ++bracket_depth;
        } else if (c == ']') {
            --bracket_depth;
        }
    }

    std::string result = input;
    while (bracket_depth > 0) {
        result += ']';
        --bracket_depth;
    }
    while (brace_depth > 0) {
        result += '}';
        --brace_depth;
    }
    return result;
}

json tryParseJson(const std::string& text) {
    try {
        return json::parse(text);
    } catch (const json::exception&) {
        return json();
    }
}

json parseToolCallJsonObject(const std::string& raw) {
    const std::string trimmed = trimWhitespace(raw);
    if (trimmed.empty()) {
        return json();
    }

    json result = tryParseJson(trimmed);
    if (!result.is_null()) {
        return result;
    }

    const std::string sanitized = sanitizeJsonStringLiterals(trimmed);
    result = tryParseJson(sanitized);
    if (!result.is_null()) {
        std::cout << "[ToolCalls DEBUG] Parsed after sanitizing JSON string literals" << std::endl;
        return result;
    }

    const std::string balanced = balanceJsonBrackets(sanitized);
    result = tryParseJson(balanced);
    if (!result.is_null()) {
        std::cout << "[ToolCalls DEBUG] Parsed after balancing JSON brackets" << std::endl;
        return result;
    }

    return json();
}

json normalizeToolCallArguments(const json& arguments) {
    if (arguments.is_string()) {
        const std::string raw = arguments.get<std::string>();
        json parsed = parseToolCallJsonObject(raw);
        if (!parsed.is_null()) {
            return parsed;
        }
        try {
            return json::parse(raw);
        } catch (const json::exception&) {
            return json{{"raw", raw}};
        }
    }
    return arguments;
}

std::string toolCallArgumentsString(const json& arguments) {
    if (arguments.is_string()) {
        return arguments.get<std::string>();
    }
    if (arguments.is_object() || arguments.is_array()) {
        return arguments.dump();
    }
    return arguments.dump();
}

std::optional<ToolCall> toolCallFromJsonObject(const json& tool_call_obj) {
    if (!tool_call_obj.is_object()) {
        return std::nullopt;
    }

    ToolCall tool_call;
    const json* fields = &tool_call_obj;

    if (tool_call_obj.contains("function") && tool_call_obj["function"].is_object()) {
        fields = &tool_call_obj["function"];
    }

    if (fields->contains("name") && (*fields)["name"].is_string()) {
        tool_call.name = (*fields)["name"];
    } else {
        std::cerr << "[WARNING] Tool call missing 'name' field, skipping" << std::endl;
        return std::nullopt;
    }

    if (fields->contains("arguments")) {
        tool_call.arguments = normalizeToolCallArguments((*fields)["arguments"]);
    } else if (fields->contains("parameters")) {
        tool_call.arguments = normalizeToolCallArguments((*fields)["parameters"]);
    } else {
        std::cerr << "[WARNING] Tool call missing 'arguments' or 'parameters' field, skipping" << std::endl;
        return std::nullopt;
    }

    return tool_call;
}

std::optional<ToolCall> parseToolCallPayload(const std::string& tool_call_json) {
    json tool_call_obj = parseToolCallJsonObject(tool_call_json);
    if (tool_call_obj.is_null()) {
        return std::nullopt;
    }
    return toolCallFromJsonObject(tool_call_obj);
}

void eraseMatch(std::string& cleaned_text, size_t match_pos, size_t match_len) {
    if (match_pos < cleaned_text.size()) {
        cleaned_text.erase(match_pos, std::min(match_len, cleaned_text.size() - match_pos));
    }
}

}  // namespace

std::string escapeJsonString(const std::string& s) {
    std::string escaped;
    escaped.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') {
            escaped += '\\';
            escaped += static_cast<char>(c);
        } else if (c < 0x20) {
            appendEscapedControlChar(escaped, c);
        } else {
            escaped.push_back(static_cast<char>(c));
        }
    }
    return escaped;
}

std::pair<std::vector<ToolCall>, std::string> extractToolCalls(const std::string& text) {
    std::vector<ToolCall> tool_calls;
    std::string cleaned_text = text;

    std::cout << "[ToolCalls DEBUG] Extracting tool calls from text (" << text.length() << " chars)" << std::endl;
    std::cout << "[ToolCalls DEBUG] Text: " << text.substr(0, std::min(size_t(300), text.length())) << std::endl;

    // Qwen-style tool calls: <tool_call>...</tool_call>
    const std::regex closed_tool_call_pattern(
        R"(<tool_call>([\s\S]*?)</tool_call>)",
        std::regex::icase | std::regex::ECMAScript);

    std::smatch match;
    std::string search_text = text;
    size_t offset = 0;
    size_t removed = 0;

    while (std::regex_search(search_text, match, closed_tool_call_pattern)) {
        const std::string tool_call_json = match[1].str();

        std::cout << "[ToolCalls DEBUG] Found Qwen-style match, JSON: " << tool_call_json << std::endl;

        if (auto tool_call = parseToolCallPayload(tool_call_json)) {
            tool_calls.push_back(*tool_call);
            eraseMatch(cleaned_text, offset + match.position() - removed, match.length());
            removed += match.length();
        } else {
            std::cerr << "[WARNING] Failed to parse tool call JSON after repair attempts" << std::endl;
        }

        search_text = match.suffix();
        offset += match.position() + match.length();
    }

    // Handle truncated generations that open <tool_call> but never emit </tool_call>.
    const std::regex open_tool_call_pattern(
        R"(<tool_call>([\s\S]+)$)",
        std::regex::icase | std::regex::ECMAScript);

    search_text = cleaned_text;
    offset = 0;
    removed = 0;
    while (std::regex_search(search_text, match, open_tool_call_pattern)) {
        const std::string tool_call_json = match[1].str();

        std::cout << "[ToolCalls DEBUG] Found unclosed Qwen-style match, JSON: " << tool_call_json << std::endl;

        if (auto tool_call = parseToolCallPayload(tool_call_json)) {
            tool_calls.push_back(*tool_call);
            eraseMatch(cleaned_text, offset + match.position() - removed, match.length());
            removed += match.length();
        } else {
            std::cerr << "[WARNING] Failed to parse unclosed tool call JSON after repair attempts" << std::endl;
        }

        search_text = match.suffix();
        offset += match.position() + match.length();
    }

    // Mistral-style: [TOOL_CALLS] [...]
    const std::regex mistral_pattern(
        R"(\[TOOL_CALLS\]\s*\[([\s\S]*?)\])",
        std::regex::icase | std::regex::ECMAScript);

    search_text = cleaned_text;
    offset = 0;
    removed = 0;

    while (std::regex_search(search_text, match, mistral_pattern)) {
        const std::string tool_calls_array_json = "[" + match[1].str() + "]";

        json tool_calls_array = parseToolCallJsonObject(tool_calls_array_json);
        if (tool_calls_array.is_array()) {
            for (const auto& tool_call_obj : tool_calls_array) {
                if (auto tool_call = toolCallFromJsonObject(tool_call_obj)) {
                    tool_calls.push_back(*tool_call);
                }
            }
            eraseMatch(cleaned_text, offset + match.position() - removed, match.length());
            removed += match.length();
        } else {
            std::cerr << "[WARNING] Failed to parse [TOOL_CALLS] JSON after repair attempts" << std::endl;
        }

        search_text = match.suffix();
        offset += match.position() + match.length();
    }

    cleaned_text = trimWhitespace(cleaned_text);

    std::cout << "[ToolCalls DEBUG] Extracted " << tool_calls.size() << " tool call(s)" << std::endl;

    return {tool_calls, cleaned_text};
}

json formatToolCallsForOpenAI(const std::vector<ToolCall>& tool_calls) {
    json openai_tool_calls = json::array();

    int index = 0;
    for (const auto& tool_call : tool_calls) {
        std::string tool_call_id = "call_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(index++);

        json openai_tool_call = {
            {"id", tool_call_id},
            {"type", "function"},
            {"function", {
                {"name", tool_call.name},
                {"arguments", toolCallArgumentsString(tool_call.arguments)}
            }}
        };
        openai_tool_calls.push_back(openai_tool_call);
    }

    return openai_tool_calls;
}

std::vector<std::string> buildToolCallStreamChunks(const json& openai_tool_calls,
                                                   const std::string& model_id) {
    std::vector<std::string> chunks;
    if (!openai_tool_calls.is_array() || openai_tool_calls.empty()) {
        return chunks;
    }

    const std::time_t created = std::time(nullptr);
    const std::string completion_id = "chatcmpl-" + std::to_string(created);

    for (size_t i = 0; i < openai_tool_calls.size(); ++i) {
        const json& tool_call = openai_tool_calls[i];
        if (!tool_call.is_object() || !tool_call.contains("function")) {
            continue;
        }

        json chunk = {
            {"id", completion_id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model_id},
            {"choices", json::array({
                {
                    {"index", 0},
                    {"delta", {
                        {"tool_calls", json::array({
                            {
                                {"index", static_cast<int>(i)},
                                {"id", tool_call.value("id", "")},
                                {"type", tool_call.value("type", "function")},
                                {"function", tool_call["function"]}
                            }
                        })}
                    }},
                    {"finish_reason", nullptr}
                }
            })}
        };
        chunks.push_back(chunk.dump());
    }

    return chunks;
}

}  // namespace ryzenai
