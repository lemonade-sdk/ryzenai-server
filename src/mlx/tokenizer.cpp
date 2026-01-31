/*
 * tokenizer.cpp
 * * Robust Universal Tokenizer
 * * Features: Special Token Handling, Universal Cleanup, Defensive Parsing
 */

#include "ryzenai/mlx/tokenizer.h"
#include <json.hpp>
#include <sentencepiece_processor.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <codecvt>
#include <locale>

namespace fs = std::filesystem;
using json = nlohmann::json;

thread_local std::string tl_decoded_buffer;

// =========================================================
// UNIVERSAL CLEANUP
// =========================================================
void universal_cleanup(std::string& text) {
    auto replace_all = [&](const std::string& from, const std::string& to) {
        if (from.empty()) return;
        size_t start_pos = 0;
        while ((start_pos = text.find(from, start_pos)) != std::string::npos) {
            text.replace(start_pos, from.length(), to);
            start_pos += to.length(); 
        }
    };
    
    // Qwen/GPT-2
    replace_all("\xC4\x8A", "\n"); // Ċ
    replace_all("\xC4\xA0", " ");  // Ġ
    replace_all("\xC4\x89", "\t"); // ĉ
    // Llama/SP
    replace_all("\xE2\x96\x81", " "); // _
}

// =========================================================
// SENTENCEPIECE
// =========================================================
class SentencePieceBackend : public TokenizerBackend {
    sentencepiece::SentencePieceProcessor processor;
public:
    SentencePieceBackend(const std::string& path) {
        if (!processor.Load(path).ok()) throw std::runtime_error("Failed to load SentencePiece model");
    }
    void Encode(const std::string& text, std::vector<int32_t>& ids) override {
        std::vector<int> sp_ids;
        processor.Encode(text, &sp_ids);
        ids.assign(sp_ids.begin(), sp_ids.end());
    }
    std::string Decode(const std::vector<int32_t>& ids) override {
        std::string text;
        std::vector<int> sp_ids(ids.begin(), ids.end());
        processor.Decode(sp_ids, &text);
        return text;
    }
};

// =========================================================
// HUGGINGFACE BPE (With Special Token Support)
// =========================================================
class HuggingFaceBackend : public TokenizerBackend {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> reverse_vocab;
    std::unordered_map<std::string, int> bpe_ranks;
    std::vector<std::pair<std::string, int32_t>> special_tokens; // For Encode
    std::regex pat;
    int32_t unk_token_id = 0;

public:
    HuggingFaceBackend(const std::string& json_path) {
        std::cout << "[Tokenizer] Parsing " << json_path << "..." << std::endl;
        std::ifstream f(json_path);
        json j = json::parse(f);

        // 1. Load Vocab
        auto load_vocab = [&](const json& v_obj) {
            for (auto& [token, id] : v_obj.items()) {
                if (id.is_number_integer()) {
                    int32_t i = id.template get<int32_t>();
                    vocab[token] = i;
                    reverse_vocab[i] = token;
                }
            }
        };

        try {
            if (j.contains("model") && j["model"].contains("vocab")) load_vocab(j["model"]["vocab"]);
            else if (j.contains("vocab")) load_vocab(j["vocab"]);
        } catch (...) {}
        
        std::cout << "[Tokenizer] Vocab size: " << reverse_vocab.size() << std::endl;

        // 2. Load Merges
        try {
            if (j.contains("model") && j["model"].contains("merges")) {
                auto& merges = j["model"]["merges"];
                if (merges.is_array()) {
                    int rank = 0;
                    for (const auto& merge : merges) {
                        if (merge.is_string()) bpe_ranks[merge.get<std::string>()] = rank++;
                    }
                }
            }
        } catch (...) {}

        // 3. Added Tokens (Special Tokens)
        try {
            if (j.contains("added_tokens")) {
                for (const auto& t : j["added_tokens"]) {
                    if (t.contains("id") && t.contains("content")) {
                        int32_t id = t["id"].get<int32_t>();
                        std::string content = t["content"].get<std::string>();
                        vocab[content] = id;
                        reverse_vocab[id] = content;
                        special_tokens.push_back({content, id});
                    }
                }
            }
        } catch (...) {}

        // Sort special tokens by length (longest first) to match greedy
        std::sort(special_tokens.begin(), special_tokens.end(), [](const auto& a, const auto& b) {
            return a.first.length() > b.first.length();
        });
        
        // 4. Regex
        pat = std::regex(R"(\s+\S+|\S+)", std::regex::optimize);
    }

    void bpe(const std::string& token, std::vector<std::string>& bpe_tokens) {
        std::vector<std::string> word;
        for (size_t i = 0; i < token.length();) {
            unsigned char c = token[i];
            size_t n = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
            if (i + n > token.length()) n = 1;
            word.push_back(token.substr(i, n));
            i += n;
        }

        if (word.empty()) return;

        while (word.size() > 1) {
            int min_rank = std::numeric_limits<int>::max();
            int best_idx = -1;

            for (size_t i = 0; i < word.size() - 1; ++i) {
                std::string pair_key = word[i] + " " + word[i+1];
                auto it = bpe_ranks.find(pair_key);
                if (it != bpe_ranks.end() && it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = i;
                }
            }

            if (best_idx == -1) break;

            std::string merged = word[best_idx] + word[best_idx+1];
            word[best_idx] = merged;
            word.erase(word.begin() + best_idx + 1);
        }
        bpe_tokens.insert(bpe_tokens.end(), word.begin(), word.end());
    }

    // UPDATED ENCODE: Handles Special Tokens correctly!
    void Encode(const std::string& text, std::vector<int32_t>& ids) override {
        size_t start = 0;
        
        while (start < text.length()) {
            // 1. Check for Special Tokens at current position
            bool found_special = false;
            for (const auto& [spec_str, spec_id] : special_tokens) {
                if (text.compare(start, spec_str.length(), spec_str) == 0) {
                    ids.push_back(spec_id);
                    start += spec_str.length();
                    found_special = true;
                    break;
                }
            }
            if (found_special) continue;

            // 2. Find next chunk of regular text (until next special token or end)
            size_t next_special_pos = std::string::npos;
            for (const auto& [spec_str, _] : special_tokens) {
                size_t pos = text.find(spec_str, start);
                if (pos != std::string::npos) {
                    if (next_special_pos == std::string::npos || pos < next_special_pos) {
                        next_special_pos = pos;
                    }
                }
            }
            
            size_t chunk_len = (next_special_pos == std::string::npos) ? std::string::npos : next_special_pos - start;
            std::string chunk = text.substr(start, chunk_len);
            
            if (!chunk.empty()) {
                // 3. Run BPE on the regular text chunk
                std::sregex_iterator it(chunk.begin(), chunk.end(), pat);
                std::sregex_iterator end;
                for (; it != end; ++it) {
                    std::string token = it->str();
                    std::vector<std::string> bpe_tokens;
                    bpe(token, bpe_tokens);
                    for (const auto& t : bpe_tokens) {
                        if (vocab.count(t)) ids.push_back(vocab[t]);
                        else {
                            // Byte fallback
                            for (unsigned char c : t) {
                                std::stringstream ss; ss << "<0x" << std::hex << std::uppercase << (c < 16 ? "0" : "") << (int)c << ">";
                                if (vocab.count(ss.str())) { ids.push_back(vocab[ss.str()]); continue; }
                                std::string raw(1, (char)c);
                                if (vocab.count(raw)) { ids.push_back(vocab[raw]); continue; }
                                ids.push_back(unk_token_id);
                            }
                        }
                    }
                }
            }
            
            if (next_special_pos == std::string::npos) break;
            start = next_special_pos;
        }
    }

    std::string Decode(const std::vector<int32_t>& ids) override {
        std::string text;
        for (int32_t id : ids) {
            if (reverse_vocab.count(id)) {
                std::string token = reverse_vocab[id];
                if (token.size() == 6 && token.substr(0,3) == "<0x" && token.back() == '>') {
                    try {
                        int b = std::stoi(token.substr(3,2), nullptr, 16);
                        text += (char)b;
                        continue;
                    } catch(...) {}
                }
                text += token;
            }
        }
        return text;
    }
};

// =========================================================
// OGA TOKENIZER
// =========================================================

std::unique_ptr<MlxOgaSequences> MlxOgaSequences::Create() { return std::make_unique<MlxOgaSequences>(); }
const int32_t* MlxOgaSequences::SequenceData(int) const { return ids.data(); }
size_t MlxOgaSequences::SequenceCount(int) const { return ids.size(); }

std::unique_ptr<MlxOgaTokenizer> MlxOgaTokenizer::Create(const MlxOgaModel& model) {
    auto tok = std::make_unique<MlxOgaTokenizer>();

    // 1. HuggingFace JSON
    std::string hf_path = model.model_path + "/tokenizer.json";
    if (fs::exists(hf_path)) {
        try {
            tok->backend = std::make_unique<HuggingFaceBackend>(hf_path);
            std::cout << "[Tokenizer] Loaded HuggingFace backend" << std::endl;
        } catch(const std::exception& e) {
            std::cerr << "[Tokenizer] JSON load failed: " << e.what() << std::endl;
        }
    }

    // 2. SentencePiece
    if (!tok->backend) {
        std::string sp_path = model.model_path + "/tokenizer.model";
        if (fs::exists(sp_path)) {
            try {
                tok->backend = std::make_unique<SentencePieceBackend>(sp_path);
                std::cout << "[Tokenizer] Loaded SentencePiece backend" << std::endl;
            } catch(...) {}
        }
    }

    std::string config_path = model.model_path + "/tokenizer_config.json";
    if (fs::exists(config_path)) {
        try {
            std::ifstream f(config_path);
            json config = json::parse(f);
            if (config.contains("chat_template") && config["chat_template"].is_string()) {
                tok->chat_template_str = config["chat_template"].get<std::string>();
            }
        } catch(...) {}
    }

    if (!tok->backend) std::cerr << "[Tokenizer] ERROR: No backend loaded!" << std::endl;
    return tok;
}

void MlxOgaTokenizer::Encode(const char* text, MlxOgaSequences& sequences) {
    if (backend) backend->Encode(text, sequences.ids);
}

oga_char_ptr MlxOgaTokenizer::Decode(const int32_t* tokens, size_t count) {
    if (!backend) return "";
    std::vector<int32_t> ids(tokens, tokens + count);
    std::string raw = backend->Decode(ids);
    universal_cleanup(raw);
    tl_decoded_buffer = raw;
    return tl_decoded_buffer.c_str();
}

oga_char_ptr MlxOgaTokenizer::ApplyChatTemplate(const char* template_str, const char* messages_json, 
                                              const char* tools_json, bool add_generation_prompt) {
    std::string active_template = (template_str && template_str[0]) ? template_str : chat_template_str;
    enum class TemplateType { ChatML, Llama2, Llama3, Phi3, Gemma, Mistral, Vicuna, Unknown };
    TemplateType type = TemplateType::Unknown;

    if (active_template.find("<|im_start|>") != std::string::npos) type = TemplateType::ChatML;
    else if (active_template.find("<|start_header_id|>") != std::string::npos) type = TemplateType::Llama3;
    else if (active_template.find("<|user|>") != std::string::npos) type = TemplateType::Phi3;
    else if (active_template.find("<start_of_turn>") != std::string::npos) type = TemplateType::Gemma;
    else if (active_template.find("[INST]") != std::string::npos) {
        type = (active_template.find("<<SYS>>") != std::string::npos) ? TemplateType::Llama2 : TemplateType::Mistral;
    }

    json messages;
    try { messages = json::parse(messages_json); } catch(...) { return ""; }
    
    std::ostringstream ss;
    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");
        std::string content = msg.value("content", "");
        switch(type) {
            case TemplateType::ChatML: ss << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n"; break;
            case TemplateType::Llama3: ss << "<|start_header_id|>" << role << "<|end_header_id|>\n\n" << content << "<|eot_id|>"; break;
            case TemplateType::Phi3: ss << "<|" << role << "|>\n" << content << "<|end|>\n"; break;
            case TemplateType::Gemma: ss << "<start_of_turn>" << (role=="assistant"?"model":role) << "\n" << content << "<end_of_turn>\n"; break;
            case TemplateType::Llama2: if (role=="system") ss << "[INST] <<SYS>>\n" << content << "\n<</SYS>>\n\n"; else if (role=="user") ss << content << " [/INST] "; else ss << content << " </s><s>[INST] "; break;
            default: ss << role << ": " << content << "\n";
        }
    }
    if (add_generation_prompt) {
        switch(type) {
            case TemplateType::ChatML: ss << "<|im_start|>assistant\n"; break;
            case TemplateType::Llama3: ss << "<|start_header_id|>assistant<|end_header_id|>\n\n"; break;
            case TemplateType::Phi3: ss << "<|assistant|>\n"; break;
            case TemplateType::Gemma: ss << "<start_of_turn>model\n"; break;
            default: ss << "Assistant:";
        }
    }
    tl_decoded_buffer = ss.str();
    return tl_decoded_buffer.c_str();
}

std::unique_ptr<MlxOgaTokenizerStream> MlxOgaTokenizerStream::Create(const MlxOgaTokenizer& tokenizer) {
    auto stream = std::make_unique<MlxOgaTokenizerStream>();
    stream->tokenizer = &tokenizer;
    return stream;
}

oga_char_ptr MlxOgaTokenizerStream::Decode(int32_t token) {
    std::string delta = const_cast<MlxOgaTokenizer*>(tokenizer)->Decode(&token, 1);
    accumulated += delta;
    decoded_buffer = delta;
    return decoded_buffer.c_str();
}