/*
 * tokenizer.cpp
 * 
 * Tokenizer implementation supporting SentencePiece and HuggingFace formats.
 * Provides text encoding, decoding, and chat template processing.
 */

#include "ryzenai/mlx/tokenizer.h"
#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

thread_local std::string decoded_buffer;


std::unique_ptr<OgaSequences> OgaSequences::Create() {
    return std::make_unique<OgaSequences>();
}


const int32_t* OgaSequences::SequenceData(int) const {
    return ids.data();
}


size_t OgaSequences::SequenceCount(int) const {
    return ids.size();
}


/*
 * OgaTokenizer::Create
 * 
 * Factory function that initializes a tokenizer from model files.
 * Priority: SentencePiece (.model) > HuggingFace (.json) > fallback
 */
std::unique_ptr<OgaTokenizer> OgaTokenizer::Create(const OgaModel& model) {
    auto tok = std::make_unique<OgaTokenizer>();

    std::string sp_model_path = model.model_path + "/tokenizer.model";
    if (fs::exists(sp_model_path)) {
        try {
            tok->sp_processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
            if (tok->sp_processor->Load(sp_model_path).ok()) {
                tok->use_sentencepiece = true;
                std::cout << "[Tokenizer] Loaded SentencePiece: " << sp_model_path << std::endl;
            } else {
                std::cerr << "[Tokenizer] SentencePiece load failed, using fallback" << std::endl;
                tok->sp_processor.reset();
                tok->use_sentencepiece = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Tokenizer] SentencePiece init error: " << e.what() << std::endl;
            tok->sp_processor.reset();
            tok->use_sentencepiece = false;
        }
    } else {
        std::cout << "[Tokenizer] No tokenizer.model, checking for HuggingFace format" << std::endl;
    }

    std::string hf_tokenizer_path = model.model_path + "/tokenizer.json";
    if (!tok->use_sentencepiece && fs::exists(hf_tokenizer_path)) {
        try {
            std::ifstream f(hf_tokenizer_path);
            nlohmann::json tokenizer_config;
            f >> tokenizer_config;

            if (tokenizer_config.contains("model") && tokenizer_config["model"].contains("vocab")) {
                auto& vocab_json = tokenizer_config["model"]["vocab"];
                for (auto& [token, id] : vocab_json.items()) {
                    int32_t token_id = id.get<int32_t>();
                    tok->vocab[token] = token_id;
                    tok->reverse_vocab[token_id] = token;
                }
                tok->use_hf_tokenizer = true;
                std::cout << "[Tokenizer] Loaded HuggingFace vocab (" 
                          << tok->vocab.size() << " tokens)" << std::endl;
            } else {
                std::cerr << "[Tokenizer] HuggingFace tokenizer missing vocab section" << std::endl;
                tok->use_hf_tokenizer = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Tokenizer] HuggingFace parse error: " << e.what() << std::endl;
            tok->use_hf_tokenizer = false;
        }
    }

    if (!tok->use_sentencepiece && !tok->use_hf_tokenizer) {
        tok->vocab["<unk>"] = 0;
        tok->vocab["<s>"] = 1;
        tok->vocab["</s>"] = 2;
        tok->reverse_vocab[0] = "<unk>";
        tok->reverse_vocab[1] = "<s>";
        tok->reverse_vocab[2] = "</s>";
        
        int id = 3;
        for (char c = 'a'; c <= 'z'; ++c) {
            std::string token(1, c);
            tok->vocab[token] = id;
            tok->reverse_vocab[id] = token;
            id++;
        }
        tok->vocab[" "] = id;
        tok->reverse_vocab[id] = " ";
        std::cout << "[Tokenizer] Using basic fallback tokenizer" << std::endl;
    }

    std::string config_path = model.model_path + "/tokenizer_config.json";
    if (fs::exists(config_path)) {
        std::ifstream f(config_path);
        nlohmann::json config;
        try {
            f >> config;
            if (config.contains("chat_template") && config["chat_template"].is_string()) {
                tok->chat_template = config["chat_template"].get<std::string>();
            }
        } catch (const std::exception& e) {
            std::cerr << "[Tokenizer] Config parse error: " << e.what() << std::endl;
        }
    }

    return tok;
}


/*
 * OgaTokenizer::Encode
 * 
 * Converts input text to a sequence of token IDs.
 * Uses SentencePiece if available, otherwise HuggingFace BPE-style encoding.
 */
void OgaTokenizer::Encode(const char* text, OgaSequences& sequences) {
    std::vector<int32_t> ids;

    if (use_sentencepiece && sp_processor) {
        std::vector<int> sp_ids;
        if (sp_processor->Encode(text, &sp_ids).ok()) {
            ids.assign(sp_ids.begin(), sp_ids.end());
            sequences.ids = std::move(ids);
            return;
        } else {
            std::cerr << "[Tokenizer] SentencePiece encode failed" << std::endl;
            use_sentencepiece = false;
        }
    }

    if (use_hf_tokenizer) {
        std::string input(text);
        size_t pos = 0;
        
        std::vector<std::pair<std::string, int32_t>> special_tokens;
        for (const auto& [token, id] : vocab) {
            if (token.size() > 2 && token[0] == '<' && token.back() == '>') {
                special_tokens.emplace_back(token, id);
            }
        }
        std::sort(special_tokens.begin(), special_tokens.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        
        bool word_start = true;
        
        while (pos < input.size()) {
            bool found_special = false;
            
            for (const auto& [token, id] : special_tokens) {
                if (input.compare(pos, token.size(), token) == 0) {
                    ids.push_back(id);
                    pos += token.size();
                    found_special = true;
                    word_start = true;
                    break;
                }
            }
            if (found_special) continue;
            
            if (input[pos] == ' ') {
                word_start = true;
                pos++;
                continue;
            }
            
            if (input[pos] == '\n') {
                auto nl_it = vocab.find("<0x0A>");
                if (nl_it != vocab.end()) {
                    ids.push_back(nl_it->second);
                } else {
                    nl_it = vocab.find("\n");
                    if (nl_it != vocab.end()) {
                        ids.push_back(nl_it->second);
                    }
                }
                word_start = true;
                pos++;
                continue;
            }
            
            size_t best_len = 0;
            int32_t best_id = 0;
            
            for (size_t len = std::min(size_t(15), input.size() - pos); len > 0; --len) {
                std::string candidate = input.substr(pos, len);
                
                if (word_start) {
                    std::string with_underscore = "▁" + candidate;
                    auto it = vocab.find(with_underscore);
                    if (it != vocab.end()) {
                        best_len = len;
                        best_id = it->second;
                        break;
                    }
                }
                
                auto it = vocab.find(candidate);
                if (it != vocab.end()) {
                    best_len = len;
                    best_id = it->second;
                    break;
                }
            }
            
            if (best_len > 0) {
                ids.push_back(best_id);
                pos += best_len;
                word_start = false;
            } else {
                std::string ch(1, input[pos]);
                
                if (word_start) {
                    std::string with_underscore = "▁" + ch;
                    auto it = vocab.find(with_underscore);
                    if (it != vocab.end()) {
                        ids.push_back(it->second);
                        pos++;
                        word_start = false;
                        continue;
                    }
                }
                
                auto it = vocab.find(ch);
                if (it != vocab.end()) {
                    ids.push_back(it->second);
                    word_start = false;
                } else {
                    std::cerr << "[Tokenizer] Unknown char: '" << ch 
                              << "' (0x" << std::hex << (int)(unsigned char)input[pos] 
                              << std::dec << ")" << std::endl;
                }
                pos++;
            }
        }
        
        sequences.ids = std::move(ids);
        return;
    }

    std::string input(text);
    size_t pos = 0;
    while (pos < input.size()) {
        size_t next_space = input.find(' ', pos);
        if (next_space == std::string::npos) {
            next_space = input.size();
        }

        std::string token = input.substr(pos, next_space - pos);
        if (!token.empty()) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(0);
            }
        }

        if (next_space < input.size()) {
            auto sp_it = vocab.find(" ");
            if (sp_it != vocab.end()) {
                ids.push_back(sp_it->second);
            }
        }

        pos = next_space + 1;
    }

    sequences.ids = std::move(ids);
}


/*
 * OgaTokenizer::Decode
 * 
 * Converts token IDs back to text.
 * Returns pointer to thread-local buffer (valid until next call).
 */
oga_char_ptr OgaTokenizer::Decode(const int32_t* tokens, size_t count) {
    std::string detok;

    if (use_sentencepiece && sp_processor) {
        std::vector<int> token_vec(tokens, tokens + count);
        if (sp_processor->Decode(token_vec, &detok).ok()) {
            decoded_buffer = std::move(detok);
            return decoded_buffer.c_str();
        } else {
            std::cerr << "[Tokenizer] SentencePiece decode failed" << std::endl;
            use_sentencepiece = false;
        }
    }

    for (size_t i = 0; i < count; ++i) {
        auto it = reverse_vocab.find(tokens[i]);
        if (it != reverse_vocab.end()) {
            detok += it->second;
        } else {
            detok += "<unk>";
        }
    }

    decoded_buffer = std::move(detok);
    return decoded_buffer.c_str();
}


/*
 * OgaTokenizer::ApplyChatTemplate
 * 
 * Formats conversation messages according to the model's chat template.
 * Auto-detects template format from template string patterns.
 */
oga_char_ptr OgaTokenizer::ApplyChatTemplate(const char* template_str, const char* messages_json, 
                                              const char* tools_json, bool add_generation_prompt) {
    if (tools_json && tools_json[0] != '\0') {
        throw std::runtime_error("Tools not supported in MLX backend");
    }
    
    std::string tmpl = (template_str && template_str[0] != '\0') ? template_str : chat_template;
    nlohmann::json messages = nlohmann::json::parse(messages_json);
    std::ostringstream result;
    
    bool is_phi3 = (tmpl.find("<|user|>") != std::string::npos && 
                    tmpl.find("<|end|>") != std::string::npos);
    bool is_chatml = (tmpl.find("<|im_start|>") != std::string::npos);
    bool is_llama2 = (tmpl.find("[INST]") != std::string::npos);
    bool is_llama3 = (tmpl.find("<|start_header_id|>") != std::string::npos);
    bool is_mistral = (tmpl.find("[INST]") != std::string::npos && 
                       tmpl.find("<<SYS>>") == std::string::npos);
    bool is_gemma = (tmpl.find("<start_of_turn>") != std::string::npos);
    bool is_vicuna = (tmpl.find("### Human:") != std::string::npos ||
                      tmpl.find("### User:") != std::string::npos);
    
    if (is_phi3) {
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            
            if (role == "system") {
                result << "<|system|>\n" << content << "<|end|>\n";
            } else if (role == "user") {
                result << "<|user|>\n" << content << "<|end|>\n";
            } else if (role == "assistant") {
                result << "<|assistant|>\n" << content << "<|end|>\n";
            }
        }
        if (add_generation_prompt) {
            result << "<|assistant|>\n";
        }
    } else if (is_chatml) {
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            result << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        }
        if (add_generation_prompt) {
            result << "<|im_start|>assistant\n";
        }
    } else if (is_llama3) {
        result << "<|begin_of_text|>";
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            result << "<|start_header_id|>" << role << "<|end_header_id|>\n\n" << content << "<|eot_id|>";
        }
        if (add_generation_prompt) {
            result << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    } else if (is_gemma) {
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            if (role == "assistant") role = "model";
            result << "<start_of_turn>" << role << "\n" << content << "<end_of_turn>\n";
        }
        if (add_generation_prompt) {
            result << "<start_of_turn>model\n";
        }
    } else if (is_vicuna) {
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            
            if (role == "system") {
                result << content << "\n\n";
            } else if (role == "user") {
                result << "### Human: " << content << "\n";
            } else if (role == "assistant") {
                result << "### Assistant: " << content << "\n";
            }
        }
        if (add_generation_prompt) {
            result << "### Assistant:";
        }
    } else if (is_llama2 || is_mistral) {
        std::string system_msg;
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            
            if (role == "system") {
                system_msg = content;
            } else if (role == "user") {
                result << "[INST] ";
                if (!system_msg.empty()) {
                    result << "<<SYS>>\n" << system_msg << "\n<</SYS>>\n\n";
                    system_msg.clear();
                }
                result << content << " [/INST]";
            } else if (role == "assistant") {
                result << " " << content << "</s>";
            }
        }
    } else {
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            
            if (role == "system") {
                result << "System: " << content << "\n\n";
            } else if (role == "user") {
                result << "User: " << content << "\n\n";
            } else if (role == "assistant") {
                result << "Assistant: " << content << "\n\n";
            }
        }
        if (add_generation_prompt) {
            result << "Assistant: ";
        }
    }
    
    decoded_buffer = result.str();
    return decoded_buffer.c_str();
}


std::unique_ptr<OgaTokenizerStream> OgaTokenizerStream::Create(const OgaTokenizer& tokenizer) {
    auto stream = std::make_unique<OgaTokenizerStream>();
    stream->tokenizer = &tokenizer;
    stream->pending_tokens.clear();
    stream->accumulated.clear();
    return stream;
}


/*
 * OgaTokenizerStream::Decode
 * 
 * Streaming decoder that returns incremental text for each token.
 */
oga_char_ptr OgaTokenizerStream::Decode(int32_t token) {
    pending_tokens.push_back(token);

    std::string delta;
    auto it = tokenizer->reverse_vocab.find(token);
    if (it != tokenizer->reverse_vocab.end()) {
        delta = it->second;
    } else {
        delta = "<unk>";
    }

    accumulated += delta;
    decoded_buffer = delta;
    return decoded_buffer.c_str();
}
