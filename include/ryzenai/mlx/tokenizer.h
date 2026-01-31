#pragma once
#include <string>
#include <vector>
#include <memory>
#include <ryzenai/mlx/model.h>

// Opaque types for C-interface compatibility if needed
using oga_char_ptr = const char*;

struct OgaSequences {
    std::vector<int32_t> ids;
    static std::unique_ptr<OgaSequences> Create();
    const int32_t* SequenceData(int index) const;
    size_t SequenceCount(int index) const;
};

// Abstract Backend Interface
struct TokenizerBackend {
public:
    virtual ~TokenizerBackend() = default;
    virtual void Encode(const std::string& text, std::vector<int32_t>& ids) = 0;
    virtual std::string Decode(const std::vector<int32_t>& ids) = 0;
};

struct OgaTokenizer {
public:
    static std::unique_ptr<OgaTokenizer> Create(const OgaModel& model);
    
    void Encode(const char* text, OgaSequences& sequences);
    oga_char_ptr Decode(const int32_t* tokens, size_t count);
    
    oga_char_ptr ApplyChatTemplate(const char* template_str, 
                                   const char* messages_json, 
                                   const char* tools_json, 
                                   bool add_generation_prompt);

private:
    std::unique_ptr<TokenizerBackend> backend;
    std::string chat_template_str;
    std::string decoded_buffer; // Thread-local storage emulation
};

class OgaTokenizerStream {
public:
    static std::unique_ptr<OgaTokenizerStream> Create(const OgaTokenizer& tokenizer);
    oga_char_ptr Decode(int32_t token);

private:
    const OgaTokenizer* tokenizer;
    std::vector<int32_t> pending_tokens;
    std::string accumulated;
    std::string decoded_buffer;
};