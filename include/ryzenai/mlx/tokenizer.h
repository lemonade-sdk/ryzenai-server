/*
 * tokenizer.h
 * 
 * Text tokenization and detokenization for MLX models.
 * Supports SentencePiece and HuggingFace tokenizer formats.
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include <sentencepiece_processor.h>
#include <unordered_map>
#include <memory>
#include <vector>

typedef const char* oga_char_ptr;

struct OgaSequences;


struct OgaTokenizer {
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> reverse_vocab;
    std::string chat_template;
    bool use_sentencepiece = false;
    bool use_hf_tokenizer = false;

    /*
     * Create
     * Factory function that loads tokenizer from model directory.
     * Attempts SentencePiece first, then HuggingFace JSON format.
     */
    static std::unique_ptr<OgaTokenizer> Create(const OgaModel& model);

    /*
     * Encode
     * Converts input text to token IDs.
     */
    void Encode(const char* text, OgaSequences& sequences);

    /*
     * Decode
     * Converts token IDs back to text.
     */
    oga_char_ptr Decode(const int32_t* tokens, size_t count);

    /*
     * ApplyChatTemplate
     * Formats messages using model-specific chat template.
     * Supports Phi-3, ChatML, Llama, Gemma, and other formats.
     */
    oga_char_ptr ApplyChatTemplate(const char* template_str, const char* messages_json, 
                                    const char* tools_json, bool add_generation_prompt);
};


struct OgaSequences {
    std::vector<int32_t> ids;

    static std::unique_ptr<OgaSequences> Create();
    const int32_t* SequenceData(int index) const;
    size_t SequenceCount(int index) const;
};


struct OgaTokenizerStream {
    const OgaTokenizer* tokenizer;
    std::vector<int32_t> pending_tokens;
    std::string accumulated;

    /*
     * Create
     * Creates a streaming decoder bound to a tokenizer.
     */
    static std::unique_ptr<OgaTokenizerStream> Create(const OgaTokenizer& tokenizer);

    /*
     * Decode
     * Decodes a single token and returns the delta string.
     */
    oga_char_ptr Decode(int32_t token);
};
