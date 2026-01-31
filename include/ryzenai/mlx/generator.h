/*
 * generator.h
 * 
 * Token generation orchestration for MLX models.
 * Manages the autoregressive generation loop and sampling parameters.
 */

#pragma once

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/tokenizer.h"
#include <memory>
#include <vector>


/*
 * OgaGeneratorParams
 * 
 * Configuration for text generation including sampling parameters.
 * Compatible with ONNX GenAI parameter interface.
 */
struct OgaGeneratorParams {
    int max_length = 512;
    float temperature = 0.7f;
    float top_p = 0.95f;
    int top_k = 50;
    float repetition_penalty = 1.0f;
    bool do_sample = true;
    double random_seed = 0.0;
    int eos_token_id = 2;  // Default EOS token ID
    int pad_token_id = 2;  // Default pad token ID

    static std::unique_ptr<OgaGeneratorParams> Create(const OgaModel& model);

    void SetSearchOption(const std::string& key, int value);
    void SetSearchOption(const std::string& key, double value);
    void SetSearchOptionBool(const std::string& key, bool value);
};


/*
 * OgaGenerator
 * 
 * Manages autoregressive token generation.
 * Holds model reference, current token sequence, and inference engine.
 */
struct OgaGenerator {
    const OgaModel* model = nullptr;
    const OgaTokenizer* tokenizer = nullptr;
    OgaGeneratorParams params;
    std::vector<int32_t> current_tokens;
    bool done = false;
    std::unique_ptr<BaseInferenceEngine> inference_engine;

    // For stop sequence detection
    size_t input_token_count = 0;
    std::string accumulated_text;
    std::vector<std::string> stop_sequences;

    /*
     * Create
     * Factory function that initializes generator with model and parameters.
     * Creates the appropriate inference engine based on model architecture.
     */
    static std::unique_ptr<OgaGenerator> Create(const OgaModel& model, const OgaGeneratorParams& p);

    /*
     * SetTokenizer
     * Associates a tokenizer for debug token decoding.
     */
    void SetTokenizer(const OgaTokenizer& tokenizer);

    /*
     * AppendTokens
     * Adds tokens to the current sequence (for prompt input).
     */
    void AppendTokens(const int32_t* tokens, size_t count);

    /*
     * GenerateNextToken
     * Runs one step of generation: forward pass + sampling.
     * Updates current_tokens with the new token.
     */
    void GenerateNextToken();

    /*
     * IsDone
     * Returns true if generation has completed (EOS or max length).
     */
    bool IsDone() const;

    /*
     * GetSequenceData
     * Returns pointer to the current token sequence.
     */
    const int32_t* GetSequenceData(int index) const;

    /*
     * GetSequenceCount
     * Returns the number of tokens in the current sequence.
     */
    size_t GetSequenceCount(int index) const;

private:
    int sample_token(const array& logits);
};
