/*
 * common.h
 * 
 * Common types and base classes for MLX inference engines.
 * Defines the abstract interface for model forward passes.
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <mlx/mlx.h>

using namespace mlx::core;

struct OgaModel;
struct OgaGeneratorParams;
struct OgaTokenizer;


/*
 * BaseInferenceEngine
 * 
 * Abstract base class for all model-specific inference implementations.
 * Subclasses implement the forward pass and token sampling logic
 * for different model architectures (Phi-3, Gemma, etc.).
 */
class BaseInferenceEngine {
public:
    virtual ~BaseInferenceEngine() = default;

    /*
     * forward
     * Runs a forward pass through the model given input tokens.
     * Returns logits for vocabulary prediction.
     */
    virtual array forward(const std::vector<int32_t>& input_tokens,
                          const OgaGeneratorParams& params) = 0;

    /*
     * sample_token
     * Samples the next token from output logits.
     * Implements sampling strategy (greedy, temperature, etc.).
     */
    virtual int sample_token(const array& logits, const OgaGeneratorParams& params) = 0;
};


/*
 * create_inference_engine
 * 
 * Factory function that creates the appropriate inference engine
 * based on the model's architecture type.
 */
std::unique_ptr<BaseInferenceEngine> create_inference_engine(const OgaModel& model);
