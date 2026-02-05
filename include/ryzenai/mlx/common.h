/*
 * common.h
 * 
 * Common types and base classes for MLX inference engines.
 * Defines the abstract interface for model forward passes.
 * Only compiled when USE_MLX is defined (macOS with Apple Silicon).
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <mlx/mlx.h>

using namespace mlx::core;

// Forward declarations for MLX OGA-compatible types
// Named MlxOga* to distinguish from ONNX OGA types
struct MlxOgaModel;
struct MlxOgaGeneratorParams;
struct MlxOgaTokenizer;


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
                          const MlxOgaGeneratorParams& params) = 0;
                          
    virtual mlx::core::array forward(const mlx::core::array& tokens, const MlxOgaGeneratorParams& params) {
        throw std::runtime_error("forward(array) not implemented for this model");
    }

    /*
     * sample_token
     * Samples the next token from output logits.
     * Implements sampling strategy (greedy, temperature, etc.).
     */
    virtual int sample_token(const array& logits, const MlxOgaGeneratorParams& params) = 0;
    
    /*
     * supports_kv_cache
     * Returns true if the engine supports incremental decoding with KV cache.
     * If true, GenerateNextToken can pass only the last token during decode.
     * If false, all tokens must be passed on every forward call.
     * Default: false (safe fallback - always send all tokens)
     */
    virtual bool supports_kv_cache() const { return false; }
};


/*
 * create_inference_engine
 * 
 * Factory function that creates the appropriate inference engine
 * based on the model's architecture type.
 */
std::unique_ptr<BaseInferenceEngine> create_inference_engine(const MlxOgaModel& model);
