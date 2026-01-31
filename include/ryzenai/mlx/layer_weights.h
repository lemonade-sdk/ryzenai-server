/*
 * layer_weights.h
 * 
 * Generic layer weight structures for efficient inference.
 * Provides direct weight references to eliminate hash map lookups in hot paths.
 * Only compiled when USE_MLX is defined (macOS with Apple Silicon).
 * 
 * Used by all model inference engines (Qwen3, LLaMA, Phi, Gemma, etc.)
 */

#pragma once

#ifdef USE_MLX

#include <vector>
#include <mlx/array.h>

namespace ryzenai::mlx {

using ::mlx::core::array;

/*
 * LinearWeights - Direct references for a single linear layer
 * Supports both quantized and non-quantized weights
 */
struct LinearWeights {
    const array* weight = nullptr;
    const array* weight_T = nullptr; // Pre-transposed weight (for non-quantized, avoids per-token transpose)
    const array* scales = nullptr;   // For quantized models
    const array* biases = nullptr;   // For quantized models (or regular bias)
    
    // Quantization config (set from model config)
    int group_size = 64;
    int bits = 4;
    
    bool is_quantized() const { return scales != nullptr; }
    bool has_bias() const { return biases != nullptr; }
    bool has_pretransposed() const { return weight_T != nullptr; }
};


/*
 * AttentionWeights - Direct references for attention layer
 * Common structure for Q/K/V/O projections + norms
 */
struct AttentionWeights {
    // Fused QKV projection (Phi3 style - single matmul for all projections)
    LinearWeights qkv_proj;
    
    // Separate Q/K/V projections (Qwen3/LLaMA style)
    LinearWeights q_proj;
    LinearWeights k_proj;
    LinearWeights v_proj;
    LinearWeights o_proj;
    
    // Q/K normalization (Qwen2.5/Qwen3 specific, optional for others)
    const array* q_norm = nullptr;
    const array* k_norm = nullptr;
    
    // Helper to check if fused QKV is available
    bool has_fused_qkv() const { return qkv_proj.weight != nullptr; }
};


/*
 * MLPWeights - Direct references for MLP/FFN layer
 * Supports SwiGLU (gate + up + down) and standard (up + down) configurations
 */
struct MLPWeights {
    LinearWeights gate_proj;  // For SwiGLU (NULL for standard MLP)
    LinearWeights up_proj;
    LinearWeights down_proj;
    
    bool has_gate() const { return gate_proj.weight != nullptr; }
};


/*
 * LayerWeights - Complete weight references for a transformer layer
 * Combines attention, MLP, and normalization weights
 */
struct LayerWeights {
    // Pre-attention normalization (input_layernorm / ln_1)
    const array* input_layernorm = nullptr;
    
    // Attention weights
    AttentionWeights attention;
    
    // Post-attention normalization (post_attention_layernorm / ln_2)
    const array* post_attn_layernorm = nullptr;
    
    // MLP/FFN weights
    MLPWeights mlp;
};


/*
 * ModelWeights - Direct references to all model weights
 * Combines embeddings, layers, and final normalization
 */
struct ModelWeights {
    // Embedding weights
    const array* embed_tokens = nullptr;
    const array* embed_tokens_transposed = nullptr;  // Pre-transposed for tie_word_embeddings
    
    // Per-layer weights
    std::vector<LayerWeights> layers;
    
    // Final normalization
    const array* final_norm = nullptr;
    
    // LM head (NULL if tie_word_embeddings)
    LinearWeights lm_head;
    
    // Pre-computed constants
    float attention_scale = 0.0f;  // 1/sqrt(head_dim)
    float rope_theta = 10000.0f;
};

} // namespace ryzenai::mlx

#endif // USE_MLX
