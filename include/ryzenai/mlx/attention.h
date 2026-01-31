/*
 * attention.h
 * 
 * Attention helper functions for MLX inference.
 * Provides a stable interface for scaled dot-product attention
 * that matches the Python mlx-lm implementation.
 */

#pragma once

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <optional>
#include <string>

namespace ryzenai {
namespace mlx {

namespace traits {
    // Helper to detect if type T has member 'scales'
    template <typename T, typename = void>
    struct has_scales : std::false_type {};

    template <typename T>
    struct has_scales<T, std::void_t<decltype(T::scales)>> : std::true_type {};

    // Helper to detect if type T has member 'biases'
    template <typename T, typename = void>
    struct has_biases : std::false_type {};

    template <typename T>
    struct has_biases<T, std::void_t<decltype(T::biases)>> : std::true_type {};

    // Helper to detect if type T has member 'weight_T'
    template <typename T, typename = void>
    struct has_weight_T : std::false_type {};

    template <typename T>
    struct has_weight_T<T, std::void_t<decltype(T::weight_T)>> : std::true_type {};
    
    // Helper to detect group_size/bits
    template <typename T, typename = void>
    struct has_group_size : std::false_type {};
    template <typename T>
    struct has_group_size<T, std::void_t<decltype(T::group_size)>> : std::true_type {};
    
    template <typename T, typename = void>
    struct has_bits : std::false_type {};
    template <typename T>
    struct has_bits<T, std::void_t<decltype(T::bits)>> : std::true_type {};
}

/**
 * Attention utilities
 */
class Attention {
public:
    /**
     * Scaled dot-product attention.
     * 
     * Matches mlx-lm's scaled_dot_product_attention function.
     * Uses efficient "causal" mask mode when no explicit mask is needed.
     * 
     * @param queries  Query tensor [B, n_heads, L, head_dim]
     * @param keys     Key tensor [B, n_kv_heads, S, head_dim]  
     * @param values   Value tensor [B, n_kv_heads, S, head_dim]
     * @param scale    Attention scale factor (typically 1/sqrt(head_dim))
     * @param mask     Optional attention mask tensor
     * @return         Output tensor [B, n_heads, L, head_dim]
     */
    static ::mlx::core::array scaled_dot_product_attention(
        const ::mlx::core::array& queries,
        const ::mlx::core::array& keys,
        const ::mlx::core::array& values,
        float scale,
        const std::optional<::mlx::core::array>& mask = std::nullopt
    ) {
        // If we have an explicit mask, pass it
        // Otherwise use empty mask_mode for no masking
        return ::mlx::core::fast::scaled_dot_product_attention(
            queries,
            keys, 
            values,
            scale,
            "",           // mask_mode
            mask,         // mask_arr
            std::nullopt  // sinks
        );
    }
    
    /**
     * Scaled dot-product attention with mask mode string.
     * 
     * Supports "causal" mask mode for efficient causal masking.
     * This is more efficient than passing an explicit mask array.
     * 
     * @param queries    Query tensor [B, n_heads, L, head_dim]
     * @param keys       Key tensor [B, n_kv_heads, S, head_dim]  
     * @param values     Value tensor [B, n_kv_heads, S, head_dim]
     * @param scale      Attention scale factor
     * @param mask_mode  "causal" for causal masking, "" for no mask
     * @return           Output tensor [B, n_heads, L, head_dim]
     */
    static ::mlx::core::array scaled_dot_product_attention_causal(
        const ::mlx::core::array& queries,
        const ::mlx::core::array& keys,
        const ::mlx::core::array& values,
        float scale,
        const std::string& mask_mode = "causal"
    ) {
        return ::mlx::core::fast::scaled_dot_product_attention(
            queries,
            keys, 
            values,
            scale,
            mask_mode,    // "causal" for causal masking
            std::nullopt, // no explicit mask array
            std::nullopt  // sinks
        );
    }

    /**
     * Generic Linear Forward.
     * Handles both Quantized and Standard weights via Duck Typing.
     * Expects 'w' to have: .weight, .scales (optional), .biases (optional), .group_size/bits (if quantized)
     */
    template<typename LinearWeightsT>
    static ::mlx::core::array linear_forward(const ::mlx::core::array& x, const LinearWeightsT& w) {
        using namespace ::mlx::core;
        // Detect quantization by checking if 'scales' member exists and is not null
        bool is_quant = false;
        if constexpr (traits::has_scales<LinearWeightsT>::value) {
            if (w.scales != nullptr) is_quant = true;
        }

        if (is_quant) {
            // Quantized Path
             int group_size = 64; 
             int bits = 4;
             
             if constexpr (traits::has_group_size<LinearWeightsT>::value) group_size = w.group_size;
             if constexpr (traits::has_bits<LinearWeightsT>::value) bits = w.bits;
             
             std::optional<array> bias_opt = std::nullopt;
             if constexpr (traits::has_biases<LinearWeightsT>::value) { 
                 if(w.biases != nullptr) bias_opt = *w.biases; 
             }

             return quantized_matmul(
                x, *w.weight, *w.scales, bias_opt,
                true, group_size, bits 
             );
        } else {
            // Standard Path
            array W = *w.weight;
            
            // Check for pre-transposed weight
            if constexpr (traits::has_weight_T<LinearWeightsT>::value) {
                if (w.weight_T != nullptr) W = *w.weight_T;
                else if (W.shape(-1) == x.shape(-1)) W = transpose(W, {1, 0});
            } else {
                 if (W.shape(-1) == x.shape(-1)) W = transpose(W, {1, 0});
            }
            
            array out = matmul(x, W);
            
            if constexpr (traits::has_biases<LinearWeightsT>::value) {
                if (w.biases != nullptr) out = out + *w.biases;
            }
            return out;
        }
    }

    /**
     * Full Self-Attention Implementation.
     * 
     * * Handles:
     * 1. QKV Projection (Fused)
     * 2. Q/K Normalization (Specific to Qwen/Cohere)
     * 3. RoPE (Rotary Positional Embeddings)
     * 4. KV Cache Management
     * 5. Scaled Dot Product Attention
     * 6. Output Projection
     * * @tparam LayerT Type of the layer weights struct (Qwen3MoELayerWeights)
     */
    template<typename LayerT>
    static ::mlx::core::array self_attention_fast_impl(
        const ::mlx::core::array& x,
        const LayerT& layer,
        int layer_idx,
        const std::string& mask_type,
        std::vector<::mlx::core::array>& k_cache,
        std::vector<::mlx::core::array>& v_cache,
        int cache_pos,
        int max_cache_len,
        int head_dim,
        float rope_theta,
        float scale,
        int num_heads,
        int num_kv_heads
    ) {
        using namespace ::mlx::core;
        int B = x.shape(0);
        int L = x.shape(1);
        int q_size = num_heads * head_dim;
        int kv_size = num_kv_heads * head_dim;

        // 1. Fused QKV Projection
        // Uses the helper to handle potential quantization automatically
        ::mlx::core::array qkv = linear_forward(x, layer.attention.qkv_proj);

        // 2. Split Q, K, V
        array queries = reshape(slice(qkv, {0, 0, 0}, {B, L, q_size}), 
                               {B, L, num_heads, head_dim});
        array keys = reshape(slice(qkv, {0, 0, q_size}, {B, L, q_size + kv_size}), 
                            {B, L, num_kv_heads, head_dim});
        array values = reshape(slice(qkv, {0, 0, q_size + kv_size}, {B, L, q_size + 2 * kv_size}), 
                              {B, L, num_kv_heads, head_dim});

        // 3. Q/K Normalization (Qwen-specific)
        // Checks if q_norm/k_norm pointers exist and applies them
        if (layer.attention.q_norm) {
            queries = fast::rms_norm(queries, *layer.attention.q_norm, 1e-6f);
        }
        if (layer.attention.k_norm) {
            keys = fast::rms_norm(keys, *layer.attention.k_norm, 1e-6f);
        }

        // 4. RoPE
        // Prepare for RoPE (needs [B, H, L, D])
        array q_roped = transpose(queries, {0, 2, 1, 3});
        array k_roped = transpose(keys, {0, 2, 1, 3});
        
        q_roped = fast::rope(q_roped, head_dim, false, rope_theta, 1.0f, cache_pos);
        k_roped = fast::rope(k_roped, head_dim, false, rope_theta, 1.0f, cache_pos);

        // 5. KV Cache Update
        // Concatenate new keys/values with cached ones
        array full_k = k_roped;
        array full_v = transpose(values, {0, 2, 1, 3}); // [B, H_kv, L, D]

        if (cache_pos > 0) {
            // Retrieve existing cache
            array past_k = k_cache[layer_idx];
            array past_v = v_cache[layer_idx];
            
            // Slice valid part if needed (simplification: assume cache is pre-filled or we append)
            // Ideally, we write into the cache buffer.
            // For MLX, we often just concat or update slice.
            
            // Simple append logic for inference
            if (cache_pos + L <= max_cache_len) {
                 // Write update logic here or just concat if we treat cache as growing array
                 // For efficiency in MLX, we usually concat with slice of past
                 // Here we assume k_cache grows or we take the relevant slice:
                 array k_past_slice = slice(past_k, {0,0,0,0}, {past_k.shape(0), past_k.shape(1), cache_pos, past_k.shape(3)});
                 array v_past_slice = slice(past_v, {0,0,0,0}, {past_v.shape(0), past_v.shape(1), cache_pos, past_v.shape(3)});
                 full_k = concatenate({k_past_slice, full_k}, 2);
                 full_v = concatenate({v_past_slice, full_v}, 2);
            }
        }
        
        // Update generic cache storage (copy)
        // In a real optimized engine, we'd write to a pre-allocated buffer in-place
        // but for functional correctness here:
        k_cache[layer_idx] = full_k; 
        v_cache[layer_idx] = full_v;

        // 6. Attention (SDPA)
        array output = 
        (
            mask_type == "none" ?
            fast::scaled_dot_product_attention(q_roped, full_k, full_v, scale) : 
            (
                mask_type == "causal" ?
                fast::scaled_dot_product_attention(q_roped, full_k, full_v, scale, "causal") :
                fast::scaled_dot_product_attention(q_roped, full_k, full_v, scale, "causal")
            )
        );

        // 7. Output Projection
        output = transpose(output, {0, 2, 1, 3});
        output = reshape(output, {B, L, q_size});
        
        return linear_forward(output, layer.attention.o_proj);
    }
    
    /**
     * Create attention mask - matches Python create_attention_mask.
     * 
     * Returns:
     *   - "none" if seq_len == 1 (no mask needed for single token)
     *   - "causal" if seq_len > 1 and no window (use efficient causal)
     *   - "array" if window_size specified (need explicit mask)
     * 
     * @param seq_len      Current sequence length
     * @param window_size  Optional sliding window size
     * @return             Mask type string
     */
    static std::string get_mask_type(int seq_len, int window_size = 0) {
        if (seq_len == 1) {
            return "none";  // No mask needed for single token generation
        }
        if (window_size > 0 && seq_len > window_size) {
            return "array"; // Need explicit mask for windowed attention
        }
        return "causal";    // Use efficient causal mask
    }
    
    /**
     * Create a causal attention mask array.
     * 
     * Creates a mask where future positions are masked out (set to -inf).
     * Only needed when get_mask_type returns "array".
     * 
     * Python reference:
     *   def create_causal_mask(N, offset=0, window_size=None):
     *       rinds = mx.arange(offset + N)
     *       linds = mx.arange(offset, offset + N) if offset else rinds
     *       linds = linds[:, None]
     *       rinds = rinds[None]
     *       mask = linds >= rinds
     *       if window_size is not None:
     *           mask = mask & (linds < rinds + window_size)
     *       return mask
     * 
     * @param seq_len      Current sequence length (N)
     * @param offset       Position offset (cached tokens)
     * @param window_size  Optional sliding window size
     * @return             Mask tensor [seq_len, offset+seq_len] as boolean
     */
    static ::mlx::core::array create_causal_mask(
        int seq_len, 
        int offset = 0, 
        int window_size = 0
    ) {
        using namespace ::mlx::core;
        
        int total_len = offset + seq_len;
        
        // rinds = mx.arange(offset + N)
        array rinds = arange(total_len, int32);
        
        // linds = mx.arange(offset, offset + N) if offset else rinds
        array linds = offset > 0 ? 
                        arange(offset, offset + seq_len, int32) : 
                        arange(seq_len, int32);
        
        // linds = linds[:, None], rinds = rinds[None]
        linds = reshape(linds, {seq_len, 1});
        rinds = reshape(rinds, {1, total_len});
        
        // mask = linds >= rinds
        array mask = greater_equal(linds, rinds);
        
        // Optional windowed attention
        if (window_size > 0) {
            // mask = mask & (linds < rinds + window_size)
            array window_mask = less(linds, rinds + array(window_size));
            mask = logical_and(mask, window_mask);
        }
        
        return mask;  // Boolean mask [seq_len, total_len]
    }
    
    /**
     * Convert boolean mask to float attention mask.
     * 
     * @param bool_mask  Boolean mask where true = attend
     * @return           Float mask where 0 = attend, -inf = masked
     */
    static ::mlx::core::array mask_to_float(const ::mlx::core::array& bool_mask) {
        using namespace ::mlx::core;
        array zero = array(0.0f);
        array neg_inf = array(-1e9f);
        return where(bool_mask, zero, neg_inf);
    }
    
    /**
     * Prepare mask for SDPA - expands dims to [1, 1, seq_len, total_len].
     * 
     * @param mask  2D mask [seq_len, total_len]
     * @return      4D mask [1, 1, seq_len, total_len]
     */
    static ::mlx::core::array expand_mask_for_sdpa(const ::mlx::core::array& mask) {
        using namespace ::mlx::core;
        return expand_dims(expand_dims(mask, 0), 0);
    }
};

} // namespace mlx
} // namespace ryzenai
