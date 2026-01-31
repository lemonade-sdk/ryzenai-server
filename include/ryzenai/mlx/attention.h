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
