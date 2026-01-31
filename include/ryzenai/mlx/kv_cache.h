/*
 * kv_cache.h
 * 
 * Shared KV cache utility for transformer inference engines.
 * Pre-allocates fixed-size buffers to avoid per-token memory allocation.
 * 
 * Used by: Qwen3, Phi3, LLaMA, Mixtral, etc.
 */

#pragma once

#include <vector>
#include <utility>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace ryzenai::mlx {

using ::mlx::core::array;
using ::mlx::core::Dtype;
using ::mlx::core::float16;
using ::mlx::core::zeros;
using ::mlx::core::slice;
using ::mlx::core::concatenate;
using ::mlx::core::eval;

/*
 * KVCache
 * 
 * Manages key-value cache for efficient autoregressive transformer inference.
 * Pre-allocates fixed-size buffers to minimize memory allocation during generation.
 * 
 * Features:
 *   - Pre-allocated buffers to reduce memory fragmentation
 *   - Position tracking for incremental updates
 *   - Sliding window support for long sequences
 *   - Clear without deallocation for conversation resets
 */
class KVCache {
public:
    /*
     * Constructor
     * Initializes empty cache (must call initialize() before use).
     */
    KVCache() : initialized_(false), position_(0), max_length_(0), num_layers_(0) {}
    
    /*
     * initialize
     * Pre-allocates KV cache buffers for all layers.
     * 
     * @param num_layers Number of transformer layers
     * @param num_kv_heads Number of key-value attention heads
     * @param max_length Maximum sequence length to cache
     * @param head_dim Dimension per attention head
     * @param dtype Data type for cache tensors
     */
    void initialize(int num_layers, int num_kv_heads, int max_length, int head_dim, Dtype dtype = float16) {
        num_layers_ = num_layers;
        max_length_ = max_length;
        position_ = 0;
        
        k_cache_.clear();
        v_cache_.clear();
        k_cache_.reserve(num_layers);
        v_cache_.reserve(num_layers);
        
        for (int i = 0; i < num_layers; ++i) {
            k_cache_.push_back(zeros({1, num_kv_heads, max_length, head_dim}, dtype));
            v_cache_.push_back(zeros({1, num_kv_heads, max_length, head_dim}, dtype));
        }
        
        eval(k_cache_);
        eval(v_cache_);
        initialized_ = true;
    }
    
    /*
     * clear
     * Resets position counter without deallocating buffers.
     * Use this between conversations to avoid memory allocation.
     */
    void clear() {
        position_ = 0;
    }
    
    /*
     * update
     * Updates the cache for a given layer and returns the full K/V tensors.
     * Handles initial fill, normal append, and sliding window overflow.
     * 
     * @param layer_idx Layer index (0 to num_layers-1)
     * @param new_k New key tensor [B, num_kv_heads, seq_len, head_dim]
     * @param new_v New value tensor [B, num_kv_heads, seq_len, head_dim]
     * @return Pair of (full_k, full_v) including cached history
     */
    std::pair<array, array> update(int layer_idx, const array& new_k, const array& new_v) {
        int seq_len = static_cast<int>(new_k.shape(2));
        int new_pos = position_ + seq_len;
        
        if (position_ == 0) {
            // First tokens - just use the new K/V directly
            k_cache_[layer_idx] = new_k;
            v_cache_[layer_idx] = new_v;
            return {new_k, new_v};
        }
        
        if (new_pos <= max_length_) {
            // Within cache limit - concatenate with existing
            int b = static_cast<int>(k_cache_[layer_idx].shape(0));
            int h = static_cast<int>(k_cache_[layer_idx].shape(1));
            int d = static_cast<int>(k_cache_[layer_idx].shape(3));
            
            array cached_k = slice(k_cache_[layer_idx], {0, 0, 0, 0}, {b, h, position_, d});
            array cached_v = slice(v_cache_[layer_idx], {0, 0, 0, 0}, {b, h, position_, d});
            
            array full_k = concatenate({cached_k, new_k}, 2);
            array full_v = concatenate({cached_v, new_v}, 2);
            
            k_cache_[layer_idx] = full_k;
            v_cache_[layer_idx] = full_v;
            return {full_k, full_v};
        }
        
        // Cache overflow - use sliding window (keep most recent tokens)
        int keep_from = position_ - (max_length_ - seq_len);
        if (keep_from < 0) keep_from = 0;
        int keep_len = position_ - keep_from;
        
        if (keep_len > 0) {
            int b = static_cast<int>(k_cache_[layer_idx].shape(0));
            int h = static_cast<int>(k_cache_[layer_idx].shape(1));
            int d = static_cast<int>(k_cache_[layer_idx].shape(3));
            
            array cached_k = slice(k_cache_[layer_idx], {0, 0, keep_from, 0}, {b, h, position_, d});
            array cached_v = slice(v_cache_[layer_idx], {0, 0, keep_from, 0}, {b, h, position_, d});
            
            array full_k = concatenate({cached_k, new_k}, 2);
            array full_v = concatenate({cached_v, new_v}, 2);
            
            k_cache_[layer_idx] = full_k;
            v_cache_[layer_idx] = full_v;
            return {full_k, full_v};
        }
        
        k_cache_[layer_idx] = new_k;
        v_cache_[layer_idx] = new_v;
        return {new_k, new_v};
    }
    
    /*
     * advance_position
     * Updates the position counter after processing all layers.
     * Call this once per forward pass after all layers have been updated.
     * 
     * @param seq_len Number of new tokens processed
     */
    void advance_position(int seq_len) {
        position_ += seq_len;
        if (position_ > max_length_) {
            position_ = max_length_;
        }
    }
    
    // Accessors
    bool is_initialized() const { return initialized_; }
    int position() const { return position_; }
    int max_length() const { return max_length_; }
    int num_layers() const { return num_layers_; }

private:
    bool initialized_;
    int position_;        // Current position in cache (number of tokens stored)
    int max_length_;      // Maximum cache length
    int num_layers_;      // Number of layers
    
    std::vector<array> k_cache_;
    std::vector<array> v_cache_;
};

} // namespace ryzenai::mlx
