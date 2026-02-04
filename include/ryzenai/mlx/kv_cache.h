/*
 * kv_cache.h
 * 
 * Shared KV cache utility for transformer inference engines.
 * Pre-allocates fixed-size buffers to avoid per-token memory allocation.
 * Only compiled when USE_MLX is defined (macOS with Apple Silicon).
 * 
 * Supports optional INT8 quantization for reduced memory bandwidth.
 * When quantized, KV values are stored as INT8 with per-head per-token FP32 scales.
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
using ::mlx::core::Shape;
using ::mlx::core::Dtype;
using ::mlx::core::int8;
using ::mlx::core::float32;
using ::mlx::core::float16;
using ::mlx::core::zeros;
using ::mlx::core::slice;
using ::mlx::core::slice_update;
using ::mlx::core::concatenate;
using ::mlx::core::eval;
using ::mlx::core::max;
using ::mlx::core::abs;
using ::mlx::core::round;
using ::mlx::core::astype;
using ::mlx::core::arange;
using ::mlx::core::scatter;
using ::mlx::core::expand_dims;

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
 *   - Optional INT8 quantization with per-head per-token scales
 */
class KVCache {
public:
    std::vector<array> k_cache_;
    std::vector<array> v_cache_;
public:
    /*
     * Constructor
     * 
     * @param quantized If true, use INT8 quantized cache (default: false)
     */
    KVCache(bool quantized = false) 
        : quantized_(quantized), initialized_(false), position_(0), max_length_(0), num_layers_(0), num_kv_heads_(0), head_dim_(0) {}
    
    /*
     * initialize
     * Pre-allocates KV cache buffers for all layers.
     * 
     * @param num_layers Number of transformer layers
     * @param num_kv_heads Number of key-value attention heads
     * @param max_length Maximum sequence length to cache
     * @param head_dim Dimension per attention head
     */
    void initialize(int num_layers, int num_kv_heads, int max_length, int head_dim) {
        num_layers_ = num_layers;
        num_kv_heads_ = num_kv_heads;
        max_length_ = max_length;
        head_dim_ = head_dim;
        position_ = 0;
        
        Dtype cache_dtype = quantized_ ? int8 : float16;
        
        k_cache_.clear();
        v_cache_.clear();
        k_cache_.reserve(num_layers);
        v_cache_.reserve(num_layers);
        
        for (int i = 0; i < num_layers; ++i) {
            k_cache_.push_back(zeros({1, num_kv_heads, 0, head_dim}, cache_dtype));
            v_cache_.push_back(zeros({1, num_kv_heads, 0, head_dim}, cache_dtype));
        }
        
        if (quantized_) {
            k_scales_.clear();
            v_scales_.clear();
            k_scales_.reserve(num_layers);
            v_scales_.reserve(num_layers);
            
            for (int i = 0; i < num_layers; ++i) {
                k_scales_.push_back(zeros({1, num_kv_heads, 0, 1}, float32));
                v_scales_.push_back(zeros({1, num_kv_heads, 0, 1}, float32));
            }
        }
        
        eval(k_cache_);
        eval(v_cache_);
        if (quantized_) {
            eval(k_scales_);
            eval(v_scales_);
        }
        initialized_ = true;
    }
    
    /*
     * clear
     * Resets position counter without deallocating buffers.
     * Use this between conversations to avoid memory allocation.
     */
    void clear() {
        position_ = 0;
        // Reset caches to empty shape without deallocation
        Dtype cache_dtype = quantized_ ? int8 : float16;
        for (int i = 0; i < num_layers_; ++i) {
            k_cache_[i] = zeros({1, num_kv_heads_, 0, head_dim_}, cache_dtype);
            v_cache_[i] = zeros({1, num_kv_heads_, 0, head_dim_}, cache_dtype);
        }
        if (quantized_) {
            for (int i = 0; i < num_layers_; ++i) {
                k_scales_[i] = zeros({1, num_kv_heads_, 0, 1}, float32);
                v_scales_[i] = zeros({1, num_kv_heads_, 0, 1}, float32);
            }
        }
    }
    
    /*
     * update
     * Updates the cache for a given layer and returns the full K/V tensors (dequantized if applicable).
     * Handles initial fill, normal append, and sliding window overflow.
     * 
     * @param layer_idx Layer index (0 to num_layers-1)
     * @param new_k New key tensor [B, num_kv_heads, seq_len, head_dim]
     * @param new_v New value tensor [B, num_kv_heads, seq_len, head_dim]
     * @return Pair of (full_k, full_v) including cached history (always FP16)
     */
    std::pair<array, array> update(int layer_idx, const array& new_k, const array& new_v) {
        int B = static_cast<int>(new_k.shape(0));
        int H = static_cast<int>(new_k.shape(1));
        int seq_len = static_cast<int>(new_k.shape(2));
        int D = static_cast<int>(new_k.shape(3));
        int new_pos = position_ + seq_len;
        
        if (position_ == 0) {
            // First tokens - just use the new K/V directly
            if (quantized_) {
                auto abs_k = abs(new_k);
                auto max_k = max(abs_k, {-2, -1}, true); // [B, H, 1, 1]
                auto scales_k = max_k / 127.0f;
                auto int8_k = astype(round(new_k / scales_k), int8);
                
                auto abs_v = abs(new_v);
                auto max_v = max(abs_v, {-2, -1}, true);
                auto scales_v = max_v / 127.0f;
                auto int8_v = astype(round(new_v / scales_v), int8);
                
                k_cache_[layer_idx] = int8_k;
                v_cache_[layer_idx] = int8_v;
                k_scales_[layer_idx] = scales_k;
                v_scales_[layer_idx] = scales_v;
                
                auto full_k = int8_k * scales_k;
                auto full_v = int8_v * scales_v;
                return {full_k, full_v};
            } else {
                k_cache_[layer_idx] = new_k;
                v_cache_[layer_idx] = new_v;
                return {new_k, new_v};
            }
        }
        
        // Compute cached up to position
        auto cached_k = slice(k_cache_[layer_idx], {0, 0, 0, 0}, {B, H, position_, D});
        auto cached_v = slice(v_cache_[layer_idx], {0, 0, 0, 0}, {B, H, position_, D});
        
        array updated_k(0), updated_v(0), full_k(0), full_v(0);
        if (quantized_) {
            auto cached_scales_k = slice(k_scales_[layer_idx], {0, 0, 0, 0}, {B, H, position_, 1});
            auto cached_scales_v = slice(v_scales_[layer_idx], {0, 0, 0, 0}, {B, H, position_, 1});
            
            auto abs_k = abs(new_k);
            auto max_k = max(abs_k, {-2, -1}, true);
            auto scales_k = max_k / 127.0f;
            auto int8_k = astype(round(new_k / scales_k), int8);
            
            auto abs_v = abs(new_v);
            auto max_v = max(abs_v, {-2, -1}, true);
            auto scales_v = max_v / 127.0f;
            auto int8_v = astype(round(new_v / scales_v), int8);
            
            updated_k = concatenate({cached_k, int8_k}, 2);
            updated_v = concatenate({cached_v, int8_v}, 2);
            
            auto updated_scales_k = concatenate({cached_scales_k, scales_k}, 2);
            auto updated_scales_v = concatenate({cached_scales_v, scales_v}, 2);
            
            full_k = updated_k * updated_scales_k;
            full_v = updated_v * updated_scales_v;
            
            k_scales_[layer_idx] = updated_scales_k;
            v_scales_[layer_idx] = updated_scales_v;
        } else {
            updated_k = concatenate({cached_k, new_k}, 2);
            updated_v = concatenate({cached_v, new_v}, 2);
            
            full_k = updated_k;
            full_v = updated_v;
        }
        
        // Handle overflow
        if (new_pos > max_length_) {
            int excess = new_pos - max_length_;
            int keep_from = excess;
            
            updated_k = slice(updated_k, {0, 0, keep_from, 0}, {B, H, new_pos, D});
            updated_v = slice(updated_v, {0, 0, keep_from, 0}, {B, H, new_pos, D});
            
            if (quantized_) {
                auto updated_scales_k = slice(k_scales_[layer_idx], {0, 0, keep_from, 0}, {B, H, new_pos, 1});
                auto updated_scales_v = slice(v_scales_[layer_idx], {0, 0, keep_from, 0}, {B, H, new_pos, 1});
                
                full_k = updated_k * updated_scales_k;
                full_v = updated_v * updated_scales_v;
                
                k_scales_[layer_idx] = updated_scales_k;
                v_scales_[layer_idx] = updated_scales_v;
            } else {
                full_k = updated_k;
                full_v = updated_v;
            }
        }
        
        // Assign back
        k_cache_[layer_idx] = updated_k;
        v_cache_[layer_idx] = updated_v;
        
        return {full_k, full_v};
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
    
    /*
     * memory_usage_bytes
     * Returns total memory used by the cache in bytes.
     */
    size_t memory_usage_bytes() const {
        size_t total = 0;
        for (const auto& a : k_cache_) total += a.nbytes();
        for (const auto& a : v_cache_) total += a.nbytes();
        if (quantized_) {
            for (const auto& a : k_scales_) total += a.nbytes();
            for (const auto& a : v_scales_) total += a.nbytes();
        }
        return total;
    }
    
    /*
     * fp16_equivalent_bytes
     * Returns memory usage if cache was FP16 (for compression ratio).
     */
    size_t fp16_equivalent_bytes() const {
        size_t elements_per_cache = static_cast<size_t>(1) * num_kv_heads_ * max_length_ * head_dim_;
        size_t total_elements = elements_per_cache * 2 * num_layers_; // k and v
        return total_elements * sizeof(float16_t);
    }

private:
    bool quantized_;
    bool initialized_;
    int position_;        // Current position in cache (number of tokens stored)
    int max_length_;      // Maximum cache length
    int num_layers_;      // Number of layers
    int num_kv_heads_;    // Number of KV heads (set in initialize)
    int head_dim_;        // Head dimension (set in initialize)
    
    
    std::vector<array> k_scales_;
    std::vector<array> v_scales_;
};

/*
 * HighPerformanceKVCache
 * 
 * Ultra-fast KV cache with O(1) updates using pre-allocated buffers.
 * Uses slice_update for in-place writes instead of concatenate.
 * 
 * This is critical for achieving 150+ TPS.
 */
class HighPerformanceKVCache {
public:
    std::vector<array> k_cache_;
    std::vector<array> v_cache_;
    
    HighPerformanceKVCache() 
        : initialized_(false), position_(0), max_length_(0), num_layers_(0), num_kv_heads_(0), head_dim_(0) {}
    
    /*
     * initialize
     * Pre-allocates FULL KV cache buffers for all layers upfront.
     * This enables O(1) slice_update writes.
     */
    void initialize(int num_layers, int num_kv_heads, int max_length, int head_dim) {
        num_layers_ = num_layers;
        num_kv_heads_ = num_kv_heads;
        max_length_ = max_length;
        head_dim_ = head_dim;
        position_ = 0;
        
        k_cache_.clear();
        v_cache_.clear();
        k_cache_.reserve(num_layers);
        v_cache_.reserve(num_layers);
        
        // Pre-allocate FULL buffers for each layer
        for (int i = 0; i < num_layers; ++i) {
            k_cache_.push_back(zeros({1, num_kv_heads, max_length, head_dim}, float16));
            v_cache_.push_back(zeros({1, num_kv_heads, max_length, head_dim}, float16));
        }
        
        // Force materialization
        eval(k_cache_);
        eval(v_cache_);
        initialized_ = true;
    }
    
    void clear() {
        position_ = 0;
        // Don't reallocate - just reset position
    }
    
    /*
     * update
     * O(1) update using slice_update - no concatenation!
     * Returns a view of the valid portion of the cache.
     */
    std::pair<array, array> update(int layer_idx, const array& new_k, const array& new_v) {
        int seq_len = static_cast<int>(new_k.shape(2));
        int new_pos = position_ + seq_len;
        
        // Clamp to max_length
        if (new_pos > max_length_) {
            // Sliding window: shift cache and insert at end
            // This is more complex - for now, just cap at max
            new_pos = max_length_;
        }
        
        // O(1) in-place update using slice_update
        // Write new_k/new_v at position [0:B, 0:H, position_:new_pos, 0:D]
        k_cache_[layer_idx] = slice_update(
            k_cache_[layer_idx], 
            new_k, 
            Shape{0, 0, position_, 0},  // start
            Shape{1, num_kv_heads_, new_pos, head_dim_}  // stop
        );
        
        v_cache_[layer_idx] = slice_update(
            v_cache_[layer_idx], 
            new_v, 
            Shape{0, 0, position_, 0},
            Shape{1, num_kv_heads_, new_pos, head_dim_}
        );
        
        // Return slice of valid cache [0:new_pos]
        array full_k = slice(k_cache_[layer_idx], {0, 0, 0, 0}, {1, num_kv_heads_, new_pos, head_dim_});
        array full_v = slice(v_cache_[layer_idx], {0, 0, 0, 0}, {1, num_kv_heads_, new_pos, head_dim_});
        
        return {full_k, full_v};
    }
    
    void advance_position(int seq_len) {
        position_ += seq_len;
        if (position_ > max_length_) {
            position_ = max_length_;
        }
    }
    
    bool is_initialized() const { return initialized_; }
    int position() const { return position_; }
    int max_length() const { return max_length_; }
    int num_layers() const { return num_layers_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }
    
    size_t memory_usage_bytes() const {
        size_t total = 0;
        for (const auto& a : k_cache_) total += a.nbytes();
        for (const auto& a : v_cache_) total += a.nbytes();
        return total;
    }

private:
    bool initialized_;
    int position_;
    int max_length_;
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
};

} // namespace ryzenai::mlx
