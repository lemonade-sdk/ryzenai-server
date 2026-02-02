/*
 * qwen3_moe_inference.cpp
 *
 * Optimized Qwen3 MoE Inference Implementation.
 * Features weight loading, fused projections, and null-safe forward pass.
 */

#include "ryzenai/mlx/models/qwen3_moe_inference.h"
#include "ryzenai/mlx/attention.h"
#include "ryzenai/mlx/quantization.h"
#include <iostream>
#include <fstream>
#include <json.hpp>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>

using namespace mlx::core;

static std::vector<std::string> layer_prefixes_;

/*
 * Qwen3MoEInference constructor
 *
 * Initializes the inference engine, sets up dimensions, loads configuration,
 * and pre-allocates KV cache.
 */
Qwen3MoEInference::Qwen3MoEInference(const MlxOgaModel& model) 
    : model_(model), cache_initialized_(false), cache_position_(0), step_(0) {
    
    actual_hidden_size_ = model_.hidden_size;
    tie_word_embeddings_ = model_.tie_word_embeddings;
    intermediate_size_ = model_.intermediate_size;

    if (model_.head_dim > 0) {
        head_dim_ = model_.head_dim;
    } else {
        head_dim_ = actual_hidden_size_ / model_.num_attention_heads;
    }
    
    std::string config_path = model_.model_path + "/config.json";
    std::ifstream f(config_path);
    if (f.is_open()) {
        nlohmann::json config;
        f >> config;
        if (config.contains("num_experts")) moe_config_.num_experts = config["num_experts"];
        if (config.contains("num_experts_per_tok")) moe_config_.num_experts_per_tok = config["num_experts_per_tok"];
        if (config.contains("decoder_sparse_step")) moe_config_.decoder_sparse_step = config["decoder_sparse_step"];
        if (config.contains("moe_intermediate_size")) {
            moe_intermediate_size_ = config["moe_intermediate_size"];
            moe_config_.moe_intermediate_size = moe_intermediate_size_;
        } else {
            moe_intermediate_size_ = intermediate_size_;
        }
        if (config.contains("mlp_only_layers")) {
            for (const auto& layer_idx : config["mlp_only_layers"]) {
                mlp_only_layers_.insert(layer_idx.get<int>());
                moe_config_.mlp_only_layers.push_back(layer_idx.get<int>());
            }
        }
    }
    
    // KV Cache Alloc
    max_cache_length_ = model_.max_context_length;
    Dtype cache_dtype = float16;
    if (model_.weights.count("embed_tokens.weight")) {
        cache_dtype = model_.weights.at("embed_tokens.weight").dtype();
    }
    
    k_cache_.reserve(model_.num_hidden_layers);
    v_cache_.reserve(model_.num_hidden_layers);
    
    // Allocate KV cache in smaller chunks to avoid GPU timeout on large models
    const int chunk_size = 8;
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        k_cache_.push_back(zeros({1, model_.num_key_value_heads, max_cache_length_, head_dim_}, cache_dtype));
        v_cache_.push_back(zeros({1, model_.num_key_value_heads, max_cache_length_, head_dim_}, cache_dtype));
        
        // Eval every chunk_size layers to prevent GPU timeout
        if ((i + 1) % chunk_size == 0 || i == model_.num_hidden_layers - 1) {
            eval(k_cache_);
            eval(v_cache_);
        }
    }
    cache_initialized_ = true;
    
    // 1. Cache Weights (Perform fusion and robust loading)
    cache_weights();
    
    // 2. Setup References (Map loaded weights to structs)
    setup_weight_references();
    
    // 3. Setup MoE Blocks (Initialize SwitchGLU objects)
    setup_moe_blocks();
    
    attention_scale_ = 1.0f / sqrt(static_cast<float>(head_dim_));
    rope_theta_ = model_.rope_theta > 0 ? model_.rope_theta : 1000000.0f;
    
    // For tie_word_embeddings, pre-transpose embedding for use as LM head
    // But only if the embedding is NOT quantized
    if (tie_word_embeddings_ && cached_weights_.count("embed_tokens.weight") && !embed_is_quantized_) {
        embed_tokens_transposed_ = transpose(cached_weights_.at("embed_tokens.weight"), {1, 0});
        eval(*embed_tokens_transposed_);
    }
    
    // Print MoE setup summary
    std::cout << "[Qwen3MoEInference] hidden_size=" << actual_hidden_size_
              << ", head_dim=" << head_dim_
              << ", num_heads=" << model_.num_attention_heads
              << ", num_kv_heads=" << model_.num_key_value_heads
              << ", num_experts=" << moe_config_.num_experts
              << ", experts_per_tok=" << moe_config_.num_experts_per_tok
              << ", decoder_sparse_step=" << moe_config_.decoder_sparse_step
              << ", tie_word_embeddings=" << (tie_word_embeddings_ ? "yes" : "no")
              << std::endl;
    
    std::cout << "[Qwen3MoEInference] Pre-allocating KV cache: " 
              << model_.num_hidden_layers << " layers x " 
              << max_cache_length_ << " tokens" << std::endl;
    
    // Count MoE blocks that were successfully set up
    int moe_block_count = 0;
    for (const auto& block : moe_blocks_) {
        if (block.has_value()) moe_block_count++;
    }
    std::cout << "[Qwen3MoEInference] Setup " << moe_block_count << " MoE blocks" << std::endl;
    
    if (embed_tokens_transposed_.has_value()) {
        std::cout << "[Qwen3MoEInference] Pre-transposed embed_tokens for LM head" << std::endl;
    }
}

/*
 * clear_cache
 *
 * Resets the KV cache position and step counter.
 */
void Qwen3MoEInference::clear_cache() { step_ = 0; cache_position_ = 0; }

/*
 * is_moe_layer
 *
 * Determines if a given layer index corresponds to a Mixture of Experts layer.
 */
bool Qwen3MoEInference::is_moe_layer(int layer_idx) const {
    if (mlp_only_layers_.count(layer_idx)) return false;
    return (moe_config_.num_experts > 0 && (layer_idx + 1) % moe_config_.decoder_sparse_step == 0);
}

/*
 * forward
 *
 * Performs a single inference step, processing input tokens through all layers.
 */
array Qwen3MoEInference::forward(const std::vector<int32_t>& input_tokens, const MlxOgaGeneratorParams& params) {
    int seq_len = static_cast<int>(input_tokens.size());
    if (seq_len > 1) clear_cache();
    
    if (!embed_tokens_) throw std::runtime_error("Critical: embed_tokens.weight missing. Model load failed.");
    
    array token_indices(input_tokens.data(), {seq_len}, int32);
    
    // Handle quantized embedding - dequantize only the rows we need
    array h = take(*embed_tokens_, token_indices, 0);
    
    if (embed_is_quantized_) {
        // Embedding is quantized - need to dequantize the selected rows
        auto scales_it = cached_weights_.find("embed_tokens.scales");
        if (scales_it != cached_weights_.end()) {
            // Take the corresponding scales for these tokens
            array token_scales = take(scales_it->second, token_indices, 0);
            
            auto biases_it = cached_weights_.find("embed_tokens.biases");
            array token_biases = biases_it != cached_weights_.end() ?
                take(biases_it->second, token_indices, 0) :
                zeros({seq_len, actual_hidden_size_}, h.dtype());
            
            h = dequantize(h, token_scales, token_biases, 
                          model_.quantization.group_size, model_.quantization.bits);
        }
    }
    
    // Get actual embedding dimension from the embedding output
    int embed_dim = static_cast<int>(h.shape(-1));
    h = reshape(h, {1, seq_len, embed_dim});
    std::string mask_type = ryzenai::mlx::Attention::get_mask_type(seq_len);

    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        const Qwen3MoELayerWeights& layer = layer_weights_[i];
        
        // Attention
        array normed = rms_norm_fast(h, layer.input_layernorm);
        h = h + self_attention_fast(normed, layer, i, mask_type);
        
        // MLP
        normed = rms_norm_fast(h, layer.post_attn_layernorm);
        
        // Lazy initialization of MoE blocks
        if (is_moe_layer(i) && !moe_blocks_[i].has_value()) {
            initialize_moe_layer_lazy(i);
        }

        // 2. Direct Initialization (Compiler executes only one branch)
        array mlp_out = is_moe_layer(i) ? 
            (*moe_blocks_[i])(normed) : 
            dense_mlp_block(normed, layer.dense_mlp);

        // 3. Accumulate
        h = h + mlp_out;
    }
    
    cache_position_ += seq_len;
    step_ += seq_len;
    if (cache_position_ > max_cache_length_) { cache_position_ = max_cache_length_; step_ = max_cache_length_; }
    
    h = rms_norm_fast(h, final_norm_);
    array last_h = take(h, array({seq_len - 1}), 1);
    
    // LM head output
    if (tie_word_embeddings_) {
        if (embed_tokens_transposed_.has_value()) {
            // Non-quantized case: use pre-transposed embedding
            return reshape(matmul(last_h, *embed_tokens_transposed_), {model_.vocab_size});
        } else if (embed_is_quantized_) {
            // Quantized case: use quantized_matmul with embedding weights
            auto scales = cached_weights_.find("embed_tokens.scales");
            auto biases = cached_weights_.find("embed_tokens.biases");
            if (scales != cached_weights_.end()) {
                std::optional<array> biases_opt = biases != cached_weights_.end() ? 
                    std::optional(biases->second) : std::nullopt;
                array output = quantized_matmul(last_h, *embed_tokens_, scales->second, biases_opt, 
                                                true, model_.quantization.group_size, model_.quantization.bits, "affine");
                return reshape(output, {model_.vocab_size});
            }
        }
    }
    
    // Fallback: use separate LM head
    return reshape(linear_fast(last_h, lm_head_), {model_.vocab_size});
}

/*
 * rms_norm_fast
 *
 * Applies root mean square normalization to the input array.
 */
array Qwen3MoEInference::rms_norm_fast(const array& x, const array* weight) {
    if (!weight) return x;
    return fast::rms_norm(x, *weight, model_.rms_norm_eps);
}

/*
 * linear_fast
 *
 * Performs matrix multiplication, supporting both quantized and full-precision weights.
 */
array Qwen3MoEInference::linear_fast(const array& x, const ryzenai::mlx::LinearWeights& w) {
    if (!w.weight) throw std::runtime_error("linear_fast: weight is null");
    
    if (w.is_quantized()) {
        std::optional<array> biases_opt = w.biases ? std::optional(*w.biases) : std::nullopt;
        return quantized_matmul(x, *w.weight, *w.scales, biases_opt, true, w.group_size, w.bits, "affine");
    }
    array output = w.has_pretransposed() ? matmul(x, *w.weight_T) : matmul(x, transpose(*w.weight, {1, 0}));
    if (w.biases) output = output + *w.biases;
    return output;
}

/*
 * self_attention_fast
 *
 * Computes self-attention with KV caching and rotary position embeddings.
 */
array Qwen3MoEInference::self_attention_fast(const array& x, const Qwen3MoELayerWeights& layer,
                                              int layer_idx, const std::string& mask_type) {
    return ryzenai::mlx::Attention::self_attention_fast_impl(
        x, layer, layer_idx, mask_type, k_cache_, v_cache_, 
        cache_position_, max_cache_length_, head_dim_, rope_theta_, 
        attention_scale_, model_.num_attention_heads, model_.num_key_value_heads
    );
}

/*
 * dense_mlp_block
 *
 * Implements a standard dense Feed-Forward Network block with SwiGLU activation.
 */
array Qwen3MoEInference::dense_mlp_block(const array& x, const ryzenai::mlx::MLPWeights& mlp) {
    array gate_up = linear_fast(x, mlp.gate_proj);
    
    auto chunks = split(gate_up, 2, -1);
    array activated = ryzenai::moe::swiglu(chunks[0], chunks[1]);
    
    return linear_fast(activated, mlp.down_proj);
}

/*
 * cache_weights
 *
 * Loads, fuses, and caches model weights from the MlxOgaModel.
 */
void Qwen3MoEInference::cache_weights() {
    cached_weights_.clear();
    const auto& w = model_.weights;
    
    if (w.empty()) {
        std::cerr << "[Qwen3MoEInference] ERROR: Model weights map is empty!" << std::endl;
        return;
    }
    
    std::string prefix_base = "";
    if (w.find("model.layers.0.input_layernorm.weight") != w.end()) {
        prefix_base = "model.";
    }

    layer_prefixes_.clear();
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        layer_prefixes_.push_back(prefix_base + "layers." + std::to_string(i) + ".");
    }

    auto cache_weight = [&](const std::string& name) {
        auto it = w.find(name + ".weight");
        if (it != w.end()) {
            cached_weights_.emplace(name + ".weight", it->second);
            auto scales = w.find(name + ".scales");
            if (scales != w.end()) cached_weights_.emplace(name + ".scales", scales->second);
            auto biases = w.find(name + ".biases");
            if (biases != w.end()) cached_weights_.emplace(name + ".biases", biases->second);
            
            // Pre-transpose for standard linear (optimization for linear_fast)
            // Only transpose if it's a 2D tensor (linear layer weight, not norm weight)
            if (scales == w.end() && it->second.ndim() == 2) {
                cached_weights_.emplace(name + ".weight_T", transpose(it->second, {1, 0}));
            }
        }
    };
    
    // Fuse Helper: Concatenates two weights along an axis
    auto fuse_tensors = [&](const std::string& key1, const std::string& key2, 
                            const std::string& dest_key, int axis) {
        auto it1 = w.find(key1);
        auto it2 = w.find(key2);
        if (it1 != w.end() && it2 != w.end()) {
            array fused = concatenate({it1->second, it2->second}, axis);
            cached_weights_.emplace(dest_key, fused);
            return true;
        }
        return false;
    };

    // Cache Embeddings - need to handle quantized embeddings
    std::string embed_key = prefix_base + "embed_tokens";
    if (w.count("embed_tokens.weight")) embed_key = "embed_tokens"; 
    else if (w.count("model.embed_tokens.weight")) embed_key = "model.embed_tokens";
    
    if (w.count(embed_key + ".weight")) {
        array embed_weight = w.at(embed_key + ".weight");
        auto scales_it = w.find(embed_key + ".scales");
        auto biases_it = w.find(embed_key + ".biases");
        
        // Check if embedding is quantized (packed dimension != hidden_size)
        int embed_dim = static_cast<int>(embed_weight.shape(1));
        bool is_quantized = embed_dim != actual_hidden_size_ && scales_it != w.end();
        
        if (is_quantized) {
            // For quantized embeddings, keep in packed format and handle in forward()
            // Dequantizing 151936×2048 tensor at init causes GPU timeout on smaller GPUs
            std::cout << "[Qwen3MoEInference] Keeping embedding in quantized format (packed dim=" 
                      << embed_dim << ") - will dequantize on-demand" << std::endl;
            cached_weights_.emplace("embed_tokens.weight", embed_weight);
            cached_weights_.emplace("embed_tokens.scales", scales_it->second);
            if (biases_it != w.end()) {
                cached_weights_.emplace("embed_tokens.biases", biases_it->second);
            }
            embed_is_quantized_ = true;
        } else {
            cached_weights_.emplace("embed_tokens.weight", embed_weight);
            embed_is_quantized_ = false;
        }
    }

    // Process Layers
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        std::string p = layer_prefixes_[i];
        
        cache_weight(p + "input_layernorm");
        cache_weight(p + "post_attention_layernorm");
        cache_weight(p + "self_attn.q_norm");
        cache_weight(p + "self_attn.k_norm");
        cache_weight(p + "self_attn.o_proj");

        // Fuse QKV - also fuse scales and biases for quantized weights
        if (w.count(p + "self_attn.q_proj.weight")) {
             array fused = concatenate({
                 w.at(p + "self_attn.q_proj.weight"),
                 w.at(p + "self_attn.k_proj.weight"),
                 w.at(p + "self_attn.v_proj.weight")
             }, 0);
             cached_weights_.emplace(p + "self_attn.qkv_proj.weight", fused);
             
             // Also fuse scales if quantized
             if (w.count(p + "self_attn.q_proj.scales")) {
                 array fused_scales = concatenate({
                     w.at(p + "self_attn.q_proj.scales"),
                     w.at(p + "self_attn.k_proj.scales"),
                     w.at(p + "self_attn.v_proj.scales")
                 }, 0);
                 cached_weights_.emplace(p + "self_attn.qkv_proj.scales", fused_scales);
             }
             
             // Also fuse biases if they exist
             if (w.count(p + "self_attn.q_proj.biases")) {
                 array fused_biases = concatenate({
                     w.at(p + "self_attn.q_proj.biases"),
                     w.at(p + "self_attn.k_proj.biases"),
                     w.at(p + "self_attn.v_proj.biases")
                 }, 0);
                 cached_weights_.emplace(p + "self_attn.qkv_proj.biases", fused_biases);
             }
        }

        if (is_moe_layer(i)) {
            // Check for router weight first
            cache_weight(p + "mlp.gate"); // Router
            
            // LAZY LOADING: Don't stack expert weights at init
            // They will be stacked on first use in setup_moe_blocks_lazy()
            // This saves significant RAM at startup
            
            // Check if weights are already in switch_mlp format (pre-stacked) 
            bool has_switch_mlp = w.count(p + "mlp.switch_mlp.gate_proj.weight") > 0;
            
            if (has_switch_mlp) {
                // Already stacked format - just fuse gate+up
                bool success = fuse_tensors(p + "mlp.switch_mlp.gate_proj.weight", 
                                            p + "mlp.switch_mlp.up_proj.weight",
                                            p + "mlp.switch_mlp.gate_up_proj.weight", 1);
                
                if (success) {
                    fuse_tensors(p + "mlp.switch_mlp.gate_proj.scales", 
                                 p + "mlp.switch_mlp.up_proj.scales",
                                 p + "mlp.switch_mlp.gate_up_proj.scales", 1);
                    fuse_tensors(p + "mlp.switch_mlp.gate_proj.biases", 
                                 p + "mlp.switch_mlp.up_proj.biases",
                                 p + "mlp.switch_mlp.gate_up_proj.biases", 1);
                }
                cache_weight(p + "mlp.switch_mlp.down_proj");
                moe_layers_initialized_.insert(i);  // Mark as already initialized
            }
            // For HuggingFace format, expert weights will be stacked lazily during first inference
        } else {
            // DENSE FUSION: [out, in] -> concat on axis 0 -> [2*out, in]
            bool success = fuse_tensors(p + "mlp.gate_proj.weight", 
                                        p + "mlp.up_proj.weight",
                                        p + "mlp.gate_up_proj.weight", 0);
            
            if (success) {
                 fuse_tensors(p + "mlp.gate_proj.scales", p + "mlp.up_proj.scales",
                              p + "mlp.gate_up_proj.scales", 0);
                 fuse_tensors(p + "mlp.gate_proj.biases", p + "mlp.up_proj.biases",
                              p + "mlp.gate_up_proj.biases", 0);
            }
            cache_weight(p + "mlp.gate_up_proj"); // Wrapper
            cache_weight(p + "mlp.down_proj");
        }
    }

    cache_weight(prefix_base + "norm");
    if (!tie_word_embeddings_) cache_weight(prefix_base + "lm_head");
    
    std::cout << "[Qwen3MoEInference] Cached " << cached_weights_.size() << " tensors." << std::endl;
}

void Qwen3MoEInference::setup_weight_references() {
    auto get_weight = [this](const std::string& name) -> const array* {
        auto it = cached_weights_.find(name);
        return it != cached_weights_.end() ? &(it->second) : nullptr;
    };
    
    auto setup_linear = [&](const std::string& prefix) -> ryzenai::mlx::LinearWeights {
        ryzenai::mlx::LinearWeights lw;
        lw.weight = get_weight(prefix + ".weight");
        lw.weight_T = get_weight(prefix + ".weight_T");
        lw.scales = get_weight(prefix + ".scales");
        lw.biases = get_weight(prefix + ".biases");
        lw.group_size = model_.quantization.group_size;
        lw.bits = model_.quantization.bits;
        return lw;
    };
    
    embed_tokens_ = get_weight("embed_tokens.weight");
    final_norm_ = get_weight("norm.weight");
    if (!tie_word_embeddings_) lm_head_ = setup_linear("lm_head");
    
    layer_weights_.resize(model_.num_hidden_layers);
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        const std::string& p = layer_prefixes_[i];
        Qwen3MoELayerWeights& layer = layer_weights_[i];
        
        layer.input_layernorm = get_weight(p + "input_layernorm.weight");
        layer.post_attn_layernorm = get_weight(p + "post_attention_layernorm.weight");
        layer.attention.qkv_proj = setup_linear(p + "self_attn.qkv_proj");
        layer.attention.o_proj = setup_linear(p + "self_attn.o_proj");
        layer.attention.q_norm = get_weight(p + "self_attn.q_norm.weight");
        layer.attention.k_norm = get_weight(p + "self_attn.k_norm.weight");
        
        layer.is_moe_layer = is_moe_layer(i);
        
        if (!layer.is_moe_layer) {
            // DENSE: Map gate_proj to the FUSED gate_up weight
            layer.dense_mlp.gate_proj = setup_linear(p + "mlp.gate_up_proj");
            layer.dense_mlp.down_proj = setup_linear(p + "mlp.down_proj");
        }
    }
}

void Qwen3MoEInference::setup_moe_blocks() {
    moe_blocks_.resize(model_.num_hidden_layers);
    
    for (int i = 0; i < model_.num_hidden_layers; ++i) {
        if (!is_moe_layer(i)) {
            moe_blocks_[i] = std::nullopt;
            continue;
        }
        
        const std::string& p = layer_prefixes_[i];
        auto router_it = cached_weights_.find(p + "mlp.gate.weight");
        if (router_it == cached_weights_.end()) continue;
        
        // Retrieve FUSED MoE Weights
        auto gate_up_it = cached_weights_.find(p + "mlp.switch_mlp.gate_up_proj.weight");
        auto down_it = cached_weights_.find(p + "mlp.switch_mlp.down_proj.weight");
        
        if (gate_up_it == cached_weights_.end() || down_it == cached_weights_.end()) {
             std::cerr << "Warning: MoE weights missing for layer " << i << std::endl;
             continue;
        }

        // Create SwitchLinear using fused weights
        auto get_opt = [&](const std::string& s) -> std::optional<array> {
            auto it = cached_weights_.find(s);
            if (it != cached_weights_.end()) return it->second;
            return std::nullopt;
        };

        ryzenai::moe::SwitchLinear gate_up_sw(
            gate_up_it->second, 
            get_opt(p + "mlp.switch_mlp.gate_up_proj.scales"),
            get_opt(p + "mlp.switch_mlp.gate_up_proj.biases")
        );
        
        ryzenai::moe::SwitchLinear down_sw(
            down_it->second,
            get_opt(p + "mlp.switch_mlp.down_proj.scales"),
            get_opt(p + "mlp.switch_mlp.down_proj.biases")
        );

        ryzenai::moe::SwitchGLU switch_mlp(gate_up_sw, down_sw);
        
        moe_blocks_[i] = ryzenai::moe::SparseMoEBlock(
            router_it->second, switch_mlp, moe_config_);
    }
}

void Qwen3MoEInference::initialize_moe_layer_lazy(int layer_idx) {
    const std::string& p = layer_prefixes_[layer_idx];
    const auto& w = model_.weights;
    
    std::cout << "[Qwen3MoEInference] Lazy-initializing MoE for layer " << layer_idx << std::endl;
    
    // Check for router weight
    auto router_it = cached_weights_.find(p + "mlp.gate.weight");
    if (router_it == cached_weights_.end()) {
        std::cerr << "[Qwen3MoEInference] Warning: No router weight for layer " << layer_idx << std::endl;
        return;
    }
    
    // Stack individual expert weights
    std::vector<array> gate_weights, up_weights, down_weights;
    std::vector<array> gate_scales, up_scales, down_scales;
    std::vector<array> gate_biases, up_biases, down_biases;
    
    for (int e = 0; e < moe_config_.num_experts; ++e) {
        std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
        
        // Gate projection
        auto gate_it = w.find(ep + "gate_proj.weight");
        if (gate_it != w.end()) gate_weights.push_back(gate_it->second);
        auto gate_s = w.find(ep + "gate_proj.scales");
        if (gate_s != w.end()) gate_scales.push_back(gate_s->second);
        auto gate_b = w.find(ep + "gate_proj.biases");
        if (gate_b != w.end()) gate_biases.push_back(gate_b->second);
        
        // Up projection
        auto up_it = w.find(ep + "up_proj.weight");
        if (up_it != w.end()) up_weights.push_back(up_it->second);
        auto up_s = w.find(ep + "up_proj.scales");
        if (up_s != w.end()) up_scales.push_back(up_s->second);
        auto up_b = w.find(ep + "up_proj.biases");
        if (up_b != w.end()) up_biases.push_back(up_b->second);
        
        // Down projection
        auto down_it = w.find(ep + "down_proj.weight");
        if (down_it != w.end()) down_weights.push_back(down_it->second);
        auto down_s = w.find(ep + "down_proj.scales");
        if (down_s != w.end()) down_scales.push_back(down_s->second);
        auto down_b = w.find(ep + "down_proj.biases");
        if (down_b != w.end()) down_biases.push_back(down_b->second);
    }
    
    if (gate_weights.size() != static_cast<size_t>(moe_config_.num_experts)) {
        std::cerr << "[Qwen3MoEInference] Warning: Only found " << gate_weights.size() 
                  << " experts for layer " << layer_idx << std::endl;
        return;
    }
    
    // Stack expert weights: [experts, out, in]
    // Evaluate in batches to avoid GPU timeout on large models
    array stacked_gate = stack(gate_weights, 0);
    eval(stacked_gate);
    array stacked_up = stack(up_weights, 0);
    eval(stacked_up);
    // Fuse gate+up: [experts, out, in] concat on axis 1 -> [experts, 2*out, in]
    array fused_gate_up = concatenate({stacked_gate, stacked_up}, 1);
    eval(fused_gate_up);
    
    // Clear intermediate arrays to free memory
    stacked_gate = array({}, float16);
    stacked_up = array({}, float16);
    
    cached_weights_.insert_or_assign(p + "mlp.switch_mlp.gate_up_proj.weight", fused_gate_up);
    
    if (!gate_scales.empty() && gate_scales.size() == static_cast<size_t>(moe_config_.num_experts)) {
        array stacked_gate_s = stack(gate_scales, 0);
        array stacked_up_s = stack(up_scales, 0);
        array fused_scales = concatenate({stacked_gate_s, stacked_up_s}, 1);
        cached_weights_.insert_or_assign(p + "mlp.switch_mlp.gate_up_proj.scales", fused_scales);
    }
    if (!gate_biases.empty() && gate_biases.size() == static_cast<size_t>(moe_config_.num_experts)) {
        array stacked_gate_b = stack(gate_biases, 0);
        array stacked_up_b = stack(up_biases, 0);
        array fused_biases = concatenate({stacked_gate_b, stacked_up_b}, 1);
        cached_weights_.insert_or_assign(p + "mlp.switch_mlp.gate_up_proj.biases", fused_biases);
    }
    
    if (!down_weights.empty()) {
        array stacked_down = stack(down_weights, 0);
        eval(stacked_down);  // Force evaluation
        cached_weights_.insert_or_assign(p + "mlp.switch_mlp.down_proj.weight", stacked_down);
        
        if (!down_scales.empty()) {
            array ds = stack(down_scales, 0);
            eval(ds);
            cached_weights_.insert_or_assign(p + "mlp.switch_mlp.down_proj.scales", ds);
        }
        if (!down_biases.empty()) {
            array db = stack(down_biases, 0);
            eval(db);
            cached_weights_.insert_or_assign(p + "mlp.switch_mlp.down_proj.biases", db);
        }
    }
    
    // Clear vectors to free memory
    gate_weights.clear(); up_weights.clear(); down_weights.clear();
    gate_scales.clear(); up_scales.clear(); down_scales.clear();
    gate_biases.clear(); up_biases.clear(); down_biases.clear();
    
    // Now create the MoE block
    auto get_opt = [&](const std::string& s) -> std::optional<array> {
        auto it = cached_weights_.find(s);
        if (it != cached_weights_.end()) return it->second;
        return std::nullopt;
    };
    
    auto gate_up_it = cached_weights_.find(p + "mlp.switch_mlp.gate_up_proj.weight");
    auto down_it = cached_weights_.find(p + "mlp.switch_mlp.down_proj.weight");
    
    if (gate_up_it != cached_weights_.end() && down_it != cached_weights_.end()) {
        ryzenai::moe::SwitchLinear gate_up_sw(
            gate_up_it->second, 
            get_opt(p + "mlp.switch_mlp.gate_up_proj.scales"),
            get_opt(p + "mlp.switch_mlp.gate_up_proj.biases")
        );
        
        ryzenai::moe::SwitchLinear down_sw(
            down_it->second,
            get_opt(p + "mlp.switch_mlp.down_proj.scales"),
            get_opt(p + "mlp.switch_mlp.down_proj.biases")
        );

        ryzenai::moe::SwitchGLU switch_mlp(gate_up_sw, down_sw);
        
        moe_blocks_[layer_idx] = ryzenai::moe::SparseMoEBlock(
            router_it->second, switch_mlp, moe_config_);
        
        moe_layers_initialized_.insert(layer_idx);
        std::cout << "[Qwen3MoEInference] Layer " << layer_idx << " MoE initialized" << std::endl;
    } else {
        std::cerr << "[Qwen3MoEInference] Failed to create MoE block for layer " << layer_idx << std::endl;
    }
}

int Qwen3MoEInference::sample_token(const ::mlx::core::array& logits, 
                                     const MlxOgaGeneratorParams& params) {
    using namespace mlx::core;
    
    if (params.do_sample) {
        if (params.temperature > 0.0f) {
            array scaled_logits = logits / params.temperature;
            array probs = softmax(scaled_logits);
            auto max_idx = argmax(probs);
            return static_cast<int>(max_idx.item<int32_t>());
        } else {
            auto max_idx = argmax(logits);
            return static_cast<int>(max_idx.item<int32_t>());
        }
    } else {
        auto max_idx = argmax(logits);
        return static_cast<int>(max_idx.item<int32_t>());
    }
}
