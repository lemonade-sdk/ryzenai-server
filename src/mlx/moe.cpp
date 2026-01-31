/*
 * moe.cpp
 * * Generic Mixture of Experts (MoE) components implementation.
 * Provides reusable MoE blocks for Qwen3-MoE, Mixtral, DeepSeek, etc.
 */

#include "ryzenai/mlx/moe.h"
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/transforms.h>
#include <iostream>

namespace ryzenai {
namespace moe {

using namespace ::mlx::core;


/*
 * swiglu - SwiGLU activation function
 * Implements: silu(gate) * up = gate * sigmoid(gate) * up
 */
array swiglu(const array& gate, const array& up) {
    return gate * sigmoid(gate) * up;
}


/*
 * gather_sort
 * Sorts indices for efficient gather_mm access.
 */
std::tuple<array, array, array> gather_sort(const array& x, const array& indices) {
    // Get the top-k dimension (last dim of indices)
    int M = indices.shape(-1);
    
    // Flatten indices to 1D
    array flat_indices = flatten(indices);
    
    // Sort indices and get inverse order
    array order = argsort(flat_indices);
    array inv_order = argsort(order);
    
    // Reorder x according to sorted indices
    auto x_shape = x.shape();
    int hidden = x_shape.back();
    
    // Flatten x to [N, hidden] where N = product of all dims except last
    array flat_x = reshape(x, {-1, hidden});
    
    // Get sorted indices divided by M (maps to token positions)
    array token_indices = order / M;
    array sorted_x = take(flat_x, token_indices, 0);
    
    // Get sorted expert indices
    array sorted_indices = take(flat_indices, order);
    
    return {sorted_x, sorted_indices, inv_order};
}


/*
 * scatter_unsort
 * Reverses the sorting done by gather_sort.
 * Uses Shape object directly to prevent std::vector overhead.
 */
array scatter_unsort(const array& x, const array& inv_order, 
                     const Shape& original_shape) {
                     
    array unsorted = take(x, inv_order, 0);
    
    // Directly reshape using the passed Shape object
    return reshape(unsorted, original_shape);
}

/*
 * top_k_routing
 * Computes top-k expert selection from router logits.
 */
std::pair<array, array> top_k_routing(const array& router_output, 
                                       int k, 
                                       bool normalize) {
    array gates = softmax(router_output, -1);
    
    // Get top-k indices
    // argpartition is more efficient than argsort for finding top-k
    array all_indices = argpartition(gates, -k, -1);
    
    int num_experts = gates.shape(-1);
    array indices = slice(all_indices, {0}, {}, {-1, num_experts - k, num_experts, 1});
    
    // Sort just the top-k for stability (optional, but good for gather_mm)
    // Actually, simple argsort is often fast enough on GPU for small num_experts
    array sorted_indices = argsort(gates, -1);
    indices = slice(sorted_indices, {-k}, {});  // Take last k
    indices = reshape(indices, {gates.shape(0), gates.shape(1), k});
    
    // Get scores
    array scores = take_along_axis(gates, indices, -1);
    
    if (normalize) {
        array score_sum = sum(scores, -1, true);
        scores = scores / score_sum;
    }
    
    return {indices, scores};
}


// ==================== SwitchLinear ====================

// ==================== SwitchLinear ====================

SwitchLinear::SwitchLinear(const array& weight,
                           const std::optional<array>& scales,
                           const std::optional<array>& biases,
                           const std::optional<array>& bias,
                           int group_size,
                           int bits)
    : weight_(weight)   // <--- FIX: Initialize weight_ explicitly here
    , scales_(scales)
    , biases_(biases)
    , bias_(bias)
    , group_size_(group_size)
    , bits_(bits) {

    // OPTIMIZATION: Check shape and transpose ONCE during construction
    // Standard Linear weights are [out, in]. Stacked experts are [experts, out, in].
    // MLX gather_mm expects [experts, in, out].
    
    // We already initialized weight_ = weight. Now checks if we need to overwrite it with a transpose.
    bool is_square = (weight.shape(-1) == weight.shape(-2));
    
    // Heuristic: If out > in (often true for Gate/Up), we likely need to transpose 
    // from [experts, out, in] -> [experts, in, out]
    if (!is_square && weight.shape(-1) < weight.shape(-2) && !scales.has_value()) {
         weight_ = transpose(weight, {0, 2, 1});
    }
    
    // Force evaluation to ensure transpose happens now, not at runtime
    eval(weight_);
}

array SwitchLinear::operator()(const array& x, const array& indices, bool sorted_indices) const {
    // weight_ is already [experts, in, out]
    array result = gather_mm(x, weight_, indices);
    
    if (bias_.has_value()) {
        array selected_bias = take(*bias_, indices, 0);
        selected_bias = expand_dims(selected_bias, -2);
        result = result + selected_bias;
    }
    
    return result;
}


// ==================== SwitchGLU ====================

SwitchGLU::SwitchGLU(const SwitchLinear& gate_up_proj,
                     const SwitchLinear& down_proj)
    : gate_up_proj_(gate_up_proj)
    , down_proj_(down_proj) {
}

/*
 * SwitchGLU forward pass (FUSED)
 */
array SwitchGLU::operator()(const array& x, const array& indices) const {
    // [batch, seq, hidden] -> [batch, seq, 1, 1, hidden]
    array x_expanded = expand_dims(expand_dims(x, -2), -2);
    
    int num_tokens = static_cast<int>(indices.size());
    bool do_sort = num_tokens >= 64;
    
    array idx = indices;
    array sorted_x = x_expanded;
    std::optional<array> inv_order_opt;
    
    if (do_sort) {
        auto [sx, si, io] = gather_sort(x_expanded, indices);
        sorted_x = sx;
        idx = si;
        inv_order_opt = io;
    }
    
    // 1. Fused Gate + Up Projection
    // Result shape: [tokens, hidden * 2]
    array gate_up = gate_up_proj_(sorted_x, idx, do_sort);
    
    // 2. Split and Apply SwiGLU
    // Split last dim into 2 chunks
    auto chunks = split(gate_up, 2, -1);
    array activated = swiglu(chunks[0], chunks[1]);
    
    // 3. Down Projection
    array result = down_proj_(activated, idx, do_sort);
    
    if (do_sort && inv_order_opt.has_value()) {
        // Fix: Use x_expanded.shape() (Shape object) directly
        result = scatter_unsort(result, *inv_order_opt, x_expanded.shape());
    }
    
    return squeeze(result, -2);
}


// ==================== SparseMoEBlock ====================

SparseMoEBlock::SparseMoEBlock(const array& router_weight,
                               const SwitchGLU& switch_mlp,
                               const MoEConfig& config)
    : router_weight_(router_weight)
    , switch_mlp_(switch_mlp)
    , config_(config) {
}

array SparseMoEBlock::operator()(const array& x) const {
    // 1. Compute router logits
    array router_logits = matmul(x, transpose(router_weight_, {1, 0}));
    
    // 2. Get top-k expert indices and scores
    auto [indices, scores] = top_k_routing(router_logits, config_.num_experts_per_tok, 
                                            config_.norm_topk_prob);
    
    // 3. Route through experts via SwitchGLU
    array expert_out = switch_mlp_(x, indices);
    
    // 4. Combine expert outputs weighted by scores
    array weights = expand_dims(scores, -1);
    array combined = sum(expert_out * weights, -2);
    
    return combined;
}


}  // namespace moe
}  // namespace ryzenai