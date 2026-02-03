/*
 * moe.cpp
 *
 * Generic Mixture of Experts (MoE) components implementation.
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

array swiglu(const array& gate, const array& up) {
    return gate * sigmoid(gate) * up;
}

std::tuple<array, array, array> gather_sort(const array& x, const array& indices) {
    int M = indices.shape(-1);

    array flat_indices = flatten(indices);

    array order = argsort(flat_indices);
    array inv_order = argsort(order);

    auto x_shape = x.shape();
    int hidden = x_shape.back();

    array flat_x = reshape(x, {-1, hidden});

    array token_indices = order / M;
    array sorted_x = take(flat_x, token_indices, 0);

    array sorted_indices = take(flat_indices, order);

    return {sorted_x, sorted_indices, inv_order};
}

array scatter_unsort(const array& x, const array& inv_order,
                     const Shape& original_shape) {

    array unsorted = take(x, inv_order, 0);

    return reshape(unsorted, original_shape);
}

std::pair<array, array> top_k_routing(const array& router_output,
                                       int k,
                                       bool normalize) {
    array gates = softmax(router_output, -1);

    array all_indices = argpartition(gates, -k, -1);

    int num_experts = gates.shape(-1);
    array indices = slice(all_indices, {0}, {}, {-1, num_experts - k, num_experts, 1});

    array sorted_indices = argsort(gates, -1);
    indices = slice(sorted_indices, {-k}, {});
    indices = reshape(indices, {gates.shape(0), gates.shape(1), k});

    array scores = take_along_axis(gates, indices, -1);

    if (normalize) {
        array score_sum = sum(scores, -1, true);
        scores = scores / score_sum;
    }

    return {indices, scores};
}

SwitchLinear::SwitchLinear(const array& weight,
                           const std::optional<array>& scales,
                           const std::optional<array>& biases,
                           const std::optional<array>& bias,
                           int group_size,
                           int bits)
    : weight_(weight)
    , scales_(scales)
    , biases_(biases)
    , bias_(bias)
    , group_size_(group_size)
    , bits_(bits) {

    bool is_square = (weight.shape(-1) == weight.shape(-2));

    if (!is_square && weight.shape(-1) < weight.shape(-2) && !scales.has_value()) {
         weight_ = transpose(weight, {0, 2, 1});
    }
}

array SwitchLinear::operator()(const array& x, const array& indices, bool sorted_indices) const {
    array result = gather_mm(x, weight_, indices);

    if (bias_.has_value()) {
        array selected_bias = take(*bias_, indices, 0);
        selected_bias = expand_dims(selected_bias, -2);
        result = result + selected_bias;
    }

    return result;
}

SwitchGLU::SwitchGLU(const SwitchLinear& gate_up_proj,
                     const SwitchLinear& down_proj)
    : gate_up_proj_(gate_up_proj)
    , down_proj_(down_proj) {
}

array SwitchGLU::operator()(const array& x, const array& indices) const {
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

    array gate_up = gate_up_proj_(sorted_x, idx, do_sort);

    auto chunks = split(gate_up, 2, -1);
    array activated = swiglu(chunks[0], chunks[1]);

    array result = down_proj_(activated, idx, do_sort);

    if (do_sort && inv_order_opt.has_value()) {
        result = scatter_unsort(result, *inv_order_opt, x_expanded.shape());
    }

    return squeeze(result, -2);
}

SparseMoEBlock::SparseMoEBlock(const array& router_weight,
                               const SwitchGLU& switch_mlp,
                               const MoEConfig& config)
    : router_weight_(router_weight)
    , switch_mlp_(switch_mlp)
    , config_(config) {
}

array SparseMoEBlock::operator()(const array& x) const {
    array router_logits = matmul(x, transpose(router_weight_, {1, 0}));

    auto [indices, scores] = top_k_routing(router_logits, config_.num_experts_per_tok,
                                            config_.norm_topk_prob);

    array expert_out = switch_mlp_(x, indices);

    array weights = expand_dims(scores, -1);
    array combined = sum(expert_out * weights, -2);

    return combined;
}

}  // namespace moe
}  // namespace ryzenai
