/*
 * moe.h
 * * Generic Mixture of Experts (MoE) components for MLX backend.
 * Optimized for GPU usage:
 * - Fused Gate/Up projections to reduce kernel launches.
 * - Pre-transposed weights for direct gather_mm usage.
 * - Shape-aware helper functions to avoid CPU-side vector conversions.
 */

#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <vector>
#include <tuple>

namespace ryzenai {
namespace moe {

using namespace ::mlx::core;

struct MoEConfig {
    int num_experts = 8;
    int num_experts_per_tok = 2;
    bool norm_topk_prob = true;
    
    // Sparse step (frequency of MoE layers)
    int decoder_sparse_step = 1;
    std::vector<int> mlp_only_layers; 
    
    // Quantization
    int router_group_size = 64;
    int router_bits = 8;
    
    int moe_intermediate_size = 0; 
};

/*
 * SwitchLinear
 * * Optimized Linear layer for Expert routing.
 * * OPTIMIZATION NOTE:
 * Weights are stored internally as [num_experts, input_dims, output_dims].
 * If passed as [num_experts, output, input], they are transposed in the constructor.
 * This avoids doing a transpose operation during every forward pass.
 */
class SwitchLinear {
public:
    /*
     * @param weight Expert weights. 
     * Preferred shape: [num_experts, in_dim, out_dim] (No transpose needed)
     * Legacy shape:    [num_experts, out_dim, in_dim] (Will be transposed)
     */
    SwitchLinear(const array& weight,
                 const std::optional<array>& scales = std::nullopt,
                 const std::optional<array>& biases = std::nullopt,
                 const std::optional<array>& bias = std::nullopt,
                 int group_size = 64,
                 int bits = 4);
    
    /*
     * Forward pass
     * @param x Input tensor [..., seq_len, hidden_dim]
     * @param indices Expert indices [..., seq_len, top_k]
     * @return Output tensor [..., seq_len, top_k, output_dim]
     */
    array operator()(const array& x, const array& indices, bool sorted_indices = false) const;
    
    bool is_quantized() const { return scales_.has_value(); }
    
    // Getters reflect the operational shape [experts, in, out]
    int num_experts() const { return weight_.shape(0); }
    int input_dims() const { return weight_.shape(1); } 
    int output_dims() const { return weight_.shape(2); }

private:
    array weight_;                    // Stored as [num_experts, in_dim, out_dim]
    std::optional<array> scales_;     
    std::optional<array> biases_;     
    std::optional<array> bias_;       
    int group_size_;
    int bits_;
};

/*
 * SwitchGLU
 * * Optimized SwiGLU with fused Gate+Up projection.
 * Instead of: down(swiglu(gate(x), up(x)))
 * We do:      down(swiglu(split(gate_up(x))))
 * * This cuts the number of heavy gather_mm kernels from 3 to 2.
 */
class SwitchGLU {
public:
    SwitchGLU() = default;
    
    /*
     * Construct with FUSED gate_up and down projection.
     */
    SwitchGLU(const SwitchLinear& gate_up_proj,
              const SwitchLinear& down_proj);
              
    array operator()(const array& x, const array& indices) const;

private:
    SwitchLinear gate_up_proj_; // Fused [experts, in, hidden*2]
    SwitchLinear down_proj_;    // [experts, hidden, out]
};

/*
 * SparseMoEBlock
 * * Generic sparse MoE block.
 * Optimized to minimize CPU synchronization.
 */
class SparseMoEBlock {
public:
    SparseMoEBlock() = default;
    
    SparseMoEBlock(const array& router_weight,
                   const SwitchGLU& switch_mlp,
                   const MoEConfig& config);
    
    array operator()(const array& x) const;
    
    int num_experts() const { return config_.num_experts; }
    int top_k() const { return config_.num_experts_per_tok; }

private:
    array router_weight_;
    SwitchGLU switch_mlp_;
    MoEConfig config_;
    
    std::optional<array> router_scales_;
    std::optional<array> router_biases_;
};


/*
 * Helper functions
 */

std::tuple<array, array, array> gather_sort(const array& x, const array& indices);

// OPTIMIZATION: Uses const Shape& to avoid std::vector conversion overhead
array scatter_unsort(const array& x, const array& inv_order, 
                     const Shape& original_shape);

std::pair<array, array> top_k_routing(const array& router_output, 
                                       int k, 
                                       bool normalize = true);

array swiglu(const array& gate, const array& up);

}  // namespace moe
}  // namespace ryzenai