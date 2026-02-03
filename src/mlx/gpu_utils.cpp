/*
 * gpu_utils.cpp
 *
 * GPU discovery and utility functions implementation for MLX backends.
 */

#include <ryzenai/mlx/gpu_utils.h>
#include <mlx/device.h>
#include <iostream>

namespace ryzenai {
namespace mlx {

// Discover available GPUs and their types
std::vector<GpuInfo> GpuUtils::discoverAvailableGpus() {
    std::vector<GpuInfo> gpus;

    int device_count = ::mlx::core::device_count(::mlx::core::Device::DeviceType::gpu);
    for (int i = 0; i < device_count; ++i) {
        try {
            auto device_info = ::mlx::core::device_info(::mlx::core::Device(::mlx::core::Device::DeviceType::gpu, i));
            GpuInfo gpu;
            gpu.index = i;

            // Extract device name
            auto name_it = device_info.find("device_name");
            gpu.name = (name_it != device_info.end()) ?
                std::get<std::string>(name_it->second) : "Unknown GPU";

            // Determine GPU type based on device properties
            BackendType gpu_type = BackendType::AUTO; // Unknown initially

            // Check for CUDA/NVIDIA GPU
            auto arch_it = device_info.find("architecture");
            if (arch_it != device_info.end()) {
                std::string arch = std::get<std::string>(arch_it->second);
                if (arch.find("sm_") == 0) {
                    gpu_type = BackendType::MLX_CUDA;
                } else if (arch.find("gfx") == 0 || arch.find("rdna") == 0) {
                    gpu_type = BackendType::MLX_ROCM;
                }
            }

            // Check for compute capability (CUDA-specific)
            auto cc_it = device_info.find("compute_capability_major");
            if (cc_it != device_info.end()) {
                gpu_type = BackendType::MLX_CUDA;
            }

            // Check for GCN arch name (ROCm-specific)
            auto gcn_it = device_info.find("gcnArchName");
            if (gcn_it != device_info.end()) {
                gpu_type = BackendType::MLX_ROCM;
            }

            gpu.type = gpu_type;
            gpu.properties = std::move(device_info);
            gpus.push_back(std::move(gpu));

        } catch (const std::exception& e) {
            // Skip GPUs we can't query
            std::cerr << "[GpuUtils] Warning: Could not query GPU " << i << ": " << e.what() << std::endl;
            continue;
        }
    }

    return gpus;
}

// Find first GPU of specified type
int GpuUtils::findFirstGpuOfType(BackendType type) {
    auto gpus = discoverAvailableGpus();
    for (const auto& gpu : gpus) {
        if (gpu.type == type) {
            return gpu.index;
        }
    }
    return -1; // Not found
}

// Set the MLX default device for the specified backend type
bool GpuUtils::setMlxDeviceForBackend(BackendType type) {
    // Check if any GPUs are available
    int gpu_count = ::mlx::core::device_count(::mlx::core::Device::DeviceType::gpu);
    bool has_gpus = (gpu_count > 0);

    switch (type) {
        case BackendType::MLX_METAL:
            if (has_gpus) {
                std::cout << "[GpuUtils] Setting MLX default device to Metal GPU" << std::endl;
                ::mlx::core::set_default_device(::mlx::core::Device::gpu);
                return true;
            } else {
                std::cout << "[GpuUtils] No GPUs available for Metal backend" << std::endl;
                return false;
            }

        case BackendType::MLX_ROCM: {
            int gpu_index = findFirstGpuOfType(BackendType::MLX_ROCM);
            if (gpu_index >= 0) {
                std::cout << "[GpuUtils] Setting MLX default device to ROCm GPU (index " << gpu_index << ")" << std::endl;
                ::mlx::core::set_default_device(::mlx::core::Device(::mlx::core::Device::gpu, gpu_index));
                // Verify the device was actually set
                auto current_device = ::mlx::core::default_device();
                if (current_device.type == ::mlx::core::Device::DeviceType::gpu &&
                    current_device.index == gpu_index) {
                    std::cout << "[GpuUtils] Successfully set device to ROCm GPU " << gpu_index << std::endl;
                    return true;
                } else {
                    std::cout << "[GpuUtils] Failed to set device to GPU " << gpu_index
                              << ", current device type: " << (current_device.type == ::mlx::core::Device::DeviceType::cpu ? "cpu" : "gpu")
                              << ", index: " << current_device.index << std::endl;
                    return false;
                }
            } else {
                std::cout << "[GpuUtils] No ROCm-compatible GPU found" << std::endl;
                return false;
            }
        }

        case BackendType::MLX_CUDA: {
            int gpu_index = findFirstGpuOfType(BackendType::MLX_CUDA);
            if (gpu_index >= 0) {
                std::cout << "[GpuUtils] Setting MLX default device to CUDA GPU (index " << gpu_index << ")" << std::endl;
                ::mlx::core::set_default_device(::mlx::core::Device(::mlx::core::Device::gpu, gpu_index));
                return true;
            } else {
                std::cout << "[GpuUtils] No CUDA-compatible GPU found" << std::endl;
                return false;
            }
        }

    default:
        // Fallback to CPU for unknown types
        std::cout << "[GpuUtils] Setting MLX default device to CPU" << std::endl;
        ::mlx::core::set_default_device(::mlx::core::Device::cpu);
        return true;
    }
}

// Get device capabilities for the current MLX device
DeviceCapabilities GpuUtils::getDeviceCapabilities() {
    DeviceCapabilities caps;

    try {
        // Check if current device is CPU (indicates device setting failed)
        auto current_device = ::mlx::core::default_device();
        if (current_device.type == ::mlx::core::Device::DeviceType::cpu) {
            std::cerr << "[GpuUtils] Warning: Current device is CPU, device setting may have failed" << std::endl;
            caps.total_memory_mb = 0;  // Indicate not capable
            caps.available_memory_mb = 0;
            caps.supports_matmul = false;
            caps.supports_quantized_matmul = false;
            caps.supports_rope = false;
            caps.supports_sdpa = false;
            caps.supports_fp16 = false;
            caps.supports_int8 = false;
            caps.supports_int4 = false;
            caps.supports_fast_matmul = false;
            return caps;
        }

        // Get current device info
        auto device_info = ::mlx::core::device_info(current_device);

        // Extract memory information
        auto mem_it = device_info.find("max_buffer_length");
        if (mem_it != device_info.end()) {
            size_t max_buffer = std::get<size_t>(mem_it->second);
            caps.total_memory_mb = max_buffer / (1024 * 1024);
            // Available memory is approximately total minus some overhead
            caps.available_memory_mb = caps.total_memory_mb * 0.9;  // Conservative estimate
        }

        // Extract architecture
        auto arch_it = device_info.find("architecture");
        if (arch_it != device_info.end()) {
            caps.architecture = std::get<std::string>(arch_it->second);
        }

        // Extract compute capability for CUDA
        auto cc_major_it = device_info.find("compute_capability_major");
        if (cc_major_it != device_info.end()) {
            caps.compute_capability_major = static_cast<int>(std::get<size_t>(cc_major_it->second));
        }

        auto cc_minor_it = device_info.find("compute_capability_minor");
        if (cc_minor_it != device_info.end()) {
            caps.compute_capability_minor = static_cast<int>(std::get<size_t>(cc_minor_it->second));
        }

        // Extract multiprocessor count
        auto mp_it = device_info.find("multiprocessor_count");
        if (mp_it != device_info.end()) {
            caps.multiprocessor_count = static_cast<int>(std::get<size_t>(mp_it->second));
        }

        // Extract max threads per block
        auto mtpb_it = device_info.find("max_threads_per_block");
        if (mtpb_it != device_info.end()) {
            caps.max_threads_per_block = static_cast<int>(std::get<size_t>(mtpb_it->second));
        }

        // Determine supported operations based on architecture and capabilities
        caps.supports_matmul = true;  // Basic matmul is always supported
        caps.supports_quantized_matmul = true;  // MLX supports quantized operations
        caps.supports_rope = true;  // RoPE is supported in MLX
        caps.supports_sdpa = true;  // Scaled dot product attention is supported

        // Determine precision support
        if (!caps.architecture.empty()) {
            std::string arch_lower = caps.architecture;
            std::transform(arch_lower.begin(), arch_lower.end(), arch_lower.begin(), ::tolower);

            if (arch_lower.find("sm_") == 0) {
                // CUDA GPU
                caps.supports_fp16 = (caps.compute_capability_major >= 6);
                caps.supports_int8 = (caps.compute_capability_major >= 6);
                caps.supports_int4 = (caps.compute_capability_major >= 7);  // Tensor cores
                caps.supports_fast_matmul = (caps.compute_capability_major >= 7);
            } else if (arch_lower.find("gfx") == 0 || arch_lower.find("rdna") == 0) {
                // ROCm GPU
                caps.supports_fp16 = true;
                caps.supports_int8 = true;
                caps.supports_int4 = true;
                caps.supports_fast_matmul = true;
            } else if (arch_lower.find("apple") != std::string::npos || caps.architecture.empty()) {
                // Apple Silicon or unknown
                caps.supports_fp16 = true;
                caps.supports_int8 = true;
                caps.supports_int4 = true;
                caps.supports_fast_matmul = true;
            }
        } else {
            // Default assumptions for unknown architectures
            caps.supports_fp16 = true;
            caps.supports_int8 = true;
            caps.supports_int4 = true;
            caps.supports_fast_matmul = true;
        }

    } catch (const std::exception& e) {
        std::cerr << "[GpuUtils] Error getting device capabilities: " << e.what() << std::endl;
        // Return default capabilities
        caps.supports_matmul = true;
        caps.supports_quantized_matmul = true;
        caps.supports_rope = true;
        caps.supports_sdpa = true;
        caps.supports_fp16 = true;
        caps.supports_int8 = true;
        caps.supports_int4 = true;
        caps.supports_fast_matmul = true;
    }

    return caps;
}

} // namespace mlx
} // namespace ryzenai
