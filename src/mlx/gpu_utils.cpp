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
    switch (type) {
        case BackendType::MLX_METAL:
            std::cout << "[GpuUtils] Setting MLX default device to Metal GPU" << std::endl;
            ::mlx::core::set_default_device(::mlx::core::Device::gpu);
            return true;

        case BackendType::MLX_ROCM: {
            int gpu_index = findFirstGpuOfType(BackendType::MLX_ROCM);
            if (gpu_index >= 0) {
                std::cout << "[GpuUtils] Setting MLX default device to ROCm GPU (index " << gpu_index << ")" << std::endl;
                ::mlx::core::set_default_device(::mlx::core::Device(::mlx::core::Device::gpu, gpu_index));
                return true;
            } else {
                std::cout << "[GpuUtils] No ROCm-compatible GPU found, using default GPU device" << std::endl;
                ::mlx::core::set_default_device(::mlx::core::Device::gpu);
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
                std::cout << "[GpuUtils] No CUDA-compatible GPU found, using default GPU device" << std::endl;
                ::mlx::core::set_default_device(::mlx::core::Device::gpu);
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

} // namespace mlx
} // namespace ryzenai
