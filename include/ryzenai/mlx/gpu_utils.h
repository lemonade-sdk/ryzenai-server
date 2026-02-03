/*
 * gpu_utils.h
 *
 * GPU discovery and utility functions for MLX backends.
 */

#pragma once

#include <ryzenai/backend/backend.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>

namespace ryzenai {
namespace mlx {

// GPU information structure
struct GpuInfo {
    int index;
    BackendType type;
    std::string name;
    std::unordered_map<std::string, std::variant<std::string, size_t>> properties;
};

// Device capabilities structure
struct DeviceCapabilities {
    size_t total_memory_mb = 0;
    size_t available_memory_mb = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    bool supports_fp16 = false;
    bool supports_int8 = false;
    bool supports_int4 = false;
    bool supports_matmul = false;
    bool supports_quantized_matmul = false;
    bool supports_fast_matmul = false;
    bool supports_rope = false;
    bool supports_sdpa = false;
    std::string architecture;
    int max_threads_per_block = 0;
    int multiprocessor_count = 0;
};

/**
 * @brief GPU utilities class for MLX backends
 *
 * Provides functions for discovering and querying GPU devices available to MLX.
 */
class GpuUtils {
public:
    /**
     * @brief Discover all available GPUs and their types
     * @return Vector of GpuInfo structures describing each GPU
     */
    static std::vector<GpuInfo> discoverAvailableGpus();

    /**
     * @brief Find the first GPU of the specified backend type
     * @param type The backend type to search for (MLX_ROCM, MLX_CUDA, etc.)
     * @return Device index of the first matching GPU, or -1 if not found
     */
    static int findFirstGpuOfType(BackendType type);

    /**
     * @brief Set the MLX default device for the specified backend type
     * @param type The backend type (MLX_METAL, MLX_ROCM, MLX_CUDA)
     * @return Device index if set successfully, -1 otherwise
     */
    static int setMlxDeviceForBackend(BackendType type);

    /**
     * @brief Get device capabilities for the current MLX device
     * @return DeviceCapabilities structure with current device capabilities
     */
    static DeviceCapabilities getDeviceCapabilities();

private:
    GpuUtils() = delete; // Static utility class
};

} // namespace mlx
} // namespace ryzenai
