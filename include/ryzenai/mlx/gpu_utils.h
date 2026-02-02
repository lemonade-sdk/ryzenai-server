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
     * @return true if device was set successfully, false otherwise
     */
    static bool setMlxDeviceForBackend(BackendType type);

private:
    GpuUtils() = delete; // Static utility class
};

} // namespace mlx
} // namespace ryzenai
