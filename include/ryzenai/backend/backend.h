/*
 * backend.h
 * 
 * Unified backend interface for inference engines.
 * Supports ONNX Runtime GenAI, MLX (Metal/ROCm/CUDA).
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>

// Include types.h using relative path
#include "../types.h"

namespace ryzenai {

// Forward declarations
struct GenerationParams;
struct CompletionTimingData;

/*
 * BackendType - Enum for available backends
 */
enum class BackendType {
    AUTO,           // Auto-detect based on platform
    ONNX_RYZENAI,   // ONNX Runtime with RyzenAI NPU (Windows)
    ONNX_CPU,       // ONNX Runtime CPU (cross-platform)
    MLX_METAL,      // Apple MLX with Metal (macOS)
    MLX_ROCM,       // MLX with ROCm (AMD GPUs on Linux)
    MLX_CUDA        // MLX with CUDA (NVIDIA GPUs)
};

/*
 * BackendCapabilities - What a backend supports
 */
struct BackendCapabilities {
    bool supports_streaming = true;
    bool supports_quantization = true;
    bool supports_kv_cache = true;
    bool supports_tool_calls = true;
    bool supports_thinking = true;
    int max_context_length = 4096;
    std::vector<std::string> supported_model_types;
};

/*
 * IBackend - Abstract interface for inference backends
 */
class IBackend {
public:
    virtual ~IBackend() = default;
    
    // Lifecycle
    virtual void loadModel(const std::string& model_path) = 0;
    virtual void setContextSize(int ctx_size) { (void)ctx_size; }  // Optional override for context size
    virtual void setKVFormat(KVCacheMode format) { (void)format; }  // Optional override for KV cache format
    virtual BackendType getType() const = 0;
    virtual std::string getName() const = 0;
    virtual BackendCapabilities getCapabilities() const = 0;
    
    // Tokenization
    virtual std::vector<int32_t> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int32_t>& tokens) = 0;
    virtual std::string applyChatTemplate(const std::string& messages_json, const std::string& tools_json = "") = 0;
    virtual int countTokens(const std::string& text) = 0;
    
    // Generation
    virtual std::string complete(const std::string& prompt, const GenerationParams& params, CompletionTimingData* out_timing = nullptr) = 0;
    virtual void streamComplete(const std::string& prompt, const GenerationParams& params, StreamCallback callback) = 0;
    
    // Model Info
    virtual std::string getModelName() const = 0;
    virtual std::string getModelType() const = 0;
    virtual int getMaxContextLength() const = 0;
    virtual int getEosTokenId() const = 0;
    virtual bool isEos(int32_t token_id) const = 0;
    virtual const std::vector<AdditionalToken>& getSpecialTokens() const = 0;
    virtual GenerationParams getDefaultParams() const = 0;
};

/*
 * BackendRegistry - Factory for backend creation
 */
using BackendCreator = std::function<std::unique_ptr<IBackend>(const std::string& model_path)>;

class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry registry;
        return registry;
    }
    
    void registerBackend(BackendType type, BackendCreator creator) {
        creators_[type] = std::move(creator);
    }
    
    std::unique_ptr<IBackend> create(BackendType type, const std::string& model_path) {
        auto it = creators_.find(type);
        if (it == creators_.end()) {
            throw std::runtime_error("Backend not registered: " + backendTypeToString(type));
        }
        return it->second(model_path);
    }
    
    BackendType detectBestBackend() const {
#if defined(__APPLE__)
        return BackendType::MLX_METAL;
#elif defined(_WIN32)
        return BackendType::ONNX_RYZENAI;
#else
        return BackendType::ONNX_CPU;
#endif
    }
    
    bool isAvailable(BackendType type) const {
        return creators_.find(type) != creators_.end();
    }
    
    std::vector<BackendType> getAvailable() const {
        std::vector<BackendType> result;
        for (const auto& [type, _] : creators_) {
            result.push_back(type);
        }
        return result;
    }
    
    static std::string backendTypeToString(BackendType type) {
        switch (type) {
            case BackendType::AUTO: return "auto";
            case BackendType::ONNX_RYZENAI: return "onnx-ryzenai";
            case BackendType::ONNX_CPU: return "onnx-cpu";
            case BackendType::MLX_METAL: return "mlx-metal";
            case BackendType::MLX_ROCM: return "mlx-rocm";
            case BackendType::MLX_CUDA: return "mlx-cuda";
            default: return "unknown";
        }
    }
    
    static BackendType parseBackendType(const std::string& str) {
        if (str == "auto") return BackendType::AUTO;
        if (str == "onnx-ryzenai" || str == "ryzenai" || str == "npu") return BackendType::ONNX_RYZENAI;
        if (str == "onnx-cpu" || str == "cpu") return BackendType::ONNX_CPU;
        if (str == "mlx" || str == "metal" || str == "mlx-metal") return BackendType::MLX_METAL;
        if (str == "rocm" || str == "mlx-rocm") return BackendType::MLX_ROCM;
        if (str == "cuda" || str == "mlx-cuda") return BackendType::MLX_CUDA;
        return BackendType::AUTO;
    }
    
private:
    BackendRegistry() = default;
    std::unordered_map<BackendType, BackendCreator> creators_;
};

// Convenience functions
inline std::string backendTypeToString(BackendType type) {
    return BackendRegistry::backendTypeToString(type);
}

inline BackendType parseBackendType(const std::string& str) {
    return BackendRegistry::parseBackendType(str);
}

} // namespace ryzenai
