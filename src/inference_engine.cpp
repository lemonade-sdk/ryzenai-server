/*
 * inference_engine.cpp
 * 
 * Multi-backend inference engine implementation.
 * Routes requests to appropriate backends based on model name.
 */

#include "ryzenai/inference_engine.h"
#include "ryzenai/backend/backend.h"

#ifdef MLX_ON
#include "ryzenai/backend/mlx_backend.h"
#include "ryzenai/mlx/gpu_utils.h"
#include <mlx/backend/gpu/available.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/rocm/rocm.h>
#endif

#ifdef RYZENAI_ON
#include "ryzenai/backend/onnx_backend.h"
#endif

#include <filesystem>
#include <iostream>
#include <cctype>
#include <json.hpp>

namespace ryzenai {

namespace fs = std::filesystem;
using json = nlohmann::json;

// Static member definitions
const std::vector<AdditionalToken> InferenceEngine::empty_tokens_;
const GenerationParams InferenceEngine::default_params_;

/**
 * @brief Constructs the InferenceEngine with specified optimization settings.
 *
 * Initializes the multi-backend inference engine, registers available backends
 * based on compilation flags, and configures optimization parameters for model inference.
 *
 * @param opt Optimization settings including context size, KV cache, and repetition penalty parameters.
 */
InferenceEngine::InferenceEngine(const OptimizationSettings& opt)
    : opt_settings_(opt),
      has_metal_available_(false),
      has_rocm_gpu_available_(false),
      has_cuda_gpu_available_(false) {
    
    std::cout << "[InferenceEngine] Initializing multi-backend engine" << std::endl;
    std::cout << "[InferenceEngine] Optimization settings:" << std::endl;
    std::cout << "  - Context size: " << opt.ctx_size << " tokens" << std::endl;
    std::cout << "  - Repetition lookback: " << opt.repetition_lookback << " tokens" << std::endl;
    std::cout << "  - KV cache: " << (opt.kv_cache ? "enabled" : "disabled") << std::endl;
    std::cout << "  - Prefill chunk: " << opt.prefill_chunk << " tokens" << std::endl;
    
    // Register available backends (only if hardware is available)
#ifdef MLX_ON
    // Discover available GPUs using GpuUtils
    auto available_gpus = ryzenai::mlx::GpuUtils::discoverAvailableGpus();

    // Check which backend types are supported by available hardware
    has_metal_available_ = ::mlx::core::metal::is_available();
    has_rocm_gpu_available_ = ryzenai::mlx::GpuUtils::findFirstGpuOfType(BackendType::MLX_ROCM) >= 0;
    has_cuda_gpu_available_ = ryzenai::mlx::GpuUtils::findFirstGpuOfType(BackendType::MLX_CUDA) >= 0;

    int registered_mlx_backends = 0;

    // Register Metal backend if available
    if (has_metal_available_) {
        BackendRegistry::instance().registerBackend(
            BackendType::MLX_METAL,
            [](const std::string& model_path) -> std::unique_ptr<IBackend> {
                auto backend = std::make_unique<MlxBackend>(BackendType::MLX_METAL);
                backend->loadModel(model_path);
                return backend;
            }
        );
        std::cout << "[InferenceEngine] Registered MLX Metal backend" << std::endl;
        registered_mlx_backends++;
    }

    // Register ROCm backend if ROCm GPUs are available
    if (has_rocm_gpu_available_) {
        std::cout << "[InferenceEngine] Available ROCm GPUs:" << std::endl;
        for (const auto& gpu : available_gpus) {
            if (gpu.type == BackendType::MLX_ROCM) {
                std::cout << "  - GPU " << gpu.index << ": " << gpu.name;
                auto arch_it = gpu.properties.find("architecture");
                if (arch_it != gpu.properties.end()) {
                    std::cout << " (" << std::get<std::string>(arch_it->second) << ")";
                }
                std::cout << std::endl;
            }
        }

        BackendRegistry::instance().registerBackend(
            BackendType::MLX_ROCM,
            [](const std::string& model_path) -> std::unique_ptr<IBackend> {
                auto backend = std::make_unique<MlxBackend>(BackendType::MLX_ROCM);
                backend->loadModel(model_path);
                return backend;
            }
        );
        std::cout << "[InferenceEngine] Registered MLX ROCm backend" << std::endl;
        registered_mlx_backends++;
    }

    // Register CUDA backend if CUDA GPUs are available
    if (has_cuda_gpu_available_) {
        std::cout << "[InferenceEngine] Available CUDA GPUs:" << std::endl;
        for (const auto& gpu : available_gpus) {
            if (gpu.type == BackendType::MLX_CUDA) {
                std::cout << "  - GPU " << gpu.index << ": " << gpu.name;
                auto arch_it = gpu.properties.find("architecture");
                if (arch_it != gpu.properties.end()) {
                    std::cout << " (" << std::get<std::string>(arch_it->second) << ")";
                }
                auto cc_it = gpu.properties.find("compute_capability_major");
                if (cc_it != gpu.properties.end()) {
                    auto cc_minor_it = gpu.properties.find("compute_capability_minor");
                    if (cc_minor_it != gpu.properties.end()) {
                        std::cout << " (CC " << std::get<size_t>(cc_it->second)
                                  << "." << std::get<size_t>(cc_minor_it->second) << ")";
                    }
                }
                std::cout << std::endl;
            }
        }

        BackendRegistry::instance().registerBackend(
            BackendType::MLX_CUDA,
            [](const std::string& model_path) -> std::unique_ptr<IBackend> {
                auto backend = std::make_unique<MlxBackend>(BackendType::MLX_CUDA);
                backend->loadModel(model_path);
                return backend;
            }
        );
        std::cout << "[InferenceEngine] Registered MLX CUDA backend" << std::endl;
        registered_mlx_backends++;
    }

    if (registered_mlx_backends == 0) {
        std::cout << "[InferenceEngine] No MLX backends available (no compatible hardware found)" << std::endl;
    } else {
        std::cout << "[InferenceEngine] Registered " << registered_mlx_backends << " MLX backend(s)" << std::endl;
    }
#endif

#ifdef RYZENAI_ON
    BackendRegistry::instance().registerBackend(
        BackendType::ONNX_RYZENAI,
        [this](const std::string& model_path) -> std::unique_ptr<IBackend> {
            auto backend = std::make_unique<OnnxBackend>(BackendType::ONNX_RYZENAI);
            backend->loadModel(model_path);
            return backend;
        }
    );
    BackendRegistry::instance().registerBackend(
        BackendType::ONNX_CPU,
        [this](const std::string& model_path) -> std::unique_ptr<IBackend> {
            auto backend = std::make_unique<OnnxBackend>(BackendType::ONNX_CPU);
            backend->loadModel(model_path);
            return backend;
        }
    );
    std::cout << "[InferenceEngine] Registered ONNX backends" << std::endl;
#endif
}

/**
 * @brief Destructor for the InferenceEngine.
 *
 * Cleans up all loaded models and clears the model index.
 */
InferenceEngine::~InferenceEngine() {
    std::lock_guard<std::mutex> lock(models_mutex_);
    std::cout << "[InferenceEngine] Shutting down, unloading " << loaded_models_.size() << " models" << std::endl;
    loaded_models_.clear();
    model_index_.clear();
}

/**
 * @brief Normalizes a model name by converting to lowercase and keeping only alphanumeric characters.
 *
 * This function creates a standardized version of the model name for indexing and lookup.
 *
 * @param name The original model name.
 * @return std::string The normalized model name.
 */
std::string InferenceEngine::normalizeModelName(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            result += std::tolower(static_cast<unsigned char>(c));
        }
    }
    
    return result;
}

std::string InferenceEngine::extractModelName(const std::string& model_path) {
    // Use the directory name as the model identifier
    // This ensures each model has a unique name even if they have the same model_type
    std::string dir_name = fs::path(model_path).filename().string();
    std::cout << "[InferenceEngine] Model name from directory: " << dir_name << std::endl;
    return dir_name;
}

LoadedModel* InferenceEngine::findModel(const std::string& model_name) {
    std::string normalized = normalizeModelName(model_name);
    
    auto it = model_index_.find(normalized);
    if (it != model_index_.end() && it->second < loaded_models_.size()) {
        return &loaded_models_[it->second];
    }
    
    // Also try partial match
    for (auto& model : loaded_models_) {
        std::string model_normalized = normalizeModelName(model.model_name);
        if (model_normalized.find(normalized) != std::string::npos ||
            normalized.find(model_normalized) != std::string::npos) {
            return &model;
        }
    }
    
    return nullptr;
}

const LoadedModel* InferenceEngine::findModel(const std::string& model_name) const {
    std::string normalized = normalizeModelName(model_name);
    
    auto it = model_index_.find(normalized);
    if (it != model_index_.end() && it->second < loaded_models_.size()) {
        return &loaded_models_[it->second];
    }
    
    // Also try partial match
    for (const auto& model : loaded_models_) {
        std::string model_normalized = normalizeModelName(model.model_name);
        if (model_normalized.find(normalized) != std::string::npos ||
            normalized.find(model_normalized) != std::string::npos) {
            return &model;
        }
    }
    
    return nullptr;
}

// ==================== Model Management ====================

std::string InferenceEngine::loadModel(const std::string& model_path, BackendType backend_type) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    // Resolve path (handle HuggingFace cache structure)
    std::string resolved_path = model_path;
    std::string snapshots_dir = model_path + "/snapshots";
    if (fs::exists(snapshots_dir) && fs::is_directory(snapshots_dir)) {
        std::cout << "[InferenceEngine] Detected Hugging Face cache structure" << std::endl;
        for (const auto& entry : fs::directory_iterator(snapshots_dir)) {
            if (entry.is_directory()) {
                resolved_path = entry.path().string();
                std::cout << "[InferenceEngine] Resolved to snapshot: " << resolved_path << std::endl;
                break;
            }
        }
    }
    
    // Validate model directory
    if (!fs::exists(resolved_path) || !fs::is_directory(resolved_path)) {
        throw std::runtime_error("Model path does not exist: " + resolved_path);
    }
    
    // Extract model name
    std::string model_name = extractModelName(resolved_path);
    std::string normalized = normalizeModelName(model_name);
    
    // Check if already loaded
    if (model_index_.find(normalized) != model_index_.end()) {
        std::cout << "[InferenceEngine] Model already loaded: " << model_name << std::endl;
        return model_name;
    }
    
    // Auto-detect backend if needed
    BackendType actual_type = backend_type;
    if (actual_type == BackendType::AUTO) {
        actual_type = BackendRegistry::instance().detectBestBackend();
        std::cout << "[InferenceEngine] Auto-detected backend: "
                  << backendTypeToString(actual_type) << std::endl;
    }

    // Check device availability based on backend type (using cached availability)
    switch (actual_type) {
        case BackendType::MLX_METAL:
            if (!has_metal_available_) {
                throw std::runtime_error("Metal device not available. Metal backend requires macOS with Apple Silicon or compatible hardware.");
            }
            std::cout << "[InferenceEngine] Metal device is available" << std::endl;
            break;

        case BackendType::MLX_ROCM:
            if (!has_rocm_gpu_available_) {
                throw std::runtime_error("ROCm-compatible GPU not available. Please check that ROCm drivers are properly installed and a compatible AMD GPU is present.");
            }
            std::cout << "[InferenceEngine] ROCm GPU is available" << std::endl;
            break;

        case BackendType::MLX_CUDA:
            if (!has_cuda_gpu_available_) {
                throw std::runtime_error("CUDA-compatible GPU not available. Please check that CUDA drivers are properly installed and a compatible NVIDIA GPU is present.");
            }
            std::cout << "[InferenceEngine] CUDA GPU is available" << std::endl;
            break;

        case BackendType::ONNX_CPU:
            // CPU is always available
            std::cout << "[InferenceEngine] CPU backend selected (always available)" << std::endl;
            break;

        case BackendType::ONNX_RYZENAI:
            // TODO: Add Ryzen AI NPU availability check
            std::cout << "[InferenceEngine] Ryzen AI NPU backend selected" << std::endl;
            break;

        case BackendType::ONNX_DIRECTML:
            // TODO: Add DirectML availability check
            std::cout << "[InferenceEngine] DirectML backend selected" << std::endl;
            break;

        default:
            // For AUTO and unknown types, assume available
            break;
    }

    // Create backend and load model
    std::cout << "[InferenceEngine] Loading model: " << model_name
              << " on backend: " << backendTypeToString(actual_type) << std::endl;

    auto backend = BackendRegistry::instance().create(actual_type, resolved_path);
    
    // Apply context size from optimization settings
    backend->setContextSize(opt_settings_.ctx_size);
    backend->setKVFormat(opt_settings_.kv_format);

    // Create LoadedModel entry
    LoadedModel loaded;
    loaded.backend = std::move(backend);
    loaded.model_name = model_name;
    loaded.model_path = resolved_path;
    loaded.backend_type = actual_type;
    
    // Add to vector and index
    size_t index = loaded_models_.size();
    loaded_models_.push_back(std::move(loaded));
    model_index_[normalized] = index;
    
    std::cout << "[InferenceEngine] Successfully loaded model: " << model_name 
              << " (total loaded: " << loaded_models_.size() << ")" << std::endl;
    
    return model_name;
}

void InferenceEngine::unloadModel(const std::string& model_name) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    std::string normalized = normalizeModelName(model_name);
    auto it = model_index_.find(normalized);
    
    if (it == model_index_.end()) {
        std::cerr << "[WARNING] Cannot unload - model not found: " << model_name << std::endl;
        return;
    }
    
    size_t index = it->second;
    std::cout << "[InferenceEngine] Unloading model: " << loaded_models_[index].model_name << std::endl;
    
    // Remove from vector
    loaded_models_.erase(loaded_models_.begin() + index);
    
    // Rebuild index
    model_index_.clear();
    for (size_t i = 0; i < loaded_models_.size(); ++i) {
        std::string norm = normalizeModelName(loaded_models_[i].model_name);
        model_index_[norm] = i;
    }
    
    std::cout << "[InferenceEngine] Model unloaded (remaining: " << loaded_models_.size() << ")" << std::endl;
}

std::vector<std::string> InferenceEngine::getLoadedModels() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    std::vector<std::string> names;
    names.reserve(loaded_models_.size());
    
    for (const auto& model : loaded_models_) {
        names.push_back(model.model_name);
    }
    
    return names;
}

IBackend* InferenceEngine::getBackendForModel(const std::string& model_name) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    LoadedModel* model = findModel(model_name);
    if (!model) {
        throw std::runtime_error("Model not loaded: " + model_name);
    }
    
    return model->backend.get();
}

const IBackend* InferenceEngine::getBackendForModel(const std::string& model_name) const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    const LoadedModel* model = findModel(model_name);
    if (!model) {
        throw std::runtime_error("Model not loaded: " + model_name);
    }
    
    return model->backend.get();
}

IBackend* InferenceEngine::getDefaultBackend() {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (loaded_models_.empty()) {
        return nullptr;
    }
    
    return loaded_models_[0].backend.get();
}

// ==================== Model-Specific Inference ====================

std::string InferenceEngine::complete(const std::string& model_name,
                                       const std::string& prompt,
                                       const GenerationParams& params,
                                       CompletionTimingData* out_timing) {
    IBackend* backend = getBackendForModel(model_name);
    return backend->complete(prompt, params, out_timing);
}

void InferenceEngine::streamComplete(const std::string& model_name,
                                      const std::string& prompt,
                                      const GenerationParams& params,
                                      StreamCallback callback) {
    IBackend* backend = getBackendForModel(model_name);
    backend->streamComplete(prompt, params, callback);
}

std::string InferenceEngine::applyChatTemplate(const std::string& model_name,
                                                const std::string& messages_json,
                                                const std::string& tools_json) {
    IBackend* backend = getBackendForModel(model_name);
    return backend->applyChatTemplate(messages_json, tools_json);
}

int InferenceEngine::countTokens(const std::string& model_name, const std::string& text) {
    IBackend* backend = getBackendForModel(model_name);
    return backend->countTokens(text);
}

// ==================== Model Info ====================

std::string InferenceEngine::getModelType(const std::string& model_name) const {
    const IBackend* backend = getBackendForModel(model_name);
    return backend->getModelType();
}

int InferenceEngine::getMaxContextLength(const std::string& model_name) const {
    const IBackend* backend = getBackendForModel(model_name);
    return backend->getMaxContextLength();
}

GenerationParams InferenceEngine::getDefaultParams(const std::string& model_name) const {
    const IBackend* backend = getBackendForModel(model_name);
    return backend->getDefaultParams();
}

const std::vector<AdditionalToken>& InferenceEngine::getSpecialTokens(const std::string& model_name) const {
    const IBackend* backend = getBackendForModel(model_name);
    return backend->getSpecialTokens();
}

// ==================== Backward Compatibility ====================

std::string InferenceEngine::complete(const std::string& prompt,
                                       const GenerationParams& params,
                                       CompletionTimingData* out_timing) {
    IBackend* backend = getDefaultBackend();
    if (!backend) {
        throw std::runtime_error("No models loaded");
    }
    return backend->complete(prompt, params, out_timing);
}

void InferenceEngine::streamComplete(const std::string& prompt,
                                      const GenerationParams& params,
                                      StreamCallback callback) {
    IBackend* backend = getDefaultBackend();
    if (!backend) {
        throw std::runtime_error("No models loaded");
    }
    backend->streamComplete(prompt, params, callback);
}

std::string InferenceEngine::applyChatTemplate(const std::string& messages_json,
                                                const std::string& tools_json) {
    IBackend* backend = getDefaultBackend();
    if (!backend) {
        throw std::runtime_error("No models loaded");
    }
    return backend->applyChatTemplate(messages_json, tools_json);
}

int InferenceEngine::countTokens(const std::string& text) {
    IBackend* backend = getDefaultBackend();
    if (!backend) {
        return 0;
    }
    return backend->countTokens(text);
}

std::string InferenceEngine::getModelName() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (loaded_models_.empty()) {
        return "";
    }
    return loaded_models_[0].model_name;
}

GenerationParams InferenceEngine::getDefaultParams() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (loaded_models_.empty()) {
        return default_params_;
    }
    return loaded_models_[0].backend->getDefaultParams();
}

const std::vector<AdditionalToken>& InferenceEngine::getAdditionalTags() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (loaded_models_.empty()) {
        return empty_tokens_;
    }
    return loaded_models_[0].backend->getSpecialTokens();
}

} // namespace ryzenai
