/*
 * mlx_backend.h
 * 
 * MLX (Metal/ROCm/CUDA) backend implementation.
 * Wraps the existing MLX inference pipeline with the IBackend interface.
 */

#pragma once

#include <ryzenai/backend/backend.h>
#include <ryzenai/mlx/model.h>
#include <ryzenai/mlx/tokenizer.h>
#include <ryzenai/mlx/generator.h>
#include <memory>
#include <string>

namespace ryzenai {

class MlxBackend : public IBackend {
public:
    explicit MlxBackend(BackendType type = BackendType::MLX_METAL);
    ~MlxBackend() override = default;
    
    // Lifecycle
    void loadModel(const std::string& model_path) override;
    void setContextSize(int ctx_size) override;
    void setKVFormat(KVCacheMode format) override;
    BackendType getType() const override { return type_; }
    std::string getName() const override;
    BackendCapabilities getCapabilities() const override;
    
    // Tokenization
    std::vector<int32_t> encode(const std::string& text) override;
    std::string decode(const std::vector<int32_t>& tokens) override;
    std::string applyChatTemplate(const std::string& messages_json, const std::string& tools_json = "") override;
    int countTokens(const std::string& text) override;
    
    // Generation
    std::string complete(const std::string& prompt, const GenerationParams& params, CompletionTimingData* out_timing = nullptr) override;
    void streamComplete(const std::string& prompt, const GenerationParams& params, StreamCallback callback) override;
    
    // Model Info
    std::string getModelName() const override;
    std::string getModelType() const override;
    int getMaxContextLength() const override;
    int getEosTokenId() const override;
    bool isEos(int32_t token_id) const override;
    const std::vector<AdditionalToken>& getSpecialTokens() const override;
    GenerationParams getDefaultParams() const override;
    
    // MLX-specific: access to underlying model/tokenizer
    MlxOgaModel* getModel() { return model_.get(); }
    const MlxOgaModel* getModel() const { return model_.get(); }
    MlxOgaTokenizer* getTokenizer() { return tokenizer_.get(); }
    
private:
    BackendType type_;
    std::string model_path_;
    std::unique_ptr<MlxOgaModel> model_;
    std::unique_ptr<MlxOgaTokenizer> tokenizer_;
    
    // Cached values
    std::string model_name_;
    GenerationParams default_params_;
    bool has_default_params_ = false;
    
    void loadDefaultParams();
    std::string resolveModelPath(const std::string& path);
};

// Factory registration helper
void registerMlxBackend();

} // namespace ryzenai
