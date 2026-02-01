/*
 * onnx_backend.h
 * 
 * ONNX Runtime GenAI backend implementation for Ryzen AI.
 * Wraps ONNX Runtime GenAI with the IBackend interface.
 * Only compiled when RYZENAI_ON is defined (Windows with Ryzen AI).
 */

#pragma once

#ifdef RYZENAI_ON

#include <ryzenai/backend/backend.h>
#include <ort_genai.h>
#include <memory>
#include <string>
#include <vector>

namespace ryzenai {

class OnnxBackend : public IBackend {
public:
    explicit OnnxBackend(BackendType type = BackendType::ONNX_RYZENAI);
    ~OnnxBackend() override = default;
    
    // Lifecycle
    void loadModel(const std::string& model_path) override;
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
    
    // ONNX-specific: access to underlying model/tokenizer
    OgaModel* getModel() { return model_.get(); }
    const OgaModel* getModel() const { return model_.get(); }
    OgaTokenizer* getTokenizer() { return tokenizer_.get(); }
    
private:
    BackendType type_;
    std::string model_path_;
    
    // ONNX Runtime GenAI types (using smart pointers with custom deleters)
    std::unique_ptr<OgaModel, decltype(&OgaDestroyModel)> model_;
    std::unique_ptr<OgaTokenizer, decltype(&OgaDestroyTokenizer)> tokenizer_;
    
    // Cached values
    std::string model_name_;
    std::string model_type_;
    std::string chat_template_;
    int max_context_length_ = 4096;
    int eos_token_id_ = 2;
    std::vector<int> eos_token_ids_;  // Additional EOS tokens (for models with multiple)
    std::vector<AdditionalToken> special_tokens_;
    GenerationParams default_params_;
    bool has_default_params_ = false;
    
    void loadConfig();
    void loadDefaultParams();
    void loadSpecialTokens();
    std::string resolveModelPath(const std::string& path);
    std::vector<int32_t> truncatePrompt(const std::vector<int32_t>& input_ids, int max_length);
};

// Factory registration helper
void registerOnnxBackend();

} // namespace ryzenai

#endif // RYZENAI_ON
