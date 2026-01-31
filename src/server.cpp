#include "ryzenai/server.h"
#include "ryzenai/tool_calls.h"
#include "ryzenai/reasoning.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <thread>

namespace ryzenai {

/**
 * @brief Constructs the Ryzen AI Server with the provided command line arguments.
 *
 * Initializes the server by loading models, setting up the HTTP server with thread pool,
 * and configuring all API routes.
 *
 * @param args Command line arguments containing model paths, server settings, and backend configurations.
 */
RyzenAIServer::RyzenAIServer(const CommandLineArgs& args)
    : args_(args) {
    
    std::cout << "\n";
    std::cout << "===============================================================\n";
    std::cout << "            Ryzen AI LLM Server                                \n";
    std::cout << "            OpenAI API Compatible                              \n";
    std::cout << "===============================================================\n";
    std::cout << "\n";
    
    // Load the model
    loadModel();
    
    // Create HTTP server
    http_server_ = std::make_unique<httplib::Server>();
    
    // Enable multi-threading for better request handling performance
    http_server_->new_task_queue = [] { 
        std::cout << "[Server] Creating thread pool with 8 threads" << std::endl;
        return new httplib::ThreadPool(8);
    };
    
    std::cout << "[Server] HTTP server initialized with thread pool (8 threads)" << std::endl;
    
    // Setup routes
    setupRoutes();
    
    std::cout << "[Server] Initialization complete\n" << std::endl;
}

/**
 * @brief Destructor for the Ryzen AI Server.
 *
 * Stops the server if it's running.
 */
RyzenAIServer::~RyzenAIServer() {
    stop();
}

/**
 * @brief Creates generation parameters from user-provided values and defaults.
 *
 * This function combines user-specified generation parameters with default values
 * from the inference engine configuration to create a complete GenerationParams object.
 *
 * @param max_tokens Maximum number of tokens to generate.
 * @param temperature Sampling temperature (higher = more random).
 * @param top_p Nucleus sampling parameter.
 * @param top_k Top-k sampling parameter.
 * @param repeat_penalty Penalty for repeating tokens.
 * @param stop List of stop sequences.
 * @return GenerationParams Configured parameters for text generation.
 */
GenerationParams RyzenAIServer::createGenerationParams(int max_tokens, float temperature, float top_p,
                                                       int top_k, float repeat_penalty,
                                                       const std::vector<std::string>& stop) const {
    // Start with defaults from genai_config.json (or hardcoded defaults if no config)
    GenerationParams params = inference_engine_->getDefaultParams();
    
    std::cout << "[createGenerationParams] Input max_tokens=" << max_tokens << std::endl;
    std::cout << "[createGenerationParams] Default params.max_length=" << params.max_length << std::endl;
    
    // Always apply user-provided values, regardless of whether they match defaults
    // The request parsing already handles providing defaults when values aren't specified
    params.max_length = max_tokens;
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.repetition_penalty = repeat_penalty;
    params.stop_sequences = stop;
    
    std::cout << "[createGenerationParams] Final params: max_length=" << params.max_length
              << ", temperature=" << params.temperature 
              << ", top_p=" << params.top_p 
              << ", top_k=" << params.top_k 
              << ", do_sample=" << params.do_sample
              << ", repetition_penalty=" << params.repetition_penalty << std::endl;
    
    return params;
}

/**
 * @brief Loads all models specified in the command line arguments.
 *
 * Creates an inference engine with optimization settings and loads each model
 * with its specified backend. Prints a summary of loaded models and their backends.
 *
 * @throws std::runtime_error If model loading fails.
 */
void RyzenAIServer::loadModel() {
    std::cout << "[Server] Loading models..." << std::endl;
    std::cout << "[Server] Number of models to load: " << args_.models.size() << std::endl;
    
    try {
        // Create optimization settings from command line args
        OptimizationSettings opt;
        opt.ctx_size = args_.ctx_size;
        opt.repetition_lookback = args_.repetition_lookback;
        opt.kv_cache = args_.kv_cache;
        opt.prefill_chunk = args_.prefill_chunk;
        
        // Create inference engine with optimization settings
        inference_engine_ = std::make_unique<InferenceEngine>(opt);
        
        // Load all models from command line
        for (const auto& model_config : args_.models) {
            std::cout << "\n[Server] Loading: " << model_config.path 
                      << " (backend: " << model_config.backend << ")" << std::endl;
            
            // Determine backend type
            BackendType backend_type = parseBackendType(model_config.backend);
            
            // Load the model
            std::string loaded_name = inference_engine_->loadModel(model_config.path, backend_type);
            
            // Use first model as primary ID for backward compatibility
            if (model_id_.empty()) {
                model_id_ = loaded_name;
            }
            
            std::cout << "[Server] [OK] Loaded: " << loaded_name << std::endl;
        }
        
        // Print summary
        auto loaded_models = inference_engine_->getLoadedModels();
        std::cout << "\n[Server] ========== Models Summary ==========" << std::endl;
        for (const auto& name : loaded_models) {
            try {
                const auto* backend = inference_engine_->getBackendForModel(name);
                std::cout << "[Server]   - " << name << " (" << backend->getName() 
                          << ", ctx: " << backend->getMaxContextLength() << ")" << std::endl;
            } catch (...) {
                std::cout << "[Server]   - " << name << std::endl;
            }
        }
        std::cout << "[Server] =========================================\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Failed to load model: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief Extracts the model name from a file path.
 *
 * This function takes a full path to a model file and returns just the filename
 * (the part after the last directory separator).
 *
 * @param model_path The full path to the model file.
 * @return std::string The extracted model name (filename).
 */
std::string RyzenAIServer::extractModelName(const std::string& model_path) {
    // Extract the last component of the path
    size_t last_slash = model_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return model_path.substr(last_slash + 1);
    }
    return model_path;
}

/**
 * @brief Sets up HTTP routes for the server API endpoints.
 *
 * This function configures all the REST API routes including CORS headers,
 * health check, completions, chat completions, responses, and models endpoints.
 */
void RyzenAIServer::setupRoutes() {
    std::cout << "[Server] Setting up routes..." << std::endl;
    
    // Set CORS headers for all responses
    http_server_->set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"}
    });
    
    // Handle OPTIONS requests (CORS preflight)
    http_server_->Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });
    
    // Health endpoint
    http_server_->Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        handleHealth(req, res);
    });
    
    // Completions endpoint
    http_server_->Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handleCompletions(req, res);
    });
    
    // Chat completions endpoint
    http_server_->Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handleChatCompletions(req, res);
    });
    
    // Responses endpoint
    http_server_->Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handleResponses(req, res);
    });
    
    // Models endpoint - OpenAI compatible
    http_server_->Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        handleModels(res);
    });
    
    // Root redirect
    http_server_->Get("/", [this](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"message", "Ryzen AI LLM Server"},
            {"version", "1.0.0"},
            {"model", model_id_},
            {"endpoints", {
                "/health",
                "/v1/completions",
                "/v1/chat/completions",
                "/v1/responses"
            }}
        };
        res.set_content(response.dump(2), "application/json");
    });
    
    std::cout << "[Server] [OK] Routes configured" << std::endl;
}

/**
 * @brief Creates a standardized error response JSON object.
 *
 * This function generates a consistent error response format for API errors.
 *
 * @param message The error message to include in the response.
 * @param type The error type identifier.
 * @return json JSON object containing the error information.
 */
json RyzenAIServer::createErrorResponse(const std::string& message, const std::string& type) {
    return {
        {"error", {
            {"message", message},
            {"type", type}
        }}
    };
}

/**
 * @brief Handles the health check endpoint request.
 *
 * This function returns server status information including loaded models,
 * backend details, and configuration.
 *
 * @param req The HTTP request object.
 * @param res The HTTP response object to be populated.
 */
void RyzenAIServer::handleHealth(const httplib::Request& req, httplib::Response& res) {
    json response = {
        {"status", "ok"},
        {"model", model_id_},
        {"execution_mode", inference_engine_->getExecutionMode()},
        {"model_path", args_.model_path},
        {"loaded_models", inference_engine_->getLoadedModels()}
    };
    
    // Add backend-specific info if available
    if (auto* backend = inference_engine_->getDefaultBackend()) {
        response["backend"] = backend->getName();
        response["max_context_length"] = backend->getMaxContextLength();
    }
    
    res.set_content(response.dump(2), "application/json");
}

/**
 * @brief Handles the OpenAI completions endpoint request.
 *
 * This function processes text completion requests, supporting both streaming
 * and non-streaming modes, with reasoning content parsing and error handling.
 *
 * @param req The HTTP request object containing the completion request.
 * @param res The HTTP response object to be populated with the completion result.
 */
void RyzenAIServer::handleCompletions(const httplib::Request& req, httplib::Response& res) {
    try {
        json request_json = json::parse(req.body);
        auto comp_req = CompletionRequest::fromJSON(request_json);
        
        if (comp_req.prompt.empty()) {
            res.status = 400;
            res.set_content(createErrorResponse("Missing prompt", "invalid_request").dump(), "application/json");
            return;
        }
        
        // Helper lambda for robust parsing using model-specific tags
        // Works for both MLX and Onyx backends - getAdditionalTags() returns appropriate tags for each
        auto extract_robust_reasoning = [this](const std::string& text) -> std::pair<std::string, std::string> {
            auto result = parseReasoningContentWithModel(text, inference_engine_->getAdditionalTags());
            return {result.reasoning_content, result.regular_content};
        };

        if (comp_req.stream) {
            // For completeness of this snippet, re-inserting the STREAMING block:
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            
            GenerationParams params = createGenerationParams(comp_req.max_tokens, comp_req.temperature, comp_req.top_p, comp_req.top_k, comp_req.repeat_penalty, comp_req.stop);
            std::string prompt = comp_req.prompt;
            std::string model_id = model_id_;
            int prompt_tokens = inference_engine_->countTokens(prompt);
            
            res.set_chunked_content_provider("text/event-stream", [this, prompt, params, model_id, prompt_tokens](size_t offset, httplib::DataSink& sink) {
                if (offset > 0) return false;
                try {
                    // Use model-specific tags for both MLX and Onyx backends
                    ReasoningStreamParser parser(inference_engine_->getAdditionalTags());
                    int token_count = 0;
                    auto start_time = std::chrono::high_resolution_clock::now();

                    inference_engine_->streamComplete(prompt, params, [&sink, model_id, &token_count, &parser](const std::string& token, bool is_final) -> bool {
                        auto parsed = parser.consume(token);

                        // Simple JSON escape
                        auto escape = [](const std::string& s) {
                            std::string out; out.reserve(s.size());
                            for(char c : s) {
                                if(c == '\n') out += "\\n";
                                else if(c == '\r') out += "\\r";
                                else if(c == '"') out += "\\\"";
                                else if(c == '\\') out += "\\\\";
                                else out += c;
                            }
                            return out;
                        };

                        if (parsed.has_reasoning && !parsed.reasoning_content.empty()) {
                            std::string chunk = "data: {\"id\":\"cmpl-" + std::to_string(std::time(nullptr)) + "\",\"object\":\"text_completion.chunk\",\"created\":" + std::to_string(std::time(nullptr)) + ",\"model\":\"" + model_id + "\",\"choices\":[{\"index\":0,\"reasoning_content\":\"" + escape(parsed.reasoning_content) + "\",\"finish_reason\":null}]}\n\n";
                            sink.write(chunk.c_str(), chunk.size());
                        }
                        if (!parsed.regular_content.empty()) {
                            std::string finish = is_final ? "\"stop\"" : "null";
                            std::string chunk = "data: {\"id\":\"cmpl-" + std::to_string(std::time(nullptr)) + "\",\"object\":\"text_completion.chunk\",\"created\":" + std::to_string(std::time(nullptr)) + ",\"model\":\"" + model_id + "\",\"choices\":[{\"index\":0,\"text\":\"" + escape(parsed.regular_content) + "\",\"finish_reason\":" + finish + "}]}\n\n";
                            sink.write(chunk.c_str(), chunk.size());
                        }

                        if (parsed.should_stop) {
                            std::cout << "[Server] parser signaled stop, ending generation." << std::endl;
                            return false; // Stop generation
                        }
                        token_count++;
                        return true;
                    });

                    // Usage & Done
                    std::string usage = "data: {\"choices\":[{\"text\":\"\",\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"completion_tokens\":" + std::to_string(token_count) + "}}\n\n";
                    sink.write(usage.c_str(), usage.size());
                    sink.write("data: [DONE]\n\n", 14);
                    sink.done();
                } catch (const std::exception& e) {
                    json err = createErrorResponse(e.what(), "inference_error");
                    std::string s = "data: " + err.dump() + "\n\n";
                    sink.write(s.c_str(), s.size());
                    sink.done();
                }
                return false;
            });

        } else {
            // NON-STREAMING (Updated with Robust Parsing)
            GenerationParams params = createGenerationParams(comp_req.max_tokens, comp_req.temperature, comp_req.top_p, comp_req.top_k, comp_req.repeat_penalty, comp_req.stop);
            CompletionTimingData timing;
            std::string output = inference_engine_->complete(comp_req.prompt, params, &timing);
            
            // USE ROBUST PARSING HERE
            auto [reasoning, content] = extract_robust_reasoning(output);
            std::string final_text = comp_req.echo ? (comp_req.prompt + content) : content;
            
            json choice = {
                {"index", 0},
                {"text", final_text},
                {"finish_reason", "stop"}
            };
            if (!reasoning.empty()) choice["reasoning_content"] = reasoning;
            
            json response = {
                {"id", "cmpl-" + std::to_string(std::time(nullptr))},
                {"object", "text_completion"},
                {"created", std::time(nullptr)},
                {"model", model_id_},
                {"choices", {choice}},
                {"usage", {{"prompt_tokens", 0}, {"completion_tokens", timing.token_count}, {"total_tokens", timing.token_count}, {"completion_time_ms", timing.total_time_ms}}}
            };
            res.set_content(response.dump(), "application/json");
        }
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(createErrorResponse(e.what(), "internal_error").dump(), "application/json");
    }
}

/**
 * @brief Handles the OpenAI chat completions endpoint request.
 *
 * This function processes chat completion requests with message history,
 * supporting both streaming and non-streaming modes, tool calls, and reasoning content parsing.
 *
 * @param req The HTTP request object containing the chat completion request.
 * @param res The HTTP response object to be populated with the completion result.
 */
void RyzenAIServer::handleChatCompletions(const httplib::Request& req, httplib::Response& res) {
    try {
        // 1. Parse Request
        json request_json = json::parse(req.body);
        auto chat_req = ChatCompletionRequest::fromJSON(request_json);
        
        if (chat_req.messages.empty()) {
            res.status = 400;
            res.set_content(createErrorResponse("Missing messages", "invalid_request").dump(), 
                          "application/json");
            return;
        }
        
        // Determine which model to use (fall back to default if not specified)
        std::string requested_model = chat_req.model.empty() ? model_id_ : chat_req.model;
        std::cout << "[Server] Chat completion request for model: " << requested_model << std::endl;
        
        // 2. Prepare Prompt (using model-specific applyChatTemplate)
        json messages_array = json::array();
        for (const auto& msg : chat_req.messages) {
            messages_array.push_back({{"role", msg.role}, {"content", msg.content}});
        }
        std::string tools_json = chat_req.tools.empty() ? "" : chat_req.tools.dump();
        std::string prompt = inference_engine_->applyChatTemplate(requested_model, messages_array.dump(), tools_json);
        
        // -----------------------------------------------------------------------
        // ROBUST PARSING LOGIC using model-specific tags
        // Works for both MLX and Onyx backends - getAdditionalTags() returns appropriate tags for each
        // -----------------------------------------------------------------------
        auto robust_parse = [this](const std::string& full_output) -> std::pair<std::string, std::string> {
            auto result = parseReasoningContentWithModel(full_output, inference_engine_->getAdditionalTags());
            return {result.reasoning_content, result.regular_content};
        };

        if (chat_req.stream) {
            // STREAMING LOGIC
            // Note: Streaming "implicit starts" is very difficult because tokens are sent 
            // before we know a tag is missing. This uses standard parsing.
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            
            GenerationParams params = createGenerationParams(
                chat_req.max_tokens, chat_req.temperature, chat_req.top_p,
                chat_req.top_k, chat_req.repeat_penalty, chat_req.stop
            );
            
            std::string model_for_response = requested_model;
            bool has_tools = !chat_req.tools.empty();
            
            res.set_chunked_content_provider("text/event-stream",
                [this, prompt, params, model_for_response, requested_model, has_tools](size_t offset, httplib::DataSink& sink) {
                    if (offset > 0) return false;
                    try {
                        // Use model-specific tags for both MLX and Onyx backends
                        ReasoningStreamParser parser(inference_engine_->getAdditionalTags());
                        std::string full_response; // Buffer for tool extraction

                        inference_engine_->streamComplete(requested_model, prompt, params,
                            [&sink, &model_for_response, &parser, &full_response, has_tools](const std::string& token, bool is_final) -> bool {
                                if (has_tools) full_response += token;

                                auto parsed = parser.consume(token);

                                // JSON Escape Helper
                                auto escape = [](const std::string& s) {
                                    std::string out;
                                    for(char c : s) {
                                        if(c == '\n') out += "\\n"; else if(c == '\r') out += "\\r";
                                        else if(c == '"') out += "\\\""; else if(c == '\\') out += "\\\\";
                                        else out += c;
                                    }
                                    return out;
                                };

                                if (parsed.has_reasoning && !parsed.reasoning_content.empty()) {
                                    std::string chunk = "data: {\"id\":\"chatcmpl-" + std::to_string(std::time(nullptr)) + "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(std::time(nullptr)) + ",\"model\":\"" + model_for_response + "\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"" + escape(parsed.reasoning_content) + "\"},\"finish_reason\":null}]}\n\n";
                                    sink.write(chunk.c_str(), chunk.size());
                                }
                                if (!parsed.regular_content.empty()) {
                                    std::string finish = is_final ? "\"stop\"" : "null";
                                    std::string chunk = "data: {\"id\":\"chatcmpl-" + std::to_string(std::time(nullptr)) + "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(std::time(nullptr)) + ",\"model\":\"" + model_for_response + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + escape(parsed.regular_content) + "\"},\"finish_reason\":" + finish + "}]}\n\n";
                                    sink.write(chunk.c_str(), chunk.size());
                                }
                                
                                // Check if parser detected CHAT_END token - signal stop
                                if (parsed.should_stop) {
                                    std::cout << "[Server] ChatCompletions parser signaled stop, ending generation." << std::endl;
                                    return false;
                                }
                                return true;
                            }
                        );
                        
                        // Stream Tool Calls if needed
                        if (has_tools) {
                            auto [extracted, cleaned] = extractToolCalls(full_response);
                            for (const auto& tc : extracted) {
                                std::string args = tc.arguments.dump();
                                std::string esc_args; 
                                for(char c : args) { if(c=='"') esc_args+="\\\""; else if(c=='\\') esc_args+="\\\\"; else esc_args+=c; }
                                std::string chunk = "data: {\"id\":\"chatcmpl-" + std::to_string(std::time(nullptr)) + "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(std::time(nullptr)) + ",\"model\":\"" + model_for_response + "\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_auto\",\"type\":\"function\",\"function\":{\"name\":\"" + tc.name + "\",\"arguments\":\"" + esc_args + "\"}}]},\"finish_reason\":null}]}\n\n";
                                sink.write(chunk.c_str(), chunk.size());
                            }
                        }
                        
                        sink.write("data: [DONE]\n\n", 14);
                        sink.done();
                    } catch (const std::exception& e) {
                        json err = {{"error", {{"message", e.what()}, {"type", "inference_error"}}}};
                        std::string s = "data: " + err.dump() + "\n\n";
                        sink.write(s.c_str(), s.size());
                        sink.done();
                    }
                    return false;
                }
            );
        } else {
            // NON-STREAMING EXECUTION
            GenerationParams params = createGenerationParams(
                chat_req.max_tokens, chat_req.temperature, chat_req.top_p,
                chat_req.top_k, chat_req.repeat_penalty, chat_req.stop
            );
            
            CompletionTimingData timing;
            // Use model-specific complete
            std::string output = inference_engine_->complete(requested_model, prompt, params, &timing);
            
            // Apply Robust Parsing
            auto [reasoning, content] = robust_parse(output);
            
            // Extract Tool Calls (Compatibility)
            json tool_calls_json = nullptr;
            if (!chat_req.tools.empty()) {
                auto [extracted, cleaned] = extractToolCalls(content);
                if (!extracted.empty()) {
                    content = cleaned;
                    tool_calls_json = formatToolCallsForOpenAI(extracted);
                }
            }
            
            // Build Message
            json message = {
                {"role", "assistant"},
                {"content", content}
            };
            if (!reasoning.empty()) message["reasoning_content"] = reasoning;
            if (!tool_calls_json.is_null()) message["tool_calls"] = tool_calls_json;

            int prompt_tokens = inference_engine_->countTokens(requested_model, prompt);
            double tps = timing.token_count / (timing.total_time_ms / 1000.0);
            double ttft = timing.total_time_ms; // Placeholder for time to first token

            json response = {
                {"id", "chatcmpl-" + std::to_string(std::time(nullptr))},
                {"object", "chat.completion"},
                {"created", std::time(nullptr)},
                {"model", requested_model},
                {"choices", {{
                    {"index", 0},
                    {"message", message},
                    {"finish_reason", "stop"}
                }}},
                {"usage", {
                    {"prompt_tokens", prompt_tokens},
                    {"completion_tokens", timing.token_count},
                    {"total_tokens", timing.token_count},
                    {"completion_time_ms", timing.total_time_ms},
                    {"tps", tps},
                    {"ttft", ttft}
                }}
            };
            res.set_content(response.dump(), "application/json");
        }
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(createErrorResponse(e.what(), "internal_error").dump(), "application/json");
    }
}

/**
 * @brief Starts the HTTP server and begins listening for requests.
 *
 * This function starts the server on the configured host and port,
 * displays server information, and blocks until the server is stopped.
 *
 * @throws std::runtime_error If the server fails to start listening.
 */
void RyzenAIServer::run() {
    running_ = true;
    
    std::cout << "\n";
    std::cout << "===============================================================\n";
    std::cout << "  Server running at: http://" << args_.host << ":" << args_.port << "\n";
    std::cout << "===============================================================\n";
    std::cout << "\n";
    std::cout << "Available endpoints:\n";
    std::cout << "  GET  http://" << args_.host << ":" << args_.port << "/health\n";
    std::cout << "  GET  http://" << args_.host << ":" << args_.port << "/v1/models\n";
    std::cout << "  POST http://" << args_.host << ":" << args_.port << "/v1/completions\n";
    std::cout << "  POST http://" << args_.host << ":" << args_.port << "/v1/chat/completions\n";
    std::cout << "  POST http://" << args_.host << ":" << args_.port << "/v1/responses\n";
    std::cout << "\n";
    std::cout << "Press Ctrl+C to stop the server\n";
    std::cout << "===============================================================\n\n";
    
    // Start listening
    if (!http_server_->listen(args_.host, args_.port)) {
        throw std::runtime_error("Failed to start server on " + args_.host + ":" + std::to_string(args_.port));
    }
}

/**
 * @brief Handles the responses endpoint request.
 *
 * This function processes responses API requests, supporting both streaming
 * and non-streaming modes, with event-based streaming format.
 *
 * @param req The HTTP request object containing the responses request.
 * @param res The HTTP response object to be populated with the response result.
 */
void RyzenAIServer::handleResponses(const httplib::Request& req, httplib::Response& res) {
    try {
        // Parse request
        json request_json = json::parse(req.body);
        
        // Extract parameters (following ResponsesRequest format)
        bool stream = request_json.value("stream", false);
        std::string model = request_json.value("model", model_id_);
        int max_output_tokens = request_json.value("max_output_tokens", 512);
        float temperature = request_json.value("temperature", 1.0f);
        float repeat_penalty = request_json.value("repeat_penalty", 1.0f);
        int top_k = request_json.value("top_k", 40);
        float top_p = request_json.value("top_p", 0.9f);
        
        // Handle input - can be string or array of messages
        std::string prompt;
        if (request_json["input"].is_string()) {
            prompt = request_json["input"].get<std::string>();
        } else if (request_json["input"].is_array()) {
            // Apply chat template to messages array
            prompt = inference_engine_->applyChatTemplate(request_json["input"].dump());
        } else {
            res.status = 400;
            res.set_content(createErrorResponse("Input must be string or messages array", "invalid_request").dump(), 
                          "application/json");
            return;
        }
        
        std::cout << "[Server] Responses request (stream=" << stream << ")" << std::endl;
        
        if (stream) {
            // STREAMING RESPONSE: Use Responses API event format
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_header("X-Accel-Buffering", "no");
            
            GenerationParams params = createGenerationParams(
                max_output_tokens, temperature, top_p,
                top_k, repeat_penalty, {}
            );
            
            std::string model_name = model;
            
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt, params, model_name](size_t offset, httplib::DataSink& sink) {
                    if (offset > 0) return false; // Only run once
                    
                    try {
                        // Send response.created event
                        std::string created_time = std::to_string(std::time(nullptr));
                        std::string created_event = 
                            "data: {\"type\":\"response.created\",\"sequence_number\":0,"
                            "\"response\":{\"id\":\"0\",\"model\":\"" + model_name + 
                            "\",\"created_at\":" + created_time + 
                            ",\"object\":\"response\",\"output\":[],"
                            "\"parallel_tool_calls\":true,\"tool_choice\":\"auto\",\"tools\":[]}}\n\n";
                        
                        if (!sink.write(created_event.c_str(), created_event.size())) {
                            std::cout << "[Server] Failed to write created event" << std::endl;
                            return false;
                        }
                        
                        // Accumulate full response for the completed event
                        std::string full_response;
                        
                        // Generate and send tokens in real-time
                        inference_engine_->streamComplete(prompt, params, 
                            [&sink, &full_response](const std::string& token, bool is_final) -> bool {
                                // Escape special characters for JSON
                                std::string escaped_token = token;
                                size_t pos = 0;
                                while ((pos = escaped_token.find('\\', pos)) != std::string::npos) {
                                    escaped_token.replace(pos, 1, "\\\\");
                                    pos += 2;
                                }
                                pos = 0;
                                while ((pos = escaped_token.find('"', pos)) != std::string::npos) {
                                    escaped_token.replace(pos, 1, "\\\"");
                                    pos += 2;
                                }
                                pos = 0;
                                while ((pos = escaped_token.find('\n', pos)) != std::string::npos) {
                                    escaped_token.replace(pos, 1, "\\n");
                                    pos += 2;
                                }
                                pos = 0;
                                while ((pos = escaped_token.find('\r', pos)) != std::string::npos) {
                                    escaped_token.replace(pos, 1, "\\r");
                                    pos += 2;
                                }
                                
                                // Accumulate unescaped token for final response
                                full_response += token;
                                
                                // Send response.output_text.delta event
                                std::string delta_event = 
                                    "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,"
                                    "\"content_index\":0,\"delta\":\"" + escaped_token + 
                                    "\",\"item_id\":\"0\",\"output_index\":0}\n\n";
                                
                                if (!sink.write(delta_event.c_str(), delta_event.size())) {
                                    std::cout << "[Server] Client disconnected during streaming" << std::endl;
                                    return false; // Client disconnected, stop generation
                                }
                                return true; // Continue generation
                            }
                        );
                        
                        std::cout << "[Server] Token generation completed, sending final events" << std::endl;
                        
                        // Escape full_response for JSON
                        std::string escaped_full_response = full_response;
                        size_t pos = 0;
                        while ((pos = escaped_full_response.find('\\', pos)) != std::string::npos) {
                            escaped_full_response.replace(pos, 1, "\\\\");
                            pos += 2;
                        }
                        pos = 0;
                        while ((pos = escaped_full_response.find('"', pos)) != std::string::npos) {
                            escaped_full_response.replace(pos, 1, "\\\"");
                            pos += 2;
                        }
                        pos = 0;
                        while ((pos = escaped_full_response.find('\n', pos)) != std::string::npos) {
                            escaped_full_response.replace(pos, 1, "\\n");
                            pos += 2;
                        }
                        pos = 0;
                        while ((pos = escaped_full_response.find('\r', pos)) != std::string::npos) {
                            escaped_full_response.replace(pos, 1, "\\r");
                            pos += 2;
                        }
                        
                        // Send response.completed event
                        std::string completed_time = std::to_string(std::time(nullptr));
                        std::string completed_event = 
                            "data: {\"type\":\"response.completed\",\"sequence_number\":0,"
                            "\"response\":{\"id\":\"0\",\"model\":\"" + model_name + 
                            "\",\"created_at\":" + completed_time + 
                            ",\"object\":\"response\",\"output\":[{\"id\":\"0\",\"content\":[{"
                            "\"type\":\"output_text\",\"text\":\"" + escaped_full_response + 
                            "\",\"annotations\":[]}],\"role\":\"assistant\",\"status\":\"completed\","
                            "\"type\":\"message\"}],\"parallel_tool_calls\":true,"
                            "\"tool_choice\":\"auto\",\"tools\":[]}}\n\n";
                        
                        if (!sink.write(completed_event.c_str(), completed_event.size())) {
                            std::cout << "[Server] Failed to write completed event" << std::endl;
                            return false;
                        }
                        
                        // Send [DONE] marker
                        const char* done_msg = "data: [DONE]\n\n";
                        if (!sink.write(done_msg, strlen(done_msg))) {
                            std::cout << "[Server] Failed to write [DONE] marker" << std::endl;
                            return false;
                        }
                        
                        std::cout << "[Server] Streaming responses completed successfully" << std::endl;
                        return true;
                        
                    } catch (const std::exception& e) {
                        std::cerr << "[Server] Error in streaming responses: " << e.what() << std::endl;
                        std::string error_msg = "data: {\"error\":\"" + std::string(e.what()) + "\"}\n\n";
                        sink.write(error_msg.c_str(), error_msg.size());
                        return false;
                    }
                }
            );
            
        } else {
            // NON-STREAMING RESPONSE
            GenerationParams params = createGenerationParams(
                max_output_tokens, temperature, top_p,
                top_k, repeat_penalty, {}
            );
            
            std::string generated_text = inference_engine_->complete(prompt, params);
            
            // Create Response object
            json response = {
                {"id", "0"},
                {"model", model},
                {"created_at", std::time(nullptr)},
                {"object", "response"},
                {"output", json::array({
                    {
                        {"id", "0"},
                        {"content", json::array({
                            {
                                {"type", "output_text"},
                                {"text", generated_text},
                                {"annotations", json::array()}
                            }
                        })},
                        {"role", "assistant"},
                        {"status", "completed"},
                        {"type", "message"}
                    }
                })},
                {"parallel_tool_calls", true},
                {"tool_choice", "auto"},
                {"tools", json::array()}
            };
            
            res.set_content(response.dump(), "application/json");
            std::cout << "[Server] Non-streaming responses completed" << std::endl;
        }
        
    } catch (const json::exception& e) {
        std::cerr << "[Server] JSON parsing error: " << e.what() << std::endl;
        res.status = 400;
        res.set_content(createErrorResponse(e.what(), "invalid_request").dump(), "application/json");
    } catch (const std::exception& e) {
        std::cerr << "[Server] Error in responses: " << e.what() << std::endl;
        res.status = 500;
        res.set_content(createErrorResponse(e.what(), "internal_error").dump(), "application/json");
    }
}

/**
 * @brief Stops the HTTP server if it's running.
 *
 * This function stops the server and sets the running flag to false.
 */
void RyzenAIServer::stop() {
    if (running_) {
        std::cout << "\n[Server] Shutting down..." << std::endl;
        http_server_->stop();
        running_ = false;
    }
}

/**
 * @brief Handles the OpenAI models endpoint request.
 *
 * This function returns a list of all loaded models in OpenAI-compatible format,
 * including model IDs, creation times, and backend information.
 *
 * @param res The HTTP response object to be populated with the models list.
 */
void RyzenAIServer::handleModels(httplib::Response& res) {
    // OpenAI-compatible /v1/models endpoint
    json models_array = json::array();
    
    auto loaded_models = inference_engine_->getLoadedModels();
    for (const auto& name : loaded_models) {
        json model_obj = {
            {"id", name},
            {"object", "model"},
            {"created", std::time(nullptr)},
            {"owned_by", "ryzenai-server"}
        };
        
        // Add backend-specific info if available
        try {
            const auto* backend = inference_engine_->getBackendForModel(name);
            model_obj["backend"] = backend->getName();
            model_obj["max_context_length"] = backend->getMaxContextLength();
        } catch (...) {
            // Backend info not available
        }
        
        models_array.push_back(model_obj);
    }
    
    json response = {
        {"object", "list"},
        {"data", models_array}
    };
    
    res.set_content(response.dump(2), "application/json");
}

} // namespace ryzenai

