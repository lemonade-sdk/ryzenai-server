#include "ryzenai/command_line.h"
#include <iostream>
#include <cstring>

namespace ryzenai {

/**
 * @brief Parses command line arguments into a CommandLineArgs structure.
 *
 * This function processes the command line arguments passed to the program,
 * validating and converting them into appropriate data types for server configuration.
 * Supports both single model and multi-model configurations with different backends.
 *
 * @param argc Number of command line arguments.
 * @param argv Array of command line argument strings.
 * @return CommandLineArgs Structure containing parsed configuration values.
 * @throws std::runtime_error If invalid arguments are provided or required values are missing.
 */
CommandLineArgs CommandLineParser::parse(int argc, char* argv[]) {
    CommandLineArgs args;
    
    // Track pending model config (for --model --backend pairs)
    ModelConfig pending_model;
    bool has_pending_model = false;
    
    auto flush_pending_model = [&]() {
        if (has_pending_model) {
            args.models.push_back(pending_model);
            pending_model = ModelConfig();
            has_pending_model = false;
        }
    };
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-m" || arg == "--model") {
            // Flush any pending model before starting a new one
            flush_pending_model();
            
            if (i + 1 < argc) {
                pending_model.path = argv[++i];
                has_pending_model = true;
                
                // Also set model_path for backward compatibility (first model)
                if (args.model_path.empty()) {
                    args.model_path = pending_model.path;
                }
            } else {
                throw std::runtime_error("Missing value for " + arg);
            }
        }
        else if (arg == "--backend" || arg == "-b") {
            if (i + 1 < argc) {
                std::string backend = argv[++i];
                
                // Validate backend - all supported types
                if (backend != "auto" && 
                    backend != "mlx" && backend != "mlx-metal" && backend != "metal" &&
                    backend != "mlx-rocm" && backend != "rocm" &&
                    backend != "mlx-cuda" && backend != "cuda" &&
                    backend != "onnx" && backend != "onnx-ryzenai" && 
                    backend != "onnx-directml" && backend != "onnx-cpu" &&
                    backend != "ryzenai" && backend != "npu" &&
                    backend != "cpu" && backend != "hybrid") {
                    throw std::runtime_error("Invalid backend: " + backend + 
                        " (see --help for valid backends)");
                }
                
                if (has_pending_model) {
                    // Apply to pending model
                    pending_model.backend = backend;
                } else {
                    // No pending model, use as default mode
                    args.mode = backend;
                }
            } else {
                throw std::runtime_error("Missing value for --backend");
            }
        }
        else if (arg == "--host") {
            flush_pending_model();
            if (i + 1 < argc) {
                args.host = argv[++i];
            } else {
                throw std::runtime_error("Missing value for --host");
            }
        }
        else if (arg == "--port" || arg == "-p") {
            flush_pending_model();
            if (i + 1 < argc) {
                args.port = std::stoi(argv[++i]);
            } else {
                throw std::runtime_error("Missing value for --port");
            }
        }
        else if (arg == "--mode") {
            flush_pending_model();
            if (i + 1 < argc) {
                args.mode = argv[++i];
                // Validate mode (same as backend)
                if (args.mode != "auto" && 
                    args.mode != "mlx" && args.mode != "mlx-metal" && args.mode != "metal" &&
                    args.mode != "mlx-rocm" && args.mode != "rocm" &&
                    args.mode != "mlx-cuda" && args.mode != "cuda" &&
                    args.mode != "onnx" && args.mode != "onnx-ryzenai" && 
                    args.mode != "onnx-directml" && args.mode != "onnx-cpu" &&
                    args.mode != "ryzenai" && args.mode != "npu" &&
                    args.mode != "cpu" && args.mode != "hybrid") {
                    throw std::runtime_error("Invalid mode: " + args.mode + 
                        " (see --help for valid modes)");
                }
            } else {
                throw std::runtime_error("Missing value for --mode");
            }
        }
        else if (arg == "--ctx-size" || arg == "-c") {
            flush_pending_model();
            if (i + 1 < argc) {
                args.ctx_size = std::stoi(argv[++i]);
            } else {
                throw std::runtime_error("Missing value for --ctx-size");
            }
        }
        else if (arg == "--threads" || arg == "-t") {
            flush_pending_model();
            if (i + 1 < argc) {
                args.threads = std::stoi(argv[++i]);
            } else {
                throw std::runtime_error("Missing value for --threads");
            }
        }
        else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        }
        else if (arg == "--rep-lookback") {
            flush_pending_model();
            if (i + 1 < argc) {
                args.repetition_lookback = std::stoi(argv[++i]);
            } else {
                throw std::runtime_error("Missing value for --rep-lookback");
            }
        }
        else if (arg == "--kv-cache") {
            args.kv_cache = true;
        }
        else if (arg == "--no-kv-cache") {
            args.kv_cache = false;
        }
        else if (arg == "--kv-format") {
            if (i + 1 < argc) {
                std::string format = argv[++i];
                if (format == "fp16") {
                    args.kv_format = KVCacheMode::FP16;
                } else if (format == "int8") {
                    args.kv_format = KVCacheMode::INT8;
                } else if (format == "int4") {
                    args.kv_format = KVCacheMode::INT4;
                } else {
                    throw std::runtime_error("Invalid KV format: " + format + 
                        " (valid options: fp16, int8, int4)");
                }
            } else {
                throw std::runtime_error("Missing value for --kv-format");
            }
        }
        else if (arg == "--prefill-chunk") {
            flush_pending_model();
            if (i + 1 < argc) {
                args.prefill_chunk = std::stoi(argv[++i]);
            } else {
                throw std::runtime_error("Missing value for --prefill-chunk");
            }
        }
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    
    // Flush any remaining pending model
    flush_pending_model();
    
    // If no models were specified via the models vector but model_path is set,
    // create a single model config for backward compatibility
    if (args.models.empty() && !args.model_path.empty()) {
        ModelConfig config;
        config.path = args.model_path;
        config.backend = args.mode;
        args.models.push_back(config);
    }
    
    return args;
}

/**
 * @brief Prints the usage information and help text for the program.
 *
 * This function displays detailed command line usage instructions, including
 * all available options, backend types, and examples of how to use the server
 * with different configurations.
 *
 * @param program_name The name of the executable program.
 */
void CommandLineParser::printUsage(const char* program_name) {
    std::cout << "Ryzen AI LLM Server - OpenAI API compatible server for multi-backend execution\n\n";
    std::cout << "Usage: " << program_name << " -m MODEL_PATH [OPTIONS]\n\n";
    
    std::cout << "Required Arguments:\n";
    std::cout << "  -m, --model PATH          Path to model directory (can be repeated)\n";
    std::cout << "  -b, --backend TYPE        Backend for preceding model (default: auto)\n\n";
    
    std::cout << "Multi-Model Usage:\n";
    std::cout << "  " << program_name << " -m /path/to/phi3 --backend mlx -m /path/to/qwen --backend mlx\n\n";
    
    std::cout << "Server Options:\n";
    std::cout << "  --host HOST               Host to bind to (default: 127.0.0.1)\n";
    std::cout << "  -p, --port PORT           Port to listen on (default: 8080)\n";
    std::cout << "  --mode MODE               Default backend/mode (default: hybrid)\n";
    std::cout << "  -t, --threads NUM         Number of threads (default: 4)\n";
    std::cout << "  -v, --verbose             Enable verbose output\n\n";
    
    std::cout << "Model Options:\n";
    std::cout << "  -c, --ctx-size SIZE       Maximum context size (default: 2048)\n\n";
    
    std::cout << "Optimization Options:\n";
    std::cout << "  --rep-lookback NUM        Repetition penalty lookback tokens (default: 64)\n";
    std::cout << "  --kv-cache                Enable KV cache (default: enabled)\n";
    std::cout << "  --no-kv-cache             Disable KV cache (slower but uses less memory)\n";
    std::cout << "  --kv-format FORMAT        KV cache format: fp16 (default), int8 (2x savings), int4\n";
    std::cout << "  --prefill-chunk SIZE      Chunk size for long prompt prefill (default: 512)\n\n";
    
    std::cout << "Backend Types:\n";
    std::cout << "  auto          - Auto-detect best backend for platform\n";
    std::cout << "  mlx, mlx-metal, metal   - Apple MLX with Metal (macOS)\n";
    std::cout << "  mlx-rocm, rocm          - MLX with ROCm (AMD GPUs on Linux)\n";
    std::cout << "  mlx-cuda, cuda          - MLX with CUDA (NVIDIA GPUs)\n";
    std::cout << "  onnx-ryzenai, ryzenai, npu  - AMD Ryzen AI NPU (Windows)\n";
    std::cout << "  onnx-directml           - ONNX with DirectML (Windows GPU)\n";
    std::cout << "  onnx-cpu, cpu           - ONNX Runtime CPU fallback\n";
    std::cout << "  hybrid                  - NPU + iGPU hybrid (Windows)\n\n";
    
    std::cout << "Other:\n";
    std::cout << "  -h, --help                Show this help message\n\n";
    
    std::cout << "Examples:\n";
    std::cout << "  # Single model (auto-detect backend)\n";
    std::cout << "  " << program_name << " -m ./models/phi-3-mini-4k-instruct\n\n";
    
    std::cout << "  # Single model with explicit backend\n";
    std::cout << "  " << program_name << " -m ./models/qwen3-0.6b --backend mlx\n\n";
    
    std::cout << "  # Multiple models on same backend\n";
    std::cout << "  " << program_name << " -m ./models/phi3 --backend mlx -m ./models/qwen --backend mlx\n\n";
    
    std::cout << "  # Multiple models on different backends\n";
    std::cout << "  " << program_name << " -m ./models/phi3-npu --backend ryzenai -m ./models/phi3-mlx --backend mlx\n\n";
    
    std::cout << "  # ROCm backend for AMD GPUs\n";
    std::cout << "  " << program_name << " -m ./models/llama --backend mlx-rocm\n\n";
}

} // namespace ryzenai
