#include "ryzenai/command_line.h"
#include <iostream>
#include <cstring>

namespace ryzenai {

CommandLineArgs CommandLineParser::parse(int argc, char* argv[]) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                args.model_path = argv[++i];
            } else {
                throw std::runtime_error("Missing value for " + arg);
            }
        }
        else if (arg == "--host") {
            if (i + 1 < argc) {
                args.host = argv[++i];
            } else {
                throw std::runtime_error("Missing value for --host");
            }
        }
        else if (arg == "--port" || arg == "-p") {
            if (i + 1 < argc) {
                args.port = std::stoi(argv[++i]);
            } else {
                throw std::runtime_error("Missing value for --port");
            }
        }
        else if (arg == "--mode") {
            if (i + 1 < argc) {
                args.mode = argv[++i];
                // Validate mode
                if (args.mode != "npu" && args.mode != "hybrid" && args.mode != "cpu") {
                    throw std::runtime_error("Invalid mode: " + args.mode + " (must be npu, hybrid, or cpu)");
                }
            } else {
                throw std::runtime_error("Missing value for --mode");
            }
        }
        else if (arg == "--ctx-size" || arg == "-c") {
            if (i + 1 < argc) {
                args.ctx_size = std::stoi(argv[++i]);
            } else {
                throw std::runtime_error("Missing value for --ctx-size");
            }
        }
        else if (arg == "--threads" || arg == "-t") {
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
        else if (arg == "--prefill-chunk") {
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
    
    return args;
}

void CommandLineParser::printUsage(const char* program_name) {
    std::cout << "Ryzen AI LLM Server - OpenAI API compatible server for NPU/Hybrid/CPU execution\n\n";
    std::cout << "Usage: " << program_name << " -m MODEL_PATH [OPTIONS]\n\n";
    std::cout << "Required Arguments:\n";
    std::cout << "  -m, --model PATH          Path to model directory\n\n";
    std::cout << "Server Options:\n";
    std::cout << "  --host HOST               Host to bind to (default: 127.0.0.1)\n";
    std::cout << "  -p, --port PORT           Port to listen on (default: 8080)\n";
    std::cout << "  --mode MODE               Execution mode: npu|hybrid|cpu (default: hybrid)\n";
    std::cout << "  -t, --threads NUM         Number of threads (default: 4)\n";
    std::cout << "  -v, --verbose             Enable verbose output\n\n";
    std::cout << "Model Options:\n";
    std::cout << "  -c, --ctx-size SIZE       Maximum context size (default: 2048)\n\n";
    std::cout << "Optimization Options:\n";
    std::cout << "  --rep-lookback NUM        Repetition penalty lookback tokens (default: 64)\n";
    std::cout << "  --kv-cache                Enable KV cache (default: enabled)\n";
    std::cout << "  --no-kv-cache             Disable KV cache (slower but uses less memory)\n";
    std::cout << "  --prefill-chunk SIZE      Chunk size for long prompt prefill (default: 512)\n\n";
    std::cout << "Other:\n";
    std::cout << "  -h, --help                Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " -m ./models/phi-3-mini-4k-instruct\n";
    std::cout << "  " << program_name << " -m ./models/qwen3-0.6b --ctx-size 4096 --rep-lookback 128\n";
    std::cout << "  " << program_name << " -m ./models/llama --no-kv-cache --verbose\n\n";
}

} // namespace ryzenai

