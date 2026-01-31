# Ryzen AI LLM Server (ONNX & MLX)

A lightweight, OpenAI API-compatible server for running LLMs with support for multiple backends including AMD Ryzen AI NPUs (Windows), Apple MLX (macOS), and ONNX Runtime (cross-platform).

## Overview

This server enables running Large Language Models on AMD Ryzen AI 300-series processors with NPU acceleration and other models through MLX. It implements the OpenAI API specification, making it compatible with existing LLM applications and tools.

**Key Features:**
- **OpenAI API Compatible:** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`
- **Tool/Function Calling:** OpenAI-compatible function calling support
- **Multiple Backends:** NPU, Hybrid (NPU+iGPU), CPU, MLX (Metal/ROCm/CUDA)
- **Multiple Model Loading:** Load and serve multiple models simultaneously
- **Streaming Support:** Real-time Server-Sent Events for all endpoints
- **Echo Parameter:** Option to include prompt in completion output
- **Stop Sequences:** Custom stop sequences for generation control
- **Minimal Dependencies:** Single executable deployment

## Supported Backends

| Backend | Platform | Hardware | Description |
|---------|----------|----------|-------------|
| `mlx-metal` | macOS | Apple Silicon | MLX with Metal GPU acceleration |
| `mlx-rocm` | Linux | AMD GPUs | MLX with ROCm (coming soon) |
| `mlx-cuda` | Linux/Windows | NVIDIA GPUs | MLX with CUDA (coming soon) |
| `onnx-ryzenai` | Windows | AMD NPU | ONNX Runtime with Ryzen AI NPU |
| `onnx-cpu` | Cross-platform | CPU | ONNX Runtime CPU fallback |
| `auto` | Any | Auto-detect | Automatically selects best backend |

## Supported Model Architectures

### ONNX Backend (Windows/Ryzen AI)
- Any ONNX model compatible with ONNX Runtime GenAI
- Optimized for AMD Ryzen AI 300/400-series NPUs

### MLX Backend
- **Phi-3** - Microsoft Phi-3 family
- **Qwen3** - Alibaba Qwen3 models
- **Qwen3-Next** - Qwen3 with linear attention

## Building from Source

### Prerequisites

*****Windows (ONNX/Ryzen AI Backend):*****
- Windows 11 (64-bit)
- Visual Studio 2022
- CMake 3.20 or higher
- **Ryzen AI Software 1.6.0** with LLM patch
  - Base installation must be at `C:\Program Files\RyzenAI\1.6.0`
  - LLM patch must be applied on top of base installation
  - Download from: https://ryzenai.docs.amd.com

****AMD (ROCm/NPU) Hardware Requirements:****
- AMD Ryzen AI 300- or 400-series processor (for NPU execution)
- Minimum 16GB RAM (32GB recommended for larger models)

**macOS (MLX Backend):**
- macOS 14+ with Apple Silicon (M1/M2/M3/M4/M5)
- Xcode Command Line Tools
- CMake 3.20 or higher

**MLX Support - Hardware Requirements:**
- **macOS:** Apple Silicon Mac with 16GB+ unified memory

### Build Options

The build system uses CMake options to configure which backends to enable:

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_RYZEN` | `ON` | Build with Ryzen AI NPU support (Windows only) |
| `BUILD_MLX` | `ON` (macOS/Linux) | Build with MLX backend support |
| `MLX_CUDA` | `ON` (Linux) | Enable CUDA support for MLX |
| `MLX_ROCM` | `OFF` | Enable ROCm support for MLX (AMD GPUs) |
| `OGA_ROOT` | `C:/Program Files/RyzenAI/1.6.0` | Custom path to Ryzen AI installation |


### Build Steps (Windows - Ryzen AI Backend)

```cmd
# Clone the repository
git clone https://github.com/lemonade-sdk/ryzenai-server.git
cd ryzenai-server

# Create and enter build directory
mkdir build
cd build

# Configure with CMake (Ryzen AI backend is auto-enabled on Windows)
cmake .. -G "Visual Studio 17 2022" -A x64

# Or with custom Ryzen AI installation path:
cmake .. -G "Visual Studio 17 2022" -A x64 -DOGA_ROOT="D:\RyzenAI\1.6.0"

# Build
cmake --build . --config Release
```

### Build Steps (macOS - MLX with Metal)

```bash
# Clone the repository
git clone https://github.com/lemonade-sdk/ryzenai-server.git
cd ryzenai-server

# Create and enter build directory
mkdir build
cd build

# Configure with CMake (MLX backend with Metal is auto-enabled on macOS)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (this will fetch MLX and SentencePiece automatically)
cmake --build . --config Release
```

**Note:** The first build will download and compile MLX and SentencePiece dependencies, which may take several minutes.

### Build Steps (Linux - MLX with CUDA)

```bash
# Clone the repository
git clone https://github.com/lemonade-sdk/ryzenai-server.git
cd ryzenai-server

mkdir build && cd build

# Enable MLX with CUDA support
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_MLX=ON -DMLX_CUDA=ON

# Build
cmake --build . --config Release
```

### Build Steps (Linux - MLX with ROCm)

```bash
# Clone the repository
git clone https://github.com/lemonade-sdk/ryzenai-server.git
cd ryzenai-server

mkdir build && cd build

# Enable MLX with ROCm support for AMD GPUs
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_MLX=ON -DMLX_ROCM=ON

# Build
cmake --build . --config Release
```


### Build Output

**macOS/Linux:**
```
build/bin/ryzenai-server
```

**Windows:**
```
build\bin\Release\ryzenai-server.exe
```

All required Ryzen AI DLLs are automatically copied to the output directory during a Windows build.

## Usage

### Starting the Server

```bash
# Single model with auto-detected backend
ryzenai-server -m /path/to/model

# Specify backend explicitly
ryzenai-server -m /path/to/model --mode onnx-ryzenai
ryzenai-server -m /path/to/model --mode cpu
ryzenai-server -m /path/to/model --mode mlx-metal

# Custom port
ryzenai-server -m /path/to/model --port 8081

# Verbose logging
ryzenai-server -m /path/to/model --verbose
```

### Multiple Model Loading

Load multiple models simultaneously with different backends:

```bash
# Load multiple models (each with optional backend specification)
ryzenai-server \
  --model /path/to/llama:onnx-cpu \
  --model /path/to/phi3:mlx \
  --model /path/to/qwen3:mlx

# Models are addressable by name in API requests
```

When multiple models are loaded:
- Each model is accessible via the `model` field in API requests
- Models can use different backends
- The first loaded model is used as the default if no model is specified

### Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model PATH[:BACKEND]` | `-m` | Model path with optional backend | (required) |
| `--host ADDRESS` | | Server host address | `127.0.0.1` |
| `--port PORT` | `-p` | Server port | `8080` |
| `--mode MODE` | | Default backend mode | `auto` |
| `--ctx-size SIZE` | `-c` | Context size in tokens | `2048` |
| `--threads NUM` | `-t` | Number of CPU threads | `4` |
| `--kv-cache` | | Enable KV cache (default) | enabled |
| `--no-kv-cache` | | Disable KV cache | |
| `--rep-lookback NUM` | | Repetition penalty lookback | `64` |
| `--prefill-chunk NUM` | | Prefill chunk size | `512` |
| `--verbose` | `-v` | Enable verbose logging | off |
| `--help` | `-h` | Show help message | |

### Model Requirements

**MLX Models:**
- SafeTensors format (`.safetensors`)
- `config.json` with model configuration
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)
- Optionally quantized (4-bit, 8-bit)

**ONNX Models:**
- `model.onnx` or `model.onnx.data`
- `genai_config.json`
- Tokenizer files

## Code Structure

```
ryzenai-server/
├── CMakeLists.txt              # Build configuration
│
├── src/                        # Source files
│   ├── main.cpp                # Entry point
│   ├── server.cpp              # HTTP server (cpp-httplib)
│   ├── inference_engine.cpp    # Multi-backend inference engine
│   ├── command_line.cpp        # CLI argument parsing
│   ├── types.cpp               # Data structures
│   ├── tool_calls.cpp          # OpenAI tool/function calling
│   ├── reasoning.cpp           # Reasoning content handling
│   │
│   ├── backend/                # Backend implementations
│   │   ├── onnx_backend.cpp    # ONNX Runtime backend
│   │   └── mlx_backend.cpp     # MLX backend
│   │
│   └── mlx/                    # MLX-specific code
│       ├── mlx_oga.cpp         # MLX model loading
│       ├── tokenizer.cpp       # Tokenizer implementation
│       ├── quantization.cpp    # Quantization support
│       └── models/             # Model architecture implementations
│           ├── llama_inference.cpp
│           ├── phi_inference.cpp
│           ├── phi3_inference.cpp
│           ├── qwen3_inference.cpp
│           ├── gemma_inference.cpp
│           ├── mixtral_inference.cpp
│           └── deepseek_inference.cpp
│
├── include/ryzenai/            # Headers
│   ├── server.h
│   ├── inference_engine.h
│   ├── command_line.h
│   ├── types.h
│   ├── tool_calls.h
│   ├── reasoning.h
│   │
│   ├── backend/                # Backend interface
│   │   ├── backend.h           # IBackend interface & registry
│   │   ├── onnx_backend.h
│   │   └── mlx_backend.h
│   │
│   └── mlx/                    # MLX headers
│       ├── model.h
│       ├── tokenizer.h
│       ├── generator.h
│       ├── kv_cache.h
│       ├── quantization.h
│       └── models/             # Model-specific headers
│
└── external/                   # Header-only dependencies
    ├── cpp-httplib/            # HTTP server (auto-downloaded)
    └── json/                   # JSON library (auto-downloaded)
```

## Architecture Overview

### Multi-Backend Design

The server uses a pluggable backend architecture:

```
┌─────────────────────────────────────────────────┐
│         HTTP Server (cpp-httplib)               │
│         OpenAI API Endpoints                    │
├─────────────────────────────────────────────────┤
│         Request Handlers                        │
│         (routing by model name)                 │
├─────────────────────────────────────────────────┤
│         Inference Engine                        │
│         (multi-model management)                │
├─────────────────────────────────────────────────┤
│    ┌──────────────┬──────────────────────┐      │
│    │ ONNX Backend │    MLX Backend       │      │
│    ├──────────────┼──────────────────────┤      │
│    │ • RyzenAI NPU│ • Metal (Apple)      │      │
│    │ • CPU        │ • ROCm (AMD GPU)     │      │
│    │              │ • CUDA (NVIDIA)      │      │
│    └──────────────┴──────────────────────┘      │
└─────────────────────────────────────────────────┘
```

### Component Layers

**Inference Engine:** Manages multiple loaded models across different backends. Routes requests to appropriate backends based on model name.

**Backend Interface (`IBackend`):** Unified interface for all backends providing:
- Model loading and lifecycle
- Tokenization (encode/decode)
- Chat template application
- Synchronous and streaming generation

**Backend Registry:** Factory pattern for backend creation with auto-detection.

### Dependencies

Auto-downloaded during build:
- **cpp-httplib** (v0.26.0) - HTTP server [MIT License]
- **nlohmann/json** (v3.11.3) - JSON parsing [MIT License]

Platform-specific:
- **MLX** - Apple's ML framework (macOS)
- **ONNX Runtime GenAI** - Microsoft inference engine (Windows)

## API Endpoints

The server implements OpenAI-compatible API endpoints.

### Health Check

```bash
GET /health
```

Returns server status and loaded models:
```json
{
  "status": "ok",
  "models": ["phi3", "qwen3"],
  "backends": {
    "phi3": "mlx-metal",
    "qwen3": "mlx-metal"
  }
}
```

### List Models

```bash
GET /v1/models
```

Returns list of available models:
```json
{
  "object": "list",
  "data": [
    {"id": "Phi-3-mini-4k-instruct-4bit", "object": "model", "owned_by": "local"},
    {"id": "Qwen3-1.7B-4bit", "object": "model", "owned_by": "local"}
  ]
}
```

### Chat Completions

```bash
POST /v1/chat/completions
```

```json
{
  "model": "Phi-3-mini-4k-instruct-4bit",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "stream": false
}
```

### Other Endpoints

- `GET /` - Server information and available endpoints
- `POST /v1/completions` - Text completions with echo parameter
- `POST /v1/responses` - OpenAI Responses API format

## Testing

### Quick Test

```bash
# Start the server
./ryzenai-server -m /path/to/model --verbose

# Test health endpoint
curl http://localhost:8080/health

# Test chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'

# Test with specific model (multi-model mode)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Integration Examples

### Python with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

# Single model mode
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(response.choices[0].message.content)

# Multi-model mode - specify which model to use
response = client.chat.completions.create(
    model="phi3",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

### Streaming Example

```python
stream = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Integration with Lemonade Server

This server is designed to be used as a backend for [Lemonade Server](https://github.com/lemonade-sdk/lemonade). When running Lemonade Server, the `ryzenai-server` executable is automatically downloaded from GitHub releases and managed by the Lemonade Router.

## Troubleshooting

### "Model not found" or "Failed to load model"

**Check:**
1. Model path is correct and contains required files
2. Model format matches backend (SafeTensors for MLX, ONNX for ONNX backend)
3. Platform-specific requirements are met

### Backend Selection Issues

If auto-detection isn't working:
```bash
# Force a specific backend
ryzenai-server -m /path/to/model --mode mlx-metal
ryzenai-server -m /path/to/model --mode onnx-cpu
```

### Missing DLLs (Windows)

All required DLLs should be automatically copied during build. If you get DLL errors:
1. Verify Ryzen AI is installed correctly
2. Rebuild with `cmake --build . --config Release`
3. Manually copy DLLs from `C:\Program Files\RyzenAI\1.6.0\deployment\`

### Port Already in Use

```bash
ryzenai-server -m /path/to/model --port 8081
```

## Development

### Code Style

- C++17 standard
- RAII for resource management
- Smart pointers (no raw pointers)
- Const correctness
- Snake_case for functions
- PascalCase for types

### Adding a New Backend

1. Implement `IBackend` interface in `include/ryzenai/backend/`
2. Add implementation in `src/backend/`
3. Register in `BackendRegistry` during initialization

### Adding a New Model Architecture (MLX)

1. Create header in `include/ryzenai/mlx/models/`
2. Implement in `src/mlx/models/`
3. Register in inference factory

## Related Projects

- **Ryzen AI Documentation:** https://ryzenai.docs.amd.com
- **ONNX Runtime GenAI:** https://github.com/microsoft/onnxruntime-genai
- **Apple MLX:** https://github.com/ml-explore/mlx
- **Lemonade Server:** https://github.com/lemonade-sdk/lemonade

## License

This project's **source code** is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

**Release Artifacts:**
- The `ryzenai-server` binary and header-only dependencies are MIT licensed
- **Ryzen AI DLLs** (Windows releases) are licensed under the AMD Software End User License Agreement - see `AMD_LICENSE` file
