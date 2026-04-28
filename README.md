# Ryzen AI LLM Server

A lightweight, OpenAI API-compatible server for running LLMs on AMD Ryzen AI NPUs using ONNX Runtime GenAI.

## Overview

This server enables running Large Language Models on AMD Ryzen AI 300-series processors with NPU acceleration. It implements the OpenAI API specification, making it compatible with existing LLM applications and tools.

**Key Features:**
- **OpenAI API Compatible:** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
- **Tool/Function Calling:** OpenAI-compatible function calling support
- **Multiple Execution Modes:** NPU, Hybrid (NPU+iGPU), CPU
- **Streaming Support:** Real-time Server-Sent Events for all endpoints
- **Echo Parameter:** Option to include prompt in completion output
- **Stop Sequences:** Custom stop sequences for generation control
- **Minimal Dependencies:** Single executable + DLLs
- **Simple Architecture:** One-model-per-process design

## Building from Source

### Prerequisites

**Windows Requirements:**
- Windows 11 (64-bit)
- Visual Studio 2022
- CMake 3.20 or higher
- **Ryzen AI Software 1.7.0**
  - Default installation path: `C:\Program Files\RyzenAI\1.7.0`
  - Download from: https://ryzenai.docs.amd.com

**Hardware Requirements:**
- AMD Ryzen AI 300- or 400-series processor (for NPU execution)
- Minimum 16GB RAM (32GB recommended for larger models)

### Build Steps (Windows)

```cmd
# Clone the repository
git clone https://github.com/lemonade-sdk/ryzenai-server.git
cd ryzenai-server

# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64

# Build
cmake --build . --config Release
```

### Build Steps (Linux)

**Linux Requirements:**
- Ubuntu 22.04+ or equivalent Linux distribution
- GCC 9+ or Clang 10+
- CMake 3.20 or higher
- **Ryzen AI Software 1.7.0 for Linux**
  - Default installation path: `/opt/ryzenai/1.7.0`
  - Download from: https://ryzenai.docs.amd.com

```bash
# Clone the repository
git clone https://github.com/lemonade-sdk/ryzenai-server.git
cd ryzenai-server

# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .
```

### Build Output

**Windows:** The executable and required DLLs will be created at:
```
build\bin\Release\ryzenai-server.exe
```

**Linux:** The executable and required shared libraries will be created at:
```
build/bin/ryzenai-server
```

All necessary Ryzen AI libraries (DLLs on Windows, .so files on Linux) are automatically copied to the output directory during build.

**Note:** The Ryzen AI libraries included in the release are licensed under the AMD Software End User License Agreement. See `AMD_LICENSE` in the release package for full terms.

### Custom Ryzen AI Installation Path

If Ryzen AI is installed in a custom location, you can specify it using either an environment variable (recommended) or a CMake option.

**Option 1: Environment Variable (works for both build and runtime)**

```bash
# Linux/macOS
export RYZENAI_INSTALL_PATH=/custom/path/ryzenai/1.7.0
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Windows (PowerShell)
$env:RYZENAI_INSTALL_PATH="C:\custom\path\RyzenAI\1.7.0"
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**Option 2: CMake Option (build-time only)**

```bash
# Linux
cmake .. -DCMAKE_BUILD_TYPE=Release -DOGA_ROOT=/custom/path/ryzenai/1.7.0

# Windows
cmake .. -G "Visual Studio 17 2022" -A x64 -DOGA_ROOT="C:\custom\path\RyzenAI\1.7.0"
```

**Priority Order:**
1. `RYZENAI_INSTALL_PATH` environment variable (highest priority)
2. `-DOGA_ROOT` CMake cache variable
3. Platform-specific defaults (`C:/Program Files/RyzenAI/1.7.0` on Windows, `/opt/ryzenai/1.7.0` on Linux)

## Code Structure

```
ryzenai-server/
├── CMakeLists.txt              # Build configuration
│
├── src/                        # Source files
│   ├── main.cpp                # Entry point
│   ├── server.cpp              # HTTP server (cpp-httplib)
│   ├── inference_engine.cpp    # ONNX Runtime GenAI wrapper
│   ├── command_line.cpp        # CLI argument parsing
│   ├── types.cpp               # Data structures
│   ├── tool_calls.cpp          # OpenAI tool/function calling
│   └── reasoning.cpp           # Reasoning content handling
│
├── include/ryzenai/            # Headers
│   ├── server.h
│   ├── inference_engine.h
│   ├── command_line.h
│   ├── types.h
│   ├── tool_calls.h
│   └── reasoning.h
│
└── external/                   # Header-only dependencies
    ├── cpp-httplib/            # HTTP server (auto-downloaded)
    └── json/                   # JSON library (auto-downloaded)
```

## Architecture Overview

### Design Principles

1. **Simplicity**: One process serves one model - no dynamic loading/unloading
2. **RAII**: Resource management follows C++ best practices with smart pointers
3. **Thread Safety**: Shared resources protected with proper synchronization
4. **Single Binary**: Minimal dependencies for easy deployment

### Component Layers

```
┌─────────────────────────────────────────────────┐
│         HTTP Server (cpp-httplib)               │
│         OpenAI API Endpoints                    │
├─────────────────────────────────────────────────┤
│         Request Handlers                        │
│         (chat, completions, streaming)          │
├─────────────────────────────────────────────────┤
│         Inference Engine                        │
│         ONNX Runtime GenAI                      │
├─────────────────────────────────────────────────┤
│         Execution Providers                     │
│         NPU / Hybrid / CPU                      │
└─────────────────────────────────────────────────┘
```

**Server:** HTTP server using cpp-httplib with OpenAI-compatible endpoints. Features:
- 8-thread pool for concurrent request handling
- Built-in CORS support (Access-Control-Allow-Origin: *)
- Request routing and response formatting
- Chunked transfer encoding for streaming

**Inference Engine:** Wraps ONNX Runtime GenAI API, managing model loading, generation parameters, and streaming callbacks. Applies chat templates and handles tool call extraction.

**Execution Providers:** Supports three modes (auto-detected from model config):
- **Hybrid**: NPU + iGPU
- **NPU**: Pure NPU execution
- **CPU**: CPU-only fallback

### Dependencies

These dependencies are automatically downloaded during build:

- **cpp-httplib** (v0.26.0) - HTTP server [MIT License]
- **nlohmann/json** (v3.11.3) - JSON parsing [MIT License]

These dependencies must be manually installed by the developer:
- **ONNX Runtime GenAI** - Inference engine

## Usage

### Starting the Server

**Windows:**
```cmd
# Start the server (execution mode is auto-detected from the model)
ryzenai-server.exe -m C:\path\to\onnx\model

# Custom port
ryzenai-server.exe -m C:\path\to\onnx\model --port 8081

# Verbose logging
ryzenai-server.exe -m C:\path\to\onnx\model --verbose
```

**Linux:**
```bash
# Start the server (execution mode is auto-detected from the model)
./ryzenai-server -m /opt/models/phi-3-mini-4k-instruct-onnx

# With custom Ryzen AI installation path
export RYZENAI_INSTALL_PATH=/custom/ryzenai/1.7.0
./ryzenai-server -m /opt/models/phi-3-mini-4k-instruct-onnx

# Custom port
./ryzenai-server -m /opt/models/phi-3-mini-4k-instruct-onnx --port 8081

# Verbose logging
./ryzenai-server -m /opt/models/phi-3-mini-4k-instruct-onnx --verbose
```

### Command-Line Arguments

- `-m, --model PATH` - Path to ONNX model directory (required)
- `--host ADDRESS` - Server host address (default: 127.0.0.1)
- `-p, --port PORT` - Server port (default: 8080)
- `-c, --ctx-size SIZE` - Context size in tokens (default: 2048)
- `-t, --threads NUM` - Number of CPU threads (default: 4)
- `-v, --verbose` - Enable verbose logging
- `-h, --help` - Show help message

The execution mode (NPU, Hybrid, or CPU) is automatically detected from the model's `genai_config.json` configuration.

### Model Requirements

Models must be in ONNX format compatible with Ryzen AI. Required files:
- `model.onnx` or `model.onnx.data`
- `genai_config.json`
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)

Models are typically cached in:

**Windows:**
```
C:\Users\<Username>\.cache\huggingface\hub\
```

**Linux:**
```
~/.cache/huggingface/hub/
```

## API Endpoints

The server implements OpenAI-compatible API endpoints.

### Health Check

```bash
GET /health
```

Returns server status and Ryzen AI-specific information:
```json
{
  "status": "ok",
  "model": "phi-3-mini-4k-instruct",
  "execution_mode": "hybrid",
  "max_prompt_length": 4096,
  "ryzenai_version": "1.7.0"
}
```

### Other Endpoints

- `GET /` - Server information and available endpoints
- `POST /v1/chat/completions` - Chat completions with tool/function calling support
- `POST /v1/completions` - Text completions with echo parameter
- `POST /v1/responses` - OpenAI Responses API format

All endpoints support both streaming and non-streaming modes. The server applies chat templates automatically and extracts tool calls from model output.

## Testing

### Quick Test

**Windows:**
```cmd
# Start the server
cd build\bin\Release
ryzenai-server.exe -m C:\path\to\model --verbose

# Test health endpoint (in another terminal)
curl http://localhost:8080/health

# Test chat completion
curl http://localhost:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 50}"
```

**Linux:**
```bash
# Start the server
cd build/bin
./ryzenai-server -m /path/to/model --verbose

# Test health endpoint (in another terminal)
curl http://localhost:8080/health

# Test chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

## Integration with Lemonade Server

This server is designed to be used as a backend for [Lemonade Server](https://github.com/lemonade-sdk/lemonade). When running Lemonade Server, the `ryzenai-server` executable is automatically downloaded from GitHub releases and managed by the Lemonade Router.

## Integration Examples

### Python with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="ignored",  # Model is already loaded
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ]
)

print(response.choices[0].message.content)
```

## Troubleshooting

### "Model not found" or "Failed to load model"

**Check:**
1. Model path is correct and contains required ONNX files
2. Ryzen AI 1.7.0 is installed at the correct path
3. NPU drivers are up to date (Windows Update)
4. Model is compatible with your Ryzen AI version

### Missing DLLs

All required DLLs should be automatically copied during build. If you get DLL errors:
1. Verify Ryzen AI is installed correctly
2. Rebuild with `cmake --build . --config Release`
3. Manually copy DLLs from `C:\Program Files\RyzenAI\1.7.0\deployment\` to the executable directory

### Port Already in Use

If port 8080 is occupied:

**Windows:**
```cmd
ryzenai-server.exe -m C:\path\to\model --port 8081
```

**Linux:**
```bash
./ryzenai-server -m /path/to/model --port 8081
```

### Linux-Specific Notes

**Driver Detection:**
On Linux, NPU driver detection is informational only. If the driver cannot be detected, the server will print a warning but continue startup. This is expected behavior as Linux driver interfaces may vary.

**Library Loading:**
The build system automatically copies required Ryzen AI libraries next to the executable and configures RPATH to search the executable's directory (`$ORIGIN`). This means:
- No `LD_LIBRARY_PATH` setup required
- The binary is relocatable - works from any directory
- To use a different Ryzen AI version, rebuild with the appropriate `RYZENAI_INSTALL_PATH` or `OGA_ROOT`

**Running from Different Directories:**
Because RPATH is configured, you can run the server from any directory:
```bash
# These all work
./build/bin/ryzenai-server -m /path/to/model
cd build/bin && ./ryzenai-server -m /path/to/model
/full/path/to/ryzenai-server -m /path/to/model
```

## Development

### Code Style

- C++17 standard
- RAII for resource management
- Smart pointers (no raw pointers)
- Const correctness
- Snake_case for functions
- PascalCase for types

### Building for Development

**Windows:**
```cmd
cmake --build . --config Debug
```

Debug executable location: `build\bin\Debug\ryzenai-server.exe`

**Linux:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

Debug executable location: `build/bin/ryzenai-server`

### Known Issues

**Streaming with JSON Library:** Creating `nlohmann::json` objects directly in ONNX Runtime streaming callbacks can cause crashes. The workaround is to manually construct JSON strings in callbacks. This is stable and performs well.

## Related Projects

- **Ryzen AI Documentation:** https://ryzenai.docs.amd.com
- **ONNX Runtime GenAI:** https://github.com/microsoft/onnxruntime-genai
- **Lemonade Server:** https://github.com/lemonade-sdk/lemonade - Parent project providing model orchestration

## License

This project's **source code** is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

**Release Artifacts (ryzenai-server.zip):**
- The `ryzenai-server.exe` binary and the header-only dependencies (**cpp-httplib**, **nlohmann/json**) are MIT licensed
- The **Ryzen AI DLLs** included in binary releases are licensed under the AMD Software End User License Agreement - see `AMD_LICENSE` file in the release package for full terms

**Note:** When you download a release, the `AMD_LICENSE` file is included alongside the DLLs. The source code in this repository does not include the DLLs - they are copied from your local Ryzen AI installation during the build process.
