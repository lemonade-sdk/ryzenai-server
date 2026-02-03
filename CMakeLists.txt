cmake_minimum_required(VERSION 3.20)
project(ryzenai-server VERSION 1.0.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_RYZEN "Build with Ryzen AI support" ON)
if(APPLE)
    option(BUILD_MLX "Build with MLX support" ON)
    option(MLX_CUDA "Enable CUDA support for MLX" OFF)
    option(MLX_ROCM "Enable ROCm support for MLX" OFF)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    option(BUILD_MLX "Build with MLX support" ON)
    option(MLX_CUDA "Enable CUDA support for MLX" OFF)
    option(MLX_ROCM "Enable ROCm support for MLX" ON)
else()
    option(BUILD_MLX "Build with MLX support" OFF)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Building Ryzen AI LLM Server")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find dependencies
find_package(Threads REQUIRED)

# Find BLAS and LAPACK for MLX
if(BUILD_MLX)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

# Ryzen AI dependencies
if(BUILD_RYZEN)
    # ONNX Runtime GenAI
    if(WIN32)
        set(OGA_ROOT "C:/Program Files/RyzenAI/1.6.0" CACHE PATH "Path to Ryzen AI installation")

        if(NOT EXISTS ${OGA_ROOT})
            message(FATAL_ERROR "Ryzen AI not found at ${OGA_ROOT}. Please install Ryzen AI 1.6.0 or set OGA_ROOT")
        endif()

        message(STATUS "Using Ryzen AI from: ${OGA_ROOT}")

        # Find ONNX Runtime GenAI library (it's in LLM subfolder)
        find_library(OGA_LIB
            NAMES onnxruntime-genai
            PATHS ${OGA_ROOT}/LLM/lib
            REQUIRED
        )

        find_path(OGA_INCLUDE
            NAMES ort_genai.h
            PATHS ${OGA_ROOT}/LLM/include
            REQUIRED
        )

        message(STATUS "ONNX Runtime GenAI library: ${OGA_LIB}")
        message(STATUS "ONNX Runtime GenAI include: ${OGA_INCLUDE}")
    else()
        message(WARNING "Ryzen AI is only supported on Windows. Disabling Ryzen AI support.")
        set(BUILD_RYZEN OFF CACHE BOOL "Build with Ryzen AI support" FORCE)
    endif()
endif()

include(FetchContent)

# MLX dependencies
if(BUILD_MLX)
    if(NOT APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "BUILD_MLX is only supported on macOS and Linux")
    endif()

    set(CMAKE_POLICY_VERSION_MINIMUM "3.5")
    # SentencePiece library
    message(STATUS "Fetching SentencePiece...")
    FetchContent_Declare(
        sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG v0.2.0
        GIT_SHALLOW FALSE      # <--- Faster download
        GIT_PROGRESS TRUE     # <--- SHOWS YOU THE DOWNLOAD BAR
    )
    FetchContent_MakeAvailable(sentencepiece)
    set(SENTENCEPIECE_INCLUDE_DIRS ${sentencepiece_SOURCE_DIR}/src)

    message(STATUS "Fetching MLX...")
    FetchContent_Declare(
        mlx
        GIT_REPOSITORY https://github.com/Geramy/mlx.git
        GIT_TAG        "geramy/rocm-performance-upgrade"
        GIT_SHALLOW FALSE      # <--- Faster download
        GIT_PROGRESS TRUE     # <--- SHOWS YOU THE DOWNLOAD BAR
    )
    # Disable MLX extras to speed up build
    set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    # ROCm support
    if(MLX_ROCM)
        set(HIP_PLATFORM "amd" CACHE STRING "" FORCE)
        set(MLX_BUILD_ROCM ON CACHE BOOL "" FORCE)
        set(CMAKE_PREFIX_PATH "/opt/rocm" CACHE PATH "" FORCE)
        set(MLX_ROCM_ARCHITECTURES "gfx1151;" CACHE STRING "" FORCE)
        # Disable CUDA when ROCm is enabled
        set(MLX_BUILD_CUDA OFF CACHE BOOL "" FORCE)
    else()
        # Disable CUDA for ROCM support branch
        set(MLX_BUILD_CUDA OFF CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(mlx)
    # Override HIP architectures after MLX configuration
    if(MLX_ROCM)
        set(MLX_ROCM_ARCHITECTURES "gfx1151;" CACHE STRING "" FORCE)
        # Add MLX source include path for HIP compilation
        set(CMAKE_HIP_FLAGS "-I${mlx_SOURCE_DIR}" CACHE STRING "" FORCE)
    endif()
    # Set include directories for MLX
    set(MLX_INCLUDE_DIRS ${mlx_SOURCE_DIR}/include)
endif()
# Validate build options after platform-specific checks
if(NOT BUILD_RYZEN AND NOT BUILD_MLX)
    message(FATAL_ERROR "At least one backend must be enabled. Set BUILD_RYZEN=ON or BUILD_MLX=ON")
endif()



# cpp-httplib
set(HTTPLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/cpp-httplib")
if(NOT EXISTS "${HTTPLIB_DIR}/httplib.h")
    file(MAKE_DIRECTORY ${HTTPLIB_DIR})
    message(STATUS "Downloading cpp-httplib...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.26.0/httplib.h"
        "${HTTPLIB_DIR}/httplib.h"
        SHOW_PROGRESS
    )
endif()

# nlohmann/json
set(JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/json")
if(NOT EXISTS "${JSON_DIR}/json.hpp")
    file(MAKE_DIRECTORY ${JSON_DIR})
    message(STATUS "Downloading nlohmann/json...")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
        "${JSON_DIR}/json.hpp"
        SHOW_PROGRESS
    )
endif()

# Model inference sources (organized by architecture family) - only for MLX
set(MODEL_SOURCES "")
set(MLX_CORE_SOURCES "")

if(BUILD_MLX)
    set(MODEL_SOURCES
        # Phi family
        src/mlx/models/phi3_inference.cpp
        # Qwen family
        src/mlx/models/qwen3_inference.cpp
        src/mlx/models/qwen3_moe_inference.cpp
    )

    # MLX core sources
    set(MLX_CORE_SOURCES
        src/mlx/mlx_oga.cpp
        src/mlx/quantization.cpp
        src/mlx/inference_factory.cpp
        src/mlx/tokenizer.cpp
        src/mlx/moe.cpp
        src/mlx/gpu_utils.cpp
    )
endif()

# Backend abstraction sources (conditional compilation)
set(BACKEND_SOURCES "")

# MLX backend (macOS/Linux)
if(BUILD_MLX)
    list(APPEND BACKEND_SOURCES src/backend/mlx_backend.cpp)
endif()

# ONNX backend (Windows with Ryzen AI)
if(BUILD_RYZEN)
    list(APPEND BACKEND_SOURCES src/backend/onnx_backend.cpp)
endif()

# Core application sources
set(CORE_SOURCES
    src/main.cpp
    src/server.cpp
    src/inference_engine.cpp
    src/command_line.cpp
    src/types.cpp
    src/tool_calls.cpp
    src/reasoning.cpp
    src/driver_check.cpp
)

# Combine all sources
set(SOURCES
    ${CORE_SOURCES}
    ${MLX_CORE_SOURCES}
    ${MODEL_SOURCES}
    ${BACKEND_SOURCES}
)

# SentencePiece is header-only for our use case

# Create executable
add_executable(ryzenai-server ${SOURCES})

# Include directories
target_include_directories(ryzenai-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HTTPLIB_DIR}
    ${JSON_DIR}
)

if(BUILD_RYZEN)
    target_include_directories(ryzenai-server PRIVATE ${OGA_INCLUDE})
endif()

if(BUILD_MLX)
    # MLX-specific include directories (library + internal headers)
    target_include_directories(ryzenai-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ryzenai/mlx
        ${MLX_INCLUDE_DIRS}
        ${SENTENCEPIECE_INCLUDE_DIRS}
    )
endif()

# Link libraries
target_link_libraries(ryzenai-server PRIVATE Threads::Threads)

if(BUILD_RYZEN)
    target_link_libraries(ryzenai-server PRIVATE ${OGA_LIB})
endif()

if(BUILD_MLX)
    target_link_libraries(ryzenai-server PRIVATE mlx sentencepiece-static)
endif()

# Enable multi-threading for cpp-httplib
target_compile_definitions(ryzenai-server PRIVATE CPPHTTPLIB_THREAD_POOL_COUNT=8)

# Backend-specific compile definitions
if(BUILD_RYZEN)
    target_compile_definitions(ryzenai-server PRIVATE BUILD_RYZEN RYZENAI_ON)
endif()

if(BUILD_MLX)
    target_compile_definitions(ryzenai-server PRIVATE BUILD_MLX MLX_ON USE_MLX)
    if(MLX_CUDA)
        target_compile_definitions(ryzenai-server PRIVATE MLX_CUDA_ON)
    endif()
    if(MLX_ROCM)
        target_compile_definitions(ryzenai-server PRIVATE MLX_ROCM_ON)
    endif()
endif()

# Windows-specific settings
if(WIN32)
    # Target Windows 10 or later (required for cpp-httplib v0.26.0)
    target_compile_definitions(ryzenai-server PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(ryzenai-server PRIVATE ws2_32 wbemuuid ole32 oleaut32 shell32)

    # Set console subsystem explicitly
    set_target_properties(ryzenai-server PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
    )

    # Ryzen AI specific Windows settings
    if(BUILD_RYZEN)
        # Copy required DLLs to output directory (they're in deployment folder)
        set(REQUIRED_DLLS
            "${OGA_ROOT}/deployment/onnxruntime-genai.dll"
            "${OGA_ROOT}/deployment/onnxruntime.dll"
            "${OGA_ROOT}/deployment/onnxruntime_providers_shared.dll"
            "${OGA_ROOT}/deployment/onnxruntime_providers_vitisai.dll"
            "${OGA_ROOT}/deployment/onnxruntime_vitisai_ep.dll"
            "${OGA_ROOT}/deployment/onnxruntime_vitis_ai_custom_ops.dll"
            "${OGA_ROOT}/deployment/onnx_custom_ops.dll"
            "${OGA_ROOT}/deployment/abseil_dll.dll"
            "${OGA_ROOT}/deployment/libutf8_validity.dll"
            "${OGA_ROOT}/deployment/dyn_dispatch_core.dll"
            "${OGA_ROOT}/deployment/DirectML.dll"
            "${OGA_ROOT}/deployment/zlib.dll"
            "${OGA_ROOT}/deployment/zstd.dll"
            "${OGA_ROOT}/deployment/ryzenai_onnx_utils.dll"
            "${OGA_ROOT}/deployment/ryzen_mm.dll"
            "${OGA_ROOT}/deployment/aiecompiler_client.dll"
        )

        foreach(DLL ${REQUIRED_DLLS})
            if(EXISTS ${DLL})
                add_custom_command(TARGET ryzenai-server POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${DLL}
                    $<TARGET_FILE_DIR:ryzenai-server>
                    COMMENT "Copying ${DLL}"
                )
            else()
                message(WARNING "DLL not found: ${DLL}")
            endif()
        endforeach()

        # Copy AMD_LICENSE to output directory
        add_custom_command(TARGET ryzenai-server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/AMD_LICENSE
            $<TARGET_FILE_DIR:ryzenai-server>/AMD_LICENSE
            COMMENT "Copying AMD_LICENSE"
        )
    endif()
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(ryzenai-server PRIVATE /W4)
else()
    target_compile_options(ryzenai-server PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Installation rules
install(TARGETS ryzenai-server
    RUNTIME DESTINATION bin
)

# Print build information
message(STATUS "==========================================================")
message(STATUS "Ryzen AI LLM Server Configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
if(BUILD_RYZEN)
    message(STATUS "  Ryzen AI:          ENABLED (${OGA_ROOT})")
else()
    message(STATUS "  Ryzen AI:          DISABLED")
endif()
if(BUILD_MLX)
    message(STATUS "  MLX:               ENABLED")
    if(MLX_ROCM)
        message(STATUS "  MLX ROCm:          ENABLED (${CMAKE_HIP_ARCHITECTURES})")
    else()
        message(STATUS "  MLX ROCm:          DISABLED")
    endif()
else()
    message(STATUS "  MLX:               DISABLED")
endif()
message(STATUS "  Output directory:  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "==========================================================")