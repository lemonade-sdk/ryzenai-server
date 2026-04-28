cmake_minimum_required(VERSION 3.20)
project(ryzenai-server VERSION 1.7.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Building Ryzen AI LLM Server")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find dependencies
find_package(Threads REQUIRED)

# ONNX Runtime GenAI
# Priority order: RYZENAI_INSTALL_PATH env var > -DOGA_ROOT > platform default
if(DEFINED ENV{RYZENAI_INSTALL_PATH})
    set(OGA_ROOT_DEFAULT "$ENV{RYZENAI_INSTALL_PATH}")
    message(STATUS "Using RYZENAI_INSTALL_PATH from environment: ${OGA_ROOT_DEFAULT}")
    # Warn if both env var and explicit -DOGA_ROOT are set
    if(DEFINED OGA_ROOT AND NOT OGA_ROOT STREQUAL "$ENV{RYZENAI_INSTALL_PATH}")
        message(WARNING "Both RYZENAI_INSTALL_PATH and -DOGA_ROOT are set. Environment variable takes precedence: $ENV{RYZENAI_INSTALL_PATH}")
    endif()
    # Force override cached value if environment variable is set
    set(OGA_ROOT "$ENV{RYZENAI_INSTALL_PATH}" CACHE PATH "Path to Ryzen AI installation" FORCE)
elseif(WIN32)
    set(OGA_ROOT_DEFAULT "C:/Program Files/RyzenAI/1.7.0")
    set(OGA_ROOT "${OGA_ROOT_DEFAULT}" CACHE PATH "Path to Ryzen AI installation")
elseif(UNIX AND NOT APPLE)
    set(OGA_ROOT_DEFAULT "/opt/ryzenai/1.7.0")
    set(OGA_ROOT "${OGA_ROOT_DEFAULT}" CACHE PATH "Path to Ryzen AI installation")
endif()

if(NOT EXISTS ${OGA_ROOT})
    message(FATAL_ERROR "Ryzen AI not found at ${OGA_ROOT}. Please install Ryzen AI 1.7.0 or set OGA_ROOT")
endif()

message(STATUS "Using Ryzen AI from: ${OGA_ROOT}")

# Find ONNX Runtime GenAI library
# - Windows installer: ${OGA_ROOT}/LLM/{lib,include}
# - Linux installer:   ${OGA_ROOT}/deployment/lib (no headers shipped)
if(WIN32)
    find_library(OGA_LIB
        NAMES onnxruntime-genai
        PATHS ${OGA_ROOT}/LLM/lib
        REQUIRED
        NO_DEFAULT_PATH
    )
    find_path(OGA_INCLUDE
        NAMES ort_genai.h
        PATHS ${OGA_ROOT}/LLM/include
        REQUIRED
        NO_DEFAULT_PATH
    )
else()
    find_library(OGA_LIB
        NAMES onnxruntime-genai
        PATHS ${OGA_ROOT}/deployment/lib
        REQUIRED
        NO_DEFAULT_PATH
    )
    # Linux SDK does not ship ort_genai.h / ort_genai_c.h. Vendor them from
    # microsoft/onnxruntime-genai matching the OGA library version (0.11.2).
    set(OGA_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/external/oga")
    foreach(HDR ort_genai.h ort_genai_c.h)
        if(NOT EXISTS "${OGA_INCLUDE}/${HDR}")
            message(STATUS "Downloading ${HDR}...")
            file(DOWNLOAD
                "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v0.11.2/src/${HDR}"
                "${OGA_INCLUDE}/${HDR}"
                SHOW_PROGRESS
                STATUS HDR_DOWNLOAD_STATUS
            )
            list(GET HDR_DOWNLOAD_STATUS 0 HDR_DOWNLOAD_CODE)
            if(NOT HDR_DOWNLOAD_CODE EQUAL 0)
                message(FATAL_ERROR "Failed to download ${HDR}: ${HDR_DOWNLOAD_STATUS}")
            endif()
        endif()
    endforeach()
endif()

message(STATUS "ONNX Runtime GenAI library: ${OGA_LIB}")
message(STATUS "ONNX Runtime GenAI include: ${OGA_INCLUDE}")

# Download header-only libraries if not present
include(FetchContent)

# cpp-httplib
set(HTTPLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/cpp-httplib")
if(NOT EXISTS "${HTTPLIB_DIR}/httplib.h")
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
    message(STATUS "Downloading nlohmann/json...")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
        "${JSON_DIR}/json.hpp"
        SHOW_PROGRESS
    )
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/server.cpp
    src/inference_engine.cpp
    src/command_line.cpp
    src/types.cpp
    src/tool_calls.cpp
    src/reasoning.cpp
    src/driver_check.cpp
)

# Create executable
add_executable(ryzenai-server ${SOURCES})

# Include directories
target_include_directories(ryzenai-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OGA_INCLUDE}
    ${HTTPLIB_DIR}
    ${JSON_DIR}
)

# Link libraries
target_link_libraries(ryzenai-server PRIVATE
    ${OGA_LIB}
    Threads::Threads
)

# Pass version info from CMake to C++ as compile definitions
target_compile_definitions(ryzenai-server PRIVATE
    CPPHTTPLIB_THREAD_POOL_COUNT=8
    RYZENAI_SERVER_VERSION="${PROJECT_VERSION}"
)

# Windows-specific settings
if(WIN32)
    # Target Windows 10 or later (required for cpp-httplib v0.26.0)
    target_compile_definitions(ryzenai-server PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(ryzenai-server PRIVATE ws2_32 wbemuuid ole32 oleaut32 shell32)
    
    # Copy required DLLs to output directory (they're in deployment folder)
    set(REQUIRED_DLLS
        "${OGA_ROOT}/deployment/onnxruntime-genai.dll"
        "${OGA_ROOT}/deployment/onnxruntime.dll"
        "${OGA_ROOT}/deployment/onnxruntime_providers_shared.dll"
        "${OGA_ROOT}/deployment/onnxruntime_providers_vitisai.dll"
        "${OGA_ROOT}/deployment/onnxruntime_providers_ryzenai.dll"
        "${OGA_ROOT}/deployment/onnxruntime_vitisai_ep.dll"
        "${OGA_ROOT}/deployment/onnxruntime_vitis_ai_custom_ops.dll"
        "${OGA_ROOT}/deployment/onnx_custom_ops.dll"
        "${OGA_ROOT}/deployment/dyn_dispatch_core.dll"
        "${OGA_ROOT}/deployment/DirectML.dll"
        "${OGA_ROOT}/deployment/D3D12Core.dll"
        "${OGA_ROOT}/deployment/flexmlrt.dll"
        "${OGA_ROOT}/deployment/cert_dtrace.dll"
        "${OGA_ROOT}/deployment/zlib.dll"
        "${OGA_ROOT}/deployment/zstd.dll"
        "${OGA_ROOT}/deployment/ryzenai_onnx_utils.dll"
        "${OGA_ROOT}/deployment/ryzen_mm.dll"
        "${OGA_ROOT}/deployment/aiecompiler_client.dll"
    )
    
    # Set console subsystem explicitly
    set_target_properties(ryzenai-server PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
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
elseif(UNIX AND NOT APPLE)
    target_link_libraries(ryzenai-server PRIVATE dl)

    # Resolve all .so files via $ORIGIN. Use DT_RPATH (not DT_RUNPATH) so the
    # path propagates to transitive deps — libonnxruntime-genai.so itself
    # NEEDs libryzen_mm.so.1, and the SDK libs ship with broken DT_RUNPATHs
    # (CI build paths) that won't exist on user systems.
    target_link_options(ryzenai-server PRIVATE "-Wl,--disable-new-dtags")
    set_target_properties(ryzenai-server PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )

    # Copy the entire deployment/lib tree next to the binary. Preserves the
    # SONAME symlink chain (e.g. libryzen_mm.so -> .so.1 -> .so.1.0.0) which
    # individual file copies would lose.
    if(NOT EXISTS "${OGA_ROOT}/deployment/lib")
        message(FATAL_ERROR "Expected ${OGA_ROOT}/deployment/lib to exist")
    endif()
    add_custom_command(TARGET ryzenai-server POST_BUILD
        COMMAND cp -a ${OGA_ROOT}/deployment/lib/. $<TARGET_FILE_DIR:ryzenai-server>/
        COMMENT "Copying Ryzen AI deployment libraries"
    )

    # Copy AMD_LICENSE to output directory
    add_custom_command(TARGET ryzenai-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/AMD_LICENSE
        $<TARGET_FILE_DIR:ryzenai-server>/AMD_LICENSE
        COMMENT "Copying AMD_LICENSE"
    )
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
message(STATUS "  Ryzen AI path:     ${OGA_ROOT}")
message(STATUS "  Output directory:  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "==========================================================")

