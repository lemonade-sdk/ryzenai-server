#!/bin/bash

# ROCm Kernel Performance Profiling Script
# This script uses rocprof-compute for detailed kernel performance analysis

echo "=== ROCm Kernel Performance Profiler ==="
echo "Press Ctrl+C to stop profiling and exit"
echo ""

# Set default model if not provided
MODEL_PATH="${1:-models/Qwen3-4B-MLX-4bit}"
BACKEND="${2:-mlx-rocm}"

echo "Profiling model: $MODEL_PATH"
echo "Backend: $BACKEND"
echo ""

# Trap SIGINT (Ctrl+C) to clean up
trap 'echo ""; echo "Stopping profiling..."; exit 0' INT

# Check if rocprof-compute is available and supports the GPU architecture
if command -v /opt/rocm/bin/rocprof-compute &> /dev/null; then
    echo "Testing rocprof-compute compatibility..."
    # Try to run rocprof-compute briefly to check if it supports the GPU
    if timeout 5 /opt/rocm/bin/rocprof-compute profile --name test 2>&1 | grep -q "supported architectures"; then
        echo "GPU architecture not supported by rocprof-compute, falling back to rocprofv3..."
        USE_ROCPROF_COMPUTE=false
    else
        echo "Using rocprof-compute for detailed kernel analysis..."
        USE_ROCPROF_COMPUTE=true
    fi
else
    echo "rocprof-compute not available, using rocprofv3..."
    USE_ROCPROF_COMPUTE=false
fi

if [ "$USE_ROCPROF_COMPUTE" = true ]; then
    /opt/rocm/bin/rocprof-compute \
        profile \
        --name ryzenai-profile \
        -- \
        ./build/bin/ryzenai-server \
        -m "$MODEL_PATH" \
        --backend "$BACKEND" \
        --ctx-size 32768
else
    /opt/rocm/bin/rocprofv3 \
        --hip-trace \
        --marker-trace \
        -S \
        -- \
        ./build/bin/ryzenai-server \
        -m "$MODEL_PATH" \
        --backend "$BACKEND" \
        --ctx-size 32768
fi

echo "Kernel profiling completed."
