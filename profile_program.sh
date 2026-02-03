#!/bin/bash

# ROCm Program Performance Profiling Script
# This script uses rocprof-sys-run for system-wide performance analysis

echo "=== ROCm Program Performance Profiler ==="
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

# Run rocprof-sys-run for system-wide profiling
echo "Starting system-wide profiling..."
/opt/rocm/bin/rocprof-sys-run \
    -- \
    ./build/bin/ryzenai-server \
    -m "$MODEL_PATH" \
    --backend "$BACKEND" \
    --ctx-size 32768

echo "System profiling completed."