/*
 * mlx_oga.h
 * 
 * Main header for the MLX OGA (ONNX GenAI compatible) backend.
 * Aggregates all MLX inference components.
 * Only compiled when USE_MLX is defined (macOS with Apple Silicon).
 */

#pragma once

#ifdef USE_MLX

#include "ryzenai/mlx/common.h"
#include "ryzenai/mlx/model.h"
#include "ryzenai/mlx/tokenizer.h"
#include "ryzenai/mlx/generator.h"

#endif // USE_MLX
