/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 17:41:24
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Allocation function signature for agfxDeviceCreate.
/// @param size The number of bytes to allocate.
/// @param userData The agfxDeviceCreateInfo::userData pointer, passed through verbatim.
typedef void* (*agfxAllocate)(uint64_t size, void* userData);

inline void* defaultAlloc(uint64_t size, void* userData) { return malloc(size); }

typedef enum agfxShaderStage {
    AGFX_SHADER_STAGE_VERTEX,
    AGFX_SHADER_STAGE_FRAGMENT,
    AGFX_SHADER_STAGE_COMPUTE,
    AGFX_SHADER_STAGE_TASK,
    AGFX_SHADER_STAGE_MESH
} agfxShaderStage;

typedef struct agfxShaderCompilerOptions {
    /// @brief The allocation function to use for device memory allocations.
    agfxAllocate allocate;
    /// @brief Optional user pointer handed back verbatim to allocate
    ///        Leave nullptr if unused.
    void* userData;
    agfxShaderStage stage;
    char entryPoint[256];

    char* defines[256];
    uint32_t definesCount;

    char* sourceCode;
    uint32_t sourceCodeSize;

    uint8_t usePointTopology;
    uint8_t addDebugSymbols;

    /// @brief Path to the dxcompiler shared library to dlopen on Linux. Ignored on Windows/macOS,
    ///        which link DXC statically. If NULL/empty on Linux, defaults to "data/dlls/libdxcompiler.so".
#if defined(__linux__) && defined(DLOPEN)
    const char* dxCompilerPath;
#endif
} agfxShaderCompilerOptions;

typedef struct agfxShaderCompilerResult {
    uint8_t* compiledCode;
    uint32_t compiledSize;

    uint32_t tgSizeX;
    uint32_t tgSizeY;
    uint32_t tgSizeZ;

    uint32_t meshSizeX;
    uint32_t meshSizeY;
    uint32_t meshSizeZ;

    uint32_t taskSizeX;
    uint32_t taskSizeY;
    uint32_t taskSizeZ;
} agfxShaderCompilerResult;

void agfxCompileShader(agfxShaderCompilerOptions* options, agfxShaderCompilerResult* output);

#ifdef __cplusplus
}
#endif
