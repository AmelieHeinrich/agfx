/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 17:41:24
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <stdint.h>

typedef enum agfxShaderStage {
    AGFX_SHADER_STAGE_VERTEX,
    AGFX_SHADER_STAGE_FRAGMENT,
    AGFX_SHADER_STAGE_COMPUTE,
    AGFX_SHADER_STAGE_TASK,
    AGFX_SHADER_STAGE_MESH
} agfxShaderStage;

typedef struct agfxShaderCompilerOptions {
    agfxShaderStage stage;
    char entryPoint[256];

    char* defines[256];
    uint32_t definesCount;

    char* sourceCode;
    uint32_t sourceCodeSize;

    uint8_t usePointTopology;
    uint8_t addDebugSymbols;
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
