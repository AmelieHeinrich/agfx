/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "culling.h"
#include "demo_file_utils.h"

#include <agfx_shader/agfx_shader_compiler.h>

#include <cstring>
#include <string>

namespace {

struct CullingPushConstants {
    glm::vec4 frustumPlanes[6];
    uint32_t gpuScene;
    uint32_t primitiveCount;
    uint64_t bundleHandle;
};

} // namespace

AgfxCulling::AgfxCulling(agfxDevice* device)
{
    m_device = device;

    std::string source = ReadFile((std::string(kDataDir) + "shaders/demo/culling.hlsl").c_str());

    agfxShaderCompilerOptions options = {};
    options.stage = AGFX_SHADER_STAGE_COMPUTE;
    strncpy(options.entryPoint, "main_cs", sizeof(options.entryPoint) - 1);
    options.sourceCode = source.empty() ? nullptr : const_cast<char*>(&source[0]);
    options.sourceCodeSize = (uint32_t)source.size();

    agfxShaderCompilerResult result = {};
    agfxCompileShader(&options, &result);

    agfxShaderModuleCreateInfo moduleInfo = {};
    moduleInfo.code = result.compiledCode;
    moduleInfo.codeSize = result.compiledSize;
    moduleInfo.entryPoint = "main_cs";
    moduleInfo.type = AGFX_SHADER_MODULE_TYPE_COMPUTE;
    computeShader = agfxShaderModuleCreate(device, &moduleInfo);
    free(result.compiledCode);

    agfxComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.name = "Culling Pipeline";
    pipelineInfo.computeShader = computeShader;
    pipelineInfo.groupSizeX = 64;
    pipelineInfo.groupSizeY = 1;
    pipelineInfo.groupSizeZ = 1;
    pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
}

AgfxCulling::~AgfxCulling()
{
    if (pipeline) agfxComputePipelineDestroy(m_device, pipeline);
    if (computeShader) agfxShaderModuleDestroy(m_device, computeShader);
}

void AgfxCulling::Cull(agfxComputePass* pass, agfxBufferView* gpuSceneView, uint32_t primitiveCount,
                        const glm::vec4 frustumPlanes[6], uint64_t bundleHandle)
{
    agfxComputePassSetPipeline(pass, pipeline);

    CullingPushConstants pc = {};
    memcpy(pc.frustumPlanes, frustumPlanes, sizeof(pc.frustumPlanes));
    pc.gpuScene = (uint32_t)agfxBufferViewGetHandle(gpuSceneView);
    pc.primitiveCount = primitiveCount;
    pc.bundleHandle = bundleHandle;
    agfxComputePassPushConstants(pass, &pc, sizeof(pc));

    agfxComputePassDispatch(pass, (primitiveCount + 63) / 64, 1, 1);
}
