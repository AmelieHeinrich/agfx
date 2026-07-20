/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "ssao.h"
#include "demo_file_utils.h"

#include <agfx_shader/agfx_shader_compiler.h>

#include <string>

namespace {

struct SSAOPushConstants {
    uint32_t normalTex;
    uint32_t depthTex;
    uint32_t aoTex;
    uint32_t sceneCB;
    uint32_t pointSampler;
};

} // namespace

AgfxSSAO::AgfxSSAO(agfxDevice* device)
{
    m_device = device;

    std::string source = ReadFile((std::string(kDataDir) + "shaders/demo/ssao.hlsl").c_str());

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
    pipelineInfo.name = "SSAO Pipeline";
    pipelineInfo.computeShader = computeShader;
    pipelineInfo.groupSizeX = 8;
    pipelineInfo.groupSizeY = 8;
    pipelineInfo.groupSizeZ = 1;
    pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
}

AgfxSSAO::~AgfxSSAO()
{
    if (pipeline) agfxComputePipelineDestroy(m_device, pipeline);
    if (computeShader) agfxShaderModuleDestroy(m_device, computeShader);
}

void AgfxSSAO::Generate(agfxDevice* device, agfxCommandBuffer* cmdBuffer,
                         agfxTextureView* normalSRV, agfxTextureView* depthSRV, agfxTextureView* aoUAV, agfxTexture* aoTexture,
                         agfxBufferView* sceneCBView, agfxSampler* pointSampler,
                         uint32_t width, uint32_t height)
{
    (void)device;

    agfxComputePass* pass = agfxComputePassBegin(cmdBuffer, "SSAO");
    agfxComputePassSetPipeline(pass, pipeline);

    SSAOPushConstants pc = {};
    pc.normalTex = (uint32_t)agfxTextureViewGetHandle(normalSRV);
    pc.depthTex = (uint32_t)agfxTextureViewGetHandle(depthSRV);
    pc.aoTex = (uint32_t)agfxTextureViewGetHandle(aoUAV);
    pc.sceneCB = (uint32_t)agfxBufferViewGetHandle(sceneCBView);
    pc.pointSampler = (uint32_t)agfxSamplerGetHandle(pointSampler);
    agfxComputePassPushConstants(pass, &pc, sizeof(pc));

    agfxComputePassDispatch(pass, (width + 7) / 8, (height + 7) / 8, 1);
    agfxComputePassTextureUAVBarrier(pass, aoTexture);
    agfxComputePassEnd(pass);
}
