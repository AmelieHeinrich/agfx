/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "reflections.h"
#include "demo_file_utils.h"

#include <agfx_shader/agfx_shader_compiler.h>

#include <string>

namespace {

struct ReflectionsPushConstants {
    uint32_t tlas;
    uint32_t gpuScene;
    uint32_t indexBuffer;
    uint32_t vertexBuffer;
    uint32_t gbufferSampler;
    uint32_t materialSampler;
    uint32_t albedoTex;
    uint32_t normalTex;
    uint32_t mraTex;
    uint32_t depthTex;
    uint32_t reflOut;
    uint32_t sceneCB;
    uint32_t shadowMap;
    uint32_t cascadeCB;
    uint32_t shadowComparisonSampler;
    uint32_t pointSampler;
    float metallicThreshold;
    float roughnessThreshold;
    uint32_t width;
    uint32_t height;
};

} // namespace

AgfxReflections::AgfxReflections(agfxDevice* device)
{
    m_device = device;

    std::string source = ReadFile((std::string(kDataDir) + "shaders/demo/reflections.hlsl").c_str());

    agfxShaderCompilerOptions options = {};
    options.stage = AGFX_SHADER_STAGE_COMPUTE;
    strncpy(options.entryPoint, "main_cs", sizeof(options.entryPoint) - 1);
    options.sourceCode = source.empty() ? nullptr : const_cast<char*>(&source[0]);
    options.sourceCodeSize = (uint32_t)source.size();
    options.addDebugSymbols = true;

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
    pipelineInfo.name = "Reflections Pipeline";
    pipelineInfo.computeShader = computeShader;
    pipelineInfo.groupSizeX = 8;
    pipelineInfo.groupSizeY = 8;
    pipelineInfo.groupSizeZ = 1;
    pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
}

AgfxReflections::~AgfxReflections()
{
    if (pipeline) agfxComputePipelineDestroy(m_device, pipeline);
    if (computeShader) agfxShaderModuleDestroy(m_device, computeShader);
}

void AgfxReflections::Generate(agfxDevice* device, agfxCommandBuffer* cmdBuffer,
                                agfxAccelerationStructure* tlas, agfxBufferView* gpuSceneView,
                                agfxBufferView* indexBufferView, agfxBufferView* vertexBufferView,
                                agfxTextureView* albedoSRV, agfxTextureView* normalSRV, agfxTextureView* mraSRV, agfxTextureView* depthSRV,
                                agfxTextureView* reflUAV, agfxTexture* reflTexture,
                                agfxBufferView* sceneCBView, agfxTextureView* shadowMapView, agfxBufferView* cascadeCBView,
                                agfxSampler* gbufferSampler, agfxSampler* materialSampler, agfxSampler* shadowComparisonSampler, agfxSampler* pointSampler,
                                float metallicThreshold, float roughnessThreshold,
                                uint32_t width, uint32_t height)
{
    (void)device;

    agfxComputePass* pass = agfxComputePassBegin(cmdBuffer, "Reflections");
    agfxComputePassSetPipeline(pass, pipeline);

    ReflectionsPushConstants pc = {};
    pc.tlas = (uint32_t)agfxAccelerationStructureGetHandle(tlas);
    pc.gpuScene = (uint32_t)agfxBufferViewGetHandle(gpuSceneView);
    pc.indexBuffer = (uint32_t)agfxBufferViewGetHandle(indexBufferView);
    pc.vertexBuffer = (uint32_t)agfxBufferViewGetHandle(vertexBufferView);
    pc.gbufferSampler = (uint32_t)agfxSamplerGetHandle(gbufferSampler);
    pc.materialSampler = (uint32_t)agfxSamplerGetHandle(materialSampler);
    pc.albedoTex = (uint32_t)agfxTextureViewGetHandle(albedoSRV);
    pc.normalTex = (uint32_t)agfxTextureViewGetHandle(normalSRV);
    pc.mraTex = (uint32_t)agfxTextureViewGetHandle(mraSRV);
    pc.depthTex = (uint32_t)agfxTextureViewGetHandle(depthSRV);
    pc.reflOut = (uint32_t)agfxTextureViewGetHandle(reflUAV);
    pc.sceneCB = (uint32_t)agfxBufferViewGetHandle(sceneCBView);
    pc.shadowMap = (uint32_t)agfxTextureViewGetHandle(shadowMapView);
    pc.cascadeCB = (uint32_t)agfxBufferViewGetHandle(cascadeCBView);
    pc.shadowComparisonSampler = (uint32_t)agfxSamplerGetHandle(shadowComparisonSampler);
    pc.pointSampler = (uint32_t)agfxSamplerGetHandle(pointSampler);
    pc.metallicThreshold = metallicThreshold;
    pc.roughnessThreshold = roughnessThreshold;
    pc.width = width;
    pc.height = height;
    agfxComputePassPushConstants(pass, &pc, sizeof(pc));

    agfxComputePassDispatch(pass, (width + 7) / 8, (height + 7) / 8, 1);
    agfxComputePassTextureUAVBarrier(pass, reflTexture);
    agfxComputePassEnd(pass);
}
