/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "deferred_renderer.h"
#include "demo_file_utils.h"

#include <agfx_shader/agfx_shader_compiler.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cfloat>
#include <cstring>
#include <string>

namespace {

struct GBufferPushConstants {
    glm::mat4 worldMatrix;
    uint32_t vertexBuffer;
    uint32_t albedoTex;
    uint32_t normalTex;
    uint32_t metallicRoughnessTex;
    uint32_t textureSampler;
    uint32_t sceneCB;
    uint32_t vertexOffset;
    float metallicFactor;
    float roughnessFactor;
};

struct GBufferSceneConstants {
    glm::mat4 viewProj;
};

struct LightingPushConstants {
    uint32_t albedoTex;
    uint32_t normalTex;
    uint32_t mraTex;
    uint32_t depthTex;
    uint32_t textureSampler;
    uint32_t lightCB;
    uint32_t shadowMap;
    uint32_t cascadeCB;
    uint32_t shadowComparisonSampler;
    uint32_t pointSampler;
    uint32_t aoTex;
};

struct SSAOSceneConstants {
    glm::mat4 viewProj;
    glm::mat4 invViewProj;
    glm::vec2 screenSize;
    float radius;
    float bias;
    float power;
    uint32_t enabled;
    float _pad0[2];
};

struct LightingSceneConstants {
    glm::mat4 invViewProj;
    glm::vec3 cameraPos;
    float _pad0;
    glm::vec3 lightDir;
    float _pad1;
    glm::vec3 lightColor;
    float lightIntensity;
};

struct CascadeConstants {
    glm::mat4 viewProj[kMaxCascades];
    glm::vec4 splitFar;
    glm::vec4 texelWorldSize;
    uint32_t cascadeCount;
    float shadowMapResolution;
    float depthBiasConstant;
    float lightSizeUV;
    float pcssMaxPenumbraUV;
    uint32_t visualizeCascades;
    float _pad0[2];
};

struct ShadowPushConstants {
    glm::mat4 worldMatrix;
    uint32_t vertexBuffer;
    uint32_t cascadeCB;
    uint32_t cascadeIndex;
    uint32_t vertexOffset;
};

struct TonemapPushConstants {
    uint32_t hdrTex;
    uint32_t textureSampler;
    uint32_t isHDR;
};

agfxShaderModule* CompileShader(agfxDevice* device, const std::string& source, agfxShaderStage stage, const char* entryPoint, agfxShaderModuleType moduleType)
{
    agfxShaderCompilerOptions options = {};
    options.stage = stage;
    strncpy(options.entryPoint, entryPoint, sizeof(options.entryPoint) - 1);
    options.sourceCode = source.empty() ? nullptr : const_cast<char*>(&source[0]);
    options.sourceCodeSize = (uint32_t)source.size();

    agfxShaderCompilerResult result = {};
    agfxCompileShader(&options, &result);

    agfxShaderModuleCreateInfo moduleInfo = {};
    moduleInfo.code = result.compiledCode;
    moduleInfo.codeSize = result.compiledSize;
    moduleInfo.entryPoint = entryPoint;
    moduleInfo.type = moduleType;
    agfxShaderModule* module = agfxShaderModuleCreate(device, &moduleInfo);

    free(result.compiledCode);
    return module;
}

} // namespace

void DeferredRenderer::CreateGBufferTargets(agfxDevice* device, uint32_t width, uint32_t height)
{
    agfxTextureCreateInfo albedoInfo = {};
    albedoInfo.type = AGFX_TEXTURE_TYPE_2D;
    albedoInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    albedoInfo.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
    albedoInfo.width = width;
    albedoInfo.height = height;
    albedoInfo.depthOrArrayLayers = 1;
    albedoInfo.mipLevels = 1;
    albedoTexture = agfxTextureCreate(device, &albedoInfo);

    agfxTextureCreateInfo normalInfo = albedoInfo;
    normalTexture = agfxTextureCreate(device, &normalInfo);

    agfxTextureCreateInfo mraInfo = albedoInfo;
    mraTexture = agfxTextureCreate(device, &mraInfo);

    agfxTextureCreateInfo depthInfo = {};
    depthInfo.type = AGFX_TEXTURE_TYPE_2D;
    depthInfo.format = AGFX_TEXTURE_FORMAT_DEPTH32F;
    depthInfo.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
    depthInfo.width = width;
    depthInfo.height = height;
    depthInfo.depthOrArrayLayers = 1;
    depthInfo.mipLevels = 1;
    depthTexture = agfxTextureCreate(device, &depthInfo);

    agfxTextureCreateInfo hdrInfo = {};
    hdrInfo.type = AGFX_TEXTURE_TYPE_2D;
    hdrInfo.format = AGFX_TEXTURE_FORMAT_RGBA16F;
    hdrInfo.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
    hdrInfo.width = width;
    hdrInfo.height = height;
    hdrInfo.depthOrArrayLayers = 1;
    hdrInfo.mipLevels = 1;
    hdrTexture = agfxTextureCreate(device, &hdrInfo);

    agfxTextureCreateInfo aoInfo = {};
    aoInfo.type = AGFX_TEXTURE_TYPE_2D;
    aoInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    aoInfo.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_SAMPLED | AGFX_TEXTURE_USAGE_STORAGE);
    aoInfo.width = width;
    aoInfo.height = height;
    aoInfo.depthOrArrayLayers = 1;
    aoInfo.mipLevels = 1;
    aoTexture = agfxTextureCreate(device, &aoInfo);

    auto makeViews = [&](agfxTexture* tex, agfxTextureFormat fmt, agfxTextureView** sampledView, agfxRenderTarget** rt, bool isDepth) {
        agfxTextureViewCreateInfo viewInfo = {};
        viewInfo.texture = tex;
        viewInfo.format = fmt;
        viewInfo.type = AGFX_TEXTURE_TYPE_2D;
        viewInfo.mipLevelCount = 1;
        viewInfo.arrayLayerCount = 1;
        *sampledView = agfxTextureViewCreate(device, &viewInfo);

        agfxRenderTargetCreateInfo rtInfo = {};
        rtInfo.texture = tex;
        rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
        rtInfo.isDepth = isDepth;
        *rt = agfxRenderTargetCreate(device, &rtInfo);
    };

    makeViews(albedoTexture, AGFX_TEXTURE_FORMAT_RGBA8_UNORM, &albedoView, &albedoRT, false);
    makeViews(normalTexture, AGFX_TEXTURE_FORMAT_RGBA8_UNORM, &normalView, &normalRT, false);
    makeViews(mraTexture, AGFX_TEXTURE_FORMAT_RGBA8_UNORM, &mraView, &mraRT, false);
    makeViews(depthTexture, AGFX_TEXTURE_FORMAT_R32F, &depthView, &depthRT, true);
    makeViews(hdrTexture, AGFX_TEXTURE_FORMAT_RGBA16F, &hdrView, &hdrRT, false);

    agfxTextureViewCreateInfo aoViewInfo = {};
    aoViewInfo.texture = aoTexture;
    aoViewInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    aoViewInfo.type = AGFX_TEXTURE_TYPE_2D;
    aoViewInfo.mipLevelCount = 1;
    aoViewInfo.arrayLayerCount = 1;
    aoViewInfo.writeable = false;
    aoView = agfxTextureViewCreate(device, &aoViewInfo);

    agfxTextureViewCreateInfo aoUAVInfo = aoViewInfo;
    aoUAVInfo.writeable = true;
    aoUAVView = agfxTextureViewCreate(device, &aoUAVInfo);

    agfxDeviceMakeResourcesResident(device);
}

void DeferredRenderer::DestroyGBufferTargets(agfxDevice* device)
{
    if (albedoRT) agfxRenderTargetDestroy(device, albedoRT);
    if (albedoView) agfxTextureViewDestroy(device, albedoView);
    if (albedoTexture) agfxTextureDestroy(device, albedoTexture);

    if (normalRT) agfxRenderTargetDestroy(device, normalRT);
    if (normalView) agfxTextureViewDestroy(device, normalView);
    if (normalTexture) agfxTextureDestroy(device, normalTexture);

    if (mraRT) agfxRenderTargetDestroy(device, mraRT);
    if (mraView) agfxTextureViewDestroy(device, mraView);
    if (mraTexture) agfxTextureDestroy(device, mraTexture);

    if (depthRT) agfxRenderTargetDestroy(device, depthRT);
    if (depthView) agfxTextureViewDestroy(device, depthView);
    if (depthTexture) agfxTextureDestroy(device, depthTexture);

    if (hdrRT) agfxRenderTargetDestroy(device, hdrRT);
    if (hdrView) agfxTextureViewDestroy(device, hdrView);
    if (hdrTexture) agfxTextureDestroy(device, hdrTexture);

    if (aoUAVView) agfxTextureViewDestroy(device, aoUAVView);
    if (aoView) agfxTextureViewDestroy(device, aoView);
    if (aoTexture) agfxTextureDestroy(device, aoTexture);

    albedoRT = normalRT = mraRT = depthRT = hdrRT = nullptr;
    albedoView = normalView = mraView = depthView = hdrView = nullptr;
    albedoTexture = normalTexture = mraTexture = depthTexture = hdrTexture = nullptr;
    aoUAVView = aoView = nullptr;
    aoTexture = nullptr;
}

void DeferredRenderer::CreateShadowTargets(agfxDevice* device, uint32_t resolution)
{
    agfxTextureCreateInfo shadowInfo = {};
    shadowInfo.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
    shadowInfo.format = AGFX_TEXTURE_FORMAT_DEPTH32F;
    shadowInfo.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
    shadowInfo.width = resolution;
    shadowInfo.height = resolution;
    shadowInfo.depthOrArrayLayers = kMaxCascades;
    shadowInfo.mipLevels = 1;
    shadowTexture = agfxTextureCreate(device, &shadowInfo);

    agfxTextureViewCreateInfo viewInfo = {};
    viewInfo.texture = shadowTexture;
    viewInfo.format = AGFX_TEXTURE_FORMAT_R32F;
    viewInfo.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
    viewInfo.mipLevelCount = 1;
    viewInfo.arrayLayerCount = kMaxCascades;
    shadowArrayView = agfxTextureViewCreate(device, &viewInfo);

    for (uint32_t i = 0; i < kMaxCascades; ++i) {
        agfxRenderTargetCreateInfo rtInfo = {};
        rtInfo.texture = shadowTexture;
        rtInfo.format = AGFX_TEXTURE_FORMAT_DEPTH32F;
        rtInfo.arrayLayer = i;
        rtInfo.isDepth = true;
        shadowCascadeRT[i] = agfxRenderTargetCreate(device, &rtInfo);
    }

    csm.shadowMapResolution = resolution;

    agfxDeviceMakeResourcesResident(device);
}

void DeferredRenderer::DestroyShadowTargets(agfxDevice* device)
{
    for (uint32_t i = 0; i < kMaxCascades; ++i) {
        if (shadowCascadeRT[i]) agfxRenderTargetDestroy(device, shadowCascadeRT[i]);
        shadowCascadeRT[i] = nullptr;
    }
    if (shadowArrayView) agfxTextureViewDestroy(device, shadowArrayView);
    if (shadowTexture) agfxTextureDestroy(device, shadowTexture);
    shadowArrayView = nullptr;
    shadowTexture = nullptr;
}

agfxRenderPipeline* DeferredRenderer::CreateGBufferPipeline(agfxDevice* device, agfxCullMode cullMode)
{
    agfxRenderPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.name = "GBuffer Pipeline";
    pipelineInfo.fillMode = AGFX_FILL_MODE_SOLID;
    pipelineInfo.cullMode = cullMode;
    pipelineInfo.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
    pipelineInfo.topology = AGFX_TOPOLOGY_TRIANGLES;
    pipelineInfo.depthTestEnable = true;
    pipelineInfo.depthWriteEnable = true;
    pipelineInfo.depthCompareOp = AGFX_COMPARISON_FUNCTION_LESS;
    pipelineInfo.depthFormat = AGFX_TEXTURE_FORMAT_DEPTH32F;
    pipelineInfo.colorFormats[0] = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    pipelineInfo.colorFormats[1] = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    pipelineInfo.colorFormats[2] = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    pipelineInfo.colorAttachmentCount = 3;
    pipelineInfo.vertexShader = gbufferVS;
    pipelineInfo.fragmentShader = gbufferPS;
    return agfxRenderPipelineCreate(device, &pipelineInfo);
}

void DeferredRenderer::Init(agfxDevice* device, agfxTextureFormat swapchainFormat, uint32_t initWidth, uint32_t initHeight)
{
    width = initWidth;
    height = initHeight;

    std::string gbufferSource = ReadFile((std::string(kDataDir) + "shaders/demo/gbuffer.hlsl").c_str());
    gbufferVS = CompileShader(device, gbufferSource, AGFX_SHADER_STAGE_VERTEX, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    gbufferPS = CompileShader(device, gbufferSource, AGFX_SHADER_STAGE_FRAGMENT, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
    gbufferPipelineCullBack = CreateGBufferPipeline(device, AGFX_CULL_MODE_BACK);
    gbufferPipelineCullNone = CreateGBufferPipeline(device, AGFX_CULL_MODE_NONE);

    std::string lightingSource = ReadFile((std::string(kDataDir) + "shaders/demo/deferred_lighting.hlsl").c_str());
    lightingVS = CompileShader(device, lightingSource, AGFX_SHADER_STAGE_VERTEX, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    lightingPS = CompileShader(device, lightingSource, AGFX_SHADER_STAGE_FRAGMENT, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);

    agfxRenderPipelineCreateInfo lightingPipelineInfo = {};
    lightingPipelineInfo.name = "Deferred Lighting Pipeline";
    lightingPipelineInfo.fillMode = AGFX_FILL_MODE_SOLID;
    lightingPipelineInfo.cullMode = AGFX_CULL_MODE_NONE;
    lightingPipelineInfo.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
    lightingPipelineInfo.topology = AGFX_TOPOLOGY_TRIANGLES;
    lightingPipelineInfo.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
    lightingPipelineInfo.colorFormats[0] = AGFX_TEXTURE_FORMAT_RGBA16F;
    lightingPipelineInfo.colorAttachmentCount = 1;
    lightingPipelineInfo.vertexShader = lightingVS;
    lightingPipelineInfo.fragmentShader = lightingPS;
    lightingPipeline = agfxRenderPipelineCreate(device, &lightingPipelineInfo);

    std::string shadowSource = ReadFile((std::string(kDataDir) + "shaders/demo/shadow_depth.hlsl").c_str());
    shadowVS = CompileShader(device, shadowSource, AGFX_SHADER_STAGE_VERTEX, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);

    agfxRenderPipelineCreateInfo shadowPipelineInfo = {};
    shadowPipelineInfo.name = "Shadow Depth Pipeline";
    shadowPipelineInfo.fillMode = AGFX_FILL_MODE_SOLID;
    shadowPipelineInfo.cullMode = AGFX_CULL_MODE_NONE;
    shadowPipelineInfo.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
    shadowPipelineInfo.topology = AGFX_TOPOLOGY_TRIANGLES;
    shadowPipelineInfo.depthTestEnable = true;
    shadowPipelineInfo.depthWriteEnable = true;
    shadowPipelineInfo.depthClampEnable = true;
    shadowPipelineInfo.depthCompareOp = AGFX_COMPARISON_FUNCTION_LESS;
    shadowPipelineInfo.depthFormat = AGFX_TEXTURE_FORMAT_DEPTH32F;
    shadowPipelineInfo.colorAttachmentCount = 0;
    shadowPipelineInfo.vertexShader = shadowVS;
    shadowPipelineInfo.fragmentShader = nullptr;
    shadowPipeline = agfxRenderPipelineCreate(device, &shadowPipelineInfo);

    std::string tonemapSource = ReadFile((std::string(kDataDir) + "shaders/demo/tonemap.hlsl").c_str());
    tonemapVS = CompileShader(device, tonemapSource, AGFX_SHADER_STAGE_VERTEX, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    tonemapPS = CompileShader(device, tonemapSource, AGFX_SHADER_STAGE_FRAGMENT, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
    RecreateTonemapPipeline(device, swapchainFormat);

    ssao = new AgfxSSAO(device);

    CreateGBufferTargets(device, width, height);
    CreateShadowTargets(device, csm.shadowMapResolution);

    agfxSamplerCreateInfo samplerInfo = {};
    samplerInfo.filter = AGFX_SAMPLER_FILTER_LINEAR;
    samplerInfo.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
    samplerInfo.maxLod = FLT_MAX;
    pointSampler = agfxSamplerCreate(device, &samplerInfo);

    agfxSamplerCreateInfo shadowSamplerInfo = {};
    shadowSamplerInfo.filter = AGFX_SAMPLER_FILTER_LINEAR;
    shadowSamplerInfo.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.maxAnisotropy = 1.0f;
    shadowSamplerInfo.comparisonFunction = AGFX_COMPARISON_FUNCTION_LESS_EQUAL;
    shadowSamplerInfo.maxLod = 0.0f;
    shadowComparisonSampler = agfxSamplerCreate(device, &shadowSamplerInfo);

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        agfxBufferCreateInfo gcbInfo = {};
        gcbInfo.size = sizeof(GBufferSceneConstants);
        gcbInfo.stride = sizeof(GBufferSceneConstants);
        gcbInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        gcbInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
        gbufferSceneCB[i] = agfxBufferCreate(device, &gcbInfo);
        agfxBufferViewCreateInfo gcbViewInfo = {};
        gcbViewInfo.buffer = gbufferSceneCB[i];
        gcbViewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        gbufferSceneCBView[i] = agfxBufferViewCreate(device, &gcbViewInfo);

        agfxBufferCreateInfo lcbInfo = {};
        lcbInfo.size = sizeof(LightingSceneConstants);
        lcbInfo.stride = sizeof(LightingSceneConstants);
        lcbInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        lcbInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
        lightingSceneCB[i] = agfxBufferCreate(device, &lcbInfo);
        agfxBufferViewCreateInfo lcbViewInfo = {};
        lcbViewInfo.buffer = lightingSceneCB[i];
        lcbViewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        lightingSceneCBView[i] = agfxBufferViewCreate(device, &lcbViewInfo);

        agfxBufferCreateInfo ccbInfo = {};
        ccbInfo.size = sizeof(CascadeConstants);
        ccbInfo.stride = sizeof(CascadeConstants);
        ccbInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        ccbInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
        cascadeCB[i] = agfxBufferCreate(device, &ccbInfo);
        agfxBufferViewCreateInfo ccbViewInfo = {};
        ccbViewInfo.buffer = cascadeCB[i];
        ccbViewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        cascadeCBView[i] = agfxBufferViewCreate(device, &ccbViewInfo);

        agfxBufferCreateInfo scbInfo = {};
        scbInfo.size = sizeof(SSAOSceneConstants);
        scbInfo.stride = sizeof(SSAOSceneConstants);
        scbInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        scbInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
        ssaoSceneCB[i] = agfxBufferCreate(device, &scbInfo);
        agfxBufferViewCreateInfo scbViewInfo = {};
        scbViewInfo.buffer = ssaoSceneCB[i];
        scbViewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        ssaoSceneCBView[i] = agfxBufferViewCreate(device, &scbViewInfo);
    }

    agfxDeviceMakeResourcesResident(device);
}

void DeferredRenderer::Shutdown(agfxDevice* device)
{
    DestroyGBufferTargets(device);
    DestroyShadowTargets(device);

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (gbufferSceneCBView[i]) agfxBufferViewDestroy(device, gbufferSceneCBView[i]);
        if (gbufferSceneCB[i]) agfxBufferDestroy(device, gbufferSceneCB[i]);
        if (lightingSceneCBView[i]) agfxBufferViewDestroy(device, lightingSceneCBView[i]);
        if (lightingSceneCB[i]) agfxBufferDestroy(device, lightingSceneCB[i]);
        if (cascadeCBView[i]) agfxBufferViewDestroy(device, cascadeCBView[i]);
        if (cascadeCB[i]) agfxBufferDestroy(device, cascadeCB[i]);
        if (ssaoSceneCBView[i]) agfxBufferViewDestroy(device, ssaoSceneCBView[i]);
        if (ssaoSceneCB[i]) agfxBufferDestroy(device, ssaoSceneCB[i]);
    }

    if (ssao) {
        delete ssao;
        ssao = nullptr;
    }

    if (pointSampler) agfxSamplerDestroy(device, pointSampler);
    if (shadowComparisonSampler) agfxSamplerDestroy(device, shadowComparisonSampler);

    if (shadowPipeline) agfxRenderPipelineDestroy(device, shadowPipeline);
    if (shadowVS) agfxShaderModuleDestroy(device, shadowVS);

    if (gbufferPipelineCullBack) agfxRenderPipelineDestroy(device, gbufferPipelineCullBack);
    if (gbufferPipelineCullNone) agfxRenderPipelineDestroy(device, gbufferPipelineCullNone);
    if (gbufferVS) agfxShaderModuleDestroy(device, gbufferVS);
    if (gbufferPS) agfxShaderModuleDestroy(device, gbufferPS);

    if (lightingPipeline) agfxRenderPipelineDestroy(device, lightingPipeline);
    if (lightingVS) agfxShaderModuleDestroy(device, lightingVS);
    if (lightingPS) agfxShaderModuleDestroy(device, lightingPS);

    if (tonemapPipeline) agfxRenderPipelineDestroy(device, tonemapPipeline);
    if (tonemapVS) agfxShaderModuleDestroy(device, tonemapVS);
    if (tonemapPS) agfxShaderModuleDestroy(device, tonemapPS);
}

void DeferredRenderer::Resize(agfxDevice* device, uint32_t newWidth, uint32_t newHeight)
{
    if (newWidth == width && newHeight == height) return;
    // Shadow targets are sized from csm.shadowMapResolution, not the screen resolution -
    // intentionally not recreated here. See RecreateShadowTargets.
    DestroyGBufferTargets(device);
    CreateGBufferTargets(device, newWidth, newHeight);
    width = newWidth;
    height = newHeight;
}

void DeferredRenderer::RecreateShadowTargets(agfxDevice* device, uint32_t resolution)
{
    DestroyShadowTargets(device);
    CreateShadowTargets(device, resolution);
}

void DeferredRenderer::RecreateTonemapPipeline(agfxDevice* device, agfxTextureFormat swapchainFormat)
{
    if (tonemapPipeline) agfxRenderPipelineDestroy(device, tonemapPipeline);

    agfxRenderPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.name = "Tonemap Pipeline";
    pipelineInfo.fillMode = AGFX_FILL_MODE_SOLID;
    pipelineInfo.cullMode = AGFX_CULL_MODE_NONE;
    pipelineInfo.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
    pipelineInfo.topology = AGFX_TOPOLOGY_TRIANGLES;
    pipelineInfo.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
    pipelineInfo.colorFormats[0] = swapchainFormat;
    pipelineInfo.colorAttachmentCount = 1;
    pipelineInfo.vertexShader = tonemapVS;
    pipelineInfo.fragmentShader = tonemapPS;
    tonemapPipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
    tonemapOutputFormat = swapchainFormat;
}

void DeferredRenderer::RenderGBuffer(agfxDevice* device, agfxCommandBuffer* cmdBuffer, const GltfScene& scene, const Camera& camera, uint32_t frameSlot)
{
    agfxCommandBufferTextureBarrier(cmdBuffer, albedoTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, normalTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, mraTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, depthTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_DEPTH_WRITE, 0, 0, true);

    GBufferSceneConstants sceneConstants = {};
    sceneConstants.viewProj = camera.GetProj() * camera.GetView();
    void* cbDst = agfxBufferMap(gbufferSceneCB[frameSlot]);
    memcpy(cbDst, &sceneConstants, sizeof(sceneConstants));
    agfxBufferUnmap(gbufferSceneCB[frameSlot]);
    agfxDeviceMakeResourcesResident(device);

    agfxRenderPassCreateInfo passInfo = {};
    passInfo.colorAttachmentCount = 3;
    passInfo.colorAttachments[0].renderTarget = albedoRT;
    passInfo.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
    passInfo.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
    passInfo.colorAttachments[1].renderTarget = normalRT;
    passInfo.colorAttachments[1].loadOp = AGFX_LOAD_OPERATION_CLEAR;
    passInfo.colorAttachments[1].storeOp = AGFX_STORE_OPERATION_STORE;
    passInfo.colorAttachments[2].renderTarget = mraRT;
    passInfo.colorAttachments[2].loadOp = AGFX_LOAD_OPERATION_CLEAR;
    passInfo.colorAttachments[2].storeOp = AGFX_STORE_OPERATION_STORE;
    passInfo.hasDepthAttachment = true;
    passInfo.depthAttachment.renderTarget = depthRT;
    passInfo.depthAttachment.loadOp = AGFX_LOAD_OPERATION_CLEAR;
    passInfo.depthAttachment.storeOp = AGFX_STORE_OPERATION_STORE;
    passInfo.width = width;
    passInfo.height = height;
    passInfo.name = "GBuffer";

    agfxRenderPass* pass = agfxRenderPassBegin(cmdBuffer, &passInfo);
    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    agfxRenderPassSetScissor(pass, 0, 0, width, height);

    for (const ScenePrimitive& prim : scene.primitives) {
        if (prim.materialIndex < 0 || prim.materialIndex >= (int)scene.materials.size())
            continue;
        const SceneMaterial& mat = scene.materials[prim.materialIndex];

        agfxRenderPassSetPipeline(pass, mat.doubleSided ? gbufferPipelineCullNone : gbufferPipelineCullBack);

        GBufferPushConstants pc = {};
        pc.worldMatrix = prim.worldMatrix;
        pc.vertexBuffer = (uint32_t)agfxBufferViewGetHandle(scene.vertexBufferView);
        pc.albedoTex = scene.textures[mat.albedoTexIndex].handle;
        pc.normalTex = scene.textures[mat.normalTexIndex].handle;
        pc.metallicRoughnessTex = scene.textures[mat.metallicRoughnessTexIndex].handle;
        pc.textureSampler = (uint32_t)agfxSamplerGetHandle(scene.defaultSampler);
        pc.sceneCB = (uint32_t)agfxBufferViewGetHandle(gbufferSceneCBView[frameSlot]);
        pc.vertexOffset = prim.vertexOffset;
        pc.metallicFactor = mat.metallicFactor;
        pc.roughnessFactor = mat.roughnessFactor;
        agfxRenderPassPushConstants(pass, &pc, sizeof(pc));

        agfxRenderPassDrawIndexed(pass, scene.indexBuffer, prim.indexCount, 1, prim.indexOffset, 0, 0);
    }

    agfxRenderPassEnd(pass);

    agfxCommandBufferTextureBarrier(cmdBuffer, albedoTexture, AGFX_RESOURCE_STATE_RENDER_TARGET, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, normalTexture, AGFX_RESOURCE_STATE_RENDER_TARGET, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, mraTexture, AGFX_RESOURCE_STATE_RENDER_TARGET, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, depthTexture, AGFX_RESOURCE_STATE_DEPTH_WRITE, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, true);
}

void DeferredRenderer::RenderSSAO(agfxDevice* device, agfxCommandBuffer* cmdBuffer, const Camera& camera, uint32_t frameSlot)
{
    glm::mat4 view = camera.GetView();
    glm::mat4 proj = camera.GetProj();
    glm::mat4 viewProj = proj * view;

    SSAOSceneConstants sceneConstants = {};
    sceneConstants.viewProj = viewProj;
    sceneConstants.invViewProj = glm::inverse(viewProj);
    sceneConstants.screenSize = glm::vec2((float)width, (float)height);
    sceneConstants.radius = ssaoSettings.radius;
    sceneConstants.bias = ssaoSettings.bias;
    sceneConstants.power = ssaoSettings.power;
    sceneConstants.enabled = ssaoSettings.enabled ? 1u : 0u;
    void* cbDst = agfxBufferMap(ssaoSceneCB[frameSlot]);
    memcpy(cbDst, &sceneConstants, sizeof(sceneConstants));
    agfxBufferUnmap(ssaoSceneCB[frameSlot]);
    agfxDeviceMakeResourcesResident(device);

    agfxCommandBufferTextureBarrier(cmdBuffer, normalTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, depthTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, aoTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, true);

    ssao->Generate(device, cmdBuffer, normalView, depthView, aoUAVView, aoTexture, ssaoSceneCBView[frameSlot], pointSampler, width, height);

    agfxCommandBufferTextureBarrier(cmdBuffer, aoTexture, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, normalTexture, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, true);
    agfxCommandBufferTextureBarrier(cmdBuffer, depthTexture, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, true);
}

void DeferredRenderer::RenderShadows(agfxDevice* device, agfxCommandBuffer* cmdBuffer, const GltfScene& scene, const Camera& camera, const LightSettings& light, uint32_t frameSlot)
{
    CascadeInfo cascades[kMaxCascades];
    csm.ComputeCascades(camera, glm::normalize(light.direction), cascades);

    CascadeConstants cascadeConstants = {};
    cascadeConstants.cascadeCount = csm.cascadeCount;
    cascadeConstants.shadowMapResolution = (float)csm.shadowMapResolution;
    cascadeConstants.depthBiasConstant = csm.depthBiasConstant;
    cascadeConstants.lightSizeUV = csm.lightSizeUV;
    cascadeConstants.pcssMaxPenumbraUV = csm.pcssMaxPenumbraUV;
    cascadeConstants.visualizeCascades = csm.visualizeCascades ? 1u : 0u;
    for (uint32_t i = 0; i < csm.cascadeCount; ++i) {
        cascadeConstants.viewProj[i] = cascades[i].viewProj;
        cascadeConstants.splitFar[i] = cascades[i].splitFar;
        cascadeConstants.texelWorldSize[i] = cascades[i].texelWorldSize;
    }
    void* ccbDst = agfxBufferMap(cascadeCB[frameSlot]);
    memcpy(ccbDst, &cascadeConstants, sizeof(cascadeConstants));
    agfxBufferUnmap(cascadeCB[frameSlot]);
    agfxDeviceMakeResourcesResident(device);

    for (uint32_t c = 0; c < csm.cascadeCount; ++c) {
        agfxCommandBufferTextureBarrier(cmdBuffer, shadowTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_DEPTH_WRITE, 0, c, false);
    }

    for (uint32_t c = 0; c < csm.cascadeCount; ++c) {
        agfxRenderPassCreateInfo passInfo = {};
        passInfo.hasDepthAttachment = true;
        passInfo.depthAttachment.renderTarget = shadowCascadeRT[c];
        passInfo.depthAttachment.loadOp = AGFX_LOAD_OPERATION_CLEAR;
        passInfo.depthAttachment.storeOp = AGFX_STORE_OPERATION_STORE;
        passInfo.width = csm.shadowMapResolution;
        passInfo.height = csm.shadowMapResolution;
        passInfo.name = "ShadowCascade";

        agfxRenderPass* pass = agfxRenderPassBegin(cmdBuffer, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)csm.shadowMapResolution, (float)csm.shadowMapResolution, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, csm.shadowMapResolution, csm.shadowMapResolution);
        agfxRenderPassSetPipeline(pass, shadowPipeline);

        for (const ScenePrimitive& prim : scene.primitives) {
            ShadowPushConstants pc = {};
            pc.worldMatrix = prim.worldMatrix;
            pc.vertexBuffer = (uint32_t)agfxBufferViewGetHandle(scene.vertexBufferView);
            pc.cascadeCB = (uint32_t)agfxBufferViewGetHandle(cascadeCBView[frameSlot]);
            pc.cascadeIndex = c;
            pc.vertexOffset = prim.vertexOffset;
            agfxRenderPassPushConstants(pass, &pc, sizeof(pc));

            agfxRenderPassDrawIndexed(pass, scene.indexBuffer, prim.indexCount, 1, prim.indexOffset, 0, 0);
        }

        agfxRenderPassEnd(pass);
    }

    for (uint32_t c = 0; c < csm.cascadeCount; ++c) {
        agfxCommandBufferTextureBarrier(cmdBuffer, shadowTexture, AGFX_RESOURCE_STATE_DEPTH_WRITE, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, c, false);
    }
}

void DeferredRenderer::RenderLighting(agfxDevice* device, agfxCommandBuffer* cmdBuffer, const LightSettings& light, const Camera& camera, uint32_t frameSlot)
{
    agfxCommandBufferTextureBarrier(cmdBuffer, hdrTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, true);

    glm::mat4 view = camera.GetView();
    glm::mat4 proj = camera.GetProj();
    glm::mat4 viewProj = proj * view;

    LightingSceneConstants sceneConstants = {};
    sceneConstants.invViewProj = glm::inverse(viewProj);
    sceneConstants.cameraPos = camera.position;
    sceneConstants.lightDir = glm::normalize(light.direction);
    sceneConstants.lightColor = light.color;
    sceneConstants.lightIntensity = light.intensity;
    void* cbDst = agfxBufferMap(lightingSceneCB[frameSlot]);
    memcpy(cbDst, &sceneConstants, sizeof(sceneConstants));
    agfxBufferUnmap(lightingSceneCB[frameSlot]);
    agfxDeviceMakeResourcesResident(device);

    agfxRenderPassCreateInfo passInfo = {};
    passInfo.colorAttachmentCount = 1;
    passInfo.colorAttachments[0].renderTarget = hdrRT;
    passInfo.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
    passInfo.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
    passInfo.hasDepthAttachment = false;
    passInfo.width = width;
    passInfo.height = height;
    passInfo.name = "DeferredLighting";

    agfxRenderPass* pass = agfxRenderPassBegin(cmdBuffer, &passInfo);
    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    agfxRenderPassSetScissor(pass, 0, 0, width, height);
    agfxRenderPassSetPipeline(pass, lightingPipeline);

    LightingPushConstants pc = {};
    pc.albedoTex = (uint32_t)agfxTextureViewGetHandle(albedoView);
    pc.normalTex = (uint32_t)agfxTextureViewGetHandle(normalView);
    pc.mraTex = (uint32_t)agfxTextureViewGetHandle(mraView);
    pc.depthTex = (uint32_t)agfxTextureViewGetHandle(depthView);
    pc.textureSampler = (uint32_t)agfxSamplerGetHandle(pointSampler);
    pc.lightCB = (uint32_t)agfxBufferViewGetHandle(lightingSceneCBView[frameSlot]);
    pc.shadowMap = (uint32_t)agfxTextureViewGetHandle(shadowArrayView);
    pc.cascadeCB = (uint32_t)agfxBufferViewGetHandle(cascadeCBView[frameSlot]);
    pc.shadowComparisonSampler = (uint32_t)agfxSamplerGetHandle(shadowComparisonSampler);
    pc.pointSampler = (uint32_t)agfxSamplerGetHandle(pointSampler);
    pc.aoTex = (uint32_t)agfxTextureViewGetHandle(aoView);
    agfxRenderPassPushConstants(pass, &pc, sizeof(pc));

    agfxRenderPassDraw(pass, 3, 1, 0, 0);

    agfxRenderPassEnd(pass);

    agfxCommandBufferTextureBarrier(cmdBuffer, hdrTexture, AGFX_RESOURCE_STATE_RENDER_TARGET, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, true);
}

void DeferredRenderer::RenderTonemap(agfxDevice* device, agfxRenderPass* backbufferPass, bool isHDR)
{
    (void)device;

    agfxRenderPassSetPipeline(backbufferPass, tonemapPipeline);

    TonemapPushConstants pc = {};
    pc.hdrTex = (uint32_t)agfxTextureViewGetHandle(hdrView);
    pc.textureSampler = (uint32_t)agfxSamplerGetHandle(pointSampler);
    pc.isHDR = isHDR ? 1 : 0;
    agfxRenderPassPushConstants(backbufferPass, &pc, sizeof(pc));

    agfxRenderPassDraw(backbufferPass, 3, 1, 0, 0);
}
