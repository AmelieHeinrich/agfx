/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <agfx/agfx.h>
#include <glm/glm.hpp>

#include "gltf_scene.h"
#include "camera.h"
#include "csm.h"
#include "ssao.h"

struct SSAOSettings {
    bool enabled = true;
    float radius = 0.5f;
    float bias = 0.025f;
    float power = 1.5f;
};

struct LightSettings {
    glm::vec3 direction = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);
    float intensity = 2.0f;
};

class DeferredRenderer {
public:
    static const uint32_t kFramesInFlight = 3;

    DeferredRenderer() = default;
    DeferredRenderer(const DeferredRenderer&) = delete;
    DeferredRenderer& operator=(const DeferredRenderer&) = delete;

    void Init(agfxDevice* device, agfxTextureFormat swapchainFormat, uint32_t width, uint32_t height);
    void Shutdown(agfxDevice* device);
    void Resize(agfxDevice* device, uint32_t width, uint32_t height);
    void RecreateTonemapPipeline(agfxDevice* device, agfxTextureFormat swapchainFormat);
    void RecreateShadowTargets(agfxDevice* device, uint32_t resolution);

    void RenderGBuffer(agfxDevice* device, agfxCommandBuffer* cmdBuffer, const GltfScene& scene, const Camera& camera, uint32_t frameSlot);
    void RenderSSAO(agfxDevice* device, agfxCommandBuffer* cmdBuffer, const Camera& camera, uint32_t frameSlot);
    void RenderShadows(agfxDevice* device, agfxCommandBuffer* cmdBuffer, const GltfScene& scene, const Camera& camera, const LightSettings& light, uint32_t frameSlot);
    void RenderLighting(agfxDevice* device, agfxCommandBuffer* cmdBuffer, const LightSettings& light, const Camera& camera, uint32_t frameSlot);
    void RenderTonemap(agfxDevice* device, agfxRenderPass* backbufferPass, bool isHDR);

    uint32_t width = 0;
    uint32_t height = 0;

    // GBuffer
    agfxTexture* albedoTexture = nullptr;
    agfxTextureView* albedoView = nullptr; // sampled view (lighting pass)
    agfxRenderTarget* albedoRT = nullptr;

    agfxTexture* normalTexture = nullptr;
    agfxTextureView* normalView = nullptr;
    agfxRenderTarget* normalRT = nullptr;

    agfxTexture* mraTexture = nullptr; // metallic (r), roughness (g), gbuffer-ao (b)
    agfxTextureView* mraView = nullptr;
    agfxRenderTarget* mraRT = nullptr;

    agfxTexture* depthTexture = nullptr;
    agfxTextureView* depthView = nullptr; // sampled view (lighting pass)
    agfxRenderTarget* depthRT = nullptr;

    agfxTexture* hdrTexture = nullptr;
    agfxTextureView* hdrView = nullptr;
    agfxRenderTarget* hdrRT = nullptr;

    agfxSampler* pointSampler = nullptr;

    // SSAO (compute)
    SSAOSettings ssaoSettings;
    AgfxSSAO* ssao = nullptr;
    agfxTexture* aoTexture = nullptr;
    agfxTextureView* aoView = nullptr; // sampled view (lighting pass)
    agfxTextureView* aoUAVView = nullptr; // writeable view (SSAO compute pass)

    // Shadows (CSM + PCSS)
    CSMSettings csm;
    agfxTexture* shadowTexture = nullptr;
    agfxTextureView* shadowArrayView = nullptr; // sampled view (lighting pass), whole array
    agfxRenderTarget* shadowCascadeRT[kMaxCascades] = {}; // per-slice render targets

    agfxSampler* shadowComparisonSampler = nullptr;

    agfxShaderModule* shadowVS = nullptr;
    agfxRenderPipeline* shadowPipeline = nullptr;

    // GBuffer pipeline
    agfxShaderModule* gbufferVS = nullptr;
    agfxShaderModule* gbufferPS = nullptr;
    agfxRenderPipeline* gbufferPipelineCullBack = nullptr;
    agfxRenderPipeline* gbufferPipelineCullNone = nullptr;

    // Lighting pipeline
    agfxShaderModule* lightingVS = nullptr;
    agfxShaderModule* lightingPS = nullptr;
    agfxRenderPipeline* lightingPipeline = nullptr;

    // Tonemap pipeline
    agfxShaderModule* tonemapVS = nullptr;
    agfxShaderModule* tonemapPS = nullptr;
    agfxRenderPipeline* tonemapPipeline = nullptr;
    agfxTextureFormat tonemapOutputFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;

    // Per-frame constant buffers, one set per frame-in-flight slot
    agfxBuffer* gbufferSceneCB[kFramesInFlight] = {};
    agfxBufferView* gbufferSceneCBView[kFramesInFlight] = {};
    agfxBuffer* lightingSceneCB[kFramesInFlight] = {};
    agfxBufferView* lightingSceneCBView[kFramesInFlight] = {};
    agfxBuffer* cascadeCB[kFramesInFlight] = {};
    agfxBufferView* cascadeCBView[kFramesInFlight] = {};
    agfxBuffer* ssaoSceneCB[kFramesInFlight] = {};
    agfxBufferView* ssaoSceneCBView[kFramesInFlight] = {};

private:
    void CreateGBufferTargets(agfxDevice* device, uint32_t width, uint32_t height);
    void DestroyGBufferTargets(agfxDevice* device);
    void CreateShadowTargets(agfxDevice* device, uint32_t resolution);
    void DestroyShadowTargets(agfxDevice* device);
    agfxRenderPipeline* CreateGBufferPipeline(agfxDevice* device, agfxCullMode cullMode);
};
