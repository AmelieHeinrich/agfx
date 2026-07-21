/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "depth_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace agfxtest
{
    namespace
    {
        constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

        /// @brief The columns draw is 6 vertices per column; the full-screen quad is one quad.
        uint32_t VertexCount(const DepthDraw& draw)
        {
            return draw.fullscreen ? 6 : kDepthColumns * 6;
        }

        const char* VertexEntryPoint(const DepthDraw& draw)
        {
            return draw.fullscreen ? "main_vs_fullscreen" : "main_vs";
        }

        agfxTextureCreateInfo ColorTargetInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kDepthColorFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kDepthWidth;
            info.height = kDepthHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxTextureCreateInfo DepthTargetInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kDepthFormat;
            info.usage = AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT;
            info.width = kDepthWidth;
            info.height = kDepthHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        /// @brief One pipeline per draw. The depth state under test lives entirely in the PSO, so a
        /// DepthState with several draws needs several pipelines even though they share a shader.
        agfxRenderPipelineCreateInfo PipelineInfo(const DepthDraw& draw, agfxShaderModule* vs,
                                                  agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "test depth";
            info.fillMode = AGFX_FILL_MODE_SOLID;
            // Culling off, as in the raster tests: a winding difference between backends must not be
            // able to drop a column and be mistaken for a depth result.
            info.cullMode = AGFX_CULL_MODE_NONE;
            info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
            info.topology = AGFX_TOPOLOGY_TRIANGLES;
            info.depthTestEnable = draw.depthTestEnable ? 1 : 0;
            info.depthWriteEnable = draw.depthWriteEnable ? 1 : 0;
            info.depthClampEnable = draw.depthClampEnable ? 1 : 0;
            info.depthCompareOp = draw.depthCompareOp;
            info.depthFormat = kDepthFormat;
            info.colorAttachmentCount = 1;
            info.colorFormats[0] = kDepthColorFormat;
            info.vertexShader = vs;
            info.fragmentShader = ps;
            return info;
        }

        agfxRenderPassCreateInfo PassInfo(const DepthState& state, agfxRenderTarget* colorTarget,
                                          agfxRenderTarget* depthTarget)
        {
            agfxRenderPassCreateInfo info{};
            info.colorAttachmentCount = 1;
            info.colorAttachments[0].renderTarget = colorTarget;
            info.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
            info.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
            memcpy(info.colorAttachments[0].clearColor, kClearColor, sizeof(kClearColor));
            info.hasDepthAttachment = 1;
            info.depthAttachment.renderTarget = depthTarget;
            info.depthAttachment.loadOp = AGFX_LOAD_OPERATION_CLEAR;
            info.depthAttachment.storeOp = AGFX_STORE_OPERATION_STORE;
            info.depthAttachment.clearDepth = state.clearDepth;
            info.width = kDepthWidth;
            info.height = kDepthHeight;
            info.name = "test depth";
            return info;
        }

        // --- C ---------------------------------------------------------------------------------

        bool RenderDepthC(const DepthState& state, const CompiledShader& vsShader,
                          const CompiledShader& fullscreenShader, const CompiledShader& psShader,
                          Image& outImage)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                return false;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo colorInfo = ColorTargetInfo();
            const agfxTextureCreateInfo depthInfo = DepthTargetInfo();
            agfxTexture* colorTexture = agfxTextureCreate(device, &colorInfo);
            agfxTexture* depthTexture = agfxTextureCreate(device, &depthInfo);
            if (!colorTexture || !depthTexture) {
                return false;
            }

            agfxRenderTargetCreateInfo colorRtInfo{};
            colorRtInfo.texture = colorTexture;
            colorRtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN; // inherit
            agfxRenderTarget* colorTarget = agfxRenderTargetCreate(device, &colorRtInfo);

            agfxRenderTargetCreateInfo depthRtInfo{};
            depthRtInfo.texture = depthTexture;
            depthRtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfxRenderTarget* depthTarget = agfxRenderTargetCreate(device, &depthRtInfo);

            std::vector<agfxRenderPipeline*> pipelines;
            pipelines.reserve(state.draws.size());
            agfxShaderModule* psModule =
                CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
            for (const DepthDraw& draw : state.draws) {
                const CompiledShader& shader = draw.fullscreen ? fullscreenShader : vsShader;
                agfxShaderModule* vsModule = CreateShaderModule(device, shader, VertexEntryPoint(draw),
                                                                AGFX_SHADER_MODULE_TYPE_VERTEX);
                const agfxRenderPipelineCreateInfo info = PipelineInfo(draw, vsModule, psModule);
                pipelines.push_back(agfxRenderPipelineCreate(device, &info));
                agfxShaderModuleDestroy(device, vsModule);
            }
            agfxShaderModuleDestroy(device, psModule);

            bool ok = colorTarget != nullptr && depthTarget != nullptr;
            for (agfxRenderPipeline* pipeline : pipelines) {
                ok = ok && pipeline != nullptr;
            }

            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, colorTexture, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 1);
                    agfxCommandBufferTextureBarrier(cmd, depthTexture, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_DEPTH_WRITE, 0, 0, 1);

                    const agfxRenderPassCreateInfo passInfo = PassInfo(state, colorTarget, depthTarget);
                    agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kDepthWidth, (float)kDepthHeight,
                                              0.0f, 1.0f);
                    agfxRenderPassSetScissor(pass, 0, 0, kDepthWidth, kDepthHeight);
                    // All draws share one pass and one depth buffer, so later draws see what earlier
                    // ones wrote. That ordering is the mechanism the depth-write tests rely on.
                    for (size_t i = 0; i < state.draws.size(); ++i) {
                        const DepthDraw& draw = state.draws[i];
                        agfxRenderPassSetPipeline(pass, pipelines[i]);
                        agfxRenderPassPushConstants(pass, &draw.constants, sizeof(draw.constants));
                        agfxRenderPassDraw(pass, VertexCount(draw), 1, 0, 0);
                    }
                    agfxRenderPassEnd(pass);
                });

                ok = ReadbackTexture2D(device, gpu.Queue(), colorTexture, kDepthWidth, kDepthHeight,
                                       kDepthColorFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
            }

            for (agfxRenderPipeline* pipeline : pipelines) {
                if (pipeline) {
                    agfxRenderPipelineDestroy(device, pipeline);
                }
            }
            if (depthTarget) {
                agfxRenderTargetDestroy(device, depthTarget);
            }
            if (colorTarget) {
                agfxRenderTargetDestroy(device, colorTarget);
            }
            agfxTextureDestroy(device, depthTexture);
            agfxTextureDestroy(device, colorTexture);
            return ok;
        }

        // --- C++ -------------------------------------------------------------------------------

        bool RenderDepthCpp(const DepthState& state, const CompiledShader& vsShader,
                            const CompiledShader& fullscreenShader, const CompiledShader& psShader,
                            Image& outImage)
        {
            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                return false;
            }

            agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture colorTexture = device.CreateTexture(ColorTargetInfo());
            agfx::Texture depthTexture = device.CreateTexture(DepthTargetInfo());
            if (!colorTexture.Get() || !depthTexture.Get()) {
                return false;
            }

            agfxRenderTargetCreateInfo colorRtInfo{};
            colorRtInfo.texture = colorTexture;
            colorRtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfx::RenderTarget colorTarget = device.CreateRenderTarget(colorRtInfo);

            agfxRenderTargetCreateInfo depthRtInfo{};
            depthRtInfo.texture = depthTexture;
            depthRtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfx::RenderTarget depthTarget = device.CreateRenderTarget(depthRtInfo);
            if (!colorTarget.Get() || !depthTarget.Get()) {
                return false;
            }

            std::vector<agfx::RenderPipeline> pipelines;
            pipelines.reserve(state.draws.size());
            {
                agfx::ShaderModule psModule(device.Get(),
                    CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
                for (const DepthDraw& draw : state.draws) {
                    const CompiledShader& shader = draw.fullscreen ? fullscreenShader : vsShader;
                    agfx::ShaderModule vsModule(device.Get(),
                        CreateShaderModule(device.Get(), shader, VertexEntryPoint(draw),
                                           AGFX_SHADER_MODULE_TYPE_VERTEX));
                    pipelines.push_back(device.CreateRenderPipeline(
                        PipelineInfo(draw, vsModule, psModule)));
                }
            }
            for (const agfx::RenderPipeline& pipeline : pipelines) {
                if (!pipeline.Get()) {
                    return false;
                }
            }

            device.MakeResourcesResident();

            cmd.Begin();
            cmd.TextureBarrier(colorTexture, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            cmd.TextureBarrier(depthTexture, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            {
                agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(state, colorTarget, depthTarget));
                pass.SetViewport(0.0f, 0.0f, (float)kDepthWidth, (float)kDepthHeight);
                pass.SetScissor(0, 0, kDepthWidth, kDepthHeight);
                for (size_t i = 0; i < state.draws.size(); ++i) {
                    const DepthDraw& draw = state.draws[i];
                    pass.SetPipeline(pipelines[i]);
                    pass.PushConstants(draw.constants);
                    pass.Draw(VertexCount(draw), 1, 0, 0);
                }
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            return ReadbackTexture2D(device.Get(), queue, colorTexture, kDepthWidth, kDepthHeight,
                                     kDepthColorFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
        }

        // --- ez --------------------------------------------------------------------------------

        bool RenderDepthEz(const DepthState& state, const CompiledShader& vsShader,
                           const CompiledShader& fullscreenShader, const CompiledShader& psShader,
                           Image& outImage)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: straight into an offscreen target
            contextInfo.width = kDepthWidth;
            contextInfo.height = kDepthHeight;
            agfx::ez::Context context(contextInfo);
            agfx::Device& device = context.GetDevice();

            // The modules must outlive the draws: ez's PipelineDesc caches on their addresses. Both
            // vertex entry points are created up front for the same reason -- a DepthState may mix
            // full-screen and column draws, and the descs referencing them all live to the pass end.
            agfx::ShaderModule vsModule(device.Get(),
                CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
            agfx::ShaderModule fullscreenModule(device.Get(),
                CreateShaderModule(device.Get(), fullscreenShader, "main_vs_fullscreen",
                                   AGFX_SHADER_MODULE_TYPE_VERTEX));
            agfx::ShaderModule psModule(device.Get(),
                CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
            if (!vsModule.Get() || !fullscreenModule.Get() || !psModule.Get()) {
                return false;
            }

            agfx::ez::Texture2D colorTarget = context.CreateTexture2D(
                kDepthWidth, kDepthHeight, kDepthColorFormat, AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT);
            agfx::ez::Texture2D depthTarget = context.CreateTexture2D(
                kDepthWidth, kDepthHeight, kDepthFormat, AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT);

            device.MakeResourcesResident();

            {
                agfx::ez::Frame frame = context.BeginFrame();
                // clearDepth is the last argument; it is what makes the GREATER-family comparisons
                // reachable, and it is plumbed through agfxRenderPassAttachment::clearDepth.
                context.SetRenderTargets({&colorTarget}, &depthTarget, AGFX_LOAD_OPERATION_CLEAR,
                                         kClearColor, AGFX_LOAD_OPERATION_CLEAR, state.clearDepth);
                context.SetViewportScissor(0, 0, kDepthWidth, kDepthHeight);

                for (const DepthDraw& draw : state.draws) {
                    agfx::ez::PipelineDesc desc;
                    desc.name = "test depth";
                    desc.vertexShader = draw.fullscreen ? &fullscreenModule : &vsModule;
                    desc.fragmentShader = &psModule;
                    desc.cullMode = AGFX_CULL_MODE_NONE;
                    desc.depthTestEnable = draw.depthTestEnable;
                    desc.depthWriteEnable = draw.depthWriteEnable;
                    desc.depthClampEnable = draw.depthClampEnable;
                    desc.depthCompareOp = draw.depthCompareOp;

                    agfx::ez::ShaderBindings bindings;
                    bindings.Write(&draw.constants, sizeof(draw.constants));

                    context.SetPipeline(desc);
                    context.PushShaderBindings(bindings);
                    context.Draw(VertexCount(draw));
                }
                context.EndActivePass();
            }
            context.DrainGPU();

            return ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), colorTarget.Raw(),
                                     kDepthWidth, kDepthHeight, kDepthColorFormat, colorTarget.State(),
                                     outImage);
        }
    } // namespace

    bool RenderDepth(TestApi api, const DepthState& state, Image& outImage)
    {
        // Both vertex entry points are compiled regardless of which the state uses: the compile is
        // cheap next to device creation, and it keeps this function from having to reason about
        // which draws a caller happened to include.
        const CompiledShader vsShader = CompileTestShader("depth.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
        const CompiledShader fullscreenShader =
            CompileTestShader("depth.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs_fullscreen");
        const CompiledShader psShader =
            CompileTestShader("depth.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        if (!vsShader.Valid() || !fullscreenShader.Valid() || !psShader.Valid()) {
            return false;
        }

        switch (api) {
        case TestApi::C:   return RenderDepthC(state, vsShader, fullscreenShader, psShader, outImage);
        case TestApi::Cpp: return RenderDepthCpp(state, vsShader, fullscreenShader, psShader, outImage);
        case TestApi::Ez:  return RenderDepthEz(state, vsShader, fullscreenShader, psShader, outImage);
        }
        return false;
    }
} // namespace agfxtest
