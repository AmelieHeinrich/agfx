/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "blend_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace agfxtest
{
    namespace
    {
        /// @brief Transparent black. The alpha matters: it is the destination alpha the DST_ALPHA
        /// family reads outside the columns, and a clear alpha of 1 would make that region
        /// indistinguishable from an opaque destination.
        constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        uint32_t VertexCount(const BlendDraw& draw)
        {
            return draw.fullscreen ? 6 : kBlendColumns * 6;
        }

        const char* VertexEntryPoint(const BlendDraw& draw)
        {
            return draw.fullscreen ? "main_vs_fullscreen" : "main_vs";
        }

        agfxTextureCreateInfo TargetInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kBlendFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kBlendWidth;
            info.height = kBlendHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        /// @brief One pipeline per draw: the blend state under test lives entirely in the PSO.
        ///
        /// The alpha blend factors are set to the same values as the color ones. Blending the alpha
        /// channel separately is a real feature, but it is not what these tests are measuring, and
        /// letting the two disagree would make the destination alpha the *next* draw reads depend on
        /// a second axis of state.
        agfxRenderPipelineCreateInfo PipelineInfo(const BlendDraw& draw, agfxShaderModule* vs,
                                                  agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "test blend";
            info.fillMode = AGFX_FILL_MODE_SOLID;
            info.cullMode = AGFX_CULL_MODE_NONE;
            info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
            info.topology = AGFX_TOPOLOGY_TRIANGLES;
            info.depthTestEnable = 0;
            info.depthWriteEnable = 0;
            info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
            info.colorAttachmentCount = 1;
            info.colorFormats[0] = kBlendFormat;
            info.blendEnable[0] = draw.blendEnable ? 1 : 0;
            info.srcColorBlendFactor[0] = draw.srcBlend;
            info.dstColorBlendFactor[0] = draw.dstBlend;
            info.colorBlendOp[0] = draw.blendOp;
            info.srcAlphaBlendFactor[0] = draw.srcBlend;
            info.dstAlphaBlendFactor[0] = draw.dstBlend;
            info.alphaBlendOp[0] = draw.blendOp;
            info.vertexShader = vs;
            info.fragmentShader = ps;
            return info;
        }

        agfxRenderPassCreateInfo PassInfo(agfxRenderTarget* renderTarget)
        {
            agfxRenderPassCreateInfo info{};
            info.colorAttachmentCount = 1;
            info.colorAttachments[0].renderTarget = renderTarget;
            info.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
            info.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
            memcpy(info.colorAttachments[0].clearColor, kClearColor, sizeof(kClearColor));
            info.width = kBlendWidth;
            info.height = kBlendHeight;
            info.name = "test blend";
            return info;
        }

        // --- C ---------------------------------------------------------------------------------

        bool RenderBlendC(const BlendState& state, const CompiledShader& vsShader,
                          const CompiledShader& fullscreenShader, const CompiledShader& psShader,
                          Image& outImage)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                return false;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo targetInfo = TargetInfo();
            agfxTexture* target = agfxTextureCreate(device, &targetInfo);
            if (!target) {
                return false;
            }

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfxRenderTarget* renderTarget = agfxRenderTargetCreate(device, &rtInfo);

            std::vector<agfxRenderPipeline*> pipelines;
            pipelines.reserve(state.draws.size());
            agfxShaderModule* psModule =
                CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
            for (const BlendDraw& draw : state.draws) {
                const CompiledShader& shader = draw.fullscreen ? fullscreenShader : vsShader;
                agfxShaderModule* vsModule = CreateShaderModule(device, shader, VertexEntryPoint(draw),
                                                                AGFX_SHADER_MODULE_TYPE_VERTEX);
                const agfxRenderPipelineCreateInfo info = PipelineInfo(draw, vsModule, psModule);
                pipelines.push_back(agfxRenderPipelineCreate(device, &info));
                agfxShaderModuleDestroy(device, vsModule);
            }
            agfxShaderModuleDestroy(device, psModule);

            bool ok = renderTarget != nullptr;
            for (agfxRenderPipeline* pipeline : pipelines) {
                ok = ok && pipeline != nullptr;
            }

            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 1);

                    const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
                    agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kBlendWidth, (float)kBlendHeight,
                                              0.0f, 1.0f);
                    agfxRenderPassSetScissor(pass, 0, 0, kBlendWidth, kBlendHeight);
                    // One pass: the source draw must see what the destination draw left in the
                    // attachment, which is the whole premise of a blend test.
                    for (size_t i = 0; i < state.draws.size(); ++i) {
                        const BlendDraw& draw = state.draws[i];
                        agfxRenderPassSetPipeline(pass, pipelines[i]);
                        agfxRenderPassPushConstants(pass, &draw.constants, sizeof(draw.constants));
                        agfxRenderPassDraw(pass, VertexCount(draw), 1, 0, 0);
                    }
                    agfxRenderPassEnd(pass);
                });

                ok = ReadbackTexture2D(device, gpu.Queue(), target, kBlendWidth, kBlendHeight,
                                       kBlendFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
            }

            for (agfxRenderPipeline* pipeline : pipelines) {
                if (pipeline) {
                    agfxRenderPipelineDestroy(device, pipeline);
                }
            }
            if (renderTarget) {
                agfxRenderTargetDestroy(device, renderTarget);
            }
            agfxTextureDestroy(device, target);
            return ok;
        }

        // --- C++ -------------------------------------------------------------------------------

        bool RenderBlendCpp(const BlendState& state, const CompiledShader& vsShader,
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

            agfx::Texture target = device.CreateTexture(TargetInfo());
            if (!target.Get()) {
                return false;
            }

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);
            if (!renderTarget.Get()) {
                return false;
            }

            std::vector<agfx::RenderPipeline> pipelines;
            pipelines.reserve(state.draws.size());
            {
                agfx::ShaderModule psModule(device.Get(),
                    CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
                for (const BlendDraw& draw : state.draws) {
                    const CompiledShader& shader = draw.fullscreen ? fullscreenShader : vsShader;
                    agfx::ShaderModule vsModule(device.Get(),
                        CreateShaderModule(device.Get(), shader, VertexEntryPoint(draw),
                                           AGFX_SHADER_MODULE_TYPE_VERTEX));
                    pipelines.push_back(device.CreateRenderPipeline(PipelineInfo(draw, vsModule, psModule)));
                }
            }
            for (const agfx::RenderPipeline& pipeline : pipelines) {
                if (!pipeline.Get()) {
                    return false;
                }
            }

            device.MakeResourcesResident();

            cmd.Begin();
            cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            {
                agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget));
                pass.SetViewport(0.0f, 0.0f, (float)kBlendWidth, (float)kBlendHeight);
                pass.SetScissor(0, 0, kBlendWidth, kBlendHeight);
                for (size_t i = 0; i < state.draws.size(); ++i) {
                    const BlendDraw& draw = state.draws[i];
                    pass.SetPipeline(pipelines[i]);
                    pass.PushConstants(draw.constants);
                    pass.Draw(VertexCount(draw), 1, 0, 0);
                }
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            return ReadbackTexture2D(device.Get(), queue, target, kBlendWidth, kBlendHeight, kBlendFormat,
                                     AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
        }

        // --- ez --------------------------------------------------------------------------------

        bool RenderBlendEz(const BlendState& state, const CompiledShader& vsShader,
                           const CompiledShader& fullscreenShader, const CompiledShader& psShader,
                           Image& outImage)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: straight into an offscreen target
            contextInfo.width = kBlendWidth;
            contextInfo.height = kBlendHeight;
            agfx::ez::Context context(contextInfo);
            agfx::Device& device = context.GetDevice();

            // The modules must outlive the draws: ez's PipelineDesc caches on their addresses.
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

            agfx::ez::Texture2D target = context.CreateTexture2D(kBlendWidth, kBlendHeight, kBlendFormat,
                                                                  AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT);

            device.MakeResourcesResident();

            {
                agfx::ez::Frame frame = context.BeginFrame();
                context.SetRenderTargets({&target}, nullptr, AGFX_LOAD_OPERATION_CLEAR, kClearColor);
                context.SetViewportScissor(0, 0, kBlendWidth, kBlendHeight);

                for (const BlendDraw& draw : state.draws) {
                    agfx::ez::PipelineDesc desc;
                    desc.name = "test blend";
                    desc.vertexShader = draw.fullscreen ? &fullscreenModule : &vsModule;
                    desc.fragmentShader = &psModule;
                    desc.cullMode = AGFX_CULL_MODE_NONE;
                    desc.depthTestEnable = false;
                    desc.depthWriteEnable = false;
                    desc.blendEnable = draw.blendEnable;
                    desc.srcBlend = draw.srcBlend;
                    desc.dstBlend = draw.dstBlend;
                    desc.blendOp = draw.blendOp;

                    agfx::ez::ShaderBindings bindings;
                    bindings.Write(&draw.constants, sizeof(draw.constants));

                    context.SetPipeline(desc);
                    context.PushShaderBindings(bindings);
                    context.Draw(VertexCount(draw));
                }
                context.EndActivePass();
            }
            context.DrainGPU();

            return ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(), kBlendWidth,
                                     kBlendHeight, kBlendFormat, target.State(), outImage);
        }
    } // namespace

    BlendDraw DestinationDraw()
    {
        BlendDraw draw;
        draw.fullscreen = false;
        draw.blendEnable = false; // the destination is written, not blended
        // Dim, to leave headroom before the UNORM clamp.
        draw.constants.columnScale[0] = 0.4f;
        draw.constants.columnScale[1] = 0.4f;
        draw.constants.columnScale[2] = 0.4f;
        // Three different destination alphas, plus 0 outside the columns from the clear: four
        // distinct destination alphas in one image for the DST_ALPHA family to separate on.
        draw.constants.columnAlpha[0] = 1.0f;
        draw.constants.columnAlpha[1] = 0.6f;
        draw.constants.columnAlpha[2] = 0.2f;
        return draw;
    }

    BlendDraw SourceDraw(agfxBlendFactor srcBlend, agfxBlendFactor dstBlend, agfxBlendOperation blendOp)
    {
        BlendDraw draw;
        draw.fullscreen = true;
        draw.blendEnable = true;
        draw.srcBlend = srcBlend;
        draw.dstBlend = dstBlend;
        draw.blendOp = blendOp;
        // Asymmetric across channels so a channel swap is visible, and dim enough that additive
        // results stay under 1.0 against the destination columns.
        draw.constants.color[0] = 0.5f;
        draw.constants.color[1] = 0.25f;
        draw.constants.color[2] = 0.125f;
        draw.constants.color[3] = 0.25f; // not 0.5 -- see the header
        return draw;
    }

    BlendState State(agfxBlendFactor srcBlend, agfxBlendFactor dstBlend, agfxBlendOperation blendOp)
    {
        BlendState state;
        state.draws = {DestinationDraw(), SourceDraw(srcBlend, dstBlend, blendOp)};
        return state;
    }

    bool RenderBlend(TestApi api, const BlendState& state, Image& outImage)
    {
        const CompiledShader vsShader = CompileTestShader("blend.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
        const CompiledShader fullscreenShader =
            CompileTestShader("blend.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs_fullscreen");
        const CompiledShader psShader =
            CompileTestShader("blend.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        if (!vsShader.Valid() || !fullscreenShader.Valid() || !psShader.Valid()) {
            return false;
        }

        switch (api) {
        case TestApi::C:   return RenderBlendC(state, vsShader, fullscreenShader, psShader, outImage);
        case TestApi::Cpp: return RenderBlendCpp(state, vsShader, fullscreenShader, psShader, outImage);
        case TestApi::Ez:  return RenderBlendEz(state, vsShader, fullscreenShader, psShader, outImage);
        }
        return false;
    }
} // namespace agfxtest
