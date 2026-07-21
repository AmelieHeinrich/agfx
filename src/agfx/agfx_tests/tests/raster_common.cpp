/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "raster_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace agfxtest
{
    namespace
    {
        constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

        agfxTextureCreateInfo TargetInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kRasterFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kRasterWidth;
            info.height = kRasterHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        /// @brief The pipeline the raster tests draw with. No depth attachment exists, so depth
        /// test/write stay off and the only thing that can change the image is the state under test.
        agfxRenderPipelineCreateInfo PipelineInfo(const RasterState& state, agfxShaderModule* vs, agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "test raster";
            info.fillMode = state.fillMode;
            info.cullMode = state.cullMode;
            info.frontFace = state.frontFace;
            info.topology = state.topology;
            info.depthTestEnable = 0;
            info.depthWriteEnable = 0;
            info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
            info.colorAttachmentCount = 1;
            info.colorFormats[0] = kRasterFormat;
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
            info.width = kRasterWidth;
            info.height = kRasterHeight;
            info.name = "test raster";
            return info;
        }

        bool RenderRasterC(const RasterState& state, const CompiledShader& vsShader,
                           const CompiledShader& psShader, Image& outImage)
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

            agfxShaderModule* vsModule = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
            agfxShaderModule* psModule = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
            const agfxRenderPipelineCreateInfo pipelineInfo = PipelineInfo(state, vsModule, psModule);
            agfxRenderPipeline* pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
            agfxShaderModuleDestroy(device, vsModule);
            agfxShaderModuleDestroy(device, psModule);

            bool ok = renderTarget != nullptr && pipeline != nullptr;
            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);

                    const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
                    agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kRasterWidth, (float)kRasterHeight, 0.0f, 1.0f);
                    agfxRenderPassSetScissor(pass, 0, 0, kRasterWidth, kRasterHeight);
                    agfxRenderPassSetPipeline(pass, pipeline);
                    agfxRenderPassPushConstants(pass, &state.constants, sizeof(state.constants));
                    agfxRenderPassDraw(pass, state.vertexCount, 1, 0, 0);
                    agfxRenderPassEnd(pass);
                });

                ok = ReadbackTexture2D(device, gpu.Queue(), target, kRasterWidth, kRasterHeight, kRasterFormat,
                                       AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
            }

            if (pipeline) {
                agfxRenderPipelineDestroy(device, pipeline);
            }
            if (renderTarget) {
                agfxRenderTargetDestroy(device, renderTarget);
            }
            agfxTextureDestroy(device, target);
            return ok;
        }

        bool RenderRasterCpp(const RasterState& state, const CompiledShader& vsShader,
                             const CompiledShader& psShader, Image& outImage)
        {
            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                return false;
            }

            agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture target = device.CreateTexture(TargetInfo());

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);

            agfx::RenderPipeline pipeline;
            {
                agfx::ShaderModule vsModule(device.Get(),
                    CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
                agfx::ShaderModule psModule(device.Get(),
                    CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
                pipeline = device.CreateRenderPipeline(PipelineInfo(state, vsModule, psModule));
            }
            if (!target.Get() || !renderTarget.Get() || !pipeline.Get()) {
                return false;
            }

            device.MakeResourcesResident();

            cmd.Begin();
            cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
            {
                agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget));
                pass.SetViewport(0.0f, 0.0f, (float)kRasterWidth, (float)kRasterHeight);
                pass.SetScissor(0, 0, kRasterWidth, kRasterHeight);
                pass.SetPipeline(pipeline);
                pass.PushConstants(&state.constants, sizeof(state.constants));
                pass.Draw(state.vertexCount);
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            return ReadbackTexture2D(device.Get(), queue, target, kRasterWidth, kRasterHeight, kRasterFormat,
                                     AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
        }

        bool RenderRasterEz(const RasterState& state, const CompiledShader& vsShader,
                            const CompiledShader& psShader, Image& outImage)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: straight into an offscreen target
            contextInfo.width = kRasterWidth;
            contextInfo.height = kRasterHeight;
            agfx::ez::Context context(contextInfo);

            agfx::Device& device = context.GetDevice();
            agfx::ShaderModule vsModule(device.Get(),
                CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
            agfx::ShaderModule psModule(device.Get(),
                CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
            if (!vsModule.Get() || !psModule.Get()) {
                return false;
            }

            agfx::ez::Texture2D target = context.CreateTexture2D(kRasterWidth, kRasterHeight, kRasterFormat,
                                                                 AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT);

            // The modules must outlive the draw: ez's PipelineDesc caches on their addresses.
            agfx::ez::PipelineDesc desc;
            desc.name = "test raster";
            desc.vertexShader = &vsModule;
            desc.fragmentShader = &psModule;
            desc.fillMode = state.fillMode;
            desc.cullMode = state.cullMode;
            desc.frontFace = state.frontFace;
            desc.topology = state.topology;
            desc.depthTestEnable = false;
            desc.depthWriteEnable = false;

            agfx::ez::ShaderBindings bindings;
            bindings.Write(&state.constants, sizeof(state.constants));

            {
                agfx::ez::Frame frame = context.BeginFrame();
                context.SetRenderTargets({&target}, nullptr, AGFX_LOAD_OPERATION_CLEAR, kClearColor);
                context.SetViewportScissor(0, 0, kRasterWidth, kRasterHeight);
                context.SetPipeline(desc);
                context.PushShaderBindings(bindings);
                context.Draw(state.vertexCount);
                context.EndActivePass();
            }
            context.DrainGPU();

            return ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(), kRasterWidth,
                                     kRasterHeight, kRasterFormat, target.State(), outImage);
        }
    } // namespace

    uint32_t CountLitPixels(const Image& image)
    {
        uint32_t count = 0;
        for (size_t i = 0; i + 3 < image.pixels.size(); i += 4) {
            // Any visible RGB energy counts; alpha is 1 everywhere thanks to the clear color, so
            // testing it would match every pixel.
            if (image.pixels[i] > 0.0f || image.pixels[i + 1] > 0.0f || image.pixels[i + 2] > 0.0f) {
                ++count;
            }
        }
        return count;
    }

    bool RenderRaster(TestApi api, const RasterState& state, Image& outImage)
    {
        const CompiledShader vsShader = CompileTestShader("raster.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
        const CompiledShader psShader = CompileTestShader("raster.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        if (!vsShader.Valid() || !psShader.Valid()) {
            return false;
        }

        switch (api) {
        case TestApi::C:   return RenderRasterC(state, vsShader, psShader, outImage);
        case TestApi::Cpp: return RenderRasterCpp(state, vsShader, psShader, outImage);
        case TestApi::Ez:  return RenderRasterEz(state, vsShader, psShader, outImage);
        }
        return false;
    }
} // namespace agfxtest
