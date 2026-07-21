/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "mrt_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace agfxtest
{
    namespace
    {
        constexpr uint32_t kGroupSize = 8; // Matches [numthreads(8,8,1)].
        constexpr uint32_t kGroupsX = (kMrtWidth + kGroupSize - 1) / kGroupSize;
        constexpr uint32_t kGroupsY = (kMrtHeight + kGroupSize - 1) / kGroupSize;

        constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

        /// @brief Mirrors MRTPushConstants in data/shaders/tests/mrt.hlsl.
        struct MrtPushConstants
        {
            uint32_t source0 = 0;
            uint32_t source1 = 0;
            uint32_t destination = 0;
            uint32_t width = kMrtWidth;
            uint32_t height = kMrtHeight;
            uint32_t padding0 = 0;
            uint32_t padding1 = 0;
            uint32_t padding2 = 0;
        };
        static_assert(sizeof(MrtPushConstants) == 32, "MrtPushConstants must match the HLSL layout");

        agfxTextureCreateInfo AttachmentInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kMrtFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kMrtWidth;
            info.height = kMrtHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxTextureCreateInfo DestInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kMrtFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kMrtWidth;
            info.height = kMrtHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxTextureViewCreateInfo ViewInfo(agfxTexture* texture, bool writeable)
        {
            agfxTextureViewCreateInfo info{};
            info.texture = texture;
            info.format = kMrtFormat;
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.mipLevelCount = 1;
            info.arrayLayerCount = 1;
            info.writeable = writeable ? 1 : 0;
            return info;
        }

        /// @brief Two color attachments, both RGBA8. colorAttachmentCount is the field under test:
        /// a backend that only ever wired up attachment 0 produces a destination missing its green.
        agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "test mrt";
            info.fillMode = AGFX_FILL_MODE_SOLID;
            info.cullMode = AGFX_CULL_MODE_NONE;
            info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
            info.topology = AGFX_TOPOLOGY_TRIANGLES;
            info.depthTestEnable = 0;
            info.depthWriteEnable = 0;
            info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
            info.colorAttachmentCount = 2;
            info.colorFormats[0] = kMrtFormat;
            info.colorFormats[1] = kMrtFormat;
            info.vertexShader = vs;
            info.fragmentShader = ps;
            return info;
        }

        agfxRenderPassCreateInfo PassInfo(agfxRenderTarget* rt0, agfxRenderTarget* rt1)
        {
            agfxRenderPassCreateInfo info{};
            info.colorAttachmentCount = 2;
            for (uint32_t i = 0; i < 2; ++i) {
                info.colorAttachments[i].renderTarget = i == 0 ? rt0 : rt1;
                info.colorAttachments[i].loadOp = AGFX_LOAD_OPERATION_CLEAR;
                info.colorAttachments[i].storeOp = AGFX_STORE_OPERATION_STORE;
                memcpy(info.colorAttachments[i].clearColor, kClearColor, sizeof(kClearColor));
            }
            info.width = kMrtWidth;
            info.height = kMrtHeight;
            info.name = "test mrt";
            return info;
        }

        agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
        {
            agfxComputePipelineCreateInfo info{};
            info.name = "mrt add";
            info.computeShader = module;
            info.groupSizeX = kGroupSize;
            info.groupSizeY = kGroupSize;
            info.groupSizeZ = 1;
            return info;
        }

        bool RunC(const CompiledShader& vsShader, const CompiledShader& psShader,
                  const CompiledShader& csShader, Image& outImage, std::string& outError)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                outError = "failed to create headless device";
                return false;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo attachmentInfo = AttachmentInfo();
            const agfxTextureCreateInfo destInfo = DestInfo();
            agfxTexture* target0 = agfxTextureCreate(device, &attachmentInfo);
            agfxTexture* target1 = agfxTextureCreate(device, &attachmentInfo);
            agfxTexture* dest = agfxTextureCreate(device, &destInfo);

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            rtInfo.texture = target0;
            agfxRenderTarget* rt0 = target0 ? agfxRenderTargetCreate(device, &rtInfo) : nullptr;
            rtInfo.texture = target1;
            agfxRenderTarget* rt1 = target1 ? agfxRenderTargetCreate(device, &rtInfo) : nullptr;

            const agfxTextureViewCreateInfo srv0Info = ViewInfo(target0, false);
            const agfxTextureViewCreateInfo srv1Info = ViewInfo(target1, false);
            const agfxTextureViewCreateInfo destUavInfo = ViewInfo(dest, true);
            agfxTextureView* srv0 = target0 ? agfxTextureViewCreate(device, &srv0Info) : nullptr;
            agfxTextureView* srv1 = target1 ? agfxTextureViewCreate(device, &srv1Info) : nullptr;
            agfxTextureView* destUav = dest ? agfxTextureViewCreate(device, &destUavInfo) : nullptr;

            agfxShaderModule* vsModule = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
            agfxShaderModule* psModule = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
            const agfxRenderPipelineCreateInfo pipelineInfo = PipelineInfo(vsModule, psModule);
            agfxRenderPipeline* pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
            agfxShaderModuleDestroy(device, vsModule);
            agfxShaderModuleDestroy(device, psModule);

            agfxShaderModule* csModule =
                CreateShaderModule(device, csShader, "main_add_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
            const agfxComputePipelineCreateInfo addPipelineInfo = ComputePipelineInfo(csModule);
            agfxComputePipeline* addPipeline = agfxComputePipelineCreate(device, &addPipelineInfo);
            agfxShaderModuleDestroy(device, csModule);

            bool ok = target0 && target1 && dest && rt0 && rt1 && srv0 && srv1 && destUav && pipeline &&
                      addPipeline;
            if (!ok) {
                outError = "resource or pipeline creation failed";
            }

            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                MrtPushConstants constants{};
                constants.source0 = (uint32_t)agfxTextureViewGetHandle(srv0);
                constants.source1 = (uint32_t)agfxTextureViewGetHandle(srv1);
                constants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, target0, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_RENDER_TARGET,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxCommandBufferTextureBarrier(cmd, target1, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_RENDER_TARGET,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

                    const agfxRenderPassCreateInfo passInfo = PassInfo(rt0, rt1);
                    agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kMrtWidth, (float)kMrtHeight, 0.0f, 1.0f);
                    agfxRenderPassSetScissor(pass, 0, 0, kMrtWidth, kMrtHeight);
                    agfxRenderPassSetPipeline(pass, pipeline);
                    agfxRenderPassDraw(pass, 3, 1, 0, 0);
                    agfxRenderPassEnd(pass);
                });

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    // agglomerate must be true: on the Metal backend a barrier recorded with false is
                    // a no-op, which would leave the render pass and the add dispatch unsynchronized.
                    agfxCommandBufferTextureBarrier(cmd, target0, AGFX_RESOURCE_STATE_RENDER_TARGET,
                                                    AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxCommandBufferTextureBarrier(cmd, target1, AGFX_RESOURCE_STATE_RENDER_TARGET,
                                                    AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxComputePass* pass = agfxComputePassBegin(cmd, "mrt add");
                    agfxComputePassSetPipeline(pass, addPipeline);
                    agfxComputePassPushConstants(pass, &constants, sizeof(constants));
                    agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
                    agfxComputePassEnd(pass);
                });

                ok = ReadbackTexture2D(device, gpu.Queue(), dest, kMrtWidth, kMrtHeight, kMrtFormat,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage);
                if (!ok) {
                    outError = "texture readback failed";
                }
            }

            if (addPipeline) agfxComputePipelineDestroy(device, addPipeline);
            if (pipeline) agfxRenderPipelineDestroy(device, pipeline);
            if (destUav) agfxTextureViewDestroy(device, destUav);
            if (srv1) agfxTextureViewDestroy(device, srv1);
            if (srv0) agfxTextureViewDestroy(device, srv0);
            if (rt1) agfxRenderTargetDestroy(device, rt1);
            if (rt0) agfxRenderTargetDestroy(device, rt0);
            if (dest) agfxTextureDestroy(device, dest);
            if (target1) agfxTextureDestroy(device, target1);
            if (target0) agfxTextureDestroy(device, target0);
            return ok;
        }

        bool RunCpp(const CompiledShader& vsShader, const CompiledShader& psShader,
                    const CompiledShader& csShader, Image& outImage, std::string& outError)
        {
            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                outError = "failed to create headless device";
                return false;
            }

            agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture target0 = device.CreateTexture(AttachmentInfo());
            agfx::Texture target1 = device.CreateTexture(AttachmentInfo());
            agfx::Texture dest = device.CreateTexture(DestInfo());

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            rtInfo.texture = target0;
            agfx::RenderTarget rt0 = device.CreateRenderTarget(rtInfo);
            rtInfo.texture = target1;
            agfx::RenderTarget rt1 = device.CreateRenderTarget(rtInfo);

            agfx::TextureView srv0 = device.CreateTextureView(ViewInfo(target0, false));
            agfx::TextureView srv1 = device.CreateTextureView(ViewInfo(target1, false));
            agfx::TextureView destUav = device.CreateTextureView(ViewInfo(dest, true));

            agfx::RenderPipeline pipeline;
            agfx::ComputePipeline addPipeline;
            {
                agfx::ShaderModule vsModule(device.Get(),
                    CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
                agfx::ShaderModule psModule(device.Get(),
                    CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
                agfx::ShaderModule csModule(device.Get(),
                    CreateShaderModule(device.Get(), csShader, "main_add_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                pipeline = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
                addPipeline = device.CreateComputePipeline(ComputePipelineInfo(csModule));
            }

            if (!target0.Get() || !target1.Get() || !dest.Get() || !rt0.Get() || !rt1.Get() ||
                !pipeline.Get() || !addPipeline.Get()) {
                outError = "resource or pipeline creation failed";
                return false;
            }

            device.MakeResourcesResident();

            MrtPushConstants constants{};
            constants.source0 = (uint32_t)agfxTextureViewGetHandle(srv0);
            constants.source1 = (uint32_t)agfxTextureViewGetHandle(srv1);
            constants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

            cmd.Begin();
            cmd.TextureBarrier(target0, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            cmd.TextureBarrier(target1, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            {
                agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(rt0, rt1));
                pass.SetViewport(0.0f, 0.0f, (float)kMrtWidth, (float)kMrtHeight);
                pass.SetScissor(0, 0, kMrtWidth, kMrtHeight);
                pass.SetPipeline(pipeline);
                pass.Draw(3);
            }
            cmd.TextureBarrier(target0, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            cmd.TextureBarrier(target1, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            {
                agfx::ComputePass pass = cmd.BeginComputePass("mrt add");
                pass.SetPipeline(addPipeline);
                pass.PushConstants(&constants, sizeof(constants));
                pass.Dispatch(kGroupsX, kGroupsY, 1);
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            if (!ReadbackTexture2D(device.Get(), queue, dest, kMrtWidth, kMrtHeight, kMrtFormat,
                                   AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage)) {
                outError = "texture readback failed";
                return false;
            }
            return true;
        }

        bool RunEz(const CompiledShader& vsShader, const CompiledShader& psShader,
                   const CompiledShader& csShader, Image& outImage, std::string& outError)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: no swap chain
            contextInfo.width = kMrtWidth;
            contextInfo.height = kMrtHeight;
            agfx::ez::Context context(contextInfo);

            agfx::Device& device = context.GetDevice();

            const agfxTextureUsage attachmentUsage =
                (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            agfx::ez::Texture2D target0 = context.CreateTexture2D(kMrtWidth, kMrtHeight, kMrtFormat, attachmentUsage);
            agfx::ez::Texture2D target1 = context.CreateTexture2D(kMrtWidth, kMrtHeight, kMrtFormat, attachmentUsage);
            agfx::ez::Texture2D dest = context.CreateTexture2D(
                kMrtWidth, kMrtHeight, kMrtFormat,
                (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED));

            agfx::ShaderModule vsModule(device.Get(),
                CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
            agfx::ShaderModule psModule(device.Get(),
                CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
            agfx::ComputePipeline addPipeline;
            {
                agfx::ShaderModule csModule(device.Get(),
                    CreateShaderModule(device.Get(), csShader, "main_add_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                addPipeline = device.CreateComputePipeline(ComputePipelineInfo(csModule));
            }
            if (!vsModule.Get() || !psModule.Get() || !addPipeline.Get()) {
                outError = "shader module or pipeline creation failed";
                return false;
            }

            // The modules must outlive the draw: ez's PipelineDesc caches on their addresses. ez
            // takes the attachment count and formats from the SetRenderTargets call, so passing two
            // targets is what makes this an MRT pipeline.
            agfx::ez::PipelineDesc desc;
            desc.name = "test mrt";
            desc.vertexShader = &vsModule;
            desc.fragmentShader = &psModule;
            desc.cullMode = AGFX_CULL_MODE_NONE;
            desc.depthTestEnable = false;
            desc.depthWriteEnable = false;

            agfx::ez::ShaderBindings addBindings;
            addBindings.BindTexture(target0.SRV());
            addBindings.BindTexture(target1.SRV());
            addBindings.BindTexture(dest.UAV());
            addBindings.Write(kMrtWidth);
            addBindings.Write(kMrtHeight);
            addBindings.Write(0u);
            addBindings.Write(0u);
            addBindings.Write(0u);

            device.MakeResourcesResident();

            {
                agfx::ez::Frame frame = context.BeginFrame();
                context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

                context.SetRenderTargets({&target0, &target1}, nullptr, AGFX_LOAD_OPERATION_CLEAR, kClearColor);
                context.SetViewportScissor(0, 0, kMrtWidth, kMrtHeight);
                context.SetPipeline(desc);
                context.Draw(3);
                context.EndActivePass();

                context.TransitionTexture(target0, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                context.TransitionTexture(target1, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

                // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
                {
                    agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("mrt add");
                    pass.SetPipeline(addPipeline);
                    pass.PushConstants(addBindings.Data(), addBindings.Size());
                    pass.Dispatch(kGroupsX, kGroupsY, 1);
                }
            }
            context.DrainGPU();

            if (!ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kMrtWidth,
                                   kMrtHeight, kMrtFormat, dest.State(), outImage)) {
                outError = "texture readback failed";
                return false;
            }
            return true;
        }
    } // namespace

    std::vector<uint8_t> ExpectedMrtSum()
    {
        std::vector<uint8_t> pixels((size_t)kMrtWidth * kMrtHeight * 4);
        for (uint32_t y = 0; y < kMrtHeight; ++y) {
            for (uint32_t x = 0; x < kMrtWidth; ++x) {
                uint8_t* texel = &pixels[((size_t)y * kMrtWidth + x) * 4];
                texel[0] = (uint8_t)x; // from attachment 0's red ramp
                texel[1] = (uint8_t)y; // from attachment 1's green ramp
                texel[2] = 0u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    bool RenderMrt(TestApi api, Image& outImage, std::string& outError)
    {
        outError.clear();

        const CompiledShader vsShader = CompileTestShader("mrt.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
        const CompiledShader psShader = CompileTestShader("mrt.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        const CompiledShader csShader = CompileTestShader("mrt.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_add_cs");
        if (!vsShader.Valid() || !psShader.Valid() || !csShader.Valid()) {
            outError = "failed to compile mrt.hlsl";
            return false;
        }

        switch (api) {
        case TestApi::C:   return RunC(vsShader, psShader, csShader, outImage, outError);
        case TestApi::Cpp: return RunCpp(vsShader, psShader, csShader, outImage, outError);
        case TestApi::Ez:  return RunEz(vsShader, psShader, csShader, outImage, outError);
        }
        outError = "unknown API flavor";
        return false;
    }
} // namespace agfxtest
