/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "queue_handoff_common.h"

#include <cstring>

namespace agfxtest
{
    namespace
    {
        constexpr uint32_t kBytesPerPixel = 4;
        constexpr uint32_t kRowBytes = kHandoffWidth * kBytesPerPixel; // 512: a multiple of D3D12's 256.
        constexpr uint32_t kGroupSize = 8;                             // Matches [numthreads(8,8,1)].
        constexpr uint32_t kGroupsX = (kHandoffWidth + kGroupSize - 1) / kGroupSize;
        constexpr uint32_t kGroupsY = (kHandoffHeight + kGroupSize - 1) / kGroupSize;
        constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

        /// @brief Mirrors QueueHandoffPushConstants in queue_handoff.hlsl.
        struct DrawConstants
        {
            uint32_t source;
            uint32_t samplerId;
            uint32_t padding0;
            uint32_t padding1;
        };

        /// @brief Mirrors TexturePushConstants in texture_ops.hlsl, for the compute producer.
        struct WriteConstants
        {
            uint32_t source;
            uint32_t destination;
            uint32_t width;
            uint32_t height;
        };

        /// @brief The copy producer's source content: a checkerboard over a two-channel ramp.
        /// Asymmetric in both axes, so a copy that flipped, transposed or half-covered the texture
        /// shows up as visibly wrong; and busy enough that a draw which sampled uninitialized
        /// memory could not pass by accident.
        std::vector<uint8_t> SourcePixels()
        {
            std::vector<uint8_t> pixels((size_t)kRowBytes * kHandoffHeight);
            for (uint32_t y = 0; y < kHandoffHeight; ++y) {
                for (uint32_t x = 0; x < kHandoffWidth; ++x) {
                    uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                    const bool checker = (((x / 16u) + (y / 16u)) % 2u) != 0u;
                    texel[0] = (uint8_t)(x * 255u / (kHandoffWidth - 1u));
                    texel[1] = (uint8_t)(y * 255u / (kHandoffHeight - 1u));
                    texel[2] = checker ? 255u : 32u;
                    texel[3] = 255u;
                }
            }
            return pixels;
        }

        /// @brief The state the shared texture is transitioned into before the producer runs, and
        /// out of once it has. The transitions are recorded on the *graphics* queue rather than
        /// around the producer's work: a D3D12 copy queue only understands COMMON/COPY_SOURCE/
        /// COPY_DEST, so a COPY_DEST -> PIXEL_SHADER_RESOURCE transition recorded there would pass
        /// on Metal and fail validation on Windows.
        agfxResourceState ProducerState(QueueProducer producer)
        {
            return producer == QueueProducer::Copy ? AGFX_RESOURCE_STATE_COPY_DEST
                                                   : AGFX_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        agfxCommandQueueType ProducerQueueType(QueueProducer producer)
        {
            return producer == QueueProducer::Copy ? AGFX_COMMAND_QUEUE_TYPE_TRANSFER
                                                   : AGFX_COMMAND_QUEUE_TYPE_COMPUTE;
        }

        agfxTextureCreateInfo SharedInfo(QueueProducer producer)
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kHandoffFormat;
            // The compute producer writes through a UAV; the copy producer only needs a copy
            // destination, which every texture already is.
            info.usage = producer == QueueProducer::Compute
                             ? (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED)
                             : AGFX_TEXTURE_USAGE_SAMPLED;
            info.width = kHandoffWidth;
            info.height = kHandoffHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxTextureCreateInfo SourceInfo()
        {
            agfxTextureCreateInfo info = SharedInfo(QueueProducer::Copy);
            return info;
        }

        agfxTextureCreateInfo TargetInfo()
        {
            agfxTextureCreateInfo info = SourceInfo();
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            return info;
        }

        agfxTextureViewCreateInfo ViewInfo(agfxTexture* texture, bool writeable)
        {
            agfxTextureViewCreateInfo info{};
            info.texture = texture;
            info.format = kHandoffFormat;
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.baseMipLevel = 0;
            info.mipLevelCount = 1;
            info.baseArrayLayer = 0;
            info.arrayLayerCount = 1;
            info.writeable = writeable ? 1 : 0;
            return info;
        }

        /// @brief NEAREST, so the 1:1 fullscreen sample reproduces the produced texels exactly and
        /// any difference in the golden is the handoff's doing rather than the filter's.
        agfxSamplerCreateInfo SamplerInfo()
        {
            agfxSamplerCreateInfo info{};
            info.filter = AGFX_SAMPLER_FILTER_NEAREST;
            info.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.maxAnisotropy = 1.0f;
            info.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
            info.minLod = 0.0f;
            info.maxLod = 0.0f;
            return info;
        }

        agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "queue handoff";
            info.fillMode = AGFX_FILL_MODE_SOLID;
            info.cullMode = AGFX_CULL_MODE_NONE;
            info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
            info.topology = AGFX_TOPOLOGY_TRIANGLES;
            info.depthTestEnable = 0;
            info.depthWriteEnable = 0;
            info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
            info.colorAttachmentCount = 1;
            info.colorFormats[0] = kHandoffFormat;
            info.vertexShader = vs;
            info.fragmentShader = ps;
            return info;
        }

        agfxComputePipelineCreateInfo WritePipelineInfo(agfxShaderModule* module)
        {
            agfxComputePipelineCreateInfo info{};
            info.name = "queue handoff producer";
            info.computeShader = module;
            info.groupSizeX = kGroupSize;
            info.groupSizeY = kGroupSize;
            info.groupSizeZ = 1;
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
            info.width = kHandoffWidth;
            info.height = kHandoffHeight;
            info.name = "queue handoff";
            return info;
        }

        agfxTextureRegion FullRegion()
        {
            agfxTextureRegion region{};
            region.width = kHandoffWidth;
            region.height = kHandoffHeight;
            region.depth = 1;
            return region;
        }

        bool RenderQueueHandoffC(QueueProducer producer, const CompiledShader& vsShader,
                                 const CompiledShader& psShader, const CompiledShader& csShader,
                                 Image& outImage)
        {
            const bool isCopy = producer == QueueProducer::Copy;

            agfxDeviceCreateInfo deviceInfo = DefaultDeviceCreateInfo();
            agfxDevice* device = agfxDeviceCreate(&deviceInfo);
            if (!device) {
                return false;
            }

            agfxCommandQueueCreateInfo graphicsQueueInfo{};
            graphicsQueueInfo.type = AGFX_COMMAND_QUEUE_TYPE_GRAPHICS;
            agfxCommandQueue* graphicsQueue = agfxCommandQueueCreate(device, &graphicsQueueInfo);

            agfxCommandQueueCreateInfo producerQueueInfo{};
            producerQueueInfo.type = ProducerQueueType(producer);
            agfxCommandQueue* producerQueue = agfxCommandQueueCreate(device, &producerQueueInfo);

            // The transitions before the producer runs and the draw after it are separate graphics
            // submissions, because the producer's work has to be ordered between them.
            agfxCommandBuffer* preCmd = graphicsQueue ? agfxCommandBufferCreate(device, graphicsQueue) : nullptr;
            agfxCommandBuffer* drawCmd = graphicsQueue ? agfxCommandBufferCreate(device, graphicsQueue) : nullptr;
            agfxCommandBuffer* producerCmd = producerQueue ? agfxCommandBufferCreate(device, producerQueue) : nullptr;
            agfxFence* fence = agfxFenceCreate(device);

            const agfxTextureCreateInfo sourceInfo = SourceInfo();
            const agfxTextureCreateInfo sharedInfo = SharedInfo(producer);
            const agfxTextureCreateInfo targetInfo = TargetInfo();
            // Only the copy producer needs a source to copy *from*; the compute one synthesizes it.
            agfxTexture* source = isCopy ? agfxTextureCreate(device, &sourceInfo) : nullptr;
            agfxTexture* shared = agfxTextureCreate(device, &sharedInfo);
            agfxTexture* target = agfxTextureCreate(device, &targetInfo);

            const agfxTextureViewCreateInfo srvInfo = ViewInfo(shared, false);
            const agfxTextureViewCreateInfo uavInfo = ViewInfo(shared, true);
            agfxTextureView* srv = shared ? agfxTextureViewCreate(device, &srvInfo) : nullptr;
            agfxTextureView* uav = (shared && !isCopy) ? agfxTextureViewCreate(device, &uavInfo) : nullptr;

            const agfxSamplerCreateInfo samplerInfo = SamplerInfo();
            agfxSampler* sampler = agfxSamplerCreate(device, &samplerInfo);

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfxRenderTarget* renderTarget = target ? agfxRenderTargetCreate(device, &rtInfo) : nullptr;

            agfxShaderModule* vsModule = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
            agfxShaderModule* psModule = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
            const agfxRenderPipelineCreateInfo pipelineInfo = PipelineInfo(vsModule, psModule);
            agfxRenderPipeline* pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
            agfxShaderModuleDestroy(device, vsModule);
            agfxShaderModuleDestroy(device, psModule);

            agfxComputePipeline* writePipeline = nullptr;
            if (!isCopy) {
                agfxShaderModule* csModule =
                    CreateShaderModule(device, csShader, "main_write_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
                const agfxComputePipelineCreateInfo writeInfo = WritePipelineInfo(csModule);
                writePipeline = agfxComputePipelineCreate(device, &writeInfo);
                agfxShaderModuleDestroy(device, csModule);
            }

            bool ok = graphicsQueue && producerQueue && preCmd && drawCmd && producerCmd && fence && shared &&
                      target && srv && sampler && renderTarget && pipeline && (isCopy ? source != nullptr : uav && writePipeline);

            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                if (isCopy) {
                    const std::vector<uint8_t> pixels = SourcePixels();
                    ok = UploadTexture2D(device, graphicsQueue, source, kHandoffWidth, kHandoffHeight,
                                         kHandoffFormat, pixels.data(), AGFX_RESOURCE_STATE_COMMON);
                }
            }

            if (ok) {
                DrawConstants drawConstants{};
                drawConstants.source = (uint32_t)agfxTextureViewGetHandle(srv);
                drawConstants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);

                // 1. Graphics queue: put the textures into the producer's states. A D3D12 copy queue
                //    requires a resource to be in COMMON when it crosses over from another queue
                //    type, so the copy producer's source/destination are left in COMMON here and
                //    picked up via the copy queue's implicit "assumed at first use" state instead.
                agfxCommandBufferBegin(preCmd);
                if (!isCopy) {
                    agfxCommandBufferTextureBarrier(preCmd, shared, AGFX_RESOURCE_STATE_COMMON,
                                                    ProducerState(producer), 0, 0, 0);
                }
                agfxCommandBufferEnd(preCmd);
                agfxCommandQueueSubmit(graphicsQueue, &preCmd, 1);
                agfxCommandQueueSignal(graphicsQueue, fence, 1);

                // 2. Producer queue: waits for those transitions, then writes the shared texture.
                agfxCommandQueueWait(producerQueue, fence, 1);
                agfxCommandBufferBegin(producerCmd);
                {
                    agfxComputePass* pass = agfxComputePassBegin(producerCmd, "queue handoff producer");
                    if (isCopy) {
                        const agfxTextureRegion region = FullRegion();
                        agfxComputePassCopyTextureToTexture(pass, source, shared, &region, 0, 0);
                    } else {
                        WriteConstants writeConstants{};
                        writeConstants.destination = (uint32_t)agfxTextureViewGetHandle(uav);
                        writeConstants.width = kHandoffWidth;
                        writeConstants.height = kHandoffHeight;

                        agfxComputePassSetPipeline(pass, writePipeline);
                        agfxComputePassPushConstants(pass, &writeConstants, sizeof(writeConstants));
                        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
                    }
                    agfxComputePassEnd(pass);
                }
                agfxCommandBufferEnd(producerCmd);
                agfxCommandQueueSubmit(producerQueue, &producerCmd, 1);
                agfxCommandQueueSignal(producerQueue, fence, 2);

                // 3. Graphics queue: waits for the producer, then samples its result. This wait is
                //    the whole test — without it the draw races the producer.
                agfxCommandQueueWait(graphicsQueue, fence, 2);
                agfxCommandBufferBegin(drawCmd);
                // Symmetric with the pre-producer transition: the copy queue's work also has to
                // decay back to COMMON crossing back to the graphics queue, so that's the "before"
                // state here rather than ProducerState for the copy case.
                agfxCommandBufferTextureBarrier(drawCmd, shared,
                                                isCopy ? AGFX_RESOURCE_STATE_COMMON : ProducerState(producer),
                                                AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, 0);
                agfxCommandBufferTextureBarrier(drawCmd, target, AGFX_RESOURCE_STATE_COMMON,
                                                AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);
                {
                    const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
                    agfxRenderPass* pass = agfxRenderPassBegin(drawCmd, &passInfo);
                    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kHandoffWidth, (float)kHandoffHeight, 0.0f, 1.0f);
                    agfxRenderPassSetScissor(pass, 0, 0, kHandoffWidth, kHandoffHeight);
                    agfxRenderPassSetPipeline(pass, pipeline);
                    agfxRenderPassPushConstants(pass, &drawConstants, sizeof(drawConstants));
                    agfxRenderPassDraw(pass, 3, 1, 0, 0);
                    agfxRenderPassEnd(pass);
                }
                agfxCommandBufferEnd(drawCmd);
                agfxCommandQueueSubmit(graphicsQueue, &drawCmd, 1);
                agfxCommandQueueSignal(graphicsQueue, fence, 3);
                agfxFenceWait(fence, 3, UINT64_MAX);

                ok = ReadbackTexture2D(device, graphicsQueue, target, kHandoffWidth, kHandoffHeight,
                                       kHandoffFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
            }

            if (writePipeline) {
                agfxComputePipelineDestroy(device, writePipeline);
            }
            if (pipeline) {
                agfxRenderPipelineDestroy(device, pipeline);
            }
            if (renderTarget) {
                agfxRenderTargetDestroy(device, renderTarget);
            }
            if (sampler) {
                agfxSamplerDestroy(device, sampler);
            }
            if (uav) {
                agfxTextureViewDestroy(device, uav);
            }
            if (srv) {
                agfxTextureViewDestroy(device, srv);
            }
            if (target) {
                agfxTextureDestroy(device, target);
            }
            if (shared) {
                agfxTextureDestroy(device, shared);
            }
            if (source) {
                agfxTextureDestroy(device, source);
            }
            if (fence) {
                agfxFenceDestroy(device, fence);
            }
            if (producerCmd) {
                agfxCommandBufferDestroy(device, producerCmd);
            }
            if (drawCmd) {
                agfxCommandBufferDestroy(device, drawCmd);
            }
            if (preCmd) {
                agfxCommandBufferDestroy(device, preCmd);
            }
            if (producerQueue) {
                agfxCommandQueueDestroy(device, producerQueue);
            }
            if (graphicsQueue) {
                agfxCommandQueueDestroy(device, graphicsQueue);
            }
            agfxDeviceDestroy(device);
            return ok;
        }

        bool RenderQueueHandoffCpp(QueueProducer producer, const CompiledShader& vsShader,
                                   const CompiledShader& psShader, const CompiledShader& csShader,
                                   Image& outImage)
        {
            const bool isCopy = producer == QueueProducer::Copy;

            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                return false;
            }

            agfx::CommandQueue graphicsQueue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandQueue producerQueue = device.CreateCommandQueue(ProducerQueueType(producer));
            if (!graphicsQueue.Get() || !producerQueue.Get()) {
                return false;
            }

            agfx::CommandBuffer preCmd = device.CreateCommandBuffer(graphicsQueue);
            agfx::CommandBuffer drawCmd = device.CreateCommandBuffer(graphicsQueue);
            agfx::CommandBuffer producerCmd = device.CreateCommandBuffer(producerQueue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture source;
            if (isCopy) {
                source = device.CreateTexture(SourceInfo());
            }
            agfx::Texture shared = device.CreateTexture(SharedInfo(producer));
            agfx::Texture target = device.CreateTexture(TargetInfo());

            agfx::TextureView srv = device.CreateTextureView(ViewInfo(shared, false));
            agfx::TextureView uav;
            if (!isCopy) {
                uav = device.CreateTextureView(ViewInfo(shared, true));
            }
            agfx::Sampler sampler = device.CreateSampler(SamplerInfo());

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
                pipeline = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
            }

            agfx::ComputePipeline writePipeline;
            if (!isCopy) {
                agfx::ShaderModule csModule(device.Get(),
                    CreateShaderModule(device.Get(), csShader, "main_write_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                writePipeline = device.CreateComputePipeline(WritePipelineInfo(csModule));
            }

            if (!shared.Get() || !target.Get() || !srv.Get() || !sampler.Get() || !renderTarget.Get() ||
                !pipeline.Get() || (isCopy ? !source.Get() : (!uav.Get() || !writePipeline.Get()))) {
                return false;
            }

            device.MakeResourcesResident();

            if (isCopy) {
                const std::vector<uint8_t> pixels = SourcePixels();
                if (!UploadTexture2D(device.Get(), graphicsQueue, source, kHandoffWidth, kHandoffHeight,
                                     kHandoffFormat, pixels.data(), AGFX_RESOURCE_STATE_COMMON)) {
                    return false;
                }
            }

            DrawConstants drawConstants{};
            drawConstants.source = (uint32_t)srv.GetHandle();
            drawConstants.samplerId = (uint32_t)sampler.GetHandle();

            // 1. Graphics queue: into the producer's states. A D3D12 copy queue requires a resource
            //    be in COMMON crossing over from another queue type, so the copy producer's
            //    source/destination are left in COMMON and picked up via the copy queue's implicit
            //    "assumed at first use" state instead.
            preCmd.Begin();
            if (!isCopy) {
                preCmd.TextureBarrier(shared, AGFX_RESOURCE_STATE_COMMON, ProducerState(producer),
                                      AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
            }
            preCmd.End();
            graphicsQueue.Submit(preCmd);
            graphicsQueue.Signal(fence, 1);

            // 2. Producer queue: waits for those transitions, then writes the shared texture.
            producerQueue.Wait(fence, 1);
            producerCmd.Begin();
            {
                agfx::ComputePass pass = producerCmd.BeginComputePass("queue handoff producer");
                if (isCopy) {
                    pass.CopyTextureToTexture(source, shared, FullRegion(), 0, 0);
                } else {
                    WriteConstants writeConstants{};
                    writeConstants.destination = (uint32_t)uav.GetHandle();
                    writeConstants.width = kHandoffWidth;
                    writeConstants.height = kHandoffHeight;

                    pass.SetPipeline(writePipeline);
                    pass.PushConstants(writeConstants);
                    pass.Dispatch(kGroupsX, kGroupsY, 1);
                }
            }
            producerCmd.End();
            producerQueue.Submit(producerCmd);
            producerQueue.Signal(fence, 2);

            // 3. Graphics queue: waits for the producer, then samples its result.
            graphicsQueue.Wait(fence, 2);
            drawCmd.Begin();
            drawCmd.TextureBarrier(shared, isCopy ? AGFX_RESOURCE_STATE_COMMON : ProducerState(producer),
                                   AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                   AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
            drawCmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                                   AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
            {
                agfx::RenderPass pass = drawCmd.BeginRenderPass(PassInfo(renderTarget));
                pass.SetViewport(0.0f, 0.0f, (float)kHandoffWidth, (float)kHandoffHeight);
                pass.SetScissor(0, 0, kHandoffWidth, kHandoffHeight);
                pass.SetPipeline(pipeline);
                pass.PushConstants(drawConstants);
                pass.Draw(3);
            }
            drawCmd.End();
            graphicsQueue.Submit(drawCmd);
            graphicsQueue.Signal(fence, 3);
            fence.Wait(3, UINT64_MAX);

            return ReadbackTexture2D(device.Get(), graphicsQueue, target, kHandoffWidth, kHandoffHeight,
                                     kHandoffFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
        }
    } // namespace

    bool RenderQueueHandoff(TestApi api, QueueProducer producer, Image& outImage)
    {
        const CompiledShader vsShader = CompileTestShader("queue_handoff.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
        const CompiledShader psShader = CompileTestShader("queue_handoff.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        if (!vsShader.Valid() || !psShader.Valid()) {
            return false;
        }

        // Only the compute producer needs it, and compiling it unconditionally would make a broken
        // texture_ops.hlsl fail the copy test too.
        CompiledShader csShader;
        if (producer == QueueProducer::Compute) {
            csShader = CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_cs");
            if (!csShader.Valid()) {
                return false;
            }
        }

        switch (api) {
        case TestApi::C:   return RenderQueueHandoffC(producer, vsShader, psShader, csShader, outImage);
        case TestApi::Cpp: return RenderQueueHandoffCpp(producer, vsShader, psShader, csShader, outImage);
        case TestApi::Ez:  return false; // Not expressible through ez -- see queue_handoff_common.h.
        }
        return false;
    }
} // namespace agfxtest
