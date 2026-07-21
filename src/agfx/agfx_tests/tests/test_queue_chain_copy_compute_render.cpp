/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy -> compute -> render queue".
//
// queue_handoff_common.cpp already covers a single producer handing off to the render queue (copy
// OR compute). This is the three-stage extension: copy queue, then compute queue, then render
// queue, chained through *two* cross-queue waits on one shared fence instead of one. Each stage's
// output is the next stage's input, so a dropped wait anywhere in the chain -- not just at the very
// end -- has to surface for the test to be doing its job:
//
//   1. Transfer queue copies a seeded, host-uploaded source into an intermediate texture.
//   2. Compute queue waits for that copy, then reads the intermediate texture with
//      texture_ops.hlsl:main_load_cs, which mirrors it horizontally and swizzles it (r,g,b) ->
//      (b,r,g), writing the result into a final texture.
//   3. Graphics queue waits for that dispatch, then samples the final texture with
//      queue_handoff.hlsl (NEAREST, 1:1) into an offscreen render target.
//
// All state transitions are recorded on the *graphics* queue, bracketing the copy and compute
// stages rather than happening inside them: a D3D12 copy queue only understands COMMON/COPY_SOURCE/
// COPY_DEST, so a transition into a shader-visible state recorded there would pass on Metal and
// fail validation on Windows. This mirrors queue_handoff_common's ProducerState handling exactly.
//
// The expected output is computed analytically from the source pixels and main_load_cs's known
// transform (mirror + swizzle), so the test does not lean on the golden as the oracle: a chain that
// silently serialized on the CPU instead of the GPU could still happen to produce the right pixels
// once in a while, but happening to match a hand-computed expectation on every run is a much higher
// bar than happening to look plausible.
//
// Only C and C++ flavors exist. ez owns its graphics queue and submits frames internally, with no
// way to interleave a second or third queue's submissions into that ordering -- same reasoning as
// queue_handoff_common.h.

#include "../test_gpu.h"

#include <cstring>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kRowBytes = kWidth * kBytesPerPixel; // 512: a multiple of D3D12's 256.
    constexpr uint32_t kGroupSize = 8;                      // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroupsX = (kWidth + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsY = (kHeight + kGroupSize - 1) / kGroupSize;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    constexpr const char* kGolden = "queue_chain_copy_compute_render.png";

    /// @brief Mirrors QueueHandoffPushConstants in queue_handoff.hlsl.
    struct DrawConstants
    {
        uint32_t source;
        uint32_t samplerId;
        uint32_t padding0;
        uint32_t padding1;
    };

    /// @brief Mirrors TexturePushConstants in texture_ops.hlsl.
    struct LoadConstants
    {
        uint32_t source;
        uint32_t destination;
        uint32_t width;
        uint32_t height;
    };

    /// @brief The chain's seed: a checkerboard over a two-channel ramp, asymmetric in both axes so a
    /// transposed, flipped or half-covered copy is visibly wrong rather than plausible.
    std::vector<uint8_t> SourcePixels()
    {
        std::vector<uint8_t> pixels((size_t)kRowBytes * kHeight);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                const bool checker = (((x / 16u) + (y / 16u)) % 2u) != 0u;
                texel[0] = (uint8_t)(x * 255u / (kWidth - 1u));
                texel[1] = (uint8_t)(y * 255u / (kHeight - 1u));
                texel[2] = checker ? 255u : 32u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief What the render queue must read back: main_load_cs's transform (horizontal mirror,
    /// then (r,g,b) -> (b,r,g)) applied to SourcePixels(), which is exactly what a correctly-ordered
    /// copy -> compute -> render chain produces. Any dropped wait either leaves this looking like the
    /// intermediate's clear/garbage state or a stale value, not this.
    std::vector<uint8_t> ExpectedFinal()
    {
        const std::vector<uint8_t> source = SourcePixels();
        std::vector<uint8_t> expected((size_t)kRowBytes * kHeight);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                const uint32_t mirroredX = kWidth - 1u - x;
                const uint8_t* src = &source[(size_t)y * kRowBytes + (size_t)mirroredX * kBytesPerPixel];
                uint8_t* dst = &expected[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                dst[0] = src[1]; // main_load_cs: dst = (source.g, source.b, source.r, source.a)
                dst[1] = src[2];
                dst[2] = src[0];
                dst[3] = src[3];
            }
        }
        return expected;
    }

    agfxTextureCreateInfo SourceInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = AGFX_TEXTURE_USAGE_SAMPLED;
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    /// @brief The copy queue's destination and the compute queue's source: only needs to be a copy
    /// destination and a (non-pixel) shader resource, never sampled by the render queue directly.
    agfxTextureCreateInfo IntermediateInfo()
    {
        agfxTextureCreateInfo info = SourceInfo();
        return info;
    }

    /// @brief The compute queue's destination and the render queue's source.
    agfxTextureCreateInfo FinalInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxTextureCreateInfo TargetInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxTextureViewCreateInfo ViewInfo(agfxTexture* texture, bool writeable)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.mipLevelCount = 1;
        info.arrayLayerCount = 1;
        info.writeable = writeable ? 1 : 0;
        return info;
    }

    /// @brief NEAREST, so the 1:1 fullscreen sample reproduces the compute stage's texels exactly
    /// and ExpectedFinal() can be an exact comparison rather than a tolerance.
    agfxSamplerCreateInfo SamplerInfo()
    {
        agfxSamplerCreateInfo info{};
        info.filter = AGFX_SAMPLER_FILTER_NEAREST;
        info.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.maxAnisotropy = 1.0f;
        info.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
        return info;
    }

    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "queue chain render";
        info.fillMode = AGFX_FILL_MODE_SOLID;
        info.cullMode = AGFX_CULL_MODE_NONE;
        info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        info.topology = AGFX_TOPOLOGY_TRIANGLES;
        info.depthTestEnable = 0;
        info.depthWriteEnable = 0;
        info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
        info.colorAttachmentCount = 1;
        info.colorFormats[0] = kFormat;
        info.vertexShader = vs;
        info.fragmentShader = ps;
        return info;
    }

    agfxComputePipelineCreateInfo LoadPipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "queue chain compute";
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
        info.width = kWidth;
        info.height = kHeight;
        info.name = "queue chain render";
        return info;
    }

    agfxTextureRegion FullRegion()
    {
        agfxTextureRegion region{};
        region.width = kWidth;
        region.height = kHeight;
        region.depth = 1;
        return region;
    }

    bool RunC(const CompiledShader& vsShader, const CompiledShader& psShader, const CompiledShader& csShader,
              Image& outImage, std::string& outError)
    {
        agfxDeviceCreateInfo deviceInfo = DefaultDeviceCreateInfo();
        agfxDevice* device = agfxDeviceCreate(&deviceInfo);
        if (!device) {
            outError = "failed to create headless device";
            return false;
        }

        agfxCommandQueueCreateInfo graphicsInfo{};
        graphicsInfo.type = AGFX_COMMAND_QUEUE_TYPE_GRAPHICS;
        agfxCommandQueue* graphicsQueue = agfxCommandQueueCreate(device, &graphicsInfo);

        agfxCommandQueueCreateInfo transferInfo{};
        transferInfo.type = AGFX_COMMAND_QUEUE_TYPE_TRANSFER;
        agfxCommandQueue* transferQueue = agfxCommandQueueCreate(device, &transferInfo);

        agfxCommandQueueCreateInfo computeInfo{};
        computeInfo.type = AGFX_COMMAND_QUEUE_TYPE_COMPUTE;
        agfxCommandQueue* computeQueue = agfxCommandQueueCreate(device, &computeInfo);

        // Three graphics submissions bracket the copy and compute stages; each is its own command
        // buffer because the fence ordering between them is what the test is about.
        agfxCommandBuffer* preCmd = graphicsQueue ? agfxCommandBufferCreate(device, graphicsQueue) : nullptr;
        agfxCommandBuffer* midCmd = graphicsQueue ? agfxCommandBufferCreate(device, graphicsQueue) : nullptr;
        agfxCommandBuffer* drawCmd = graphicsQueue ? agfxCommandBufferCreate(device, graphicsQueue) : nullptr;
        agfxCommandBuffer* copyCmd = transferQueue ? agfxCommandBufferCreate(device, transferQueue) : nullptr;
        agfxCommandBuffer* computeCmd = computeQueue ? agfxCommandBufferCreate(device, computeQueue) : nullptr;
        agfxFence* fence = agfxFenceCreate(device);

        const agfxTextureCreateInfo sourceInfo = SourceInfo();
        const agfxTextureCreateInfo intermediateInfo = IntermediateInfo();
        const agfxTextureCreateInfo finalInfo = FinalInfo();
        const agfxTextureCreateInfo targetInfo = TargetInfo();
        agfxTexture* source = agfxTextureCreate(device, &sourceInfo);
        agfxTexture* intermediate = agfxTextureCreate(device, &intermediateInfo);
        agfxTexture* final_ = agfxTextureCreate(device, &finalInfo);
        agfxTexture* target = agfxTextureCreate(device, &targetInfo);

        const agfxTextureViewCreateInfo intermediateSrvInfo = ViewInfo(intermediate, false);
        const agfxTextureViewCreateInfo finalUavInfo = ViewInfo(final_, true);
        const agfxTextureViewCreateInfo finalSrvInfo = ViewInfo(final_, false);
        agfxTextureView* intermediateSrv = intermediate ? agfxTextureViewCreate(device, &intermediateSrvInfo) : nullptr;
        agfxTextureView* finalUav = final_ ? agfxTextureViewCreate(device, &finalUavInfo) : nullptr;
        agfxTextureView* finalSrv = final_ ? agfxTextureViewCreate(device, &finalSrvInfo) : nullptr;

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

        agfxShaderModule* csModule = CreateShaderModule(device, csShader, "main_load_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
        const agfxComputePipelineCreateInfo loadPipelineInfo = LoadPipelineInfo(csModule);
        agfxComputePipeline* loadPipeline = agfxComputePipelineCreate(device, &loadPipelineInfo);
        agfxShaderModuleDestroy(device, csModule);

        bool ok = graphicsQueue && transferQueue && computeQueue && preCmd && midCmd && drawCmd && copyCmd &&
                  computeCmd && fence && source && intermediate && final_ && target && intermediateSrv &&
                  finalUav && finalSrv && sampler && renderTarget && pipeline && loadPipeline;
        if (!ok) {
            outError = "resource or pipeline creation failed";
        }

        if (ok) {
            agfxDeviceMakeResourcesResident(device);

            const std::vector<uint8_t> seed = SourcePixels();
            ok = UploadTexture2D(device, graphicsQueue, source, kWidth, kHeight, kFormat, seed.data(),
                                 AGFX_RESOURCE_STATE_COMMON);
            if (!ok) {
                outError = "failed to seed the source texture";
            }
        }

        if (ok) {
            DrawConstants drawConstants{};
            drawConstants.source = (uint32_t)agfxTextureViewGetHandle(finalSrv);
            drawConstants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);

            LoadConstants loadConstants{};
            loadConstants.source = (uint32_t)agfxTextureViewGetHandle(intermediateSrv);
            loadConstants.destination = (uint32_t)agfxTextureViewGetHandle(finalUav);
            loadConstants.width = kWidth;
            loadConstants.height = kHeight;

            // 1. Graphics queue: nothing to transition here -- a D3D12 copy queue requires a
            //    resource to be in COMMON when it crosses over from another queue type, so source
            //    and intermediate are left in COMMON and picked up via the transfer queue's implicit
            //    "assumed at first use" COPY_SOURCE/COPY_DEST state instead. Still recorded and
            //    submitted so the transfer queue's wait below has a fence value to wait on.
            agfxCommandBufferBegin(preCmd);
            agfxCommandBufferEnd(preCmd);
            agfxCommandQueueSubmit(graphicsQueue, &preCmd, 1);
            agfxCommandQueueSignal(graphicsQueue, fence, 1);

            // 2. Transfer queue: waits for those transitions, copies source -> intermediate.
            agfxCommandQueueWait(transferQueue, fence, 1);
            agfxCommandBufferBegin(copyCmd);
            {
                const agfxTextureRegion region = FullRegion();
                agfxComputePass* pass = agfxComputePassBegin(copyCmd, "queue chain copy");
                agfxComputePassCopyTextureToTexture(pass, source, intermediate, &region, 0, 0);
                agfxComputePassEnd(pass);
            }
            agfxCommandBufferEnd(copyCmd);
            agfxCommandQueueSubmit(transferQueue, &copyCmd, 1);
            agfxCommandQueueSignal(transferQueue, fence, 2);

            // 3. Graphics queue: waits for the copy, into the compute stage's states. Recorded on
            //    graphics rather than the transfer or compute queue -- a transfer queue only
            //    understands COMMON/COPY_SOURCE/COPY_DEST, and this is a shader-visible state.
            agfxCommandQueueWait(graphicsQueue, fence, 2);
            agfxCommandBufferBegin(midCmd);
            // The transfer queue's COPY_DEST usage decays back to COMMON crossing back to the
            // graphics queue, so that's the "before" state here, not COPY_DEST.
            agfxCommandBufferTextureBarrier(midCmd, intermediate, AGFX_RESOURCE_STATE_COMMON,
                                            AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 0, 0, 0);
            agfxCommandBufferTextureBarrier(midCmd, final_, AGFX_RESOURCE_STATE_COMMON,
                                            AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, 0);
            agfxCommandBufferEnd(midCmd);
            agfxCommandQueueSubmit(graphicsQueue, &midCmd, 1);
            agfxCommandQueueSignal(graphicsQueue, fence, 3);

            // 4. Compute queue: waits for those transitions, mirrors+swizzles intermediate -> final.
            agfxCommandQueueWait(computeQueue, fence, 3);
            agfxCommandBufferBegin(computeCmd);
            {
                agfxComputePass* pass = agfxComputePassBegin(computeCmd, "queue chain compute");
                agfxComputePassSetPipeline(pass, loadPipeline);
                agfxComputePassPushConstants(pass, &loadConstants, sizeof(loadConstants));
                agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
                agfxComputePassEnd(pass);
            }
            agfxCommandBufferEnd(computeCmd);
            agfxCommandQueueSubmit(computeQueue, &computeCmd, 1);
            agfxCommandQueueSignal(computeQueue, fence, 4);

            // 5. Graphics queue: waits for the compute dispatch, samples final into target. This
            //    wait and the one at step 3 are the whole test -- drop either and the draw or the
            //    dispatch races its producer.
            agfxCommandQueueWait(graphicsQueue, fence, 4);
            agfxCommandBufferBegin(drawCmd);
            agfxCommandBufferTextureBarrier(drawCmd, final_, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                            AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, 0, 0, 0);
            agfxCommandBufferTextureBarrier(drawCmd, target, AGFX_RESOURCE_STATE_COMMON,
                                            AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);
            {
                const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
                agfxRenderPass* pass = agfxRenderPassBegin(drawCmd, &passInfo);
                agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kWidth, (float)kHeight, 0.0f, 1.0f);
                agfxRenderPassSetScissor(pass, 0, 0, kWidth, kHeight);
                agfxRenderPassSetPipeline(pass, pipeline);
                agfxRenderPassPushConstants(pass, &drawConstants, sizeof(drawConstants));
                agfxRenderPassDraw(pass, 3, 1, 0, 0);
                agfxRenderPassEnd(pass);
            }
            agfxCommandBufferEnd(drawCmd);
            agfxCommandQueueSubmit(graphicsQueue, &drawCmd, 1);
            agfxCommandQueueSignal(graphicsQueue, fence, 5);
            agfxFenceWait(fence, 5, UINT64_MAX);

            ok = ReadbackTexture2D(device, graphicsQueue, target, kWidth, kHeight, kFormat,
                                   AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
            if (!ok) {
                outError = "texture readback failed";
            }
        }

        if (loadPipeline) agfxComputePipelineDestroy(device, loadPipeline);
        if (pipeline) agfxRenderPipelineDestroy(device, pipeline);
        if (renderTarget) agfxRenderTargetDestroy(device, renderTarget);
        if (sampler) agfxSamplerDestroy(device, sampler);
        if (finalSrv) agfxTextureViewDestroy(device, finalSrv);
        if (finalUav) agfxTextureViewDestroy(device, finalUav);
        if (intermediateSrv) agfxTextureViewDestroy(device, intermediateSrv);
        if (target) agfxTextureDestroy(device, target);
        if (final_) agfxTextureDestroy(device, final_);
        if (intermediate) agfxTextureDestroy(device, intermediate);
        if (source) agfxTextureDestroy(device, source);
        if (fence) agfxFenceDestroy(device, fence);
        if (computeCmd) agfxCommandBufferDestroy(device, computeCmd);
        if (copyCmd) agfxCommandBufferDestroy(device, copyCmd);
        if (drawCmd) agfxCommandBufferDestroy(device, drawCmd);
        if (midCmd) agfxCommandBufferDestroy(device, midCmd);
        if (preCmd) agfxCommandBufferDestroy(device, preCmd);
        if (computeQueue) agfxCommandQueueDestroy(device, computeQueue);
        if (transferQueue) agfxCommandQueueDestroy(device, transferQueue);
        if (graphicsQueue) agfxCommandQueueDestroy(device, graphicsQueue);
        agfxDeviceDestroy(device);
        return ok;
    }

    bool RunCpp(const CompiledShader& vsShader, const CompiledShader& psShader, const CompiledShader& csShader,
                Image& outImage, std::string& outError)
    {
        agfx::Device device(DefaultDeviceCreateInfo());
        if (!device.Get()) {
            outError = "failed to create headless device";
            return false;
        }

        agfx::CommandQueue graphicsQueue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
        agfx::CommandQueue transferQueue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_TRANSFER);
        agfx::CommandQueue computeQueue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_COMPUTE);
        if (!graphicsQueue.Get() || !transferQueue.Get() || !computeQueue.Get()) {
            outError = "queue creation failed";
            return false;
        }

        agfx::CommandBuffer preCmd = device.CreateCommandBuffer(graphicsQueue);
        agfx::CommandBuffer midCmd = device.CreateCommandBuffer(graphicsQueue);
        agfx::CommandBuffer drawCmd = device.CreateCommandBuffer(graphicsQueue);
        agfx::CommandBuffer copyCmd = device.CreateCommandBuffer(transferQueue);
        agfx::CommandBuffer computeCmd = device.CreateCommandBuffer(computeQueue);
        agfx::Fence fence = device.CreateFence();

        agfx::Texture source = device.CreateTexture(SourceInfo());
        agfx::Texture intermediate = device.CreateTexture(IntermediateInfo());
        agfx::Texture final_ = device.CreateTexture(FinalInfo());
        agfx::Texture target = device.CreateTexture(TargetInfo());

        agfx::TextureView intermediateSrv = device.CreateTextureView(ViewInfo(intermediate, false));
        agfx::TextureView finalUav = device.CreateTextureView(ViewInfo(final_, true));
        agfx::TextureView finalSrv = device.CreateTextureView(ViewInfo(final_, false));
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

        agfx::ComputePipeline loadPipeline;
        {
            agfx::ShaderModule csModule(device.Get(),
                CreateShaderModule(device.Get(), csShader, "main_load_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
            loadPipeline = device.CreateComputePipeline(LoadPipelineInfo(csModule));
        }

        if (!source.Get() || !intermediate.Get() || !final_.Get() || !target.Get() || !intermediateSrv.Get() ||
            !finalUav.Get() || !finalSrv.Get() || !sampler.Get() || !renderTarget.Get() || !pipeline.Get() ||
            !loadPipeline.Get()) {
            outError = "resource or pipeline creation failed";
            return false;
        }

        device.MakeResourcesResident();

        const std::vector<uint8_t> seed = SourcePixels();
        if (!UploadTexture2D(device.Get(), graphicsQueue, source, kWidth, kHeight, kFormat, seed.data(),
                             AGFX_RESOURCE_STATE_COMMON)) {
            outError = "failed to seed the source texture";
            return false;
        }

        DrawConstants drawConstants{};
        drawConstants.source = (uint32_t)finalSrv.GetHandle();
        drawConstants.samplerId = (uint32_t)sampler.GetHandle();

        LoadConstants loadConstants{};
        loadConstants.source = (uint32_t)intermediateSrv.GetHandle();
        loadConstants.destination = (uint32_t)finalUav.GetHandle();
        loadConstants.width = kWidth;
        loadConstants.height = kHeight;

        // 1. Graphics queue: nothing to transition here -- a D3D12 copy queue requires a resource
        //    to be in COMMON when it crosses over from another queue type, so source and
        //    intermediate are left in COMMON and picked up via the transfer queue's implicit
        //    "assumed at first use" COPY_SOURCE/COPY_DEST state instead.
        preCmd.Begin();
        preCmd.End();
        graphicsQueue.Submit(preCmd);
        graphicsQueue.Signal(fence, 1);

        // 2. Transfer queue: waits for those transitions, copies source -> intermediate.
        transferQueue.Wait(fence, 1);
        copyCmd.Begin();
        {
            agfx::ComputePass pass = copyCmd.BeginComputePass("queue chain copy");
            pass.CopyTextureToTexture(source, intermediate, FullRegion(), 0, 0);
        }
        copyCmd.End();
        transferQueue.Submit(copyCmd);
        transferQueue.Signal(fence, 2);

        // 3. Graphics queue: waits for the copy, into the compute stage's states.
        graphicsQueue.Wait(fence, 2);
        midCmd.Begin();
        // The transfer queue's COPY_DEST usage decays back to COMMON crossing back to the graphics
        // queue, so that's the "before" state here, not COPY_DEST.
        midCmd.TextureBarrier(intermediate, AGFX_RESOURCE_STATE_COMMON,
                              AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                              AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
        midCmd.TextureBarrier(final_, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                              AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
        midCmd.End();
        graphicsQueue.Submit(midCmd);
        graphicsQueue.Signal(fence, 3);

        // 4. Compute queue: waits for those transitions, mirrors+swizzles intermediate -> final.
        computeQueue.Wait(fence, 3);
        computeCmd.Begin();
        {
            agfx::ComputePass pass = computeCmd.BeginComputePass("queue chain compute");
            pass.SetPipeline(loadPipeline);
            pass.PushConstants(&loadConstants, sizeof(loadConstants));
            pass.Dispatch(kGroupsX, kGroupsY, 1);
        }
        computeCmd.End();
        computeQueue.Submit(computeCmd);
        computeQueue.Signal(fence, 4);

        // 5. Graphics queue: waits for the compute dispatch, samples final into target.
        graphicsQueue.Wait(fence, 4);
        drawCmd.Begin();
        drawCmd.TextureBarrier(final_, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                               AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
        drawCmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
        {
            agfx::RenderPass pass = drawCmd.BeginRenderPass(PassInfo(renderTarget));
            pass.SetViewport(0.0f, 0.0f, (float)kWidth, (float)kHeight);
            pass.SetScissor(0, 0, kWidth, kHeight);
            pass.SetPipeline(pipeline);
            pass.PushConstants(&drawConstants, sizeof(drawConstants));
            pass.Draw(3);
        }
        drawCmd.End();
        graphicsQueue.Submit(drawCmd);
        graphicsQueue.Signal(fence, 5);
        fence.Wait(5, UINT64_MAX);

        if (!ReadbackTexture2D(device.Get(), graphicsQueue, target, kWidth, kHeight, kFormat,
                               AGFX_RESOURCE_STATE_RENDER_TARGET, outImage)) {
            outError = "texture readback failed";
            return false;
        }
        return true;
    }

    bool Run(TestApi api, Image& outImage, std::string& outError)
    {
        outError.clear();

        const CompiledShader vsShader = CompileTestShader("queue_handoff.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
        const CompiledShader psShader = CompileTestShader("queue_handoff.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        const CompiledShader csShader = CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_load_cs");
        if (!vsShader.Valid() || !psShader.Valid() || !csShader.Valid()) {
            outError = "failed to compile shaders";
            return false;
        }

        switch (api) {
        case TestApi::C:   return RunC(vsShader, psShader, csShader, outImage, outError);
        case TestApi::Cpp: return RunCpp(vsShader, psShader, csShader, outImage, outError);
        case TestApi::Ez:  return false; // Not expressible through ez -- see the file comment.
        }
        outError = "unknown API flavor";
        return false;
    }
} // namespace

AGFX_TEST_TEXTURE(QueueChainCopyComputeRender, C, kWidth, kHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(Run(TestApi::C, image, error), error.c_str());
    AGFX_EXPECT_MSG(ImageEqualsRgba8(image, ExpectedFinal()),
                    "the render queue's result is not the compute stage's mirrored/swizzled copy of the seed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(QueueChainCopyComputeRender, Cpp, kWidth, kHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(Run(TestApi::Cpp, image, error), error.c_str());
    AGFX_EXPECT_MSG(ImageEqualsRgba8(image, ExpectedFinal()),
                    "the render queue's result is not the compute stage's mirrored/swizzled copy of the seed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
