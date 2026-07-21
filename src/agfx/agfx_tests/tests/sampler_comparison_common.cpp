/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "sampler_comparison_common.h"

#include <agfx/agfx_ez.hpp>

namespace agfxtest
{
    namespace
    {
        constexpr uint32_t kGroupSize = 8; // Matches [numthreads(8,8,1)].
        constexpr uint32_t kGroupsX = (kCmpWidth + kGroupSize - 1) / kGroupSize;
        constexpr uint32_t kGroupsY = (kCmpHeight + kGroupSize - 1) / kGroupSize;

        constexpr agfxTextureFormat kDepthFormat = AGFX_TEXTURE_FORMAT_DEPTH32F;
        /// @brief The format the depth texture is *viewed* as for sampling. D32 is not a readable
        /// format; the single channel is read back as R32F, which is the same reinterpretation the
        /// demo's shadow maps do in DeferredRenderer::CreateShadowTargets.
        constexpr agfxTextureFormat kDepthViewFormat = AGFX_TEXTURE_FORMAT_R32F;
        constexpr agfxTextureFormat kDestFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

        /// @brief Mirrors SamplingComparisonPushConstants in sampling_comparison.hlsl. Field order
        /// and padding must match exactly -- this is memcpy'd into the push-constant block.
        struct CmpPushConstants
        {
            uint32_t source = 0;
            uint32_t samplerId = 0;
            uint32_t destination = 0;
            uint32_t width = kCmpWidth;
            uint32_t height = kCmpHeight;
            uint32_t bandCount = kCmpBandCount;
            uint32_t padding0 = 0;
            uint32_t padding1 = 0;
            float references[4] = {kCmpNearer, kCmpStoredDepth, kCmpFarther, 0.0f};
        };
        static_assert(sizeof(CmpPushConstants) == 48, "CmpPushConstants must match the HLSL layout");

        agfxSamplerCreateInfo SamplerInfo(agfxComparisonFunction compareOp)
        {
            agfxSamplerCreateInfo info{};
            // NEAREST, not LINEAR: a linear comparison sampler does PCF, averaging the comparison
            // results of four taps. With a uniform source depth every tap agrees so the average
            // would still be 0 or 1, but pinning it to a single tap leaves the comparison function
            // as the only thing that can move the result.
            info.filter = AGFX_SAMPLER_FILTER_NEAREST;
            info.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.maxAnisotropy = 1.0f;
            info.comparisonFunction = compareOp;
            return info;
        }

        agfxTextureCreateInfo DepthInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kDepthFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kCmpWidth;
            info.height = kCmpHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxTextureCreateInfo DestInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kDestFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kCmpWidth;
            info.height = kCmpHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxTextureViewCreateInfo DepthViewInfo(agfxTexture* texture)
        {
            agfxTextureViewCreateInfo info{};
            info.texture = texture;
            info.format = kDepthViewFormat; // D32 -> R32F, as the demo's shadow map view does
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.mipLevelCount = 1;
            info.arrayLayerCount = 1;
            info.writeable = 0;
            return info;
        }

        agfxTextureViewCreateInfo DestViewInfo(agfxTexture* texture)
        {
            agfxTextureViewCreateInfo info{};
            info.texture = texture;
            info.format = kDestFormat;
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.mipLevelCount = 1;
            info.arrayLayerCount = 1;
            info.writeable = 1;
            return info;
        }

        agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
        {
            agfxComputePipelineCreateInfo info{};
            info.name = "sampler comparison";
            info.computeShader = module;
            info.groupSizeX = kGroupSize;
            info.groupSizeY = kGroupSize;
            info.groupSizeZ = 1;
            return info;
        }

        /// @brief The depth-only pass that establishes the source. There is no draw and no color
        /// attachment: clearing to kCmpStoredDepth is the entire content of the pass, which keeps
        /// the stored depth exact rather than whatever a rasterized triangle would interpolate to.
        agfxRenderPassCreateInfo DepthPassInfo(agfxRenderTarget* depthTarget)
        {
            agfxRenderPassCreateInfo info{};
            info.colorAttachmentCount = 0;
            info.hasDepthAttachment = 1;
            info.depthAttachment.renderTarget = depthTarget;
            info.depthAttachment.loadOp = AGFX_LOAD_OPERATION_CLEAR;
            info.depthAttachment.storeOp = AGFX_STORE_OPERATION_STORE;
            info.depthAttachment.clearDepth = kCmpStoredDepth;
            info.width = kCmpWidth;
            info.height = kCmpHeight;
            info.name = "sampler comparison depth";
            return info;
        }

        bool RunC(agfxComparisonFunction compareOp, const CompiledShader& shader, Image& outImage,
                  std::string& outError)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                outError = "failed to create headless device";
                return false;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo depthInfo = DepthInfo();
            const agfxTextureCreateInfo destInfo = DestInfo();
            agfxTexture* depth = agfxTextureCreate(device, &depthInfo);
            agfxTexture* dest = agfxTextureCreate(device, &destInfo);

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = depth;
            rtInfo.format = kDepthFormat;
            rtInfo.isDepth = 1;
            agfxRenderTarget* depthTarget = depth ? agfxRenderTargetCreate(device, &rtInfo) : nullptr;

            const agfxTextureViewCreateInfo depthViewInfo = DepthViewInfo(depth);
            const agfxTextureViewCreateInfo destViewInfo = DestViewInfo(dest);
            agfxTextureView* depthSrv = depth ? agfxTextureViewCreate(device, &depthViewInfo) : nullptr;
            agfxTextureView* destUav = dest ? agfxTextureViewCreate(device, &destViewInfo) : nullptr;

            const agfxSamplerCreateInfo samplerInfo = SamplerInfo(compareOp);
            agfxSampler* sampler = agfxSamplerCreate(device, &samplerInfo);

            agfxShaderModule* module =
                CreateShaderModule(device, shader, "main_sample_cmp_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
            const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
            agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
            agfxShaderModuleDestroy(device, module);

            bool ok = depth && dest && depthTarget && depthSrv && destUav && sampler && pipeline;
            if (!ok) {
                outError = "resource or pipeline creation failed";
            }

            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                CmpPushConstants constants{};
                constants.source = (uint32_t)agfxTextureViewGetHandle(depthSrv);
                constants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
                constants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, depth, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_DEPTH_WRITE,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

                    const agfxRenderPassCreateInfo passInfo = DepthPassInfo(depthTarget);
                    agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                    agfxRenderPassEnd(pass);
                });

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    // agglomerate must be true: on the Metal backend a barrier recorded with false
                    // is a no-op, which would leave the depth clear and the sampling dispatch
                    // unsynchronized.
                    agfxCommandBufferTextureBarrier(cmd, depth, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                                                    AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxComputePass* pass = agfxComputePassBegin(cmd, "sampler comparison");
                    agfxComputePassSetPipeline(pass, pipeline);
                    agfxComputePassPushConstants(pass, &constants, sizeof(constants));
                    agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
                    agfxComputePassEnd(pass);
                });

                ok = ReadbackTexture2D(device, gpu.Queue(), dest, kCmpWidth, kCmpHeight, kDestFormat,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage);
                if (!ok) {
                    outError = "texture readback failed";
                }
            }

            if (pipeline) {
                agfxComputePipelineDestroy(device, pipeline);
            }
            if (sampler) {
                agfxSamplerDestroy(device, sampler);
            }
            if (destUav) {
                agfxTextureViewDestroy(device, destUav);
            }
            if (depthSrv) {
                agfxTextureViewDestroy(device, depthSrv);
            }
            if (depthTarget) {
                agfxRenderTargetDestroy(device, depthTarget);
            }
            if (dest) {
                agfxTextureDestroy(device, dest);
            }
            if (depth) {
                agfxTextureDestroy(device, depth);
            }
            return ok;
        }

        bool RunCpp(agfxComparisonFunction compareOp, const CompiledShader& shader, Image& outImage,
                    std::string& outError)
        {
            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                outError = "failed to create headless device";
                return false;
            }

            agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture depth = device.CreateTexture(DepthInfo());
            agfx::Texture dest = device.CreateTexture(DestInfo());

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = depth;
            rtInfo.format = kDepthFormat;
            rtInfo.isDepth = 1;
            agfx::RenderTarget depthTarget = device.CreateRenderTarget(rtInfo);

            agfx::TextureView depthSrv = device.CreateTextureView(DepthViewInfo(depth));
            agfx::TextureView destUav = device.CreateTextureView(DestViewInfo(dest));
            agfx::Sampler sampler = device.CreateSampler(SamplerInfo(compareOp));

            agfx::ComputePipeline pipeline;
            {
                agfx::ShaderModule module(device.Get(),
                    CreateShaderModule(device.Get(), shader, "main_sample_cmp_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
            }

            if (!depth.Get() || !dest.Get() || !depthTarget.Get() || !depthSrv.Get() || !destUav.Get() ||
                !sampler.Get() || !pipeline.Get()) {
                outError = "resource or pipeline creation failed";
                return false;
            }

            device.MakeResourcesResident();

            CmpPushConstants constants{};
            constants.source = (uint32_t)agfxTextureViewGetHandle(depthSrv);
            constants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
            constants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

            cmd.Begin();
            cmd.TextureBarrier(depth, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            {
                agfx::RenderPass pass = cmd.BeginRenderPass(DepthPassInfo(depthTarget));
            }
            cmd.TextureBarrier(depth, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                               AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            {
                agfx::ComputePass pass = cmd.BeginComputePass("sampler comparison");
                pass.SetPipeline(pipeline);
                pass.PushConstants(&constants, sizeof(constants));
                pass.Dispatch(kGroupsX, kGroupsY, 1);
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            if (!ReadbackTexture2D(device.Get(), queue, dest, kCmpWidth, kCmpHeight, kDestFormat,
                                   AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage)) {
                outError = "texture readback failed";
                return false;
            }
            return true;
        }

        bool RunEz(agfxComparisonFunction compareOp, const CompiledShader& shader, Image& outImage,
                   std::string& outError)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: no swap chain
            contextInfo.width = kCmpWidth;
            contextInfo.height = kCmpHeight;
            agfx::ez::Context context(contextInfo);

            agfx::Device& device = context.GetDevice();

            agfx::ez::Texture2D depth = context.CreateTexture2D(
                kCmpWidth, kCmpHeight, kDepthFormat,
                (agfxTextureUsage)(AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED));
            agfx::ez::Texture2D dest = context.CreateTexture2D(
                kCmpWidth, kCmpHeight, kDestFormat,
                (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED));

            // ez's TextureBase::SRV() always views a texture as its own format, so it cannot express
            // the D32 -> R32F reinterpretation comparison sampling needs. The view is built off the
            // raw texture instead; everything else on this path stays ez.
            agfx::TextureView depthSrv = device.CreateTextureView(DepthViewInfo(depth.Raw()));

            // ez has no sampler sugar; samplers come straight off the device.
            agfx::Sampler sampler = device.CreateSampler(SamplerInfo(compareOp));

            agfx::ComputePipeline pipeline;
            {
                agfx::ShaderModule module(device.Get(),
                    CreateShaderModule(device.Get(), shader, "main_sample_cmp_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
            }
            if (!depthSrv.Get() || !sampler.Get() || !pipeline.Get()) {
                outError = "resource or pipeline creation failed";
                return false;
            }

            // Mirrors CmpPushConstants field for field.
            agfx::ez::ShaderBindings bindings;
            bindings.BindTexture(depthSrv);
            bindings.BindSampler(sampler);
            bindings.BindTexture(dest.UAV());
            bindings.Write(kCmpWidth);
            bindings.Write(kCmpHeight);
            bindings.Write(kCmpBandCount);
            bindings.Write(0u); // padding0
            bindings.Write(0u); // padding1
            bindings.Write(kCmpNearer);
            bindings.Write(kCmpStoredDepth);
            bindings.Write(kCmpFarther);
            bindings.Write(0.0f);

            device.MakeResourcesResident();

            {
                agfx::ez::Frame frame = context.BeginFrame();
                context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

                // A depth-only pass: no color targets, and the clear is the whole of it.
                context.SetRenderTargets({}, &depth, AGFX_LOAD_OPERATION_CLEAR, nullptr,
                                         AGFX_LOAD_OPERATION_CLEAR, kCmpStoredDepth);
                context.EndActivePass();

                context.TransitionTexture(depth, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

                // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
                {
                    agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("sampler comparison");
                    pass.SetPipeline(pipeline);
                    pass.PushConstants(bindings.Data(), bindings.Size());
                    pass.Dispatch(kGroupsX, kGroupsY, 1);
                }
            }
            context.DrainGPU();

            if (!ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kCmpWidth,
                                   kCmpHeight, kDestFormat, dest.State(), outImage)) {
                outError = "texture readback failed";
                return false;
            }
            return true;
        }
    } // namespace

    void ExpectedBands(agfxComparisonFunction compareOp, bool outBands[kCmpBandCount])
    {
        // Band 0 compares kCmpNearer against the stored depth, band 1 kCmpStoredDepth, band 2
        // kCmpFarther -- so the three bands are exactly the `<`, `==` and `>` cases of `reference OP
        // storedDepth`, and each comparison function is the union of some subset of them.
        switch (compareOp) {
        case AGFX_COMPARISON_FUNCTION_NEVER:
            outBands[0] = false; outBands[1] = false; outBands[2] = false; break;
        case AGFX_COMPARISON_FUNCTION_LESS:
            outBands[0] = true;  outBands[1] = false; outBands[2] = false; break;
        case AGFX_COMPARISON_FUNCTION_EQUAL:
            outBands[0] = false; outBands[1] = true;  outBands[2] = false; break;
        case AGFX_COMPARISON_FUNCTION_LESS_EQUAL:
            outBands[0] = true;  outBands[1] = true;  outBands[2] = false; break;
        case AGFX_COMPARISON_FUNCTION_GREATER:
            outBands[0] = false; outBands[1] = false; outBands[2] = true;  break;
        case AGFX_COMPARISON_FUNCTION_NOT_EQUAL:
            outBands[0] = true;  outBands[1] = false; outBands[2] = true;  break;
        case AGFX_COMPARISON_FUNCTION_GREATER_EQUAL:
            outBands[0] = false; outBands[1] = true;  outBands[2] = true;  break;
        case AGFX_COMPARISON_FUNCTION_ALWAYS:
        default:
            outBands[0] = true;  outBands[1] = true;  outBands[2] = true;  break;
        }
    }

    bool ImageMatchesBands(const Image& image, agfxComparisonFunction compareOp)
    {
        if (!image.Valid() || image.width != kCmpWidth || image.height != kCmpHeight) {
            return false;
        }

        bool bands[kCmpBandCount];
        ExpectedBands(compareOp, bands);

        for (uint32_t y = 0; y < kCmpHeight; ++y) {
            for (uint32_t x = 0; x < kCmpWidth; ++x) {
                // Must match the band split in sampling_comparison.hlsl exactly.
                uint32_t band = (x * kCmpBandCount) / kCmpWidth;
                if (band >= kCmpBandCount) {
                    band = kCmpBandCount - 1;
                }
                const float expected = bands[band] ? 1.0f : 0.0f;
                const float* texel = &image.pixels[((size_t)y * kCmpWidth + x) * 4];
                // The comparison result is 0 or 1 exactly, and both survive an RGBA8 round trip
                // exactly, so this stays an exact comparison rather than a tolerance.
                if (texel[0] != expected || texel[1] != expected || texel[2] != expected) {
                    return false;
                }
            }
        }
        return true;
    }

    bool RenderSamplerComparison(TestApi api, agfxComparisonFunction compareOp, Image& outImage,
                                 std::string& outError)
    {
        outError.clear();

        const CompiledShader shader =
            CompileTestShader("sampling_comparison.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_cmp_cs");
        if (!shader.Valid()) {
            outError = "failed to compile sampling_comparison.hlsl";
            return false;
        }

        switch (api) {
        case TestApi::C:   return RunC(compareOp, shader, outImage, outError);
        case TestApi::Cpp: return RunCpp(compareOp, shader, outImage, outError);
        case TestApi::Ez:  return RunEz(compareOp, shader, outImage, outError);
        }
        outError = "unknown API flavor";
        return false;
    }
} // namespace agfxtest
