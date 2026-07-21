/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "sampling_common.h"

#include <agfx/agfx_ez.hpp>

namespace agfxtest
{
    namespace
    {
        constexpr uint32_t kGroupSize = 8; // Matches [numthreads(8,8,1)].
        constexpr uint32_t kGroupsX = (kSamplerWidth + kGroupSize - 1) / kGroupSize;
        constexpr uint32_t kGroupsY = (kSamplerHeight + kGroupSize - 1) / kGroupSize;

        /// @brief Mirrors TexturePushConstants in texture_ops.hlsl (the seeding dispatch).
        struct SeedPushConstants
        {
            uint32_t source;
            uint32_t destination;
            uint32_t width;
            uint32_t height;
        };

        /// @brief Mirrors SamplingPushConstants in sampling.hlsl (the sampling dispatch).
        struct SamplingPushConstants
        {
            uint32_t source;
            uint32_t samplerId;
            uint32_t destination;
            uint32_t width;
            uint32_t height;
            uint32_t sliceCount;
            float uvScale[2];
            float uvOffset[2];
            float padding0[2];
        };

        SamplingPushConstants MakeSamplingConstants(const SampleState& state)
        {
            SamplingPushConstants constants{};
            constants.width = kSamplerWidth;
            constants.height = kSamplerHeight;
            constants.sliceCount = 1; // Plain 2D: the layered paths have their own tests.
            constants.uvScale[0] = state.uvScale[0];
            constants.uvScale[1] = state.uvScale[1];
            constants.uvOffset[0] = state.uvOffset[0];
            constants.uvOffset[1] = state.uvOffset[1];
            return constants;
        }

        agfxTextureCreateInfo TextureInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kSamplerFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kSamplerWidth;
            info.height = kSamplerHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxTextureViewCreateInfo ViewInfo(agfxTexture* texture, bool writeable)
        {
            agfxTextureViewCreateInfo info{};
            info.texture = texture;
            info.format = kSamplerFormat;
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.baseMipLevel = 0;
            info.mipLevelCount = 1;
            info.baseArrayLayer = 0;
            info.arrayLayerCount = 1;
            info.writeable = writeable ? 1 : 0;
            return info;
        }

        agfxComputePipelineCreateInfo ComputePipelineInfo(const char* name, agfxShaderModule* module)
        {
            agfxComputePipelineCreateInfo info{};
            info.name = name;
            info.computeShader = module;
            info.groupSizeX = kGroupSize;
            info.groupSizeY = kGroupSize;
            info.groupSizeZ = 1;
            return info;
        }

        bool RenderSampleC(const SampleState& state, const CompiledShader& seedShader,
                           const CompiledShader& sampleShader, Image& outImage)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                return false;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo textureInfo = TextureInfo();
            agfxTexture* source = agfxTextureCreate(device, &textureInfo);
            agfxTexture* dest = agfxTextureCreate(device, &textureInfo);

            const agfxTextureViewCreateInfo sourceUavInfo = ViewInfo(source, /*writeable*/ true);
            const agfxTextureViewCreateInfo sourceSrvInfo = ViewInfo(source, /*writeable*/ false);
            const agfxTextureViewCreateInfo destUavInfo = ViewInfo(dest, /*writeable*/ true);
            agfxTextureView* sourceUav = agfxTextureViewCreate(device, &sourceUavInfo);
            agfxTextureView* sourceSrv = agfxTextureViewCreate(device, &sourceSrvInfo);
            agfxTextureView* destUav = agfxTextureViewCreate(device, &destUavInfo);

            agfxSampler* sampler = agfxSamplerCreate(device, &state.sampler);

            agfxShaderModule* seedModule =
                CreateShaderModule(device, seedShader, "main_write_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
            agfxShaderModule* sampleModule =
                CreateShaderModule(device, sampleShader, "main_sample_2d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
            const agfxComputePipelineCreateInfo seedPipelineInfo = ComputePipelineInfo("sampler seed", seedModule);
            const agfxComputePipelineCreateInfo samplePipelineInfo = ComputePipelineInfo("sampler sample", sampleModule);
            agfxComputePipeline* seedPipeline = agfxComputePipelineCreate(device, &seedPipelineInfo);
            agfxComputePipeline* samplePipeline = agfxComputePipelineCreate(device, &samplePipelineInfo);
            agfxShaderModuleDestroy(device, seedModule);
            agfxShaderModuleDestroy(device, sampleModule);

            bool ok = source && dest && sourceUav && sourceSrv && destUav && sampler && seedPipeline &&
                      samplePipeline;
            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                SeedPushConstants seedConstants{};
                seedConstants.destination = (uint32_t)agfxTextureViewGetHandle(sourceUav);
                seedConstants.width = kSamplerWidth;
                seedConstants.height = kSamplerHeight;

                SamplingPushConstants sampleConstants = MakeSamplingConstants(state);
                sampleConstants.source = (uint32_t)agfxTextureViewGetHandle(sourceSrv);
                sampleConstants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
                sampleConstants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    // agglomerate must be true: on the Metal backend a barrier recorded with false is
                    // a no-op, which would leave the seed and sample dispatches unsynchronized.
                    agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxComputePass* pass = agfxComputePassBegin(cmd, "sampler seed");
                    agfxComputePassSetPipeline(pass, seedPipeline);
                    agfxComputePassPushConstants(pass, &seedConstants, sizeof(seedConstants));
                    agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
                    agfxComputePassEnd(pass);
                });

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                    AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                    agfxComputePass* pass = agfxComputePassBegin(cmd, "sampler sample");
                    agfxComputePassSetPipeline(pass, samplePipeline);
                    agfxComputePassPushConstants(pass, &sampleConstants, sizeof(sampleConstants));
                    agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
                    agfxComputePassEnd(pass);
                });

                ok = ReadbackTexture2D(device, gpu.Queue(), dest, kSamplerWidth, kSamplerHeight, kSamplerFormat,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage);
            }

            if (samplePipeline) {
                agfxComputePipelineDestroy(device, samplePipeline);
            }
            if (seedPipeline) {
                agfxComputePipelineDestroy(device, seedPipeline);
            }
            if (sampler) {
                agfxSamplerDestroy(device, sampler);
            }
            if (destUav) {
                agfxTextureViewDestroy(device, destUav);
            }
            if (sourceSrv) {
                agfxTextureViewDestroy(device, sourceSrv);
            }
            if (sourceUav) {
                agfxTextureViewDestroy(device, sourceUav);
            }
            if (dest) {
                agfxTextureDestroy(device, dest);
            }
            if (source) {
                agfxTextureDestroy(device, source);
            }
            return ok;
        }

        bool RenderSampleCpp(const SampleState& state, const CompiledShader& seedShader,
                             const CompiledShader& sampleShader, Image& outImage)
        {
            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                return false;
            }

            agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture source = device.CreateTexture(TextureInfo());
            agfx::Texture dest = device.CreateTexture(TextureInfo());

            agfx::TextureView sourceUav = device.CreateTextureView(ViewInfo(source, /*writeable*/ true));
            agfx::TextureView sourceSrv = device.CreateTextureView(ViewInfo(source, /*writeable*/ false));
            agfx::TextureView destUav = device.CreateTextureView(ViewInfo(dest, /*writeable*/ true));

            agfx::Sampler sampler = device.CreateSampler(state.sampler);

            agfx::ComputePipeline seedPipeline;
            agfx::ComputePipeline samplePipeline;
            {
                agfx::ShaderModule seedModule(device.Get(),
                    CreateShaderModule(device.Get(), seedShader, "main_write_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                agfx::ShaderModule sampleModule(device.Get(),
                    CreateShaderModule(device.Get(), sampleShader, "main_sample_2d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                seedPipeline = device.CreateComputePipeline(ComputePipelineInfo("sampler seed", seedModule));
                samplePipeline = device.CreateComputePipeline(ComputePipelineInfo("sampler sample", sampleModule));
            }

            if (!source.Get() || !dest.Get() || !sourceUav.Get() || !sourceSrv.Get() || !destUav.Get() ||
                !sampler.Get() || !seedPipeline.Get() || !samplePipeline.Get()) {
                return false;
            }

            device.MakeResourcesResident();

            SeedPushConstants seedConstants{};
            seedConstants.destination = (uint32_t)sourceUav.GetHandle();
            seedConstants.width = kSamplerWidth;
            seedConstants.height = kSamplerHeight;

            SamplingPushConstants sampleConstants = MakeSamplingConstants(state);
            sampleConstants.source = (uint32_t)sourceSrv.GetHandle();
            sampleConstants.samplerId = (uint32_t)sampler.GetHandle();
            sampleConstants.destination = (uint32_t)destUav.GetHandle();

            cmd.Begin();
            cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            {
                agfx::ComputePass pass = cmd.BeginComputePass("sampler seed");
                pass.SetPipeline(seedPipeline);
                pass.PushConstants(seedConstants);
                pass.Dispatch(kGroupsX, kGroupsY, 1);
            }
            cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                               AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, AGFX_SUBRESOURCE_ALL_MIPS,
                               AGFX_SUBRESOURCE_ALL_LAYERS, true);
            {
                agfx::ComputePass pass = cmd.BeginComputePass("sampler sample");
                pass.SetPipeline(samplePipeline);
                pass.PushConstants(sampleConstants);
                pass.Dispatch(kGroupsX, kGroupsY, 1);
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            return ReadbackTexture2D(device.Get(), queue, dest, kSamplerWidth, kSamplerHeight, kSamplerFormat,
                                     AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage);
        }

        bool RenderSampleEz(const SampleState& state, const CompiledShader& seedShader,
                            const CompiledShader& sampleShader, Image& outImage)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: no swap chain
            contextInfo.width = kSamplerWidth;
            contextInfo.height = kSamplerHeight;
            agfx::ez::Context context(contextInfo);

            agfx::Device& device = context.GetDevice();

            const agfxTextureUsage usage =
                (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
            agfx::ez::Texture2D source = context.CreateTexture2D(kSamplerWidth, kSamplerHeight, kSamplerFormat, usage);
            agfx::ez::Texture2D dest = context.CreateTexture2D(kSamplerWidth, kSamplerHeight, kSamplerFormat, usage);

            // ez has no sampler sugar; samplers come straight off the device.
            agfx::Sampler sampler = device.CreateSampler(state.sampler);
            if (!sampler.Get()) {
                return false;
            }

            agfx::ComputePipeline seedPipeline;
            agfx::ComputePipeline samplePipeline;
            {
                agfx::ShaderModule seedModule(device.Get(),
                    CreateShaderModule(device.Get(), seedShader, "main_write_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                agfx::ShaderModule sampleModule(device.Get(),
                    CreateShaderModule(device.Get(), sampleShader, "main_sample_2d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
                seedPipeline = device.CreateComputePipeline(ComputePipelineInfo("sampler seed", seedModule));
                samplePipeline = device.CreateComputePipeline(ComputePipelineInfo("sampler sample", sampleModule));
            }
            if (!seedPipeline.Get() || !samplePipeline.Get()) {
                return false;
            }

            // Mirrors SeedPushConstants field for field.
            agfx::ez::ShaderBindings seedBindings;
            seedBindings.Write(0u); // unused source slot
            seedBindings.BindTexture(source.UAV());
            seedBindings.Write(kSamplerWidth);
            seedBindings.Write(kSamplerHeight);

            // Mirrors SamplingPushConstants field for field.
            agfx::ez::ShaderBindings sampleBindings;
            sampleBindings.BindTexture(source.SRV());
            sampleBindings.BindSampler(sampler);
            sampleBindings.BindTexture(dest.UAV());
            sampleBindings.Write(kSamplerWidth);
            sampleBindings.Write(kSamplerHeight);
            sampleBindings.Write(1u); // sliceCount
            sampleBindings.Write(state.uvScale[0]);
            sampleBindings.Write(state.uvScale[1]);
            sampleBindings.Write(state.uvOffset[0]);
            sampleBindings.Write(state.uvOffset[1]);

            device.MakeResourcesResident();

            {
                agfx::ez::Frame frame = context.BeginFrame();
                context.TransitionTexture(source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);
                context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

                // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
                {
                    agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("sampler seed");
                    pass.SetPipeline(seedPipeline);
                    pass.PushConstants(seedBindings.Data(), seedBindings.Size());
                    pass.Dispatch(kGroupsX, kGroupsY, 1);
                }

                // The seed writes the source through its UAV; the sample reads it through its SRV.
                context.TransitionTexture(source, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

                {
                    agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("sampler sample");
                    pass.SetPipeline(samplePipeline);
                    pass.PushConstants(sampleBindings.Data(), sampleBindings.Size());
                    pass.Dispatch(kGroupsX, kGroupsY, 1);
                }
            }
            context.DrainGPU();

            return ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kSamplerWidth,
                                     kSamplerHeight, kSamplerFormat, dest.State(), outImage);
        }
    } // namespace

    agfxSamplerCreateInfo DefaultSamplerInfo()
    {
        agfxSamplerCreateInfo info{};
        info.filter = AGFX_SAMPLER_FILTER_LINEAR;
        info.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.maxAnisotropy = 1.0f;
        info.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
        info.minLod = 0.0f;
        info.maxLod = 0.0f;
        return info;
    }

    bool RenderSample(TestApi api, const SampleState& state, Image& outImage)
    {
        const CompiledShader seedShader =
            CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_cs");
        const CompiledShader sampleShader =
            CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_2d_cs");
        if (!seedShader.Valid() || !sampleShader.Valid()) {
            return false;
        }

        switch (api) {
        case TestApi::C:   return RenderSampleC(state, seedShader, sampleShader, outImage);
        case TestApi::Cpp: return RenderSampleCpp(state, seedShader, sampleShader, outImage);
        case TestApi::Ez:  return RenderSampleEz(state, seedShader, sampleShader, outImage);
        }
        return false;
    }
} // namespace agfxtest
