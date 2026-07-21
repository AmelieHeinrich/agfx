/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sample 3D texture".
//
// Two dispatches: texture_volume.hlsl:main_write_3d_cs seeds a 64x64x4 source 3D texture, then
// sampling.hlsl:main_sample_3d_cs samples it through a sampler and writes each depth slice into a
// vertical band of one 64x256 destination.
//
// The counterpart to test_sample_2d_array.cpp, and the contrast is the point: a 3D texture's third
// sample coordinate is *normalized and filtered*, where a 2D array's is an unfiltered integer layer
// index. The shader aims at each slice's center ((z + 0.5) / depth) so the expected result is that
// slice rather than a blend of two, which makes a backend that treats w as a layer index fail
// visibly — flooring 0.125 to layer 0 collapses the gradient the seed writes along z.
//
// Seeding with the write shader rather than a host upload isn't a shortcut: a 3D texture's depth
// slices are addressed by copy-region z rather than by subresource layer, and the suite's host-side
// upload helpers only speak layers. The write path is pinned independently by
// test_compute_texture_write_3d.cpp, so a failure here is attributable to the sample.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;
    constexpr uint32_t kDepth = 4;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kGroupSize = 8; // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroupsX = (kWidth + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsY = (kHeight + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsZ = kDepth;
    constexpr const char* kGolden = "sample_3d.png";

    // The magnification window: the source's centre quarter, blown up to fill each band.
    constexpr float kUvScale = 0.25f;
    constexpr float kUvOffset = 0.375f;

    /// @brief Mirrors VolumePushConstants in texture_volume.hlsl (the seeding dispatch).
    struct VolumePushConstants
    {
        uint32_t source;
        uint32_t destination;
        uint32_t width;
        uint32_t height;
        uint32_t sliceCount;
        uint32_t padding0;
        uint32_t padding1;
        uint32_t padding2;
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

    SamplingPushConstants MakeSamplingConstants()
    {
        SamplingPushConstants constants{};
        constants.width = kWidth;
        constants.height = kHeight;
        constants.sliceCount = kDepth;
        constants.uvScale[0] = kUvScale;
        constants.uvScale[1] = kUvScale;
        constants.uvOffset[0] = kUvOffset;
        constants.uvOffset[1] = kUvOffset;
        return constants;
    }

    agfxSamplerCreateInfo SamplerInfo()
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

    agfxTextureCreateInfo SourceInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_3D;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = kDepth; // Depth, not layers.
        info.mipLevels = 1;
        return info;
    }

    /// @brief The destination is a flat 2D texture holding all four bands stacked, so one golden
    /// covers the whole volume.
    agfxTextureCreateInfo DestInfo()
    {
        agfxTextureCreateInfo info = SourceInfo();
        info.type = AGFX_TEXTURE_TYPE_2D; // Flat, not a volume: the bands are stacked in Y.
        info.height = kHeight * kDepth;
        info.depthOrArrayLayers = 1;
        return info;
    }

    agfxTextureViewCreateInfo VolumeViewInfo(agfxTexture* texture, bool writeable)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_3D;
        info.baseMipLevel = 0;
        info.mipLevelCount = 1;
        info.baseArrayLayer = 0;
        info.arrayLayerCount = 1; // A 3D texture has no array layers; depth lives in the texture.
        info.writeable = writeable ? 1 : 0;
        return info;
    }

    agfxTextureViewCreateInfo FlatViewInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info = VolumeViewInfo(texture, /*writeable*/ true);
        info.type = AGFX_TEXTURE_TYPE_2D; // The stacked destination is a plain 2D texture.
        info.arrayLayerCount = 1;
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
} // namespace

AGFX_TEST_TEXTURE(Sample3D, C, kWidth, kHeight* kDepth)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader writeShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_3d_cs");
    const CompiledShader sampleShader =
        CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_3d_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile texture_volume.hlsl:main_write_3d_cs");
    AGFX_EXPECT_MSG(sampleShader.Valid(), "failed to compile sampling.hlsl:main_sample_3d_cs");

    const agfxTextureCreateInfo sourceInfo = SourceInfo();
    const agfxTextureCreateInfo destInfo = DestInfo();
    agfxTexture* source = agfxTextureCreate(device, &sourceInfo);
    agfxTexture* dest = agfxTextureCreate(device, &destInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(dest);

    const agfxTextureViewCreateInfo sourceUavInfo = VolumeViewInfo(source, /*writeable*/ true);
    const agfxTextureViewCreateInfo sourceSrvInfo = VolumeViewInfo(source, /*writeable*/ false);
    const agfxTextureViewCreateInfo destUavInfo = FlatViewInfo(dest);
    agfxTextureView* sourceUav = agfxTextureViewCreate(device, &sourceUavInfo);
    agfxTextureView* sourceSrv = agfxTextureViewCreate(device, &sourceSrvInfo);
    agfxTextureView* destUav = agfxTextureViewCreate(device, &destUavInfo);
    AGFX_EXPECT_NOT_NULL(sourceUav);
    AGFX_EXPECT_NOT_NULL(sourceSrv);
    AGFX_EXPECT_NOT_NULL(destUav);

    const agfxSamplerCreateInfo samplerInfo = SamplerInfo();
    agfxSampler* sampler = agfxSamplerCreate(device, &samplerInfo);
    AGFX_EXPECT_NOT_NULL(sampler);

    agfxShaderModule* writeModule =
        CreateShaderModule(device, writeShader, "main_write_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    agfxShaderModule* sampleModule =
        CreateShaderModule(device, sampleShader, "main_sample_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo writePipelineInfo = ComputePipelineInfo("volume seed", writeModule);
    const agfxComputePipelineCreateInfo samplePipelineInfo = ComputePipelineInfo("volume sample", sampleModule);
    agfxComputePipeline* writePipeline = agfxComputePipelineCreate(device, &writePipelineInfo);
    agfxComputePipeline* samplePipeline = agfxComputePipelineCreate(device, &samplePipelineInfo);
    agfxShaderModuleDestroy(device, writeModule);
    agfxShaderModuleDestroy(device, sampleModule);
    AGFX_EXPECT_NOT_NULL(writePipeline);
    AGFX_EXPECT_NOT_NULL(samplePipeline);

    agfxDeviceMakeResourcesResident(device);

    VolumePushConstants seedConstants{};
    seedConstants.destination = (uint32_t)agfxTextureViewGetHandle(sourceUav);
    seedConstants.width = kWidth;
    seedConstants.height = kHeight;
    seedConstants.sliceCount = kDepth;

    SamplingPushConstants sampleConstants = MakeSamplingConstants();
    sampleConstants.source = (uint32_t)agfxTextureViewGetHandle(sourceSrv);
    sampleConstants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
    sampleConstants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        // Every layer must be transitioned, not just layer 0, and agglomerate must be true: on the
        // Metal backend a barrier recorded with agglomerate=false is a no-op, so passing false here
        // left the seed and sample dispatches with no synchronization at all between them.
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "volume seed");
        agfxComputePassSetPipeline(pass, writePipeline);
        agfxComputePassPushConstants(pass, &seedConstants, sizeof(seedConstants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, kGroupsZ);
        agfxComputePassEnd(pass);
    });

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "volume sample");
        agfxComputePassSetPipeline(pass, samplePipeline);
        agfxComputePassPushConstants(pass, &sampleConstants, sizeof(sampleConstants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, kGroupsZ);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), dest, kWidth, kHeight * kDepth, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, samplePipeline);
    agfxComputePipelineDestroy(device, writePipeline);
    agfxSamplerDestroy(device, sampler);
    agfxTextureViewDestroy(device, destUav);
    agfxTextureViewDestroy(device, sourceSrv);
    agfxTextureViewDestroy(device, sourceUav);
    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(Sample3D, Cpp, kWidth, kHeight* kDepth)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader writeShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_3d_cs");
    const CompiledShader sampleShader =
        CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_3d_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile texture_volume.hlsl:main_write_3d_cs");
    AGFX_EXPECT_MSG(sampleShader.Valid(), "failed to compile sampling.hlsl:main_sample_3d_cs");

    agfx::Texture source = device.CreateTexture(SourceInfo());
    agfx::Texture dest = device.CreateTexture(DestInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(dest.Get());

    agfx::TextureView sourceUav = device.CreateTextureView(VolumeViewInfo(source, /*writeable*/ true));
    agfx::TextureView sourceSrv = device.CreateTextureView(VolumeViewInfo(source, /*writeable*/ false));
    agfx::TextureView destUav = device.CreateTextureView(FlatViewInfo(dest));
    AGFX_EXPECT_NOT_NULL(sourceUav.Get());
    AGFX_EXPECT_NOT_NULL(sourceSrv.Get());
    AGFX_EXPECT_NOT_NULL(destUav.Get());

    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());
    AGFX_EXPECT_NOT_NULL(sampler.Get());

    agfx::ComputePipeline writePipeline;
    agfx::ComputePipeline samplePipeline;
    {
        agfx::ShaderModule writeModule(device.Get(),
            CreateShaderModule(device.Get(), writeShader, "main_write_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule sampleModule(device.Get(),
            CreateShaderModule(device.Get(), sampleShader, "main_sample_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        writePipeline = device.CreateComputePipeline(ComputePipelineInfo("volume seed", writeModule));
        samplePipeline = device.CreateComputePipeline(ComputePipelineInfo("volume sample", sampleModule));
    }
    AGFX_EXPECT_NOT_NULL(writePipeline.Get());
    AGFX_EXPECT_NOT_NULL(samplePipeline.Get());

    device.MakeResourcesResident();

    VolumePushConstants seedConstants{};
    seedConstants.destination = (uint32_t)sourceUav.GetHandle();
    seedConstants.width = kWidth;
    seedConstants.height = kHeight;
    seedConstants.sliceCount = kDepth;

    SamplingPushConstants sampleConstants = MakeSamplingConstants();
    sampleConstants.source = (uint32_t)sourceSrv.GetHandle();
    sampleConstants.samplerId = (uint32_t)sampler.GetHandle();
    sampleConstants.destination = (uint32_t)destUav.GetHandle();

    cmd.Begin();
    // agglomerate must be true: a barrier recorded with false is a no-op on the Metal backend, which
    // left the seed and sample dispatches unsynchronized and made this test intermittently fail.
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("volume seed");
        pass.SetPipeline(writePipeline);
        pass.PushConstants(seedConstants);
        pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
    }
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, AGFX_SUBRESOURCE_ALL_MIPS,
                       AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("volume sample");
        pass.SetPipeline(samplePipeline);
        pass.PushConstants(sampleConstants);
        pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, dest, kWidth, kHeight * kDepth, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(Sample3D, Ez, kWidth, kHeight* kDepth)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight * kDepth;
    agfx::ez::Context context(contextInfo);

    const CompiledShader writeShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_3d_cs");
    const CompiledShader sampleShader =
        CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_3d_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile texture_volume.hlsl:main_write_3d_cs");
    AGFX_EXPECT_MSG(sampleShader.Valid(), "failed to compile sampling.hlsl:main_sample_3d_cs");

    agfx::Device& device = context.GetDevice();

    const agfxTextureUsage usage =
        (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
    agfx::ez::Texture3D source = context.CreateTexture3D(kWidth, kHeight, kDepth, kFormat, usage);
    // The stacked destination is a plain 2D texture, so its default UAV is already FlatViewInfo().
    agfx::ez::Texture2D dest = context.CreateTexture2D(kWidth, kHeight * kDepth, kFormat, usage);

    // ez has no sampler sugar; samplers come straight off the device.
    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());
    AGFX_EXPECT_NOT_NULL(sampler.Get());

    agfx::ComputePipeline writePipeline;
    agfx::ComputePipeline samplePipeline;
    {
        agfx::ShaderModule writeModule(device.Get(),
            CreateShaderModule(device.Get(), writeShader, "main_write_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule sampleModule(device.Get(),
            CreateShaderModule(device.Get(), sampleShader, "main_sample_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        writePipeline = device.CreateComputePipeline(ComputePipelineInfo("volume seed", writeModule));
        samplePipeline = device.CreateComputePipeline(ComputePipelineInfo("volume sample", sampleModule));
    }
    AGFX_EXPECT_NOT_NULL(writePipeline.Get());
    AGFX_EXPECT_NOT_NULL(samplePipeline.Get());

    agfx::ez::ShaderBindings seedBindings;
    seedBindings.Write(0u); // unused source slot
    seedBindings.BindTexture(source.UAV());
    seedBindings.Write(kWidth);
    seedBindings.Write(kHeight);
    seedBindings.Write(kDepth);

    // Mirrors SamplingPushConstants field for field.
    agfx::ez::ShaderBindings sampleBindings;
    sampleBindings.BindTexture(source.SRV());
    sampleBindings.BindSampler(sampler);
    sampleBindings.BindTexture(dest.UAV());
    sampleBindings.Write(kWidth);
    sampleBindings.Write(kHeight);
    sampleBindings.Write(kDepth);
    sampleBindings.Write(kUvScale);
    sampleBindings.Write(kUvScale);
    sampleBindings.Write(kUvOffset);
    sampleBindings.Write(kUvOffset);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        {
            agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("volume seed");
            pass.SetPipeline(writePipeline);
            pass.PushConstants(seedBindings.Data(), seedBindings.Size());
            pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
        }

        // The seed writes the source through its UAV; the sample reads it through its SRV.
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        {
            agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("volume sample");
            pass.SetPipeline(samplePipeline);
            pass.PushConstants(sampleBindings.Data(), sampleBindings.Size());
            pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
        }
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kWidth,
                                          kHeight * kDepth, kFormat, dest.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
