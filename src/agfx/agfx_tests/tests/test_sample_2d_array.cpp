/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sample 2D array texture".
//
// Two dispatches: texture_volume.hlsl:main_write_array_cs seeds a 64x64x4 source array texture,
// then sampling.hlsl:main_sample_2d_array_cs samples it through a sampler and writes each layer
// into a vertical band of one 64x256 destination.
//
// What this pins down over the Load()-based "compute texture load 2D array" test is the array
// coordinate's role in a *sampled* fetch: the third component is a layer index, not a normalized
// coordinate, and it is not filtered between layers. A backend that normalizes it, or that filters
// across the array axis, blends neighbouring layers together and produces bands the golden does not
// have. The 4x magnification in XY additionally keeps in-layer filtering under test.
//
// Seeding with the write shader rather than a host upload matches the load tests' reasoning: that
// path is pinned independently by test_compute_texture_write_2d_array.cpp, so a failure here is
// attributable to the sample.
//
// There is no Ez variant: ez::Context only exposes CreateTexture2D.

#include "../test_gpu.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;
    constexpr uint32_t kLayers = 4;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kGroupSize = 8; // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroupsX = (kWidth + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsY = (kHeight + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsZ = kLayers;
    constexpr const char* kGolden = "sample_2d_array.png";

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
        constants.sliceCount = kLayers;
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
        info.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = kLayers;
        info.mipLevels = 1;
        return info;
    }

    /// @brief The destination is a flat 2D texture holding all four bands stacked, so one golden
    /// covers the whole array.
    agfxTextureCreateInfo DestInfo()
    {
        agfxTextureCreateInfo info = SourceInfo();
        info.type = AGFX_TEXTURE_TYPE_2D; // Flat, not an array: the bands are stacked in Y.
        info.height = kHeight * kLayers;
        info.depthOrArrayLayers = 1;
        return info;
    }

    agfxTextureViewCreateInfo ArrayViewInfo(agfxTexture* texture, bool writeable)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
        info.baseMipLevel = 0;
        info.mipLevelCount = 1;
        info.baseArrayLayer = 0;
        info.arrayLayerCount = kLayers;
        info.writeable = writeable ? 1 : 0;
        return info;
    }

    agfxTextureViewCreateInfo FlatViewInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info = ArrayViewInfo(texture, /*writeable*/ true);
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

AGFX_TEST_TEXTURE(Sample2DArray, C, kWidth, kHeight* kLayers)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader writeShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_array_cs");
    const CompiledShader sampleShader =
        CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_2d_array_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile texture_volume.hlsl:main_write_array_cs");
    AGFX_EXPECT_MSG(sampleShader.Valid(), "failed to compile sampling.hlsl:main_sample_2d_array_cs");

    const agfxTextureCreateInfo sourceInfo = SourceInfo();
    const agfxTextureCreateInfo destInfo = DestInfo();
    agfxTexture* source = agfxTextureCreate(device, &sourceInfo);
    agfxTexture* dest = agfxTextureCreate(device, &destInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(dest);

    const agfxTextureViewCreateInfo sourceUavInfo = ArrayViewInfo(source, /*writeable*/ true);
    const agfxTextureViewCreateInfo sourceSrvInfo = ArrayViewInfo(source, /*writeable*/ false);
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
        CreateShaderModule(device, writeShader, "main_write_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    agfxShaderModule* sampleModule =
        CreateShaderModule(device, sampleShader, "main_sample_2d_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo writePipelineInfo = ComputePipelineInfo("array seed", writeModule);
    const agfxComputePipelineCreateInfo samplePipelineInfo = ComputePipelineInfo("array sample", sampleModule);
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
    seedConstants.sliceCount = kLayers;

    SamplingPushConstants sampleConstants = MakeSamplingConstants();
    sampleConstants.source = (uint32_t)agfxTextureViewGetHandle(sourceSrv);
    sampleConstants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
    sampleConstants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, 0);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, 0);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "array seed");
        agfxComputePassSetPipeline(pass, writePipeline);
        agfxComputePassPushConstants(pass, &seedConstants, sizeof(seedConstants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, kGroupsZ);
        agfxComputePassEnd(pass);
    });

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 0, 0, 0);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "array sample");
        agfxComputePassSetPipeline(pass, samplePipeline);
        agfxComputePassPushConstants(pass, &sampleConstants, sizeof(sampleConstants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, kGroupsZ);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), dest, kWidth, kHeight * kLayers, kFormat,
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

AGFX_TEST_TEXTURE(Sample2DArray, Cpp, kWidth, kHeight* kLayers)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader writeShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_array_cs");
    const CompiledShader sampleShader =
        CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_2d_array_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile texture_volume.hlsl:main_write_array_cs");
    AGFX_EXPECT_MSG(sampleShader.Valid(), "failed to compile sampling.hlsl:main_sample_2d_array_cs");

    agfx::Texture source = device.CreateTexture(SourceInfo());
    agfx::Texture dest = device.CreateTexture(DestInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(dest.Get());

    agfx::TextureView sourceUav = device.CreateTextureView(ArrayViewInfo(source, /*writeable*/ true));
    agfx::TextureView sourceSrv = device.CreateTextureView(ArrayViewInfo(source, /*writeable*/ false));
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
            CreateShaderModule(device.Get(), writeShader, "main_write_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule sampleModule(device.Get(),
            CreateShaderModule(device.Get(), sampleShader, "main_sample_2d_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        writePipeline = device.CreateComputePipeline(ComputePipelineInfo("array seed", writeModule));
        samplePipeline = device.CreateComputePipeline(ComputePipelineInfo("array sample", sampleModule));
    }
    AGFX_EXPECT_NOT_NULL(writePipeline.Get());
    AGFX_EXPECT_NOT_NULL(samplePipeline.Get());

    device.MakeResourcesResident();

    VolumePushConstants seedConstants{};
    seedConstants.destination = (uint32_t)sourceUav.GetHandle();
    seedConstants.width = kWidth;
    seedConstants.height = kHeight;
    seedConstants.sliceCount = kLayers;

    SamplingPushConstants sampleConstants = MakeSamplingConstants();
    sampleConstants.source = (uint32_t)sourceSrv.GetHandle();
    sampleConstants.samplerId = (uint32_t)sampler.GetHandle();
    sampleConstants.destination = (uint32_t)destUav.GetHandle();

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("array seed");
        pass.SetPipeline(writePipeline);
        pass.PushConstants(seedConstants);
        pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
    }
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, AGFX_SUBRESOURCE_ALL_MIPS,
                       AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("array sample");
        pass.SetPipeline(samplePipeline);
        pass.PushConstants(sampleConstants);
        pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, dest, kWidth, kHeight * kLayers, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
