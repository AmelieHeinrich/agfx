/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sample 2D texture".
//
// Dispatches data/shaders/tests/sampling.hlsl:main_sample_2d_cs, which pulls a sampler out of
// SamplerDescriptorHeap and SampleLevel()s a seeded source texture into a storage texture. Where
// the "compute texture load 2D" test covers texel addressing with Load(), this covers the sampler
// path on top of it: a broken heap lookup, an ignored filter, or a half-texel coordinate offset all
// change the result.
//
// The source is magnified 4x (a quarter of it stretched across the whole destination) so LINEAR
// filtering actually has something to interpolate -- at 1:1 a nearest and a linear sampler agree
// everywhere and the test would pass with filtering entirely unimplemented.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kRowBytes = kWidth * kBytesPerPixel; // 256: the D3D12 row-pitch alignment.
    constexpr uint64_t kImageBytes = uint64_t(kRowBytes) * kHeight;
    constexpr uint32_t kGroupSize = 8;                   // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroupsX = (kWidth + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsY = (kHeight + kGroupSize - 1) / kGroupSize;
    constexpr const char* kGolden = "sample_2d.png";

    // The magnification window: the source's centre quarter, blown up to fill the destination.
    constexpr float kUvScale = 0.25f;
    constexpr float kUvOffset = 0.375f;

    /// @brief Mirrors SamplingPushConstants in sampling.hlsl.
    struct PushConstants
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

    PushConstants MakeConstants()
    {
        PushConstants constants{};
        constants.width = kWidth;
        constants.height = kHeight;
        constants.sliceCount = 1;
        constants.uvScale[0] = kUvScale;
        constants.uvScale[1] = kUvScale;
        constants.uvOffset[0] = kUvOffset;
        constants.uvOffset[1] = kUvOffset;
        return constants;
    }

    /// @brief A coarse 8x8-block checker over a ramp: the blocks give the linear filter hard edges
    /// to ramp across under magnification, and the ramp keeps the image asymmetric so a flipped or
    /// transposed sample is visible.
    std::vector<uint8_t> SourcePixels()
    {
        std::vector<uint8_t> pixels((size_t)kImageBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                const bool checker = (((x / 8u) + (y / 8u)) % 2u) != 0u;
                texel[0] = (uint8_t)(x * 255u / (kWidth - 1u));
                texel[1] = (uint8_t)(y * 255u / (kHeight - 1u));
                texel[2] = checker ? 255u : 0u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief LINEAR with CLAMP_TO_EDGE. The magnification window sits well inside [0,1], so the
    /// address mode never fires here -- that is the address mode test's job, and keeping it out of
    /// the way means a failure in this test points at filtering or at the heap lookup.
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
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = AGFX_TEXTURE_USAGE_SAMPLED;
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxTextureCreateInfo DestInfo()
    {
        agfxTextureCreateInfo info = SourceInfo();
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        return info;
    }

    agfxTextureViewCreateInfo ViewInfo(agfxTexture* texture, bool writeable)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.baseMipLevel = 0;
        info.mipLevelCount = 1;
        info.baseArrayLayer = 0;
        info.arrayLayerCount = 1;
        info.writeable = writeable ? 1 : 0;
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "sample 2d";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(Sample2D, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_2d_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile sampling.hlsl:main_sample_2d_cs");

    const agfxTextureCreateInfo sourceInfo = SourceInfo();
    const agfxTextureCreateInfo destInfo = DestInfo();
    agfxTexture* source = agfxTextureCreate(device, &sourceInfo);
    agfxTexture* dest = agfxTextureCreate(device, &destInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(dest);

    const agfxTextureViewCreateInfo srvInfo = ViewInfo(source, false);
    const agfxTextureViewCreateInfo uavInfo = ViewInfo(dest, true);
    agfxTextureView* srv = agfxTextureViewCreate(device, &srvInfo);
    agfxTextureView* uav = agfxTextureViewCreate(device, &uavInfo);
    AGFX_EXPECT_NOT_NULL(srv);
    AGFX_EXPECT_NOT_NULL(uav);

    const agfxSamplerCreateInfo samplerInfo = SamplerInfo();
    agfxSampler* sampler = agfxSamplerCreate(device, &samplerInfo);
    AGFX_EXPECT_NOT_NULL(sampler);

    agfxShaderModule* module = CreateShaderModule(device, shader, "main_sample_2d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint8_t> pixels = SourcePixels();
    const bool seeded = UploadTexture2D(device, gpu.Queue(), source, kWidth, kHeight, kFormat,
                                        pixels.data(), AGFX_RESOURCE_STATE_COMMON);

    PushConstants constants = MakeConstants();
    constants.source = (uint32_t)agfxTextureViewGetHandle(srv);
    constants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
    constants.destination = (uint32_t)agfxTextureViewGetHandle(uav);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 0, 0, 0);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "sample 2d");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), dest, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, pipeline);
    agfxSamplerDestroy(device, sampler);
    agfxTextureViewDestroy(device, uav);
    agfxTextureViewDestroy(device, srv);
    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(seeded, "failed to seed the source texture");
    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(Sample2D, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_2d_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile sampling.hlsl:main_sample_2d_cs");

    agfx::Texture source = device.CreateTexture(SourceInfo());
    agfx::Texture dest = device.CreateTexture(DestInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(dest.Get());

    agfx::TextureView srv = device.CreateTextureView(ViewInfo(source, false));
    agfx::TextureView uav = device.CreateTextureView(ViewInfo(dest, true));
    AGFX_EXPECT_NOT_NULL(srv.Get());
    AGFX_EXPECT_NOT_NULL(uav.Get());

    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());
    AGFX_EXPECT_NOT_NULL(sampler.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_sample_2d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    const std::vector<uint8_t> pixels = SourcePixels();
    AGFX_EXPECT_MSG(UploadTexture2D(device.Get(), queue, source, kWidth, kHeight, kFormat, pixels.data(),
                                    AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the source texture");

    PushConstants constants = MakeConstants();
    constants.source = (uint32_t)srv.GetHandle();
    constants.samplerId = (uint32_t)sampler.GetHandle();
    constants.destination = (uint32_t)uav.GetHandle();

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("sample 2d");
        pass.SetPipeline(pipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupsX, kGroupsY, 1);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(Sample2D, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader = CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_2d_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile sampling.hlsl:main_sample_2d_cs");

    agfx::Device& device = context.GetDevice();

    const std::vector<uint8_t> pixels = SourcePixels();
    agfx::ez::Texture2D source =
        context.CreateTexture2D(kWidth, kHeight, kFormat, AGFX_TEXTURE_USAGE_SAMPLED, pixels.data(), kRowBytes);
    agfx::ez::Texture2D dest = context.CreateTexture2D(kWidth, kHeight, kFormat, AGFX_TEXTURE_USAGE_STORAGE);

    // ez has no sampler wrapper: samplers come straight off the Context's device and are bound
    // through ShaderBindings, which is exactly the handle the C/C++ flavors pack by hand.
    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());
    AGFX_EXPECT_NOT_NULL(sampler.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_sample_2d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    agfx::ez::ShaderBindings bindings;
    bindings.BindTexture(source.SRV());
    bindings.BindSampler(sampler);
    bindings.BindTexture(dest.UAV());
    bindings.Write(kWidth);
    bindings.Write(kHeight);
    bindings.Write(1u); // sliceCount
    bindings.Write(kUvScale);
    bindings.Write(kUvScale);
    bindings.Write(kUvOffset);
    bindings.Write(kUvOffset);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("sample 2d");
        pass.SetPipeline(pipeline);
        pass.PushConstants(bindings.Data(), bindings.Size());
        pass.Dispatch(kGroupsX, kGroupsY, 1);
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(),
                                          kWidth, kHeight, kFormat, dest.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
