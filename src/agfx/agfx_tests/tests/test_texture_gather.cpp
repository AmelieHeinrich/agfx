/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "texture gather".
//
// Gather returns the four texels a bilinear tap would blend, unfiltered. This test aims the sample
// point at the exact corner shared by four texels — so all four are named unambiguously — and writes
// the four gathered red values into the destination's RGBA channels, one per channel. The expected
// result is then stated analytically from the host-uploaded source, which makes the *ordering* of
// Gather's four results the thing under test.
//
// Ordering is the interesting part: the HLSL contract is counter-clockwise from the lower-left,
// (-,+) (+,+) (+,-) (-,-) relative to the sample point, and it is the detail most likely to differ
// between backends. A test that only checked the set of four values, or that compared a filtered
// blend, could not see a permutation at all.
//
// The right and bottom edges gather a clamped texel twice (CLAMP_TO_EDGE at uv == 1.0); the host
// expectation replicates that clamp rather than skipping those texels, so the edge behaviour is
// covered too.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kSize = 64; // 64 * 4 bytes = 256: D3D12's row-pitch alignment.
    constexpr uint32_t kGroupSize = 8; // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroups = (kSize + kGroupSize - 1) / kGroupSize;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    constexpr const char* kGolden = "texture_gather.png";

    /// @brief Mirrors SamplingPushConstants in data/shaders/tests/sampling.hlsl.
    struct GatherPushConstants
    {
        uint32_t source = 0;
        uint32_t samplerId = 0;
        uint32_t destination = 0;
        uint32_t width = kSize;
        uint32_t height = kSize;
        uint32_t sliceCount = 1;
        float uvScale[2] = {1.0f, 1.0f};   // unused by main_gather_cs
        float uvOffset[2] = {0.0f, 0.0f};  // unused by main_gather_cs
        float padding0[2] = {0.0f, 0.0f};
    };

    /// @brief The red channel of source texel (x, y). Deliberately not separable in x and y: a
    /// permutation of the four gathered results has to change the answer, which it would not if red
    /// depended on x alone.
    uint8_t SourceRed(uint32_t x, uint32_t y)
    {
        return (uint8_t)((x * 5u + y * 37u + ((x * y) & 0x1Fu)) & 0xFFu);
    }

    std::vector<uint8_t> SourcePixels()
    {
        std::vector<uint8_t> pixels((size_t)kSize * kSize * 4);
        for (uint32_t y = 0; y < kSize; ++y) {
            for (uint32_t x = 0; x < kSize; ++x) {
                uint8_t* texel = &pixels[((size_t)y * kSize + x) * 4];
                texel[0] = SourceRed(x, y);
                // The other channels are set but never gathered; GatherRed must ignore them, and a
                // backend gathering the wrong channel produces an obviously wrong image.
                texel[1] = (uint8_t)(255u - x);
                texel[2] = (uint8_t)(255u - y);
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief What main_gather_cs must produce: the four texels around the corner at (x+1, y+1), in
    /// HLSL's counter-clockwise-from-lower-left order, with CLAMP_TO_EDGE applied at the far edges.
    std::vector<uint8_t> ExpectedPixels()
    {
        std::vector<uint8_t> pixels((size_t)kSize * kSize * 4);
        for (uint32_t y = 0; y < kSize; ++y) {
            for (uint32_t x = 0; x < kSize; ++x) {
                const uint32_t x1 = x + 1 < kSize ? x + 1 : kSize - 1; // CLAMP_TO_EDGE
                const uint32_t y1 = y + 1 < kSize ? y + 1 : kSize - 1;

                uint8_t* texel = &pixels[((size_t)y * kSize + x) * 4];
                texel[0] = SourceRed(x,  y1); // lower-left  (-,+)
                texel[1] = SourceRed(x1, y1); // lower-right (+,+)
                texel[2] = SourceRed(x1, y);  // upper-right (+,-)
                texel[3] = SourceRed(x,  y);  // upper-left  (-,-)
            }
        }
        return pixels;
    }

    agfxTextureCreateInfo TextureInfo(bool storage)
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = storage ? (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED)
                             : AGFX_TEXTURE_USAGE_SAMPLED;
        info.width = kSize;
        info.height = kSize;
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

    agfxSamplerCreateInfo SamplerInfo()
    {
        agfxSamplerCreateInfo info{};
        // Gather ignores the min/mag filter -- it always returns the four unfiltered texels -- but
        // the address mode very much applies, and CLAMP_TO_EDGE is what the edge expectation above
        // assumes.
        info.filter = AGFX_SAMPLER_FILTER_LINEAR;
        info.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.maxAnisotropy = 1.0f;
        info.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS; // "not a comparison sampler"
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "texture gather";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(TextureGather, C, kSize, kSize)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_gather_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile sampling.hlsl:main_gather_cs");

    const agfxTextureCreateInfo sourceInfo = TextureInfo(false);
    const agfxTextureCreateInfo destInfo = TextureInfo(true);
    agfxTexture* source = agfxTextureCreate(device, &sourceInfo);
    agfxTexture* dest = agfxTextureCreate(device, &destInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(dest);

    const agfxTextureViewCreateInfo srvInfo = ViewInfo(source, false);
    const agfxTextureViewCreateInfo uavInfo = ViewInfo(dest, true);
    agfxTextureView* srv = agfxTextureViewCreate(device, &srvInfo);
    agfxTextureView* uav = agfxTextureViewCreate(device, &uavInfo);

    const agfxSamplerCreateInfo samplerInfo = SamplerInfo();
    agfxSampler* sampler = agfxSamplerCreate(device, &samplerInfo);

    agfxShaderModule* module =
        CreateShaderModule(device, shader, "main_gather_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint8_t> sourcePixels = SourcePixels();
    AGFX_EXPECT_MSG(UploadTexture2D(device, gpu.Queue(), source, kSize, kSize, kFormat,
                                    sourcePixels.data(), AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the source");

    GatherPushConstants constants{};
    constants.source = (uint32_t)agfxTextureViewGetHandle(srv);
    constants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
    constants.destination = (uint32_t)agfxTextureViewGetHandle(uav);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "texture gather");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroups, kGroups, 1);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), dest, kSize, kSize, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, pipeline);
    agfxSamplerDestroy(device, sampler);
    agfxTextureViewDestroy(device, uav);
    agfxTextureViewDestroy(device, srv);
    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(image, ExpectedPixels()),
                    "the gathered texels are not the four corner neighbours in HLSL's order");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(TextureGather, Cpp, kSize, kSize)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_gather_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile sampling.hlsl:main_gather_cs");

    agfx::Texture source = device.CreateTexture(TextureInfo(false));
    agfx::Texture dest = device.CreateTexture(TextureInfo(true));
    agfx::TextureView srv = device.CreateTextureView(ViewInfo(source, false));
    agfx::TextureView uav = device.CreateTextureView(ViewInfo(dest, true));
    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_gather_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    const std::vector<uint8_t> sourcePixels = SourcePixels();
    AGFX_EXPECT_MSG(UploadTexture2D(device.Get(), queue, source, kSize, kSize, kFormat,
                                    sourcePixels.data(), AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the source");

    GatherPushConstants constants{};
    constants.source = (uint32_t)agfxTextureViewGetHandle(srv);
    constants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
    constants.destination = (uint32_t)agfxTextureViewGetHandle(uav);

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("texture gather");
        pass.SetPipeline(pipeline);
        pass.PushConstants(&constants, sizeof(constants));
        pass.Dispatch(kGroups, kGroups, 1);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    AGFX_EXPECT_MSG(ReadbackTexture2D(device.Get(), queue, dest, kSize, kSize, kFormat,
                                      AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image),
                    "texture readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(image, ExpectedPixels()),
                    "the gathered texels are not the four corner neighbours in HLSL's order");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(TextureGather, Ez, kSize, kSize)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kSize;
    contextInfo.height = kSize;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const CompiledShader shader = CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_gather_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile sampling.hlsl:main_gather_cs");

    const std::vector<uint8_t> sourcePixels = SourcePixels();
    agfx::ez::Texture2D source = context.CreateTexture2D(kSize, kSize, kFormat, AGFX_TEXTURE_USAGE_SAMPLED,
                                                          sourcePixels.data(), kSize * 4);
    agfx::ez::Texture2D dest = context.CreateTexture2D(
        kSize, kSize, kFormat, (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED));

    // ez has no sampler sugar; samplers come straight off the device.
    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_gather_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    // Mirrors GatherPushConstants field for field.
    agfx::ez::ShaderBindings bindings;
    bindings.BindTexture(source.SRV());
    bindings.BindSampler(sampler);
    bindings.BindTexture(dest.UAV());
    bindings.Write(kSize);
    bindings.Write(kSize);
    bindings.Write(1u); // sliceCount
    bindings.Write(1.0f);
    bindings.Write(1.0f);
    bindings.Write(0.0f);
    bindings.Write(0.0f);
    bindings.Write(0.0f);
    bindings.Write(0.0f);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        {
            agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("texture gather");
            pass.SetPipeline(pipeline);
            pass.PushConstants(bindings.Data(), bindings.Size());
            pass.Dispatch(kGroups, kGroups, 1);
        }
    }
    context.DrainGPU();

    Image image;
    AGFX_EXPECT_MSG(ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kSize, kSize,
                                      kFormat, dest.State(), image),
                    "texture readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(image, ExpectedPixels()),
                    "the gathered texels are not the four corner neighbours in HLSL's order");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
