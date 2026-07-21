/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute texture load 2D".
//
// Dispatches data/shaders/tests/texture_ops.hlsl:main_load_cs, which reads a seeded source texture
// with Load() — no sampler, so texel addressing is under test without filtering to hide behind —
// and writes a mirrored, channel-swizzled copy into a storage texture. The mirror catches a Load()
// that ignores its coordinate or a view bound to the wrong texture; the swizzle catches a channel
// order mixup between the sampled and storage paths.

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
    constexpr const char* kGolden = "compute_texture_load.png";

    /// @brief Mirrors TexturePushConstants in texture_ops.hlsl.
    struct PushConstants
    {
        uint32_t source;
        uint32_t destination;
        uint32_t width;
        uint32_t height;
    };

    /// @brief Left-right asymmetric on purpose: the shader mirrors in X, so a no-op mirror would be
    /// invisible against a symmetric source.
    std::vector<uint8_t> SourcePixels()
    {
        std::vector<uint8_t> pixels((size_t)kImageBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 255u / (kWidth - 1u)); // ramp: reverses under the mirror
                texel[1] = (uint8_t)(y * 255u / (kHeight - 1u));
                texel[2] = x < kWidth / 4u ? 255u : 0u;          // a marker block near the left edge
                texel[3] = 255u;
            }
        }
        return pixels;
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
        info.name = "compute texture load";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(ComputeTextureLoad2D, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_load_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_ops.hlsl:main_load_cs");

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

    agfxShaderModule* module = CreateShaderModule(device, shader, "main_load_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint8_t> pixels = SourcePixels();
    const bool seeded = UploadTexture2D(device, gpu.Queue(), source, kWidth, kHeight, kFormat,
                                        pixels.data(), AGFX_RESOURCE_STATE_COMMON);

    PushConstants constants{};
    constants.source = (uint32_t)agfxTextureViewGetHandle(srv);
    constants.destination = (uint32_t)agfxTextureViewGetHandle(uav);
    constants.width = kWidth;
    constants.height = kHeight;

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 0, 0, 0);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "compute texture load");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), dest, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, pipeline);
    agfxTextureViewDestroy(device, uav);
    agfxTextureViewDestroy(device, srv);
    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(seeded, "failed to seed the source texture");
    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(ComputeTextureLoad2D, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_load_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_ops.hlsl:main_load_cs");

    agfx::Texture source = device.CreateTexture(SourceInfo());
    agfx::Texture dest = device.CreateTexture(DestInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(dest.Get());

    agfx::TextureView srv = device.CreateTextureView(ViewInfo(source, false));
    agfx::TextureView uav = device.CreateTextureView(ViewInfo(dest, true));
    AGFX_EXPECT_NOT_NULL(srv.Get());
    AGFX_EXPECT_NOT_NULL(uav.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_load_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    const std::vector<uint8_t> pixels = SourcePixels();
    AGFX_EXPECT_MSG(UploadTexture2D(device.Get(), queue, source, kWidth, kHeight, kFormat, pixels.data(),
                                    AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the source texture");

    PushConstants constants{};
    constants.source = (uint32_t)srv.GetHandle();
    constants.destination = (uint32_t)uav.GetHandle();
    constants.width = kWidth;
    constants.height = kHeight;

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("compute texture load");
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

AGFX_TEST_TEXTURE(ComputeTextureLoad2D, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader = CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_load_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_ops.hlsl:main_load_cs");

    agfx::Device& device = context.GetDevice();

    const std::vector<uint8_t> pixels = SourcePixels();
    agfx::ez::Texture2D source =
        context.CreateTexture2D(kWidth, kHeight, kFormat, AGFX_TEXTURE_USAGE_SAMPLED, pixels.data(), kRowBytes);
    agfx::ez::Texture2D dest = context.CreateTexture2D(kWidth, kHeight, kFormat, AGFX_TEXTURE_USAGE_STORAGE);

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_load_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    // The SRV/UAV split ez hands out is exactly the pair the C and C++ flavors build by hand.
    agfx::ez::ShaderBindings bindings;
    bindings.BindTexture(source.SRV());
    bindings.BindTexture(dest.UAV());
    bindings.Write(kWidth);
    bindings.Write(kHeight);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("compute texture load");
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
