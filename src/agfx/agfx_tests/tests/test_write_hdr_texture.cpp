/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "write to HDR texture".
//
// Dispatches data/shaders/tests/texture_ops.hlsl:main_hdr_cs into an RGBA32F storage texture and
// compares against a radiance-HDR golden through FLIP's HDR mode. The shader writes every channel
// into [1, 9], so a target that is secretly 8-bit — or a readback path that normalizes on the way
// out — clamps the whole image to white and fails. This is the only test in the suite whose golden
// is not representable in an LDR image.
//
// The range is deliberately modest: the golden is stored as radiance HDR, which shares a single
// 8-bit exponent across RGB and cannot represent negatives, so a texel mixing 0.0 with ~100.0 (or
// holding a negative channel) would not survive the trip to disk and the test would fail against
// its own output. Values above 1.0 are what this test needs; extreme values are not.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA32F;
    constexpr uint32_t kGroupSize = 8;                   // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroupsX = (kWidth + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsY = (kHeight + kGroupSize - 1) / kGroupSize;
    // The .hdr extension is what routes the golden through the radiance writer and FLIP's HDR mode.
    constexpr const char* kGolden = "write_hdr_texture.hdr";

    /// @brief Mirrors TexturePushConstants in texture_ops.hlsl.
    struct PushConstants
    {
        uint32_t source;
        uint32_t destination;
        uint32_t width;
        uint32_t height;
    };

    agfxTextureCreateInfo TargetInfo()
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

    agfxTextureViewCreateInfo UavInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.baseMipLevel = 0;
        info.mipLevelCount = 1;
        info.baseArrayLayer = 0;
        info.arrayLayerCount = 1;
        info.writeable = 1;
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "write hdr texture";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(WriteHDRTexture, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_hdr_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_ops.hlsl:main_hdr_cs");

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    const agfxTextureViewCreateInfo uavInfo = UavInfo(target);
    agfxTextureView* uav = agfxTextureViewCreate(device, &uavInfo);
    AGFX_EXPECT_NOT_NULL(uav);

    agfxShaderModule* module = CreateShaderModule(device, shader, "main_hdr_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    PushConstants constants{};
    constants.destination = (uint32_t)agfxTextureViewGetHandle(uav);
    constants.width = kWidth;
    constants.height = kHeight;

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "write hdr texture");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, pipeline);
    agfxTextureViewDestroy(device, uav);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(WriteHDRTexture, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_hdr_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_ops.hlsl:main_hdr_cs");

    agfx::Texture target = device.CreateTexture(TargetInfo());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfx::TextureView uav = device.CreateTextureView(UavInfo(target));
    AGFX_EXPECT_NOT_NULL(uav.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_hdr_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    PushConstants constants{};
    constants.destination = (uint32_t)uav.GetHandle();
    constants.width = kWidth;
    constants.height = kHeight;

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("write hdr texture");
        pass.SetPipeline(pipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupsX, kGroupsY, 1);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(WriteHDRTexture, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader = CompileTestShader("texture_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_hdr_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_ops.hlsl:main_hdr_cs");

    agfx::Device& device = context.GetDevice();

    agfx::ez::Texture2D target = context.CreateTexture2D(kWidth, kHeight, kFormat, AGFX_TEXTURE_USAGE_STORAGE);

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_hdr_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    agfx::ez::ShaderBindings bindings;
    bindings.Write(0u); // unused source slot
    bindings.BindTexture(target.UAV());
    bindings.Write(kWidth);
    bindings.Write(kHeight);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(target, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("write hdr texture");
        pass.SetPipeline(pipeline);
        pass.PushConstants(bindings.Data(), bindings.Size());
        pass.Dispatch(kGroupsX, kGroupsY, 1);
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                          kWidth, kHeight, kFormat, target.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
