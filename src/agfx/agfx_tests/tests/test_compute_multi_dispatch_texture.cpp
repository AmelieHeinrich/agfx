/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute multi dispatch texture".
//
// The texture half of "compute multi dispatch buffer": drives
// data/shaders/tests/multi_dispatch_texture.hlsl four times back to back within one compute pass,
// with a texture UAV barrier between each. Every pass read-modify-writes what the previous one
// wrote and folds its own pass index in, so the golden only reproduces if the barriers really
// serialize the dispatches — a backend that lets them overlap lands somewhere below it, and the
// per-pass index means a dropped or reordered dispatch is distinguishable from a lost one.
//
// Worth having separately from the buffer test: the buffer and texture UAV barriers are distinct
// entry points, and on both backends they lower to different code.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kGroupSize = 8;                   // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroupsX = (kWidth + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsY = (kHeight + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kPassCount = 4;
    constexpr const char* kGolden = "compute_multi_dispatch_texture.png";

    /// @brief Mirrors MultiDispatchTexturePushConstants in multi_dispatch_texture.hlsl.
    struct PushConstants
    {
        uint32_t rwTexture;
        uint32_t width;
        uint32_t height;
        uint32_t passIndex;
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
        info.name = "multi dispatch texture";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }

    /// @brief Pass 0 reads the target before writing it, so it has to start from a known value
    /// rather than from whatever the allocator handed back.
    std::vector<uint8_t> ZeroPixels()
    {
        return std::vector<uint8_t>((size_t)kWidth * kHeight * 4u, 0u);
    }
} // namespace

AGFX_TEST_TEXTURE(ComputeMultiDispatchTexture, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader =
        CompileTestShader("multi_dispatch_texture.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_texture_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile multi_dispatch_texture.hlsl:main_texture_cs");

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    const agfxTextureViewCreateInfo uavInfo = UavInfo(target);
    agfxTextureView* uav = agfxTextureViewCreate(device, &uavInfo);
    AGFX_EXPECT_NOT_NULL(uav);

    agfxShaderModule* module = CreateShaderModule(device, shader, "main_texture_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint8_t> zeros = ZeroPixels();
    const bool seeded = UploadTexture2D(device, gpu.Queue(), target, kWidth, kHeight, kFormat,
                                        zeros.data(), AGFX_RESOURCE_STATE_COMMON);

    PushConstants constants{};
    constants.rwTexture = (uint32_t)agfxTextureViewGetHandle(uav);
    constants.width = kWidth;
    constants.height = kHeight;

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "multi dispatch texture");
        agfxComputePassSetPipeline(pass, pipeline);
        for (uint32_t i = 0; i < kPassCount; ++i) {
            if (i > 0) {
                agfxComputePassTextureUAVBarrier(pass, target);
            }
            constants.passIndex = i;
            agfxComputePassPushConstants(pass, &constants, sizeof(constants));
            agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
        }
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, pipeline);
    agfxTextureViewDestroy(device, uav);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(seeded, "failed to zero the target texture");
    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(ComputeMultiDispatchTexture, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader =
        CompileTestShader("multi_dispatch_texture.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_texture_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile multi_dispatch_texture.hlsl:main_texture_cs");

    agfx::Texture target = device.CreateTexture(TargetInfo());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfx::TextureView uav = device.CreateTextureView(UavInfo(target));
    AGFX_EXPECT_NOT_NULL(uav.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_texture_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    const std::vector<uint8_t> zeros = ZeroPixels();
    AGFX_EXPECT_MSG(UploadTexture2D(device.Get(), queue, target, kWidth, kHeight, kFormat, zeros.data(),
                                    AGFX_RESOURCE_STATE_COMMON),
                    "failed to zero the target texture");

    PushConstants constants{};
    constants.rwTexture = (uint32_t)uav.GetHandle();
    constants.width = kWidth;
    constants.height = kHeight;

    cmd.Begin();
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("multi dispatch texture");
        pass.SetPipeline(pipeline);
        for (uint32_t i = 0; i < kPassCount; ++i) {
            if (i > 0) {
                pass.TextureUAVBarrier(target);
            }
            constants.passIndex = i;
            pass.PushConstants(constants);
            pass.Dispatch(kGroupsX, kGroupsY, 1);
        }
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

AGFX_TEST_TEXTURE(ComputeMultiDispatchTexture, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader =
        CompileTestShader("multi_dispatch_texture.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_texture_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile multi_dispatch_texture.hlsl:main_texture_cs");

    agfx::Device& device = context.GetDevice();

    // Seeded at creation: pass 0 reads before it writes, so the zeros have to be there already.
    const std::vector<uint8_t> zeros = ZeroPixels();
    agfx::ez::Texture2D target = context.CreateTexture2D(kWidth, kHeight, kFormat,
                                                        AGFX_TEXTURE_USAGE_STORAGE, zeros.data(), kWidth * 4u);

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_texture_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(target, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("multi dispatch texture");
        pass.SetPipeline(pipeline);
        for (uint32_t i = 0; i < kPassCount; ++i) {
            if (i > 0) {
                pass.TextureUAVBarrier(target.Raw());
            }

            // Rebuilt per pass: the pass index is part of the constants the shader folds in.
            agfx::ez::ShaderBindings bindings;
            bindings.BindTexture(target.UAV());
            bindings.Write(kWidth);
            bindings.Write(kHeight);
            bindings.Write(i);
            pass.PushConstants(bindings.Data(), bindings.Size());

            pass.Dispatch(kGroupsX, kGroupsY, 1);
        }
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                          kWidth, kHeight, kFormat, target.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
