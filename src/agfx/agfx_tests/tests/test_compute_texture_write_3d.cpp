/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute texture write 3D".
//
// The volumetric counterpart of test_compute_texture_write.cpp: dispatches
// data/shaders/tests/texture_volume.hlsl:main_write_3d_cs over a 64x64x4 RGBA8 3D texture in one
// dispatch, then reads the depth slices back and stacks them into a single 64x256 golden. Each
// slice is tinted by its z, so a UAV that collapses onto slice 0 or a dispatch whose Z dimension is
// dropped shows up as a uniform stack.
//
// Worth keeping separate from the 2D array test even though the shaders are near-identical: a 3D
// texture's slices are a different addressing mechanism from array layers on both backends (region
// z origin vs. subresource layer), and only one of the two paths being right is a real failure mode.

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
    constexpr uint32_t kGroupsZ = kDepth; // The shader's Z group size is 1, so one group per slice.
    constexpr const char* kGolden = "compute_texture_write_3d.png";

    /// @brief Mirrors VolumePushConstants in texture_volume.hlsl.
    struct PushConstants
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

    agfxTextureCreateInfo TargetInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_3D;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = kDepth; // Depth, not layers, for a 3D texture.
        info.mipLevels = 1;
        return info;
    }

    /// @brief A 3D texture has no array layers: the depth is part of the texture itself, so the
    /// view covers one "layer" and the shader addresses slices through its z coordinate.
    agfxTextureViewCreateInfo UavInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_3D;
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
        info.name = "compute texture write 3d";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(ComputeTextureWrite3D, C, kWidth, kHeight* kDepth)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_3d_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_volume.hlsl:main_write_3d_cs");

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    const agfxTextureViewCreateInfo uavInfo = UavInfo(target);
    agfxTextureView* uav = agfxTextureViewCreate(device, &uavInfo);
    AGFX_EXPECT_NOT_NULL(uav);

    agfxShaderModule* module =
        CreateShaderModule(device, shader, "main_write_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    PushConstants constants{};
    constants.destination = (uint32_t)agfxTextureViewGetHandle(uav);
    constants.width = kWidth;
    constants.height = kHeight;
    constants.sliceCount = kDepth;

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "compute texture write 3d");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, kGroupsZ);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture3DStack(device, gpu.Queue(), target, kWidth, kHeight, kDepth,
                                               kFormat, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, pipeline);
    agfxTextureViewDestroy(device, uav);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(readOk, "3d texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(ComputeTextureWrite3D, Cpp, kWidth, kHeight* kDepth)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_3d_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_volume.hlsl:main_write_3d_cs");

    agfx::Texture target = device.CreateTexture(TargetInfo());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfx::TextureView uav = device.CreateTextureView(UavInfo(target));
    AGFX_EXPECT_NOT_NULL(uav.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_write_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    PushConstants constants{};
    constants.destination = (uint32_t)uav.GetHandle();
    constants.width = kWidth;
    constants.height = kHeight;
    constants.sliceCount = kDepth;

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("compute texture write 3d");
        pass.SetPipeline(pipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture3DStack(device.Get(), queue, target, kWidth, kHeight, kDepth,
                                               kFormat, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);
    AGFX_EXPECT_MSG(readOk, "3d texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(ComputeTextureWrite3D, Ez, kWidth, kHeight* kDepth)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_3d_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_volume.hlsl:main_write_3d_cs");

    agfx::Device& device = context.GetDevice();

    agfx::ez::Texture3D target = context.CreateTexture3D(
        kWidth, kHeight, kDepth, kFormat, (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED));

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_write_3d_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    // target.UAV() defaults to the whole resource, which for a 3D texture is the one-layer,
    // one-mip view the C and C++ flavors spell out by hand in UavInfo().
    agfx::ez::ShaderBindings bindings;
    bindings.Write(0u); // unused source slot
    bindings.BindTexture(target.UAV());
    bindings.Write(kWidth);
    bindings.Write(kHeight);
    bindings.Write(kDepth);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(target, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("compute texture write 3d");
        pass.SetPipeline(pipeline);
        pass.PushConstants(bindings.Data(), bindings.Size());
        pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture3DStack(device.Get(), context.GetGraphicsQueue(), target.Raw(), kWidth,
                                               kHeight, kDepth, kFormat, target.State(), image);
    AGFX_EXPECT_MSG(readOk, "3d texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
