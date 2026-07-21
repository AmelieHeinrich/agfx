/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute texture write 2D array".
//
// The layered counterpart of test_compute_texture_write.cpp: dispatches
// data/shaders/tests/texture_volume.hlsl:main_write_array_cs over a 64x64x4 RGBA8 array texture, one
// dispatch covering every layer at once, then reads the layers back and stacks them into a single
// 64x256 golden. Each layer is tinted by its index, so a UAV view that collapses onto layer 0, a
// dispatch whose Z dimension is dropped, or a backend that writes the layers in the wrong order all
// come out as a visibly wrong stack rather than a plausible image.
//
// There is no Ez variant: ez::Context only exposes CreateTexture2D, so array textures aren't
// expressible through that layer.

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
    constexpr uint32_t kGroupsZ = kLayers; // The shader's Z group size is 1, so one group per layer.
    constexpr const char* kGolden = "compute_texture_write_2d_array.png";

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
        info.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = kLayers;
        info.mipLevels = 1;
        return info;
    }

    /// @brief One UAV covering every layer, so the single dispatch can address all of them through
    /// the shader's z coordinate.
    agfxTextureViewCreateInfo UavInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
        info.baseMipLevel = 0;
        info.mipLevelCount = 1;
        info.baseArrayLayer = 0;
        info.arrayLayerCount = kLayers;
        info.writeable = 1;
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "compute texture write 2d array";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(ComputeTextureWrite2DArray, C, kWidth, kHeight* kLayers)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_array_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_volume.hlsl:main_write_array_cs");

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    const agfxTextureViewCreateInfo uavInfo = UavInfo(target);
    agfxTextureView* uav = agfxTextureViewCreate(device, &uavInfo);
    AGFX_EXPECT_NOT_NULL(uav);

    agfxShaderModule* module =
        CreateShaderModule(device, shader, "main_write_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    PushConstants constants{};
    constants.destination = (uint32_t)agfxTextureViewGetHandle(uav);
    constants.width = kWidth;
    constants.height = kHeight;
    constants.sliceCount = kLayers;

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "compute texture write 2d array");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, kGroupsZ);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2DArrayStack(device, gpu.Queue(), target, kWidth, kHeight, kLayers,
                                                    kFormat, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, pipeline);
    agfxTextureViewDestroy(device, uav);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(readOk, "array texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(ComputeTextureWrite2DArray, Cpp, kWidth, kHeight* kLayers)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_array_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile texture_volume.hlsl:main_write_array_cs");

    agfx::Texture target = device.CreateTexture(TargetInfo());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfx::TextureView uav = device.CreateTextureView(UavInfo(target));
    AGFX_EXPECT_NOT_NULL(uav.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_write_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    PushConstants constants{};
    constants.destination = (uint32_t)uav.GetHandle();
    constants.width = kWidth;
    constants.height = kHeight;
    constants.sliceCount = kLayers;

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("compute texture write 2d array");
        pass.SetPipeline(pipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2DArrayStack(device.Get(), queue, target, kWidth, kHeight, kLayers,
                                                    kFormat, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);
    AGFX_EXPECT_MSG(readOk, "array texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
