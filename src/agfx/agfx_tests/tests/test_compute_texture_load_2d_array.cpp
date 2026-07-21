/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute texture load 2D array".
//
// Two dispatches over data/shaders/tests/texture_volume.hlsl: main_write_array_cs seeds a 64x64x4
// source array texture, then main_load_array_cs reads it through a read-only AGFXTexture2DArray
// view and writes a mirrored, slice-reversed, channel-swizzled copy into a second array texture.
// The result is stacked into one 64x256 golden.
//
// Seeding with the write shader rather than a host upload is deliberate: UploadTextureSubresource
// would work per layer here, but the 3D twin has no equivalent host path (a 3D texture's slices are
// addressed by copy-region z, not by layer), and having both tests take the same shape keeps the
// pair comparable. The write path is pinned independently by test_compute_texture_write_2d_array.cpp,
// so a failure here is attributable to the load path.
//
// The slice reversal is what this test adds over the 2D case: a Load() that ignores the slice
// component of its coordinate reproduces the source stack unreversed, which the golden catches.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

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
    constexpr const char* kGolden = "compute_texture_load_2d_array.png";

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

    agfxTextureCreateInfo TextureInfo()
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

    agfxTextureViewCreateInfo ViewInfo(agfxTexture* texture, bool writeable)
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

AGFX_TEST_TEXTURE(ComputeTextureLoad2DArray, C, kWidth, kHeight* kLayers)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader writeShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_array_cs");
    const CompiledShader loadShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_load_array_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile texture_volume.hlsl:main_write_array_cs");
    AGFX_EXPECT_MSG(loadShader.Valid(), "failed to compile texture_volume.hlsl:main_load_array_cs");

    const agfxTextureCreateInfo textureInfo = TextureInfo();
    agfxTexture* source = agfxTextureCreate(device, &textureInfo);
    agfxTexture* target = agfxTextureCreate(device, &textureInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(target);

    const agfxTextureViewCreateInfo sourceUavInfo = ViewInfo(source, /*writeable*/ true);
    const agfxTextureViewCreateInfo sourceSrvInfo = ViewInfo(source, /*writeable*/ false);
    const agfxTextureViewCreateInfo targetUavInfo = ViewInfo(target, /*writeable*/ true);
    agfxTextureView* sourceUav = agfxTextureViewCreate(device, &sourceUavInfo);
    agfxTextureView* sourceSrv = agfxTextureViewCreate(device, &sourceSrvInfo);
    agfxTextureView* targetUav = agfxTextureViewCreate(device, &targetUavInfo);
    AGFX_EXPECT_NOT_NULL(sourceUav);
    AGFX_EXPECT_NOT_NULL(sourceSrv);
    AGFX_EXPECT_NOT_NULL(targetUav);

    agfxShaderModule* writeModule =
        CreateShaderModule(device, writeShader, "main_write_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    agfxShaderModule* loadModule =
        CreateShaderModule(device, loadShader, "main_load_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo writePipelineInfo = ComputePipelineInfo("array seed", writeModule);
    const agfxComputePipelineCreateInfo loadPipelineInfo = ComputePipelineInfo("array load", loadModule);
    agfxComputePipeline* writePipeline = agfxComputePipelineCreate(device, &writePipelineInfo);
    agfxComputePipeline* loadPipeline = agfxComputePipelineCreate(device, &loadPipelineInfo);
    agfxShaderModuleDestroy(device, writeModule);
    agfxShaderModuleDestroy(device, loadModule);
    AGFX_EXPECT_NOT_NULL(writePipeline);
    AGFX_EXPECT_NOT_NULL(loadPipeline);

    PushConstants seedConstants{};
    seedConstants.destination = (uint32_t)agfxTextureViewGetHandle(sourceUav);
    seedConstants.width = kWidth;
    seedConstants.height = kHeight;
    seedConstants.sliceCount = kLayers;

    PushConstants loadConstants{};
    loadConstants.source = (uint32_t)agfxTextureViewGetHandle(sourceSrv);
    loadConstants.destination = (uint32_t)agfxTextureViewGetHandle(targetUav);
    loadConstants.width = kWidth;
    loadConstants.height = kHeight;
    loadConstants.sliceCount = kLayers;

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "compute texture load 2d array");
        agfxComputePassSetPipeline(pass, writePipeline);
        agfxComputePassPushConstants(pass, &seedConstants, sizeof(seedConstants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, kGroupsZ);
        agfxComputePassEnd(pass);

        // The seed writes through a UAV and the load reads through an SRV, so the source needs a
        // real state transition between the two passes, not just a UAV barrier.
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

        agfxComputePass* loadPass = agfxComputePassBegin(cmd, "compute texture load 2d array");
        agfxComputePassSetPipeline(loadPass, loadPipeline);
        agfxComputePassPushConstants(loadPass, &loadConstants, sizeof(loadConstants));
        agfxComputePassDispatch(loadPass, kGroupsX, kGroupsY, kGroupsZ);
        agfxComputePassEnd(loadPass);
    });

    Image image;
    const bool readOk = ReadbackTexture2DArrayStack(device, gpu.Queue(), target, kWidth, kHeight, kLayers,
                                                    kFormat, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, loadPipeline);
    agfxComputePipelineDestroy(device, writePipeline);
    agfxTextureViewDestroy(device, targetUav);
    agfxTextureViewDestroy(device, sourceSrv);
    agfxTextureViewDestroy(device, sourceUav);
    agfxTextureDestroy(device, target);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(readOk, "array texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(ComputeTextureLoad2DArray, Cpp, kWidth, kHeight* kLayers)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader writeShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_array_cs");
    const CompiledShader loadShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_load_array_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile texture_volume.hlsl:main_write_array_cs");
    AGFX_EXPECT_MSG(loadShader.Valid(), "failed to compile texture_volume.hlsl:main_load_array_cs");

    agfx::Texture source = device.CreateTexture(TextureInfo());
    agfx::Texture target = device.CreateTexture(TextureInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfx::TextureView sourceUav = device.CreateTextureView(ViewInfo(source, true));
    agfx::TextureView sourceSrv = device.CreateTextureView(ViewInfo(source, false));
    agfx::TextureView targetUav = device.CreateTextureView(ViewInfo(target, true));
    AGFX_EXPECT_NOT_NULL(sourceUav.Get());
    AGFX_EXPECT_NOT_NULL(sourceSrv.Get());
    AGFX_EXPECT_NOT_NULL(targetUav.Get());

    agfx::ComputePipeline writePipeline;
    agfx::ComputePipeline loadPipeline;
    {
        agfx::ShaderModule writeModule(device.Get(),
            CreateShaderModule(device.Get(), writeShader, "main_write_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule loadModule(device.Get(),
            CreateShaderModule(device.Get(), loadShader, "main_load_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        writePipeline = device.CreateComputePipeline(ComputePipelineInfo("array seed", writeModule));
        loadPipeline = device.CreateComputePipeline(ComputePipelineInfo("array load", loadModule));
    }
    AGFX_EXPECT_NOT_NULL(writePipeline.Get());
    AGFX_EXPECT_NOT_NULL(loadPipeline.Get());

    PushConstants seedConstants{};
    seedConstants.destination = (uint32_t)sourceUav.GetHandle();
    seedConstants.width = kWidth;
    seedConstants.height = kHeight;
    seedConstants.sliceCount = kLayers;

    PushConstants loadConstants{};
    loadConstants.source = (uint32_t)sourceSrv.GetHandle();
    loadConstants.destination = (uint32_t)targetUav.GetHandle();
    loadConstants.width = kWidth;
    loadConstants.height = kHeight;
    loadConstants.sliceCount = kLayers;

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("compute texture load 2d array");
        pass.SetPipeline(writePipeline);
        pass.PushConstants(seedConstants);
        pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
    }
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("compute texture load 2d array");
        pass.SetPipeline(loadPipeline);
        pass.PushConstants(loadConstants);
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

AGFX_TEST_TEXTURE(ComputeTextureLoad2DArray, Ez, kWidth, kHeight* kLayers)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader writeShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_write_array_cs");
    const CompiledShader loadShader =
        CompileTestShader("texture_volume.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_load_array_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile texture_volume.hlsl:main_write_array_cs");
    AGFX_EXPECT_MSG(loadShader.Valid(), "failed to compile texture_volume.hlsl:main_load_array_cs");

    agfx::Device& device = context.GetDevice();

    const agfxTextureUsage usage =
        (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
    agfx::ez::Texture2DArray source = context.CreateTexture2DArray(kWidth, kHeight, kLayers, kFormat, usage);
    agfx::ez::Texture2DArray target = context.CreateTexture2DArray(kWidth, kHeight, kLayers, kFormat, usage);

    agfx::ComputePipeline writePipeline;
    agfx::ComputePipeline loadPipeline;
    {
        agfx::ShaderModule writeModule(device.Get(),
            CreateShaderModule(device.Get(), writeShader, "main_write_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule loadModule(device.Get(),
            CreateShaderModule(device.Get(), loadShader, "main_load_array_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        writePipeline = device.CreateComputePipeline(ComputePipelineInfo("array seed", writeModule));
        loadPipeline = device.CreateComputePipeline(ComputePipelineInfo("array load", loadModule));
    }
    AGFX_EXPECT_NOT_NULL(writePipeline.Get());
    AGFX_EXPECT_NOT_NULL(loadPipeline.Get());

    // Both ez views span every layer, matching ViewInfo()'s arrayLayerCount = kLayers.
    agfx::ez::ShaderBindings seedBindings;
    seedBindings.Write(0u); // unused source slot
    seedBindings.BindTexture(source.UAV());
    seedBindings.Write(kWidth);
    seedBindings.Write(kHeight);
    seedBindings.Write(kLayers);

    agfx::ez::ShaderBindings loadBindings;
    loadBindings.BindTexture(source.SRV());
    loadBindings.BindTexture(target.UAV());
    loadBindings.Write(kWidth);
    loadBindings.Write(kHeight);
    loadBindings.Write(kLayers);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);
        context.TransitionTexture(target, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        {
            agfx::ComputePass pass =
                context.GetCurrentCommandBuffer().BeginComputePass("compute texture load 2d array");
            pass.SetPipeline(writePipeline);
            pass.PushConstants(seedBindings.Data(), seedBindings.Size());
            pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
        }

        // UAV write then SRV read: a real state transition, not just a UAV barrier.
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        {
            agfx::ComputePass pass =
                context.GetCurrentCommandBuffer().BeginComputePass("compute texture load 2d array");
            pass.SetPipeline(loadPipeline);
            pass.PushConstants(loadBindings.Data(), loadBindings.Size());
            pass.Dispatch(kGroupsX, kGroupsY, kGroupsZ);
        }
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2DArrayStack(device.Get(), context.GetGraphicsQueue(), target.Raw(), kWidth,
                                                    kHeight, kLayers, kFormat, target.State(), image);
    AGFX_EXPECT_MSG(readOk, "array texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
