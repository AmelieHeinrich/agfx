/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "scissor rect".
//
// The mirror of test_viewport.cpp: the same triangle drawn through a full-target viewport, but
// with the scissor clipped to a 64x64 rect offset into the middle of the 128x128 target. The
// triangle must come out at full size with everything outside the rect still at the clear color —
// a scissor that is ignored leaves the whole triangle, and one that is confused with the viewport
// squeezes the triangle down instead of cropping it. The rect is deliberately not axis-symmetric
// so a swapped x/y or a width/height mixup shifts the crop visibly.
//
// There is no Ez variant: ez only exposes the combined SetViewportScissor, which cannot set a
// scissor that disagrees with the viewport.

#include "../test_gpu.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr const char* kGolden = "scissor_rect.png";
    constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    // Off-center on purpose: x != y and the rect doesn't straddle the target's midpoint evenly.
    constexpr uint32_t kScissorX = 24;
    constexpr uint32_t kScissorY = 40;
    constexpr uint32_t kScissorWidth = 64;
    constexpr uint32_t kScissorHeight = 64;

    agfxTextureCreateInfo TargetInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "test scissor rect";
        info.fillMode = AGFX_FILL_MODE_SOLID;
        info.cullMode = AGFX_CULL_MODE_NONE;
        info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        info.topology = AGFX_TOPOLOGY_TRIANGLES;
        info.depthTestEnable = 0;
        info.depthWriteEnable = 0;
        info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
        info.colorAttachmentCount = 1;
        info.colorFormats[0] = kFormat;
        info.vertexShader = vs;
        info.fragmentShader = ps;
        return info;
    }

    agfxRenderPassCreateInfo PassInfo(agfxRenderTarget* renderTarget)
    {
        agfxRenderPassCreateInfo info{};
        info.colorAttachmentCount = 1;
        info.colorAttachments[0].renderTarget = renderTarget;
        info.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
        info.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
        memcpy(info.colorAttachments[0].clearColor, kClearColor, sizeof(kClearColor));
        info.width = kWidth;
        info.height = kHeight;
        info.name = "test scissor rect";
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(ScissorRect, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader vsShader = CompileTestShader("triangle.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("triangle.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile triangle.hlsl:main_vs");
    AGFX_EXPECT_MSG(psShader.Valid(), "failed to compile triangle.hlsl:main_ps");

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target;
    rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
    agfxRenderTarget* renderTarget = agfxRenderTargetCreate(device, &rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget);

    agfxShaderModule* vsModule = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfxShaderModule* psModule = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
    const agfxRenderPipelineCreateInfo pipelineInfo = PipelineInfo(vsModule, psModule);
    agfxRenderPipeline* pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, vsModule);
    agfxShaderModuleDestroy(device, psModule);
    AGFX_EXPECT_NOT_NULL(pipeline);

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 1);

        const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kWidth, (float)kHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, kScissorX, kScissorY, kScissorWidth, kScissorHeight);
        agfxRenderPassSetPipeline(pass, pipeline);
        agfxRenderPassDraw(pass, 3, 1, 0, 0);
        agfxRenderPassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_RENDER_TARGET, image);

    agfxRenderPipelineDestroy(device, pipeline);
    agfxRenderTargetDestroy(device, renderTarget);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(ScissorRect, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader vsShader = CompileTestShader("triangle.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("triangle.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile triangle.hlsl:main_vs");
    AGFX_EXPECT_MSG(psShader.Valid(), "failed to compile triangle.hlsl:main_ps");

    agfx::Texture target = device.CreateTexture(TargetInfo());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target;
    rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
    agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget.Get());

    agfx::RenderPipeline pipeline;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        agfx::ShaderModule psModule(device.Get(),
            CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
        pipeline = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget));
        pass.SetViewport(0.0f, 0.0f, (float)kWidth, (float)kHeight);
        pass.SetScissor(kScissorX, kScissorY, kScissorWidth, kScissorHeight);
        pass.SetPipeline(pipeline);
        pass.Draw(3);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_RENDER_TARGET, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
