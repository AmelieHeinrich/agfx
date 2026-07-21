/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "viewport".
//
// Draws the same triangle as test_draw_triangle.cpp, but through a viewport occupying only the
// bottom-right quadrant of a 128x128 target, with the scissor left wide open. The triangle should
// come out squeezed into that quadrant with everything else still at the clear color: a viewport
// that is ignored fills the whole target, and one whose origin is measured from the other corner
// lands the triangle in the wrong quadrant. Keeping the scissor full-target is what separates this
// test from test_scissor_rect.cpp — the clipping there must come from the viewport alone.
//
// There is no Ez variant: ez only exposes the combined SetViewportScissor, which cannot set a
// viewport and scissor that disagree, so it cannot express what this test is checking.

#include "../test_gpu.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr const char* kGolden = "viewport.png";
    constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    // Bottom-right quadrant, in the API's top-left-origin viewport space.
    constexpr float kViewportX = (float)(kWidth / 2);
    constexpr float kViewportY = (float)(kHeight / 2);
    constexpr float kViewportWidth = (float)(kWidth / 2);
    constexpr float kViewportHeight = (float)(kHeight / 2);

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
        info.name = "test viewport";
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
        info.name = "test viewport";
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(Viewport, C, kWidth, kHeight)
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
        agfxRenderPassSetViewport(pass, kViewportX, kViewportY, kViewportWidth, kViewportHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kWidth, kHeight); // wide open: the viewport does the work
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

AGFX_TEST_TEXTURE(Viewport, Cpp, kWidth, kHeight)
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
        pass.SetViewport(kViewportX, kViewportY, kViewportWidth, kViewportHeight);
        pass.SetScissor(0, 0, kWidth, kHeight);
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
