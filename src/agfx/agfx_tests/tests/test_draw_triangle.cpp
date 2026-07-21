/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw triangle".
//
// The smallest end-to-end raster test there is: clear a 128x128 offscreen RGBA8 target to black and
// draw one RGB gradient triangle from data/shaders/tests/triangle.hlsl (no vertex buffer, no
// bindless resources), then FLIP the readback against the golden PNG. Catches broken render pass
// setup, wrong clear/load ops, flipped winding, inverted Y, and format mixups in one shot.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr const char* kGolden = "draw_triangle.png";
    constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

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

    /// @brief The pipeline state the triangle draws with: no depth (there's no depth attachment),
    /// no culling (so a winding-order difference between backends can't silently drop the
    /// primitive and turn this into a "cleared to black" pass).
    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "test triangle";
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
} // namespace

AGFX_TEST_TEXTURE(DrawTriangle, C, kWidth, kHeight)
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
    rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN; // inherit the texture's format
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
                                        AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);

        agfxRenderPassCreateInfo passInfo{};
        passInfo.colorAttachmentCount = 1;
        passInfo.colorAttachments[0].renderTarget = renderTarget;
        passInfo.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
        passInfo.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
        memcpy(passInfo.colorAttachments[0].clearColor, kClearColor, sizeof(kClearColor));
        passInfo.width = kWidth;
        passInfo.height = kHeight;
        passInfo.name = "test triangle";

        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kWidth, (float)kHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kWidth, kHeight);
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

AGFX_TEST_TEXTURE(DrawTriangle, Cpp, kWidth, kHeight)
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
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfxRenderPassCreateInfo passInfo{};
        passInfo.colorAttachmentCount = 1;
        passInfo.colorAttachments[0].renderTarget = renderTarget;
        passInfo.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
        passInfo.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
        memcpy(passInfo.colorAttachments[0].clearColor, kClearColor, sizeof(kClearColor));
        passInfo.width = kWidth;
        passInfo.height = kHeight;
        passInfo.name = "test triangle";

        agfx::RenderPass pass = cmd.BeginRenderPass(passInfo);
        pass.SetViewport(0.0f, 0.0f, (float)kWidth, (float)kHeight);
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

AGFX_TEST_TEXTURE(DrawTriangle, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: render straight into an offscreen target
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader vsShader = CompileTestShader("triangle.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("triangle.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile triangle.hlsl:main_vs");
    AGFX_EXPECT_MSG(psShader.Valid(), "failed to compile triangle.hlsl:main_ps");

    agfx::Device& device = context.GetDevice();

    agfx::ShaderModule vsModule(device.Get(),
        CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
    agfx::ShaderModule psModule(device.Get(),
        CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
    AGFX_EXPECT_NOT_NULL(vsModule.Get());
    AGFX_EXPECT_NOT_NULL(psModule.Get());

    agfx::ez::Texture2D target = context.CreateTexture2D(kWidth, kHeight, kFormat,
                                                         AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT);

    // ez's PipelineDesc caches on its contents, so the modules must stay alive for the draw.
    agfx::ez::PipelineDesc desc;
    desc.name = "test triangle";
    desc.vertexShader = &vsModule;
    desc.fragmentShader = &psModule;
    desc.cullMode = AGFX_CULL_MODE_NONE;
    desc.depthTestEnable = false;
    desc.depthWriteEnable = false;

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.SetRenderTargets({&target}, nullptr, AGFX_LOAD_OPERATION_CLEAR, kClearColor);
        context.SetViewportScissor(0, 0, kWidth, kHeight);
        context.SetPipeline(desc);
        context.Draw(3);
        context.EndActivePass();
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                          kWidth, kHeight, kFormat, target.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
