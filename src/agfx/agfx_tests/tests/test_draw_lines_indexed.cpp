/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw lines indexed".
//
// The intersection of the two paths that are already pinned separately: an index buffer (as in
// test_draw_indexed.cpp) feeding AGFX_TOPOLOGY_LINES (as in test_draw_lines.cpp). Both work on their
// own; what this adds is that index lookup and line assembly compose — indices are consumed in
// *pairs* here, not triples, so a backend that hardcodes 3 indices per primitive when reading the
// buffer assembles the wrong segments from the right vertices.
//
// The index list traces the quad's outline (0-1, 1-2, 2-3, 3-0) and then both diagonals (0-2, 1-3).
// The diagonals matter: an outline alone is what a line-*strip* interpretation would also produce,
// so without them a backend that ignored the LINES topology could pass. With them the golden holds
// an X inside the box, which neither a strip nor a triangle interpretation can reproduce.

#include "raster_common.h" // for CountLitPixels

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr const char* kGolden = "draw_lines_indexed.png";
    constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    /// @brief Mirrors Vertex in indexed.hlsl. The padding keeps the HLSL float4 color 16-byte
    /// aligned, so the structured view's stride is the same 32 bytes on both sides.
    struct Vertex
    {
        float position[2];
        float padding[2];
        float color[4];
    };
    static_assert(sizeof(Vertex) == 32, "Vertex must match the HLSL layout");

    /// @brief The quad's four corners, one primary-ish color each so every corner is identifiable
    /// in the golden on its own.
    constexpr Vertex kVertices[4] = {
        {{-0.8f, -0.8f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}, // 0: bottom left,  red
        {{ 0.8f, -0.8f}, {0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}}, // 1: bottom right, green
        {{ 0.8f,  0.8f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}, // 2: top right,    blue
        {{-0.8f,  0.8f}, {0.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}}, // 3: top left,     yellow
    };

    /// @brief Six segments as index *pairs*: the four outline edges, then the two diagonals. Not a
    /// sequential run, so an implementation that ignores the index buffer and consumes vertex IDs
    /// straight through draws three unrelated segments instead of a boxed X.
    constexpr uint32_t kIndices[12] = {0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3};
    constexpr uint32_t kIndexCount = 12;

    /// @brief Six one-pixel-wide segments over a 128x128 target. A floor rather than an exact count,
    /// for the reason test_draw_lines.cpp gives: the golden's mean-error comparison cannot separate
    /// ~2% coverage from an empty image, while an exact count would be hostage to each backend's
    /// legitimate freedom in line endpoint rules.
    constexpr uint32_t kMinLinePixels = 400;

    /// @brief Mirrors IndexedPushConstants in indexed.hlsl.
    struct PushConstants
    {
        uint32_t vertices;
        uint32_t padding0;
        uint32_t padding1;
        uint32_t padding2;
    };

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

    agfxBufferCreateInfo VertexBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = sizeof(kVertices);
        info.stride = sizeof(Vertex);
        info.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    /// @brief The index buffer's stride is what selects the index type: 4 bytes here, 2 in the u16
    /// twin. There is no separate index-type parameter on agfxRenderPassDrawIndexed.
    agfxBufferCreateInfo IndexBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = sizeof(kIndices);
        info.stride = sizeof(uint32_t);
        info.usage = AGFX_BUFFER_USAGE_INDEX;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    /// @brief Culling is off, as it is for every raster test here. Lines have no winding to cull on,
    /// so this is belt-and-braces rather than load-bearing — but it keeps the pipeline identical to
    /// test_draw_indexed.cpp's except for the topology, which is the only difference under test.
    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "test draw lines indexed";
        info.fillMode = AGFX_FILL_MODE_SOLID;
        info.cullMode = AGFX_CULL_MODE_NONE;
        info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        info.topology = AGFX_TOPOLOGY_LINES;
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
        info.name = "test draw lines indexed";
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(DrawLinesIndexed, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader vsShader = CompileTestShader("indexed.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("indexed.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile indexed.hlsl:main_vs");
    AGFX_EXPECT_MSG(psShader.Valid(), "failed to compile indexed.hlsl:main_ps");

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target;
    rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN; // inherit the texture's format
    agfxRenderTarget* renderTarget = agfxRenderTargetCreate(device, &rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget);

    const agfxBufferCreateInfo vertexInfo = VertexBufferInfo();
    const agfxBufferCreateInfo indexInfo = IndexBufferInfo();
    agfxBuffer* vertexBuffer = agfxBufferCreate(device, &vertexInfo);
    agfxBuffer* indexBuffer = agfxBufferCreate(device, &indexInfo);
    AGFX_EXPECT_NOT_NULL(vertexBuffer);
    AGFX_EXPECT_NOT_NULL(indexBuffer);

    agfxBufferViewCreateInfo viewInfo{};
    viewInfo.buffer = vertexBuffer;
    viewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
    viewInfo.offset = 0;
    viewInfo.writeable = 0;
    agfxBufferView* vertexView = agfxBufferViewCreate(device, &viewInfo);
    AGFX_EXPECT_NOT_NULL(vertexView);

    // Both buffers are GPU-only, so they have to be staged in before the draw can see them.
    AGFX_EXPECT_MSG(UploadBuffer(device, gpu.Queue(), vertexBuffer, kVertices, sizeof(kVertices),
                                 AGFX_RESOURCE_STATE_COMMON), "vertex buffer upload failed");
    AGFX_EXPECT_MSG(UploadBuffer(device, gpu.Queue(), indexBuffer, kIndices, sizeof(kIndices),
                                 AGFX_RESOURCE_STATE_COMMON), "index buffer upload failed");

    agfxShaderModule* vsModule = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfxShaderModule* psModule = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
    const agfxRenderPipelineCreateInfo pipelineInfo = PipelineInfo(vsModule, psModule);
    agfxRenderPipeline* pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, vsModule);
    agfxShaderModuleDestroy(device, psModule);
    AGFX_EXPECT_NOT_NULL(pipeline);

    PushConstants constants{};
    constants.vertices = (uint32_t)agfxBufferViewGetHandle(vertexView);

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 1);
        agfxCommandBufferBufferBarrier(cmd, vertexBuffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 1);
        agfxCommandBufferBufferBarrier(cmd, indexBuffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_INDEX_BUFFER, 1);

        const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kWidth, (float)kHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kWidth, kHeight);
        agfxRenderPassSetPipeline(pass, pipeline);
        agfxRenderPassPushConstants(pass, &constants, sizeof(constants));
        agfxRenderPassDrawIndexed(pass, indexBuffer, kIndexCount, 1, 0, 0, 0);
        agfxRenderPassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_RENDER_TARGET, image);

    agfxRenderPipelineDestroy(device, pipeline);
    agfxBufferViewDestroy(device, vertexView);
    agfxBufferDestroy(device, indexBuffer);
    agfxBufferDestroy(device, vertexBuffer);
    agfxRenderTargetDestroy(device, renderTarget);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    AGFX_EXPECT(CountLitPixels(image) > kMinLinePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawLinesIndexed, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader vsShader = CompileTestShader("indexed.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("indexed.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile indexed.hlsl:main_vs");
    AGFX_EXPECT_MSG(psShader.Valid(), "failed to compile indexed.hlsl:main_ps");

    agfx::Texture target = device.CreateTexture(TargetInfo());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target;
    rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
    agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget.Get());

    agfx::Buffer vertexBuffer = device.CreateBuffer(VertexBufferInfo());
    agfx::Buffer indexBuffer = device.CreateBuffer(IndexBufferInfo());
    AGFX_EXPECT_NOT_NULL(vertexBuffer.Get());
    AGFX_EXPECT_NOT_NULL(indexBuffer.Get());

    agfxBufferViewCreateInfo viewInfo{};
    viewInfo.buffer = vertexBuffer;
    viewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
    viewInfo.offset = 0;
    viewInfo.writeable = 0;
    agfx::BufferView vertexView = device.CreateBufferView(viewInfo);
    AGFX_EXPECT_NOT_NULL(vertexView.Get());

    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, vertexBuffer, kVertices, sizeof(kVertices),
                                 AGFX_RESOURCE_STATE_COMMON), "vertex buffer upload failed");
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, indexBuffer, kIndices, sizeof(kIndices),
                                 AGFX_RESOURCE_STATE_COMMON), "index buffer upload failed");

    agfx::RenderPipeline pipeline;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        agfx::ShaderModule psModule(device.Get(),
            CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
        pipeline = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    PushConstants constants{};
    constants.vertices = (uint32_t)vertexView.GetHandle();

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    cmd.BufferBarrier(vertexBuffer, AGFX_RESOURCE_STATE_COMMON,
                      AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
    cmd.BufferBarrier(indexBuffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_INDEX_BUFFER, true);
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget));
        pass.SetViewport(0.0f, 0.0f, (float)kWidth, (float)kHeight);
        pass.SetScissor(0, 0, kWidth, kHeight);
        pass.SetPipeline(pipeline);
        pass.PushConstants(constants);
        pass.DrawIndexed(indexBuffer, kIndexCount, 1, 0, 0, 0);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_RENDER_TARGET, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    AGFX_EXPECT(CountLitPixels(image) > kMinLinePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawLinesIndexed, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: render straight into an offscreen target
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader vsShader = CompileTestShader("indexed.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("indexed.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile indexed.hlsl:main_vs");
    AGFX_EXPECT_MSG(psShader.Valid(), "failed to compile indexed.hlsl:main_ps");

    agfx::Device& device = context.GetDevice();

    agfx::ShaderModule vsModule(device.Get(),
        CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
    agfx::ShaderModule psModule(device.Get(),
        CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
    AGFX_EXPECT_NOT_NULL(vsModule.Get());
    AGFX_EXPECT_NOT_NULL(psModule.Get());

    agfx::ez::Texture2D target = context.CreateTexture2D(kWidth, kHeight, kFormat,
                                                         AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT);

    agfx::ez::Buffer vertexBuffer = context.CreateStructuredBuffer(kVertices, sizeof(kVertices), sizeof(Vertex));
    // The stride selects the index type; 4 is CreateIndexBuffer's default. The u16 twin passes 2.
    agfx::ez::Buffer indexBuffer = context.CreateIndexBuffer(kIndices, sizeof(kIndices), sizeof(uint32_t));

    agfx::ez::ShaderBindings bindings;
    bindings.BindBuffer(vertexBuffer.View(AGFX_BUFFER_VIEW_TYPE_STRUCTURED));

    agfx::ez::PipelineDesc desc;
    desc.name = "test draw lines indexed";
    desc.vertexShader = &vsModule;
    desc.fragmentShader = &psModule;
    desc.topology = AGFX_TOPOLOGY_LINES;
    desc.cullMode = AGFX_CULL_MODE_NONE;
    desc.depthTestEnable = false;
    desc.depthWriteEnable = false;

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(vertexBuffer, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.TransitionBuffer(indexBuffer, AGFX_RESOURCE_STATE_INDEX_BUFFER);
        context.SetRenderTargets({&target}, nullptr, AGFX_LOAD_OPERATION_CLEAR, kClearColor);
        context.SetViewportScissor(0, 0, kWidth, kHeight);
        context.SetPipeline(desc);
        context.PushShaderBindings(bindings);
        context.DrawIndexed(indexBuffer, kIndexCount);
        context.EndActivePass();
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                          kWidth, kHeight, kFormat, target.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    AGFX_EXPECT(CountLitPixels(image) > kMinLinePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
