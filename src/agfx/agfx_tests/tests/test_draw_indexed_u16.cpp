/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw indexed u16".
//
// The 16-bit twin of test_draw_indexed.cpp: identical geometry, identical shader, identical golden
// — only the index buffer's stride changes, from 4 to 2. Sharing draw_indexed.png as the golden is
// the point of the test: index width is supposed to be invisible in the output, so if a backend
// derives the index type from anything other than the buffer stride, or reads u16 indices as u32,
// the two tests disagree against the same expectation.
//
// There is no Ez variant: ez's CreateIndexBuffer hardcodes a stride of 0, which the backends read
// as 32-bit, so a u16 index buffer isn't expressible through that layer.

#include "../test_gpu.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr const char* kGolden = "draw_indexed.png"; // Shared with the u32 test, deliberately.
    constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    /// @brief Mirrors Vertex in indexed.hlsl.
    struct Vertex
    {
        float position[2];
        float padding[2];
        float color[4];
    };
    static_assert(sizeof(Vertex) == 32, "Vertex must match the HLSL layout");

    constexpr Vertex kVertices[4] = {
        {{-0.8f, -0.8f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}, // 0: bottom left,  red
        {{ 0.8f, -0.8f}, {0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}}, // 1: bottom right, green
        {{ 0.8f,  0.8f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}, // 2: top right,    blue
        {{-0.8f,  0.8f}, {0.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}}, // 3: top left,     yellow
    };

    constexpr uint16_t kIndices[6] = {0, 1, 2, 0, 2, 3};
    constexpr uint32_t kIndexCount = 6;

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

    /// @brief The one line that differs from the u32 test: a 2-byte stride is how both backends
    /// select R16_UINT / MTLIndexTypeUInt16.
    agfxBufferCreateInfo IndexBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = sizeof(kIndices);
        info.stride = sizeof(uint16_t);
        info.usage = AGFX_BUFFER_USAGE_INDEX;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "test draw indexed u16";
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
        info.name = "test draw indexed u16";
        return info;
    }
} // namespace

AGFX_TEST_TEXTURE(DrawIndexedU16, C, kWidth, kHeight)
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
    rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
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
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawIndexedU16, Cpp, kWidth, kHeight)
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

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
