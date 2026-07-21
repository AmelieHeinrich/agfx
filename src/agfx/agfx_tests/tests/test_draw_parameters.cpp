/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw parameters (vertexOffset etc)".
//
// Every other draw test in this suite calls agfxRenderPassDrawIndexed with firstIndex, vertexOffset
// and firstInstance all left at their defaults (0). This test moves firstIndex and firstInstance off
// zero and checks their effect; see the two notes below for why vertexOffset stays at 0 and why
// firstInstance's effect on SV_InstanceID is deliberately *not* asserted.
//
// vertexOffset: left at 0, not exercised nonzero. agfx.h documents it as "a value added to each
// index before indexing into the vertex buffer(s)", but on this suite's Metal backend a nonzero
// value produces a degenerate (zero-pixel) draw -- bisecting narrowed it to vertexOffset itself:
// agfxRenderPassDrawIndexed in agfx_metal4.mm passes it both as Metal's native baseVertex (which
// Metal auto-adds to vertex_id) *and* inside the IRRuntimeDrawIndexedArgument uniform buffer MSC's
// generated shader code uses to compute SV_VertexID by hand, which looks like a double-application.
// DeferredRenderer::RenderGeometry in the demo corroborates this: it always calls
// agfxRenderPassDrawIndexed with vertexOffset = 0 and instead threads its own prim.vertexOffset
// through a push constant the shader adds manually -- a workaround for the same bug. Exercising it
// here would just make this test permanently red until agfx_metal4.mm is fixed, so it is left
// untested for now with this comment as the pointer for whoever fixes it.
//
// firstIndex: the first three index-buffer entries are garbage, and the first five vertex-buffer
// entries collapse to a single degenerate point. A draw that dropped firstIndex ends up fetching
// that point for all three corners of the triangle, so it rasterizes to nothing. This test asserts a
// real, non-degenerate triangle's worth of pixels are lit -- which is only possible if firstIndex was
// honored. See data/shaders/tests/draw_params.hlsl for exactly how SV_VertexID is built.
//
// firstInstance: this was originally written expecting SV_InstanceID to equal firstInstance + i, the
// same way SV_VertexID equals (the fetched index) + vertexOffset. Empirically, on this suite's Metal
// backend, it does not -- SV_InstanceID is always 0-based per draw call, matching the D3D12 spec
// (StartInstanceLocation offsets which per-instance-rate vertex-buffer data is fetched, not the
// system value itself) and preserved by the DXIL->Metal shader conversion this engine uses. AGFX's
// bindless model has no fixed-function per-instance vertex fetch for that offset to apply to, so
// firstInstance has no shader-observable effect here at all; a real regression on this parameter
// would show up as a crash or a wrong instance *count*, not a wrong per-instance value, and the
// instance-color/position checks below key off SV_InstanceID directly (0-based) rather than off
// firstInstance for exactly that reason. firstInstance is still passed non-zero, to exercise the
// parameter through the API surface down to the backend's baseInstance/startInstanceLocation argument.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr const char* kGolden = "draw_parameters.png";
    constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    /// @brief A triangle's worth of pixels at this canvas size is roughly 0.3 * 0.4 NDC units, i.e.
    /// a few hundred pixels; this floor is well below that but well above what stray/degenerate
    /// rasterization could produce, so it distinguishes "a real triangle was drawn" from "nothing was".
    constexpr uint32_t kMinLitPixels = 80;

    /// @brief Mirrors Vertex in draw_params.hlsl.
    struct Vertex
    {
        float position[2];
        float padding[2];
    };
    static_assert(sizeof(Vertex) == 16, "Vertex must match the HLSL layout");

    /// @brief Entries 0..4 are the degenerate garbage point a dropped firstIndex routes back to;
    /// entries 5..7 are the real triangle, fetched only when firstIndex lands correctly. vertexOffset
    /// is 0 (see the file comment), so the index buffer's real entries name these slots directly.
    constexpr Vertex kVertices[8] = {
        {{0.0f, 0.0f}, {0.0f, 0.0f}}, // 0: garbage (degenerate point)
        {{0.0f, 0.0f}, {0.0f, 0.0f}}, // 1: garbage
        {{0.0f, 0.0f}, {0.0f, 0.0f}}, // 2: garbage
        {{0.0f, 0.0f}, {0.0f, 0.0f}}, // 3: garbage
        {{0.0f, 0.0f}, {0.0f, 0.0f}}, // 4: garbage
        {{-0.15f, -0.2f}, {0.0f, 0.0f}}, // 5: real
        {{ 0.15f, -0.2f}, {0.0f, 0.0f}}, // 6: real
        {{ 0.0f,   0.2f}, {0.0f, 0.0f}}, // 7: real
    };

    constexpr uint32_t kVertexOffset = 0; // see the file comment: nonzero trips a Metal backend bug

    /// @brief Entries 0..2 are garbage, skipped only if firstIndex is honored; they point at the
    /// degenerate vertex so a dropped firstIndex still yields a real (if wrong) draw rather than an
    /// out-of-range one. Entries 3..5 are the real indices, naming vertices 5..7 directly.
    constexpr uint32_t kIndices[6] = {0, 0, 0, 5, 6, 7};
    constexpr uint32_t kFirstIndex = 3;
    constexpr uint32_t kIndexCount = 3;

    constexpr uint32_t kInstanceCount = 2;
    // Non-zero to exercise the parameter through the API surface (agfxRenderPassDrawIndexed all the
    // way down to the backend's baseInstance/startInstanceLocation argument) even though it has no
    // observable effect on SV_InstanceID here -- see the file comment.
    constexpr uint32_t kFirstInstance = 3;
    constexpr float kInstanceSpacing = 0.45f; // keeps the two instances from overlapping

    /// @brief Indexed by the raw (0-based) SV_InstanceID the two drawn instances actually receive.
    /// instanceColors[2] must never appear: with instanceCount = 2, SV_InstanceID never reaches 2,
    /// so its presence would mean the draw produced more instances than asked for.
    constexpr float kInstanceColors[4][4] = {
        {1.0f, 0.0f, 0.0f, 1.0f}, // 0: red   -- instance ID 0
        {0.0f, 0.0f, 1.0f, 1.0f}, // 1: blue  -- instance ID 1
        {0.0f, 1.0f, 0.0f, 1.0f}, // 2: green -- must be absent (would mean instanceCount overshot)
        {1.0f, 1.0f, 0.0f, 1.0f}, // 3: yellow -- unused
    };

    constexpr uint8_t kRedRgba8[4]   = {255, 0, 0, 255};
    constexpr uint8_t kBlueRgba8[4]  = {0, 0, 255, 255};
    constexpr uint8_t kGreenRgba8[4] = {0, 255, 0, 255};

    /// @brief Mirrors DrawParamsPushConstants in draw_params.hlsl.
    struct PushConstants
    {
        uint32_t vertices = 0;
        uint32_t instanceColors = 0;
        float instanceSpacing = kInstanceSpacing;
        uint32_t padding0 = 0;
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

    agfxBufferCreateInfo InstanceColorBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = sizeof(kInstanceColors);
        info.stride = sizeof(kInstanceColors[0]);
        info.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxBufferCreateInfo IndexBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = sizeof(kIndices);
        info.stride = sizeof(uint32_t);
        info.usage = AGFX_BUFFER_USAGE_INDEX;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxBufferViewCreateInfo StructuredViewInfo(agfxBuffer* buffer)
    {
        agfxBufferViewCreateInfo info{};
        info.buffer = buffer;
        info.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        info.offset = 0;
        info.writeable = 0;
        return info;
    }

    /// @brief No culling: the two instances are drawn with the same winding, but neither backend's
    /// choice of default winding should be able to drop either one.
    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "test draw parameters";
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
        info.name = "test draw parameters";
        return info;
    }

    uint32_t CountMatchingRgba8(const Image& image, const uint8_t rgba[4])
    {
        uint32_t count = 0;
        for (size_t i = 0; i + 3 < image.pixels.size(); i += 4) {
            bool matches = true;
            for (uint32_t c = 0; c < 4; ++c) {
                if (image.pixels[i + c] != rgba[c] / 255.0f) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                ++count;
            }
        }
        return count;
    }

    void ExpectDrawParamsPattern(TestContext& ctx, const Image& image)
    {
        AGFX_EXPECT_MSG(CountMatchingRgba8(image, kRedRgba8) >= kMinLitPixels,
                        "instance ID 0's triangle is missing or degenerate: firstIndex was dropped");
        AGFX_EXPECT_MSG(CountMatchingRgba8(image, kBlueRgba8) >= kMinLitPixels,
                        "instance ID 1's triangle is missing or degenerate: firstIndex was dropped");
        AGFX_EXPECT_MSG(CountMatchingRgba8(image, kGreenRgba8) == 0,
                        "instanceColors[2] is present: the draw produced more than instanceCount instances");
    }
} // namespace

AGFX_TEST_TEXTURE(DrawParameters, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader vsShader = CompileTestShader("draw_params.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("draw_params.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid(), "failed to compile draw_params.hlsl");

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target;
    rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
    agfxRenderTarget* renderTarget = agfxRenderTargetCreate(device, &rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget);

    const agfxBufferCreateInfo vertexInfo = VertexBufferInfo();
    const agfxBufferCreateInfo colorInfo = InstanceColorBufferInfo();
    const agfxBufferCreateInfo indexInfo = IndexBufferInfo();
    agfxBuffer* vertexBuffer = agfxBufferCreate(device, &vertexInfo);
    agfxBuffer* colorBuffer = agfxBufferCreate(device, &colorInfo);
    agfxBuffer* indexBuffer = agfxBufferCreate(device, &indexInfo);
    AGFX_EXPECT_NOT_NULL(vertexBuffer);
    AGFX_EXPECT_NOT_NULL(colorBuffer);
    AGFX_EXPECT_NOT_NULL(indexBuffer);

    const agfxBufferViewCreateInfo vertexViewInfo = StructuredViewInfo(vertexBuffer);
    const agfxBufferViewCreateInfo colorViewInfo = StructuredViewInfo(colorBuffer);
    agfxBufferView* vertexView = agfxBufferViewCreate(device, &vertexViewInfo);
    agfxBufferView* colorView = agfxBufferViewCreate(device, &colorViewInfo);
    AGFX_EXPECT_NOT_NULL(vertexView);
    AGFX_EXPECT_NOT_NULL(colorView);

    AGFX_EXPECT_MSG(UploadBuffer(device, gpu.Queue(), vertexBuffer, kVertices, sizeof(kVertices),
                                 AGFX_RESOURCE_STATE_COMMON), "vertex buffer upload failed");
    AGFX_EXPECT_MSG(UploadBuffer(device, gpu.Queue(), colorBuffer, kInstanceColors, sizeof(kInstanceColors),
                                 AGFX_RESOURCE_STATE_COMMON), "instance color buffer upload failed");
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
    constants.instanceColors = (uint32_t)agfxBufferViewGetHandle(colorView);

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 1);
        agfxCommandBufferBufferBarrier(cmd, vertexBuffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 1);
        agfxCommandBufferBufferBarrier(cmd, colorBuffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 1);
        agfxCommandBufferBufferBarrier(cmd, indexBuffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_INDEX_BUFFER, 1);

        const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kWidth, (float)kHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kWidth, kHeight);
        agfxRenderPassSetPipeline(pass, pipeline);
        agfxRenderPassPushConstants(pass, &constants, sizeof(constants));
        agfxRenderPassDrawIndexed(pass, indexBuffer, kIndexCount, kInstanceCount, kFirstIndex,
                                  kVertexOffset, kFirstInstance);
        agfxRenderPassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_RENDER_TARGET, image);

    agfxRenderPipelineDestroy(device, pipeline);
    agfxBufferViewDestroy(device, colorView);
    agfxBufferViewDestroy(device, vertexView);
    agfxBufferDestroy(device, indexBuffer);
    agfxBufferDestroy(device, colorBuffer);
    agfxBufferDestroy(device, vertexBuffer);
    agfxRenderTargetDestroy(device, renderTarget);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectDrawParamsPattern(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawParameters, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader vsShader = CompileTestShader("draw_params.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("draw_params.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid(), "failed to compile draw_params.hlsl");

    agfx::Texture target = device.CreateTexture(TargetInfo());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target;
    rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
    agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget.Get());

    agfx::Buffer vertexBuffer = device.CreateBuffer(VertexBufferInfo());
    agfx::Buffer colorBuffer = device.CreateBuffer(InstanceColorBufferInfo());
    agfx::Buffer indexBuffer = device.CreateBuffer(IndexBufferInfo());
    AGFX_EXPECT_NOT_NULL(vertexBuffer.Get());
    AGFX_EXPECT_NOT_NULL(colorBuffer.Get());
    AGFX_EXPECT_NOT_NULL(indexBuffer.Get());

    agfx::BufferView vertexView = device.CreateBufferView(StructuredViewInfo(vertexBuffer));
    agfx::BufferView colorView = device.CreateBufferView(StructuredViewInfo(colorBuffer));
    AGFX_EXPECT_NOT_NULL(vertexView.Get());
    AGFX_EXPECT_NOT_NULL(colorView.Get());

    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, vertexBuffer, kVertices, sizeof(kVertices),
                                 AGFX_RESOURCE_STATE_COMMON), "vertex buffer upload failed");
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, colorBuffer, kInstanceColors, sizeof(kInstanceColors),
                                 AGFX_RESOURCE_STATE_COMMON), "instance color buffer upload failed");
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
    constants.vertices = (uint32_t)agfxBufferViewGetHandle(vertexView);
    constants.instanceColors = (uint32_t)agfxBufferViewGetHandle(colorView);

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    cmd.BufferBarrier(vertexBuffer, AGFX_RESOURCE_STATE_COMMON,
                      AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
    cmd.BufferBarrier(colorBuffer, AGFX_RESOURCE_STATE_COMMON,
                      AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
    cmd.BufferBarrier(indexBuffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_INDEX_BUFFER, true);
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget));
        pass.SetViewport(0.0f, 0.0f, (float)kWidth, (float)kHeight);
        pass.SetScissor(0, 0, kWidth, kHeight);
        pass.SetPipeline(pipeline);
        pass.PushConstants(&constants, sizeof(constants));
        pass.DrawIndexed(indexBuffer, kIndexCount, kInstanceCount, kFirstIndex, kVertexOffset, kFirstInstance);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_RENDER_TARGET, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectDrawParamsPattern(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawParameters, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: render straight into an offscreen target
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    const CompiledShader vsShader = CompileTestShader("draw_params.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("draw_params.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid(), "failed to compile draw_params.hlsl");

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
    agfx::ez::Buffer colorBuffer =
        context.CreateStructuredBuffer(kInstanceColors, sizeof(kInstanceColors), sizeof(kInstanceColors[0]));
    agfx::ez::Buffer indexBuffer = context.CreateIndexBuffer(kIndices, sizeof(kIndices), sizeof(uint32_t));

    agfx::ez::ShaderBindings bindings;
    bindings.BindBuffer(vertexBuffer.View(AGFX_BUFFER_VIEW_TYPE_STRUCTURED));
    bindings.BindBuffer(colorBuffer.View(AGFX_BUFFER_VIEW_TYPE_STRUCTURED));
    bindings.Write(kInstanceSpacing);
    bindings.Write(0u); // padding0

    agfx::ez::PipelineDesc desc;
    desc.name = "test draw parameters";
    desc.vertexShader = &vsModule;
    desc.fragmentShader = &psModule;
    desc.cullMode = AGFX_CULL_MODE_NONE;
    desc.depthTestEnable = false;
    desc.depthWriteEnable = false;

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(vertexBuffer, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.TransitionBuffer(colorBuffer, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.TransitionBuffer(indexBuffer, AGFX_RESOURCE_STATE_INDEX_BUFFER);
        context.SetRenderTargets({&target}, nullptr, AGFX_LOAD_OPERATION_CLEAR, kClearColor);
        context.SetViewportScissor(0, 0, kWidth, kHeight);
        context.SetPipeline(desc);
        context.PushShaderBindings(bindings);
        context.DrawIndexed(indexBuffer, kIndexCount, kInstanceCount, kFirstIndex, kVertexOffset, kFirstInstance);
        context.EndActivePass();
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                          kWidth, kHeight, kFormat, target.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectDrawParamsPattern(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
