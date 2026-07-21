/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "mesh_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace agfxtest
{
    namespace
    {
        constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

        agfxTextureCreateInfo TargetInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kMeshFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kMeshWidth;
            info.height = kMeshHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        /// @brief A mesh pipeline: meshShader (plus optional taskShader) instead of vertexShader.
        ///
        /// The group sizes are copied from what the compiler reflected out of the shaders rather
        /// than hardcoded to (3,1,1)/(1,1,1). They must match the declared [numthreads(...)] or the
        /// backends disagree about dispatch shape, and reflecting them means editing mesh.hlsl's
        /// thread counts cannot silently desync the host side.
        agfxRenderPipelineCreateInfo PipelineInfo(const MeshState& state, const CompiledShader& meshShader,
                                                  const CompiledShader& taskShader, agfxShaderModule* ms,
                                                  agfxShaderModule* as, agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "test mesh";
            info.fillMode = AGFX_FILL_MODE_SOLID;
            info.cullMode = AGFX_CULL_MODE_NONE; // As in the raster tests: winding must not drop a triangle.
            info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
            info.topology = AGFX_TOPOLOGY_TRIANGLES;
            info.depthTestEnable = 0;
            info.depthWriteEnable = 0;
            info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
            info.colorAttachmentCount = 1;
            info.colorFormats[0] = kMeshFormat;
            info.meshShader = ms;
            info.fragmentShader = ps;
            info.meshGroupSizeX = meshShader.meshSizeX;
            info.meshGroupSizeY = meshShader.meshSizeY;
            info.meshGroupSizeZ = meshShader.meshSizeZ;
            if (state.useTaskShader) {
                info.taskShader = as;
                info.taskGroupSizeX = taskShader.taskSizeX;
                info.taskGroupSizeY = taskShader.taskSizeY;
                info.taskGroupSizeZ = taskShader.taskSizeZ;
            }
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
            info.width = kMeshWidth;
            info.height = kMeshHeight;
            info.name = "test mesh";
            return info;
        }

        /// @brief With a task shader the draw dispatches *task* groups, and the task shader amplifies
        /// to kMeshGroupCount mesh groups itself (from the groupCount push constant). Dispatching
        /// kMeshGroupCount task groups instead would amplify to four triangles, not two.
        uint32_t DrawGroupCount(const MeshState& state)
        {
            return state.useTaskShader ? 1u : kMeshGroupCount;
        }

        MeshResult RenderMeshC(const MeshState& state, const CompiledShader& meshShader,
                               const CompiledShader& taskShader, const CompiledShader& psShader,
                               Image& outImage)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                return MeshResult::Failed;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo targetInfo = TargetInfo();
            agfxTexture* target = agfxTextureCreate(device, &targetInfo);
            if (!target) {
                return MeshResult::Failed;
            }

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfxRenderTarget* renderTarget = agfxRenderTargetCreate(device, &rtInfo);

            agfxShaderModule* msModule =
                CreateShaderModule(device, meshShader, state.useTaskShader ? "main_ms_payload" : "main_ms",
                                   AGFX_SHADER_MODULE_TYPE_MESH);
            agfxShaderModule* asModule =
                state.useTaskShader
                    ? CreateShaderModule(device, taskShader, "main_as", AGFX_SHADER_MODULE_TYPE_TASK)
                    : nullptr;
            agfxShaderModule* psModule =
                CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
            const agfxRenderPipelineCreateInfo pipelineInfo =
                PipelineInfo(state, meshShader, taskShader, msModule, asModule, psModule);
            agfxRenderPipeline* pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
            agfxShaderModuleDestroy(device, msModule);
            if (asModule) {
                agfxShaderModuleDestroy(device, asModule);
            }
            agfxShaderModuleDestroy(device, psModule);

            bool ok = renderTarget != nullptr && pipeline != nullptr;
            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);

                    const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
                    agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kMeshWidth, (float)kMeshHeight, 0.0f, 1.0f);
                    agfxRenderPassSetScissor(pass, 0, 0, kMeshWidth, kMeshHeight);
                    agfxRenderPassSetPipeline(pass, pipeline);
                    agfxRenderPassPushConstants(pass, &state.constants, sizeof(state.constants));
                    agfxRenderPassDrawMesh(pass, DrawGroupCount(state), 1, 1);
                    agfxRenderPassEnd(pass);
                });

                ok = ReadbackTexture2D(device, gpu.Queue(), target, kMeshWidth, kMeshHeight, kMeshFormat,
                                       AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
            }

            if (pipeline) {
                agfxRenderPipelineDestroy(device, pipeline);
            }
            if (renderTarget) {
                agfxRenderTargetDestroy(device, renderTarget);
            }
            agfxTextureDestroy(device, target);
            return ok ? MeshResult::Ok : MeshResult::Failed;
        }

        MeshResult RenderMeshCpp(const MeshState& state, const CompiledShader& meshShader,
                                 const CompiledShader& taskShader, const CompiledShader& psShader,
                                 Image& outImage)
        {
            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                return MeshResult::Failed;
            }

            agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture target = device.CreateTexture(TargetInfo());

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);

            agfx::RenderPipeline pipeline;
            {
                agfx::ShaderModule msModule(device.Get(),
                    CreateShaderModule(device.Get(), meshShader,
                                       state.useTaskShader ? "main_ms_payload" : "main_ms",
                                       AGFX_SHADER_MODULE_TYPE_MESH));
                agfx::ShaderModule asModule(device.Get(),
                    state.useTaskShader
                        ? CreateShaderModule(device.Get(), taskShader, "main_as", AGFX_SHADER_MODULE_TYPE_TASK)
                        : nullptr);
                agfx::ShaderModule psModule(device.Get(),
                    CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
                pipeline = device.CreateRenderPipeline(
                    PipelineInfo(state, meshShader, taskShader, msModule, asModule, psModule));
            }
            if (!target.Get() || !renderTarget.Get() || !pipeline.Get()) {
                return MeshResult::Failed;
            }

            device.MakeResourcesResident();

            cmd.Begin();
            cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
            {
                agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget));
                pass.SetViewport(0.0f, 0.0f, (float)kMeshWidth, (float)kMeshHeight);
                pass.SetScissor(0, 0, kMeshWidth, kMeshHeight);
                pass.SetPipeline(pipeline);
                pass.PushConstants(&state.constants, sizeof(state.constants));
                pass.DrawMesh(DrawGroupCount(state), 1, 1);
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            const bool ok = ReadbackTexture2D(device.Get(), queue, target, kMeshWidth, kMeshHeight,
                                              kMeshFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
            return ok ? MeshResult::Ok : MeshResult::Failed;
        }

        MeshResult RenderMeshEz(const MeshState& state, const CompiledShader& meshShader,
                                const CompiledShader& taskShader, const CompiledShader& psShader,
                                Image& outImage)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: straight into an offscreen target
            contextInfo.width = kMeshWidth;
            contextInfo.height = kMeshHeight;
            agfx::ez::Context context(contextInfo);

            agfx::Device& device = context.GetDevice();
            agfx::ShaderModule msModule(device.Get(),
                CreateShaderModule(device.Get(), meshShader,
                                   state.useTaskShader ? "main_ms_payload" : "main_ms",
                                   AGFX_SHADER_MODULE_TYPE_MESH));
            agfx::ShaderModule asModule(device.Get(),
                state.useTaskShader
                    ? CreateShaderModule(device.Get(), taskShader, "main_as", AGFX_SHADER_MODULE_TYPE_TASK)
                    : nullptr);
            agfx::ShaderModule psModule(device.Get(),
                CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
            if (!msModule.Get() || !psModule.Get() || (state.useTaskShader && !asModule.Get())) {
                return MeshResult::Failed;
            }

            agfx::ez::Texture2D target = context.CreateTexture2D(kMeshWidth, kMeshHeight, kMeshFormat,
                                                                 AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT);

            // The modules must outlive the draw: ez's PipelineDesc caches on their addresses.
            agfx::ez::PipelineDesc desc;
            desc.name = "test mesh";
            desc.meshShader = &msModule;
            desc.fragmentShader = &psModule;
            // ez does not reflect the group sizes, so they are copied in explicitly -- the header on
            // PipelineDesc spells out that they must match the shader's [numthreads(...)].
            desc.meshGroupSizeX = meshShader.meshSizeX;
            desc.meshGroupSizeY = meshShader.meshSizeY;
            desc.meshGroupSizeZ = meshShader.meshSizeZ;
            if (state.useTaskShader) {
                desc.taskShader = &asModule;
                desc.taskGroupSizeX = taskShader.taskSizeX;
                desc.taskGroupSizeY = taskShader.taskSizeY;
                desc.taskGroupSizeZ = taskShader.taskSizeZ;
            }
            desc.cullMode = AGFX_CULL_MODE_NONE;
            desc.depthTestEnable = false;
            desc.depthWriteEnable = false;

            agfx::ez::ShaderBindings bindings;
            bindings.Write(&state.constants, sizeof(state.constants));

            {
                agfx::ez::Frame frame = context.BeginFrame();
                context.SetRenderTargets({&target}, nullptr, AGFX_LOAD_OPERATION_CLEAR, kClearColor);
                context.SetViewportScissor(0, 0, kMeshWidth, kMeshHeight);
                context.SetPipeline(desc);
                context.PushShaderBindings(bindings);
                context.DrawMesh(DrawGroupCount(state), 1, 1);
                context.EndActivePass();
            }
            context.DrainGPU();

            const bool ok = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                              kMeshWidth, kMeshHeight, kMeshFormat, target.State(), outImage);
            return ok ? MeshResult::Ok : MeshResult::Failed;
        }
    } // namespace

    bool DeviceSupportsMeshShaders()
    {
        GpuFixture gpu;
        if (!gpu.Valid()) {
            return false;
        }

        agfxDeviceInfo info{};
        agfxDeviceGetInfo(gpu.Device(), &info);
        return info.supportsMeshShaders != 0;
    }

    MeshResult RenderMesh(TestApi api, const MeshState& state, Image& outImage)
    {
        if (!DeviceSupportsMeshShaders()) {
            return MeshResult::Unsupported;
        }

        // The mesh entry point differs between the two paths: the task path's takes the payload,
        // which is part of the signature and so cannot be one shared entry point.
        const CompiledShader meshShader = CompileTestShader(
            "mesh.hlsl", AGFX_SHADER_STAGE_MESH, state.useTaskShader ? "main_ms_payload" : "main_ms");
        const CompiledShader psShader =
            CompileTestShader("mesh.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        CompiledShader taskShader;
        if (state.useTaskShader) {
            taskShader = CompileTestShader("mesh.hlsl", AGFX_SHADER_STAGE_TASK, "main_as");
        }

        if (!meshShader.Valid() || !psShader.Valid() || (state.useTaskShader && !taskShader.Valid())) {
            return MeshResult::Failed;
        }

        switch (api) {
        case TestApi::C:   return RenderMeshC(state, meshShader, taskShader, psShader, outImage);
        case TestApi::Cpp: return RenderMeshCpp(state, meshShader, taskShader, psShader, outImage);
        case TestApi::Ez:  return RenderMeshEz(state, meshShader, taskShader, psShader, outImage);
        }
        return MeshResult::Failed;
    }
} // namespace agfxtest
