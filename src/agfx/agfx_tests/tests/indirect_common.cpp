/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "indirect_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>
#include <vector>

namespace agfxtest
{
    namespace
    {
        constexpr float kClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        constexpr uint32_t kDispatchGroupSize = 8; // Matches main_dispatch_cs's [numthreads(8,8,1)].

        bool IsMeshKind(IndirectKind kind)
        {
            return kind == IndirectKind::DrawMesh || kind == IndirectKind::DrawTaskMesh;
        }

        bool IsDrawKind(IndirectKind kind)
        {
            return kind != IndirectKind::Dispatch;
        }

        agfxIndirectBundleType BundleType(IndirectKind kind)
        {
            switch (kind) {
            case IndirectKind::Draw:         return AGFX_INDIRECT_BUNDLE_TYPE_DRAW;
            case IndirectKind::DrawIndexed:  return AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED;
            case IndirectKind::DrawMesh:     return AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH;
            case IndirectKind::DrawTaskMesh: return AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH;
            case IndirectKind::Dispatch:     return AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH;
            }
            return AGFX_INDIRECT_BUNDLE_TYPE_DRAW;
        }

        /// @brief The producer entry point that authors this bundle type's commands. Both mesh kinds
        /// share one: the difference between them is the consumer pipeline, not the command.
        const char* ProducerEntryPoint(IndirectKind kind)
        {
            switch (kind) {
            case IndirectKind::Draw:         return "main_produce_draw_cs";
            case IndirectKind::DrawIndexed:  return "main_produce_draw_indexed_cs";
            case IndirectKind::DrawMesh:
            case IndirectKind::DrawTaskMesh: return "main_produce_draw_mesh_cs";
            case IndirectKind::Dispatch:     return "main_produce_dispatch_cs";
            }
            return "";
        }

        // --- Geometry for the indexed path ----------------------------------------------------

        /// @brief Mirrors IndirectVertex in indirect.hlsl. The padding keeps the structured view's
        /// stride at 16 bytes on both sides.
        struct Vertex
        {
            float position[2];
            float padding[2];
        };
        static_assert(sizeof(Vertex) == 16, "Vertex must match the HLSL layout");

        /// @brief The three columns' corners, four each, matching IndirectColumnCorner in
        /// indirect.hlsl. Duplicated here rather than derived on the GPU because the point of the
        /// indexed path is that the *host's* buffers are reached through a GPU-authored firstIndex.
        std::vector<Vertex> ColumnVertices()
        {
            std::vector<Vertex> vertices;
            vertices.reserve(kIndirectColumns * 4);
            for (uint32_t c = 0; c < kIndirectColumns; ++c) {
                const float x0 = -0.85f + 0.6f * (float)c;
                const float x1 = x0 + 0.5f;
                vertices.push_back({{x0, -0.5f}, {0.0f, 0.0f}}); // 0: bottom left
                vertices.push_back({{x1, -0.5f}, {0.0f, 0.0f}}); // 1: bottom right
                vertices.push_back({{x1,  0.5f}, {0.0f, 0.0f}}); // 2: top right
                vertices.push_back({{x0,  0.5f}, {0.0f, 0.0f}}); // 3: top left
            }
            return vertices;
        }

        /// @brief Six indices per column, two triangles sharing the 0-2 diagonal. Command i reads
        /// the run at firstIndex = i * 6, so a per-command index offset that fails to round trip
        /// puts a column in the wrong place rather than merely recoloring it.
        std::vector<uint32_t> ColumnIndices()
        {
            std::vector<uint32_t> indices;
            indices.reserve(kIndirectColumns * 6);
            for (uint32_t c = 0; c < kIndirectColumns; ++c) {
                const uint32_t base = c * 4;
                indices.insert(indices.end(), {base + 0, base + 1, base + 2,
                                               base + 0, base + 2, base + 3});
            }
            return indices;
        }

        // --- Create infos ---------------------------------------------------------------------

        agfxTextureCreateInfo TargetInfo(IndirectKind kind)
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kIndirectFormat;
            // The dispatch consumer writes through a UAV; the draw consumers rasterize.
            info.usage = (agfxTextureUsage)((IsDrawKind(kind) ? AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT
                                                              : AGFX_TEXTURE_USAGE_STORAGE) |
                                            AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kIndirectWidth;
            info.height = kIndirectHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxIndirectBundleCreateInfo BundleInfo(IndirectKind kind)
        {
            agfxIndirectBundleCreateInfo info{};
            info.type = BundleType(kind);
            info.maxCommandCount = kIndirectColumns;
            info.maxCountCount = 1;
            return info;
        }

        /// @brief The consumer render pipeline. supportsIndirect is the whole reason this is not
        /// just a copy of the raster tests' pipeline -- see the header.
        agfxRenderPipelineCreateInfo PipelineInfo(const IndirectState& state,
                                                  const CompiledShader& meshShader,
                                                  const CompiledShader& taskShader,
                                                  agfxShaderModule* vs, agfxShaderModule* ms,
                                                  agfxShaderModule* as, agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "test indirect";
            info.supportsIndirect = 1;
            info.fillMode = AGFX_FILL_MODE_SOLID;
            // As in the raster and mesh tests: culling off, so a winding difference between backends
            // cannot drop a column and be mistaken for a lost command.
            info.cullMode = AGFX_CULL_MODE_NONE;
            info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
            info.topology = AGFX_TOPOLOGY_TRIANGLES;
            info.depthTestEnable = 0;
            info.depthWriteEnable = 0;
            info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
            info.colorAttachmentCount = 1;
            info.colorFormats[0] = kIndirectFormat;
            info.fragmentShader = ps;
            if (IsMeshKind(state.kind)) {
                info.meshShader = ms;
                info.meshGroupSizeX = meshShader.meshSizeX;
                info.meshGroupSizeY = meshShader.meshSizeY;
                info.meshGroupSizeZ = meshShader.meshSizeZ;
                if (state.kind == IndirectKind::DrawTaskMesh) {
                    info.taskShader = as;
                    info.taskGroupSizeX = taskShader.taskSizeX;
                    info.taskGroupSizeY = taskShader.taskSizeY;
                    info.taskGroupSizeZ = taskShader.taskSizeZ;
                }
            } else {
                info.vertexShader = vs;
            }
            return info;
        }

        agfxComputePipelineCreateInfo ComputePipelineInfo(const char* name, const CompiledShader& shader,
                                                          agfxShaderModule* module)
        {
            agfxComputePipelineCreateInfo info{};
            info.name = name;
            info.computeShader = module;
            // Reflected rather than hardcoded, so editing the shader's [numthreads(...)] cannot
            // silently desync the host.
            info.groupSizeX = shader.groupSizeX;
            info.groupSizeY = shader.groupSizeY;
            info.groupSizeZ = shader.groupSizeZ;
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
            info.width = kIndirectWidth;
            info.height = kIndirectHeight;
            info.name = "test indirect";
            return info;
        }

        /// @brief The one place an agfxIndirectBundleExecuteInfo is built.
        ///
        /// Prepare and Execute are both handed the result of this call. They must agree field for
        /// field -- most sharply on pushConstants, which Metal bakes into the pre-encoded commands
        /// at prepare time and cannot patch later, while D3D12 reads them only at execute time.
        /// Building them separately is the single easiest way to write an indirect path that works
        /// on Windows and renders black on macOS, so there is deliberately no second constructor.
        agfxIndirectBundleExecuteInfo ExecuteInfo(const IndirectState& state,
                                                  const IndirectConstants& consumerConstants,
                                                  agfxRenderPipeline* renderPipeline,
                                                  agfxComputePipeline* computePipeline,
                                                  agfxBuffer* indexBuffer)
        {
            agfxIndirectBundleExecuteInfo info{};
            info.countIndex = 0;
            info.commandOffset = 0;
            info.maxCommandCount = kIndirectColumns;
            memcpy(info.pushConstants, &consumerConstants, sizeof(consumerConstants));
            info.renderPipeline = renderPipeline;
            info.computePipeline = computePipeline;
            info.indexBuffer = state.kind == IndirectKind::DrawIndexed ? indexBuffer : nullptr;
            return info;
        }

        /// @brief The push constants the *producer* dispatch reads. Only the bundle handle and the
        /// command bound; the consumer's half stays zeroed.
        IndirectConstants ProducerConstants(const IndirectState& state, uint64_t bundleHandle)
        {
            IndirectConstants constants{};
            constants.bundleHandle = bundleHandle;
            constants.commandCount = state.commandCount;
            return constants;
        }

        // --- C ---------------------------------------------------------------------------------

        IndirectResult RenderIndirectC(const IndirectState& state, const CompiledShader& producerShader,
                                       const CompiledShader& consumerShader,
                                       const CompiledShader& taskShader, const CompiledShader& psShader,
                                       const char* consumerEntry, const char* psEntry, Image& outImage)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                return IndirectResult::Failed;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo targetInfo = TargetInfo(state.kind);
            agfxTexture* target = agfxTextureCreate(device, &targetInfo);
            if (!target) {
                return IndirectResult::Failed;
            }

            agfxRenderTarget* renderTarget = nullptr;
            agfxTextureView* uav = nullptr;
            if (IsDrawKind(state.kind)) {
                agfxRenderTargetCreateInfo rtInfo{};
                rtInfo.texture = target;
                rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
                renderTarget = agfxRenderTargetCreate(device, &rtInfo);
            } else {
                agfxTextureViewCreateInfo viewInfo{};
                viewInfo.texture = target;
                viewInfo.format = kIndirectFormat;
                viewInfo.type = AGFX_TEXTURE_TYPE_2D;
                viewInfo.mipLevelCount = 1;
                viewInfo.arrayLayerCount = 1;
                viewInfo.writeable = 1;
                uav = agfxTextureViewCreate(device, &viewInfo);
            }

            const agfxIndirectBundleCreateInfo bundleInfo = BundleInfo(state.kind);
            agfxIndirectBundle* bundle = agfxIndirectBundleCreate(device, &bundleInfo);

            // Indexed-path geometry. Created unconditionally-but-empty elsewhere would complicate
            // the teardown; these stay null for every other kind.
            agfxBuffer* vertexBuffer = nullptr;
            agfxBuffer* indexBuffer = nullptr;
            agfxBufferView* vertexView = nullptr;
            const std::vector<Vertex> vertices = ColumnVertices();
            const std::vector<uint32_t> indices = ColumnIndices();
            if (state.kind == IndirectKind::DrawIndexed) {
                agfxBufferCreateInfo vbInfo{};
                vbInfo.size = vertices.size() * sizeof(Vertex);
                vbInfo.stride = sizeof(Vertex);
                vbInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
                vbInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
                vertexBuffer = agfxBufferCreate(device, &vbInfo);

                agfxBufferCreateInfo ibInfo{};
                ibInfo.size = indices.size() * sizeof(uint32_t);
                ibInfo.stride = sizeof(uint32_t);
                ibInfo.usage = AGFX_BUFFER_USAGE_INDEX;
                ibInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
                indexBuffer = agfxBufferCreate(device, &ibInfo);

                agfxBufferViewCreateInfo viewInfo{};
                viewInfo.buffer = vertexBuffer;
                viewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
                viewInfo.offset = 0;
                viewInfo.writeable = 0;
                vertexView = agfxBufferViewCreate(device, &viewInfo);
            }

            // Pipelines.
            agfxShaderModule* producerModule = CreateShaderModule(device, producerShader,
                ProducerEntryPoint(state.kind), AGFX_SHADER_MODULE_TYPE_COMPUTE);
            const agfxComputePipelineCreateInfo producerPipelineInfo =
                ComputePipelineInfo("test indirect producer", producerShader, producerModule);
            agfxComputePipeline* producerPipeline = agfxComputePipelineCreate(device, &producerPipelineInfo);
            agfxShaderModuleDestroy(device, producerModule);

            agfxRenderPipeline* renderPipeline = nullptr;
            agfxComputePipeline* consumerPipeline = nullptr;
            if (IsDrawKind(state.kind)) {
                const agfxShaderModuleType consumerType =
                    IsMeshKind(state.kind) ? AGFX_SHADER_MODULE_TYPE_MESH : AGFX_SHADER_MODULE_TYPE_VERTEX;
                agfxShaderModule* consumerModule =
                    CreateShaderModule(device, consumerShader, consumerEntry, consumerType);
                agfxShaderModule* asModule =
                    state.kind == IndirectKind::DrawTaskMesh
                        ? CreateShaderModule(device, taskShader, "main_as", AGFX_SHADER_MODULE_TYPE_TASK)
                        : nullptr;
                agfxShaderModule* psModule =
                    CreateShaderModule(device, psShader, psEntry, AGFX_SHADER_MODULE_TYPE_FRAGMENT);

                const agfxRenderPipelineCreateInfo pipelineInfo = PipelineInfo(
                    state, consumerShader, taskShader,
                    IsMeshKind(state.kind) ? nullptr : consumerModule,
                    IsMeshKind(state.kind) ? consumerModule : nullptr, asModule, psModule);
                renderPipeline = agfxRenderPipelineCreate(device, &pipelineInfo);

                agfxShaderModuleDestroy(device, consumerModule);
                if (asModule) {
                    agfxShaderModuleDestroy(device, asModule);
                }
                agfxShaderModuleDestroy(device, psModule);
            } else {
                agfxShaderModule* consumerModule =
                    CreateShaderModule(device, consumerShader, consumerEntry, AGFX_SHADER_MODULE_TYPE_COMPUTE);
                const agfxComputePipelineCreateInfo info =
                    ComputePipelineInfo("test indirect consumer", consumerShader, consumerModule);
                consumerPipeline = agfxComputePipelineCreate(device, &info);
                agfxShaderModuleDestroy(device, consumerModule);
            }

            bool ok = bundle != nullptr && producerPipeline != nullptr &&
                      (IsDrawKind(state.kind) ? (renderTarget != nullptr && renderPipeline != nullptr)
                                              : (uav != nullptr && consumerPipeline != nullptr));
            if (ok && state.kind == IndirectKind::DrawIndexed) {
                ok = vertexBuffer && indexBuffer && vertexView &&
                     UploadBuffer(device, gpu.Queue(), vertexBuffer, vertices.data(),
                                  vertices.size() * sizeof(Vertex), AGFX_RESOURCE_STATE_COMMON) &&
                     UploadBuffer(device, gpu.Queue(), indexBuffer, indices.data(),
                                  indices.size() * sizeof(uint32_t), AGFX_RESOURCE_STATE_COMMON);
            }

            if (ok) {
                agfxBuffer* commandsBuffer = agfxIndirectBundleGetCommandsBuffer(bundle);
                agfxBuffer* countBuffer = agfxIndirectBundleGetCountBuffer(bundle);

                // The count slot must start at zero: the producer only ever InterlockedAdds into
                // it, so a stale count would replay garbage commands. The commands buffer needs no
                // reset -- both backends clamp execution to the live count, so whatever is beyond
                // it is never read.
                const uint32_t zero = 0;
                ok = UploadBuffer(device, gpu.Queue(), countBuffer, &zero, sizeof(zero),
                                  AGFX_RESOURCE_STATE_COMMON);

                IndirectConstants consumerConstants{};
                if (vertexView) {
                    consumerConstants.vertices = (uint32_t)agfxBufferViewGetHandle(vertexView);
                }
                if (uav) {
                    consumerConstants.destination = (uint32_t)agfxTextureViewGetHandle(uav);
                }
                const IndirectConstants producerConstants =
                    ProducerConstants(state, agfxIndirectBundleGetHandle(bundle));
                const agfxIndirectBundleExecuteInfo executeInfo =
                    ExecuteInfo(state, consumerConstants, renderPipeline, consumerPipeline, indexBuffer);

                agfxDeviceMakeResourcesResident(device);

                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                                    IsDrawKind(state.kind)
                                                        ? AGFX_RESOURCE_STATE_RENDER_TARGET
                                                        : AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    0, 0, 1);
                    if (state.kind == IndirectKind::DrawIndexed) {
                        agfxCommandBufferBufferBarrier(cmd, vertexBuffer, AGFX_RESOURCE_STATE_COMMON,
                                                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 1);
                        agfxCommandBufferBufferBarrier(cmd, indexBuffer, AGFX_RESOURCE_STATE_COMMON,
                                                       AGFX_RESOURCE_STATE_INDEX_BUFFER, 1);
                    }
                    agfxCommandBufferBufferBarrier(cmd, commandsBuffer, AGFX_RESOURCE_STATE_COMMON,
                                                   AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 1);
                    agfxCommandBufferBufferBarrier(cmd, countBuffer, AGFX_RESOURCE_STATE_COMMON,
                                                   AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 1);

                    agfxComputePass* producePass = agfxComputePassBegin(cmd, "test indirect produce");
                    agfxComputePassSetPipeline(producePass, producerPipeline);
                    agfxComputePassPushConstants(producePass, &producerConstants, sizeof(producerConstants));
                    // One group: the producer's [numthreads] is already the column count.
                    agfxComputePassDispatch(producePass, 1, 1, 1);

                    // The prepare kernel reads what the producer just wrote.
                    agfxComputePassBufferUAVBarrier(producePass, commandsBuffer);
                    agfxComputePassBufferUAVBarrier(producePass, countBuffer);

                    // No-op on D3D12; builds the ICB on Metal. Must be inside a compute pass.
                    agfxComputePassPrepareIndirectBundle(producePass, bundle, &executeInfo);
                    agfxComputePassEnd(producePass);

                    agfxCommandBufferBufferBarrier(cmd, commandsBuffer, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                   AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT, 1);
                    agfxCommandBufferBufferBarrier(cmd, countBuffer, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                   AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT, 1);

                    if (IsDrawKind(state.kind)) {
                        const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget);
                        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kIndirectWidth,
                                                  (float)kIndirectHeight, 0.0f, 1.0f);
                        agfxRenderPassSetScissor(pass, 0, 0, kIndirectWidth, kIndirectHeight);
                        // No SetPipeline/PushConstants here on purpose: ExecuteIndirectBundle binds
                        // the pipeline, push constants and index buffer from executeInfo itself.
                        agfxRenderPassExecuteIndirectBundle(pass, bundle, &executeInfo);
                        agfxRenderPassEnd(pass);
                    } else {
                        agfxComputePass* pass = agfxComputePassBegin(cmd, "test indirect execute");
                        agfxComputePassExecuteIndirectBundle(pass, bundle, &executeInfo);
                        agfxComputePassEnd(pass);
                    }
                });

                ok = ok && ReadbackTexture2D(device, gpu.Queue(), target, kIndirectWidth, kIndirectHeight,
                                             kIndirectFormat,
                                             IsDrawKind(state.kind) ? AGFX_RESOURCE_STATE_RENDER_TARGET
                                                                    : AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                             outImage);
            }

            if (renderPipeline) {
                agfxRenderPipelineDestroy(device, renderPipeline);
            }
            if (consumerPipeline) {
                agfxComputePipelineDestroy(device, consumerPipeline);
            }
            if (producerPipeline) {
                agfxComputePipelineDestroy(device, producerPipeline);
            }
            if (bundle) {
                agfxIndirectBundleDestroy(device, bundle);
            }
            if (vertexView) {
                agfxBufferViewDestroy(device, vertexView);
            }
            if (indexBuffer) {
                agfxBufferDestroy(device, indexBuffer);
            }
            if (vertexBuffer) {
                agfxBufferDestroy(device, vertexBuffer);
            }
            if (uav) {
                agfxTextureViewDestroy(device, uav);
            }
            if (renderTarget) {
                agfxRenderTargetDestroy(device, renderTarget);
            }
            agfxTextureDestroy(device, target);
            return ok ? IndirectResult::Ok : IndirectResult::Failed;
        }

        // --- C++ -------------------------------------------------------------------------------

        IndirectResult RenderIndirectCpp(const IndirectState& state, const CompiledShader& producerShader,
                                         const CompiledShader& consumerShader,
                                         const CompiledShader& taskShader, const CompiledShader& psShader,
                                         const char* consumerEntry, const char* psEntry, Image& outImage)
        {
            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                return IndirectResult::Failed;
            }

            agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture target = device.CreateTexture(TargetInfo(state.kind));
            if (!target.Get()) {
                return IndirectResult::Failed;
            }

            agfx::RenderTarget renderTarget;
            agfx::TextureView uav;
            if (IsDrawKind(state.kind)) {
                agfxRenderTargetCreateInfo rtInfo{};
                rtInfo.texture = target;
                rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
                renderTarget = device.CreateRenderTarget(rtInfo);
            } else {
                agfxTextureViewCreateInfo viewInfo{};
                viewInfo.texture = target;
                viewInfo.format = kIndirectFormat;
                viewInfo.type = AGFX_TEXTURE_TYPE_2D;
                viewInfo.mipLevelCount = 1;
                viewInfo.arrayLayerCount = 1;
                viewInfo.writeable = 1;
                uav = device.CreateTextureView(viewInfo);
            }

            agfx::IndirectBundle bundle = device.CreateIndirectBundle(BundleInfo(state.kind));
            if (!bundle.Get()) {
                return IndirectResult::Failed;
            }

            agfx::Buffer vertexBuffer;
            agfx::Buffer indexBuffer;
            agfx::BufferView vertexView;
            const std::vector<Vertex> vertices = ColumnVertices();
            const std::vector<uint32_t> indices = ColumnIndices();
            if (state.kind == IndirectKind::DrawIndexed) {
                agfxBufferCreateInfo vbInfo{};
                vbInfo.size = vertices.size() * sizeof(Vertex);
                vbInfo.stride = sizeof(Vertex);
                vbInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
                vbInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
                vertexBuffer = device.CreateBuffer(vbInfo);

                agfxBufferCreateInfo ibInfo{};
                ibInfo.size = indices.size() * sizeof(uint32_t);
                ibInfo.stride = sizeof(uint32_t);
                ibInfo.usage = AGFX_BUFFER_USAGE_INDEX;
                ibInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
                indexBuffer = device.CreateBuffer(ibInfo);

                agfxBufferViewCreateInfo viewInfo{};
                viewInfo.buffer = vertexBuffer;
                viewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
                viewInfo.offset = 0;
                viewInfo.writeable = 0;
                vertexView = device.CreateBufferView(viewInfo);

                if (!UploadBuffer(device.Get(), queue, vertexBuffer, vertices.data(),
                                  vertices.size() * sizeof(Vertex), AGFX_RESOURCE_STATE_COMMON) ||
                    !UploadBuffer(device.Get(), queue, indexBuffer, indices.data(),
                                  indices.size() * sizeof(uint32_t), AGFX_RESOURCE_STATE_COMMON)) {
                    return IndirectResult::Failed;
                }
            }

            agfx::ComputePipeline producerPipeline;
            {
                agfx::ShaderModule module(device.Get(),
                    CreateShaderModule(device.Get(), producerShader, ProducerEntryPoint(state.kind),
                                       AGFX_SHADER_MODULE_TYPE_COMPUTE));
                producerPipeline = device.CreateComputePipeline(
                    ComputePipelineInfo("test indirect producer", producerShader, module));
            }

            agfx::RenderPipeline renderPipeline;
            agfx::ComputePipeline consumerPipeline;
            if (IsDrawKind(state.kind)) {
                const agfxShaderModuleType consumerType =
                    IsMeshKind(state.kind) ? AGFX_SHADER_MODULE_TYPE_MESH : AGFX_SHADER_MODULE_TYPE_VERTEX;
                agfx::ShaderModule consumerModule(device.Get(),
                    CreateShaderModule(device.Get(), consumerShader, consumerEntry, consumerType));
                agfx::ShaderModule asModule(device.Get(),
                    state.kind == IndirectKind::DrawTaskMesh
                        ? CreateShaderModule(device.Get(), taskShader, "main_as", AGFX_SHADER_MODULE_TYPE_TASK)
                        : nullptr);
                agfx::ShaderModule psModule(device.Get(),
                    CreateShaderModule(device.Get(), psShader, psEntry, AGFX_SHADER_MODULE_TYPE_FRAGMENT));
                renderPipeline = device.CreateRenderPipeline(PipelineInfo(
                    state, consumerShader, taskShader,
                    IsMeshKind(state.kind) ? nullptr : consumerModule.Get(),
                    IsMeshKind(state.kind) ? consumerModule.Get() : nullptr,
                    asModule.Get(), psModule.Get()));
            } else {
                agfx::ShaderModule module(device.Get(),
                    CreateShaderModule(device.Get(), consumerShader, consumerEntry,
                                       AGFX_SHADER_MODULE_TYPE_COMPUTE));
                consumerPipeline = device.CreateComputePipeline(
                    ComputePipelineInfo("test indirect consumer", consumerShader, module));
            }

            if (!producerPipeline.Get() ||
                (IsDrawKind(state.kind) ? !renderPipeline.Get() : !consumerPipeline.Get())) {
                return IndirectResult::Failed;
            }

            agfxBuffer* commandsBuffer = bundle.CommandsBuffer();
            agfxBuffer* countBuffer = bundle.CountBuffer();

            const uint32_t zero = 0;
            if (!UploadBuffer(device.Get(), queue, countBuffer, &zero, sizeof(zero),
                              AGFX_RESOURCE_STATE_COMMON)) {
                return IndirectResult::Failed;
            }

            IndirectConstants consumerConstants{};
            if (vertexView.Get()) {
                consumerConstants.vertices = (uint32_t)vertexView.GetHandle();
            }
            if (uav.Get()) {
                consumerConstants.destination = (uint32_t)uav.GetHandle();
            }
            const IndirectConstants producerConstants = ProducerConstants(state, bundle.GetHandle());
            const agfxIndirectBundleExecuteInfo executeInfo = ExecuteInfo(
                state, consumerConstants, renderPipeline.Get(), consumerPipeline.Get(), indexBuffer.Get());

            device.MakeResourcesResident();

            cmd.Begin();
            cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON,
                               IsDrawKind(state.kind) ? AGFX_RESOURCE_STATE_RENDER_TARGET
                                                      : AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            if (state.kind == IndirectKind::DrawIndexed) {
                cmd.BufferBarrier(vertexBuffer, AGFX_RESOURCE_STATE_COMMON,
                                  AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
                cmd.BufferBarrier(indexBuffer, AGFX_RESOURCE_STATE_COMMON,
                                  AGFX_RESOURCE_STATE_INDEX_BUFFER, true);
            }
            cmd.BufferBarrier(commandsBuffer, AGFX_RESOURCE_STATE_COMMON,
                              AGFX_RESOURCE_STATE_UNORDERED_ACCESS, true);
            cmd.BufferBarrier(countBuffer, AGFX_RESOURCE_STATE_COMMON,
                              AGFX_RESOURCE_STATE_UNORDERED_ACCESS, true);
            {
                agfx::ComputePass pass = cmd.BeginComputePass("test indirect produce");
                pass.SetPipeline(producerPipeline);
                pass.PushConstants(producerConstants);
                pass.Dispatch(1, 1, 1);
                pass.BufferUAVBarrier(commandsBuffer);
                pass.BufferUAVBarrier(countBuffer);
                pass.PrepareIndirectBundle(bundle, executeInfo);
            }
            cmd.BufferBarrier(commandsBuffer, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                              AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT, true);
            cmd.BufferBarrier(countBuffer, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                              AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT, true);
            if (IsDrawKind(state.kind)) {
                agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget));
                pass.SetViewport(0.0f, 0.0f, (float)kIndirectWidth, (float)kIndirectHeight);
                pass.SetScissor(0, 0, kIndirectWidth, kIndirectHeight);
                pass.ExecuteIndirectBundle(bundle, executeInfo);
            } else {
                agfx::ComputePass pass = cmd.BeginComputePass("test indirect execute");
                pass.ExecuteIndirectBundle(bundle, executeInfo);
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            const bool ok = ReadbackTexture2D(device.Get(), queue, target, kIndirectWidth, kIndirectHeight,
                                              kIndirectFormat,
                                              IsDrawKind(state.kind) ? AGFX_RESOURCE_STATE_RENDER_TARGET
                                                                     : AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                              outImage);
            return ok ? IndirectResult::Ok : IndirectResult::Failed;
        }

        // --- ez --------------------------------------------------------------------------------

        IndirectResult RenderIndirectEz(const IndirectState& state, const CompiledShader& producerShader,
                                        const CompiledShader& consumerShader,
                                        const CompiledShader& taskShader, const CompiledShader& psShader,
                                        const char* consumerEntry, const char* psEntry, Image& outImage)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: straight into an offscreen target
            contextInfo.width = kIndirectWidth;
            contextInfo.height = kIndirectHeight;
            agfx::ez::Context context(contextInfo);
            agfx::Device& device = context.GetDevice();

            // ez has no compute-pass sugar by design, so the producer pipeline, its dispatch and the
            // prepare all go through the raw API on ez's own command buffer. Only the render-side
            // replay uses Context::ExecuteIndirectBundle.
            agfx::ComputePipeline producerPipeline;
            {
                agfx::ShaderModule module(device.Get(),
                    CreateShaderModule(device.Get(), producerShader, ProducerEntryPoint(state.kind),
                                       AGFX_SHADER_MODULE_TYPE_COMPUTE));
                producerPipeline = device.CreateComputePipeline(
                    ComputePipelineInfo("test indirect producer", producerShader, module));
            }
            if (!producerPipeline.Get()) {
                return IndirectResult::Failed;
            }

            // The modules must outlive the draw: ez's PipelineDesc caches on their addresses.
            const agfxShaderModuleType consumerType =
                IsMeshKind(state.kind)   ? AGFX_SHADER_MODULE_TYPE_MESH
                : IsDrawKind(state.kind) ? AGFX_SHADER_MODULE_TYPE_VERTEX
                                         : AGFX_SHADER_MODULE_TYPE_COMPUTE;
            agfx::ShaderModule consumerModule(device.Get(),
                CreateShaderModule(device.Get(), consumerShader, consumerEntry, consumerType));
            agfx::ShaderModule asModule(device.Get(),
                state.kind == IndirectKind::DrawTaskMesh
                    ? CreateShaderModule(device.Get(), taskShader, "main_as", AGFX_SHADER_MODULE_TYPE_TASK)
                    : nullptr);
            agfx::ShaderModule psModule(device.Get(),
                IsDrawKind(state.kind)
                    ? CreateShaderModule(device.Get(), psShader, psEntry, AGFX_SHADER_MODULE_TYPE_FRAGMENT)
                    : nullptr);
            if (!consumerModule.Get()) {
                return IndirectResult::Failed;
            }

            agfx::ComputePipeline consumerPipeline;
            if (!IsDrawKind(state.kind)) {
                consumerPipeline = device.CreateComputePipeline(
                    ComputePipelineInfo("test indirect consumer", consumerShader, consumerModule));
                if (!consumerPipeline.Get()) {
                    return IndirectResult::Failed;
                }
            }

            agfx::ez::Texture2D target = context.CreateTexture2D(
                kIndirectWidth, kIndirectHeight, kIndirectFormat,
                IsDrawKind(state.kind) ? AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT : AGFX_TEXTURE_USAGE_STORAGE);

            agfx::ez::Buffer vertexBuffer;
            agfx::ez::Buffer indexBuffer;
            const std::vector<Vertex> vertices = ColumnVertices();
            const std::vector<uint32_t> indices = ColumnIndices();
            if (state.kind == IndirectKind::DrawIndexed) {
                vertexBuffer = context.CreateStructuredBuffer(vertices.data(),
                                                              vertices.size() * sizeof(Vertex), sizeof(Vertex));
                indexBuffer = context.CreateIndexBuffer(indices.data(), indices.size() * sizeof(uint32_t),
                                                        sizeof(uint32_t));
            }

            agfx::ez::IndirectBundle bundle =
                context.CreateIndirectBundle(BundleType(state.kind), kIndirectColumns);

            const uint32_t zero = 0;
            if (!UploadBuffer(device.Get(), context.GetGraphicsQueue(), bundle.Raw().CountBuffer(), &zero,
                              sizeof(zero), AGFX_RESOURCE_STATE_COMMON)) {
                return IndirectResult::Failed;
            }

            agfx::ez::PipelineDesc desc;
            desc.name = "test indirect";
            desc.supportsIndirect = true; // Required for the ICB replay on Metal; see PipelineDesc.
            desc.fragmentShader = &psModule;
            if (IsMeshKind(state.kind)) {
                desc.meshShader = &consumerModule;
                desc.meshGroupSizeX = consumerShader.meshSizeX;
                desc.meshGroupSizeY = consumerShader.meshSizeY;
                desc.meshGroupSizeZ = consumerShader.meshSizeZ;
                if (state.kind == IndirectKind::DrawTaskMesh) {
                    desc.taskShader = &asModule;
                    desc.taskGroupSizeX = taskShader.taskSizeX;
                    desc.taskGroupSizeY = taskShader.taskSizeY;
                    desc.taskGroupSizeZ = taskShader.taskSizeZ;
                }
            } else {
                desc.vertexShader = &consumerModule;
            }
            desc.cullMode = AGFX_CULL_MODE_NONE;
            desc.depthTestEnable = false;
            desc.depthWriteEnable = false;

            IndirectConstants consumerConstants{};
            if (state.kind == IndirectKind::DrawIndexed) {
                consumerConstants.vertices =
                    (uint32_t)vertexBuffer.View(AGFX_BUFFER_VIEW_TYPE_STRUCTURED).GetHandle();
            }
            if (!IsDrawKind(state.kind)) {
                consumerConstants.destination = (uint32_t)target.UAV().GetHandle();
            }
            const IndirectConstants producerConstants = ProducerConstants(state, bundle.GetHandle());

            device.MakeResourcesResident();

            {
                agfx::ez::Frame frame = context.BeginFrame();

                if (state.kind == IndirectKind::DrawIndexed) {
                    context.TransitionBuffer(vertexBuffer, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    context.TransitionBuffer(indexBuffer, AGFX_RESOURCE_STATE_INDEX_BUFFER);
                }
                context.TransitionTexture(target, IsDrawKind(state.kind)
                                                      ? AGFX_RESOURCE_STATE_RENDER_TARGET
                                                      : AGFX_RESOURCE_STATE_UNORDERED_ACCESS);
                context.TransitionIndirectBundle(bundle, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

                // ExecuteInfo is built after the pipeline exists, because the render pipeline comes
                // out of ez's cache rather than being created up front. It is still the *same*
                // helper the other two flavors use, so prepare and execute cannot disagree.
                agfx::RenderPipeline* cachedPipeline =
                    IsDrawKind(state.kind) ? &context.GetOrCreatePipeline(desc, {&target}, nullptr) : nullptr;
                const agfxIndirectBundleExecuteInfo executeInfo = ExecuteInfo(
                    state, consumerConstants, cachedPipeline ? cachedPipeline->Get() : nullptr,
                    consumerPipeline.Get(),
                    state.kind == IndirectKind::DrawIndexed ? indexBuffer.Raw().Get() : nullptr);

                {
                    agfx::ComputePass pass =
                        context.GetCurrentCommandBuffer().BeginComputePass("test indirect produce");
                    pass.SetPipeline(producerPipeline);
                    pass.PushConstants(producerConstants);
                    pass.Dispatch(1, 1, 1);
                    pass.BufferUAVBarrier(bundle.Raw().CommandsBuffer());
                    pass.BufferUAVBarrier(bundle.Raw().CountBuffer());
                    pass.PrepareIndirectBundle(bundle.Raw(), executeInfo);
                }

                context.TransitionIndirectBundle(bundle, AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT);

                if (IsDrawKind(state.kind)) {
                    context.SetRenderTargets({&target}, nullptr, AGFX_LOAD_OPERATION_CLEAR, kClearColor);
                    context.SetViewportScissor(0, 0, kIndirectWidth, kIndirectHeight);
                    context.SetPipeline(desc);
                    context.ExecuteIndirectBundle(bundle, executeInfo);
                    context.EndActivePass();
                } else {
                    agfx::ComputePass pass =
                        context.GetCurrentCommandBuffer().BeginComputePass("test indirect execute");
                    pass.ExecuteIndirectBundle(bundle.Raw(), executeInfo);
                }
            }
            context.DrainGPU();

            const bool ok = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                              kIndirectWidth, kIndirectHeight, kIndirectFormat,
                                              target.State(), outImage);
            return ok ? IndirectResult::Ok : IndirectResult::Failed;
        }
    } // namespace

    bool DeviceSupportsIndirect(IndirectKind kind)
    {
        GpuFixture gpu;
        if (!gpu.Valid()) {
            return false;
        }

        agfxDeviceInfo info{};
        agfxDeviceGetInfo(gpu.Device(), &info);
        if (!info.supportsMultiDrawIndirect) {
            return false;
        }
        // A mesh bundle needs both capabilities, and they are independent: a device can support
        // indirect submission without supporting mesh shading.
        return !IsMeshKind(kind) || info.supportsMeshShaders != 0;
    }

    IndirectResult RenderIndirect(TestApi api, const IndirectState& state, Image& outImage)
    {
        if (!DeviceSupportsIndirect(state.kind)) {
            return IndirectResult::Unsupported;
        }

        // The consumer entry point is what distinguishes the five kinds beyond the bundle type: the
        // indexed path pulls vertices through a structured buffer, and the task path's mesh entry
        // takes a payload (part of the signature, so it cannot be one shared entry point).
        const char* consumerEntry = nullptr;
        const char* psEntry = "main_ps";
        agfxShaderStage consumerStage = AGFX_SHADER_STAGE_VERTEX;
        switch (state.kind) {
        case IndirectKind::Draw:
            consumerEntry = "main_vs";
            break;
        case IndirectKind::DrawIndexed:
            consumerEntry = "main_vs_indexed";
            break;
        case IndirectKind::DrawMesh:
            consumerEntry = "main_ms";
            consumerStage = AGFX_SHADER_STAGE_MESH;
            psEntry = "main_ms_ps";
            break;
        case IndirectKind::DrawTaskMesh:
            consumerEntry = "main_ms_payload";
            consumerStage = AGFX_SHADER_STAGE_MESH;
            psEntry = "main_ms_ps";
            break;
        case IndirectKind::Dispatch:
            consumerEntry = "main_dispatch_cs";
            consumerStage = AGFX_SHADER_STAGE_COMPUTE;
            break;
        }

        const CompiledShader producerShader = CompileTestShader(
            "indirect.hlsl", AGFX_SHADER_STAGE_COMPUTE, ProducerEntryPoint(state.kind));
        const CompiledShader consumerShader =
            CompileTestShader("indirect.hlsl", consumerStage, consumerEntry);
        CompiledShader psShader;
        if (IsDrawKind(state.kind)) {
            psShader = CompileTestShader("indirect.hlsl", AGFX_SHADER_STAGE_FRAGMENT, psEntry);
        }
        CompiledShader taskShader;
        if (state.kind == IndirectKind::DrawTaskMesh) {
            taskShader = CompileTestShader("indirect.hlsl", AGFX_SHADER_STAGE_TASK, "main_as");
        }

        if (!producerShader.Valid() || !consumerShader.Valid() ||
            (IsDrawKind(state.kind) && !psShader.Valid()) ||
            (state.kind == IndirectKind::DrawTaskMesh && !taskShader.Valid())) {
            return IndirectResult::Failed;
        }

        switch (api) {
        case TestApi::C:
            return RenderIndirectC(state, producerShader, consumerShader, taskShader, psShader,
                                   consumerEntry, psEntry, outImage);
        case TestApi::Cpp:
            return RenderIndirectCpp(state, producerShader, consumerShader, taskShader, psShader,
                                     consumerEntry, psEntry, outImage);
        case TestApi::Ez:
            return RenderIndirectEz(state, producerShader, consumerShader, taskShader, psShader,
                                    consumerEntry, psEntry, outImage);
        }
        return IndirectResult::Failed;
    }
} // namespace agfxtest
