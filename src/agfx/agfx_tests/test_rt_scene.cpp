/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "test_rt_scene.h"

namespace agfxtest
{
    // --- Geometry ---------------------------------------------------------------------------

    const std::vector<float>& RtTriangleVertices()
    {
        // Apex above centre, base below it, and wider than it is tall — an equilateral triangle
        // would look the same transposed, which is exactly the failure a build bug produces.
        static const std::vector<float> vertices = {
            0.0f,  0.75f, 0.0f, //
            -0.8f, -0.5f, 0.0f, //
            0.8f,  -0.5f, 0.0f, //
        };
        return vertices;
    }

    const std::vector<uint32_t>& RtTriangleIndices()
    {
        static const std::vector<uint32_t> indices = {0, 1, 2};
        return indices;
    }

    const std::vector<float>& RtMultiTriangleVertices()
    {
        // Three triangles staggered in x and stepped back in z. The z steps are what the depth
        // channel of the golden picks up, so primitive order is observable and not just coverage.
        static const std::vector<float> vertices = {
            // Left, nearest.
            -0.9f, 0.6f,  0.2f, //
            -0.9f, -0.6f, 0.2f, //
            -0.1f, -0.6f, 0.2f, //
            // Centre, middle depth.
            0.0f,  0.7f,  0.6f, //
            -0.4f, -0.3f, 0.6f, //
            0.4f,  -0.3f, 0.6f, //
            // Right, furthest.
            0.2f,  0.5f,  1.0f, //
            0.2f,  -0.7f, 1.0f, //
            0.9f,  -0.7f, 1.0f, //
        };
        return vertices;
    }

    const std::vector<uint32_t>& RtMultiTriangleIndices()
    {
        static const std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        return indices;
    }

    const std::vector<float>& RtAabbData()
    {
        // (min, max) pairs. Given some depth so a ray along +z actually passes through it.
        static const std::vector<float> aabbs = {
            -0.5f, -0.5f, -0.25f, //
            0.5f,  0.5f,  0.25f,  //
        };
        return aabbs;
    }

    const std::vector<float>& RtMultiAabbData()
    {
        static const std::vector<float> aabbs = {
            -0.8f, -0.4f, -0.25f, //
            -0.2f, 0.4f,  0.25f,  //
            0.2f,  -0.4f, -0.25f, //
            0.8f,  0.4f,  0.25f,  //
        };
        return aabbs;
    }

    void RtIdentityTransform(float outTransform[12])
    {
        RtTranslationTransform(outTransform, 0.0f, 0.0f, 0.0f);
    }

    void RtTranslationTransform(float outTransform[12], float x, float y, float z)
    {
        // Row-major 3x4: three rows of (basis, basis, basis, translation).
        const float values[12] = {
            1.0f, 0.0f, 0.0f, x, //
            0.0f, 1.0f, 0.0f, y, //
            0.0f, 0.0f, 1.0f, z, //
        };
        for (int i = 0; i < 12; ++i) {
            outTransform[i] = values[i];
        }
    }

    // --- Buffers ----------------------------------------------------------------------------

    agfxBuffer* RtCreateInputBuffer(GpuFixture& gpu, const void* data, uint64_t size, uint32_t stride,
                                    const char* name)
    {
        (void)name;

        agfxBufferCreateInfo info{};
        info.size = size;
        info.stride = stride;
        info.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;

        agfxBuffer* buffer = agfxBufferCreate(gpu.Device(), &info);
        if (!buffer) {
            return nullptr;
        }

        // GPU_ONLY means the contents have to come in through a staging copy; a build reading an
        // uninitialised buffer would otherwise produce a plausible-looking but arbitrary AS.
        if (!UploadBuffer(gpu.Device(), gpu.Queue(), buffer, data, size,
                          AGFX_RESOURCE_STATE_COMMON)) {
            agfxBufferDestroy(gpu.Device(), buffer);
            return nullptr;
        }
        return buffer;
    }

    agfxBuffer* RtCreateScratchBuffer(agfxDevice* device, uint64_t size, const char* name)
    {
        (void)name;

        agfxBufferCreateInfo info{};
        info.size = size;
        info.stride = 0;
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return agfxBufferCreate(device, &info);
    }

    // --- Acceleration structures ------------------------------------------------------------

    RtBlas::~RtBlas()
    {
        if (!device) {
            return;
        }
        if (blas) agfxAccelerationStructureDestroy(device, blas);
        if (aabbBuffer) agfxBufferDestroy(device, aabbBuffer);
        if (indexBuffer) agfxBufferDestroy(device, indexBuffer);
        if (vertexBuffer) agfxBufferDestroy(device, vertexBuffer);
    }

    RtBlas& RtBlas::operator=(RtBlas&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        this->~RtBlas();
        device = other.device;
        blas = other.blas;
        vertexBuffer = other.vertexBuffer;
        indexBuffer = other.indexBuffer;
        aabbBuffer = other.aabbBuffer;
        other.device = nullptr;
        other.blas = nullptr;
        other.vertexBuffer = nullptr;
        other.indexBuffer = nullptr;
        other.aabbBuffer = nullptr;
        return *this;
    }

    RtTlas::~RtTlas()
    {
        if (device && tlas) {
            agfxAccelerationStructureDestroy(device, tlas);
        }
    }

    RtTlas& RtTlas::operator=(RtTlas&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        this->~RtTlas();
        device = other.device;
        tlas = other.tlas;
        other.device = nullptr;
        other.tlas = nullptr;
        return *this;
    }

    bool RtBuildAndWait(GpuFixture& gpu, agfxAccelerationStructure* as)
    {
        agfxAccelerationStructureSizes sizes{};
        agfxAccelerationStructureGetSizes(gpu.Device(), as, &sizes);

        agfxBuffer* scratch = RtCreateScratchBuffer(gpu.Device(), sizes.scratchBufferSize, "scratch");
        if (!scratch) {
            return false;
        }

        gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
            agfxComputePass* pass = agfxComputePassBegin(cmd, "build acceleration structure");
            agfxComputePassBuildAccelerationStructure(pass, as, scratch, 0);
            agfxComputePassEnd(pass);
            // The trace dispatch reads the AS as a shader resource; without this the read can race
            // the build on both backends.
            agfxCommandBufferAccelerationStructureBarrier(
                cmd, as, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
        });

        agfxBufferDestroy(gpu.Device(), scratch);
        return true;
    }

    namespace
    {
        RtBlas BuildTriangleBlasFrom(GpuFixture& gpu, const std::vector<float>& vertices,
                                     const std::vector<uint32_t>& indices, bool opaque,
                                     bool allowUpdate)
        {
            RtBlas result;
            result.device = gpu.Device();

            result.vertexBuffer =
                RtCreateInputBuffer(gpu, vertices.data(), vertices.size() * sizeof(float),
                                    sizeof(float) * 3, "rt vertices");
            result.indexBuffer =
                RtCreateInputBuffer(gpu, indices.data(), indices.size() * sizeof(uint32_t),
                                    sizeof(uint32_t), "rt indices");
            if (!result.vertexBuffer || !result.indexBuffer) {
                return result;
            }

            agfxAccelerationStructureGeometry geometry{};
            geometry.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES;
            geometry.opaque = opaque ? 1 : 0;
            geometry.triangles.vertexBuffer = result.vertexBuffer;
            geometry.triangles.vertexOffset = 0;
            geometry.triangles.vertexCount = (uint32_t)(vertices.size() / 3);
            geometry.triangles.indexBuffer = result.indexBuffer;
            geometry.triangles.indexOffset = 0;
            geometry.triangles.indexCount = (uint32_t)indices.size();

            agfxAccelerationStructureCreateInfo info{};
            info.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            info.bottomLevel.geometries = &geometry;
            info.bottomLevel.geometryCount = 1;
            info.allowUpdate = allowUpdate ? 1 : 0;
            info.name = "rt triangle blas";

            result.blas = agfxAccelerationStructureCreate(gpu.Device(), &info);
            if (!result.blas || !RtBuildAndWait(gpu, result.blas)) {
                if (result.blas) {
                    agfxAccelerationStructureDestroy(gpu.Device(), result.blas);
                    result.blas = nullptr;
                }
            }
            return result;
        }
    } // namespace

    RtBlas RtBuildTriangleBlas(GpuFixture& gpu, bool opaque, bool allowUpdate)
    {
        return BuildTriangleBlasFrom(gpu, RtTriangleVertices(), RtTriangleIndices(), opaque,
                                     allowUpdate);
    }

    RtBlas RtBuildMultiTriangleBlas(GpuFixture& gpu, bool opaque, bool allowUpdate)
    {
        return BuildTriangleBlasFrom(gpu, RtMultiTriangleVertices(), RtMultiTriangleIndices(), opaque,
                                     allowUpdate);
    }

    RtBlas RtBuildAabbBlas(GpuFixture& gpu, const std::vector<float>& aabbs, uint32_t aabbCount,
                           bool opaque)
    {
        RtBlas result;
        result.device = gpu.Device();

        // An AABB is two float3s, so the stride is six floats regardless of how many there are.
        const uint32_t stride = sizeof(float) * 6;
        result.aabbBuffer = RtCreateInputBuffer(gpu, aabbs.data(), aabbs.size() * sizeof(float),
                                                stride, "rt aabbs");
        if (!result.aabbBuffer) {
            return result;
        }

        agfxAccelerationStructureGeometry geometry{};
        geometry.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_AABBS;
        geometry.opaque = opaque ? 1 : 0;
        geometry.aabbs.aabbBuffer = result.aabbBuffer;
        geometry.aabbs.aabbOffset = 0;
        geometry.aabbs.aabbCount = aabbCount;
        geometry.aabbs.aabbStride = stride;

        agfxAccelerationStructureCreateInfo info{};
        info.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        info.bottomLevel.geometries = &geometry;
        info.bottomLevel.geometryCount = 1;
        info.allowUpdate = 0;
        info.name = "rt aabb blas";

        result.blas = agfxAccelerationStructureCreate(gpu.Device(), &info);
        if (!result.blas || !RtBuildAndWait(gpu, result.blas)) {
            if (result.blas) {
                agfxAccelerationStructureDestroy(gpu.Device(), result.blas);
                result.blas = nullptr;
            }
        }
        return result;
    }

    RtTlas RtBuildTlas(GpuFixture& gpu, const std::vector<agfxAccelerationStructureInstance>& instances,
                       uint32_t maxInstanceCount)
    {
        RtTlas result;
        result.device = gpu.Device();

        agfxAccelerationStructureCreateInfo info{};
        info.type = AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        info.topLevel.maxInstanceCount =
            maxInstanceCount ? maxInstanceCount : (uint32_t)instances.size();
        info.allowUpdate = 0;
        info.name = "rt tlas";

        result.tlas = agfxAccelerationStructureCreate(gpu.Device(), &info);
        if (!result.tlas) {
            return result;
        }

        agfxAccelerationStructureAddInstances(result.tlas, instances.data(),
                                              (uint32_t)instances.size());

        if (!RtBuildAndWait(gpu, result.tlas)) {
            agfxAccelerationStructureDestroy(gpu.Device(), result.tlas);
            result.tlas = nullptr;
        }
        return result;
    }

    RtTlas RtBuildSingleInstanceTlas(GpuFixture& gpu, agfxAccelerationStructure* blas, uint32_t userID,
                                     bool opaque)
    {
        agfxAccelerationStructureInstance instance{};
        instance.blas = blas;
        instance.userID = userID;
        instance.opaque = opaque ? 1 : 0;
        RtIdentityTransform(instance.transform);

        return RtBuildTlas(gpu, {instance});
    }

    // --- Tracing ----------------------------------------------------------------------------

    bool RtTraceToImage(GpuFixture& gpu, agfxAccelerationStructure* tlas, const char* entryPoint,
                        Image& outImage)
    {
        agfxDevice* device = gpu.Device();

        const CompiledShader shader =
            CompileTestShader("raytracing.hlsl", AGFX_SHADER_STAGE_COMPUTE, entryPoint);
        if (!shader.Valid()) {
            return false;
        }

        agfxTextureCreateInfo targetInfo{};
        targetInfo.type = AGFX_TEXTURE_TYPE_2D;
        targetInfo.format = kRtFormat;
        targetInfo.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        targetInfo.width = kRtWidth;
        targetInfo.height = kRtHeight;
        targetInfo.depthOrArrayLayers = 1;
        targetInfo.mipLevels = 1;

        agfxTexture* target = agfxTextureCreate(device, &targetInfo);
        if (!target) {
            return false;
        }

        agfxTextureViewCreateInfo uavInfo{};
        uavInfo.texture = target;
        uavInfo.format = kRtFormat;
        uavInfo.type = AGFX_TEXTURE_TYPE_2D;
        uavInfo.baseMipLevel = 0;
        uavInfo.mipLevelCount = 1;
        uavInfo.baseArrayLayer = 0;
        uavInfo.arrayLayerCount = 1;
        uavInfo.writeable = 1;

        agfxTextureView* uav = agfxTextureViewCreate(device, &uavInfo);
        if (!uav) {
            agfxTextureDestroy(device, target);
            return false;
        }

        agfxShaderModule* module =
            CreateShaderModule(device, shader, entryPoint, AGFX_SHADER_MODULE_TYPE_COMPUTE);

        agfxComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.name = "rt trace";
        pipelineInfo.computeShader = module;
        pipelineInfo.groupSizeX = kRtGroupSize;
        pipelineInfo.groupSizeY = kRtGroupSize;
        pipelineInfo.groupSizeZ = 1;

        agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
        agfxShaderModuleDestroy(device, module);
        if (!pipeline) {
            agfxTextureViewDestroy(device, uav);
            agfxTextureDestroy(device, target);
            return false;
        }

        RtPushConstants constants{};
        constants.tlas = (uint32_t)agfxAccelerationStructureGetHandle(tlas);
        constants.output = (uint32_t)agfxTextureViewGetHandle(uav);
        constants.width = kRtWidth;
        constants.height = kRtHeight;

        agfxDeviceMakeResourcesResident(device);

        gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
            agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                            AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                            AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

            agfxComputePass* pass = agfxComputePassBegin(cmd, "rt trace");
            agfxComputePassSetPipeline(pass, pipeline);
            agfxComputePassPushConstants(pass, &constants, sizeof(constants));
            agfxComputePassDispatch(pass, kRtGroups, kRtGroups, 1);
            agfxComputePassEnd(pass);
        });

        const bool readOk =
            ReadbackTexture2D(device, gpu.Queue(), target, kRtWidth, kRtHeight, kRtFormat,
                              AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage);

        agfxComputePipelineDestroy(device, pipeline);
        agfxTextureViewDestroy(device, uav);
        agfxTextureDestroy(device, target);
        return readOk;
    }
} // namespace agfxtest
