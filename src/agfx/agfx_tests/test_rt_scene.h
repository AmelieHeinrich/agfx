/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include "test_gpu.h"

#include <agfx/agfx.h>

#include <vector>

// The scene scaffolding every raytracing test shares.
//
// The RT tests differ almost entirely in how an acceleration structure is *produced* — built,
// rebuilt from a reused scratch buffer, updated in place, compacted, copied, instanced — and barely
// at all in the geometry that goes into it or in what tracing it should yield. Keeping the geometry,
// the buffer creation, the build submission and the trace-to-image step here means a test body is
// only the operation under test plus a golden comparison, and means a change to (say) how vertices
// are laid out cannot drift between seven files.
//
// Everything lives in world space inside x,y in [-1,1] around z = 0, which is exactly the
// orthographic frustum data/shaders/tests/raytracing.hlsl traces through.

namespace agfxtest
{
    /// @brief The dimensions every RT golden is rendered at. 64x64 keeps readback cheap while still
    /// being large enough that a triangle's edges span enough texels for a misplaced build to show.
    constexpr uint32_t kRtWidth = 64;
    constexpr uint32_t kRtHeight = 64;
    constexpr agfxTextureFormat kRtFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kRtGroupSize = 8; // Matches [numthreads(8,8,1)] in raytracing.hlsl.
    constexpr uint32_t kRtGroups = (kRtWidth + kRtGroupSize - 1) / kRtGroupSize;

    /// @brief Mirrors RaytracingPushConstants in data/shaders/tests/raytracing.hlsl.
    struct RtPushConstants
    {
        uint32_t tlas;
        uint32_t output;
        uint32_t width;
        uint32_t height;
        uint32_t padding0;
        uint32_t padding1;
        uint32_t padding2;
        uint32_t padding3;
    };

    // --- Geometry ---------------------------------------------------------------------------

    /// @brief One large triangle centred on the origin, as position-only float3 vertices.
    /// Deliberately asymmetric in y so a build that flips or transposes the geometry is visible.
    const std::vector<float>& RtTriangleVertices();
    const std::vector<uint32_t>& RtTriangleIndices();

    /// @brief Three smaller triangles at distinct z depths, used by the multi-geometry and
    /// multi-primitive tests. Separating them in z means the depth channel of the golden orders
    /// them, so a BLAS that merges or reorders primitives changes the image rather than just its
    /// silhouette.
    const std::vector<float>& RtMultiTriangleVertices();
    const std::vector<uint32_t>& RtMultiTriangleIndices();

    /// @brief One AABB as the (min, max) float3 pair D3D12 and Metal both expect, covering the
    /// middle of the frustum.
    const std::vector<float>& RtAabbData();

    /// @brief Two AABBs side by side, so a procedural BLAS that collapses its primitives differs
    /// from one that keeps them.
    const std::vector<float>& RtMultiAabbData();

    /// @brief The identity 3x4 row-major transform agfxAccelerationStructureInstance wants.
    void RtIdentityTransform(float outTransform[12]);

    /// @brief An identity transform translated by (x, y, z); the instance tests place copies of one
    /// BLAS with these so each instance occupies its own part of the image.
    void RtTranslationTransform(float outTransform[12], float x, float y, float z);

    // --- Buffers ----------------------------------------------------------------------------

    /// @brief Creates a GPU buffer seeded with `data` and left in a state an AS build can read.
    ///
    /// AS build inputs need no dedicated usage flag: both backends take the buffer's raw GPU
    /// address, so SHADER_READ is what a vertex/index/AABB buffer is created with.
    agfxBuffer* RtCreateInputBuffer(GpuFixture& gpu, const void* data, uint64_t size, uint32_t stride,
                                    const char* name);

    /// @brief Creates a scratch buffer of at least `size` bytes for AS builds.
    agfxBuffer* RtCreateScratchBuffer(agfxDevice* device, uint64_t size, const char* name);

    // --- Acceleration structures ------------------------------------------------------------

    /// @brief A BLAS plus the buffers backing it, torn down in reverse order on destruction.
    ///
    /// The buffers have to outlive the BLAS handle across the build submission, and forgetting one
    /// leaks silently rather than failing, so ownership is bundled rather than left to each test.
    struct RtBlas
    {
        agfxDevice* device = nullptr;
        agfxAccelerationStructure* blas = nullptr;
        agfxBuffer* vertexBuffer = nullptr;
        agfxBuffer* indexBuffer = nullptr;
        agfxBuffer* aabbBuffer = nullptr;

        RtBlas() = default;
        ~RtBlas();

        RtBlas(const RtBlas&) = delete;
        RtBlas& operator=(const RtBlas&) = delete;
        RtBlas(RtBlas&& other) noexcept { *this = static_cast<RtBlas&&>(other); }
        RtBlas& operator=(RtBlas&& other) noexcept;

        bool Valid() const { return blas != nullptr; }
    };

    /// @brief Creates and builds a triangle BLAS from `RtTriangleVertices`, submitting the build and
    /// waiting for it. `opaque` sets the geometry's opaque flag; `allowUpdate` makes it refittable.
    RtBlas RtBuildTriangleBlas(GpuFixture& gpu, bool opaque = true, bool allowUpdate = false);

    /// @brief As RtBuildTriangleBlas but with the three-triangle geometry.
    RtBlas RtBuildMultiTriangleBlas(GpuFixture& gpu, bool opaque = true, bool allowUpdate = false);

    /// @brief Creates and builds a procedural BLAS over `aabbCount` AABBs from `aabbs`.
    RtBlas RtBuildAabbBlas(GpuFixture& gpu, const std::vector<float>& aabbs, uint32_t aabbCount,
                           bool opaque = true);

    /// @brief Submits a build for `as` on a transient scratch buffer, barriers it for shader reads,
    /// and waits. The tests that own their scratch buffer (reuse, offset) do not use this; it is for
    /// the ones where the build itself is only setup for the operation under test.
    bool RtBuildAndWait(GpuFixture& gpu, agfxAccelerationStructure* as);

    /// @brief A TLAS plus its scratch buffer.
    struct RtTlas
    {
        agfxDevice* device = nullptr;
        agfxAccelerationStructure* tlas = nullptr;

        RtTlas() = default;
        ~RtTlas();

        RtTlas(const RtTlas&) = delete;
        RtTlas& operator=(const RtTlas&) = delete;
        RtTlas(RtTlas&& other) noexcept { *this = static_cast<RtTlas&&>(other); }
        RtTlas& operator=(RtTlas&& other) noexcept;

        bool Valid() const { return tlas != nullptr; }
    };

    /// @brief Creates a TLAS over `instances`, builds it, and waits. `maxInstanceCount` defaults to
    /// the instance count; the instance tests pass a larger value to check overallocation is fine.
    RtTlas RtBuildTlas(GpuFixture& gpu, const std::vector<agfxAccelerationStructureInstance>& instances,
                       uint32_t maxInstanceCount = 0);

    /// @brief The common case: one BLAS at the origin, wrapped in a single-instance TLAS.
    RtTlas RtBuildSingleInstanceTlas(GpuFixture& gpu, agfxAccelerationStructure* blas,
                                     uint32_t userID = 0, bool opaque = true);

    // --- Tracing ----------------------------------------------------------------------------

    /// @brief Compiles `entryPoint` from raytracing.hlsl, traces `tlas` into a 64x64 RGBA8 texture,
    /// and reads it back into `outImage`. Returns false if any step failed.
    ///
    /// This is the step that turns "the AS is not null" into an actual assertion: nothing about a
    /// built acceleration structure is observable from the host, so every RT test's real check is
    /// the image this produces.
    bool RtTraceToImage(GpuFixture& gpu, agfxAccelerationStructure* tlas, const char* entryPoint,
                        Image& outImage);
} // namespace agfxtest
