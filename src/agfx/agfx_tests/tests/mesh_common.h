/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the mesh-shading tests (mesh-only, and task + mesh).
//
// Same reasoning as raster_common.h: both tests build the same render target, render pass and
// fragment stage and differ only in whether the pipeline carries a task shader, so the setup lives
// here once and each test file holds only the state under test and its golden.
//
// Unlike the raster tests, these can legitimately not run: mesh shading is optional. RenderMesh
// reports that as MeshResult::Unsupported so the caller can skip rather than fail -- a hard failure
// on a device without mesh shaders would be a false alarm, and silently passing would be worse.

#include "../test_gpu.h"

namespace agfxtest
{
    constexpr uint32_t kMeshWidth = 128;
    constexpr uint32_t kMeshHeight = 128;
    constexpr agfxTextureFormat kMeshFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    /// @brief Mesh groups dispatched, one triangle each. Both tests emit the same two triangles;
    /// see data/shaders/tests/mesh.hlsl for why the geometry is per-group rather than tabulated.
    constexpr uint32_t kMeshGroupCount = 2;

    /// @brief Mirrors MeshPushConstants in data/shaders/tests/mesh.hlsl. Field order and padding
    /// must match exactly -- this is memcpy'd into the push-constant block.
    struct MeshConstants
    {
        float tint[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        uint32_t groupCount = kMeshGroupCount;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
        uint32_t padding2 = 0;
    };

    /// @brief What one mesh test varies: whether the pipeline amplifies through a task shader.
    struct MeshState
    {
        bool useTaskShader = false;
        MeshConstants constants{};
    };

    enum class MeshResult
    {
        Ok,
        Unsupported, ///< The device reports no mesh shader support; the caller should skip.
        Failed,      ///< Device creation, shader compilation, pipeline creation or readback failed.
    };

    /// @brief Renders one MeshState through the given API flavor into `outImage`.
    MeshResult RenderMesh(TestApi api, const MeshState& state, Image& outImage);

    /// @brief True if a headless device on this machine reports mesh shader support. Used by the
    /// tests to decide whether to skip before doing any other work.
    bool DeviceSupportsMeshShaders();
} // namespace agfxtest
