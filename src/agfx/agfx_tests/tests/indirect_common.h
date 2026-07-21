/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the indirect (GPU-driven submission) tests: draw, draw indexed, draw mesh,
// draw task+mesh, and dispatch, each in a one-command and a three-command flavor.
//
// Same reasoning as raster_common.h and mesh_common.h, with more force behind it. Every indirect
// test runs the identical seven-step sequence -- zero the count, barrier the bundle to
// UNORDERED_ACCESS, dispatch a producer that appends commands, UAV-barrier, prepare, barrier to
// INDIRECT_ARGUMENT, replay -- and differs only in which bundle type it asks for and how many
// commands it appends. Nine tests times three API flavors is 27 copies of that sequence, and the
// sequence is exactly the part that is easy to get subtly wrong, so it lives here once.
//
// Two things make centralizing this more than a convenience:
//
// 1. Prepare and Execute must be handed *identical* agfxIndirectBundleExecuteInfo contents. Metal
//    bakes the push constants into each pre-encoded ICB command at prepare time and cannot patch
//    them at execute time, while D3D12 only reads them at execute time -- so a host that fills them
//    on one and not the other passes on Windows and renders black on macOS. ExecuteInfo() below is
//    the single constructor both calls go through, so they cannot drift.
//
// 2. The consumer render pipeline must set supportsIndirect. Without it Metal cannot legally
//    reference the PSO from an ICB and faults the GPU rather than reporting a validation error;
//    D3D12 has no such requirement, so forgetting it is invisible on Windows.
//
// Indirect support is optional, so RenderIndirect reports an unsupporting device as
// IndirectResult::Unsupported and the caller skips -- as mesh_common.h does for mesh shading. The
// mesh-bundle kinds require *both* capabilities.

#include "../test_gpu.h"

namespace agfxtest
{
    constexpr uint32_t kIndirectWidth = 128;
    constexpr uint32_t kIndirectHeight = 128;
    constexpr agfxTextureFormat kIndirectFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    /// @brief Columns in the scene, and so the largest command count any test asks for. Matches
    /// AGFX_TEST_INDIRECT_COLUMNS in data/shaders/tests/indirect.hlsl.
    constexpr uint32_t kIndirectColumns = 3;

    /// @brief Which bundle type is under test. DrawMesh and DrawTaskMesh both build a
    /// DRAW_MESH bundle and differ only in whether the consumer pipeline carries a task shader.
    enum class IndirectKind
    {
        Draw,
        DrawIndexed,
        DrawMesh,
        DrawTaskMesh,
        Dispatch,
    };

    /// @brief Mirrors IndirectPushConstants in data/shaders/tests/indirect.hlsl. Field order and
    /// padding must match exactly -- this is memcpy'd into the 128-byte push-constant block, and
    /// into agfxIndirectBundleExecuteInfo::pushConstants.
    ///
    /// Producers and consumers share one block because AGFX_PUSH_CONSTANTS is fixed to register(b0);
    /// each side leaves the other's fields zeroed.
    struct IndirectConstants
    {
        float tint[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        uint64_t bundleHandle = 0;
        uint32_t commandCount = 0;
        uint32_t vertices = 0;
        uint32_t destination = 0;
        uint32_t width = kIndirectWidth;
        uint32_t height = kIndirectHeight;
        uint32_t padding0 = 0;
    };
    static_assert(sizeof(IndirectConstants) == 48, "IndirectConstants must match the HLSL layout");

    /// @brief What one indirect test varies.
    struct IndirectState
    {
        IndirectKind kind = IndirectKind::Draw;
        /// @brief Commands the producer appends, 1..kIndirectColumns. The producer always dispatches
        /// kIndirectColumns threads and bounds itself against this, so a producer that ignored the
        /// bound would append three commands where the test asked for one.
        uint32_t commandCount = 1;
    };

    enum class IndirectResult
    {
        Ok,
        Unsupported, ///< The device reports no multi-draw-indirect (or mesh) support; caller skips.
        Failed,      ///< Device creation, shader compilation, pipeline creation or readback failed.
    };

    /// @brief Runs one IndirectState through the given API flavor into `outImage`.
    IndirectResult RenderIndirect(TestApi api, const IndirectState& state, Image& outImage);

    /// @brief True if a headless device on this machine supports what `kind` needs. Used by the
    /// tests to decide whether to skip before doing any other work.
    bool DeviceSupportsIndirect(IndirectKind kind);
} // namespace agfxtest
