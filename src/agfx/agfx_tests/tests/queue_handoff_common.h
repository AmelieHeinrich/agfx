/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the cross-queue handoff tests ("copy queue to render queue", "compute
// queue to render queue").
//
// Both are the same three-submission shape and differ only in who produces the texture: the
// graphics queue transitions it into the producer's state and signals, a second queue (transfer or
// compute) writes it and signals, and the graphics queue waits on that and samples the result in a
// render pass. The queues are ordered purely by GPU-side agfxCommandQueueSignal/Wait on one shared
// fence — the CPU blocks once, at the very end — so nothing is accidentally serialized by a CPU
// stall the way a fence.Wait() between steps would do.
//
// What they pin down: that work recorded on a non-graphics queue is visible to a later graphics
// submission that waited on its fence. Drop the cross-queue wait and the draw samples a texture the
// producer has not landed in yet, so the golden comes back cleared or torn.
//
// Only the C and C++ flavors exist. ez owns its own graphics queue and submits frames internally,
// with no way to interleave a submission from a second queue into that ordering — the handoff these
// tests are about is not expressible through it, so neither test registers an Ez case.
//
// Same reasoning as raster_common.h: routing both producers and both flavors through one entry
// point makes "the four combinations agree" a property of the code rather than something kept in
// sync by hand across four copies of ~250 lines.

#include "../test_gpu.h"

namespace agfxtest
{
    constexpr uint32_t kHandoffWidth = 128;
    constexpr uint32_t kHandoffHeight = 128;
    constexpr agfxTextureFormat kHandoffFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    /// @brief Which queue produces the texture the render queue then samples.
    enum class QueueProducer
    {
        Copy,    ///< AGFX_COMMAND_QUEUE_TYPE_TRANSFER: a texture-to-texture copy of a seeded source.
        Compute, ///< AGFX_COMMAND_QUEUE_TYPE_COMPUTE: a dispatch of texture_ops.hlsl:main_write_cs.
    };

    /// @brief Runs the handoff for one producer through one API flavor and writes the render
    /// queue's output into `outImage`. Returns false if device creation, shader compilation or
    /// readback failed; the caller should AGFX_EXPECT on it before comparing against a golden.
    bool RenderQueueHandoff(TestApi api, QueueProducer producer, Image& outImage);
} // namespace agfxtest
