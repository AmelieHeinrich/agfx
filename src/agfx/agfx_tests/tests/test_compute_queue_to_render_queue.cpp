/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute queue to render queue".
//
// The compute-producer half of the cross-queue handoff pair: a dedicated compute queue dispatches
// texture_ops.hlsl:main_write_cs into a storage texture, and the graphics queue waits on its fence
// and samples the result in a render pass. Where the copy-queue variant covers a transfer-queue
// copy becoming visible to a draw, this covers UAV writes from an async compute queue doing the
// same — a different queue class, and on both backends a different barrier path out of
// UNORDERED_ACCESS rather than out of COPY_DEST.
//
// See queue_handoff_common.h for the full submission shape and for why there is no Ez flavor.

#include "queue_handoff_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "compute_queue_to_render_queue.png";
} // namespace

AGFX_TEST_TEXTURE(ComputeQueueToRenderQueue, C, kHandoffWidth, kHandoffHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderQueueHandoff(TestApi::C, QueueProducer::Compute, image), "handoff failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(ComputeQueueToRenderQueue, Cpp, kHandoffWidth, kHandoffHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderQueueHandoff(TestApi::Cpp, QueueProducer::Compute, image), "handoff failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
