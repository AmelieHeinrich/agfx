/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy queue to render queue".
//
// A transfer queue copies a seeded texture, the graphics queue waits on its fence and samples the
// result in a render pass, so the golden is the seeded pattern round-tripped through a copy queue.
// Drop the cross-queue wait and the draw races the copy: the golden comes back cleared or torn.
// See queue_handoff_common.h for the full submission shape and for why there is no Ez flavor.

#include "queue_handoff_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "copy_queue_to_render_queue.png";
} // namespace

AGFX_TEST_TEXTURE(CopyQueueToRenderQueue, C, kHandoffWidth, kHandoffHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderQueueHandoff(TestApi::C, QueueProducer::Copy, image), "handoff failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(CopyQueueToRenderQueue, Cpp, kHandoffWidth, kHandoffHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderQueueHandoff(TestApi::Cpp, QueueProducer::Copy, image), "handoff failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
