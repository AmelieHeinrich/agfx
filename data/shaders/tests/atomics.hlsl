//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: every Interlocked* op AGFXRWByteAddressBuffer exposes, each hammered by the whole
// dispatch at one shared address. The results are order-independent by construction (sum, and, or,
// xor of a symmetric set, min, max), so the golden is deterministic even though the execution order
// is not — a broken atomic shows up as a lost update, not as flakiness.

#include "data/shaders/agfx.h"

struct AtomicsPushConstants
{
    ResourceHandle rwBuffer; // AGFX_BUFFER_VIEW_TYPE_RAW, writeable
    uint threadCount;        // Total threads participating.
    uint padding0;
    uint padding1;
};

AGFX_PUSH_CONSTANTS(AtomicsPushConstants, g_Constants);

// Slot layout, in 4-byte words. Slot 0 is seeded by the host; the rest start at their identity.
#define SLOT_ADD 0
#define SLOT_AND 1
#define SLOT_OR  2
#define SLOT_XOR 3
#define SLOT_MIN 4
#define SLOT_MAX 5
#define SLOT_EXCHANGE_COUNT 6

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWByteAddressBuffer dst = AGFXRWByteAddressBuffer::Create(g_Constants.rwBuffer);

    const uint index = id.x;
    if (index >= g_Constants.threadCount) {
        return;
    }

    uint original = 0;

    // Sum of 1..threadCount — a dropped update is immediately visible in the total.
    dst.InterlockedAdd(SLOT_ADD * 4u, index + 1u, original);

    // AND clears one distinct bit per thread; OR sets one; both converge regardless of order.
    dst.InterlockedAnd(SLOT_AND * 4u, ~(1u << (index & 31u)), original);
    dst.InterlockedOr(SLOT_OR * 4u, 1u << (index & 31u), original);

    // XOR over a set where every value appears an even number of times must cancel out to the seed.
    dst.InterlockedXor(SLOT_XOR * 4u, 1u << (index & 15u), original);

    dst.InterlockedMin(SLOT_MIN * 4u, index + 1u, original);
    dst.InterlockedMax(SLOT_MAX * 4u, index + 1u, original);

    // CompareExchange: exactly one thread may win the transition from 0 to 1, and that winner is
    // the only one allowed to bump the counter.
    uint previous = 0;
    dst.InterlockedCompareExchange(SLOT_EXCHANGE_COUNT * 4u + 4u, 0u, 1u, previous);
    if (previous == 0u) {
        dst.InterlockedAdd(SLOT_EXCHANGE_COUNT * 4u, 1u, original);
    }
}
