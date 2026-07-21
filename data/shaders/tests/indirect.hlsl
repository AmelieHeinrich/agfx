//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the producer/consumer pair behind the indirect (GPU-driven submission) tests --
// draw indirect, draw indexed indirect, draw mesh indirect, draw task+mesh indirect, and dispatch
// indirect, each in a one-command and a three-command flavor.
//
// Every indirect test is two shaders, not one. A *producer* compute shader appends commands into an
// indirect bundle (the thing under test: the host never writes a command), and a *consumer*
// vertex/mesh/compute shader runs once per replayed command. That split is the whole point: if the
// host filled the commands buffer itself the test would only prove that ExecuteIndirect reads
// memory, not that a shader-authored bundle round-trips.
//
// The scene is three columns. Command i draws column i in its own color, so the image encodes both
// *how many* commands replayed and *which drawID each one carried*:
//
//   - one command    -> only the left column, red.
//   - three commands -> red, green, blue left to right.
//
// That makes the two goldens for a given bundle type differ structurally rather than subtly, and it
// makes the failure modes distinguishable by eye: a dropped count clamp draws too many columns, a
// zeroed count draws none, and a drawID that never reaches the consumer paints all three columns
// red (or stacks them all in the left slot) instead of producing the gradient.
//
// drawID reaches the consumer through AGFX_DRAW_ID(), which is the b1 root constant the backends
// patch per command -- see the drawID note in agfx.h. Nothing else distinguishes one replayed
// command from another, which is exactly why a broken drawID cannot hide here.

#include "data/shaders/agfx.h"

// ---------------------------------------------------------------------------------------------
// Shared scene definition
// ---------------------------------------------------------------------------------------------

#define AGFX_TEST_INDIRECT_COLUMNS 3

// Column c spans x in [-0.85 + 0.6c, -0.35 + 0.6c], y in [-0.5, 0.5]: three non-touching bands with
// clear background between them, so a column drawn one slot over never overlaps the one that should
// have been there and cannot masquerade as a pass.
float2 IndirectColumnCorner(uint column, uint corner)
{
    const float x0 = -0.85f + 0.6f * (float)column;
    const float x1 = x0 + 0.5f;
    const float2 corners[4] = {
        float2(x0, -0.5f), // 0: bottom left
        float2(x1, -0.5f), // 1: bottom right
        float2(x1,  0.5f), // 2: top right
        float2(x0,  0.5f)  // 3: top left
    };
    return corners[min(corner, 3u)];
}

// One primary per command index. Chosen so any two columns differ in every channel: a swapped pair
// is as visible as a missing one.
float3 IndirectDrawColor(uint drawID)
{
    const float3 colors[AGFX_TEST_INDIRECT_COLUMNS] = {
        float3(1.0f, 0.0f, 0.0f), // command 0: red
        float3(0.0f, 1.0f, 0.0f), // command 1: green
        float3(0.0f, 0.0f, 1.0f)  // command 2: blue
    };
    return colors[min(drawID, (uint)(AGFX_TEST_INDIRECT_COLUMNS - 1))];
}

// ---------------------------------------------------------------------------------------------
// Producers: the compute shaders that author the bundle
// ---------------------------------------------------------------------------------------------

// AGFX_PUSH_CONSTANTS is fixed to register(b0), so every entry point in this file shares one block
// rather than declaring a per-stage struct. Producers read the top half, consumers the bottom; each
// leaves the other's fields zeroed. Mirrors IndirectConstants in indirect_common.h -- field order
// and padding must match exactly, this is memcpy'd into the 128-byte push-constant block.
struct IndirectPushConstants
{
    // --- consumer ---
    float4 tint;      // Multiplied into every column's color.
    // --- producer ---
    uint64_t bundleHandle; // Packed commands/count handles; see agfxIndirectBundleGetHandle.
    uint commandCount;     // Commands to append. The count buffer is what bounds the replay.
    // --- consumer ---
    uint vertices;    // Structured buffer of column corners; indexed draw path only.
    uint destination; // Storage texture handle; dispatch path only.
    uint width;
    uint height;
    uint padding0;
};

AGFX_PUSH_CONSTANTS(IndirectPushConstants, g_Constants);
AGFX_DECLARE_DRAW_ID();

// One thread per command, all appending concurrently through the bundle's InterlockedAdd. The
// dispatch is deliberately wider than commandCount (a fixed AGFX_TEST_INDIRECT_COLUMNS threads) so
// the early-out is doing real work: a producer that ignored the bound would append three commands
// where the test asked for one, and the one-command golden would catch it.
//
// Because the append order across threads is not deterministic, drawID is written explicitly from
// the thread index rather than inferred from the slot the atomic handed back. Command *slots* may
// come out in any order; the drawID each command carries may not.

[numthreads(AGFX_TEST_INDIRECT_COLUMNS, 1, 1)]
void main_produce_draw_cs(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= g_Constants.commandCount) {
        return;
    }

    AGFXIndirectDrawBundle bundle = AGFXIndirectDrawBundle::Create(g_Constants.bundleHandle);
    // firstVertex stays 0: the consumer derives its column from drawID, not from a vertex base.
    // SV_VertexID's treatment of a non-zero base differs between the backends, and pinning that
    // down is not what this test is for.
    bundle.Draw(/*commandOffset*/0, /*countIndex*/0, /*drawId*/dtid.x,
                /*vertexCount*/6, /*instanceCount*/1, /*firstVertex*/0, /*firstInstance*/0);
}

[numthreads(AGFX_TEST_INDIRECT_COLUMNS, 1, 1)]
void main_produce_draw_indexed_cs(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= g_Constants.commandCount) {
        return;
    }

    AGFXIndirectDrawIndexedBundle bundle = AGFXIndirectDrawIndexedBundle::Create(g_Constants.bundleHandle);
    // firstIndex *is* per-command here (6 indices per column), so the indexed path tests something
    // the non-indexed one cannot: that a per-command index offset survives the round trip. Position
    // then comes from the index buffer and color from drawID, and the two can fail independently --
    // a bad firstIndex shifts a column, a bad drawID recolors one.
    bundle.DrawIndexed(/*commandOffset*/0, /*countIndex*/0, /*drawId*/dtid.x,
                       /*indexCount*/6, /*instanceCount*/1, /*firstIndex*/dtid.x * 6,
                       /*vertexOffset*/0, /*firstInstance*/0);
}

[numthreads(AGFX_TEST_INDIRECT_COLUMNS, 1, 1)]
void main_produce_draw_mesh_cs(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= g_Constants.commandCount) {
        return;
    }

    AGFXIndirectDrawMeshBundle bundle = AGFXIndirectDrawMeshBundle::Create(g_Constants.bundleHandle);
    // One group per command for both the mesh-only and the task+mesh pipeline. With a task shader
    // the group count is the *task* group count, and the task stage amplifies 1 -> 1; keeping it at
    // one either way means the two goldens differ only if the task stage itself misbehaves.
    bundle.DrawMesh(/*commandOffset*/0, /*countIndex*/0, /*drawId*/dtid.x,
                    /*groupSizeX*/1, /*groupSizeY*/1, /*groupSizeZ*/1);
}

[numthreads(1, 1, 1)]
void main_produce_dispatch_cs()
{
    AGFXIndirectDispatchBundle bundle = AGFXIndirectDispatchBundle::Create(g_Constants.bundleHandle);
    // agfxDispatchCommand carries no drawID (see agfx.h), so unlike the draw bundles there is no
    // per-command identity to test -- only that the group counts themselves round trip. The
    // consumer covers the whole target, so a group count that arrives short leaves a black band.
    // Group counts cover the whole target at the consumer's [numthreads(8,8,1)]. They are computed
    // here rather than passed in so they are genuinely GPU-authored: a host that wrote the command
    // itself would prove nothing about the bundle.
    bundle.Dispatch(/*commandOffset*/0, /*countIndex*/0,
                    (g_Constants.width + 7) / 8, (g_Constants.height + 7) / 8, 1);
}

// ---------------------------------------------------------------------------------------------
// Consumers: what one replayed command actually runs
// ---------------------------------------------------------------------------------------------

// The tint is not decoration. On Metal the push constants are baked into every pre-encoded ICB
// command at *prepare* time and cannot be patched at execute time, so a host that fills
// agfxIndirectBundleExecuteInfo::pushConstants only on the execute info hands the GPU a zeroed
// block -- which on D3D12 works anyway. A non-white tint makes that host-side asymmetry show up as
// black columns in the golden instead of passing on one backend and silently breaking on the other.

struct vs_out
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

// Vertex corners for one quad, two triangles: 0,1,2 and 0,2,3.
static const uint kQuadCorners[6] = {0, 1, 2, 0, 2, 3};

vs_out main_vs(uint vertexID : SV_VertexID)
{
    const uint drawID = AGFX_DRAW_ID();

    vs_out output;
    output.position = float4(IndirectColumnCorner(drawID, kQuadCorners[min(vertexID, 5u)]), 0.0f, 1.0f);
    output.color = IndirectDrawColor(drawID) * g_Constants.tint.rgb;
    return output;
}

// Mirrors the Vertex struct in indirect_common.cpp: position padded out so the structured view's
// stride is 16 bytes on both sides.
struct IndirectVertex
{
    float2 position;
    float2 padding;
};

// The indexed consumer pulls positions out of a structured buffer through the index buffer, so the
// column it lands in is decided by the per-command firstIndex rather than by drawID. Color still
// comes from drawID -- see main_produce_draw_indexed_cs.
vs_out main_vs_indexed(uint vertexID : SV_VertexID)
{
    AGFXStructuredBuffer<IndirectVertex> vertices =
        AGFXStructuredBuffer<IndirectVertex>::Create((ResourceHandle)g_Constants.vertices);

    vs_out output;
    output.position = float4(vertices.Load(vertexID).position, 0.0f, 1.0f);
    output.color = IndirectDrawColor(AGFX_DRAW_ID()) * g_Constants.tint.rgb;
    return output;
}

float4 main_ps(vs_out input) : SV_Target0
{
    return float4(input.color, 1.0f);
}

// --- Mesh consumers ---------------------------------------------------------------------------

struct ms_out
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

struct IndirectPayload
{
    uint drawID; // Carried explicitly so the task path proves the payload crossed the stage boundary.
};

void EmitColumn(uint drawID, uint threadID, out vertices ms_out verts[4], out indices uint3 tris[2])
{
    verts[threadID].position = float4(IndirectColumnCorner(drawID, threadID), 0.0f, 1.0f);
    verts[threadID].color = IndirectDrawColor(drawID) * g_Constants.tint.rgb;

    if (threadID == 0) {
        tris[0] = uint3(0, 1, 2);
        tris[1] = uint3(0, 2, 3);
    }
}

// Four threads, one per quad corner -- the shape a real mesh shader would use, and it exercises
// SV_GroupThreadID rather than letting one thread write every vertex.
[outputtopology("triangle")]
[numthreads(4, 1, 1)]
void main_ms(uint threadID : SV_GroupThreadID, out vertices ms_out verts[4], out indices uint3 tris[2])
{
    SetMeshOutputCounts(4, 2);
    EmitColumn(AGFX_DRAW_ID(), threadID, verts, tris);
}

// The task stage amplifies 1 -> 1 and forwards drawID through the payload. The forwarding is the
// assertion: drawID has to survive the b1 patch *and* the task->mesh boundary, and routing it
// through the payload means a mesh stage that quietly re-read AGFX_DRAW_ID() itself would not be
// what the test measured.
groupshared IndirectPayload s_payload;

[numthreads(1, 1, 1)]
void main_as()
{
    s_payload.drawID = AGFX_DRAW_ID();
    DispatchMesh(1, 1, 1, s_payload);
}

[outputtopology("triangle")]
[numthreads(4, 1, 1)]
void main_ms_payload(uint threadID : SV_GroupThreadID, in payload IndirectPayload payload,
                     out vertices ms_out verts[4], out indices uint3 tris[2])
{
    SetMeshOutputCounts(4, 2);
    EmitColumn(payload.drawID, threadID, verts, tris);
}

float4 main_ms_ps(ms_out input) : SV_Target0
{
    return float4(input.color, 1.0f);
}

// --- Dispatch consumer ------------------------------------------------------------------------

// Writes the same three-column scene the draw consumers rasterize, so the dispatch golden is
// comparable to theirs by eye. Column comes from the pixel's x, not from a command ID: dispatch
// commands carry no drawID.
[numthreads(8, 8, 1)]
void main_dispatch_cs(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= g_Constants.width || dtid.y >= g_Constants.height) {
        return;
    }

    AGFXRWTexture2D<float4> destination =
        AGFXRWTexture2D<float4>::Create((ResourceHandle)g_Constants.destination);

    // NDC, y down (image space) -- matches the rasterized goldens' orientation.
    const float2 ndc = float2((dtid.x + 0.5f) / g_Constants.width, (dtid.y + 0.5f) / g_Constants.height)
                     * 2.0f - 1.0f;

    float3 color = float3(0.0f, 0.0f, 0.0f);
    for (uint c = 0; c < AGFX_TEST_INDIRECT_COLUMNS; ++c) {
        const float2 bl = IndirectColumnCorner(c, 0);
        const float2 tr = IndirectColumnCorner(c, 2);
        if (ndc.x >= bl.x && ndc.x <= tr.x && ndc.y >= bl.y && ndc.y <= tr.y) {
            color = IndirectDrawColor(c);
        }
    }

    destination.Store(dtid.xy, float4(color, 1.0f));
}
