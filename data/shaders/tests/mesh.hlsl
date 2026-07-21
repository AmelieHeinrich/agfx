//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the mesh-shading counterpart to raster.hlsl's vertex/fragment pair, behind the
// "draw mesh shader" and "draw task + mesh shader" tests.
//
// Both entry points emit the *same geometry raster.hlsl's vertex shader does* -- two triangles on
// opposite halves of the target, with the same per-corner colors. That is deliberate: it means a
// mesh pipeline's golden can be eyeballed against the classic pipeline's, and it keeps the
// difference between the two paths down to how the vertices get there rather than to what they are.
//
// Geometry is a function of SV_GroupID, not a flat table: group 0 emits the left triangle and group
// 1 the right, so a backend that ignores the group ID stacks both triangles in the same place and a
// backend that dispatches the wrong group count drops one. Neither can be mistaken for a pass, and
// neither would be caught if one group emitted everything.

#include "data/shaders/agfx.h"

struct MeshPushConstants
{
    float4 tint;      // Multiplied into every vertex color. The task path uses it to prove that push
                      // constants reach the amplification stage, not just the mesh stage.
    uint groupCount;  // Triangles the task shader should amplify to. Unused by the mesh-only path.
    uint padding0;
    uint padding1;
    uint padding2;
};

AGFX_PUSH_CONSTANTS(MeshPushConstants, g_Constants);

struct ms_out
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

// The payload the task stage hands to the mesh stage: one color index per mesh group. One field,
// deliberately -- this test is about the payload arriving and being per-group correct, not about
// payload layout rules.
struct MeshPayload
{
    uint colorIndex[2];
};

// Corner positions and colors for both triangles, indexed [group][corner]. The left triangle is
// wound counter-clockwise and the right clockwise, matching raster.hlsl -- culling is off in these
// tests, but keeping the winding identical means a mesh golden and a classic golden differ only if
// the mesh path itself is wrong.
static const float2 kPositions[2][3] = {
    { float2(-0.9f, -0.8f), float2(-0.1f, -0.8f), float2(-0.5f,  0.8f) },
    { float2( 0.1f, -0.8f), float2( 0.5f,  0.8f), float2( 0.9f, -0.8f) }
};

static const float3 kColors[2][3] = {
    { float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f) },
    { float3(1.0f, 1.0f, 0.0f), float3(0.0f, 1.0f, 1.0f), float3(1.0f, 0.0f, 1.0f) }
};

// Shared by both mesh entry points so the two tests cannot drift apart in what they emit.
//
// Position and color are selected by *separate* indices on purpose. The obvious design -- one index
// picking both -- makes the task payload untestable: if the payload only decided which group draws
// which whole triangle, reversing it would swap two groups' work and produce a pixel-identical
// image, since each triangle still lands at its own position. (That is not hypothetical; the first
// version of this test did exactly that and its two goldens came out byte-identical.) Splitting the
// indices lets the task path keep position tied to SV_GroupID while routing *color* through the
// payload, so a dropped payload changes the image instead of being invisible.
void EmitTriangle(uint positionIndex, uint colorIndex, uint threadID,
                  out vertices ms_out verts[3], out indices uint3 tris[1])
{
    const uint pos = min(positionIndex, 1u);
    const uint col = min(colorIndex, 1u);

    verts[threadID].position = float4(kPositions[pos][threadID], 0.0f, 1.0f);
    verts[threadID].color = kColors[col][threadID] * g_Constants.tint.rgb;

    if (threadID == 0) {
        tris[0] = uint3(0, 1, 2);
    }
}

// One group per triangle, three threads per group -- one thread per vertex, which is the shape a
// real mesh shader would use and exercises SV_GroupThreadID rather than letting one thread write
// all three vertices.
[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main_ms(uint groupID : SV_GroupID, uint threadID : SV_GroupThreadID,
             out vertices ms_out verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    EmitTriangle(groupID, groupID, threadID, verts, tris);
}

// The task (amplification) stage. One thread, which fans out to groupCount mesh groups. The payload
// carries each mesh group's color index explicitly rather than letting the mesh stage re-derive it
// from SV_GroupID: that is what makes the test prove the payload actually crossed the stage
// boundary. Reversed on purpose (group 0 gets group 1's colors) so a backend that ignores the
// payload and falls back to the group ID renders each triangle in its *own* colors -- which is
// exactly the mesh-only test's image, so the two goldens differing is itself the assertion.
groupshared MeshPayload s_payload;

[numthreads(1, 1, 1)]
void main_as()
{
    s_payload.colorIndex[0] = 1;
    s_payload.colorIndex[1] = 0;

    DispatchMesh(g_Constants.groupCount, 1, 1, s_payload);
}

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main_ms_payload(uint groupID : SV_GroupID, uint threadID : SV_GroupThreadID,
                     in payload MeshPayload payload,
                     out vertices ms_out verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    // Position follows the group, color follows the payload -- see EmitTriangle.
    EmitTriangle(groupID, payload.colorIndex[min(groupID, 1u)], threadID, verts, tris);
}

float4 main_ps(ms_out input) : SV_Target0
{
    return float4(input.color, 1.0f);
}
