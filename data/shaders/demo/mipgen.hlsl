/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct MipGenPushConstants {
    ResourceHandle srcTex;
    ResourceHandle dstTex;
    uint dstWidth;
    uint dstHeight;
};

AGFX_PUSH_CONSTANTS(MipGenPushConstants, g_Constants);

[numthreads(8, 8, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.dstWidth || id.y >= g_Constants.dstHeight)
        return;

    AGFXTexture2D<float4> src = AGFXTexture2D<float4>::Create(g_Constants.srcTex);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.dstTex);
    Texture2D<float4> srcTex = src.Resource();

    int2 srcCoord = int2(id.xy) * 2;
    float4 c0 = srcTex.Load(int3(srcCoord + int2(0, 0), 0));
    float4 c1 = srcTex.Load(int3(srcCoord + int2(1, 0), 0));
    float4 c2 = srcTex.Load(int3(srcCoord + int2(0, 1), 0));
    float4 c3 = srcTex.Load(int3(srcCoord + int2(1, 1), 0));
    float4 avg = (c0 + c1 + c2 + c3) * 0.25f;

    dst.Store(int2(id.xy), avg);
}
