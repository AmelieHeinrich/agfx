---
name: agfx-writing-bindless-shaders
description: ALWAYS use when writing or modifying HLSL shaders for AGFX, or wiring the agfxShaderCompiler / agfxShaderModule / agfxRenderPipeline / agfxComputePipeline pipeline around them. Trigger for AGFX_PUSH_CONSTANTS, ResourceHandle, AGFXTexture2D/AGFXRWTexture2D/AGFXStructuredBuffer/AGFXByteAddressBuffer/AGFXSampler, ResourceDescriptorHeap/SamplerDescriptorHeap, agfxCompileShader, agfxShaderCompilerOptions, agfxShaderModuleCreate, register(b0)/register(b1), "bindless", "push constants", writing a new .hlsl file for AGFX, mesh/task/compute shader entry points (main_vs/main_ps/main_cs/main_ms/main_as). Do NOT trigger for render pass/attachment authoring in host code — use agfx-render-targets-and-passes. Do NOT trigger for resource-state barriers/fences — use agfx-synchronization. Do NOT trigger for swap chain/present — use agfx-presentation-and-swapchain.
---

# AGFX Bindless HLSL Shaders

## Overview

AGFX shaders are HLSL, compiled with DXC to DXIL (SM 6.6) and then, on macOS, translated to Metal IR via the Metal shader converter (`agfx_shader/agfx_shader_compiler_mac.mm`) using `IRRootSignatureFlagSamplerHeapDirectlyIndexed | IRRootSignatureFlagCBVSRVUAVHeapDirectlyIndexed` — i.e. **fully bindless, direct-indexed heaps**. There is no per-draw descriptor table, no `register(t0, space0)` binding model, and no `Bind*` API on the C side beyond push constants. Every resource a shader touches — textures, buffers, samplers — is accessed by a `ResourceHandle` (a plain `uint` index) pulled out of `ResourceDescriptorHeap`/`SamplerDescriptorHeap` and wrapped in one of the `AGFX*` helper classes declared in `data/shaders/<project>/agfx.h`.

The host side hands shaders these handles two ways: almost always via push constants (`agfxRenderPassPushConstants`/`agfxComputePassPushConstants`, bound at `register(b0)`), or, for structured scene/per-draw constant data, by putting the handle to a constant buffer *inside* the push constants and loading it as an `AGFXStructuredBuffer` in the shader (see `sceneCB` pattern below) rather than a second root CBV.

## Ownership

**Owns:**
- The bindless resource-access pattern: `ResourceHandle`, `AGFXTexture1D/2D/2DArray/3D/Cube<T>`, `AGFXRWTexture1D/2D/3D<T>`, `AGFXStructuredBuffer<T>`/`AGFXRWStructuredBuffer<T>`, `AGFXByteAddressBuffer`/`AGFXRWByteAddressBuffer`, `AGFXSampler`/`AGFXComparisonSampler`, `AGFXRaytracingAccelerationStructure`
- Push constants: `AGFX_PUSH_CONSTANTS(type, name)` at `register(b0)`, and the optional per-draw ID at `register(b1)` (`AGFX_DECLARE_DRAW_ID()`/`AGFX_DRAW_ID()`)
- Entry point / stage conventions (`main_vs`, `main_ps`, `main_cs`, `main_ms`, `main_as`) and matching `agfxShaderModuleType`
- `agfxShaderCompilerOptions`/`agfxShaderCompilerResult` and `agfxCompileShader` — the HLSL → DXIL → (macOS) Metal IR pipeline
- Wiring compiled `agfxShaderModule`s into `agfxRenderPipelineCreateInfo`/`agfxComputePipelineCreateInfo`

**Doesn't own:**
- Render pass/attachment setup the pipeline is later bound and drawn within → `agfx-render-targets-and-passes`
- Barriers needed before a shader can safely read/write a resource (state transitions, UAV hazard barriers) → `agfx-synchronization`
- Swap chain / back buffer acquisition → `agfx-presentation-and-swapchain`

## References

The bindless helper header lives per-project at `data/shaders/<project>/agfx.h` (e.g. `data/shaders/demo/agfx.h`, `data/shaders/game/agfx.h`) — **always `#include` it first** in a new shader and read it before inventing a new resource-access pattern; it is the complete list of what's available. Real shader examples: `data/shaders/demo/gbuffer.hlsl` (vertex+fragment, structured vertex pulling, textures+sampler), `data/shaders/demo/ssao.hlsl` (compute, RW texture output, scene CB), `data/shaders/demo/mipgen.hlsl` (minimal compute), `data/shaders/demo/deferred_lighting.hlsl`, `data/shaders/demo/shadow_depth.hlsl`, `data/shaders/demo/tonemap.hlsl`, `data/shaders/demo/imgui.hlsl`. Host-side compile+load pattern: `agfx_demo/deferred_renderer.cpp`'s `CompileShader` helper, `agfx_demo/ssao.cpp`, `agfx_demo/agfx_mipgen.cpp`. Compiler internals: `agfx_shader/agfx_shader_compiler.h` and `agfx_shader/agfx_shader_compiler_mac.mm`.

## Design Patterns

### Minimal shader skeleton

```hlsl
#include "data/shaders/demo/agfx.h"   // path is project-relative; match the project's existing shaders

struct MyPushConstants {
    ResourceHandle someTex;
    ResourceHandle someSampler;
    ResourceHandle sceneCB;
    float someScalar;
};

AGFX_PUSH_CONSTANTS(MyPushConstants, g_Constants);
```

`AGFX_PUSH_CONSTANTS` expands to `ConstantBuffer<type> name : register(b0)` — this is the *only* resource binding declaration a shader normally needs. Everything else is created inline from a `ResourceHandle` field on that struct.

### Reading resources: create-from-handle, then use

```hlsl
AGFXTexture2D<float4> albedoTex = AGFXTexture2D<float4>::Create(g_Constants.albedoTex);
AGFXSampler samp = AGFXSampler::Create(g_Constants.textureSampler);
float4 c = albedoTex.Sample(samp, uv);

AGFXRWTexture2D<float4> aoOut = AGFXRWTexture2D<float4>::Create(g_Constants.aoTex);
aoOut.Store(int2(id.xy), float4(ao, ao, ao, 1.0f));

AGFXStructuredBuffer<SceneVertex> vertices = AGFXStructuredBuffer<SceneVertex>::Create(g_Constants.vertexBuffer);
SceneVertex v = vertices.Load(vertexID);
```

Pick the wrapper by both dimensionality and read/write need: `AGFXTexture2D<T>` (read-only, sampled) vs `AGFXRWTexture2D<T>` (read/write, `Load`/`Store` only, no filtering) — matching `AGFX_TEXTURE_USAGE_SAMPLED` vs `AGFX_TEXTURE_USAGE_STORAGE` on the host-side `agfxTextureCreateInfo`. Use `AGFXStructuredBuffer<T>` for typed per-element buffer reads (scene constants, vertex pulling) and `AGFXByteAddressBuffer`/`AGFXRWByteAddressBuffer` for raw/untyped access — matching `AGFX_BUFFER_VIEW_TYPE_STRUCTURED` vs `AGFX_BUFFER_VIEW_TYPE_RAW` host-side.

### Scene/per-frame constants: no second CBV — nest a handle in push constants

There is no root-level CBV beyond `b0`'s push constants. To pass a larger, per-frame constant buffer, put its `ResourceHandle` as a field in the push-constant struct and load it as a one-element `AGFXStructuredBuffer` inside the shader:

```hlsl
struct GBufferSceneConstants { float4x4 viewProj; };
struct GBufferPushConstants {
    float4x4 worldMatrix;
    ResourceHandle sceneCB;
    // ...
};
AGFX_PUSH_CONSTANTS(GBufferPushConstants, g_Constants);

vs_out main_vs(uint vertexID : SV_VertexID) {
    AGFXStructuredBuffer<GBufferSceneConstants> sceneCB = AGFXStructuredBuffer<GBufferSceneConstants>::Create(g_Constants.sceneCB);
    GBufferSceneConstants scene = sceneCB.Load(0);
    // ...
}
```

Host side, this `sceneCB` handle comes from `agfxBufferViewGetHandle` on an `agfxBufferView` created with `AGFX_BUFFER_VIEW_TYPE_CONSTANT` (or `STRUCTURED`, since the shader reads it as a structured buffer either way) over an upload-heap `agfxBuffer`.

### Vertex pulling instead of input-assembler vertex buffers

AGFX vertex shaders don't use IA vertex attributes — they take `SV_VertexID` and manually pull from a structured buffer, since bindless makes an explicit vertex-buffer handle no more expensive than an IA binding and lets one pipeline draw meshes with arbitrary vertex layouts:

```hlsl
vs_out main_vs(uint vertexID : SV_VertexID) {
    AGFXStructuredBuffer<SceneVertex> vertices = AGFXStructuredBuffer<SceneVertex>::Create(g_Constants.vertexBuffer);
    SceneVertex v = vertices.Load(vertexID + g_Constants.vertexOffset);
    // ...
}
```

`vertexOffset` in push constants lets one shared vertex buffer serve multiple meshes/draws without rebinding.

### Compute shaders: bounds check, then dispatch-sized work

```hlsl
struct MyPushConstants { ResourceHandle srcTex; ResourceHandle dstTex; uint dstWidth; uint dstHeight; };
AGFX_PUSH_CONSTANTS(MyPushConstants, g_Constants);

[numthreads(8, 8, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.dstWidth || id.y >= g_Constants.dstHeight)
        return;
    // ...
}
```

The `[numthreads(x, y, z)]` values must match the `groupSizeX/Y/Z` passed to `agfxComputePipelineCreateInfo` host-side (or, for mesh/task shaders, the reflected `meshSizeX/Y/Z`/`taskSizeX/Y/Z` the compiler extracts automatically — see below). Always bounds-check against actual target dimensions since dispatch group counts are typically rounded up.

### Entry point / stage naming convention

Existing shaders use `main_vs`, `main_ps`, `main_cs`, and (for mesh pipelines) `main_ms`/`main_as`. Match this convention for new shaders and pass the matching `agfxShaderStage`/`agfxShaderModuleType` pair host-side — the DXC target profile (`vs_6_6`, `ps_6_6`, `cs_6_6`, `ms_6_6`, `as_6_6`) is derived from `agfxShaderStage` in `agfx_shader_compiler_mac.mm`'s `ProfileFromType`, so stage and entry point must agree.

### Host-side: compile → shader module → pipeline

```cpp
// Typical helper (see deferred_renderer.cpp's CompileShader)
agfxShaderCompilerOptions options = {};
options.stage = AGFX_SHADER_STAGE_FRAGMENT;
strncpy(options.entryPoint, "main_ps", sizeof(options.entryPoint) - 1);
options.sourceCode = source.data();
options.sourceCodeSize = (uint32_t)source.size();

agfxShaderCompilerResult result = {};
agfxCompileShader(&options, &result);

agfxShaderModuleCreateInfo moduleInfo = {};
moduleInfo.code = result.compiledCode;
moduleInfo.codeSize = result.compiledSize;
moduleInfo.entryPoint = "main_ps";
moduleInfo.type = AGFX_SHADER_MODULE_TYPE_FRAGMENT;
agfxShaderModule* fragmentModule = agfxShaderModuleCreate(device, &moduleInfo);
```

Compile vertex+fragment (or mesh[+task]) modules separately and attach both to `agfxRenderPipelineCreateInfo::vertexShader`/`fragmentShader` (or `meshShader`/`taskShader`); a single `computeShader` module goes into `agfxComputePipelineCreateInfo`. `agfxShaderModuleDestroy` is safe immediately after the pipeline(s) built from it are created — the module isn't referenced afterward.

For mesh/task shaders, read `result.meshSizeX/Y/Z`/`taskSizeX/Y/Z` (populated via Metal shader-converter reflection, not something you hand-specify) and feed them into `agfxRenderPipelineCreateInfo::meshGroupSizeX/Y/Z`/`taskGroupSizeX/Y/Z` — these must match what the shader actually declares or dispatch counts silently disagree between the two backends.

### Common mistakes

- Declaring a second `register(bN)`/`register(tN)`/`register(sN)` resource binding instead of routing everything through push constants + `ResourceDescriptorHeap`/`SamplerDescriptorHeap` — AGFX's root signature only has two root-constants parameters (`b0` push constants, `b1` draw ID) and direct-indexed heaps; anything else won't be bound.
- Using `AGFXTexture2D` (read-only) where the texture was created with `AGFX_TEXTURE_USAGE_STORAGE` and needs `Store`, or vice versa — pick the wrapper matching the host-side `agfxTextureUsage`/`agfxTextureViewCreateInfo::writeable`.
- Forgetting the bounds check in a compute shader before writing to an `AGFXRWTexture2D` sized smaller than `numthreads`-rounded dispatch dimensions.
- Mismatching entry-point name and `agfxShaderStage`/`agfxShaderModuleType` between the compile options and the module create info — the DXC profile is derived from the stage, and a mismatch will misdirect the compiler or produce a module that doesn't bind to the intended pipeline slot.
