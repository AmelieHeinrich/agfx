---
name: agfx-raytracing
description: ALWAYS use when building or tracing against acceleration structures in AGFX — creating agfxAccelerationStructure (BLAS/TLAS) objects, filling agfxAccelerationStructureGeometry / agfxAccelerationStructureInstance, building/updating/compacting them in a compute pass, or writing inline-raytracing (RayQuery) HLSL compute shaders that trace an AGFXRaytracingAccelerationStructure. Trigger for agfxAccelerationStructureCreate/Destroy/GetSizes/GetHandle/AddInstances/ResetInstances, agfxComputePassBuildAccelerationStructure/UpdateAccelerationStructure/CopyAccelerationStructure/CompactAccelerationStructure, agfxCommandBufferAccelerationStructureBarrier, AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL/TOP_LEVEL, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, supportsRayTracing, RayQuery/TraceRayInline/CommittedStatus, "ray tracing", "reflections", "BLAS", "TLAS", "acceleration structure", agfx::AccelerationStructure, agfx::ez CreateBottomLevel/CreateTopLevel. Do NOT trigger for general compute dispatch or barriers unrelated to acceleration structures — use agfx-synchronization. Do NOT trigger for the bindless shader-authoring mechanics themselves (ResourceHandle, push constants) — use agfx-writing-bindless-shaders.
---

# AGFX Ray Tracing

## Overview

AGFX exposes **inline ray tracing** (DXR 1.1 `RayQuery` / `TraceRayInline`) run from **compute** shaders — there is no separate ray-generation/hit/miss pipeline or shader binding table. You build a two-level acceleration structure (one or more **BLAS** = bottom-level, over triangle or AABB geometry; one **TLAS** = top-level, over instances that reference BLAS), then a compute shader reads the TLAS bindlessly and traces rays against it. The reference consumer is the demo's raytraced mirror reflections (`data/shaders/demo/reflections.hlsl` + `src/agfx/agfx_demo/reflections.cpp`, wired in `deferred_renderer.cpp::RenderReflections`) and the scene-side build in `src/agfx/agfx_demo/gltf_scene.cpp`.

On macOS everything maps to Metal 4 acceleration structures (`agfx_metal4.mm`); on Windows to D3D12 DXR. The C API is in `src/agfx/agfx/agfx.h`; RAII C++ wrappers in `agfx.hpp`; a one-call immediate-mode helper in `agfx_ez.hpp`.

## Ownership

**Owns:**
- Acceleration-structure objects and their lifecycle: `agfxAccelerationStructureCreate` / `CreateCompacted` / `Destroy` / `GetSizes` / `GetHandle` / `AddInstances` / `ResetInstances`
- Geometry & instance description: `agfxAccelerationStructureGeometry` (triangles/AABBs), `agfxAccelerationStructureInstance`, the BLAS/TLAS create-info structs
- Build-time compute-pass commands: `agfxComputePassBuildAccelerationStructure` / `UpdateAccelerationStructure` / `CopyAccelerationStructure` / `CompactAccelerationStructure` / `WriteCompactedSizeToBuffer`
- The AS residency/scratch/barrier build recipe and its ordering hazards
- Capability gating via `agfxDeviceGetInfo().supportsRayTracing`
- Shader-side inline tracing: `AGFXRaytracingAccelerationStructure`, `RayQuery`, `TraceRayInline`, hit-attribute reconstruction
- C++/ez wrappers: `agfx::AccelerationStructure`, `agfx::ez::AccelerationStructure`, `Context::CreateBottomLevel/CreateTopLevel/SupportsRayTracing`

**Does NOT own** (delegate):
- General barrier stage/agglomerate semantics, fences, frames-in-flight → **agfx-synchronization** (this skill only covers the AS-specific `AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE` transition)
- Push-constant / `ResourceHandle` / `AGFX*` bindless mechanics, DXC→Metal IR compile → **agfx-writing-bindless-shaders**
- Vertex/index/structured buffer creation and views → the general buffer APIs

## Capability gate (do this first)

Ray tracing is optional. Query it once and skip **all** BLAS/TLAS building and tracing when absent — never create an AS on an unsupported device.

```cpp
agfxDeviceInfo info = {};
agfxDeviceGetInfo(device, &info);
if (!info.supportsRayTracing) { /* skip AS build; leave tlas == nullptr */ }
```

Downstream passes should early-out on a null TLAS (`if (!scene.tlas) return;`). In the demo, `gltf_scene.cpp` returns before the build loop, and `RenderReflections` guards on `!scene.tlas`. The lighting shader also takes a `reflectionsEnabled` flag so it never composites a **stale** reflection texture when the pass didn't run this frame.

## Build lifecycle (host)

The full, correct sequence — every step matters:

1. **Create BLAS** per mesh: fill `agfxAccelerationStructureGeometry`, set `bottomLevel.geometries/geometryCount`, `agfxAccelerationStructureCreate`.
2. **Create TLAS**: `topLevel.maxInstanceCount`, then `agfxAccelerationStructureAddInstances(tlas, instances, count)` (call `ResetInstances` first if refilling).
3. **Size the scratch buffer**: `agfxAccelerationStructureGetSizes` on each AS; allocate one GPU-only scratch buffer ≥ the max `scratchBufferSize`.
4. **`agfxDeviceMakeResourcesResident(device)`** — the AS storage, TLAS instance buffer, and scratch buffer were added to the residency set *after* any earlier commit, so re-commit or the GPU faults ("MTLAccelerationStructure is not resident at the time of commit").
5. **Record builds** in a compute pass, **with an AS→AS barrier after each** (see below).
6. **Submit + fence-wait** (builds are typically one-time at load).

Reference: `gltf_scene.cpp` lines ~455–553.

```cpp
agfxAccelerationStructureGeometry geo = {};
geo.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES;
geo.opaque = true;
geo.triangles.vertexBuffer = vertexBuffer;
geo.triangles.vertexOffset = firstVertex * sizeof(Vertex);   // BYTES — see gotcha
geo.triangles.vertexCount  = vertexCount;
geo.triangles.indexBuffer  = indexBuffer;
geo.triangles.indexCount   = indexCount;
geo.triangles.indexOffset  = firstIndex * sizeof(uint32_t);  // BYTES

agfxAccelerationStructureCreateInfo blasInfo = {};
blasInfo.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
blasInfo.bottomLevel.geometries = &geo;
blasInfo.bottomLevel.geometryCount = 1;
agfxAccelerationStructure* blas = agfxAccelerationStructureCreate(device, &blasInfo);
```

## Critical gotchas (all learned the hard way)

### 1. Barrier source state MUST be `RAYTRACING_ACCELERATION_STRUCTURE`, never `COMMON`
The barrier tracker drops any barrier whose **source** state has an empty producer stage — and `AGFX_RESOURCE_STATE_COMMON` produces stage 0. A `COMMON → RAYTRACING_ACCELERATION_STRUCTURE` barrier is silently a **no-op**, so builds sharing one scratch buffer run concurrently and the TLAS reads half-built BLAS → GPU memory corruption → **device lost / window-server hang**. Use AS→AS:

```cpp
agfxCommandBufferAccelerationStructureBarrier(cmd, as,
    AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,   // producer = AccelerationStructure stage
    AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, true);
```
Emit one after each BLAS build (serializes shared-scratch reuse) and one before the TLAS build (orders BLAS-before-TLAS). Give each build its own scratch region instead if you want them to run in parallel.

### 2. `vertexOffset` / `indexOffset` in the geometry struct are **byte** offsets
The C API documents them as "offset" but the backend adds them straight to the buffer GPU address. Pass `elementIndex * stride`, not the raw element index. A raw index gives both a wrong and a misaligned address ("Vertex buffer address must be a multiple of 4 bytes"). The **shader** side, by contrast, indexes by **element** (`vertices.Load(vertexOffset + index)`), so the GPUScene copy of the offset stays an element index — only the AS geometry wants bytes.

### 3. Instance transform is **row-major 3×4**
`agfxAccelerationStructureInstance.transform` is `float[12]`, row-major (`transform[row*4 + col]`). The Metal backend transposes it into `MTLPackedFloat4x3` (column-major) element-by-element. If you build this list yourself, lay it out row-major. A transpose bug here flings every instance to a garbage world position — symptom: the graphics debugger shows absurd (~1e8) "vertex" values (world-space, viewer-applied) even though the BLAS geometry is fine.

### 4. Inline tracing runs in compute → **no implicit-LOD sampling**
Reflection/hit shading in a `[numthreads(8,8,1)]` (non-1D) compute shader cannot use `.Sample()` / `.SampleCmp()` (they need pixel derivatives → "dx.op.sample.f32 ... only supported for 1D threadgroup size"). Use `SampleLevel(..., 0.0f)` and `SampleCmpLevelZero(...)`. Shadow maps have no mips so LOD 0 is exact anyway.

### 5. Re-commit residency after every post-commit allocation
Any AS, instance buffer, or scratch buffer created after your last `MakeResourcesResident` must be made resident again before the build references it. See lifecycle step 4.

## Shader side (inline RayQuery)

Trace against the TLAS handle passed in push constants. Pattern from `reflections.hlsl`:

```hlsl
AGFXRaytracingAccelerationStructure tlas = AGFXRaytracingAccelerationStructure::Create(g_Constants.tlas);

RayDesc ray;
ray.Origin = worldPos + normal * 0.01f;   // offset off the surface
ray.Direction = reflect(-viewDir, normal);
ray.TMin = 0.001f; ray.TMax = 1000.0f;

RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
q.TraceRayInline(tlas.Resource(), RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
while (q.Proceed()) {}

if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { /* miss: sky/env */ return; }

uint instance = q.CommittedInstanceID();          // == userID set on the instance
uint prim     = q.CommittedPrimitiveIndex();
float2 bary   = q.CommittedTriangleBarycentrics();
// Reconstruct hit attributes: look up per-instance offsets in a GPUScene structured buffer,
// fetch the 3 indices then 3 vertices, interpolate with barycentrics, shade.
```

`CommittedInstanceID()` returns the `userID` you set in `agfxAccelerationStructureInstance` — use it to index a parallel "GPU scene" buffer holding each instance's vertex/index offsets and material handles. `AGFXRaytracingAccelerationStructure` is declared in `data/shaders/<project>/agfx.h` (owned by **agfx-writing-bindless-shaders**).

## C++ wrapper (`agfx.hpp`)

```cpp
agfx::AccelerationStructure blas = device.CreateAccelerationStructure(blasInfo);
agfx::AccelerationStructure tlas = device.CreateAccelerationStructure(tlasInfo);
tlas.AddInstances(instances, count);
auto sizes = blas.GetSizes();                    // scratch sizing
// build:
{
    agfx::ComputePass pass = cmd.BeginComputePass("Build BLAS");
    pass.BuildAccelerationStructure(blas, scratch, 0);
}
cmd.AccelerationStructureBarrier(blas,
    AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
uint32_t handle = (uint32_t)tlas.GetHandle();    // → push constant
```
Also on `ComputePass`: `UpdateAccelerationStructure` (refit; needs `allowUpdate`), `CopyAccelerationStructure`, `CompactAccelerationStructure` (dest from `Device::CreateAccelerationStructureCompacted`), `WriteCompactedSizeToBuffer`.

## ez layer (`agfx_ez.hpp`) — one call, fully synchronous

The ez helpers hide scratch allocation, residency re-commit, the AS→AS barrier, submit, and the fence wait:

```cpp
if (ctx.SupportsRayTracing()) {
    ez::AccelerationStructure blas = ctx.CreateBottomLevel(
        vtxBuf, vtxCount, /*vertexOffset*/0, idxBuf, idxCount, /*indexOffset*/0);
    agfxAccelerationStructureInstance inst = {};
    inst.blas = blas.Raw();  WriteRowMajor3x4(world, inst.transform);  inst.userID = 0; inst.opaque = true;
    ez::AccelerationStructure tlas = ctx.CreateTopLevel(&inst, 1);
    uint32_t tlasHandle = tlas.Bind();   // 0 if the device has no RT
}
```
Here `vertexOffset`/`indexOffset` are **element** indices — the ez wrapper multiplies by the buffer stride for you. `Bind()` returns the bindless handle (0 when invalid). `CreateBottomLevel`/`CreateTopLevel` return an invalid AS on non-RT devices, so guard with `SupportsRayTracing()`.

## Backend implementation notes (`agfx_metal4.mm`, if editing the Metal backend)

Metal 4 replaced object+offset bindings with **address+length ranges** and typed instance buffers — a bare address leaves `length == 0` and produces a **degenerate/empty** structure the GPU debugger can't even open:
- `MTL4AccelerationStructureTriangleGeometryDescriptor.vertexBuffer` / `.indexBuffer` / `.boundingBoxBuffer` are `MTL4BufferRange` — build with `MTL4BufferRangeMake(gpuAddress + byteOffset, elementCount * stride)`.
- The TLAS `MTL4InstanceAccelerationStructureDescriptor` needs `instanceDescriptorBuffer` (a `MTL4BufferRange`), `instanceDescriptorStride = sizeof(MTLIndirectAccelerationStructureInstanceDescriptor)`, and `instanceDescriptorType = ...TypeIndirect` — not just `instanceCount`.
- Instance transform: unpack row-major 3×4 → `MTLPackedFloat4x3` (column-major, `columns[col][row]`) element-by-element; a `memcpy` transposes it.
- The bindless AS descriptor must be written to the **allocated slot** (`entry[slotIndex].gpuVA = resourceIDBuffer.gpuAddress`), not slot 0.
- Any texture that will have a view created needs `MTLTextureUsagePixelFormatView` in its usage (the backend ORs it in for all textures) — unrelated to AS but in the same RT bring-up path.

## Cross-references
- **agfx-synchronization** — barrier agglomeration/stage model, fences, frames-in-flight (this skill only names the AS-specific transition).
- **agfx-writing-bindless-shaders** — `AGFXRaytracingAccelerationStructure`, push constants, the `AGFX*` resource wrappers, DXC→Metal IR compile.
- **using-agfx-ez** — the immediate-mode Context this skill's `CreateBottomLevel/CreateTopLevel` extend.
