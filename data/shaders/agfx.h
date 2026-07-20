/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 18:18:31
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#ifndef AGFX_SHADER_HLSL
#define AGFX_SHADER_HLSL

static const int AGFX_INVALID_DESCRIPTOR = -1;
typedef uint ResourceHandle;

#define AGFX_PUSH_CONSTANTS(type, name) ConstantBuffer<type> name : register(b0)

class AGFXSampler
{
    ResourceHandle handle;
    SamplerState state;

    static AGFXSampler Create(ResourceHandle id)
    {
        AGFXSampler s;
        s.handle = id;
        s.state  = SamplerDescriptorHeap[id];
        return s;
    }

    ResourceHandle Handle() { return handle; }
    SamplerState   Resource() { return state; }
};

class AGFXComparisonSampler
{
    ResourceHandle handle;
    SamplerComparisonState state;

    static AGFXComparisonSampler Create(ResourceHandle id)
    {
        AGFXComparisonSampler s;
        s.handle = id;
        s.state  = SamplerDescriptorHeap[id];
        return s;
    }

    ResourceHandle Handle() { return handle; }
    SamplerComparisonState   Resource() { return state; }
};

// RO Textures

template<typename T>
class AGFXTexture1D
{
    ResourceHandle handle;
    Texture1D<T> texture;

    static AGFXTexture1D<T> Create(ResourceHandle id)
    {
        AGFXTexture1D<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    Texture1D<T>   Resource() { return texture; }

    T    Load(int2 location)                                                         { return texture.Load(location); }
    T    Sample(AGFXSampler s, float location)                                   { return texture.Sample(s.state, location); }
    T    SampleLevel(AGFXSampler s, float location, float lod)                   { return texture.SampleLevel(s.state, location, lod); }
    T    SampleBias(AGFXSampler s, float location, float bias)                   { return texture.SampleBias(s.state, location, bias); }
    T    SampleGrad(AGFXSampler s, float location, float ddx, float ddy)         { return texture.SampleGrad(s.state, location, ddx, ddy); }
    void GetDimensions(uint mip, out uint width, out uint numLevels)                 { texture.GetDimensions(mip, width, numLevels); }
};

template<typename T>
class AGFXTexture2D
{
    ResourceHandle handle;
    Texture2D<T> texture;

    static AGFXTexture2D<T> Create(ResourceHandle id)
    {
        AGFXTexture2D<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    Texture2D<T>   Resource() { return texture; }

    T    Load(int2 location)                                                         { return texture.Load(location); }
    T    Sample(AGFXSampler s, float2 location)                                   { return texture.Sample(s.state, location); }
    T    SampleLevel(AGFXSampler s, float2 location, float lod)                   { return texture.SampleLevel(s.state, location, lod); }
    T    SampleBias(AGFXSampler s, float2 location, float bias)                   { return texture.SampleBias(s.state, location, bias); }
    T    SampleGrad(AGFXSampler s, float2 location, float2 ddx, float2 ddy)         { return texture.SampleGrad(s.state, location, ddx, ddy); }
    float SampleCmp(AGFXComparisonSampler s, float2 location, float cmp)             { return texture.SampleCmp(s.state, location, cmp); }
    float SampleCmpLevelZero(AGFXComparisonSampler s, float2 location, float cmp)     { return texture.SampleCmpLevelZero(s.state, location, cmp); }
    void GetDimensions(uint mip, out uint width, out uint height, out uint numLevels)                 { texture.GetDimensions(mip, width, height, numLevels); }
};

template<typename T>
class AGFXTexture2DArray
{
    ResourceHandle handle;
    Texture2DArray<T> texture;

    static AGFXTexture2DArray<T> Create(ResourceHandle id)
    {
        AGFXTexture2DArray<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    Texture2DArray<T>   Resource() { return texture; }

    T    Load(int3 location)                                                         { return texture.Load(location); }
    T    Sample(AGFXSampler s, float3 location)                                   { return texture.Sample(s.state, location); }
    T    SampleLevel(AGFXSampler s, float3 location, float lod)                   { return texture.SampleLevel(s.state, location, lod); }
    T    SampleBias(AGFXSampler s, float3 location, float bias)                   { return texture.SampleBias(s.state, location, bias); }
    T    SampleGrad(AGFXSampler s, float3 location, float3 ddx, float3 ddy)         { return texture.SampleGrad(s.state, location, ddx, ddy); }
    float SampleCmp(AGFXComparisonSampler s, float3 location, float cmp)             { return texture.SampleCmp(s.state, location, cmp); }
    float SampleCmpLevelZero(AGFXComparisonSampler s, float3 location, float cmp)     { return texture.SampleCmpLevelZero(s.state, location, cmp); }
    void GetDimensions(uint mip, out uint width, out uint height, out uint numLevels, out uint arraySize)                 { texture.GetDimensions(mip, width, height, numLevels, arraySize); }
};

template<typename T>
class AGFXTexture3D
{
    ResourceHandle handle;
    Texture3D<T> texture;

    static AGFXTexture3D<T> Create(ResourceHandle id)
    {
        AGFXTexture3D<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    Texture3D<T>   Resource() { return texture; }

    T    Load(int4 location)                                                         { return texture.Load(location); }
    T    Sample(AGFXSampler s, float3 location)                                   { return texture.Sample(s.state, location); }
    T    SampleLevel(AGFXSampler s, float3 location, float lod)                   { return texture.SampleLevel(s.state, location, lod); }
    T    SampleBias(AGFXSampler s, float3 location, float bias)                   { return texture.SampleBias(s.state, location, bias); }
    T    SampleGrad(AGFXSampler s, float3 location, float3 ddx, float3 ddy)         { return texture.SampleGrad(s.state, location, ddx, ddy); }
    void GetDimensions(uint mip, out uint width, out uint height, out uint depth, out uint numLevels)                 { texture.GetDimensions(mip, width, height, depth, numLevels); }
};

template<typename T>
class AGFXTextureCube
{
    ResourceHandle handle;
    TextureCube<T> texture;

    static AGFXTextureCube<T> Create(ResourceHandle id)
    {
        AGFXTextureCube<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    TextureCube<T>   Resource() { return texture; }

    T    Sample(AGFXSampler s, float3 location)                                   { return texture.Sample(s.state, location); }
    T    SampleLevel(AGFXSampler s, float3 location, float lod)                   { return texture.SampleLevel(s.state, location, lod); }
    T    SampleBias(AGFXSampler s, float3 location, float bias)                   { return texture.SampleBias(s.state, location, bias); }
    T    SampleGrad(AGFXSampler s, float3 location, float3 ddx, float3 ddy)         { return texture.SampleGrad(s.state, location, ddx, ddy); }
    void GetDimensions(uint mip, out uint width, out uint height, out uint numLevels)                 { texture.GetDimensions(mip, width, height, numLevels); }
};

// RW Textures

template<typename T>
class AGFXRWTexture1D
{
    ResourceHandle handle;
    RWTexture1D<T> texture;

    static AGFXRWTexture1D<T> Create(ResourceHandle id)
    {
        AGFXRWTexture1D<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    RWTexture1D<T>   Resource() { return texture; }

    T Load(int location) { return texture[location]; }
    void Store(int location, T value) { texture[location] = value; }
    void GetDimensions(out uint width) { texture.GetDimensions(width); }
};

template<typename T>
class AGFXRWTexture2D
{
    ResourceHandle handle;
    RWTexture2D<T> texture;

    static AGFXRWTexture2D<T> Create(ResourceHandle id)
    {
        AGFXRWTexture2D<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    RWTexture2D<T>   Resource() { return texture; }

    T Load(int2 location) { return texture[location]; }
    void Store(int2 location, T value) { texture[location] = value; }
    void GetDimensions(out uint width, out uint height) { texture.GetDimensions(width, height); }
};

template<typename T>
class AGFXRWTexture2DArray
{
    ResourceHandle handle;
    RWTexture2DArray<T> texture;

    static AGFXRWTexture2DArray<T> Create(ResourceHandle id)
    {
        AGFXRWTexture2DArray<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    RWTexture2DArray<T>   Resource() { return texture; }

    T Load(int3 location) { return texture[location]; }
    void Store(int3 location, T value) { texture[location] = value; }
    void GetDimensions(out uint width, out uint height, out uint arraySize) { texture.GetDimensions(width, height, arraySize); }
};

template<typename T>
class AGFXRWTexture3D
{
    ResourceHandle handle;
    RWTexture3D<T> texture;

    static AGFXRWTexture3D<T> Create(ResourceHandle id)
    {
        AGFXRWTexture3D<T> t;
        t.handle  = id;
        t.texture = ResourceDescriptorHeap[id];
        return t;
    }

    ResourceHandle Handle() { return handle; }
    RWTexture3D<T>   Resource() { return texture; }

    T Load(uint3 location) { return texture[location]; }
    void Store(uint3 location, T value) { texture[location] = value; }
    void GetDimensions(out uint width, out uint height, out uint depth) { texture.GetDimensions(width, height, depth); }
};

// Buffers

template<typename T>
class AGFXStructuredBuffer
{
    ResourceHandle handle;
    StructuredBuffer<T> buffer;

    static AGFXStructuredBuffer<T> Create(ResourceHandle id)
    {
        AGFXStructuredBuffer<T> b;
        b.handle  = id;
        b.buffer = ResourceDescriptorHeap[id];
        return b;
    }

    ResourceHandle Handle() { return handle; }
    StructuredBuffer<T>   Resource() { return buffer; }

    T Load(uint location) { return buffer[location]; }
    void GetDimensions(out uint count) { buffer.GetDimensions(count); }
};

template<typename T>
class AGFXRWStructuredBuffer
{
    ResourceHandle handle;
    RWStructuredBuffer<T> buffer;

    static AGFXRWStructuredBuffer<T> Create(ResourceHandle id)
    {
        AGFXRWStructuredBuffer<T> b;
        b.handle  = id;
        b.buffer = ResourceDescriptorHeap[id];
        return b;
    }

    ResourceHandle Handle() { return handle; }
    RWStructuredBuffer<T>   Resource() { return buffer; }

    T Load(uint location) { return buffer[location]; }
    void Store(uint location, T value) { buffer[location] = value; }
    void GetDimensions(out uint count) { buffer.GetDimensions(count); }
};

class AGFXByteAddressBuffer
{
    ResourceHandle handle;
    ByteAddressBuffer buffer;

    static AGFXByteAddressBuffer Create(ResourceHandle id)
    {
        AGFXByteAddressBuffer b;
        b.handle  = id;
        b.buffer = ResourceDescriptorHeap[id];
        return b;
    }

    ResourceHandle Handle() { return handle; }
    ByteAddressBuffer   Resource() { return buffer; }

    uint Load(uint location) { return buffer.Load(location); }
    uint2 Load2(uint location) { return buffer.Load2(location); }
    uint3 Load3(uint location) { return buffer.Load3(location); }
    uint4 Load4(uint location) { return buffer.Load4(location); }
};

class AGFXRWByteAddressBuffer
{
    ResourceHandle handle;
    RWByteAddressBuffer buffer;

    static AGFXRWByteAddressBuffer Create(ResourceHandle id)
    {
        AGFXRWByteAddressBuffer b;
        b.handle  = id;
        b.buffer = ResourceDescriptorHeap[id];
        return b;
    }

    ResourceHandle Handle() { return handle; }
    RWByteAddressBuffer   Resource() { return buffer; }

    uint Load(uint location) { return buffer.Load(location); }
    uint2 Load2(uint location) { return buffer.Load2(location); }
    uint3 Load3(uint location) { return buffer.Load3(location); }
    uint4 Load4(uint location) { return buffer.Load4(location); }

    void Store(uint location, uint value) { buffer.Store(location, value); }
    void Store2(uint location, uint2 value) { buffer.Store2(location, value); }
    void Store3(uint location, uint3 value) { buffer.Store3(location, value); }
    void Store4(uint location, uint4 value) { buffer.Store4(location, value); }

    void InterlockedAdd(uint addr, uint v, out uint original) { buffer.InterlockedAdd(addr, v, original); }
    void InterlockedAnd(uint addr, uint v, out uint original) { buffer.InterlockedAnd(addr, v, original); }
    void InterlockedOr(uint addr, uint v, out uint original) { buffer.InterlockedOr(addr, v, original); }
    void InterlockedXor(uint addr, uint v, out uint original) { buffer.InterlockedXor(addr, v, original); }
    void InterlockedMax(uint addr, uint v, out uint original) { buffer.InterlockedMax(addr, v, original); }
    void InterlockedMin(uint addr, uint v, out uint original) { buffer.InterlockedMin(addr, v, original); }
    void InterlockedExchange(uint addr, uint v, out uint original) { buffer.InterlockedExchange(addr, v, original); }
    void InterlockedCompareExchange(uint addr, uint compareValue, uint value, out uint original) { buffer.InterlockedCompareExchange(addr, compareValue, value, original); }
};

// RT

class AGFXRaytracingAccelerationStructure
{
    ResourceHandle handle;
    RaytracingAccelerationStructure accelerationStructure;

    static AGFXRaytracingAccelerationStructure Create(ResourceHandle id)
    {
        AGFXRaytracingAccelerationStructure b;
        b.handle  = id;
        b.accelerationStructure = ResourceDescriptorHeap[id];
        return b;
    }

    ResourceHandle Handle() { return handle; }
    RaytracingAccelerationStructure   Resource() { return accelerationStructure; }
};

struct __agfx_draw_id { uint id; };
#define AGFX_DECLARE_DRAW_ID() ConstantBuffer<__agfx_draw_id> __agfx_draw_id_binding : register(b1)
#define AGFX_DRAW_ID() __agfx_draw_id_binding.id

// Indirect bundles
//
// Command struct field order must match agfxDrawIndexedCommand (agfx.h): drawID is the leading
// field. The D3D12 command signature declares the drawID CONSTANT argument at index 0 and the
// terminal DRAW_INDEXED argument at index 1, so the argument buffer's per-command memory layout is
// [drawID, D3D12_DRAW_INDEXED_ARGUMENTS-shaped fields...].
class AGFXIndirectDrawIndexedBundle
{
    AGFXRWByteAddressBuffer commands;
    AGFXRWByteAddressBuffer count;

    static AGFXIndirectDrawIndexedBundle Create(uint64_t bundleHandle)
    {
        AGFXIndirectDrawIndexedBundle b;
        b.commands = AGFXRWByteAddressBuffer::Create((ResourceHandle)(bundleHandle & 0xFFFFFFFF));
        b.count    = AGFXRWByteAddressBuffer::Create((ResourceHandle)(bundleHandle >> 32));
        return b;
    }

    // commandOffset/countIndex are in command-struct/uint32 units, matching agfxIndirectBundleExecuteInfo.
    void DrawIndexed(uint commandOffset, uint countIndex, uint drawId, uint indexCount, uint instanceCount, uint firstIndex, int vertexOffset, uint firstInstance)
    {
        uint slot;
        count.InterlockedAdd(countIndex * 4, 1, slot);
        uint byteOffset = (commandOffset + slot) * 24; // sizeof(agfxDrawIndexedCommand): 6 x 4 bytes
        commands.Store(byteOffset + 0,  drawId);
        commands.Store(byteOffset + 4,  indexCount);
        commands.Store(byteOffset + 8,  instanceCount);
        commands.Store(byteOffset + 12, firstIndex);
        commands.Store(byteOffset + 16, (uint)vertexOffset);
        commands.Store(byteOffset + 20, firstInstance);
    }
};

// Command struct field order must match agfxDrawCommand (agfx.h): drawID leading, matching
// AGFXIndirectDrawIndexedBundle's note above.
class AGFXIndirectDrawBundle
{
    AGFXRWByteAddressBuffer commands;
    AGFXRWByteAddressBuffer count;

    static AGFXIndirectDrawBundle Create(uint64_t bundleHandle)
    {
        AGFXIndirectDrawBundle b;
        b.commands = AGFXRWByteAddressBuffer::Create((ResourceHandle)(bundleHandle & 0xFFFFFFFF));
        b.count    = AGFXRWByteAddressBuffer::Create((ResourceHandle)(bundleHandle >> 32));
        return b;
    }

    void Draw(uint commandOffset, uint countIndex, uint drawId, uint vertexCount, uint instanceCount, uint firstVertex, uint firstInstance)
    {
        uint slot;
        count.InterlockedAdd(countIndex * 4, 1, slot);
        uint byteOffset = (commandOffset + slot) * 20; // sizeof(agfxDrawCommand): 5 x 4 bytes
        commands.Store(byteOffset + 0,  drawId);
        commands.Store(byteOffset + 4,  vertexCount);
        commands.Store(byteOffset + 8,  instanceCount);
        commands.Store(byteOffset + 12, firstVertex);
        commands.Store(byteOffset + 16, firstInstance);
    }
};

// Command struct field order must match agfxDrawMeshCommand (agfx.h): drawID leading, matching
// AGFXIndirectDrawIndexedBundle's note above.
class AGFXIndirectDrawMeshBundle
{
    AGFXRWByteAddressBuffer commands;
    AGFXRWByteAddressBuffer count;

    static AGFXIndirectDrawMeshBundle Create(uint64_t bundleHandle)
    {
        AGFXIndirectDrawMeshBundle b;
        b.commands = AGFXRWByteAddressBuffer::Create((ResourceHandle)(bundleHandle & 0xFFFFFFFF));
        b.count    = AGFXRWByteAddressBuffer::Create((ResourceHandle)(bundleHandle >> 32));
        return b;
    }

    void DrawMesh(uint commandOffset, uint countIndex, uint drawId, uint groupSizeX, uint groupSizeY, uint groupSizeZ)
    {
        uint slot;
        count.InterlockedAdd(countIndex * 4, 1, slot);
        uint byteOffset = (commandOffset + slot) * 16; // sizeof(agfxDrawMeshCommand): 4 x 4 bytes
        commands.Store(byteOffset + 0,  drawId);
        commands.Store(byteOffset + 4,  groupSizeX);
        commands.Store(byteOffset + 8,  groupSizeY);
        commands.Store(byteOffset + 12, groupSizeZ);
    }
};

// agfxDispatchCommand has no drawID field - plain indirect compute dispatch is expected to carry
// its own addressing scheme (SV_DispatchThreadID plus a caller-managed per-dispatch data buffer),
// so no CONSTANT/drawID patch stage exists for this bundle type. See notes/mdi.md open question #5.
class AGFXIndirectDispatchBundle
{
    AGFXRWByteAddressBuffer commands;
    AGFXRWByteAddressBuffer count;

    static AGFXIndirectDispatchBundle Create(uint64_t bundleHandle)
    {
        AGFXIndirectDispatchBundle b;
        b.commands = AGFXRWByteAddressBuffer::Create((ResourceHandle)(bundleHandle & 0xFFFFFFFF));
        b.count    = AGFXRWByteAddressBuffer::Create((ResourceHandle)(bundleHandle >> 32));
        return b;
    }

    void Dispatch(uint commandOffset, uint countIndex, uint groupCountX, uint groupCountY, uint groupCountZ)
    {
        uint slot;
        count.InterlockedAdd(countIndex * 4, 1, slot);
        uint byteOffset = (commandOffset + slot) * 12; // sizeof(agfxDispatchCommand): 3 x 4 bytes
        commands.Store(byteOffset + 0, groupCountX);
        commands.Store(byteOffset + 4, groupCountY);
        commands.Store(byteOffset + 8, groupCountZ);
    }
};

#endif
