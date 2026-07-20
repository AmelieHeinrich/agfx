/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 12:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// dear imgui: Renderer Backend for agfx
// This needs to be used along with a Platform Backend (e.g. imgui_impl_sdl3)

// Implemented features:
//  [X] Renderer: User texture binding. Use 'agfxTextureViewGetHandle()' cast to ImTextureID.
//  [X] Renderer: Multi-viewport / platform windows are not supported by this backend.
//  [X] Renderer: Large meshes support (64k+ vertices) with 'ImGuiBackendFlags_RendererHasVtxOffset'.

#pragma once
#include "imgui.h"
#ifndef IMGUI_DISABLE

#include <agfx/agfx.h>

// Initialization data, for ImGui_ImplAGFX_Init()
struct ImGui_ImplAGFX_InitInfo
{
    agfxDevice*         Device = nullptr;
    agfxCommandQueue*   CommandQueue = nullptr;
    agfxTextureFormat   ColorAttachmentFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
    agfxShaderModule*   VertexShaderModule = nullptr; // It is your job to compile the ImGui shader
    agfxShaderModule*   FragmentShaderModule = nullptr;
    uint32_t            FramesInFlight = 3;
};

IMGUI_IMPL_API bool ImGui_ImplAGFX_Init(ImGui_ImplAGFX_InitInfo* info);
IMGUI_IMPL_API void ImGui_ImplAGFX_Shutdown();
IMGUI_IMPL_API void ImGui_ImplAGFX_NewFrame();
IMGUI_IMPL_API void ImGui_ImplAGFX_RenderDrawData(ImDrawData* draw_data, agfxRenderPass* render_pass, uint32_t fb_width, uint32_t fb_height, uint32_t frame_index);

// Called by ImGui_ImplAGFX_Init(), but can also be called directly to force a device object rebuild.
IMGUI_IMPL_API bool ImGui_ImplAGFX_CreateDeviceObjects();
IMGUI_IMPL_API void ImGui_ImplAGFX_InvalidateDeviceObjects();

// Convenience helper: recreate the render pipeline against a new color attachment format
// (e.g. after a HDR toggle changes the output format).
IMGUI_IMPL_API void ImGui_ImplAGFX_RecreatePipeline(agfxTextureFormat new_color_format);

#endif // #ifndef IMGUI_DISABLE
