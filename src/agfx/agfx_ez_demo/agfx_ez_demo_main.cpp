/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// A minimal agfx_ez.hpp sample: spins an RGB cube into an offscreen render target,
// runs a compute pass over it (hue cycle / vignette / scanlines), then blits the result
// to the swap chain and draws an ImGui overlay on top. Demonstrates Context's render
// passes, compute passes, one-call resource creation, and the ShaderBindings helper.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <agfx/agfx_ez.hpp>
#include <agfx_shader/agfx_shader_compiler.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "imgui_impl_sdl3.h"
#include <agfx_imgui/imgui_impl_agfx.h>

namespace
{
    constexpr uint32_t kFramesInFlight = 3;
    constexpr const char* kDataDir = "data/";

    std::string ReadFile(const char* path)
    {
        SDL_IOStream* io = SDL_IOFromFile(path, "rb");
        if (!io) {
            return {};
        }
        Sint64 size = SDL_GetIOSize(io);
        std::string contents(static_cast<size_t>(size), '\0');
        SDL_ReadIO(io, contents.data(), contents.size());
        SDL_CloseIO(io);
        return contents;
    }

    void* Allocate(uint64_t size) { return malloc(size); }
    void Deallocate(void* ptr) { free(ptr); }

    agfx::ShaderModule CompileShader(agfx::Device& device, const std::string& source, agfxShaderStage stage,
                                      const char* entryPoint, agfxShaderModuleType moduleType)
    {
        agfxShaderCompilerOptions options{};
        options.stage = stage;
        strncpy(options.entryPoint, entryPoint, sizeof(options.entryPoint) - 1);
        options.sourceCode = source.empty() ? nullptr : const_cast<char*>(&source[0]);
        options.sourceCodeSize = static_cast<uint32_t>(source.size());

        agfxShaderCompilerResult result{};
        agfxCompileShader(&options, &result);
        if (!result.compiledCode || result.compiledSize == 0) {
            std::cerr << "Failed to compile shader entry point " << entryPoint << std::endl;
            exit(1);
        }

        agfxShaderModuleCreateInfo moduleInfo{};
        moduleInfo.code = result.compiledCode;
        moduleInfo.codeSize = result.compiledSize;
        moduleInfo.entryPoint = entryPoint;
        moduleInfo.type = moduleType;
        agfx::ShaderModule module = device.CreateShaderModule(moduleInfo);
        free(result.compiledCode);
        return module;
    }

    struct CubeVertex
    {
        glm::vec3 pos;
        glm::vec3 color;
    };

    // Offscreen targets that the cube is rendered into, the compute filter runs over, and the
    // blit pass reads from. Recreated whenever the window resizes.
    struct SceneTargets
    {
        agfx::ez::Texture2D color;
        agfx::ez::Texture2D depth;
        agfx::ez::Texture2D filtered;
        uint32_t width = 0, height = 0;

        void Create(agfx::ez::Context& ctx, uint32_t w, uint32_t h)
        {
            width = w;
            height = h;
            color = ctx.CreateTexture2D(w, h, AGFX_TEXTURE_FORMAT_RGBA16F, AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT);
            depth = ctx.CreateTexture2D(w, h, AGFX_TEXTURE_FORMAT_DEPTH32F, AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT);
            filtered = ctx.CreateTexture2D(w, h, AGFX_TEXTURE_FORMAT_RGBA16F, AGFX_TEXTURE_USAGE_STORAGE);
        }
    };
}

int main()
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return 1;
    }

#if GAME_MAC
    SDL_Window* window = SDL_CreateWindow("agfx_ez_demo", 1280, 720, SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
#else
    SDL_Window* window = SDL_CreateWindow("agfx_ez_demo", 1280, 720, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
#endif
    if (!window) {
        std::cerr << "Failed to create SDL window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

#if GAME_MAC
    SDL_MetalView metalView = SDL_Metal_CreateView(window);
    void* metalLayer = SDL_Metal_GetLayer(metalView);
#endif

    int drawableWidth = 0, drawableHeight = 0;
    SDL_GetWindowSizeInPixels(window, &drawableWidth, &drawableHeight);

    // Every agfx object below (Context and everything created through it) must be destroyed
    // before SDL_Metal_DestroyView()/SDL_DestroyWindow() release the CAMetalLayer the swap
    // chain references -- hence the explicit block scope, closed right after the game loop.
    {

    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo.allocate = Allocate;
    contextInfo.deviceInfo.free = Deallocate;
    contextInfo.deviceInfo.tempAllocate = Allocate;
    contextInfo.deviceInfo.tempFree = Deallocate;
    contextInfo.deviceInfo.enableValidation = false;
    contextInfo.width = static_cast<uint32_t>(drawableWidth);
    contextInfo.height = static_cast<uint32_t>(drawableHeight);
    contextInfo.vsync = true;
    contextInfo.hdr = false;
    contextInfo.framesInFlight = kFramesInFlight;
#if GAME_MAC
    contextInfo.windowHandle = metalLayer;
#else
    contextInfo.windowHandle = (void*)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
#endif
    agfx::ez::Context ctx(contextInfo);
    agfx::Device& device = ctx.GetDevice();

    // --- ImGui ---
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplSDL3_InitForMetal(window);

    std::string imguiSource = ReadFile((std::string(kDataDir) + "shaders/agfx_imgui/imgui.hlsl").c_str());
    // ImGui_ImplAGFX_Shutdown() destroys the shader modules it's handed at Init itself, so
    // ownership is released to it here rather than kept in an agfx::ShaderModule RAII wrapper
    // (which would double-free them once this scope ends).
    agfx::ShaderModule imguiVSOwned = CompileShader(device, imguiSource, AGFX_SHADER_STAGE_VERTEX, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfx::ShaderModule imguiPSOwned = CompileShader(device, imguiSource, AGFX_SHADER_STAGE_FRAGMENT, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);

    ImGui_ImplAGFX_InitInfo imguiInit{};
    imguiInit.Device = device;
    imguiInit.CommandQueue = ctx.GetGraphicsQueue();
    imguiInit.ColorAttachmentFormat = ctx.GetSwapChainFormat();
    imguiInit.VertexShaderModule = imguiVSOwned.Release();
    imguiInit.FragmentShaderModule = imguiPSOwned.Release();
    imguiInit.FramesInFlight = kFramesInFlight;
    if (!ImGui_ImplAGFX_Init(&imguiInit)) {
        std::cerr << "Failed to initialize ImGui AGFX backend" << std::endl;
        return 1;
    }

    // --- Shaders & pipelines ---
    std::string cubeSource = ReadFile((std::string(kDataDir) + "shaders/ez_demo/cube.hlsl").c_str());
    agfx::ShaderModule cubeVS = CompileShader(device, cubeSource, AGFX_SHADER_STAGE_VERTEX, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfx::ShaderModule cubePS = CompileShader(device, cubeSource, AGFX_SHADER_STAGE_FRAGMENT, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);

    std::string blitSource = ReadFile((std::string(kDataDir) + "shaders/ez_demo/blit.hlsl").c_str());
    agfx::ShaderModule blitVS = CompileShader(device, blitSource, AGFX_SHADER_STAGE_VERTEX, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfx::ShaderModule blitPS = CompileShader(device, blitSource, AGFX_SHADER_STAGE_FRAGMENT, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);

    std::string filterSource = ReadFile((std::string(kDataDir) + "shaders/ez_demo/color_filter.hlsl").c_str());
    agfx::ShaderModule filterCS = CompileShader(device, filterSource, AGFX_SHADER_STAGE_COMPUTE, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);

    agfxComputePipelineCreateInfo filterPipelineInfo{};
    filterPipelineInfo.name = "Color Filter";
    filterPipelineInfo.computeShader = filterCS;
    filterPipelineInfo.groupSizeX = 8;
    filterPipelineInfo.groupSizeY = 8;
    filterPipelineInfo.groupSizeZ = 1;
    agfx::ComputePipeline filterPipeline = device.CreateComputePipeline(filterPipelineInfo);

    agfx::ez::PipelineDesc cubePipelineDesc{};
    cubePipelineDesc.name = "Cube";
    cubePipelineDesc.vertexShader = &cubeVS;
    cubePipelineDesc.fragmentShader = &cubePS;
    cubePipelineDesc.cullMode = AGFX_CULL_MODE_NONE;

    agfx::ez::PipelineDesc blitPipelineDesc{};
    blitPipelineDesc.name = "Blit";
    blitPipelineDesc.vertexShader = &blitVS;
    blitPipelineDesc.fragmentShader = &blitPS;
    blitPipelineDesc.cullMode = AGFX_CULL_MODE_NONE;
    blitPipelineDesc.depthTestEnable = false;
    blitPipelineDesc.depthWriteEnable = false;

    // --- Geometry ---
    const CubeVertex cubeVertices[8] = {
        {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}},
        {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},
        {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
        {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
        {{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}},
    };
    const uint32_t cubeIndices[36] = {
        0, 1, 2, 0, 2, 3, // back
        5, 4, 7, 5, 7, 6, // front
        4, 0, 3, 4, 3, 7, // left
        1, 5, 6, 1, 6, 2, // right
        3, 2, 6, 3, 6, 7, // top
        4, 5, 1, 4, 1, 0, // bottom
    };

    agfx::ez::Buffer cubeVertexBuffer = ctx.CreateStructuredBuffer(cubeVertices, sizeof(cubeVertices), sizeof(CubeVertex));
    agfx::ez::Buffer cubeIndexBuffer = ctx.CreateIndexBuffer(cubeIndices, sizeof(cubeIndices));

    agfxSamplerCreateInfo samplerInfo{};
    samplerInfo.filter = AGFX_SAMPLER_FILTER_LINEAR;
    samplerInfo.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    agfx::Sampler linearSampler = device.CreateSampler(samplerInfo);

    SceneTargets targets;
    targets.Create(ctx, static_cast<uint32_t>(drawableWidth), static_cast<uint32_t>(drawableHeight));

    float rotationSpeed = 0.8f;
    float hueSpeed = 0.6f;
    bool running = true;
    uint64_t lastTicks = SDL_GetTicks();
    float time = 0.0f;

    while (running) {
#if GAME_MAC
        @autoreleasepool {
#endif
            uint64_t nowTicks = SDL_GetTicks();
            float dt = static_cast<float>(nowTicks - lastTicks) / 1000.0f;
            lastTicks = nowTicks;
            time += dt;

            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                ImGui_ImplSDL3_ProcessEvent(&event);
                if (event.type == SDL_EVENT_QUIT) {
                    running = false;
                } else if (event.type == SDL_EVENT_WINDOW_RESIZED || event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                    SDL_GetWindowSizeInPixels(window, &drawableWidth, &drawableHeight);
                    ctx.Resize(static_cast<uint32_t>(drawableWidth), static_cast<uint32_t>(drawableHeight));
                    targets.Create(ctx, static_cast<uint32_t>(drawableWidth), static_cast<uint32_t>(drawableHeight));
                }
            }

            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();
            ImGui::Begin("agfx_ez_demo");
            ImGui::Text("RGB cube -> compute color filter -> blit -> ImGui");
            ImGui::SliderFloat("Rotation Speed", &rotationSpeed, 0.0f, 3.0f);
            ImGui::SliderFloat("Hue Cycle Speed", &hueSpeed, 0.0f, 3.0f);
            ImGui::End();
            ImGui::Render();

            agfx::ez::Frame frame = ctx.BeginFrame();
            uint32_t w = targets.width, h = targets.height;

            // --- Pass 1: render the spinning cube into an offscreen HDR target ---
            const float clearColor[4] = {0.05f, 0.05f, 0.08f, 1.0f};
            ctx.SetRenderTargets({&targets.color}, &targets.depth, AGFX_LOAD_OPERATION_CLEAR, clearColor);
            ctx.SetViewportScissor(0.0f, 0.0f, static_cast<float>(w), static_cast<float>(h));
            ctx.SetPipeline(cubePipelineDesc);

            glm::mat4 model = glm::rotate(glm::mat4(1.0f), time * rotationSpeed, glm::vec3(0.3f, 1.0f, 0.2f));
            glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 1.2f, 2.5f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 proj = glm::perspective(glm::radians(60.0f), static_cast<float>(w) / static_cast<float>(h), 0.1f, 100.0f);
            glm::mat4 mvp = proj * view * model;

            agfx::ez::ShaderBindings cubeBindings;
            cubeBindings.Write(mvp);
            cubeBindings.BindBuffer(cubeVertexBuffer.View(AGFX_BUFFER_VIEW_TYPE_STRUCTURED));
            ctx.PushShaderBindings(cubeBindings);
            ctx.DrawIndexed(cubeIndexBuffer, 36);

            // --- Pass 2: compute color filter (reads the cube target, writes the filtered target) ---
            ctx.EndActivePass();
            ctx.TransitionTexture(targets.color, AGFX_RESOURCE_STATE_ALL_SHADER_RESOURCE);
            ctx.TransitionTexture(targets.filtered, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);
            {
                agfx::ComputePass computePass = ctx.GetCurrentCommandBuffer().BeginComputePass("Color Filter");
                computePass.SetPipeline(filterPipeline);

                agfx::ez::ShaderBindings filterBindings;
                filterBindings.BindTexture(targets.color.SRV());
                filterBindings.BindTexture(targets.filtered.UAV());
                filterBindings.BindSampler(linearSampler);
                filterBindings.Write(time * hueSpeed);
                filterBindings.Write(glm::vec2(static_cast<float>(w), static_cast<float>(h)));
                computePass.PushConstants(filterBindings.Data(), filterBindings.Size());
                computePass.Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            }
            ctx.TransitionTexture(targets.filtered, AGFX_RESOURCE_STATE_ALL_SHADER_RESOURCE);

            // --- Pass 3: blit the filtered result to the back buffer and draw ImGui on top ---
            ctx.SetBackBufferRenderTarget(AGFX_LOAD_OPERATION_CLEAR, clearColor);
            ctx.SetViewportScissor(0.0f, 0.0f, static_cast<float>(w), static_cast<float>(h));
            ctx.SetPipeline(blitPipelineDesc);

            agfx::ez::ShaderBindings blitBindings;
            blitBindings.BindTexture(targets.filtered.SRV());
            blitBindings.BindSampler(linearSampler);
            ctx.PushShaderBindings(blitBindings);
            ctx.Draw(3);

            ImGui_ImplAGFX_RenderDrawData(ImGui::GetDrawData(), ctx.GetActiveRenderPass(), w, h, ctx.CurrentFrameSlot());

            // frame ends automatically here (Frame's destructor calls Context::EndFrame())
#if GAME_MAC
        }
#endif
    }

    // ImGui_ImplAGFX_Shutdown() destroys the ImGui vertex/index buffers immediately, so the
    // GPU must be drained first -- otherwise a still in-flight command buffer referencing them
    // (e.g. still executing on Windows when the swap chain hasn't caught up) races their release.
    ctx.DrainGPU();
    ImGui_ImplAGFX_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    } // end of agfx object scope -- ctx and everything created through it are now destroyed

#if GAME_MAC
    SDL_Metal_DestroyView(metalView);
#endif
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
