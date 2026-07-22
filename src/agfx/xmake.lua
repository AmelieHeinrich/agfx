-- 
-- @ Author: Amélie Heinrich @ Amélie Heinrich
-- @ Create Time: 2026-07-18 20:59:28
-- @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
-- 

target("agfx_shader")
    set_kind("static")
    
    if is_plat("macosx") then
        add_files("agfx_shader/agfx_shader_compiler_mac.mm")
        add_links("metalirconverter", "dxcompiler")
    elseif is_plat("windows") then
        add_files("agfx_shader/agfx_shader_compiler_windows.cpp")
        add_syslinks("dxcompiler", { public = true })
    end

target("agfx_shader_cli")
    set_kind("binary")
    add_files("agfx_shader_cli/**.cpp")

    add_deps("agfx_shader")

target("agfx")
    set_kind("static")

    if is_plat("windows") then
        set_runtimes("MT")
    end

    if is_plat("macosx") then
        add_files("agfx/agfx_metal4.mm")
        add_frameworks("Metal", "QuartzCore", "CoreGraphics", { public = true })
    elseif is_plat("windows") then
        add_files("agfx/agfx_d3d12.cpp")
        add_syslinks("d3d12", "dxgi", "dxguid", { public = true })
    end

target("agfx_imgui")
    set_kind("static")
    add_files("agfx_imgui/*.cpp")

    add_deps("agfx", "agfx_shader")
    add_packages("imgui", { public = true })

target("agfx_demo")
    set_kind("binary")
    add_files("agfx_demo/**.cpp")

    add_deps("agfx", "agfx_shader", "agfx_imgui")
    add_packages("libsdl3", "cgltf", "stb", "imgui", "glm", { public = true })

target("agfx_tests")
    set_kind("binary")
    add_files("agfx_tests/**.cpp")

    add_deps("agfx", "agfx_shader")
    add_packages("stb", "nlohmann_json", "glm", { public = true })

target("agfx_ez_demo")
    set_kind("binary")
    add_files("agfx_ez_demo/**.cpp")
    add_files("agfx_demo/imgui_impl_sdl3.cpp")
    add_includedirs("agfx_demo")

    add_deps("agfx", "agfx_shader", "agfx_imgui")
    add_packages("libsdl3", "imgui", "glm", { public = true })
