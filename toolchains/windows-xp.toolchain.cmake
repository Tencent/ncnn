# using MinGW-w64 with appropriate flags

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_FLAGS "-DWIN32_LEAN_AND_MEAN -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-DWIN32_LEAN_AND_MEAN -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

set(NCNN_RUNTIME_CPU OFF CACHE BOOL "Disable runtime CPU dispatch for XP compatibility" FORCE)
set(NCNN_AVX OFF CACHE BOOL "Disable AVX for XP compatibility" FORCE)
set(NCNN_AVX2 OFF CACHE BOOL "Disable AVX2 for XP compatibility" FORCE)
set(NCNN_AVX512 OFF CACHE BOOL "Disable AVX512 for XP compatibility" FORCE)
set(NCNN_FMA OFF CACHE BOOL "Disable FMA for XP compatibility" FORCE)
set(NCNN_F16C OFF CACHE BOOL "Disable F16C for XP compatibility" FORCE)

set(NCNN_SSE2 ON CACHE BOOL "Enable SSE2 for XP compatibility" FORCE)
set(NCNN_VULKAN OFF CACHE BOOL "Disable Vulkan for XP compatibility" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc -static-libstdc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++")

set(CMAKE_GENERATOR_PLATFORM Win32 CACHE STRING "Force 32-bit build for XP compatibility" FORCE) 