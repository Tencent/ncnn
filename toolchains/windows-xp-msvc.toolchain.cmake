# Windows XP MSVC Toolchain
# Supports Visual Studio 2017 (v141) and later with XP compatibility

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 5.1)
set(CMAKE_C_COMPILER cl)
set(CMAKE_CXX_COMPILER cl)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_WIN32_WINNT=0x0501 /DWINVER=0x0501")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_WIN32_WINNT=0x0501 /DWINVER=0x0501")
set(CMAKE_GENERATOR_TOOLSET "v141_xp" CACHE STRING "Platform Toolset" FORCE)

# Force disable all advanced CPU features for Windows XP compatibility
# These must be set before CMakeLists.txt processes the options
set(NCNN_RUNTIME_CPU OFF CACHE BOOL "Disable runtime CPU dispatch for XP compatibility" FORCE)
set(NCNN_AVX OFF CACHE BOOL "Disable AVX for Windows XP" FORCE)
set(NCNN_AVX2 OFF CACHE BOOL "Disable AVX2 for Windows XP" FORCE)
set(NCNN_AVX512 OFF CACHE BOOL "Disable AVX512 for Windows XP" FORCE)
set(NCNN_AVX512VNNI OFF CACHE BOOL "Disable AVX512VNNI for Windows XP" FORCE)
set(NCNN_FMA OFF CACHE BOOL "Disable FMA for Windows XP" FORCE)
set(NCNN_F16C OFF CACHE BOOL "Disable F16C for Windows XP" FORCE)
set(NCNN_XOP OFF CACHE BOOL "Disable XOP for Windows XP" FORCE)
set(NCNN_AVXVNNI OFF CACHE BOOL "Disable AVXVNNI for Windows XP" FORCE)
set(NCNN_AVXVNNIINT8 OFF CACHE BOOL "Disable AVXVNNIINT8 for Windows XP" FORCE)
set(NCNN_AVXVNNIINT16 OFF CACHE BOOL "Disable AVXVNNIINT16 for Windows XP" FORCE)
set(NCNN_AVXNECONVERT OFF CACHE BOOL "Disable AVXNECONVERT for Windows XP" FORCE)
set(NCNN_AVX512BF16 OFF CACHE BOOL "Disable AVX512BF16 for Windows XP" FORCE)
set(NCNN_AVX512FP16 OFF CACHE BOOL "Disable AVX512FP16 for Windows XP" FORCE)

# Enable only basic SSE2 support
set(NCNN_SSE2 ON CACHE BOOL "Enable SSE2 for Windows XP" FORCE)
set(NCNN_VULKAN OFF CACHE BOOL "Disable Vulkan for Windows XP" FORCE)
set(CMAKE_GENERATOR_PLATFORM "Win32" CACHE STRING "Platform" FORCE)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "Runtime Library" FORCE)
set(NCNN_SIMPLEOCV ON CACHE BOOL "Use simple OpenCV implementation" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Windows XP MSVC Toolchain configured")
message(STATUS "  Platform Toolset: ${CMAKE_GENERATOR_TOOLSET}")
message(STATUS "  Platform: ${CMAKE_GENERATOR_PLATFORM}")
message(STATUS "  Runtime Library: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
message(STATUS "  Vulkan: ${NCNN_VULKAN}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}") 