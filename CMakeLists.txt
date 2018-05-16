
if(CMAKE_TOOLCHAIN_FILE)
set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
# get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 2.8.10)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

project(ncnn)

option(NCNN_OPENMP "openmp support" ON)
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)
option(NCNN_OPENCV "minimal opencv structure emulation" OFF)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)

if(NCNN_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

if(WIN32)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
else()
    add_definitions(-Wall -Wextra -Wno-unused-function)

    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        add_definitions(-fPIC)
        add_definitions(-Ofast)

        add_definitions(-ffast-math)
    endif()
    # add_definitions(-march=native)

    # add_definitions(-flto)

    add_definitions(-fvisibility=hidden -fvisibility-inlines-hidden)
endif()

if(ANDROID)
    # disable shared library on android
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")
elseif(IOS)
    # disable shared library on xcode ios
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")
endif()

##############################################

# add_subdirectory(examples)
# add_subdirectory(benchmark)
add_subdirectory(src)
if(NOT ANDROID AND NOT IOS)
add_subdirectory(tools)
endif()
