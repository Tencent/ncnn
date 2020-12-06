
##############################################

configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h)

# Add source file to list, and add to special visual folder
function(ncnn_src_group ncnn_src_string folder)
    string(REPLACE " " ";" _ncnn_src_list ${ncnn_src_string})

    string(REGEX REPLACE "/" "\\\\" _target_folder "${folder}")

    foreach(_file IN LISTS ${_ncnn_src_list})
        source_group ("${_target_folder}" FILES "${_file}")
    endforeach ()
endfunction()

set(ncnn_SRCS
    allocator.cpp
    benchmark.cpp
    blob.cpp
    c_api.cpp
    command.cpp
    cpu.cpp
    datareader.cpp
    gpu.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    mat_pixel_affine.cpp
    mat_pixel_resize.cpp
    mat_pixel_rotate.cpp
    modelbin.cpp
    net.cpp
    option.cpp
    paramdict.cpp
    pipeline.cpp
    pipelinecache.cpp
    simpleocv.cpp
    simpleomp.cpp
    simplestl.cpp
)

if(ANDROID)
    list(APPEND ncnn_SRCS mat_pixel_android.cpp)
endif()

ncnn_src_group(ncnn_SRCS "sources")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}")

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_shader_spv_header.cmake)

# ncnn macro
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_add_shader.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_add_layer.cmake)

# look for vulkan compute shader and compile
set(NCNN_SHADER_SPV_HEX_FILES)

set(__LAYER_TYPE_ENUM_INDEX 0)
set(__LAYER_SHADER_TYPE_ENUM_INDEX 0)

# layer implementation
ncnn_add_layer(AbsVal)
ncnn_add_layer(ArgMax OFF)
ncnn_add_layer(BatchNorm)
ncnn_add_layer(Bias)
ncnn_add_layer(BNLL)
ncnn_add_layer(Concat)
ncnn_add_layer(Convolution)
ncnn_add_layer(Crop)
ncnn_add_layer(Deconvolution)
ncnn_add_layer(Dropout)
ncnn_add_layer(Eltwise)
ncnn_add_layer(ELU)
ncnn_add_layer(Embed)
ncnn_add_layer(Exp)
ncnn_add_layer(Flatten)
ncnn_add_layer(InnerProduct)
ncnn_add_layer(Input)
ncnn_add_layer(Log)
ncnn_add_layer(LRN)
ncnn_add_layer(MemoryData)
ncnn_add_layer(MVN)
ncnn_add_layer(Pooling)
ncnn_add_layer(Power)
ncnn_add_layer(PReLU)
ncnn_add_layer(Proposal)
ncnn_add_layer(Reduction)
ncnn_add_layer(ReLU)
ncnn_add_layer(Reshape)
ncnn_add_layer(ROIPooling)
ncnn_add_layer(Scale)
ncnn_add_layer(Sigmoid)
ncnn_add_layer(Slice)
ncnn_add_layer(Softmax)
ncnn_add_layer(Split)
ncnn_add_layer(SPP OFF)
ncnn_add_layer(TanH)
ncnn_add_layer(Threshold)
ncnn_add_layer(Tile OFF)
ncnn_add_layer(RNN OFF)
ncnn_add_layer(LSTM)
ncnn_add_layer(BinaryOp)
ncnn_add_layer(UnaryOp)
ncnn_add_layer(ConvolutionDepthWise)
ncnn_add_layer(Padding)
ncnn_add_layer(Squeeze)
ncnn_add_layer(ExpandDims)
ncnn_add_layer(Normalize)
ncnn_add_layer(Permute)
ncnn_add_layer(PriorBox)
ncnn_add_layer(DetectionOutput)
ncnn_add_layer(Interp)
ncnn_add_layer(DeconvolutionDepthWise)
ncnn_add_layer(ShuffleChannel)
ncnn_add_layer(InstanceNorm)
ncnn_add_layer(Clip)
ncnn_add_layer(Reorg)
ncnn_add_layer(YoloDetectionOutput)
ncnn_add_layer(Quantize)
ncnn_add_layer(Dequantize)
ncnn_add_layer(Yolov3DetectionOutput)
ncnn_add_layer(PSROIPooling)
ncnn_add_layer(ROIAlign)
ncnn_add_layer(Packing)
ncnn_add_layer(Requantize)
ncnn_add_layer(Cast)
ncnn_add_layer(HardSigmoid)
ncnn_add_layer(SELU)
ncnn_add_layer(HardSwish)
ncnn_add_layer(Noop)
ncnn_add_layer(PixelShuffle)
ncnn_add_layer(DeepCopy)
ncnn_add_layer(Mish)
ncnn_add_layer(StatisticsPooling)
ncnn_add_layer(Swish)
ncnn_add_layer(Gemm)
ncnn_add_layer(GroupNorm)
ncnn_add_layer(LayerNorm)
ncnn_add_layer(Softplus)

if(NCNN_VULKAN)
    ncnn_add_shader(${CMAKE_CURRENT_SOURCE_DIR}/convert_ycbcr.comp)
endif()

add_custom_target(ncnn-generate-spirv DEPENDS ${NCNN_SHADER_SPV_HEX_FILES})

# create new
configure_file(layer_declaration.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
configure_file(layer_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
configure_file(layer_registry_avx2.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_registry_avx2.h)
configure_file(layer_registry_arm82.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_registry_arm82.h)
configure_file(layer_type_enum.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)
configure_file(layer_shader_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_registry.h)
configure_file(layer_shader_spv_data.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_spv_data.h)
configure_file(layer_shader_type_enum.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_type_enum.h)

add_library(ncnn STATIC ${ncnn_SRCS})
set_target_properties(ncnn PROPERTIES DEBUG_POSTFIX "d")

if(NCNN_SIMPLESTL)
    # link math lib explicitly
    target_link_libraries(ncnn PUBLIC m)

    # do not link gcc lib on darwin and emscripten
    if(CMAKE_CXX_COMPILER_ID MATCHES "(Clang|GNU)" AND NOT CMAKE_SYSTEM_NAME MATCHES "(Darwin|Emscripten)")
        # for cpu_support_x86_avx2
        target_link_libraries(ncnn PUBLIC gcc)
    endif()
endif()

target_include_directories(ncnn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/ncnn>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/layer>)

if(NCNN_OPENMP)
    if(NOT NCNN_SIMPLEOMP)
        find_package(OpenMP)
        if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
            target_compile_options(ncnn PRIVATE ${OpenMP_CXX_FLAGS})
        endif()
    endif()

    if(NCNN_SIMPLEOMP OR OpenMP_CXX_FOUND OR OPENMP_FOUND)
        if(NCNN_CMAKE_VERBOSE)
            message("Building with OpenMP")
        endif()

        if(NCNN_SIMPLEOMP)
            if(IOS OR APPLE)
                target_compile_options(ncnn PRIVATE -Xpreprocessor -fopenmp)
            else()
                target_compile_options(ncnn PRIVATE -fopenmp)
            endif()
            target_link_libraries(ncnn PUBLIC pthread)
        elseif(ANDROID_NDK_MAJOR AND (ANDROID_NDK_MAJOR GREATER 20))
            target_compile_options(ncnn PRIVATE -fopenmp)
            target_link_libraries(ncnn PUBLIC -fopenmp -static-openmp)
        elseif(OpenMP_CXX_FOUND)
            target_link_libraries(ncnn PUBLIC OpenMP::OpenMP_CXX)
        else()
            target_link_libraries(ncnn PRIVATE "${OpenMP_CXX_FLAGS}")
        endif()
    endif()
endif()

if(NCNN_VULKAN)
    find_package(Vulkan QUIET)
    if(NOT Vulkan_FOUND)
        message(STATUS "=== CMAKE_SYSTEM_NAME is: ${CMAKE_SYSTEM_NAME}")
        if(DEFINED ENV{VULKAN_SDK})
            if(CMAKE_SYSTEM_NAME MATCHES "Linux")
                list(APPEND CMAKE_MODULE_PATH "$ENV{VULKAN_SDK}/../source/VulkanTools/cmake")
            elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
                list(APPEND CMAKE_MODULE_PATH "$ENV{VULKAN_SDK}/Samples/cmake")
            elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
                message(WARNING "Failed to find vulkan since cmake too old\n"
                    "cmake >= 3.7 required. Consider `brew upgrade cmake`")
            endif()
        else()
            message(FATAL_ERROR "!! CMake didn't find Vulkan. Please set VULKAN_SDK env var, e.g.:\n"
                "Linux: export VULKAN_SDK=~/soft/vulkansdk/1.2.148.0/x86_64\n"
                "Windows: set VULKAN_SDK=E:/lib/VulkanSDK/1.2.148.0\n"
                "MacOS: export VULKAN_SDK=~/soft/vulkansdk/1.2.148.0/macOS\n"
            )
        endif()
        find_package(Vulkan REQUIRED)
    endif()

    target_link_libraries(ncnn PUBLIC Vulkan::Vulkan)

    target_include_directories(ncnn PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>)
    target_link_libraries(ncnn PRIVATE glslang SPIRV)
endif()

if(ANDROID_NDK)
    target_link_libraries(ncnn PUBLIC android log)
endif()

if(WIN32)
    target_compile_definitions(ncnn PUBLIC NOMINMAX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC"))
    target_compile_definitions(ncnn PRIVATE _SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE)

    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        target_compile_options(ncnn PRIVATE /fp:fast)
    endif()
else()
    target_compile_options(ncnn PRIVATE -Wall -Wextra -Wno-unused-function)

    if(NOT NCNN_DISABLE_PIC)
        set_target_properties(ncnn PROPERTIES POSITION_INDEPENDENT_CODE ON INTERFACE_POSITION_INDEPENDENT_CODE ON)
    endif()

    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
            target_compile_options(ncnn PRIVATE -Ofast)
        endif()

        target_compile_options(ncnn PRIVATE -ffast-math)
    endif()

    if(PI3)
        target_compile_options(ncnn PRIVATE -march=native -mfpu=neon -mfloat-abi=hard)
        target_compile_definitions(ncnn PRIVATE __ARM_NEON __ANDROID__)
    endif()
    # target_compile_options(ncnn PRIVATE -march=native)
    # set_target_properties(ncnn PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    target_compile_options(ncnn PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif()

if(NCNN_DISABLE_RTTI)
    target_compile_options(ncnn PUBLIC -fno-rtti)
endif()

if(NCNN_DISABLE_EXCEPTION)
    target_compile_options(ncnn PUBLIC -fno-exceptions)
endif()

if(NCNN_TARGET_ARCH STREQUAL "x86")
    if(NCNN_SSE2)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC"))
            target_compile_options(ncnn PRIVATE /arch:SSE2 /D__SSE2__)
        else()
            target_compile_options(ncnn PRIVATE -msse2 -msse)
            if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
                target_compile_options(ncnn PRIVATE -msimd128)
            endif()
        endif()
    endif()

    if(NOT NCNN_RUNTIME_CPU AND NCNN_AVX2)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC"))
            target_compile_options(ncnn PRIVATE /arch:AVX2)
        else()
            target_compile_options(ncnn PRIVATE -mfma -mf16c -mavx2)
        endif()
    endif()
endif()

if(((IOS AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64") OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64)")))
    if (NOT NCNN_RUNTIME_CPU AND NCNN_ARM82)
        target_compile_options(ncnn PRIVATE -march=armv8.2-a+fp16)
    endif()
endif()

if(NCNN_COVERAGE)
    target_compile_options(ncnn PUBLIC -coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(ncnn PUBLIC -coverage -lgcov)
endif()

if(ANDROID OR IOS)
    # disable shared library on android and xcode ios
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
endif()

add_dependencies(ncnn ncnn-generate-spirv)

if(NCNN_INSTALL_SDK)
    install(TARGETS ncnn EXPORT ncnn ARCHIVE DESTINATION lib)
    install(FILES
        allocator.h
        benchmark.h
        blob.h
        c_api.h
        command.h
        cpu.h
        datareader.h
        gpu.h
        layer.h
        layer_shader_type.h
        layer_type.h
        mat.h
        modelbin.h
        net.h
        option.h
        paramdict.h
        pipeline.h
        pipelinecache.h
        simpleocv.h
        simpleomp.h
        simplestl.h
        ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_type_enum.h
        ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h
        ${CMAKE_CURRENT_BINARY_DIR}/platform.h
        DESTINATION include/ncnn
    )
    install(EXPORT ncnn DESTINATION lib/cmake/ncnn)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../cmake/ncnnConfig.cmake.in ncnnConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ncnnConfig.cmake DESTINATION lib/cmake/ncnn)
endif()

# add ncnn and generate-spirv to a virtual project group
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET ncnn PROPERTY FOLDER "libncnn")
set_property(TARGET ncnn-generate-spirv PROPERTY FOLDER "libncnn")
