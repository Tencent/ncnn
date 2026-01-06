
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
    expression.cpp
    gpu.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    mat_pixel_affine.cpp
    mat_pixel_drawing.cpp
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
    simplemath.cpp
    simplevk.cpp
)

if(ANDROID)
    list(APPEND ncnn_SRCS mat_pixel_android.cpp)
endif()

ncnn_src_group(ncnn_SRCS "sources")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}")

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
ncnn_add_layer(Tile)
ncnn_add_layer(RNN)
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
ncnn_add_layer(GRU)
ncnn_add_layer(MultiHeadAttention)
ncnn_add_layer(GELU)
ncnn_add_layer(Convolution1D)
ncnn_add_layer(Pooling1D)
ncnn_add_layer(ConvolutionDepthWise1D)
ncnn_add_layer(Convolution3D)
ncnn_add_layer(ConvolutionDepthWise3D)
ncnn_add_layer(Pooling3D)
ncnn_add_layer(MatMul)
ncnn_add_layer(Deconvolution1D)
ncnn_add_layer(DeconvolutionDepthWise1D)
ncnn_add_layer(Deconvolution3D)
ncnn_add_layer(DeconvolutionDepthWise3D)
ncnn_add_layer(Einsum)
ncnn_add_layer(DeformableConv2D)
ncnn_add_layer(GLU)
ncnn_add_layer(Fold)
ncnn_add_layer(Unfold)
ncnn_add_layer(GridSample)
ncnn_add_layer(CumulativeSum)
ncnn_add_layer(CopyTo)
ncnn_add_layer(Erf)
ncnn_add_layer(Diag)
ncnn_add_layer(CELU)
ncnn_add_layer(Shrink)
ncnn_add_layer(RMSNorm)
ncnn_add_layer(Spectrogram)
ncnn_add_layer(InverseSpectrogram)
ncnn_add_layer(Flip)
ncnn_add_layer(SDPA)
ncnn_add_layer(RotaryEmbed)

if(NCNN_VULKAN)
    ncnn_add_shader(${CMAKE_CURRENT_SOURCE_DIR}/convert_ycbcr.comp)
    ncnn_add_shader(${CMAKE_CURRENT_SOURCE_DIR}/layer/vulkan/shader/vulkan_activation.comp)
endif()

add_custom_target(ncnn-generate-spirv DEPENDS ${NCNN_SHADER_SPV_HEX_FILES})

# create new
configure_file(layer_declaration.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
configure_file(layer_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
configure_file(layer_type_enum.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)
configure_file(layer_shader_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_registry.h)
configure_file(layer_shader_spv_data.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_spv_data.h)
configure_file(layer_shader_type_enum.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_type_enum.h)

if(NCNN_SHARED_LIB)
    add_library(ncnn SHARED ${ncnn_SRCS})
else()
    add_library(ncnn STATIC ${ncnn_SRCS})
endif()
set_target_properties(ncnn PROPERTIES DEBUG_POSTFIX "d")
if(APPLE OR IOS)
    # macos / ios only accepts a.b.c.d.e where a=24bit b/c/d/e=10bit
    # 20201228 to 20.12.28
    string(SUBSTRING ${NCNN_VERSION} 2 2 NCNN_VERSION_YEAR)
    string(SUBSTRING ${NCNN_VERSION} 4 2 NCNN_VERSION_MONTH)
    string(SUBSTRING ${NCNN_VERSION} 6 2 NCNN_VERSION_DAY)
    set(NCNN_VERSION_APPLE_STRING ${NCNN_VERSION_MAJOR}.${NCNN_VERSION_MINOR}.${NCNN_VERSION_YEAR}.${NCNN_VERSION_MONTH}.${NCNN_VERSION_DAY})
    set_target_properties(ncnn PROPERTIES VERSION ${NCNN_VERSION_APPLE_STRING} SOVERSION ${NCNN_VERSION_MAJOR})
else()
    set_target_properties(ncnn PROPERTIES VERSION ${NCNN_VERSION_STRING} SOVERSION ${NCNN_VERSION_MAJOR})
endif()

include(GenerateExportHeader)
generate_export_header(ncnn)

if(NOT NCNN_SHARED_LIB)
    set_target_properties(ncnn PROPERTIES COMPILE_FLAGS -DNCNN_STATIC_DEFINE)
endif()

if(NCNN_SIMPLESTL AND NOT NCNN_SIMPLEMATH)
    # link math lib explicitly
    target_link_libraries(ncnn PUBLIC m)
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

if(NCNN_THREADS)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    if(TARGET Threads::Threads)
        target_link_libraries(ncnn PUBLIC Threads::Threads)
    endif()
    if(NCNN_SIMPLEOMP OR NCNN_SIMPLESTL)
        target_link_libraries(ncnn PUBLIC pthread)
    endif()
endif()

if(NCNN_VULKAN)
    if(NCNN_SIMPLEVK)
        if(APPLE)
            # simplevk always use static vulkan linkage on apple platform
            if(NOT DEFINED Vulkan_LIBRARY)
                message(WARNING "Vulkan_LIBRARY shall be defined for simplevk static linkage on APPLE platforms")
            endif()
            target_link_libraries(ncnn PUBLIC ${Vulkan_LIBRARY})
        else()
            target_link_libraries(ncnn PRIVATE ${CMAKE_DL_LIBS})
        endif()
    else()
        find_package(Vulkan QUIET)
        if(NOT Vulkan_FOUND)
            if(DEFINED ENV{VULKAN_SDK})
                if(CMAKE_SYSTEM_NAME MATCHES "Linux")
                    list(APPEND CMAKE_MODULE_PATH "$ENV{VULKAN_SDK}/../source/VulkanTools/cmake")
                elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
                    list(APPEND CMAKE_MODULE_PATH "$ENV{VULKAN_SDK}/Samples/cmake")
                elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
                    message(WARNING "Failed to find vulkan since cmake is too old\n"
                        "cmake >= 3.7 required. Consider `brew upgrade cmake`")
                endif()
            else()
                message(FATAL_ERROR "Error: CMake didn't find Vulkan. Please set VULKAN_SDK env var, e.g.:\n"
                    "Linux: export VULKAN_SDK=~/soft/vulkansdk/1.2.148.0/x86_64\n"
                    "Windows: set VULKAN_SDK=E:/lib/VulkanSDK/1.2.148.0\n"
                    "MacOS: export VULKAN_SDK=~/soft/vulkansdk/1.2.148.0/macOS\n"
                )
            endif()
            find_package(Vulkan REQUIRED)
        endif()

        target_link_libraries(ncnn PUBLIC Vulkan::Vulkan)
    endif()

    # Support mac platform static library compilation
    # https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Runtime_UserGuide.md#optionally-link-to-required-system-libraries
    if(NOT NCNN_SHARED_LIB AND APPLE)
        find_library(Metal NAMES Metal)
        find_library(Foundation NAMES Foundation)
        find_library(QuartzCore NAMES QuartzCore)
        find_library(CoreGraphics NAMES CoreGraphics)
        find_library(IOSurface NAMES IOSurface)
        list(APPEND vulkan_dependent_LINK_LIBRARIES ${Metal} ${Foundation} ${QuartzCore} ${CoreGraphics} ${IOSurface})
        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            if(NOT IOS)
                find_library(AppKit NAMES AppKit)
                list(APPEND vulkan_dependent_LINK_LIBRARIES ${AppKit})
            endif()
            find_library(IOKit NAMES IOKit)
            list(APPEND vulkan_dependent_LINK_LIBRARIES ${IOKit})
        endif()
        if(IOS OR CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "tvOS")
            find_library(UIKit NAMES UIKit)
            list(APPEND vulkan_dependent_LINK_LIBRARIES ${UIKit})
        endif()
        target_link_libraries(ncnn PRIVATE ${vulkan_dependent_LINK_LIBRARIES})
    endif()

    # link in-house glslang
    target_include_directories(ncnn PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>)
    target_link_libraries(ncnn PRIVATE glslang SPIRV)
endif()

if(NCNN_PLATFORM_API AND ANDROID)
    target_link_libraries(ncnn PUBLIC android jnigraphics log)
endif()

if(WIN32)
    target_compile_definitions(ncnn PUBLIC NOMINMAX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
    target_compile_definitions(ncnn PRIVATE _SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE)

    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        target_compile_options(ncnn PRIVATE /fp:fast)
    endif()

    if(NCNN_TARGET_ARCH STREQUAL "arm")
        # disable msvc svml optimization on arm target as it produces wrong result
        target_compile_options(ncnn PRIVATE /d2Qvec-mathlib-)
    endif()

    if(NCNN_SHARED_LIB)
        # msvc argues about stl string and vector uses in exported functions
        target_compile_options(ncnn PRIVATE /wd4251)
    endif()
else()
    target_compile_options(ncnn PRIVATE -Wall -Wextra -Wno-unused-function)

    if(NOT NCNN_DISABLE_PIC)
        set_target_properties(ncnn PROPERTIES POSITION_INDEPENDENT_CODE ON INTERFACE_POSITION_INDEPENDENT_CODE ON)
    endif()

    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.6))
            target_compile_options(ncnn PRIVATE -Ofast)
        endif()

        target_compile_options(ncnn PRIVATE -ffast-math)
    endif()

    # target_compile_options(ncnn PRIVATE -march=native)
    target_compile_options(ncnn PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    if(NCNN_SHARED_LIB AND NCNN_ENABLE_LTO)
        set_target_properties(ncnn PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()

if(NCNN_DISABLE_RTTI)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
        target_compile_options(ncnn PUBLIC /GR-)
    else()
        target_compile_options(ncnn PUBLIC -fno-rtti)
    endif()
endif()

if(NCNN_DISABLE_EXCEPTION)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
        target_compile_options(ncnn PUBLIC /EHsc /D_HAS_EXCEPTIONS=0)
    else()
        target_compile_options(ncnn PUBLIC -fno-exceptions)
    endif()
endif()

if(NCNN_TARGET_ARCH STREQUAL "x86")
    if(NCNN_SSE2)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                target_compile_options(ncnn PRIVATE /arch:SSE2)
            endif()
            target_compile_options(ncnn PRIVATE /D__SSE2__)
        else()
            if(NOT CMAKE_SYSTEM_NAME MATCHES "WASI")
                target_compile_options(ncnn PRIVATE -msse2 -msse)
            endif ()
            if(CMAKE_SYSTEM_NAME MATCHES "Emscripten|WASI")
                target_compile_options(ncnn PRIVATE -msimd128)
            endif()
        endif()

        if(NCNN_COMPILER_SUPPORT_X86_RECIP_NONE)
            # recip optimization causes precision loss
            target_compile_options(ncnn PRIVATE -mrecip=none)
        endif()
    endif()

    if(NOT NCNN_RUNTIME_CPU AND NCNN_AVX512)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(ncnn PRIVATE /arch:AVX512 /D__SSSE3__ /D__SSE4_1__ /D__FMA__ /D__F16C__)
            if(NCNN_AVX512VNNI)
                target_compile_options(ncnn PRIVATE /D__AVX512VNNI__)
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
            target_compile_options(ncnn PRIVATE /arch:AVX512 -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mfma -mf16c /D__SSSE3__ /D__SSE4_1__ /D__FMA__ /D__F16C__)
            if(NCNN_AVX512VNNI)
                target_compile_options(ncnn PRIVATE -mavx512vnni /D__AVX512VNNI__)
            endif()
            if(NCNN_AVX512BF16)
                target_compile_options(ncnn PRIVATE -mavx512bf16 /D__AVX512BF16__)
            endif()
            if(NCNN_AVX512FP16)
                target_compile_options(ncnn PRIVATE -mavx512fp16 /D__AVX512FP16__)
            endif()
        else()
            target_compile_options(ncnn PRIVATE -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mfma -mf16c)
            if(NCNN_AVX512VNNI)
                target_compile_options(ncnn PRIVATE -mavx512vnni)
            endif()
            if(NCNN_AVX512BF16)
                target_compile_options(ncnn PRIVATE -mavx512bf16)
            endif()
            if(NCNN_AVX512FP16)
                target_compile_options(ncnn PRIVATE -mavx512fp16)
            endif()
        endif()
    elseif(NOT NCNN_RUNTIME_CPU AND NCNN_FMA)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            if(NCNN_AVX2)
                target_compile_options(ncnn PRIVATE /arch:AVX2 /D__SSSE3__ /D__SSE4_1__ /D__FMA__)
            else()
                target_compile_options(ncnn PRIVATE /arch:AVX /D__SSSE3__ /D__SSE4_1__ /D__FMA__)
            endif()
            if(NCNN_AVXVNNIINT8)
                target_compile_options(ncnn PRIVATE /D__AVXVNNIINT8__)
            endif()
            if(NCNN_AVXVNNIINT16)
                target_compile_options(ncnn PRIVATE /D__AVXVNNIINT16__)
            endif()
            if(NCNN_AVXNECONVERT)
                target_compile_options(ncnn PRIVATE /D__AVXNECONVERT__)
            endif()
            if(NCNN_AVXVNNI)
                target_compile_options(ncnn PRIVATE /D__AVXVNNI__)
            elseif(NCNN_XOP)
                target_compile_options(ncnn PRIVATE /D__XOP__)
            endif()
            if(NCNN_F16C)
                target_compile_options(ncnn PRIVATE /D__F16C__)
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
            if(NCNN_AVX2)
                target_compile_options(ncnn PRIVATE /arch:AVX2 -mfma /D__SSSE3__ /D__SSE4_1__ /D__FMA__)
            else()
                target_compile_options(ncnn PRIVATE /arch:AVX -mfma /D__SSSE3__ /D__SSE4_1__ /D__FMA__)
            endif()
            if(NCNN_AVXVNNIINT8)
                target_compile_options(ncnn PRIVATE -mavxvnniint8 /D__AVXVNNIINT8__)
            endif()
            if(NCNN_AVXVNNIINT16)
                target_compile_options(ncnn PRIVATE -mavxvnniint16 /D__AVXVNNIINT16__)
            endif()
            if(NCNN_AVXNECONVERT)
                target_compile_options(ncnn PRIVATE -mavxneconvert /D__AVXNECONVERT__)
            endif()
            if(NCNN_AVXVNNI)
                target_compile_options(ncnn PRIVATE -mavxvnni /D__AVXVNNI__)
            elseif(NCNN_XOP)
                target_compile_options(ncnn PRIVATE -mxop /D__XOP__)
            endif()
            if(NCNN_F16C)
                target_compile_options(ncnn PRIVATE -mf16c /D__F16C__)
            endif()
        else()
            if(NCNN_AVX2)
                target_compile_options(ncnn PRIVATE -mavx2 -mfma)
            else()
                target_compile_options(ncnn PRIVATE -mavx -mfma)
            endif()
            if(NCNN_AVXVNNIINT8)
                target_compile_options(ncnn PRIVATE -mavxvnniint8)
            endif()
            if(NCNN_AVXVNNIINT16)
                target_compile_options(ncnn PRIVATE -mavxvnniint16)
            endif()
            if(NCNN_AVXNECONVERT)
                target_compile_options(ncnn PRIVATE -mavxneconvert)
            endif()
            if(NCNN_AVXVNNI)
                target_compile_options(ncnn PRIVATE -mavxvnni)
            elseif(NCNN_XOP)
                target_compile_options(ncnn PRIVATE -mxop)
            endif()
            if(NCNN_F16C)
                target_compile_options(ncnn PRIVATE -mf16c)
            endif()
        endif()
    elseif(NOT NCNN_RUNTIME_CPU AND NCNN_AVX)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(ncnn PRIVATE /arch:AVX /D__SSSE3__ /D__SSE4_1__)
            if(NCNN_XOP)
                target_compile_options(ncnn PRIVATE /D__XOP__)
            endif()
            if(NCNN_F16C)
                target_compile_options(ncnn PRIVATE /D__F16C__)
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
            target_compile_options(ncnn PRIVATE /arch:AVX /D__SSSE3__ /D__SSE4_1__)
            if(NCNN_XOP)
                target_compile_options(ncnn PRIVATE -mxop /D__XOP__)
            endif()
            if(NCNN_F16C)
                target_compile_options(ncnn PRIVATE -mf16c /D__F16C__)
            endif()
        else()
            target_compile_options(ncnn PRIVATE -mavx)
            if(NCNN_XOP)
                target_compile_options(ncnn PRIVATE -mxop)
            endif()
            if(NCNN_F16C)
                target_compile_options(ncnn PRIVATE -mf16c)
            endif()
        endif()
    endif()
endif()

if(NCNN_TARGET_ARCH STREQUAL "arm" AND (CMAKE_SIZEOF_VOID_P EQUAL 4 AND NOT NCNN_TARGET_ILP32))
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
        # always enable neon for msvc arm
        target_compile_options(ncnn PRIVATE /D__arm__ /D__ARM_NEON)
    endif()

    if(NOT NCNN_RUNTIME_CPU AND NCNN_VFPV4)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(ncnn PRIVATE /arch:VFPv4 /D__ARM_FP=0x0E)
        else()
            if(NCNN_COMPILER_SUPPORT_ARM_VFPV4)
                target_compile_options(ncnn PRIVATE -mfpu=neon-vfpv4)
            elseif(NCNN_COMPILER_SUPPORT_ARM_VFPV4_FP16)
                target_compile_options(ncnn PRIVATE -mfpu=neon-vfpv4 -mfp16-format=ieee)
            endif()
        endif()
    elseif(NOT NCNN_RUNTIME_CPU)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(ncnn PRIVATE /D__ARM_FP=0x0C)
        endif()
    endif()
endif()

if(NCNN_TARGET_ARCH STREQUAL "arm" AND (CMAKE_SIZEOF_VOID_P EQUAL 8 OR NCNN_TARGET_ILP32))
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
        # always enable neon and vfpv4 for msvc arm64
        target_compile_options(ncnn PRIVATE /D__arm__ /D__aarch64__ /D__ARM_NEON /D__ARM_FP=0x0E)
    endif()

    if(NOT NCNN_RUNTIME_CPU AND NCNN_ARM86SVE)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            # TODO add support for sve family
            target_compile_options(ncnn PRIVATE /arch:armv8.6 /D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC /D__ARM_FEATURE_DOTPROD /D__ARM_FEATURE_FP16_FML /D__ARM_FEATURE_BF16_VECTOR_ARITHMETIC /D__ARM_FEATURE_MATMUL_INT8)
            if(NCNN_ARM86SVE2)
            endif()
            if(NCNN_ARM86SVEBF16)
            endif()
            if(NCNN_ARM86SVEI8MM)
            endif()
            if(NCNN_ARM86SVEF32MM)
            endif()
        endif()
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(ARM_MARCH_FLAG "-march=armv8.6-a+fp16+dotprod+sve")
            if(NCNN_ARM86SVE2)
                set(ARM_MARCH_FLAG "${ARM_MARCH_FLAG}+sve2")
            endif()
            if(NCNN_ARM86SVEBF16)
                set(ARM_MARCH_FLAG "${ARM_MARCH_FLAG}+bf16")
            endif()
            if(NCNN_ARM86SVEI8MM)
                set(ARM_MARCH_FLAG "${ARM_MARCH_FLAG}+i8mm")
            endif()
            if(NCNN_ARM86SVEF32MM)
                set(ARM_MARCH_FLAG "${ARM_MARCH_FLAG}+f32mm")
            endif()
        endif()
    elseif(NOT NCNN_RUNTIME_CPU AND (NCNN_ARM84BF16 OR NCNN_ARM84I8MM))
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(ncnn PRIVATE /arch:armv8.4 /D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC /D__ARM_FEATURE_DOTPROD /D__ARM_FEATURE_FP16_FML)
            if(NCNN_ARM84BF16)
                target_compile_options(ncnn PRIVATE /D__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
            endif()
            if(NCNN_ARM84I8MM)
                target_compile_options(ncnn PRIVATE /D__ARM_FEATURE_MATMUL_INT8)
            endif()
        endif()
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(ARM_MARCH_FLAG "-march=armv8.4-a+fp16+dotprod")
            if(NCNN_ARM84BF16)
                set(ARM_MARCH_FLAG "${ARM_MARCH_FLAG}+bf16")
            endif()
            if(NCNN_ARM84I8MM)
                set(ARM_MARCH_FLAG "${ARM_MARCH_FLAG}+i8mm")
            endif()
        endif()
    elseif(NOT NCNN_RUNTIME_CPU AND NCNN_ARM82)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(ncnn PRIVATE /arch:armv8.2 /D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            if(NCNN_ARM82DOT)
                target_compile_options(ncnn PRIVATE /D__ARM_FEATURE_DOTPROD)
            endif()
            if(NCNN_ARM82FP16FML)
                target_compile_options(ncnn PRIVATE /D__ARM_FEATURE_FP16_FML)
            endif()
        endif()
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(ARM_MARCH_FLAG "-march=armv8.2-a+fp16")
            if(NCNN_ARM82DOT)
                set(ARM_MARCH_FLAG "${ARM_MARCH_FLAG}+dotprod")
            endif()
            if(NCNN_ARM82FP16FML)
                set(ARM_MARCH_FLAG "${ARM_MARCH_FLAG}+fp16fml")
                # clang 9.0.9 shipped with android ndk-r21 is missing __ARM_FEATURE_FP16_FML macro for asimdfhm target
                target_compile_options(ncnn PRIVATE -D__ARM_FEATURE_FP16_FML)
            endif()
        endif()
    endif()
    target_compile_options(ncnn PRIVATE ${ARM_MARCH_FLAG})

    if(ANDROID_NDK_MAJOR AND (ANDROID_NDK_MAJOR GREATER_EQUAL 23))
        # llvm 12 in ndk-23 enables out-of-line atomics by default
        # disable this feature for fixing linking atomic builtins issue with old ndk
        target_compile_options(ncnn PRIVATE -mno-outline-atomics)
    endif()
endif()

if(NCNN_TARGET_ARCH STREQUAL "mips")
    if(NOT NCNN_RUNTIME_CPU AND NCNN_MSA)
        target_compile_options(ncnn PRIVATE -mmsa)
    endif()
    if(NOT NCNN_RUNTIME_CPU AND NCNN_MMI)
        target_compile_options(ncnn PRIVATE -mloongson-mmi)
    endif()
endif()

if(NCNN_TARGET_ARCH STREQUAL "loongarch")
    if(NOT NCNN_RUNTIME_CPU AND NCNN_LSX)
        target_compile_options(ncnn PRIVATE -mlsx)
        if(NCNN_LASX)
            target_compile_options(ncnn PRIVATE -mlasx)
        endif()
    endif()
endif()

if(NCNN_TARGET_ARCH STREQUAL "riscv" AND NOT C906)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        if(NOT NCNN_RUNTIME_CPU AND NCNN_RVV)
            set(RISCV_MARCH_FLAG "-march=rv64gcv")
            if(NCNN_ZFH)
                set(RISCV_MARCH_FLAG "${RISCV_MARCH_FLAG}_zfh")
                target_compile_options(ncnn PRIVATE -D__fp16=_Float16)
            endif()
            if(NCNN_ZVFH)
                set(RISCV_MARCH_FLAG "${RISCV_MARCH_FLAG}_zvfh")
            endif()
        elseif(NOT NCNN_RUNTIME_CPU AND NCNN_XTHEADVECTOR)
            set(RISCV_MARCH_FLAG "-march=rv64gc_xtheadvector")
            if(NCNN_ZFH)
                set(RISCV_MARCH_FLAG "${RISCV_MARCH_FLAG}_zfh")
                target_compile_options(ncnn PRIVATE -D__riscv_zvfh=1 -D__fp16=_Float16)
            endif()
        endif()
        target_compile_options(ncnn PRIVATE ${RISCV_MARCH_FLAG})
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        if(NOT NCNN_RUNTIME_CPU AND NCNN_RVV)
            set(RISCV_MARCH_FLAG "-march=rv32gcv")
            if(NCNN_ZFH)
                set(RISCV_MARCH_FLAG "${RISCV_MARCH_FLAG}_zfh")
                target_compile_options(ncnn PRIVATE -D__fp16=_Float16)
            endif()
            if(NCNN_ZVFH)
                set(RISCV_MARCH_FLAG "${RISCV_MARCH_FLAG}_zvfh")
            endif()
        elseif(NOT NCNN_RUNTIME_CPU AND NCNN_XTHEADVECTOR)
            set(RISCV_MARCH_FLAG "-march=rv32gc_xtheadvector")
            if(NCNN_ZFH)
                set(RISCV_MARCH_FLAG "${RISCV_MARCH_FLAG}_zfh")
                target_compile_options(ncnn PRIVATE -D__riscv_zvfh=1 -D__fp16=_Float16)
            endif()
        endif()
        target_compile_options(ncnn PRIVATE ${RISCV_MARCH_FLAG})
    endif()
endif()

if(NCNN_PPC64LE_VSX)
    # Auto-translate SSE2 to VSX if compiler is new enough.
    if(NCNN_VSX_SSE2)
        target_compile_options(ncnn PRIVATE -DNO_WARN_X86_INTRINSICS -D__SSE2__)
    endif()

    # Auto-translate SSE4.1 to VSX if compiler is new enough.
    if(NCNN_VSX_SSE41)
        target_compile_options(ncnn PRIVATE -DNO_WARN_X86_INTRINSICS -D__SSE4_1__)
    endif()
endif()

if(NCNN_COVERAGE)
    target_compile_options(ncnn PUBLIC -coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(ncnn PUBLIC -coverage -lgcov)
endif()

if(NCNN_ASAN)
    target_compile_options(ncnn PUBLIC -fsanitize=address)
    target_link_libraries(ncnn PUBLIC -fsanitize=address)
endif()

add_dependencies(ncnn ncnn-generate-spirv)

if(NCNN_INSTALL_SDK)
    include(GNUInstallDirs)

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ncnnConfigVersion.cmake
        VERSION ${NCNN_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    install(TARGETS ncnn EXPORT ncnn
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(FILES
        allocator.h
        benchmark.h
        blob.h
        c_api.h
        command.h
        cpu.h
        datareader.h
        expression.h
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
        simplemath.h
        simplevk.h
        vulkan_header_fix.h
        ${CMAKE_CURRENT_BINARY_DIR}/ncnn_export.h
        ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_type_enum.h
        ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h
        ${CMAKE_CURRENT_BINARY_DIR}/platform.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ncnn
    )
    install(EXPORT ncnn DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ncnn)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../cmake/ncnnConfig.cmake.in ncnnConfig.cmake @ONLY)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ncnnConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ncnnConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ncnn)
    # pkgconfig
    configure_file(ncnn.pc.in ${CMAKE_CURRENT_BINARY_DIR}/ncnn.pc @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ncnn.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

# add ncnn and generate-spirv to a virtual project group
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET ncnn PROPERTY FOLDER "libncnn")
set_property(TARGET ncnn-generate-spirv PROPERTY FOLDER "libncnn")
