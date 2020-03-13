
##############################################

configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h)

if(NCNN_VULKAN)
    find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH)
    message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")
endif()

set(ncnn_SRCS
    allocator.cpp
    blob.cpp
    command.cpp
    cpu.cpp
    datareader.cpp
    gpu.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    mat_pixel_resize.cpp
    mat_pixel_rotate.cpp
    modelbin.cpp
    net.cpp
    opencv.cpp
    option.cpp
    paramdict.cpp
    pipeline.cpp
    benchmark.cpp
)

if(ANDROID)
    list(APPEND ncnn_SRCS mat_pixel_android.cpp)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/ncnn_generate_shader_spv_header.cmake)

macro(ncnn_add_shader SHADER_SRC)
    ncnn_generate_shader_spv_header(SHADER_SPV_HEADER SHADER_SPV_HEX_HEADERS ${SHADER_SRC})

    get_filename_component(SHADER_SPV_HEADER_NAME ${SHADER_SPV_HEADER} NAME)
    string(APPEND layer_shader_spv_data "#include \"${SHADER_SPV_HEADER_NAME}\"\n")

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    string(APPEND layer_shader_registry "{\"${SHADER_SRC_NAME_WE}\",${SHADER_SRC_NAME_WE}_spv_data,sizeof(${SHADER_SRC_NAME_WE}_spv_data)},\n")
    string(APPEND layer_shader_registry "{\"${SHADER_SRC_NAME_WE}_fp16p\",${SHADER_SRC_NAME_WE}_fp16p_spv_data,sizeof(${SHADER_SRC_NAME_WE}_fp16p_spv_data)},\n")
    string(APPEND layer_shader_registry "{\"${SHADER_SRC_NAME_WE}_fp16pa\",${SHADER_SRC_NAME_WE}_fp16pa_spv_data,sizeof(${SHADER_SRC_NAME_WE}_fp16pa_spv_data)},\n")
    string(APPEND layer_shader_registry "{\"${SHADER_SRC_NAME_WE}_fp16s\",${SHADER_SRC_NAME_WE}_fp16s_spv_data,sizeof(${SHADER_SRC_NAME_WE}_fp16s_spv_data)},\n")
    string(APPEND layer_shader_registry "{\"${SHADER_SRC_NAME_WE}_fp16sa\",${SHADER_SRC_NAME_WE}_fp16sa_spv_data,sizeof(${SHADER_SRC_NAME_WE}_fp16sa_spv_data)},\n")

    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEADER})
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_HEADERS})
endmacro()

macro(ncnn_add_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    if(NCNN_CMAKE_VERBOSE)
        message(STATUS "WITH_LAYER_${name} = ${WITH_LAYER_${name}}")
    endif()

    if(WITH_LAYER_${name})
        list(APPEND ncnn_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp)

        # look for arch specific implementation and append source
        # optimized implementation for armv7, aarch64 or x86
        if((IOS AND CMAKE_OSX_ARCHITECTURES MATCHES "arm")
            OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)"))
            set(arch arm)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(mips)")
            set(arch mips)
        else()
            set(arch x86)
        endif()

        set(LAYER_ARCH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/${arch}/${name}_${arch}.cpp)
        if(EXISTS ${LAYER_ARCH_SRC})
            set(WITH_LAYER_${name}_${arch} 1)
            list(APPEND ncnn_SRCS ${LAYER_ARCH_SRC})
        endif()

        set(LAYER_VULKAN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/vulkan/${name}_vulkan.cpp)
        if(NCNN_VULKAN AND EXISTS ${LAYER_VULKAN_SRC})
            set(WITH_LAYER_${name}_vulkan 1)
            list(APPEND ncnn_SRCS ${LAYER_VULKAN_SRC})
        endif()
    endif()

    # generate layer_declaration and layer_registry file
    if(WITH_LAYER_${name})
        set(layer_declaration "${layer_declaration}#include \"layer/${name}.h\"\n")
        set(layer_declaration_class "class ${class}_final : virtual public ${class}")
        set(create_pipeline_content "        { int ret = ${class}::create_pipeline(opt); if (ret) return ret; }\n")
        set(destroy_pipeline_content "        { int ret = ${class}::destroy_pipeline(opt); if (ret) return ret; }\n")
    endif()

    if(WITH_LAYER_${name}_${arch})
        set(layer_declaration "${layer_declaration}#include \"layer/${arch}/${name}_${arch}.h\"\n")
        set(layer_declaration_class "${layer_declaration_class}, virtual public ${class}_${arch}")
        set(create_pipeline_content "${create_pipeline_content}        { int ret = ${class}_${arch}::create_pipeline(opt); if (ret) return ret; }\n")
        set(destroy_pipeline_content "        { int ret = ${class}_${arch}::destroy_pipeline(opt); if (ret) return ret; }\n${destroy_pipeline_content}")
    endif()

    if(WITH_LAYER_${name}_vulkan)
        set(layer_declaration "${layer_declaration}#include \"layer/vulkan/${name}_vulkan.h\"\n")
        set(layer_declaration_class "${layer_declaration_class}, virtual public ${class}_vulkan")
        set(create_pipeline_content "${create_pipeline_content}        if (vkdev) { int ret = ${class}_vulkan::create_pipeline(opt); if (ret) return ret; }\n")
        set(destroy_pipeline_content "        if (vkdev) { int ret = ${class}_vulkan::destroy_pipeline(opt); if (ret) return ret; }\n${destroy_pipeline_content}")

        file(GLOB_RECURSE SHADER_SRCS "layer/vulkan/shader/${name}.comp")
        file(GLOB_RECURSE SHADER_SUBSRCS "layer/vulkan/shader/${name}_*.comp")
        list(APPEND SHADER_SRCS ${SHADER_SUBSRCS})
        foreach(SHADER_SRC ${SHADER_SRCS})
            ncnn_add_shader(${SHADER_SRC})
        endforeach()
    endif()

    if(WITH_LAYER_${name})
        set(layer_declaration "${layer_declaration}namespace ncnn {\n${layer_declaration_class}\n{\n")
        set(layer_declaration "${layer_declaration}public:\n")
        set(layer_declaration "${layer_declaration}    virtual int create_pipeline(const Option& opt) {\n${create_pipeline_content}        return 0;\n    }\n")
        set(layer_declaration "${layer_declaration}    virtual int destroy_pipeline(const Option& opt) {\n${destroy_pipeline_content}        return 0;\n    }\n")
        set(layer_declaration "${layer_declaration}};\n")
        set(layer_declaration "${layer_declaration}DEFINE_LAYER_CREATOR(${class}_final)\n} // namespace ncnn\n\n")
    endif()

    if(WITH_LAYER_${name})
        set(layer_registry "${layer_registry}#if NCNN_STRING\n{\"${class}\",${class}_final_layer_creator},\n#else\n{${class}_final_layer_creator},\n#endif\n")
    else()
        set(layer_registry "${layer_registry}#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()

    # generate layer_type_enum file
    set(layer_type_enum "${layer_type_enum}${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()

# look for vulkan compute shader and compile
set(SHADER_SPV_HEX_FILES)

set(__LAYER_TYPE_ENUM_INDEX 0)

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
ncnn_add_layer(ROIAlign OFF)
ncnn_add_layer(Packing)
ncnn_add_layer(Requantize)
ncnn_add_layer(Cast)
ncnn_add_layer(HardSigmoid)
ncnn_add_layer(SELU)
ncnn_add_layer(HardSwish)
ncnn_add_layer(Noop)
ncnn_add_layer(PixelShuffle)

if(NCNN_VULKAN)
    ncnn_add_shader(${CMAKE_CURRENT_SOURCE_DIR}/convert_ycbcr.comp)
endif()

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

# create new
configure_file(layer_declaration.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
configure_file(layer_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
configure_file(layer_type_enum.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)
configure_file(layer_shader_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_registry.h)
configure_file(layer_shader_spv_data.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_spv_data.h)

add_library(ncnn STATIC ${ncnn_SRCS})

target_include_directories(ncnn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/ncnn>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/layer>)

if(NCNN_OPENMP)
    find_package(OpenMP)
    if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
        target_compile_options(ncnn PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
endif()

if(NCNN_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(ncnn PUBLIC Vulkan::Vulkan)
endif()

if(ANDROID_NDK)
    target_link_libraries(ncnn PUBLIC android)
endif()

if(WIN32)
    target_compile_definitions(ncnn
        PUBLIC NOMINMAX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC"
    OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC"))
    target_compile_definitions(ncnn
        PRIVATE _SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE)
else()
    target_compile_options(ncnn
        PRIVATE -Wall -Wextra -Wno-unused-function)
    if(NOT NCNN_DISABLE_PIC)
        set_target_properties(ncnn
            PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                INTERFACE_POSITION_INDEPENDENT_CODE ON)
    endif()

    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
            target_compile_options(ncnn PRIVATE -Ofast)
        endif()

        target_compile_options(ncnn PRIVATE -ffast-math)
    endif()

    if(PI3)
        target_compile_options(ncnn
            PRIVATE -march=native -mfpu=neon -mfloat-abi=hard)
        target_compile_definitions(ncnn
            PRIVATE __ARM_NEON __ANDROID__)
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

if(ANDROID OR IOS)
    # disable shared library on android and xcode ios
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
elseif(NCNN_AVX2)
    #set up according to your own system environment
    #Windows
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC"
        OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC"))
        target_compile_options(ncnn PRIVATE /arch:AVX2 /DAVX2)
    #Linux
    else()
        target_compile_options(ncnn PRIVATE -mfma -mf16c)
    endif()
endif()

add_dependencies(ncnn generate-spirv)

if(NCNN_OPENMP AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
    if(NCNN_CMAKE_VERBOSE)
        message("Building with OpenMP")
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ncnn PUBLIC OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(ncnn PRIVATE "${OpenMP_CXX_FLAGS}")
    endif()
endif()

if(NCNN_INSTALL_SDK)
    install(TARGETS ncnn EXPORT ncnn ARCHIVE DESTINATION lib)
    install(FILES
        allocator.h
        blob.h
        command.h
        cpu.h
        datareader.h
        gpu.h
        layer.h
        layer_type.h
        mat.h
        modelbin.h
        net.h
        opencv.h
        option.h
        paramdict.h
        pipeline.h
        benchmark.h
        ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h
        ${CMAKE_CURRENT_BINARY_DIR}/platform.h
        DESTINATION include/ncnn
    )
    install(EXPORT ncnn
        DESTINATION lib/cmake/ncnn)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../cmake/ncnnConfig.cmake.in
        ncnnConfig.cmake
        @ONLY)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ncnnConfig.cmake
        DESTINATION lib/cmake/ncnn)
endif()

# add ncnn and generate-spirv to a virtual project group
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET ncnn PROPERTY FOLDER "libncnn")
set_property(TARGET generate-spirv PROPERTY FOLDER "libncnn")
