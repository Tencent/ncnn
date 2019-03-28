
##############################################

configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/layer)

if(NCNN_VULKAN)
    find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH)
    message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")
endif()

set(ncnn_SRCS
    allocator.cpp
    blob.cpp
    command.cpp
    cpu.cpp
    gpu.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    mat_pixel_resize.cpp
    modelbin.cpp
    net.cpp
    opencv.cpp
    paramdict.cpp
    pipeline.cpp
    benchmark.cpp
)

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
        list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp")

        # look for arch specific implementation and append source
        # optimized implementation for armv7, aarch64 or x86
        if((IOS AND CMAKE_OSX_ARCHITECTURES MATCHES "arm")
            OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)"))
            set(arch arm)
        else()
            set(arch x86)
        endif()
        set(LAYER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/${arch}/${name}_${arch}.cpp)
        if(EXISTS ${LAYER_SRC})
            set(WITH_LAYER_${name}_${arch} 1)
            list(APPEND ncnn_SRCS ${LAYER_SRC})
            if(NCNN_CMAKE_VERBOSE)
                message(STATUS "Adding layer: ${LAYER_SRC}")
            endif()
        endif()
    endif()

    # generate layer_declaration and layer_registry file
    if(WITH_LAYER_${name})
        if(WITH_LAYER_${name}_${arch})
            string(APPEND layer_declaration
                "extern Layer* ${class}_${arch}_layer_creator();\n")
            string(APPEND layer_registry
                "#if NCNN_STRING\n{\"${class}\",${class}_${arch}_layer_creator},\n#else\n{${class}_${arch}_layer_creator},\n#endif\n")
        else()
            string(APPEND layer_declaration
                "extern Layer* ${class}_layer_creator();\n")
            string(APPEND layer_registry
                "#if NCNN_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
        endif()
    else()
        string(APPEND layer_registry "#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()

    if(WITH_LAYER_${name} AND NCNN_VULKAN)
        file(GLOB_RECURSE SHADER_SRCS "layer/shader/${name}.comp")
        file(GLOB_RECURSE SHADER_SUBSRCS "layer/shader/${name}_*.comp")
        list(APPEND SHADER_SRCS ${SHADER_SUBSRCS})
        foreach(SHADER_SRC ${SHADER_SRCS})
            get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)

            set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
            add_custom_command(
                OUTPUT ${SHADER_SPV_HEX_FILE}
                COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
                ARGS -Dsfp=float -Dsfpvec2=vec2 -Dsfpvec4=vec4 -Dsfpmat4=mat4 -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4 -Dafpmat4=mat4 -V -s -e ${SHADER_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC}
                DEPENDS ${SHADER_SRC}
                COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
                VERBATIM
            )
            set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

            string(APPEND layer_shader_spv_data "static const uint32_t ${SHADER_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_SRC_NAME_WE}.spv.hex.h\"\n};\n")
            string(APPEND layer_shader_registry "{\"${SHADER_SRC_NAME_WE}\",${SHADER_SRC_NAME_WE}_spv_data,sizeof(${SHADER_SRC_NAME_WE}_spv_data)},\n")

            list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_FILE})

            # fp16 storage
            set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

            set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
            add_custom_command(
                OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
                COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
                ARGS -Dsfp=float16_t -Dsfpvec2=f16vec2 -Dsfpvec4=f16vec4 -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4 -Dafpmat4=mat4 -DNCNN_fp16_storage=1 -V -s -e ${SHADER_fp16s_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC}
                DEPENDS ${SHADER_SRC}
                COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
                VERBATIM
            )
            set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

            string(APPEND layer_shader_spv_data "static const uint32_t ${SHADER_fp16s_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h\"\n};\n")
            string(APPEND layer_shader_registry "{\"${SHADER_fp16s_SRC_NAME_WE}\",${SHADER_fp16s_SRC_NAME_WE}_spv_data,sizeof(${SHADER_fp16s_SRC_NAME_WE}_spv_data)},\n")

            list(APPEND SHADER_SPV_HEX_FILES ${SHADER_fp16s_SPV_HEX_FILE})

            # fp16 storage + fp16 arithmetic
            set(SHADER_fp16a_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16a")

            set(SHADER_fp16a_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16a_SRC_NAME_WE}.spv.hex.h)
            add_custom_command(
                OUTPUT ${SHADER_fp16a_SPV_HEX_FILE}
                COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
                ARGS -Dsfp=float16_t -Dsfpvec2=f16vec2 -Dsfpvec4=f16vec4 -Dsfpmat4=f16mat4 -Dafp=float16_t -Dafpvec2=f16vec2 -Dafpvec4=f16vec4 -Dafpmat4=f16mat4 -DNCNN_fp16_storage=1 -DNCNN_fp16_arithmetic=1 -V -s -e ${SHADER_fp16a_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_fp16a_SPV_HEX_FILE} ${SHADER_SRC}
                DEPENDS ${SHADER_SRC}
                COMMENT "Building SPIR-V module ${SHADER_fp16a_SRC_NAME_WE}.spv"
                VERBATIM
            )
            set_source_files_properties(${SHADER_fp16a_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

            string(APPEND layer_shader_spv_data "static const uint32_t ${SHADER_fp16a_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16a_SRC_NAME_WE}.spv.hex.h\"\n};\n")
            string(APPEND layer_shader_registry "{\"${SHADER_fp16a_SRC_NAME_WE}\",${SHADER_fp16a_SRC_NAME_WE}_spv_data,sizeof(${SHADER_fp16a_SRC_NAME_WE}_spv_data)},\n")

            list(APPEND SHADER_SPV_HEX_FILES ${SHADER_fp16a_SPV_HEX_FILE})
        endforeach()
    endif()

    # generate layer_type_enum file
    string(APPEND layer_type_enum "${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
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
ncnn_add_layer(LSTM OFF)
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

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

# create new
configure_file(layer_declaration.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
configure_file(layer_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
configure_file(layer_type_enum.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)
configure_file(layer_shader_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_registry.h)
configure_file(layer_shader_spv_data.h.in ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_spv_data.h)

add_library(ncnn STATIC ${ncnn_SRCS})

add_dependencies(ncnn generate-spirv)

if(NCNN_OPENMP AND OpenMP_CXX_FOUND)
    if(NCNN_CMAKE_VERBOSE)
        message("Building with OpenMP")
    endif()
    target_link_libraries(ncnn PUBLIC OpenMP::OpenMP_CXX)
endif()

if(NCNN_INSTALL_SDK)
  install(TARGETS ncnn ARCHIVE DESTINATION lib)
  install(FILES
    allocator.h
    blob.h
    command.h
    cpu.h
    gpu.h
    layer.h
    layer_type.h
    mat.h
    modelbin.h
    net.h
    opencv.h
    paramdict.h
    pipeline.h
    benchmark.h
    ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h
    ${CMAKE_CURRENT_BINARY_DIR}/platform.h
    DESTINATION include
  )
endif()
