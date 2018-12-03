
##############################################

configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/layer)

set(ncnn_SRCS
    allocator.cpp
    blob.cpp
    cpu.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    modelbin.cpp
    net.cpp
    opencv.cpp
    paramdict.cpp
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
        if((IOS AND ${CMAKE_OSX_ARCHITECTURES} MATCHES "arm")
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
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h
                "extern Layer* ${class}_${arch}_layer_creator();\n")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
                "#if NCNN_STRING\n{\"${class}\",${class}_${arch}_layer_creator},\n#else\n{${class}_${arch}_layer_creator},\n#endif\n")
        else()
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h
                "extern Layer* ${class}_layer_creator();\n")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
                "#if NCNN_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
        endif()
    else()
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h "#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()

    # generate layer_type_enum file
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h "${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()

# create new
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)
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

add_library(ncnn STATIC ${ncnn_SRCS})

if(NCNN_OPENMP AND OpenMP_CXX_FOUND)
    if(NCNN_CMAKE_VERBOSE)
        message("Building with OpenMP")
    endif()
    target_link_libraries(ncnn PUBLIC OpenMP::OpenMP_CXX)
endif()

if(COVERAGE)
    target_compile_options(ncnn PRIVATE --coverage)
endif()

install(TARGETS ncnn ARCHIVE DESTINATION lib)
install(FILES
    allocator.h
    blob.h
    cpu.h
    layer.h
    layer_type.h
    mat.h
    modelbin.h
    net.h
    opencv.h
    paramdict.h
    benchmark.h
    ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h
    ${CMAKE_CURRENT_BINARY_DIR}/platform.h
    DESTINATION include
)
