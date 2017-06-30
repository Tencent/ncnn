
##############################################

configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/layer)

set(ncnn_SRCS
    blob.cpp
    cpu.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    net.cpp
    opencv.cpp
)

macro(ncnn_add_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    message("WITH_LAYER_${name} = ${WITH_LAYER_${name}}")

    if(WITH_LAYER_${name})
        list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp")

        # look for arch specific implementation and append source
        # optimized implementation for armv7 aarch64
        if((ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
            OR (ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "armv7"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "arm64"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "armv7;arm64")))
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/layer/arm/${name}_arm.cpp")
                list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/arm/${name}_arm.cpp")
                set(WITH_LAYER_${name}_arm 1)
            endif()
        else()
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/layer/x86/${name}_x86.cpp")
                list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/x86/${name}_x86.cpp")
                set(WITH_LAYER_${name}_x86 1)
            endif()
        endif()
    endif()

    # generate layer_declaration and layer_registry file
    if(WITH_LAYER_${name})
        if(WITH_LAYER_${name}_arm)
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h
                "extern Layer* ${class}_arm_layer_creator();\n")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
                "#if NCNN_STRING\n{\"${class}\",${class}_arm_layer_creator},\n#else\n{${class}_arm_layer_creator},\n#endif\n")
        elseif(WITH_LAYER_${name}_x86)
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h
                "extern Layer* ${class}_x86_layer_creator();\n")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
                "#if NCNN_STRING\n{\"${class}\",${class}_x86_layer_creator},\n#else\n{${class}_x86_layer_creator},\n#endif\n")
        else()
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h
                "extern Layer* ${class}_layer_creator();\n")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
                "#if NCNN_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
        endif()
    else()
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h "#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()
endmacro()

# create new
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_declaration.h)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)

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
ncnn_add_layer(Embed OFF)
ncnn_add_layer(Exp)
ncnn_add_layer(Flatten)
ncnn_add_layer(InnerProduct)
ncnn_add_layer(Input)
ncnn_add_layer(Log)
ncnn_add_layer(LRN)
ncnn_add_layer(MemoryData OFF)
ncnn_add_layer(MVN)
ncnn_add_layer(Pooling)
ncnn_add_layer(Power)
ncnn_add_layer(PReLU)
ncnn_add_layer(Proposal OFF)
ncnn_add_layer(Reduction OFF)
ncnn_add_layer(ReLU)
ncnn_add_layer(Reshape OFF)
ncnn_add_layer(ROIPooling OFF)
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

add_library(ncnn STATIC ${ncnn_SRCS})

install(TARGETS ncnn ARCHIVE DESTINATION lib)
install(FILES
    blob.h
    cpu.h
    layer.h
    mat.h
    net.h
    opencv.h
    ${CMAKE_CURRENT_BINARY_DIR}/platform.h
    DESTINATION include
)
