
if(MSVC)
    add_definitions(/wd4996)
endif()

add_library(ncnnperfutil STATIC perfutil.cpp)
target_link_libraries(ncnnperfutil PUBLIC ncnn)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/layer)

macro(ncnn_add_perf name)
    add_executable(perf_${name} perf_${name}.cpp)
    target_link_libraries(perf_${name} PRIVATE ncnnperfutil ncnn)

    # add to a virtual project group
    set_property(TARGET perf_${name} PROPERTY FOLDER "perftests")
endmacro()

macro(ncnn_add_layer_perf class)
    string(TOLOWER ${class} name)

    # enable if WITH_LAYER_xxx option ON
    if(${WITH_LAYER_${name}})
        ncnn_add_perf(${name})
    endif()
endmacro()

ncnn_add_layer_perf(ReLU)
ncnn_add_layer_perf(Convolution)
ncnn_add_layer_perf(ConvolutionDepthWise)
ncnn_add_layer_perf(Pooling)
ncnn_add_layer_perf(InnerProduct)
ncnn_add_layer_perf(Softmax)
ncnn_add_layer_perf(BinaryOp)
ncnn_add_layer_perf(Concat)
ncnn_add_layer_perf(Sigmoid)
ncnn_add_layer_perf(BatchNorm)
