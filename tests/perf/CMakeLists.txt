
if(MSVC)
    add_definitions(/wd4996)
endif()

add_library(ncnnperftestutil STATIC perftestutil.cpp)
target_link_libraries(ncnnperftestutil PUBLIC ncnn)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/layer)

macro(ncnn_add_perftest name)
    add_executable(perf_${name} test_${name}.cpp)
    target_link_libraries(perf_${name} PRIVATE ncnnperftestutil ncnn)

    # add to a virtual project group
    set_property(TARGET perf_${name} PROPERTY FOLDER "perftests")
endmacro()

macro(ncnn_add_layer_perftest class)
    string(TOLOWER ${class} name)

    # enable if WITH_LAYER_xxx option ON
    if(${WITH_LAYER_${name}})
        ncnn_add_perftest(${name})
    endif()
endmacro()

ncnn_add_layer_perftest(ReLU)
ncnn_add_layer_perftest(Convolution)
ncnn_add_layer_perftest(ConvolutionDepthWise)
ncnn_add_layer_perftest(Pooling)
ncnn_add_layer_perftest(InnerProduct)
ncnn_add_layer_perftest(Softmax)
ncnn_add_layer_perftest(BinaryOp)
ncnn_add_layer_perftest(Concat)
ncnn_add_layer_perftest(Sigmoid)
ncnn_add_layer_perftest(BatchNorm)
