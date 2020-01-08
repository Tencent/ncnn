
macro(ncnn_add_layer_test class)
    string(TOLOWER ${class} name)

    # enable if WITH_LAYER_xxx option ON
    if(${WITH_LAYER_${name}})
        add_executable(test_${name} test_${name}.cpp)
        target_link_libraries(test_${name} PRIVATE ncnn)
        add_test(test_${name} test_${name})
    endif()
endmacro()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

ncnn_add_layer_test(Concat)
ncnn_add_layer_test(Convolution)
ncnn_add_layer_test(ConvolutionDepthWise)
ncnn_add_layer_test(Deconvolution)
ncnn_add_layer_test(DeconvolutionDepthWise)
ncnn_add_layer_test(InnerProduct)
ncnn_add_layer_test(Packing)
ncnn_add_layer_test(ReLU)
ncnn_add_layer_test(ShuffleChannel)
