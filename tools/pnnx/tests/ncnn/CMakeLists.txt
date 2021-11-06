
find_package(Python3 REQUIRED COMPONENTS Interpreter)

macro(pnnx_ncnn_add_test name)
    add_test(NAME test_ncnn_${name} COMMAND ${CMAKE_COMMAND} -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE} -DPYTHON_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py -P ${CMAKE_CURRENT_SOURCE_DIR}/../run_test.cmake)
endmacro()

pnnx_ncnn_add_test(nn_BatchNorm2d)
pnnx_ncnn_add_test(nn_ChannelShuffle)
pnnx_ncnn_add_test(nn_GroupNorm)
pnnx_ncnn_add_test(nn_InstanceNorm2d)
pnnx_ncnn_add_test(nn_LayerNorm)
pnnx_ncnn_add_test(nn_LocalResponseNorm)
pnnx_ncnn_add_test(nn_MultiheadAttention)
pnnx_ncnn_add_test(nn_PixelShuffle)
pnnx_ncnn_add_test(nn_PixelUnshuffle)

pnnx_ncnn_add_test(mobilenet_v2)
pnnx_ncnn_add_test(mobilenet_v3_small)
pnnx_ncnn_add_test(resnet18)
pnnx_ncnn_add_test(shufflenet_v2_x1_0)
pnnx_ncnn_add_test(squeezenet1_1)
