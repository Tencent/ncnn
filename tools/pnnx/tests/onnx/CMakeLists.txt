
find_package(Python3 REQUIRED COMPONENTS Interpreter)

macro(pnnx_onnx_add_test name)
    add_test(NAME test_onnx_${name} COMMAND ${CMAKE_COMMAND} -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE} -DPYTHON_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py -P ${CMAKE_CURRENT_SOURCE_DIR}/../run_test.cmake)
endmacro()

pnnx_onnx_add_test(F_relu)

pnnx_onnx_add_test(convnext_tiny)
pnnx_onnx_add_test(mobilenet_v2)
pnnx_onnx_add_test(mobilenet_v3_small)
pnnx_onnx_add_test(resnet18)
pnnx_onnx_add_test(shufflenet_v2_x1_0)
pnnx_onnx_add_test(squeezenet1_1)
pnnx_onnx_add_test(swin_t)
pnnx_onnx_add_test(vit_b_32)
