
find_package(Python3 REQUIRED COMPONENTS Interpreter)

macro(pnnx_onnx_add_test name)
    add_test(NAME test_onnx_${name} COMMAND ${CMAKE_COMMAND} -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE} -DPYTHON_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py -P ${CMAKE_CURRENT_SOURCE_DIR}/../run_test.cmake)
endmacro()

pnnx_onnx_add_test(F_relu)

pnnx_onnx_add_test(resnet18)
