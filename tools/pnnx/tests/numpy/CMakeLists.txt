
find_package(Python3 REQUIRED COMPONENTS Interpreter)

macro(pnnx_numpy_add_test name)
    add_test(NAME test_${name} COMMAND ${CMAKE_COMMAND} -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE} -DPYTHON_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py -P ${CMAKE_CURRENT_SOURCE_DIR}/../run_test.cmake)
endmacro()

pnnx_numpy_add_test(convnext_tiny)
pnnx_numpy_add_test(convnext_tiny_endian)
pnnx_numpy_add_test(convnext_tiny_fortran_array)
pnnx_numpy_add_test(nn_Conv3d)
pnnx_numpy_add_test(pnnx_eliminate_noop_expand)
pnnx_numpy_add_test(pnnx_fuse_multiheadattention)
