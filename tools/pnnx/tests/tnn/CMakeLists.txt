add_test(NAME test_tnn_level2 COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_level2.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/level2)
