
set(ENV{PYTHONPATH} "ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}")
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_SCRIPT} RESULT_VARIABLE result)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Test failed with return value '${result}'")
endif()
