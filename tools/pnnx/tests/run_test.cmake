
set(ENV{PYTHONPATH} "ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}")
# Pass the built frontend capability; standalone runs keep the Python version gate.
if(DEFINED PNNX_BUILD_PT2)
    set(ENV{PNNX_BUILD_PT2} "${PNNX_BUILD_PT2}")
endif()
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_SCRIPT} RESULT_VARIABLE result)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Test failed with return value '${result}'")
endif()
