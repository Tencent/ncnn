
if(WIN32)
    set(path_separator ";")
else()
    set(path_separator ":")
endif()

if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
    set(ENV{PYTHONPATH} "$ENV{PYTHONPATH}${path_separator}${CMAKE_CURRENT_BINARY_DIR}")
else()
    set(ENV{PYTHONPATH} "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(DEFINED PNNX_TEST_FORMAT)
    set(ENV{PNNX_TEST_FORMAT} "${PNNX_TEST_FORMAT}")
endif()

if(DEFINED PNNX_EXECUTABLE)
    set(ENV{PNNX_TEST_PNNX} "${PNNX_EXECUTABLE}")
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_SCRIPT} RESULT_VARIABLE result)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Test failed with return value '${result}'")
endif()
