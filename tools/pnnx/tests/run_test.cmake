# Explicit CTest registrations override an ambient format selection. Unchanged
# ncnn/onnx registrations have no format argument and must remain TS-only.
if(NOT DEFINED PNNX_TEST_FORMAT)
    set(PNNX_TEST_FORMAT torchscript)
endif()
if(NOT PNNX_TEST_FORMAT STREQUAL "torchscript" AND NOT PNNX_TEST_FORMAT STREQUAL "pt2")
    message(FATAL_ERROR "Unknown PNNX_TEST_FORMAT '${PNNX_TEST_FORMAT}'")
endif()
set(ENV{PNNX_TEST_FORMAT} "${PNNX_TEST_FORMAT}")

# Preserve the caller's search path and use the platform's actual separator.
if(WIN32)
    set(path_separator ";")
else()
    set(path_separator ":")
endif()
if("$ENV{PYTHONPATH}" STREQUAL "")
    set(ENV{PYTHONPATH} "${CMAKE_CURRENT_BINARY_DIR}")
else()
    set(ENV{PYTHONPATH} "$ENV{PYTHONPATH}${path_separator}${CMAKE_CURRENT_BINARY_DIR}")
endif()
set(ENV{PYTHONUNBUFFERED} "1")

if(NOT DEFINED PNNX_TEST_CASE_TIMEOUT)
    set(PNNX_TEST_CASE_TIMEOUT 1800)
endif()
# Leave stdout/stderr attached so a crash or timeout cannot hide diagnostics.
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${PYTHON_SCRIPT}"
    RESULT_VARIABLE result TIMEOUT "${PNNX_TEST_CASE_TIMEOUT}")
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "${PYTHON_SCRIPT} [${PNNX_TEST_FORMAT}] failed with return value '${result}'")
endif()
