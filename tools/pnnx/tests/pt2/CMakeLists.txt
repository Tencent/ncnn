# ExportedProgram tests use the installed PyTorch and the normal optlevel=2 pipeline.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys, torch; from packaging import version; sys.exit(version.parse(torch.__version__) < version.parse('2.11'))"
    RESULT_VARIABLE PNNX_PYTORCH_PT2_TEST_STATUS
    ERROR_QUIET
)
if(NOT PNNX_PYTORCH_PT2_TEST_STATUS EQUAL 0)
    message(STATUS "Skipping pt2 tests (Python PyTorch >= 2.11 required)")
    return()
endif()

foreach(name api archive arguments backend constants copy_fill dtype dynamic effects expression loader models ops rules signature tensor)
    add_test(NAME test_pt2_${name}
        COMMAND ${CMAKE_COMMAND} -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE} -DPYTHON_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py -DPNNX_BUILD_PT2=${PNNX_BUILD_PT2} -P ${CMAKE_CURRENT_SOURCE_DIR}/../run_test.cmake)
endforeach()
