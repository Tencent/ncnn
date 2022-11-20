cmake_minimum_required(VERSION 3.4)

project(pyncnn)

set(PACKAGE_VERSION ${NCNN_VERSION_STRING})
add_definitions(-DVERSION_INFO="${PACKAGE_VERSION}")

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "ARM64")
    option(PYBIND11_PYTHONLIBS_OVERWRITE "" OFF)

    set(PYTHON_PREFIX "$ENV{LOCALAPPDATA}/pypa/cibuildwheel/Cache/nuget-cpython/pythonarm64.$ENV{PYTHON_VERSION}/tools")
    if(NOT DEFINED $ENV{CIBUILDWHEEL})
        message(WARNING
            " This is hack for cibuildwheel on github action\n"
            " Use the right way to cross-compile python module for windows arm64 like follows\n"
            " set(PYTHON_PREFIX \"<your-pythonarm64-root-path>\")\n"
        )
    endif()
endif()

add_subdirectory(pybind11)

if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    if(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/ncnn/)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/ncnn/)
    endif(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
pybind11_add_module(pyncnn src/main.cpp)
set_target_properties(pyncnn PROPERTIES OUTPUT_NAME "ncnn")
target_link_libraries(pyncnn PUBLIC ncnn)
set_target_properties(pyncnn PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ncnn")
set_property(TARGET pyncnn PROPERTY FOLDER "python")
if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    add_custom_command(TARGET pyncnn POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/ncnn/ncnn${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION} 
        ${PROJECT_SOURCE_DIR}/ncnn/ncnn${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION})
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

configure_file(setup.py.i ${PROJECT_SOURCE_DIR}/setup.py)
