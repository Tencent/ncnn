cmake_minimum_required(VERSION 3.4)

project(pypnnx)

set(PACKAGE_VERSION {PNNX_VERSION_STRING})
add_definitions(-DVERSION_INFO="${PACKAGE_VERSION}")

set( CMAKE_CXX_STANDARD  14)
set( CMAKE_CXX_STANDARD_REQUIRED  ON)

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