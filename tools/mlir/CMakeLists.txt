project(mlir2ncnn)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)

set(LLVM_DIR "/home/nihui/osd/llvm-project/build/install/lib/cmake/llvm")
find_package(LLVM REQUIRED)

set(MLIR_DIR "/home/nihui/osd/llvm-project/build/install/lib/cmake/mlir")
find_package(MLIR REQUIRED)

add_definitions(-fno-rtti -fno-exceptions)

include_directories("/home/nihui/osd/llvm-project/build/install/include")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(${LLVM_DIR}/TableGen.cmake)
include(${MLIR_DIR}/AddMLIR.cmake)

add_mlir_dialect(tf_ops tf)
add_mlir_interface(tf_op_interfaces)

add_executable(mlir2ncnn
    mlir2ncnn.cpp
    tf_attributes.cc
    tf_types.cc
)

add_dependencies(mlir2ncnn
    MLIRtf_opsIncGen
    MLIRtf_op_interfacesIncGen
)

target_link_libraries(mlir2ncnn
    MLIRIR
    MLIRDialect
    MLIRInferTypeOpInterface
    MLIRParser
    MLIRStandardOps
)
