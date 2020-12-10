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

set(LLVM_TARGET_DEFINITIONS tf_ops.td)
mlir_tablegen(tf_all_ops.h.inc -gen-op-decls)
mlir_tablegen(tf_all_ops.cc.inc -gen-op-defs)
add_public_tablegen_target(tf_opsIncGen)

set(LLVM_TARGET_DEFINITIONS tf_op_interfaces.td)
mlir_tablegen(tf_op_interfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(tf_op_interfaces.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(tf_op_interfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ncnn_ops.td)
mlir_tablegen(ncnn_ops.h.inc -gen-op-decls)
mlir_tablegen(ncnn_ops.cc.inc -gen-op-defs)
add_public_tablegen_target(ncnn_opsIncGen)

set(LLVM_TARGET_DEFINITIONS ncnn_rewriter.td)
mlir_tablegen(ncnn_rewriter.inc -gen-rewriters)
add_public_tablegen_target(ncnn_rewriterIncGen)

add_executable(mlir2ncnn
    mlir2ncnn.cpp
    ncnn_dialect.cpp
    ncnn_rewriter.cpp
    tf_dialect.cpp
    tf_attributes.cc
    tf_types.cc
)

add_dependencies(mlir2ncnn
    tf_opsIncGen
    tf_op_interfacesIncGen
    ncnn_opsIncGen
    ncnn_rewriterIncGen
)

target_link_libraries(mlir2ncnn
    MLIRIR
    MLIRDialect
    MLIRInferTypeOpInterface
    MLIRParser
    MLIRPass
    MLIRStandard
    MLIRTransforms
)
