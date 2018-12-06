
find_package(Protobuf)

if(PROTOBUF_FOUND)
    include_directories(${PROTOBUF_INCLUDE_DIR})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
    add_executable(onnx2ncnn onnx2ncnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
    target_link_libraries(onnx2ncnn ${PROTOBUF_LIBRARIES})
else()
    message(WARNING "Protobuf not found, onnx model convert tool won't be built")
endif()
