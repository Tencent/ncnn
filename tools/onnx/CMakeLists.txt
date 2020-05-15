
find_package(Protobuf)

if(PROTOBUF_FOUND)
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
    add_executable(onnx2ncnn onnx2ncnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
    target_include_directories(onnx2ncnn
        PRIVATE
            ${PROTOBUF_INCLUDE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(onnx2ncnn PRIVATE ${PROTOBUF_LIBRARIES})

    # add all onnx2ncnn tool to a virtual project group
    set_property(TARGET onnx2ncnn PROPERTY FOLDER "tools/converter")
else()
    message(WARNING "Protobuf not found, onnx model convert tool won't be built")
endif()
