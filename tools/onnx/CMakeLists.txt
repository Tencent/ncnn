
find_package(Protobuf)

if(PROTOBUF_FOUND)
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
    add_library(libonnx2ncnn onnx2ncnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
    target_include_directories(libonnx2ncnn
        PUBLIC
            ${PROTOBUF_INCLUDE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(libonnx2ncnn PRIVATE ${PROTOBUF_LIBRARIES})

    add_executable(onnx2ncnn onnx2ncnn_exe.cpp)
    target_link_libraries(onnx2ncnn libonnx2ncnn)

    # add all onnx2ncnn tool to a virtual project group
    set_property(TARGET libonnx2ncnn onnx2ncnn PROPERTY FOLDER "tools/converter")
else()
    message(WARNING "Protobuf not found, onnx model convert tool won't be built")
endif()
