
find_package(Protobuf QUIET)

if(NOT PROTOBUF_FOUND)
    # fallback to config mode
    option(protobuf_MODULE_COMPATIBLE "CMake built-in FindProtobuf.cmake module compatible" ON)
    find_package(protobuf CONFIG)
endif()

if(PROTOBUF_FOUND)
    if(DEFINED Protobuf_VERSION AND Protobuf_VERSION VERSION_GREATER_EQUAL 3.22)
        set(CMAKE_CXX_STANDARD 17)
    endif()

    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
    add_executable(onnx2ncnn onnx2ncnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
    target_include_directories(onnx2ncnn
        PRIVATE
            ${PROTOBUF_INCLUDE_DIRS}
            ${CMAKE_CURRENT_BINARY_DIR})
    if(TARGET protobuf::libprotobuf)
        target_link_libraries(caffe2ncnn PRIVATE protobuf::libprotobuf)
    else()
        target_link_libraries(caffe2ncnn PRIVATE ${PROTOBUF_LIBRARIES})
    endif()

    # add all onnx2ncnn tool to a virtual project group
    set_property(TARGET onnx2ncnn PROPERTY FOLDER "tools/converter")
    ncnn_install_tool(onnx2ncnn)
else()
    message(WARNING "Protobuf not found, onnx model convert tool won't be built")
endif()
