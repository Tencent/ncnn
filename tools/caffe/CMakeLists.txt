
find_package(protobuf CONFIG)

if(protobuf_FOUND)
    set(PROTOBUF_FOUND ${protobuf_FOUND})
    set(PROTOBUF_VERSION ${protobuf_VERSION})
else()
    # fallback to system
    find_package(Protobuf)
    set(PROTOBUF_FOUND ${Protobuf_FOUND})
    set(PROTOBUF_VERSION ${Protobuf_VERSION})
    if(TARGET protobuf::protoc)
        set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION_RELEASE "${PROTOBUF_PROTOC_EXECUTABLE}")
    endif()
endif()

if(PROTOBUF_FOUND)
    get_filename_component(CAFFE_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../pnnx/src" ABSOLUTE)
    set(CAFFE_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/caffe.pb.cc")
    set(CAFFE_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/caffe.pb.h")
    if(TARGET protobuf::protoc)
        set(CAFFE_PROTOC_EXECUTABLE protobuf::protoc)
    else()
        set(CAFFE_PROTOC_EXECUTABLE "${PROTOBUF_PROTOC_EXECUTABLE}")
    endif()
    # Older protobuf CMake helpers misplace outputs for out-of-source schemas.
    add_custom_command(
        OUTPUT ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS}
        COMMAND ${CAFFE_PROTOC_EXECUTABLE}
        ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}" -I "${CAFFE_PROTO_DIR}" "${CAFFE_PROTO_DIR}/caffe.proto"
        DEPENDS "${CAFFE_PROTO_DIR}/caffe.proto" ${CAFFE_PROTOC_EXECUTABLE}
        VERBATIM)

    if(DEFINED PROTOBUF_VERSION AND PROTOBUF_VERSION VERSION_GREATER_EQUAL 3.22)
        set(CMAKE_CXX_STANDARD 17)
    endif()

    add_executable(caffe2ncnn caffe2ncnn.cpp ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS})
    if(Protobuf_FOUND OR protobuf_MODULE_COMPATIBLE)
        target_include_directories(caffe2ncnn PRIVATE ${PROTOBUF_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(caffe2ncnn PRIVATE ${PROTOBUF_LIBRARIES})
    else()
        target_include_directories(caffe2ncnn PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(caffe2ncnn PRIVATE protobuf::libprotobuf)
    endif()

    # add all caffe2ncnn tool to a virtual project group
    set_property(TARGET caffe2ncnn PROPERTY FOLDER "tools/converter")
    ncnn_install_tool(caffe2ncnn)
else()
    message(WARNING "Protobuf not found, caffe model convert tool won't be built")
endif()
