set(CAFFE2NCNN_EXECUTABLE "caffe2ncnn" CACHE FILEPATH "Legacy caffe2ncnn executable for Caffe regression tests")
set(CAFFE_TEST_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../examples" CACHE PATH "Directory containing the Caffe SqueezeNet example")
set(CAFFE_SSD_MODEL_DIR "" CACHE PATH "Optional external MobileNet-SSD source fixture directory")

macro(pnnx_caffe_add_test name)
    add_test(NAME test_caffe_${name} COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py
        --pnnx $<TARGET_FILE:pnnx>
        ${ARGN}
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/${name})
endmacro()

if(TARGET protobuf::protoc)
    set(caffe_test_protoc $<TARGET_FILE:protobuf::protoc>)
else()
    set(caffe_test_protoc ${PROTOBUF_PROTOC_EXECUTABLE})
endif()
set(caffe_test_protobuf_args
    --protoc ${caffe_test_protoc}
    --schema ${CMAKE_CURRENT_SOURCE_DIR}/../../src/caffe.proto)

pnnx_caffe_add_test(squeezenet
    --caffe2ncnn ${CAFFE2NCNN_EXECUTABLE}
    --model-dir ${CAFFE_TEST_MODEL_DIR})
pnnx_caffe_add_test(operators ${caffe_test_protobuf_args})
pnnx_caffe_add_test(names ${caffe_test_protobuf_args})

# Export failure tests create their own temporary output directory.
add_test(NAME test_caffe_export_errors COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_export_errors.py
    --pnnx $<TARGET_FILE:pnnx>
    ${caffe_test_protobuf_args})
set_tests_properties(test_caffe_export_errors PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP: /dev/full is unavailable")

pnnx_caffe_add_test(argmax)
pnnx_caffe_add_test(power)
pnnx_caffe_add_test(lrn)
pnnx_caffe_add_test(reduction)
pnnx_caffe_add_test(recurrent ${caffe_test_protobuf_args})
pnnx_caffe_add_test(embed ${caffe_test_protobuf_args})
pnnx_caffe_add_test(detection)
pnnx_caffe_add_test(roi)
pnnx_caffe_add_test(proposal)
pnnx_caffe_add_test(proposal_roi)
pnnx_caffe_add_test(roi_head ${caffe_test_protobuf_args})
pnnx_caffe_add_test(yolo)

if(CAFFE_SSD_MODEL_DIR)
    pnnx_caffe_add_test(mobilenet_ssd
        --caffe2ncnn ${CAFFE2NCNN_EXECUTABLE}
        --model-dir ${CAFFE_SSD_MODEL_DIR})
endif()
