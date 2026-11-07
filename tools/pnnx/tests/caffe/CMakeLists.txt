set(CAFFE2NCNN_EXECUTABLE "caffe2ncnn" CACHE FILEPATH "Legacy caffe2ncnn executable for Caffe regression tests")
set(CAFFE_TEST_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../examples" CACHE PATH "Directory containing the Caffe SqueezeNet example")
set(CAFFE_SSD_MODEL_DIR "" CACHE PATH "Optional external MobileNet-SSD source fixture directory")

add_test(NAME test_caffe_squeezenet COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_squeezenet.py
    --pnnx $<TARGET_FILE:pnnx>
    --caffe2ncnn ${CAFFE2NCNN_EXECUTABLE}
    --model-dir ${CAFFE_TEST_MODEL_DIR}
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/squeezenet)

if(TARGET protobuf::protoc)
    set(caffe_test_protoc $<TARGET_FILE:protobuf::protoc>)
else()
    set(caffe_test_protoc ${PROTOBUF_PROTOC_EXECUTABLE})
endif()

add_test(NAME test_caffe_operators COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_operators.py
    --pnnx $<TARGET_FILE:pnnx>
    --protoc ${caffe_test_protoc}
    --schema ${CMAKE_CURRENT_SOURCE_DIR}/../../src/caffe.proto
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/operators)

add_test(NAME test_caffe_names COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_names.py
    --pnnx $<TARGET_FILE:pnnx>
    --protoc ${caffe_test_protoc}
    --schema ${CMAKE_CURRENT_SOURCE_DIR}/../../src/caffe.proto
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/names)

add_test(NAME test_caffe_export_errors COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_export_errors.py
    --pnnx $<TARGET_FILE:pnnx>
    --protoc ${caffe_test_protoc}
    --schema ${CMAKE_CURRENT_SOURCE_DIR}/../../src/caffe.proto)
set_tests_properties(test_caffe_export_errors PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP: /dev/full is unavailable")

add_test(NAME test_caffe_argmax COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_argmax.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/argmax)

add_test(NAME test_caffe_power COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_power.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/power)

add_test(NAME test_caffe_lrn COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_lrn.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/lrn)

add_test(NAME test_caffe_reduction COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reduction.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/reduction)

add_test(NAME test_caffe_recurrent COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_recurrent.py
    --pnnx $<TARGET_FILE:pnnx>
    --protoc ${caffe_test_protoc}
    --schema ${CMAKE_CURRENT_SOURCE_DIR}/../../src/caffe.proto
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/recurrent)

add_test(NAME test_caffe_embed COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_embed.py
    --pnnx $<TARGET_FILE:pnnx>
    --protoc ${caffe_test_protoc}
    --schema ${CMAKE_CURRENT_SOURCE_DIR}/../../src/caffe.proto
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/embed)

add_test(NAME test_caffe_detection COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_detection.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/detection)

add_test(NAME test_caffe_roi COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_roi.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/roi)

add_test(NAME test_caffe_proposal COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_proposal.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/proposal)

add_test(NAME test_caffe_proposal_roi COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_proposal_roi.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/proposal_roi)

add_test(NAME test_caffe_roi_head COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_roi_head.py
    --pnnx $<TARGET_FILE:pnnx>
    --protoc ${caffe_test_protoc}
    --schema ${CMAKE_CURRENT_SOURCE_DIR}/../../src/caffe.proto
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/roi_head)

add_test(NAME test_caffe_yolo COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_yolo.py
    --pnnx $<TARGET_FILE:pnnx>
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}/yolo)

if(CAFFE_SSD_MODEL_DIR)
    add_test(NAME test_caffe_mobilenet_ssd COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_mobilenet_ssd.py
        --pnnx $<TARGET_FILE:pnnx>
        --caffe2ncnn ${CAFFE2NCNN_EXECUTABLE}
        --model-dir ${CAFFE_SSD_MODEL_DIR}
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd)
endif()
