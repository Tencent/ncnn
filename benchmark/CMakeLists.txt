
if(MSVC)
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4996)
endif()

# ncnn macro
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_add_param.cmake)

set(benchncnn_PARAMS
    alexnet.param
    blazeface.param
    efficientnet_b0.param
    efficientnetv2_b0.param
    FastestDet.param
    googlenet_int8.param
    googlenet.param
    mnasnet.param
    mobilenet_int8.param
    mobilenet_ssd_int8.param
    mobilenet_ssd.param
    mobilenet_v2.param
    mobilenet_v3.param
    mobilenet_yolo.param
    mobilenet.param
    mobilenetv2_yolov3.param
    nanodet_m.param
    proxylessnasnet.param
    regnety_400m.param
    resnet18_int8.param
    resnet18.param
    resnet50_int8.param
    resnet50.param
    shufflenet_v2.param
    shufflenet.param
    squeezenet_int8.param
    squeezenet_ssd_int8.param
    squeezenet_ssd.param
    squeezenet.param
    vgg16_int8.param
    vgg16.param
    vision_transformer.param
    yolo-fastest-1.1.param
    yolo-fastestv2.param
    yolov4-tiny.param
)

foreach(PARAM_FILE ${benchncnn_PARAMS})
    ncnn_add_param("${CMAKE_CURRENT_SOURCE_DIR}/${PARAM_FILE}")
endforeach()

add_custom_target(ncnn-generate-param DEPENDS ${NCNN_PARAM_HEX_FILES})

configure_file(benchncnn_param_data.h.in ${CMAKE_CURRENT_BINARY_DIR}/benchncnn_param_data.h)

add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)

target_include_directories(benchncnn PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(benchncnn PRIVATE nodefs.js)
endif()

add_dependencies(benchncnn ncnn-generate-param)

# add benchncnn to a virtual project group
set_property(TARGET benchncnn PROPERTY FOLDER "benchmark")
