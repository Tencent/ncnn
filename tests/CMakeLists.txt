
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(test_convolution test_convolution.cpp)
target_link_libraries(test_convolution PRIVATE ncnn)
add_test(test_convolution test_convolution)

add_executable(test_deconvolution test_deconvolution.cpp)
target_link_libraries(test_deconvolution PRIVATE ncnn)
add_test(test_deconvolution test_deconvolution)

add_executable(test_relu test_relu.cpp)
target_link_libraries(test_relu PRIVATE ncnn)
add_test(test_relu test_relu)
