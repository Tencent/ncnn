
find_package(OpenCV REQUIRED core highgui imgproc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(squeezenet squeezenet.cpp)

target_link_libraries(squeezenet ncnn ${OpenCV_LIBS})
