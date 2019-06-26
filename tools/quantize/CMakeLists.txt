find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../src)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(ncnn2table ncnn2table.cpp)
target_link_libraries(ncnn2table PRIVATE ncnn ${OpenCV_LIBS})

add_executable(ncnn2int8 ncnn2int8.cpp)
target_link_libraries(ncnn2int8 PRIVATE ncnn)