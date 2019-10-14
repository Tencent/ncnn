find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

add_executable(ncnn2table ncnn2table.cpp)
target_link_libraries(ncnn2table PRIVATE ncnn ${OpenCV_LIBS})
target_compile_definitions(ncnn2table PRIVATE -DOpenCV_VERSION_MAJOR=${OpenCV_VERSION_MAJOR})

add_executable(ncnn2int8 ncnn2int8.cpp)
target_link_libraries(ncnn2int8 PRIVATE ncnn)

