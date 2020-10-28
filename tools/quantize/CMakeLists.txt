find_package(OpenCV QUIET COMPONENTS opencv_world)
# for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
if("${OpenCV_LIBS}" STREQUAL "")
    set(OpenCV_FOUND FALSE)
endif()
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
endif()
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
endif()

if(OpenCV_FOUND)
    if(NCNN_PIXEL)
        add_executable(ncnn2table ncnn2table.cpp)
        target_compile_definitions(ncnn2table PRIVATE -DOpenCV_VERSION_MAJOR=${OpenCV_VERSION_MAJOR})
        target_include_directories(ncnn2table PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(ncnn2table PRIVATE ncnn ${OpenCV_LIBS})

        # add ncnn2table tool to a virtual project group
        set_property(TARGET ncnn2table PROPERTY FOLDER "tools/optimization")
    endif()

    add_executable(ncnn2int8 ncnn2int8.cpp)
    target_link_libraries(ncnn2int8 PRIVATE ncnn)

    # add ncnn2int8 tool to a virtual project group
    set_property(TARGET ncnn2int8 PROPERTY FOLDER "tools/optimization")
else()
    message(WARNING "OpenCV not found, quantize tools won't be built")
endif()
