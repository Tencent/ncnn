
if(NCNN_PIXEL)
    if(NOT NCNN_SIMPLEOCV)
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
    endif()

    set(OpenCV_FOUND FALSE)

    if(OpenCV_FOUND)
        add_executable(ncnn2table ncnn2table.cpp)
        target_include_directories(ncnn2table PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(ncnn2table PRIVATE ncnn ${OpenCV_LIBS})
    elseif(NCNN_SIMPLEOCV)
        add_executable(ncnn2table ncnn2table.cpp)
        target_compile_definitions(ncnn2table PUBLIC USE_NCNN_SIMPLEOCV)
        target_link_libraries(ncnn2table PRIVATE ncnn)
    else()
        add_executable(ncnn2table ncnn2table.cpp imreadwrite.cpp)
        target_compile_definitions(ncnn2table PUBLIC USE_LOCAL_IMREADWRITE)
        target_link_libraries(ncnn2table PRIVATE ncnn)
    endif()

    # add ncnn2table tool to a virtual project group
    set_property(TARGET ncnn2table PROPERTY FOLDER "tools/optimization")
endif()

add_executable(ncnn2int8 ncnn2int8.cpp)
target_link_libraries(ncnn2int8 PRIVATE ncnn)

# add ncnn2int8 tool to a virtual project group
set_property(TARGET ncnn2int8 PROPERTY FOLDER "tools/optimization")
ncnn_install_tool(ncnn2table)
ncnn_install_tool(ncnn2int8)
