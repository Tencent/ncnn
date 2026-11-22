
if(NCNN_WEIGHT_QUANT)
    add_executable(ncnnllm2int ncnnllm2int.cpp)
    target_link_libraries(ncnnllm2int PRIVATE ncnn)

    add_executable(ncnnllm2table ncnnllm2table.cpp)
    target_link_libraries(ncnnllm2table PRIVATE ncnn)

    set_property(TARGET ncnnllm2int PROPERTY FOLDER "tools/optimization")
    set_property(TARGET ncnnllm2table PROPERTY FOLDER "tools/optimization")
    ncnn_install_tool(ncnnllm2int)
    ncnn_install_tool(ncnnllm2table)

    add_executable(ncnngguf2ncnn ncnngguf2ncnn.cpp)
    target_link_libraries(ncnngguf2ncnn PRIVATE ncnn)
    if(NOT WIN32)
        target_compile_definitions(ncnngguf2ncnn PRIVATE _FILE_OFFSET_BITS=64)
    endif()
    set_property(TARGET ncnngguf2ncnn PROPERTY FOLDER "tools/optimization")
    ncnn_install_tool(ncnngguf2ncnn)

    add_executable(ncnndit2int ncnndit2int.cpp)
    target_link_libraries(ncnndit2int PRIVATE ncnn)

    add_executable(ncnndit2table ncnndit2table.cpp)
    target_link_libraries(ncnndit2table PRIVATE ncnn)

    set_property(TARGET ncnndit2int PROPERTY FOLDER "tools/optimization")
    set_property(TARGET ncnndit2table PROPERTY FOLDER "tools/optimization")
    ncnn_install_tool(ncnndit2int)
    ncnn_install_tool(ncnndit2table)

    add_executable(ncnncomfy2int8 ncnncomfy2int8.cpp ../pnnx/src/json.cpp)
    target_link_libraries(ncnncomfy2int8 PRIVATE ncnn)
    set_property(TARGET ncnncomfy2int8 PROPERTY FOLDER "tools/optimization")
    ncnn_install_tool(ncnncomfy2int8)
endif()

if(NCNN_INT8)
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
        ncnn_install_tool(ncnn2table)
    endif()

    add_executable(ncnn2int8 ncnn2int8.cpp)
    target_link_libraries(ncnn2int8 PRIVATE ncnn)

    # add ncnn2int8 tool to a virtual project group
    set_property(TARGET ncnn2int8 PROPERTY FOLDER "tools/optimization")
    ncnn_install_tool(ncnn2int8)
endif()
