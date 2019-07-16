add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)

if(NCNN_VULKAN)
    target_link_libraries(benchncnn PRIVATE ${Vulkan_LIBRARY})
endif()
