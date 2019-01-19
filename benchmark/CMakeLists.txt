
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

if(NCNN_VULKAN)
    include_directories(${Vulkan_INCLUDE_DIR})
endif()

add_executable(benchncnn benchncnn.cpp)
set_property(TARGET benchncnn PROPERTY COMPILE_FLAGS "-fpie")
set_property(TARGET benchncnn PROPERTY LINK_FLAGS "-pie")

if(NCNN_VULKAN)
    target_link_libraries(benchncnn ncnn ${Vulkan_LIBRARY})
else()
    target_link_libraries(benchncnn ncnn)
endif()
