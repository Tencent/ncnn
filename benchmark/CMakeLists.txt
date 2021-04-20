
if(MSVC)
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4996)
endif()

add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)

add_executable(benchncnn_c_api benchncnn_c_api.c)
target_link_libraries(benchncnn_c_api PRIVATE ncnn)


if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(benchncnn PRIVATE nodefs.js)
endif()

# add benchncnn to a virtual project group
set_property(TARGET benchncnn PROPERTY FOLDER "benchmark")
set_property(TARGET benchncnn_c_api PROPERTY FOLDER "benchmark")