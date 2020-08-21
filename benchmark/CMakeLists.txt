add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn -lgcc)

# add benchncnn to a virtual project group
set_property(TARGET benchncnn PROPERTY FOLDER "benchmark")
