add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)

# add benchncnn to a virtual project group
SET_PROPERTY(TARGET benchncnn PROPERTY FOLDER "benchmark")
