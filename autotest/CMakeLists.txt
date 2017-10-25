include(CTest)
include(cmake/DownloadProject.cmake)

enable_testing()

# download gtest at first
download_project(
    PROJ googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG master
    UPDATE_DISCONNECTED 1
)
add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

# include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

# add cpp and lib files
add_executable(autotest autotest.cpp)

target_link_libraries(autotest ${project_library_target_name} ${REQUIRED_LIBRARIES} gtest gmock)
target_link_libraries(autotest ncnn)
