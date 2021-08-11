# vcpkg

## [vcpkg](https://github.com/microsoft/vcpkg) pr ncnn

## ./vcpkg install ncnn


## use CMakeLists.txt

```cmake
set(VCPKG_ROOT "${CMAKE_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE PATH "")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT})
 
project(main)
find_package(ncnn CONFIG REQUIRED)
```