# Compile with Visual Studio

[TOC]

## Prepare

Community Edition version of Visual Studio 2015 / 2017 / 2019 / **2022** Preview, using dynamic CRT runtime

CMake, version >= 3.17 recommended

## Compiling

### minimal compilation

https://github.com/Tencent/ncnn.git

#### command prompt version

```batch
mkdir build-vs2022
cd build-vs2022
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
cmake --install . --config Release
cmake --build . --config Debug
cmake --install . --config Debug
```

It will be installed in build-vs2022/install, and the debug version of the library will have a `d` suffix.

#### x64 native tools command prompt version (no x64 for VS2022)
ncnn
protobuf defines parameters with reference to the following

```batch
mkdir build-vs2022
cd build-vs2022
cmake ..
cmake --build . 
cmake --install .  --config Debug

// The default build generates the Debug version; the default install installs the Release version. Refer to the command prompt version
```


### Compile and install the ncnn library with Vulkan support

#### Device and Vulkan preparation
Confirm that the device supports Vulkan, and install the graphics driver.

Download and install the Vulkan SDK : https://vulkan.lunarg.com/sdk/home

Along with submodules, get the source code:
- The "ncnn-YYYYMMDD-full-source.zip" download can be found at http://github.com/Tencent/ncnn/releases
- Or get the latest version with git:

```batch
git clone https://github.com/tencent/ncnn
git submodule update --init
```

#### Compile and install ncnn
```batch
mkdir build-vs2022
cd build-vs2022
cmake -G "Visual Studio 17 2022"  -A x64  -DCMAKE_INSTALL_PREFIX="%cd%/install"  -DNCNN_VULKAN=ON
cmake --build . --config Release
cmake --install . --config Release
cmake --build . --config Debug
cmake --install . --config Debug
```

### Compile and install ncnn library and model conversion tool

- This step is used to compile the model conversion tool, which can be skipped and converted directly using the https://convertmodel.com tool

- The following command lines all use **x64 Native Tools Command Prompt for VS 2022**

*Note: If it is done in cmd / PowerShell, it needs to be modified:*
- `-G"NMake Makefile"` Change to a suitable Generator such as `-G "Visual Studio 17 2022" -A x64`
- `nmake` change to `cmake --build . --config Release`， or open `.sln` manually trigger the build of `protobuf` / `ncnn` items
- `nmake install` change to `cmake --install . --config Release`or open `.sln` manually trigger the build of `INSTALL` items


#### Compile and install protobuf

Tools for generating caffe2ncnn and onnx2ncnn

https://github.com/google/protobuf/archive/v3.4.0.zip

Download it to C:/Users/{{user-name}}/source and unzipped it

```batch
mkdir build-vs2022
cd build-vs2022
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%cd%/install" ^
    -Dprotobuf_BUILD_TESTS=OFF ^
    -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
nmake
nmake install
```

protobuf will be installed in build-vs2022/install

#### Compile and install ncnn

https://github.com/Tencent/ncnn.git

The protobuf path in the cmake command should be modified accordingly to its own

```batch
mkdir build-vs2022
cd build-vs2022
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%cd%/install" ^
    -DProtobuf_INCLUDE_DIR=C:/Users/{{user-name}}/source/protobuf-3.4.0/build-vs2022/install/include ^
    -DProtobuf_LIBRARIES=C:/Users/{{user-name}}/source/protobuf-3.4.0/build-vs2022/install/lib/libprotobuf.lib ^
    -DProtobuf_PROTOC_EXECUTABLE=C:/Users/{{user-name}}/source/protobuf-3.4.0/build-vs2022/install/bin/protoc.exe ..
nmake
nmake install
```

ncnn will be installed in build-vs2022/install

The ncnn conversion tool is in build-vs2022/tools

#### mlir2ncnn

see [build-mlir2ncnn](build-mlir2ncnn.md)

## Use the compiled ncnn library

Write in CMakeLists
```cmake
set(ncnn_DIR "C:/Users/{{user-name}}/source/ncnn/build-vs2022/install/lib/cmake/ncnn" CACHE PATH "Directory containing ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(my_target ncnn)
```

Learn more about [use-ncnn-with-own-project](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-own-project)

