# 用 Visual Studio 编译

[TOC]

## 预先准备

Visual Studio 2015 / 2017 / 2019 / 2022 Preview 的 Community Edition 版本， 使用动态的 CRT 运行库

CMake,  推荐 >= 3.17 的版本

## 开始编译

### 最简编译

https://github.com/Tencent/ncnn.git

#### 命令提示符版本

```batch
mkdir build-vs2019
cd build-vs2019
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
cmake --install . --config Release
cmake --build . --config Debug
cmake --install . --config Debug
```

会安装在 build-vs2019/install 里头，debug 版本的库会带有 `d` 后缀。

#### x64 本机工具命令提示符 版本 （VS2022无X64）
ncnn
protobuf参照后文定义参数

```batch
mkdir build-vs2019
cd build-vs2019
cmake ..
cmake --build . 
cmake --install .  --config Debug

//默认build生成Debug版本；默认install安装Relase版本。 参照命令提示符版本
```


### 编译安装带 Vulkan 支持的 ncnn 库

#### 设备和 Vulkan 准备
确认设备支持 Vulkan， 安装显卡驱动。

下载和安装 Vulkan SDK: https://vulkan.lunarg.com/sdk/home

连同子模块一起，获取源码：
- 可从 http://github.com/Tencent/ncnn/releases 找到 "ncnn-YYYYMMDD-full-source.zip" 下载
- 或用 git 获取最新版本：

```batch
git clone https://github.com/tencent/ncnn
git submodule update --init
```

#### 编译安装 ncnn
```batch
mkdir build-vs2019
cd build-vs2019
cmake -G "Visual Studio 16 2019"  -A x64  -DCMAKE_INSTALL_PREFIX="%cd%/install"  -DNCNN_VULKAN=ON
cmake --build . --config Release
cmake --install . --config Release
cmake --build . --config Debug
cmake --install . --config Debug
```

### 编译安装 ncnn 库和模型转换工具

- 此步骤用于编译模型转换工具，可跳过，直接使用 https://convertmodel.com 工具转换

- 以下命令行均使用  **适用于 VS 2019 的 x64 本机工具命令提示** 

*注：若在 cmd / PowerShell 进行，需修改：*
- `-G"NMake Makefile"` 改为合适的 Generator 如 `-G "Visual Studio 16 2019" -A x64`
- `nmake` 改为 `cmake --build . --config Release`， 或打开 `.sln` 手动触发 `protobuf` / `ncnn` 项的构建
- `nmake install` 改为 `cmake --install . --config Release`，或打开 `.sln` 手动触发 `INSTALL` 项的构建


#### 编译安装 protobuf

用于生成 caffe2ncnn 和 onnx2ncnn 工具

https://github.com/google/protobuf/archive/v3.4.0.zip

我下载到 C:/Users/shuiz/source 解压缩

```batch
mkdir build-vs2019
cd build-vs2019
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%cd%/install" ^
    -Dprotobuf_BUILD_TESTS=OFF ^
    -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
nmake
nmake install
```

protobuf 会安装在 build-vs2019/install 里头

#### 编译安装 ncnn

https://github.com/Tencent/ncnn.git

cmake 命令中的 protobuf 路径要相应修改成自己的

```batch
mkdir build-vs2019
cd build-vs2019
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%cd%/install" ^
    -DProtobuf_INCLUDE_DIR=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2019/install/include ^
    -DProtobuf_LIBRARIES=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2019/install/lib/libprotobuf.lib ^
    -DProtobuf_PROTOC_EXECUTABLE=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2019/install/bin/protoc.exe ..
nmake
nmake install
```

ncnn 会安装在 build-vs2019/install 里头

ncnn 转换工具在 build-vs2019/tools 里头

#### mlir2ncnn

见 [build-mlir2ncnn](build-mlir2ncnn.md)

## 使用编译好的 ncnn 库

CMakeLists 里写
```cmake
set(ncnn_DIR "C:/Users/shuiz/source/ncnn/build-vs2019/install/lib/cmake/ncnn" CACHE PATH "包含 ncnnConfig.cmake 的目录")
find_package(ncnn REQUIRED)
target_link_libraries(my_target ncnn)
```

进一步了解 [use-ncnn-with-own-project](../how-to-use-and-FAQ/use-ncnn-with-own-project.md)
