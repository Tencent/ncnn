## 预先准备

Visual Studio 2017 Community Edition，使用动态的 CRT 运行库

以下命令行均使用  **适用于 VS 2017 的 x64 本机工具命令提示**

## 编译安装 protobuf

https://github.com/google/protobuf/archive/v3.4.0.zip

我下载到 C:/Users/shuiz/source 解压缩

```batch
mkdir build-vs2017
cd build-vs2017
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install ^
    -Dprotobuf_BUILD_TESTS=OFF ^
    -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
nmake
nmake install
```

protobuf 会安装在 build-vs2017/install 里头

## 编译安装 ncnn

https://github.com/Tencent/ncnn.git

cmake 命令中的 protobuf 路径要相应修改成自己的

```batch
mkdir build-vs2017
cd build-vs2017
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install ^
    -DProtobuf_INCLUDE_DIR=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2017/install/include ^
    -DProtobuf_LIBRARIES=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2017/install/lib/libprotobuf.lib ^
    -DProtobuf_PROTOC_EXECUTABLE=C:/Users/shuiz/source/protobuf-3.4.0/build-vs2017/install/bin/protoc.exe ..
nmake
nmake install
```

ncnn 会安装在 build-vs2017/install 里头

ncnn 转换工具在 build-vs2017/tools 里头

