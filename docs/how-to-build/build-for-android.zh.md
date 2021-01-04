### 安装 android-ndk

传送门 http://developer.android.com/ndk/downloads/index.html

比如我把 android-ndk 解压到 /home/nihui/android-ndk-r15c
```bash
export ANDROID_NDK=/home/nihui/android-ndk-r15c
```

### 准备 android toolchain 文件

android.toolchain.cmake 这个文件可以从 $ANDROID_NDK/build/cmake 找到

(可选) 删除debug编译参数，缩小二进制体积 [android-ndk issue](https://github.com/android-ndk/ndk/issues/243)
```bash
# 用编辑器打开 $ANDROID_NDK/build/cmake/android.toolchain.cmake
# 删除 "-g" 这行
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

### 编译方法
```bash
mkdir build-android
cd build-android
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
    -DANDROID_PLATFORM=android-14 ..
make
make install
make package
```
没有遇到错误的话，sdk 包已经静静地在 build-android/dist 目录里等你了

这里简要介绍几个参数

ANDROID_ABI 是架构名字，"armeabi-v7a" 支持绝大部分手机硬件

ANDROID_ARM_NEON 是否使用 NEON 指令集，设为 ON 支持绝大部分手机硬件

ANDROID_PLATFORM 指定最低系统版本，"android-14" 就是 android-4.0

armv5的参数
```bash
ANDROID_ABI="armeabi"
```
armv8的参数
```bash
ANDROID_ABI="arm64-v8a"
```
x86的参数
```bash
ANDROID_ABI="x86"
```
x86_64的参数
```bash
ANDROID_ABI="x86_64"
```

### CMakeLists.txt 要注意的地方

开头 project(XXX) 之前要加
```cmake
if(CMAKE_TOOLCHAIN_FILE)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Installation Directory")
endif()
```
交叉编译通常不需要把文件装在编译主机上的，所以 CMAKE_INSTALL_PREFIX 设置为 build-android/install

使用 opencv 这类第三方库的时候，也要指定 android 版本的路径
```cmake
set(OpenCV_DIR "/home/nihui/opencv-2.4.11/sdk/native/jni")
find_package(OpenCV REQUIRED)
```

CMakeLists.txt 里头可以用 if(ANDROID) .... endif() 来判断是否给 android 编译

android 并不全是 arm 架构，如果要编译 neon 优化的源码文件，还要判断下处理器架构
当然最好还是别分成两个文件，在同一个 cpp 里用 __ARM_NEON 围起来
```cmake
set(XXX_SRCS matrix_test.cpp)
if((ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
    OR (ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")))
    # 这里是 arm 专门的源代码文件
    set(XXX_SRCS ${XXX_SRCS} matrix_mul_neon.cpp)
else()
    set(XXX_SRCS ${XXX_SRCS} matrix_mul_c.cpp)
endif()
```