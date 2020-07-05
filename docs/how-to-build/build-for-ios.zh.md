### 安装 xcode 和 cmake

传送门 https://developer.apple.com/xcode/download

传送门 https://cmake.org/download

默认情况 cmake 命令行可能用不了，需要手工加在 PATH 里面
```bash
export PATH=/Applications/CMake.app/Contents/bin/:$PATH
```
 
### 准备 ios toolchain 文件

把 ios.toolchain.cmake 放到和 CMakeLists.txt 同一级的项目目录里
可以去 opencv 的 github 上弄来，自己稍微调整下。
传送门 https://github.com/Itseez/opencv/tree/master/platforms/ios/cmake

### 编译方法
```bash
mkdir build-ios
cd build-ios
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=../ios.toolchain.cmake \
    -DIOS_PLATFORM=iPhoneOS \
    -DCMAKE_OSX_ARCHITECTURES=armv7 ..
make
make install
make package
```
没有遇到错误的话，手机平台 armv7 库已经编好了

这里简要介绍几个参数

IOS_PLATFORM 是平台名字，iPhoneOS 是真实机器的 ios，iPhoneSimulator 是模拟器平台

CMAKE_OSX_ARCHITECTURES 指定架构，iPhoneOS 配套 armv7 armv7s arm64，iPhoneSimulator 配套 i386 x86_64

### 打包成 framework

手工新建目录 XXX.framework/Versions/A，还有些软链接
```bash
mkdir -p XXX.framework/Versions/A/Headers
mkdir -p XXX.framework/Versions/A/Resources
ln -s A XXX.framework/Versions/Current
ln -s Versions/Current/Headers XXX.framework/Headers
ln -s Versions/Current/Resources XXX.framework/Resources
ln -s Versions/Current/XXX XXX.framework/XXX
```
framework 里面的库是多架构的，得先把 5 种架构都编译出来，比如分别编译在

build-iPhoneOS-armv7

build-iPhoneOS-armv7s

build-iPhoneOS-arm64

build-iPhoneSimulator-i386

build-iPhoneSimulator-x86_64

### 合成胖子库(fat)
```bash
lipo -create \
    build-iPhoneOS-armv7/install/lib/libXXX.a \
    build-iPhoneOS-armv7s/install/lib/libXXX.a \
    build-iPhoneOS-arm64/install/lib/libXXX.a \
    build-iPhoneSimulator-i386/install/lib/libXXX.a \
    build-iPhoneSimulator-x86_64/install/lib/libXXX.a \
    -o XXX.framework/Versions/A/XXX
```
复制头文件和 Info.plist
```bash
cp -r build-iPhoneOS-armv7/install/include/* XXX.framework/Versions/A/Headers/
cp Info.plist XXX.framework/Versions/A/Resources/
```
压缩成 zip
```bash
zip -y -r XXX.framework.zip XXX.framework
```

### CMakeLists.txt 要注意的地方

开头 project(XXX) 之前要加
```cmake
if(CMAKE_TOOLCHAIN_FILE)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Installation Directory")
endif()
```
交叉编译通常不需要把文件装在主机上的，所以 CMAKE_INSTALL_PREFIX 设置为 build-android/install 这里

使用 opencv2.framework 这类第三方 sdk，需要指定这些 framework 的位置
```cmake
set(CMAKE_FRAMEWORK_PATH "/Users/nihui/Downloads")
add_definitions(-F ${CMAKE_FRAMEWORK_PATH})
```

ios 平台默认不允许生成动态库

add_library(XXX SHARED ${XXX_SRCS}) 并没有效果，越狱设备除外

本文中的 ios toolchain 文件默认指定使用 libc++，最低系统需求为 ios 6.0

如果要修改这个配置，在 ios.toolchain.cmake 文件里
```cmake
set (CMAKE_C_FLAGS_INIT "-isysroot ${CMAKE_OSX_SYSROOT} -miphoneos-version-min=6.0")
set (CMAKE_CXX_FLAGS_INIT "-stdlib=libc++ -fvisibility=hidden -fvisibility-inlines-hidden -isysroot ${CMAKE_OSX_SYSROOT} -miphoneos-version-min=6.0")
```

CMakeLists.txt 里头可以用 if(IOS) .... endif() 来判断是否给 ios 编译

ios 并不全是 arm 架构，如果要编译 neon 优化的源码文件，还要判断下处理器架构
当然最好还是别分成两个文件，在同一个 cpp 里用 __ARM_NEON 围起来
```cmake
set(XXX_SRCS matrix_test.cpp)
if((IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "armv7"))
    OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "armv7s"))
    OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "arm64")))
    # 这里是 arm 专门的源代码文件
    set(XXX_SRCS ${XXX_SRCS} matrix_mul_neon.cpp)
else()
    set(XXX_SRCS ${XXX_SRCS} matrix_mul_c.cpp)
endif()
```
