* [Build for Linux x86](#build-for-linux-x86)
* [Build for Windows x64 using VS2017](#build-for-windows-x64-using-visual-studio-community-2017)
* [Build for MacOSX](#build-for-macosx)
* [Build for Raspberry Pi 3](#build-for-raspberry-pi-3)
* [Build for NVIDIA Jetson](#build-for-nvidia-jetson)
* [Build for ARM Cortex-A family with cross-compiling](#build-for-arm-cortex-a-family-with-cross-compiling)
* [Build for Android](#build-for-android)
* [Build for iOS on MacOSX with xcode](#build-for-ios-on-macosx-with-xcode)
* [Build for iOS on Linux with cctools-port](#build-for-ios-on-linux-with-cctools-port)
* [Build for Hisilicon platform with cross-compiling](#build-for-hisilicon-platform-with-cross-compiling)

***

### Build for Linux x86
install g++ cmake protobuf

(optional) download and install vulkan-sdk from https://vulkan.lunarg.com/sdk/home
```
$ wget https://sdk.lunarg.com/sdk/download/1.1.114.0/linux/vulkansdk-linux-x86_64-1.1.114.0.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.1.114.0.tar.gz
$ tar -xf vulkansdk-linux-x86_64-1.1.114.0.tar.gz

# setup env
$ export VULKAN_SDK=`pwd`/1.1.114.0/x86_64
```

```
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build

# cmake option NCNN_VULKAN for enabling vulkan
$ cmake -DNCNN_VULKAN=ON ..

$ make -j4
```
install opencv for building example
```
$ cd <ncnn-root-dir>

uncomment add_subdirectory(examples)
 in CMakeLists.txt with your favourite editor

$ mkdir -p build
$ cd build
$ cmake ..
$ make -j4

copy examples/squeezenet_v1.1.param to build/examples
copy examples/squeezenet_v1.1.bin to build/examples

$ cd build/examples
$ ./squeezenet yourimage.jpg 

output top-3 class-id and score
you may refer examples/synset_words.txt to find the class name
404 = 0.990290
908 = 0.004464
405 = 0.003941
```

***

### Build for Windows x64 using Visual Studio Community 2017

install Visual Studio Community 2017
```
download Visual Studio Community 2017 from https://visualstudio.microsoft.com/vs/community/
install it
Start → Programs → Visual Studio 2017 → Visual Studio Tools → x64 Native Tools Command Prompt for VS 2017
```
build protobuf library
```
download protobuf-3.4.0 from https://github.com/google/protobuf/archive/v3.4.0.zip
> cd <protobuf-root-dir>
> mkdir build-vs2017
> cd build-vs2017
> cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
> nmake
> nmake install
```
(optional) download and install vulkan-sdk from https://vulkan.lunarg.com/sdk/home

launch VulkanSDK-1.1.114.0-Installer.exe and install

build ncnn library (replace <protobuf-root-dir> with your path)
```
> cd <ncnn-root-dir>
> mkdir -p build-vs2017
> cd build-vs2017

# cmake option NCNN_VULKAN for enabling vulkan
> cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=<protobuf-root-dir>/build-vs2017/install/include -DProtobuf_LIBRARIES=<protobuf-root-dir>/build-vs2017/install/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=<protobuf-root-dir>/build-vs2017/install/bin/protoc.exe -DNCNN_VULKAN=ON ..

> nmake
> nmake install

pick build-vs2017/install folder for further usage
```

***

### Build for MacOSX
install xcode and protobuf

**Because the compiler bundled with xcode do not support openmp feature, you cannot enable the multithreading inference feature of ncnn library, if you build with xcode.**

```
# install protobuf via homebrew
$ brew install protobuf
```

(optional) download and install vulkan-sdk from https://vulkan.lunarg.com/sdk/home
```
$ wget https://sdk.lunarg.com/sdk/download/1.1.114.0/mac/vulkansdk-macos-1.1.114.0.tar.gz?Human=true -O vulkansdk-macos-1.1.114.0.tar.gz
$ tar -xf vulkansdk-macos-1.1.114.0.tar.gz

# setup env
$ export VULKAN_SDK=`pwd`/vulkansdk-macos-1.1.114.0/macOS
```

```
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build

# cmake option NCNN_VULKAN for enabling vulkan
$ cmake -DNCNN_VULKAN=ON ..

$ make -j4
$ make install
```

pick build/install folder for further usage

***

### Build for Raspberry Pi 3
install g++ cmake protobuf
```
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake -DPI3=ON ..
$ make -j4
$ make install
```

pick build/install folder for further usage

***

### Build for NVIDIA Jetson
#### download Vulkan SDK from NVIDIA
please click the `Vulkan SDK File` link on [https://developer.nvidia.com/embedded/vulkan](https://developer.nvidia.com/embedded/vulkan), at the time of writing we got `Vulkan_loader_demos_1.1.100.tar.gz`

scp the downloaded SDK to your Jetson device

```bash
scp Vulkan_loader_demos_1.1.100.tar.gz USERNAME@JETSON_IP:~/
```

from this monment on, we will work on the Jetson device
```bash
ssh USERNAME@JETSON_IP
```

#### install Vulkan SDK

```bash
cd ~/Vulkanloader_demos_1.1.100
sudo cp loader/libvulkan.so.1.1.100 /usr/lib/aarch64-linux-gnu/
cd /usr/lib/aarch64-linux-gnu/
sudo rm -rf libvulkan.so.1 libvulkan.so
sudo ln -s libvulkan.so.1.1.100 libvulkan.so
sudo ln -s libvulkan.so.1.1.100 libvulkan.so.1
cd ~/
```

#### install glslang dependency
```
# glslang is a dependency of Tencent/ncnn
git clone --depth=1 https://github.com/KhronosGroup/glslang.git
cd glslang
# assure that SPIR-V generated from HLSL is legal for Vulkan
./update_glslang_sources.py
mkdir -p build && cd build
sudo make -j`nproc` install && cd ..
```

#### compile ncnn
```
git clone https://github.com/Tencent/ncnn.git
# while aarch64-linux-gnu.toolchain.cmake would compile Tencent/ncnn as well
# but why not compile with more native features w
cd ncnn && mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake -DNCNN_VULKAN=ON -DCMAKE_BUILD_TYPE=Release ..
make -j`nproc`
sudo make install
```

***

### Build for ARM Cortex-A family with cross-compiling
download ARM toolchain from https://developer.arm.com/open-source/gnu-toolchain/gnu-a/downloads
```
$ export PATH=<your-toolchain-compiler-path>:$PATH
```
AArch32 target with soft float (arm-linux-gnueabi)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-arm-linux-gnueabi
$ cd build-arm-linux-gnueabi
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabi.toolchain.cmake ..
$ make -j4
$ make install
```
AArch32 target with hard float (arm-linux-gnueabihf)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-arm-linux-gnueabihf
$ cd build-arm-linux-gnueabihf
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake ..
$ make -j4
$ make install
```
AArch64 GNU/Linux target (aarch64-linux-gnu)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-aarch64-linux-gnu
$ cd build-aarch64-linux-gnu
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
$ make -j4
$ make install
```

pick build-XXXXX/install folder for further usage

***

### Build for Android
you can use the pre-build ncnn-android-lib.zip from https://github.com/Tencent/ncnn/releases

install android-ndk
```
download android-ndk from http://developer.android.com/ndk/downloads/index.html
$ unzip android-ndk-r21d-linux-x86_64.zip
$ export ANDROID_NDK=<your-ndk-root-path>
```
(optional) drop debug compile flag to reduce binary size due to [android-ndk issue](https://github.com/android-ndk/ndk/issues/243)
```
# edit $ANDROID_NDK/build/cmake/android.toolchain.cmake with your favorite editor
# remove "-g" line
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

(optional) download and install vulkan-sdk from https://vulkan.lunarg.com/sdk/home
```
$ wget https://sdk.lunarg.com/sdk/download/1.1.114.0/linux/vulkansdk-linux-x86_64-1.1.114.0.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.1.114.0.tar.gz
$ tar -xf vulkansdk-linux-x86_64-1.1.114.0.tar.gz

# setup env
$ export VULKAN_SDK=`pwd`/1.1.114.0/x86_64
```

build armv7 library
```
$ cd <ncnn-root-dir>
$ mkdir -p build-android-armv7
$ cd build-android-armv7

$ cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
    -DANDROID_PLATFORM=android-14 ..

# if you want to enable vulkan, platform api version >= android-24 is needed
$ cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
    -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..

$ make -j4
$ make install

pick build-android-armv7/install folder for further jni usage
```
build aarch64 library
```
$ cd <ncnn-root-dir>
$ mkdir -p build-android-aarch64
$ cd build-android-aarch64

$ cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 ..

# if you want to enable vulkan, platform api version >= android-24 is needed
$ cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..

$ make -j4
$ make install

pick build-android-aarch64/install folder for further jni usage
```

***

### Build for iOS on MacOSX with xcode
you can use the pre-build ncnn.framework and openmp.framework from https://github.com/Tencent/ncnn/releases

install xcode

**Because the compiler bundled with xcode do not support openmp feature, you cannot enable the multithreading inference feature of ncnn library, if you build with xcode.**

(optional) download and install vulkan-sdk from https://vulkan.lunarg.com/sdk/home
```
$ wget https://sdk.lunarg.com/sdk/download/1.1.114.0/mac/vulkansdk-macos-1.1.114.0.tar.gz?Human=true -O vulkansdk-macos-1.1.114.0.tar.gz
$ tar -xf vulkansdk-macos-1.1.114.0.tar.gz

# setup env
$ export VULKAN_SDK=`pwd`/vulkansdk-macos-1.1.114.0/macOS
```

build library for iPhoneOS
```
$ cd <ncnn-root-dir>
$ mkdir build-ios
$ cd build-ios

$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=OS ..

# vulkan is only available on arm64 devices
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=OS64 -DVulkan_INCLUDE_DIR=`pwd`/vulkansdk-macos-1.1.114.0/MoltenVK/include -DVulkan_LIBRARY=`pwd`/vulkansdk-macos-1.1.114.0/MoltenVK/iOS/dynamic/libMoltenVK.dylib -DNCNN_VULKAN=ON ..

$ make -j4
$ make install
```

build library for iPhoneSimulator
```
$ cd <ncnn-root-dir>
$ mkdir build-ios-sim
$ cd build-ios-sim
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=SIMULATOR ..
$ make -j4
$ make install
```
package framework
```
$ cd <ncnn-root-dir>
$ mkdir -p ncnn.framework/Versions/A/Headers
$ mkdir -p ncnn.framework/Versions/A/Resources
$ ln -s A ncnn.framework/Versions/Current
$ ln -s Versions/Current/Headers ncnn.framework/Headers
$ ln -s Versions/Current/Resources ncnn.framework/Resources
$ ln -s Versions/Current/ncnn ncnn.framework/ncnn
$ lipo -create \
    build-ios/install/lib/libncnn.a \
    build-ios-sim/install/lib/libncnn.a \
    -o ncnn.framework/Versions/A/ncnn
$ cp -r build-ios/install/include/* ncnn.framework/Versions/A/Headers/
$ cp Info.plist ncnn.framework/Versions/A/Resources/

pick ncnn.framework folder for app development
```

***

### Build for iOS on Linux with cctools-port
you can use the pre-build ncnn.framework and openmp.framework from https://github.com/Tencent/ncnn/releases

setup cross-compiling environment with https://github.com/tpoechtrager/cctools-port

**you can enable the multithreading inference feature of ncnn library, if you build with cctools-port.**

```
$ cd <ncnn-root-dir>

change CMAKE_IOS_SDK_ROOT variable to your cctools-port target path
 in iosxc.toolchain.cmake and iossimxc.toolchain.cmake with your favourite editor
```
build armv7 arm64 library
```
$ cd <ncnn-root-dir>
$ mkdir -p build-ios
$ cd build-ios
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc.toolchain.cmake ..
$ make
$ make install
```
build i386 x86_64 simulator library
```
$ cd <ncnn-root-dir>
$ mkdir -p build-ios-sim
$ cd build-ios-sim
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iossimxc.toolchain.cmake ..
$ make
$ make install
```
package framework
```
$ cd <ncnn-root-dir>
$ mkdir -p ncnn.framework/Versions/A/Headers
$ mkdir -p ncnn.framework/Versions/A/Resources
$ ln -s A ncnn.framework/Versions/Current
$ ln -s Versions/Current/Headers ncnn.framework/Headers
$ ln -s Versions/Current/Resources ncnn.framework/Resources
$ ln -s Versions/Current/ncnn ncnn.framework/ncnn
$ lipo -create \
    build-ios/install/lib/libncnn.a \
    build-ios-sim/install/lib/libncnn.a \
    -o ncnn.framework/Versions/A/ncnn
$ cp -r build-ios/install/include/* ncnn.framework/Versions/A/Headers/
$ cp Info.plist ncnn.framework/Versions/A/Resources/

pick ncnn.framework folder for app development
```

***

### Build for Hisilicon platform with cross-compiling
download and install Hisilicon SDK
```
# the path that toolchain should be installed in
$ ls /opt/hisi-linux/x86-arm
```
```
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build

# choose one cmake toolchain file depends on your target platform
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv300.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv500.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix100.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix200.toolchain.cmake ..

$ make -j4
$ make install
```

pick build/install folder for further usage
