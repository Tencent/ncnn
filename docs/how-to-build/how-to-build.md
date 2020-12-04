### Git clone ncnn repo with submodule

```
$ git clone https://github.com/Tencent/ncnn.git
$ cd ncnn
$ git submodule update --init
```

* [Build for Linux / NVIDIA Jetson / Raspberry Pi](#build-for-linux)
* [Build for Windows x64 using VS2017](#build-for-windows-x64-using-visual-studio-community-2017)
* [Build for MacOSX](#build-for-macosx)
* [Build for ARM Cortex-A family with cross-compiling](#build-for-arm-cortex-a-family-with-cross-compiling)
* [Build for Hisilicon platform with cross-compiling](#build-for-hisilicon-platform-with-cross-compiling)
* [Build for Android](#build-for-android)
* [Build for iOS on MacOSX with xcode](#build-for-ios-on-macosx-with-xcode)
* [Build for iOS on Linux with cctools-port](#build-for-ios-on-linux-with-cctools-port)
* [Build for Windows x32 using C++Builder](#build-for-windows-x32-using-CBuilder-Community-Edition)
***

### Build for Linux

Install required build dependencies:

* git
* g++
* cmake
* protocol buffer (protobuf) headers files and protobuf compiler
* vulkan header files and loader library
* glslang
* (optional) opencv  # For building examples

Generally if you have Intel, AMD or Nvidia GPU from last 10 years, Vulkan can be easily used.

On some systems there are no Vulkan drivers easily available at the moment (October 2020), so you might need to disable use of Vulkan on them. This applies to Raspberry Pi 3 (but there is experimental open source Vulkan driver in the works, which is not ready yet). Nvidia Tegra series devices (like Nvidia Jetson) should support Vulkan. Ensure you have most recent software installed for best expirience.

On Debian, Ubuntu or Raspberry Pi OS, you can install all required dependencies using: `sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev`

To use Vulkan backend install Vulkan header files, a vulkan driver loader, GLSL to SPIR-V compiler and vulkaninfo tool. Preferably from your distribution repositories. Alternatively download and install full Vulkan SDK (about 200MB in size; it contains all header files, documentation and prebuilt loader, as well some extra tools and source code of everything) from https://vulkan.lunarg.com/sdk/home

```
$ wget https://sdk.lunarg.com/sdk/download/1.2.154.0/linux/vulkansdk-linux-x86_64-1.2.154.0.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.154.0.tar.gz
$ tar -xf vulkansdk-linux-x86_64-1.2.154.0.tar.gz
$ export VULKAN_SDK=$(pwd)/1.2.154.0/x86_64
```

To use Vulkan after building ncnn later, you will also need to have Vulkan driver for your GPU. For AMD and Intel GPUs these can be found in Mesa graphics driver, which usually is installed by default on all distros (i.e. `sudo apt install mesa-vulkan-drivers` on Debian/Ubuntu). For Nvidia GPUs the propietary Nvidia driver must be downloaded and installed (some distros will allow easier installation in some way). After installing Vulkan driver, confirm Vulkan libraries and driver are working, by using `vulkaninfo` or `vulkaninfo | grep deviceType`, it should list GPU device type. If there are more than one GPU installed (including the case of integrated GPU and discrete GPU, commonly found in laptops), you might need to note the order of devices to use later on.

Nvidia Jetson devices the Vulkan support should be present in Nvidia provided SDK (Jetpack) or prebuild OS images.

Raspberry Pi Vulkan drivers do exists, but are not mature. You are free to experiment at your own discretion, and report results and performance.

```
$ cd ncnn
$ mkdir -p build
$ cd build
build$ cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..
build$ make -j$(nproc)
```

You can add `-GNinja` to `cmake` above to use Ninja build system (invoke build using `ninja` or `cmake --build .`).

For Nvidia Jetson devices, add `-DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake` to cmake.

For Rasberry Pi 3, add `-DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake -DPI3=ON` to cmake. You can also consider disabling Vulkan support as the Vulkan drivers for Rasberry Pi are still not mature, but it doesn't hurt to build the support in, but not use it.

Verify build by running some examples:

```
build$ cd ../examples
examples$ ../build/examples/squeezenet ../images/256-ncnn.png
[0 AMD RADV FIJI (LLVM 10.0.1)]  queueC=1[4]  queueG=0[1]  queueT=0[1]
[0 AMD RADV FIJI (LLVM 10.0.1)]  bugsbn1=0  buglbia=0  bugcopc=0  bugihfa=0
[0 AMD RADV FIJI (LLVM 10.0.1)]  fp16p=1  fp16s=1  fp16a=0  int8s=1  int8a=1
532 = 0.163452
920 = 0.093140
716 = 0.061584
example$
```

You can also run benchmarks (the 4th argument is a GPU device index to use, refer to `vulkaninfo`, if you have more than one GPU):

```
build$ cd ../benchmark
benchmark$ ../build/benchmark/benchncnn 10 $(nproc) 0 0
[0 AMD RADV FIJI (LLVM 10.0.1)]  queueC=1[4]  queueG=0[1]  queueT=0[1]
[0 AMD RADV FIJI (LLVM 10.0.1)]  bugsbn1=0  buglbia=0  bugcopc=0  bugihfa=0
[0 AMD RADV FIJI (LLVM 10.0.1)]  fp16p=1  fp16s=1  fp16a=0  int8s=1  int8a=1
warmup_loop_count = 5
[1] loop_count = 10
[2] num_threads = 32
[3] powersave = 0
[4] gpu_device = 0
[5] cooling_down_time = 1
          squeezenet        max =  297.54/s  median =  271.51/s
           mobilenet        max =  222.22/s  median =  215.89/s
        mobilenet_v2        max =  151.51/s  median =  138.20/s
...
```

To run benchmarks on a CPU, set the 5th argument to `-1`.


***

### Build for Windows x64 using Visual Studio Community 2017

Download and Install Visual Studio Community 2017 from https://visualstudio.microsoft.com/vs/community/

Start the command prompt: `Start → Programs → Visual Studio 2017 → Visual Studio Tools → x64 Native Tools Command Prompt for VS 2017`

Download protobuf-3.4.0 from https://github.com/google/protobuf/archive/v3.4.0.zip

Build protobuf library:

```
> cd <protobuf-root-dir>
> mkdir build
> cd build
> cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
> nmake
> nmake install
```
(optional) Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home

Build ncnn library (replace <protobuf-root-dir> with a proper path):

```
> cd <ncnn-root-dir>
> mkdir -p build
> cd build
> cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=<protobuf-root-dir>/build/install/include -DProtobuf_LIBRARIES=<protobuf-root-dir>/build/install/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=<protobuf-root-dir>/build/install/bin/protoc.exe -DNCNN_VULKAN=ON ..
> nmake
> nmake install
```

Note: To speed up compilation process on multi core machines, configuring `cmake` to use `jom` or `ninja` using `-G` flag is recommended.

***

### Build for MacOSX
Install xcode and protobuf

**Because the compiler bundled with xcode do not support OpenMP feature, you cannot enable the multithreading inference feature of ncnn library, if you build with xcode.**

```
# Install protobuf via homebrew
$ brew install protobuf
```

(optional) Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
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
$ cmake -DNCNN_VULKAN=ON ..
$ make -j4
$ make install
```

***

### Build for ARM Cortex-A family with cross-compiling
Download ARM toolchain from https://developer.arm.com/open-source/gnu-toolchain/gnu-a/downloads

```
$ export PATH="<your-toolchain-compiler-path>:${PATH}"
```

Alternatively install a cross-compiler provided by the distribution (i.e. on Debian / Ubuntu, you can do `sudo apt install g++-arm-linux-gnueabi g++-arm-linux-gnueabihf g++-aarch64-linux-gnu`).

Depending on your needs build one ore more of the below targets.

AArch32 target with soft float (arm-linux-gnueabi)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-arm-linux-gnueabi
$ cd build-arm-linux-gnueabi
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabi.toolchain.cmake ..
$ make -j$(nproc)
```

AArch32 target with hard float (arm-linux-gnueabihf)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-arm-linux-gnueabihf
$ cd build-arm-linux-gnueabihf
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake ..
$ make -j$(nproc)
```

AArch64 GNU/Linux target (aarch64-linux-gnu)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-aarch64-linux-gnu
$ cd build-aarch64-linux-gnu
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
$ make -j$(nproc)
```

***

### Build for Hisilicon platform with cross-compiling
Download and install Hisilicon SDK. The oolchain should be in `/opt/hisi-linux/x86-arm`

```
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build

# Choose one cmake toolchain file depends on your target platform
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv300.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv500.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix100.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix200.toolchain.cmake ..

$ make -j$(nproc)
$ make install
```

***

### Build for Android
You can use the pre-build ncnn-android-lib.zip from https://github.com/Tencent/ncnn/releases

Download Android NDK from http://developer.android.com/ndk/downloads/index.html and install it, for example:

```
$ unzip android-ndk-r21d-linux-x86_64.zip
$ export ANDROID_NDK=<your-ndk-root-path>
```

(optional) Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
```
$ https://sdk.lunarg.com/sdk/download/1.2.154.0/linux/vulkansdk-linux-x86_64-1.2.154.0.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.154.0.tar.gz
$ tar -xf vulkansdk-linux-x86_64-1.2.154.0.tar.gz

# setup env
$ export VULKAN_SDK=`pwd`/1.2.154.0/x86_64
```

Build armv7 library

```
$ cd <ncnn-root-dir>
$ mkdir -p build-android-armv7
$ cd build-android-armv7

$ cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
    -DANDROID_PLATFORM=android-14 ..

# If you want to enable Vulkan, platform api version >= android-24 is needed
$ cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
    -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..
$ make -j$(nproc)
$ make install
```

Pick `build-android-armv7/install` folder for further JNI usage.


Build aarch64 library:

```
$ cd <ncnn-root-dir>
$ mkdir -p build-android-aarch64
$ cd build-android-aarch64

$ cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"\
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 ..

# If you want to enable Vulkan, platform api version >= android-24 is needed
$ cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..
$ make -j$(nproc)
$ make install
```

Pick `build-android-aarch64/install` folder for further JNI usage.

***

### Build for iOS on MacOSX with xcode
You can use the pre-build ncnn.framework and openmp.framework from https://github.com/Tencent/ncnn/releases

Install xcode

**Because the compiler bundled with xcode do not support openmp feature, you cannot enable the multithreading inference feature of ncnn library, if you build with xcode.**

(optional) Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
```
$ wget https://sdk.lunarg.com/sdk/download/1.1.114.0/mac/vulkansdk-macos-1.1.114.0.tar.gz?Human=true -O vulkansdk-macos-1.1.114.0.tar.gz
$ tar -xf vulkansdk-macos-1.1.114.0.tar.gz

# setup env
$ export VULKAN_SDK=`pwd`/vulkansdk-macos-1.1.114.0/macOS
```

Build library for iPhoneOS:

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

Build library for iPhoneSimulator:

```
$ cd <ncnn-root-dir>
$ mkdir build-ios-sim
$ cd build-ios-sim
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=SIMULATOR ..
$ make -j4
$ make install
```

Package framework:
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

```

Pick `ncnn.framework` folder for app development.

***

### Build for iOS on Linux with cctools-port
You can use the pre-build `ncnn.framework` and `openmp.framework` from https://github.com/Tencent/ncnn/releases

Setup cross-compiling environment with https://github.com/tpoechtrager/cctools-port

**You can enable the multithreading inference feature of ncnn library, if you build with cctools-port.**

```
$ cd <ncnn-root-dir>
```

Edit `iosxc.toolchain.cmake` and `iossimxc.toolchain.cmake` files, and update `CMAKE_IOS_SDK_ROOT` variable to your cctools-port target path.

Build armv7 arm64 library:

```
$ cd <ncnn-root-dir>
$ mkdir -p build-ios
$ cd build-ios
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc.toolchain.cmake ..
$ make -j$(nproc)
$ make install
```

Build i386 / x86_64 simulator library:
```
$ cd <ncnn-root-dir>
$ mkdir -p build-ios-sim
$ cd build-ios-sim
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iossimxc.toolchain.cmake ..
$ make -j$(nproc)
$ make install
```

Package framework:
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

```

Pick `ncnn.framework` folder for app development.

***

### build for windows x32 using C++Builder Community Edition

Download and install CMake 3.10 from https://cmake.org/download/

```
Locate your CMake installation folder and the Modules\Platform subfolder. E.g. C:\Program Files\CMake\share\cmake-3.10\Modules\Platform
Locate the Windows-Embarcadero.cmake file and make a backup.
Copy Windows-Embarcadero.cmake from the Studio\20.0\cmake folder and overwrite the version in the CMake folder.
```

Download and install Ninja 1.8.2 from https://ninja-build.org/

Download and Install C++Builder Community Edition 10.3.3 from https://www.embarcadero.com/products/cbuilder/starter/free-download/

Start the command prompt: `Start → Programs → Embarcadero RAD Studio 10.3 → RAD Studio Command Prompt`


Build ncnn library :

```
> cd <ncnn-root-dir>
> mkdir -p build
> cd build
> cmake -DCMAKE_C_COMPILER=bcc32x.exe -DCMAKE_CXX_COMPILER=bcc32x.exe  -DCMAKE_BUILD_TYPE_INIT=Release -G Ninja -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF -DNCNN_THREADS=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF ..
> ninja
```

Note: To speed up compilation process on multi core machines, configuring `cmake` to use `jom` or `ninja` using `-G` flag is recommended.

***
