### Git clone ncnn repo with submodule

```
$ git clone https://github.com/Tencent/ncnn.git
$ cd ncnn
$ git submodule update --init
```

* [Build for Linux / NVIDIA Jetson / Raspberry Pi](#build-for-linux)
* [Build for Windows x64 using VS2017](#build-for-windows-x64-using-visual-studio-community-2017)
* [Build for macOS](#build-for-macos)
* [Build for ARM Cortex-A family with cross-compiling](#build-for-arm-cortex-a-family-with-cross-compiling)
* [Build for Hisilicon platform with cross-compiling](#build-for-hisilicon-platform-with-cross-compiling)
* [Build for Android](#build-for-android)
* [Build for iOS on macOS with xcode](#build-for-ios-on-macos-with-xcode)
* [Build for WebAssembly](#build-for-webassembly)
* [Build for AllWinner D1](#build-for-allwinner-d1)
* [Build for Loongson 2K1000](#build-for-loongson-2k1000)
* [Build for Termux on Android](#Build-for-Termux-on-Android)

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

On some systems there are no Vulkan drivers easily available at the moment (October 2020), so you might need to disable use of Vulkan on them. This applies to Raspberry Pi 3 (but there is experimental open source Vulkan driver in the works, which is not ready yet). Nvidia Tegra series devices (like Nvidia Jetson) should support Vulkan. Ensure you have most recent software installed for best experience.

On Debian, Ubuntu or Raspberry Pi OS, you can install all required dependencies using: 
```shell
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev
```
To use Vulkan backend install Vulkan header files, a vulkan driver loader, GLSL to SPIR-V compiler and vulkaninfo tool. Preferably from your distribution repositories. Alternatively download and install full Vulkan SDK (about 200MB in size; it contains all header files, documentation and prebuilt loader, as well some extra tools and source code of everything) from https://vulkan.lunarg.com/sdk/home

```shell
wget https://sdk.lunarg.com/sdk/download/1.2.189.0/linux/vulkansdk-linux-x86_64-1.2.189.0.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.189.0.tar.gz
tar -xf vulkansdk-linux-x86_64-1.2.189.0.tar.gz
export VULKAN_SDK=$(pwd)/1.2.189.0/x86_64
```

To use Vulkan after building ncnn later, you will also need to have Vulkan driver for your GPU. For AMD and Intel GPUs these can be found in Mesa graphics driver, which usually is installed by default on all distros (i.e. `sudo apt install mesa-vulkan-drivers` on Debian/Ubuntu). For Nvidia GPUs the proprietary Nvidia driver must be downloaded and installed (some distros will allow easier installation in some way). After installing Vulkan driver, confirm Vulkan libraries and driver are working, by using `vulkaninfo` or `vulkaninfo | grep deviceType`, it should list GPU device type. If there are more than one GPU installed (including the case of integrated GPU and discrete GPU, commonly found in laptops), you might need to note the order of devices to use later on.

#### Nvidia Jetson

The Vulkan driver is a default component of the Linux For Tegra BSP release, check [the device list](https://developer.nvidia.com/embedded/vulkan).

```shell
cd ncnn
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
```

#### Raspberry Pi

Vulkan drivers do exists, but are not mature. You are free to experiment at your own discretion, and report results and performance.

```shell
cd ncnn
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
```

You can add `-GNinja` to `cmake` above to use Ninja build system (invoke build using `ninja` or `cmake --build .`).

For Rasberry Pi 3, add `-DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake -DPI3=ON` to cmake. You can also consider disabling Vulkan support as the Vulkan drivers for Rasberry Pi are still not mature, but it doesn't hurt to build the support in, but not use it.

#### Verification

Verify build by running some examples:

```shell
cd ../examples
../build/examples/squeezenet ../images/256-ncnn.png
[0 AMD RADV FIJI (LLVM 10.0.1)]  queueC=1[4]  queueG=0[1]  queueT=0[1]
[0 AMD RADV FIJI (LLVM 10.0.1)]  bugsbn1=0  buglbia=0  bugcopc=0  bugihfa=0
[0 AMD RADV FIJI (LLVM 10.0.1)]  fp16p=1  fp16s=1  fp16a=0  int8s=1  int8a=1
532 = 0.163452
920 = 0.093140
716 = 0.061584
```

You can also run benchmarks (the 4th argument is a GPU device index to use, refer to `vulkaninfo`, if you have more than one GPU):

```shell
cd ../benchmark
../build/benchmark/benchncnn 10 $(nproc) 0 0
[0 AMD RADV FIJI (LLVM 10.0.1)]  queueC=1[4]  queueG=0[1]  queueT=0[1]
[0 AMD RADV FIJI (LLVM 10.0.1)]  bugsbn1=0  buglbia=0  bugcopc=0  bugihfa=0
[0 AMD RADV FIJI (LLVM 10.0.1)]  fp16p=1  fp16s=1  fp16a=0  int8s=1  int8a=1
num_threads = 4
powersave = 0
gpu_device = 0
cooling_down = 1
          squeezenet  min =    4.68  max =    4.99  avg =    4.85
     squeezenet_int8  min =   38.52  max =   66.90  avg =   48.52
...
```

To run benchmarks on a CPU, set the 5th argument to `-1`.


***

### Build for Windows x64 using Visual Studio Community 2017

Download and Install Visual Studio Community 2017 from https://visualstudio.microsoft.com/vs/community/

Start the command prompt: `Start → Programs → Visual Studio 2017 → Visual Studio Tools → x64 Native Tools Command Prompt for VS 2017`

Download protobuf-3.4.0 from https://github.com/google/protobuf/archive/v3.4.0.zip

Build protobuf library:

```shell
cd <protobuf-root-dir>
mkdir build
cd build
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
nmake
nmake install
```
(optional) Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home

Build ncnn library (replace <protobuf-root-dir> with a proper path):

```shell
cd <ncnn-root-dir>
mkdir -p build
cd build
cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=<protobuf-root-dir>/build/install/include -DProtobuf_LIBRARIES=<protobuf-root-dir>/build/install/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=<protobuf-root-dir>/build/install/bin/protoc.exe -DNCNN_VULKAN=ON ..
nmake
nmake install
```

Note: To speed up compilation process on multi core machines, configuring `cmake` to use `jom` or `ninja` using `-G` flag is recommended.

***
### Build for macOS

We've published ncnn to [brew](https://formulae.brew.sh/formula/ncnn#default) now, you can just use following method to install ncnn if you have the Xcode Command Line Tools installed.

```shell
brew update
brew install ncnn
```

Or if you want to compile and build ncnn locally, first install Xcode or Xcode Command Line Tools according to your needs.

Then install `protobuf` and `libomp` via homebrew

```shell
brew install protobuf libomp
```

Download and install Vulkan SDK from <https://vulkan.lunarg.com/sdk/home>


```shell
wget https://sdk.lunarg.com/sdk/download/1.2.189.0/mac/vulkansdk-macos-1.2.189.0.dmg?Human=true -O vulkansdk-macos-1.2.189.0.dmg
hdiutil attach vulkansdk-macos-1.2.189.0.dmg
sudo /Volumes/vulkansdk-macos-1.2.189.0/InstallVulkan.app/Contents/MacOS/InstallVulkan --root `pwd`/vulkansdk-macos-1.2.189.0 --accept-licenses --default-answer --confirm-command install
hdiutil detach /Volumes/vulkansdk-macos-1.2.189.0

# setup env
export VULKAN_SDK=`pwd`/vulkansdk-macos-1.2.189.0/macOS
```

```shell
cd <ncnn-root-dir>
mkdir -p build
cd build

cmake -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
    -DVulkan_INCLUDE_DIR=`pwd`/../vulkansdk-macos-1.2.189.0/MoltenVK/include \
    -DVulkan_LIBRARY=`pwd`/../vulkansdk-macos-1.2.189.0/MoltenVK/dylib/macOS/libMoltenVK.dylib \
    -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..

cmake --build . -j 4
cmake --build . --target install
```

*Note: If you encounter `libomp` related errors during installation, you can also check our GitHub Actions at [here](https://github.com/Tencent/ncnn/blob/d91cccf/.github/workflows/macos-x64-gpu.yml#L50-L68) to install and use `openmp`.*
***

### Build for ARM Cortex-A family with cross-compiling
Download ARM toolchain from https://developer.arm.com/open-source/gnu-toolchain/gnu-a/downloads

```shell
export PATH="<your-toolchain-compiler-path>:${PATH}"
```

Alternatively install a cross-compiler provided by the distribution (i.e. on Debian / Ubuntu, you can do `sudo apt install g++-arm-linux-gnueabi g++-arm-linux-gnueabihf g++-aarch64-linux-gnu`).

Depending on your needs build one or more of the below targets.

AArch32 target with soft float (arm-linux-gnueabi)
```shell
cd <ncnn-root-dir>
mkdir -p build-arm-linux-gnueabi
cd build-arm-linux-gnueabi
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabi.toolchain.cmake ..
make -j$(nproc)
```

AArch32 target with hard float (arm-linux-gnueabihf)
```shell
cd <ncnn-root-dir>
mkdir -p build-arm-linux-gnueabihf
cd build-arm-linux-gnueabihf
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake ..
make -j$(nproc)
```

AArch64 GNU/Linux target (aarch64-linux-gnu)
```shell
cd <ncnn-root-dir>
mkdir -p build-aarch64-linux-gnu
cd build-aarch64-linux-gnu
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j$(nproc)
```

***

### Build for Hisilicon platform with cross-compiling
Download and install Hisilicon SDK. The toolchain should be in `/opt/hisi-linux/x86-arm`

```shell
cd <ncnn-root-dir>
mkdir -p build
cd build

# Choose one cmake toolchain file depends on your target platform
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv300.toolchain.cmake ..
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv500.toolchain.cmake ..
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix100.toolchain.cmake ..
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix200.toolchain.cmake ..

make -j$(nproc)
make install
```

***

### Build for Android
You can use the pre-build ncnn-android-lib.zip from https://github.com/Tencent/ncnn/releases

Download Android NDK from http://developer.android.com/ndk/downloads/index.html and install it, for example:

```shell
unzip android-ndk-r21d-linux-x86_64.zip
export ANDROID_NDK=<your-ndk-root-path>
```

(optional) remove the hardcoded debug flag in Android NDK [android-ndk issue](https://github.com/android-ndk/ndk/issues/243)
```
# open $ANDROID_NDK/build/cmake/android.toolchain.cmake
# delete "-g" line
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

Build armv7 library

```shell
cd <ncnn-root-dir>
mkdir -p build-android-armv7
cd build-android-armv7

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
    -DANDROID_PLATFORM=android-14 ..

# If you want to enable Vulkan, platform api version >= android-24 is needed
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
  -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..

make -j$(nproc)
make install
```

Pick `build-android-armv7/install` folder for further JNI usage.


Build aarch64 library:

```shell
cd <ncnn-root-dir>
mkdir -p build-android-aarch64
cd build-android-aarch64

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"\
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM=android-21 ..

# If you want to enable Vulkan, platform api version >= android-24 is needed
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..

make -j$(nproc)
make install
```

Pick `build-android-aarch64/install` folder for further JNI usage.

***

### Build for iOS on macOS with xcode
You can use the pre-build ncnn.framework glslang.framework and openmp.framework from https://github.com/Tencent/ncnn/releases

Install xcode

You can replace ```-DENABLE_BITCODE=0``` to ```-DENABLE_BITCODE=1``` in the following cmake arguments if you want to build bitcode enabled libraries.

Download and install openmp for multithreading inference feature on iPhoneOS
```shell
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/openmp-11.0.0.src.tar.xz
tar -xf openmp-11.0.0.src.tar.xz
cd openmp-11.0.0.src

# apply some compilation fix
sed -i'' -e '/.size __kmp_unnamed_critical_addr/d' runtime/src/z_Linux_asm.S
sed -i'' -e 's/__kmp_unnamed_critical_addr/___kmp_unnamed_critical_addr/g' runtime/src/z_Linux_asm.S

mkdir -p build-ios
cd build-ios

cmake -DCMAKE_TOOLCHAIN_FILE=<ncnn-root-dir>/toolchains/ios.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install \
    -DIOS_PLATFORM=OS -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 -DIOS_ARCH="armv7;arm64;arm64e" \
    -DPERL_EXECUTABLE=/usr/local/bin/perl \
    -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_USE_HWLOC=OFF ..

cmake --build . -j 4
cmake --build . --target install

# copy openmp library and header files to xcode toolchain sysroot
# <xcode-dir> is usually /Applications/Xcode.app or /Applications/Xcode-beta.app depends on your Xcode version
sudo cp install/include/* <xcode-dir>/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/include
sudo cp install/lib/libomp.a <xcode-dir>/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib
```

Download and install openmp for multithreading inference feature on iPhoneSimulator
```shell
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/openmp-11.0.0.src.tar.xz
tar -xf openmp-11.0.0.src.tar.xz
cd openmp-11.0.0.src

# apply some compilation fix
sed -i'' -e '/.size __kmp_unnamed_critical_addr/d' runtime/src/z_Linux_asm.S
sed -i'' -e 's/__kmp_unnamed_critical_addr/___kmp_unnamed_critical_addr/g' runtime/src/z_Linux_asm.S

mkdir -p build-ios-sim
cd build-ios-sim

cmake -DCMAKE_TOOLCHAIN_FILE=<ncnn-root-dir>/toolchains/ios.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install \
    -DIOS_PLATFORM=SIMULATOR -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 -DIOS_ARCH="i386;x86_64" \
    -DPERL_EXECUTABLE=/usr/local/bin/perl \
    -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_USE_HWLOC=OFF ..

cmake --build . -j 4
cmake --build . --target install

# copy openmp library and header files to xcode toolchain sysroot
# <xcode-dir> is usually /Applications/Xcode.app or /Applications/Xcode-beta.app depends on your Xcode version
sudo cp install/include/* <xcode-dir>/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/include
sudo cp install/lib/libomp.a <xcode-dir>/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/lib
```

Package openmp framework:
```shell
cd <openmp-root-dir>

mkdir -p openmp.framework/Versions/A/Headers
mkdir -p openmp.framework/Versions/A/Resources
ln -s A openmp.framework/Versions/Current
ln -s Versions/Current/Headers openmp.framework/Headers
ln -s Versions/Current/Resources openmp.framework/Resources
ln -s Versions/Current/openmp openmp.framework/openmp
lipo -create build-ios/install/lib/libomp.a build-ios-sim/install/lib/libomp.a -o openmp.framework/Versions/A/openmp
cp -r build-ios/install/include/* openmp.framework/Versions/A/Headers/
sed -e 's/__NAME__/openmp/g' -e 's/__IDENTIFIER__/org.llvm.openmp/g' -e 's/__VERSION__/11.0/g' <ncnn-root-dir>/Info.plist > openmp.framework/Versions/A/Resources/Info.plist
```

Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
```shell
wget https://sdk.lunarg.com/sdk/download/1.2.189.0/mac/vulkansdk-macos-1.2.189.0.dmg?Human=true -O vulkansdk-macos-1.2.189.0.dmg
hdiutil attach vulkansdk-macos-1.2.189.0.dmg
sudo /Volumes/vulkansdk-macos-1.2.189.0/InstallVulkan.app/Contents/MacOS/InstallVulkan --root `pwd`/vulkansdk-macos-1.2.189.0 --accept-licenses --default-answer --confirm-command install
hdiutil detach /Volumes/vulkansdk-macos-1.2.189.0

# setup env
export VULKAN_SDK=`pwd`/vulkansdk-macos-1.2.189.0/macOS
```

Build library for iPhoneOS:

```shell
cd <ncnn-root-dir>
git submodule update --init
mkdir -p build-ios
cd build-ios

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=OS -DIOS_ARCH="armv7;arm64;arm64e" \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib/libomp.a" \
    -DNCNN_BUILD_BENCHMARK=OFF ..

# vulkan is only available on arm64 devices
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=OS64 -DIOS_ARCH="arm64;arm64e" \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib/libomp.a" \
    -DVulkan_INCLUDE_DIR=$VULKAN_SDK/../MoltenVK/include \
    -DVulkan_LIBRARY=$VULKAN_SDK/../MoltenVK/dylib/iOS/libMoltenVK.dylib \
    -DNCNN_VULKAN=ON -DNCNN_BUILD_BENCHMARK=OFF ..

cmake --build . -j 4
cmake --build . --target install
```

Build library for iPhoneSimulator:

```shell
cd <ncnn-root-dir>
mkdir -p build-ios-sim
cd build-ios-sim

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=SIMULATOR -DIOS_ARCH="i386;x86_64" \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/lib/libomp.a" \
    -DNCNN_BUILD_BENCHMARK=OFF ..

cmake --build . -j 4
cmake --build . --target install
```

Package glslang framework:
```shell
cd <ncnn-root-dir>

mkdir -p glslang.framework/Versions/A/Headers
mkdir -p glslang.framework/Versions/A/Resources
ln -s A glslang.framework/Versions/Current
ln -s Versions/Current/Headers glslang.framework/Headers
ln -s Versions/Current/Resources glslang.framework/Resources
ln -s Versions/Current/glslang glslang.framework/glslang
libtool -static build-ios/install/lib/libglslang.a build-ios/install/lib/libSPIRV.a build-ios/install/lib/libOGLCompiler.a build-ios/install/lib/libOSDependent.a -o build-ios/install/lib/libglslang_combined.a
libtool -static build-ios-sim/install/lib/libglslang.a build-ios-sim/install/lib/libSPIRV.a build-ios-sim/install/lib/libOGLCompiler.a build-ios-sim/install/lib/libOSDependent.a -o build-ios-sim/install/lib/libglslang_combined.a
lipo -create build-ios/install/lib/libglslang_combined.a build-ios-sim/install/lib/libglslang_combined.a -o glslang.framework/Versions/A/glslang
cp -r build/install/include/glslang glslang.framework/Versions/A/Headers/
sed -e 's/__NAME__/glslang/g' -e 's/__IDENTIFIER__/org.khronos.glslang/g' -e 's/__VERSION__/1.0/g' Info.plist > glslang.framework/Versions/A/Resources/Info.plist
```

Package ncnn framework:
```shell
cd <ncnn-root-dir>

mkdir -p ncnn.framework/Versions/A/Headers
mkdir -p ncnn.framework/Versions/A/Resources
ln -s A ncnn.framework/Versions/Current
ln -s Versions/Current/Headers ncnn.framework/Headers
ln -s Versions/Current/Resources ncnn.framework/Resources
ln -s Versions/Current/ncnn ncnn.framework/ncnn
lipo -create build-ios/install/lib/libncnn.a build-ios-sim/install/lib/libncnn.a -o ncnn.framework/Versions/A/ncnn
cp -r build-ios/install/include/* ncnn.framework/Versions/A/Headers/
sed -e 's/__NAME__/ncnn/g' -e 's/__IDENTIFIER__/com.tencent.ncnn/g' -e 's/__VERSION__/1.0/g' Info.plist > ncnn.framework/Versions/A/Resources/Info.plist
```

Pick `ncnn.framework` `glslang.framework` and `openmp.framework` folder for app development.

***

### Build for WebAssembly

Install Emscripten

```shell
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 2.0.8
./emsdk activate 2.0.8

source emsdk/emsdk_env.sh
```

Build without any extension for general compatibility:
```shell
mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF -DNCNN_AVX=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install
```

Build with WASM SIMD extension:
```shell
mkdir -p build-simd
cd build-simd
cmake -DCMAKE_TOOLCHAIN_FILE=../emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=ON -DNCNN_AVX2=OFF -DNCNN_AVX=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install
```

Build with WASM Thread extension:
```shell
mkdir -p build-threads
cd build-threads
cmake -DCMAKE_TOOLCHAIN_FILE=../emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DNCNN_THREADS=ON -DNCNN_OPENMP=ON -DNCNN_SIMPLEOMP=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF -DNCNN_AVX=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install
```

Build with WASM SIMD and Thread extension:
```shell
mkdir -p build-simd-threads
cd build-simd-threads
cmake -DCMAKE_TOOLCHAIN_FILE=../emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DNCNN_THREADS=ON -DNCNN_OPENMP=ON -DNCNN_SIMPLEOMP=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=ON -DNCNN_AVX2=OFF -DNCNN_AVX=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install
```

Pick `build-XYZ/install` folder for further usage.

***

### Build for AllWinner D1

Download c906 toolchain package from https://occ.t-head.cn/community/download?id=3913221581316624384

```shell
tar -xf riscv64-linux-x86_64-20210512.tar.gz
export RISCV_ROOT_PATH=/home/nihui/osd/riscv64-linux-x86_64-20210512
```

Build ncnn with riscv-v vector and simpleocv enabled:
```shell
mkdir -p build-c906
cd build-c906
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/c906.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=relwithdebinfo -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=ON \
    -DNCNN_SIMPLEOCV=ON -DNCNN_BUILD_EXAMPLES=ON ..
cmake --build . -j 4
cmake --build . --target install
```

Pick `build-c906/install` folder for further usage.

You can upload binary inside `build-c906/examples` folder and run on D1 board for testing.

***

### Build for Loongson 2K1000

For gcc version < 8.5, you need to fix msa.h header for workaround msa fmadd/fmsub/maddv/msubv bug.

Open ```/usr/lib/gcc/mips64el-linux-gnuabi64/8/include/msa.h```, find ```__msa_fmadd``` and ```__msa_fmsub``` and apply changes as the following
```c
// #define __msa_fmadd_w __builtin_msa_fmadd_w
// #define __msa_fmadd_d __builtin_msa_fmadd_d
// #define __msa_fmsub_w __builtin_msa_fmsub_w
// #define __msa_fmsub_d __builtin_msa_fmsub_d
#define __msa_fmadd_w(a, b, c) __builtin_msa_fmadd_w(c, b, a)
#define __msa_fmadd_d(a, b, c) __builtin_msa_fmadd_d(c, b, a)
#define __msa_fmsub_w(a, b, c) __builtin_msa_fmsub_w(c, b, a)
#define __msa_fmsub_d(a, b, c) __builtin_msa_fmsub_d(c, b, a)
```

find ```__msa_maddv``` and ```__msa_msubv``` and apply changes as the following
```c
// #define __msa_maddv_b __builtin_msa_maddv_b
// #define __msa_maddv_h __builtin_msa_maddv_h
// #define __msa_maddv_w __builtin_msa_maddv_w
// #define __msa_maddv_d __builtin_msa_maddv_d
// #define __msa_msubv_b __builtin_msa_msubv_b
// #define __msa_msubv_h __builtin_msa_msubv_h
// #define __msa_msubv_w __builtin_msa_msubv_w
// #define __msa_msubv_d __builtin_msa_msubv_d
#define __msa_maddv_b(a, b, c) __builtin_msa_maddv_b(c, b, a)
#define __msa_maddv_h(a, b, c) __builtin_msa_maddv_h(c, b, a)
#define __msa_maddv_w(a, b, c) __builtin_msa_maddv_w(c, b, a)
#define __msa_maddv_d(a, b, c) __builtin_msa_maddv_d(c, b, a)
#define __msa_msubv_b(a, b, c) __builtin_msa_msubv_b(c, b, a)
#define __msa_msubv_h(a, b, c) __builtin_msa_msubv_h(c, b, a)
#define __msa_msubv_w(a, b, c) __builtin_msa_msubv_w(c, b, a)
#define __msa_msubv_d(a, b, c) __builtin_msa_msubv_d(c, b, a)
```

Build ncnn with mips msa and simpleocv enabled:
```shell
mkdir -p build
cd build
cmake -DNCNN_DISABLE_RTTI=ON -DNCNN_DISABLE_EXCEPTION=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_MSA=ON -DNCNN_MMI=ON -DNCNN_SIMPLEOCV=ON ..
cmake --build . -j 2
cmake --build . --target install
```

Pick `build/install` folder for further usage.

You can run binary inside `build/examples` folder for testing.

***

### Build for Termux on Android

Install app Termux on your phone,and install Ubuntu in Termux.

 If you want use ssh, just install openssh in Termux

```
pkg install proot-distro
proot-distro install ubuntu
```

or you can see what system can be installed using `proot-distro list` 

while you install ubuntu successfully, using `proot-distro login ubuntu` to login Ubuntu.

Then make ncnn,no need to install any other dependencies.

```
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_EXAMPLES=ON -DNCNN_PLATFORM_API=OFF -DNCNN_SIMPLEOCV=ON ..
make -j$(nproc)
```

Then you can run a test

> on my Pixel 3 XL using Qualcomm 845,cant load `256-ncnn.png`

```
cd ../examples
../build/examples/squeezenet ../images/128-ncnn.png
```

