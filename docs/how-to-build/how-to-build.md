### Git clone ncnn repo with submodule

```
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
```

- [Git clone ncnn repo with submodule](#git-clone-ncnn-repo-with-submodule)
- [Build for Linux](#build-for-linux)
  - [Nvidia Jetson](#nvidia-jetson)
  - [Raspberry Pi](#raspberry-pi)
  - [POWER](#power)
  - [Intel oneAPI](#intel-oneapi)
  - [Cross compile: Riscv-gnu-toolchain](#cross-compile-riscv-gnu-toolchain)
  - [Verification](#verification)
- [Build for Windows x64 using Visual Studio Community 2017](#build-for-windows-x64-using-visual-studio-community-2017)
- [Build for Windows x64 using MinGW-w64](#build-for-windows-x64-using-mingw-w64)
- [Build for Windows XP (x86)](#build-for-windows-xp-x86)
  - [Using MinGW-w64](#using-mingw-w64)
  - [Using Clang](#using-clang)
  - [Using Visual Studio (MSVC)](#using-visual-studio-msvc)
- [Build for macOS](#build-for-macos)
- [Build for ARM Cortex-A family with cross-compiling](#build-for-arm-cortex-a-family-with-cross-compiling)
- [Build for Hisilicon platform with cross-compiling](#build-for-hisilicon-platform-with-cross-compiling)
- [Build for AnyCloud platform with cross-compiling](#build-for-AnyCloud-platform-with-cross-compiling)
- [Build for Android](#build-for-android)
- [Build for iOS on macOS with xcode](#build-for-ios-on-macos-with-xcode)
- [Build for WebAssembly](#build-for-webassembly)
- [Build for AllWinner D1](#build-for-allwinner-d1)
- [Build for Loongson 2K1000](#build-for-loongson-2k1000)
- [Build for Termux on Android](#build-for-termux-on-android)
- [Build for QNX](#build-for-qnx)
- [Build for Nintendo 3DS Homebrew Launcher](#build-for-nintendo-3ds-homebrew-launcher)
- [Build for HarmonyOS with cross-compiling](#build-for-harmonyos-with-cross-compiling)
- [Build for ESP32 with cross-compiling](#build-for-esp32-with-cross-compiling)

***

### Build for Linux

Install required build dependencies:

* git
* g++
* cmake
* protocol buffer (protobuf) headers files and protobuf compiler
* (optional) LLVM OpenMP header files # If building with Clang, and multithreaded CPU inference is desired
* (optional) opencv  # For building examples

Generally if you have Intel, AMD or Nvidia GPU from last 10 years, Vulkan can be easily used.

On some systems there are no Vulkan drivers easily available at the moment (October 2020), so you might need to disable use of Vulkan on them. This applies to Raspberry Pi 3 (but there is experimental open source Vulkan driver in the works, which is not ready yet). Nvidia Tegra series devices (like Nvidia Jetson) should support Vulkan. Ensure you have most recent software installed for best experience.

On Debian, Ubuntu, or Raspberry Pi OS, you can install all required dependencies using:
```shell
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libomp-dev libopencv-dev
```
On Redhat or Centos, you can install all required dependencies using:
```shell
sudo yum install build-essential git cmake libprotobuf-dev protobuf-compiler libopencv-dev
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
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
```

You can add `-GNinja` to `cmake` above to use Ninja build system (invoke build using `ninja` or `cmake --build .`).

For Raspberry Pi 3 on 32bit OS, add `-DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake` to cmake. You can also consider disabling Vulkan support as the Vulkan drivers for Raspberry Pi are still not mature, but it doesn't hurt to build the support in, but not use it.

#### POWER

For POWER9 with Clang:

```shell
cd ncnn
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchains/power9le-linux-gnu-vsx.clang.toolchain.cmake -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
```

To use GCC instead, use the `power9le-linux-gnu-vsx.toolchain.cmake` toolchain file instead. Note that according to benchmarks, Clang appears to produce noticeably faster CPU inference than GCC for POWER9 targets. For fastest inference, use Clang 18 or higher; earlier versions of Clang may have impaired inference speed due to [Bug 49864](https://github.com/llvm/llvm-project/issues/49864) and [Bug 64664](https://github.com/llvm/llvm-project/issues/64664).

For POWER8 instead of POWER9, use the `power8le-linux-gnu-vsx.clang.toolchain.cmake` or `power8le-linux-gnu-vsx.toolchain.cmake` toolchain file instead. POWER8 will be slower than POWER9.

Note that the POWER toolchain files only support little-endian mode.

#### Intel oneAPI

Besides the prerequests in this section, Intel oneAPI BaseKit and HPCKit should be installed. They are available from https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html and https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html freely.

Intel oneAPI offers two kinds of compilers, the classic `icc/icpc` and the LLVM based `icx/icpx`. To build with these compilers, add `CC=icc CXX=icpc` or `CC=icx CXX=icpx` before the `cmake` command. When compiling with `icc/icpc`, cmake will warn that `xop`, `avx512`, and `bf16` extensions are not supported by the compiler, while `icx/icpx` works well.

Both of these compilers have been tested and passed the ncnn benchmark successfully. The results have been included in ncnn benchmark readme. Generally, `icx/icpx` are likely to show better performance than `icc/icpc` and the quantized models can benefit from the extensions `icx/icpx` supports.

#### Cross compile: Riscv-gnu-toolchain
Before compiling the whole project, toolchain must be installed.
[Reference: Riscv-gnu-toolchain build guide](https://github.com/riscv-collab/riscv-gnu-toolchain/blob/master/README.md)
```shell

# configure with vector extension.
./configure --prefix=/opt/riscv --enable-multilib --with-arch=rv64gcv

# configure without vector extension.
./configure --prefix=/opt/riscv --enable-multilib --with-arch=rv64gc

# it takes quite a long time:(
sudo make linux

```
Now you can build the project:
```shell
mkdir build-riscv
cd build-riscv
cmake -DDCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-unknown-linux-gnu.toolchain.cmake -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc) # or `make -j2` if your cpu isn't powerful enough.
```

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

> You can also search `x64 Native Tools Command Prompt for VS 2017` directly.

Download protobuf-3.11.2 from https://github.com/google/protobuf/archive/v3.11.2.zip

Build protobuf library:

```shell
cd <protobuf-root-dir>
mkdir protobuf_build
cd protobuf_build
cmake -A x64 -DCMAKE_INSTALL_PREFIX=%cd%/install -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```

Build ncnn library (replace `<protobuf-root-dir>` with a proper path):

```shell
cd <ncnn-root-dir>
mkdir -p protobuf_build
cd protobuf_build
cmake -A x64 -DCMAKE_INSTALL_PREFIX=%cd%/install -Dprotobuf_DIR=<protobuf-root-dir>/protobuf_build/install/cmake -DNCNN_VULKAN=ON ..
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```

Note: To speed up compilation process on multi core machines, configuring `cmake` to use `jom` or `ninja` using `-G` flag is recommended.

Note: For protobuf >=22.0 (Take v25.3 for example):

Build zlib:
```shell
git clone -b -v1.3.1 https://github.com/madler/zlib.git
cd zlib
mkdir build
cd build
cmake -A x64 -DCMAKE_INSTALL_PREFIX=%cd%/install ..
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```

Build protobuf library (replace `<zlib-root-dir>` with a proper path):
```shell
git clone -b v25.3 https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive

mkdir protobuf_build
cd protobuf_build
cmake -A x64 -DCMAKE_INSTALL_PREFIX=%cd%/install -DCMAKE_CXX_STANDARD=14 -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DZLIB_INCLUDE_DIR=<zlib-root-dir>\build\install\include -DZLIB_LIBRARY=<zlib-root-dir>\build\install\lib\zlib.lib -DABSL_PROPAGATE_CXX_STD=ON ../cmake
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```

Build ncnn library (replace `<zlib-root-dir>` and `<protobuf-root-dir>` with a proper path):

```shell
cd <ncnn-root-dir>
mkdir -p build
cd build
cmake -A x64 -DCMAKE_INSTALL_PREFIX=%cd%/install -DCMAKE_PREFIX_PATH=<protobuf-root-dir>/protobuf_build\install\cmake -DZLIB_INCLUDE_DIR=<zlib-root-dir>\build\install\include -DZLIB_LIBRARY=<zlib-root-dir>\build\install\lib\zlib.lib -Dabsl_DIR=<protobuf-root-dir>/protobuf_build\install\lib\cmake\absl -Dutf8_range_DIR=<protobuf-root-dir>/protobuf_build\install\lib\cmake\utf8_range -DNCNN_VULKAN=ON ..
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```

***

### Build for Windows x64 using MinGW-w64

Download MinGW-w64 toolchain from [winlibs](https://winlibs.com/) or [w64devkit](https://github.com/skeeto/w64devkit), add `bin` folder to environment variables.

Build ncnn library:

```shell
cd <ncnn-root-dir>
mkdir build
cd build
cmake -DNCNN_VULKAN=ON -G "MinGW Makefiles" ..
cmake --build . --config Release -j 4
cmake --build . --config Release --target install
```

***

### Build for Windows XP (x86)

> **Note:** Windows XP support is provided through collaborative contributions from [@Sugar-Baby](https://github.com/Sugar-Baby) and [@AtomAlpaca](https://github.com/AtomAlpaca).

#### Using MinGW-w64

Download mingw toolchain targeting 32 bit from [sourceforge](https://jaist.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/8.1.0/threads-posix/dwarf/i686-8.1.0-release-posix-dwarf-rt_v6-rev0.7z), extract and add environment variable named `MINGW32_ROOT_PATH` valued by `<your-path-to-mingw-root-path>`, and add `<your-path-to-mingw-root-path>/bin` to `PATH`.

```shell
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE="../toolchains/windows-xp-mingw.toolchain.cmake" -DNCNN_WINXP=ON -DNCNN_SIMPLEOCV=ON -DNCNN_AVX2=OFF -DNCNN_AVX=OFF -DNCNN_VULKAN=OFF .. -G "MinGW Makefiles"
cmake --build . --config Release -j 4
cmake --build . --config Release --target install
```

#### Using Clang

Clang requires libraries from mingw. Configure mingw toolchain targeting 32-bit as described in the [MinGW-w64 section](#using-mingw-w64).

Install Clang 6.0 or later.

```shell
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE="../toolchains/windows-xp-clang.toolchain.cmake" -DNCNN_WINXP=ON -DNCNN_SIMPLEOCV=ON -DNCNN_SIMPLEOMP=ON -DNCNN_AVX2=OFF -DNCNN_AVX=OFF .. -G "MinGW Makefiles"
cmake --build . --config Release -j 4
cmake --build . --config Release --target install
```

#### Using Visual Studio (MSVC)

Install v141_xp toolset for Windows XP:

1. Bring up the Visual Studio installer (Tools → Get Tools and Features)
2. Select Desktop development with C++
3. Select Windows XP support for C++ from the Summary section
4. Click Modify

```shell
mkdir build
cd build
cmake -A WIN32 -G "Visual Studio 17 2022" -T v141_xp -DNCNN_WINXP=ON -DNCNN_SIMPLEOCV=ON -DNCNN_OPENMP=OFF -DNCNN_AVX2=OFF -DNCNN_AVX=OFF -DNCNN_BUILD_WITH_STATIC_CRT=ON -DCMAKE_TOOLCHAIN_FILE="../toolchains/windows-xp-msvc.toolchain.cmake" ..
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```

**Note:** The MSVC toolchain uses the `v141_xp` platform toolset for Windows XP compatibility. Vulkan is disabled for XP compatibility, and advanced CPU features (AVX, AVX2, AVX512) are disabled to ensure compatibility with older processors.

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
wget https://sdk.lunarg.com/sdk/download/1.3.280.1/mac/vulkansdk-macos-1.3.280.1.dmg -O vulkansdk-macos-1.3.280.1.dmg
hdiutil attach vulkansdk-macos-1.3.280.1.dmg
sudo /Volumes/vulkansdk-macos-1.3.280.1/InstallVulkan.app/Contents/MacOS/InstallVulkan --root `pwd`/vulkansdk-macos-1.3.280.1 --accept-licenses --default-answer --confirm-command install
hdiutil detach /Volumes/vulkansdk-macos-1.3.280.1

# setup env
export VULKAN_SDK=`pwd`/vulkansdk-macos-1.3.280.1/macOS
```

```shell
cd <ncnn-root-dir>
mkdir -p build
cd build

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DPLATFORM=MAC -DARCHS="x86_64;arm64" \
    -DVulkan_LIBRARY=`pwd`/../vulkansdk-macos-1.3.280.1/macOS/lib/libMoltenVK.dylib \
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
new version of Hisilicon toolchain should be in `/opt/linux/x86-arm/` 

```shell
cd <ncnn-root-dir>
mkdir -p build
cd build

# Choose one cmake toolchain file depends on your target platform
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv300.toolchain.cmake ..
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv500.toolchain.cmake ..
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix100.toolchain.cmake ..
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix200.toolchain.cmake ..
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix210.toolchain.cmake ..

make -j$(nproc)
make install
```

***

### Build for AnyCloud platform with cross-compiling
Download and install AnyCloud SDK. And load env to set toolchain can access in shell

```shell
cd <ncnn-root-dir>
mkdir -p build
cd build

# Choose one cmake toolchain file depends on your target platform
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/anykav500.toolchain.cmake ..

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
# open $ANDROID_NDK/build/cmake/android.toolchain.cmake for ndk < r23
# or $ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake for ndk >= r23
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
    -DANDROID_PLATFORM=android-14 -DNCNN_VULKAN=ON ..

# If you use cmake >= 3.21 and ndk-r23
# you need to add -DANDROID_USE_LEGACY_TOOLCHAIN_FILE=False option for working optimization flags

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
    -DANDROID_PLATFORM=android-21 -DNCNN_VULKAN=ON ..

# If you use cmake >= 3.21 and ndk-r23
# you need to add -DANDROID_USE_LEGACY_TOOLCHAIN_FILE=False option for working optimization flags

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
    -DPLATFORM=OS64 -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 -DARCHS="arm64;arm64e" \
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
    -DPLATFORM=SIMULATORARM64 -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 -DARCHS="x86_64;arm64" \
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

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DPLATFORM=OS64 -DARCHS="arm64;arm64e" \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib/libomp.a" \
    -DNCNN_VULKAN=ON -DNCNN_BUILD_BENCHMARK=OFF ..

cmake --build . -j 4
cmake --build . --target install
```

Build library for iPhoneSimulator:

```shell
cd <ncnn-root-dir>
mkdir -p build-ios-sim
cd build-ios-sim

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DPLATFORM=SIMULATORARM64 -DARCHS="x86_64;arm64" \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/lib/libomp.a" \
    -DNCNN_BUILD_BENCHMARK=OFF ..

cmake --build . -j 4
cmake --build . --target install
```

Package glslang framework for iPhoneOS:
```shell
cd <ncnn-root-dir>

mkdir -p glslang.framework/Versions/A/Headers
mkdir -p glslang.framework/Versions/A/Resources
ln -s A glslang.framework/Versions/Current
ln -s Versions/Current/Headers glslang.framework/Headers
ln -s Versions/Current/Resources glslang.framework/Resources
ln -s Versions/Current/glslang glslang.framework/glslang
libtool -static build-ios/install/lib/libglslang.a build-ios/install/lib/libSPIRV.a -o build-ios/install/lib/libglslang_combined.a
lipo -create build-ios/install/lib/libglslang_combined.a -o glslang.framework/Versions/A/glslang
cp -r build/install/include/glslang glslang.framework/Versions/A/Headers/
sed -e 's/__NAME__/glslang/g' -e 's/__IDENTIFIER__/org.khronos.glslang/g' -e 's/__VERSION__/1.0/g' Info.plist > glslang.framework/Versions/A/Resources/Info.plist
```

Package ncnn framework for iPhoneOS:
```shell
cd <ncnn-root-dir>

mkdir -p ncnn.framework/Versions/A/Headers
mkdir -p ncnn.framework/Versions/A/Resources
ln -s A ncnn.framework/Versions/Current
ln -s Versions/Current/Headers ncnn.framework/Headers
ln -s Versions/Current/Resources ncnn.framework/Resources
ln -s Versions/Current/ncnn ncnn.framework/ncnn
lipo -create build-ios/install/lib/libncnn.a -o ncnn.framework/Versions/A/ncnn
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
./emsdk install 3.1.28
./emsdk activate 3.1.28

source emsdk_env.sh
```

Build without any extension for general compatibility:
```shell
mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=OFF -DNCNN_SIMPLEOCV=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF -DNCNN_AVX=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install
```

Build with WASM SIMD extension:
```shell
mkdir -p build-simd
cd build-simd
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=OFF -DNCNN_SIMPLEOCV=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=ON -DNCNN_AVX2=OFF -DNCNN_AVX=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install
```

Build with WASM Thread extension:
```shell
mkdir -p build-threads
cd build-threads
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DNCNN_THREADS=ON -DNCNN_OPENMP=ON -DNCNN_SIMPLEOMP=ON -DNCNN_SIMPLEOCV=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=OFF -DNCNN_AVX2=OFF -DNCNN_AVX=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install
```

Build with WASM SIMD and Thread extension:
```shell
mkdir -p build-simd-threads
cd build-simd-threads
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DNCNN_THREADS=ON -DNCNN_OPENMP=ON -DNCNN_SIMPLEOMP=ON -DNCNN_SIMPLEOCV=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_SSE2=ON -DNCNN_AVX2=OFF -DNCNN_AVX=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF ..
cmake --build . -j 4
cmake --build . --target install
```

Pick `build-XYZ/install` folder for further usage.

***

### Build for AllWinner D1

Download c906 toolchain package from https://www.xrvm.cn/community/download?id=4453617141140230144

```shell
tar -xf Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.1.0-20250522.tar.gz
export RISCV_ROOT_PATH=/home/nihui/osd/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.1.0
```

Build ncnn with riscv-v vector and simpleocv enabled:
```shell
mkdir -p build-c906
cd build-c906
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/c906-v310.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=release -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=OFF -DNCNN_XTHEADVECTOR=ON -DNCNN_ZFH=ON -DNCNN_ZVFH=OFF \
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

```shell
pkg install proot-distro
proot-distro install ubuntu
```

or you can see what system can be installed using `proot-distro list`

while you install ubuntu successfully, using `proot-distro login ubuntu` to login Ubuntu.

Then make ncnn,no need to install any other dependencies.

```shell
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

```shell
cd ../examples
../build/examples/squeezenet ../images/128-ncnn.png
```

### Build for QNX

Request license and download SDP from QNX Software Center: https://www.qnx.com/products/everywhere/ .

Setup QNX environment by invoking SDP's bundled script:

on Windows, open cmd and run
```batch
call C:\Users\zz\qnx800\qnxsdp-env.bat
```

on Linux, use /bin/bash and run
```shell
source /home/zz/qnx800/qnxsdp-env.sh
```

If it gives error `cannot find ld` on Linux, solve it by creaing link file:
```shell
cd ${QNX_HOST}/usr/bin/
ln -s aarch64-unknown-nto-qnx7.1.0-ld ld
```

Build ncnn with cmake in same shell:

```shell
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
mkdir -p build-qnx
cd build-qnx
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-qnx.toolchain.cmake ..
make -j$(nproc)
make install
```

Pick `build-qnx/install` folder for further usage.

### Build for Nintendo 3DS Homebrew Launcher
Install DevkitPRO toolchains
- If you are working on windows, download DevkitPro installer from [DevkitPro](https://devkitpro.org/wiki/Getting_Started).
- If you are using Ubuntu, the official guidelines from DevkitPro might not work for you. Try using the lines below to install
```shell
sudo apt-get update
sudo apt-get upgrade
wget https://apt.devkitpro.org/install-devkitpro-pacman
chmod +x ./install-devkitpro-pacman
sudo ./install-devkitpro-pacman
```

```shell
export DEVKITPRO=/opt/devkitpro
export DEVKITARM=/opt/devkitpro/devkitARM
export DEVKITPPC=/opt/devkitpro/devkitPPC
export export PATH=$/opt/devkitpro/tools/bin:$PATH
source ~/.profile
```
```shell
sudo dkp-pacman -Sy
sudo dkp-pacman -Syu
sudo dkp-pacman -S 3ds-dev
```
Copy the toolchain files from [3DS-cmake](https://github.com/Xtansia/3ds-cmake)(DevitARM3DS.cmake and the cmake folder) to NCNN's toolchains folder.
```
├── toolchains
│   ├── cmake
│   │   ├── bin2s_header.h.in
│   │   ├── FindCITRO3D.cmake
│   │   ├── FindCTRULIB.cmake
│   │   ├── FindFreetype.cmake
│   │   ├── FindJPEG.cmake
│   │   ├── FindPNG.cmake
│   │   ├── FindSF2D.cmake
│   │   ├── FindSFIL.cmake
│   │   ├── FindSFTD.cmake
│   │   ├── FindZLIB.cmake
│   │   ├── LibFindMacros.cmake
│   │   ├── Tools3DS.cmake
│   │   ├── ToolsGBA.cmake
│   │   └── try_add_imported_target.cmake
│   ├── DevkitArm3DS.cmake
...

```
Build with:
```shell
cd ncnn
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/DevkitArm3DS.cmake .. -DNCNN_SIMPLEOCV=ON -DNCNN_OPENMP=OFF -DNCNN_VFPV4=OFF ..
make -j4
make install
```
Modify the Makefile in Homebrew example to link and use NCNN in your 3DS Homebrew app.

***

### Build for HarmonyOS with cross-compiling
Download and install HarmonyOS SDK. The sdk installation directory is `/opt/ohos-sdk/linux`

```shell
cd <ncnn-root-dir>
mkdir -p build
cd build

export HM_SDK=/opt/ohos-sdk/linux

# Choose HarmonyOS sdk cmake toolchain file.
# If you want to enable vulkan, set -DNCNN_VULKAN=ON
# The HarmonyOS sdk does not support openmp, use ncnn simpleomp instead.
# Cross-compiling with CMake must use the one provided by the HarmonyOS SDK; otherwise, it won't recognize parameters like OHOS_PLATFORM, leading to compilation errors.
${HM_SDK}/native/build-tools/cmake/bin/cmake -DOHOS_STL=c++_static -DOHOS_ARCH=arm64-v8a -DOHOS_PLATFORM=OHOS -DCMAKE_TOOLCHAIN_FILE=${HM_SDK}/native/build/cmake/ohos.toolchain.cmake -DNCNN_VULKAN=ON -DNCNN_SIMPLEOMP=ON ..

make -j$(nproc)
make install
```

***

### Build for ESP32 with cross-compiling
Download esp-idf sdk
```shell
git clone https://github.com/espressif/esp-idf
cd esp-idf
git submodule update --init --recursive
```
Install esp-idf sdk and configure the environment
```shell
./install.sh
source export.sh
```
And for Windows, you should use:
```bash
install.bat # or `install.ps1`
export.bat
```
Note: python>=3.8, cmake>=3.24.0

Build ncnn library:
```shell
mkdir build-esp32
cd build-esp32
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/esp32.toolchain.cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
make install
```
Note: Make sure to compile in esp-idf environment.

The compiled ncnn library and headers can be put to the esp32 project to test.
