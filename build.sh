#!/usr/bin/env bash

##### android armv7 without neon
mkdir -p build-android-armv7-without-neon
pushd build-android-armv7-without-neon
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=OFF -DANDROID_PLATFORM=android-19 -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-19 -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### android x86
mkdir -p build-android-x86
pushd build-android-x86
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="x86" -DANDROID_PLATFORM=android-19 -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### android x86_64
mkdir -p build-android-x86_64
pushd build-android-x86_64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="x86_64" -DANDROID_PLATFORM=android-21 -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### linux of hisiv300 (forgot the chip name) toolchain with neon and openmp
mkdir -p build-hisiv300-linux
pushd build-hisiv300-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv300.toolchain.cmake ..
make -j4
make install
popd

##### linux of hisiv500 (Hi3516CV200 and Hi3519V101) toolchain with neon and openmp
mkdir -p build-hisiv500-linux
pushd build-hisiv500-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv500.toolchain.cmake ..
make -j4
make install
popd

##### linux of hisiv600 (Hi3559V100) toolchain with neon and no openmp (due to only one cpu, close openmp)
mkdir -p build-hisiv600-linux
pushd build-hisiv600-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv600.toolchain.cmake ..
make -j4
make install
popd

##### linux of himix100 (Hi3559a) toolchain with neon and openmp
mkdir -p build-himix100-linux
pushd build-himix100-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix100.toolchain.cmake ..
make -j4
make install
popd

##### linux of arm-linux-gnueabi toolchain
mkdir -p build-arm-linux-gnueabi
pushd build-arm-linux-gnueabi
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabi.toolchain.cmake ..
make -j4
make install
popd

##### linux of arm-linux-gnueabihf toolchain
mkdir -p build-arm-linux-gnueabihf
pushd build-arm-linux-gnueabihf
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake ..
make -j4
make install
popd

##### linux of v831 toolchain with neon and openmp
mkdir -p build-v831-linux
pushd build-v831-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/v831.toolchain.cmake ..
make -j4
make install
popd

##### linux for aarch64-linux-gnu toolchain
mkdir -p build-aarch64-linux-gnu
pushd build-aarch64-linux-gnu
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j4
make install
popd

##### linux host system with gcc/g++
mkdir -p build-host-gcc-linux
pushd build-host-gcc-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/host.gcc.toolchain.cmake ..
make -j4
make install
popd

##### MacOS
mkdir -p build-mac
pushd build-mac
cmake   -DNCNN_OPENMP=OFF \
        -DNCNN_BENCHMARK=ON \
        ..
make -j8
make install
popd
