#!/usr/bin/env bash

##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-19 ..
make -j4
make install
popd

##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
make -j4
make install
popd

##### android armv7 without neon
mkdir -p build-android-armv7-without-neon
pushd build-android-armv7-without-neon
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_PLATFORM=android-19 ..
make -j4
make install
popd

##### android x86
mkdir -p build-android-x86
pushd build-android-x86
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="x86" -DANDROID_PLATFORM=android-19 ..
make -j4
make install
popd

##### android x86_64
mkdir -p build-android-x86_64
pushd build-android-x86_64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="x86_64" -DANDROID_PLATFORM=android-21 ..
make -j4
make install
popd

##### android armv7 vulkan
mkdir -p build-android-armv7-vulkan
pushd build-android-armv7-vulkan
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### android aarch64 vulkan
mkdir -p build-android-aarch64-vulkan
pushd build-android-aarch64-vulkan
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### android x86 vulkan
mkdir -p build-android-x86-vulkan
pushd build-android-x86-vulkan
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="x86" -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### android x86_64 vulkan
mkdir -p build-android-x86_64-vulkan
pushd build-android-x86_64-vulkan
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="x86_64" -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..
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

##### ios armv7 arm64
mkdir -p build-ios
pushd build-ios
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc.toolchain.cmake -DENABLE_BITCODE=OFF ..
make -j4
make install
popd

##### ios armv7 arm64 bitcode
mkdir -p build-ios-bitcode
pushd build-ios-bitcode
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc.toolchain.cmake -DENABLE_BITCODE=ON ..
make -j4
make install
popd

##### ios simulator i386 x86_64
mkdir -p build-ios-sim
pushd build-ios-sim
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iossimxc.toolchain.cmake -DENABLE_BITCODE=OFF ..
make -j4
make install
popd

##### ios simulator i386 x86_64 bitcode
mkdir -p build-ios-sim-bitcode
pushd build-ios-sim-bitcode
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iossimxc.toolchain.cmake -DENABLE_BITCODE=ON ..
make -j4
make install
popd

##### ios arm64 vulkan
mkdir -p build-ios-vulkan
pushd build-ios-vulkan
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc-arm64.toolchain.cmake -DENABLE_BITCODE=OFF -DVulkan_INCLUDE_DIR=${VULKAN_SDK}/MoltenVK/include -DVulkan_LIBRARY=${VULKAN_SDK}/MoltenVK/iOS/MoltenVK.framework/MoltenVK -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### ios arm64 vulkan bitcode
mkdir -p build-ios-vulkan-bitcode
pushd build-ios-vulkan-bitcode
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc-arm64.toolchain.cmake -DENABLE_BITCODE=ON -DVulkan_INCLUDE_DIR=${VULKAN_SDK}/MoltenVK/include -DVulkan_LIBRARY=${VULKAN_SDK}/MoltenVK/iOS/MoltenVK.framework/MoltenVK -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### ios simulator x86_64 vulkan
mkdir -p build-ios-sim-vulkan
pushd build-ios-sim-vulkan
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iossimxc-x64.toolchain.cmake -DENABLE_BITCODE=OFF -DVulkan_INCLUDE_DIR=${VULKAN_SDK}/MoltenVK/include -DVulkan_LIBRARY=${VULKAN_SDK}/MoltenVK/iOS/MoltenVK.framework/MoltenVK -DNCNN_VULKAN=ON ..
make
make install
popd

##### ios simulator x86_64 vulkan bitcode
mkdir -p build-ios-sim-vulkan-bitcode
pushd build-ios-sim-vulkan-bitcode
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/iossimxc-x64.toolchain.cmake -DENABLE_BITCODE=ON -DVulkan_INCLUDE_DIR=${VULKAN_SDK}/MoltenVK/include -DVulkan_LIBRARY=${VULKAN_SDK}/MoltenVK/iOS/MoltenVK.framework/MoltenVK -DNCNN_VULKAN=ON ..
make -j4
make install
popd

##### MacOS
mkdir -p build-mac
pushd build-mac
cmake   -DNCNN_OPENMP=OFF \
        -DNCNN_OPENCV=ON \
        -DNCNN_BENCHMARK=ON \
        ..
make -j 8
make install
popd
