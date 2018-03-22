#!/usr/bin/env bash

##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-14 ..
make
make install
popd

##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
make
make install
popd

##### android armv7 without neon
mkdir -p build-android-armv7-without-neon
pushd build-android-armv7-without-neon
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_PLATFORM=android-14 ..
make
make install
popd

##### ios armv7 arm64
mkdir -p build-ios
pushd build-ios
cmake -DCMAKE_TOOLCHAIN_FILE=../iosxc.toolchain.cmake ..
make
make install
popd

##### ios simulator i386 x86_64
mkdir -p build-ios-sim
pushd build-ios-sim
cmake -DCMAKE_TOOLCHAIN_FILE=../iossimxc.toolchain.cmake ..
make
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
