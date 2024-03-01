:: Set android ndk root
@ECHO OFF
@SETLOCAL
@SET ANDROID_NDK=<your-ndk-root_path, such as"E:\android-ndk-r18b">

:: Set ninja.exe
:: @SET NINJA_EXE=<your-ninja-exe_path, such as"D:\android\sdk\cmake\3.10.2.4988404\bin\ninja.exe">

:: android armv7
mkdir build-android-armv7-vulkan
pushd build-android-armv7-vulkan
cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK%/build/cmake/android.toolchain.cmake -DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%/prebuilt/windows-x86_64/bin/make.exe" -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-19 -DNCNN_VULKAN=ON ..
cmake --build . --parallel %NUMBER_OF_PROCESSORS%
cmake --build . --target install
popd

:: android aarch64
mkdir build-android-aarch64-vulkan
pushd build-android-aarch64-vulkan
cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK%/build/cmake/android.toolchain.cmake -DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%/prebuilt/windows-x86_64/bin/make.exe" -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -DNCNN_VULKAN=ON ..
cmake --build . --parallel %NUMBER_OF_PROCESSORS%
cmake --build . --target install
popd

:: android x86
mkdir build-android-x86
pushd build-android-x86
cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK%/build/cmake/android.toolchain.cmake -DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%/prebuilt/windows-x86_64/bin/make.exe" -DANDROID_ABI="x86" -DANDROID_PLATFORM=android-19 -DNCNN_VULKAN=ON ..
cmake --build . --parallel %NUMBER_OF_PROCESSORS%
cmake --build . --target install
popd

:: android x86_64
mkdir build-android-x86_64
pushd build-android-x86_64
cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK%/build/cmake/android.toolchain.cmake -DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%/prebuilt/windows-x86_64/bin/make.exe" -DANDROID_ABI="x86_64" -DANDROID_PLATFORM=android-21 -DNCNN_VULKAN=ON ..
cmake --build . --parallel %NUMBER_OF_PROCESSORS%
cmake --build . --target install
popd

@ENDLOCAL
