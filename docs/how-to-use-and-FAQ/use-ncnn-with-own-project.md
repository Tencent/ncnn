### use ncnn with own project

After building ncnn, there is one or more library files generated. Consider integrating ncnn into your own project, you may use ncnn's installating provided cmake config file, or by manually specify library path(s).

**with cmake**

Ensure your project is built by cmake. Then in your project's CMakeLists.txt, add these lines:

```cmake
set(ncnn_DIR "<ncnn_install_dir>/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(my_target ncnn)
```
After this, both the header file search path ("including directories") and library paths are configured automatically, including vulkan related dependencies.

Note: you have to change `<ncnn_install_dir>` to your machine's directory, it is the directory that contains `ncnnConfig.cmake`.

For the prebuilt ncnn release packages, ncnnConfig is located in:
- for `ncnn-YYYYMMDD-windows-vs2019`, it is `lib/cmake/ncnn`
- for `ncnn-YYYYMMDD-android-vulkan`, it is `${ANDROID_ABI}/lib/cmake/ncnn` (`${ANDROID_ABI}` is defined in NDK's cmake toolchain file)
- other prebuilt release packages are with similar condition

**manually specify**

You may also manually specify ncnn library path and including directory. Note that if you use ncnn with vulkan, it is also required to specify vulkan related dependencies.

For example, on Visual Studio debug mode with vulkan required, the lib paths are:
```
E:\github\ncnn\build\vs2019-x64\install\lib\ncnnd.lib
E:\lib\VulkanSDK\1.2.148.0\Lib\vulkan-1.lib
E:\github\ncnn\build\vs2019-x64\install\lib\SPIRVd.lib
E:\github\ncnn\build\vs2019-x64\install\lib\glslangd.lib
E:\github\ncnn\build\vs2019-x64\install\lib\OGLCompilerd.lib
E:\github\ncnn\build\vs2019-x64\install\lib\OSDependentd.lib
```
And for its release mode, lib paths are:
```
E:\github\ncnn\build\vs2019-x64\install\lib\ncnn.lib
E:\lib\VulkanSDK\1.2.148.0\Lib\vulkan-1.lib
E:\github\ncnn\build\vs2019-x64\install\lib\SPIRV.lib
E:\github\ncnn\build\vs2019-x64\install\lib\glslang.lib
E:\github\ncnn\build\vs2019-x64\install\lib\OGLCompiler.lib
E:\github\ncnn\build\vs2019-x64\install\lib\OSDependent.lib
```