For some reason, if you're not happy with the binary size of the ncnn library, then here is the cheatsheet that helps you to build a minimal ncnn :P

### disable c++ rtti and exceptions

```
cmake -DNCNN_DISABLE_RTTI=ON -DNCNN_DISABLE_EXCEPTION=ON ..
```
* Cannot use RTTI and Exceptions when ncnn functions are called.

### disable vulkan support

```
cmake -DNCNN_VULKAN=OFF ..
```

* Cannot use GPU acceleration.

### disable NCNN_STDIO

```
cmake -DNCNN_STDIO=OFF ..
```

* Cannot load model from files, but can load model from memory or by Android Assets.

    Read more [here](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-alexnet.md#load-model).

### disable NCNN_STRING

```
cmake -DNCNN_STRING=OFF ..
```

* Cannot load human-readable param files with visible strings, but can load binary param.bin files.

    Read more [here](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-alexnet.md#strip-visible-string)

* Cannot identify blobs by string name when calling `Extractor::input / extract`, but can identify them by enum value in `id.h`.

    Read more [here](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-alexnet.md#input-and-output).

### drop pixel rotate and affine functions

```
cmake -DNCNN_PIXEL_ROTATE=OFF -DNCNN_PIXEL_AFFINE=OFF ..
```

* Cannot use functions doing rotatation and affine transformation like `ncnn::kanna_rotate_xx / ncnn::warpaffine_bilinear_xx`, but functions like `Mat::from_pixels / from_pixels_resize` are still available. 

### drop pixel functions

```
cmake -DNCNN_PIXEL_ROTATE=OFF -DNCNN_PIXEL_AFFINE=OFF ..
```

* Cannot use functions transfering from image to pixels like `Mat::from_pixels / from_pixels_resize / to_pixels / to_pixels_resize`, and need create a Mat and fill in data by hand.

### disable openmp

```
cmake -DNCNN_OPENMP=OFF ..
```

* Cannot use openmp multi-threading acceleration. If you want to run a model in single thread on your target machine, it is recommended to close the option.

### disable avx2 and arm82 optimized kernel

```
cmake -DNCNN_AVX2=OFF -DNCNN_ARM82=OFF ..
```

* Do not compile optimized kernels using avx2 / arm82 instruction set extensions. If your target machine does not support some of them, it is recommended to close the related options.

### disable runtime cpu instruction dispatch

```
cmake -DNCNN_RUNTIME_CPU=OFF ..
```

* Cannot check supported cpu instruction set extensions and use related optimized kernels in runtime.
* If you know which instruction set extensions are supported on your target machine like avx2 / arm82, you can open related options like `-DNCNN_AVX2=ON / -DNCNN_ARM82=ON` by hand and then sse2 / arm8 version kernels will not be compiled.

### drop layers not used

```
cmake -DWITH_LAYER_absval=OFF -DWITH_LAYER_bnll=OFF ..
```

* If your model does not include some layers, taking absval / bnll as a example above, you can drop them.
* Some key or dependency layers should not be dropped, like convolution / innerproduct, their dependency like padding / flatten, and activation like relu / clip.

### disable c++ stl

```
cmake -DNCNN_SIMPLESTL=ON ..
```

* STL provided by compiler is no longer depended on, and use `simplestl` provided by ncnn as a replacement. Users also can only use `simplestl` when ncnn functions are called.
* Usually with compiler parameters `-nodefaultlibs -fno-builtin -nostdinc++ -lc`
* Need cmake parameters `cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_STL=system` to avoid STL conflict when compiling to Android.

### drop optimized kernel not used

* Modify the source code under `ncnn/src/layer/arm/` to delete unnecessary optimized kernels or replace them with empty functions.
* You can also drop layers and related optimized kernels by `-DWITH_LAYER_absval=OFF` as mentioned above.

### drop operators from BinaryOp UnaryOp

* Modify `ncnn/src/layer/binaryop.cpp unaryop.cpp` and `ncnn/src/layer/arm/binaryop.cpp unaryop_arm.cpp` by hand to delete unnecessary operators.
