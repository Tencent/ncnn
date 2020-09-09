### how to enable ncnn vulkan capablity

follow [the build and install instruction](how-to-build)

make sure you have installed vulkan sdk from [lunarg vulkan sdk website](https://vulkan.lunarg.com/sdk/home)

Usually, you can enable the vulkan compute inference feature by adding only one line of code to your application.

```cpp
// enable vulkan compute feature before loading
ncnn::Net net;
net.opt.use_vulkan_compute = 1;
```

### does my graphics device support vulkan

Some platforms have been tested and known working. In theory, if your platform support vulkan api, either 1.0 or 1.1, it shall work.

* Y = known work
* ? = shall work, not confirmed
* / = not applied

|    |windows|linux|android|mac|ios|
|---|---|---|---|---|---|
|intel|Y|Y|?|?|/|
|amd|Y|Y|/|?|/|
|nvidia|Y|Y|?|/|/|
|qcom|/|/|Y|/|/|
|apple|/|/|/|?|Y|
|arm|/|?|Y|/|/|

You can search [the vulkan database](https://vulkan.gpuinfo.org) to see if your device supports vulkan.

Some old buggy drivers may produce wrong result, that are blacklisted in ncnn and treated as non-vulkan capable device.
You could check if your device and driver have this issue with  [my conformance test here](vulkan-conformance-test).
Most of these systems are android with version lower than 8.1.

### why using vulkan over cuda/opencl/metal

In the beginning, I had no GPGPU programming experience, and I had to learn one.

vulkan is considered more portable and well supported by venders and the cross-platform low-overhead graphics api. As a contrast, cuda is only available on nvidia device, metal is only available on macos and ios, while loading opencl library is banned in android 7.0+ and does not work on ios.

### I got errors like "vkCreateComputePipelines failed -1000012000" or random stalls or crashes

Upgrade your vulkan driver.

[intel https://downloadcenter.intel.com/product/80939/Graphics-Drivers](https://downloadcenter.intel.com/product/80939/Graphics-Drivers)

[amd https://www.amd.com/en/support](https://www.amd.com/en/support)

[nvidia https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

### how to use ncnn vulkan on android

minimum android ndk version: android-ndk-r18b

minimum sdk platform api version: android-24

link your jni project with libvulkan.so

[The squeezencnn example](https://github.com/Tencent/ncnn/tree/master/examples/squeezencnn) have equipped gpu inference, you could take it as reference.

### how to use ncnn vulkan on ios

setup vulkan sdk (https://vulkan.lunarg.com/sdk/home#mac)

metal only works on real device with arm64 cpu (iPhone 5s and later)

link your project with MoltenVK framework and Metal

### what about the layers without vulkan support

These layers have vulkan support currently

AbsVal, BatchNorm, BinaryOp, Cast, Clip, Concat, Convolution, ConvolutionDepthWise, Crop, Deconvolution, DeconvolutionDepthWise, Dropout, Eltwise, Flatten, HardSigmoid, InnerProduct, Interp, LRN, Packing, Padding, Permute, Pooling(pad SAME not supported), PReLU, PriorBox, ReLU, Reorg, Reshape, Scale, ShuffleChannel, Sigmoid, Softmax, TanH, UnaryOp

For these layers without vulkan support, ncnn inference engine will automatically fallback to cpu path.

Thus, it is usually not a serious issue if your network only has some special head layers like SSD or YOLO. All examples in ncnn are known working properly with vulkan enabled.

### my model runs slower on gpu than cpu

The current vulkan inference implementation is far from the preferred state. Many handful optimization techniques are planned, such as winograd convolution, operator fusion, fp16 storage and arithmetic etc.

It is common that your model runs slower on gpu than cpu on arm devices like mobile phones, since we have quite good arm optimization in ncnn ;)
