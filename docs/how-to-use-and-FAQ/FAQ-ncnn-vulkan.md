### how to enable ncnn vulkan capability

follow [the build and install instruction](https://github.com/Tencent/ncnn/blob/master/docs/how-to-build/how-to-build.md)

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
|apple|/|/|/|Y|Y|
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

### vulkan device not found / extra high cpu utility while vulkan is enabled on nvidia gpu

There are severel reasons could lead to this outcome. First please check your driver status with `nvidia-smi`. If you have correctly installed your driver, you should see something like this:

```bash
$ nvidia-smi
Sat Mar 06 19:53:16 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 451.48       Driver Version: 451.48       CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1060   WDDM  | 00000000:02:00.0 Off |                  N/A |
| N/A   31C    P8     5W /  N/A |     90MiB /  6144MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

If `nvidia-smi` crashes or cannot be found, please reinstall your graphics driver.

If ncnn *is* utilizing the Tesla GPU, you can see your program in the `Processes` block at the bottom. In that case, it's likely some operators are not yet supported in Vulkan, and have fallbacked to the CPU, thus leading to a low utilization of the GPU.

If you *couldn't* find your process running, plase check the active driver model, which can be found to the right of your device name. For Geforce and Titan GPUs, the default driver model is WDDM (Windows Desktop Driver Model), which supports both rendering graphics as well as computing. But for Tesla GPUs, without configuration, the driver model is defualted to TCC ([Tesla Computing Cluster](https://docs.nvidia.com/gameworks/content/developertools/desktop/tesla_compute_cluster.htm)). NVIDIA's TCC driver does not support Vulkan, so you need to use the following command to set the driver model back to WDDM, to use Vulkan:

```bash
$ nvidia-smi -g 0 -dm 0
```

The number following `-g` is the GPU ID (which can be found to the left of your device name in `nvidia-smi` output); and `-dm` stands for driver model, 0 refers to WDDM and 1 means TCC.
