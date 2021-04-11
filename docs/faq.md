

# 如何加入技术交流QQ群？

# 如何看作者b站直播？
nihui的bilibili直播间：https://live.bilibili.com/1264617
# 编译

## 怎样下载源码？

github
gitee
full-source.zip

## 怎么交叉编译？

cmake 工具链怎么设置啊？

## The submodules were not downloaded! Please update submodules with "git submodule update --init" and try again

## Could NOT find Protobuf (missing: Protobuf_INCLUDE_DIR)

sudo apt-get install libprotobuf-dev protobuf-compiler

## Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY)

https://github.com/Tencent/ncnn/issues/1873

## Could not find a package configuration file provided by "OpenCV" with any of the following names: OpenCVConfig.cmake opencv-config.cmake

set(OpenCV_DIR)

## Could not find a package configuration file provided by "ncnn" with any of the following names: ncnnConfig.cmake ncnn-config.cmake

set(ncnn_DIR)

## 找不到 Vulkan

如何安装 vulkan sdk
cmake版本 3.10，否则没有带 FindVulkan.cmake
android-api >= 24
macos 要先执行安装脚本

## undefined reference to __kmpc_for_static_init_4 __kmpc_for_static_fini __kmpc_fork_call ...

## undefined reference to vkEnumerateInstanceExtensionProperties vkGetInstanceProcAddr vkQueueSubmit ...

## undefined reference to glslang::InitializeProcess() glslang::TShader::TShader(EShLanguage) ...

## undefined reference to AAssetManager_fromJava AAssetManager_open AAsset_seek ...

find_package(ncnn)

## undefined reference to typeinfo for ncnn::Layer

opencv rtti -> opencv-mobile

## undefined reference to __cpu_model

升级编译器 / libgcc_s libgcc

## unrecognized command line option "-mavx2"

升级 gcc


## 为啥自己编译的ncnn android库特别大？


## rtti/exceptions冲突


# 怎样添加ncnn库到项目中？cmake方式怎么用？


## android

## ios

## linux

## windows

## macos

## arm linux


# 转模型问题

## caffe

## mxnet

## darknet

## pytorch - onnx

## tensorflow 1.x/2.x - keras

## tensorflow 2.x - mlir

## Shape not supported yet! Gather not supported yet! Cast not supported yet!

onnx-simplifier 静态shape

## convertmodel

## netron

## 怎么生成有固定 shape 信息的模型？

Input      0=w 1=h 2=c

why gpu能更快

## ncnnoptimize 怎么转成 fp16 模型

## ncnnoptimize 怎样查看模型的 FLOPS / 内存占用情况

## 怎么修改模型支持动态 shape？

Interp Reshape

## 如何将模型转换为代码内嵌到程序里？

ncnn2mem

## 如何加密模型？

https://zhuanlan.zhihu.com/p/268327784

## Linux下转的ncnn模型，Windows/MacOS/Android/.. 也能直接用吗？

Yes，全平台通用

## 如何去掉后处理，再导出 onnx？

检测

## pytorch 有的层导不出 onnx 怎么办？

ONNX_ATEN_FALLBACK
完全自定义的op，先改成能导出的（如 concat slice），转到 ncnn 后再修改 param

# 使用

## vkEnumeratePhysicalDevices failed -3

## vkCreateInstance failed -9

驱动
opengl

## ModuleNotFoundError: No module named 'ncnn.ncnn'

python setup.py develop

## fopen nanodet-m.param failed

文件路径
working dir

File not found or not readable. Make sure that XYZ.param/XYZ.bin is accessible.

## find_blob_index_by_name data / output / ... failed

layer name vs blob name
param.bin 应该用 xxx.id.h 的枚举

## parse magic failed

## param is too old, please regenerate

模型本身有问题

Your model file is being the old format converted by an old caffe2ncnn tool.

Checkout the latest ncnn code, build it and regenerate param and model binary files, and that should work.

Make sure that your param file starts with the magic number 7767517.

you may find more info on use-ncnn-with-alexnet

## set_vulkan_compute failed, network use_vulkan_compute disabled

你应该在 load_param / load_model 之前设置 net.opt.use_vulkan_compute = true;

## 多个blob输入，多个blob输出，怎么做？
多次执行`ex.input()` 和 `ex.extract()`
```
ex.input();
ex.input();
ex.extract();
ex.extract();
```
## Extractor extract 多次会重复计算吗？

## 如何看每一层的耗时？

cmake -DNCNN_BENCHMARK=ON ..

## 如何转换 cv::Mat CV_8UC3 BGR 图片

from_pixels to_pixels

## 如何转换 float 数据为 ncnn::Mat

https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx

## 如何初始化 ncnn::Mat 为全 0

`mat.fill(0.f);`

## 如何转换 yuv 数据

## 如何 resize crop rotate 图片

## 如何人脸5点对齐

get_affine_transform
warpaffine_bilinear_c3

## 如何获得中间层的blob输出

    ncnn::Mat output;
    ex.extract("your_blob_name", output);

## 为什么我使用GPU，但是GPU占用为0

windows 10 资源管理器显示 TAB 切换到 Compute_0 / Compute_1
GPU-Z

## layer XYZ not exists or registered

Your network contains some operations that are not implemented in ncnn.

You may implement them as custom layer followed in how-to-implement-custom-layer-step-by-step.

Or you could simply register them as no-op if you are sure those operations make no sense.

```
class Noop : public ncnn::Layer {};
DEFINE_LAYER_CREATOR(Noop)

net.register_custom_layer("LinearRegressionOutput", Noop_layer_creator);
net.register_custom_layer("MAERegressionOutput", Noop_layer_creator);
```

## network graph not ready

You shall call Net::load_param() first, then Net::load_model().

This error may also happens when Net::load_param() failed, but not properly handled.

For more information about the ncnn model load api, see ncnn-load-model

## memory not 32-bit aligned at XYZ

The pointer passed to Net::load_param() or Net::load_model() is not 32bit aligned.

In practice, the head pointer of std::vector is not guaranteed to be 32bit aligned.

you can store your binary buffer in ncnn::Mat structure, its internal memory is aligned.

## crash on android with '__kmp_abort_process'

This usually happens if you bundle multiple shared library with openmp linked

It is actually an issue of the android ndk https://github.com/android/ndk/issues/1028

On old android ndk, modify the link flags as

-Wl,-Bstatic -lomp -Wl,-Bdynamic

For recent ndk >= 21

-fstatic-openmp

## dlopen failed: library "libomp.so" not found
Newer android ndk defaults to dynamic openmp runtime

modify the link flags as

-fstatic-openmp -fopenmp

## crash when freeing a ncnn dynamic library(.dll/.so) built with openMP

for optimal performance, the openmp threadpool spin waits for about a second prior to shutting down in case more work becomes available.

If you unload a dynamic library that's in the process of spin-waiting, it will crash in the manner you see (most of the time).

Just set OMP_WAIT_POLICY=passive in your environment, before calling loadlibrary. or Just wait a few seconds before calling freelibrary.

You can also use the following method to set environment variables in your code:

for msvc++:

    SetEnvironmentVariable(_T("OMP_WAIT_POLICY"), _T("passive"));

for g++:

    setenv("OMP_WAIT_POLICY", "passive", 1)

    reference: https://stackoverflow.com/questions/34439956/vc-crash-when-freeing-a-dll-built-with-openmp

# 跑出来的结果对不上

## 如何打印 ncnn::Mat 的值？


## 总是输出第一张图的结果
复用 Extractor？！

## 启用fp16时的精度差异

net.opt.use_fp16_packed = false;
net.opt.use_fp16_storage = false;
net.opt.use_fp16_arithmetic = false;

https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result


# 如何跑得更快？内存占用更少？库体积更小？

## fp32 fp16

## 大小核绑定

## get_cpu_count get_gpu_count

## ncnnoptimize

## 如何使用量化工具？

## 如何设置线程数？

## 如何降低CPU占用率？

net.opt.openmp_blocktime = 0;
OMP_WAIT_POLICY=passive

## 如何 batch inference？

## partial graph inference

先 extract 分类，判断后，再 extract bbox

## 如何启用 bf16s 加速？

A53

## 如何裁剪更小的 ncnn 库？

https://github.com/Tencent/ncnn/wiki/build-minimal-library

## net.opt sgemm winograd fp16_storage 各是有什么作用？

对内存消耗的影响

# 白嫖项目

## nanodet

# 其他

## up主用的什么系统/编辑器/开发环境？

| 软件类型     |   软件名称  |
| ------------| ----------- |
| 系统        | Fedora       |
| 桌面环境     | KDE         |
| 编辑器       | Kate        |
| 画草图       | kolourpaint |
| 画函数图像   | kmplot      |
| bilibili直播 |  OBS         |