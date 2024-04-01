

# How to join the technical Community Groups with QQ  ？

- Open QQ -> click the group chat search-> search group number 637093648, enter the answer to the question: conv conv conv conv conv → join the group chat → ready to accept the Turing test(a joke)
- Open QQ -> search Pocky group: 677104663 (lots experts), the answer to the question

# How to watch the author's on live in Bilibili？

- nihui：[水竹院落](https://live.bilibili.com/1264617)

# Compilation

- ## How to download the full source code？

   git clone --recursive https://github.com/Tencent/ncnn/
   
   or
   
   download [ncnn-xxxxx-full-source.zip](https://github.com/Tencent/ncnn/releases)

- ## How to cross-compile？How to set the cmake toolchain？
  
   See https://github.com/Tencent/ncnn/wiki/how-to-build

- ## The submodules were not downloaded! Please update submodules with "git submodule update --init" and try again

   As above, download the full source code. Or follow the prompts to execute: git submodule update --init

- ## Could NOT find Protobuf (missing: Protobuf_INCLUDE_DIR)
  
   sudo apt-get install libprotobuf-dev protobuf-compiler

- ## Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY)

   https://github.com/Tencent/ncnn/issues/1873

- ## Could not find a package configuration file provided by "OpenCV" with any of the following names: OpenCVConfig.cmake opencv-config.cmake

   sudo apt-get install libopencv-dev

   or customized compile and install ，with set(OpenCV_DIR {the dir OpenCVConfig.cmake exist})

- ## Could not find a package configuration file provided by "ncnn" with any of the following names: ncnnConfig.cmake ncnn-config.cmake

   set(ncnn_DIR { the dir ncnnConfig.cmake exist})

- ## Vulkan not found, 

   - cmake requires version >= 3.10, otherwise there is no FindVulkan.cmake

   - android-api >= 24

   - macos has to run the install script first

- ## How to install vulkan sdk

    - See https://www.vulkan.org/tools#download-these-essential-development-tools
    - But There was a frequent problem that the project need glslang lib in ncnn not official vulkan

- ## xxx.lib not found（be specified by system/compiler）

   undefined reference to __kmpc_for_static_init_4 __kmpc_for_static_fini __kmpc_fork_call ...

   Need to link openmp

   undefined reference to vkEnumerateInstanceExtensionProperties vkGetInstanceProcAddr vkQueueSubmit ...

   need vulkan-1.lib

   undefined reference to glslang::InitializeProcess() glslang::TShader::TShader(EShLanguage) ...

   need glslang.lib OGLCompiler.lib SPIRV.lib OSDependent.lib

   undefined reference to AAssetManager_fromJava AAssetManager_open AAsset_seek ...

   Add android to find_library and target_like_libraries 

   find_package(ncnn)

- ## undefined reference to typeinfo for ncnn::Layer

   opencv rtti -> opencv-mobile

- ## undefined reference to __cpu_model

   upgrade compiler / libgcc_s libgcc

- ## unrecognized command line option "-mavx2"

   upgrade gcc

- ## Why is the compiled ncnn-android library so large？

   See https://github.com/Tencent/ncnn/wiki/build-for-android.zh and see How to trim smaller ncnn

- ## ncnnoptimize and custom layer

   ncnnoptimize first before adding a custom layer to avoid ncnnoptimize not being able to handle custom layer saves.


- ## rtti/exceptions Conflict

   The reason for the conflict is that the libraries used in the project are configured differently, so analyze whether you need to turn them on or off according to your actual situation. ncnn is ON by default, add the following two parameters when recompiling ncnn.
   - ON: -DNCNN_DISABLE_RTTI=OFF -DNCNN_DISABLE_EXCEPTION=OFF
   - OFF: -DNCNN_DISABLE_RTTI=ON -DNCNN_DISABLE_EXCEPTION=ON


- ## error: undefined symbol: ncnn::Extractor::extract(char const*, ncnn::Mat&)

   Possible scenarios.
   - Try upgrading the NDK version of Android Studio


# How do I add the ncnn library to my project and how does the cmake method work?

Compile ncnn,and make install. linux/windows should set/export ncnn_DIR points to the directory containing ncnnConfig.cmake under the install directory

- ## android

- ## ios

- ## linux

- ## windows

- ## macos

- ## arm linux


# Convert model issues

- ## caffe

   `./caffe2ncnn caffe.prototxt caffe.caffemodel ncnn.param ncnn.bin`

- ## mxnet

   ` ./mxnet2ncnn mxnet-symbol.json mxnet.params ncnn.param ncnn.bin`

- ## darknet

   [https://github.com/xiangweizeng/darknet2ncnn](https://github.com/xiangweizeng/darknet2ncnn)

- ## pytorch - onnx

   [use ncnn with pytorch or onnx](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx)

- ## tensorflow 1.x/2.x - keras

   [https://github.com/MarsTechHAN/keras2ncnn](https://github.com/MarsTechHAN/keras2ncnn) **[@MarsTechHAN](https://github.com/MarsTechHAN)**

- ## tensorflow 2.x - mlir

   [Converting tensorflow2 models to ncnn via MLIR](https://zhuanlan.zhihu.com/p/152535430) **@[nihui](https://www.zhihu.com/people/nihui-2)**

- ## Shape not supported yet! Gather not supported yet! Cast not supported yet!

   onnx-simplifier shape

- ## convertmodel

   [https://convertmodel.com/](https://convertmodel.com/) **[@大老师](https://github.com/daquexian)**

- ## netron

   [https://github.com/lutzroeder/netron](https://github.com/lutzroeder/netron)

- ## How to generate a model with fixed shape？

   Input      0=w 1=h 2=c

- ## why gpu can speedup

- ## How to convert ncnnoptimize to fp16 model

   `ncnnoptimize model.param model.bin yolov5s-opt.param yolov5s-opt.bin 65536`

- ## How to use ncnnoptimize  checking the FLOPS / memory usage of your model

- ## How to modify the model to support dynamics shape？

   Interp Reshape

- ## How to convert a model into code embedded in a program？

   use ncnn2mem

- ## How to encrypt the model？

   See https://zhuanlan.zhihu.com/p/268327784

- ## The ncnn model transferred under Linux, Windows/MacOS/Android/... Can I use it directly?

   Yes, for all platforms

- ## How to remove post-processing and export onnx？

   Ref：

   Referring to an article by UP <https://zhuanlan.zhihu.com/p/128974102>, step 3 is to remove the post-processing and then export the onnx, where removing the post-processing can be the result of removing the subsequent steps when testing within the project.

- ## pytorch layers can't export to onnx？

 Mode 1:

   ONNX_ATEN_FALLBACK
Fully customizable op, first change to one that can export (e.g. concat slice), go to ncnn and then modify param

 Way 2.

 You can try this with PNNX, see the following article for a general description:

   1. [Windows/Linux/macOS steps for compiling PNNX](https://zhuanlan.zhihu.com/p/431833958)

   2. [Learn in 5 minutes! Converting TorchScript models to ncnn models with PNNX](https://zhuanlan.zhihu.com/p/427512763)

# Using

- ## vkEnumeratePhysicalDevices failed -3

- ## vkCreateInstance failed -9

   Please upgrade your GPU driver if you meet this crash or error.
   Here are the download sites for some brands of GPU drivers. We have provided some driver download pages here.
   [Intel](https://downloadcenter.intel.com/product/80939/Graphics-Drivers), [AMD](https://www.amd.com/en/support), [Nvidia](https://) www.nvidia.com/Download/index.aspx)

- ## ModuleNotFoundError: No module named 'ncnn.ncnn'

   python setup.py develop

- ## fopen nanodet-m.param failed

   path should be working dir

   File not found or not readable. Make sure that XYZ.param/XYZ.bin is accessible.

- ## find_blob_index_by_name data / output / ... failed

   layer name vs blob name
   
   param.bin use xxx.id.h enum

- ## parse magic failed

- ## param is too old, please regenerate

   The model maybe has problems

   Your model file is being the old format converted by an old caffe2ncnn tool.

   Checkout the latest ncnn code, build it and regenerate param and model binary files, and that should work.

   Make sure that your param file starts with the magic number 7767517.

   you may find more info on use-ncnn-with-alexnet
   
   When adding the softmax layer yourself, you need to add 1=1

- ## set_vulkan_compute failed, network use_vulkan_compute disabled

   Set net.opt.use_vulkan_compute = true before load_param / load_model;

- ## How to ececute multiple blob inputs, multiple blob outputs？
   Multiple execute `ex.input()` and `ex.extract()` like following
    ```
    ex.input("data1", in_1);
    ex.input("data2", in_2);
    ex.extract("output1", out_1);
    ex.extract("output2", out_2);
    ```
- ## Multiple executions of Extractor extract double the calculation？

   No

- ## How to see the elapsed time for every layer？

   cmake -DNCNN_BENCHMARK=ON ..

- ## How to convert a cv::Mat CV_8UC3 BGR image

   from_pixels to_pixels

- ## How to convert float data to ncnn::Mat

   First of all, you need to manage the memory you request yourself, at this point ncnn::Mat will not automatically free up the float data you pass over to it
   ``` c++
   std::vector<float> testData(60, 1.0); // use std::vector<float> to manage memory requests and releases yourself
   ncnn::Mat in1 = ncnn::Mat(60, (void*)testData.data()).reshape(4, 5, 3); // just pass the pointer to the float data as a void*, and even specify the dimension (up says it's best to use reshape to solve the channel gap)
   float* a = new float[60]; // New a piece of memory yourself, you need to release it later
   ncnn::Mat in2 = ncnn::Mat(60, (void*)a).reshape(4, 5, 3).clone(); // use the same method as above, clone() to transfer data owner
   ```
