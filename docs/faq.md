

# 如何加入技术交流QQ群？

# 如何看作者b站直播？
   
- nihui的bilibili直播间：[水竹院落](https://live.bilibili.com/1264617)

# 编译

- ## 怎样下载完整源码？

   git clone --recursive https://github.com/Tencent/ncnn/
   
   或者
   
   下载 [ncnn-xxxxx-full-source.zip](https://github.com/Tencent/ncnn/releases)

- ## 怎么交叉编译？cmake 工具链怎么设置啊？
   
   参见 https://github.com/Tencent/ncnn/wiki/how-to-build

- ## The submodules were not downloaded! Please update submodules with "git submodule update --init" and try again

   如上，下载完整源码。或者按提示执行: git submodule update --init

- ## Could NOT find Protobuf (missing: Protobuf_INCLUDE_DIR)
   
   sudo apt-get install libprotobuf-dev protobuf-compiler

- ## Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY)

   https://github.com/Tencent/ncnn/issues/1873

- ## Could not find a package configuration file provided by "OpenCV" with any of the following names: OpenCVConfig.cmake opencv-config.cmake

   sudo apt-get install libopencv-dev

   或者自行编译安装，set(OpenCV_DIR {OpenCVConfig.cmake所在目录})

- ## Could not find a package configuration file provided by "ncnn" with any of the following names: ncnnConfig.cmake ncnn-config.cmake

   set(ncnn_DIR {ncnnConfig.cmake所在目录})

- ## 找不到 Vulkan, 

   cmake版本 3.10，否则没有带 FindVulkan.cmake

   android-api >= 24

   macos 要先执行安装脚本

- ## 如何安装 vulkan sdk

- ## 找不到库（需要根据系统/编译器指定）

   undefined reference to __kmpc_for_static_init_4 __kmpc_for_static_fini __kmpc_fork_call ...

   需要链接openmp库 

   undefined reference to vkEnumerateInstanceExtensionProperties vkGetInstanceProcAddr vkQueueSubmit ...

   需要 vulkan-1.lib

   undefined reference to glslang::InitializeProcess() glslang::TShader::TShader(EShLanguage) ...

   需要 glslang.lib OGLCompiler.lib SPIRV.lib OSDependent.lib

   undefined reference to AAssetManager_fromJava AAssetManager_open AAsset_seek ...

   find_library和target_like_libraries中增加 android 

   find_package(ncnn)

- ## undefined reference to typeinfo for ncnn::Layer

   opencv rtti -> opencv-mobile

- ## undefined reference to __cpu_model

   升级编译器 / libgcc_s libgcc

- ## unrecognized command line option "-mavx2"

   升级 gcc

- ## 为啥自己编译的ncnn android库特别大？

   https://github.com/Tencent/ncnn/wiki/build-for-android.zh 以及见 如何裁剪更小的 ncnn 库

- ## ncnnoptimize和自定义层

   先ncnnoptimize再增加自定义层，避免ncnnoptimize不能处理自定义层保存。


- ## rtti/exceptions冲突

   产生原因是项目工程中使用的库配置不一样导致冲突，根据自己的实际情况分析是需要开启还是关闭。ncnn默认是ON，在重新编译ncnn时增加以下2个参数即可：
   - 开启：-DNCNN_DISABLE_RTTI=OFF -DNCNN_DISABLE_EXCEPTION=OFF
   - 关闭：-DNCNN_DISABLE_RTTI=ON -DNCNN_DISABLE_EXCEPTION=ON


- ## error: undefined symbol: ncnn::Extractor::extract(char const*, ncnn::Mat&)

   可能的情况：
   - 尝试升级 Android Studio 的 NDK 版本


# 怎样添加ncnn库到项目中？cmake方式怎么用？

编译ncnn，make install。linux/windows set/export ncnn_DIR 指向 isntall目录下下包含ncnnConfig.cmake 的目录

- ## android

- ## ios

- ## linux

- ## windows

- ## macos

- ## arm linux


# 转模型问题

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

   [通过MLIR将tensorflow2模型转换到ncnn](https://zhuanlan.zhihu.com/p/152535430) **@[nihui](https://www.zhihu.com/people/nihui-2)**

- ## Shape not supported yet! Gather not supported yet! Cast not supported yet!

   onnx-simplifier 静态shape

- ## convertmodel

   [https://convertmodel.com/](https://convertmodel.com/) **[@大老师](https://github.com/daquexian)**

- ## netron

   [https://github.com/lutzroeder/netron](https://github.com/lutzroeder/netron)

- ## 怎么生成有固定 shape 信息的模型？

   Input      0=w 1=h 2=c

- ## why gpu能更快

- ## ncnnoptimize 怎么转成 fp16 模型

   `ncnnoptimize model.param model.bin yolov5s-opt.param yolov5s-opt.bin 65536`

- ## ncnnoptimize 怎样查看模型的 FLOPS / 内存占用情况

- ## 怎么修改模型支持动态 shape？

   Interp Reshape

- ## 如何将模型转换为代码内嵌到程序里？

   ncnn2mem

- ## 如何加密模型？

   https://zhuanlan.zhihu.com/p/268327784

- ## Linux下转的ncnn模型，Windows/MacOS/Android/.. 也能直接用吗？

   Yes，全平台通用

- ## 如何去掉后处理，再导出 onnx？

   检测：

   参考up的一篇文章(https://zhuanlan.zhihu.com/p/128974102)，步骤三就是去掉后处理,再导出onnx,其中去掉后处理可以是项目内测试时去掉后续步骤的结果。

- ## pytorch 有的层导不出 onnx 怎么办？

   ONNX_ATEN_FALLBACK
完全自定义的op，先改成能导出的（如 concat slice），转到 ncnn 后再修改 param

# 使用

- ## vkEnumeratePhysicalDevices failed -3

- ## vkCreateInstance failed -9

   驱动

- ## ModuleNotFoundError: No module named 'ncnn.ncnn'

   python setup.py develop

- ## fopen nanodet-m.param failed

   文件路径 working dir

   File not found or not readable. Make sure that XYZ.param/XYZ.bin is accessible.

- ## find_blob_index_by_name data / output / ... failed

   layer name vs blob name
   
   param.bin 应该用 xxx.id.h 的枚举

- ## parse magic failed

- ## param is too old, please regenerate

   模型本身有问题

   Your model file is being the old format converted by an old caffe2ncnn tool.

   Checkout the latest ncnn code, build it and regenerate param and model binary files, and that should work.

   Make sure that your param file starts with the magic number 7767517.

   you may find more info on use-ncnn-with-alexnet

- ## set_vulkan_compute failed, network use_vulkan_compute disabled

   你应该在 load_param / load_model 之前设置 net.opt.use_vulkan_compute = true;

- ## 多个blob输入，多个blob输出，怎么做？
   多次执行`ex.input()` 和 `ex.extract()`
```
ex.input("data1", in);
ex.input("data2", in);
ex.extract("output1", out);
ex.extract("output2", out);
```
- ## Extractor extract 多次会重复计算吗？

   不会

- ## 如何看每一层的耗时？

   cmake -DNCNN_BENCHMARK=ON ..

- ## 如何转换 cv::Mat CV_8UC3 BGR 图片

   from_pixels to_pixels

- ## 如何转换 float 数据为 ncnn::Mat

   首先，自己申请的内存需要自己管理，此时ncnn::Mat不会自动给你释放你传过来的float数据
   ``` c++
   std::vector<float> testData(60, 1.0);                                // 利用std::vector<float>自己管理内存的申请和释放
   ncnn::Mat in1(60, (void*)testData.data()).reshape(4, 5, 3);          // 把float数据的指针转成void*传过去即可，甚至还可以指定维度(up说最好使用reshape用来解决channel gap)
   float* a = new float[60];                                            // 自己new一块内存，后续需要自己释放
   ncnn::Mat in2 = ncnn::Mat(60, (void*)a).reshape(4, 5, 3).clone();    // 使用方法和上面相同，clone() to transfer data owner
   ```

- ## 如何初始化 ncnn::Mat 为全 0

   `mat.fill(0.f);`

- ## 如何查看／获取版本号

   cmake时会打印

   c_api.h ncnn_version()

   自己拼 1.0+yyyymmdd

- ## 如何转换 yuv 数据

   yuv420sp2rgb yuv420sp2rgb_nv12

   **[@zz大佬](https://github.com/zchrissirhcz/xxYUV)**

- ## 如何 resize crop rotate 图片

   [efficient roi resize rotate](https://github.com/Tencent/ncnn/wiki/efficient-roi-resize-rotate)

- ## 如何人脸5点对齐

   get_affine_transform

   warpaffine_bilinear_c3

```c
// 计算变换矩阵 并且求逆变换
int type = 0;       // 0->区域外填充为v[0],v[1],v[2], -233->区域外不处理
unsigned int v = 0;
float tm[6];
float tm_inv[6];
// 人脸区域在原图上的坐标和宽高
float src_x = target->det.rect.x / target->det.w * pIveImageU8C3->u32Width;
float src_y = target->det.rect.y / target->det.h * pIveImageU8C3->u32Height;
float src_w = target->det.rect.w / target->det.w * pIveImageU8C3->u32Width;
float src_h = target->det.rect.h / target->det.h * pIveImageU8C3->u32Height;
float point_src[10] = {
src_x + src_w * target->attr.land[0][0], src_x + src_w * target->attr.land[0][1],
src_x + src_w * target->attr.land[1][0], src_x + src_w * target->attr.land[1][1],
src_x + src_w * target->attr.land[2][0], src_x + src_w * target->attr.land[2][1],
src_x + src_w * target->attr.land[3][0], src_x + src_w * target->attr.land[3][1],
src_x + src_w * target->attr.land[4][0], src_x + src_w * target->attr.land[4][1],
};
float point_dst[10] = { // +8 是因为我们处理112*112的图
30.2946f + 8.0f, 51.6963f,
65.5318f + 8.0f, 51.5014f,
48.0252f + 8.0f, 71.7366f,
33.5493f + 8.0f, 92.3655f,
62.7299f + 8.0f, 92.2041f,
};
// 第一种方式：先计算变换在求逆
AffineTrans::get_affine_transform(point_src, point_dst, 5, tm);
AffineTrans::invert_affine_transform(tm, tm_inv);
// 第二种方式：直接拿到求逆的结果
// AffineTrans::get_affine_transform(point_dst, point_src, 5, tm_inv);
// rgb 分离的，所以要单独处理
for(int c = 0; c < 3; c++)
{
    unsigned char* pSrc = malloc(xxx);
    unsigned char* pDst = malloc(xxx);
    ncnn::warpaffine_bilinear_c1(pSrc, SrcWidth, SrcHeight, SrcStride[c], pDst, DstWidth, DstHeight, DstStride[c], tm_inv, type, v);
}
// rgb packed则可以一次处理
ncnn::warpaffine_bilinear_c3(pSrc, SrcWidth, SrcHeight, SrcStride, pDst, DstWidth, DstHeight, DstStride, tm_inv, type, v);
```

- ## 如何获得中间层的blob输出
   
   ncnn::Mat output;
   
   ex.extract("your_blob_name", output);

- ## 为什么我使用GPU，但是GPU占用为0

   windows 10 任务管理器 - 性能选项卡 - GPU - 选择其中一个视图左上角的下拉箭头切换到 Compute_0 / Compute_1 / Cuda

   你还可以安装软件：GPU-Z 

- ## layer XYZ not exists or registered

   Your network contains some operations that are not implemented in ncnn.

   You may implement them as custom layer followed in how-to-implement-custom-layer-step-by-step.

   Or you could simply register them as no-op if you are sure those operations make no sense.

```
class Noop : public ncnn::Layer {};
DEFINE_LAYER_CREATOR(Noop)

net.register_custom_layer("LinearRegressionOutput", Noop_layer_creator);
net.register_custom_layer("MAERegressionOutput", Noop_layer_creator);
```

- ## network graph not ready

   You shall call Net::load_param() first, then Net::load_model().

   This error may also happens when Net::load_param() failed, but not properly handled.

   For more information about the ncnn model load api, see ncnn-load-model

- ## memory not 32-bit aligned at XYZ

   The pointer passed to Net::load_param() or Net::load_model() is not 32bit aligned.

   In practice, the head pointer of std::vector is not guaranteed to be 32bit aligned.

   you can store your binary buffer in ncnn::Mat structure, its internal memory is aligned.

- ## crash on android with '__kmp_abort_process'

   This usually happens if you bundle multiple shared library with openmp linked

   It is actually an issue of the android ndk https://github.com/android/ndk/issues/1028

   On old android ndk, modify the link flags as

   -Wl,-Bstatic -lomp -Wl,-Bdynamic

   For recent ndk >= 21

   -fstatic-openmp

- ## dlopen failed: library "libomp.so" not found
   Newer android ndk defaults to dynamic openmp runtime

   modify the link flags as

   -fstatic-openmp -fopenmp

- ## crash when freeing a ncnn dynamic library(.dll/.so) built with openMP

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

[ncnn-produce-wrong-result](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result)

- ## 如何打印 ncnn::Mat 的值？

```C++
void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}
```
In Android Studio, `printf` will not work, you can use `__android_log_print` instead. Example :
```C++
#include <android/log.h>  // Don't forget this

void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                __android_log_print(ANDROID_LOG_DEBUG,"LOG_TAG","ncnn Mat is : %f", m.channel(q).row(y)[x]);
            }
        }
    }
}
```

- ## 如何可视化 ncnn::Mat 的值？

```
void visualize(const char* title, const ncnn::Mat& m)
{
    std::vector<cv::Mat> normed_feats(m.c);

    for (int i=0; i<m.c; i++)
    {
        cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y=0; y<m.h; y++)
        {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x=0; x<m.w; x++)
            {
                float v = tp[x];
                if (v != v)
                {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }

                sp += 3;
            }
        }
    }

    int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
    int th = (m.c - 1) / tw + 1;

    cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
    show_map = cv::Scalar(127);

    // tile
    for (int i=0; i<m.c; i++)
    {
        int ty = i / tw;
        int tx = i % tw;

        normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
    }

    cv::resize(show_map, show_map, cv::Size(0,0), 2, 2, cv::INTER_NEAREST);
    cv::imshow(title, show_map);
}
```

- ## 总是输出第一张图的结果

   复用 Extractor？！

- ## 启用fp16时的精度有差异

   net.opt.use_fp16_packed = false;

   net.opt.use_fp16_storage = false;

   net.opt.use_fp16_arithmetic = false;

   [ncnn-produce-wrong-result](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result)


# 如何跑得更快？内存占用更少？库体积更小？

- ## fp32 fp16

- ## 大小核绑定
   ncnn::set_cpu_powersave(int)绑定大核或小核
   注意windows系统不支持绑核。
   ncnn支持不同的模型运行在不同的核心。假设硬件平台有2个大核，4个小核，你想把netA运行在大核，netB运行在小核。
   可以通过std::thread or pthread创建两个线程，运行如下代码：
   0:全部
   1:小核
   2:大核
```
   void thread_1()
   {
      ncnn::set_cpu_powersave(2); // bind to big cores
      netA.opt.num_threads = 2;
   }

   void thread_2()
   {
      ncnn::set_cpu_powersave(1); // bind to little cores
      netB.opt.num_threads = 4;
   }
```

   [openmp-best-practice.zh.md](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/openmp-best-practice.zh.md)

- ## 查看 CPU 或 GPU 数量
   get_cpu_count
   
   get_gpu_count

- ## ncnnoptimize

   使用方式一：
    - ./ncnnoptimize ncnn.param ncnn.bin new.param new.bin flag
    <br/>注意这里的flag指的是fp32和fp16，其中0指的是fp32，1指的是fp16

   使用方式二：
    - ./ncnnoptimize ncnn.param ncnn.bin new.param new.bin flag cutstartname cutendname
    <br/>cutstartname：模型截取的起点
    <br/>cutendname：模型截取的终点


- ## 如何使用量化工具？

   [Post Training Quantization Tools](https://github.com/Tencent/ncnn/tree/master/tools/quantize)

- ## 如何设置线程数？

   opt.num_threads

- ## 如何降低CPU占用率？

   net.opt.openmp_blocktime = 0;
   
   OMP_WAIT_POLICY=passive

- ## 如何 batch inference？

- ## partial graph inference

   先 extract 分类，判断后，再 extract bbox

- ## 如何启用 bf16s 加速？

```
net.opt.use_packing_layout = true;
net.opt.use_bf16_storage = true;
```

   [用bf16加速ncnn](https://zhuanlan.zhihu.com/p/112564372) **@[nihui](https://www.zhihu.com/people/nihui-2)**

   A53

- ## 如何裁剪更小的 ncnn 库？

   [build-minimal-library](https://github.com/Tencent/ncnn/wiki/build-minimal-library)

- ## net.opt sgemm winograd fp16_storage 各是有什么作用？

   对内存消耗的影响

# 白嫖项目

- ## nanodet

# 其他

- ## up主用的什么系统/编辑器/开发环境？

   | 软件类型     |   软件名称  |
   | ------------| ----------- |
   | 系统        | Fedora       |
   | 桌面环境     | KDE         |
   | 编辑器       | Kate        |
   | 画草图       | kolourpaint |
   | 画函数图像   | kmplot      |
   | bilibili直播 |  OBS         |
