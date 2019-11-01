### input data and extract output
```
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "net.h"

int main()
{
    cv::Mat img = cv::imread("image.ppm", CV_LOAD_IMAGE_GRAYSCALE);
    int w = img.cols;
    int h = img.rows;

    // subtract 128, norm to -1 ~ 1
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_GRAY, w, h, 60, 60);
    float mean[1] = { 128.f };
    float norm[1] = { 1/128.f };
    in.substract_mean_normalize(mean, norm);

    ncnn::Net net;
    net.load_param("model.param");
    net.load_model("model.bin");

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat feat;
    ex.extract("output", feat);

    return 0;
}

```

### print Mat content
```
void pretty_print(const Mat& m)
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

### caffe-android-lib+openblas vs ncnn
use squeezenet v1.1, nexus6p, android 7.1.2

memory usage is the RSS item in top utility output

|compare item|caffe-android-lib+openblas|ncnn|
|---|---|---|
|inference time(1 thread)|228ms|88ms|
|inference time(8 thread)|152ms|38ms|
|memory usage|138.16M|21.56M|
|library binary size|6.9M|<500K|
|compability|armeabi-v7a-hard with neon or arm64-v8a|armeabi-v7a with neon or arm64-v8a|
|thirdparty dependency|boost gflags glog lmdb openblas opencv protobuf|none|

### FAQ
Q ncnn的起源

A 深度学习算法要在手机上落地，caffe依赖太多，手机上也没有cuda，需要个又快又小的前向网络实现


Q ncnn名字的来历

A cnn就是卷积神经网络的缩写，开头的n算是一语n关。比如new/next(全新的实现)，naive(ncnn是naive实现)，neon(ncnn最初为手机优化)，up主名字(←_←)


Q 支持哪些平台

A 跨平台，主要支持 android，次要支持 ios / linux / windows


Q 计算精度如何

A armv7 neon float 不遵照 ieee754 标准，有些采用快速实现(如exp sin等)，速度快但确保精度足够高


Q pc 上的速度很慢

A pc都是x86架构的，基本没做什么优化，主要用来核对结果，毕竟up主精力是有限的（


Q 为何没有 logo

A up主是mc玩家，所以开始是找了萌萌的苦力怕当看板娘的，但是这样子会侵权对吧，只好空出来了...
