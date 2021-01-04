# NCNN增加自定义层

## 举例

这里举个例子添加自定义层次 如Relu6，即 std::min(6, std::max(0, val))

```
Input            input   0 1 input
Convolution      conv2d  1 1 input conv2d 0=32 1=1 2=1 3=1 4=0 5=0 6=768
Relu6            relu6   1 1 conv2d relu6
Pooling          maxpool 1 1 relu6 maxpool 0=0 1=3 2=2 3=-233 4=0
```



## 定义源码h文件：src/layer/relu6.h

```CPP
#ifndef LAYER_RELU6_H
#define LAYER_RELU6_H

#include "layer.h"

namespace ncnn {

class Relu6 : public Layer
{
public:
    Relu6();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RELU6_H
```



## 定义源码CPP文件：src/layer/relu6.cpp

```CPP
#include "relu6.h"

#include <math.h>

namespace ncnn {

Relu6::Relu6()
{
    one_blob_only = true;
    support_inplace = true;
}

int Relu6::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = std::min(6, std::max(0, ptr[i]));
            }
        }

        return 0;
}

} // namespace ncnn

```



## 修改 src/CMakeLists.txt 注册Relu6

```CPP
ncnn_add_layer(GroupNorm)
ncnn_add_layer(LayerNorm)
ncnn_add_layer(Relu6)
```



## 定义测试用例CPP文件 src/test_relu6.cpp 

```CPP
#include "layer/relu6.h"
#include "testutil.h"

static int test_relu6(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Relu6>("Relu6", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_relu6 failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_relu6_0()
{
    return 0
           || test_relu6(RandomMat(5, 7, 24))
           || test_relu6(RandomMat(7, 9, 12))
           || test_relu6(RandomMat(3, 5, 13));
}

static int test_relu6_1()
{
    return 0
           || test_relu6(RandomMat(15, 24))
           || test_relu6(RandomMat(17, 12))
           || test_relu6(RandomMat(19, 15));
}

static int test_relu6_2()
{
    return 0
           || test_relu6(RandomMat(128))
           || test_relu6(RandomMat(124))
           || test_relu6(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_relu6_0()
           || test_relu6_1()
           || test_relu6_2();
}

```



## 修改tests/CMakeLists.txt 注册Relu6测试用例

```CPP
ncnn_add_layer_test(LSTM)
ncnn_add_layer_test(Yolov3DetectionOutput)
ncnn_add_layer_test(Relu6)
```



## 编译

```
按原NCNN步骤编译
```



## 单元测试

```
./test_relu6
```

