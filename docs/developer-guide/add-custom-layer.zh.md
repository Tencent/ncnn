这里举个例子添加 Relu6，即 std::min(6, std::max(0, val))

```
Input            input   0 1 input
Convolution      conv2d  1 1 input conv2d 0=32 1=1 2=1 3=1 4=0 5=0 6=768
Relu6            relu6   1 1 conv2d relu6
Pooling          maxpool 1 1 relu6 maxpool 0=0 1=3 2=2 3=-233 4=0
```

## method 1 -- 注册自定义层
```cpp
#include "layer.h"

class Relu6 : public ncnn::Layer
{
public:
    Relu6()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* outptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = std::min(6, std::max(0, outptr[i]));
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Relu6)
```

```cpp
ncnn::Net net;
net.register_custom_layer("Relu6", Relu6_layer_creator);

net.load_param("model.param");
net.load_model("model.bin");

ncnn::Extractor ex = net.create_extractor();

ex.input("input", inputmat);
ex.extract("maxpool", maxpoolmat);
```


## method 2 -- 处理中间 blob
```cpp
ncnn::Net net;
net.load_param("model.param");
net.load_model("model.bin");

ncnn::Extractor ex = net.create_extractor();

ex.input("input", inputmat);
ex.extract("conv2d", conv2dmat);

// relu6
ncnn::Mat relu6mat = conv2dmat.clone();
{
    int w = relu6mat.w;
    int h = relu6mat.h;
    int channels = relu6mat.c;
    int size = w * h;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* outptr = relu6mat.channel(q);
        for (int i=0; i<size; i++)
        {
            outptr[i] = std::min(6, std::max(0, outptr[i]));
        }
    }
}

ex.input("relu6", relu6mat);
ex.extract("maxpool", maxpoolmat);

```

## method 3 -- 直接修改 ncnn
实现 src/layer/relu6.h

实现 src/layer/relu6.cpp

修改 src/CMakeLists.txt
```cmake
ncnn_add_layer(UnaryOp)
ncnn_add_layer(ConvolutionDepthWise)
ncnn_add_layer(Padding)
ncnn_add_layer(Relu6)
```
