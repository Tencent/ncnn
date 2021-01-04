# step1 create a new empty class
```cpp
// mylayer.h
#include "layer.h"
using namespace ncnn;

// a new layer type called MyLayer
class MyLayer : public Layer
{
};

// mylayer.cpp
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer)
```

# step2 declare layer parameters and weights
```cpp
// mylayer.h
#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{
private:
    int channels;// new code
    float gamma;// new code
    Mat weight;// new code
};

// mylayer.cpp
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer)
```

# step3 implment load functions for parameters and weights
```cpp
// mylayer.h
#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{
public:
    virtual int load_param(const ParamDict& pd);// new code
    virtual int load_model(const ModelBin& mb);// new code

private:
    int channels;
    float eps;
    Mat gamma_data;
};

// mylayer.cpp
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer)

// new routine for loading parameters
int MyLayer::load_param(const ParamDict& pd)
{
    // details about the relations with param file
    // https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
    //
    channels = pd.get(0, 0);// parse 0=<int value> entry, default value 0
    eps = pd.get(1, 0.001f);// parse 1=<float value> entry, default value 0.001f

    return 0;// return zero if success
}

// new routine for loading weights
int MyLayer::load_model(const ModelBin& mb)
{
    // details about the relations with model file
    // https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
    //
    // read weights with length of channels * sizeof(float)
    // the second argument explains as follows
    // 0 judge the value type automatically, you may get float or float16 or uint8 etc
    //   depends on the model storage and the supporting target hardware
    // 1 read float values anyway
    // 2 read float16 values anyway
    // 3 read uint8 values anyway
    gamma_data = mb.load(channels, 1);
    if (gamma_data.empty())
        return -100;// return non-zero on error, -100 indicates out-of-memory

    return 0;// return zero if success
}
```

# step4 determine forward behavior
```cpp
// mylayer.h
#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{
public:
    MyLayer();// new code
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);

private:
    int channels;
    float eps;
    Mat gamma_data;
};

// mylayer.cpp
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer)

// new routine for setting forward behavior
MyLayer::MyLayer()
{
    // one input and one output
    // typical one_blob_only type: Convolution, Pooling, ReLU, Softmax ...
    // typical non-non_blob_only type: Eltwise, Split, Concat, Slice ...
    one_blob_only = true;

    // do not change the blob size, modify data in-place
    // typical support_inplace type: ReLU, Sigmoid ...
    // typical non-support_inplace type: Convolution, Pooling ...
    support_inplace = true;
}

int MyLayer::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.001f);

    // you could alter the behavior based on loaded parameter
    // if (eps == 0.001f)
    // {
    //     one_blob_only = false;
    //     support_inplace = false;
    // }

    return 0;
}

int MyLayer::load_model(const ModelBin& mb)
{
    gamma_data = mb.load(channels, 1);
    if (gamma_data.empty())
        return -100;

    // you could alter the behavior based on loaded weight
    // if (gamma_data[0] == 0.f)
    // {
    //     one_blob_only = false;
    //     support_inplace = false;
    // }

    return 0;
}
```

# step5 choose proper interface based on forward behavior
```cpp
// The base class Layer defines four interfaces for each forward behavior combination

// 1
virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

// 2
virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

// 3
virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;

// 4
virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
```
**must** = layer must implement this function

**optional** = layer may implement this function for optimal performance

sometimes the graph inference path cannot call forward_inplace directly due to data sharing, in this situation the non-inplace forward routine will be used, which deep-copy the input blob and call inplace forward on it if the optional routine is not implemented. Thus, you could avoid this deep-copy by process input to output on-the-fly.

|one_blob_only|support_inplace|1|2|3|4|
|---|---|---|---|---|---|
|false|false|must| | | |
|false|true|optional| |must| |
|true|false| |must| | |
|true|true| |optional| |must|

# step6 implement forward function
```cpp
// mylayer.h
#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{
public:
    MyLayer();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;// new code, optional
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;// new code

private:
    int channels;
    float eps;
    Mat gamma_data;
};

// mylayer.cpp
#include "mylayer.h"
DEFINE_LAYER_CREATOR(MyLayer)

MyLayer::MyLayer()
{
    one_blob_only = true;
    support_inplace = true;
}

int MyLayer::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.001f);

    return 0;
}

int MyLayer::load_model(const ModelBin& mb)
{
    gamma_data = mb.load(channels, 1);
    if (gamma_data.empty())
        return -100;

    return 0;
}

// optional new routine for layer forward function, non-inplace version
int MyLayer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // check input dims, return non-zero on error
    if (bottom_blob.c != channels)
        return -1;

    // x = (x + eps) * gamma_per_channel

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;// return non-zero on error, -100 indicates out-of-memory

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        const float gamma = gamma_data[q];

        for (int i=0; i<size; i++)
        {
            outptr[i] = (ptr[i] + eps) * gamma ;
        }
    }

    return 0;
}

// new routine for layer forward function
int MyLayer::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // check input dims, return non-zero on error
    if (bottom_top_blob.c != channels)
        return -1;

    // x = (x + eps) * gamma_per_channel

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        const float gamma = gamma_data[q];

        for (int i=0; i<size; i++)
        {
            ptr[i] = (ptr[i] + eps) * gamma ;
        }
    }

    return 0;
}
```

# step7 integret with ncnn library
you may probably need to modify caffe2ncnn or mxnet2ncnn etc. to write your layer specific parameters and weights into ncnn param and model file

the param and model file structure [param-and-model-file-structure](param-and-model-file-structure)

```
// example param file content
Input            input   0 1 input
Convolution      conv2d  1 1 input conv2d 0=32 1=1 2=1 3=1 4=0 5=0 6=768
MyLayer          mylayer 1 1 conv2d mylayer0
Pooling          maxpool 1 1 mylayer0 maxpool 0=0 1=3 2=2 3=-233 4=0
```

```cpp
ncnn::Net net;

// register custom layer before load param and model
// the layer creator function signature is always XYZ_layer_creator, which defined in DEFINE_LAYER_CREATOR macro
net.register_custom_layer("MyLayer", MyLayer_layer_creator);

net.load_param("model.param");
net.load_model("model.bin");
```
