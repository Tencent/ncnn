首先，非常感谢大家对 ncnn 组件的关注
为了方便大家使用 ncnn 组件，up主特意写了这篇使用指北，以烂大街的 alexnet 作为例子


### 准备caffe网络和模型

caffe 的网络和模型通常是搞深度学习的研究者训练出来的，一般来说训练完会有
```
train.prototxt
deploy.prototxt
snapshot_10000.caffemodel
```
部署的时候只需要 TEST 过程，所以有 deploy.prototxt 和 caffemodel 就足够了

alexnet 的 deploy.prototxt 可以在这里下载
https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet

alexnet 的 caffemodel 可以在这里下载
http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

### 转换ncnn网络和模型

caffe 自带了工具可以把老版本的 caffe 网络和模型转换为新版（ncnn的工具只认识新版
```
upgrade_net_proto_text [老prototxt] [新prototxt]
upgrade_net_proto_binary [老caffemodel] [新caffemodel]
```
输入层改用 Input，因为每次只需要做一个图片，所以第一个 dim 设为 1
```
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 3 dim: 227 dim: 227 } }
}
```
使用 caffe2ncnn 工具转换为 ncnn 的网络描述和模型
```
caffe2ncnn deploy.prototxt bvlc_alexnet.caffemodel alexnet.param alexnet.bin
```
### 去除可见字符串

有 param 和 bin 文件其实已经可以用了，但是 param 描述文件是明文的，如果放在 APP 分发出去容易被窥探到网络结构（说得好像不明文就看不到一样
使用 ncnn2mem 工具转换为二进制描述文件和内存模型，生成 alexnet.param.bin 和两个静态数组的代码文件
```
ncnn2mem alexnet.param alexnet.bin alexnet.id.h alexnet.mem.h
```
### 加载模型

直接加载 param 和 bin，适合快速验证效果使用
```cpp
ncnn::Net net;
net.load_param("alexnet.param");
net.load_model("alexnet.bin");
```
加载二进制的 param.bin 和 bin，没有可见字符串，适合 APP 分发模型资源
```cpp
ncnn::Net net;
net.load_param_bin("alexnet.param.bin");
net.load_model("alexnet.bin");
```
从内存引用加载网络和模型，没有可见字符串，模型数据全在代码里头，没有任何外部文件
另外，android apk 打包的资源文件读出来也是内存块
```cpp
#include "alexnet.mem.h"
ncnn::Net net;
net.load_param(alexnet_param_bin);
net.load_model(alexnet_bin);
```
以上三种都可以加载模型，其中内存引用方式加载是 zero-copy 的，所以使用 net 模型的来源内存块必须存在

### 卸载模型
```cpp
net.clear();
```

### 输入和输出

ncnn 用自己的数据结构 Mat 来存放输入和输出数据
输入图像的数据要转换为 Mat，依需要减去均值和乘系数
```cpp
#include "mat.h"
unsigned char* rgbdata;// data pointer to RGB image pixels
int w;// image width
int h;// image height
ncnn::Mat in = ncnn::Mat::from_pixels(rgbdata, ncnn::Mat::PIXEL_RGB, w, h);

const float mean_vals[3] = {104.f, 117.f, 123.f};
in.substract_mean_normalize(mean_vals, 0);
```
执行前向网络，获得计算结果
```cpp
#include "net.h"
ncnn::Mat in;// input blob as above
ncnn::Mat out;
ncnn::Extractor ex = net.create_extractor();
ex.set_light_mode(true);
ex.input("data", in);
ex.extract("prob", out);
```
如果是二进制的 param.bin 方式，没有可见字符串，利用 alexnet.id.h 的枚举来代替 blob 的名字
```cpp
#include "net.h"
#include "alexnet.id.h"
ncnn::Mat in;// input blob as above
ncnn::Mat out;
ncnn::Extractor ex = net.create_extractor();
ex.set_light_mode(true);
ex.input(alexnet_param_id::BLOB_data, in);
ex.extract(alexnet_param_id::BLOB_prob, out);
```
获取 Mat 中的输出数据，Mat 内部的数据通常是三维的，c / h / w，遍历所有获得全部分类的分数
```cpp
ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
std::vector<float> scores;
scores.resize(out_flatterned.w);
for (int j=0; j<out_flatterned.w; j++)
{
    scores[j] = out_flatterned[j];
}
```
### 某些使用技巧

Extractor 有个多线程加速的开关，设置线程数能加快计算
```cpp
ex.set_num_threads(4);
```
Mat 转换图像的时候可以顺便转换颜色和缩放大小，这些顺带的操作也是有优化的
支持 RGB2GRAY GRAY2RGB RGB2BGR 等常用转换，支持缩小和放大
```cpp
#include "mat.h"
unsigned char* rgbdata;// data pointer to RGB image pixels
int w;// image width
int h;// image height
int target_width = 227;// target resized width
int target_height = 227;// target resized height
ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgbdata, ncnn::Mat::PIXEL_RGB2GRAY, w, h, target_width, target_height);
```
Net 有从 FILE* 文件描述加载的接口，可以利用这点把多个网络和模型文件合并为一个，分发时能方便些，内存引用就无所谓了

> $ cat alexnet.param.bin alexnet.bin > alexnet-all.bin

```cpp
#include "net.h"
FILE* fp = fopen("alexnet-all.bin", "rb");
net.load_param_bin(fp);
net.load_model(fp);
fclose(fp);
```
