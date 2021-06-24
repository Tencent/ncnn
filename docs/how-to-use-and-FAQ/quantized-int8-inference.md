# Post Training Quantization Tools

To support int8 model deployment on mobile devices,we provide the universal post training quantization tools which can convert the float32 model to int8 model.

## User Guide

Example with mobilenet, just need three steps.

### 1. Optimize model

```shell
./ncnnoptimize mobilenet.param mobilenet.bin mobilenet-opt.param mobilenet-opt.bin 0
```

### 2. Create the calibration table file

We suggest that using the verification dataset for calibration, which is more than 5000 images.

Some imagenet sample images here https://github.com/nihui/imagenet-sample-images

```shell
find images/ -type f > imagelist.txt
./ncnn2table mobilenet-opt.param mobilenet-opt.bin imagelist.txt mobilenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
```

* mean and norm are the values you passed to ```Mat::substract_mean_normalize()```
* shape is the blob shape of your model, [w,h] or [w,h,c]

>
    * if w and h both are given, image will be resized to exactly size.
    * if w and h both are zero or negative, image will not be resized.
    * if only h is zero or negative, image's width will scaled resize to w, keeping aspect ratio.
    * if only w is zero or negative, image's height will scaled resize to h

* pixel is the pixel format of your model, image pixels will be converted to this type before ```Extractor::input()```
* thread is the CPU thread count that could be used for parallel inference
* method is the post training quantization algorithm, kl and aciq are currently supported

If your model has multiple input nodes, you can use multiple list files and other parameters

```shell
./ncnn2table mobilenet-opt.param mobilenet-opt.bin imagelist-bgr.txt,imagelist-depth.txt mobilenet.table mean=[104,117,123],[128] norm=[0.017,0.017,0.017],[0.0078125] shape=[224,224,3],[224,224,1] pixel=BGR,GRAY thread=8 method=kl
```

### 3. Quantize model

```shell
./ncnn2int8 mobilenet-opt.param mobilenet-opt.bin mobilenet-int8.param mobilenet-int8.bin mobilenet.table
```

## use ncnn int8 inference

the ncnn library would use int8 inference automatically, nothing changed in your code

```cpp
ncnn::Net mobilenet;
mobilenet.load_param("mobilenet-int8.param");
mobilenet.load_model("mobilenet-int8.bin");
```

## mixed precision inference

Before quantize your model, comment the layer weight scale line in table file, then the layer will do the float32 inference

```
conv1_param_0 156.639840536
```

```
#conv1_param_0 156.639840536
```
