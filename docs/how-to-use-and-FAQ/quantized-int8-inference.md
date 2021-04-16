# Post Training Quantization Tools

To support int8 model deployment on mobile devices,we provide the universal post training quantization tools which can convert the float32 model to int8 model.

## User Guide

Example with mobilenet, just need three steps.

### 1. Optimize model

```
./ncnnoptimize mobilenet.param mobilenet.bin mobilenet-opt.param mobilenet-opt.bin 0
```

### 2. Create the calibration table file

We suggest that using the verification dataset for calibration, which is more than 5000 images.

Some imagenet sample images here https://github.com/nihui/imagenet-sample-images

```
./ncnn2table --param=mobilenet-opt.param --bin=mobilenet-opt.bin --images=images/ --output=mobilenet.table --mean=104,117,123 --norm=0.017,0.017,0.017 --size=224,224
```

### 3. Quantize model

```
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
