under construction ...

## caffe-int8-convert-tools
https://github.com/BUG1989/caffe-int8-convert-tools

## convert caffe model to ncnn quantized int8 model
### the offline way, reduce model binary size down to 25%

|sample model binary|size|
|---|---|
|squeezenet.bin|4.7M|
|squeezenet-int8.bin|1.2M|
|mobilenet_ssd_voc.bin|22.1M|
|mobilenet_ssd_voc-int8.bin|5.6M|

```
./caffe2ncnn resnet.prototxt resnet.caffemodel resnet-int8.param resnet-int8.bin 256 resnet.table
```
### the runtime way, no model binary reduction
```
./caffe2ncnn resnet.prototxt resnet.caffemodel resnet-fp32-int8.param resnet-fp32-int8.bin 0 resnet.table
```

## use ncnn int8 inference
the ncnn library would use int8 inference automatically, nothing changed in your code
```cpp
ncnn::Net resnet;
resnet.load_param("resnet-int8.param");
resnet.load_model("resnet-int8.bin");
```
### turn off int8 inference, the runtime model only
```cpp
ncnn::Net resnet;
resnet.use_int8_inference = 0;// set the switch before loading, force int8 inference off
resnet.load_param("resnet-fp32-int8.param");
resnet.load_model("resnet-fp32-int8.bin");
```

## mixed precision inference
before converting your model files, delete the layer weight scale line in table file, and that layer will do the float32 inference
```
conv1_param_0 156.639840536
```
```
```
