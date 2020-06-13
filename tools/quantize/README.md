# Post Training Quantization Tools

To support int8 model deployment on mobile devices,we provide the universal post training quantization tools which can convert the float32 model to int8 model.

The old convert tool and technical detail is in [Caffe-Int8-Convert-Tools](https://github.com/BUG1989/caffe-int8-convert-tools)

## User Guide

Example with mobilenet,just need three steps.

### 1. Optimization graphic 

```
./ncnnoptimize mobilenet-fp32.param mobilenet-fp32.bin mobilenet-nobn-fp32.param mobilenet-nobn-fp32.bin 0
```

### 2. Create the calibration table file

We suggest that using the verification dataset for calibration, which is more than 5000 images.

```
./ncnn2table --param=mobilenet-nobn-fp32.param --bin=mobilenet-nobn-fp32.bin --images=images/ --output=mobilenet-nobn.table --mean=104,117,123 --norm=0.017,0.017,0.017 --size=224,224 --thread=2
```

### 3. Quantization

```
./ncnn2int8 mobilenet-nobn-fp32.param mobilenet-nobn-fp32.bin mobilenet-int8.param mobilenet-int8.bin mobilenet-nobn.table
```

