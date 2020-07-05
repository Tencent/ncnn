# Darknet To NCNN Conversion Tools

This is a standalone darknet2ncnn converter without additional dependency.

Support yolov4, yolov4-tiny, yolov3, yolov3-tiny and enet-coco.cfg (EfficientNetB0-Yolov3).

Another conversion tool based on darknet can be found at: [darknet2ncnn](https://github.com/xiangweizeng/darknet2ncnn)

## Usage

```
Usage: darknet2ncnn [darknetcfg] [darknetweights] [ncnnparam] [ncnnbin] [merge_output]
        [darknetcfg]     .cfg file of input darknet model.
        [darknetweights] .weights file of input darknet model.
        [cnnparam]       .param file of output ncnn model.
        [ncnnbin]        .bin file of output ncnn model.
        [merge_output]   merge all output yolo layers into one, enabled by default.
```

## Example

### 1. Convert yolov4-tiny cfg and weights

Download pre-trained [yolov4-tiny.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg) and [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights) or with your own trained weight.

Convert cfg and weights:
```
./darknet2ncnn yolov4-tiny.cfg yolov4-tiny.weights yolov4-tiny.param yolov4-tiny.bin 1
```

If succeeded, the output would be:
```
Loading cfg...
WARNING: The ignore_thresh=0.700000 of yolo0 is too high. An alternative value 0.25 is written instead.
WARNING: The ignore_thresh=0.700000 of yolo1 is too high. An alternative value 0.25 is written instead.
Loading weights...
Converting model...
83 layers, 91 blobs generated.
NOTE: The input of darknet uses: mean_vals=0 and norm_vals=1/255.f.
NOTE: Remeber to use ncnnoptimize for better performance.
```

### 2. Optimize graphic

```
./ncnnoptimize yolov4-tiny.param yolov4-tiny.bin yolov4-tiny-opt.param yolov4-tiny-opt.bin 0
```

### 3. Test

build examples/yolov4.cpp and test with:

```
./yolov4 dog.jpg
```

The result will be:

![](https://github.com/Tencent/ncnn/blob/master/tools/darknet/output.jpg)


## How to run with benchncnn

Set 2=0.3 for Yolov3DetectionOutput layer.

