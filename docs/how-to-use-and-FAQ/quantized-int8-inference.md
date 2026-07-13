# Post Training Quantization Tools

To support int8 model deployment on mobile devices,we provide the universal post training quantization tools which can convert the float32 model to int8 model.

## User Guide

Example with mobilenet, just need three steps.

### 1. Optimize model

NOTE: **If your model is converted via pnnx, skip this step.**

```shell
./ncnnoptimize mobilenet.param mobilenet.bin mobilenet-opt.param mobilenet-opt.bin 0
```

### 2. Create the calibration table file

#### 2.1 From image

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

#### 2.2 From npy

We suggest that using the validation(development) set for calibration.

Use the same preprocessing as the training set to get the input vectors, in the case of batchsize=1, store each input vector as an npy file, n inputs correspond to n npy files, the actual stored vectors to remove the batch dimension.


test net, shape is in NCHW format, but there's no `N`.
```txt
in0, shape=[512]
in1, shape=[2, 1, 64]
in2, shape=[2, 1, 64]
```

filelist_in0.txt
```txt
0_in0.npy
1_in0.npy
2_in0.npy
...
```

filelist_in1.txt
```txt
0_in1.npy
1_in1.npy
2_in1.npy
...
```

filelist_in2.txt
```txt
0_in2.npy
1_in2.npy
2_in2.npy
...
```

```shell
./ncnn2table test.param test.bin filelist_in0.txt,filelist_in1.txt,filelist_in2.txt test.table shape=[512],[64,1,2],[64,1,2] thread=8 method=kl type=1
```
**Here shape is WHC, because the order of the arguments to `ncnn::Mat`.**

ncnn2table can generate static weight scales without a calibration dataset for RNN,GRU,LSTM,MultiHeadAttention and Embed layers

```shell
./ncnn2table rnn.param rnn.bin rnn.table method=kl
```

### 3. Quantize model

```shell
./ncnn2int8 mobilenet-opt.param mobilenet-opt.bin mobilenet-int8.param mobilenet-int8.bin mobilenet.table
```

## Weight-only block quantized Gemm and MultiHeadAttention

LLM-oriented `Gemm` and `MultiHeadAttention` weight-only block quantization is separate from the post training int8 flow above. It stores weight as signed int4/int6/int8 blocks and keeps activation/output in fp32.

The workflow is similar to `ncnn2table` and `ncnn2int8`:

```shell
./ncnnllm2table in.param in.bin model.llm.table method=minmax bits=6 block=64
./ncnnllm2int in.param in.bin out.param out.bin model.llm.table
```

method can be minmax,mseclip,awq,gptq. bits can be 4,6,8. block can be 32,64,128. thread is the CPU thread count.

awq and gptq need calibration data, same as npy calibration in ncnn2table.

```shell
./ncnnllm2table in.param in.bin calib.list awq.llm.table method=awq bits=4 block=64 type=1 shape=[...]
./ncnnllm2int in.param in.bin awq.param awq.bin awq.llm.table
```

```shell
./ncnnllm2table in.param in.bin calib.list gptq.llm.table method=gptq bits=4 block=128 type=1 shape=[...]
./ncnnllm2int in.param in.bin gptq.param gptq.bin gptq.llm.table
```

The calibration list format follows `ncnn2table`.

```text
gemm_name_param_1 bits=4 block=64 method=mseclip scale0 scale1 ...
mha_name_param_0  bits=4 block=64 method=mseclip scale0 scale1 ...
mha_name_param_1  bits=4 block=64 method=mseclip scale0 scale1 ...
mha_name_param_2  bits=4 block=64 method=mseclip scale0 scale1 ...
mha_name_param_3  bits=4 block=64 method=mseclip scale0 scale1 ...
```

For `MultiHeadAttention`, `_param_0/_param_1/_param_2/_param_3` are q/k/v/out weights.

```text
gemm_name_param_1_input_scale method=awq scale0 scale1 ...
mha_name_param_0_input_scale  method=awq scale0 scale1 ...
```

`method=gptq` uses fixed symmetric block scales and standard GPTQ error compensation. It writes packed qweight files and records them in the table.

```text
gemm_name_param_1 bits=4 block=128 method=gptq qweight=gemm.qweight scale0 scale1 ...
```

The table is text and may be edited before conversion. Missing `Gemm` rows are skipped. `MultiHeadAttention` q/k/v/out rows must exist together. Unused rows are rejected and at least one layer must be quantized.

The generated layer `quantize_term` is `bits * 100 + input_scale * 10 + block_code`, and block_code 0/1/2 means block 32/64/128.

For quick conversion without saving a table, `ncnnllm2int` can still compute scales directly:

```shell
./ncnnllm2int in.param in.bin out.param out.bin method=minmax bits=6 block=64
```

This format is signed symmetric scale-only. Zero point is not used.

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
