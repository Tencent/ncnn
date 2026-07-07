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

LLM-oriented `Gemm` and `MultiHeadAttention` weight-only block quantization is separate from the post training int8 activation/weight inference flow above. It stores weights as signed symmetric int4/int6/int8 blocks with one fp32 scale per K block and output remains fp32.

The recommended workflow mirrors `ncnn2table` and `ncnn2int8`:

```shell
./ncnnllm2table in.param in.bin model.llm.table method=minmax bits=6 block=64
./ncnnllm2int468 in.param in.bin out.param out.bin model.llm.table
```

`ncnnllm2table` options follow the same trailing key-value style as `ncnn2table`:

| key | values | default | description |
| --- | ------ | ------- | ----------- |
| `method` | `minmax`, `mseclip` | `minmax` | weight quantization scale search method |
| `bits` | `4`, `6`, `8` | `6` | signed weight bit width |
| `block` | `32`, `64`, `128` | `64` | K block size |
| `thread` | positive integer | `1` | worker threads |

`method=minmax` uses per-block absmax scaling.

`method=mseclip` searches clipped absmax candidates and picks the scale with the smallest block weight reconstruction error. It is calibration-free and still exports the same symmetric scale-only runtime format. It is not AWQ and does not add activation rescaling metadata.

The llm table uses one line for each quantized weight:

```text
gemm_name_param_1 format=block_symmetric dtype=int4 block=64 scale_dtype=fp32 scale_encoding=quant method=mseclip scale0 scale1 ...
mha_name_param_0  format=block_symmetric dtype=int4 block=64 scale_dtype=fp32 scale_encoding=quant method=mseclip scale0 scale1 ...
mha_name_param_1  format=block_symmetric dtype=int4 block=64 scale_dtype=fp32 scale_encoding=quant method=mseclip scale0 scale1 ...
mha_name_param_2  format=block_symmetric dtype=int4 block=64 scale_dtype=fp32 scale_encoding=quant method=mseclip scale0 scale1 ...
mha_name_param_3  format=block_symmetric dtype=int4 block=64 scale_dtype=fp32 scale_encoding=quant method=mseclip scale0 scale1 ...
```

For `MultiHeadAttention`, `_param_0/_param_1/_param_2/_param_3` are q/k/v/out weights. All four rows must use the same dtype and block size.

`format`, `dtype`, `block`, `scale_dtype`, and `scale_encoding` are required for rows consumed by `ncnnllm2int468`. `dtype` and `block` are row-local, so one table can mix int4/int6/int8 and block32/block64/block128 for different layers. `ncnnllm2table` writes `method`, but `ncnnllm2int468` only logs it as offline provenance and does not require or validate it. Unused rows and unknown metadata are ignored with a warning, leaving room for future table extensions.

The first table format supports only `format=block_symmetric` with `dtype=int4/int6/int8`, `block=32/64/128`, `scale_dtype=fp32`, and `scale_encoding=quant`.

The runtime quantize term is stored in param id `18`.

```text
400/401/402  int4 block=32/64/128
600/601/602  int6 block=32/64/128
800/801/802  int8 block=32/64/128
```

Optional per-input-channel multipliers can be stored by adding separate `_input_scale` rows. They are applied to the current input channel inside the dot product:

```text
sum += (x[k] * input_scale[k]) * (qweight[n,k] / block_scale[n,k/block])
```

```text
gemm_name_param_1_input_scale format=input_scale scale_dtype=fp32 scale_encoding=mul method=awq coeff0 coeff1 ...
mha_name_param_0_input_scale  format=input_scale scale_dtype=fp32 scale_encoding=mul method=awq coeff0 coeff1 ...
mha_name_param_1_input_scale  format=input_scale scale_dtype=fp32 scale_encoding=mul method=awq coeff0 coeff1 ...
mha_name_param_2_input_scale  format=input_scale scale_dtype=fp32 scale_encoding=mul method=awq coeff0 coeff1 ...
mha_name_param_3_input_scale  format=input_scale scale_dtype=fp32 scale_encoding=mul method=awq coeff0 coeff1 ...
```

For `Gemm`, the input scale count is `constantK`. For `MultiHeadAttention`, q/k/v/out input scale counts are qdim/kdim/vdim/embed_dim, and either all four rows exist or none exist. `ncnnllm2table` does not write input scale rows by default.

Input-scale models use the same bits/block terms with tens digit `1`: `410/411/412`, `610/611/612`, and `810/811/812`.

For quick conversion without saving a table, `ncnnllm2int468` can still compute scales directly:

```shell
./ncnnllm2int468 in.param in.bin out.param out.bin method=minmax bits=6 block=64
```

The scale-only `format=block_symmetric` table does not claim exact GPTQ import support. Exact GPTQ qweight import needs a separate qweight payload design. AWQ and SmoothQuant must remain offline tooling or graph rewrite steps and must not become runtime `quantize_term` values. If offline tooling emits pre-scaled weights such as `W / s`, it may write `_input_scale` rows so runtime computes `x * s * dequant(W / s)`, which is equivalent to `x * dequant(W / s) * s`.

This format does not use zero point or asymmetric dequantization metadata.

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
