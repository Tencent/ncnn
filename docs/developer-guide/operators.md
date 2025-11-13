
* [AbsVal](#absval)
* [ArgMax](#argmax)
* [BatchNorm](#batchnorm)
* [Bias](#bias)
* [BinaryOp](#binaryop)
* [BNLL](#bnll)
* [Cast](#cast)
* [CELU](#celu)
* [Clip](#clip)
* [Concat](#concat)
* [Convolution](#convolution)
* [Convolution1D](#convolution1d)
* [Convolution3D](#convolution3d)
* [ConvolutionDepthWise](#convolutiondepthwise)
* [ConvolutionDepthWise1D](#convolutiondepthwise1d)
* [ConvolutionDepthWise3D](#convolutiondepthwise3d)
* [CopyTo](#copyto)
* [Crop](#crop)
* [CumulativeSum](#cumulativesum)
* [Deconvolution](#deconvolution)
* [Deconvolution1D](#deconvolution1d)
* [Deconvolution3D](#deconvolution3d)
* [DeconvolutionDepthWise](#deconvolutiondepthwise)
* [DeconvolutionDepthWise1D](#deconvolutiondepthwise1d)
* [DeconvolutionDepthWise3D](#deconvolutiondepthwise3d)
* [DeformableConv2D](#deformableconv2d)
* [Dequantize](#dequantize)
* [Diag](#diag)
* [Dropout](#dropout)
* [Eltwise](#eltwise)
* [ELU](#elu)
* [Embed](#embed)
* [Exp](#exp)
* [ExpandDims](#expanddims)
* [Flatten](#flatten)
* [Flip](#flip)
* [Fold](#fold)
* [GELU](#gelu)
* [GLU](#glu)
* [Gemm](#gemm)
* [GridSample](#gridsample)
* [GroupNorm](#groupnorm)
* [GRU](#gru)
* [HardSigmoid](#hardsigmoid)
* [HardSwish](#hardswish)
* [InnerProduct](#innerproduct)
* [Input](#input)
* [InstanceNorm](#instancenorm)
* [Interp](#interp)
* [InverseSpectrogram](#inversespectrogram)
* [LayerNorm](#layernorm)
* [Log](#log)
* [LRN](#lrn)
* [LSTM](#lstm)
* [MemoryData](#memorydata)
* [Mish](#mish)
* [MultiHeadAttention](#multiheadattention)
* [MVN](#mvn)
* [Noop](#noop)
* [Normalize](#normalize)
* [Packing](#packing)
* [Padding](#padding)
* [Permute](#permute)
* [PixelShuffle](#pixelshuffle)
* [Pooling](#pooling)
* [Pooling1D](#pooling1d)
* [Pooling3D](#pooling3d)
* [Power](#power)
* [PReLU](#prelu)
* [Quantize](#quantize)
* [Reduction](#reduction)
* [ReLU](#relu)
* [Reorg](#reorg)
* [Requantize](#requantize)
* [Reshape](#reshape)
* [RMSNorm](#rmsnorm)
* [RNN](#rnn)
* [Scale](#scale)
* [SDPA](#sdpa)
* [SELU](#selu)
* [Shrink](#shrink)
* [ShuffleChannel](#shufflechannel)
* [Sigmoid](#sigmoid)
* [Slice](#slice)
* [Softmax](#softmax)
* [Softplus](#softplus)
* [Spectrogram](#spectrogram)
* [Split](#split)
* [Squeeze](#squeeze)
* [Swish](#swish)
* [TanH](#tanh)
* [Threshold](#threshold)
* [Tile](#tile)
* [UnaryOp](#unaryop)
* [Unfold](#unfold)

# AbsVal
```
y = abs(x)
```

* one_blob_only
* support_inplace

# ArgMax
```
y = argmax(x, out_max_val, topk)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | out_max_val   | int   | 0         |                   |
| 1         | topk          | int   | 1         |                   |

# BatchNorm
```
y = (x - mean) / sqrt(var + eps) * slope + bias
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | channels      | int   | 0         |                   |
| 1         | eps           | float | 0.f       |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| slope_data    | float | [channels]            |
| mean_data     | float | [channels]            |
| var_data      | float | [channels]            |
| bias_data     | float | [channels]            |

# Bias
```
y = x + bias
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | bias_data_size| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| bias_data     | float | [channels]            |

# BinaryOp
 This operation is used for binary computation, and the calculation rule depends on the [broadcasting rule](https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting).
```
C = binaryop(A, B)
```
if with_scalar = 1:
- one_blob_only
- support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | op_type       | int   | 0         | Operation type as follows |
| 1         | with_scalar   | int   | 0         | with_scalar=0 B is a matrix, with_scalar=1 B is a scalar |
| 2         | b             | float | 0.f       | When B is a scalar, B = b |

Operation type:
- 0 = ADD
- 1 = SUB
- 2 = MUL
- 3 = DIV
- 4 = MAX
- 5 = MIN
- 6 = POW
- 7 = RSUB
- 8 = RDIV
- 9 = RPOW
- 10 = ATAN2
- 11 = RATAN2

# BNLL
```
y = log(1 + e^(-x)) , x > 0
y = log(1 + e^x),     x < 0
```

* one_blob_only
* support_inplace

# Cast
```
y = cast(x)
```

* one_blob_only
* support_packing

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | type_from     | int   | 0         |                   |
| 1         | type_to       | int   | 0         |                   |

Element type:
- 0 = auto
- 1 = float32
- 2 = float16
- 3 = int8
- 4 = bfloat16

# CELU
```
if x < 0    y = (exp(x / alpha) - 1.f) * alpha
else        y = x
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | alpha         | float | 1.f       |                   |

# Clip
```
y = clamp(x, min, max)
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | min           | float | -FLT_MAX  |                   |
| 1         | max           | float | FLT_MAX   |                   |

# Concat
```
y = concat(x0, x1, x2, ...) by axis
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | axis          | int   | 0         |                   |

# Convolution
```
x2 = pad(x, pads, pad_value)
x3 = conv(x2, weight, kernel, stride, dilation) + bias
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 8         | int8_scale_term| int  | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 18        | pad_value     | float | 0.f       |                   |
| 19        | dynamic_weight| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16/int8 | [kernel_w, kernel_h, num_input, num_output] |
| bias_data     | float | [num_output]          |
| weight_data_int8_scales| float | [num_output] |
| bottom_blob_int8_scales| float | [1]          |
| top_blob_int8_scales| float | [1]             |

# Convolution1D
```
x2 = pad(x, pads, pad_value)
x3 = conv1d(x2, weight, kernel, stride, dilation) + bias
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 18        | pad_value     | float | 0.f       |                   |
| 19        | dynamic_weight| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16/int8 | [kernel_w, num_input, num_output] |
| bias_data     | float | [num_output]          |

# Convolution3D
```
x2 = pad(x, pads, pad_value)
x3 = conv3d(x2, weight, kernel, stride, dilation) + bias
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 17        | pad_behind    | int   | pad_front |                   |
| 18        | pad_value     | float | 0.f       |                   |
| 21        | kernel_d      | int   | kernel_w  |                   |
| 22        | dilation_d    | int   | dilation_w |                  |
| 23        | stride_d      | int   | stride_w  |                   |
| 24        | pad_front     | int   | pad_left  |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16/int8 | [kernel_w, kernel_h, kernel_d, num_input, num_output] |
| bias_data     | float | [num_output]          |

# ConvolutionDepthWise
```
x2 = pad(x, pads, pad_value)
x3 = conv(x2, weight, kernel, stride, dilation, group) + bias
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 7         | group         | int   | 1         |                   |
| 8         | int8_scale_term| int  | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 18        | pad_value     | float | 0.f       |                   |
| 19        | dynamic_weight| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16/int8 | [kernel_w, kernel_h, num_input / group, num_output / group, group] |
| bias_data     | float | [num_output]          |
| weight_data_int8_scales| float | [group]      |
| bottom_blob_int8_scales| float | [1]          |
| top_blob_int8_scales| float | [1]             |

# ConvolutionDepthWise1D
```
x2 = pad(x, pads, pad_value)
x3 = conv1d(x2, weight, kernel, stride, dilation, group) + bias
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 7         | group         | int   | 1         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 18        | pad_value     | float | 0.f       |                   |
| 19        | dynamic_weight| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16/int8 | [kernel_w, num_input / group, num_output / group, group] |
| bias_data     | float | [num_output]          |

# ConvolutionDepthWise3D
```
x2 = pad(x, pads, pad_value)
x3 = conv3d(x2, weight, kernel, stride, dilation, group) + bias
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 7         | group         | int   | 1         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 17        | pad_behind    | int   | pad_front |                   |
| 18        | pad_value     | float | 0.f       |                   |
| 21        | kernel_d      | int   | kernel_w  |                   |
| 22        | dilation_d    | int   | dilation_w |                  |
| 23        | stride_d      | int   | stride_w  |                   |
| 24        | pad_front     | int   | pad_left  |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16/int8 | [kernel_w, kernel_h, kernel_d, num_input / group, num_output / group, group] |
| bias_data     | float | [num_output]          |

# CopyTo
```
self[offset] = src
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | woffset       | int   | 0         |                   |
| 1         | hoffset       | int   | 0         |                   |
| 13        | doffset       | int   | 0         |                   |
| 2         | coffset       | int   | 0         |                   |
| 9         | starts        | array | [ ]       |                   |
| 11        | axes          | array | [ ]       |                   |

# Crop
```
y = crop(x)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | woffset       | int   | 0         |                   |
| 1         | hoffset       | int   | 0         |                   |
| 13        | doffset       | int   | 0         |                   |
| 2         | coffset       | int   | 0         |                   |
| 3         | outw          | int   | 0         |                   |
| 4         | outh          | int   | 0         |                   |
| 14        | outd          | int   | 0         |                   |
| 5         | outc          | int   | 0         |                   |
| 6         | woffset2      | int   | 0         |                   |
| 7         | hoffset2      | int   | 0         |                   |
| 15        | doffset2      | int   | 0         |                   |
| 8         | coffset2      | int   | 0         |                   |
| 9         | starts        | array | [ ]       |                   |
| 10        | ends          | array | [ ]       |                   |
| 11        | axes          | array | [ ]       |                   |
| 19        | starts_expr   | str   | ""        |                   |
| 20        | ends_expr     | str   | ""        |                   |
| 21        | axes_expr     | str   | ""        |                   |

# CumulativeSum

If axis < 0, we use axis = x.dims + axis

It implements https://pytorch.org/docs/stable/generated/torch.cumsum.html

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | axis          | int   | 0         |                   |


# Deconvolution
```
x2 = deconv(x, weight, kernel, stride, dilation) + bias
x3 = depad(x2, pads, pad_value)
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 18        | output_pad_right| int | 0         |                   |
| 19        | output_pad_bottom| int | output_pad_right |           |
| 20        | output_w      | int   | 0         |                   |
| 21        | output_h      | int   | output_w  |                   |
| 28        | dynamic_weight| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16 | [kernel_w, kernel_h, num_input, num_output] |
| bias_data     | float | [num_output]          |

# Deconvolution1D
```
x2 = deconv1d(x, weight, kernel, stride, dilation) + bias
x3 = depad(x2, pads, pad_value)
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 18        | output_pad_right| int | 0         |                   |
| 20        | output_w      | int   | 0         |                   |
| 28        | dynamic_weight| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16 | [kernel_w, num_input, num_output] |
| bias_data     | float | [num_output]          |

# Deconvolution3D
```
x2 = deconv3d(x, weight, kernel, stride, dilation) + bias
x3 = depad(x2, pads, pad_value)
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 17        | pad_behind    | int   | pad_front |                   |
| 18        | output_pad_right| int | 0         |                   |
| 19        | output_pad_bottom| int | output_pad_right |           |
| 20        | output_pad_behind| int | output_pad_right |           |
| 21        | kernel_d      | int   | kernel_w  |                   |
| 22        | dilation_d    | int   | dilation_w |                  |
| 23        | stride_d      | int   | stride_w  |                   |
| 24        | pad_front     | int   | pad_left  |                   |
| 25        | output_w      | int   | 0         |                   |
| 26        | output_h      | int   | output_w  |                   |
| 27        | output_d      | int   | output_w  |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16 | [kernel_w, kernel_h, kernel_d, num_input, num_output] |
| bias_data     | float | [num_output]          |

# DeconvolutionDepthWise
```
x2 = deconv(x, weight, kernel, stride, dilation, group) + bias
x3 = depad(x2, pads, pad_value)
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 7         | group         | int   | 1         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 18        | output_pad_right| int | 0         |                   |
| 19        | output_pad_bottom| int | output_pad_right |           |
| 20        | output_w      | int   | 0         |                   |
| 21        | output_h      | int   | output_w  |                   |
| 28        | dynamic_weight| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16 | [kernel_w, kernel_h, num_input / group, num_output / group, group] |
| bias_data     | float | [num_output]          |

# DeconvolutionDepthWise1D
```
x2 = deconv1d(x, weight, kernel, stride, dilation, group) + bias
x3 = depad(x2, pads, pad_value)
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 7         | group         | int   | 1         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 18        | output_pad_right| int | 0         |                   |
| 20        | output_w      | int   | 0         |                   |
| 28        | dynamic_weight| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16 | [kernel_w, num_input / group, num_output / group, group] |
| bias_data     | float | [num_output]          |

# DeconvolutionDepthWise3D
```
x2 = deconv3d(x, weight, kernel, stride, dilation, group) + bias
x3 = depad(x2, pads, pad_value)
y = activation(x3, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 7         | group         | int   | 1         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 17        | pad_behind    | int   | pad_front |                   |
| 18        | output_pad_right| int | 0         |                   |
| 19        | output_pad_bottom| int | output_pad_right |           |
| 20        | output_pad_behind| int | output_pad_right |           |
| 21        | kernel_d      | int   | kernel_w  |                   |
| 22        | dilation_d    | int   | dilation_w |                  |
| 23        | stride_d      | int   | stride_w  |                   |
| 24        | pad_front     | int   | pad_left  |                   |
| 25        | output_w      | int   | 0         |                   |
| 26        | output_h      | int   | output_w  |                   |
| 27        | output_d      | int   | output_w  |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16 | [kernel_w, kernel_h, kernel_d, num_input / group, num_output / group, group] |
| bias_data     | float | [num_output]          |

# DeformableConv2D
```
x2 = deformableconv2d(x, offset, mask, weight, kernel, stride, dilation) + bias
y = activation(x2, act_type, act_params)
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 5         | bias_term     | int   | 0         |                   |
| 6         | weight_data_size| int | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16/int8 | [kernel_w, kernel_h, num_input, num_output] |
| bias_data     | float | [num_output]          |

# Dequantize
```
y = x * scale + bias
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | scale_data_size| int  | 1         |                   |
| 1         | bias_data_size| int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| scale_data    | float | [scale_data_size]     |
| bias_data     | float | [bias_data_size]      |

# Diag
```
y = diag(x, diagonal)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | diagonal      | int   | 0         |                   |

# Dropout
```
y = x * scale
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | scale         | float | 1.f       |                   |

# Eltwise
```
y = elementwise_op(x0, x1, ...)
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | op_type       | int   | 0         |                   |
| 1         | coeffs        | array | [ ]       |                   |

Operation type:
- 0 = PROD
- 1 = SUM
- 2 = MAX

# ELU
```
if x < 0    y = (exp(x) - 1) * alpha
else        y = x
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | alpha         | float | 0.1f      |                   |

# Embed
```
y = embedding(x)
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | input_dim     | int   | 0         |                   |
| 2         | bias_term     | int   | 0         |                   |
| 3         | weight_data_size | int | 0        |                   |
| 18        | int8_scale_term| int  | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float | [weight_data_size]    |
| bias_term     | float | [num_output]          |
| weight_data_int8_scales| float | [1]          |

# Exp
```
if base == -1   y = exp(shift + x * scale)
else            y = pow(base, (shift + x * scale))
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | base          | float | -1.f      |                   |
| 1         | scale         | float | 1.f       |                   |
| 2         | shift         | float | 0.f       |                   |

# ExpandDims

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 3         | axes          | array | [ ]       |                   |

# Flatten
Reshape blob to 1 dimension

* one_blob_only

# Flip

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | axes          | array | [ ]       |                   |

# Fold
```
y = fold(x)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
| 20        | output_w      | int   | 0         |                   |
| 21        | output_h      | int   | output_w  |                   |

# GELU
```
if fast_gelu == 1   y = 0.5 * x * (1 + tanh(0.79788452 * (x + 0.044715 * x * x * x)));
else                y = 0.5 * x * erfc(-0.70710678 * x)
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | fast_gelu     | int   | 0         | use approximation |

# GLU

If axis < 0, we use axis = x.dims + axis

GLU(a,b)=a⊗σ(b)

where a is the first half of the input matrix and b is the second half.

axis specifies the dimension to split the input

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | axis          | int   | 0         |                   |

# Gemm
```
a = transA ? transpose(x0) : x0
b = transb ? transpose(x1) : x1
c = x2
y = (gemm(a, b) + c * beta) * alpha
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | alpha         | float | 1.f       |                   |
| 1         | beta          | float | 1.f       |                   |
| 2         | transA        | int   | 0         |                   |
| 3         | transb        | int   | 0         |                   |
| 4         | constantA     | int   | 0         |                   |
| 5         | constantB     | int   | 0         |                   |
| 6         | constantC     | int   | 0         |                   |
| 7         | constantM     | int   | 0         |                   |
| 8         | constantN     | int   | 0         |                   |
| 9         | constantK     | int   | 0         |                   |
| 10        | constant_broadcast_type_C | int | 0 |                 |
| 11        | output_N1M    | int   | 0         |                   |
| 12        | output_elempack | int | 0         |                   |
| 13        | output_elemtype | int | 0         |                   |
| 14        | output_transpose | int| 0         |                   |
| 18        | int8_scale_term | int | 0         |                   |
| 20        | constant_TILE_M | int | 0         |                   |
| 21        | constant_TILE_N | int | 0         |                   |
| 22        | constant_TILE_K | int | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| A_data        | float/fp16/int8 | [M, K] or [K, M] |
| B_data        | float/fp16/int8 | [N, K] or [K, N] |
| C_data        | float | [1], [M] or [N] or [1, M] or [N,1] or [N, M] |
| A_data_int8_scales| float | [M]               |
| B_data_int8_scales| float | [1]               |

# GridSample
```
Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.

For each output location output[:, h2, w2], the size-2 vector grid[h2, w2, 2] specifies input pixel[:, h1, w1] locations x and y, 
which are used to interpolate the output value output[:, h2, w2]

This function is often used in conjunction with affine_grid() to build Spatial Transformer Networks .
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | sample_type   | int   | 1         |                   |
| 1         | padding_mode  | int   | 1         |                   |
| 2         | align_corner  | int   | 0         |                   |
| 3         | permute_fusion| int   | 0         | fuse with permute |


Sample type:
- 1 = Nearest
- 2 = Bilinear
- 3 = Bicubic

Padding mode:
- 1 = zeros
- 2 = border
- 3 = reflection


# GroupNorm
```
split x along channel axis into group x0, x1 ...
l2 normalize for each group x0, x1 ...
y = x * gamma + beta
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | group         | int   | 1         |                   |
| 1         | channels      | int   | 0         |                   |
| 2         | eps           | float | 0.001f    | x = x / sqrt(var + eps) |
| 3         | affine        | int   | 1         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| gamma_data    | float | [channels]            |
| beta_data     | float | [channels]            |

# GRU
Apply a single-layer GRU to a feature sequence of `T` timesteps. The input blob shape is `[w=input_size, h=T]` and the output blob shape is `[w=num_output, h=T]`.

```
y = gru(x)
y0, hidden y1 = gru(x0, hidden x1)
```

* one_blob_only if bidirectional

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         | hidden size of output |
| 1         | weight_data_size| int | 0         | total size of weight matrix |
| 2         | direction     | int   | 0         | 0=forward, 1=reverse, 2=bidirectional |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_xc_data| float/fp16/int8 | [input_size, num_output * 3, num_directions] |
| bias_c_data   | float/fp16/int8 | [num_output, 4, num_directions] |
| weight_hc_data| float/fp16/int8 | [num_output, num_output * 3, num_directions] |

Direction flag:
- 0 = forward only
- 1 = reverse only
- 2 = bidirectional

# HardSigmoid
```
y = clamp(x * alpha + beta, 0, 1)
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | alpha         | float | 0.2f      |                   |
| 1         | beta          | float | 0.5f      |                   |

# HardSwish
```
y = x * clamp(x * alpha + beta, 0, 1)
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | alpha         | float | 0.2f      |                   |
| 1         | beta          | float | 0.5f      |                   |

# InnerProduct
```
x2 = innerproduct(x, weight) + bias
y = activation(x2, act_type, act_params)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | bias_term     | int   | 0         |                   |
| 2         | weight_data_size| int | 0         |                   |
| 8         | int8_scale_term| int  | 0         |                   |
| 9         | activation_type| int  | 0         |                   |
| 10        | activation_params| array | [ ]    |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_data   | float/fp16/int8 | [num_input, num_output] |
| bias_data     | float | [num_output]          |
| weight_data_int8_scales| float | [num_output] |
| bottom_blob_int8_scales| float | [1]          |

# Input
```
y = input
```

* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | w             | int   | 0         |                   |
| 1         | h             | int   | 0         |                   |
| 11        | d             | int   | 0         |                   |
| 2         | c             | int   | 0         |                   |

# InstanceNorm
```
split x along channel axis into instance x0, x1 ...
l2 normalize for each channel instance x0, x1 ...
y = x * gamma + beta
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | channels      | int   | 0         |                   |
| 1         | eps           | float | 0.001f    | x = x / sqrt(var + eps) |
| 2         | affine        | int   | 1         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| gamma_data    | float | [channels]            |
| beta_data     | float | [channels]            |

# Interp
```
if dynamic_target_size == 0     y = resize(x) by fixed size or scale
else                            y = resize(x0, size(x1))
```

* one_blob_only if dynamic_target_size == 0

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | resize_type   | int   | 0         |                   |
| 1         | height_scale  | float | 1.f       |                   |
| 2         | width_scale   | float | 1.f       |                   |
| 3         | output_height | int   | 0         |                   |
| 4         | output_width  | int   | 0         |                   |
| 5         | dynamic_target_size| int | 0      |                   |
| 6         | align_corner  | int   | 0         |                   |
| 9         | size_expr     | str   | ""        |                   |

Resize type:
- 1 = Nearest
- 2 = Bilinear
- 3 = Bicubic

# InverseSpectrogram
```
x1 = x as complex
x1 = x1 * sqrt(norm) if normalized
y = istft(x1)
y1 = unpad(y) if center

if returns == 0 return y1 as complex
if returns == 1 return y1 real
if returns == 2 return y1 imag
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | n_fft         | int   | 0         |                   |
| 1         | returns       | int   | 1         |                   |
| 2         | hoplen        | int   | n_fft / 4 |                   |
| 3         | winlen        | int   | n_fft     |                   |
| 4         | window_type   | int   | 0         | 0=ones 1=hann 2=hamming |
| 5         | center        | int   | 1         |                   |
| 7         | normalized    | int   | 0         | 0=no 1=n_fft 2=window-l2-energy |

# LayerNorm
```
split x along outmost axis into part x0, x1 ...
l2 normalize for each part x0, x1 ...
y = x * gamma + beta by elementwise
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | affine_size   | int   | 0         |                   |
| 1         | eps           | float | 0.001f    | x = x / sqrt(var + eps) |
| 2         | affine        | int   | 1         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| gamma_data    | float | [affine_size]         |
| beta_data     | float | [affine_size]         |

# Log
```
if base == -1   y = log(shift + x * scale)
else            y = log(shift + x * scale) / log(base)
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | base          | float | -1.f      |                   |
| 1         | scale         | float | 1.f       |                   |
| 2         | shift         | float | 0.f       |                   |

# LRN
```
if region_type == ACROSS_CHANNELS   square_sum = sum of channel window of local_size
if region_type == WITHIN_CHANNEL    square_sum = sum of spatial window of local_size
y = x * pow(bias + alpha * square_sum / (local_size * local_size), -beta)
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | region_type   | int   | 0         |                   |
| 1         | local_size    | int   | 5         |                   |
| 2         | alpha         | float | 1.f       |                   |
| 3         | beta          | float | 0.75f     |                   |
| 4         | bias          | float | 1.f       |                   |

Region type:
- 0 = ACROSS_CHANNELS
- 1 = WITHIN_CHANNEL

# LSTM
Apply a single-layer LSTM to a feature sequence of `T` timesteps. The input blob shape is `[w=input_size, h=T]` and the output blob shape is `[w=num_output, h=T]`.

```
y = lstm(x)
y0, hidden y1, cell y2 = lstm(x0, hidden x1, cell x2)
```

* one_blob_only if bidirectional

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         | output size of output |
| 1         | weight_data_size| int | 0         | total size of IFOG weight matrix |
| 2         | direction     | int   | 0         | 0=forward, 1=reverse, 2=bidirectional |
| 3         | hidden_size   | int   | num_output| hidden size       |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_xc_data| float/fp16/int8 | [input_size, hidden_size * 4, num_directions] |
| bias_c_data   | float/fp16/int8 | [hidden_size, 4, num_directions] |
| weight_hc_data| float/fp16/int8 | [num_output, hidden_size * 4, num_directions] |
| weight_hr_data| float/fp16/int8 | [hidden_size, num_output, num_directions] |

Direction flag:
- 0 = forward only
- 1 = reverse only
- 2 = bidirectional

# MemoryData
```
y = data
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | w             | int   | 0         |                   |
| 1         | h             | int   | 0         |                   |
| 11        | d             | int   | 0         |                   |
| 2         | c             | int   | 0         |                   |
| 21        | load_type     | int   | 1         | 1=fp32            |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| data          | float | [w, h, d, c]          |

# Mish
```
y = x * tanh(log(exp(x) + 1))
```

* one_blob_only
* support_inplace

# MultiHeadAttention
```
q_affine = affine(q) / (embed_dim / num_head)
k_affine = affine(k) or reuse kv_cache part
v_affine = affine(v) or reuse kv_cache part
split q k v into num_head part q0, k0, v0, q1, k1, v1 ...
for each num_head part
    qk = q * k
    qk = qk + attn_mask if attn_mask exists
    softmax(qk)
    qkv = qk * v
    merge qkv to out
y = affine(out)
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | embed_dim     | int   | 0         |                   |
| 1         | num_heads     | int   | 1         |                   |
| 2         | weight_data_size| int | 0         | qdim = weight_data_size / embed_dim |
| 3         | kdim          | int   | embed_dim |                   |
| 4         | vdim          | int   | embed_dim |                   |
| 5         | attn_mask     | int   | 0         |                   |
| 6         | scale         | float | 1.f / sqrt(embed_dim / num_heads) | |
| 7         | kv_cache      | int   | 0         |                   |
| 18        | int8_scale_term | int | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| q_weight_data | float/fp16/int8 | [embed_dim * qdim] |
| q_bias_data   | float | [embed_dim]           |
| k_weight_data | float/fp16/int8 | [embed_dim * kdim] |
| k_bias_data   | float | [embed_dim]           |
| v_weight_data | float/fp16/int8 | [embed_dim * vdim] |
| v_bias_data   | float | [embed_dim]           |
| out_weight_data| float/fp16/int8 | [qdim * embed_dim] |
| out_bias_data | float | [qdim]                |
| q_weight_data_int8_scales| float | [embed_dim] |
| k_weight_data_int8_scales| float | [embed_dim] |
| v_weight_data_int8_scales| float | [embed_dim] |
| out_weight_data_int8_scales| float | [1]      |

# MVN
```
if normalize_variance == 1 && across_channels == 1      y = (x - mean) / (sqrt(var) + eps) of whole blob
if normalize_variance == 1 && across_channels == 0      y = (x - mean) / (sqrt(var) + eps) of each channel
if normalize_variance == 0 && across_channels == 1      y = x - mean of whole blob
if normalize_variance == 0 && across_channels == 0      y = x - mean of each channel
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | normalize_variance| int | 0       |                   |
| 1         | across_channels| int  | 0         |                   |
| 2         | eps           | float | 0.0001f   | x = x / (sqrt(var) + eps) |

# Noop
```
y = x
```

# Normalize
```
if across_spatial == 1 && across_channel == 1      x2 = normalize(x) of whole blob
if across_spatial == 1 && across_channel == 0      x2 = normalize(x) of each channel
if across_spatial == 0 && across_channel == 1      x2 = normalize(x) of each position
y = x2 * scale
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | across_spatial| int   | 0         |                   |
| 1         | channel_shared| int   | 0         |                   |
| 2         | eps           | float | 0.0001f   | see eps mode      |
| 3         | scale_data_size| int  | 0         |                   |
| 4         | across_channel| int   | 0         |                   |
| 9         | eps_mode      | int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| scale_data    | float | [scale_data_size]     |

Eps Mode:
- 0 = caffe/mxnet   x = x / sqrt(var + eps)
- 1 = pytorch       x = x / max(sqrt(var), eps)
- 2 = tensorflow    x = x / sqrt(max(var, eps))

# Packing
```
y = wrap_packing(x)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | out_elempack  | int   | 1         |                   |
| 1         | use_padding   | int   | 0         |                   |
| 2         | cast_type_from| int   | 0         |                   |
| 3         | cast_type_to  | int   | 0         |                   |
| 4         | storage_type_from| int | 0        |                   |
| 5         | storage_type_to| int  | 0         |                   |

# Padding
```
y = pad(x, pads)
```

| param id  | name          | type | default   | description       |
| --------- | ------------- | ---- | --------- | ----------------- |
| 0         | top           | int  | 0         |                   |
| 1         | bottom        | int  | 0         |                   |
| 2         | left          | int  | 0         |                   |
| 3         | right         | int  | 0         |                   |
| 4         | type          | int  | 0         |                   |
| 5         | value         | float | 0         |                   |
| 6         | per_channel_pad_data_size| int | 0 |                 |
| 7         | front         | int  | stride_w  |                   |
| 8         | behind        | int  | pad_left  |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| per_channel_pad_data| float | [per_channel_pad_data_size] |

Padding type:
- 0 = CONSTANT
- 1 = REPLICATE
- 2 = REFLECT

# Permute
```
y = reorder(x)
```

| param id  | name          | type | default   | description       |
| --------- | ------------- | ---- | --------- | ----------------- |
| 0         | order_type    | int  | 0         |                   |

Order Type:
- 0 = WH WHC WHDC
- 1 = HW HWC HWDC
- 2 = WCH WDHC
- 3 = CWH DWHC
- 4 = HCW HDWC
- 5 = CHW DHWC
- 6 = WHCD
- 7 = HWCD
- 8 = WCHD
- 9 = CWHD
- 10 = HCWD
- 11 = CHWD
- 12 = WDCH
- 13 = DWCH
- 14 = WCDH
- 15 = CWDH
- 16 = DCWH
- 17 = CDWH
- 18 = HDCW
- 19 = DHCW
- 20 = HCDW
- 21 = CHDW
- 22 = DCHW
- 23 = CDHW

# PixelShuffle
```
if mode == 0    y = depth_to_space(x) where x channel order is sw-sh-outc
if mode == 1    y = depth_to_space(x) where x channel order is outc-sw-sh
```

* one_blob_only

| param id  | name          | type | default   | description       |
| --------- | ------------- | ---- | --------- | ----------------- |
| 0         | upscale_factor| int  | 1         |                   |
| 1         | mode          | int  | 0         |                   |

# Pooling
```
x2 = pad(x, pads)
x3 = pooling(x2, kernel, stride)
```

| param id  | name          | type | default   | description       |
| --------- | --------------| ---- | --------- | ----------------- |
| 0         | pooling_type  | int  | 0         |                   |
| 1         | kernel_w      | int  | 0         |                   |
| 2         | stride_w      | int  | 1         |                   |
| 3         | pad_left      | int  | 0         |                   |
| 4         | global_pooling| int  | 0         |                   |
| 5         | pad_mode      | int  | 0         |                   |
| 6         | avgpool_count_include_pad| int | 0 |                 |
| 7         | adaptive_pooling| int | 0        |                   |
| 8         | out_w         | int  | 0         |                   |
| 11        | kernel_h      | int  | kernel_w  |                   |
| 12        | stride_h      | int  | stride_w  |                   |
| 13        | pad_top       | int  | pad_left  |                   |
| 14        | pad_right     | int  | pad_left  |                   |
| 15        | pad_bottom    | int  | pad_top   |                   |
| 18        | out_h         | int  | out_w     |                   |

Pooling type:
- 0 = MAX
- 1 = AVG

Pad mode:
- 0 = full padding
- 1 = valid padding
- 2 = tensorflow padding=SAME or onnx padding=SAME_UPPER
- 3 = onnx padding=SAME_LOWER

# Pooling1D
```
x2 = pad(x, pads)
x3 = pooling1d(x2, kernel, stride)
```

| param id  | name          | type | default   | description       |
| --------- | --------------| ---- | --------- | ----------------- |
| 0         | pooling_type  | int  | 0         |                   |
| 1         | kernel_w      | int  | 0         |                   |
| 2         | stride_w      | int  | 1         |                   |
| 3         | pad_left      | int  | 0         |                   |
| 4         | global_pooling| int  | 0         |                   |
| 5         | pad_mode      | int  | 0         |                   |
| 6         | avgpool_count_include_pad| int | 0 |                 |
| 7         | adaptive_pooling| int | 0        |                   |
| 8         | out_w         | int  | 0         |                   |
| 14        | pad_right     | int  | pad_left  |                   |

Pooling type:
- 0 = MAX
- 1 = AVG

Pad mode:
- 0 = full padding
- 1 = valid padding
- 2 = tensorflow padding=SAME or onnx padding=SAME_UPPER
- 3 = onnx padding=SAME_LOWER

# Pooling3D
```
x2 = pad(x, pads)
x3 = pooling3d(x2, kernel, stride)
```

| param id  | name          | type | default   | description       |
| --------- | --------------| ---- | --------- | ----------------- |
| 0         | pooling_type  | int  | 0         |                   |
| 1         | kernel_w      | int  | 0         |                   |
| 2         | stride_w      | int  | 1         |                   |
| 3         | pad_left      | int  | 0         |                   |
| 4         | global_pooling| int  | 0         |                   |
| 5         | pad_mode      | int  | 0         |                   |
| 6         | avgpool_count_include_pad| int | 0 |                 |
| 7         | adaptive_pooling| int | 0        |                   |
| 8         | out_w         | int  | 0         |                   |
| 11        | kernel_h      | int  | kernel_w  |                   |
| 12        | stride_h      | int  | stride_w  |                   |
| 13        | pad_top       | int  | pad_left  |                   |
| 14        | pad_right     | int  | pad_left  |                   |
| 15        | pad_bottom    | int  | pad_top   |                   |
| 16        | pad_behind    | int  | pad_front |                   |
| 18        | out_h         | int  | out_w     |                   |
| 21        | kernel_d      | int  | kernel_w  |                   |
| 22        | stride_d      | int  | stride_w  |                   |
| 23        | pad_front     | int  | pad_left  |                   |
| 28        | out_d         | int  | out_w     |                   |

Pooling type:
- 0 = MAX
- 1 = AVG

Pad mode:
- 0 = full padding
- 1 = valid padding
- 2 = tensorflow padding=SAME or onnx padding=SAME_UPPER
- 3 = onnx padding=SAME_LOWER

# Power
```
y = pow((shift + x * scale), power)
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | power         | float | 1.f       |                   |
| 1         | scale         | float | 1.f       |                   |
| 2         | shift         | float | 0.f       |                   |

# PReLU
```
if x < 0    y = x * slope
else        y = x
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_slope     | int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| slope_data    | float | [num_slope]           |

# Quantize
```
y = float2int8(x * scale)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | scale_data_size| int  | 1         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| scale_data    | float | [scale_data_size]     |

# Reduction
```
y = reduce_op(x * coeff)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | operation     | int   | 0         |                   |
| 1         | reduce_all    | int   | 1         |                   |
| 2         | coeff         | float | 1.f       |                   |
| 3         | axes          | array | [ ]       |                   |
| 4         | keepdims      | int   | 0         |                   |
| 5         | fixbug0       | int   | 0         | hack for bug fix, should be 1 |

Operation type:
- 0 = SUM
- 1 = ASUM
- 2 = SUMSQ
- 3 = MEAN
- 4 = MAX
- 5 = MIN
- 6 = PROD
- 7 = L1
- 8 = L2
- 9 = LogSum
- 10 = LogSumExp

# ReLU
```
if x < 0    y = x * slope
else        y = x
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | slope         | float | 0.f       |                   |

# Reorg
```
if mode == 0    y = space_to_depth(x) where x channel order is sw-sh-outc
if mode == 1    y = space_to_depth(x) where x channel order is outc-sw-sh
```

* one_blob_only

| param id  | name          | type | default   | description       |
| --------- | ------------- | ---- | --------- | ----------------- |
| 0         | stride        | int  | 1         |                   |
| 1         | mode          | int  | 0         |                   |

# Requantize
```
x2 = x * scale_in + bias
x3 = activation(x2)
y = float2int8(x3 * scale_out)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | scale_in_data_size| int | 1       |                   |
| 1         | scale_out_data_size| int | 1      |                   |
| 2         | bias_data_size| int   | 0         |                   |
| 3         | activation_type| int  | 0         |                   |
| 4         | activation_params| int | [ ]      |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| scale_in_data | float | [scale_in_data_size]  |
| scale_out_data| float | [scale_out_data_size] |
| bias_data     | float | [bias_data_size]      |

# Reshape
```
y = reshape(x)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | w             | int   | -233      |                   |
| 1         | h             | int   | -233      |                   |
| 11        | d             | int   | -233      |                   |
| 2         | c             | int   | -233      |                   |
| 6         | shape_expr    | str   | ""        |                   |

Reshape flag:
- 0 = copy from bottom
- -1 = remaining
- -233 = drop this dim(default)

# RMSNorm
```
split x along outmost axis into part x0, x1 ...
root mean square normalize for each part x0, x1 ...
y = x * gamma by elementwise
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | affine_size   | int   | 0         |                   |
| 1         | eps           | float | 0.001f    | x = x / sqrt(var + eps) |
| 2         | affine        | int   | 1         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| gamma_data    | float | [affine_size]         |

# RNN
Apply a single-layer RNN to a feature sequence of `T` timesteps. The input blob shape is `[w=input_size, h=T]` and the output blob shape is `[w=num_output, h=T]`.

```
y = rnn(x)
y0, hidden y1 = rnn(x0, hidden x1)
```

* one_blob_only if bidirectional

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         | hidden size of output |
| 1         | weight_data_size| int | 0         | total size of weight matrix |
| 2         | direction     | int   | 0         | 0=forward, 1=reverse, 2=bidirectional |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| weight_xc_data| float/fp16/int8 | [input_size, num_output, num_directions] |
| bias_c_data   | float/fp16/int8 | [num_output, 1, num_directions] |
| weight_hc_data| float/fp16/int8 | [num_output, num_output, num_directions] |

Direction flag:
- 0 = forward only
- 1 = reverse only
- 2 = bidirectional

# Scale
```
if scale_data_size == -233  y = x0 * x1
else                        y = x * scale + bias
```

* one_blob_only if scale_data_size != -233
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | scale_data_size| int  | 0         |                   |
| 1         | bias_term     | int   | 0         |                   |

| weight        | type  | shape                 |
| ------------- | ----- | --------------------- |
| scale_data    | float | [scale_data_size]     |
| bias_data     | float | [scale_data_size]     |

# SDPA
```
scaled dot product attention
for each num_head part
    qk = q * k
    qk = qk + attn_mask if attn_mask exists
    softmax(qk)
    qkv = qk * v
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 5         | attn_mask     | int   | 0         |                   |
| 6         | scale         | float | 0.f       | auto = 1.f / sqrt(embed_dim) |
| 7         | kv_cache      | int   | 0         |                   |
| 18        | int8_scale_term | int | 0         |                   |

# SELU
```
if x < 0    y = (exp(x) - 1.f) * alpha * lambda
else        y = x * lambda
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | alpha         | float | 1.67326324f|                  |
| 1         | lambda        | float | 1.050700987f|                 |

# Shrink
```
if x < -lambd y = x + bias
if x >  lambd y = x - bias
else          y = x
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | bias          | float | 0.0f      |                   |
| 1         | lambd         | float | 0.5f      |                   |

# ShuffleChannel
```
if reverse == 0     y = shufflechannel(x) by group
if reverse == 1     y = shufflechannel(x) by channel / group
```

* one_blob_only

| param id  | name          | type | default   | description       |
| --------- | ------------- | ---- | --------- | ----------------- |
| 0         | group         | int  | 1         |                   |
| 1         | reverse       | int  | 0         |                   |

# Sigmoid
```
y = 1 / (1 + exp(-x))
```

* one_blob_only
* support_inplace

# Slice
```
split x along axis into slices, each part slice size is based on slices array
```

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | slices        | array | [ ]       |                   |
| 1         | axis          | int   | 0         |                   |
| 2         | indices       | array | [ ]       |                   |

# Softmax
```
softmax(x, axis)
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | axis          | int   | 0         |                   |
| 1         | fixbug0       | int   | 0         | hack for bug fix, should be 1 |

# Softplus
```
y = log(exp(x) + 1)
```

* one_blob_only
* support_inplace

# Spectrogram
```
x1 = pad(x) if center
y = stft(x1)
y = y / sqrt(norm) if normalized

if power == 0 return y as real
if power == 1 return magnitude
if power == 2 return square of magnitude
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | n_fft         | int   | 0         |                   |
| 1         | power         | int   | 0         |                   |
| 2         | hoplen        | int   | n_fft / 4 |                   |
| 3         | winlen        | int   | n_fft     |                   |
| 4         | window_type   | int   | 0         | 0=ones 1=hann 2=hamming |
| 5         | center        | int   | 1         |                   |
| 6         | pad_type      | int   | 2         | 0=CONSTANT 1=REPLICATE 2=REFLECT |
| 7         | normalized    | int   | 0         | 0=no 1=n_fft 2=window-l2-energy |
| 8         | onesided      | int   | 1         |                   |

# Split
```
y0, y1 ... = x
```

# Squeeze

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | squeeze_w     | int   | 0         |                   |
| 1         | squeeze_h     | int   | 0         |                   |
| 11        | squeeze_d     | int   | 0         |                   |
| 2         | squeeze_c     | int   | 0         |                   |
| 3         | axes          | array | [ ]       |                   |

# Swish
```
y = x / (1 + exp(-x))
```

* one_blob_only
* support_inplace

# TanH
```
y = tanh(x)
```

* one_blob_only
* support_inplace

# Threshold
```
if x > threshold    y = 1
else                y = 0
```

* one_blob_only
* support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | threshold     | float | 0.f       |                   |

# Tile
```
y = repeat tiles along axis for x
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | axis          | int   | 0         |                   |
| 1         | tiles         | int   | 1         |                   |
| 2         | repeats       | array | [ ]       |                   |

# UnaryOp
```
y = unaryop(x)
```

- one_blob_only
- support_inplace

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | op_type       | int   | 0         | Operation type as follows |

Operation type:
- 0 = ABS
- 1 = NEG
- 2 = FLOOR
- 3 = CEIL
- 4 = SQUARE
- 5 = SQRT
- 6 = RSQ
- 7 = EXP
- 8 = LOG
- 9 = SIN
- 10 = COS
- 11 = TAN
- 12 = ASIN
- 13 = ACOS
- 14 = ATAN
- 15 = RECIPROCAL
- 16 = TANH
- 17 = LOG10
- 18 = ROUND
- 19 = TRUNC

# Unfold
```
y = unfold(x)
```

* one_blob_only

| param id  | name          | type  | default   | description       |
| --------- | ------------- | ----- | --------- | ----------------- |
| 0         | num_output    | int   | 0         |                   |
| 1         | kernel_w      | int   | 0         |                   |
| 2         | dilation_w    | int   | 1         |                   |
| 3         | stride_w      | int   | 1         |                   |
| 4         | pad_left      | int   | 0         |                   |
| 11        | kernel_h      | int   | kernel_w  |                   |
| 12        | dilation_h    | int   | dilation_w |                  |
| 13        | stride_h      | int   | stride_w  |                   |
| 14        | pad_top       | int   | pad_left  |                   |
| 15        | pad_right     | int   | pad_left  |                   |
| 16        | pad_bottom    | int   | pad_top   |                   |
