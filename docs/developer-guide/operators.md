
* [absval](#absval)
* [argmax](#argmax)
* [batchnorm](#batchnorm)
* [bias](#bias)
* [binaryop](#binaryop)
* [bnll](#bnll)
* [clip](#clip)
* [concat](#concat)
* [convolution](#convolution)
* [dequantize](#dequantize)
* [lstm](#lstm)
* [softmax](#softmax)

# absval
```
y = abs(x)
```

* one_blob_only
* support_inplace

# argmax
```
y = argmax(x, out_max_val, topk)
```

* one_blob_only

|param id|name|type|default|
|--|--|--|--|
|0|out_max_val|int|0|
|1|topk|int|1|

# batchnorm
```
y = (x - mean) / sqrt(var + eps) * slope + bias
```

* one_blob_only
* support_inplace

|param id|name|type|default|
|--|--|--|--|
|0|channels|int|0|
|1|eps|float|0.f|

|weight|type|
|--|--|
|slope_data|float|
|mean_data|float|
|var_data|float|
|bias_data|float|

# bias
```
y = x + bias
```

* one_blob_only
* support_inplace

|param id|name|type|default|
|--|--|--|--|
|0|bias_data_size|int|0|

|weight|type|
|--|--|
|bias_data|float|

# binaryop
 This operation is used for binary computation, and the calculation rule depends on the [broadcasting rule](https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting).
```
C = binaryop(A, B)
```
if with_scalar = 1:
- one_blob_only
- support_inplace

|param id|name|type|default|description|
|--|--|--|--|--|
|0|op_type|int|0|Operation type as follows|
|1|with_scalar|int|0|with_scalar=0 B is a matrix, with_scalar=1 B is a scalar|
|2|b|float|0.f|When B is a scalar, B = b|

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

# bnll
```
y = log(1 + e^(-x)) , x > 0
y = log(1 + e^x),     x < 0
```

* one_blob_only
* support_inplace

# clip
```
y = clamp(x, min, max)
```

* one_blob_only
* support_inplace

|param id|name|type|default|
|--|--|--|--|
|0|min|float|-FLT_MAX|
|1|max|float|FLT_MAX|

# concat
```
y = concat(x0, x1, x2, ...) by axis
```

|param id|name|type|default|
|--|--|--|--|
|0|axis|int|0|

# convolution
```
x2 = pad(x, pads, pad_value)
x3 = conv(x2, weight, kernel, stride, dilation) + bias
y = activation(x3, act_type, act_params)
```

* one_blob_only

|param id|name|type|default|
|--|--|--|--|
|0|num_output|int|0|
|1|kernel_w|int|0|
|2|dilation_w|int|1|
|3|stride_w|int|1|
|4|pad_left|int|0|
|5|bias_term|int|0|
|6|weight_data_size|int|0|
|8|int8_scale_term|int|0|
|9|activation_type|int|0|
|10|activation_params|array|[ ]|
|11|kernel_h|int|kernel_w|
|12|dilation_h|int|dilation_w|
|13|stride_h|int|stride_w|
|15|pad_right|int|pad_left|
|14|pad_top|int|pad_left|
|16|pad_bottom|int|pad_top|
|18|pad_value|float|0.f|

|weight|type|
|--|--|
|weight_data|float/fp16/int8|
|bias_data|float|

# dequantize
```
 y = x * scale + bias
```
* one_blob_only
* support_inplace

|param id|name|type|default|
|--|--|--|--|
|0|scale|float|1.f|
|1|bias_term|int|0|
|2|bias_data_size|int|0|

# lstm
Apply a single-layer LSTM to a feature sequence of `T` timesteps. The input blob shape is `[w=input_size, h=T]` and the output blob shape is `[w=num_output, h=T]`.

* one_blob_only

|param id|name|type|default|description|
|--|--|--|--|--|
|0|num_output|int|0|hidden size of output|
|1|weight_data_size|int|0|total size of IFOG weight matrix|
|2|direction|int|0|0=forward, 1=reverse, 2=bidirectional|

|weight|type|shape|description|
|--|--|--|--|
|weight_xc_data|float|`[w=input_size, h=num_output * 4, c=num_directions]`||
|bias_c_data|float|`[w=num_output, h=4, c=num_directions]`||
|weight_hc_data|float|`[w=num_output, h=num_output * 4, c=num_directions]`||

# softmax
```
softmax(x, axis)
```

* one_blob_only
* support_inplace

|param id|name|type|default|description|
|--|--|--|--|--|
|0|axis|int|0||
|1|fixbug0|int|0|hack for bug fix, should be 1|
