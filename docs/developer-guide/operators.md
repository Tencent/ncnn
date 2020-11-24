
* [absval](#absval)
* [argmax](#argmax)
* [batchnorm](#batchnorm)
* [bias](#bias)
* [clip](#clip)
* [concat](#concat)
* [convolution](#convolution)
* [dequantize](#dequantize)

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

