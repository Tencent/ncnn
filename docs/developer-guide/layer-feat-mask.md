# layer feature mask

Each ncnn layer allows a special parameter pair `31=X` to control specific bahavior.

X is an unsigned integer with each bit contributing a feature mask.

We usually use it to configuring fine-graded behaviors for certain layers to maintain accuracy, reduce memory usage or optimize performance.

|bit|value|mask|rationale|
|---|---|---|---|
|1<<0|1|no fp16 arithmetic|precision concern|
|1<<1|2|no fp16 storage|precision concern|
|1<<2|4|no bf16 storage|precision concern|
|1<<3|8|no int8|debug dynamic quantized model|
|1<<4|16|no vulkan|reduce overhead for cpu op - gpu split - cpu op|
|1<<5|32|no sgemm|reduce some memory|
|1<<6|64|no winograd|reduce some memory|
|1<<7|128|no threading|force single thread|

These bits can be OR-combined into one value to control multiple behaviors simultaneously.

For example, `31=17` means disabling both vulkan and fp16 arithmetic.

## disable fp16 for certain layer to fix overflow

```ruby
7767517
3 3
Input           input   0 1 input0 0=22 1=22 2=32
Convolution     conv0   1 1 input0 conv0 0=32 1=1 6=1024 9=1
Convolution     conv1   1 1 conv0 conv1 0=128 1=3 6=36864 9=1
```

Typically, we use fp16 computation to improve inference speed.
However, since the weight value of `conv1` is very large, fp16 accumulation may cause numerical overflow, so fp16 needs to be disabled individually for `conv1`, while other layers continue to use fp16 mode

Add `31=3` to disable fp16 storage and arithmetic.

```ruby
7767517
3 3
Input           input   0 1 input0 0=22 1=22 2=32
Convolution     conv0   1 1 input0 conv0 0=32 1=1 6=1024 9=1
Convolution     conv1   1 1 conv0 conv1 0=128 1=3 6=36864 9=1 31=3
```

## disable vulkan for certain layer to improve performance

```ruby
7767517
5 5
Input           input   0 1 input0 0=22 1=22 2=32
Convolution     conv0   1 1 input0 conv0 0=32 1=1 6=1024 9=1
SomeCPULayer    c0      1 1 conv0 c0 0=32
ReLU            relu0   1 1 c0 relu0
SomeCPULayer    c1      1 1 relu0 c1 0=32
```

Between the CPU layers, there is a simple calculation layer that supports vulkan. We can set `31=16` to force it to run on CPU. This can avoid the overhead of data upload, download and storage layout conversion between CPU and GPU. After all, CPU is fast enough for simple operations.

```ruby
7767517
5 5
Input           input   0 1 input0 0=22 1=22 2=32
Convolution     conv0   1 1 input0 conv0 0=32 1=1 6=1024 9=1
SomeCPULayer    c0      1 1 conv0 c0 0=32
ReLU            relu0   1 1 c0 relu0 31=16
SomeCPULayer    c1      1 1 relu0 c1 0=32
```

## disable winograd for certain layer to reduce memory usage

```ruby
7767517
3 3
Input           input   0 1 input0 0=22 1=22 2=32
Convolution     conv0   1 1 input0 conv0 0=32 1=1 6=1024 9=1
Convolution     conv1   1 1 conv0 conv1 0=128 1=3 6=36864 9=1
```

The winograd technology uses more memory for the purpose of improving convolution performance, but this is not always true. In some memory-constrained situations, or memory IO bottlenecks, we can disable the use of winograd on some layers in exchange for a smaller memory footprint. Add `31=64` to Convolution layer, which forces it to use implcit-gemm or tiled im2col-gemm implementation, reducing memory usage and sometimes improving vulkan performance.

```ruby
7767517
3 3
Input           input   0 1 input0 0=22 1=22 2=32
Convolution     conv0   1 1 input0 conv0 0=32 1=1 6=1024 9=1
Convolution     conv1   1 1 conv0 conv1 0=128 1=3 6=36864 9=1 31=64
```

## disable threading for certain layer to improve performance

```ruby
7767517
4 4
Input           input   0 1 input0 0=22 1=22 2=3
Convolution     conv0   1 1 input0 conv0 0=16 1=3 6=432
HardSigmoid     hs      1 1 conv0 hs0
Convolution     conv1   1 1 hs0 conv1 0=16 1=3 6=2304
```

The overhead of multi-thread dispatch and merging is too large for small tensors. Add `31=128` to HardSigmoid layer, which forces it to execute in a single thread, reducing power consumption and improving performance.

```ruby
7767517
4 4
Input           input   0 1 input0 0=22 1=22 2=3
Convolution     conv0   1 1 input0 conv0 0=16 1=3 6=432
HardSigmoid     hs      1 1 conv0 hs0 31=128
Convolution     conv1   1 1 hs0 conv1 0=16 1=3 6=2304
```
