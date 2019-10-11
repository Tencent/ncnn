## ncnn optimize: auto pack model 技术文档

### 功能
在目标硬件上自动调优模型，挑选卷积的实现方案。结果会是全局最优的。

### 前置条件

1. 请先准备一块 arm linux 开发板，直接用手机步骤会麻烦点。假设想把模型放到 qcom845 芯片手机上执行，尽量准备 845 开发板。若条件有限，大的架构(armv8还是armv7)一致也行，最终优化速度用 CPU 主频直接折算;

2. 使用 ncnn/tools 下的工具，把训练好的模型转换为 ncnn 支持的格式，例如 ncnn.param 和 ncnn.bin;

### 使用方法

执行命令
```
ncnn optimize ncnn.param ncnn.bin out.param out.bin 0 data 227 227 3

Input  [w h nc]: 227 227 3
Kernel [w h nc]: 3 3 192
Output [w h nc]: 113 113 64
im2col cost 14.188ms
direct cost 9.394ms
conv3x3s2 cost 6.555ms
conv1 use conv3x3s2

Input  [w h nc]: 56 56 64
Kernel [w h nc]: 1 1 1024
Output [w h nc]: 56 56 16
im2col cost 1.812ms
direct cost 1.995ms
fire2/squeeze1x1 use im2col

Input  [w h nc]: 56 56 16
Kernel [w h nc]: 1 1 1024
Output [w h nc]: 56 56 64
im2col cost 1.223ms
direct cost 2.169ms
fire2/expand1x1 use im2col

Input  [w h nc]: 58 58 16
Kernel [w h nc]: 3 3 1024
Output [w h nc]: 56 56 64
winograd cost 5.853ms
im2col cost 10.480ms
direct cost 6.752ms
fire2/expand3x3 use winograd
...
```

其中 data 是输入层的名字，由于常见的输入层只有一个，暂时只支持一个；
227 227 3 是实际要使用的 WHC 格式的尺寸，尺寸不同最终选择的方案也不同。考虑到 N 在移动端推理没有任何应用场景，因此我们认为 N = 1。

### 工作原理

首先要认同一件事儿：卷积优化不是某种单一的方法就能搞定的，不存在“一招鲜吃遍天”。
在这个认知基础上，每种方法（MEC/FFT/direct/winograd）在不同的情况（尺寸、内存、核数、能耗等）下都有各自的速度优势。
至于哪种是最快的，实际跑一遍就知道。用黑盒处理黑盒，没必要用一堆判断条件。
此原理保证了 auto pack model 后不会比之前慢。

同理，想知道哪种FC/pooling/dwConv是最快的/能耗最低的，也可以用同样的方法。

### 感谢
最后感谢 up 主自 16 年底开源的 ncnn。
