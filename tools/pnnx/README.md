# PNNX
PyTorch Neural Network eXchange(PNNX) is an open standard for PyTorch model interoperability. PNNX provides an open model format for PyTorch. It defines computation graph as well as high level operators strictly matches PyTorch.

# Rationale
PyTorch is currently one of the most popular machine learning frameworks. We need to deploy the trained AI model to various hardware and environments more conveniently and easily.

Before PNNX, we had the following methods:

1. export to ONNX, and deploy with ONNX-runtime
2. export to ONNX, and convert onnx to inference-framework specific format, and deploy with TensorRT/OpenVINO/ncnn/etc.
3. export to TorchScript, and deploy with libtorch

As far as we know, ONNX has the ability to express the PyTorch model and it is an open standard. People usually use ONNX as an intermediate  representation between PyTorch and the inference platform. However, ONNX still has the following fatal problems, which makes the birth of PNNX necessary:

1. ONNX does not have a human-readable and editable file representation, making it difficult for users to easily modify the computation graph or add custom operators.
2. The operator definition of ONNX is not completely in accordance with PyTorch. When exporting some PyTorch operators, glue operators are often added passively by ONNX, which makes the computation graph inconsistent with PyTorch and may impact the inference efficiency.
3. There are a large number of additional parameters designed to be compatible with various ML frameworks in the operator definition in ONNX. These parameters increase the burden of inference implementation on hardware and software.

PNNX tries to define a set of operators and a simple and easy-to-use format that are completely contrasted with the python api of PyTorch, so that the conversion and interoperability of PyTorch models are more convenient.

# Features

1. [Human readable and editable format](#the-pnnxparam-format)
2. [Plain model binary in storage zip](#the-pnnxbin-format)
3. [One-to-one mapping of PNNX operators and PyTorch python api](#pnnx-operator)
4. [Preserve math expression as one operator](#pnnx-expression-operator)
5. [Preserve torch function as one operator](#pnnx-torch-function-operator)
6. [Preserve miscellaneous module as one operator](#pnnx-module-operator)
7. [Inference via exported PyTorch python code](#pnnx-python-inference)
8. [Tensor shape propagation](#pnnx-shape-propagation)
9. [Model optimization](#pnnx-model-optimization)
10. [Custom operator support](#pnnx-custom-operator)

# Build TorchScript to PNNX converter

1. Install PyTorch and TorchVision c++ library
2. Build PNNX with cmake

# Usage

1. Export your model to TorchScript

```python
import torch
import torchvision.models as models

net = models.resnet18(pretrained=True)
net = net.eval()

x = torch.rand(1, 3, 224, 224)

mod = torch.jit.trace(net, x)
torch.jit.save(mod, "resnet18.pt")
```

2. Convert TorchScript to PNNX

```shell
pnnx resnet18.pt resnet18.pnnx.param resnet18.pnnx.bin
```

3. Visualize PNNX with Netron

Open https://netron.app/ in browser, and drag resnet18.pnnx.param into it.

# The pnnx.param format
### example
```
7767517
4 3
Input       pnnx_input_1    0 1 x.1
nn.Conv2d   conv_0_0        1 1 x.1 19 bias=1 dilation=(1,1) groups=1 in_channels=12 kernel_size=(3,3) out_channels=16 padding=(0,0) stride=(1,1) @bias=(16) @weight=(16,12,3,3)
nn.Conv2d   conv_0_1        1 1 19 20 bias=1 dilation=(1,1) groups=1 in_channels=16 kernel_size=(2,2) out_channels=20 padding=(2,2) stride=(2,2) @bias=(20) @weight=(20,16,2,2)
Output      pnnx_output_0   1 0 20
```
### overview
```
[magic]
```
* magic number : 7767517
```
[operator count] [operand count]
```
* operator count : count of the operator line follows
* operand count : count of all operands
### operator line
```
[type] [name] [input count] [output count] [input operands] [output operands] [operator params]
```
* type : type name, such as Conv2d ReLU etc
* name : name of this operator
* input count : count of the operands this operator needs as input
* output count : count of the operands this operator produces as output
* input operands : name list of all the input blob names, separated by space
* output operands : name list of all the output blob names, separated by space
* operator params : key=value pair list, separated by space, operator weights are prefixed by ```@``` symbol, tensor shapes are prefixed by ```#``` symbol, input parameter keys are prefixed by ```$```

# The pnnx.bin format

pnnx.bin file is a zip file with store-only mode(no compression)

weight binary file has its name composed by operator name and weight name

For example, ```nn.Conv2d   conv_0_0        1 1 x.1 19 bias=1 dilation=(1,1) groups=1 in_channels=12 kernel_size=(3,3) out_channels=16 padding=(0,0) stride=(1,1) @bias=(16) @weight=(16,12,3,3)``` would pull conv_0_0.weight and conv_0_0.bias into pnnx.bin zip archive.

# PNNX operator
PNNX always preserve operators from what PyTorch python api provides.

Here is the netron visualization comparision among ONNX, TorchScript and PNNX with the original PyTorch python code shown.

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=32)

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        return x
```

![MultiheadAttention.onnx](https://raw.githubusercontent.com/nihui/ncnn/pnnx/tools/pnnx/assets/MultiheadAttention.onnx.png)

![MultiheadAttention.pt](https://raw.githubusercontent.com/nihui/ncnn/pnnx/tools/pnnx/assets/MultiheadAttention.pt.png)

![MultiheadAttention.pnnx](https://raw.githubusercontent.com/nihui/ncnn/pnnx/tools/pnnx/assets/MultiheadAttention.pnnx.png)

# PNNX expression operator
PNNX trys to preserve expression from what PyTorch python code writes.

Here is the netron visualization comparision among ONNX, TorchScript and PNNX with the original PyTorch python code shown.

```python
import torch

def foo(x, y):
    return torch.sqrt((2 * x + y) / 12)
```

![math.onnx](https://raw.githubusercontent.com/nihui/ncnn/pnnx/tools/pnnx/assets/math.onnx.png)

![math.pt](https://raw.githubusercontent.com/nihui/ncnn/pnnx/tools/pnnx/assets/math.pt.png)

![math.pnnx](https://raw.githubusercontent.com/nihui/ncnn/pnnx/tools/pnnx/assets/math.pnnx.png)


# PNNX module operator
Users could ask PNNX to keep module as one big operator when it has complex logic.

Here is the netron visualization comparision among ONNX, TorchScript and PNNX with the original PyTorch python code shown.

TBD


# Supported PyTorch operator status

| torch.nn        | Is Supported |
|---------------------------|----|
|nn.AdaptiveAvgPool1d       | :heavy_check_mark: |
|nn.AdaptiveAvgPool2d       | :heavy_check_mark: |
|nn.AdaptiveAvgPool3d       | :heavy_check_mark: |
|nn.AdaptiveMaxPool1d       | :heavy_check_mark: |
|nn.AdaptiveMaxPool2d       | :heavy_check_mark: |
|nn.AdaptiveMaxPool3d       | :heavy_check_mark: |
|nn.AlphaDropout            |   |
|nn.AvgPool1d               | :heavy_check_mark: |
|nn.AvgPool2d               | :heavy_check_mark: |
|nn.AvgPool3d               | :heavy_check_mark: |
|nn.BatchNorm1d             | :heavy_check_mark: |
|nn.BatchNorm2d             | :heavy_check_mark: |
|nn.BatchNorm3d             | :heavy_check_mark: |
|nn.Bilinear                |   |
|nn.CELU                    | :heavy_check_mark: |
|nn.ChannelShuffle          | :heavy_check_mark: |
|nn.ConstantPad1d           | :heavy_check_mark: |
|nn.ConstantPad2d           | :heavy_check_mark: |
|nn.ConstantPad3d           | :heavy_check_mark: |
|nn.Conv1d                  | :heavy_check_mark: |
|nn.Conv2d                  | :heavy_check_mark: |
|nn.Conv3d                  | :heavy_check_mark: |
|nn.ConvTranspose1d         | :heavy_check_mark: |
|nn.ConvTranspose2d         | :heavy_check_mark: |
|nn.ConvTranspose3d         | :heavy_check_mark: |
|nn.CosineSimilarity        |   |
|nn.Dropout                 |   |
|nn.Dropout2d               |   |
|nn.Dropout3d               |   |
|nn.ELU                     | :heavy_check_mark: |
|nn.Embedding               |   |
|nn.EmbeddingBag            |   |
|nn.Flatten                 | :heavy_check_mark: |
|nn.Fold                    |   |
|nn.FractionalMaxPool2d     |   |
|nn.FractionalMaxPool3d     |   |
|nn.GELU                    | :heavy_check_mark: |
|nn.GroupNorm               | :heavy_check_mark: |
|nn.GRU                     | :heavy_check_mark: |
|nn.GRUCell                 |   |
|nn.Hardshrink              | :heavy_check_mark: |
|nn.Hardsigmoid             | :heavy_check_mark: |
|nn.Hardswish               | :heavy_check_mark: |
|nn.Hardtanh                | :heavy_check_mark: |
|nn.Identity                |   |
|nn.InstanceNorm1d          | :heavy_check_mark: |
|nn.InstanceNorm2d          | :heavy_check_mark: |
|nn.InstanceNorm3d          | :heavy_check_mark: |
|nn.LayerNorm               | :heavy_check_mark: |
|nn.LazyBatchNorm1d         |   |
|nn.LazyBatchNorm2d         |   |
|nn.LazyBatchNorm3d         |   |
|nn.LazyConv1d              |   |
|nn.LazyConv2d              |   |
|nn.LazyConv3d              |   |
|nn.LazyConvTranspose1d     |   |
|nn.LazyConvTranspose2d     |   |
|nn.LazyConvTranspose3d     |   |
|nn.LazyLinear              |   |
|nn.LeakyReLU               | :heavy_check_mark: |
|nn.Linear                  | :heavy_check_mark: |
|nn.LocalResponseNorm       | :heavy_check_mark: |
|nn.LogSigmoid              | :heavy_check_mark: |
|nn.LogSoftmax              | :heavy_check_mark: |
|nn.LPPool1d                | :heavy_check_mark: |
|nn.LPPool2d                | :heavy_check_mark: |
|nn.LSTM                    | :heavy_check_mark: |
|nn.LSTMCell                |   |
|nn.MaxPool1d               | :heavy_check_mark: |
|nn.MaxPool2d               | :heavy_check_mark: |
|nn.MaxPool3d               | :heavy_check_mark: |
|nn.MaxUnpool1d             |   |
|nn.MaxUnpool2d             |   |
|nn.MaxUnpool3d             |   |
|nn.Mish                    | :heavy_check_mark: |
|nn.MultiheadAttention      | :heavy_check_mark: |
|nn.PairwiseDistance        |   |
|nn.PixelShuffle            | :heavy_check_mark: |
|nn.PixelUnshuffle          | :heavy_check_mark: |
|nn.PReLU                   | :heavy_check_mark: |
|nn.ReflectionPad1d         | :heavy_check_mark: |
|nn.ReflectionPad2d         | :heavy_check_mark: |
|nn.ReLU                    | :heavy_check_mark: |
|nn.ReLU6                   | :heavy_check_mark: |
|nn.ReplicationPad1d        | :heavy_check_mark: |
|nn.ReplicationPad2d        | :heavy_check_mark: |
|nn.ReplicationPad3d        | :heavy_check_mark: |
|nn.RNN                     | :heavy_check_mark: |
|nn.RNNBase                 |   |
|nn.RNNCell                 |   |
|nn.RReLU                   | :heavy_check_mark: |
|nn.SELU                    | :heavy_check_mark: |
|nn.Sigmoid                 | :heavy_check_mark: |
|nn.SiLU                    | :heavy_check_mark: |
|nn.Softmax                 | :heavy_check_mark: |
|nn.Softmax2d               |   |
|nn.Softmin                 | :heavy_check_mark: |
|nn.Softplus                | :heavy_check_mark: |
|nn.Softshrink              | :heavy_check_mark: |
|nn.Softsign                | :heavy_check_mark: |
|nn.SyncBatchNorm           |   |
|nn.Tanh                    | :heavy_check_mark: |
|nn.Tanhshrink              | :heavy_check_mark: |
|nn.Threshold               | :heavy_check_mark: |
|nn.Transformer             |   |
|nn.TransformerDecoder      |   |
|nn.TransformerDecoderLayer |   |
|nn.TransformerEncoder      |   |
|nn.TransformerEncoderLayer |   |
|nn.Unflatten               |   |
|nn.Unfold                  |   |
|nn.Upsample                | :heavy_check_mark: |
|nn.UpsamplingBilinear2d    | :heavy_check_mark: |
|nn.UpsamplingNearest2d     | :heavy_check_mark: |
|nn.ZeroPad2d               | :heavy_check_mark: |


| torch.nn.functional | Is Supported |
|---------------------------|----|
|F.adaptive_avg_pool1d      | :heavy_check_mark: |
|F.adaptive_avg_pool2d      | :heavy_check_mark: |
|F.adaptive_avg_pool3d      | :heavy_check_mark: |
|F.adaptive_max_pool1d      | :heavy_check_mark: |
|F.adaptive_max_pool2d      | :heavy_check_mark: |
|F.adaptive_max_pool3d      | :heavy_check_mark: |
|F.affine_grid              |  |
|F.alpha_dropout            |  |
|F.avg_pool1d               | :heavy_check_mark: |
|F.avg_pool2d               | :heavy_check_mark: |
|F.avg_pool3d               | :heavy_check_mark: |
|F.batch_norm               | :heavy_check_mark: |
|F.bilinear                 |  |
|F.celu                     | :heavy_check_mark: |
|F.conv1d                   | :heavy_check_mark: |
|F.conv2d                   | :heavy_check_mark: |
|F.conv3d                   | :heavy_check_mark: |
|F.conv_transpose1d         | :heavy_check_mark: |
|F.conv_transpose2d         | :heavy_check_mark: |
|F.conv_transpose3d         | :heavy_check_mark: |
|F.cosine_similarity        |  |
|F.dropout                  |  |
|F.dropout2d                |  |
|F.dropout3d                |  |
|F.elu                      | :heavy_check_mark: |
|F.elu_                     | :heavy_check_mark: |
|F.embedding                |  |
|F.embedding_bag            |  |
|F.feature_alpha_dropout    |  |
|F.fold                     |  |
|F.fractional_max_pool2d    |  |
|F.fractional_max_pool3d    |  |
|F.gelu                     | :heavy_check_mark: |
|F.glu                      |  |
|F.grid_sample              | :heavy_check_mark: |
|F.group_norm               | :heavy_check_mark: |
|F.gumbel_softmax           |  |
|F.hardshrink               | :heavy_check_mark: |
|F.hardsigmoid              | :heavy_check_mark: |
|F.hardswish                | :heavy_check_mark: |
|F.hardtanh                 | :heavy_check_mark: |
|F.hardtanh_                | :heavy_check_mark: |
|F.instance_norm            | :heavy_check_mark: |
|F.interpolate              | :heavy_check_mark: |
|F.layer_norm               | :heavy_check_mark: |
|F.leaky_relu               | :heavy_check_mark: |
|F.leaky_relu_              | :heavy_check_mark: |
|F.linear                   | :heavy_check_mark: |
|F.local_response_norm      | :heavy_check_mark: |
|F.logsigmoid               | :heavy_check_mark: |
|F.log_softmax              | :heavy_check_mark: |
|F.lp_pool1d                | :heavy_check_mark: |
|F.lp_pool2d                | :heavy_check_mark: |
|F.max_pool1d               | :heavy_check_mark: |
|F.max_pool2d               | :heavy_check_mark: |
|F.max_pool3d               | :heavy_check_mark: |
|F.max_unpool1d             |  |
|F.max_unpool2d             |  |
|F.max_unpool3d             |  |
|F.mish                     | :heavy_check_mark: |
|F.normalize                | :heavy_check_mark: |
|F.one_hot                  |  |
|F.pad                      | :heavy_check_mark: |
|F.pairwise_distance        |  |
|F.pdist                    |  |
|F.pixel_shuffle            | :heavy_check_mark: |
|F.pixel_unshuffle          | :heavy_check_mark: |
|F.prelu                    | :heavy_check_mark: |
|F.relu                     | :heavy_check_mark: |
|F.relu_                    | :heavy_check_mark: |
|F.relu6                    | :heavy_check_mark: |
|F.rrelu                    | :heavy_check_mark: |
|F.rrelu_                   | :heavy_check_mark: |
|F.selu                     | :heavy_check_mark: |
|F.sigmoid                  | :heavy_check_mark: |
|F.silu                     | :heavy_check_mark: |
|F.softmax                  | :heavy_check_mark: |
|F.softmin                  | :heavy_check_mark: |
|F.softplus                 | :heavy_check_mark: |
|F.softshrink               | :heavy_check_mark: |
|F.softsign                 | :heavy_check_mark: |
|F.tanh                     | :heavy_check_mark: |
|F.tanhshrink               | :heavy_check_mark: |
|F.threshold                | :heavy_check_mark: |
|F.threshold_               | :heavy_check_mark: |
|F.unfold                   |  |
|F.upsample                 | :heavy_check_mark: |
|F.upsample_bilinear        | :heavy_check_mark: |
|F.upsample_nearest         | :heavy_check_mark: |
