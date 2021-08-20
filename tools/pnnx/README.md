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
2. [Plain model binary storage](#the-pnnxbin-format)
3. [One-to-one mapping of PNNX operators and PyTorch python api](#pnnx-operator)
4. [Preserve math expression as one operator](#pnnx-expression-operator)
5. [Preserve torch function as one operator](#pnnx-torch-function-operator)
6. [Preserve miscellaneous module as one operator](#pnnx-module-operator)
7. [Inference via exported PyTorch python code](#pnnx-python-inference)

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
Input     pnnx_input_1    0 1 x.1
Conv2d    conv_0_0        1 1 x.1 19 bias=1 dilation=(1,1) groups=1 in_channels=12 kernel_size=(3,3) out_channels=16 padding=(0,0) stride=(1,1) @bias=(16) @weight=(16,12,3,3)
Conv2d    conv_0_1        1 1 19 20 bias=1 dilation=(1,1) groups=1 in_channels=16 kernel_size=(2,2) out_channels=20 padding=(2,2) stride=(2,2) @bias=(20) @weight=(20,16,2,2)
Output    pnnx_output_0   1 0 20
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
* operator params : key=value pair list, separated by space, operator weights are prefixed by ```@``` symbol

# The pnnx.bin format
```
  +---------+---------+---------+---------+---------+---------+
  | weight1 | weight2 | weight3 | weight4 | ....... | weightN |
  +---------+---------+---------+---------+---------+---------+
```
the model binary is the concatenation of all weight data

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
