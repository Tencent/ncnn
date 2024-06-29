Here is a practical guide for converting pytorch model to ncnn

resnet18 is used as the example

# use pnnx (recommand)
[more about pnnx](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)
## get pnnx
[get pre-built executable file](https://github.com/pnnx/pnnx/releases)

## pytorch to torchscript

```
import torch
import torchvision.models as models

net = models.resnet18(pretrained=True)
net = net.eval()

x = torch.rand(1, 3, 224, 224)

# You could try disabling checking when tracing raises error
# mod = torch.jit.trace(net, x, check_trace=False)
mod = torch.jit.trace(net, x)

mod.save("resnet18.pt")
```

## Convert TorchScript to ncnn
```
pnnx resnet18.pt inputshape=[1,3,224,224]
```
the resnet18.ncnn.param and resnet18.ncnn.bin is all you need.

## some tips
### mutilple input
```pnnx xxx.pt inputshape=[n1,c1,h1,w1],[n2,c2,h2,w2]...```

example:
```
# step 1, trace model
import torch
import mymodel

# “nchw” is the input you need to populate for your own model
input1 = torch.rand(n1, c1, h1, w1)
input2 = torch.rand(n2, c2, h2, w2)

net = mymodel()

mod = torch.trace(net, (input1, input2))
mod.save("mymodel.pt")
```
```
# step 2, use pnnx convert model
pnnx mymodel.pt inputshape=[n1,c1,h1,w1],[n2,c2,h2,w2]
```
### dynamic input
```pnnx xxx.pt inputshape=[n,c,h,w] inputshape2=[n',c',h',w']```

### only have onnx model
use [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch), convert onnx to torchscript

```
# step 1, install onnx2torch

pip install onnx2torch
```

```
# step 2, convert onnx to torch and trace into torchscript
# there are two ways to use onnx2torch

import onnx
import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '/some/path/mobile_net_v2.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)

input = torch.rand(1, 3, 224, 224)

mod1 = torch.trace(torch_model_1, input)
mod1.save("mobile_net_v2.pt")

# “nchw” is the input you need to populate for your own model
input = torch.rand(n, c, h, w)
mod2 = torch.trace(torch_model_2, input)
mod2.save("model.pt")
```

#### verify whether the model is exported successfully
1. pnnx generated the corresponding `xxx.ncnn.param` and `xxx.ncnn.bin`
2. open `xxx.ncnn.param` in txt, and there is no `pnnx.XXX` operator in the first column
3. If you are familiar with this model, you can check it against the [operator table](https://github.com/Tencent/ncnn/wiki/operators)

and also, you can run xxx_pnnx.py and xxx_ncnn.py to verify whether your model was successfully converted

# use onnx2ncnn
## pytorch to onnx

The official pytorch tutorial for exporting onnx model

https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html

```python
import torch
import torchvision
import torch.onnx

# An instance of your model
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 3, 224, 224)

# Export the model
torch_out = torch.onnx._export(model, x, "resnet18.onnx", export_params=True)
```

## simplify onnx model

The exported resnet18.onnx model may contains many redundant operators such as Shape, Gather and Unsqueeze that is not supported in ncnn

```
Shape not supported yet!
Gather not supported yet!
  # axis=0
Unsqueeze not supported yet!
  # axes 7
Unsqueeze not supported yet!
  # axes 7
```

Fortunately, daquexian developed a handy tool to eliminate them. cheers!

https://github.com/daquexian/onnx-simplifier

```
python3 -m onnxsim resnet18.onnx resnet18-sim.onnx
```

## onnx to ncnn

Finally, you can convert the model to ncnn using tools/onnx2ncnn

```
onnx2ncnn resnet18-sim.onnx resnet18.param resnet18.bin
```

