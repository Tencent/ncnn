Here is a practical guide for converting pytorch model to ncnn

resnet18 is used as the example

# use pnnx (recommand)
[more about pnnx](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)
## get pnnx
[build on your own](https://zhuanlan.zhihu.com/p/431833958)

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
#### only have onnx model
use [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch), convert onnx to torchscript

#### verify whether the model is exported successfully
1. pnnx generated the corresponding `xxx.ncnn.param` and `xxx.ncnn.bin`
2. open `xxx.ncnn.param` in txt, and there is no `pnnx.XXX` operator in the first column
3. If you are familiar with this model, you can check it against the [operator table](https://github.com/Tencent/ncnn/wiki/operators)

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

