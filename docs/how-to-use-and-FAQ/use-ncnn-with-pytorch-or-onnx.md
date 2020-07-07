Here is a practical guide for converting pytorch model to ncnn

resnet18 is used as the example

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

