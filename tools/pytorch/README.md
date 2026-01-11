# pytorch to ncnn Conversion Guide

This guide is for pytorch users who want to convert their models to the ncnn format.

## The Recommended Tool: pnnx

The recommended and most robust method is to use **[pnnx](https://github.com/pnnx/pnnx)**.

pnnx is the new-generation model converter that is actively developed and maintained. It offers a more robust and flexible solution for converting models from various deep learning frameworks into ncnn.

* pnnx: https://github.com/pnnx/pnnx
* ncnn: https://github.com/Tencent/ncnn
* supported pytorch operators: https://github.com/Tencent/ncnn/tree/master/tools/pnnx#supported-pytorch-operator-status

## Quick Start: Direct Conversion with `pnnx.export` (Recommended)

This is the simplest and most recommended workflow. It allows you to convert a `torch.nn.Module` object into ncnn files without leaving your Python environment.

Install pnnx and use `pnnx.export` in your python script.

```bash
pip3 install pnnx
```

Modify your script to call `pnnx.export` after defining your model. You need to provide the model instance and a dummy input tensor that defines the input shape.

Here is a complete example:

```python
import torch
import torch.nn as nn
import pnnx

# 1. Define your pytorch model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 224 * 224, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 2. Instantiate your model and set it to evaluation mode
model = MyModel()
model.eval()

# 3. Create a dummy input tensor with the correct shape
#    Format: [batch, channels, height, width]
input_tensor = torch.rand(1, 3, 224, 224)

# 4. Export the model to ncnn format
#    The first argument is the model instance.
#    The second argument is a tuple of input tensors.
#    The third argument is the base path for the output files.
pnnx.export(model, "my_model.pt", (input_tensor,))

print("Conversion finished! Check for my_model.ncnn.param and my_model.ncnn.bin")
```

After running this script, you will get `my_model.ncnn.param` and `my_model.ncnn.bin` in the same directory.

## Alternative Workflow: Using TorchScript

This method involves two steps: first exporting your model to a TorchScript (`.pt`) file, and then using the pnnx command-line tool to perform the conversion. This can be useful for workflows where you already have TorchScript models.

### 1. Export to TorchScript

In your Python script, use `torch.jit.trace` to create a `.pt` file.

```python
import torch
import torch.nn as nn

# Define or load your model as in the example above
class MyModel(nn.Module):
    # ... (same model definition)
    pass

model = MyModel()
model.eval()

# Create a dummy input
input_tensor = torch.rand(1, 3, 224, 224)

# Trace the model to generate a TorchScript file
traced_module = torch.jit.trace(model, input_tensor)
traced_module.save("my_model.pt")

print("TorchScript model saved to my_model.pt")
```

### 2. Convert with pnnx Command-Line Tool

Install pnnx and run the following command in your terminal

```bash
pip3 install pnnx

# Syntax: pnnx <torchscript_model_path>
# Example:
pnnx my_model.pt
```

This command will read `my_model.pt` and generate the `my_model.ncnn.param` and `my_model.ncnn.bin` files, ready for use with ncnn.
