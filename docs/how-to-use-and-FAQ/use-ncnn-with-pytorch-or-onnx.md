# A Guide to Converting pytorch / onnx Models to ncnn

This guide is designed to help pytorch and onnx users use the new-generation model conversion tool, **pnnx**, to efficiently and reliably convert models to the ncnn format for high-performance inference on the edge.

This document is written and revised based on the **official pnnx documentation**.

* pnnx project: https://github.com/pnnx/pnnx
* ncnn project: https://github.com/Tencent/ncnn
* supported pytorch operators: https://github.com/Tencent/ncnn/tree/master/tools/pnnx#supported-pytorch-operator-status
* supported onnx operators: https://github.com/Tencent/ncnn/tree/master/tools/pnnx#supported-onnx-operator-status

---

## Why is pnnx Highly Recommended?

Regardless of which framework you come from, pnnx offers significant advantages over traditional tools (like `onnx2ncnn`):

*   **Forget the Hassles of onnx**: The traditional `pytorch -> onnx -> ncnn` pipeline often fails due to onnx operator compatibility issues and dynamic shape problems. pnnx can convert directly from pytorch, completely bypassing the unstable intermediate step of onnx.
*   **Core Framework Support**: pnnx focuses on supporting **pytorch** and **onnx**, providing you with a unified and consistent conversion experience.
*   **More Stable and Powerful**: pnnx can handle a wider range of modern operators and complex model architectures, generating cleaner and more accurate ncnn graphs.
*   **Active and Continuous Development**: pnnx is under active development, constantly adding support for the latest operators and features from both source frameworks and the ncnn engine.
*   **Richer Graph Information**: pnnx preserves the original model's structural information during the conversion process, which is highly beneficial for model analysis and subsequent optimization.

---

## Workflow 1: Guide for pytorch Users (Recommended)

For pytorch users, converting directly from a pytorch model is the most stable and efficient path.

### Method A: Direct Conversion in Python with `pnnx.export` (Most Recommended)

This is the simplest and most recommended workflow, allowing you to complete the model conversion with a single command without leaving your Python environment.

#### 1. Install pnnx

First, install the pnnx Python package. This command installs both the `pnnx` Python library and the `pnnx` command-line tool.

```bash
pip3 install pnnx
```

#### 2. Call `pnnx.export` in Your Python Script

Calling the `pnnx.export` function will generate both a TorchScript (`.pt`) file and the `.param` and `.bin` files required by ncnn.

**Complete Code Example:**

```python
import torch
import torch.nn as nn
import pnnx

# 1. Define or load your pytorch model
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

# 2. Instantiate the model and set it to evaluation mode
model = MyModel()
model.eval()

# 3. Create a dummy input tensor with the correct input shape
input_tensor = torch.rand(1, 3, 224, 224)

# 4. Call pnnx.export to export the model
pnnx.export(model, "my_model.pt", (input_tensor,))

print("Conversion complete!")
print("Please check for the generated my_model.pt, my_model.ncnn.param, and my_model.ncnn.bin files.")
```

### Method B: Using the Command-Line Tool (Alternative)

#### 1. Get the pnnx Command-Line Tool

If you have already run `pip install pnnx`, the `pnnx` command is available, and you can proceed to the next step.

For non-Python environments or users who prefer a standalone executable, you can manually download the latest binary from the [pnnx Releases page](https://github.com/pnnx/pnnx/releases).

#### 2. Export to TorchScript (Skip if you already have a .pt file)

```python
import torch
# ... (model definition from above)
model = MyModel()
model.eval()
input_tensor = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, input_tensor)
traced_script_module.save("my_model.pt")
```

#### 3. Run the pnnx Command for Conversion

Run the following command in your terminal.

```bash
# Syntax: pnnx <torchscript_model_path>
pnnx my_model.pt
```

---

## Workflow 2: Guide for onnx Users

For users who already have an `.onnx` file, please use pnnx for conversion.

### 1. Get the pnnx Command-Line Tool

*   **Method 1 (Recommended):** If you have Python in your environment, install it directly via pip.
    ```bash
    pip3 install pnnx
    ```
    The `pnnx` command will be automatically added to your system's path.

*   **Method 2 (Alternative):** For non-Python environments or to use a standalone program, you can download the latest executable from the [pnnx Releases page](https://github.com/pnnx/pnnx/releases).

### 2. Run the Command-Line Conversion

Open a terminal, navigate to the directory containing your model file, and run the following command.

**Basic Command Example:**

```bash
# Syntax: pnnx <onnx_model_path>
pnnx my_model.onnx
```
After the command executes successfully, you will get the `my_model.ncnn.param` and `my_model.ncnn.bin` files, which can be directly loaded and used in your ncnn project.
