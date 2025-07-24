Here is a practical guide for converting pytorch model to ncnn

resnet18 is used as the example

## pytorch to ncnn, onnx to ncnn

### What's the pnnx?
PyTorch Neural Network eXchange(PNNX) is an open standard for PyTorch model interoperability. PNNX provides an open model format for PyTorch. It defines computation graph as well as high level operators strictly matches PyTorch.
It is recommended to use the `pnnx` tool to convert your `onnx` or `pytorch` model into a ncnn model now.

### How to install pnnx?
* A. python pip (recommended)
  * Windows/Linux/macOS 64bit
  * python 3.7 or later

  ```shell
  pip3 install pnnx
  ```

* B. portable binary package (recommended if you hate python)
  * Windows/Linux/macOS 64bit
  * For Linux, glibc 2.17+

  Download portable pnnx binary package from https://github.com/pnnx/pnnx/releases and extract it.

* C. build from source
  1. install pytorch
  2. (optional) install torchvision for pnnx torchvision operator support
  3. (optional) install protobuf for pnnx onnx-zero support
  4. clone https://github.com/Tencent/ncnn.git
  5. build pnnx in ncnn/tools/pnnx with cmake

  You will probably refer https://github.com/pnnx/pnnx/blob/main/.github/workflows/release.yml for detailed steps

  ```shell
  git clone https://github.com/Tencent/ncnn.git
  mkdir ncnn/tools/pnnx/build
  cd ncnn/tools/pnnx/build
  cmake -DCMAKE_INSTALL_PREFIX=install -DTorch_INSTALL_DIR=<your libtorch install dir> -DTorchVision_INSTALL_DIR=<your torchvision install dir> ..
  cmake --build . --config Release -j 4
  cmake --build . --config Release --target install
  ```

### How to use pnnx?
* A. python
  1. optimize and export your torch model with pnnx.export()
      ```python
      import torch
      import torchvision.models as models
      import pnnx

      model = models.resnet18(pretrained=True)

      x = torch.rand(1, 3, 224, 224)

      opt_model = pnnx.export(model, "resnet18.pt", x)

      # use tuple for model with multiple inputs
      # opt_model = pnnx.export(model, "resnet18.pt", (x, y, z))
      ```
  2. use optimized module just like the normal one
      ```python
      result = opt_model(x) 
      ```
  3. pick resnet18_pnnx.py for pnnx-optimized torch model
  4. pick resnet18.ncnn.param and resnet18.ncnn.bin for ncnn inference

B. command line
  1. export your torch model to torchscript / onnx
      ```python
      import torch
      import torchvision.models as models

      net = models.resnet18(pretrained=True)
      net = net.eval()

      x = torch.rand(1, 3, 224, 224)

      # You could try disabling checking when tracing raises error
      # mod = torch.jit.trace(net, x, check_trace=False)
      mod = torch.jit.trace(net, x)

      mod.save("resnet18.pt")

      # You could also try exporting to the good-old onnx
      torch.onnx.export(net, x, 'resnet18.onnx')
      ```

  2. pnnx convert torchscript / onnx to optimized pnnx model and ncnn model files
      ```shell
      ./pnnx resnet18.pt inputshape=[1,3,224,224]
      ./pnnx resnet18.onnx inputshape=[1,3,224,224]
      ```
      macOS zsh user may need double quotes to prevent ambiguity
      ```shell
      ./pnnx resnet18.pt "inputshape=[1,3,224,224]"
      ```
      For model with multiple inputs, use list
      ```shell
      ./pnnx resnet18.pt inputshape=[1,3,224,224],[1,32]
      ```
      For model with non-fp32 input data type, add type suffix
      ```shell
      ./pnnx resnet18.pt inputshape=[1,3,224,224]f32,[1,32]i64
      ```
  3. pick resnet18_pnnx.py for pnnx-optimized torch model
  4. pick resnet18.ncnn.param and resnet18.ncnn.bin for ncnn inference

see more pnnx informations: https://github.com/pnnx/pnnx

## pytorch to onnx (deprecated)
<details><summary>pytorch to onnx</summary>
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
</details>

## simplify onnx model (deprecated)
<details><summary>simplify onnx model</summary>
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

### onnxsim

Fortunately, [@daquexian](https://github.com/daquexian) developed a handy tool to eliminate them. cheers!

#### how to use onnxsim?
```shell
pip install onnxsim
python -m onnxsim resnet18.onnx resnet18-sim.onnx
```
more informations: https://github.com/daquexian/onnx-simplifier

### onnxslim

Or you can use another powerful model simplification tool implemented in pure Python development by [@inisis](https://github.com/inisis):

#### how to use onnxslim?
```shell
pip install onnxslim
python -m onnxslim resnet18.onnx resnet18-slim.onnx
```
more informations: https://github.com/inisis/OnnxSlim
</details>

## onnx2ncnn (deprecated)

~~The onnx2ncnn tool has stopped maintenance. It is recommended to use the PNNX tool~~

<details><summary>onnx2ncnn tool</summary>

~~Finally, you can convert the model to ncnn using tools/onnx2ncnn~~

~~onnx2ncnn resnet18-sim.onnx resnet18.param resnet18.bin~~
</details>