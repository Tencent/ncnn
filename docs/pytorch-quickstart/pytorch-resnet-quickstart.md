# Deploy PyTorch ResNet Model with ncnn

This guide walks through converting a PyTorch-trained ResNet model to ncnn
format and running inference, step by step.

## Prerequisites

- Python 3.8+ with PyTorch installed
- ncnn built from source (see [build guide](https://github.com/Tencent/ncnn/wiki/how-to-build))
- pnnx (PyTorch Neural Network eXchange) tool

```bash
pip install torch torchvision
git clone https://github.com/Tencent/ncnn.git
cd ncnn && mkdir build && cd build && cmake .. && make -j$(nproc)
```

## Step 1: Train/Obtain a ResNet Model

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

# Trace with a dummy input
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
print("ONNX exported: resnet18.onnx")
```

## Step 2: Convert ONNX to ncnn with pnnx

```bash
# pnnx performs operator fusion and optimization beyond basic onnx2ncnn
pnnx resnet18.onnx inputshape=[1,3,224,224]
```

This produces:
- `resnet18.ncnn.param` — model structure
- `resnet18.ncnn.bin` — model weights

## Step 3: Inference in C++

```cpp
#include "net.h"
#include <opencv2/opencv.hpp>

int main() {
    // Load model
    ncnn::Net net;
    net.load_param("resnet18.ncnn.param");
    net.load_model("resnet18.ncnn.bin");

    // Preprocess image (ImageNet normalization)
    cv::Mat img = cv::imread("cat.jpg");
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 224, 224
    );
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1.f / (0.229f * 255.f), 1.f / (0.224f * 255.f), 1.f / (0.225f * 255.f)};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // Inference
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);

    // Post-process: find class with highest score
    int predicted = 0;
    float max_score = out[0];
    for (int c = 1; c < out.w; c++) {
        if (out[c] > max_score) { max_score = out[c]; predicted = c; }
    }
    printf("Predicted class: %d (score: %.4f)\n", predicted, max_score);
    return 0;
}
```

## Step 4: Cross-Platform Build

```cmake
cmake_minimum_required(VERSION 3.10)
project(resnet_ncnn)
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(resnet_ncnn main.cpp)
target_link_libraries(resnet_ncnn ncnn opencv_core opencv_imgproc opencv_imgcodecs)
```

Build on Linux/Windows/macOS:

```bash
# Linux/macOS — set ncnn_DIR to the build directory
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/ncnn/build/install

# Or: install ncnn first, then build
# cd /path/to/ncnn/build && cmake --install .
# mkdir build && cd build && cmake ..

# Windows (MSVC)
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH=C:/path/to/ncnn/build/install
cmake --build . --config Release
```

## Common Issues

| Issue | Solution |
|-------|----------|
| pnnx not found | Build from source: `cd ncnn/tools/pnnx && mkdir build && cd build && cmake .. && make` |
| ONNX opset mismatch | Export with `opset_version=11` for maximum compatibility |
| Shape mismatch | Use `inputshape` parameter in pnnx: `pnnx model.onnx inputshape=[1,3,224,224]` |

## References

- [ncnn Wiki: use-ncnn-with-alexnet](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh)
- [pnnx Documentation](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)
