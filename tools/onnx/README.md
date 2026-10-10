# onnx to ncnn Conversion Guide

This guide is for users who want to convert their onnx models to the ncnn format.

## The Recommended Tool: pnnx

The `onnx2ncnn` tool is now considered legacy. We strongly recommend using **[pnnx](https://github.com/pnnx/pnnx)** for all model conversion tasks.

pnnx is the new-generation model converter that is actively developed and maintained. It offers a more robust and flexible solution for converting models from various deep learning frameworks into ncnn.

* pnnx: https://github.com/pnnx/pnnx
* ncnn: https://github.com/Tencent/ncnn
* supported onnx operators: https://github.com/Tencent/ncnn/tree/master/tools/pnnx#supported-onnx-operator-status

## Quick Start: Basic Usage

Using pnnx to convert an onnx model is straightforward.

Install pnnx and run the conversion.

```shell
pip3 install pnnx

# Syntax: pnnx <onnx_model_path>
# Example:
pnnx my_model.onnx
```

After running the command, you will get `my_model.ncnn.param` and `my_model.ncnn.bin`, ready for use with ncnn.
