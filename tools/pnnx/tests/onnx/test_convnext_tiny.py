# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version

def test():
    if version.parse(torchvision.__version__) < version.parse('0.12'):
        return True

    net = models.convnext_tiny()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export onnx
    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x,), "test_convnext_tiny.onnx", dynamo=False)
    else:
        torch.onnx.export(net, (x,), "test_convnext_tiny.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_convnext_tiny.onnx inputshape=[1,3,224,224]")

    # pnnx inference
    import test_convnext_tiny_pnnx
    b = test_convnext_tiny_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
