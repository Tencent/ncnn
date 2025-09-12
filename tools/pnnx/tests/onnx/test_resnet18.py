# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models
from packaging import version

def test():
    net = models.resnet18()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export onnx
    torch.onnx.export(net, (x,), "test_resnet18.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_resnet18.onnx inputshape=[1,3,224,224]")

    # pnnx inference
    import test_resnet18_pnnx
    b = test_resnet18_pnnx.test_inference()

    if not torch.allclose(a, b, 1e-4, 1e-4):
        return False

    if version.parse(torch.__version__) < version.parse('2.8'):
        return True

    # export dynamo onnx
    torch.onnx.dynamo_export(net, x).save("test_resnet18_dynamo.onnx")

    # onnx to pnnx
    os.system("../../src/pnnx test_resnet18_dynamo.onnx inputshape=[1,3,224,224]")

    # pnnx inference
    import test_resnet18_dynamo_pnnx
    b = test_resnet18_dynamo_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
