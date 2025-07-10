# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models
from packaging import version

def test():
    if version.parse(torch.__version__) < version.parse('1.9'):
        return True

    net = models.mobilenet_v3_small()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export onnx
    torch.onnx.export(net, (x,), "test_mobilenet_v3_small.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_mobilenet_v3_small.onnx inputshape=[1,3,224,224]")

    # pnnx inference
    import test_mobilenet_v3_small_pnnx
    b = test_mobilenet_v3_small_pnnx.test_inference()

    if not torch.allclose(a, b, 1e-4, 1e-4):
        return False

    if version.parse(torch.__version__) < version.parse('2.8'):
        return True

    # export dynamo onnx
    torch.onnx.dynamo_export(net, x).save("test_mobilenet_v3_small_dynamo.onnx")

    # onnx to pnnx
    os.system("../../src/pnnx test_mobilenet_v3_small_dynamo.onnx inputshape=[1,3,224,224]")

    # pnnx inference
    import test_mobilenet_v3_small_dynamo_pnnx
    b = test_mobilenet_v3_small_dynamo_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
