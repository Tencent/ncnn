# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version

def test():
    if version.parse(torchvision.__version__) < version.parse('0.12'):
        return True

    net = models.convnext_tiny().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_convnext_tiny.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_convnext_tiny.pt inputshape=[1,3,224,224]")

    # ncnn inference
    import test_convnext_tiny_ncnn
    b = test_convnext_tiny_ncnn.test_inference()

    return torch.allclose(a, b, 1e-3, 1e-3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
