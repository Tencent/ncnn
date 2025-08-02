# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
import numpy as npy
from packaging import version

def test():
    if version.parse(torchvision.__version__) < version.parse('0.12'):
        return True

    net = models.convnext_tiny()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    np = x.numpy()
    r_np = np.astype('>f4')
    npy.save("input1.npy", r_np)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_convnext_tiny.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_convnext_tiny.pt input=input1.npy")

    # pnnx inference
    import test_convnext_tiny_pnnx
    b = test_convnext_tiny_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        print("YES")
        exit(0)
    else:
        exit(1)
