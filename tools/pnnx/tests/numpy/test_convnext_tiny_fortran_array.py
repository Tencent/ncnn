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
    f_np = npy.asfortranarray(np);
    npy.save("test_convnext_tiny_fortran_array_input1.npy", f_np)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_convnext_tiny_fortran_array.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_convnext_tiny_fortran_array.pt input=test_convnext_tiny_fortran_array_input1.npy")

    # pnnx inference
    import test_convnext_tiny_fortran_array_pnnx
    b = test_convnext_tiny_fortran_array_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
