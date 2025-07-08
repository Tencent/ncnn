# Copyright 2021 Tencent
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

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_resnet18.pt")

    # torchscript to pnnx
    import os
    if version.parse(torch.__version__) >= version.parse('2.0'):
        os.system("../src/pnnx test_resnet18.pt")
    else:
        os.system("../src/pnnx test_resnet18.pt inputshape=[1,3,224,224]")

    # pnnx inference
    import test_resnet18_pnnx
    b = test_resnet18_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
