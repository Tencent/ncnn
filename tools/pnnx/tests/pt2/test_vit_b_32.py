# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version

def test():
    if version.parse(torchvision.__version__) < version.parse('0.12'):
        return True

    net = models.vit_b_32()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export pt2
    mod = torch.export.export(net, (x,))
    torch.export.save(mod, "test_vit_b_32.pt2")

    # pt2 to pnnx
    import os
    if version.parse(torch.__version__) >= version.parse('2.0'):
        os.system(os.path.normpath("../../src/pnnx") + " test_vit_b_32.pt2")
    else:
        os.system(os.path.normpath("../../src/pnnx") + " test_vit_b_32.pt2 inputshape=[1,3,224,224]")

    # pnnx inference
    import test_vit_b_32_pnnx
    b = test_vit_b_32_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
