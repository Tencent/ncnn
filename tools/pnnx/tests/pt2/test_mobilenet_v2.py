# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models
from packaging import version

def test():
    net = models.mobilenet_v2()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export pt2
    mod = torch.export.export(net, (x,))
    torch.export.save(mod, "test_mobilenet_v2.pt2")

    # pt2 to pnnx
    import os
    if version.parse(torch.__version__) >= version.parse('2.0'):
        os.system(os.path.normpath("../../src/pnnx") + " test_mobilenet_v2.pt2")
    else:
        os.system(os.path.normpath("../../src/pnnx") + " test_mobilenet_v2.pt2 inputshape=[1,3,224,224]")

    # pnnx inference
    import test_mobilenet_v2_pnnx
    b = test_mobilenet_v2_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
