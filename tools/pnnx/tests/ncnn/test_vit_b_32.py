# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version

def test():
    if version.parse(torchvision.__version__) < version.parse('0.12'):
        return True

    net = models.vit_b_32().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export torchscript
    if version.parse(torch.__version__) >= version.parse('1.12.0'):
        mod = torch.jit.trace(net, x, check_trace=False)
    else:
        mod = torch.jit.trace(net, x)
    mod.save("test_vit_b_32.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_vit_b_32.pt inputshape=[1,3,224,224]")

    # ncnn inference
    import test_vit_b_32_ncnn
    b = test_vit_b_32_ncnn.test_inference()

    if not torch.allclose(a, b, 1e-4, 1e-4):
        return False

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x,), ["[1,3,224,224]"], "test_vit_b_32", atol=1e-4, rtol=1e-4)
    if pt2_ok is False:
        return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
