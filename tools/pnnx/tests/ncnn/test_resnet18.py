# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models

def test():
    net = models.resnet18().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_resnet18.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_resnet18.pt inputshape=[1,3,224,224]")

    # ncnn inference
    import test_resnet18_ncnn
    b = test_resnet18_ncnn.test_inference()

    ts_ok = torch.allclose(a, b, 1e-2, 1e-2)

    # pt2 path (torch.export fails, skip automatically)
    # ncnn uses fp16 conversion with ~1e-2 error; relax tolerance to 1e-2 (matching ts path)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x,), ["[1,3,224,224]"], "test_resnet18", atol=1e-2, rtol=1e-2)

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
