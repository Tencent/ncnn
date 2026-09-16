# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models
from packaging import version

def test():
    net = models.squeezenet1_1()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_squeezenet1_1.pt")

    # torchscript to pnnx
    import os
    if version.parse(torch.__version__) >= version.parse('2.0'):
        os.system(os.path.join("..", "src", "pnnx") + " test_squeezenet1_1.pt")
    else:
        os.system(os.path.join("..", "src", "pnnx") + " test_squeezenet1_1.pt inputshape=[1,3,224,224]")

    # pnnx inference
    import test_squeezenet1_1_pnnx
    b = test_squeezenet1_1_pnnx.test_inference()

    ts_ok = torch.equal(a, b)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[1,3,224,224]"], "test_squeezenet1_1")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
