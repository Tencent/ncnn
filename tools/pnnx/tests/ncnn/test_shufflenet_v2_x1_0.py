# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models

def test():
    net = models.shufflenet_v2_x1_0().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_shufflenet_v2_x1_0.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_shufflenet_v2_x1_0.pt inputshape=[1,3,224,224]")

    # ncnn inference
    import test_shufflenet_v2_x1_0_ncnn
    b = test_shufflenet_v2_x1_0_ncnn.test_inference()

    return torch.allclose(a, b, 1e-2, 1e-2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
