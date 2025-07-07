# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models

def test():
    net = models.quantization.shufflenet_v2_x1_0(quantize=True)
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_quantization_shufflenet_v2_x1_0.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_quantization_shufflenet_v2_x1_0.pt inputshape=[1,3,224,224]")

    # pnnx inference
    import test_quantization_shufflenet_v2_x1_0_pnnx
    b = test_quantization_shufflenet_v2_x1_0_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
