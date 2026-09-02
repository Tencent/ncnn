# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models
from pnnx_test_utils import test_model_formats

def test():
    net = models.squeezenet1_1()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    return test_model_formats(net, (x,), a, "test_squeezenet1_1")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
