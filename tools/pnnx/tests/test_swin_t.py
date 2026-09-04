# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version
from pnnx_test_utils import test_model_formats

def test():
    if version.parse(torchvision.__version__) < version.parse('0.13'):
        return True

    net = models.swin_t()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    return test_model_formats(
        net,
        (x,),
        a,
        "test_swin_t",
        lambda expected, actual: torch.allclose(expected, actual, 1e-4, 1e-4),
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
