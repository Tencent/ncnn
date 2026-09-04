# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version
from pnnx_test_utils import test_model_formats

def test():
    if version.parse(torchvision.__version__) < version.parse('0.12'):
        return True

    net = models.vit_b_32()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    check_trace = version.parse(torch.__version__) < version.parse('1.12.0')
    return test_model_formats(
        net,
        (x,),
        a,
        "test_vit_b_32",
        lambda expected, actual: torch.allclose(expected, actual, 1e-4, 1e-4),
        check_trace=check_trace,
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
