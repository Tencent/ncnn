# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models
from packaging import version

from pnnx_test_utils import convert_and_import

def test():
    net = models.squeezenet1_1()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_squeezenet1_1",
        pnnx_args=() if version.parse(torch.__version__) >= version.parse('2.0') else ("inputshape=[1,3,224,224]",),
    )
    b = mod.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
