# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version

from pnnx_test_utils import convert_and_import

def test():
    if version.parse(torchvision.__version__) < version.parse('0.12'):
        return True

    net = models.vit_b_32()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_vit_b_32",
        pnnx_args=() if version.parse(torch.__version__) >= version.parse('2.0') else ("inputshape=[1,3,224,224]",),
        trace_kwargs={"check_trace": False} if version.parse(torch.__version__) >= version.parse('1.12.0') else {},
    )
    b = mod.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
