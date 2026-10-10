# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version

from pnnx_test_utils import convert_and_import

def test():
    if version.parse(torchvision.__version__) < version.parse('0.13'):
        return True

    net = models.swin_t()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_swin_t",
        pnnx_args=() if version.parse(torch.__version__) >= version.parse('2.0') else ("inputshape=[1,3,224,224]",),
    )
    if (version.parse(torch.__version__) >= version.parse('1.12') and version.parse(torch.__version__) < version.parse('1.13') or
        version.parse(torch.__version__) >= version.parse('2.0') and version.parse(torch.__version__) < version.parse('2.1')):
        # torch-1.12 / 2.0 breaks 3d attention mask in no grad mode
        net_pnnx = mod.Model().float().eval()
        b = net_pnnx(x)
    else:
        b = mod.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
