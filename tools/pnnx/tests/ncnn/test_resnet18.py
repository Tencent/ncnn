# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision.models as models
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnnx_test_utils import convert_and_import_ncnn

def test():
    net = models.resnet18().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    module = convert_and_import_ncnn(net, (x,), "test_resnet18", pnnx_args=("inputshape=[1,3,224,224]",))

    # ncnn inference
    b = module.test_inference()

    return torch.allclose(a, b, 1e-2, 1e-2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
