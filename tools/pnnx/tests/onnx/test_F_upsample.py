# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        if version.parse(torch.__version__) < version.parse('1.12'):
            x0 = F.upsample(x, size=60)
            x0 = F.upsample(x0, scale_factor=2, mode='nearest')
            x1 = F.upsample(x, size=(40), mode='nearest')
            x1 = F.upsample(x1, scale_factor=(4), mode='nearest')
            x2 = F.upsample(x, size=60, mode='linear')
            x2 = F.upsample(x2, scale_factor=2, mode='linear')

            y0 = F.upsample(y, size=60)
            y0 = F.upsample(y0, scale_factor=2, mode='nearest')
            y1 = F.upsample(y, size=(40,40), mode='nearest')
            y1 = F.upsample(y1, scale_factor=(4,4), mode='nearest')
            y2 = F.upsample(y, size=(60,50), mode='nearest')
            y2 = F.upsample(y2, scale_factor=(2,3), mode='nearest')
            y3 = F.upsample(y, size=60, mode='bilinear')
            y3 = F.upsample(y3, scale_factor=2, mode='bilinear')

            z0 = F.upsample(z, size=60)
            z0 = F.upsample(z0, scale_factor=2, mode='nearest')
            z1 = F.upsample(z, size=(40,40,40), mode='nearest')
            z1 = F.upsample(z1, scale_factor=(4,4,4), mode='nearest')
            z2 = F.upsample(z, size=(60,50,40), mode='nearest')
            z2 = F.upsample(z2, scale_factor=(2,3,4), mode='nearest')
            z3 = F.upsample(z, size=60, mode='trilinear')
            z3 = F.upsample(z3, scale_factor=2, mode='trilinear')

            w = F.upsample(w, scale_factor=(1.499,1.499), mode='nearest')

            return x0, x1, x2, y0, y1, y2, y3, z0, z1, z2, z3, w
        else:
            x = F.upsample(x, size=16)
            x = F.upsample(x, scale_factor=2, mode='nearest')
            x = F.upsample(x, size=(20), mode='nearest')
            x = F.upsample(x, scale_factor=(4), mode='nearest')
            x = F.upsample(x, size=16, mode='linear')
            x = F.upsample(x, scale_factor=2, mode='linear')
            x = F.upsample(x, size=(24), mode='linear', align_corners=True)
            x = F.upsample(x, scale_factor=(3), mode='linear', align_corners=True)

            y = F.upsample(y, size=16)
            y = F.upsample(y, scale_factor=2, mode='nearest')
            y = F.upsample(y, size=(20,20), mode='nearest')
            y = F.upsample(y, scale_factor=(4,4), mode='nearest')
            y = F.upsample(y, size=(16,24), mode='nearest')
            y = F.upsample(y, scale_factor=(2,3), mode='nearest')
            y = F.upsample(y, size=16, mode='bilinear')
            y = F.upsample(y, scale_factor=2, mode='bilinear')
            y = F.upsample(y, size=(20,20), mode='bilinear', align_corners=False)
            y = F.upsample(y, scale_factor=(4,4), mode='bilinear', align_corners=False)
            y = F.upsample(y, size=(16,24), mode='bilinear', align_corners=True)
            y = F.upsample(y, scale_factor=(2,3), mode='bilinear', align_corners=True)
            y = F.upsample(y, size=16, mode='bicubic')
            y = F.upsample(y, scale_factor=2, mode='bicubic')
            y = F.upsample(y, size=(20,20), mode='bicubic', align_corners=False)
            y = F.upsample(y, scale_factor=(4,4), mode='bicubic', align_corners=False)
            y = F.upsample(y, size=(16,24), mode='bicubic', align_corners=True)
            y = F.upsample(y, scale_factor=(2,3), mode='bicubic', align_corners=True)

            z = F.upsample(z, size=16)
            z = F.upsample(z, scale_factor=2, mode='nearest')
            z = F.upsample(z, size=(20,20,20), mode='nearest')
            z = F.upsample(z, scale_factor=(4,4,4), mode='nearest')
            z = F.upsample(z, size=(16,24,20), mode='nearest')
            z = F.upsample(z, scale_factor=(2,3,4), mode='nearest')
            z = F.upsample(z, size=16, mode='trilinear')
            z = F.upsample(z, scale_factor=2, mode='trilinear')
            z = F.upsample(z, size=(20,20,20), mode='trilinear', align_corners=False)
            z = F.upsample(z, scale_factor=(4,4,4), mode='trilinear', align_corners=False)
            z = F.upsample(z, size=(16,24,20), mode='trilinear', align_corners=True)
            z = F.upsample(z, scale_factor=(2,3,4), mode='trilinear', align_corners=True)

            w = F.upsample(w, scale_factor=(1.499,1.499), mode='nearest')

            return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 32)
    y = torch.rand(1, 3, 32, 32)
    z = torch.rand(1, 3, 32, 32, 32)
    w = torch.rand(1, 8, 12, 12)

    a = net(x, y, z, w)

    # export onnx
    torch.onnx.export(net, (x, y, z, w), "test_F_upsample.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_upsample.onnx inputshape=[1,3,32],[1,3,32,32],[1,3,32,32,32],[1,8,12,12]")

    # pnnx inference
    import test_F_upsample_pnnx
    b = test_F_upsample_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
