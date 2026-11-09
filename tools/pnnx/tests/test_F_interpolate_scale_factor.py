# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.up = nn.Upsample(scale_factor=1.5, mode='bilinear', align_corners=False)
        self.up_recompute = None
        self.up_recompute_nearest = None
        if version.parse(torch.__version__) >= version.parse('1.11'):
            self.up_recompute = nn.Upsample(scale_factor=1.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            self.up_recompute_nearest = nn.Upsample(scale_factor=(1.5, 0.8), mode='nearest', recompute_scale_factor=True)

    def forward(self, x, y, z):
        outputs = [
            F.interpolate(x, scale_factor=1.5, mode='linear', align_corners=False),
            F.interpolate(x, scale_factor=0.8, mode='linear', align_corners=False),
            F.interpolate(x, scale_factor=1.5, mode='linear', align_corners=False, recompute_scale_factor=True),
            F.interpolate(x, scale_factor=1.1, mode='linear', align_corners=False),
            F.interpolate(y, scale_factor=1.5, mode='nearest', recompute_scale_factor=True),
            self.up(y),
        ]
        for mode in ['bilinear', 'bicubic']:
            outputs += [
                F.interpolate(y, scale_factor=1.5, mode=mode, align_corners=False),
                F.interpolate(y, scale_factor=0.8, mode=mode, align_corners=False),
                F.interpolate(y, scale_factor=1.5, mode=mode, align_corners=False, recompute_scale_factor=True),
                F.interpolate(y, scale_factor=0.8, mode=mode, align_corners=False, recompute_scale_factor=True),
                F.interpolate(y, scale_factor=(1.5, 0.8), mode=mode, align_corners=False, recompute_scale_factor=True),
                F.interpolate(y, scale_factor=(0.8, 1.5), mode=mode, align_corners=False, recompute_scale_factor=True),
                F.interpolate(y, scale_factor=1.1, mode=mode, align_corners=False),
                F.interpolate(y, scale_factor=(1.1, 1.5), mode=mode, align_corners=False),
                F.interpolate(y, scale_factor=(1.5, 1.1), mode=mode, align_corners=False),
                F.interpolate(y, size=(10, 10), mode=mode, align_corners=False),
                F.interpolate(y, scale_factor=1.5, mode=mode, align_corners=True),
                F.interpolate(y, scale_factor=0.2, mode=mode, align_corners=True),
                F.interpolate(z, scale_factor=1.1, mode=mode, align_corners=False),
                F.interpolate(z, scale_factor=1.5, mode=mode, align_corners=False),
            ]
        if self.up_recompute is not None:
            outputs.append(self.up_recompute(y))
            outputs.append(self.up_recompute_nearest(y))
        return tuple(outputs)

def test():
    net = Model()
    net.eval()
    torch.manual_seed(0)
    inputs = (torch.rand(1, 3, 7), torch.rand(1, 3, 7, 7), torch.rand(1, 3, 2, 3))
    inputs2 = (torch.rand(1, 3, 9), torch.rand(1, 3, 9, 11), torch.rand(1, 3, 3, 2))

    mod = torch.jit.trace(net, inputs)

    import os
    for dynamic in [False, True]:
        name = 'test_F_interpolate_scale_factor_dynamic' if dynamic else 'test_F_interpolate_scale_factor_static'
        mod.save(name + '.pt')

        command = '../src/pnnx ' + name + '.pt inputshape=[1,3,7],[1,3,7,7],[1,3,2,3]'
        if dynamic:
            command += ' inputshape2=[1,3,9],[1,3,9,11],[1,3,3,2]'
        if os.system(command) != 0:
            return False

        import importlib
        converted = importlib.import_module(name + '_pnnx').Model()
        converted.eval()
        for data in [inputs, inputs2] if dynamic else [inputs]:
            a = net(*data)
            b = converted(*data)
            if len(a) != len(b):
                print('dynamic', dynamic, 'input shapes', [tuple(x.shape) for x in data], 'output count', len(a), len(b))
                return False
            for i, (a0, b0) in enumerate(zip(a, b)):
                if a0.shape != b0.shape:
                    print('dynamic', dynamic, 'input shapes', [tuple(x.shape) for x in data], 'output', i, 'shape mismatch', tuple(a0.shape), tuple(b0.shape))
                    return False
                if not torch.allclose(a0, b0, 1e-4, 1e-4):
                    print('dynamic', dynamic, 'input shapes', [tuple(x.shape) for x in data], 'output', i, 'shapes', tuple(a0.shape), tuple(b0.shape), 'max diff', (a0 - b0).abs().max().item())
                    return False
    return True

if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
