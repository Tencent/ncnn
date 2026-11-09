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
        if version.parse(torch.__version__) >= version.parse('2.11') and os.environ.get('PNNX_BUILD_PT2', '1') != '0':
            # export exported program
            if dynamic:
                dynamic_shapes = {
                    'x': {2: torch.export.Dim('x_length', min=7, max=9)},
                    'y': {2: torch.export.Dim('y_height', min=7, max=9), 3: torch.export.Dim('y_width', min=7, max=11)},
                    'z': {2: torch.export.Dim('z_height', min=2, max=3), 3: torch.export.Dim('z_width', min=2, max=3)},
                }
                try:
                    ep = torch.export.export(net, inputs, dynamic_shapes=dynamic_shapes)
                except torch._dynamo.exc.UserError as e:
                    # scale_factor=0.2 crosses output width 1 -> 2 for these inputs;
                    # export specializes the align_corners=True branch to width 1.
                    if 'Constraints violated (y_width)' not in str(e) or 'math.trunc(0.2*' not in str(e):
                        raise
                    print('skip pt2 dynamic export: torch', torch.__version__,
                          'input shapes', [tuple(x.shape) for x in inputs], [tuple(x.shape) for x in inputs2], e)
                    continue
            else:
                ep = torch.export.export(net, inputs)
            torch.export.save(ep, name + '_export.pt2')

            # exported program to pnnx
            command = '../src/pnnx ' + name + '_export.pt2 fp16=0'
            if dynamic:
                command += ' inputshape=[1,3,7],[1,3,7,7],[1,3,2,3] inputshape2=[1,3,9],[1,3,9,11],[1,3,3,2]'
            os.system(command)

            # pnnx inference
            exported = importlib.import_module(name + '_export_pnnx')
            if dynamic:
                exported_model = exported.Model()
                exported_model.eval()
            for data in [inputs, inputs2] if dynamic else [inputs]:
                a = net(*data)
                if dynamic:
                    c = exported_model(*data)
                else:
                    c = exported.test_inference()
                if len(a) != len(c):
                    print('export dynamic', dynamic, 'input shapes', [tuple(x.shape) for x in data], 'output count', len(a), len(c))
                    return False
                for i, (a0, c0) in enumerate(zip(a, c)):
                    if a0.shape != c0.shape:
                        print('export dynamic', dynamic, 'input shapes', [tuple(x.shape) for x in data], 'output', i, 'shape mismatch', tuple(a0.shape), tuple(c0.shape))
                        return False
                    if not torch.allclose(a0, c0, 1e-4, 1e-4):
                        print('export dynamic', dynamic, 'input shapes', [tuple(x.shape) for x in data], 'output', i, 'shapes', tuple(a0.shape), tuple(c0.shape), 'max diff', (a0 - c0).abs().max().item())
                        return False
    return True

if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
