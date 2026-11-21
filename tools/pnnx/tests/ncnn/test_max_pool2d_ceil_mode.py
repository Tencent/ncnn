# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import ncnn
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2, 1, ceil_mode=True)

    def forward(self, x):
        outputs = [self.maxpool(x), F.max_pool2d(x, 2, 2, 1, ceil_mode=True)]
        for kernel, stride, padding, ceil_mode in [((2, 3), (3, 4), 1, True),
                                                   (2, 4, 0, True),
                                                   (2, 2, 1, False),
                                                   (3, 2, 1, True)]:
            outputs.append(F.max_pool2d(x, kernel, stride, padding, ceil_mode=ceil_mode))
        return tuple(outputs)

def test():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = Model().eval()
    inputs = [torch.randn(shape) for shape in [(1, 1, 3, 3), (1, 4, 3, 4), (2, 8, 4, 3), (2, 16, 4, 4)]]
    traced = torch.jit.trace(model, inputs[0])
    matched = 0
    total = 0
    for dynamic in [False, True]:
        name = 'test_max_pool2d_ceil_mode_dynamic' if dynamic else 'test_max_pool2d_ceil_mode_static'
        traced.save(name + '.pt')
        command = '../../src/pnnx ' + name + '.pt inputshape=[1,1,3,3]'
        if dynamic:
            command += ' inputshape2=[2,16,4,4]'
        if os.system(command) != 0:
            return False
        with ncnn.Net() as converted:
            converted.opt.num_threads = 1
            converted.opt.use_fp16_packed = False
            converted.opt.use_fp16_storage = False
            converted.opt.use_fp16_arithmetic = False
            converted.opt.use_bf16_storage = False
            if converted.load_param(name + '.ncnn.param') != 0 or converted.load_model(name + '.ncnn.bin') != 0:
                return False
            for x in inputs if dynamic else inputs[:1]:
                expected = model(x)
                with converted.create_extractor() as ex:
                    if ex.input('in0', ncnn.Mat(x.numpy(), batch_index=0).clone()) != 0:
                        return False
                    for i, a in enumerate(expected):
                        ret, out = ex.extract('out%d' % i)
                        if ret != 0:
                            return False
                        b = torch.from_numpy(out.numpy(batch_index=0))
                        total += 1
                        if a.shape == b.shape and torch.allclose(a, b, 1e-4, 1e-4):
                            matched += 1
                        else:
                            print('input', tuple(x.shape), 'output', i, 'mismatch', tuple(a.shape), tuple(b.shape))
    print('max_pool2d ceil-mode:', matched, '/', total, 'outputs match')
    return matched == total

if __name__ == '__main__':
    exit(0 if test() else 1)
