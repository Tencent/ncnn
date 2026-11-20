# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
import torch
import ncnn


class Model(torch.nn.Module):
    def __init__(self, sections, dim):
        super().__init__()
        self.sections = sections
        self.dim = dim

    def forward(self, x):
        return torch.tensor_split(x, self.sections, dim=self.dim)


def test_case(sections, dim, shapes, dynamic, index):
    model = Model(sections, dim).eval()
    name = 'test_torch_tensor_split_uneven_' + str(index)
    x = torch.arange(np.prod(shapes[0]), dtype=torch.float32).reshape(shapes[0])
    torch.jit.trace(model, x).save(name + '.pt')
    command = '../../src/pnnx ' + name + '.pt inputshape=[' + ','.join(map(str, shapes[0])) + '] fp16=0'
    if dynamic:
        command += ' inputshape2=[' + ','.join(map(str, shapes[1])) + ']'
    if os.system(command) != 0:
        return False

    for shape in shapes:
        x = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        expected = model(x)
        with ncnn.Net() as net:
            net.opt.use_vulkan_compute = False
            if net.load_param(name + '.ncnn.param') != 0 or net.load_model(name + '.ncnn.bin') != 0:
                return False
            with net.create_extractor() as ex:
                if ex.input('in0', ncnn.Mat(x.numpy()).clone()) != 0:
                    return False
                for i, value in enumerate(expected):
                    ret, output = ex.extract('out' + str(i))
                    if ret != 0 or not np.array_equal(output.numpy(), value.numpy()):
                        print('uneven tensor_split mismatch', sections, dim, shape, i)
                        return False
    return True


def test():
    cases = [
        (3, 0, [(10,)], False),
        (3, 0, [(12,)], False),
        (4, 0, [(9,)], False),
        (3, 1, [(2, 10)], False),
        (3, -1, [(2, 10)], False),
        (3, -1, [(2, 10), (3, 11), (4, 8)], True),
        (4, 1, [(2, 10, 3), (3, 11, 5), (4, 13, 2)], True),
    ]
    for index, (sections, dim, shapes, dynamic) in enumerate(cases):
        if not test_case(sections, dim, shapes, dynamic, index):
            return False
    return True


if __name__ == '__main__':
    exit(0 if test() else 1)
