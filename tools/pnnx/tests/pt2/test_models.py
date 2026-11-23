# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os

import torch
from packaging import version


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return self.fc(torch.flatten(x, 1))


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(8, 8)
        self.k = torch.nn.Linear(8, 8)
        self.v = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        a = torch.softmax(self.q(x) @ self.k(x).transpose(-2, -1) * 0.3535533905932738, -1)
        return self.norm(x + a @ self.v(x))


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.set_num_threads(1)
    torch.manual_seed(0)
    for name, model, shape in [
        ('linear', torch.nn.Linear(8, 4), (1, 8)),
        ('mlp', torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)), (1, 8)),
        ('cnn', CNN(), (1, 3, 8, 8)),
        ('attention', Attention(), (1, 4, 8)),
    ]:
        model.eval()
        torch.manual_seed(0)
        x = torch.rand(*shape)
        with torch.no_grad():
            expected = model(x)

        prefix = 'test_pt2_models_' + name
        program = torch.export.export(model, (x,))
        torch.export.save(program, prefix + '.pt2')
        os.system('../../src/pnnx ' + prefix + '.pt2 fp16=0')

        actual = importlib.import_module(prefix + '_pnnx').test_inference()
        if actual.shape != expected.shape or not torch.allclose(actual, expected, 2e-5, 2e-5):
            return False

        actual = importlib.import_module(prefix + '_ncnn').test_inference()
        if actual.shape != expected.shape or not torch.allclose(actual, expected, 1e-3, 1e-4):
            return False
    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
