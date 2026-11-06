# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import pathlib
import subprocess
import sys

import torch
from common import rewrite_archive


def test():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'python'))
    convert_module = importlib.import_module('pnnx.utils.convert')
    convert_module.EXEC_PATH = '../../src/pnnx'

    net = torch.nn.Linear(4, 3)
    net.eval()
    torch.manual_seed(0)
    x = torch.rand(1, 4)
    a = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, 'test_pt2_api.pt2')
    converted = convert_module.convert('test_pt2_api.pt2', fp16=False)
    b = converted(x)
    if not torch.allclose(a, b, 2e-5, 2e-5):
        return False

    rewrite_archive('test_pt2_api.pt2', 'test_pt2_api_bad.pt2', lambda p, e: p['opset_version'].update(aten=999))
    try:
        convert_module.convert('test_pt2_api_bad.pt2')
    except subprocess.CalledProcessError:
        pass
    else:
        return False

    # Reading valid weights must not clear an earlier graph metadata error.
    rewrite_archive('test_pt2_api.pt2', 'test_pt2_api_bad_guards.pt2', lambda p, e: p.update(guards_code=None))
    try:
        convert_module.convert('test_pt2_api_bad_guards.pt2')
    except subprocess.CalledProcessError:
        return True
    return False


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
