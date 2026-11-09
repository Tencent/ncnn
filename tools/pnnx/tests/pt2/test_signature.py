# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess

import torch
from packaging import version
from common import rewrite_archive


class Nested(torch.nn.Module):
    def forward(self, x, more, scale=3):
        y = x + more['right'][0] * scale
        return {'values': (y, x), 'again': [y]}


class Direct(torch.nn.Module):
    def forward(self, x):
        return x, x


class BadOutput(torch.nn.Module):
    def forward(self, x):
        return x, None


def check_reject(path, message, options=()):
    result = subprocess.run(['../../src/pnnx', path, 'fp16=0', *options], capture_output=True, text=True)
    if result.returncode == 0 or message not in result.stderr:
        print(result.stderr)
        return False
    return True


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    net = Nested().eval()
    torch.manual_seed(0)
    x = torch.rand(2, 3)
    y = torch.rand(2, 3)
    a = net(x, {'right': [y]}, 3)
    ep = torch.export.export(net, (x, {'right': [y]}, 3))
    torch.export.save(ep, 'test_pt2_signature_nested.pt2')
    os.system('../../src/pnnx test_pt2_signature_nested.pt2 fp16=0')

    import test_pt2_signature_nested_pnnx
    b = test_pt2_signature_nested_pnnx.test_inference()
    a = (*a['values'], *a['again'])
    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    net = Direct().eval()
    torch.manual_seed(0)
    x = torch.rand(2, 3)
    a = net(x)
    ep = torch.export.export(net, (x,))
    path = 'test_pt2_signature_direct.pt2'
    torch.export.save(ep, path)
    os.system('../../src/pnnx test_pt2_signature_direct.pt2 fp16=0')

    import test_pt2_signature_direct_pnnx
    b = test_pt2_signature_direct_pnnx.test_inference()
    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    ep = torch.export.export(BadOutput(), (x,))
    torch.export.save(ep, 'test_pt2_signature_none_output.pt2')
    if not check_reject('test_pt2_signature_none_output.pt2', 'non-Tensor user output'):
        return False
    if not check_reject(path, 'input static dimension mismatch', ['inputshape=[3,3]']):
        return False
    if not check_reject(path, 'input dtype/rank mismatch', ['inputshape=[2,3]i64']):
        return False

    rewrite_archive(path, 'test_pt2_signature_bad_ref.pt2', lambda p, e: p['graph_module']['graph']['outputs'][0]['as_tensor'].update(name='missing'))
    if not check_reject('test_pt2_signature_bad_ref.pt2', 'output signature mismatch'):
        return False
    rewrite_archive(path, 'test_pt2_signature_bad_count.pt2', lambda p, e: p['graph_module']['signature']['input_specs'].clear())
    return check_reject('test_pt2_signature_bad_count.pt2', 'count mismatch')


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
