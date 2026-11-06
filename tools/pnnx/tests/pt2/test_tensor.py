# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import subprocess

import torch

from common import rewrite_archive


class Views(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        self.register_buffer('a', base[:, 1:4].t())
        self.register_buffer('b', base[:, 2:5].t(), persistent=False)
        self.register_buffer('z', torch.tensor(0.25).expand(3, 4))
        self.register_buffer('scalar', torch.tensor(2.0))
        self.register_buffer('empty', torch.empty(0, 4))
        self.p = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))

    def forward(self, x):
        return x + self.a + self.b + self.z + self.p, self.scalar, self.empty, torch.tensor([2., 3.])


class Typed(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer('a', torch.arange(6).reshape(2, 3).to(dtype))

    def forward(self, x):
        return x + self.a, self.a


class ScalarTensor(torch.nn.Module):
    def __init__(self, value, dtype):
        super().__init__()
        self.register_buffer('value', torch.tensor([value], dtype=dtype))

    def forward(self, x):
        return x + self.value


class IntegerBounds(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.tensor([-2**63], dtype=torch.int64))
        self.register_buffer('b', torch.tensor([-2**31 - 1], dtype=torch.int64))
        self.register_buffer('c', torch.tensor([-2**31], dtype=torch.int64))
        self.register_buffer('d', torch.tensor([2**31 - 1], dtype=torch.int64))
        self.register_buffer('e', torch.tensor([2**31], dtype=torch.int64))
        self.register_buffer('f', torch.tensor([2**63 - 1], dtype=torch.int64))

    def forward(self, x):
        return x + self.a, x + self.b, x + self.c, x + self.d, x + self.e, x + self.f


class IntegerConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.tensor([2**31 - 1], dtype=torch.int32))
        self.register_buffer('b', torch.tensor([1], dtype=torch.int32))

    def forward(self, x):
        return x + (self.a + self.b)


class BFloatConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('bias', torch.tensor([0.5, -0.25], dtype=torch.bfloat16))

    def forward(self, x, w):
        return torch.nn.functional.conv1d(x, w, self.bias)


def test_views():
    model = Views().eval()
    torch.manual_seed(0)
    x = torch.rand(3, 4)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_tensor_views.pt2')
    os.system('../../src/pnnx test_pt2_tensor_views.pt2 fp16=0')

    import test_pt2_tensor_views_pnnx
    actual = test_pt2_tensor_views_pnnx.test_inference()
    if len(actual) != len(expected):
        print('views output count mismatch')
        return False
    for a, b in zip(expected, actual):
        if a.dtype != b.dtype or a.shape != b.shape:
            print('views output dtype or shape mismatch')
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            print('views output values differ')
            return False
    return True


def test_typed_floating():
    for dtype in (torch.float16, torch.bfloat16):
        model = Typed(dtype).eval()
        x = torch.ones(2, 3, dtype=dtype)
        program = torch.export.export(model, (x,))
        prefix = 'test_pt2_tensor_typed_' + str(dtype).split('.')[1]
        torch.export.save(program, prefix + '.pt2')
        os.system('../../src/pnnx ' + prefix + '.pt2 fp16=0')

    import test_pt2_tensor_typed_float16_pnnx
    import test_pt2_tensor_typed_bfloat16_pnnx
    for dtype, module, tolerance in (
        (torch.float16, test_pt2_tensor_typed_float16_pnnx, 3e-3),
        (torch.bfloat16, test_pt2_tensor_typed_bfloat16_pnnx, 2e-2),
    ):
        model = Typed(dtype).eval()
        x = torch.ones(2, 3, dtype=dtype)
        torch.manual_seed(0)
        sample = torch.rand(2, 3, dtype=dtype)
        for actual, expected in (
            (module.Model().eval()(x), model(x)),
            (module.test_inference(), model(sample)),
        ):
            if len(actual) != len(expected):
                print(dtype, 'typed output count mismatch')
                return False
            for a, b in zip(expected, actual):
                if a.dtype != b.dtype or a.shape != b.shape:
                    print(dtype, 'typed output dtype or shape mismatch')
                    return False
                if not torch.allclose(a, b, rtol=tolerance, atol=tolerance):
                    print(dtype, 'typed output values differ')
                    return False
    return True


def test_typed_integer():
    for dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8, torch.bool):
        model = Typed(dtype).eval()
        x = torch.ones(2, 3, dtype=dtype)
        program = torch.export.export(model, (x,))
        prefix = 'test_pt2_tensor_typed_' + str(dtype).split('.')[1]
        torch.export.save(program, prefix + '.pt2')
        os.system('../../src/pnnx ' + prefix + '.pt2 fp16=0')

    import test_pt2_tensor_typed_int32_pnnx
    import test_pt2_tensor_typed_int64_pnnx
    import test_pt2_tensor_typed_int16_pnnx
    import test_pt2_tensor_typed_int8_pnnx
    import test_pt2_tensor_typed_uint8_pnnx
    import test_pt2_tensor_typed_bool_pnnx
    for dtype, module in (
        (torch.int32, test_pt2_tensor_typed_int32_pnnx),
        (torch.int64, test_pt2_tensor_typed_int64_pnnx),
        (torch.int16, test_pt2_tensor_typed_int16_pnnx),
        (torch.int8, test_pt2_tensor_typed_int8_pnnx),
        (torch.uint8, test_pt2_tensor_typed_uint8_pnnx),
        (torch.bool, test_pt2_tensor_typed_bool_pnnx),
    ):
        model = Typed(dtype).eval()
        x = torch.ones(2, 3, dtype=dtype)
        torch.manual_seed(0)
        sample = torch.randint(2 if dtype == torch.bool else 10, (2, 3), dtype=dtype)
        for actual, expected in (
            (module.Model().eval()(x), model(x)),
            (module.test_inference(), model(sample)),
        ):
            if len(actual) != len(expected):
                print(dtype, 'typed output count mismatch')
                return False
            for a, b in zip(expected, actual):
                if a.dtype != b.dtype or a.shape != b.shape:
                    print(dtype, 'typed output dtype or shape mismatch')
                    return False
                if not torch.equal(a, b):
                    print(dtype, 'typed output values differ')
                    return False
    return True


def test_mixed_half():
    model = ScalarTensor(0.25, torch.float32).eval()
    x = torch.ones(2, 3, dtype=torch.float16)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_tensor_mixed_half.pt2')
    os.system('../../src/pnnx test_pt2_tensor_mixed_half.pt2 fp16=0')

    import test_pt2_tensor_mixed_half_pnnx
    actual = test_pt2_tensor_mixed_half_pnnx.Model().eval()(x)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        print('mixed_half output dtype or shape mismatch')
        return False
    if not torch.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        print('mixed_half output values differ')
        return False
    return True


def test_scalar_half():
    model = ScalarTensor(0.25, torch.float16).eval()
    x = torch.ones(2, 3, dtype=torch.float16)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_tensor_scalar_half.pt2')
    os.system('../../src/pnnx test_pt2_tensor_scalar_half.pt2 fp16=0')

    import test_pt2_tensor_scalar_half_pnnx
    actual = test_pt2_tensor_scalar_half_pnnx.Model().eval()(x)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        print('scalar_half output dtype or shape mismatch')
        return False
    if not torch.allclose(actual, expected, rtol=3e-3, atol=3e-3):
        print('scalar_half output values differ')
        return False
    return True


def test_scalar_bfloat16():
    model = ScalarTensor(0.25, torch.bfloat16).eval()
    x = torch.ones(2, 3, dtype=torch.bfloat16)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_tensor_scalar_bfloat16.pt2')
    os.system('../../src/pnnx test_pt2_tensor_scalar_bfloat16.pt2 fp16=0')

    import test_pt2_tensor_scalar_bfloat16_pnnx
    actual = test_pt2_tensor_scalar_bfloat16_pnnx.Model().eval()(x)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        print('scalar_bfloat16 output dtype or shape mismatch')
        return False
    if not torch.allclose(actual, expected, rtol=2e-2, atol=2e-2):
        print('scalar_bfloat16 output values differ')
        return False
    return True


def test_large_integer():
    model = ScalarTensor(2**40 + 1, torch.int64).eval()
    x = torch.ones(2, 3, dtype=torch.int64)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_tensor_large_integer.pt2')
    os.system('../../src/pnnx test_pt2_tensor_large_integer.pt2 fp16=0')

    import test_pt2_tensor_large_integer_pnnx
    actual = test_pt2_tensor_large_integer_pnnx.Model().eval()(x)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        print('large_integer output dtype or shape mismatch')
        return False
    if not torch.equal(actual, expected):
        print('large_integer output values differ')
        return False
    return True


def test_scalar_integer():
    for dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8, torch.bool):
        model = ScalarTensor(3, dtype).eval()
        x = torch.ones(2, 3, dtype=dtype)
        name = 'test_pt2_tensor_scalar_' + str(dtype).split('.')[-1]
        program = torch.export.export(model, (x,))
        torch.export.save(program, name + '.pt2')
        os.system('../../src/pnnx ' + name + '.pt2 fp16=0')

    import test_pt2_tensor_scalar_int32_pnnx
    import test_pt2_tensor_scalar_int64_pnnx
    import test_pt2_tensor_scalar_int16_pnnx
    import test_pt2_tensor_scalar_int8_pnnx
    import test_pt2_tensor_scalar_uint8_pnnx
    import test_pt2_tensor_scalar_bool_pnnx
    for dtype, module in (
        (torch.int32, test_pt2_tensor_scalar_int32_pnnx),
        (torch.int64, test_pt2_tensor_scalar_int64_pnnx),
        (torch.int16, test_pt2_tensor_scalar_int16_pnnx),
        (torch.int8, test_pt2_tensor_scalar_int8_pnnx),
        (torch.uint8, test_pt2_tensor_scalar_uint8_pnnx),
        (torch.bool, test_pt2_tensor_scalar_bool_pnnx),
    ):
        x = torch.ones(2, 3, dtype=dtype)
        expected = ScalarTensor(3, dtype).eval()(x)
        actual = module.Model().eval()(x)
        if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual, expected):
            return False
    return True


def test_integer_bounds():
    model = IntegerBounds().eval()
    x = torch.zeros(2, 3, dtype=torch.int64)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_tensor_integer_bounds.pt2')
    os.system('../../src/pnnx test_pt2_tensor_integer_bounds.pt2 fp16=0')

    import test_pt2_tensor_integer_bounds_pnnx
    actual = test_pt2_tensor_integer_bounds_pnnx.Model().eval()(x)
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return False
    return True


def test_integer_constants():
    model = IntegerConstants().eval()
    x = torch.ones(2, 3, dtype=torch.int32)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_tensor_integer_constants.pt2')
    os.system('../../src/pnnx test_pt2_tensor_integer_constants.pt2 fp16=0')

    import test_pt2_tensor_integer_constants_pnnx
    actual = test_pt2_tensor_integer_constants_pnnx.Model().eval()(x)
    return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)


def test_scalar_broadcast():
    model = ScalarTensor(3, torch.int32).eval()
    x = torch.tensor(2, dtype=torch.int32)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_tensor_scalar_broadcast.pt2')
    os.system('../../src/pnnx test_pt2_tensor_scalar_broadcast.pt2 fp16=0')

    import test_pt2_tensor_scalar_broadcast_pnnx
    actual = test_pt2_tensor_scalar_broadcast_pnnx.Model().eval()(x)
    return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)


def test_bfloat_conv_bias():
    model = BFloatConv().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 2, 7, dtype=torch.bfloat16)
    w = torch.rand(2, 2, 3, dtype=torch.bfloat16)
    expected = model(x, w)
    program = torch.export.export(model, (x, w))
    torch.export.save(program, 'test_pt2_tensor_bfloat_conv_bias.pt2')
    os.system('../../src/pnnx test_pt2_tensor_bfloat_conv_bias.pt2 fp16=0')

    import test_pt2_tensor_bfloat_conv_bias_pnnx
    actual = test_pt2_tensor_bfloat_conv_bias_pnnx.test_inference()
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        print('bfloat_conv_bias output dtype or shape mismatch')
        return False
    if not torch.allclose(actual, expected, rtol=2e-2, atol=2e-2):
        print('bfloat_conv_bias output values differ')
        return False
    return True


def test_linear():
    expected = {}
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3).to(dtype).eval()
        torch.manual_seed(0)
        x = torch.rand(1, 4, dtype=dtype)
        expected[dtype] = model(x)
        program = torch.export.export(model, (x,))
        prefix = 'test_pt2_tensor_linear_' + str(dtype).split('.')[1]
        torch.export.save(program, prefix + '.pt2')
        os.system('../../src/pnnx ' + prefix + '.pt2 fp16=0')

    import test_pt2_tensor_linear_float16_pnnx
    import test_pt2_tensor_linear_bfloat16_pnnx
    for dtype, module, tolerance in (
        (torch.float16, test_pt2_tensor_linear_float16_pnnx, 3e-3),
        (torch.bfloat16, test_pt2_tensor_linear_bfloat16_pnnx, 2e-2),
    ):
        actual = module.test_inference()
        if actual.dtype != expected[dtype].dtype or actual.shape != expected[dtype].shape:
            print(dtype, 'linear output dtype or shape mismatch')
            return False
        if not torch.allclose(actual, expected[dtype], rtol=tolerance, atol=tolerance):
            print(dtype, 'linear output values differ')
            return False
    return True


def corrupt_payload(p, entries, field, value):
    key = next(n for n in entries if n.endswith('_weights_config.json'))
    config = json.loads(entries[key])
    payload = next(iter(config['config'].values()))
    if field in ('dtype', 'storage_offset', 'strides'):
        payload['tensor_meta'][field] = value
    else:
        payload[field] = value
    entries[key] = json.dumps(config).encode()


def test_archive_errors():
    for name, change, message in (
        ('pickle', lambda p, e: corrupt_payload(p, e, 'use_pickle', True), 'pickle'),
        ('escape', lambda p, e: corrupt_payload(p, e, 'path_name', '../escape'), 'path'),
        ('dtype', lambda p, e: corrupt_payload(p, e, 'dtype', 999), 'dtype'),
        ('storage_offset', lambda p, e: corrupt_payload(p, e, 'storage_offset', {'as_int': 2**40}), 'exceeds storage'),
        ('storage_stride', lambda p, e: corrupt_payload(p, e, 'strides', [{'as_int': 2**63 - 1}]), 'address overflow'),
        ('byteorder', lambda p, e: e.update({next(n for n in e if n.endswith('/byteorder')): b'big'}), 'little-endian'),
    ):
        path = 'test_pt2_tensor_' + name + '.pt2'
        rewrite_archive('test_pt2_tensor_views.pt2', path, change)
        result = subprocess.run(['../../src/pnnx', path, 'fp16=0'], capture_output=True, text=True, timeout=90)
        if result.returncode == 0 or message not in result.stderr:
            print(name, result.stderr)
            return False
    return True


@torch.no_grad()
def test():
    torch.set_num_threads(1)
    return (
        test_views()
        and test_typed_floating()
        and test_typed_integer()
        and test_mixed_half()
        and test_scalar_half()
        and test_scalar_bfloat16()
        and test_large_integer()
        and test_scalar_integer()
        and test_integer_bounds()
        and test_integer_constants()
        and test_scalar_broadcast()
        and test_bfloat_conv_bias()
        and test_linear()
        and test_archive_errors()
    )


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
