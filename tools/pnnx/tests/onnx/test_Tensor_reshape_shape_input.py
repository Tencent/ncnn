# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import subprocess

import onnx
from onnx import TensorProto, helper
import torch


def test():
    # Keep shape as a genuine tensor input rather than a folded initializer or
    # a Python ListConstruct. This exercises the shared Python writer contract.
    for name, data_shape, shape_size, output_shape, samples in [
        ('tensor', [49], 2, ['a', 'b'], [[7, 7], [1, 49], [49, 1]]),
        ('scalar', [1], 0, [], [[]]),
    ]:
        prefix = 'test_Tensor_reshape_shape_input_' + name
        graph = helper.make_graph(
            [helper.make_node('Reshape', ['x', 'shape'], ['y'])], prefix,
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, data_shape),
             helper.make_tensor_value_info('shape', TensorProto.INT64, [shape_size])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, output_shape)])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)], ir_version=9)
        onnx.save(model, prefix + '.onnx')
        subprocess.run(['../../src/pnnx', prefix + '.onnx', 'fp16=0'], check=True)
        converted = importlib.import_module(prefix + '_pnnx').Model().eval()
        x = torch.arange(data_shape[0], dtype=torch.float32)
        for shape in samples:
            actual = converted(x, torch.tensor(shape, dtype=torch.int64))
            expected = x.reshape(tuple(shape))
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                return False
    return True


if __name__ == '__main__':
    assert test()
