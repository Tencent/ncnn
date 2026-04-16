# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
if sys.version_info < (3, 9):
    sys.exit(0)

import torch
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper

def build_model():
    X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 1, 1, 8])
    Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 8])

    axes_np = np.array([[2], [3]], dtype=np.int64)
    axes_init = numpy_helper.from_array(axes_np, name='axes')
    axes_const = helper.make_node('Constant', inputs=[], outputs=['axes'], value=axes_init)

    squeeze_node = helper.make_node('Squeeze', inputs=['x', 'axes'], outputs=['y'])

    graph = helper.make_graph(
        [axes_const, squeeze_node],
        'test_fuse_2d_constant',
        [X],
        [Y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    return model

def test():
    model = build_model()
    onnx.save(model, 'test_onnx_fuse_2d_constant.onnx')

    torch.manual_seed(0)
    x = torch.rand(1, 3, 1, 1, 8)
    a = (x.squeeze(2).squeeze(2),)

    import os
    os.system('../../src/pnnx test_onnx_fuse_2d_constant.onnx inputshape=[1,3,1,1,8] fp16=0')

    import test_onnx_fuse_2d_constant_pnnx
    b = test_onnx_fuse_2d_constant_pnnx.test_inference()

    import test_onnx_fuse_2d_constant_ncnn
    c = test_onnx_fuse_2d_constant_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-3, 1e-3) or not torch.allclose(a0, c0, 1e-3, 1e-3):
            return False
    return True

if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
