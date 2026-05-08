# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
if sys.version_info < (3, 9):
    sys.exit(0)

import torch
import onnx
import onnxruntime as ort
from onnxscript import FLOAT, script
from onnxscript import opset19 as op

@script()
def Model(x: FLOAT["N",12,14,15]):

    bn_scale = op.RandomNormal(seed=0.0, shape=[12])
    bn_bias = op.RandomNormal(seed=1.0, shape=[12])
    bn_mean = op.RandomNormal(seed=2.0, shape=[12])
    bn_var = op.RandomNormal(seed=3.0, mean=5.0, shape=[12])

    in_scale = op.RandomNormal(seed=6.0, shape=[12])
    in_bias = op.RandomNormal(seed=7.0, shape=[12])

    ln_scale = op.RandomNormal(seed=8.0, shape=[15])
    ln_bias = op.RandomNormal(seed=9.0, shape=[15])

    ln_scale2 = op.RandomNormal(seed=10.0, shape=[14,15])
    ln_bias2 = op.RandomNormal(seed=11.0, shape=[14,15])

    return (op.BatchNormalization(x, bn_scale, bn_bias, bn_mean, bn_var),
        op.InstanceNormalization(x, in_scale, in_bias, epsilon=0.1),
        op.InstanceNormalization(x, in_scale, in_bias),
        op.LayerNormalization(x, ln_scale, ln_bias, epsilon=0.1),
        op.LayerNormalization(x, ln_scale2, ln_bias2, axis=-2),
        op.LRN(x, alpha=0.1, beta=0.5, bias=0.75, size=5),
        op.LRN(x, size=3),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_normalize_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(1, 12, 14, 15)

    # ort inference
    sess = ort.InferenceSession("test_onnx_normalize_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_normalize_ops.onnx inputshape=[1,12,14,15]")

    # pnnx inference
    import test_onnx_normalize_ops_pnnx
    b = test_onnx_normalize_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_normalize_ops_ncnn
    c = test_onnx_normalize_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-3, 1e-3) or not torch.allclose(a0, c0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
