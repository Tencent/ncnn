# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import onnx
import onnxruntime as ort
from onnxscript import FLOAT, script
from onnxscript import opset19 as op

@script()
def Model(x: FLOAT[1,12,13,14]):
    return (
        # op.DepthToSpace(x, blocksize=2),
        op.DepthToSpace(x, blocksize=2, mode='CRD'),

        op.Flatten(x),
        # op.Flatten(x, axis=-1),

        # op.Pad(x, pads=[0, 0, 0, 0, 1, 1, 1, 1], mode='constant', constant_value=0.0),
        # op.Pad(x, pads=[0, 0, 1, 1, 0, 0, 1, 1], mode='reflect'),
        # op.Pad(x, pads=[1, 0, 0, 2, 0, 0, 3, 3], mode='edge'),

        # op.Concat(x),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_layout_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(1, 12, 13, 14)

    # ort inference
    sess = ort.InferenceSession("test_onnx_layout_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy()}))

    # onnx to ncnn
    import os
    os.system("../../src/pnnx test_onnx_layout_ops.onnx")

    # ncnn inference
    import test_onnx_layout_ops_ncnn
    b = test_onnx_layout_ops_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
