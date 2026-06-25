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
def Model(x: FLOAT["N",12,"H","W"]):

    a = op.Reshape(x, shape=[1,-1,1,1])
    b = op.Reshape(x, shape=[1,-1])

    a0, a1 = op.Split(x, axis=1, num_outputs=2)
    b0, b1, b2, b3, b4, b5, b6 = op.Split(x, axis=-1, num_outputs=7)

    return (
        # op.DepthToSpace(x, blocksize=2),
        op.DepthToSpace(x, blocksize=2, mode='CRD'),

        # op.SpaceToDepth(x, blocksize=2),

        op.Flatten(x),
        op.Flatten(x, axis=-2),

        op.Pad(x, pads=[0, 0, 0, 0, 0, 1, 1, 1], mode='constant', constant_value=0.0),
        op.Pad(x, pads=[0, 0, 1, 1, 0, 0, 1, 1], mode='reflect'),
        op.Pad(x, pads=[0, 1, 0, 2, 0, 0, 3, 3], mode='edge'),

        op.Reshape(x, shape=[0,2,-1,0]),

        op.Squeeze(a),
        op.Squeeze(a, axes=[-1]),

        op.Unsqueeze(b, axes=[2]),
        op.Unsqueeze(b, axes=[-1,1]),
        op.Unsqueeze(b, axes=[2,3]),
        op.Unsqueeze(b, axes=[-2,-1]),

        op.Concat(x, x, axis=1),
        op.Concat(a, a, a, axis=-1),

        op.Slice(x, [1], [3], [1], [1]),
        op.Slice(x, [1,2], [5,-2], [1,-1], [1,1]),

        op.Transpose(x, perm=[2,0,1,3]),

        # op.Upsample(x, [1.0, 1.0, 2.0, 2.0], mode='nearest'),
        # op.Upsample(x, [1.0, 1.0, 3.0, 4.0], mode='linear'),

        op.Resize(x, None, [1.0, 1.0, 2.0, 2.0], None, mode='nearest'),
        op.Resize(x, None, None, [1, 12, 13, 13], mode='linear', coordinate_transformation_mode="half_pixel"),
        op.Resize(x, None, None, [1, 12, 18, 18], mode='linear', coordinate_transformation_mode='align_corners'),

        a0, a1,
        b0, b1, b2, b3, b4, b5, b6,
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_layout_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(1, 12, 16, 14)

    # ort inference
    sess = ort.InferenceSession("test_onnx_layout_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_layout_ops.onnx inputshape=[1,12,16,14] inputshape2=[1,12,48,66]")

    # pnnx inference
    import test_onnx_layout_ops_pnnx
    b = test_onnx_layout_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_layout_ops_ncnn
    c = test_onnx_layout_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-4, 1e-4) or not torch.allclose(a0, c0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
