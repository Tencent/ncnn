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
def Model(x: FLOAT["L","N",64]):

    rnn_W = op.RandomNormal(seed=0.0, shape=[2,44,64])
    rnn_R = op.RandomNormal(seed=1.0, shape=[2,44,44])
    rnn_B = op.RandomNormal(seed=2.0, shape=[2,2*44])

    rnn_W2 = op.RandomNormal(seed=3.0, shape=[1,16,64])
    rnn_R2 = op.RandomNormal(seed=4.0, shape=[1,16,16])

    lstm_W = op.RandomNormal(seed=5.0, shape=[2,4*44,64])
    lstm_R = op.RandomNormal(seed=6.0, shape=[2,4*44,44])
    lstm_B = op.RandomNormal(seed=7.0, shape=[2,8*44])

    lstm_W2 = op.RandomNormal(seed=8.0, shape=[1,4*16,64])
    lstm_R2 = op.RandomNormal(seed=9.0, shape=[1,4*16,16])

    gru_W = op.RandomNormal(seed=10.0, mean=0.5, shape=[2,3*44,64])
    gru_R = op.RandomNormal(seed=11.0, mean=0.5, shape=[2,3*44,44])
    gru_B = op.RandomNormal(seed=12.0, shape=[2,6*44])

    gru_W2 = op.RandomNormal(seed=13.0, mean=0.5, shape=[1,3*16,64])
    gru_R2 = op.RandomNormal(seed=14.0, mean=0.5, shape=[1,3*16,16])

    return (
        op.Reshape(op.Transpose(op.RNN(x, rnn_W, rnn_R, rnn_B, hidden_size=44, direction='bidirectional'), perm=[0,2,1,3]), shape=[0,0,-1]),
        op.Squeeze(op.RNN(x, rnn_W2, rnn_R2, None, hidden_size=16), axes=[1]),

        op.Reshape(op.Transpose(op.LSTM(x, lstm_W, lstm_R, lstm_B, hidden_size=44, direction='bidirectional'), perm=[0,2,1,3]), shape=[0,0,-1]),
        op.Squeeze(op.LSTM(x, lstm_W2, lstm_R2, None, hidden_size=16), axes=[1]),

        op.Reshape(op.Transpose(op.GRU(x, gru_W, gru_R, gru_B, hidden_size=44, direction='bidirectional'), perm=[0,2,1,3]), shape=[0,0,-1]),
        op.Squeeze(op.GRU(x, gru_W2, gru_R2, None, hidden_size=16), axes=[1]),

        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_rnn_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(9, 1, 64)

    # ort inference
    sess = ort.InferenceSession("test_onnx_rnn_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_rnn_ops.onnx inputshape=[9,1,64] fp16=0")

    # pnnx inference
    import test_onnx_rnn_ops_pnnx
    b = test_onnx_rnn_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_rnn_ops_ncnn
    c = test_onnx_rnn_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-3, 1e-3) or not torch.allclose(a0, c0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
