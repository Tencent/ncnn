# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
if sys.version_info < (3, 9):
    sys.exit(0)

import torch
import torch.nn as nn
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=16, num_layers=1, bidirectional=True)

    def forward(self, x, h0):
        x0, h1 = self.gru(x, h0)
        return x0, h1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(9, 1, 32)
    h0 = torch.rand(2, 1, 16)

    a = net(x, h0)

    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x, h0), "test_nn_GRU_initial_states.onnx", dynamo=False, input_names=["x", "h0"])
    else:
        torch.onnx.export(net, (x, h0), "test_nn_GRU_initial_states.onnx", dynamo=False, input_names=["x", "h0"])

    import os
    os.system("../../src/pnnx test_nn_GRU_initial_states.onnx inputshape=[9,1,32],[2,1,16] fp16=0")

    import test_nn_GRU_initial_states_pnnx
    b = test_nn_GRU_initial_states_pnnx.test_inference()

    import test_nn_GRU_initial_states_ncnn
    c = test_nn_GRU_initial_states_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-3, 1e-3) or not torch.allclose(a0, c0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
