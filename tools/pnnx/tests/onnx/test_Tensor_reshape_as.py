# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.reshape_as(y)
        y = y.reshape_as(z)
        z = z.reshape_as(x)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(6, 2, 2, 2)
    z = torch.rand(48)

    a = net(x, y, z)

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_Tensor_reshape_as.onnx", input_names = ['x','y','z'], output_names = ['a','b','c'],
                      dynamic_axes={'x' : {0 : 'x0', 1 : 'x1', 2 : 'x2'}, 'y' : {0 : 'y0', 1 : 'y1', 2 : 'y2', 3 : 'y3'}, 'z' : {0 : 'z0'}})

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_reshape_as.onnx inputshape=[1,3,16],[6,2,2,2],[48] inputshape2=[1,24,8],[12,1,4,4],[192]")

    # pnnx inference
    import test_Tensor_reshape_as_pnnx
    netb = test_Tensor_reshape_as_pnnx.Model().float().eval()
    b = netb(x, y, z)

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
