# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = nn.Linear(17, 8)
        self.linear_no_bias = nn.Linear(17, 8, bias=False)

    def forward(self, x, y, z):
        z = F.max_pool2d(z, 1)
        return self.linear(x), self.linear_no_bias(x), self.linear(y), self.linear_no_bias(y), self.linear(z), self.linear_no_bias(z)

def test():
    torch.manual_seed(0)
    net = Model().half().float()
    net.eval()

    inputs = (torch.rand(1, 5, 17), torch.rand(2, 3, 5, 17), torch.rand(2, 3, 5, 17))
    inputs2 = (torch.rand(4, 7, 17), torch.rand(3, 4, 7, 17), torch.rand(3, 4, 7, 17))

    # export torchscript
    mod = torch.jit.trace(net, inputs)
    mod.save("test_nn_Linear_leading_dims.pt")

    # torchscript to pnnx
    import os
    if os.system("../../src/pnnx test_nn_Linear_leading_dims.pt inputshape=[1,5,17],[2,3,5,17],[2,3,5,17] inputshape2=[4,7,17],[3,4,7,17],[3,4,7,17] fp16=0") != 0:
        return False

    # ncnn inference
    import ncnn
    with ncnn.Net() as ncnn_net:
        ncnn_net.opt.num_threads = 2
        if ncnn_net.load_param("test_nn_Linear_leading_dims.ncnn.param") != 0 or ncnn_net.load_model("test_nn_Linear_leading_dims.ncnn.bin") != 0:
            return False
        for input_index, args in enumerate((inputs, inputs2)):
            expected = net(*args)
            with ncnn_net.create_extractor() as ex:
                for i, x in enumerate(args):
                    if ex.input("in%d" % i, ncnn.Mat(x.numpy(), batch_index=0 if i == 2 else 233).clone()) != 0:
                        return False
                for i, a in enumerate(expected):
                    ret, out = ex.extract("out%d" % i)
                    if ret != 0:
                        return False
                    b = torch.from_numpy(out.numpy(batch_index=0 if a.dim() == out.dims + 1 else 233).copy())
                    if a.shape != b.shape:
                        print("input set %d output %d shape mismatch: torch %s ncnn %s" % (input_index, i, tuple(a.shape), tuple(b.shape)))
                        return False
                    if not torch.allclose(a, b, 1e-3, 1e-3):
                        print("input set %d output %d shape %s max error %g" % (input_index, i, tuple(a.shape), (a - b).abs().max().item()))
                        return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
