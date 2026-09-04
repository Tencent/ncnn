# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
import torch
from packaging import version

# torch.onnx.export(..., dynamo=True) is only available on recent torch, and it
# is what lowers gelu into the float16 scalar constants this test needs.
if sys.version_info < (3, 9) or version.parse(torch.__version__) < version.parse('2.5'):
    sys.exit(0)

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def test():
    net = Model().half().eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16, 16, 16, dtype=torch.float16)

    a = net(x).float()

    # opset 18 predates the native Gelu op, so the dynamo exporter lowers gelu
    # into primitive ops whose 0.5 / sqrt(2) coefficients become float16 scalar
    # constants, which is what exercises the float16 scalar-constant path in the
    # onnx importer
    torch.onnx.export(net, (x,), "test_F_gelu_fp16.onnx", opset_version=18, dynamo=True)

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_gelu_fp16.onnx inputshape=[1,16,16,16]f16")

    # pnnx inference
    import test_F_gelu_fp16_pnnx
    b = test_F_gelu_fp16_pnnx.test_inference()
    b = b.float() if torch.is_tensor(b) else torch.from_numpy(b).float()

    return torch.allclose(a, b, 1e-2, 1e-2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
