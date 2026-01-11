# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torchvision
import torchvision.models as models
from packaging import version

def test():
    if version.parse(torchvision.__version__) < version.parse('0.13'):
        return True

    net = models.swin_t()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)

    a = net(x)

    # export onnx
    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x,), "test_swin_t.onnx", dynamo=False)
    else:
        torch.onnx.export(net, (x,), "test_swin_t.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_swin_t.onnx inputshape=[1,3,224,224]")

    # pnnx inference
    import test_swin_t_pnnx
    if (version.parse(torch.__version__) >= version.parse('1.12') and version.parse(torch.__version__) < version.parse('1.13') or
        version.parse(torch.__version__) >= version.parse('2.0') and version.parse(torch.__version__) < version.parse('2.1')):
        # torch-1.12 / 2.0 breaks 3d attention mask in no grad mode
        net_pnnx = test_swin_t_pnnx.Model().float().eval()
        b = net_pnnx(x)
    else:
        b = test_swin_t_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
