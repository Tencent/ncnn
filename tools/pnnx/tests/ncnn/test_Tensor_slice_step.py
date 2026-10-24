# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from packaging import version

class Model(nn.Module):
    def forward(self, x, y, z, w, b, c):
        b = torch.nn.functional.max_pool2d(b, 2, 1)
        c = torch.nn.functional.max_pool2d(c, 1)
        return (x[..., ::2], x[..., 1::2],
                x[..., 2:x.size(-1)-1:3], x[..., -5:-1:2],
                x[..., -100:100:3], x[1::2, ::3],
                x[1:x.size(0)-1:2, 2:x.size(1)-1:3],
                x[:, ::2][:, 1::3], y[:-233:2],
                z[:, 1::2, ::3], x[1, ::2],
                x[:y.size(0)//100:2, ::2],
                w[::2, 1::2, ::3, 1::2],
                w[:, 1:-1:2, :, 2:w.size(-1)-1:3],
                torch.relu(b[:, :, 1::2, ::3]),
                y[(y.size(0)-312)//2::2],
                y[:b.size(2):2],
                c[0, :, 1:], c[:1, :, :c.size(2)//2],
                torch.relu(x[..., ::2])[..., :8])

def make_inputs(height, width):
    return (torch.rand(height, width), torch.rand(301 + width),
            torch.rand(5, height, width), torch.rand(5, 4, height, width),
            torch.rand(2, 3, height, width) - 0.5, torch.rand(1, 3, height, width))

def test():
    net = Model().eval()
    torch.manual_seed(0)
    shapes = [(7, 15), (9, 16), (11, 24), (7, 7), (13, 13), (13, 11)]
    inputs = make_inputs(*shapes[0])

    if version.parse(torch.__version__) < version.parse('2.0'):
        mod = torch.jit.trace(net, inputs)
    else:
        mod = torch.jit.trace(net, inputs, _store_inputs=False)
    mod.save("test_Tensor_slice_step.pt")

    # torchscript to pnnx
    import subprocess
    shape_args = []
    for key, shape in zip(("inputshape", "inputshape2"), shapes[:2]):
        values = make_inputs(*shape)
        shape_args.append(key + "=" + ",".join(str(list(value.shape)).replace(" ", "") for value in values))
    subprocess.run(["../../src/pnnx", "test_Tensor_slice_step.pt", *shape_args, "fp16=0"], check=True)

    # ncnn inference with shapes not used for conversion
    import numpy as np
    import ncnn
    modes = [False]
    if hasattr(ncnn, "get_gpu_count") and ncnn.get_gpu_count() > 0:
        modes.append(True)
    for use_vulkan in modes:
        with ncnn.Net() as ncnn_net:
            ncnn_net.opt.use_vulkan_compute = use_vulkan
            ncnn_net.opt.use_fp16_storage = False
            ncnn_net.opt.use_fp16_packed = False
            ncnn_net.opt.use_fp16_arithmetic = False
            if ncnn_net.load_param("test_Tensor_slice_step.ncnn.param") != 0:
                return False
            if ncnn_net.load_model("test_Tensor_slice_step.ncnn.bin") != 0:
                return False
            for shape in shapes:
                inputs = make_inputs(*shape)
                expected = net(*inputs)
                with ncnn_net.create_extractor() as ex:
                    for i, value in enumerate(inputs):
                        mat = ncnn.Mat(value.numpy(), batch_index=0) if i >= 4 else ncnn.Mat(value.numpy())
                        if ex.input("in%d" % i, mat.clone()) != 0:
                            return False
                    for i, a in enumerate(expected):
                        ret, out = ex.extract("out%d" % i)
                        if ret != 0:
                            return False
                        batched = i in (14, 18)
                        if out.n != (a.shape[0] if batched else 1):
                            return False
                        b = torch.from_numpy(out.numpy(batch_index=0) if batched else np.array(out))
                        if a.shape != b.shape or not torch.equal(a, b):
                            print("slice mismatch", i, a.shape, b.shape, shape)
                            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
