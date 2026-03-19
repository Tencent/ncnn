# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, u, v):
        x_values, x_indices = torch.topk(
            x, 2, dim=1, largest=True, sorted=True
        )
        x_k1_values, x_k1_indices = torch.topk(
            x, 1, dim=1, largest=True, sorted=True
        )
        x_k0_values, x_k0_indices = torch.topk(
            x, 0, dim=1, largest=True, sorted=True
        )
        x_unsorted_values, x_unsorted_indices = torch.topk(
            x, 2, dim=1, largest=True, sorted=False
        )
        x_values_only = torch.topk(
            x, 3, dim=1, largest=True, sorted=True
        )[0]
        y_values, y_indices = torch.topk(
            y, 4, dim=3, largest=False, sorted=True
        )
        z_values, z_indices = torch.topk(
            z, 3, dim=0, largest=True, sorted=True
        )
        z_unsorted_values, z_unsorted_indices = torch.topk(
            z, 3, dim=0, largest=True, sorted=False
        )
        u_values, u_indices = torch.topk(
            u, 2, dim=-1, largest=True, sorted=True
        )
        v_values, v_indices = torch.topk(
            v, 2, dim=1, largest=True, sorted=True
        )

        return (
            x_values,
            x_indices,
            x_k1_values,
            x_k1_indices,
            x_k0_values,
            x_k0_indices,
            x_unsorted_values,
            x_unsorted_indices,
            x_values_only,
            y_values,
            y_indices,
            z_values,
            z_indices,
            z_unsorted_values,
            z_unsorted_indices,
            u_values,
            u_indices,
            v_values,
            v_indices,
        )


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)
    u = torch.rand(2, 8, 4)
    v = torch.rand(2, 4, 3)

    a = net(x, y, z, u, v)

    # export onnx
    torch.onnx.export(net, (x, y, z, u, v), "test_torch_topk.onnx")

    # onnx to pnnx
    import os

    os.system(
        "../../src/pnnx test_torch_topk.onnx "
        "inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10],[2,8,4],[2,4,3]"
    )

    # pnnx inference
    import test_torch_topk_pnnx
    b = test_torch_topk_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
