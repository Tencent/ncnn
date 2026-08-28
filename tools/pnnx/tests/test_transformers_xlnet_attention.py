# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import XLNetConfig
from transformers.models.xlnet.modeling_xlnet import XLNetRelativeAttention

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = XLNetConfig(d_model=192, n_head=12, d_head=16)
        self.attn0 = XLNetRelativeAttention(config0)

        torch.nn.init.xavier_uniform_(self.attn0.q)
        torch.nn.init.xavier_uniform_(self.attn0.k)
        torch.nn.init.xavier_uniform_(self.attn0.v)
        torch.nn.init.xavier_uniform_(self.attn0.o)
        torch.nn.init.xavier_uniform_(self.attn0.r)

        torch.nn.init.xavier_uniform_(self.attn0.r_r_bias)
        torch.nn.init.xavier_uniform_(self.attn0.r_s_bias)
        torch.nn.init.xavier_uniform_(self.attn0.r_w_bias)
        torch.nn.init.xavier_uniform_(self.attn0.seg_embed)

    def forward(self, x, r, mask0):

        mask0 = torch.zeros_like(mask0)

        out0 = self.attn0(h=x, g=None, attn_mask_h=mask0, attn_mask_g=None, r=r, seg_mat=None, head_mask=None, output_attentions=True)

        return out0[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16, 3, 192)
    r = torch.rand(32, 3, 192)
    mask0 = torch.rand(16, 16, 3, 12)

    a = net(x, r, mask0)

    mod = convert_and_import(
        net,
        (x, r, mask0),
        "test_transformers_xlnet_attention",
        pnnx_args=("inputshape=[16,3,192],[32,3,192],[16,16,3,12]",),
    )
    b = mod.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
