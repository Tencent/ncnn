# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import subprocess
import sys

from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(6)
        self.prelu = nn.PReLU(6)
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(24, 8)
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.prelu(x)
        x = F.avg_pool2d(x, kernel_size=2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.ln(x)
        x = F.gelu(x)

        x = x.reshape(1, 2, 4)
        x = torch.matmul(x, x.transpose(-1, -2))
        x = F.softmax(x, dim=-1)
        x = F.pad(x, (1, 1))
        return x


class MultiInputModel(nn.Module):
    def __init__(self):
        super(MultiInputModel, self).__init__()

    def forward(self, x, y, weight, bias):
        x = x + y
        x = torch.matmul(x, weight)
        x = x + bias
        x = F.relu(x)
        return x


class ViewMaterializeModel(nn.Module):
    def __init__(self):
        super(ViewMaterializeModel, self).__init__()

    def forward(self, x, y):
        a = x.reshape(6)
        b = x.view(2, 3)
        c = x.permute(1, 0)
        d = x[:, 1:]
        e = torch.cat((x, y), dim=0)
        f = torch.clone(y)
        return a, b, c, d, e, f


class ExpressionModel(nn.Module):
    def __init__(self):
        super(ExpressionModel, self).__init__()

    def forward(self, x):
        y = torch.sin(x) + torch.cos(x) * torch.sqrt(torch.abs(x) + 1)
        return y


class AddmmModel(nn.Module):
    def __init__(self):
        super(AddmmModel, self).__init__()

    def forward(self, x, y, z):
        return torch.addmm(x, y, z)


class ScaledAddmmBaddbmmModel(nn.Module):
    def __init__(self):
        super(ScaledAddmmBaddbmmModel, self).__init__()

    def forward(self, x, y, z, a, b, c):
        x = torch.addmm(x, y, z, beta=0, alpha=0.7)
        a = torch.baddbmm(a, b, c, beta=1.4, alpha=0.7)
        return x, a


class UnbatchedConvolutionModel(nn.Module):
    def __init__(self):
        super(UnbatchedConvolutionModel, self).__init__()

        self.conv1d = nn.Conv1d(2, 3, kernel_size=3, padding=1, bias=True)
        self.conv2d = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
        self.conv3d = nn.Conv3d(1, 2, kernel_size=3, padding=1, bias=False)
        self.deconv1d = nn.ConvTranspose1d(2, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x1, x2, x3, x4):
        return self.conv1d(x1), self.conv2d(x2), self.conv3d(x3), self.deconv1d(x4)


class FunctionalNormNoAffineModel(nn.Module):
    def __init__(self):
        super(FunctionalNormNoAffineModel, self).__init__()

        self.register_buffer("running_mean", torch.zeros(4))
        self.register_buffer("running_var", torch.ones(4))

    def forward(self, x):
        x = F.layer_norm(x, (3, 3), weight=None, bias=None)
        x = F.group_norm(x, 2, weight=None, bias=None)
        x = F.batch_norm(x, self.running_mean, self.running_var, weight=None, bias=None, training=False)
        return x


class FunctionalBatchNormModel(nn.Module):
    def __init__(self):
        super(FunctionalBatchNormModel, self).__init__()

    def forward(self, x, running_mean, running_var, weight, bias):
        return F.batch_norm(x, running_mean, running_var, weight, bias, training=False)


class InstanceNormStatsModel(nn.Module):
    def __init__(self):
        super(InstanceNormStatsModel, self).__init__()

        self.in_stats = nn.InstanceNorm2d(4, affine=True, track_running_stats=False)
        self.in_running = nn.InstanceNorm2d(4, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.in_stats(x), self.in_running(x)


class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()

        self.embed = nn.Embedding(num_embeddings=10, embedding_dim=4)

    def forward(self, x):
        return self.embed(x)


class MultiheadAttentionMaskModel(nn.Module):
    def __init__(self):
        super(MultiheadAttentionMaskModel, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=True)

    def forward(self, x, mask):
        x, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return x


class UnbatchedMultiheadAttentionModel(nn.Module):
    def __init__(self):
        super(UnbatchedMultiheadAttentionModel, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=True)

    def forward(self, x):
        x, _ = self.attn(x, x, x, need_weights=False)
        return x


class ScaledDotProductAttentionMaskModel(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttentionMaskModel, self).__init__()

    def forward(self, q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class ScaledDotProductAttentionNoMaskModel(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttentionNoMaskModel, self).__init__()

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


class ScaledDotProductAttentionCausalModel(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttentionCausalModel, self).__init__()

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class NormalizeModel(nn.Module):
    def __init__(self):
        super(NormalizeModel, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class ReductionStatModel(nn.Module):
    def __init__(self):
        super(ReductionStatModel, self).__init__()

    def forward(self, x):
        a = torch.mean(x, dim=1)
        b = torch.var(x, dim=1, unbiased=False)
        c = torch.std(x, dim=1, unbiased=False)
        d = torch.logsumexp(x, dim=1)
        e = torch.norm(x, p=2, dim=1)
        return a, b, c, d, e


class FusedFunctionalStatModel(nn.Module):
    def __init__(self):
        super(FusedFunctionalStatModel, self).__init__()

    def forward(self, x_lrn, x_lp, x1, x2):
        a = F.local_response_norm(x_lrn, size=3)
        b = F.lp_pool2d(x_lp, norm_type=2, kernel_size=2, stride=2)
        c = F.pairwise_distance(x1, x2)
        return a, b, c


class FoldModel(nn.Module):
    def __init__(self):
        super(FoldModel, self).__init__()

    def forward(self, x):
        return F.fold(x, output_size=(3, 3), kernel_size=(2, 2), stride=1)


class AdaptivePoolModel(nn.Module):
    def __init__(self):
        super(AdaptivePoolModel, self).__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (3, 3))


class UpsampleNearestModel(nn.Module):
    def __init__(self):
        super(UpsampleNearestModel, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


class MultiheadAttentionExtraModel(nn.Module):
    def __init__(self):
        super(MultiheadAttentionExtraModel, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=True, add_bias_kv=True, add_zero_attn=True)

    def forward(self, x):
        x, _ = self.attn(x, x, x, need_weights=False)
        return x


class LSTMProjStateModel(nn.Module):
    def __init__(self):
        super(LSTMProjStateModel, self).__init__()

        self.lstm = nn.LSTM(input_size=3, hidden_size=4, proj_size=2, bias=False)

    def forward(self, x, h, c):
        y, (hn, cn) = self.lstm(x, (h, c))
        return y, hn, cn


class UnbatchedLSTMModel(nn.Module):
    def __init__(self):
        super(UnbatchedLSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=3, hidden_size=4, bias=False)

    def forward(self, x):
        y, (hn, cn) = self.lstm(x)
        return y, hn, cn


def _make_input(spec):
    if isinstance(spec, torch.Tensor):
        return spec

    if isinstance(spec, tuple) and len(spec) == 4 and spec[0] == "randint":
        return torch.randint(spec[1], spec[2], dtype=spec[3])

    return torch.rand(spec)


def _format_ops(ops):
    units = ["", "K", "M", "G", "T", "P"]
    unit_scales = [1, 1000, 1000000, 1000000000, 1000000000000, 1000000000000000]

    unit_index = 0
    while unit_index + 1 < len(unit_scales) and ops >= unit_scales[unit_index + 1]:
        unit_index = unit_index + 1

    if unit_index == 0:
        return str(ops)

    scale = unit_scales[unit_index]
    integer = ops // scale
    fraction = ((ops % scale) * 1000 + scale // 2) // scale

    if fraction == 1000:
        integer = integer + 1
        fraction = 0

    s = str(integer)
    if fraction != 0:
        s = s + "." + str(fraction).rjust(3, "0").rstrip("0")

    return s + units[unit_index]


def _check_stat_text(text, expected_inputshape, expected_flops, expected_memops):
    inputshape = re.findall(r"(?:^|\n)#? ?model inputshape = (.+)", text)
    flops = re.findall(r"(?:^|\n)#? ?FLOPS = ([0-9.]+[KMGTPE]?)", text)
    memops = re.findall(r"(?:^|\n)#? ?memory OPS = ([0-9.]+[KMGTPE]?)", text)
    if not inputshape or not flops or not memops:
        return False

    return inputshape[-1] == expected_inputshape and \
        flops[-1] == _format_ops(expected_flops) and \
        memops[-1] == _format_ops(expected_memops)


def _allclose(a, b):
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return False

        for a0, b0 in zip(a, b):
            if not _allclose(a0, b0):
                return False

        return True

    if not torch.is_floating_point(a):
        return torch.equal(a, b)

    return torch.allclose(a, b, 1e-4, 1e-4)


def _run_case(name, net, inputs, inputshape, expected_inputshape, expected_flops, expected_memops):
    net.eval()

    torch.manual_seed(0)
    inputs = tuple(_make_input(spec) for spec in inputs)

    a = net(*inputs)

    # export torchscript
    mod = torch.jit.trace(net, inputs)
    mod.save(name + ".pt")

    # torchscript to pnnx
    cmd = ["../src/pnnx", name + ".pt", "inputshape=" + inputshape]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        return False

    if not _check_stat_text(p.stdout + p.stderr, expected_inputshape, expected_flops, expected_memops):
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        return False

    with open(name + "_pnnx.py", "r") as f:
        pnnx_py = f.read()

    if "# pnnx model stat" not in pnnx_py:
        return False
    if not _check_stat_text(pnnx_py, expected_inputshape, expected_flops, expected_memops):
        return False

    # pnnx inference
    pnnx_module = __import__(name + "_pnnx")
    b = pnnx_module.test_inference()

    return _allclose(a, b)


def test():
    if not _run_case("test_pnnx_model_stat", Model(),
                     ((1, 3, 8, 8),),
                     "[1,3,8,8]", "[1,3,8,8]f32",
                     23708, 4166):
        return False

    if not _run_case("test_pnnx_model_stat_multi_input", MultiInputModel(),
                     ((2, 3), (2, 3), (3, 4), (4,)),
                     "[2,3],[2,3],[3,4],[4]",
                     "[2,3]f32,[2,3]f32,[3,4]f32,[4]f32",
                     70, 80):
        return False

    if not _run_case("test_pnnx_model_stat_view_materialize", ViewMaterializeModel(),
                     ((2, 3), (2, 3)),
                     "[2,3],[2,3]", "[2,3]f32,[2,3]f32",
                     0, 36):
        return False

    if not _run_case("test_pnnx_model_stat_expression", ExpressionModel(),
                     ((2, 3),),
                     "[2,3]", "[2,3]f32",
                     96, 12):
        return False

    if not _run_case("test_pnnx_model_stat_addmm", AddmmModel(),
                     ((2, 4), (2, 3), (3, 4)),
                     "[2,4],[2,3],[3,4]",
                     "[2,4]f32,[2,3]f32,[3,4]f32",
                     56, 34):
        return False

    if not _run_case("test_pnnx_model_stat_scaled_addmm_baddbmm", ScaledAddmmBaddbmmModel(),
                     ((2, 4), (2, 3), (3, 4), (2, 2, 4), (2, 2, 3), (2, 3, 4)),
                     "[2,4],[2,3],[3,4],[2,2,4],[2,2,3],[2,3,4]",
                     "[2,4]f32,[2,3]f32,[3,4]f32,[2,2,4]f32,[2,2,3]f32,[2,3,4]f32",
                     200, 94):
        return False

    if version.parse(torch.__version__) >= version.parse('1.12'):
        if not _run_case("test_pnnx_model_stat_unbatched_convolution", UnbatchedConvolutionModel(),
                         ((2, 5), (2, 4, 4), (1, 3, 3, 3), (2, 5)),
                         "[2,5],[2,4,4],[1,3,3,3],[2,5]",
                         "[2,5]f32,[2,4,4]f32,[1,3,3,3]f32,[2,5]f32",
                         5019, 358):
            return False

    if not _run_case("test_pnnx_model_stat_norm_no_affine", FunctionalNormNoAffineModel(),
                     ((2, 4, 3, 3),),
                     "[2,4,3,3]", "[2,4,3,3]f32",
                     1152, 440):
        return False

    if not _run_case("test_pnnx_model_stat_functional_batch_norm", FunctionalBatchNormModel(),
                     ((2, 4, 3, 3), (4,), (4,), (4,), (4,)),
                     "[2,4,3,3],[4],[4],[4],[4]",
                     "[2,4,3,3]f32,[4]f32,[4]f32,[4]f32,[4]f32",
                     288, 160):
        return False

    if not _run_case("test_pnnx_model_stat_instance_norm_stats", InstanceNormStatsModel(),
                     ((2, 4, 3, 3),),
                     "[2,4,3,3]", "[2,4,3,3]f32",
                     936, 312):
        return False

    if not _run_case("test_pnnx_model_stat_embedding", EmbeddingModel(),
                     (("randint", 10, (2, 3), torch.int),),
                     "[2,3]i32", "[2,3]i32",
                     0, 54):
        return False

    if not _run_case("test_pnnx_model_stat_multihead_attention_mask", MultiheadAttentionMaskModel(),
                     ((3, 1, 4), (3, 3)),
                     "[3,1,4],[3,3]", "[3,1,4]f32,[3,3]f32",
                     684, 155):
        return False

    if not _run_case("test_pnnx_model_stat_multihead_attention_extra", MultiheadAttentionExtraModel(),
                     ((3, 1, 4),),
                     "[3,1,4]", "[3,1,4]f32",
                     822, 166):
        return False

    if version.parse(torch.__version__) >= version.parse('1.12'):
        if not _run_case("test_pnnx_model_stat_multihead_attention_unbatched", UnbatchedMultiheadAttentionModel(),
                         ((3, 4),),
                         "[3,4]", "[3,4]f32",
                         666, 146):
            return False

    if not _run_case("test_pnnx_model_stat_normalize", NormalizeModel(),
                     ((2, 3, 4),),
                     "[2,3,4]", "[2,3,4]f32",
                     112, 112):
        return False

    if not _run_case("test_pnnx_model_stat_reduction", ReductionStatModel(),
                     ((2, 3),),
                     "[2,3]", "[2,3]f32",
                     130, 40):
        return False

    fused_functional_flops = 324 if version.parse(torch.__version__) < version.parse('2.0') else 308
    fused_functional_memops = 206 if version.parse(torch.__version__) < version.parse('2.0') else 126
    if not _run_case("test_pnnx_model_stat_fused_functional", FusedFunctionalStatModel(),
                     ((1, 4, 2, 2), (1, 1, 4, 4), (2, 3), (2, 3)),
                     "[1,4,2,2],[1,1,4,4],[2,3],[2,3]",
                     "[1,4,2,2]f32,[1,1,4,4]f32,[2,3]f32,[2,3]f32",
                     fused_functional_flops, fused_functional_memops):
        return False

    if not _run_case("test_pnnx_model_stat_fold", FoldModel(),
                     ((1, 4, 4),),
                     "[1,4,4]", "[1,4,4]f32",
                     7, 25):
        return False

    if not _run_case("test_pnnx_model_stat_adaptive_pool", AdaptivePoolModel(),
                     ((1, 1, 5, 5),),
                     "[1,1,5,5]", "[1,1,5,5]f32",
                     49, 34):
        return False

    if not _run_case("test_pnnx_model_stat_upsample_nearest", UpsampleNearestModel(),
                     ((1, 1, 2, 2),),
                     "[1,1,2,2]", "[1,1,2,2]f32",
                     0, 20):
        return False

    if not _run_case("test_pnnx_model_stat_lstm_proj_state", LSTMProjStateModel(),
                     ((2, 1, 3), (1, 1, 2), (1, 1, 4)),
                     "[2,1,3],[1,1,2],[1,1,4]",
                     "[2,1,3]f32,[1,1,2]f32,[1,1,4]f32",
                     432, 110):
        return False

    if version.parse(torch.__version__) >= version.parse('1.12'):
        if not _run_case("test_pnnx_model_stat_lstm_unbatched", UnbatchedLSTMModel(),
                         ((2, 3),),
                         "[2,3]", "[2,3]f32",
                         528, 134):
            return False

    if hasattr(F, "scaled_dot_product_attention"):
        if not _run_case("test_pnnx_model_stat_scaled_dot_product_attention_no_mask", ScaledDotProductAttentionNoMaskModel(),
                         ((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 5)),
                         "[1,2,3,4],[1,2,3,4],[1,2,3,5]",
                         "[1,2,3,4]f32,[1,2,3,4]f32,[1,2,3,5]f32",
                         414, 126):
            return False

        if not _run_case("test_pnnx_model_stat_scaled_dot_product_attention_mask", ScaledDotProductAttentionMaskModel(),
                         ((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 3, 3)),
                         "[1,2,3,4],[1,2,3,4],[1,2,3,5],[1,2,3,3]",
                         "[1,2,3,4]f32,[1,2,3,4]f32,[1,2,3,5]f32,[1,2,3,3]f32",
                         432, 144):
            return False

        if not _run_case("test_pnnx_model_stat_scaled_dot_product_attention_causal", ScaledDotProductAttentionCausalModel(),
                         ((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 5)),
                         "[1,2,3,4],[1,2,3,4],[1,2,3,5]",
                         "[1,2,3,4]f32,[1,2,3,4]f32,[1,2,3,5]f32",
                         432, 126):
            return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
