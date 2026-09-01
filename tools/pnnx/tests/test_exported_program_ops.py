# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export


class LinearNorm(nn.Module):
    def __init__(self):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(8, 12)
        self.norm = nn.LayerNorm(12)

    def forward(self, x):
        return F.gelu(self.norm(self.linear(x)))


class ConvolutionNd(nn.Module):
    def __init__(self):
        super(ConvolutionNd, self).__init__()
        self.conv1 = nn.Conv1d(3, 4, 3, padding=1)
        self.conv3 = nn.Conv3d(2, 3, 3, padding=1)

    def forward(self, x1, x3):
        return F.relu(self.conv1(x1)), F.silu(self.conv3(x3))


class Pooling(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 2, 2), F.adaptive_max_pool2d(x, (2, 3))


class ShapeOps(nn.Module):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).reshape(1, -1, 3)
        return torch.cat((x, x + 1), dim=1)


class PadResize(nn.Module):
    def forward(self, x):
        x = F.pad(x, (1, 2, 2, 1), mode="constant", value=0.25)
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)


class EmbeddingBatchNorm(nn.Module):
    def __init__(self):
        super(EmbeddingBatchNorm, self).__init__()
        self.embedding = nn.Embedding(16, 5)
        self.norm = nn.BatchNorm2d(4)

    def forward(self, indices, x):
        return self.embedding(indices), self.norm(x)


class DynamicReshape(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])


def close(a, b):
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return torch.allclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True)


def run_case(model, inputs, basename, dynamic_shapes=None):
    model.eval()
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    expected = model(*inputs)
    converted = exported_program_to_pnnx(model, inputs, basename, dynamic_shapes=dynamic_shapes)
    actual = converted(*inputs)
    return close(expected, actual)


def test():
    if not has_torch_export():
        return True

    torch.manual_seed(0)
    cases = (
        (LinearNorm(), torch.rand(2, 4, 8), "test_exported_program_linear_norm"),
        (ConvolutionNd(), (torch.rand(1, 3, 9), torch.rand(1, 2, 4, 5, 6)), "test_exported_program_convolution_nd"),
        (Pooling(), torch.rand(1, 3, 8, 10), "test_exported_program_pooling"),
        (ShapeOps(), torch.rand(1, 3, 4, 5), "test_exported_program_shape_ops"),
        (PadResize(), torch.rand(1, 3, 4, 5), "test_exported_program_pad_resize"),
        (EmbeddingBatchNorm(), (torch.randint(0, 16, (2, 3)), torch.rand(1, 4, 5, 6)), "test_exported_program_embedding_batch_norm"),
    )

    if not all(run_case(model, inputs, basename) for model, inputs, basename in cases):
        return False

    if hasattr(torch.export, "Dim"):
        batch = torch.export.Dim("batch", min=1, max=4)
        height = torch.export.Dim("height", min=2, max=8)
        width = torch.export.Dim("width", min=2, max=8)
        model = DynamicReshape().eval()
        x = torch.rand(2, 3, 4, 5)
        converted = exported_program_to_pnnx(model, x, "test_exported_program_dynamic_shape", ({0: batch, 2: height, 3: width},))
        if not close(model(x), converted(x)):
            return False
        x = torch.rand(3, 3, 5, 6)
        return close(model(x), converted(x))

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
