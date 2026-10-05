# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version


class ModelMiddleBatch(nn.Module):
    def __init__(self):
        super(ModelMiddleBatch, self).__init__()

    def forward(self, x):
        x = x.unflatten(dim=0, sizes=(3, 2))
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelReshapeMiddleBatch(nn.Module):
    def __init__(self):
        super(ModelReshapeMiddleBatch, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelMiddleBatchWithOrdinaryPermute(nn.Module):
    def __init__(self):
        super(ModelMiddleBatchWithOrdinaryPermute, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        x = x.permute(1, 2, 0, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelBatchToMiddleOutput(nn.Module):
    def __init__(self):
        super(ModelBatchToMiddleOutput, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        x = x.permute(1, 0, 2, 3)
        return x


class ModelBatchToMiddleOutputSameDim(nn.Module):
    def __init__(self):
        super(ModelBatchToMiddleOutputSameDim, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        x = x.permute(1, 0, 2, 3)
        return x


class ModelMiddleBatchReshapeFoldAmbiguousAxis(nn.Module):
    def __init__(self):
        super(ModelMiddleBatchReshapeFoldAmbiguousAxis, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        x = x.permute(1, 0, 2, 3)
        x = F.relu(x)
        x = x.reshape(2, 105)
        return x


class ModelBatchMiddleRoundTrip(nn.Module):
    def __init__(self):
        super(ModelBatchMiddleRoundTrip, self).__init__()

    def forward(self, x):
        x = x.permute(1, 0, 2, 3)
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelFlattenRoundTrip(nn.Module):
    def __init__(self):
        super(ModelFlattenRoundTrip, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 0, 1)
        x = x.reshape(2, 3, 5, 7)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelFlattenBackwardBatch(nn.Module):
    def __init__(self):
        super(ModelFlattenBackwardBatch, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 0, 1)
        x = x.permute(1, 0, 2)
        x = F.max_pool1d(x, 1)
        return x


class ModelMiddleBatchFlattenFold(nn.Module):
    def __init__(self):
        super(ModelMiddleBatchFlattenFold, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.permute(1, 0, 2, 3)
        x = x.unsqueeze(1)
        x = torch.flatten(x, 1, 2)
        return x


class ModelMiddleBatchFlattenFoldAmbiguousAxis(nn.Module):
    def __init__(self):
        super(ModelMiddleBatchFlattenFoldAmbiguousAxis, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.permute(1, 0, 2, 3)
        x = x.unsqueeze(1)
        x = torch.flatten(x, 1, 2)
        return x


class ModelMiddleBatchUnflattenFold(nn.Module):
    def __init__(self):
        super(ModelMiddleBatchUnflattenFold, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.permute(1, 0, 2, 3)
        x = x.reshape(3, 2, 35)
        x = x.unflatten(dim=1, sizes=(1, 2))
        return x


class ModelMiddleBatchUnflattenFoldAmbiguousAxis(nn.Module):
    def __init__(self):
        super(ModelMiddleBatchUnflattenFoldAmbiguousAxis, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.permute(1, 0, 2, 3)
        x = x.reshape(2, 2, 35)
        x = x.unflatten(dim=1, sizes=(1, 2))
        return x


class ModelTwoBatchAxisReshapes(nn.Module):
    def __init__(self):
        super(ModelTwoBatchAxisReshapes, self).__init__()

    def forward(self, x, y):
        x = x.reshape(3, 2, 5, 7).permute(1, 0, 2, 3)
        y = y.reshape(4, 2, 3, 5).permute(1, 0, 2, 3)
        return F.max_pool2d(x, 3, stride=1, padding=1), F.max_pool2d(y, 3, stride=1, padding=1)


class ModelTwoDifferentMiddleBatchAxes(nn.Module):
    def __init__(self):
        super(ModelTwoDifferentMiddleBatchAxes, self).__init__()

    def forward(self, x, y):
        x = x.reshape(3, 2, 5, 7).permute(1, 0, 2, 3)
        y = y.reshape(4, 3, 2, 5).permute(2, 0, 1, 3)
        return F.max_pool2d(x, 3, stride=1, padding=1), F.max_pool2d(y, 3, stride=1, padding=1)


class ModelComputeBarrier(nn.Module):
    def __init__(self):
        super(ModelComputeBarrier, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        x = F.relu(x)
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelMultiConsumer(nn.Module):
    def __init__(self):
        super(ModelMultiConsumer, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        y = x.permute(1, 0, 2, 3)
        z = x.reshape(3, 2, 35)
        return y, z


class ModelBranchLayoutSplit(nn.Module):
    def __init__(self):
        super(ModelBranchLayoutSplit, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        y = x.permute(1, 0, 2, 3)
        y = F.max_pool2d(y, 3, stride=1, padding=1)
        z = x.reshape(3, 2, 35)
        return y, z


class ModelAxisSensitiveNonBatchOps(nn.Module):
    def __init__(self):
        super(ModelAxisSensitiveNonBatchOps, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.permute(1, 0, 2, 3)
        y = F.softmax(x, dim=0)
        z = torch.sum(x, dim=0, keepdim=False)
        q = torch.cumsum(x, dim=3)
        r = torch.flip(x, [2])
        s = x[:, :, 1:4, 2:6]
        return y, z, q, r, s


class ModelBinaryLayoutAgreement(nn.Module):
    def __init__(self):
        super(ModelBinaryLayoutAgreement, self).__init__()

    def forward(self, x, y, z):
        x = F.max_pool2d(x, 1)
        y = F.max_pool2d(y, 1)
        q = x.permute(1, 0, 2, 3)
        r = y.permute(1, 0, 2, 3)
        z0 = z
        z = z.unsqueeze(1)
        out0 = q + r
        out1 = q + z
        out2 = x + z0
        return out0, out1, out2


class ModelDuplicateInputLayout(nn.Module):
    def __init__(self):
        super(ModelDuplicateInputLayout, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.permute(1, 0, 2, 3)
        return x + x


class ModelCatStackSplitLayout(nn.Module):
    def __init__(self):
        super(ModelCatStackSplitLayout, self).__init__()

    def forward(self, x, y):
        x = F.max_pool1d(x, 1)
        y = F.max_pool1d(y, 1)
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        out0 = torch.cat((x, y), dim=0)
        out1 = torch.stack((x, y), dim=0)
        out2, out3 = torch.split(x, split_size_or_sections=[1, 2], dim=0)
        out4, out5 = torch.chunk(y, chunks=2, dim=2)
        out6, out7 = torch.tensor_split(x, (1,), dim=0)
        out8, out9, out10 = torch.unbind(x, dim=0)
        return out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10


class ModelUnbindBeforeBatchLayout(nn.Module):
    def __init__(self):
        super(ModelUnbindBeforeBatchLayout, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.permute(1, 0, 2, 3)
        return torch.unbind(x, dim=0)


class ModelSliceMultiSelectLayout(nn.Module):
    def __init__(self):
        super(ModelSliceMultiSelectLayout, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.permute(1, 0, 2, 3)
        return x[:, :, 1, 2]


class ModelPhysical5DReshape(nn.Module):
    def __init__(self):
        super(ModelPhysical5DReshape, self).__init__()

    def forward(self, x):
        return x.reshape(2, 3, 4, 5, 6)


class ModelPackedBatchReshapeBetweenConv(nn.Module):
    def __init__(self):
        super(ModelPackedBatchReshapeBetweenConv, self).__init__()

        self.conv0 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv1 = nn.Conv2d(4, 8, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = torch.flatten(x, 0, 1)
        x = x.reshape(4, 4, 5, 7)
        x = self.conv1(x)
        return x


class ModelSameBatchAxisReshapeCompat(nn.Module):
    def __init__(self):
        super(ModelSameBatchAxisReshapeCompat, self).__init__()

    def forward(self, x, y):
        out0 = x.reshape(x.size(0), x.size(1), -1)
        out1 = x.reshape_as(y)
        out2 = torch.flatten(x, 1, 2)
        return F.max_pool1d(out0, 1), F.max_pool1d(out1, 1), F.max_pool1d(out2, 1)


class ModelDynamicReshapeReuseBatch(nn.Module):
    def __init__(self):
        super(ModelDynamicReshapeReuseBatch, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.reshape(x.size(0), x.size(0), -1)
        x = F.max_pool1d(x, 1)
        return x


class ModelDynamicReshapeAsReference(nn.Module):
    def __init__(self):
        super(ModelDynamicReshapeAsReference, self).__init__()

    def forward(self, x, y):
        x = F.max_pool2d(x, 1)
        y = F.max_pool2d(y, 1)
        y = y.permute(1, 0, 2, 3)
        x = x.reshape_as(y)
        x = F.max_pool2d(x, 1)
        return x


class ModelSameBatchAxisUnflattenCompat(nn.Module):
    def __init__(self):
        super(ModelSameBatchAxisUnflattenCompat, self).__init__()

    def forward(self, x):
        x = x.unflatten(dim=1, sizes=(3, 4))
        x = F.max_pool2d(x, 1)
        return x


def compare(a, b):
    if isinstance(a, tuple):
        if not isinstance(b, tuple) or len(a) != len(b):
            return False
        for a0, b0 in zip(a, b):
            if not torch.allclose(a0, b0, 1e-3, 1e-3):
                return False
        return True

    return torch.allclose(a, b, 1e-3, 1e-3)


def no_batch_reshape_param(name):
    with open(name + ".ncnn.param") as f:
        for line in f:
            if line.startswith("Reshape ") and (" 12=" in line or " 13=" in line):
                return False

    return True


def run_model(name, net, inputs, inputs2=None):
    net.eval()

    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    if inputs2 is not None and not isinstance(inputs2, tuple):
        inputs2 = (inputs2,)

    a = net(*inputs)

    mod = torch.jit.trace(net, inputs)
    mod.save(name + ".pt")

    inputshape = ",".join([str(list(x.shape)).replace(" ", "") for x in inputs])
    pnnxcmd = "../../src/pnnx " + name + ".pt inputshape=" + inputshape
    if inputs2 is not None:
        inputshape2 = ",".join([str(list(x.shape)).replace(" ", "") for x in inputs2])
        pnnxcmd += " inputshape2=" + inputshape2
    if os.system(pnnxcmd) != 0:
        return False

    ncnnpy = __import__(name + "_ncnn")
    b = ncnnpy.test_inference()

    return compare(a, b)


def run_convert_only(name, net, inputs):
    net.eval()

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    mod = torch.jit.trace(net, inputs)
    mod.save(name + ".pt")

    inputshape = ",".join([str(list(x.shape)).replace(" ", "") for x in inputs])
    if os.system("../../src/pnnx " + name + ".pt inputshape=" + inputshape) != 0:
        return False

    return True


def test():
    if version.parse(torch.__version__) >= version.parse('1.13'):
        torch.manual_seed(0)
        x = torch.rand(6, 5, 7)
        if not run_model("test_ncnn_batch_layout_middle_batch", ModelMiddleBatch(), x):
            return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_reshape_middle_batch", ModelReshapeMiddleBatch(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_ordinary_permute", ModelMiddleBatchWithOrdinaryPermute(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_batch_to_middle", ModelBatchToMiddleOutput(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 2, 5, 7)
    if not run_model("test_ncnn_batch_layout_batch_to_middle_same_dim", ModelBatchToMiddleOutputSameDim(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_middle_batch_reshape_fold_ambiguous_axis", ModelMiddleBatchReshapeFoldAmbiguousAxis(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_roundtrip", ModelBatchMiddleRoundTrip(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_flatten_roundtrip", ModelFlattenRoundTrip(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(3, 5, 2, 7)
    if not run_model("test_ncnn_batch_layout_flatten_backward", ModelFlattenBackwardBatch(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_middle_batch_flatten_fold", ModelMiddleBatchFlattenFold(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 2, 5, 7)
    if not run_model("test_ncnn_batch_layout_middle_batch_flatten_fold_ambiguous_axis", ModelMiddleBatchFlattenFoldAmbiguousAxis(), x):
        return False

    if version.parse(torch.__version__) >= version.parse('1.13'):
        torch.manual_seed(0)
        x = torch.rand(2, 3, 5, 7)
        if not run_model("test_ncnn_batch_layout_middle_batch_unflatten_fold", ModelMiddleBatchUnflattenFold(), x):
            return False

        torch.manual_seed(0)
        x = torch.rand(2, 2, 5, 7)
        if not run_model("test_ncnn_batch_layout_middle_batch_unflatten_fold_ambiguous_axis", ModelMiddleBatchUnflattenFoldAmbiguousAxis(), x):
            return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    y = torch.rand(8, 3, 5)
    if not run_model("test_ncnn_batch_layout_two_reshapes", ModelTwoBatchAxisReshapes(), (x, y)):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    y = torch.rand(12, 2, 5)
    if not run_model("test_ncnn_batch_layout_two_different_middle_batch_axes", ModelTwoDifferentMiddleBatchAxes(), (x, y)):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_compute_barrier", ModelComputeBarrier(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_multi_consumer", ModelMultiConsumer(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_branch_layout_split", ModelBranchLayoutSplit(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_axis_sensitive_nonbatch_ops", ModelAxisSensitiveNonBatchOps(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    y = torch.rand(2, 3, 5, 7)
    z = torch.rand(3, 1, 1)
    if not run_model("test_ncnn_batch_layout_binary_layout_agreement", ModelBinaryLayoutAgreement(), (x, y, z)):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_duplicate_input", ModelDuplicateInputLayout(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 8)
    y = torch.rand(2, 3, 8)
    if not run_model("test_ncnn_batch_layout_cat_stack_split", ModelCatStackSplitLayout(), (x, y)):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_unbind_before_batch", ModelUnbindBeforeBatchLayout(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_slice_multi_select", ModelSliceMultiSelectLayout(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(720)
    if not run_convert_only("test_ncnn_batch_layout_physical5d_reshape", ModelPhysical5DReshape(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_packed_between_conv", ModelPackedBatchReshapeBetweenConv(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 4, 5)
    y = torch.rand(2, 3, 20)
    x2 = torch.rand(4, 3, 6, 7)
    y2 = torch.rand(4, 3, 42)
    name = "test_ncnn_batch_layout_same_batch_axis_reshape_compat"
    if not run_model(name, ModelSameBatchAxisReshapeCompat(), (x, y), (x2, y2)):
        return False
    if not no_batch_reshape_param(name):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 4, 5, 7)
    x2 = torch.rand(4, 4, 5, 7)
    if not run_model("test_ncnn_batch_layout_dynamic_reshape_reuse_batch", ModelDynamicReshapeReuseBatch(), x, x2):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    y = torch.rand(3, 2, 5, 7)
    x2 = torch.rand(4, 6, 8, 10)
    y2 = torch.rand(6, 4, 8, 10)
    if not run_model("test_ncnn_batch_layout_dynamic_reshape_as_reference", ModelDynamicReshapeAsReference(), (x, y), (x2, y2)):
        return False

    if version.parse(torch.__version__) >= version.parse('1.13'):
        torch.manual_seed(0)
        x = torch.rand(2, 12, 5)
        name = "test_ncnn_batch_layout_same_batch_axis_unflatten_compat"
        if not run_model(name, ModelSameBatchAxisUnflattenCompat(), x):
            return False
        if not no_batch_reshape_param(name):
            return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
