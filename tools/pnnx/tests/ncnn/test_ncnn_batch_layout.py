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


class ModelBatchFoldToConv1d(nn.Module):
    def __init__(self, mode):
        super(ModelBatchFoldToConv1d, self).__init__()
        self.pre = nn.Conv2d(4, 4, 1)
        self.conv = nn.Conv1d(4, 6, 3, padding=1)
        self.mode = mode

    def forward(self, x):
        x = self.pre(x)
        batch, channels, frequencies, frames = x.shape
        x = x.permute(0, 2, 1, 3)
        if self.mode == "flatten":
            x = torch.flatten(x, 0, 1)
        else:
            x = x.reshape(batch * frequencies, channels, frames)
        return self.conv(x)


class ModelBatchFoldToUnbatchedPool2d(nn.Module):
    def __init__(self):
        super(ModelBatchFoldToUnbatchedPool2d, self).__init__()
        self.pre = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        x = self.pre(x)
        x = torch.flatten(x, 0, 1)
        return F.max_pool2d(x, 1)


class ModelChannelFoldToConv1d(nn.Module):
    def __init__(self):
        super(ModelChannelFoldToConv1d, self).__init__()
        self.pre = nn.Conv2d(4, 4, 1)
        self.conv = nn.Conv1d(32, 6, 3, padding=1)

    def forward(self, x):
        x = self.pre(x)
        batch, channels, frequencies, frames = x.shape
        x = x.reshape(batch, channels * frequencies, frames)
        return self.conv(x)


class ModelBatchUnflattenAfterConv1d(nn.Module):
    def __init__(self):
        super(ModelBatchUnflattenAfterConv1d, self).__init__()
        self.pre = nn.Conv2d(4, 4, 1)
        self.conv = nn.Conv1d(4, 6, 3, padding=1)
        self.post = nn.Conv2d(6, 6, 1)

    def forward(self, x):
        x = self.pre(x)
        batch, _, frequencies, _ = x.shape
        x = x.permute(0, 2, 1, 3)
        x = torch.flatten(x, 0, 1)
        x = self.conv(x)
        x = x.unflatten(0, (batch, frequencies))
        x = x.permute(0, 2, 1, 3)
        return self.post(x)


class ModelDynamicBatchUnflatten(nn.Module):
    def __init__(self):
        super(ModelDynamicBatchUnflatten, self).__init__()
        self.conv = nn.Conv1d(4, 6, 1)
        self.post = nn.Conv2d(6, 6, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.unflatten(0, (-1, 2))
        return self.post(x.permute(0, 2, 1, 3))


class ModelMiddleBatchReshapePartition(nn.Module):
    def forward(self, x):
        x = F.max_pool1d(x, 1).transpose(0, 1)
        x = x.reshape(4, 2, 6).transpose(0, 1)
        return F.max_pool1d(x, 1)


class ModelMiddleBatchShapeExpression(nn.Module):
    def forward(self, x):
        x = F.max_pool1d(x, 1).transpose(0, 1)
        x = x.reshape(x.size(0), x.size(1), 2, x.size(2) // 2)
        return F.max_pool2d(x.permute(1, 0, 2, 3), 1)


class ModelExternalShapeReference(nn.Module):
    def forward(self, x, y):
        y = F.max_pool1d(y, 1)
        return x.reshape(-1, y.size(2))


class ModelDifferentBatchShapeReferences(nn.Module):
    def forward(self, x, y, z):
        y = F.max_pool1d(y, 1)
        z = F.max_pool1d(z, 1)
        return x.reshape(y.size(0), z.size(0), -1)


class ModelDynamicAdjacentReshape(nn.Module):
    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.reshape(x.size(0), x.size(1), 2, -1)
        return F.max_pool2d(x, 1)


class ModelDynamicLinear(nn.Module):
    def __init__(self):
        super(ModelDynamicLinear, self).__init__()
        self.linear = nn.Linear(8, 6)

    def forward(self, x):
        return self.linear(F.max_pool2d(x, 1))


def compare(a, b):
    if not isinstance(a, tuple):
        a = (a,)
    if not isinstance(b, tuple):
        b = (b,)
    if len(a) != len(b):
        print("output count mismatch", len(a), len(b))
        return False
    for i, (a0, b0) in enumerate(zip(a, b)):
        if a0.shape != b0.shape:
            print("output", i, "shape mismatch", a0.shape, b0.shape)
            return False
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            print("output", i, "max error", (a0 - b0).abs().max().item())
            return False
    return True


def no_batch_reshape_param(name):
    with open(name + ".ncnn.param") as f:
        for line in f:
            if line.startswith("Reshape ") and (" 12=" in line or " 13=" in line):
                return False

    return True


def has_batch_reshape_param(name, input_axis=0, output_axis=0, layer_prefix=None):
    with open(name + ".ncnn.param") as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] != "Reshape":
                continue
            if layer_prefix is not None and not fields[1].startswith(layer_prefix):
                continue
            param_start = 4 + int(fields[2]) + int(fields[3])
            params = dict(x.split("=", 1) for x in fields[param_start:])
            if params.get("12") == str(input_axis) and params.get("13") == str(output_axis):
                return True

    print(name, "missing batch reshape", layer_prefix, input_axis, output_axis)
    return False


def run_model(name, net, inputs, inputs2=None):
    net.eval()

    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    if inputs2 is not None and not isinstance(inputs2, tuple):
        inputs2 = (inputs2,)

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
    import ncnn
    with ncnn.Net() as ncnnnet:
        if ncnnnet.load_param(name + ".ncnn.param") != 0 or ncnnnet.load_model(name + ".ncnn.bin") != 0:
            return False
        for data in (inputs, inputs2):
            if data is not None and not compare(net(*data), ncnnpy.inference(ncnnnet, *data)):
                print(name, "inputs", [tuple(x.shape) for x in data])
                return False
    return True


def run_convert_warning(name, net, inputs, warning):
    net.eval()

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    mod = torch.jit.trace(net, inputs)
    mod.save(name + ".pt")

    inputshape = ",".join([str(list(x.shape)).replace(" ", "") for x in inputs])
    logpath = name + ".log"
    pnnxcmd = "../../src/pnnx " + name + ".pt inputshape=" + inputshape + " 2>" + logpath
    ret = os.system(pnnxcmd)
    with open(logpath) as f:
        log = f.read()
    print(log, end="")
    if ret != 0:
        return False

    return warning in log


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
    if not run_convert_warning("test_ncnn_batch_layout_physical5d_reshape", ModelPhysical5DReshape(), x, "target exceeds ncnn physical rank"):
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
    if not has_batch_reshape_param(name):
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

    torch.manual_seed(0)
    x = torch.rand(1, 4, 8, 16)
    name = "test_ncnn_batch_layout_flatten_to_conv1d_batch1"
    if not run_model(name, ModelBatchFoldToConv1d("flatten"), x):
        return False
    if not has_batch_reshape_param(name):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 4, 5, 7)
    name = "test_ncnn_batch_layout_flatten_to_unbatched_pool2d"
    if not run_model(name, ModelBatchFoldToUnbatchedPool2d(), x):
        return False
    if not has_batch_reshape_param(name, input_axis=0, output_axis=233):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 4, 8, 16)
    name = "test_ncnn_batch_layout_flatten_to_conv1d_batch2"
    if not run_model(name, ModelBatchFoldToConv1d("flatten"), x):
        return False
    if not has_batch_reshape_param(name):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 4, 8, 16)
    name = "test_ncnn_batch_layout_reshape_to_conv1d"
    if not run_model(name, ModelBatchFoldToConv1d("reshape"), x):
        return False
    if not has_batch_reshape_param(name):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 4, 8, 16)
    name = "test_ncnn_batch_layout_channel_fold_to_conv1d"
    if not run_model(name, ModelChannelFoldToConv1d(), x):
        return False
    if not no_batch_reshape_param(name):
        return False

    if version.parse(torch.__version__) >= version.parse('1.13'):
        torch.manual_seed(0)
        x = torch.rand(2, 4, 8, 16)
        name = "test_ncnn_batch_layout_unflatten_after_conv1d"
        if not run_model(name, ModelBatchUnflattenAfterConv1d(), x):
            return False
        if not has_batch_reshape_param(name, layer_prefix=("unflatten_", "Tensor.unflatten_")):
            return False

    for mode in ("flatten", "reshape"):
        for label, shape2 in (("batch", (3, 4, 8, 16)), ("frequency", (2, 4, 5, 16)), ("batch_frames", (3, 4, 5, 19))):
            x = torch.rand(2, 4, 8, 16)
            x2 = torch.rand(*shape2)
            name = "test_ncnn_batch_layout_dynamic_" + mode + "_" + label
            if not run_model(name, ModelBatchFoldToConv1d(mode), x, x2):
                return False
            if not has_batch_reshape_param(name, layer_prefix=(mode + "_", "Tensor." + mode + "_")):
                return False

    if version.parse(torch.__version__) >= version.parse('1.13'):
        name = "test_ncnn_batch_layout_dynamic_unflatten"
        if not run_model(name, ModelDynamicBatchUnflatten(), torch.rand(4, 4, 16), torch.rand(6, 4, 19)):
            return False
        if not has_batch_reshape_param(name, layer_prefix=("unflatten_", "Tensor.unflatten_")):
            return False

    name = "test_ncnn_batch_layout_middle_partition"
    if not run_model(name, ModelMiddleBatchReshapePartition(), torch.arange(48).float().reshape(2, 3, 8)):
        return False
    if not has_batch_reshape_param(name, 1, 1, "reshape_"):
        return False

    if not run_model("test_ncnn_batch_layout_middle_expression", ModelMiddleBatchShapeExpression(), torch.rand(2, 3, 8), torch.rand(2, 5, 12)):
        return False
    if not run_model("test_ncnn_batch_layout_external_reference", ModelExternalShapeReference(), (torch.rand(24), torch.rand(2, 1, 12)), (torch.rand(24), torch.rand(3, 1, 8))):
        return False
    if not run_model("test_ncnn_batch_layout_different_batch_references", ModelDifferentBatchShapeReferences(), (torch.rand(24), torch.rand(2, 1, 4), torch.rand(3, 1, 4)), (torch.rand(24), torch.rand(4, 1, 4), torch.rand(2, 1, 4))):
        return False
    if not run_model("test_ncnn_batch_layout_dynamic_adjacent", ModelDynamicAdjacentReshape(), torch.rand(2, 3, 4, 5), torch.rand(3, 3, 6, 7)):
        return False
    if not run_model("test_ncnn_batch_layout_dynamic_linear", ModelDynamicLinear(), torch.rand(2, 3, 4, 8), torch.rand(3, 5, 7, 8)):
        return False
    if not run_model("test_ncnn_batch_layout_dynamic_unbind", ModelUnbindBeforeBatchLayout(), torch.rand(2, 3, 5, 7), torch.rand(4, 3, 6, 8)):
        return False
    if not run_model("test_ncnn_batch_layout_dynamic_slice", ModelSliceMultiSelectLayout(), torch.rand(2, 3, 5, 7), torch.rand(4, 6, 8, 10)):
        return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
