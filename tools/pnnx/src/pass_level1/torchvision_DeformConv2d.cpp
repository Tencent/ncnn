// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class DeformConv2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torchvision.ops.deform_conv.DeformConv2d";
    }

    const char* type_str() const
    {
        return "torchvision.ops.DeformConv2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        const TorchNodeProxy* deform_conv2d = graph.find_node_by_kind("torchvision::deform_conv2d");

        const TorchTensorProxy& weight = mod.attr("weight");

        const Parameter stride_w = deform_conv2d->namedInput("stride_w");
        const Parameter stride_h = deform_conv2d->namedInput("stride_h");
        const Parameter pad_w = deform_conv2d->namedInput("pad_w");
        const Parameter pad_h = deform_conv2d->namedInput("pad_h");
        const Parameter dilation_w = deform_conv2d->namedInput("dilation_w");
        const Parameter dilation_h = deform_conv2d->namedInput("dilation_h");

        op->params["groups"] = deform_conv2d->namedInput("groups");
        op->params["in_channels"] = weight.size(1) * op->params["groups"].i;
        op->params["out_channels"] = weight.size(0);
        op->params["kernel_size"] = Parameter{weight.size(2), weight.size(3)};
        op->params["stride"] = {stride_h.i, stride_w.i};
        op->params["padding"] = {pad_h.i, pad_w.i};
        op->params["dilation"] = {dilation_h.i, dilation_w.i};
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(DeformConv2d)

} // namespace pnnx
