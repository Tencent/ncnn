// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torchvision_DeformConv2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 offset
pnnx.Input              input_2     0 1 mask
pnnx.Attribute          weight      0 1 weight @data=(%out_channels,%weight_in_channels,%kernel_h,%kernel_w)f32
pnnx.Attribute          bias        0 1 bias @data=(%out_channels)f32
prim::Constant          op_0        0 1 stride_h value=%stride_h
prim::Constant          op_1        0 1 stride_w value=%stride_w
prim::Constant          op_2        0 1 pad_h value=%pad_h
prim::Constant          op_3        0 1 pad_w value=%pad_w
prim::Constant          op_4        0 1 dilation_h value=%dilation_h
prim::Constant          op_5        0 1 dilation_w value=%dilation_w
prim::Constant          op_6        0 1 groups value=%groups
prim::Constant          op_7        0 1 offset_groups value=*
prim::Constant          op_8        0 1 use_mask value=True
torchvision::deform_conv2d op_9      14 1 input weight offset mask bias stride_h stride_w pad_h pad_w dilation_h dilation_w groups offset_groups use_mask out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchvision.ops.DeformConv2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const int groups = captured_params.at("groups").i;
        op->params["in_channels"] = captured_params.at("weight_in_channels").i * groups;
        op->params["out_channels"] = captured_params.at("out_channels");
        op->params["kernel_size"] = {captured_params.at("kernel_h").i, captured_params.at("kernel_w").i};
        op->params["stride"] = {captured_params.at("stride_h").i, captured_params.at("stride_w").i};
        op->params["padding"] = {captured_params.at("pad_h").i, captured_params.at("pad_w").i};
        op->params["dilation"] = {captured_params.at("dilation_h").i, captured_params.at("dilation_w").i};
        op->params["groups"] = groups;
        op->params["bias"] = true;
        op->attrs["weight"] = captured_attrs.at("weight.data");
        op->attrs["bias"] = captured_attrs.at("bias.data");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchvision_DeformConv2d, 140)

class torchvision_DeformConv2d_nomask : public torchvision_DeformConv2d
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 offset
pnnx.Input              input_2     0 1 mask
pnnx.Attribute          weight      0 1 weight @data=(%out_channels,%weight_in_channels,%kernel_h,%kernel_w)f32
pnnx.Attribute          bias        0 1 bias @data=(%out_channels)f32
prim::Constant          op_0        0 1 stride_h value=%stride_h
prim::Constant          op_1        0 1 stride_w value=%stride_w
prim::Constant          op_2        0 1 pad_h value=%pad_h
prim::Constant          op_3        0 1 pad_w value=%pad_w
prim::Constant          op_4        0 1 dilation_h value=%dilation_h
prim::Constant          op_5        0 1 dilation_w value=%dilation_w
prim::Constant          op_6        0 1 groups value=%groups
prim::Constant          op_7        0 1 offset_groups value=*
prim::Constant          op_8        0 1 use_mask value=False
torchvision::deform_conv2d op_9      14 1 input weight offset mask bias stride_h stride_w pad_h pad_w dilation_h dilation_w groups offset_groups use_mask out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        torchvision_DeformConv2d::write(op, captured_params, captured_attrs);
        op->inputs[2]->remove_consumer(op);
        op->inputs.resize(2);
        if (!op->inputnames.empty())
            op->inputnames.resize(2);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchvision_DeformConv2d_nomask, 140)

} // namespace pnnx
