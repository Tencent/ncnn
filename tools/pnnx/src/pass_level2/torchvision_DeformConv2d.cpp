// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torchvision_DeformConv2d_export : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input       0 1 input
pnnx.Attribute          weight      0 1 weight @data=(%out_channels,%in_channels_per_group,%kh,%kw)f32
pnnx.Input              offset      0 1 offset
pnnx.Input              mask        0 1 mask
pnnx.Attribute          bias        0 1 bias @data=(%out_channels)f32
prim::Constant          stride_w    0 1 stride_w value=%stride_w
prim::Constant          stride_h    0 1 stride_h value=%stride_h
prim::Constant          pad_w       0 1 pad_w value=%pad_w
prim::Constant          pad_h       0 1 pad_h value=%pad_h
prim::Constant          dilation_w  0 1 dilation_w value=%dilation_w
prim::Constant          dilation_h  0 1 dilation_h value=%dilation_h
prim::Constant          groups      0 1 groups value=%groups
prim::Constant          offset_groups 0 1 offset_groups value=*
prim::Constant          use_mask    0 1 use_mask value=False
torchvision::deform_conv2d op_0     14 1 input weight offset mask bias stride_w stride_h pad_w pad_h dilation_w dilation_h groups offset_groups use_mask out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchvision.ops.DeformConv2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["in_channels"] = captured_params.at("in_channels_per_group").i * captured_params.at("groups").i;
        op->params["out_channels"] = captured_params.at("out_channels");
        op->params["kernel_size"] = Parameter{captured_params.at("kh").i, captured_params.at("kw").i};
        op->params["stride"] = Parameter{captured_params.at("stride_h").i, captured_params.at("stride_w").i};
        op->params["padding"] = Parameter{captured_params.at("pad_h").i, captured_params.at("pad_w").i};
        op->params["dilation"] = Parameter{captured_params.at("dilation_h").i, captured_params.at("dilation_w").i};
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = true;
        op->attrs["weight"] = captured_attrs.at("weight.data");
        op->attrs["bias"] = captured_attrs.at("bias.data");

        if (!use_mask())
        {
            Operand* mask = op->inputs[2];
            mask->remove_consumer(op);
            op->inputs.resize(2);
        }
        op->inputnames = use_mask() ? std::vector<std::string>{"input", "offset", "mask"} : std::vector<std::string>{"input", "offset"};
    }

protected:
    virtual bool use_mask() const { return false; }
};

class torchvision_DeformConv2d_export_mask : public torchvision_DeformConv2d_export
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input       0 1 input
pnnx.Attribute          weight      0 1 weight @data=(%out_channels,%in_channels_per_group,%kh,%kw)f32
pnnx.Input              offset      0 1 offset
pnnx.Input              mask        0 1 mask
pnnx.Attribute          bias        0 1 bias @data=(%out_channels)f32
prim::Constant          stride_w    0 1 stride_w value=%stride_w
prim::Constant          stride_h    0 1 stride_h value=%stride_h
prim::Constant          pad_w       0 1 pad_w value=%pad_w
prim::Constant          pad_h       0 1 pad_h value=%pad_h
prim::Constant          dilation_w  0 1 dilation_w value=%dilation_w
prim::Constant          dilation_h  0 1 dilation_h value=%dilation_h
prim::Constant          groups      0 1 groups value=%groups
prim::Constant          offset_groups 0 1 offset_groups value=*
prim::Constant          use_mask    0 1 use_mask value=True
torchvision::deform_conv2d op_0     14 1 input weight offset mask bias stride_w stride_h pad_w pad_h dilation_w dilation_h groups offset_groups use_mask out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

protected:
    bool use_mask() const { return true; }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchvision_DeformConv2d_export, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchvision_DeformConv2d_export_mask, 110)

} // namespace pnnx
