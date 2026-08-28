// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

#include <limits.h>

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
pnnx.Attribute          weight      0 1 weight @data
pnnx.Attribute          bias        0 1 bias @data
prim::Constant          op_0        0 1 stride_h value=%stride_h
prim::Constant          op_1        0 1 stride_w value=%stride_w
prim::Constant          op_2        0 1 pad_h value=%pad_h
prim::Constant          op_3        0 1 pad_w value=%pad_w
prim::Constant          op_4        0 1 dilation_h value=%dilation_h
prim::Constant          op_5        0 1 dilation_w value=%dilation_w
prim::Constant          op_6        0 1 groups value=%groups
prim::Constant          op_7        0 1 offset_groups value=%offset_groups
prim::Constant          op_8        0 1 use_mask value=%use_mask
torchvision::deform_conv2d op_9     14 1 input weight offset mask bias stride_h stride_w pad_h pad_w dilation_h dilation_w groups offset_groups use_mask out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchvision.ops.DeformConv2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const char* int_params[] = {"stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w", "groups", "offset_groups"};
        for (size_t i = 0; i < sizeof(int_params) / sizeof(int_params[0]); i++)
        {
            const std::map<std::string, Parameter>::const_iterator it = captured_params.find(int_params[i]);
            if (it == captured_params.end() || it->second.type != 2)
                return false;
        }

        const std::map<std::string, Parameter>::const_iterator use_mask = captured_params.find("use_mask");
        if (use_mask == captured_params.end() || use_mask->second.type != 1)
            return false;

        const Attribute& weight = captured_attrs.at("weight.data");
        const Attribute& bias = captured_attrs.at("bias.data");
        return weight.shape.size() == 4
               && bias.shape.size() == 1
               && weight.shape[0] > 0
               && weight.shape[1] > 0
               && weight.shape[2] > 0
               && weight.shape[3] > 0
               && bias.shape[0] == weight.shape[0]
               && captured_params.at("groups").i > 0
               && captured_params.at("groups").i <= INT_MAX / weight.shape[1]
               && captured_params.at("offset_groups").i > 0;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Attribute& weight = captured_attrs.at("weight.data");
        const int groups = captured_params.at("groups").i;
        op->params["in_channels"] = weight.shape[1] * groups;
        op->params["out_channels"] = weight.shape[0];
        op->params["kernel_size"] = Parameter{weight.shape[2], weight.shape[3]};
        op->params["stride"] = Parameter{captured_params.at("stride_h").i, captured_params.at("stride_w").i};
        op->params["padding"] = Parameter{captured_params.at("pad_h").i, captured_params.at("pad_w").i};
        op->params["dilation"] = Parameter{captured_params.at("dilation_h").i, captured_params.at("dilation_w").i};
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = true;
        op->attrs["weight"] = weight;
        op->attrs["bias"] = captured_attrs.at("bias.data");

        if (!captured_params.at("use_mask").b)
        {
            for (size_t i = 0; i < op->inputnames.size(); i++)
            {
                if (op->inputnames[i] != "mask")
                    continue;

                op->inputs[i]->remove_consumer(op);
                op->inputs.erase(op->inputs.begin() + i);
                op->inputnames.erase(op->inputnames.begin() + i);
                break;
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchvision_DeformConv2d, 20)

} // namespace pnnx
