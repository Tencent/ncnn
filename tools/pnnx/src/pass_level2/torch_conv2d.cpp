// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_conv2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // pt2: nn.Conv2d is expanded by dynamo into aten::conv2d (weight is an Attribute input)
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
prim::Constant          op_0        0 1 stride value=%stride
prim::Constant          op_1        0 1 padding value=%padding
prim::Constant          op_2        0 1 dilation value=%dilation
prim::Constant          op_3        0 1 groups value=%groups
aten::conv2d            op_4        6 1 input weight stride padding dilation groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("stride").type == 5 && captured_params.at("stride").ai.size() == 2;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);
        // the pass_ncnn F_conv2d_4 pattern expects a bias=None argument
        op->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_conv2d, 20)

class torch_conv2d_bias : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
prim::Constant          op_0        0 1 stride value=%stride
prim::Constant          op_1        0 1 padding value=%padding
prim::Constant          op_2        0 1 dilation value=%dilation
prim::Constant          op_3        0 1 groups value=%groups
aten::conv2d            op_4        7 1 input weight bias stride padding dilation groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("stride").type == 5 && captured_params.at("stride").ai.size() == 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_conv2d_bias, 20)

} // namespace pnnx
