// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_AdaptiveAvgPool2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.AdaptiveAvgPool2d    op_0        1 1 input out output_size=(1,1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling";
    }

    const char* name_str() const
    {
        return "gap";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 1;
        op->params["4"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_AdaptiveAvgPool2d, 20)

class nn_AdaptiveAvgPool2d_n : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.AdaptiveAvgPool2d    op_0        1 1 input out output_size=%output_size
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling";
    }

    const char* name_str() const
    {
        return "aap";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& output_size = captured_params.at("output_size").ai;

        op->params["0"] = 1;
        op->params["7"] = 1;
        op->params["8"] = output_size[1] == 0 ? -233 : output_size[1];
        op->params["18"] = output_size[0] == 0 ? -233 : output_size[0];
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_AdaptiveAvgPool2d_n, 21)

} // namespace ncnn

} // namespace pnnx
