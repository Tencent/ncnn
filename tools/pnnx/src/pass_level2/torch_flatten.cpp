// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_flatten : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 start_dim
pnnx.Input              input_2     0 1 end_dim
aten::flatten           op_0        3 1 input start_dim end_dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.flatten";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_flatten, 60)

class torch_flatten_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
Flatten                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.flatten";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axis") != captured_params.end())
        {
            op->params["start_dim"] = captured_params.at("op_0.axis");
        }
        else
        {
            op->params["start_dim"] = 1;
        }
        op->params["end_dim"] = -1;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_flatten_onnx, 60)

} // namespace pnnx
