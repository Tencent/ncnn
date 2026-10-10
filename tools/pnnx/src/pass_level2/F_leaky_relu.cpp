// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_leaky_relu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 negative_slope
aten::leaky_relu        op_0        2 1 input negative_slope out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.leaky_relu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_leaky_relu, 100)

class F_leaky_relu_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
LeakyRelu               op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.leaky_relu";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.alpha") != captured_params.end())
        {
            op->params["negative_slope"] = captured_params.at("op_0.alpha");
        }
        else
        {
            op->params["negative_slope"] = 0.01f;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_leaky_relu_onnx, 100)

} // namespace pnnx
