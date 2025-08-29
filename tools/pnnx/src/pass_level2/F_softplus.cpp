// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_softplus : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 beta
pnnx.Input              input_2     0 1 threshold
aten::softplus          op_0        3 1 input beta threshold out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softplus";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softplus, 100)

class F_softplus_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
Softplus                op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softplus";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["beta"] = 1.f;
        op->params["threshold"] = 20.f;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softplus_onnx, 101)

class F_softplus_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 beta value=%beta
aten::mul               op_1        2 1 input beta a
Softplus                op_2        1 1 a b
prim::Constant          op_3        0 1 beta2 value=%beta
aten::div               op_4        2 1 b beta2 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softplus";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["beta"] = captured_params.at("beta");
        op->params["threshold"] = 20.f;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softplus_onnx_1, 100)

} // namespace pnnx
