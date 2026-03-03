// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_softshrink : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 lambd
aten::softshrink        op_0        2 1 input lambd out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softshrink";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softshrink, 100)

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

class F_softshrink_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 lambd value=%lambd
torch.gt                op_1        2 1 input lambd 8
prim::Constant          op_2        0 1 lambd2 value=%lambd
aten::sub               op_3        2 1 input lambd2 9
prim::Constant          op_4        0 1 zero value=0
torch.where             op_5        3 1 8 9 zero a
prim::Constant          op_6        0 1 mlambd value=%lambd2
torch.lt                op_7        2 1 input mlambd 11
prim::Constant          op_8        0 1 lambd3 value=%lambd
aten::add               op_9        2 1 input lambd3 12
prim::Constant          op_10       0 1 zero2 value=0
torch.where             op_11       3 1 11 12 zero2 b
aten::add               op_12       2 1 a b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softshrink";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float lambd = captured_params.at("lambd").f;
        float lambd2 = captured_params.at("lambd2").f;
        return NearlyEqual(lambd, -lambd2, 0.001);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["lambd"] = captured_params.at("lambd");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softshrink_onnx, 100)

class F_softshrink_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Shrink                  op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softshrink";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float bias = 0.f;
        if (captured_params.find("op_0.bias") != captured_params.end())
        {
            bias = captured_params.at("op_0.bias").f;
        }

        float lambda = 0.5f;
        if (captured_params.find("op_0.lambda") != captured_params.end())
        {
            lambda = captured_params.at("op_0.lambda").f;
        }

        return bias == lambda;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float lambda = 0.5f;
        if (captured_params.find("op_0.lambda") != captured_params.end())
        {
            lambda = captured_params.at("op_0.lambda").f;
        }

        op->params["lambd"] = lambda;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softshrink_onnx_1, 100)

class F_softshrink_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input       0 1 input
aten::abs               op_0        1 1 input abs
prim::Constant          op_1        0 1 val_2 value=%lambd
torch.gt                op_2        2 1 abs val_2 pnnx_9
aten::sign              op_3        1 1 input pnnx_10
prim::Constant          op_4        0 1 val_2_pnnxshadow1 value=%lambd
aten::mul               op_5        2 1 pnnx_10 val_2_pnnxshadow1 pnnx_11
aten::sub               op_6        2 1 input pnnx_11 pnnx_12
prim::Constant          op_7        0 1 zero value=0.0
aten::mul               op_8        2 1 input zero zero2
torch.where             op_9        3 1 pnnx_9 pnnx_12 zero2 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softshrink";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["lambd"] = captured_params.at("lambd");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softshrink_onnx_2, 100)

} // namespace pnnx
