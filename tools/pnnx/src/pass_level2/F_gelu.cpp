// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

#if _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

namespace pnnx {

class F_gelu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::gelu              op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.gelu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu, 101)

class F_gelu_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 approximate value=%approximate
aten::gelu              op_1        2 1 input approximate out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.gelu";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& approximate = captured_params.at("approximate").s;
        if (approximate != "none")
            op->params["approximate"] = approximate;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu_1, 101)

class F_gelu_1_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::gelu_approximate_none op_0    1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.gelu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu_1_1, 101)

class F_gelu_2 : public GraphRewriterPass
{
public:
    // x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 12 value=%0p5
aten::mul               op_1        2 1 input 12 13
prim::Constant          op_2        0 1 15 value=%sqrt2
aten::div               op_3        2 1 input 15 16
aten::erf               op_4        1 1 16 17
prim::Constant          op_5        0 1 20 value=%1
prim::Constant          op_6        0 1 21 value=1
aten::add               op_7        3 1 17 20 21 22
aten::mul               op_8        2 1 13 22 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("0p5").f != 0.5f)
            return false;

        if (fabs(captured_params.at("sqrt2").f - sqrt(2.f)) > 0.0001f)
            return false;

        if ((captured_params.at("1").type == 2 && captured_params.at("1").i != 1) || (captured_params.at("1").type == 3 && captured_params.at("1").f != 1.f))
            return false;

        return true;
    }

    const char* type_str() const
    {
        return "F.gelu";
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu_2, 100)

class F_gelu_3 : public GraphRewriterPass
{
public:
    // 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 60 value=%0p5
aten::mul               op_1        2 1 input 60 26
prim::Constant          op_2        0 1 28 value=%3
aten::pow               op_3        2 1 input 28 29
prim::Constant          op_4        0 1 30 value=%0p044715
aten::mul               op_5        2 1 29 30 31
prim::Constant          op_6        0 1 61 value=1
aten::add               op_7        3 1 input 31 61 35
prim::Constant          op_8        0 1 36 value=%sqrt2dpi
aten::mul               op_9        2 1 35 36 37
aten::tanh              op_10       1 1 37 39
prim::Constant          op_11       0 1 62 value=%1
prim::Constant          op_12       0 1 63 value=%1_1
aten::add               op_13       3 1 39 62 63 42
aten::mul               op_14       2 1 26 42 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("0p5").f != 0.5f)
            return false;

        if (fabs(captured_params.at("0p044715").f - 0.044715f) > 0.0001f)
            return false;

        if (fabs(captured_params.at("sqrt2dpi").f - sqrt(2.f / M_PI)) > 0.0001f)
            return false;

        if ((captured_params.at("1").type == 2 && captured_params.at("1").i != 1) || (captured_params.at("1").type == 3 && captured_params.at("1").f != 1.f))
            return false;

        if ((captured_params.at("3").type == 2 && captured_params.at("3").i != 3) || (captured_params.at("3").type == 3 && captured_params.at("3").f != 3.f))
            return false;

        if ((captured_params.at("1_1").type == 2 && captured_params.at("1_1").i != 1) || (captured_params.at("1_1").type == 3 && captured_params.at("1_1").f != 1.f))
            return false;

        return true;
    }

    const char* type_str() const
    {
        return "F.gelu";
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu_3, 100)

class F_gelu_4 : public GraphRewriterPass
{
public:
    // (x * 0.5) * (tanh((x + (torch.pow(x, 3.0) * 0.044715)) * 7.978846e-01) + 1)
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 zp5 value=0.5
aten::mul               op_1        2 1 input zp5 pnnx_90
prim::Constant          op_2        0 1 three value=%3
aten::pow               op_3        2 1 input three pnnx_91
prim::Constant          op_4        0 1 zp044715 value=%0p044715
aten::mul               op_5        2 1 pnnx_91 zp044715 pnnx_92
aten::add               op_6        2 1 input pnnx_92 pnnx_93
prim::Constant          op_7        0 1 sqrt2dpi value=%sqrt2dpi
aten::mul               op_8        2 1 pnnx_93 sqrt2dpi pnnx_94
aten::tanh              op_9        1 1 pnnx_94 pnnx_95
prim::Constant          op_10       0 1 one value=%1
aten::add               op_11       2 1 pnnx_95 one pnnx_96
aten::mul               op_12       2 1 pnnx_90 pnnx_96 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (fabs(captured_params.at("0p044715").f - 0.044715f) > 0.0001f)
            return false;

        if (fabs(captured_params.at("sqrt2dpi").f - sqrt(2.f / M_PI)) > 0.0001f)
            return false;

        if ((captured_params.at("1").type == 2 && captured_params.at("1").i != 1) || (captured_params.at("1").type == 3 && captured_params.at("1").f != 1.f))
            return false;

        if ((captured_params.at("3").type == 2 && captured_params.at("3").i != 3) || (captured_params.at("3").type == 3 && captured_params.at("3").f != 3.f))
            return false;

        return true;
    }

    const char* type_str() const
    {
        return "F.gelu";
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu_4, 100)

class F_gelu_onnx : public GraphRewriterPass
{
public:
    // x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 sqrt2 value=%sqrt2
aten::div               op_1        2 1 input sqrt2 16
aten::erf               op_2        1 1 16 17
prim::Constant          op_3        0 1 one value=%1
aten::add               op_4        2 1 17 one 22
aten::mul               op_5        2 1 input 22 25
prim::Constant          op_6        0 1 0p5 value=%0p5
aten::mul               op_7        2 1 25 0p5 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("0p5").f != 0.5f)
            return false;

        if (fabs(captured_params.at("sqrt2").f - sqrt(2.f)) > 0.0001f)
            return false;

        if (captured_params.at("1").f != 1.f)
            return false;

        return true;
    }

    const char* type_str() const
    {
        return "F.gelu";
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu_onnx, 100)

class F_gelu_onnx_1 : public F_gelu_onnx
{
public:
    // (x * 0.5) * (1.0 + torch.erf(x / math.sqrt(2.0)))
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 0p5 value=%0p5
aten::mul               op_1        2 1 input 0p5 15
prim::Constant          op_2        0 1 sqrt2 value=%sqrt2
aten::div               op_3        2 1 input sqrt2 16
aten::erf               op_4        1 1 16 17
prim::Constant          op_5        0 1 one value=%1
aten::add               op_6        2 1 17 one 22
aten::mul               op_7        2 1 15 22 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu_onnx_1, 100)

class F_gelu_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Gelu                    op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.gelu";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.approximate") != captured_params.end())
        {
            const std::string& approximate = captured_params.at("op_0.approximate").s;
            if (approximate != "none")
                op->params["approximate"] = approximate;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_gelu_onnx_2, 101)

} // namespace pnnx
