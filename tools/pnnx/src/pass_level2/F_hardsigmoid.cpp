// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

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

class F_hardsigmoid : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::hardsigmoid       op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid, 100)

class F_hardsigmoid_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 410 value=3
prim::Constant          op_1        0 1 412 value=1
aten::add               op_2        3 1 input 410 412 a
prim::Constant          op_3        0 1 413 value=0
prim::Constant          op_4        0 1 414 value=6
torch.clamp             op_5        3 1 a 413 414 b
prim::Constant          op_6        0 1 409 value=6
aten::div               op_7        2 1 b 409 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_2, 100)

class F_hardsigmoid_2_1 : public F_hardsigmoid_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 410 value=3
aten::add               op_1        2 1 input 410 a
prim::Constant          op_2        0 1 413 value=0
prim::Constant          op_3        0 1 414 value=6
torch.clamp             op_4        3 1 a 413 414 b
prim::Constant          op_5        0 1 409 value=6
aten::div               op_6        2 1 b 409 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_2_1, 100)

class F_hardsigmoid_2_2 : public F_hardsigmoid_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 410 value=3
prim::Constant          op_1        0 1 412 value=1
aten::add               op_2        3 1 input 410 412 a
prim::Constant          op_3        0 1 413 value=0
prim::Constant          op_4        0 1 414 value=6
torch.clamp             op_5        3 1 a 413 414 b
prim::Constant          op_6        0 1 409 value=%v1p6
aten::mul               op_7        2 1 b 409 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float v1p6 = captured_params.at("v1p6").f;
        return NearlyEqual(v1p6, 1.f / 6, 0.001);
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_2_2, 100)

class F_hardsigmoid_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 12 value=3
prim::Constant          op_1        0 1 13 value=1
aten::add               op_2        3 1 input 12 13 a
prim::Constant          op_3        0 1 16 value=0
prim::Constant          op_4        0 1 17 value=6
aten::hardtanh          op_5        3 1 a 16 17 b
prim::Constant          op_6        0 1 19 value=6
aten::div               op_7        2 1 b 19 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_3, 100)

class F_hardsigmoid_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 12 value=3
prim::Constant          op_1        0 1 13 value=1
aten::add               op_2        3 1 input 12 13 a
aten::relu6             op_3        1 1 a b
prim::Constant          op_4        0 1 19 value=6
aten::div               op_5        2 1 b 19 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_4, 100)

class F_hardsigmoid_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
prim::Constant          op_1        0 1 22 value=1
prim::Constant          op_2        0 1 23 value=3
aten::add               op_3        3 1 input 23 22 a
nn.ReLU6                op_4        1 1 a b
prim::Constant          op_0        0 1 21 value=6
aten::div               op_5        2 1 b 21 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_5, 100)

class F_hardsigmoid_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
HardSigmoid             op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = 0.2f;
        if (captured_params.find("op_0.alpha") != captured_params.end())
        {
            alpha = captured_params.at("op_0.alpha").f;
        }

        float beta = 0.5f;
        if (captured_params.find("op_0.beta") != captured_params.end())
        {
            beta = captured_params.at("op_0.beta").f;
        }

        return NearlyEqual(alpha, 1.f / 6, 0.001) && NearlyEqual(beta, 0.5f, 0.001);
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_onnx, 101)

class F_hardsigmoid_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
HardSigmoid             op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (alpha_scale && !beta_offset)
            return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          alpha2      0 1 alpha2
aten::mul               mul         2 1 input alpha2 a
F.hardsigmoid           hs          1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        else if (!alpha_scale && beta_offset)
            return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          beta2       0 1 beta2
aten::add               add         2 1 input beta2 a
F.hardsigmoid           hs          1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        else // if (alpha_scale && beta_offset)
            return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
prim::Constant          alpha2      0 1 alpha2
aten::mul               mul         2 1 input alpha2 a
prim::Constant          beta2       0 1 beta2
aten::add               add         2 1 a beta2 b
F.hardsigmoid           hs          1 1 b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = 0.2f;
        if (captured_params.find("op_0.alpha") != captured_params.end())
        {
            alpha = captured_params.at("op_0.alpha").f;
        }

        float beta = 0.5f;
        if (captured_params.find("op_0.beta") != captured_params.end())
        {
            beta = captured_params.at("op_0.beta").f;
        }

        alpha_scale = !NearlyEqual(alpha, 1.f / 6, 0.001);
        beta_offset = !NearlyEqual(beta, 0.5f, 0.001);

        return alpha_scale || beta_offset;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = 0.2f;
        if (captured_params.find("op_0.alpha") != captured_params.end())
        {
            alpha = captured_params.at("op_0.alpha").f;
        }

        float beta = 0.5f;
        if (captured_params.find("op_0.beta") != captured_params.end())
        {
            beta = captured_params.at("op_0.beta").f;
        }

        if (alpha_scale)
            ops.at("alpha2")->params["value"] = alpha / (1.f / 6);
        if (beta_offset)
            ops.at("beta2")->params["value"] = (beta - 0.5f) / (1.f / 6);
    }

protected:
    mutable bool alpha_scale;
    mutable bool beta_offset;
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_onnx_1, 101)

class F_hardsigmoid_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scalar_tensor_default_8 value=3.0
aten::add               op_1        2 1 input scalar_tensor_default_8 pnnx_8
torch.clamp             op_2        1 1 pnnx_8 pnnx_9 max=6.0 min=0.0
prim::Constant          op_3        0 1 max_val_cast value=6.0
aten::div               op_4        2 1 pnnx_9 max_val_cast out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_onnx_2, 100)

class F_hardsigmoid_onnx_3 : public F_hardsigmoid_2_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scalar_tensor_default_8_pnnxshadow3 value=3.0
aten::add               op_1        2 1 input scalar_tensor_default_8_pnnxshadow3 pnnx_16
torch.clamp             op_2        1 1 pnnx_16 pnnx_17 max=6.0 min=0.0
prim::Constant          op_3        0 1 val_12 value=%v1p6
aten::mul               op_4        2 1 pnnx_17 val_12 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_onnx_3, 100)

} // namespace pnnx
