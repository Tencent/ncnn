// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pass_level2.h"

namespace pnnx {

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

class F_hardsigmoid_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
HardSigmoid             op_0        1 1 input out alpha=%alpha
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = captured_params.at("alpha").f;
        return NearlyEqual(alpha, 1.f / 6, 0.001);
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
HardSigmoid             op_0        1 1 input out alpha=%alpha beta=%beta
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardsigmoid";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = captured_params.at("alpha").f;
        float beta = captured_params.at("beta").f;
        return NearlyEqual(alpha, 1.f / 6, 0.001) && NearlyEqual(beta, 0.5f, 0.001);
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_onnx_1, 101)

class F_hardsigmoid_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
HardSigmoid             op_0        1 1 input out alpha=%alpha
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          alpha2      0 1 alpha2
aten::mul               mul         2 1 input alpha2 a
F.hardsigmoid           hs          1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = captured_params.at("alpha").f;
        return !NearlyEqual(alpha, 1.f / 6, 0.001);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        const float alpha = captured_params.at("alpha").f;
        ops.at("alpha2")->params["value"] = alpha / (1.f / 6);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_onnx_2, 101)

class F_hardsigmoid_onnx_3 : public F_hardsigmoid_onnx_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
HardSigmoid             op_0        1 1 input out alpha=%alpha beta=%beta
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = captured_params.at("alpha").f;
        float beta = captured_params.at("beta").f;
        return !NearlyEqual(alpha, 1.f / 6, 0.001) && NearlyEqual(beta, 0.5f, 0.001);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardsigmoid_onnx_3, 101)

} // namespace pnnx
