// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_mish : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::mish              op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.mish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_mish, 101)

class F_mish_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 11 value=1
prim::Constant          op_1        0 1 12 value=20
aten::softplus          op_2        3 1 input 11 12 a
aten::tanh              op_3        1 1 a b
aten::mul               op_4        2 1 input b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.mish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_mish_1, 100)

class F_mish_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
F.softplus              op_0        1 1 input a beta=1.0 threshold=20.0
F.tanh                  op_1        1 1 a b
aten::mul               op_2        2 1 input b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.mish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_mish_onnx, 103)

class F_mish_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Mish                    op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.mish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_mish_onnx_1, 100)

} // namespace pnnx
