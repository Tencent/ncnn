// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_relu6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::relu6             op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.relu6";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_relu6, 100)

class F_relu6_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
F.relu                  op_0        1 1 input pnnx_8
prim::Constant          op_1        0 1 val_3 value=6.0
aten::min               op_2        2 1 pnnx_8 val_3 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.relu6";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_relu6_onnx, 101)

} // namespace pnnx
