// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_elu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 alpha
prim::Constant          op_0        0 1 scale value=1
prim::Constant          op_1        0 1 input_scale value=1
aten::elu               op_2        4 1 input alpha scale input_scale out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.elu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_elu, 100)

class F_elu_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Elu                     op_0        1 1 input out alpha=%alpha
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.elu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_elu_onnx, 100)

} // namespace pnnx
