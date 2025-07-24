// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_celu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 alpha
aten::celu              op_0        2 1 input alpha out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.celu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_celu, 100)

class F_celu_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Celu                    op_0        1 1 input out alpha=%alpha
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.celu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_celu_onnx, 100)

} // namespace pnnx
