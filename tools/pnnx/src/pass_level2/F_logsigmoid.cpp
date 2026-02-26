// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_logsigmoid : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::log_sigmoid       op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.logsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_logsigmoid, 100)

class F_logsigmoid_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
aten::sigmoid           op_0        1 1 input a
aten::log               op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.logsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_logsigmoid_onnx, 100)

} // namespace pnnx
