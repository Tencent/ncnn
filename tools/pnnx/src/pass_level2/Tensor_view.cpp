// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_view : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 shape
aten::view              op_0        2 1 input shape out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.view";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_view, 60)

} // namespace pnnx
