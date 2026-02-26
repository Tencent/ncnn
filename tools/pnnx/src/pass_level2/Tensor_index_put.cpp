// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_index_put : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 indices
pnnx.Input              input_2     0 1 values
prim::Constant          op_0        0 1 accumulate value=%accumulate
aten::index_put         op_1        4 1 input indices values accumulate out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.index_put";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_index_put, 70)

} // namespace pnnx
