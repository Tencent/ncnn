// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_transpose : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim0
pnnx.Input              input_2     0 1 dim1
aten::transpose         op_0        3 1 input dim0 dim1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.transpose";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_transpose, 60)

} // namespace pnnx
