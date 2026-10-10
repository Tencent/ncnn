// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_baddbmm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 batch1
pnnx.Input              input_2     0 1 batch2
pnnx.Input              input_3     0 1 beta
pnnx.Input              input_4     0 1 alpha
aten::baddbmm           op_0        5 1 input batch1 batch2 beta alpha out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.baddbmm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_baddbmm, 90)

} // namespace pnnx
