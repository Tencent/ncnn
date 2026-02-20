// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_where : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 condition
pnnx.Input              input_1     0 1 input
pnnx.Input              input_2     0 1 other
torch.where             op_0        3 1 condition input other out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Where";
    }

    const char* name_str() const
    {
        return "where";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_where, 20)

} // namespace ncnn

} // namespace pnnx
