// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_gather : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 index
prim::Constant          op_0        0 1 sparse_grad value=*
aten::gather            op_1        4 1 input dim index sparse_grad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.gather";
    }
};

// torch.export-exported gather usually omits the default sparse_grad (3-input variant)
class torch_gather_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 index
aten::gather            op_1        3 1 input dim index out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.gather";
    }

    void write(Operator* op, const std::map<std::string, Parameter>&, const std::map<std::string, int>&) const
    {
        op->params["sparse_grad"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_gather, 70)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_gather_3, 70)

} // namespace pnnx
