// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_std : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 unbiased value=%unbiased
prim::Constant          op_1        0 1 keepdim value=%keepdim
aten::std               op_2        4 1 input dim unbiased keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("unbiased").type == 1;
    }

    const char* type_str() const
    {
        return "torch.std";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_std, 50)

class torch_std_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 correction value=%correction
prim::Constant          op_1        0 1 keepdim value=%keepdim
aten::std               op_2        4 1 input dim correction keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("correction").type == 2;
    }

    const char* type_str() const
    {
        return "torch.std";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_std_1, 50)

} // namespace pnnx
