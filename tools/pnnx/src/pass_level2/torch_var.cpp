// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_var : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 unbiased
pnnx.Input              input_3     0 1 keepdim
aten::var               op_0        4 1 input dim unbiased keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.var";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_var, 50)

class torch_var_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // var.correction overload: self dim correction keepdim
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 correction value=%correction
prim::Constant          op_1        0 1 keepdim value=%keepdim
aten::var               op_2        4 1 input dim correction keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        // correction is an integer for var(x, dim, correction=0/1/...), but
        // torch also accepts a fractional scalar (var(x, dim, correction=0.5))
        // which is serialized as a float Parameter; accept both
        int t = captured_params.at("correction").type;
        return t == 2 || t == 3;
    }

    const char* type_str() const
    {
        return "torch.var";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_var_1, 49)

class torch_var_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // reduce-all torch.var(x) is exported under var.correction with no dim
        // input: self correction keepdim
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 correction value=%correction
prim::Constant          op_1        0 1 keepdim value=%keepdim
aten::var               op_2        3 1 input correction keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        // accept integer and fractional corrections (see torch_var_1)
        int t = captured_params.at("correction").type;
        return t == 2 || t == 3;
    }

    const char* type_str() const
    {
        return "torch.var";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_var_2, 49)

} // namespace pnnx
