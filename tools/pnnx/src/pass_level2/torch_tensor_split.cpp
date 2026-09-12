// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_tensor_split_indices : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 indices value=%indices
aten::tensor_split      op_2        3 1 input indices dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tensor_split";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("indices").type == 5;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tensor_split_indices, 60)

class torch_tensor_split_sections : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 sections value=%sections
aten::tensor_split      op_2        3 1 input sections dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tensor_split";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("sections").type == 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tensor_split_sections, 60)

// dynamo output order is input indices/sections dim (indices/sections first, dim last)
class torch_tensor_split_indices_direct : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 indices
pnnx.Input              input_2     0 1 dim
aten::tensor_split      op_0        3 1 input indices dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tensor_split";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tensor_split_indices_direct, 60)

class torch_tensor_split_sections_direct : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 sections
pnnx.Input              input_2     0 1 dim
aten::tensor_split      op_0        3 1 input sections dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tensor_split";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tensor_split_sections_direct, 60)

} // namespace pnnx
