// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_hann_window : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 window_length
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 pin_memory value=*
aten::hann_window       op_4        5 1 window_length dtype layout device pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.hann_window";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("periodic") != captured_params.end())
            op->params["periodic"] = captured_params.at("periodic");

        const Parameter& dtype = captured_params.at("dtype");
        if (dtype.type == 0)
        {
            op->params["dtype"] = Parameter();
        }
        else
        {
            op->params["dtype"] = "torch.float";
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_hann_window, 20)

class torch_hann_window_periodic : public torch_hann_window
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 window_length
prim::Constant          op_0        0 1 periodic value=%periodic
prim::Constant          op_1        0 1 dtype value=%dtype
prim::Constant          op_2        0 1 layout value=*
prim::Constant          op_3        0 1 device value=*
prim::Constant          op_4        0 1 pin_memory value=*
aten::hann_window       op_5        6 1 window_length periodic dtype layout device pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_hann_window_periodic, 20)

} // namespace pnnx
