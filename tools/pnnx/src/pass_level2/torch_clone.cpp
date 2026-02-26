// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_clone : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 memory_format value=%memory_format
aten::clone             op_1        2 1 input memory_format out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.clone";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("memory_format").i == 0)
            op->params["memory_format"] = "torch.contiguous_format";
        if (captured_params.at("memory_format").i == 1)
            op->params["memory_format"] = "torch.preserve_format";
        if (captured_params.at("memory_format").i == 2)
            op->params["memory_format"] = "torch.channels_last";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_clone, 20)

class torch_clone_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::clone             op_1        1 1 input out memory_format=%memory_format
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.clone";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("memory_format").type == 4 && captured_params.at("memory_format").s.empty())
        {
            op->params["memory_format"] = "torch.contiguous_format";
        }
        else
        {
            if (captured_params.at("memory_format").i == 0)
                op->params["memory_format"] = "torch.contiguous_format";
            if (captured_params.at("memory_format").i == 1)
                op->params["memory_format"] = "torch.preserve_format";
            if (captured_params.at("memory_format").i == 2)
                op->params["memory_format"] = "torch.channels_last";
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_clone_1, 20)

} // namespace pnnx
