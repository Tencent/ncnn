// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_contiguous : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 memory_format value=%memory_format
aten::contiguous        op_1        2 1 input memory_format out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.contiguous";
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_contiguous, 60)

} // namespace pnnx
