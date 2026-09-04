// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_new_full : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
pnnx.Input              input_2     0 1 fill_value
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 pin_memory value=*
aten::new_full          op_4        7 1 input size fill_value dtype layout device pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.new_full";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const Parameter& dtype = captured_params.at("dtype");
        if (dtype.type == 0) op->params["dtype"] = Parameter();
        if (dtype.type == 2 && dtype.i == 0) op->params["dtype"] = "torch.uint8";
        if (dtype.type == 2 && dtype.i == 1) op->params["dtype"] = "torch.int8";
        if (dtype.type == 2 && dtype.i == 2) op->params["dtype"] = "torch.short";
        if (dtype.type == 2 && dtype.i == 3) op->params["dtype"] = "torch.int";
        if (dtype.type == 2 && dtype.i == 4) op->params["dtype"] = "torch.long";
        if (dtype.type == 2 && dtype.i == 5) op->params["dtype"] = "torch.half";
        if (dtype.type == 2 && dtype.i == 6) op->params["dtype"] = "torch.float";
        if (dtype.type == 2 && dtype.i == 7) op->params["dtype"] = "torch.double";
        if (dtype.type == 2 && dtype.i == 8) op->params["dtype"] = "torch.complex32";
        if (dtype.type == 2 && dtype.i == 9) op->params["dtype"] = "torch.complex64";
        if (dtype.type == 2 && dtype.i == 10) op->params["dtype"] = "torch.complex128";
        if (dtype.type == 2 && dtype.i == 11) op->params["dtype"] = "torch.bool";
        if (dtype.type == 2 && dtype.i == 15) op->params["dtype"] = "torch.bfloat16";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_new_full, 20)

} // namespace pnnx