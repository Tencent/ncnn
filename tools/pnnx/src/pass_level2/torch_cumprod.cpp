// Copyright 2021 Tencent
// Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_cumprod : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim 
prim::Constant          op_1        0 1 dtype value=%dtype
aten::cumprod           op_2        3 1 input dim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.cumprod";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // preserve an explicit dtype (e.g. torch.cumprod(x, 1, dtype=torch.float64)):
        // dynamo emits it as a prim::Constant input; without propagating it the
        // generated code would accumulate in the input's default dtype instead
        if (captured_params.at("dtype").type == 0)
        {
            op->params["dtype"] = Parameter();
        }
        else
        {
            if (captured_params.at("dtype").i == 0) op->params["dtype"] = "torch.uint8";
            if (captured_params.at("dtype").i == 1) op->params["dtype"] = "torch.int8";
            if (captured_params.at("dtype").i == 2) op->params["dtype"] = "torch.short";
            if (captured_params.at("dtype").i == 3) op->params["dtype"] = "torch.int";
            if (captured_params.at("dtype").i == 4) op->params["dtype"] = "torch.long";
            if (captured_params.at("dtype").i == 5) op->params["dtype"] = "torch.half";
            if (captured_params.at("dtype").i == 6) op->params["dtype"] = "torch.float";
            if (captured_params.at("dtype").i == 7) op->params["dtype"] = "torch.double";
            if (captured_params.at("dtype").i == 8) op->params["dtype"] = "torch.complex32";
            if (captured_params.at("dtype").i == 9) op->params["dtype"] = "torch.complex64";
            if (captured_params.at("dtype").i == 10) op->params["dtype"] = "torch.complex128";
            if (captured_params.at("dtype").i == 11) op->params["dtype"] = "torch.bool";
            if (captured_params.at("dtype").i == 15) op->params["dtype"] = "torch.bfloat16";
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_cumprod, 90)

} // namespace pnnx
