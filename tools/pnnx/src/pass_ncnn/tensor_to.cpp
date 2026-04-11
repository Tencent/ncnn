// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_to : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 2
pnnx.Input              input_0     0 1 input
Tensor.to               op_0        1 1 input out copy=%copy dtype=%dtype
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Cast";
    }

    const char* name_str() const
    {
        return "to";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // Map torch dtype to ncnn cast type
        // torch.float = 1 (float32), torch.int64 = 5 (int64), torch.int32 = 6 (int32), etc.
        // The input type is auto-detected, we only need to set the target type
        std::string dtype = "torch.float";
        if (captured_params.find("dtype") != captured_params.end())
        {
            dtype = captured_params.at("dtype").s;
        }

        int type_to = 0;
        if (dtype == "torch.float" || dtype == "torch.float32")
            type_to = 1;
        else if (dtype == "torch.float16" || dtype == "torch.half")
            type_to = 2;
        else if (dtype == "torch.int8")
            type_to = 3;
        else if (dtype == "torch.bfloat16")
            type_to = 4;
        else if (dtype == "torch.int64" || dtype == "torch.long")
            type_to = 5;
        else if (dtype == "torch.int32" || dtype == "torch.int")
            type_to = 6;

        op->params["0"] = 0; // auto-detect input type
        op->params["1"] = type_to;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_to, 20)

} // namespace ncnn

} // namespace pnnx
