// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_clamp : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 min
pnnx.Input              input_2     0 1 max
aten::clamp             op_0        3 1 input min max out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.clamp";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_clamp, 40)

class torch_clamp_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Clip                    op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.clamp";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.min") != captured_params.end())
        {
            op->params["min"] = captured_params.at("op_0.min");
        }
        else
        {
            op->params["min"] = Parameter();
        }
        if (captured_params.find("op_0.max") != captured_params.end())
        {
            op->params["max"] = captured_params.at("op_0.max");
        }
        else
        {
            op->params["max"] = Parameter();
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_clamp_onnx, 40)

class torch_clamp_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.Clip                op_0        1 1 input out arg0=%min arg1=%max
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.clamp";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_clamp_tnn, 40)

} // namespace pnnx
