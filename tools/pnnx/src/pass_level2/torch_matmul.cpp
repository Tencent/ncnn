// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_matmul : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 other
aten::matmul            op_0        2 1 input other out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.matmul";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_matmul, 90)

class torch_matmul_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 other
MatMul                  op_0        2 1 input other out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.matmul";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_matmul_onnx, 90)

class torch_matmul_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 other
tnn.MatMul              op_0        2 1 input other out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.matmul";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.arg0") != captured_params.end())
        {
            const int weight_position = captured_params.at("op_0.arg0").i;
            if (weight_position == 0)
            {
                // swap input and weight
                std::swap(op->inputs[0], op->inputs[1]);
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_matmul_tnn, 90)

} // namespace pnnx
