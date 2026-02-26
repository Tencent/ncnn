// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_einsum : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 equation
pnnx.Input              input_1     0 1 operands
aten::einsum            op_0        2 1 equation operands out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.einsum";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_einsum, 90)

class torch_einsum_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 equation
pnnx.Input              input_1     0 1 operands
pnnx.Input              input_2     0 1 path
aten::einsum            op_0        3 1 equation operands path out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.einsum";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        // drop path input
        op->inputs[2]->remove_consumer(op);
        op->inputs.resize(2);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_einsum_1, 90)

class torch_einsum_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 operands
aten::einsum            op_0        1 1 operands out equation=%equation
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.einsum";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_einsum_onnx, 90)

} // namespace pnnx
