// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_convert_swiglu.h"

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class fuse_swiglu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 gate
Swish                   op_0        1 1 gate silu_gate
pnnx.Input              input_1     0 1 up
BinaryOp                op_1        2 1 silu_gate up out 0=2
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "SwiGLU";
    }

    const char* name_str() const
    {
        return "swiglu";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op_swish = matched_operators.at("op_0");
        const Operator* op_mul = matched_operators.at("op_1");

        // pnnx_graph_rewrite already rejects a Swish output with extra consumers
        const Operand* silu_gate = op_swish->outputs[0];
        const Operand* gate = op_swish->inputs[0];
        const Operand* up = op_mul->inputs[0] == silu_gate ? op_mul->inputs[1] : op_mul->inputs[0];

        // SwiGLU requires two inputs with identical shape, reject broadcast cases
        // matching dynamic dims are accepted like fuse_binaryop_eltwise,
        // so llm graphs exported with a dynamic sequence length still fuse
        return !gate->shape.empty() && gate->shape == up->shape;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        // spell out the defaults, so the fused node does not silently follow
        // future changes of the SwiGLU layer defaults
        op->params["0"] = 1.f;
        op->params["1"] = 0.f;
        op->params["2"] = 0.f;
    }
};

// up * silu(gate)
class fuse_swiglu_pass_1 : public fuse_swiglu_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 gate
Swish                   op_0        1 1 gate silu_gate
pnnx.Input              input_1     0 1 up
BinaryOp                op_1        2 1 up silu_gate out 0=2
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_convert_swiglu(Graph& graph)
{
    fuse_swiglu_pass a;
    fuse_swiglu_pass_1 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace ncnn

} // namespace pnnx
