// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

// x < constants
class torch_lt_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.lt                torch.lt_0  1 1 input out other=%other
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "CompareOp";
    }

    const char* name_str() const
    {
        return "lt";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float other = captured_params.at("other").type == 3 ? captured_params.at("other").f : captured_params.at("other").i;

        op->params["0"] = 0; // op_type LT
        op->params["1"] = 1; // with_scalar
        op->params["2"] = other; // b
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_lt_0, 20)

// x < y
class torch_lt_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_x       0 1 x
pnnx.Input              input_y       0 1 y
torch.lt                torch.lt_0    2 1 x y out
pnnx.Output             output        1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "CompareOp";
    }

    const char* name_str() const
    {
        return "lt";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = 0; // op_type LT
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_lt_1, 20)

} // namespace ncnn

} // namespace pnnx
