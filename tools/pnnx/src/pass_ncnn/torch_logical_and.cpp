// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_logical_and : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 a
pnnx.Input              input_1     0 1 b
torch.logical_and       op_0        2 1 a b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Logical";
    }

    const char* name_str() const
    {
        return "logical_and";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 1; // op_type AND
        op->params["1"] = 0; // with_scalar
        op->params["2"] = 0; // b
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_logical_and, 20)

} // namespace ncnn

} // namespace pnnx
