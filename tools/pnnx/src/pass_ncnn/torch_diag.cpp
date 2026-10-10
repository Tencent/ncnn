// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_diag : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.diag              op_0        1 1 input out diagonal=%diagonal
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Diag";
    }

    const char* name_str() const
    {
        return "diag";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int diagonal = captured_params.at("diagonal").i;
        int input_rank = op->inputs[0]->shape.size();

        if (input_rank > 2)
        {
            fprintf(stderr, "diag %d-rank tensor is not supported yet!\n", input_rank);
            return;
        }

        op->params["0"] = diagonal;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_diag, 20)

} // namespace ncnn

} // namespace pnnx
