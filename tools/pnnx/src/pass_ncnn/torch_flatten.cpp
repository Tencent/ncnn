// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_flatten : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.flatten           op_0        1 1 input out start_dim=1 end_dim=-1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Flatten";
    }

    const char* name_str() const
    {
        return "flatten";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_flatten, 20)

class torch_flatten_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.flatten           op_0        1 1 input out start_dim=%start_dim end_dim=-1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "flatten";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int start_dim = captured_params.at("start_dim").i;

        const int input_rank = op->inputs[0]->shape.size();

        if (start_dim < 0)
            start_dim += input_rank;

        if (input_rank <= start_dim)
        {
            fprintf(stderr, "flatten %d to -1 not possible for %d-rank tensor\n", start_dim, input_rank);
            return;
        }

        op->params["0"] = -1;
        op->params["1"] = op->inputs[0]->shape[1];
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_flatten_2, 20)

} // namespace ncnn

} // namespace pnnx
