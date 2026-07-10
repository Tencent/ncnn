// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_gather : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 index
torch.gather            op_0        2 1 input index out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Gather";
    }

    const char* name_str() const
    {
        return "gather";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int axis = 0;
        if (captured_params.find("dim") != captured_params.end())
        {
            const Parameter& dim_p = captured_params.at("dim");
            if (dim_p.type == 2)
                axis = dim_p.i;
            else if (dim_p.type == 5 && !dim_p.ai.empty())
                axis = dim_p.ai[0];
        }

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        if (axis == batch_index)
        {
            fprintf(stderr, "Gather along batch axis is not supported\n");
            return;
        }

        int new_axis = axis;
        if (axis >= 0)
            new_axis = axis > batch_index ? axis - 1 : axis;

        op->params["0"] = new_axis;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_gather, 20)

} // namespace ncnn

} // namespace pnnx
