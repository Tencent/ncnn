// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_unsqueeze : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.unsqueeze         op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ExpandDims";
    }

    const char* name_str() const
    {
        return "unsqueeze";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        if (captured_params.at("dim").type == 2)
        {
            int dim = captured_params.at("dim").i;
            if (dim == batch_index)
            {
                fprintf(stderr, "unsqueeze batch dim %d is not supported yet!\n", batch_index);
                return;
            }

            int input_rank = op->inputs[0]->shape.size();

            if (input_rank > 4)
            {
                fprintf(stderr, "unsqueeze %d-rank tensor is not supported yet!\n", input_rank);
                return;
            }

            if (dim > batch_index)
                dim -= 1;

            std::vector<int> axes = {dim};
            op->params["3"] = axes;
        }
        else // if (captured_params.at("dim").type == 5)
        {
            std::vector<int> axes = captured_params.at("dim").ai;
            for (size_t i = 0; i < axes.size(); i++)
            {
                if (axes[i] == batch_index)
                {
                    fprintf(stderr, "unsqueeze batch dim %d is not supported yet!\n", batch_index);
                    return;
                }

                if (axes[i] > batch_index)
                    axes[i] -= 1;
            }
            op->params["3"] = axes;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_unsqueeze, 20)

} // namespace ncnn

} // namespace pnnx
