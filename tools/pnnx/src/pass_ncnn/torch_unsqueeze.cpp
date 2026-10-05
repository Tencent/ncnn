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
        const int input_ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;
        int input_rank = op->inputs[0]->shape.size();
        if (input_rank == 0 && op->outputs[0]->shape.size() != 0)
            input_rank = (int)op->outputs[0]->shape.size() - 1;

        int inner_rank = input_rank;
        if (input_ncnn_batch_axis >= 0 && input_ncnn_batch_axis < input_rank)
            inner_rank -= 1;

        if (inner_rank > 4)
        {
            fprintf(stderr, "unsqueeze %d-rank tensor is not supported yet!\n", inner_rank);
        }

        if (captured_params.at("dim").type == 2)
        {
            int dim = captured_params.at("dim").i;
            if (dim < 0 && input_rank > 0)
                dim += input_rank + 1;

            const int output_ncnn_batch_axis = op->outputs[0]->params["__ncnn_batch_axis"].i;
            if (output_ncnn_batch_axis != 233 && dim > output_ncnn_batch_axis)
                dim -= 1;

            std::vector<int> axes = {dim};
            op->params["3"] = axes;
        }
        else // if (captured_params.at("dim").type == 5)
        {
            std::vector<int> axes = captured_params.at("dim").ai;
            int output_rank = op->outputs[0]->shape.size();
            if (output_rank == 0)
                output_rank = input_rank + axes.size();

            const int output_ncnn_batch_axis = op->outputs[0]->params["__ncnn_batch_axis"].i;
            std::vector<int> new_axes;
            for (size_t i = 0; i < axes.size(); i++)
            {
                int axis = axes[i];
                if (axis < 0 && output_rank > 0)
                    axis += output_rank;

                if (output_ncnn_batch_axis >= 0 && output_ncnn_batch_axis < output_rank)
                {
                    if (axis == output_ncnn_batch_axis)
                        continue;
                    if (axis > output_ncnn_batch_axis)
                        axis -= 1;
                }
                else if (input_ncnn_batch_axis != 233 && axis > input_ncnn_batch_axis)
                    axis -= 1;

                new_axes.push_back(axis);
            }
            if (new_axes.empty())
            {
                op->type = "Noop";
                return;
            }
            op->params["3"] = new_axes;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_unsqueeze, 20)

} // namespace ncnn

} // namespace pnnx
