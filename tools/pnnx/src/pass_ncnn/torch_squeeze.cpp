// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_squeeze : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.squeeze           op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Squeeze";
    }

    const char* name_str() const
    {
        return "squeeze";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

        int input_rank = op->inputs[0]->shape.size();

        if (input_rank > 5)
        {
            fprintf(stderr, "squeeze %d-rank tensor is not supported yet!\n", input_rank);
        }

        if (captured_params.at("dim").type == 2)
        {
            int dim = captured_params.at("dim").i;
            if (dim < 0 && input_rank > 0)
                dim += input_rank;

            if (dim == ncnn_batch_axis)
            {
                int output_ncnn_batch_axis = 233;
                if (!op->inputs[0]->shape.empty() && dim >= 0 && dim < (int)op->inputs[0]->shape.size() && op->inputs[0]->shape[dim] != 1)
                    output_ncnn_batch_axis = ncnn_batch_axis;
                if (output_ncnn_batch_axis == ncnn_batch_axis)
                {
                    op->inputs[0]->params["__ncnn_batch_axis"] = output_ncnn_batch_axis;
                    op->outputs[0]->params["__ncnn_batch_axis"] = output_ncnn_batch_axis;
                    op->type = "Noop";
                    return;
                }
                if (op->inputs[0]->shape.empty())
                {
                    fprintf(stderr, "squeeze along batch axis %d with unknown shape is not supported yet\n", ncnn_batch_axis);
                    op->params["3"] = std::vector<int>{dim};
                    return;
                }

                op->outputs[0]->params["__ncnn_batch_axis"] = output_ncnn_batch_axis;
                op->type = "Noop";
                return;
            }

            if (ncnn_batch_axis != 233 && dim > ncnn_batch_axis)
                dim -= 1;

            std::vector<int> axes = {dim};
            op->params["3"] = axes;
        }
        else // if (captured_params.at("dim").type == 5)
        {
            std::vector<int> axes = captured_params.at("dim").ai;
            std::vector<int> new_axes;
            for (size_t i = 0; i < axes.size(); i++)
            {
                int dim = axes[i];
                if (dim < 0 && input_rank > 0)
                    dim += input_rank;

                if (dim == ncnn_batch_axis)
                {
                    continue;
                }

                if (ncnn_batch_axis != 233 && dim > ncnn_batch_axis)
                    dim -= 1;

                new_axes.push_back(dim);
            }

            if (new_axes.empty())
            {
                fprintf(stderr, "squeeze along batch axis %d is not supported yet\n", ncnn_batch_axis);
                op->params["3"] = axes;
                return;
            }

            op->params["3"] = new_axes;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_squeeze, 20)

class torch_squeeze_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.squeeze           op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Squeeze";
    }

    const char* name_str() const
    {
        return "squeeze";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 1;
        op->params["1"] = 1;
        op->params["11"] = 1;
        op->params["2"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_squeeze_0, 20)

} // namespace ncnn

} // namespace pnnx
