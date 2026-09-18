// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_slice_scatter : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 src
torch.slice_scatter     op_0        2 1 input src out dim=%dim start=%start end=%end step=%step
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "CopyTo";
    }

    const char* name_str() const
    {
        return "slice_scatter";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

        int dim = captured_params.at("dim").i;
        int input_rank = op->inputs[0]->shape.size();
        if (input_rank == 0)
            input_rank = op->outputs[0]->shape.size();
        if (dim < 0 && input_rank > 0)
            dim += input_rank;

        bool axis_is_batch = false;
        if (ncnn_batch_axis != 233 && dim == ncnn_batch_axis)
        {
            fprintf(stderr, "slice_scatter batch dim %d is not supported yet!\n", ncnn_batch_axis);
            axis_is_batch = true;
        }

        int start = captured_params.at("start").type == 2 ? captured_params.at("start").i : 0;
        // int end = captured_params.at("end").type == 2 ? captured_params.at("end").i : INT_MAX;
        int step = captured_params.at("step").type == 2 ? captured_params.at("step").i : 1;
        if (step != 1)
        {
            fprintf(stderr, "slice_scatter step %d is not supported yet!\n", step);
        }

        if (input_rank > 5)
        {
            fprintf(stderr, "slice_scatter %d-rank tensor is not supported yet!\n", input_rank);
        }

        if (!axis_is_batch && ncnn_batch_axis != 233 && dim > ncnn_batch_axis)
            dim -= 1;

        op->params["9"] = std::vector<int>{start};
        // op->params["10"] = ends; // ncnn always resolve ends from src blob
        if (!axis_is_batch)
            op->params["11"] = std::vector<int>{dim};
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_slice_scatter, 20)

} // namespace ncnn

} // namespace pnnx
