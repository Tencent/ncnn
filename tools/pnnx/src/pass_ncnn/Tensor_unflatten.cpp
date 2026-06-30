// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_unflatten : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.unflatten         op_0        1 1 input out dim=%dim sizes=%sizes
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "unflatten";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim = captured_params.at("dim").i;
        std::vector<int> sizes = captured_params.at("sizes").ai;

        const int input_rank = op->inputs[0]->shape.size();

        if (dim < 0)
            dim += input_rank;

        if (input_rank <= dim)
        {
            fprintf(stderr, "unflatten %d not possible for %d-rank tensor\n", dim, input_rank);
            return;
        }

        const std::vector<int> shape = op->outputs[0]->shape;

        const int input_ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;
        const int output_ncnn_batch_axis = op->outputs[0]->params["__ncnn_batch_axis"].i;

        std::vector<int> new_shape = shape;

        if (new_shape.size() == 5 && output_ncnn_batch_axis == 233)
        {
            if (new_shape[0] == 1)
            {
                fprintf(stderr, "assume reshape 5-rank tensor has batch_index 0\n");
                new_shape.erase(new_shape.begin());
            }
        }

        const int shape_rank = (int)new_shape.size();

        if (shape_rank > 5 || (shape_rank == 5 && output_ncnn_batch_axis == 233))
        {
            fprintf(stderr, "reshape to %d-rank tensor is not supported yet!\n", shape_rank);
            return;
        }

        if (shape_rank == 1)
        {
            op->params["0"] = new_shape[0];
        }
        if (shape_rank == 2)
        {
            op->params["0"] = new_shape[1];
            op->params["1"] = new_shape[0];
        }
        if (shape_rank == 3)
        {
            op->params["0"] = new_shape[2];
            op->params["1"] = new_shape[1];
            op->params["2"] = new_shape[0];
        }
        if (shape_rank == 4)
        {
            op->params["0"] = new_shape[3];
            op->params["1"] = new_shape[2];
            op->params["11"] = new_shape[1];
            op->params["2"] = new_shape[0];
        }
        if (shape_rank == 5)
        {
            std::string shape_expr = std::to_string(new_shape[4]);
            for (int i = 3; i >= 0; i--)
            {
                shape_expr += ",";
                shape_expr += std::to_string(new_shape[i]);
            }
            op->params["6"] = shape_expr;
        }

        if (input_ncnn_batch_axis != 233 || output_ncnn_batch_axis != 233)
        {
            op->params["12"] = input_ncnn_batch_axis;
            op->params["13"] = output_ncnn_batch_axis;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_unflatten, 20)

} // namespace ncnn

} // namespace pnnx
