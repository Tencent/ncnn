// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_reshape_as : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 other
Tensor.reshape_as       op_0        2 1 input other out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "reshape_as";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        const int input_ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;
        const int other_ncnn_batch_axis = op->inputs[1]->params["__ncnn_batch_axis"].i;
        const int output_ncnn_batch_axis = op->outputs[0]->params["__ncnn_batch_axis"].i;

        int shape_rank = (int)op->outputs[0]->shape.size();
        if (shape_rank == 0)
            shape_rank = (int)op->inputs[1]->shape.size();

        if (shape_rank == 0)
        {
            fprintf(stderr, "reshape_as tensor with unknown rank is not supported yet, fallback to other width\n");
            op->params["6"] = "1w";
            if (input_ncnn_batch_axis != 233 || output_ncnn_batch_axis != 233)
            {
                op->params["12"] = input_ncnn_batch_axis;
                op->params["13"] = output_ncnn_batch_axis;
            }
            return;
        }

        if (shape_rank > 5 || (shape_rank == 5 && (other_ncnn_batch_axis == 233 || output_ncnn_batch_axis == 233)))
        {
            fprintf(stderr, "reshape_as to %d-rank physical tensor is not supported by ncnn runtime yet\n", shape_rank);
            if (!op->outputs[0]->shape.empty())
            {
                const std::vector<int>& shape = op->outputs[0]->shape;
                std::string shape_expr = std::to_string(shape[shape.size() - 1]);
                for (int i = (int)shape.size() - 2; i >= 0; i--)
                {
                    shape_expr += ",";
                    shape_expr += std::to_string(shape[i]);
                }
                op->params["6"] = shape_expr;
            }
            else
            {
                op->params["6"] = "1w";
            }

            if (input_ncnn_batch_axis != 233 || output_ncnn_batch_axis != 233)
            {
                op->params["12"] = input_ncnn_batch_axis;
                op->params["13"] = output_ncnn_batch_axis;
            }
            return;
        }

        std::string shape_expr;
        for (int i = shape_rank - 1; i >= 0; i--)
        {
            if (!shape_expr.empty())
                shape_expr += ",";

            if (i == other_ncnn_batch_axis)
            {
                shape_expr += "1n";
                continue;
            }

            int other_axis = i;
            int other_rank = shape_rank;
            if (other_ncnn_batch_axis != 233)
            {
                other_rank -= 1;
                if (other_axis > other_ncnn_batch_axis)
                    other_axis -= 1;
            }

            if (other_rank == 1 && other_axis == 0)
                shape_expr += "1w";
            else if (other_rank == 2 && other_axis == 0)
                shape_expr += "1h";
            else if (other_rank == 2 && other_axis == 1)
                shape_expr += "1w";
            else if (other_rank == 3 && other_axis == 0)
                shape_expr += "1c";
            else if (other_rank == 3 && other_axis == 1)
                shape_expr += "1h";
            else if (other_rank == 3 && other_axis == 2)
                shape_expr += "1w";
            else if (other_rank == 4 && other_axis == 0)
                shape_expr += "1c";
            else if (other_rank == 4 && other_axis == 1)
                shape_expr += "1d";
            else if (other_rank == 4 && other_axis == 2)
                shape_expr += "1h";
            else if (other_rank == 4 && other_axis == 3)
                shape_expr += "1w";
        }

        op->params["6"] = shape_expr;

        if (input_ncnn_batch_axis != 233 || output_ncnn_batch_axis != 233)
        {
            op->params["12"] = input_ncnn_batch_axis;
            op->params["13"] = output_ncnn_batch_axis;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_reshape_as, 20)

} // namespace ncnn

} // namespace pnnx
