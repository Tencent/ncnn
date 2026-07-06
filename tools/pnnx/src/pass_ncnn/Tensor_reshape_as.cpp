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
        const bool batch_reshape = input_ncnn_batch_axis != output_ncnn_batch_axis;

        std::vector<int> shape = op->outputs[0]->shape;
        int shape_ncnn_batch_axis = output_ncnn_batch_axis;
        if (shape.empty())
        {
            shape = op->inputs[1]->shape;
            shape_ncnn_batch_axis = other_ncnn_batch_axis;
        }

        if (shape.empty())
        {
            fprintf(stderr, "reshape_as tensor with unknown rank is not supported yet, fallback to other width\n");
            op->params["6"] = "1w";
            if (batch_reshape)
            {
                op->params["12"] = input_ncnn_batch_axis;
                op->params["13"] = output_ncnn_batch_axis;
            }
            return;
        }

        std::vector<int> new_shape = shape;
        if (!batch_reshape && shape_ncnn_batch_axis != 233 && shape_ncnn_batch_axis >= 0 && shape_ncnn_batch_axis < (int)new_shape.size())
            new_shape.erase(new_shape.begin() + shape_ncnn_batch_axis);

        const int shape_rank = (int)shape.size();
        const int new_shape_rank = (int)new_shape.size();
        if (new_shape_rank > 5 || (new_shape_rank == 5 && (!batch_reshape || other_ncnn_batch_axis == 233 || output_ncnn_batch_axis == 233)))
        {
            fprintf(stderr, "reshape_as to %d-rank physical tensor is not supported by ncnn runtime yet\n", new_shape_rank);
            if (!new_shape.empty())
            {
                std::string shape_expr = std::to_string(new_shape[new_shape_rank - 1]);
                for (int i = new_shape_rank - 2; i >= 0; i--)
                {
                    shape_expr += ",";
                    shape_expr += std::to_string(new_shape[i]);
                }
                op->params["6"] = shape_expr;
            }
            else
            {
                op->params["6"] = "1w";
            }

            if (batch_reshape)
            {
                op->params["12"] = input_ncnn_batch_axis;
                op->params["13"] = output_ncnn_batch_axis;
            }
            return;
        }

        // use static output shape when other only serves as shape reference
        if (!batch_reshape && other_ncnn_batch_axis != output_ncnn_batch_axis && !op->outputs[0]->shape.empty())
        {
            int dynamic_count = 0;
            for (int x : new_shape)
            {
                if (x == -1)
                    dynamic_count++;
            }

            if (dynamic_count <= 1)
            {
                op->inputs[1]->remove_consumer(op);
                op->inputs.resize(1);
                op->inputnames.resize(1);

                const int rank = (int)new_shape.size();
                if (rank == 1)
                {
                    op->params["0"] = new_shape[0];
                }
                if (rank == 2)
                {
                    op->params["0"] = new_shape[1];
                    op->params["1"] = new_shape[0];
                }
                if (rank == 3)
                {
                    op->params["0"] = new_shape[2];
                    op->params["1"] = new_shape[1];
                    op->params["2"] = new_shape[0];
                }
                if (rank == 4)
                {
                    op->params["0"] = new_shape[3];
                    op->params["1"] = new_shape[2];
                    op->params["11"] = new_shape[1];
                    op->params["2"] = new_shape[0];
                }
                if (rank >= 5)
                {
                    std::string shape_expr = std::to_string(new_shape[rank - 1]);
                    for (int i = rank - 2; i >= 0; i--)
                    {
                        shape_expr += ",";
                        shape_expr += std::to_string(new_shape[i]);
                    }
                    op->params["6"] = shape_expr;
                }

                return;
            }
        }

        std::string shape_expr;
        bool shape_expr_reference_batch = false;
        for (int i = shape_rank - 1; i >= 0; i--)
        {
            if (!batch_reshape && i == output_ncnn_batch_axis)
            {
                if (other_ncnn_batch_axis == output_ncnn_batch_axis)
                    continue;

                if (!shape_expr.empty())
                    shape_expr += ",";
                shape_expr += "0n";
                shape_expr_reference_batch = true;
                continue;
            }

            if (!shape_expr.empty())
                shape_expr += ",";

            if (i == other_ncnn_batch_axis)
            {
                shape_expr += "1n";
                shape_expr_reference_batch = true;
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

        if (batch_reshape || (shape_expr_reference_batch && (input_ncnn_batch_axis != 233 || output_ncnn_batch_axis != 233)))
        {
            op->params["12"] = input_ncnn_batch_axis;
            op->params["13"] = output_ncnn_batch_axis;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_reshape_as, 20)

} // namespace ncnn

} // namespace pnnx
