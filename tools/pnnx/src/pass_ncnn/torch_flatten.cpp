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
torch.flatten           op_0        1 1 input out start_dim=%start_dim end_dim=%end_dim
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op = matched_operators.at("op_0");
        const int input_rank = op->inputs[0]->shape.size();
        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

        const int start_dim = captured_params.at("start_dim").i;
        if ((start_dim == 0 && ncnn_batch_axis == 233) || (start_dim == 1 && ncnn_batch_axis == 0))
        {
            const int end_dim = captured_params.at("end_dim").i;
            if (end_dim == -1)
                return true;

            if (end_dim == input_rank - 1)
                return true;
        }

        return false;
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
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
torch.flatten           op_0        1 1 input out start_dim=%start_dim end_dim=%end_dim
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
        int end_dim = captured_params.at("end_dim").i;

        const int input_rank = op->inputs[0]->shape.size();
        const int input_ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;
        const int output_ncnn_batch_axis = op->outputs[0]->params["__ncnn_batch_axis"].i;

        std::vector<int> new_shape = op->outputs[0]->shape;

        if (new_shape.empty())
        {
            if (input_rank == 0)
            {
                fprintf(stderr, "flatten unknown-rank tensor is not supported yet, fallback to flatten all\n");
                new_shape.push_back(-1);
            }
            else
            {
                if (start_dim < 0)
                    start_dim += input_rank;

                if (end_dim < 0)
                    end_dim += input_rank;

                if (input_rank <= start_dim || input_rank <= end_dim)
                {
                    fprintf(stderr, "flatten %d to %d not possible for %d-rank tensor, fallback to flatten all\n", start_dim, end_dim, input_rank);
                    new_shape.push_back(-1);
                }
                else
                {
                    std::vector<int> shape_flattened;
                    for (int i = 0; i < start_dim; i++)
                    {
                        shape_flattened.push_back(op->inputs[0]->shape[i]);
                    }
                    int flattened_dimsize = 1;
                    for (int i = start_dim; i <= end_dim; i++)
                    {
                        if (op->inputs[0]->shape[i] == -1)
                        {
                            // flatten includes dynamic axis
                            flattened_dimsize = -1;
                            break;
                        }

                        flattened_dimsize *= op->inputs[0]->shape[i];
                    }
                    shape_flattened.push_back(flattened_dimsize);
                    for (int i = end_dim + 1; i < input_rank; i++)
                    {
                        shape_flattened.push_back(op->inputs[0]->shape[i]);
                    }

                    for (int i = 0; i < (int)shape_flattened.size(); i++)
                    {
                        new_shape.push_back(shape_flattened[i]);
                    }
                }
            }
        }

        if (new_shape.size() == 5 && output_ncnn_batch_axis == 233)
        {
            if (new_shape[0] == 1)
            {
                fprintf(stderr, "assume flatten 5-rank tensor has batch_index 0\n");
                new_shape.erase(new_shape.begin());
            }
        }

        int shape_rank = (int)new_shape.size();
        if (shape_rank == 0)
        {
            fprintf(stderr, "flatten to unknown-rank tensor is not supported yet, fallback to flatten all\n");
            new_shape.push_back(-1);
            shape_rank = 1;
        }

        // handle multiple dynamic dimension
        int dynamic_dimension_count = 0;
        for (size_t i = 0; i < new_shape.size(); i++)
        {
            if (new_shape[i] == -1)
                dynamic_dimension_count++;
        }

        if (dynamic_dimension_count > 1)
        {
            const int flattened_index = start_dim;

            int in_shape_rank = op->inputs[0]->shape.size();

            std::string shape_expr;
            for (int i = shape_rank - 1; i >= 0; i--)
            {
                if (!shape_expr.empty())
                    shape_expr += ",";

                if (i == flattened_index)
                {
                    shape_expr += "-1";
                    continue;
                }

                int input_axis = i;
                if (i > flattened_index)
                    input_axis += end_dim - start_dim;

                if (input_axis == input_ncnn_batch_axis)
                {
                    shape_expr += "0n";
                    continue;
                }

                int rank = in_shape_rank;
                if (input_ncnn_batch_axis != 233)
                {
                    rank -= 1;
                    if (input_axis > input_ncnn_batch_axis)
                        input_axis -= 1;
                }

                if (rank == 1 && input_axis == 0)
                    shape_expr += "0w";
                else if (rank == 2 && input_axis == 0)
                    shape_expr += "0h";
                else if (rank == 2 && input_axis == 1)
                    shape_expr += "0w";
                else if (rank == 3 && input_axis == 0)
                    shape_expr += "0c";
                else if (rank == 3 && input_axis == 1)
                    shape_expr += "0h";
                else if (rank == 3 && input_axis == 2)
                    shape_expr += "0w";
                else if (rank == 4 && input_axis == 0)
                    shape_expr += "0c";
                else if (rank == 4 && input_axis == 1)
                    shape_expr += "0d";
                else if (rank == 4 && input_axis == 2)
                    shape_expr += "0h";
                else if (rank == 4 && input_axis == 3)
                    shape_expr += "0w";
            }

            if (shape_expr.empty())
            {
                fprintf(stderr, "flatten dynamic shape is not supported yet, fallback to flatten all\n");
                shape_expr = "-1";
            }

            op->params["6"] = shape_expr;
            if (input_ncnn_batch_axis != 233 || output_ncnn_batch_axis != 233)
            {
                op->params["12"] = input_ncnn_batch_axis;
                op->params["13"] = output_ncnn_batch_axis;
            }
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
        if (shape_rank >= 5)
        {
            if (shape_rank > 5 || (shape_rank == 5 && output_ncnn_batch_axis == 233))
                fprintf(stderr, "reshape to %d-rank physical tensor is not supported by ncnn runtime yet\n", shape_rank);

            std::string shape_expr = std::to_string(new_shape[shape_rank - 1]);
            for (int i = shape_rank - 2; i >= 0; i--)
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

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_flatten_2, 21)

} // namespace ncnn

} // namespace pnnx
