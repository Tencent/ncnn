// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_reshape_linear.h"
#include "reshape_shape.h"

namespace pnnx {

namespace ncnn {

void insert_reshape_linear(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.Linear")
                continue;

            const int input_rank = (int)op->inputs[0]->shape.size();
            const int ncnn_batch_axis = op->inputs[0]->params.at("__ncnn_batch_axis").i;
            const int physical_rank = input_rank - (ncnn_batch_axis != 233 ? 1 : 0);
            if (physical_rank <= 2 || physical_rank > 4)
                continue;

            // linear applies to the last dimension, flatten the physical leading dimensions
            fprintf(stderr, "insert_reshape_linear %d\n", input_rank);

            Operand* linear_in = op->inputs[0];
            Operand* linear_out = op->outputs[0];

            // fold physical leading dimensions into rows and retain native batch
            const auto input_shape = get_logical_shape_expr(linear_in, 0);
            auto leading_shape = input_shape;
            leading_shape.pop_back();
            if (ncnn_batch_axis != 233)
                leading_shape.erase(leading_shape.begin() + ncnn_batch_axis);
            std::vector<std::string> folded_shape = {make_shape_product_expr(leading_shape), input_shape.back()};
            const int folded_batch_axis = ncnn_batch_axis == 233 ? 233 : 0;
            std::vector<int> reshape0_shape = {-1, linear_in->shape.back()};
            std::vector<int> reshape1_shape = {-1, linear_out->shape.back()};
            if (ncnn_batch_axis != 233)
            {
                folded_shape.insert(folded_shape.begin(), input_shape[ncnn_batch_axis]);
                reshape0_shape.insert(reshape0_shape.begin(), linear_in->shape[ncnn_batch_axis]);
                reshape1_shape.insert(reshape1_shape.begin(), linear_in->shape[ncnn_batch_axis]);
            }

            auto output_shape = get_logical_shape_expr(linear_in, 1);
            output_shape.back() = std::to_string(linear_out->shape.back());
            std::map<std::string, Parameter> params0;
            std::map<std::string, Parameter> params1;
            const auto folded_output_shape = get_logical_shape_expr((int)reshape1_shape.size(), folded_batch_axis, 0);
            if (!resolve_reshape_params(input_shape, get_ncnn_batch_axis(linear_in), folded_shape, folded_batch_axis, 1, params0)
                    || !resolve_reshape_params(folded_output_shape, folded_batch_axis, output_shape, get_ncnn_batch_axis(linear_out), 2, params1))
                continue;

            matched = true;

            Operator* reshape0 = graph.new_operator_before("Reshape", op->name + "_ncnnreshape0", op);
            Operator* reshape1 = graph.new_operator_after("Reshape", op->name + "_ncnnreshape1", op);

            Operand* reshape0_out = graph.new_operand(op->name + "_ncnnreshape0_out");
            Operand* reshape1_in = graph.new_operand(op->name + "_ncnnreshape1_in");

            reshape0->inputs.push_back(linear_in);
            reshape0->outputs.push_back(reshape0_out);
            reshape1->inputs.push_back(reshape1_in);
            reshape1->inputs.push_back(linear_in);
            reshape1->outputs.push_back(linear_out);

            for (size_t j = 0; j < linear_in->consumers.size(); j++)
            {
                if (linear_in->consumers[j] == op)
                {
                    linear_in->consumers[j] = reshape0;
                    break;
                }
            }
            linear_in->consumers.push_back(reshape1);
            linear_out->producer = reshape1;

            op->inputs[0] = reshape0_out;
            op->outputs[0] = reshape1_in;

            reshape0_out->producer = reshape0;
            reshape0_out->consumers.push_back(op);
            reshape1_in->producer = op;
            reshape1_in->consumers.push_back(reshape1);

            reshape0_out->params = linear_in->params;
            reshape0_out->params["__batch_index"] = folded_batch_axis;
            reshape0_out->params["__ncnn_batch_axis"] = folded_batch_axis;
            reshape0_out->shape = reshape0_shape;
            reshape0_out->type = linear_in->type;
            reshape1_in->params = linear_out->params;
            reshape1_in->params["__batch_index"] = folded_batch_axis;
            reshape1_in->params["__ncnn_batch_axis"] = folded_batch_axis;
            reshape1_in->shape = reshape1_shape;
            reshape1_in->type = linear_out->type;
            write_reshape_params(reshape0, params0);
            write_reshape_params(reshape1, params1);

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
