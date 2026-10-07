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

            int input_rank = op->inputs[0]->shape.size();
            if (input_rank == 0)
                continue;

            // nn.Linear    4d-2d-4d
            // nn.Linear    5d-2d-5d
            bool insert_reshape = false;
            if (op->type == "nn.Linear" && (input_rank == 4 || input_rank == 5))
            {
                insert_reshape = true;
            }

            if (!insert_reshape)
                continue;

            fprintf(stderr, "insert_reshape_linear %d\n", input_rank);

            matched = true;

            Operand* linear_in = op->inputs[0];
            Operand* linear_out = op->outputs[0];

            Operator* reshape0 = graph.new_operator_before("Reshape", op->name + "_ncnnreshape0", op);
            Operator* reshape1 = graph.new_operator_after("Reshape", op->name + "_ncnnreshape1", op);

            Operand* reshape0_out = graph.new_operand(op->name + "_ncnnreshape0_out");
            Operand* reshape1_in = graph.new_operand(op->name + "_ncnnreshape1_in");

            reshape0->inputs.push_back(linear_in);
            reshape0->outputs.push_back(reshape0_out);
            reshape1->inputs.push_back(reshape1_in);
            reshape1->outputs.push_back(linear_out);

            for (size_t j = 0; j < linear_in->consumers.size(); j++)
            {
                if (linear_in->consumers[j] == op)
                {
                    linear_in->consumers[j] = reshape0;
                    break;
                }
            }
            linear_out->producer = reshape1;

            op->inputs[0] = reshape0_out;
            op->outputs[0] = reshape1_in;

            reshape0_out->producer = reshape0;
            reshape0_out->consumers.push_back(op);
            reshape1_in->producer = op;
            reshape1_in->consumers.push_back(reshape1);

            // fold physical leading dimensions into rows and retain native batch
            const int batch_axis = linear_in->params["__ncnn_batch_axis"].i;
            const auto input_shape = logical_shape(linear_in, 0);
            auto leading_shape = input_shape;
            leading_shape.pop_back();
            if (batch_axis != 233)
                leading_shape.erase(leading_shape.begin() + batch_axis);
            std::vector<std::string> folded_shape = {shape_product(leading_shape), input_shape.back()};
            const int folded_batch_axis = batch_axis == 233 ? 233 : 0;
            reshape0_out->params["__batch_index"] = folded_batch_axis;
            reshape1_in->params["__batch_index"] = folded_batch_axis;
            reshape0_out->params["__ncnn_batch_axis"] = folded_batch_axis;
            reshape1_in->params["__ncnn_batch_axis"] = folded_batch_axis;
            reshape0_out->type = linear_in->type;
            reshape1_in->type = linear_out->type;
            reshape0_out->shape = {-1, linear_in->shape.back()};
            reshape1_in->shape = {-1, linear_out->shape.back()};
            if (batch_axis != 233)
            {
                folded_shape.insert(folded_shape.begin(), input_shape[batch_axis]);
                reshape0_out->shape.insert(reshape0_out->shape.begin(), linear_in->shape[batch_axis]);
                reshape1_in->shape.insert(reshape1_in->shape.begin(), linear_in->shape[batch_axis]);
            }
            write_reshape_shape(reshape0, folded_shape);

            reshape1->inputs.push_back(linear_in);
            linear_in->consumers.push_back(reshape1);
            auto output_shape = logical_shape(linear_in, 1);
            output_shape.back() = std::to_string(linear_out->shape.back());
            write_reshape_shape(reshape1, output_shape);

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
