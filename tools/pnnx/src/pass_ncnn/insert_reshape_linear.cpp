// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_reshape_linear.h"
#include "pass_ncnn.h"

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

            matched = true;

            Operand* linear_in = op->inputs[0];
            Operand* linear_out = op->outputs[0];

            const int batch_index = linear_in->params["__batch_index"].i;

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

            reshape0_out->params["__batch_index"] = batch_index;
            reshape1_in->params["__batch_index"] = batch_index;
            reshape0_out->params["__ncnn_batch_axis"] = ncnn_batch_axis;
            reshape1_in->params["__ncnn_batch_axis"] = ncnn_batch_axis;

            int reshape_h = 1;
            for (size_t j = 0; j < linear_in->shape.size() - 1; j++)
            {
                if ((int)j == ncnn_batch_axis)
                    continue;
                if (linear_in->shape[j] <= 0 || reshape_h > INT_MAX / linear_in->shape[j])
                {
                    reshape_h = -1;
                    break;
                }
                reshape_h *= linear_in->shape[j];
            }

            std::vector<int> reshape0_out_shape;
            std::vector<int> reshape1_in_shape;
            if (ncnn_batch_axis == 0)
            {
                reshape0_out_shape = {linear_in->shape[0], reshape_h, linear_in->shape[input_rank - 1]};
                reshape1_in_shape = {linear_out->shape[0], reshape_h, linear_out->shape[input_rank - 1]};
            }
            else
            {
                reshape0_out_shape = {reshape_h, linear_in->shape[input_rank - 1]};
                reshape1_in_shape = {reshape_h, linear_out->shape[input_rank - 1]};
            }
            reshape0->params["6"] = "0w,-1";
            reshape1->params["6"] = physical_rank == 3 ? "0w,1h,1c" : "0w,1h,1d,1c";
            reshape0_out->type = linear_in->type;
            reshape0_out->shape = reshape0_out_shape;
            reshape1_in->type = linear_out->type;
            reshape1_in->shape = reshape1_in_shape;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
