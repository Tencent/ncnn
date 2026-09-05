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

            int input_rank = op->inputs[0]->shape.size();
            if (input_rank == 0)
                continue;

            // nn.Linear    4d-2d-4d
            // nn.Linear    5d-2d-5d
            // nn.Linear    3d(S,B,F) batch>1 -> 2d -> 3d ((S,B,F) layout, batch on dim1)
            // note the (B,S,F) layout (batch on dim0) must not trigger; distinguish by shape[0]>1 && shape[1]>1
            bool insert_reshape = false;
            if (op->type == "nn.Linear" && (input_rank == 4 || input_rank == 5))
            {
                insert_reshape = true;
            }
            if (op->type == "nn.Linear" && input_rank == 3 && op->inputs[0]->shape.size() >= 2 && op->inputs[0]->shape[0] > 1 && op->inputs[0]->shape[1] > 1)
            {
                insert_reshape = true;
            }

            if (!insert_reshape)
                continue;

            matched = true;

            Operand* linear_in = op->inputs[0];
            Operand* linear_out = op->outputs[0];

            const int batch_index = linear_in->params["__batch_index"].i;
            const int ncnn_batch_axis = linear_in->params["__ncnn_batch_axis"].i;

            Operator* reshape0 = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnreshape0", op);
            Operator* reshape1 = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape1", op);

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

            reshape0_out->params["__batch_index"] = batch_index;
            reshape1_in->params["__batch_index"] = batch_index;
            reshape0_out->params["__ncnn_batch_axis"] = ncnn_batch_axis;
            reshape1_in->params["__ncnn_batch_axis"] = ncnn_batch_axis;

            int reshape_h = 1;
            for (size_t j = 0; j < linear_in->shape.size() - 1; j++)
            {
                if (linear_in->shape[j] == -1)
                {
                    reshape_h = -1;
                    break;
                }
                reshape_h *= linear_in->shape[j];
            }

            // output shape may be unknown; guard the out-rank access
            const int out_last = linear_out->shape.size() >= (size_t)input_rank ? linear_out->shape[input_rank - 1] : -1;
            std::vector<int> reshape0_out_shape;
            std::vector<int> reshape1_in_shape;
            if (input_rank == 3 && linear_in->shape.size() >= 2 && linear_in->shape[0] > 1 && linear_in->shape[1] > 1)
            {
                // 3D (S,B,F) batch>1: flatten to 2D (S*B, F), Gemm 2D, then reshape back to (S,B,N)
                reshape0_out_shape = {reshape_h, linear_in->shape[input_rank - 1]};
                reshape1_in_shape = {reshape_h, out_last};
            }
            else if (ncnn_batch_axis == 0)
            {
                reshape0_out_shape = {1, reshape_h, linear_in->shape[input_rank - 1]};
                reshape1_in_shape = {1, reshape_h, out_last};
            }
            else
            {
                reshape0_out_shape = {reshape_h, linear_in->shape[input_rank - 1]};
                reshape1_in_shape = {reshape_h, out_last};
            }
            std::vector<int> reshape1_out_shape = linear_out->shape;

            reshape0->params["shape"] = reshape0_out_shape;
            reshape1->params["shape"] = reshape1_out_shape;
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
