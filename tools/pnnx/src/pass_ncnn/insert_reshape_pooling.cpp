// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_reshape_pooling.h"
#include "reshape_shape.h"

namespace pnnx {

namespace ncnn {

void insert_reshape_pooling(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.MaxPool1d" && op->type != "nn.MaxPool2d" && op->type != "nn.MaxPool3d")
                continue;

            int input_rank = op->inputs[0]->shape.size();
            if (input_rank == 0)
                continue;

            // nn.MaxPool1d    2d-3d-2d
            // nn.MaxPool2d    3d-4d-3d
            // nn.MaxPool3d    4d-5d-4d
            bool insert_reshape = false;
            if ((op->type == "nn.MaxPool1d" && input_rank == 2)
                    || (op->type == "nn.MaxPool2d" && input_rank == 3)
                    || (op->type == "nn.MaxPool3d" && input_rank == 4))
            {
                insert_reshape = true;
            }

            if (!insert_reshape)
                continue;

            fprintf(stderr, "insert_reshape_pooling %d\n", input_rank);

            Operand* pooling_in = op->inputs[0];
            Operand* pooling_out = op->outputs[0];

            auto input_shape = get_logical_shape_expr(pooling_in, 0);
            auto target0 = input_shape;
            target0.insert(target0.begin(), "1");

            auto pooled_shape = get_logical_shape_expr((int)pooling_out->shape.size() + 1, 0, 0);
            pooled_shape[0] = "1";
            auto target1 = pooled_shape;
            target1.erase(target1.begin());
            std::map<std::string, Parameter> params0;
            std::map<std::string, Parameter> params1;
            if (!resolve_reshape_params(input_shape, get_ncnn_batch_axis(pooling_in), target0, 0, 1, params0)
                    || !resolve_reshape_params(pooled_shape, 0, target1, get_ncnn_batch_axis(pooling_out), 1, params1))
                continue;

            matched = true;

            Operator* reshape0 = graph.new_operator_before("Reshape", op->name + "_ncnnreshape0", op);
            Operator* reshape1 = graph.new_operator_after("Reshape", op->name + "_ncnnreshape1", op);

            Operand* reshape0_out = graph.new_operand(op->name + "_ncnnreshape0_out");
            Operand* reshape1_in = graph.new_operand(op->name + "_ncnnreshape1_in");

            reshape0->inputs.push_back(pooling_in);
            reshape0->outputs.push_back(reshape0_out);
            reshape1->inputs.push_back(reshape1_in);
            reshape1->outputs.push_back(pooling_out);

            for (size_t j = 0; j < pooling_in->consumers.size(); j++)
            {
                if (pooling_in->consumers[j] == op)
                {
                    pooling_in->consumers[j] = reshape0;
                    break;
                }
            }
            pooling_out->producer = reshape1;

            op->inputs[0] = reshape0_out;
            op->outputs[0] = reshape1_in;

            reshape0_out->producer = reshape0;
            reshape0_out->consumers.push_back(op);
            reshape1_in->producer = op;
            reshape1_in->consumers.push_back(reshape1);

            reshape0_out->params["__batch_index"] = 0;
            reshape0_out->params["__ncnn_batch_axis"] = 0;
            reshape0_out->shape = pooling_in->shape;
            reshape0_out->shape.insert(reshape0_out->shape.begin(), 1);
            reshape0_out->type = pooling_in->type;
            reshape1_in->params["__batch_index"] = 0;
            reshape1_in->params["__ncnn_batch_axis"] = 0;
            reshape1_in->shape = pooling_out->shape;
            reshape1_in->shape.insert(reshape1_in->shape.begin(), 1);
            reshape1_in->type = pooling_out->type;
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
