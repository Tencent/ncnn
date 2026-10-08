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

            Operand reshape0_shape = *linear_in;
            Operand reshape1_shape = *linear_out;
            reshape0_shape.producer = op;
            reshape1_shape.producer = op;
            // fold physical leading dimensions into rows and retain native batch
            const auto input_shape = logical_shape(linear_in, 0);
            auto leading_shape = input_shape;
            leading_shape.pop_back();
            if (ncnn_batch_axis != 233)
                leading_shape.erase(leading_shape.begin() + ncnn_batch_axis);
            std::vector<std::string> folded_shape = {shape_product(leading_shape), input_shape.back()};
            const int folded_batch_axis = ncnn_batch_axis == 233 ? 233 : 0;
            reshape0_shape.params["__batch_index"] = folded_batch_axis;
            reshape1_shape.params["__batch_index"] = folded_batch_axis;
            reshape0_shape.params["__ncnn_batch_axis"] = folded_batch_axis;
            reshape1_shape.params["__ncnn_batch_axis"] = folded_batch_axis;
            reshape0_shape.type = linear_in->type;
            reshape1_shape.type = linear_out->type;
            reshape0_shape.shape = {-1, linear_in->shape.back()};
            reshape1_shape.shape = {-1, linear_out->shape.back()};
            if (ncnn_batch_axis != 233)
            {
                folded_shape.insert(folded_shape.begin(), input_shape[ncnn_batch_axis]);
                reshape0_shape.shape.insert(reshape0_shape.shape.begin(), linear_in->shape[ncnn_batch_axis]);
                reshape1_shape.shape.insert(reshape1_shape.shape.begin(), linear_in->shape[ncnn_batch_axis]);
            }

            auto output_shape = logical_shape(linear_in, 1);
            output_shape.back() = std::to_string(linear_out->shape.back());
            std::map<std::string, Parameter> params0;
            std::map<std::string, Parameter> params1;
            if (!resolve_reshape_shape({linear_in}, &reshape0_shape, folded_shape, params0)
                || !resolve_reshape_shape({&reshape1_shape, linear_in}, linear_out, output_shape, params1))
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

            reshape0_out->params = reshape0_shape.params;
            reshape0_out->shape = reshape0_shape.shape;
            reshape0_out->type = reshape0_shape.type;
            reshape1_in->params = reshape1_shape.params;
            reshape1_in->shape = reshape1_shape.shape;
            reshape1_in->type = reshape1_shape.type;
            write_reshape_shape(reshape0, params0);
            write_reshape_shape(reshape1, params1);

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
