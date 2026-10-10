// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_torch_stack.h"

namespace pnnx {

namespace ncnn {

void convert_torch_stack(Graph& graph)
{
    int op_index = 0;

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "torch.stack")
                continue;

            matched = true;

            op->type = "Concat";
            op->name = std::string("stack_") + std::to_string(op_index++);

            const int batch_index = op->inputs[0]->params["__batch_index"].i;
            const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;
            int input_rank = op->inputs[0]->shape.size();
            if (input_rank == 0 && op->outputs[0]->shape.size() != 0)
                input_rank = (int)op->outputs[0]->shape.size() - 1;

            int axis = op->params.at("dim").i;
            if (axis < 0 && input_rank > 0)
            {
                axis = input_rank + 1 + axis;
            }
            else if (axis < 0 && ncnn_batch_axis != 233)
            {
                fprintf(stderr, "stack axis around batch axis %d is unknown\n", ncnn_batch_axis);
            }

            bool axis_is_batch = false;
            if (ncnn_batch_axis != 233 && axis == ncnn_batch_axis)
            {
                fprintf(stderr, "stack along batch axis %d is not supported\n", ncnn_batch_axis);
                axis_is_batch = true;
            }

            if (axis_is_batch)
            {
                // keep Concat op for future across-batch support
                op->params["0"] = -233;

                op->params.erase("dim");
                break;
            }

            bool stack_inner_most = axis == input_rank;
            const int axis0 = axis;

            if (ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
                axis -= 1;

            op->params["0"] = axis;

            if (stack_inner_most)
            {
                // stack -> reshape(x,y,..,1) + concat
                // reshape for input, expand the stack dim
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    Operand* in = op->inputs[i];

                    Operator* reshape = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnreshape_" + std::to_string(i), op);

                    Operand* reshape_out = graph.new_operand(op->name + "_ncnnreshape_in");
                    reshape_out->params["__batch_index"] = batch_index;
                    reshape_out->params["__ncnn_batch_axis"] = ncnn_batch_axis;

                    reshape->inputs.push_back(in);
                    reshape->outputs.push_back(reshape_out);

                    op->inputs[i] = reshape_out;

                    in->remove_consumer(op);
                    in->consumers.push_back(reshape);
                    reshape_out->producer = reshape;
                    reshape_out->consumers.push_back(op);

                    std::vector<int> shape = in->shape;
                    if (shape.size() != 0)
                    {
                        shape.push_back(1);
                        reshape->params["shape"] = shape;
                    }
                    else
                    {
                        reshape->params["shape"] = std::vector<int> {-1, 1};
                    }
                    reshape_out->shape = shape;
                }
            }
            else
            {
                // reshape for output, expand the stack dim
                Operand* out = op->outputs[0];

                Operator* reshape = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape", op);

                Operand* reshape_in = graph.new_operand(op->name + "_ncnnreshape_in");

                std::vector<int> shape = op->inputs[0]->shape;
                if (shape.size() != 0)
                {
                    if (axis0 >= 0 && axis0 < (int)shape.size() && shape[axis0] != -1)
                        shape[axis0] *= op->inputs.size();
                    reshape_in->shape = shape;
                }
                reshape_in->params["__batch_index"] = batch_index;
                reshape_in->params["__ncnn_batch_axis"] = ncnn_batch_axis;

                reshape->inputs.push_back(reshape_in);
                reshape->outputs.push_back(out);

                op->outputs[0] = reshape_in;

                out->producer = reshape;
                reshape_in->producer = op;
                reshape_in->consumers.push_back(reshape);

                if (!out->shape.empty())
                    reshape->params["shape"] = out->shape;
                else
                    reshape->params["shape"] = std::vector<int> {-1};
            }

            op->params.erase("dim");

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
