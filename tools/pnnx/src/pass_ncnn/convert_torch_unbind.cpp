// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_torch_unbind.h"

namespace pnnx {

namespace ncnn {

void convert_torch_unbind(Graph& graph)
{
    int op_index = 0;

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "torch.unbind")
                continue;

            matched = true;

            op->type = "Slice";
            op->name = std::string("unbind_") + std::to_string(op_index++);

            const int batch_index = op->inputs[0]->params["__batch_index"].i;

            int axis = op->params.at("dim").i;
            if (axis == batch_index)
            {
                fprintf(stderr, "unbind along batch axis %d is not supported\n", batch_index);
                continue;
            }

            if (axis < 0)
            {
                int input_rank = op->inputs[0]->shape.size();
                axis = input_rank + axis;
            }

            int output_size = (int)op->outputs.size();

            if (axis > batch_index)
                axis -= 1;

            op->params["0"].type = 5;
            op->params["0"].ai.resize(output_size, -233);

            op->params["1"] = axis;

            op->params.erase("dim");

            // reshape for each output, squeezing the unbind dim
            for (int i = 0; i < output_size; i++)
            {
                Operand* out = op->outputs[i];

                Operator* reshape = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape" + std::to_string(i), op);

                Operand* reshape_in = graph.new_operand(op->name + "_ncnnreshape" + std::to_string(i) + "_in");

                reshape->inputs.push_back(reshape_in);
                reshape->outputs.push_back(out);

                op->outputs[i] = reshape_in;

                out->producer = reshape;
                reshape_in->producer = op;
                reshape_in->consumers.push_back(reshape);

                reshape->params["shape"] = out->shape;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
