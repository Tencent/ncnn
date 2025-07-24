// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_Tensor_select.h"

namespace pnnx {

namespace ncnn {

void convert_Tensor_select(Graph& graph)
{
    int op_index = 0;

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "Tensor.select")
                continue;

            matched = true;

            op->type = "Crop";
            op->name = std::string("select_") + std::to_string(op_index++);

            const int batch_index = op->inputs[0]->params["__batch_index"].i;

            int axis = op->params.at("dim").i;
            if (axis == batch_index)
            {
                fprintf(stderr, "select along batch axis %d is not supported\n", batch_index);
                continue;
            }

            if (axis < 0)
            {
                int input_rank = op->inputs[0]->shape.size();
                axis = input_rank + axis;
            }

            if (axis > batch_index)
                axis -= 1;

            int dim;
            int index;
            if (op->has_param("dim"))
            {
                dim = op->params.at("dim").i;
            }
            else
            {
                fprintf(stderr, "select with dynamic dim is not supported\n");
                continue;
            }

            if (op->has_param("index"))
            {
                index = op->params.at("index").i;
            }
            else
            {
                fprintf(stderr, "select with dynamic index is not supported\n");
                continue;
            }

            op->params["9"] = std::vector<int> {index};
            op->params["10"] = std::vector<int> {index + 1};
            op->params["11"] = std::vector<int> {axis};

            op->params.erase("dim");
            op->params.erase("index");

            // squeezing the select dim
            {
                Operand* out = op->outputs[0];

                Operator* squeeze = graph.new_operator_after("torch.squeeze", op->name + "_ncnnsqueeze", op);

                Operand* squeeze_in = graph.new_operand(op->name + "_ncnnsqueeze_in");

                squeeze->inputs.push_back(squeeze_in);
                squeeze->outputs.push_back(out);

                op->outputs[0] = squeeze_in;

                out->producer = squeeze;
                squeeze_in->producer = op;
                squeeze_in->consumers.push_back(squeeze);

                squeeze->params["dim"] = dim;

                squeeze_in->params["__batch_index"] = batch_index;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
