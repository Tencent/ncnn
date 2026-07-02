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

            const int batch_index = op->inputs[0]->params.at("__batch_index").i;
            const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

            int axis = op->params.at("dim").i;
            if (axis < 0)
            {
                int input_rank = op->inputs[0]->shape.size();
                if (input_rank == 0 && !op->outputs.empty())
                    input_rank = op->outputs[0]->shape.size() + 1;
                if (input_rank > 0)
                    axis = input_rank + axis;
                else if (ncnn_batch_axis != 233)
                    fprintf(stderr, "select axis around batch axis %d is unknown\n", ncnn_batch_axis);
            }

            bool axis_is_batch = false;
            if (ncnn_batch_axis != 233 && axis == ncnn_batch_axis)
            {
                fprintf(stderr, "select along batch axis %d is not supported\n", ncnn_batch_axis);
                axis_is_batch = true;
            }

            const int axis_in_shape = axis;

            if (axis_is_batch)
            {
                matched = true;

                op->type = "Crop";
                op->name = std::string("select_") + std::to_string(op_index++);
                // ignore batch-axis select for now
                op->params["9"] = std::vector<int> {0};
                op->params["10"] = std::vector<int> {-233};
                op->params["11"] = std::vector<int> {0};

                op->params.erase("dim");
                op->params.erase("index");

                break;
            }

            if (ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
                axis -= 1;

            int dim;
            int index;
            if (op->has_param("dim"))
            {
                dim = axis_in_shape;
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

            matched = true;

            op->type = "Crop";
            op->name = std::string("select_") + std::to_string(op_index++);

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
                squeeze_in->params["__ncnn_batch_axis"] = ncnn_batch_axis;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
