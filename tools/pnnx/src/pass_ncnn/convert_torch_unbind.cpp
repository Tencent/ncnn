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
                    fprintf(stderr, "unbind axis around batch axis %d is unknown\n", ncnn_batch_axis);
            }

            const int axis0 = axis;

            bool axis_is_batch = false;
            if (ncnn_batch_axis != 233 && axis == ncnn_batch_axis)
            {
                fprintf(stderr, "unbind along batch axis %d is not supported\n", ncnn_batch_axis);
                axis_is_batch = true;
            }

            if (axis_is_batch)
            {
                // keep Slice op for future across-batch support
                int output_size = (int)op->outputs.size();

                op->params["0"].type = 5;
                op->params["0"].ai.resize(output_size, -233);

                op->params["1"] = -233;

                op->params.erase("dim");
                break;
            }

            int output_size = (int)op->outputs.size();

            if (ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
                axis -= 1;

            int output_batch_index = batch_index;
            if (batch_index != 233 && axis0 >= 0 && axis0 < batch_index)
                output_batch_index -= 1;

            int output_ncnn_batch_axis = ncnn_batch_axis;
            if (ncnn_batch_axis != 233 && axis0 >= 0 && axis0 < ncnn_batch_axis)
                output_ncnn_batch_axis -= 1;

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
                reshape_in->type = out->type;
                reshape_in->shape = out->shape;
                reshape_in->params["__batch_index"] = batch_index;
                reshape_in->params["__ncnn_batch_axis"] = ncnn_batch_axis;
                out->params["__batch_index"] = output_batch_index;
                out->params["__ncnn_batch_axis"] = output_ncnn_batch_axis;

                if (!out->shape.empty())
                    reshape->params["shape"] = out->shape;
                else
                    reshape->params["shape"] = std::vector<int> {-1};
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
