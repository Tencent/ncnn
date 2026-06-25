// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_unflatten : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.unflatten         op_0        1 1 input out dim=%dim sizes=%sizes
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "unflatten";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim = captured_params.at("dim").i;
        std::vector<int> sizes = captured_params.at("sizes").ai;

        const int input_rank = op->inputs[0]->shape.size();

        if (dim < 0)
            dim += input_rank;

        if (input_rank <= dim)
        {
            fprintf(stderr, "unflatten %d not possible for %d-rank tensor\n", dim, input_rank);
            return;
        }

        const std::vector<int> shape = op->outputs[0]->shape;

        const int input_batch_index = op->inputs[0]->params["__batch_index"].i;
        const int batch_index = op->outputs[0]->params["__batch_index"].i;

        int batch_mode = 0;
        int batch_axis = 0;
        if (input_batch_index == 233 && batch_index != 233)
        {
            batch_mode = 2;
            batch_axis = batch_index;
        }
        if (input_batch_index == 0 && batch_index == 233)
        {
            batch_mode = 1;
        }
        if (input_batch_index != 233 && input_batch_index > 0 && batch_index == 233)
        {
            batch_mode = 1;
            const std::vector<int>& input_shape = op->inputs[0]->shape;
            const std::vector<int>& output_shape = op->outputs[0]->shape;
            if (input_shape.empty() || output_shape.empty() || input_batch_index >= (int)input_shape.size())
            {
                fprintf(stderr, "unflatten tensor with batch index %d folded is not supported yet!\n", input_batch_index);
                return;
            }

            const int batch_size = input_shape[input_batch_index];
            if (batch_size <= 0)
            {
                fprintf(stderr, "unflatten tensor with batch index %d folded is not supported yet!\n", input_batch_index);
                return;
            }

            int left = 1;
            for (int i = 0; i < input_batch_index; i++)
            {
                if (input_shape[i] <= 0)
                {
                    left = -1;
                    break;
                }
                left *= input_shape[i];
            }

            int right = 1;
            for (int i = input_batch_index + 1; i < (int)input_shape.size(); i++)
            {
                if (input_shape[i] <= 0)
                {
                    right = -1;
                    break;
                }
                right *= input_shape[i];
            }
            if (left <= 0 || right <= 0)
            {
                fprintf(stderr, "unflatten tensor with batch index %d folded is not supported yet!\n", input_batch_index);
                return;
            }

            int batch_axis_count = 0;
            for (int i = 0; i < (int)output_shape.size(); i++)
            {
                if (output_shape[i] != batch_size)
                    continue;

                int left2 = 1;
                for (int j = 0; j < i; j++)
                {
                    if (output_shape[j] <= 0)
                    {
                        left2 = -1;
                        break;
                    }
                    left2 *= output_shape[j];
                }

                int right2 = 1;
                for (int j = i + 1; j < (int)output_shape.size(); j++)
                {
                    if (output_shape[j] <= 0)
                    {
                        right2 = -1;
                        break;
                    }
                    right2 *= output_shape[j];
                }
                if (left2 <= 0 || right2 <= 0)
                    continue;

                if (left2 == left && right2 == right)
                {
                    batch_axis = i;
                    batch_axis_count += 1;
                }
            }

            if (batch_axis_count != 1)
            {
                fprintf(stderr, "unflatten tensor with batch index %d folded is not supported yet!\n", input_batch_index);
                return;
            }
        }

        if (batch_mode == 0 && batch_index != 0 && batch_index != 233)
        {
            if (op->outputs[0]->shape.empty() || op->outputs[0]->shape[batch_index] != 1)
            {
                fprintf(stderr, "unflatten tensor to batch index %d is not supported yet!\n", batch_index);
                return;
            }
        }

        // drop shape batch index
        std::vector<int> new_shape;
        for (int i = 0; i < (int)shape.size(); i++)
        {
            if (batch_mode == 2 && i == batch_index)
                continue;

            if (batch_mode == 0 && i == batch_index)
                continue;

            new_shape.push_back(shape[i]);
        }

        if (new_shape.size() == 5 && batch_index == 233)
        {
            if (new_shape[0] == 1)
            {
                fprintf(stderr, "assume reshape 5-rank tensor has batch_index 0\n");
                new_shape.erase(new_shape.begin());
            }
        }

        const int shape_rank = (int)new_shape.size();

        if (shape_rank > 4)
        {
            fprintf(stderr, "reshape to %d-rank tensor is not supported yet!\n", shape_rank);
            return;
        }

        if (shape_rank == 1)
        {
            op->params["0"] = new_shape[0];
        }
        if (shape_rank == 2)
        {
            op->params["0"] = new_shape[1];
            op->params["1"] = new_shape[0];
        }
        if (shape_rank == 3)
        {
            op->params["0"] = new_shape[2];
            op->params["1"] = new_shape[1];
            op->params["2"] = new_shape[0];
        }
        if (shape_rank == 4)
        {
            op->params["0"] = new_shape[3];
            op->params["1"] = new_shape[2];
            op->params["11"] = new_shape[1];
            op->params["2"] = new_shape[0];
        }

        if (batch_mode != 0)
            op->params["12"] = batch_mode;

        if (batch_mode != 0 && batch_axis != 0)
            op->params["13"] = batch_axis;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_unflatten, 20)

} // namespace ncnn

} // namespace pnnx
