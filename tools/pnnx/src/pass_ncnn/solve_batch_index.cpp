// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "solve_batch_index.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

static bool is_known_operator_with_batch_index_0(const Operator* op)
{
    static const char* operator_with_batch_index_0[] = {
        "F.adaptive_avg_pool1d",
        "F.adaptive_avg_pool2d",
        "F.adaptive_avg_pool3d",
        "F.adaptive_max_pool1d",
        "F.adaptive_max_pool2d",
        "F.adaptive_max_pool3d",
        "F.affine_grid",
        "F.avg_pool1d",
        "F.avg_pool2d",
        "F.avg_pool3d",
        "F.batch_norm",
        "F.conv_transpose1d",
        "F.conv_transpose2d",
        "F.conv_transpose3d",
        "F.conv1d",
        "F.conv2d",
        "F.conv3d",
        "F.embedding",
        "F.fold",
        "F.grid_sample",
        "F.group_norm",
        "F.instance_norm",
        "F.interpolate",
        "F.layer_norm",
        "F.linear",
        "F.local_response_norm",
        "F.lp_pool1d",
        "F.lp_pool2d",
        "F.max_pool1d",
        "F.max_pool2d",
        "F.max_pool3d",
        "F.pixel_shuffle",
        "F.pixel_unshuffle",
        "F.prelu",
        "F.rms_norm",
        "F.scaled_dot_product_attention",
        "F.unfold",
        "F.upsample_bilinear",
        "F.upsample_nearest",
        "F.upsample",

        "torch.bmm",
        "torch.istft",
        "torch.stft",
        "torchaudio.functional.inverse_spectrogram",
        "torchaudio.functional.spectrogram",

        "nn.AdaptiveAvgPool1d",
        "nn.AdaptiveAvgPool2d",
        "nn.AdaptiveAvgPool3d",
        "nn.AdaptiveMaxPool1d",
        "nn.AdaptiveMaxPool2d",
        "nn.AdaptiveMaxPool3d",
        "nn.AvgPool1d",
        "nn.AvgPool2d",
        "nn.AvgPool3d",
        "nn.BatchNorm1d",
        "nn.BatchNorm2d",
        "nn.BatchNorm3d",
        "nn.ChannelShuffle",
        "nn.ConstantPad1d",
        "nn.ConstantPad2d",
        "nn.ConstantPad3d",
        "nn.Conv1d",
        "nn.Conv2d",
        "nn.Conv3d",
        "nn.ConvTranspose1d",
        "nn.ConvTranspose2d",
        "nn.ConvTranspose3d",
        "nn.Embedding",
        "nn.Fold",
        "nn.GroupNorm",
        "nn.InstanceNorm1d",
        "nn.InstanceNorm2d",
        "nn.InstanceNorm3d",
        "nn.LocalResponseNorm",
        "nn.LayerNorm",
        "nn.LPPool1d",
        "nn.LPPool2d",
        "nn.MaxPool1d",
        "nn.MaxPool2d",
        "nn.MaxPool3d",
        "nn.PixelShuffle",
        "nn.PixelUnshuffle",
        "nn.PReLU",
        "nn.ReflectionPad1d",
        "nn.ReflectionPad2d",
        "nn.ReplicationPad1d",
        "nn.ReplicationPad2d",
        "nn.ReplicationPad3d",
        "nn.RMSNorm",
        "nn.Softmax2d",
        "nn.Unfold",
        "nn.Upsample",
        "nn.UpsamplingBilinear2d",
        "nn.UpsamplingNearest2d",
        "nn.ZeroPad2d",
    };

    const size_t operator_with_batch_index_0_count = sizeof(operator_with_batch_index_0) / sizeof(const char*);
    for (size_t i = 0; i < operator_with_batch_index_0_count; i++)
    {
        if (op->type == operator_with_batch_index_0[i])
            return true;
    }

    return false;
}

static int get_known_operator_batch_index(const Operator* op)
{
    if (!is_known_operator_with_batch_index_0(op))
        return 233;

    if (op->inputs.empty())
        return 0;

    const int input_rank = op->inputs[0]->shape.size();

    if (op->type.find("pool1d") != std::string::npos || op->type.find("Pool1d") != std::string::npos)
    {
        if (input_rank == 2)
            return 233;
    }
    if (op->type.find("pool2d") != std::string::npos || op->type.find("Pool2d") != std::string::npos)
    {
        if (input_rank == 3)
            return 233;
    }
    if (op->type.find("pool3d") != std::string::npos || op->type.find("Pool3d") != std::string::npos)
    {
        if (input_rank == 4)
            return 233;
    }
    if (op->type == "torch.stft" || op->type == "torchaudio.functional.spectrogram")
    {
        if (input_rank == 1)
            return 233;
    }
    if (op->type == "torch.istft" || op->type == "torchaudio.functional.inverse_spectrogram")
    {
        if (input_rank == 2)
            return 233;
    }

    return 0;
}

static bool is_known_operator_with_batch_first_param(const Operator* op)
{
    return op->type == "nn.RNN" || op->type == "nn.LSTM" || op->type == "nn.GRU" || op->type == "nn.MultiheadAttention";
}

static int normalize_axis(int axis, int rank)
{
    if (axis < 0 && rank > 0)
        axis += rank;

    return axis;
}

static int solve_select_batch_index_forward(const Operator* op, int batch_index, int input_rank)
{
    int dim = normalize_axis(op->params.at("dim").i, input_rank);
    if (dim < 0)
        return batch_index;

    if (dim == batch_index)
        return 233;

    if (dim < batch_index)
        return batch_index - 1;

    return batch_index;
}

static int solve_select_batch_index_backward(const Operator* op, int batch_index, int input_rank)
{
    int dim = normalize_axis(op->params.at("dim").i, input_rank);
    if (dim < 0)
        return batch_index;

    if (dim <= batch_index)
        return batch_index + 1;

    return batch_index;
}

static std::vector<int> get_slice_selected_axes(const Operator* op, int input_rank)
{
    std::vector<int> axes;
    if (op->has_param("dims"))
        axes = op->params.at("dims").ai;
    else if (op->has_param("dim"))
        axes = std::vector<int> {op->params.at("dim").i};
    else
        return std::vector<int>();

    std::vector<int> steps;
    if (op->has_param("steps"))
        steps = op->params.at("steps").ai;
    else if (op->has_param("step"))
        steps = std::vector<int> {op->params.at("step").i};
    else
        return std::vector<int>();

    if (axes.size() != steps.size())
        return std::vector<int>();

    std::vector<int> selected_axes;
    for (int i = 0; i < (int)axes.size(); i++)
    {
        if (steps[i] != 0)
            continue;

        int axis = normalize_axis(axes[i], input_rank);
        if (axis < 0)
            continue;

        selected_axes.push_back(axis);
    }

    std::sort(selected_axes.begin(), selected_axes.end());
    selected_axes.erase(std::unique(selected_axes.begin(), selected_axes.end()), selected_axes.end());

    return selected_axes;
}

static int solve_slice_batch_index_forward(const Operator* op, int batch_index, int input_rank)
{
    std::vector<int> selected_axes = get_slice_selected_axes(op, input_rank);

    int select_before_batch = 0;
    for (int axis : selected_axes)
    {
        if (axis == batch_index)
            return 233;

        if (axis < batch_index)
            select_before_batch += 1;
    }

    return batch_index - select_before_batch;
}

static int solve_slice_batch_index_backward(const Operator* op, int batch_index, int input_rank, int output_rank)
{
    std::vector<int> selected_axes = get_slice_selected_axes(op, input_rank);
    if (selected_axes.empty())
        return batch_index;

    if (input_rank == 0 && output_rank > 0)
        input_rank = output_rank + selected_axes.size();
    if (input_rank == 0)
        return batch_index;

    int output_axis = 0;
    for (int i = 0; i < input_rank; i++)
    {
        if (std::find(selected_axes.begin(), selected_axes.end(), i) != selected_axes.end())
            continue;

        if (output_axis == batch_index)
            return i;

        output_axis += 1;
    }

    return batch_index;
}

static void solve_batch_index_backward(Operand* operand);
static void solve_batch_index_forward(Operand* operand)
{
    if (operand->params.find("__batch_index") == operand->params.end())
        return;

    int batch_index = operand->params["__batch_index"].i;
    if (batch_index < 0 || batch_index == 233)
        return;

    for (Operator* op : operand->consumers)
    {
        if (is_known_operator_with_batch_index_0(op))
            continue;

        if (is_known_operator_with_batch_first_param(op))
            continue;

        const int input_rank0 = op->inputs.empty() ? 0 : (int)op->inputs[0]->shape.size();
        const int output_rank0 = op->outputs.empty() ? 0 : (int)op->outputs[0]->shape.size();

        if (op->type == "Tensor.permute")
        {
            const std::vector<int>& dims = op->params.at("dims").ai;

            int batch_index_permuted = -1;
            for (int i = 0; i < (int)dims.size(); i++)
            {
                int dim = dims[i];
                if (dim < 0)
                    dim += input_rank0;

                if (dim >= 0 && dim == batch_index)
                {
                    batch_index_permuted = i;
                    break;
                }
            }

            for (Operand* r : op->outputs)
            {
                if (r->params.find("__batch_index") != r->params.end())
                    continue;

                r->params["__batch_index"] = batch_index_permuted;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else if (op->type == "torch.transpose")
        {
            int dim0 = op->params.at("dim0").i;
            int dim1 = op->params.at("dim1").i;
            if (dim0 < 0)
                dim0 += input_rank0;
            if (dim1 < 0)
                dim1 += input_rank0;

            int batch_index_transposed = batch_index;
            if (dim0 >= 0 && dim0 == batch_index)
            {
                batch_index_transposed = dim1;
            }
            else if (dim1 >= 0 && dim1 == batch_index)
            {
                batch_index_transposed = dim0;
            }

            for (Operand* r : op->outputs)
            {
                if (r->params.find("__batch_index") != r->params.end())
                    continue;

                r->params["__batch_index"] = batch_index_transposed;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else if (op->type == "torch.t")
        {
            int batch_index_transposed = batch_index;
            if (batch_index == 0)
            {
                batch_index_transposed = 1;
            }
            else if (batch_index == 1)
            {
                batch_index_transposed = 0;
            }

            for (Operand* r : op->outputs)
            {
                if (r->params.find("__batch_index") != r->params.end())
                    continue;

                r->params["__batch_index"] = batch_index_transposed;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else if (op->type == "Tensor.reshape" || op->type == "Tensor.reshape_as")
        {
            if (operand != op->inputs[0])
                continue;

            std::vector<int> shape;
            if (op->type == "Tensor.reshape_as")
            {
                shape = op->outputs[0]->shape;
            }
            else if (op->params.find("shape") == op->params.end())
            {
                // dynamic reshape
                const Operator* op_expr = op->inputs[1]->producer;
                std::string expr = op_expr->params.at("expr").s;
                {
                    int expr_stack = 0;
                    std::string t;
                    for (size_t i = 0; i < expr.size(); i++)
                    {
                        char ch = expr[i];

                        if (ch == '[') // list
                        {
                            t.clear();
                        }
                        else if (ch == '(')
                        {
                            expr_stack += 1;
                        }
                        else if (ch == ')')
                        {
                            expr_stack -= 1;
                            t.clear();
                        }
                        else if (ch == ',' || ch == ']')
                        {
                            if (expr_stack > 0)
                            {
                                shape.push_back(-1);
                            }
                            else if (!t.empty())
                            {
                                shape.push_back(std::stoi(t));
                            }
                            t.clear();
                        }
                        else
                        {
                            t += ch;
                        }
                    }

                    if (!t.empty())
                    {
                        shape.push_back(std::stoi(t));
                    }
                }
            }
            else
            {
                shape = op->params.at("shape").ai;
            }

            if (batch_index >= input_rank0 || output_rank0 == 0 || shape.empty())
                continue;

            std::vector<int> output_shape = op->outputs[0]->shape;
            if (output_shape.size() == shape.size())
            {
                for (size_t i = 0; i < shape.size(); i++)
                {
                    if (output_shape[i] == -1 && shape[i] > 0)
                        output_shape[i] = shape[i];
                }
            }

            const int batch_size = op->inputs[0]->shape[batch_index];
            const int batch_axis_size = batch_size > 0 ? batch_size : -1;
            bool keep_batch_index = false;
            int batch_index_reshaped = batch_index;
            if (batch_index == 0 && ((batch_size == 1 && shape[0] == 1) || (batch_size > 1 && output_rank0 > 0 && output_shape[0] == batch_size)))
            {
                keep_batch_index = true;
            }
            else if (output_rank0 > 0)
            {
                if (batch_index == input_rank0 - 1 && ((batch_size == 1 && shape[shape.size() - 1] == 1) || (batch_size > 1 && output_shape[output_rank0 - 1] == batch_size)))
                {
                    keep_batch_index = true;
                    batch_index_reshaped = output_rank0 - 1;
                }
                else
                {
                    batch_index_reshaped = -1;

                    // batch index is in the middle, let's consider the left and right parts
                    int left = 1;
                    int right = 1;
                    for (int i = 0; i < batch_index; i++)
                    {
                        if (op->inputs[0]->shape[i] == -1)
                        {
                            left = -1;
                            break;
                        }
                        left *= op->inputs[0]->shape[i];
                    }
                    for (int i = batch_index + 1; i < (int)op->inputs[0]->shape.size(); i++)
                    {
                        if (op->inputs[0]->shape[i] == -1)
                        {
                            right = -1;
                            break;
                        }
                        right *= op->inputs[0]->shape[i];
                    }

                    // try to find batch index in the output shape
                    if (left > 0)
                    {
                        int left2 = 1;
                        for (int i = 0; i < output_rank0 - 1; i++)
                        {
                            if (output_shape[i] == -1)
                            {
                                left2 = -1;
                                break;
                            }
                            left2 *= output_shape[i];
                            if (left2 == left && output_shape[i + 1] == batch_axis_size)
                            {
                                batch_index_reshaped = i + 1;
                                if (batch_index_reshaped + 1 < output_rank0 && output_shape[batch_index_reshaped + 1] == batch_axis_size)
                                {
                                    // multiple axes can be batch index, give up
                                    batch_index_reshaped = -1;
                                }
                                break;
                            }
                        }
                    }
                    if (right > 0)
                    {
                        int right2 = 1;
                        for (int i = output_rank0 - 1; i >= 1; i--)
                        {
                            if (output_shape[i] == -1)
                            {
                                right2 = -1;
                                break;
                            }
                            right2 *= output_shape[i];
                            if (right2 == right && output_shape[i - 1] == batch_axis_size)
                            {
                                batch_index_reshaped = i - 1;
                                if (batch_index_reshaped - 1 >= 0 && output_shape[batch_index_reshaped - 1] == batch_axis_size)
                                {
                                    // multiple axes can be batch index, give up
                                    batch_index_reshaped = -1;
                                }
                                break;
                            }
                        }
                    }

                    if (batch_index_reshaped >= 0)
                        keep_batch_index = true;
                }
            }

            if (keep_batch_index)
            {
                for (Operand* r : op->outputs)
                {
                    if (r->params.find("__batch_index") != r->params.end())
                        continue;

                    r->params["__batch_index"] = batch_index_reshaped;

                    solve_batch_index_forward(r);
                    solve_batch_index_backward(r);
                }
            }
            else
            {
                // give up reshape across batch index
            }
        }
        else if (op->type == "Tensor.unflatten")
        {
            int dim = op->params.at("dim").i;
            const int sizes_rank = (int)op->params.at("sizes").ai.size();
            if (dim < 0)
                dim += input_rank0;

            int batch_index_unflattened = batch_index;
            if (dim == batch_index)
                batch_index_unflattened = 233;
            else if (dim < batch_index)
                batch_index_unflattened = batch_index + sizes_rank - 1;

            Operand* r = op->outputs[0];
            if (r->params.find("__batch_index") == r->params.end())
            {
                r->params["__batch_index"] = batch_index_unflattened;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else if (op->type == "torch.flatten")
        {
            int start_dim = op->params.at("start_dim").i;
            int end_dim = op->params.at("end_dim").i;
            if (start_dim < 0)
                start_dim += input_rank0;
            if (end_dim < 0)
                end_dim += input_rank0;

            int batch_index_flattened = batch_index;
            if (start_dim <= batch_index && batch_index <= end_dim)
                batch_index_flattened = 233;
            else if (end_dim < batch_index)
                batch_index_flattened = batch_index - (end_dim - start_dim);

            Operand* r = op->outputs[0];
            if (r->params.find("__batch_index") == r->params.end())
            {
                r->params["__batch_index"] = batch_index_flattened;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else if (op->type == "Tensor.slice")
        {
            int batch_index_sliced = solve_slice_batch_index_forward(op, batch_index, input_rank0);

            Operand* r = op->outputs[0];
            if (r->params.find("__batch_index") == r->params.end())
            {
                r->params["__batch_index"] = batch_index_sliced;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else if (op->type == "Tensor.select")
        {
            int batch_index_selected = solve_select_batch_index_forward(op, batch_index, input_rank0);

            Operand* r = op->outputs[0];
            if (r->params.find("__batch_index") == r->params.end())
            {
                r->params["__batch_index"] = batch_index_selected;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else if (op->type == "pnnx.SliceIndexes")
        {
            // pass
        }
        else if (op->type == "torch.squeeze")
        {
            int batch_index_squeezed = batch_index;
            const std::vector<int>& input_shape = op->inputs[0]->shape;

            if (op->has_param("dim"))
            {
                int squeeze_before_batch = 0;
                if (op->params.at("dim").type == 2)
                {
                    int dim = op->params.at("dim").i;
                    if (dim < 0 && input_rank0 > 0)
                        dim += input_rank0;

                    bool squeezed = dim >= 0 && dim < input_rank0 && input_shape[dim] == 1;
                    if (squeezed && dim >= 0 && dim == batch_index)
                    {
                        batch_index_squeezed = 233;
                    }
                    else if (squeezed && dim >= 0 && dim < batch_index)
                    {
                        squeeze_before_batch += 1;
                    }
                }
                else
                {
                    const std::vector<int>& dims = op->params.at("dim").ai;
                    for (auto d : dims)
                    {
                        int dim = d;
                        if (dim < 0 && input_rank0 > 0)
                            dim += input_rank0;

                        bool squeezed = dim >= 0 && dim < input_rank0 && input_shape[dim] == 1;
                        if (squeezed && dim >= 0 && dim == batch_index)
                        {
                            batch_index_squeezed = 233;
                            break;
                        }
                        else if (squeezed && dim >= 0 && dim < batch_index)
                        {
                            squeeze_before_batch += 1;
                        }
                    }
                }
                if (batch_index_squeezed != 233)
                    batch_index_squeezed = batch_index - squeeze_before_batch;
            }
            else
            {
                // squeeze all
                if (input_shape.empty())
                {
                    // unknown rank, keep batch axis as-is
                }
                else if (batch_index >= 0 && batch_index < input_rank0 && input_shape[batch_index] == 1)
                {
                    batch_index_squeezed = 233;
                }
                else
                {
                    int squeeze_before_batch = 0;
                    for (int i = 0; i < batch_index; i++)
                    {
                        if (input_shape[i] == 1)
                            squeeze_before_batch += 1;
                    }
                    batch_index_squeezed = batch_index - squeeze_before_batch;
                }
            }

            Operand* r = op->outputs[0];
            if (r->params.find("__batch_index") == r->params.end())
            {
                r->params["__batch_index"] = batch_index_squeezed;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else if (op->type == "torch.unsqueeze")
        {
            int dim = op->params.at("dim").i;
            if (dim < 0 && input_rank0 > 0)
                dim += input_rank0 + 1;
            if (dim < 0 && input_rank0 == 0 && output_rank0 > 0)
                dim += output_rank0;

            if (batch_index == 233)
            {
                // give up
                return;
            }

            int batch_index_unsqueezed = batch_index;
            if (dim >= 0 && dim <= batch_index)
            {
                batch_index_unsqueezed = batch_index + 1;
            }

            Operand* r = op->outputs[0];
            if (r->params.find("__batch_index") == r->params.end())
            {
                r->params["__batch_index"] = batch_index_unsqueezed;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
        else
        {
            for (Operand* r : op->outputs)
            {
                if (r->params.find("__batch_index") != r->params.end())
                    continue;

                r->params["__batch_index"] = batch_index;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }
        }
    }
}

static void solve_batch_index_backward(Operand* operand)
{
    if (operand->params.find("__batch_index") == operand->params.end())
        return;

    int batch_index = operand->params["__batch_index"].i;
    if (batch_index < 0 || batch_index == 233)
        return;

    Operator* op = operand->producer;
    if (is_known_operator_with_batch_index_0(op))
        return;

    if (is_known_operator_with_batch_first_param(op))
        return;

    const int input_rank0 = op->inputs.empty() ? 0 : (int)op->inputs[0]->shape.size();
    const int output_rank0 = op->outputs.empty() ? 0 : (int)op->outputs[0]->shape.size();

    if (op->type == "Tensor.permute")
    {
        const std::vector<int>& dims = op->params.at("dims").ai;

        int batch_index_permuted = dims[batch_index];
        if (batch_index_permuted < 0)
            batch_index_permuted += input_rank0;

        for (Operand* r : op->inputs)
        {
            if (r->params.find("__batch_index") != r->params.end())
                continue;

            r->params["__batch_index"] = batch_index_permuted;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else if (op->type == "torch.transpose")
    {
        int dim0 = op->params.at("dim0").i;
        int dim1 = op->params.at("dim1").i;
        if (dim0 < 0)
            dim0 += input_rank0;
        if (dim1 < 0)
            dim1 += input_rank0;

        int batch_index_transposed = batch_index;
        if (dim0 >= 0 && dim0 == batch_index)
        {
            batch_index_transposed = dim1;
        }
        else if (dim1 >= 0 && dim1 == batch_index)
        {
            batch_index_transposed = dim0;
        }

        for (Operand* r : op->inputs)
        {
            if (r->params.find("__batch_index") != r->params.end())
                continue;

            r->params["__batch_index"] = batch_index_transposed;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else if (op->type == "torch.t")
    {
        int batch_index_transposed = batch_index;
        if (batch_index == 0)
        {
            batch_index_transposed = 1;
        }
        else if (batch_index == 1)
        {
            batch_index_transposed = 0;
        }

        for (Operand* r : op->inputs)
        {
            if (r->params.find("__batch_index") != r->params.end())
                continue;

            r->params["__batch_index"] = batch_index_transposed;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else if (op->type == "Tensor.reshape" || op->type == "Tensor.reshape_as")
    {
        if (operand != op->outputs[0])
            return;

        std::vector<int> shape;
        if (op->type == "Tensor.reshape_as")
        {
            shape = op->outputs[0]->shape;
        }
        else if (op->params.find("shape") == op->params.end())
        {
            // dynamic reshape
            const Operator* op_expr = op->inputs[1]->producer;
            std::string expr = op_expr->params.at("expr").s;
            {
                int expr_stack = 0;
                std::string t;
                for (size_t i = 0; i < expr.size(); i++)
                {
                    char ch = expr[i];

                    if (ch == '[') // list
                    {
                        t.clear();
                    }
                    else if (ch == '(')
                    {
                        expr_stack += 1;
                    }
                    else if (ch == ')')
                    {
                        expr_stack -= 1;
                        t.clear();
                    }
                    else if (ch == ',' || ch == ']')
                    {
                        if (expr_stack > 0)
                        {
                            shape.push_back(-1);
                        }
                        else if (!t.empty())
                        {
                            shape.push_back(std::stoi(t));
                        }
                        t.clear();
                    }
                    else
                    {
                        t += ch;
                    }
                }

                if (!t.empty())
                {
                    shape.push_back(std::stoi(t));
                }
            }
        }
        else
        {
            shape = op->params.at("shape").ai;
        }

        if (batch_index >= output_rank0 || input_rank0 == 0 || shape.empty())
            return;

        std::vector<int> output_shape = op->outputs[0]->shape;
        if (output_shape.size() == shape.size())
        {
            for (size_t i = 0; i < shape.size(); i++)
            {
                if (output_shape[i] == -1 && shape[i] > 0)
                    output_shape[i] = shape[i];
            }
        }

        const int batch_size = output_shape[batch_index];
        const int batch_axis_size = batch_size > 0 ? batch_size : -1;
        bool keep_batch_index = false;
        int batch_index_unreshaped = batch_index;
        if (input_rank0 > 0)
        {
            if (batch_index == 0 && ((batch_size == 1 && shape[0] == 1 && op->inputs[0]->shape[0] == 1) || (batch_size > 1 && op->inputs[0]->shape[0] == batch_size)))
            {
                keep_batch_index = true;
            }
            else if (output_rank0 > 0)
            {
                if (batch_index == output_rank0 - 1 && ((batch_size == 1 && shape[shape.size() - 1] == 1 && op->inputs[0]->shape[input_rank0 - 1] == 1) || (batch_size > 1 && op->inputs[0]->shape[input_rank0 - 1] == batch_size)))
                {
                    keep_batch_index = true;
                    batch_index_unreshaped = input_rank0 - 1;
                }
                else
                {
                    batch_index_unreshaped = -1;

                    // batch index is in the middle, let's consider the left and right parts
                    int left = 1;
                    int right = 1;
                    for (int i = 0; i < batch_index; i++)
                    {
                        if (output_shape[i] == -1)
                        {
                            left = -1;
                            break;
                        }
                        left *= output_shape[i];
                    }
                    for (int i = batch_index + 1; i < (int)output_shape.size(); i++)
                    {
                        if (output_shape[i] == -1)
                        {
                            right = -1;
                            break;
                        }
                        right *= output_shape[i];
                    }

                    // try to find batch index in the output shape
                    if (left > 0)
                    {
                        int left2 = 1;
                        for (int i = 0; i < input_rank0 - 1; i++)
                        {
                            if (op->inputs[0]->shape[i] == -1)
                            {
                                left2 = -1;
                                break;
                            }
                            left2 *= op->inputs[0]->shape[i];
                            if (left2 == left && op->inputs[0]->shape[i + 1] == batch_axis_size)
                            {
                                batch_index_unreshaped = i + 1;
                                if (batch_index_unreshaped + 1 < input_rank0 && op->inputs[0]->shape[batch_index_unreshaped + 1] == batch_axis_size)
                                {
                                    // multiple axes can be batch index, give up
                                    batch_index_unreshaped = -1;
                                }
                                break;
                            }
                        }
                    }
                    if (right > 0)
                    {
                        int right2 = 1;
                        for (int i = input_rank0 - 1; i >= 1; i--)
                        {
                            if (op->inputs[0]->shape[i] == -1)
                            {
                                right2 = -1;
                                break;
                            }
                            right2 *= op->inputs[0]->shape[i];
                            if (right2 == right && op->inputs[0]->shape[i - 1] == batch_axis_size)
                            {
                                batch_index_unreshaped = i - 1;
                                if (batch_index_unreshaped - 1 >= 0 && op->inputs[0]->shape[batch_index_unreshaped - 1] == batch_axis_size)
                                {
                                    // multiple axes can be batch index, give up
                                    batch_index_unreshaped = -1;
                                }
                                break;
                            }
                        }
                    }

                    if (batch_index_unreshaped >= 0)
                        keep_batch_index = true;
                }
            }
        }

        if (keep_batch_index)
        {
            Operand* r = op->inputs[0];
            if (r->params.find("__batch_index") == r->params.end())
            {
                r->params["__batch_index"] = batch_index_unreshaped;

                solve_batch_index_backward(r);
                solve_batch_index_forward(r);
            }
        }
        else
        {
            // give up reshape across batch index
        }
    }
    else if (op->type == "Tensor.unflatten")
    {
        int dim = op->params.at("dim").i;
        const int sizes_rank = (int)op->params.at("sizes").ai.size();
        if (dim < 0)
            dim += input_rank0;

        int batch_index_unflattened = batch_index;
        if (dim <= batch_index && batch_index < dim + sizes_rank)
            batch_index_unflattened = 233;
        else if (batch_index >= dim + sizes_rank)
            batch_index_unflattened = batch_index - sizes_rank + 1;

        Operand* r = op->inputs[0];
        if (r->params.find("__batch_index") == r->params.end())
        {
            r->params["__batch_index"] = batch_index_unflattened;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else if (op->type == "torch.flatten")
    {
        int start_dim = op->params.at("start_dim").i;
        int end_dim = op->params.at("end_dim").i;
        if (start_dim < 0)
            start_dim += input_rank0;
        if (end_dim < 0)
            end_dim += input_rank0;

        if (input_rank0 <= start_dim || input_rank0 <= end_dim)
            return;

        int batch_index_unflattened = batch_index;
        if (batch_index == start_dim && start_dim != end_dim)
            return;
        if (batch_index > start_dim)
            batch_index_unflattened = batch_index + end_dim - start_dim;

        Operand* r = op->inputs[0];
        if (r->params.find("__batch_index") == r->params.end())
        {
            r->params["__batch_index"] = batch_index_unflattened;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else if (op->type == "Tensor.slice")
    {
        int batch_index_sliced = solve_slice_batch_index_backward(op, batch_index, input_rank0, output_rank0);

        Operand* r = op->inputs[0];
        if (r->params.find("__batch_index") == r->params.end())
        {
            r->params["__batch_index"] = batch_index_sliced;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else if (op->type == "Tensor.select")
    {
        int input_rank = input_rank0;
        if (input_rank == 0 && output_rank0 > 0)
            input_rank = output_rank0 + 1;

        int batch_index_selected = solve_select_batch_index_backward(op, batch_index, input_rank);

        Operand* r = op->inputs[0];
        if (r->params.find("__batch_index") == r->params.end())
        {
            r->params["__batch_index"] = batch_index_selected;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else if (op->type == "pnnx.SliceIndexes")
    {
        // pass
    }
    else if (op->type == "torch.squeeze")
    {
        if (batch_index == 233)
        {
            // give up
            return;
        }

        int batch_index_unsqueezed = batch_index;
        const std::vector<int>& input_shape = op->inputs[0]->shape;
        if (!input_shape.empty())
        {
            std::vector<int> squeezed(input_rank0, 0);
            if (op->has_param("dim"))
            {
                if (op->params.at("dim").type == 2)
                {
                    int dim = op->params.at("dim").i;
                    if (dim < 0 && input_rank0 > 0)
                        dim += input_rank0;

                    if (dim >= 0 && dim < input_rank0 && input_shape[dim] == 1)
                        squeezed[dim] = 1;
                }
                else
                {
                    const std::vector<int>& dims = op->params.at("dim").ai;
                    for (auto d : dims)
                    {
                        int dim = d;
                        if (dim < 0 && input_rank0 > 0)
                            dim += input_rank0;

                        if (dim >= 0 && dim < input_rank0 && input_shape[dim] == 1)
                            squeezed[dim] = 1;
                    }
                }
            }
            else
            {
                for (int i = 0; i < input_rank0; i++)
                {
                    if (input_shape[i] == 1)
                        squeezed[i] = 1;
                }
            }

            batch_index_unsqueezed = -1;
            int output_axis = 0;
            for (int i = 0; i < input_rank0; i++)
            {
                if (squeezed[i])
                    continue;

                if (output_axis == batch_index)
                {
                    batch_index_unsqueezed = i;
                    break;
                }

                output_axis += 1;
            }

            if (batch_index_unsqueezed < 0)
                return;
        }
        else if (op->has_param("dim"))
        {
            int squeeze_before_batch = 0;
            if (op->params.at("dim").type == 2)
            {
                int dim = op->params.at("dim").i;
                if (dim < 0 && input_rank0 > 0)
                    dim += input_rank0;

                if (dim >= 0 && dim <= batch_index)
                {
                    squeeze_before_batch += 1;
                }
            }
            else
            {
                const std::vector<int>& dims = op->params.at("dim").ai;
                for (auto d : dims)
                {
                    int dim = d;
                    if (dim < 0 && input_rank0 > 0)
                        dim += input_rank0;

                    if (dim >= 0 && dim <= batch_index)
                    {
                        squeeze_before_batch += 1;
                    }
                }
            }
            batch_index_unsqueezed = batch_index + squeeze_before_batch;
        }
        else
        {
            // give up for squeezing all
            return;
        }

        Operand* r = op->inputs[0];
        if (r->params.find("__batch_index") == r->params.end())
        {
            r->params["__batch_index"] = batch_index_unsqueezed;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else if (op->type == "torch.unsqueeze")
    {
        int dim = op->params.at("dim").i;
        if (dim < 0 && input_rank0 > 0)
            dim += input_rank0 + 1;
        if (dim < 0 && input_rank0 == 0 && output_rank0 > 0)
            dim += output_rank0;

        int batch_index_squeezed = batch_index;
        if (dim >= 0 && dim == batch_index)
        {
            batch_index_squeezed = 233;
        }
        else if (dim >= 0 && dim <= batch_index)
        {
            batch_index_squeezed = batch_index - 1;
        }

        Operand* r = op->inputs[0];
        if (r->params.find("__batch_index") == r->params.end())
        {
            r->params["__batch_index"] = batch_index_squeezed;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
    else
    {
        for (Operand* r : op->inputs)
        {
            if (r->params.find("__batch_index") != r->params.end())
                continue;

            if (op->type == "BinaryOp" && batch_index >= 0 && batch_index < output_rank0 && (int)r->shape.size() <= output_rank0 - 1)
            {
                bool broadcast_without_batch = true;
                int j = (int)r->shape.size() - 1;
                for (int i = output_rank0 - 1; i >= 0; i--)
                {
                    if (i == batch_index)
                        continue;

                    int dim = 1;
                    if (j >= 0)
                        dim = r->shape[j--];

                    if (dim != 1 && dim != op->outputs[0]->shape[i])
                    {
                        broadcast_without_batch = false;
                        break;
                    }
                }

                if (broadcast_without_batch)
                {
                    r->params["__batch_index"] = 233;
                    continue;
                }
            }

            r->params["__batch_index"] = batch_index;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
}

void solve_batch_index(Graph& graph)
{
    // assign known operator
    for (Operator* op : graph.ops)
    {
        if (op->type == std::string("F.scaled_dot_product_attention"))
        {
            for (Operand* r : op->inputs)
            {
                r->params["__batch_index"] = r->shape.empty() || r->shape.size() > 3 ? 0 : 233;
            }
            for (Operand* r : op->outputs)
            {
                r->params["__batch_index"] = r->shape.empty() || r->shape.size() > 3 ? 0 : 233;
            }
            continue;
        }

        if (is_known_operator_with_batch_index_0(op))
        {
            const int batch_index = get_known_operator_batch_index(op);

            if (op->type == std::string("F.grid_sample"))
            {
                op->inputs[1]->params["__batch_index"] = batch_index;
            }
            if (op->type == std::string("torch.bmm"))
            {
                op->inputs[1]->params["__batch_index"] = batch_index;
            }

            op->inputs[0]->params["__batch_index"] = batch_index;
            op->outputs[0]->params["__batch_index"] = batch_index;
        }

        if (is_known_operator_with_batch_first_param(op))
        {
            bool batch_first = false;
            if (op->params.find("batch_first") != op->params.end())
            {
                batch_first = op->params["batch_first"].b;
            }

            op->inputs[0]->params["__batch_index"] = batch_first ? 0 : 1;
            op->outputs[0]->params["__batch_index"] = batch_first ? 0 : 1;

            for (size_t j = 1; j < op->inputs.size(); j++)
            {
                if (op->type == "nn.MultiheadAttention")
                {
                    if (op->inputnames.size() == op->inputs.size() && op->inputnames[j] == "attn_mask")
                    {
                        // no batch for mha attn_mask
                        op->inputs[j]->params["__batch_index"] = 233;
                    }
                    else
                    {
                        op->inputs[j]->params["__batch_index"] = batch_first ? 0 : 1;
                    }

                    continue;
                }

                op->inputs[j]->params["__batch_index"] = 1;
            }

            for (size_t j = 1; j < op->outputs.size(); j++)
            {
                op->outputs[j]->params["__batch_index"] = 1;
            }
        }
    }

    // batch index propagate
    for (Operator* op : graph.ops)
    {
        for (Operand* r : op->inputs)
        {
            solve_batch_index_backward(r);
        }

        for (Operand* r : op->outputs)
        {
            solve_batch_index_forward(r);
        }
    }

    // always treat 1-dim tensor as no batch axis
    for (Operand* r : graph.operands)
    {
        if (r->shape.size() == 1)
        {
            r->params["__batch_index"] = 233;
        }
    }

    // fallback axis 233 for unknown
    for (Operand* r : graph.operands)
    {
        if (r->params.find("__batch_index") == r->params.end())
        {
            r->params["__batch_index"] = 233;
        }
    }
}

} // namespace ncnn

} // namespace pnnx
