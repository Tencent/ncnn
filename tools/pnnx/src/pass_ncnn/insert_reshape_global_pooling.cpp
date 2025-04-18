// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "insert_reshape_global_pooling.h"

#include <algorithm>
#include <set>

namespace pnnx {

namespace ncnn {

static bool is_known_operator_handle_flatten_0(const Operator* op)
{
    // opeartors that have similar behavior for (1,c,1,1,1)/(1,c,1,1)/(1,c,1) and (1,c)

    static const char* operator_handle_flatten_0[] = {
        "F.batch_norm",
        "F.celu",
        "F.conv1d",
        "F.conv2d",
        "F.conv3d",
        "F.elu",
        "F.gelu",
        "F.glu",
        "F.hardshrink",
        "F.hardsigmoid",
        "F.hardswish",
        "F.hardtanh",
        "F.leaky_relu",
        "F.linear",
        "F.log_softmax",
        "F.logsigmoid",
        "F.prelu",
        "F.relu",
        "F.relu6",
        "F.rrelu",
        "F.selu",
        "F.sigmoid",
        "F.silu",
        "F.softmax",
        "F.softmin",
        "F.softplus",
        "F.softshrink",
        "F.softsign",
        "F.tanh",
        "F.tanhshrink",
        "F.threshold",

        "nn.BatchNorm1d",
        "nn.BatchNorm2d",
        "nn.BatchNorm3d",
        "nn.CELU",
        "nn.Conv1d",
        "nn.Conv2d",
        "nn.Conv3d",
        "nn.ELU",
        "nn.GELU",
        "nn.GLU",
        "nn.Hardshrink",
        "nn.Hardsigmoid",
        "nn.Hardswish",
        "nn.Hardtanh",
        "nn.LeakyReLU",
        "nn.Linear",
        "nn.LogSigmoid",
        "nn.LogSoftmax",
        "nn.PReLU",
        "nn.ReLU",
        "nn.ReLU6",
        "nn.RReLU",
        "nn.SELU",
        "nn.Sigmoid",
        "nn.SiLU",
        "nn.Softmax",
        "nn.Softmin",
        "nn.Softplus",
        "nn.Softshrink",
        "nn.Softsign",
        "nn.Tanh",
        "nn.Tanhshrink",
        "nn.Threshold",

        "torch.abs",
        "torch.acos",
        "torch.acosh",
        "torch.asin",
        "torch.asinh",
        "torch.atan",
        "torch.atanh",
        "torch.atan2",
        "torch.ceil",
        "torch.clamp",
        "torch.cos",
        "torch.cosh",
        "torch.exp",
        "torch.floor",
        "torch.imag",
        "torch.log",
        "torch.log10",
        "torch.neg",
        "torch.pow",
        "torch.real",
        "torch.reciprocal",
        "torch.rsqrt",
        "torch.sign",
        "torch.sin",
        "torch.sinh",
        "torch.sqrt",
        "torch.square",
        "torch.tan",
        "torch.tanh",
        "torch.trunc",
    };

    const size_t operator_handle_flatten_0_count = sizeof(operator_handle_flatten_0) / sizeof(const char*);
    for (size_t i = 0; i < operator_handle_flatten_0_count; i++)
    {
        if (op->type == operator_handle_flatten_0[i])
            return true;
    }

    return false;
}

static int is_global_pooling(const Operator* op)
{
    static const char* operator_with_flatten_state_0[] = {
        "F.adaptive_avg_pool2d",
        "F.adaptive_avg_pool3d",
        "F.adaptive_max_pool2d",
        "F.adaptive_max_pool3d",
        "nn.AdaptiveAvgPool2d",
        "nn.AdaptiveAvgPool3d",
        "nn.AdaptiveMaxPool2d",
        "nn.AdaptiveMaxPool3d",
    };

    const size_t operator_with_flatten_state_0_count = sizeof(operator_with_flatten_state_0) / sizeof(const char*);
    for (size_t i = 0; i < operator_with_flatten_state_0_count; i++)
    {
        if (op->type == operator_with_flatten_state_0[i])
        {
            // output_size=(1,1)
            // output_size=(1,1,1)
            const std::vector<int>& output_size = op->params.at("output_size").ai;
            if (output_size == std::vector<int> {1, 1})
                return 3;
            if (output_size == std::vector<int> {1, 1, 1})
                return 4;
        }
    }

    return 0;
}

static int insert_reshape_global_pooling_forward(Operand* operand, int pooled_rank, Graph& graph)
{
    for (size_t i = 0; i < operand->consumers.size(); i++)
    {
        Operator* op = operand->consumers[i];

        if (op->type == "Tensor.reshape" || op->type == "Tensor.view")
        {
            // reshape discard flatten state
            break;
        }

        if (is_known_operator_handle_flatten_0(op))
        {
            for (Operand* r : op->outputs)
            {
                int ret = insert_reshape_global_pooling_forward(r, pooled_rank, graph);
                if (ret)
                    return ret;
            }
            continue;
        }

        if (op->type == "pnnx.Expression")
        {
            // if it can be auto-broadcast
            // (1,c) with (1,c,d,h,w)/(1,c,h,w)/(1,c,w)/(1,c)
            if (operand->shape.size() == 4 && op->outputs[0]->shape.size() >= 2)
            {
                if (operand->shape[1] == op->outputs[0]->shape[1])
                    break;
            }
        }

        fprintf(stderr, "insert_reshape_global_pooling_forward %s %s\n", op->name.c_str(), operand->name.c_str());

        // insert reshape (1,c,1,1) before op
        Operator* reshape0 = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnreshape0", op);

        Operand* reshape0_out = graph.new_operand(op->name + "_ncnnreshape0_out");

        reshape0->inputs.push_back(operand);
        reshape0->outputs.push_back(reshape0_out);

        operand->consumers[i] = reshape0;

        for (size_t j = 0; j < op->inputs.size(); j++)
        {
            if (op->inputs[j] == operand)
            {
                op->inputs[j] = reshape0_out;
            }
        }

        reshape0_out->producer = reshape0;
        reshape0_out->consumers.push_back(op);

        reshape0_out->params["__batch_index"] = 0;

        if (pooled_rank == 3)
            reshape0->params["shape"] = std::vector<int> {1, -1, 1, 1};
        if (pooled_rank == 4)
            reshape0->params["shape"] = std::vector<int> {1, -1, 1, 1, 1};

        return 1;
    }

    return 0;
}

void insert_reshape_global_pooling(Graph& graph)
{
    int inserted = 0;

    while (1)
    {
        inserted = 0;

        for (Operator* op : graph.ops)
        {
            int pooled_rank = is_global_pooling(op);
            if (pooled_rank == 0)
                continue;

            // look for all output consumers
            // insert reshape (1,c,1,1) if it cannot handle flatten
            inserted = insert_reshape_global_pooling_forward(op->outputs[0], pooled_rank, graph);
            if (inserted)
            {
                break;
            }
        }

        if (inserted == 0)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
