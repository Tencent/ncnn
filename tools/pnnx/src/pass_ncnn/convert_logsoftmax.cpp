// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convert_logsoftmax.h"

namespace pnnx {

namespace ncnn {

void convert_logsoftmax(Graph& graph)
{
    int index = 0;

    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];
        if (op->type != "nn.LogSoftmax")
            continue;

        op->type = "nn.Softmax";
        auto old_output = op->outputs[0];
        auto operand = graph.new_operand("logsoftmax_intermediate" + std::to_string(index));
        op->outputs[0] = operand;
        auto log = graph.new_operator_after("UnaryOp", "logsoftmax_log" + std::to_string(index), op);
        log->inputs.push_back(operand);
        log->outputs.push_back(old_output);
        log->params["0"] = 8;

        operand->consumers.push_back(log);
        operand->producer = op;

        index++;
    }
}

} // namespace ncnn

} // namespace pnnx
