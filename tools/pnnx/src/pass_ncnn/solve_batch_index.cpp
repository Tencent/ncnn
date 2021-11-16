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

#include "solve_batch_index.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void solve_batch_index_backward(Operand* operand);
void solve_batch_index_forward(Operand* operand)
{
    int batch_index = operand->params["__batch_index"].i;

    for (Operator* op : operand->consumers)
    {
        if (op->type == "torch.permute" || op->type == "Tensor.permute")
        {
            const std::vector<int>& dims = op->params.at("dims").ai;

            int batch_index_permuted = -1;
            for (int i = 0; i < (int)dims.size(); i++)
            {
                if (dims[i] == batch_index)
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
        else if (op->type == "nn.RNN" || op->type == "nn.LSTM" || op->type == "nn.GRU")
        {
            {
                Operand* r = op->outputs[0];
                if (r->params.find("__batch_index") != r->params.end())
                    continue;

                r->params["__batch_index"] = batch_index;

                solve_batch_index_forward(r);
                solve_batch_index_backward(r);
            }

            for (size_t i = 1; i < op->outputs.size(); i++)
            {
                Operand* r = op->outputs[i];
                if (r->params.find("__batch_index") != r->params.end())
                    continue;

                r->params["__batch_index"] = 1;

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

void solve_batch_index_backward(Operand* operand)
{
    int batch_index = operand->params["__batch_index"].i;

    Operator* op = operand->producer;

    if (op->type == "torch.permute")
    {
        const std::vector<int>& dims = op->params.at("dims").ai;

        int batch_index_permuted = dims[batch_index];

        for (Operand* r : op->inputs)
        {
            if (r->params.find("__batch_index") != r->params.end())
                continue;

            r->params["__batch_index"] = batch_index_permuted;

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

            r->params["__batch_index"] = batch_index;

            solve_batch_index_backward(r);
            solve_batch_index_forward(r);
        }
    }
}

void solve_batch_index(Graph& graph)
{
    // assign input and ongoing
    for (int i = 0; i < (int)graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "pnnx.Input")
            continue;

        for (Operand* r : op->outputs)
        {
            r->params["__batch_index"] = 0;

            solve_batch_index_forward(r);
        }
    }
}

} // namespace ncnn

} // namespace pnnx
