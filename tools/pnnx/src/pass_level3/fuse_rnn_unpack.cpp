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

#include "fuse_rnn_unpack.h"
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_rnn_unpack(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.RNN" && op->type != "nn.LSTM" && op->type != "nn.GRU")
                continue;

            if (op->outputs.size() != 1)
                continue;

            if (op->outputs[0]->consumers.size() != 1)
                continue;

            Operator* op2 = op->outputs[0]->consumers[0];
            if (op2->type != "prim::TupleUnpack")
                continue;

            matched = true;

            op->outputs[0]->producer = 0;
            op->outputs[0]->remove_consumer(op2);

            for (auto& x : op2->outputs)
            {
                x->producer = op;
            }

            op->outputs = op2->outputs;

            // outputs may be swapped, fix the ugly order
            if (op->params.find("pnnx_rnn_output_swapped") != op->params.end() && op->params.at("pnnx_rnn_output_swapped").i == 1)
            {
                op->params.erase("pnnx_rnn_output_swapped");
                if (op->type == "nn.RNN" || op->type == "nn.GRU")
                {
                    std::swap(op->outputs[0], op->outputs[1]);
                }
                if (op->type == "nn.LSTM")
                {
                    std::swap(op->outputs[0], op->outputs[2]);
                    std::swap(op->outputs[1], op->outputs[2]);
                }
            }

            op2->inputs.clear();
            op2->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op2));

            delete op2;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
