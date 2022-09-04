// Copyright (c) 2022 Xiaomi Corp.        (author: Fangjun Kuang)
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

#include "fuse_input_unpack.h"
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_input_unpack(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "prim::TupleUnpack")
                continue;

            if (op->inputs.size() != 1)
                continue;

            Operator* op2 = op->inputs[0]->producer;


            if (op2->type != "pnnx.Input")
                continue;

            if (op2->outputs[0]->consumers.size() != 1)
                continue;

            for (int i = 0; i != op->outputs.size(); ++i) {
              Operator* new_op = graph.new_operator_before("pnnx.Input", op2->name + std::to_string(i), op2);
              op->outputs[i]->producer = new_op;
              new_op->outputs.push_back(op->outputs[i]);
            }

            // erase op and op2

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
            delete op;

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op2));
            delete op2;

            matched = true;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
