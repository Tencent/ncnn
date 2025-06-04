// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_multiheadattention_sameqkv.h"

namespace pnnx {

void fuse_multiheadattention_sameqkv(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "nn.MultiheadAttention")
            continue;

        const int input_count = (int)op->inputs.size();
        if (input_count != 3 && input_count != 4)
            continue;

        bool same_qkv = false;
        if (op->inputs[0] == op->inputs[1] && op->inputs[1] == op->inputs[2])
        {
            if (input_count == 4)
                same_qkv = true;
            if (input_count == 3 && (op->inputnames.empty() || op->inputnames[2] != "attn_mask"))
                same_qkv = true;
        }

        if (!same_qkv)
            continue;

        if (input_count == 3)
        {
            op->inputs.resize(1);

            op->inputnames.resize(1);
            op->inputnames[0] = "input";
        }
        else // if (input_count == 4)
        {
            op->inputs[1] = op->inputs[3];
            op->inputs.resize(2);

            op->inputnames.resize(2);
            op->inputnames[0] = "input";
            op->inputnames[1] = "attn_mask";
        }

        // reset consumer references to 1
        Operand* qkv_input = op->inputs[0];
        qkv_input->remove_consumer(op);
        qkv_input->remove_consumer(op);
    }
}

} // namespace pnnx
