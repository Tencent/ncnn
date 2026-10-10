// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

            op->inputnames = {"input"};
        }
        else // if (input_count == 4)
        {
            op->inputs[1] = op->inputs[3];
            op->inputs.resize(2);

            op->inputnames = {"input", "attn_mask"};
        }

        // reset consumer references to 1
        Operand* qkv_input = op->inputs[0];
        qkv_input->remove_consumer(op);
        qkv_input->remove_consumer(op);
    }
}

} // namespace pnnx
