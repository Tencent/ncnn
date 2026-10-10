// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_maxpool_unpack.h"
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_maxpool_unpack(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.MaxPool1d" && op->type != "nn.MaxPool2d" && op->type != "nn.MaxPool3d")
                continue;

            Operator* op2 = op->outputs[0]->consumers[0];

            if (op->outputs.size() == 1 && op2->type != "prim::TupleUnpack")
            {
                if (op->params.find("return_indices") == op->params.end())
                    continue;

                if (op->params.at("return_indices").b == false)
                    continue;

                matched = true;

                // no indices returned actually
                op->params["return_indices"] = false;
                break;
            }

            if (op->outputs.size() != 1)
                continue;

            if (op->outputs[0]->consumers.size() != 1)
                continue;

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
