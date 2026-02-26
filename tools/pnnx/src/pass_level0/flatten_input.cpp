// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "flatten_input.h"

namespace pnnx {

void flatten_input(std::shared_ptr<torch::jit::Graph>& graph)
{
    while (1)
    {
        bool matched = false;

        for (torch::jit::Node* n : graph->nodes())
        {
            if (n->kind() != c10::prim::TupleUnpack && n->kind() != c10::prim::ListUnpack)
                continue;

            for (size_t i = 1; i < graph->inputs().size(); i++)
            {
                if (n->input(0) == graph->inputs()[i])
                {
                    matched = true;

                    for (size_t j = 0; j < n->outputs().size(); j++)
                    {
                        torch::jit::Value* v2 = graph->insertInput(i + 1 + j);
                        n->output(j)->replaceAllUsesWith(v2);
                    }
                    n->destroy();
                    graph->eraseInput(i);
                    break;
                }
            }

            if (matched)
                break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
