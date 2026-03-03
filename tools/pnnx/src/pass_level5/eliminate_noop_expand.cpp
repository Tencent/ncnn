// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop_expand.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_noop_expand(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.expand_as" && op->type != "Tensor.expand")
                continue;

            Operand* expand_out = op->outputs[0];

            bool all_consumers_are_expr = true;
            for (auto& x : expand_out->consumers)
            {
                if (x->type != "pnnx.Expression")
                {
                    all_consumers_are_expr = false;
                    break;
                }
            }

            if (!all_consumers_are_expr)
                continue;

            // Tensor.expand_as  expand   2 1 in b in2
            // pnnx.Expression   add      2 1 in2 b out

            const std::vector<int>& inshape = op->inputs[0]->shape;
            if (inshape.empty())
                continue;

            bool noop_expand = true;
            for (auto& x : expand_out->consumers)
            {
                const std::vector<int>& outshape = x->outputs[0]->shape;
                if (outshape.empty())
                {
                    noop_expand = false;
                    break;
                }

                // check if inshape can be binary broadcast to outshape
                if (inshape.size() != outshape.size())
                {
                    noop_expand = false;
                    break;
                }

                for (size_t j = 0; j < inshape.size(); j++)
                {
                    if ((inshape[j] == outshape[j] && outshape[j] != -1) || inshape[j] == 1 || outshape[j] == 1)
                        continue;

                    noop_expand = false;
                    break;
                }
            }

            // check if our expand is the base shape
            // so we do not drop expand for add(expand(x,shape),1.2)
            for (auto& x : expand_out->consumers)
            {
                const std::vector<int>& outshape = x->outputs[0]->shape;

                std::vector<int> broadcasted_shape = inshape;
                for (const auto& r : x->inputs)
                {
                    if (r == expand_out)
                        continue;

                    if (r->shape.size() != inshape.size())
                        continue;

                    for (size_t j = 0; j < broadcasted_shape.size(); j++)
                    {
                        broadcasted_shape[j] = std::max(broadcasted_shape[j], r->shape[j]);
                    }
                }

                if (broadcasted_shape != outshape)
                {
                    noop_expand = false;
                    break;
                }
            }

            if (!noop_expand)
                continue;

            // delete noop-like expand
            matched = true;

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            for (auto& x : expand_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == expand_out)
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            op->inputs[0]->name = expand_out->name;

            expand_out->producer = 0;
            expand_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), expand_out));
            delete expand_out;

            op->inputs.clear();
            op->outputs.clear();

            graph.ops.erase(graph.ops.begin() + i);
            delete op;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
