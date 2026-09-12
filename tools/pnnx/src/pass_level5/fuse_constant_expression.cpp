// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_constant_expression.h"

#include <string.h>
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_constant_expression(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Expression")
                continue;

            if (op->inputs.size() != 0)
            {
                // dynamic expression
                continue;
            }

            Operand* expr_output = op->outputs[0];

            // Tensor.index / Tensor.index_put receive a tensor-list as the
            // "indices" argument; a constant there (e.g. a single-element i64
            // Attribute [1] inlined by fuse_expression) must stay a tensor
            // operand - folding it into a scalar param would change advanced
            // indexing x[[1]] into scalar indexing x[1] (or produce an illegal
            // index_put(indices=(1,)) call).
            auto is_index_op_indices = [&](const Operator* x) {
                if (x->type != "Tensor.index" && x->type != "Tensor.index_put")
                    return false;
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == expr_output && !x->inputnames[j].empty() && x->inputnames[j] == "indices")
                        return true;
                }
                return false;
            };

            std::vector<Operator*> new_consumers;
            for (auto x : expr_output->consumers)
            {
                if (x->inputnames.empty())
                {
                    // x is not a function
                    new_consumers.push_back(x);
                    continue;
                }

                if (is_index_op_indices(x))
                {
                    // keep the tensor-list operand for indexing ops
                    new_consumers.push_back(x);
                    continue;
                }

                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == expr_output && x->inputnames[j].empty())
                    {
                        // no param key
                        new_consumers.push_back(x);
                        break;
                    }
                }
            }

            if (new_consumers == expr_output->consumers)
                continue;

            matched = true;

            Parameter ep = Parameter::parse_from_string(op->params.at("expr").s);

            for (auto& x : expr_output->consumers)
            {
                if (x->inputnames.empty())
                {
                    // x is not a function
                    continue;
                }

                std::vector<Operand*> new_inputs;
                std::vector<std::string> new_inputnames;
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == expr_output && !x->inputnames[j].empty())
                    {
                        if (is_index_op_indices(x))
                        {
                            // keep the tensor-list operand for indexing ops
                            new_inputs.push_back(x->inputs[j]);
                            new_inputnames.push_back(x->inputnames[j]);
                            continue;
                        }

                        // fuse constant
                        x->params[x->inputnames[j]] = ep;
                    }
                    else
                    {
                        new_inputs.push_back(x->inputs[j]);
                        new_inputnames.push_back(x->inputnames[j]);
                    }
                }

                x->inputs = new_inputs;
                x->inputnames = new_inputnames;
            }

            expr_output->consumers = new_consumers;

            if (expr_output->consumers.empty())
            {
                // delete expression and expr_output

                expr_output->producer = 0;
                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), expr_output));
                delete expr_output;

                op->inputs.clear();
                op->outputs.clear();

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
                delete op;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
