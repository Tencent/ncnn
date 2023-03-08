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

            std::vector<Operator*> new_consumers;
            for (auto x : expr_output->consumers)
            {
                if (x->inputnames.empty())
                {
                    // x is not a function
                    new_consumers.push_back(x);
                    continue;
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
                    if (x->inputs[j] == expr_output)
                    {
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
