// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_slice_copy.h"

#include <limits.h>
#include <algorithm>
#include <stack>
#include "pass_level2.h"

namespace pnnx {

void fuse_slice_copy(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.copy")
                continue;

            // collect slice / select op chain
            std::stack<Operator*> slice_select_ops;
            int descent_dim_current = INT_MAX;
            Operand* in0 = op->inputs[0];
            while (in0->producer->type == "Tensor.slice" || in0->producer->type == "Tensor.select")
            {
                Operator* sop = in0->producer;
                if (sop->type == "Tensor.slice")
                {
                    if (!sop->has_param("dim") && !sop->has_param("dims"))
                    {
                        fprintf(stderr, "dynamic dims in slice copy chain is not supported\n");
                        break;
                    }

                    int dims0 = sop->has_param("dim") ? sop->params.at("dim").i : sop->params.at("dims").ai[0];
                    if (descent_dim_current < dims0)
                    {
                        break;
                    }

                    descent_dim_current = dims0;
                }

                if (sop->type == "Tensor.select")
                {
                    if (!sop->has_param("dim"))
                    {
                        fprintf(stderr, "dynamic dim in select copy chain is not supported\n");
                        break;
                    }

                    int dim = sop->params.at("dim").i;
                    if (descent_dim_current < dim)
                    {
                        break;
                    }

                    descent_dim_current = dim;
                }

                slice_select_ops.push(sop);
                in0 = sop->inputs[0];
            }

            matched = true;

            if (slice_select_ops.empty())
            {
                // eliminate noop copy
                Operand* out = op->outputs[0];

                for (auto& x : out->consumers)
                {
                    for (size_t j = 0; j < x->inputs.size(); j++)
                    {
                        if (x->inputs[j] == out)
                            x->inputs[j] = op->inputs[1];
                    }

                    op->inputs[1]->consumers.push_back(x);
                }

                op->inputs[0]->remove_consumer(op);
                op->inputs[1]->remove_consumer(op);

                op->inputs[1]->name = out->name;

                out->producer = 0;
                out->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), out));
                delete out;

                op->inputs.clear();
                op->outputs.clear();

                graph.ops.erase(graph.ops.begin() + i);
                delete op;

                break;
            }

            Operator* top_sop = slice_select_ops.top();

            op->type = "Tensor.slice_copy";

            // insert clone before any slices
            Operator* op_clone = graph.new_operator_before("Tensor.clone", op->name + "_ncnnclone", top_sop);
            Operand* clone_out = graph.new_operand(op->name + "_ncnnclone_out");

            clone_out->type = top_sop->inputs[0]->type;
            clone_out->shape = top_sop->inputs[0]->shape;

            op_clone->inputs.push_back(top_sop->inputs[0]);
            top_sop->inputs[0]->consumers.push_back(op_clone);

            op_clone->outputs.push_back(clone_out);
            clone_out->producer = op_clone;

            op->inputs[0]->remove_consumer(op);
            op->inputs[0] = clone_out;
            clone_out->consumers.push_back(op);

            if (top_sop->type == "Tensor.slice")
            {
                // shadow slice params and inputs to slice_copy
                op->params = top_sop->params;
                // self op->inputs[0]
                // src  op->inputs[1]

                if (top_sop->has_input("start"))
                {
                    Operand* start = top_sop->named_input("start");
                    op->inputs.push_back(start);
                    op->inputnames.push_back("start");
                    start->consumers.push_back(op);
                }
                if (top_sop->has_input("starts"))
                {
                    Operand* starts = top_sop->named_input("starts");
                    op->inputs.push_back(starts);
                    op->inputnames.push_back("starts");
                    starts->consumers.push_back(op);
                }
                if (top_sop->has_input("end"))
                {
                    Operand* end = top_sop->named_input("end");
                    op->inputs.push_back(end);
                    op->inputnames.push_back("end");
                    end->consumers.push_back(op);
                }
                if (top_sop->has_input("ends"))
                {
                    Operand* ends = top_sop->named_input("ends");
                    op->inputs.push_back(ends);
                    op->inputnames.push_back("ends");
                    ends->consumers.push_back(op);
                }
                if (top_sop->has_input("step"))
                {
                    Operand* step = top_sop->named_input("step");
                    op->inputs.push_back(step);
                    op->inputnames.push_back("step");
                    step->consumers.push_back(op);
                }
                if (top_sop->has_input("steps"))
                {
                    Operand* steps = top_sop->named_input("steps");
                    op->inputs.push_back(steps);
                    op->inputnames.push_back("steps");
                    steps->consumers.push_back(op);
                }
                if (top_sop->has_input("select"))
                {
                    Operand* select = top_sop->named_input("select");
                    op->inputs.push_back(select);
                    op->inputnames.push_back("select");
                    select->consumers.push_back(op);
                }
                if (top_sop->has_input("selects"))
                {
                    Operand* selects = top_sop->named_input("selects");
                    op->inputs.push_back(selects);
                    op->inputnames.push_back("selects");
                    selects->consumers.push_back(op);
                }
            }
            else // if (top_sop->type == "Tensor.select")
            {
                op->params["dim"] = top_sop->params.at("dim").i;
                op->params["start"] = 0;
                op->params["end"] = 0;
                op->params["step"] = 0;

                if (top_sop->has_param("index"))
                {
                    op->params["select"] = top_sop->params.at("index").i;
                }
                else // if (top_sop->has_input("index"))
                {
                    Operand* index = top_sop->named_input("index");
                    op->inputs.push_back(index);
                    op->inputnames.push_back("select");
                    index->consumers.push_back(op);
                }
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
