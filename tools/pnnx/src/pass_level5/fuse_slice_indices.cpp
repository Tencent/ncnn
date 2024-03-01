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

#include "fuse_slice_indices.h"

#include <string.h>
#include <algorithm>
#include <stack>
#include <vector>
#include "pass_level2.h"

namespace pnnx {

void fuse_slice_indices(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.slice" && op->type != "Tensor.select")
                continue;

            if (op->has_param("dims"))
            {
                // skip fused ones
                continue;
            }

            if (!op->has_param("dim"))
            {
                fprintf(stderr, "dynamic dim in slice/select chain is not supported\n");
                continue;
            }

            bool static_starts = true;
            bool static_ends = true;
            bool static_steps = true;
            bool static_selects = true;

            if (op->type == "Tensor.slice")
            {
                if (!op->has_param("start")) static_starts = false;
                if (!op->has_param("end")) static_ends = false;
                if (!op->has_param("step")) static_steps = false;
            }
            else // if (op->type == "Tensor.select")
            {
                if (!op->has_param("index")) static_selects = false;
            }

            int descent_dim_current = op->params.at("dim").i;

            // collect slice op chain
            std::stack<Operator*> slice_select_ops;
            Operator* top_sop = op;
            Operand* in0 = op->inputs[0];
            while (in0->producer->type == "Tensor.slice" || in0->producer->type == "Tensor.select")
            {
                Operator* sop = in0->producer;
                if (in0->consumers.size() != 1)
                {
                    // not single chain
                    break;
                }

                if (sop->has_param("dims"))
                {
                    // skip fused ones
                    break;
                }

                if (!sop->has_param("dim"))
                {
                    fprintf(stderr, "dynamic dim in slice/select chain is not supported\n");
                    break;
                }

                if (sop->type == "Tensor.slice")
                {
                    if (!sop->has_param("start")) static_starts = false;
                    if (!sop->has_param("end")) static_ends = false;
                    if (!sop->has_param("step")) static_steps = false;
                }
                else // if (sop->type == "Tensor.select")
                {
                    if (!sop->has_param("index")) static_selects = false;
                }

                int dim = sop->params.at("dim").i;

                if (dim < 0 && descent_dim_current <= dim)
                {
                    // not adjacent slice/select in chain
                    break;
                }

                if (dim < 0 && descent_dim_current >= 0)
                {
                    // not adjacent slice/select in chain
                    break;
                }

                // only allow select on same dim
                if (sop->type == "Tensor.select")
                {
                    if (descent_dim_current >= 0 && dim >= 0 && descent_dim_current < dim)
                    {
                        // not adjacent slice/select in chain
                        break;
                    }
                }
                else
                {
                    if (descent_dim_current >= 0 && dim >= 0 && descent_dim_current <= dim)
                    {
                        // not adjacent slice/select in chain
                        break;
                    }
                }

                descent_dim_current = dim;

                slice_select_ops.push(sop);
                top_sop = sop;
                in0 = sop->inputs[0];
            }

            if (slice_select_ops.empty())
            {
                // single orphaned slice/select
                continue;
            }

            matched = true;

            // construct one-step slice
            std::vector<int> new_dims;
            std::vector<int> new_starts;
            std::vector<int> new_ends;
            std::vector<int> new_steps;
            std::vector<int> new_selects;
            Operator* op_starts = 0;
            Operator* op_ends = 0;
            Operator* op_steps = 0;
            Operator* op_selects = 0;
            if (!static_starts) op_starts = graph.new_operator_before("pnnx.SliceIndexes", op->name + "_ncnnstarts", op);
            if (!static_ends) op_ends = graph.new_operator_before("pnnx.SliceIndexes", op->name + "_ncnnends", op);
            if (!static_steps) op_steps = graph.new_operator_before("pnnx.SliceIndexes", op->name + "_ncnnsteps", op);
            if (!static_selects) op_selects = graph.new_operator_before("pnnx.SliceIndexes", op->name + "_ncnnselects", op);

            std::vector<std::string> starts_indexes;
            std::vector<std::string> ends_indexes;
            std::vector<std::string> steps_indexes;
            std::vector<std::string> selects_indexes;

            while (!slice_select_ops.empty())
            {
                Operator* sop = slice_select_ops.top();
                slice_select_ops.pop();

                new_dims.push_back(sop->params.at("dim").i);

                if (sop->type == "Tensor.slice")
                {
                    if (static_starts)
                    {
                        new_starts.push_back(sop->params.at("start").type == 0 ? 0 : sop->params.at("start").i);
                    }
                    else if (sop->has_param("start"))
                    {
                        char tmp[32];
                        if (sop->params.at("start").type == 0)
                        {
                            sprintf(tmp, "0");
                        }
                        else
                        {
                            sprintf(tmp, "%d", sop->params.at("start").i);
                        }
                        starts_indexes.push_back(tmp);
                    }
                    else
                    {
                        char tmp[32];
                        sprintf(tmp, "@%d", (int)op_starts->inputs.size());
                        starts_indexes.push_back(tmp);
                        Operand* start = sop->named_input("start");
                        op_starts->inputs.push_back(start);
                        start->remove_consumer(sop);
                        start->consumers.push_back(op_starts);
                    }

                    if (static_ends)
                    {
                        new_ends.push_back(sop->params.at("end").type == 0 ? INT_MAX : sop->params.at("end").i);
                    }
                    else if (sop->has_param("end"))
                    {
                        char tmp[32];
                        if (sop->params.at("end").type == 0)
                        {
                            sprintf(tmp, "%d", INT_MAX);
                        }
                        else
                        {
                            sprintf(tmp, "%d", sop->params.at("end").i);
                        }
                        ends_indexes.push_back(tmp);
                    }
                    else
                    {
                        char tmp[32];
                        sprintf(tmp, "@%d", (int)op_ends->inputs.size());
                        ends_indexes.push_back(tmp);
                        Operand* end = sop->named_input("end");
                        op_ends->inputs.push_back(end);
                        end->remove_consumer(sop);
                        end->consumers.push_back(op_ends);
                    }

                    if (static_steps)
                    {
                        new_steps.push_back(sop->params.at("step").type == 0 ? 1 : sop->params.at("step").i);
                    }
                    else if (sop->has_param("step"))
                    {
                        char tmp[32];
                        if (sop->params.at("step").type == 0)
                        {
                            sprintf(tmp, "1");
                        }
                        else
                        {
                            sprintf(tmp, "%d", sop->params.at("step").i);
                        }
                        steps_indexes.push_back(tmp);
                    }
                    else
                    {
                        char tmp[32];
                        sprintf(tmp, "@%d", (int)op_steps->inputs.size());
                        steps_indexes.push_back(tmp);
                        Operand* step = sop->named_input("step");
                        op_steps->inputs.push_back(step);
                        step->remove_consumer(sop);
                        step->consumers.push_back(op_steps);
                    }

                    if (static_selects)
                    {
                        new_selects.push_back(INT_MAX);
                    }
                    else
                    {
                        char tmp[32];
                        sprintf(tmp, "%d", INT_MAX);
                        selects_indexes.push_back(tmp);
                    }
                }
                else // if (sop->type == "Tensor.select")
                {
                    if (static_starts)
                    {
                        new_starts.push_back(0);
                    }
                    else
                    {
                        starts_indexes.push_back("0");
                    }

                    if (static_ends)
                    {
                        new_ends.push_back(0);
                    }
                    else
                    {
                        ends_indexes.push_back("0");
                    }

                    if (static_steps)
                    {
                        new_steps.push_back(0);
                    }
                    else
                    {
                        steps_indexes.push_back("0");
                    }

                    if (static_selects)
                    {
                        new_selects.push_back(sop->params.at("index").type == 0 ? 0 : sop->params.at("index").i);
                    }
                    else if (sop->has_param("index"))
                    {
                        char tmp[32];
                        if (sop->params.at("index").type == 0)
                        {
                            sprintf(tmp, "0");
                        }
                        else
                        {
                            sprintf(tmp, "%d", sop->params.at("index").i);
                        }
                        selects_indexes.push_back(tmp);
                    }
                    else
                    {
                        char tmp[32];
                        sprintf(tmp, "@%d", (int)op_selects->inputs.size());
                        selects_indexes.push_back(tmp);
                        Operand* index = sop->named_input("index");
                        op_selects->inputs.push_back(index);
                        index->remove_consumer(sop);
                        index->consumers.push_back(op_selects);
                    }
                }

                {
                    // drop sop and sop output
                    Operand* sop_out = sop->outputs[0];

                    graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), sop_out));

                    delete sop_out;

                    graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), sop));

                    delete sop;
                }
            }

            new_dims.push_back(op->params.at("dim").i);

            if (op->type == "Tensor.slice")
            {
                if (static_starts)
                {
                    new_starts.push_back(op->params.at("start").type == 0 ? 0 : op->params.at("start").i);
                }
                else if (op->has_param("start"))
                {
                    char tmp[32];
                    if (op->params.at("start").type == 0)
                    {
                        sprintf(tmp, "0");
                    }
                    else
                    {
                        sprintf(tmp, "%d", op->params.at("start").i);
                    }
                    starts_indexes.push_back(tmp);
                }
                else
                {
                    char tmp[32];
                    sprintf(tmp, "@%d", (int)op_starts->inputs.size());
                    starts_indexes.push_back(tmp);
                    Operand* start = op->named_input("start");
                    op_starts->inputs.push_back(start);
                    start->remove_consumer(op);
                    start->consumers.push_back(op_starts);
                }

                if (static_ends)
                {
                    new_ends.push_back(op->params.at("end").type == 0 ? INT_MAX : op->params.at("end").i);
                }
                else if (op->has_param("end"))
                {
                    char tmp[32];
                    if (op->params.at("end").type == 0)
                    {
                        sprintf(tmp, "%d", INT_MAX);
                    }
                    else
                    {
                        sprintf(tmp, "%d", op->params.at("end").i);
                    }
                    ends_indexes.push_back(tmp);
                }
                else
                {
                    char tmp[32];
                    sprintf(tmp, "@%d", (int)op_ends->inputs.size());
                    ends_indexes.push_back(tmp);
                    Operand* end = op->named_input("end");
                    op_ends->inputs.push_back(end);
                    end->remove_consumer(op);
                    end->consumers.push_back(op_ends);
                }

                if (static_steps)
                {
                    new_steps.push_back(op->params.at("step").type == 0 ? 1 : op->params.at("step").i);
                }
                else if (op->has_param("step"))
                {
                    char tmp[32];
                    if (op->params.at("step").type == 0)
                    {
                        sprintf(tmp, "1");
                    }
                    else
                    {
                        sprintf(tmp, "%d", op->params.at("step").i);
                    }
                    steps_indexes.push_back(tmp);
                }
                else
                {
                    char tmp[32];
                    sprintf(tmp, "@%d", (int)op_steps->inputs.size());
                    steps_indexes.push_back(tmp);
                    Operand* step = op->named_input("step");
                    op_steps->inputs.push_back(step);
                    step->remove_consumer(op);
                    step->consumers.push_back(op_steps);
                }

                if (static_selects)
                {
                    new_selects.push_back(INT_MAX);
                }
                else if (op->has_param("index"))
                {
                    char tmp[32];
                    sprintf(tmp, "%d", INT_MAX);
                    selects_indexes.push_back(tmp);
                }
            }
            else // if (op->type == "Tensor.select")
            {
                if (static_starts)
                {
                    new_starts.push_back(0);
                }
                else
                {
                    starts_indexes.push_back("0");
                }

                if (static_ends)
                {
                    new_ends.push_back(0);
                }
                else
                {
                    ends_indexes.push_back("0");
                }

                if (static_steps)
                {
                    new_steps.push_back(0);
                }
                else
                {
                    steps_indexes.push_back("0");
                }

                if (static_selects)
                {
                    new_selects.push_back(op->params.at("index").type == 0 ? 0 : op->params.at("index").i);
                }
                else if (op->has_param("index"))
                {
                    char tmp[32];
                    if (op->params.at("index").type == 0)
                    {
                        sprintf(tmp, "0");
                    }
                    else
                    {
                        sprintf(tmp, "%d", op->params.at("index").i);
                    }
                    selects_indexes.push_back(tmp);
                }
                else
                {
                    char tmp[32];
                    sprintf(tmp, "@%d", (int)op_selects->inputs.size());
                    selects_indexes.push_back(tmp);
                    Operand* index = op->named_input("index");
                    op_selects->inputs.push_back(index);
                    index->remove_consumer(op);
                    index->consumers.push_back(op_selects);
                }
            }

            op->type = "Tensor.slice";

            op->params.clear();
            op->params["dims"] = new_dims;

            op->inputs.clear();
            op->inputnames.clear();

            op->inputs.push_back(in0);
            op->inputnames.push_back("input");

            in0->remove_consumer(top_sop);
            in0->consumers.push_back(op);

            if (static_starts)
            {
                op->params["starts"] = new_starts;
            }
            else
            {
                op_starts->params["indexes"] = starts_indexes;

                Operand* starts_out = graph.new_operand(op->name + "_ncnnstarts_out");
                starts_out->producer = op_starts;
                op_starts->outputs.push_back(starts_out);
                starts_out->consumers.push_back(op);
                op->inputs.push_back(starts_out);
                op->inputnames.push_back("starts");
            }

            if (static_ends)
            {
                op->params["ends"] = new_ends;
            }
            else
            {
                op_ends->params["indexes"] = ends_indexes;

                Operand* ends_out = graph.new_operand(op->name + "_ncnnends_out");
                ends_out->producer = op_ends;
                op_ends->outputs.push_back(ends_out);
                ends_out->consumers.push_back(op);
                op->inputs.push_back(ends_out);
                op->inputnames.push_back("ends");
            }

            if (static_steps)
            {
                op->params["steps"] = new_steps;
            }
            else
            {
                op_steps->params["indexes"] = steps_indexes;

                Operand* steps_out = graph.new_operand(op->name + "_ncnnsteps_out");
                steps_out->producer = op_steps;
                op_steps->outputs.push_back(steps_out);
                steps_out->consumers.push_back(op);
                op->inputs.push_back(steps_out);
                op->inputnames.push_back("steps");
            }

            if (static_selects)
            {
                op->params["selects"] = new_selects;
            }
            else
            {
                op_selects->params["indexes"] = selects_indexes;

                Operand* selects_out = graph.new_operand(op->name + "_ncnnselects_out");
                selects_out->producer = op_selects;
                op_selects->outputs.push_back(selects_out);
                selects_out->consumers.push_back(op);
                op->inputs.push_back(selects_out);
                op->inputnames.push_back("selects");
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
