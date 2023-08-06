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
            std::stack<const Operator*> slice_select_ops;
            int descent_dim_current = INT_MAX;
            const Operand* in0 = op->inputs[0];
            while (in0->producer->type == "Tensor.slice" || in0->producer->type == "Tensor.select")
            {
                const Operator* sop = in0->producer;
                if (sop->type == "Tensor.slice")
                {
                    if (sop->params.find("dims") == sop->params.end()
                            || sop->params.find("starts") == sop->params.end()
                            || sop->params.find("ends") == sop->params.end()
                            || sop->params.find("steps") == sop->params.end())
                    {
                        fprintf(stderr, "dynamic index in slice copy chain is not supported\n");
                        break;
                    }

                    int dims0 = sop->params.at("dims").ai[0];
                    if (descent_dim_current < dims0)
                    {
                        break;
                    }

                    descent_dim_current = dims0;
                }

                if (sop->type == "Tensor.select")
                {
                    if (sop->params.find("dim") == sop->params.end()
                            || sop->params.find("index") == sop->params.end())
                    {
                        fprintf(stderr, "dynamic index in select copy chain is not supported\n");
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

            const Operator* top_sop = slice_select_ops.top();

            // construct one-step slice
            std::vector<int> new_dims;
            std::vector<int> new_starts;
            std::vector<int> new_ends;
            std::vector<int> new_steps;

            int select_dims_offset = 0;
            while (!slice_select_ops.empty())
            {
                const Operator* sop = slice_select_ops.top();
                slice_select_ops.pop();

                if (sop->type == "Tensor.slice")
                {
                    std::vector<int> dims = sop->params.at("dims").ai;
                    std::vector<int> starts = sop->params.at("starts").ai;
                    std::vector<int> ends = sop->params.at("ends").ai;
                    std::vector<int> steps = sop->params.at("steps").ai;

                    for (size_t j = 0; j < dims.size(); j++)
                    {
                        dims[j] += select_dims_offset;
                    }

                    new_dims.insert(new_dims.end(), dims.begin(), dims.end());
                    new_starts.insert(new_starts.end(), starts.begin(), starts.end());
                    new_ends.insert(new_ends.end(), ends.begin(), ends.end());
                    new_steps.insert(new_steps.end(), steps.begin(), steps.end());
                }
                else if (sop->type == "Tensor.select")
                {
                    int dim = sop->params.at("dim").i;
                    int index = sop->params.at("index").i;

                    dim += select_dims_offset;
                    int end = index + 1;
                    if (index == -1)
                        end = INT_MAX;

                    new_dims.push_back(dim);
                    new_starts.push_back(index);
                    new_ends.push_back(end);
                    new_steps.push_back(1);

                    select_dims_offset += 1;
                }
            }

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

            op->params["dims"] = new_dims;
            op->params["starts"] = new_starts;
            op->params["ends"] = new_ends;
            op->params["steps"] = new_steps;

            int input_rank = (int)op->inputs[0]->shape.size();
            if (input_rank == 0)
            {
                // insert view_as(sliced) for different or unknown rank
                Operator* op_slice = graph.new_operator_before("Tensor.slice", op->name + "_ncnnslice", op);
                Operator* op_view_as = graph.new_operator_before("Tensor.view_as", op->name + "_ncnnview_as", op);

                Operand* slice_out = graph.new_operand(op->name + "_ncnnslice_out");
                Operand* view_as_out = graph.new_operand(op->name + "_ncnnview_as_out");

                op_slice->params["dims"] = new_dims;
                op_slice->params["starts"] = new_starts;
                op_slice->params["ends"] = new_ends;
                op_slice->params["steps"] = new_steps;

                op_slice->inputs.push_back(op->inputs[0]);
                op->inputs[0]->consumers.push_back(op_slice);

                op_slice->outputs.push_back(slice_out);
                slice_out->producer = op_slice;

                op_view_as->inputs.push_back(op->inputs[1]);
                op->inputs[1]->consumers.push_back(op_view_as);
                op->inputs[1]->remove_consumer(op);
                op_view_as->inputs.push_back(slice_out);
                slice_out->consumers.push_back(op_view_as);

                op_view_as->outputs.push_back(view_as_out);
                view_as_out->producer = op_view_as;

                op->inputs[1] = view_as_out;
                view_as_out->consumers.push_back(op);
            }
            else if (input_rank != (int)op->inputs[1]->shape.size())
            {
                // solve the target shape
                std::vector<int> target_shape = op->inputs[0]->shape;
                for (size_t j = 0; j < new_dims.size(); j++)
                {
                    int dim = new_dims[j];
                    int start = new_starts[j];
                    int end = new_ends[j];
                    int step = new_steps[j];

                    if (dim < 0)
                        dim = input_rank + dim;
                    if (start < 0)
                        start = target_shape[dim] + start;
                    if (end < 0)
                        end = target_shape[dim] + end;
                    if (end == INT_MAX)
                        end = target_shape[dim];

                    target_shape[dim] = (end - start + (step - 1)) / step;
                }

                Operator* op_view = graph.new_operator_before("Tensor.view", op->name + "_ncnnview", op);
                Operand* view_out = graph.new_operand(op->name + "_ncnnview_out");

                op_view->params["shape"] = target_shape;

                view_out->type = op->inputs[1]->type;
                view_out->shape = target_shape;

                op_view->inputs.push_back(op->inputs[1]);
                op->inputs[1]->consumers.push_back(op_view);
                op->inputs[1]->remove_consumer(op);

                op_view->outputs.push_back(view_out);
                view_out->producer = op_view;

                op->inputs[1] = view_out;
                view_out->consumers.push_back(op);
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
