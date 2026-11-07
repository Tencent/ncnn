// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_slice_copy.h"

#include <algorithm>
#include <vector>

namespace pnnx {

void fuse_slice_copy(Graph& graph)
{
    const std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
    {
        if (op->type != "Tensor.copy")
            continue;

        // Reverse each view in order. Choosing only the outermost slice loses
        // nested/overlapping slices and slices of the same dimension. A select
        // also changes the rank seen by the next writeback.
        std::vector<Operator*> views;
        Operand* base = op->inputs[0];
        while (base->producer->type == "Tensor.slice" || base->producer->type == "Tensor.select")
        {
            views.push_back(base->producer);
            base = base->producer->inputs[0];
        }

        Operand* value = op->inputs[1];
        for (size_t i = 0; i < views.size(); i++)
        {
            Operator* view = views[i];
            Operand* parent = view->inputs[0];
            const std::string name = op->name + "_writeback_" + std::to_string(i);
            Operator* clone = graph.new_operator_before("Tensor.clone", name + "_clone", op);
            Operand* cloned = graph.new_operand(clone->name + "_out");
            cloned->type = parent->type;
            cloned->shape = parent->shape;
            cloned->params = parent->params;
            cloned->producer = clone;
            clone->inputs.push_back(parent);
            parent->consumers.push_back(clone);
            clone->outputs.push_back(cloned);

            Operator* copy = graph.new_operator_before("Tensor.slice_copy", name, op);
            copy->inputs = {cloned, value};
            copy->inputnames = {"self", "src"};
            cloned->consumers.push_back(copy);
            value->consumers.push_back(copy);
            copy->params = view->params;
            if (view->type == "Tensor.select")
            {
                copy->params["start"] = 0;
                copy->params["end"] = 0;
                copy->params["step"] = 0;
                if (view->has_param("index"))
                {
                    copy->params["select"] = view->params.at("index");
                    copy->params.erase("index");
                }
            }
            for (size_t j = 1; j < view->inputs.size(); j++)
            {
                Operand* input = view->inputs[j];
                copy->inputs.push_back(input);
                std::string inputname = view->inputnames[j];
                if (inputname == "index")
                    inputname = "select";
                copy->inputnames.push_back(inputname);
                input->consumers.push_back(copy);
            }
            Operand* out = graph.new_operand(name + "_out");
            out->type = parent->type;
            out->shape = parent->shape;
            out->params = parent->params;
            out->producer = copy;
            copy->outputs.push_back(out);
            value = out;
        }

        Operand* out = op->outputs[0];
        for (Operator* consumer : out->consumers)
        {
            for (Operand*& input : consumer->inputs)
                if (input == out)
                    input = value;
            value->consumers.push_back(consumer);
        }
        for (Operand* input : op->inputs)
            input->remove_consumer(op);
        value->name = out->name;
        graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), out));
        delete out;
        graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
        delete op;
    }
}

} // namespace pnnx
