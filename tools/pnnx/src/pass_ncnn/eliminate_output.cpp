// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_output.h"

namespace pnnx {

namespace ncnn {

void eliminate_output(Graph& graph, bool preserve_output_set)
{
    int output_index = 0;
    std::set<Operand*> outputs;

    for (;;)
    {
        bool need_eliminate = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Output")
                continue;

            need_eliminate = true;

            // canonicalize output name
            for (int j = 0; j < (int)op->inputs.size(); j++)
            {
                op->inputs[j]->name = std::string("out") + std::to_string(output_index);
                output_index++;
            }

            for (Operand* r : op->inputs)
            {
                outputs.insert(r);
                r->remove_consumer(op);
            }

            op->inputs.clear();

            for (Operand* r : op->outputs)
            {
                r->producer = 0;
            }

            op->outputs.clear();

            graph.ops.erase(graph.ops.begin() + i);
            delete op;

            break;
        }

        if (!need_eliminate)
            break;
    }

    if (preserve_output_set)
    {
        // ncnn otherwise exposes unused tops of multi-output layers as outputs
        // consume values discarded by the source graph, such as Caffe Silence
        int silence_index = 0;
        for (Operand* r : graph.operands)
        {
            if (!r->consumers.empty() || outputs.count(r))
                continue;
            Operator* op = graph.new_operator("Noop", "silence_" + std::to_string(silence_index++));
            op->inputs.push_back(r);
            r->consumers.push_back(op);
        }
    }
}

} // namespace ncnn

} // namespace pnnx
