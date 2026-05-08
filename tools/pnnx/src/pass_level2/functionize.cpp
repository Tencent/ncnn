// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "functionize.h"

#include <algorithm>
#include <vector>

namespace pnnx {

static bool is_alias_op(const Operator* op)
{
    if (op->type == "aten::slice" || op->type == "aten::select")
        return true;

    if (op->type == "aten::view")
        return true;

    return false;
}

void functionize(Graph& graph)
{
    // graph.save("0.param", "0.bin");

    // 1. create shadow view/slice/select/... for each consumer
    // 2. replace inplace op, append copy
    // 3. tag operand alias for view/slice/select/... output
    // 4. scan inplace op, collect affacted alias
    //     5. look for any op after the inplace op with alias input
    //     6. collect ops on the chain back to alias
    //     7. move chain after copy op
    //     8. update all alias uses after copy op, retag alias
    // 9. clear all alias tag

    // 1. create shadow view/slice/select/... for each consumer
    {
        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (!is_alias_op(op))
                continue;

            Operand* out0 = op->outputs[0];

            if (out0->consumers.size() == 1)
                continue;

            bool all_consumers_are_same = true;
            for (size_t j = 1; j < out0->consumers.size(); j++)
            {
                if (out0->consumers[j] != out0->consumers[0])
                {
                    all_consumers_are_same = false;
                    break;
                }
            }
            if (all_consumers_are_same)
                continue;

            for (int j = (int)out0->consumers.size() - 1; j > 0; j--)
            {
                Operator* op1 = out0->consumers[j];

                Operator* op_shadow = graph.new_operator_after(op->type, op->name + "_pnnxshadow_" + std::to_string(j), op);

                Operand* shadow_out = graph.new_operand(op_shadow->name + "_out");

                op_shadow->inputs = op->inputs;
                op_shadow->params = op->params;
                op_shadow->outputs.push_back(shadow_out);

                for (Operand* x : op->inputs)
                {
                    x->consumers.push_back(op_shadow);
                }

                shadow_out->producer = op_shadow;
                shadow_out->type = out0->type;
                shadow_out->shape = out0->shape;
                shadow_out->params = out0->params;

                shadow_out->consumers.push_back(op1);

                for (size_t k = 0; k < op1->inputs.size(); k++)
                {
                    if (op1->inputs[k] == out0)
                        op1->inputs[k] = shadow_out;
                }
            }

            out0->consumers.resize(1);
        }
    }

    // graph.save("1.param", "1.bin");

    // 2. replace inplace op, append copy
    // 3. tag operand alias for view/slice/select/... output
    {
        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            bool is_inplace_op = op->type.size() > 2 && op->type[op->type.size() - 2] != '_' && op->type[op->type.size() - 1] == '_';

            if (op->type != "aten::copy_" && !is_alias_op(op) && !is_inplace_op)
                continue;

            Operand* in = op->inputs[0];

            int alias_index;
            if (in->params.find("__alias__") != in->params.end())
            {
                alias_index = in->params.at("__alias__").i;
            }
            else
            {
                alias_index = std::find(graph.operands.begin(), graph.operands.end(), in) - graph.operands.begin();
            }

            if (op->type == "aten::copy_")
            {
                op->outputs[0]->params["__alias__"] = alias_index;
                // fprintf(stderr, "operand %s is alias of %s\n", op->outputs[0]->name.c_str(), graph.operands[alias_index]->name.c_str());

                // set copy output shape as the alias one
                op->outputs[0]->type = graph.operands[alias_index]->type;
                op->outputs[0]->shape = graph.operands[alias_index]->shape;

                continue;
            }

            if (is_alias_op(op))
            {
                op->outputs[0]->params["__alias__"] = alias_index;
                // fprintf(stderr, "operand %s is alias of %s\n", op->outputs[0]->name.c_str(), graph.operands[alias_index]->name.c_str());
                continue;
            }

            if (is_inplace_op)
            {
                // replace with non-inplace version, create copy op
                op->type = op->type.substr(0, op->type.size() - 1);

                // append aten::copy_
                if (graph.operands[alias_index]->consumers.size() > 1)
                {
                    Operand* in0 = op->inputs[0];
                    Operand* out0 = op->outputs[0];

                    Operator* op_copy = graph.new_operator_after("aten::copy_", op->name + "_copy", op);
                    Operand* copy_out = graph.new_operand(op->name + "_copy_out");

                    op_copy->inputs.push_back(in0);
                    op_copy->inputs.push_back(out0);
                    in0->consumers.push_back(op_copy);
                    out0->consumers.push_back(op_copy);

                    op_copy->outputs.push_back(copy_out);
                    copy_out->producer = op_copy;
                }
            }
        }
    }

    // graph.save("3.param", "3.bin");

    // 4. scan inplace copy op, collect affacted alias
    {
        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "aten::copy_")
                continue;

            op->type = "aten::copy";

            Operand* out0 = op->outputs[0];

            // inplace op output always alias with the input
            const int alias_index = out0->params.at("__alias__").i;
            Operand* alias_in0 = graph.operands[alias_index];

            // fprintf(stderr, "\n---> %s  for %s\n", op->name.c_str(), alias_in0->name.c_str());

            size_t i_advanced = 0;

            // 5. look for any op after the inplace op with alias input
            for (size_t j = i + 1; j < graph.ops.size(); j++)
            {
                Operator* op1 = graph.ops[j];

                bool affacted = false;
                for (Operand* x : op1->inputs)
                {
                    if (x == alias_in0)
                    {
                        affacted = true;
                        break;
                    }

                    if (x->params.find("__alias__") == x->params.end())
                        continue;

                    int alias_index_1 = x->params.at("__alias__").i;
                    if (alias_index_1 == alias_index)
                    {
                        affacted = true;
                        break;
                    }
                }

                if (!affacted)
                    continue;

                // 6. collect ops on the chain back to alias
                std::set<size_t> chainsx_op_indexes;
                {
                    size_t op1_index = std::find(graph.ops.begin(), graph.ops.end(), op1) - graph.ops.begin();

                    if (op1_index < i - i_advanced)
                    {
                        chainsx_op_indexes.insert(op1_index);
                        // fprintf(stderr, "affacted op %s for %s\n", op1->name.c_str(), graph.operands[alias_index]->name.c_str());
                    }

                    while (1)
                    {
                        Operand* x = op1->inputs[0];
                        if (x->params.find("__alias__") == x->params.end())
                            break;

                        int alias_index_1 = x->params.at("__alias__").i;
                        if (alias_index_1 != alias_index)
                            break;

                        op1 = x->producer;
                        size_t op1_index = std::find(graph.ops.begin(), graph.ops.end(), op1) - graph.ops.begin();

                        if (op1_index < i - i_advanced)
                        {
                            chainsx_op_indexes.insert(op1_index);
                            // fprintf(stderr, "affacted op %s for %s   chained\n", op1->name.c_str(), graph.operands[alias_index]->name.c_str());
                        }
                    }
                }

                // 7. move chain after copy op
                {
                    int k = 0;
                    for (size_t doi : chainsx_op_indexes)
                    {
                        doi -= k;
                        // fprintf(stderr, "---> move %s after %s\n", graph.ops[doi]->name.c_str(), graph.ops[i - i_advanced]->name.c_str());

                        for (size_t l = doi; l < i - i_advanced; l++)
                        {
                            std::swap(graph.ops[l], graph.ops[l + 1]);
                        }

                        k += 1;
                    }

                    i_advanced += chainsx_op_indexes.size();
                }

                // 8. update all alias uses after copy op, retag alias
                out0->params.erase("__alias__");
                const int new_alias_index = std::find(graph.operands.begin(), graph.operands.end(), out0) - graph.operands.begin();
                for (size_t k = i - i_advanced + 1; k < graph.ops.size(); k++)
                {
                    Operator* op2 = graph.ops[k];

                    // bool use_in0 = false;
                    for (size_t l = 0; l < op2->inputs.size(); l++)
                    {
                        if (op2->inputs[l] == alias_in0)
                        {
                            // fprintf(stderr, "---> replace %s input %s to %s\n", op2->name.c_str(), op2->inputs[l]->name.c_str(), out0->name.c_str());

                            op2->inputs[l] = out0;
                            alias_in0->remove_consumer(op2);
                            out0->consumers.push_back(op2);
                        }
                    }

                    for (Operand* x : op2->outputs)
                    {
                        if (x->params.find("__alias__") != x->params.end() && x->params.at("__alias__").i == alias_index)
                        {
                            x->params["__alias__"] = new_alias_index;
                        }
                    }
                }

                // rewind to the updated copy operator
                j -= chainsx_op_indexes.size();
            }
        }
    }

    // graph.save("4.param", "4.bin");

    // 9. clear all alias tag
    {
        for (Operand* x : graph.operands)
        {
            x->params.erase("__alias__");
        }
    }
}

} // namespace pnnx
