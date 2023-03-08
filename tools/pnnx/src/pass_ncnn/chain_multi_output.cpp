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

#include "chain_multi_output.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void chain_multi_output(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Output")
                continue;

            // prim::TupleConstruct     pnnx_791                 2 1 a b out
            // pnnx.Expression          pnnx_expr_0              3 1 a b c out expr=[@0,@1,@2]
            // pnnx.Output              pnnx_output_0            1 0 out
            bool match_tuple_expr_output = false;
            for (int j = 0; j < (int)op->inputs.size(); j++)
            {
                Operand* r = op->inputs[j];

                if (r->consumers.size() != 1)
                    continue;

                Operator* op0 = r->producer;

                if (op0->type == "prim::TupleConstruct")
                {
                    match_tuple_expr_output = true;
                }
                else if (op0->type == "pnnx.Expression")
                {
                    const int op_expr_input_count = (int)op0->inputs.size();
                    const std::string& expr = op0->params.at("expr").s;

                    std::string pattern_expr = "[";
                    for (int k = 0; k < op_expr_input_count; k++)
                    {
                        pattern_expr += std::string("@") + std::to_string(k);

                        if (k != op_expr_input_count - 1)
                            pattern_expr += ",";
                    }
                    pattern_expr += "]";

                    if (expr == pattern_expr)
                    {
                        match_tuple_expr_output = true;
                    }
                }

                if (!match_tuple_expr_output)
                    continue;

                // chain op0 as output and delete op0
                std::vector<Operand*> new_inputs;
                for (int k = 0; k < j; k++)
                {
                    new_inputs.push_back(op->inputs[k]);
                }

                for (Operand* r : op0->inputs)
                {
                    r->remove_consumer(op0);
                    r->consumers.push_back(op);
                    new_inputs.push_back(r);
                }

                for (int k = j + 1; k < (int)op->inputs.size(); k++)
                {
                    new_inputs.push_back(op->inputs[k]);
                }

                op->inputs = new_inputs;

                Operand* op0_out = op0->outputs[0];
                op0_out->producer = 0;
                op0_out->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op0_out));
                delete op0_out;

                op0->inputs.clear();
                op0->outputs.clear();

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op0));
                delete op0;

                break;
            }

            if (match_tuple_expr_output)
                need_eliminate = true;

            break;
        }

        if (!need_eliminate)
            break;
    }

    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Output")
                continue;

            // prim::DictConstruct      pnnx_68                  4 1 key0 out0 key1 out1 out
            // pnnx.Output              pnnx_output_0            1 0 out

            bool match_dict_output = false;
            for (int j = 0; j < (int)op->inputs.size(); j++)
            {
                Operand* r = op->inputs[j];

                if (r->consumers.size() != 1)
                    continue;

                Operator* op0 = r->producer;

                if (op0->type == "prim::DictConstruct")
                {
                    match_dict_output = true;
                }

                if (!match_dict_output)
                    continue;

                // chain op0 odd ones as output and delete op0
                std::vector<Operand*> new_inputs;
                for (int k = 0; k < j; k++)
                {
                    new_inputs.push_back(op->inputs[k]);
                }

                for (int k = 0; k < (int)op0->inputs.size(); k++)
                {
                    Operand* r = op0->inputs[k];

                    if (k % 2 == 0)
                    {
                        // ignore key
                        r->remove_consumer(op0);
                    }
                    else
                    {
                        r->remove_consumer(op0);
                        r->consumers.push_back(op);
                        new_inputs.push_back(r);
                    }
                }

                for (int k = j + 1; k < (int)op->inputs.size(); k++)
                {
                    new_inputs.push_back(op->inputs[k]);
                }

                op->inputs = new_inputs;

                Operand* op0_out = op0->outputs[0];
                op0_out->producer = 0;
                op0_out->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op0_out));
                delete op0_out;

                op0->inputs.clear();
                op0->outputs.clear();

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op0));
                delete op0;

                break;
            }

            if (match_dict_output)
                need_eliminate = true;

            break;
        }

        if (!need_eliminate)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
