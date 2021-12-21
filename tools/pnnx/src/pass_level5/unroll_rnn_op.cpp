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

#include "unroll_rnn_op.h"

#include <algorithm>

namespace pnnx {

void unroll_rnn_op(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.RNN" && op->type != "nn.LSTM" && op->type != "nn.GRU")
                continue;

            int num_layers = op->params["num_layers"].i;
            if (num_layers == 1)
                continue;

            matched = true;

            bool has_input_hidden = op->inputs.size() >= 2;
            bool has_input_cell = op->inputs.size() == 3;
            bool has_output_hidden = op->outputs.size() >= 2;
            bool has_output_cell = op->outputs.size() == 3;
            const int hidden_size = op->params["hidden_size"].i;
            bool has_bias = op->params["bias"].b;
            bool is_bidirectional = op->params["bidirectional"].b;

            std::vector<Operand*> input_hiddens(num_layers);
            std::vector<Operand*> input_cells(num_layers);
            std::vector<Operand*> output_hiddens(num_layers);
            std::vector<Operand*> output_cells(num_layers);

            // slice input hidden cell
            if (has_input_hidden)
            {
                std::string opname = op->name + "_chunk_in_hidden";

                Operator* op1 = graph.new_operator_before("torch.chunk", opname, op);

                op1->params["chunks"] = num_layers;
                op1->params["dim"] = 0;

                op1->inputs.push_back(op->inputs[1]);
                op->inputs[1]->remove_consumer(op);
                op->inputs[1]->consumers.push_back(op1);

                for (int j = 0; j < num_layers; j++)
                {
                    Operand* r0 = graph.new_operand(op1->name + "_in_hidden_" + std::to_string(j));
                    r0->producer = op1;
                    op1->outputs.push_back(r0);

                    input_hiddens[j] = r0;
                }
            }
            if (has_input_cell)
            {
                std::string opname = op->name + "_chunk_in_cell";

                Operator* op1 = graph.new_operator_before("torch.chunk", opname, op);

                op1->params["chunks"] = num_layers;
                op1->params["dim"] = 0;

                op1->inputs.push_back(op->inputs[2]);
                op->inputs[2]->remove_consumer(op);
                op->inputs[2]->consumers.push_back(op1);

                for (int j = 0; j < num_layers; j++)
                {
                    Operand* r0 = graph.new_operand(op1->name + "_in_cell_" + std::to_string(j));
                    r0->producer = op1;
                    op1->outputs.push_back(r0);

                    input_cells[j] = r0;
                }
            }

            // unroll
            std::vector<Operator*> unrolled_ops(num_layers);
            for (int j = 0; j < num_layers; j++)
            {
                std::string opname = op->name + "_unroll_" + std::to_string(j);

                Operator* op1 = graph.new_operator_before(op->type, opname, op);

                op1->params = op->params;
                op1->params["num_layers"] = 1;

                // link
                if (j == 0)
                {
                    op1->inputs.push_back(op->inputs[0]);
                    op1->inputs[0]->remove_consumer(op);
                    op1->inputs[0]->consumers.push_back(op1);
                }
                else
                {
                    op1->params["input_size"] = is_bidirectional ? hidden_size * 2 : hidden_size;

                    op1->inputs.push_back(unrolled_ops[j - 1]->outputs[0]);
                    op1->inputs[0]->consumers.push_back(op1);
                }

                if (has_input_hidden)
                {
                    op1->inputs.push_back(input_hiddens[j]);
                    op1->inputs[1]->consumers.push_back(op1);
                }
                if (has_input_cell)
                {
                    op1->inputs.push_back(input_cells[j]);
                    op1->inputs[2]->consumers.push_back(op1);
                }

                if (j == num_layers - 1)
                {
                    op1->outputs.push_back(op->outputs[0]);
                    op1->outputs[0]->producer = op1;
                }
                else
                {
                    Operand* r0 = graph.new_operand(op1->name + "_out");
                    r0->producer = op1;
                    op1->outputs.push_back(r0);
                }

                if (has_output_hidden)
                {
                    Operand* r1 = graph.new_operand(op1->name + "_out_hidden");
                    r1->producer = op1;
                    op1->outputs.push_back(r1);

                    output_hiddens[j] = r1;
                }
                if (has_output_cell)
                {
                    Operand* r1 = graph.new_operand(op1->name + "_out_cell");
                    r1->producer = op1;
                    op1->outputs.push_back(r1);

                    output_cells[j] = r1;
                }

                op1->attrs["weight_hh_l0"] = op->attrs["weight_hh_l" + std::to_string(j)];
                op1->attrs["weight_ih_l0"] = op->attrs["weight_ih_l" + std::to_string(j)];

                if (has_bias)
                {
                    op1->attrs["bias_hh_l0"] = op->attrs["bias_hh_l" + std::to_string(j)];
                    op1->attrs["bias_ih_l0"] = op->attrs["bias_ih_l" + std::to_string(j)];
                }

                if (is_bidirectional)
                {
                    op1->attrs["weight_hh_l0_reverse"] = op->attrs["weight_hh_l" + std::to_string(j) + "_reverse"];
                    op1->attrs["weight_ih_l0_reverse"] = op->attrs["weight_ih_l" + std::to_string(j) + "_reverse"];

                    if (has_bias)
                    {
                        op1->attrs["bias_hh_l0_reverse"] = op->attrs["bias_hh_l" + std::to_string(j) + "_reverse"];
                        op1->attrs["bias_ih_l0_reverse"] = op->attrs["bias_ih_l" + std::to_string(j) + "_reverse"];
                    }
                }

                unrolled_ops[j] = op1;
            }

            // concat output hidden cell
            if (has_output_hidden)
            {
                std::string opname = op->name + "_cat_out_hidden";

                Operator* op1 = graph.new_operator_before("torch.cat", opname, op);

                op1->params["dim"] = 0;

                for (int j = 0; j < num_layers; j++)
                {
                    Operand* r0 = output_hiddens[j];
                    r0->consumers.push_back(op1);
                    op1->inputs.push_back(r0);
                }

                op1->outputs.push_back(op->outputs[1]);
                op1->outputs[0]->producer = op1;
            }
            if (has_output_cell)
            {
                std::string opname = op->name + "_cat_out_cell";

                Operator* op1 = graph.new_operator_before("torch.cat", opname, op);

                op1->params["dim"] = 0;

                for (int j = 0; j < num_layers; j++)
                {
                    Operand* r0 = output_cells[j];
                    r0->consumers.push_back(op1);
                    op1->inputs.push_back(r0);
                }

                op1->outputs.push_back(op->outputs[2]);
                op1->outputs[0]->producer = op1;
            }

            op->inputs.clear();
            op->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));

            delete op;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
