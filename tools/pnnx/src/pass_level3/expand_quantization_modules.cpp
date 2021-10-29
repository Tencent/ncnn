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

#include "expand_quantization_modules.h"
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void expand_quantization_modules(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.intrinsic.quantized.ConvReLU2d")
                continue;

            matched = true;

            // expand to nn.quantized.Conv2d + nn.ReLU + nn.quantized.Quantize
            op->type = "nn.quantized.Conv2d";

            // insert new operator before all output consumers
            const Operator* cur = 0;
            {
                int cur_index = graph.ops.size() - 1;
                for (auto& c : op->outputs[0]->consumers)
                {
                    int c_index = std::find(graph.ops.begin(), graph.ops.end(), c) - graph.ops.begin();
                    cur_index = std::min(cur_index, c_index);
                }

                cur = graph.ops[cur_index];
            }

            Operator* op_relu = graph.new_operator_before("nn.ReLU", op->name + "_relu", cur);
            Operator* op_quantize = graph.new_operator_before("nn.quantized.Quantize", op->name + "_output_quantize", cur);

            op_quantize->params["dtype"] = "torch.qint8";
            op_quantize->params["scale"] = op->params["scale"];
            op_quantize->params["zero_point"] = op->params["zero_point"];

            op->params.erase("scale");
            op->params.erase("zero_point");

            Operand* r0 = graph.new_operand(op->name + "_norelu");
            Operand* r1 = graph.new_operand(op->name + "_relu");

            r0->producer = op;
            r0->consumers.push_back(op_relu);

            r1->producer = op_relu;
            r1->consumers.push_back(op_quantize);

            op_relu->inputs.push_back(r0);
            op_relu->outputs.push_back(r1);
            op_relu->outputs[0]->producer = op_relu;

            op_quantize->inputs.push_back(r1);
            op_quantize->outputs.push_back(op->outputs[0]);
            op_quantize->outputs[0]->producer = op_quantize;

            op->outputs[0] = r0;

            break;
        }

        if (!matched)
            break;
    }

    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.quantized.Conv2d")
                continue;

            if (op->params.find("scale") == op->params.end() && op->params.find("zero_point") == op->params.end())
            {
                continue;
            }

            matched = true;

            // expand to nn.quantized.Conv2d + nn.quantized.Quantize
            op->type = "nn.quantized.Conv2d";

            // insert new operator before all output consumers
            const Operator* cur = 0;
            {
                int cur_index = graph.ops.size() - 1;
                for (auto& c : op->outputs[0]->consumers)
                {
                    int c_index = std::find(graph.ops.begin(), graph.ops.end(), c) - graph.ops.begin();
                    cur_index = std::min(cur_index, c_index);
                }

                cur = graph.ops[cur_index];
            }

            Operator* op_quantize = graph.new_operator_before("nn.quantized.Quantize", op->name + "_output_quantize", cur);

            op_quantize->params["dtype"] = "torch.qint8";
            op_quantize->params["scale"] = op->params["scale"];
            op_quantize->params["zero_point"] = op->params["zero_point"];

            op->params.erase("scale");
            op->params.erase("zero_point");

            Operand* r0 = graph.new_operand(op->name + "_fp");

            r0->producer = op;
            r0->consumers.push_back(op_quantize);

            op_quantize->inputs.push_back(r0);
            op_quantize->outputs.push_back(op->outputs[0]);
            op_quantize->outputs[0]->producer = op_quantize;

            op->outputs[0] = r0;

            break;
        }

        if (!matched)
            break;
    }

    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.quantized.Linear")
                continue;

            if (op->params.find("scale") == op->params.end() && op->params.find("zero_point") == op->params.end())
            {
                continue;
            }

            matched = true;

            // expand to nn.quantized.Linear + nn.quantized.Quantize
            op->type = "nn.quantized.Linear";

            // insert new operator before all output consumers
            const Operator* cur = 0;
            {
                int cur_index = graph.ops.size() - 1;
                for (auto& c : op->outputs[0]->consumers)
                {
                    int c_index = std::find(graph.ops.begin(), graph.ops.end(), c) - graph.ops.begin();
                    cur_index = std::min(cur_index, c_index);
                }

                cur = graph.ops[cur_index];
            }

            Operator* op_quantize = graph.new_operator_before("nn.quantized.Quantize", op->name + "_output_quantize", cur);

            op_quantize->params["dtype"] = "torch.qint8";
            op_quantize->params["scale"] = op->params["scale"];
            op_quantize->params["zero_point"] = op->params["zero_point"];

            op->params.erase("scale");
            op->params.erase("zero_point");

            Operand* r0 = graph.new_operand(op->name + "_fp");

            r0->producer = op;
            r0->consumers.push_back(op_quantize);

            op_quantize->inputs.push_back(r0);
            op_quantize->outputs.push_back(op->outputs[0]);
            op_quantize->outputs[0]->producer = op_quantize;

            op->outputs[0] = r0;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
