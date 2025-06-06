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

#include "eliminate_noop_upsample.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_noop_upsample(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "F.upsample" && op->type != "F.upsample_bilinear" && op->type != "F.upsample_nearest" && op->type != "F.interpolate"
                    && op->type != "nn.Upsample" && op->type != "nn.UpsamplingBilinear2d" && op->type != "nn.UpsamplingNearest2d")
                continue;

            if (op->inputs.size() != 1)
                continue;

            if (op->params.find("scale_factor") != op->params.end())
            {
                matched = true;

                std::vector<float> scale_factor;
                if (op->params.at("scale_factor").type == 3)
                {
                    scale_factor.push_back(op->params.at("scale_factor").f);
                }
                else
                {
                    scale_factor = op->params.at("scale_factor").af;
                }

                if (scale_factor.empty())
                    matched = false;

                for (auto s : scale_factor)
                {
                    if (s != 1.f)
                    {
                        matched = false;
                        break;
                    }
                }
            }

            if (!op->inputs[0]->shape.empty() && op->inputs[0]->shape == op->outputs[0]->shape)
            {
                matched = true;

                // dynamic shape comparison always fail
                for (auto s : op->inputs[0]->shape)
                {
                    if (s == -1)
                    {
                        matched = false;
                        break;
                    }
                }
            }

            // delete noop-like upsample
            if (matched)
            {
                for (auto& x : op->inputs)
                {
                    x->remove_consumer(op);
                }

                Operand* upsample_out = op->outputs[0];

                for (auto& x : upsample_out->consumers)
                {
                    for (size_t j = 0; j < x->inputs.size(); j++)
                    {
                        if (x->inputs[j] == upsample_out)
                            x->inputs[j] = op->inputs[0];
                    }

                    op->inputs[0]->consumers.push_back(x);
                }

                op->inputs[0]->name = upsample_out->name;

                upsample_out->producer = 0;
                upsample_out->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), upsample_out));
                delete upsample_out;

                op->inputs.clear();
                op->outputs.clear();

                graph.ops.erase(graph.ops.begin() + i);
                delete op;

                break;
            }
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
