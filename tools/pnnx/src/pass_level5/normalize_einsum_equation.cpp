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

#include "normalize_einsum_equation.h"

#include <algorithm>
#include <map>
#include <vector>
#include "pass_level2.h"

namespace pnnx {

void normalize_einsum_equation(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "torch.einsum")
            continue;

        std::string equation = op->params.at("equation").s;
        const size_t equation_len = equation.size();

        std::map<char, char> xset;

        // find arrow ->
        size_t arrow = equation.find("->");
        if (arrow == std::string::npos)
        {
            // normalize to ijkl...
            for (size_t i = 0; i < equation_len; i++)
            {
                char x = equation[i];

                if ((x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z'))
                {
                    if (xset.find(x) == xset.end())
                        xset[x] = '\0';
                }
            }

            char x = 'i';
            for (auto& _xset : xset)
            {
                _xset.second = x++;
            }
        }
        else
        {
            // normalize and sort to ijkl...
            std::vector<char> xs;
            for (size_t i = 0; i < equation_len; i++)
            {
                char x = equation[i];

                if ((x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z'))
                {
                    if (xset.find(x) == xset.end())
                    {
                        xset[x] = '\0';
                        xs.push_back(x);
                    }
                }
            }

            for (auto& _xset : xset)
            {
                char x = _xset.first;
                size_t xsi = 0;
                for (size_t i = 0; i < xs.size(); i++)
                {
                    if (xs[i] == x)
                    {
                        xsi = i;
                        break;
                    }
                }

                _xset.second = 'i' + xsi;
            }
        }

        for (size_t i = 0; i < equation_len; i++)
        {
            char x = equation[i];

            if ((x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z'))
                equation[i] = xset[x];
        }

        op->params["equation"] = equation;
    }
}

} // namespace pnnx
