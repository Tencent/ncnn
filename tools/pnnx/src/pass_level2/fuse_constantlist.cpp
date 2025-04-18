// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_constantlist.h"

#include <algorithm>
#include <vector>

namespace pnnx {

void fuse_constantlist(Graph& graph)
{
    // from prim::Constant * N - prim::ListConstruct
    //   to prim::Constant (x0,x1,...)

    // from prim::Constant * 0 - prim::ListConstruct
    //   to prim::Constant None

    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];
            if (op->type != "prim::ListConstruct")
                continue;

            if (op->inputs.empty())
            {
                op->type = "prim::Constant";
                op->params["value"] = Parameter();
                continue;
            }

            bool is_constant_list = true;
            std::vector<int> constants_i;
            std::vector<float> constants_f;
            std::vector<std::string> constants_s;
            for (auto x : op->inputs)
            {
                const Operator* op_constant = x->producer;
                if (op_constant->type != "prim::Constant")
                {
                    is_constant_list = false;
                    break;
                }

                if (!op_constant->has_param("value"))
                {
                    is_constant_list = false;
                    break;
                }

                if (op_constant->params.at("value").type == 2)
                {
                    constants_i.push_back(op_constant->params.at("value").i);
                }
                else if (op_constant->params.at("value").type == 3)
                {
                    constants_f.push_back(op_constant->params.at("value").f);
                }
                else if (op_constant->params.at("value").type == 4)
                {
                    constants_s.push_back(op_constant->params.at("value").s);
                }
                else
                {
                    // other typed constant
                    is_constant_list = false;
                    break;
                }
            }

            int listtype = 0;
            if (constants_i.size() == op->inputs.size())
                listtype = 5;
            if (constants_f.size() == op->inputs.size())
                listtype = 6;
            if (constants_s.size() == op->inputs.size())
                listtype = 7;

            if (!is_constant_list || listtype == 0)
                continue;

            need_eliminate = true;

            op->type = "prim::Constant";

            if (listtype == 5)
                op->params["value"] = constants_i;
            if (listtype == 6)
                op->params["value"] = constants_f;
            if (listtype == 7)
                op->params["value"] = constants_s;

            for (auto x : op->inputs)
            {
                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), x->producer));
                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), x));
            }

            op->inputs.clear();
        }

        if (!need_eliminate)
            break;
    }
}

} // namespace pnnx
