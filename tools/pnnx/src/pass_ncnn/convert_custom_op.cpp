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

#include "convert_custom_op.h"

namespace pnnx {

namespace ncnn {

void convert_custom_op(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        if (op->type.substr(0, 15) == "pnnx.custom_op.")
        {
            op->type = op->type.substr(15);

            // handle arg_N
            std::map<std::string, Parameter> new_params;
            for (const auto& it : op->params)
            {
                fprintf(stderr, "%s  %d\n", it.first.c_str(), it.second.type);
                if (it.first.substr(0, 4) == "arg_")
                {
                    new_params[it.first.substr(4)] = it.second;
                }
            }

            op->params = new_params;
        }
    }
}

} // namespace ncnn

} // namespace pnnx
