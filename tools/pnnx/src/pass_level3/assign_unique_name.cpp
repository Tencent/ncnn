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

#include "assign_unique_name.h"
#include <unordered_set>

namespace pnnx {

void assign_unique_name(Graph& graph)
{
    // assign unique name for all operators
    {
        std::unordered_set<std::string> names;
        int make_unique_index = 0;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];
            const std::string& name = op->name;

            if (names.find(name) == names.end())
            {
                names.insert(name);
            }
            else
            {
                // duplicated found
                std::string new_name = std::string("pnnx_unique_") + std::to_string(make_unique_index);
                fprintf(stderr, "assign unique operator name %s to %s\n", new_name.c_str(), name.c_str());
                op->name = new_name;
                names.insert(new_name);

                make_unique_index++;
            }
        }
    }
}

} // namespace pnnx
