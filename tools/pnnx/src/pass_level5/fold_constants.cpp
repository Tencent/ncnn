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

#include "fold_constants.h"
#include <unordered_set>

#include "pass_level4/dead_code_elimination.h"

namespace pnnx {

void fold_constants(Graph& graph, const std::map<std::string, Attribute>& foldable_constants)
{
    for (size_t i = 0; i < graph.operands.size(); i++)
    {
        Operand* operand = graph.operands[i];
        const std::string& name = operand->name;

        if (foldable_constants.find(name) == foldable_constants.end())
            continue;

        Operator* op = operand->producer;
        if (op->type == "pnnx.Attribute")
            continue;

        // replace producer with attribute
        Operator* op_new = graph.new_operator_before("pnnx.Attribute", std::string("pnnx_fold_") + name, op);

        op_new->attrs[std::string("pnnx_fold_") + name] = foldable_constants.at(name);
        op_new->outputs.push_back(operand);
        operand->producer = op_new;

        op->outputs.clear();
    }

    // dce
    dead_code_elimination(graph);
}

} // namespace pnnx
