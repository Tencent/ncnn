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

#include "storezip.h"
#include "pass_level4/dead_code_elimination.h"

namespace pnnx {

void fold_constants(Graph& graph, const std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath)
{
    if (foldable_constants.empty())
        return;

    StoreZipReader zip;
    zip.open(foldable_constants_zippath);

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

        op_new->attrs["data"] = Attribute();

        Attribute& t2 = op_new->attrs["data"];
        t2.type = operand->type;
        t2.shape = operand->shape;
        size_t size = zip.get_file_size(name);
        t2.data.resize(size);
        zip.read_file(name, t2.data.data());

        op_new->outputs.push_back(operand);
        operand->producer = op_new;

        op->outputs.clear();
    }

    zip.close();

    // dce
    dead_code_elimination(graph);
}

} // namespace pnnx
