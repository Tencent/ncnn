// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
