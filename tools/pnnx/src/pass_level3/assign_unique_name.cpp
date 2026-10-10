// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
                // fprintf(stderr, "assign unique operator name %s to %s\n", new_name.c_str(), name.c_str());
                op->name = new_name;
                names.insert(new_name);

                make_unique_index++;
            }
        }
    }
}

} // namespace pnnx
