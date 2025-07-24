// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
