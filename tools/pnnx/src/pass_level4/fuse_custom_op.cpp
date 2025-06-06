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

#include "fuse_custom_op.h"

#include <set>

namespace pnnx {

void fuse_custom_op(Graph& graph)
{
    std::set<std::string> custom_ops;

    for (;;)
    {
        bool need_fuse = false;

        // fuse in reverse order
        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type.find("::") == std::string::npos)
                continue;

            std::string op_type_namespace = op->type.substr(0, op->type.find_first_of(':'));

            if (op_type_namespace == "aten" || op_type_namespace == "prim")
                continue;

            custom_ops.insert(op->type);

            std::string op_type_name = op->type.substr(op->type.find_last_of(':') + 1);

            need_fuse = true;

            op->type = std::string("pnnx.custom_op.") + op_type_namespace + '.' + op_type_name;

            std::vector<Operand*> new_inputs;
            std::vector<std::string> new_inputnames;
            for (size_t j = 0; j < op->inputs.size(); j++)
            {
                Operator* arg = op->inputs[j]->producer;

                if (!arg->inputs.empty())
                {
                    new_inputs.push_back(op->inputs[j]);
                    new_inputnames.push_back(std::string("arg_") + std::to_string(j));
                    continue;
                }

                if (arg->type == "prim::Constant")
                {
                    op->params[std::string("arg_") + std::to_string(j)] = arg->params["value"];
                    op->inputs[j]->remove_consumer(op);
                }
                else if (arg->type == "pnnx.Expression")
                {
                    op->params[std::string("arg_") + std::to_string(j)] = Parameter::parse_from_string(arg->params["expr"].s);
                    op->inputs[j]->remove_consumer(op);
                }
                else
                {
                    new_inputs.push_back(op->inputs[j]);
                    new_inputnames.push_back(std::string("arg_") + std::to_string(j));
                }
            }

            op->inputs = new_inputs;
            op->inputnames = new_inputnames;
        }

        if (!need_fuse)
            break;
    }

    for (auto x : custom_ops)
    {
        fprintf(stderr, "custom op = %s\n", x.c_str());
    }
}

} // namespace pnnx
