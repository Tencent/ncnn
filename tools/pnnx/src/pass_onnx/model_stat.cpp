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

#include "model_stat.h"

namespace pnnx {

namespace onnx2pnnx {

static bool string_starts_with(const std::string& s, const std::string& s2)
{
    return strncmp(s.c_str(), s2.c_str(), s2.size()) == 0;
}

ModelStat get_model_stat(const onnx::ModelProto& model)
{
    ModelStat stat;

    const onnx::GraphProto& graph = model.graph();

    stat.node_size = graph.node_size();
    for (int i = 0; i < model.functions_size(); i++)
    {
        stat.node_size += model.functions(i).node_size();
    }

    stat.initializer_size = graph.initializer_size();
    stat.functions_size = model.functions_size();

    for (int i = 0; i < graph.node_size(); i++)
    {
        const onnx::NodeProto& node = graph.node(i);

        std::string op_type = node.op_type();

        // drop |folded_N suffix
        if (op_type.size() > 8)
        {
            size_t folded_N_index = op_type.rfind("|folded_");
            if (folded_N_index != std::string::npos)
            {
                op_type = op_type.substr(0, folded_N_index);
            }
        }

        if (node.domain().empty())
        {
            // native onnx op
            stat.onnx_count += 1;

            if (stat.onnx_op_count.find(op_type) == stat.onnx_op_count.end())
            {
                stat.onnx_op_count[op_type] = 1;
            }
            else
            {
                stat.onnx_op_count[op_type] = stat.onnx_op_count[op_type] + 1;
            }
            continue;
        }

        if (string_starts_with(op_type, "aten_") || string_starts_with(op_type, "_aten_"))
        {
            // aten_view
            stat.aten_count += 1;

            std::string simname = op_type;
            if (simname[0] == '_')
                simname = simname.substr(1);
            simname[4] = '.';

            if (stat.aten_op_count.find(simname) == stat.aten_op_count.end())
            {
                stat.aten_op_count[simname] = 1;
            }
            else
            {
                stat.aten_op_count[simname] = stat.aten_op_count[simname] + 1;
            }
            continue;
        }

        if (string_starts_with(op_type, "prims_"))
        {
            // prims_convert_element_type
            stat.prims_count += 1;

            std::string simname = op_type;
            simname[5] = '.';

            if (stat.prims_op_count.find(simname) == stat.prims_op_count.end())
            {
                stat.prims_op_count[simname] = 1;
            }
            else
            {
                stat.prims_op_count[simname] = stat.prims_op_count[simname] + 1;
            }
            continue;
        }

        if (string_starts_with(op_type, "torch_nn_modules_") || string_starts_with(op_type, "nn_"))
        {
            // torch_nn_modules_conv_Conv2d                 _conv1_1
            stat.nn_module_count += 1;

            std::string simname;
            if (string_starts_with(op_type, "nn_"))
            {
                // nn_Conv2d_i -> nn.Conv2d
                simname = op_type;
                simname[2] = '.';
                if (simname.find_first_of('_') != std::string::npos)
                    simname = simname.substr(0, simname.find_first_of('_'));
            }
            else
            {
                // torch_nn_modules_conv_Conv2d_xyz -> nn.Conv2d
                char nn_type[256];
                sscanf(op_type.c_str() + sizeof("torch_nn_modules_") - 1, "%*[^_]_%255[^_]", nn_type);
                simname = std::string("nn.") + nn_type;
            }

            if (stat.nn_module_op_count.find(simname) == stat.nn_module_op_count.end())
            {
                stat.nn_module_op_count[simname] = 1;
            }
            else
            {
                stat.nn_module_op_count[simname] = stat.nn_module_op_count[simname] + 1;
            }
            continue;
        }

        // custom module op
        stat.custom_module_count += 1;
    }

    // collect called functions
    std::unordered_set<std::string> called_functions;
    {
        for (int i = 0; i < graph.node_size(); i++)
        {
            const onnx::NodeProto& node = graph.node(i);

            std::string op_type = node.op_type();

            // drop |folded_N suffix
            if (op_type.size() > 8)
            {
                size_t folded_N_index = op_type.rfind("|folded_");
                if (folded_N_index != std::string::npos)
                {
                    op_type = op_type.substr(0, folded_N_index);
                }
            }

            if (node.domain().empty())
            {
                // native onnx op
                continue;
            }

            if (string_starts_with(op_type, "aten_") || string_starts_with(op_type, "_aten_"))
            {
                // aten_view
                continue;
            }

            if (string_starts_with(op_type, "prims_"))
            {
                // prims_convert_element_type
                continue;
            }

            if ((string_starts_with(op_type, "torch_nn_modules_") && !string_starts_with(op_type, "torch_nn_modules_container_")) || string_starts_with(op_type, "nn_"))
            {
                // torch_nn_modules_conv_Conv2d                 _conv1_1
                continue;
            }

            called_functions.insert(op_type);
        }

        while (1)
        {
            bool new_called_function = false;

            for (int i = 0; i < model.functions_size(); i++)
            {
                const onnx::FunctionProto& function = model.functions(i);

                if (called_functions.find(function.name()) == called_functions.end())
                    continue;

                for (int j = 0; j < function.node_size(); j++)
                {
                    const onnx::NodeProto& node = function.node(j);

                    std::string op_type = node.op_type();

                    // drop |folded_N suffix
                    if (op_type.size() > 8)
                    {
                        size_t folded_N_index = op_type.rfind("|folded_");
                        if (folded_N_index != std::string::npos)
                        {
                            op_type = op_type.substr(0, folded_N_index);
                        }
                    }

                    if (node.domain().empty())
                    {
                        // native onnx op
                        continue;
                    }

                    if (string_starts_with(op_type, "aten_") || string_starts_with(op_type, "_aten_"))
                    {
                        // aten_view
                        continue;
                    }

                    if (string_starts_with(op_type, "prims_"))
                    {
                        // prims_convert_element_type
                        continue;
                    }

                    if ((string_starts_with(op_type, "torch_nn_modules_") && !string_starts_with(op_type, "torch_nn_modules_container_")) || string_starts_with(op_type, "nn_"))
                    {
                        // torch_nn_modules_conv_Conv2d                 _conv1_1
                        continue;
                    }

                    if (called_functions.find(op_type) == called_functions.end())
                    {
                        called_functions.insert(op_type);
                        new_called_function = true;
                    }
                }
            }

            if (!new_called_function)
                break;
        }
    }

    for (int i = 0; i < model.functions_size(); i++)
    {
        const onnx::FunctionProto& function = model.functions(i);

        if (called_functions.find(function.name()) == called_functions.end())
            continue;

        for (int j = 0; j < function.node_size(); j++)
        {
            const onnx::NodeProto& node = function.node(j);

            std::string op_type = node.op_type();

            // drop |folded_N suffix
            if (op_type.size() > 8)
            {
                size_t folded_N_index = op_type.rfind("|folded_");
                if (folded_N_index != std::string::npos)
                {
                    op_type = op_type.substr(0, folded_N_index);
                }
            }

            if (node.domain().empty())
            {
                // native onnx op
                stat.onnx_count += 1;

                if (stat.onnx_op_count.find(op_type) == stat.onnx_op_count.end())
                {
                    stat.onnx_op_count[op_type] = 1;
                }
                else
                {
                    stat.onnx_op_count[op_type] = stat.onnx_op_count[op_type] + 1;
                }
                continue;
            }

            if (string_starts_with(op_type, "aten_") || string_starts_with(op_type, "_aten_"))
            {
                // aten_view
                stat.aten_count += 1;

                std::string simname = op_type;
                if (simname[0] == '_')
                    simname = simname.substr(1);
                simname[4] = '.';

                if (stat.aten_op_count.find(simname) == stat.aten_op_count.end())
                {
                    stat.aten_op_count[simname] = 1;
                }
                else
                {
                    stat.aten_op_count[simname] = stat.aten_op_count[simname] + 1;
                }
                continue;
            }

            if (string_starts_with(op_type, "prims_"))
            {
                // prims_convert_element_type
                stat.prims_count += 1;

                std::string simname = op_type;
                simname[5] = '.';

                if (stat.prims_op_count.find(simname) == stat.prims_op_count.end())
                {
                    stat.prims_op_count[simname] = 1;
                }
                else
                {
                    stat.prims_op_count[simname] = stat.prims_op_count[simname] + 1;
                }
                continue;
            }

            if (string_starts_with(op_type, "torch_nn_modules_") || string_starts_with(op_type, "nn_"))
            {
                // torch_nn_modules_conv_Conv2d                 _conv1_1
                stat.nn_module_count += 1;

                std::string simname;
                if (string_starts_with(op_type, "nn_"))
                {
                    simname = op_type;
                    simname[2] = '.';
                    if (simname.find_first_of('_') != std::string::npos)
                        simname = simname.substr(0, simname.find_first_of('_'));
                }
                else
                {
                    // torch_nn_modules_conv_Conv2d_xyz -> nn_Conv2d_i
                    char nn_type[256];
                    sscanf(op_type.c_str() + sizeof("torch_nn_modules_") - 1, "%*[^_]_%255[^_]", nn_type);
                    simname = std::string("nn.") + nn_type;
                }

                if (stat.nn_module_op_count.find(simname) == stat.nn_module_op_count.end())
                {
                    stat.nn_module_op_count[simname] = 1;
                }
                else
                {
                    stat.nn_module_op_count[simname] = stat.nn_module_op_count[simname] + 1;
                }
                continue;
            }

            // custom module op
            stat.custom_module_count += 1;
        }
    }

    return stat;
}

void print_model_stat(const ModelStat& oldstat, const ModelStat& newstat)
{
    std::set<std::string> nn_module_op_count;
    std::set<std::string> aten_op_count;
    std::set<std::string> prims_op_count;
    std::set<std::string> onnx_op_count;
    {
        for (auto& x : oldstat.nn_module_op_count)
        {
            nn_module_op_count.insert(x.first);
        }
        for (auto& x : newstat.nn_module_op_count)
        {
            nn_module_op_count.insert(x.first);
        }

        for (auto& x : oldstat.aten_op_count)
        {
            aten_op_count.insert(x.first);
        }
        for (auto& x : newstat.aten_op_count)
        {
            aten_op_count.insert(x.first);
        }

        for (auto& x : oldstat.prims_op_count)
        {
            prims_op_count.insert(x.first);
        }
        for (auto& x : newstat.prims_op_count)
        {
            prims_op_count.insert(x.first);
        }

        for (auto& x : oldstat.onnx_op_count)
        {
            onnx_op_count.insert(x.first);
        }
        for (auto& x : newstat.onnx_op_count)
        {
            onnx_op_count.insert(x.first);
        }
    }

    // resolve longest text
    int max_op_name_length = 16;
    for (auto& x : nn_module_op_count)
    {
        max_op_name_length = std::max(max_op_name_length, (int)x.size());
    }
    for (auto& x : aten_op_count)
    {
        max_op_name_length = std::max(max_op_name_length, (int)x.size());
    }
    for (auto& x : prims_op_count)
    {
        max_op_name_length = std::max(max_op_name_length, (int)x.size());
    }
    for (auto& x : onnx_op_count)
    {
        max_op_name_length = std::max(max_op_name_length, (int)x.size());
    }

    fprintf(stderr, "┌─");
    for (int i = 0; i < max_op_name_length; i++)
        fprintf(stderr, "─");
    fprintf(stderr, "─┬──────────┬──────────┐\n");

    fprintf(stderr, "│ %-*s │ orig     │ opt      │\n", max_op_name_length, "");

    fprintf(stderr, "├─");
    for (int i = 0; i < max_op_name_length; i++)
        fprintf(stderr, "─");
    fprintf(stderr, "─┼──────────┼──────────┤\n");

    if (newstat.node_size < oldstat.node_size)
        fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, "node", oldstat.node_size, newstat.node_size);
    else
        fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, "node", oldstat.node_size, newstat.node_size);

    fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, "initializer", oldstat.initializer_size, newstat.initializer_size);

    if (newstat.functions_size < oldstat.functions_size)
        fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, "functions", oldstat.functions_size, newstat.functions_size);
    else
        fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, "functions", oldstat.functions_size, newstat.functions_size);

    fprintf(stderr, "├─");
    for (int i = 0; i < max_op_name_length; i++)
        fprintf(stderr, "─");
    fprintf(stderr, "─┼──────────┼──────────┤\n");

    if (newstat.nn_module_count < oldstat.nn_module_count)
        fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, "nn module op", oldstat.nn_module_count, newstat.nn_module_count);
    else
        fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, "nn module op", oldstat.nn_module_count, newstat.nn_module_count);

    if (newstat.custom_module_count < oldstat.custom_module_count)
        fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, "custom module op", oldstat.custom_module_count, newstat.custom_module_count);
    else
        fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, "custom module op", oldstat.custom_module_count, newstat.custom_module_count);

    if (newstat.aten_count < oldstat.aten_count)
        fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, "aten op", oldstat.aten_count, newstat.aten_count);
    else
        fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, "aten op", oldstat.aten_count, newstat.aten_count);

    if (newstat.prims_count < oldstat.prims_count)
        fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, "prims op", oldstat.prims_count, newstat.prims_count);
    else
        fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, "prims op", oldstat.prims_count, newstat.prims_count);

    if (newstat.onnx_count < oldstat.onnx_count)
        fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, "onnx native op", oldstat.onnx_count, newstat.onnx_count);
    else
        fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, "onnx native op", oldstat.onnx_count, newstat.onnx_count);

    fprintf(stderr, "├─");
    for (int i = 0; i < max_op_name_length; i++)
        fprintf(stderr, "─");
    fprintf(stderr, "─┼──────────┼──────────┤\n");

    // merge nn_module_op_count
    {
        for (auto x : nn_module_op_count)
        {
            int oldcount = 0;
            int newcount = 0;
            if (oldstat.nn_module_op_count.find(x) != oldstat.nn_module_op_count.end())
            {
                oldcount = oldstat.nn_module_op_count.at(x);
            }
            if (newstat.nn_module_op_count.find(x) != newstat.nn_module_op_count.end())
            {
                newcount = newstat.nn_module_op_count.at(x);
            }

            if (newcount < oldcount)
                fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, x.c_str(), oldcount, newcount);
            else
                fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, x.c_str(), oldcount, newcount);
        }

        if (!nn_module_op_count.empty())
        {
            fprintf(stderr, "├─");
            for (int i = 0; i < max_op_name_length; i++)
                fprintf(stderr, "─");
            fprintf(stderr, "─┼──────────┼──────────┤\n");
        }
    }

    // merge aten_op_count
    {
        for (auto x : aten_op_count)
        {
            int oldcount = 0;
            int newcount = 0;
            if (oldstat.aten_op_count.find(x) != oldstat.aten_op_count.end())
            {
                oldcount = oldstat.aten_op_count.at(x);
            }
            if (newstat.aten_op_count.find(x) != newstat.aten_op_count.end())
            {
                newcount = newstat.aten_op_count.at(x);
            }

            if (newcount < oldcount)
                fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, x.c_str(), oldcount, newcount);
            else
                fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, x.c_str(), oldcount, newcount);
        }

        if (!aten_op_count.empty())
        {
            fprintf(stderr, "├─");
            for (int i = 0; i < max_op_name_length; i++)
                fprintf(stderr, "─");
            fprintf(stderr, "─┼──────────┼──────────┤\n");
        }
    }

    // merge prims_op_count
    {
        for (auto x : prims_op_count)
        {
            int oldcount = 0;
            int newcount = 0;
            if (oldstat.prims_op_count.find(x) != oldstat.prims_op_count.end())
            {
                oldcount = oldstat.prims_op_count.at(x);
            }
            if (newstat.prims_op_count.find(x) != newstat.prims_op_count.end())
            {
                newcount = newstat.prims_op_count.at(x);
            }

            if (newcount < oldcount)
                fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, x.c_str(), oldcount, newcount);
            else
                fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, x.c_str(), oldcount, newcount);
        }

        if (!prims_op_count.empty())
        {
            fprintf(stderr, "├─");
            for (int i = 0; i < max_op_name_length; i++)
                fprintf(stderr, "─");
            fprintf(stderr, "─┼──────────┼──────────┤\n");
        }
    }

    // merge onnx_op_count
    {
        for (auto x : onnx_op_count)
        {
            int oldcount = 0;
            int newcount = 0;
            if (oldstat.onnx_op_count.find(x) != oldstat.onnx_op_count.end())
            {
                oldcount = oldstat.onnx_op_count.at(x);
            }
            if (newstat.onnx_op_count.find(x) != newstat.onnx_op_count.end())
            {
                newcount = newstat.onnx_op_count.at(x);
            }

            if (newcount < oldcount)
                fprintf(stderr, "│ %-*s │ %-8d │ \033[32m%-8d\033[0m │\n", max_op_name_length, x.c_str(), oldcount, newcount);
            else
                fprintf(stderr, "│ %-*s │ %-8d │ %-8d │\n", max_op_name_length, x.c_str(), oldcount, newcount);
        }
    }

    fprintf(stderr, "└─");
    for (int i = 0; i < max_op_name_length; i++)
        fprintf(stderr, "─");
    fprintf(stderr, "─┴──────────┴──────────┘\n");
}

} // namespace onnx2pnnx

} // namespace pnnx
