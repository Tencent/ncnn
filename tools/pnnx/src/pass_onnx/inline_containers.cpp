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

#include "inline_containers.h"

#include <map>
#include <string>

namespace pnnx {

namespace onnx2pnnx {

static bool string_starts_with(const std::string& s, const std::string& s2)
{
    return strncmp(s.c_str(), s2.c_str(), s2.size()) == 0;
}

void inline_containers(onnx::ModelProto& model)
{
    onnx::GraphProto* graph = model.mutable_graph();

    for (int i = 0; i < graph->node_size(); i++)
    {
        onnx::NodeProto* node = graph->mutable_node(i);

        const std::string& op_type = node->op_type();

        if (node->domain().empty())
        {
            // native onnx op

            // Constant
            // fprintf(stderr, "   node = onnx %s\n", op_type.c_str());
            continue;
        }

        if (string_starts_with(op_type, "torch_nn_modules_") && !string_starts_with(op_type, "torch_nn_modules_container_"))
        {
            // torch_nn_modules_conv_Conv2d                 _conv1_1
            // torch_nn_modules_batchnorm_BatchNorm2d       _bn1_1
            // torch_nn_modules_pooling_MaxPool2d           _maxpool_1_3
            // torch_nn_modules_linear_Linear               _fc_1

            // std::vector<std::string> tokens = string_split(op_type, '_');

            // fprintf(stderr, "   node = nn.%s\n", tokens[4].c_str());
            continue;
        }

        if (string_starts_with(op_type, "aten_") || string_starts_with(op_type, "_aten_"))
        {
            // aten_view

            // std::vector<std::string> tokens = string_split(op_type, '_');

            // fprintf(stderr, "   node = aten::%s\n", tokens[1].c_str());
            continue;
        }

        if (string_starts_with(op_type, "prims_"))
        {
            // prims_convert_element_type
            continue;
        }

        // find function
        int function_index = -1;
        for (int j = 0; j < model.functions_size(); j++)
        {
            const onnx::FunctionProto& function = model.functions(j);
            if (function.name() == op_type)
            {
                function_index = j;
                break;
            }
        }

        if (function_index == -1)
        {
            fprintf(stderr, "no such function with name %s\n", op_type.c_str());
            continue;
        }

        // ok, this is a function, inline it at node
        // fprintf(stderr, "inline %s\n", op_type.c_str());

        const onnx::FunctionProto& function = model.functions(function_index);

        // build function input and output name remap
        std::map<std::string, std::string> input_output_remap;
        {
            for (int j = 0; j < node->input_size(); j++)
            {
                const std::string& node_input = node->input(j);
                const std::string& func_input = function.input(j);

                input_output_remap[func_input] = node_input;
            }
            for (int j = 0; j < node->output_size(); j++)
            {
                const std::string& node_output = node->output(j);
                const std::string& func_output = function.output(j);

                input_output_remap[func_output] = node_output;
            }
        }

        // append function nodes to graph
        {
            graph->mutable_node()->Reserve(graph->node_size() + function.node_size());
            for (int j = 0; j < function.node_size(); j++)
            {
                onnx::NodeProto* inlined_node = graph->add_node();
                inlined_node->CopyFrom(function.node(j));

                // prefix with caller node name
                inlined_node->set_name(node->name() + "/" + inlined_node->name());

                // reset input output
                for (int j = 0; j < inlined_node->input_size(); j++)
                {
                    const std::string& node_input = inlined_node->input(j);
                    if (input_output_remap.find(node_input) != input_output_remap.end())
                    {
                        inlined_node->set_input(j, input_output_remap.at(node_input));
                    }
                    else
                    {
                        // graph->add_value_info()->set_name(node->name() + "/" + node_input);
                        inlined_node->set_input(j, node->name() + "/" + node_input);
                    }
                }
                for (int j = 0; j < inlined_node->output_size(); j++)
                {
                    const std::string& node_output = inlined_node->output(j);
                    if (input_output_remap.find(node_output) != input_output_remap.end())
                    {
                        inlined_node->set_output(j, input_output_remap.at(node_output));
                    }
                    else
                    {
                        // graph->add_value_info()->set_name(node->name() + "/" + node_output);
                        inlined_node->set_output(j, node->name() + "/" + node_output);
                    }
                }
            }
        }

        // swap inlined function nodes to caller
        {
            //  ..... cni ....... 0 1 2 3 4
            const int graph_node_size = graph->node_size();
            for (int j = 0; j < function.node_size(); j++)
            {
                for (int k = graph_node_size - 1; k > i; k--)
                {
                    graph->mutable_node()->SwapElements(k, k - 1);
                }
            }

            //  ..... 0 1 2 3 4 cni .......
            for (int j = i + function.node_size(); j < graph_node_size - 1; j++)
            {
                graph->mutable_node()->SwapElements(j, j + 1);
            }

            //  ..... 0 1 2 3 4 ....... cni
            graph->mutable_node()->RemoveLast();
        }

        // inlined node may be function
        i -= 1;
    }
}

} // namespace onnx2pnnx

} // namespace pnnx
