// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dead_code_elimination.h"

#include <string>
#include <unordered_set>

namespace pnnx {

namespace onnx2pnnx {

static void collect_dead_nodes(const onnx::GraphProto& graph, std::vector<std::string>& dead_outputs, std::vector<int>& dead_node_indexes, std::unordered_set<std::string>& live_inputs)
{
    for (int i = 0; i < graph.output_size(); i++)
    {
        live_inputs.insert(graph.output(i).name());
    }

    for (int i = graph.node_size() - 1; i >= 0; i--)
    {
        const onnx::NodeProto& node = graph.node(i);

        bool is_outputs_live = false;
        for (int j = 0; j < node.output_size(); j++)
        {
            if (live_inputs.find(node.output(j)) != live_inputs.end())
            {
                is_outputs_live = true;
                break;
            }
        }

        if (is_outputs_live)
        {
            bool is_outputs_optional = true;

            // some operator output be discarded even if not used
            const std::string& op_type = node.op_type();
            if (op_type == "Split") is_outputs_optional = false;

            if (is_outputs_optional)
            {
                for (int j = node.output_size() - 1; j >= 0; j--)
                {
                    if (live_inputs.find(node.output(j)) == live_inputs.end())
                    {
                        dead_outputs.push_back(node.output(j));
                    }
                    else
                    {
                        // leading outputs cannot be optional
                        break;
                    }
                }
            }

            for (int j = 0; j < node.input_size(); j++)
            {
                live_inputs.insert(node.input(j));
            }
        }
        else
        {
            dead_node_indexes.push_back(i);

            for (int j = node.output_size() - 1; j >= 0; j--)
            {
                dead_outputs.push_back(node.output(j));
            }
        }

        if (is_outputs_live)
        {
            for (int j = 0; j < node.attribute_size(); j++)
            {
                const onnx::AttributeProto& attr = node.attribute(j);

                if (attr.type() == onnx::AttributeProto::GRAPH)
                {
                    const onnx::GraphProto& sg = attr.g();

                    std::vector<std::string> sg_dead_outputs;
                    std::vector<int> sg_dead_node_indexes;
                    collect_dead_nodes(sg, sg_dead_outputs, sg_dead_node_indexes, live_inputs);
                }
                if (attr.type() == onnx::AttributeProto::GRAPHS)
                {
                    for (int k = 0; k < attr.graphs().size(); k++)
                    {
                        const onnx::GraphProto& sg = attr.graphs().at(k);

                        std::vector<std::string> sg_dead_outputs;
                        std::vector<int> sg_dead_node_indexes;
                        collect_dead_nodes(sg, sg_dead_outputs, sg_dead_node_indexes, live_inputs);
                    }
                }
            }
        }
    }
}

void dead_code_elimination(onnx::ModelProto& model)
{
    // collect all nodes that have no links with graph outputs
    std::vector<std::string> dead_outputs;
    std::vector<int> dead_node_indexes;
    {
        const onnx::GraphProto& graph = model.graph();

        std::unordered_set<std::string> live_inputs;
        collect_dead_nodes(graph, dead_outputs, dead_node_indexes, live_inputs);
    }

    // eliminate dead nodes
    {
        onnx::GraphProto* graph = model.mutable_graph();

        for (size_t i = 0; i < dead_node_indexes.size(); i++)
        {
            const int dead_node_index = dead_node_indexes[i];

            //  ..... dni .......
            const int graph_node_size = graph->node_size();
            for (int j = dead_node_index; j < graph_node_size - 1; j++)
            {
                graph->mutable_node()->SwapElements(j, j + 1);
            }

            //  ..... ....... dni
            graph->mutable_node()->RemoveLast();
        }
    }

    // eliminate dead value info
    {
        onnx::GraphProto* graph = model.mutable_graph();

        for (size_t i = 0; i < dead_outputs.size(); i++)
        {
            const std::string& dead_output = dead_outputs[i];

            for (int j = 0; j < graph->value_info_size(); j++)
            {
                if (graph->value_info(j).name() == dead_output)
                {
                    //  ..... j .......
                    const int graph_value_info_size = graph->value_info_size();
                    for (int k = j; k < graph_value_info_size - 1; k++)
                    {
                        graph->mutable_value_info()->SwapElements(k, k + 1);
                    }

                    //  ..... ....... j
                    graph->mutable_value_info()->RemoveLast();

                    break;
                }
            }
        }

        for (int i = 0; i < graph->node_size(); i++)
        {
            onnx::NodeProto* node = graph->mutable_node(i);

            for (int j = 0; j < node->output_size(); j++)
            {
                bool has_dead_output = false;
                for (auto x : dead_outputs)
                {
                    if (x == node->output(j))
                    {
                        has_dead_output = true;
                        break;
                    }
                }

                if (has_dead_output)
                {
                    // drop optional unused outputs
                    const int node_output_size = node->output_size();
                    for (int k = j; k < node_output_size; k++)
                    {
                        node->mutable_output()->RemoveLast();
                    }

                    break;
                }
            }
        }
    }

    // collect all dead functions
    std::vector<int> dead_function_indexes;
    {
        const onnx::GraphProto& graph = model.graph();

        std::unordered_set<int> live_function_indexes;
        for (int i = 0; i < graph.node_size(); i++)
        {
            const std::string& op_type = graph.node(i).op_type();

            for (int j = 0; j < model.functions_size(); j++)
            {
                const onnx::FunctionProto& function = model.functions(j);

                if (function.name() == op_type)
                {
                    live_function_indexes.insert(j);
                    break;
                }
            }
        }

        // find nested live functions
        while (1)
        {
            bool new_nested_live_function = false;

            for (int i = 0; i < model.functions_size(); i++)
            {
                if (live_function_indexes.find(i) == live_function_indexes.end())
                    continue;

                const onnx::FunctionProto& function = model.functions(i);

                for (int j = 0; j < function.node_size(); j++)
                {
                    const std::string& op_type = function.node(j).op_type();

                    for (int k = 0; k < model.functions_size(); k++)
                    {
                        const onnx::FunctionProto& nested_function = model.functions(k);

                        if (nested_function.name() == op_type && live_function_indexes.find(k) == live_function_indexes.end())
                        {
                            // nested live function added
                            live_function_indexes.insert(k);

                            new_nested_live_function = true;
                        }
                    }
                }
            }

            if (!new_nested_live_function)
                break;
        }

        for (int i = model.functions_size() - 1; i >= 0; i--)
        {
            if (live_function_indexes.find(i) == live_function_indexes.end())
            {
                dead_function_indexes.push_back(i);
            }
        }
    }

    // eliminate dead funtions
    {
        for (size_t i = 0; i < dead_function_indexes.size(); i++)
        {
            const int dead_function_index = dead_function_indexes[i];

            //  ..... dfi .......
            const int model_functions_size = model.functions_size();
            for (int j = dead_function_index; j < model_functions_size - 1; j++)
            {
                model.mutable_functions()->SwapElements(j, j + 1);
            }

            //  ..... ....... dfi
            model.mutable_functions()->RemoveLast();
        }
    }

    // eliminate dead initializers
    {
        onnx::GraphProto* graph = model.mutable_graph();

        std::unordered_set<std::string> live_inputs;
        for (int i = 0; i < graph->node_size(); i++)
        {
            const onnx::NodeProto& node = graph->node(i);

            for (int j = 0; j < node.input_size(); j++)
            {
                live_inputs.insert(node.input(j));
            }
        }

        // find live inputs in functions
        for (int i = 0; i < model.functions_size(); i++)
        {
            const onnx::FunctionProto& function = model.functions(i);

            for (int j = 0; j < function.node_size(); j++)
            {
                const onnx::NodeProto& node = function.node(j);

                for (int k = 0; k < node.input_size(); k++)
                {
                    live_inputs.insert(node.input(k));
                }
            }
        }

        std::vector<int> dead_initializer_indexes;
        for (int i = graph->initializer_size() - 1; i >= 0; i--)
        {
            if (live_inputs.find(graph->initializer(i).name()) == live_inputs.end())
            {
                dead_initializer_indexes.push_back(i);
            }
        }

        for (size_t i = 0; i < dead_initializer_indexes.size(); i++)
        {
            const int dead_initializer_index = dead_initializer_indexes[i];

            //  ..... dii .......
            const int graph_initializer_size = graph->initializer_size();
            for (int j = dead_initializer_index; j < graph_initializer_size - 1; j++)
            {
                graph->mutable_initializer()->SwapElements(j, j + 1);
            }

            //  ..... ....... dii
            graph->mutable_initializer()->RemoveLast();
        }
    }
}

} // namespace onnx2pnnx

} // namespace pnnx
