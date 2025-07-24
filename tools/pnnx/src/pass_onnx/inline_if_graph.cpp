// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "inline_containers.h"

#include <map>
#include <string>

namespace pnnx {

namespace onnx2pnnx {

int inline_if_graph(onnx::ModelProto& model)
{
    int inlined = 0;

    // collect initializers
    std::unordered_map<std::string, int> initializers;
    {
        const onnx::GraphProto& graph = model.graph();
        for (int i = 0; i < graph.initializer_size(); i++)
        {
            initializers.insert(std::make_pair(graph.initializer(i).name(), i));
        }
    }

    onnx::GraphProto* graph = model.mutable_graph();

    for (int i = 0; i < graph->node_size(); i++)
    {
        onnx::NodeProto* node = graph->mutable_node(i);

        const std::string& op_type = node->op_type();

        if (op_type != "If")
            continue;

        // find constant cond
        if (initializers.find(node->input(0)) == initializers.end())
            continue;

        bool cond;
        {
            const onnx::TensorProto& tensor = graph->initializer(initializers.at(node->input(0)));
            if (tensor.has_raw_data())
            {
                // assert tensor.raw_data().size() == 2
                cond = ((uint16_t*)tensor.raw_data().data())[0] ? true : false;
            }
            else
            {
                // assert tensor.int32_data().size() == 1
                cond = tensor.int32_data().at(0) ? true : false;
            }
        }

        onnx::GraphProto* sg = 0;
        for (int j = 0; j < node->attribute_size(); j++)
        {
            if (cond == true && node->attribute(j).name() == "then_branch")
            {
                sg = node->mutable_attribute(j)->mutable_g();
                break;
            }
            if (cond == false && node->attribute(j).name() == "else_branch")
            {
                sg = node->mutable_attribute(j)->mutable_g();
                break;
            }
        }

        if (!sg)
            continue;

        // build subgraph output name remap
        std::map<std::string, std::string> output_remap;
        {
            for (int j = 0; j < node->output_size(); j++)
            {
                const std::string& node_output = node->output(j);
                const std::string& sg_output = sg->output(j).name();

                output_remap[sg_output] = node_output;
            }
        }

        // append subgraph nodes to graph
        {
            std::map<std::string, std::string> input_remap;

            graph->mutable_node()->Reserve(graph->node_size() + sg->node_size());
            for (int j = 0; j < sg->node_size(); j++)
            {
                onnx::NodeProto* inlined_node = graph->add_node();
                inlined_node->CopyFrom(sg->node(j));

                // prefix with caller node name
                inlined_node->set_name(node->name() + "/" + inlined_node->name());

                // reset input output
                for (int j = 0; j < inlined_node->input_size(); j++)
                {
                    const std::string& node_input = inlined_node->input(j);
                    if (input_remap.find(node_input) != input_remap.end())
                    {
                        std::string new_name = input_remap.at(node_input);
                        inlined_node->set_input(j, new_name);
                    }
                }
                for (int j = 0; j < inlined_node->output_size(); j++)
                {
                    const std::string& node_output = inlined_node->output(j);
                    if (output_remap.find(node_output) != output_remap.end())
                    {
                        inlined_node->set_output(j, output_remap.at(node_output));
                    }
                    else
                    {
                        std::string new_name = node->name() + "/" + node_output;
                        input_remap[node_output] = new_name;
                        inlined_node->set_output(j, new_name);
                    }
                }
            }
        }

        // swap inlined subgraph nodes to caller
        {
            //  ..... cni ....... 0 1 2 3 4
            const int graph_node_size = graph->node_size();
            for (int j = 0; j < sg->node_size(); j++)
            {
                for (int k = graph_node_size - 1; k > i; k--)
                {
                    graph->mutable_node()->SwapElements(k, k - 1);
                }
            }

            //  ..... 0 1 2 3 4 cni .......
            for (int j = i + sg->node_size(); j < graph_node_size - 1; j++)
            {
                graph->mutable_node()->SwapElements(j, j + 1);
            }

            //  ..... 0 1 2 3 4 ....... cni
            graph->mutable_node()->RemoveLast();
        }

        inlined = 1;

        // inlined node may be another subgraph
        i -= 1;
    }

    return inlined;
}

} // namespace onnx2pnnx

} // namespace pnnx
