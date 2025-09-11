// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "canonicalize.h"

#include <map>
#include <string>
#include <unordered_set>

namespace pnnx {

namespace onnx2pnnx {

static bool string_starts_with(const std::string& s, const std::string& s2)
{
    return strncmp(s.c_str(), s2.c_str(), s2.size()) == 0;
}

static void canonicalize_subgraph_nodes(onnx::GraphProto* graph, const std::unordered_set<std::string>& initializers, const std::map<std::string, std::string>& input_output_remap)
{
    for (int i = 0; i < graph->node_size(); i++)
    {
        onnx::NodeProto* node = graph->mutable_node(i);

        // canonicalize name
        // node->set_name(std::string("op_") + std::to_string(i));

        // canonicalize node input output
        {
            for (int j = 0; j < node->input_size(); j++)
            {
                const std::string& node_input = node->input(j);

                // some input/output may have empty name, it causes trouble, skip it
                if (node_input.empty())
                    continue;

                // skip initializer
                if (initializers.find(node_input) != initializers.end())
                    continue;

                if (input_output_remap.find(node_input) != input_output_remap.end())
                {
                    node->set_input(j, input_output_remap.at(node_input));
                }
            }
            for (int j = 0; j < node->output_size(); j++)
            {
                const std::string& node_output = node->output(j);

                // some input/output may have empty name, it causes trouble, skip it
                if (node_output.empty())
                    continue;

                if (input_output_remap.find(node_output) != input_output_remap.end())
                {
                    node->set_output(j, input_output_remap.at(node_output));
                }
            }
        }
    }
}

void canonicalize(onnx::ModelProto& model)
{
    // collect initializers
    std::unordered_set<std::string> initializers;
    {
        const onnx::GraphProto& graph = model.graph();
        for (int i = 0; i < graph.initializer_size(); i++)
        {
            initializers.insert(graph.initializer(i).name());
        }
    }

    onnx::GraphProto* graph = model.mutable_graph();

    std::map<std::string, std::string> function_remap;

    std::map<std::string, std::string> input_output_remap;
    int input_output_index = 0;

    // canonicalize graph input output
    {
        for (int i = 0; i < graph->input_size(); i++)
        {
            onnx::ValueInfoProto* input = graph->mutable_input(i);

            std::string new_name = std::string("in") + std::to_string(i);

            // fprintf(stderr, "%s -> %s\n", input->name().c_str(), new_name.c_str());
            input_output_remap[input->name()] = new_name;
            input->set_name(new_name);
        }
        for (int i = 0; i < graph->output_size(); i++)
        {
            onnx::ValueInfoProto* output = graph->mutable_output(i);

            std::string new_name = std::string("out") + std::to_string(i);

            // fprintf(stderr, "%s -> %s\n", output->name().c_str(), new_name.c_str());
            input_output_remap[output->name()] = new_name;
            output->set_name(new_name);
        }
    }

    for (int i = 0; i < graph->node_size(); i++)
    {
        onnx::NodeProto* node = graph->mutable_node(i);

        // simplify type
        {
            const std::string& op_type = node->op_type();

            if (node->domain().empty())
            {
                // native onnx op
                // Constant
                node->set_name(op_type + "_" + std::to_string(i));
            }
            else if (string_starts_with(op_type, "aten_"))
            {
                // aten_view
                node->set_name(op_type.substr(5) + "_" + std::to_string(i));
            }
            else if (string_starts_with(op_type, "_aten_"))
            {
                node->set_name(op_type.substr(6) + "_" + std::to_string(i));
            }
            else if (string_starts_with(op_type, "prims_"))
            {
                // prims_convert_element_type
                node->set_name(op_type.substr(6) + "_" + std::to_string(i));
            }
            else if (string_starts_with(op_type, "torch_nn_modules_") && !string_starts_with(op_type, "torch_nn_modules_container_"))
            {
                // torch_nn_modules_conv_Conv2d                 _conv1_1
                // torch_nn_modules_batchnorm_BatchNorm2d       _bn1_1
                // torch_nn_modules_pooling_MaxPool2d           _maxpool_1_3
                // torch_nn_modules_linear_Linear               _fc_1

                if (function_remap.find(op_type) != function_remap.end())
                {
                    node->set_op_type(function_remap.at(op_type));
                }
                else
                {
                    // torch_nn_modules_conv_Conv2d_xyz -> nn_Conv2d_i
                    char nn_type[256];
                    int nconsumed = 0;
                    sscanf(op_type.c_str() + sizeof("torch_nn_modules_") - 1, "%*[^_]_%255[^_]_%n", nn_type, &nconsumed);

                    std::string new_op_type = std::string("nn_") + nn_type + "_" + std::to_string(i);

                    function_remap[op_type] = new_op_type;

                    node->set_op_type(new_op_type);
                }
                node->set_name(node->op_type().substr(3));
            }
            else
            {
                // unknown module ?
                fprintf(stderr, "unexpected op_type %s\n", op_type.c_str());
                node->set_name(std::string("op_") + std::to_string(i));
            }
        }

        // canonicalize name
        // node->set_name(std::string("op_") + std::to_string(i));

        // canonicalize node input output
        {
            for (int j = 0; j < node->input_size(); j++)
            {
                const std::string& node_input = node->input(j);

                // some input/output may have empty name, it causes trouble, skip it
                if (node_input.empty())
                    continue;

                // skip initializer
                if (initializers.find(node_input) != initializers.end())
                    continue;

                if (input_output_remap.find(node_input) != input_output_remap.end())
                {
                    node->set_input(j, input_output_remap.at(node_input));
                }
                else
                {
                    std::string new_name = std::string("pnnx_") + std::to_string(input_output_index);

                    // fprintf(stderr, "%s -> pnnx_%s\n", node_input.c_str(), new_name.c_str());

                    input_output_remap[node_input] = new_name;
                    node->set_input(j, new_name);
                    input_output_index++;
                }
            }
            for (int j = 0; j < node->output_size(); j++)
            {
                const std::string& node_output = node->output(j);

                // some input/output may have empty name, it causes trouble, skip it
                if (node_output.empty())
                    continue;

                if (input_output_remap.find(node_output) != input_output_remap.end())
                {
                    node->set_output(j, input_output_remap.at(node_output));
                }
                else
                {
                    std::string new_name = std::string("pnnx_") + std::to_string(input_output_index);

                    // fprintf(stderr, "%s -> pnnx_%s\n", node_output.c_str(), new_name.c_str());

                    input_output_remap[node_output] = new_name;
                    node->set_output(j, new_name);
                    input_output_index++;
                }
            }
        }

        // canonicalize node input output in subgraph
        for (int j = 0; j < node->attribute_size(); j++)
        {
            onnx::AttributeProto* attr = node->mutable_attribute(j);

            if (attr->type() == onnx::AttributeProto::GRAPH)
            {
                onnx::GraphProto* sg = attr->mutable_g();

                canonicalize_subgraph_nodes(sg, initializers, input_output_remap);
            }
            if (attr->type() == onnx::AttributeProto::GRAPHS)
            {
                for (int k = 0; k < attr->graphs().size(); k++)
                {
                    onnx::GraphProto* sg = attr->mutable_graphs(k);

                    canonicalize_subgraph_nodes(sg, initializers, input_output_remap);
                }
            }
        }
    }

    // canonicalize all functions
    for (int i = 0; i < model.functions_size(); i++)
    {
        onnx::FunctionProto* function = model.mutable_functions(i);

        if (function_remap.find(function->name()) != function_remap.end())
        {
            function->set_name(function_remap.at(function->name()));
        }

        if (!string_starts_with(function->name(), "nn_"))
            continue;

        // simplify function input
        int function_input_index = 0;
        int function_output_index = 0;
        std::map<std::string, std::string> function_input_output_remap;
        for (int j = 0; j < function->input_size(); j++)
        {
            const std::string& func_input = function->input(j);

            if (initializers.find(func_input) == initializers.end())
            {
                // input tensor
                std::string new_name = std::string("in") + std::to_string(function_input_index);
                function_input_output_remap[func_input] = new_name;
                function->set_input(j, new_name);
                function_input_index++;
            }
            else
            {
                // weights
                // layer2.0.bn1.running_mean
                size_t last_dot = func_input.find_last_of('.');
                if (last_dot != std::string::npos)
                {
                    std::string new_name = func_input.substr(last_dot + 1);
                    function_input_output_remap[func_input] = new_name;
                    function->set_input(j, new_name);
                }
            }
        }
        for (int j = 0; j < function->output_size(); j++)
        {
            const std::string& func_output = function->output(j);

            // output tensor
            std::string new_name = std::string("out") + std::to_string(function_output_index);
            function_input_output_remap[func_output] = new_name;
            function->set_output(j, new_name);
            function_output_index++;
        }

        for (int j = 0; j < function->node_size(); j++)
        {
            onnx::NodeProto* node = function->mutable_node(j);

            for (int k = 0; k < node->input_size(); k++)
            {
                const std::string& input = node->input(k);

                if (function_input_output_remap.find(input) != function_input_output_remap.end())
                {
                    node->set_input(k, function_input_output_remap[input]);
                }
            }
            for (int k = 0; k < node->output_size(); k++)
            {
                const std::string& output = node->output(k);

                if (function_input_output_remap.find(output) != function_input_output_remap.end())
                {
                    node->set_output(k, function_input_output_remap[output]);
                }
            }
        }
    }

    // canonicalize all initializers
    // for (int i = 0; i < graph->initializer_size(); i++)
    // {
    //     onnx::TensorProto* initializer = graph->mutable_initializer(i);
    //
    //     if (input_output_remap.find(initializer->name()) == input_output_remap.end())
    //     {
    //         // skip initializers inside module function
    //         continue;
    //     }
    //
    //     initializer->set_name(input_output_remap.at(initializer->name()));
    // }

    // canonicalize all values
    for (int i = 0; i < graph->value_info_size(); i++)
    {
        onnx::ValueInfoProto* value = graph->mutable_value_info(i);

        if (input_output_remap.find(value->name()) == input_output_remap.end())
        {
            // skip values inside module function
            continue;
        }

        value->set_name(input_output_remap.at(value->name()));
    }
}

} // namespace onnx2pnnx

} // namespace pnnx
