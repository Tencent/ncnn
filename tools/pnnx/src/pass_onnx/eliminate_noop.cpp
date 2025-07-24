// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop.h"

#include <sstream>
#include <string>
#include <unordered_set>

#include "dead_code_elimination.h"

namespace pnnx {

namespace onnx2pnnx {

static onnx::ValueInfoProto* find_value_info_by_name(onnx::GraphProto* graph, const std::string& name)
{
    if (name.empty())
        return NULL;

    // input
    for (int i = 0; i < graph->input_size(); i++)
    {
        if (graph->input(i).name() == name)
        {
            return graph->mutable_input(i);
        }
    }

    // output
    for (int i = 0; i < graph->output_size(); i++)
    {
        if (graph->output(i).name() == name)
        {
            return graph->mutable_output(i);
        }
    }

    for (int i = 0; i < graph->value_info_size(); i++)
    {
        if (graph->mutable_value_info(i)->name() == name)
        {
            return graph->mutable_value_info(i);
        }
    }

    return NULL;
}

void eliminate_noop(onnx::ModelProto& model)
{
    onnx::GraphProto* graph = model.mutable_graph();

    for (int i = 0; i < graph->node_size(); i++)
    {
        const onnx::NodeProto& node = graph->node(i);
        const std::string& op_type = node.op_type();

        bool noop = false;
        if (op_type == "Identity" || op_type == "Dropout" || op_type == "aten_copy")
        {
            noop = true;
        }

        if (!noop)
            continue;

        const std::string& input_name = node.input(0);
        const std::string& output_name = node.output(0);

        for (int j = i + 1; j < graph->node_size(); j++)
        {
            onnx::NodeProto* node2 = graph->mutable_node(j);

            for (int k = 0; k < node2->input_size(); k++)
            {
                if (node2->input(k) == output_name)
                {
                    node2->set_input(k, input_name);
                }
            }
        }

        for (int j = 0; j < graph->output_size(); j++)
        {
            if (graph->output(j).name() == output_name)
            {
                graph->mutable_output(j)->set_name(input_name);
            }
        }
    }

    onnx2pnnx::dead_code_elimination(model);
}

void eliminate_noop_with_shape(onnx::ModelProto& model)
{
    onnx::GraphProto* graph = model.mutable_graph();

    for (int i = 0; i < graph->node_size(); i++)
    {
        const onnx::NodeProto& node = graph->node(i);
        const std::string& op_type = node.op_type();

        bool noop = false;

        if (op_type == "Cast")
        {
            onnx::ValueInfoProto* input_value = find_value_info_by_name(graph, node.input(0));
            onnx::ValueInfoProto* output_value = find_value_info_by_name(graph, node.output(0));

            if (!input_value || !output_value)
                continue;

            if (input_value->type().has_tensor_type() && output_value->type().has_tensor_type())
            {
                if (input_value->type().tensor_type().elem_type() == output_value->type().tensor_type().elem_type())
                    noop = true;
            }
        }

        if (op_type == "Reshape")
        {
            onnx::ValueInfoProto* input_value = find_value_info_by_name(graph, node.input(0));
            onnx::ValueInfoProto* output_value = find_value_info_by_name(graph, node.output(0));

            if (!input_value || !output_value)
                continue;

            if (input_value->type().has_tensor_type() && output_value->type().has_tensor_type())
            {
                const onnx::TensorShapeProto& input_tsp = input_value->type().tensor_type().shape();
                const onnx::TensorShapeProto& output_tsp = output_value->type().tensor_type().shape();
                if (input_tsp.dim_size() == output_tsp.dim_size())
                {
                    bool is_shape_same = true;

                    int dynamic_index_count = 0;
                    for (int j = 0; j < input_tsp.dim_size(); j++)
                    {
                        if (input_tsp.dim(j).has_dim_value() && output_tsp.dim(j).has_dim_value())
                        {
                            if (input_tsp.dim(j).dim_value() != output_tsp.dim(j).dim_value())
                            {
                                is_shape_same = false;
                                break;
                            }
                        }
                        else
                        {
                            dynamic_index_count++;
                            if (dynamic_index_count > 1)
                            {
                                is_shape_same = false;
                                break;
                            }
                        }
                    }

                    if (is_shape_same)
                        noop = true;
                }
            }
        }

        if (!noop)
            continue;

        const std::string& input_name = node.input(0);
        const std::string& output_name = node.output(0);

        for (int j = i + 1; j < graph->node_size(); j++)
        {
            onnx::NodeProto* node2 = graph->mutable_node(j);

            for (int k = 0; k < node2->input_size(); k++)
            {
                if (node2->input(k) == output_name)
                {
                    node2->set_input(k, input_name);
                }
            }
        }

        for (int j = 0; j < graph->output_size(); j++)
        {
            if (graph->output(j).name() == output_name)
            {
                graph->mutable_output(j)->set_name(input_name);
            }
        }
    }

    onnx2pnnx::dead_code_elimination(model);
}

} // namespace onnx2pnnx

} // namespace pnnx
