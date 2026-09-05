// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_graph.h"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pnnx {

struct ExportedGraphNormalizationContext
{
    ExportedGraph* graph;
    std::set<std::string> tensor_names;
    std::set<std::string> defined_tensor_names;
    size_t subgraph_index;
};

static int graph_error(const ExportedNode& node, const std::string& message, std::string& error)
{
    error = node.target + ": " + message;
    return -1;
}

static bool tensor_meta_equal(const ExportedTensorMeta& a, const ExportedTensorMeta& b)
{
    return a.dtype == b.dtype
           && a.sizes == b.sizes
           && a.strides == b.strides
           && a.storage_offset == b.storage_offset
           && a.layout == b.layout
           && a.requires_grad == b.requires_grad
           && a.device_type == b.device_type
           && a.device_index == b.device_index
           && a.has_device_index == b.has_device_index;
}

static std::string unique_tensor_name(const std::string& requested, std::set<std::string>& names)
{
    if (names.insert(requested).second)
        return requested;

    for (size_t suffix = 1;; suffix++)
    {
        std::ostringstream candidate;
        candidate << requested << '_' << suffix;
        if (names.insert(candidate.str()).second)
            return candidate.str();
    }
}

static void remember_tensor_names(const ExportedArgument& argument,
                                  std::map<std::string, std::string>& tensor_names,
                                  std::set<std::string>& used_names)
{
    if (argument.type == EXPORTED_ARGUMENT_TENSOR)
    {
        tensor_names[argument.name] = argument.name;
        used_names.insert(argument.name);
        return;
    }

    if (argument.type == EXPORTED_ARGUMENT_TENSOR_LIST)
    {
        for (size_t i = 0; i < argument.tensor_names.size(); i++)
        {
            tensor_names[argument.tensor_names[i]] = argument.tensor_names[i];
            used_names.insert(argument.tensor_names[i]);
        }
    }
}

static void remember_defined_tensor_names(const ExportedArgument& argument, std::set<std::string>& defined_names)
{
    if (argument.type == EXPORTED_ARGUMENT_TENSOR)
    {
        defined_names.insert(argument.name);
        return;
    }

    if (argument.type == EXPORTED_ARGUMENT_TENSOR_LIST)
    {
        for (size_t i = 0; i < argument.tensor_names.size(); i++)
            defined_names.insert(argument.tensor_names[i]);
    }
}

static int bind_tensor_name(const std::string& source_name,
                            const std::string& target_name,
                            std::map<std::string, std::string>& tensor_names,
                            const ExportedNode& node,
                            std::string& error)
{
    const std::map<std::string, std::string>::const_iterator existing = tensor_names.find(source_name);
    if (existing != tensor_names.end())
    {
        if (existing->second != target_name)
            return graph_error(node, "subgraph tensor " + source_name + " has conflicting bindings", error);
        return 0;
    }

    tensor_names[source_name] = target_name;
    return 0;
}

static int validate_bound_tensor_metadata(const ExportedGraph& subgraph,
        const std::string& source_name,
        const std::string& target_name,
        const ExportedNode& node,
        const ExportedGraphNormalizationContext& context,
        std::string& error)
{
    const std::map<std::string, ExportedTensorMeta>::const_iterator source_meta = subgraph.tensor_values.find(source_name);
    if (source_meta == subgraph.tensor_values.end())
        return graph_error(node, "bound subgraph tensor " + source_name + " is missing metadata", error);

    const std::map<std::string, ExportedTensorMeta>::const_iterator target_meta = context.graph->tensor_values.find(target_name);
    if (target_meta == context.graph->tensor_values.end())
        return graph_error(node, "bound tensor " + target_name + " is missing metadata", error);
    if (!tensor_meta_equal(source_meta->second, target_meta->second))
        return graph_error(node, "subgraph tensor metadata does not match bound value " + target_name, error);

    return 0;
}

static int map_tensor_name(const std::string& source_name,
                           const std::map<std::string, std::string>& tensor_names,
                           const ExportedNode& node,
                           std::string& target_name,
                           std::string& error)
{
    const std::map<std::string, std::string>::const_iterator mapped = tensor_names.find(source_name);
    if (mapped == tensor_names.end())
        return graph_error(node, "subgraph tensor " + source_name + " is missing metadata", error);

    target_name = mapped->second;
    return 0;
}

static int map_argument(const ExportedArgument& source,
                        const std::map<std::string, std::string>& tensor_names,
                        const ExportedNode& node,
                        ExportedArgument& result,
                        std::string& error)
{
    result = source;
    if (source.type == EXPORTED_ARGUMENT_TENSOR)
        return map_tensor_name(source.name, tensor_names, node, result.name, error);

    if (source.type == EXPORTED_ARGUMENT_TENSOR_LIST)
    {
        result.tensor_names.clear();
        result.tensor_names.reserve(source.tensor_names.size());
        for (size_t i = 0; i < source.tensor_names.size(); i++)
        {
            std::string mapped_name;
            if (map_tensor_name(source.tensor_names[i], tensor_names, node, mapped_name, error) != 0)
                return -1;
            result.tensor_names.push_back(mapped_name);
        }
        return 0;
    }

    if (source.type == EXPORTED_ARGUMENT_GRAPH)
        return graph_error(node, "graph arguments are only supported by known higher-order wrappers", error);

    return 0;
}

static int validate_positional_wrapper(const ExportedNode& node, std::string& error)
{
    for (size_t i = 0; i < node.inputs.size(); i++)
    {
        if (node.inputs[i].kind != EXPORTED_ARGUMENT_KIND_POSITIONAL)
            return graph_error(node, "higher-order wrapper arguments must be positional", error);
    }

    return 0;
}

static int prepare_subgraph_names(const ExportedGraph& subgraph,
                                  const std::map<std::string, std::string>& bindings,
                                  const std::string& prefix,
                                  const ExportedNode& node,
                                  ExportedGraphNormalizationContext& context,
                                  std::map<std::string, std::string>& tensor_names,
                                  std::string& error)
{
    tensor_names = bindings;

    for (std::map<std::string, ExportedTensorMeta>::const_iterator it = subgraph.tensor_values.begin(); it != subgraph.tensor_values.end(); ++it)
    {
        const std::map<std::string, std::string>::const_iterator binding = tensor_names.find(it->first);
        if (binding != tensor_names.end())
        {
            const std::map<std::string, ExportedTensorMeta>::const_iterator target_meta = context.graph->tensor_values.find(binding->second);
            if (target_meta == context.graph->tensor_values.end())
                return graph_error(node, "bound tensor " + binding->second + " is missing metadata", error);
            if (!tensor_meta_equal(it->second, target_meta->second))
                return graph_error(node, "subgraph tensor metadata does not match bound value " + binding->second, error);
            continue;
        }

        const std::string target_name = unique_tensor_name(prefix + '_' + it->first, context.tensor_names);
        tensor_names[it->first] = target_name;
        context.graph->tensor_values[target_name] = it->second;
    }

    for (std::map<std::string, std::string>::const_iterator it = bindings.begin(); it != bindings.end(); ++it)
    {
        if (subgraph.tensor_values.find(it->first) == subgraph.tensor_values.end())
            return graph_error(node, "bound subgraph tensor " + it->first + " is missing metadata", error);
    }

    return 0;
}

static int append_normalized_nodes(const ExportedGraph& source,
                                   const std::map<std::string, std::string>& tensor_names,
                                   ExportedGraphNormalizationContext& context,
                                   std::string& error);

static int append_higher_order_graph(const ExportedNode& node,
                                     const std::map<std::string, std::string>& parent_tensor_names,
                                     ExportedGraphNormalizationContext& context,
                                     std::string& error)
{
    if (validate_positional_wrapper(node, error) != 0)
        return -1;

    size_t graph_index = 0;
    size_t capture_index = 0;
    if (node.target == "torch.ops.higher_order.wrap_with_set_grad_enabled")
    {
        if (node.inputs.size() < 2)
            return graph_error(node, "set-grad wrapper requires a flag and graph", error);
        if (node.inputs[0].arg.type != EXPORTED_ARGUMENT_BOOL)
            return graph_error(node, "set-grad wrapper flag must be boolean", error);
        if (node.inputs[0].arg.bool_value)
            return graph_error(node, "enabled set-grad higher-order graph is unsupported", error);
        graph_index = 1;
        capture_index = 2;
    }
    else if (node.target == "torch.ops.higher_order.wrap_with_autocast")
    {
        if (node.inputs.size() < 5)
            return graph_error(node, "autocast wrapper requires device, dtype, flags, and graph", error);
        if (node.inputs[0].arg.type != EXPORTED_ARGUMENT_STRING || node.inputs[0].arg.string_value.empty())
            return graph_error(node, "autocast wrapper device must be a non-empty string", error);
        if (node.inputs[2].arg.type != EXPORTED_ARGUMENT_BOOL)
            return graph_error(node, "autocast wrapper enabled flag must be boolean", error);
        if (node.inputs[2].arg.bool_value)
            return graph_error(node, "enabled autocast higher-order graph is unsupported", error);
        if (node.inputs[1].arg.type != EXPORTED_ARGUMENT_NONE && node.inputs[1].arg.type != EXPORTED_ARGUMENT_SCALAR_TYPE)
            return graph_error(node, "autocast wrapper dtype must be none or a scalar type", error);
        if (node.inputs[3].arg.type != EXPORTED_ARGUMENT_NONE && node.inputs[3].arg.type != EXPORTED_ARGUMENT_BOOL)
            return graph_error(node, "autocast wrapper cache flag must be none or boolean", error);
        graph_index = 4;
        capture_index = 5;
    }
    else
    {
        return graph_error(node, "unsupported exported higher-order operator", error);
    }

    const ExportedArgument& graph_argument = node.inputs[graph_index].arg;
    if (graph_argument.type != EXPORTED_ARGUMENT_GRAPH || !graph_argument.graph_value)
        return graph_error(node, "higher-order wrapper graph argument is invalid", error);

    const ExportedGraph& subgraph = *graph_argument.graph_value;
    if (!subgraph.custom_obj_values.empty())
        return graph_error(node, "higher-order subgraph custom objects are unsupported", error);
    if (subgraph.inputs.size() != node.inputs.size() - capture_index)
        return graph_error(node, "captured argument count does not match subgraph input count", error);
    if (subgraph.outputs.size() != node.outputs.size())
        return graph_error(node, "wrapper output count does not match subgraph output count", error);

    std::map<std::string, std::string> bindings;
    for (size_t i = 0; i < subgraph.inputs.size(); i++)
    {
        if (subgraph.inputs[i].type != EXPORTED_ARGUMENT_TENSOR)
            return graph_error(node, "higher-order subgraph inputs must be tensors", error);
        if (node.inputs[capture_index + i].arg.type != EXPORTED_ARGUMENT_TENSOR)
            return graph_error(node, "higher-order captured arguments must be tensors", error);

        std::string captured_name;
        if (map_tensor_name(node.inputs[capture_index + i].arg.name, parent_tensor_names, node, captured_name, error) != 0)
            return -1;
        if (bind_tensor_name(subgraph.inputs[i].name, captured_name, bindings, node, error) != 0)
            return -1;
    }

    std::vector<std::string> wrapper_output_names;
    wrapper_output_names.reserve(subgraph.outputs.size());
    std::set<std::string> distinct_wrapper_outputs;
    for (size_t i = 0; i < subgraph.outputs.size(); i++)
    {
        if (subgraph.outputs[i].type != EXPORTED_ARGUMENT_TENSOR || node.outputs[i].type != EXPORTED_ARGUMENT_TENSOR)
            return graph_error(node, "higher-order graph outputs must be tensors", error);

        std::string output_name;
        if (map_tensor_name(node.outputs[i].name, parent_tensor_names, node, output_name, error) != 0)
            return -1;
        if (!distinct_wrapper_outputs.insert(output_name).second)
            return graph_error(node, "wrapper output tensor " + output_name + " is defined more than once", error);
        if (context.defined_tensor_names.find(output_name) != context.defined_tensor_names.end())
            return graph_error(node, "wrapper output tensor " + output_name + " is defined more than once", error);
        if (validate_bound_tensor_metadata(subgraph, subgraph.outputs[i].name, output_name, node, context, error) != 0)
            return -1;

        if (bindings.find(subgraph.outputs[i].name) == bindings.end())
            bindings[subgraph.outputs[i].name] = output_name;
        wrapper_output_names.push_back(output_name);
    }

    std::ostringstream prefix;
    prefix << "pnnx_subgraph_" << context.subgraph_index++;
    std::map<std::string, std::string> subgraph_tensor_names;
    if (prepare_subgraph_names(subgraph, bindings, prefix.str(), node, context, subgraph_tensor_names, error) != 0)
        return -1;

    if (append_normalized_nodes(subgraph, subgraph_tensor_names, context, error) != 0)
        return -1;

    for (size_t i = 0; i < subgraph.outputs.size(); i++)
    {
        const std::map<std::string, std::string>::const_iterator canonical = subgraph_tensor_names.find(subgraph.outputs[i].name);
        if (canonical == subgraph_tensor_names.end())
            return graph_error(node, "subgraph output tensor " + subgraph.outputs[i].name + " has no canonical mapping", error);
        if (context.defined_tensor_names.find(canonical->second) == context.defined_tensor_names.end())
            return graph_error(node, "subgraph output tensor " + subgraph.outputs[i].name + " has no producer", error);
        if (canonical->second == wrapper_output_names[i])
            continue;
        if (!context.defined_tensor_names.insert(wrapper_output_names[i]).second)
            return graph_error(node, "wrapper output tensor " + wrapper_output_names[i] + " is defined more than once", error);

        ExportedNode alias;
        alias.target = "torch.ops.aten.alias.default";
        ExportedNamedArgument input;
        input.name = "self";
        input.kind = EXPORTED_ARGUMENT_KIND_POSITIONAL;
        input.arg.type = EXPORTED_ARGUMENT_TENSOR;
        input.arg.name = canonical->second;
        alias.inputs.push_back(input);
        ExportedArgument output;
        output.type = EXPORTED_ARGUMENT_TENSOR;
        output.name = wrapper_output_names[i];
        alias.outputs.push_back(output);
        context.graph->nodes.push_back(alias);
    }

    return 0;
}

static int append_normalized_nodes(const ExportedGraph& source,
                                   const std::map<std::string, std::string>& tensor_names,
                                   ExportedGraphNormalizationContext& context,
                                   std::string& error)
{
    static const std::string higher_order_prefix = "torch.ops.higher_order.";

    for (size_t i = 0; i < source.nodes.size(); i++)
    {
        const ExportedNode& node = source.nodes[i];
        if (node.target.compare(0, higher_order_prefix.size(), higher_order_prefix) == 0)
        {
            if (append_higher_order_graph(node, tensor_names, context, error) != 0)
                return -1;
            continue;
        }

        ExportedNode normalized_node = node;
        for (size_t j = 0; j < normalized_node.inputs.size(); j++)
        {
            if (map_argument(node.inputs[j].arg, tensor_names, node, normalized_node.inputs[j].arg, error) != 0)
                return -1;
        }
        for (size_t j = 0; j < normalized_node.outputs.size(); j++)
        {
            if (map_argument(node.outputs[j], tensor_names, node, normalized_node.outputs[j], error) != 0)
                return -1;
        }
        for (size_t j = 0; j < normalized_node.outputs.size(); j++)
        {
            if (normalized_node.outputs[j].type == EXPORTED_ARGUMENT_TENSOR)
            {
                if (!context.defined_tensor_names.insert(normalized_node.outputs[j].name).second)
                    return graph_error(node, "tensor value " + normalized_node.outputs[j].name + " is defined more than once", error);
                continue;
            }

            if (normalized_node.outputs[j].type == EXPORTED_ARGUMENT_TENSOR_LIST)
            {
                for (size_t k = 0; k < normalized_node.outputs[j].tensor_names.size(); k++)
                {
                    if (!context.defined_tensor_names.insert(normalized_node.outputs[j].tensor_names[k]).second)
                        return graph_error(node, "tensor value " + normalized_node.outputs[j].tensor_names[k] + " is defined more than once", error);
                }
            }
        }
        context.graph->nodes.push_back(normalized_node);
    }

    return 0;
}

int normalize_exported_program_graph(const ExportedGraph& graph, ExportedGraph& normalized_graph, std::string& error)
{
    error.clear();
    normalized_graph = ExportedGraph();

    ExportedGraph candidate;
    candidate.inputs = graph.inputs;
    candidate.outputs = graph.outputs;
    candidate.tensor_values = graph.tensor_values;
    candidate.custom_obj_values = graph.custom_obj_values;
    candidate.is_single_tensor_return = graph.is_single_tensor_return;

    ExportedGraphNormalizationContext context;
    context.graph = &candidate;
    context.subgraph_index = 0;

    std::map<std::string, std::string> tensor_names;
    for (std::map<std::string, ExportedTensorMeta>::const_iterator it = graph.tensor_values.begin(); it != graph.tensor_values.end(); ++it)
    {
        context.tensor_names.insert(it->first);
        tensor_names[it->first] = it->first;
    }

    for (size_t i = 0; i < graph.inputs.size(); i++)
    {
        remember_tensor_names(graph.inputs[i], tensor_names, context.tensor_names);
        remember_defined_tensor_names(graph.inputs[i], context.defined_tensor_names);
    }
    for (size_t i = 0; i < graph.outputs.size(); i++)
        remember_tensor_names(graph.outputs[i], tensor_names, context.tensor_names);
    for (size_t i = 0; i < graph.nodes.size(); i++)
    {
        for (size_t j = 0; j < graph.nodes[i].inputs.size(); j++)
            remember_tensor_names(graph.nodes[i].inputs[j].arg, tensor_names, context.tensor_names);
        for (size_t j = 0; j < graph.nodes[i].outputs.size(); j++)
            remember_tensor_names(graph.nodes[i].outputs[j], tensor_names, context.tensor_names);
    }

    if (append_normalized_nodes(graph, tensor_names, context, error) != 0)
        return -1;

    normalized_graph = std::move(candidate);
    return 0;
}

} // namespace pnnx
