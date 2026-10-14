// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_sequences.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <vector>

namespace pnnx {

namespace onnx2pnnx {

static bool is_native_node(const onnx::NodeProto& node)
{
    return node.domain().empty() || node.domain() == "ai.onnx";
}

static int64_t get_integer_attribute(const onnx::NodeProto& node, const char* name, int64_t default_value)
{
    for (int i = 0; i < node.attribute_size(); i++)
    {
        if (node.attribute(i).name() == name && node.attribute(i).type() == onnx::AttributeProto::INT)
        {
            return node.attribute(i).i();
        }
    }
    return default_value;
}

static void add_integer_attribute(onnx::NodeProto& node, const char* name, int64_t value)
{
    onnx::AttributeProto* attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(onnx::AttributeProto::INT);
    attr->set_i(value);
}

static bool get_constant_integers(const onnx::GraphProto& graph, const std::string& name, std::vector<int64_t>& values, bool scalar)
{
    // Initializers that are also inputs are overridable, not constants.
    for (int i = 0; i < graph.input_size(); i++)
    {
        if (graph.input(i).name() == name)
        {
            return false;
        }
    }
    const onnx::TensorProto* tensor = 0;
    for (int i = 0; i < graph.initializer_size(); i++)
    {
        if (graph.initializer(i).name() == name)
        {
            tensor = &graph.initializer(i);
        }
    }
    for (int i = 0; i < graph.node_size(); i++)
    {
        const onnx::NodeProto& node = graph.node(i);
        if (!is_native_node(node) || node.op_type() != "Constant" || node.output_size() != 1 || node.output(0) != name)
        {
            continue;
        }
        for (int j = 0; j < node.attribute_size(); j++)
        {
            if (node.attribute(j).name() == "value" && node.attribute(j).type() == onnx::AttributeProto::TENSOR)
            {
                tensor = &node.attribute(j).t();
            }
        }
    }
    if (!tensor || tensor->data_type() != onnx::TensorProto::INT64 || tensor->data_location() == onnx::TensorProto::EXTERNAL || tensor->external_data_size())
    {
        return false;
    }
    if (scalar ? tensor->dims_size() != 0 : tensor->dims_size() > 1)
    {
        return false;
    }
    int64_t count = tensor->dims_size() ? tensor->dims(0) : 1;
    if (count < 0 || count > 100000)
    {
        return false;
    }
    values.clear();
    if (!tensor->raw_data().empty())
    {
        if (tensor->raw_data().size() != (size_t)count * 8)
        {
            return false;
        }
        for (int64_t i = 0; i < count; i++)
        {
            uint64_t bits = 0;
            for (int k = 0; k < 8; k++)
            {
                bits |= (uint64_t)(unsigned char)tensor->raw_data()[i * 8 + k] << (k * 8);
            }
            int64_t value;
            memcpy(&value, &bits, 8);
            values.push_back(value);
        }
    }
    else
    {
        if (tensor->int64_data_size() != count)
        {
            return false;
        }
        values.assign(tensor->int64_data().begin(), tensor->int64_data().end());
    }
    return true;
}

struct SequenceNameGenerator
{
    std::set<std::string> used_names;
    unsigned int next_id;

    explicit SequenceNameGenerator(const onnx::GraphProto& graph)
        : next_id(0)
    {
        reserve(graph);
    }

    void reserve(const onnx::GraphProto& graph)
    {
        for (int i = 0; i < graph.node_size(); i++)
        {
            const onnx::NodeProto& node = graph.node(i);
            used_names.insert(node.name());
            used_names.insert(node.input().begin(), node.input().end());
            used_names.insert(node.output().begin(), node.output().end());
            for (int j = 0; j < node.attribute_size(); j++)
            {
                const onnx::AttributeProto& attr = node.attribute(j);
                if (attr.has_g())
                    reserve(attr.g());
                for (int k = 0; k < attr.graphs_size(); k++)
                    reserve(attr.graphs(k));
            }
        }
        for (int i = 0; i < graph.sparse_initializer_size(); i++)
        {
            used_names.insert(graph.sparse_initializer(i).values().name());
            used_names.insert(graph.sparse_initializer(i).indices().name());
        }
        for (int i = 0; i < graph.initializer_size(); i++)
        {
            used_names.insert(graph.initializer(i).name());
        }
        for (int i = 0; i < graph.input_size(); i++)
        {
            used_names.insert(graph.input(i).name());
        }
        for (int i = 0; i < graph.output_size(); i++)
        {
            used_names.insert(graph.output(i).name());
        }
        for (int i = 0; i < graph.value_info_size(); i++)
        {
            used_names.insert(graph.value_info(i).name());
        }
    }

    std::string generate()
    {
        std::string name;
        do
        {
            name = "pnnx_seq_" + std::to_string(next_id++);
        } while (!used_names.insert(name).second);
        return name;
    }
};

static onnx::NodeProto make_node(const char* op_type, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, SequenceNameGenerator& names)
{
    onnx::NodeProto node;
    node.set_op_type(op_type);
    node.set_name(names.generate());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        node.add_input(inputs[i]);
    }
    for (size_t i = 0; i < outputs.size(); i++)
    {
        node.add_output(outputs[i]);
    }
    return node;
}

static onnx::NodeProto make_constant_node(const std::vector<int64_t>& values, bool scalar, const std::string& output, SequenceNameGenerator& names)
{
    onnx::NodeProto node = make_node("Constant", {}, {output}, names);
    onnx::AttributeProto* attr = node.add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    onnx::TensorProto* tensor = attr->mutable_t();
    tensor->set_data_type(onnx::TensorProto::INT64);
    if (!scalar)
    {
        tensor->add_dims(values.size());
    }
    for (size_t i = 0; i < values.size(); i++)
    {
        tensor->add_int64_data(values[i]);
    }
    return node;
}

static void append_axes_node(const char* op_type, const std::string& input, const std::string& output, int64_t axis, int opset, SequenceNameGenerator& names, std::vector<onnx::NodeProto>& nodes)
{
    onnx::NodeProto node = make_node(op_type, {input}, {output}, names);
    if (opset >= 13)
    {
        std::string axes = names.generate();
        nodes.push_back(make_constant_node({axis}, false, axes, names));
        node.add_input(axes);
    }
    else
    {
        onnx::AttributeProto* attr = node.add_attribute();
        attr->set_name("axes");
        attr->set_type(onnx::AttributeProto::INTS);
        attr->add_ints(axis);
    }
    nodes.push_back(node);
}

static bool is_value_captured(const onnx::GraphProto& graph, const std::string& name)
{
    for (int i = 0; i < graph.node_size(); i++)
    {
        const onnx::NodeProto& node = graph.node(i);
        for (int j = 0; j < node.input_size(); j++)
        {
            if (node.input(j) == name)
            {
                return true;
            }
        }
        for (int j = 0; j < node.attribute_size(); j++)
        {
            const onnx::AttributeProto& attr = node.attribute(j);
            if (attr.has_g() && is_value_captured(attr.g(), name))
            {
                return true;
            }
            for (int k = 0; k < attr.graphs_size(); k++)
            {
                if (is_value_captured(attr.graphs(k), name))
                {
                    return true;
                }
            }
        }
    }
    for (int i = 0; i < graph.output_size(); i++)
    {
        if (graph.output(i).name() == name)
        {
            return true;
        }
    }
    return false;
}

static bool get_static_shape(const onnx::GraphProto& graph, const std::string& name, std::vector<int64_t>& shape)
{
    const onnx::ValueInfoProto* value_info = 0;
    for (int i = 0; i < graph.input_size(); i++)
    {
        if (graph.input(i).name() == name)
        {
            value_info = &graph.input(i);
        }
    }
    for (int i = 0; i < graph.value_info_size(); i++)
    {
        if (graph.value_info(i).name() == name)
        {
            value_info = &graph.value_info(i);
        }
    }
    for (int i = 0; i < graph.output_size(); i++)
    {
        if (graph.output(i).name() == name)
        {
            value_info = &graph.output(i);
        }
    }
    if (!value_info || !value_info->type().has_tensor_type() || !value_info->type().tensor_type().has_shape())
    {
        return false;
    }
    const onnx::TensorShapeProto& tensor_shape = value_info->type().tensor_type().shape();
    shape.clear();
    for (int i = 0; i < tensor_shape.dim_size(); i++)
    {
        if (!tensor_shape.dim(i).has_dim_value() || tensor_shape.dim(i).dim_value() < 0)
        {
            return false;
        }
        shape.push_back(tensor_shape.dim(i).dim_value());
    }
    return true;
}

static void eliminate_sequences_in_graph(onnx::GraphProto& graph, int opset)
{
    SequenceNameGenerator names(graph);
    for (int producer_index = 0; producer_index < graph.node_size(); producer_index++)
    {
        const onnx::NodeProto producer = graph.node(producer_index);
        if (!is_native_node(producer) || producer.output_size() != 1)
        {
            continue;
        }
        bool is_split = producer.op_type() == "SplitToSequence";
        if (!is_split && producer.op_type() != "SequenceConstruct")
        {
            continue;
        }
        const std::string sequence_name = producer.output(0);
        bool safe = true;
        for (int i = 0; i < graph.output_size(); i++)
        {
            if (graph.output(i).name() == sequence_name)
            {
                safe = false;
            }
        }
        for (int i = 0; i < graph.node_size(); i++)
        {
            for (int j = 0; j < graph.node(i).attribute_size(); j++)
            {
                const onnx::AttributeProto& attr = graph.node(i).attribute(j);
                if (attr.has_g() && is_value_captured(attr.g(), sequence_name))
                {
                    safe = false;
                }
                for (int k = 0; k < attr.graphs_size(); k++)
                {
                    if (is_value_captured(attr.graphs(k), sequence_name))
                    {
                        safe = false;
                    }
                }
            }
        }
        if (!safe)
        {
            continue;
        }
        std::vector<std::string> elements;
        std::vector<onnx::NodeProto> prefix;
        if (!is_split)
        {
            elements.assign(producer.input().begin(), producer.input().end());
        }
        else
        {
            if (producer.input_size() < 1 || producer.input_size() > 2)
            {
                continue;
            }
            std::vector<int64_t> shape;
            if (!get_static_shape(graph, producer.input(0), shape))
            {
                continue;
            }
            int64_t axis = get_integer_attribute(producer, "axis", 0);
            if (axis < 0)
            {
                axis += (int64_t)shape.size();
            }
            if (axis < 0 || axis >= (int64_t)shape.size() || shape[axis] <= 0)
            {
                continue;
            }
            int64_t keepdims = get_integer_attribute(producer, "keepdims", 1);
            if (keepdims != 0 && keepdims != 1)
            {
                continue;
            }
            std::vector<int64_t> sizes;
            bool explicit_split = producer.input_size() == 2 && !producer.input(1).empty();
            if (explicit_split && !get_constant_integers(graph, producer.input(1), sizes, false))
            {
                continue;
            }
            // Some runtimes apply keepdims to explicit split inputs despite
            // the schema saying it is ignored. Leave that combination intact.
            if (explicit_split && keepdims == 0)
            {
                continue;
            }
            bool needs_squeeze = !explicit_split && !keepdims;
            bool scalar_split = !explicit_split;
            if (explicit_split)
            {
                std::vector<int64_t> scalar;
                scalar_split = get_constant_integers(graph, producer.input(1), scalar, true);
            }
            if (scalar_split)
            {
                int64_t chunk = explicit_split ? sizes[0] : 1;
                if (chunk <= 0 || 1 + (shape[axis] - 1) / chunk > 100000)
                {
                    continue;
                }
                sizes.clear();
                for (int64_t remaining = shape[axis]; remaining > 0; remaining -= std::min(remaining, chunk))
                {
                    sizes.push_back(std::min(remaining, chunk));
                }
            }
            int64_t total = 0;
            for (size_t i = 0; i < sizes.size(); i++)
            {
                if (sizes[i] <= 0 || sizes[i] > shape[axis] - total)
                {
                    safe = false;
                    break;
                }
                total += sizes[i];
            }
            if (!safe || sizes.empty() || total != shape[axis])
            {
                continue;
            }
            std::vector<std::string> outputs;
            for (size_t i = 0; i < sizes.size(); i++)
            {
                outputs.push_back(names.generate());
            }
            onnx::NodeProto node = make_node("Split", {producer.input(0)}, outputs, names);
            add_integer_attribute(node, "axis", axis);
            if (opset >= 13)
            {
                std::string sizes_name = names.generate();
                prefix.push_back(make_constant_node(sizes, false, sizes_name, names));
                node.add_input(sizes_name);
            }
            else
            {
                onnx::AttributeProto* attr = node.add_attribute();
                attr->set_name("split");
                attr->set_type(onnx::AttributeProto::INTS);
                for (size_t i = 0; i < sizes.size(); i++)
                {
                    attr->add_ints(sizes[i]);
                }
            }
            prefix.push_back(node);
            elements = outputs;
            if (needs_squeeze)
            {
                for (size_t i = 0; i < outputs.size(); i++)
                {
                    elements[i] = names.generate();
                    append_axes_node("Squeeze", outputs[i], elements[i], axis, opset, names, prefix);
                }
            }
        }
        if (elements.empty())
        {
            continue;
        }
        std::map<int, std::vector<onnx::NodeProto> > replacements;
        for (int i = 0; i < graph.node_size() && safe; i++)
        {
            const onnx::NodeProto& consumer = graph.node(i);
            bool uses_sequence = std::find(consumer.input().begin(), consumer.input().end(), sequence_name) != consumer.input().end();
            if (!uses_sequence)
            {
                continue;
            }
            if (i <= producer_index || !is_native_node(consumer) || consumer.output_size() != 1 || consumer.input(0) != sequence_name)
            {
                safe = false;
                break;
            }
            std::vector<onnx::NodeProto>& replacement_nodes = replacements[i];
            if (consumer.op_type() == "SequenceAt" && consumer.input_size() == 2)
            {
                std::vector<int64_t> index;
                if (!get_constant_integers(graph, consumer.input(1), index, true))
                {
                    safe = false;
                    break;
                }
                int64_t element_index = index[0];
                if (element_index < 0)
                {
                    element_index += (int64_t)elements.size();
                }
                if (element_index < 0 || element_index >= (int64_t)elements.size())
                {
                    safe = false;
                    break;
                }
                replacement_nodes.push_back(make_node("Identity", {elements[element_index]}, {consumer.output(0)}, names));
            }
            else if (consumer.op_type() == "SequenceLength" && consumer.input_size() == 1)
            {
                replacement_nodes.push_back(make_constant_node({(int64_t)elements.size()}, true, consumer.output(0), names));
            }
            else if (consumer.op_type() == "ConcatFromSequence" && consumer.input_size() == 1)
            {
                int64_t axis = get_integer_attribute(consumer, "axis", INT64_MAX);
                int64_t new_axis = get_integer_attribute(consumer, "new_axis", 0);
                if (axis == INT64_MAX || (new_axis != 0 && new_axis != 1))
                {
                    safe = false;
                    break;
                }
                std::vector<std::string> inputs = elements;
                if (new_axis)
                {
                    for (size_t j = 0; j < elements.size(); j++)
                    {
                        inputs[j] = names.generate();
                        append_axes_node("Unsqueeze", elements[j], inputs[j], axis, opset, names, replacement_nodes);
                    }
                }
                onnx::NodeProto node = make_node("Concat", inputs, {consumer.output(0)}, names);
                add_integer_attribute(node, "axis", axis);
                replacement_nodes.push_back(node);
            }
            else
            {
                safe = false;
            }
        }
        if (!safe)
        {
            continue;
        }
        std::vector<onnx::NodeProto> nodes;
        for (int i = 0; i < graph.node_size(); i++)
        {
            if (i == producer_index)
            {
                nodes.insert(nodes.end(), prefix.begin(), prefix.end());
            }
            else if (replacements.count(i))
            {
                nodes.insert(nodes.end(), replacements[i].begin(), replacements[i].end());
            }
            else
            {
                nodes.push_back(graph.node(i));
            }
        }
        graph.clear_node();
        for (size_t i = 0; i < nodes.size(); i++)
        {
            *graph.add_node() = nodes[i];
        }
        for (int i = graph.value_info_size() - 1; i >= 0; i--)
        {
            if (graph.value_info(i).name() == sequence_name)
            {
                graph.mutable_value_info()->DeleteSubrange(i, 1);
            }
        }
        producer_index = -1;
    }
}

static bool expand_sequence_function(const onnx::ModelProto& model, const onnx::NodeProto& call, const std::set<std::string>& sequences, int opset, SequenceNameGenerator& names, std::vector<onnx::NodeProto>& body)
{
    const onnx::FunctionProto* function = 0;
    for (int j = 0; j < model.functions_size(); j++)
    {
        if (model.functions(j).domain() == call.domain() && model.functions(j).name() == call.op_type() && model.functions(j).overload() == call.overload())
        {
            function = &model.functions(j);
        }
    }
    if (!function || function->input_size() != call.input_size() || function->output_size() != call.output_size() || function->attribute_size() || function->attribute_proto_size() || call.attribute_size())
    {
        return false;
    }
    bool safe = false;
    for (int j = 0; j < function->opset_import_size(); j++)
    {
        if (function->opset_import(j).domain().empty() || function->opset_import(j).domain() == "ai.onnx")
        {
            safe = function->opset_import(j).version() == opset;
        }
    }
    if (!safe)
    {
        return false;
    }
    std::map<std::string, std::string> remap;
    for (int j = 0; j < call.input_size(); j++)
    {
        remap[function->input(j)] = call.input(j);
    }
    for (int j = 0; j < call.output_size(); j++)
    {
        if (!remap.count(function->output(j)))
        {
            remap[function->output(j)] = call.output(j);
        }
    }
    std::set<std::string> available(function->input().begin(), function->input().end());
    for (int j = 0; j < function->node_size() && safe; j++)
    {
        onnx::NodeProto node = function->node(j);
        const std::string op_type = node.op_type();
        if (!is_native_node(node) || (op_type != "Constant" && op_type != "SequenceAt" && op_type != "SequenceLength" && op_type != "ConcatFromSequence" && op_type != "Einsum" && op_type != "Identity"))
        {
            safe = false;
            break;
        }
        if ((op_type == "SequenceAt" || op_type == "SequenceLength" || op_type == "ConcatFromSequence") && (node.input_size() == 0 || !sequences.count(remap[node.input(0)])))
        {
            safe = false;
            break;
        }
        for (int k = 0; k < node.attribute_size(); k++)
        {
            if (!node.attribute(k).ref_attr_name().empty() || node.attribute(k).has_g() || node.attribute(k).graphs_size())
            {
                safe = false;
            }
        }
        for (int k = 0; k < node.input_size(); k++)
        {
            const std::string value_name = node.input(k);
            if (!available.count(value_name))
            {
                safe = false;
                break;
            }
            node.set_input(k, remap[value_name]);
        }
        for (int k = 0; k < node.output_size(); k++)
        {
            const std::string value_name = node.output(k);
            if (value_name.empty() || available.count(value_name))
            {
                safe = false;
                break;
            }
            available.insert(value_name);
            if (!remap.count(value_name))
            {
                remap[value_name] = names.generate();
            }
            node.set_output(k, remap[value_name]);
        }
        node.set_name(names.generate());
        body.push_back(node);
    }
    for (int j = 0; j < function->output_size(); j++)
    {
        const std::string value_name = function->output(j);
        if (!available.count(value_name))
        {
            safe = false;
        }
        else if (remap[value_name] != call.output(j))
        {
            onnx::NodeProto identity;
            identity.set_op_type("Identity");
            identity.set_name(names.generate());
            identity.add_input(remap[value_name]);
            identity.add_output(call.output(j));
            body.push_back(identity);
        }
    }
    if (!safe)
    {
        return false;
    }
    return true;
}

static void inline_sequence_functions(onnx::ModelProto& model, int opset)
{
    onnx::GraphProto& graph = *model.mutable_graph();
    for (int call_index = 0; call_index < graph.node_size(); call_index++)
    {
        if (is_native_node(graph.node(call_index)))
            continue;

        std::map<std::string, int> sequence_producers;
        std::set<std::string> sequence_values;
        std::set<std::string> tensor_values;
        for (int j = 0; j < graph.initializer_size(); j++)
            tensor_values.insert(graph.initializer(j).name());
        for (int j = 0; j < graph.input_size(); j++)
        {
            if (graph.input(j).type().has_sequence_type())
                sequence_values.insert(graph.input(j).name());
            if (graph.input(j).type().has_tensor_type())
                tensor_values.insert(graph.input(j).name());
        }
        for (int j = 0; j < graph.value_info_size(); j++)
        {
            if (graph.value_info(j).type().has_sequence_type())
                sequence_values.insert(graph.value_info(j).name());
            if (graph.value_info(j).type().has_tensor_type())
                tensor_values.insert(graph.value_info(j).name());
        }
        for (int j = 0; j < graph.output_size(); j++)
        {
            if (graph.output(j).type().has_sequence_type())
                sequence_values.insert(graph.output(j).name());
            if (graph.output(j).type().has_tensor_type())
                tensor_values.insert(graph.output(j).name());
        }
        for (int j = 0; j < graph.node_size(); j++)
        {
            const onnx::NodeProto& node = graph.node(j);
            if (!is_native_node(node))
                continue;
            if ((node.op_type() == "SequenceConstruct" || node.op_type() == "SplitToSequence") && node.output_size() == 1)
                sequence_producers[node.output(0)] = j;
            if (node.op_type() == "SequenceConstruct" || node.op_type() == "SplitToSequence" || node.op_type() == "SequenceEmpty" || node.op_type() == "SequenceInsert" || node.op_type() == "SequenceErase" || (node.op_type() == "Identity" && node.input_size() == 1 && sequence_values.count(node.input(0))))
                sequence_values.insert(node.output().begin(), node.output().end());
        }

        std::set<std::string> sequences;
        const onnx::NodeProto& initial_call = graph.node(call_index);
        for (int j = 0; j < initial_call.input_size(); j++)
        {
            std::map<std::string, int>::const_iterator producer = sequence_producers.find(initial_call.input(j));
            if (producer != sequence_producers.end() && producer->second < call_index)
                sequences.insert(producer->first);
        }
        if (sequences.empty())
            continue;

        // Close the region over calls and all their sequence arguments before
        // expanding anything. A shared producer cannot lower one call at a time.
        std::set<int> calls;
        bool safe = true;
        bool changed = true;
        while (changed && safe)
        {
            changed = false;
            for (int j = 0; j < graph.node_size() && safe; j++)
            {
                const onnx::NodeProto& call = graph.node(j);
                if (is_native_node(call) || calls.count(j))
                    continue;
                bool uses_sequence = false;
                for (int k = 0; k < call.input_size(); k++)
                {
                    if (sequences.count(call.input(k)))
                        uses_sequence = true;
                }
                if (!uses_sequence)
                    continue;
                calls.insert(j);
                changed = true;
                for (int k = 0; k < call.input_size(); k++)
                {
                    const std::string& input = call.input(k);
                    std::map<std::string, int>::const_iterator producer = sequence_producers.find(input);
                    if (producer != sequence_producers.end() && producer->second < j)
                        sequences.insert(input);
                    else if (sequence_values.count(input) || !tensor_values.count(input))
                        safe = false;
                }
            }
        }
        if (!safe)
            continue;

        SequenceNameGenerator names(graph);
        std::map<int, std::vector<onnx::NodeProto> > replacements;
        for (std::set<int>::const_iterator it = calls.begin(); it != calls.end(); ++it)
        {
            if (!expand_sequence_function(model, graph.node(*it), sequences, opset, names, replacements[*it]))
            {
                safe = false;
                break;
            }
        }
        if (!safe)
            continue;

        onnx::GraphProto candidate = graph;
        candidate.clear_node();
        for (int j = 0; j < graph.node_size(); j++)
        {
            if (replacements.count(j))
            {
                const std::vector<onnx::NodeProto>& body = replacements[j];
                for (size_t k = 0; k < body.size(); k++)
                    *candidate.add_node() = body[k];
            }
            else
                *candidate.add_node() = graph.node(j);
        }
        eliminate_sequences_in_graph(candidate, opset);
        // Commit only when every producer in the connected region is gone.
        for (std::set<std::string>::const_iterator it = sequences.begin(); it != sequences.end(); ++it)
        {
            if (is_value_captured(candidate, *it))
                safe = false;
            for (int j = 0; j < candidate.node_size(); j++)
            {
                if (std::find(candidate.node(j).output().begin(), candidate.node(j).output().end(), *it) != candidate.node(j).output().end())
                    safe = false;
            }
        }
        if (!safe)
            continue;
        graph.Swap(&candidate);
        call_index = -1;
    }
}

void eliminate_sequences(onnx::ModelProto& model)
{
    const char* disable = std::getenv("PNNX_ONNX_DISABLE_SEQUENCE_ELIMINATION");
    if (disable && std::string(disable) == "1")
    {
        return;
    }
    int opset = 0;
    for (int i = 0; i < model.opset_import_size(); i++)
    {
        if (model.opset_import(i).domain().empty() || model.opset_import(i).domain() == "ai.onnx")
        {
            opset = model.opset_import(i).version();
        }
    }
    if (opset < 11)
    {
        return;
    }
    inline_sequence_functions(model, opset);
    eliminate_sequences_in_graph(*model.mutable_graph(), opset);
}

} // namespace onnx2pnnx

} // namespace pnnx
