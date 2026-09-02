// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_defaults.h"

#include <ATen/core/dispatch/Dispatcher.h>

namespace pnnx {
namespace pt2 {

static bool from_ivalue(const c10::IValue& value, Argument& argument)
{
    if (value.isNone())
    {
        argument.type = Argument::None;
        return true;
    }
    if (value.isBool())
    {
        argument.type = Argument::Boolean;
        argument.boolean = value.toBool();
        return true;
    }
    if (value.isInt())
    {
        argument.type = Argument::Integer;
        argument.integer = value.toInt();
        return true;
    }
    if (value.isDouble())
    {
        argument.type = Argument::FloatingPoint;
        argument.floating_point = value.toDouble();
        return true;
    }
    if (value.isString())
    {
        argument.type = Argument::String;
        argument.string = value.toStringRef();
        return true;
    }
    if (value.isIntList())
    {
        argument.type = Argument::Integers;
        const c10::List<int64_t> values = value.toIntList();
        for (size_t i = 0; i < values.size(); i++)
        {
            Argument item;
            item.type = Argument::Integer;
            item.integer = values.get(i);
            argument.values.push_back(item);
        }
        return true;
    }
    if (value.isDoubleList())
    {
        argument.type = Argument::FloatingPoints;
        const c10::List<double> values = value.toDoubleList();
        for (size_t i = 0; i < values.size(); i++)
        {
            Argument item;
            item.type = Argument::FloatingPoint;
            item.floating_point = values.get(i);
            argument.values.push_back(item);
        }
        return true;
    }
    if (value.isBoolList())
    {
        argument.type = Argument::Booleans;
        const c10::List<bool> values = value.toBoolList();
        for (size_t i = 0; i < values.size(); i++)
        {
            Argument item;
            item.type = Argument::Boolean;
            item.boolean = values.get(i);
            argument.values.push_back(item);
        }
        return true;
    }
    return false;
}

static bool parse_target(const std::string& target, std::string& name, std::string& overload)
{
    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return false;
    const size_t namespace_end = target.find('.', prefix.size());
    const size_t operator_end = namespace_end == std::string::npos ? std::string::npos : target.find('.', namespace_end + 1);
    if (namespace_end == std::string::npos || operator_end == std::string::npos)
        return false;
    name = target.substr(prefix.size(), namespace_end - prefix.size()) + "::" + target.substr(namespace_end + 1, operator_end - namespace_end - 1);
    overload = target.substr(operator_end + 1);
    if (overload == "default")
        overload.clear();
    return true;
}

bool append_default_arguments(ExportedProgram& program, std::string& error)
{
    error.clear();
    for (size_t node_index = 0; node_index < program.graph.nodes.size(); node_index++)
    {
        Node& node = program.graph.nodes[node_index];
        std::string name;
        std::string overload;
        if (!parse_target(node.target, name, overload))
            continue;

        c10::optional<c10::OperatorHandle> handle = c10::Dispatcher::singleton().findSchema({name, overload});
        if (!handle.has_value())
        {
            error = node.name + ": operator schema was not found for " + node.target;
            return false;
        }

        const std::vector<c10::Argument>& schema_arguments = handle->schema().arguments();
        std::map<std::string, size_t> present;
        for (size_t i = 0; i < node.inputs.size(); i++)
        {
            if (present.find(node.inputs[i].name) != present.end())
            {
                error = node.name + ": duplicate argument " + node.inputs[i].name;
                return false;
            }
            present[node.inputs[i].name] = i;
        }

        std::vector<NamedArgument> ordered_inputs;
        ordered_inputs.reserve(schema_arguments.size());
        size_t matched_input_count = 0;
        for (size_t i = 0; i < schema_arguments.size(); i++)
        {
            const c10::Argument& schema_argument = schema_arguments[i];
            std::map<std::string, size_t>::const_iterator it = present.find(schema_argument.name());
            if (it != present.end())
            {
                ordered_inputs.push_back(node.inputs[it->second]);
                matched_input_count++;
                continue;
            }
            if (!schema_argument.default_value().has_value())
            {
                error = node.name + ": required argument " + schema_argument.name() + " is missing";
                return false;
            }

            NamedArgument argument;
            argument.name = schema_argument.name();
            argument.kind = schema_argument.kwarg_only() ? NamedArgument::Keyword : NamedArgument::Positional;
            if (!from_ivalue(*schema_argument.default_value(), argument.argument))
            {
                error = node.name + ": unsupported default value for argument " + schema_argument.name();
                return false;
            }
            ordered_inputs.push_back(argument);
        }
        if (matched_input_count != node.inputs.size())
        {
            error = node.name + ": argument was not found in operator schema";
            return false;
        }
        node.inputs.swap(ordered_inputs);
    }
    return true;
}

} // namespace pt2
} // namespace pnnx