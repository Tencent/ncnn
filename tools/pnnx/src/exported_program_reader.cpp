// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program.h"

#include <limits>

#include "json.h"
#include "model_format.h"
#include "storezip.h"

namespace pnnx {
namespace pt2 {

class ExportedProgramDecoder
{
public:
    ExportedProgramDecoder(std::string& error)
        : error(error)
    {
    }

    bool decode(const JsonValue& root, ExportedProgram& program)
    {
        const JsonValue* graph_module = member(root, "graph_module", "exported_program");
        const JsonValue* opset_version = member(root, "opset_version", "exported_program");
        const JsonValue* range_constraints = member(root, "range_constraints", "exported_program");
        const JsonValue* schema_version = member(root, "schema_version", "exported_program");
        if (!graph_module || !opset_version || !range_constraints || !schema_version)
            return false;

        if (!decode_graph_module(*graph_module, program) || !decode_int_map(*opset_version, program.opset_version, "opset_version") || !decode_ranges(*range_constraints, program.range_constraints) || !decode_schema_version(*schema_version, program.schema_version))
            return false;

        const JsonValue* torch_version = root.get("torch_version");
        if (torch_version && !get_string(*torch_version, program.torch_version, "torch_version"))
            return false;

        if (program.schema_version.major != 8)
            return fail("schema_version", "unsupported exported program schema major");

        return true;
    }

    bool decode_payload_config(const JsonValue& root, std::map<std::string, PayloadMeta>& payloads, const std::string& path)
    {
        const JsonValue* config = member(root, "config", path);
        if (!config)
            return false;

        const std::map<std::string, JsonValue>* object = config->get_object();
        if (!object)
            return fail(path + ".config", "expected object");

        for (std::map<std::string, JsonValue>::const_iterator it = object->begin(); it != object->end(); ++it)
        {
            const std::string item_path = path + ".config." + it->first;
            const JsonValue* path_name = member(it->second, "path_name", item_path);
            const JsonValue* is_parameter = member(it->second, "is_param", item_path);
            const JsonValue* use_pickle = member(it->second, "use_pickle", item_path);
            const JsonValue* tensor_meta = member(it->second, "tensor_meta", item_path);
            if (!path_name || !is_parameter || !use_pickle || !tensor_meta)
                return false;

            PayloadMeta payload;
            if (!get_string(*path_name, payload.path, item_path + ".path_name")
                || !get_bool(*is_parameter, payload.is_parameter, item_path + ".is_param")
                || !get_bool(*use_pickle, payload.use_pickle, item_path + ".use_pickle"))
                return false;

            if (!tensor_meta->is_null())
            {
                if (!decode_tensor_meta(*tensor_meta, payload.tensor_meta, item_path + ".tensor_meta"))
                    return false;
                payload.has_tensor_meta = true;
            }
            payloads[it->first] = payload;
        }
        return true;
    }

private:
    const JsonValue* member(const JsonValue& value, const char* name, const std::string& path)
    {
        if (!value.get_object())
        {
            fail(path, "expected object");
            return 0;
        }

        const JsonValue* result = value.get(name);
        if (!result)
            fail(path + "." + name, "missing required field");
        return result;
    }

    bool get_string(const JsonValue& value, std::string& result, const std::string& path)
    {
        const std::string* string = value.get_string();
        if (!string)
            return fail(path, "expected string");
        result = *string;
        return true;
    }

    bool get_int(const JsonValue& value, int64_t& result, const std::string& path)
    {
        if (!value.get_int(result))
            return fail(path, "expected integer");
        return true;
    }

    bool get_bool(const JsonValue& value, bool& result, const std::string& path)
    {
        if (!value.get_bool(result))
            return fail(path, "expected boolean");
        return true;
    }

    bool get_enum(const JsonValue& value, int& result, const std::string& path)
    {
        int64_t integer = 0;
        if (!get_int(value, integer, path) || integer < 0 || integer > INT_MAX)
            return error.empty() ? fail(path, "enum value is out of range") : false;
        result = (int)integer;
        return true;
    }

    bool decode_graph_module(const JsonValue& value, ExportedProgram& program)
    {
        const JsonValue* graph = member(value, "graph", "graph_module");
        const JsonValue* signature = member(value, "signature", "graph_module");
        return graph && signature && decode_graph(*graph, program.graph) && decode_signature(*signature, program.signature);
    }

    bool decode_graph(const JsonValue& value, Graph& graph)
    {
        const JsonValue* inputs = member(value, "inputs", "graph");
        const JsonValue* outputs = member(value, "outputs", "graph");
        const JsonValue* nodes = member(value, "nodes", "graph");
        const JsonValue* tensor_values = member(value, "tensor_values", "graph");
        const JsonValue* sym_int_values = member(value, "sym_int_values", "graph");
        if (!inputs || !outputs || !nodes || !tensor_values || !sym_int_values)
            return false;

        if (!decode_arguments(*inputs, graph.inputs, "graph.inputs") || !decode_arguments(*outputs, graph.outputs, "graph.outputs") || !decode_nodes(*nodes, graph.nodes) || !decode_tensor_map(*tensor_values, graph.tensor_values) || !decode_sym_int_map(*sym_int_values, graph.sym_int_values))
            return false;

        const JsonValue* single_return = value.get("is_single_tensor_return");
        if (single_return && !get_bool(*single_return, graph.is_single_tensor_return, "graph.is_single_tensor_return"))
            return false;
        return true;
    }

    bool decode_nodes(const JsonValue& value, std::vector<Node>& nodes)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail("graph.nodes", "expected array");

        for (size_t i = 0; i < array->size(); i++)
        {
            const std::string path = "graph.nodes[" + std::to_string(i) + "]";
            const JsonValue* target = member((*array)[i], "target", path);
            const JsonValue* inputs = member((*array)[i], "inputs", path);
            const JsonValue* outputs = member((*array)[i], "outputs", path);
            const JsonValue* metadata = member((*array)[i], "metadata", path);
            if (!target || !inputs || !outputs || !metadata)
                return false;

            Node node;
            if (!get_string(*target, node.target, path + ".target") || !decode_named_arguments(*inputs, node.inputs, path + ".inputs") || !decode_arguments(*outputs, node.outputs, path + ".outputs") || !decode_string_map(*metadata, node.metadata, path + ".metadata"))
                return false;

            const JsonValue* name = (*array)[i].get("name");
            if (name && !name->is_null() && !get_string(*name, node.name, path + ".name"))
                return false;
            nodes.push_back(node);
        }
        return true;
    }

    bool decode_named_arguments(const JsonValue& value, std::vector<NamedArgument>& arguments, const std::string& path)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail(path, "expected array");

        for (size_t i = 0; i < array->size(); i++)
        {
            const std::string item_path = path + "[" + std::to_string(i) + "]";
            const JsonValue* name = member((*array)[i], "name", item_path);
            const JsonValue* argument = member((*array)[i], "arg", item_path);
            if (!name || !argument)
                return false;

            NamedArgument result;
            if (!get_string(*name, result.name, item_path + ".name") || !decode_argument(*argument, result.argument, item_path + ".arg"))
                return false;

            const JsonValue* kind = (*array)[i].get("kind");
            if (kind && !kind->is_null())
            {
                int enum_value = 0;
                if (!get_enum(*kind, enum_value, item_path + ".kind") || enum_value > 2)
                    return error.empty() ? fail(item_path + ".kind", "unknown argument kind") : false;
                result.kind = (NamedArgument::Kind)enum_value;
            }
            arguments.push_back(result);
        }
        return true;
    }

    bool decode_arguments(const JsonValue& value, std::vector<Argument>& arguments, const std::string& path)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail(path, "expected array");
        for (size_t i = 0; i < array->size(); i++)
        {
            Argument argument;
            if (!decode_argument((*array)[i], argument, path + "[" + std::to_string(i) + "]"))
                return false;
            arguments.push_back(argument);
        }
        return true;
    }

    bool decode_argument(const JsonValue& value, Argument& argument, const std::string& path)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object || object->size() != 1)
            return fail(path, "expected single argument variant");

        const std::string& type = object->begin()->first;
        const JsonValue& data = object->begin()->second;
        if (type == "as_none")
        {
            argument.type = Argument::None;
            return true;
        }
        if (type == "as_tensor")
        {
            argument.type = Argument::Tensor;
            return decode_named_reference(data, argument.name, path + ".as_tensor");
        }
        if (type == "as_tensors")
        {
            argument.type = Argument::Tensors;
            return decode_reference_list(data, argument.values, path + ".as_tensors");
        }
        if (type == "as_int")
        {
            argument.type = Argument::Integer;
            return get_int(data, argument.integer, path + ".as_int");
        }
        if (type == "as_float")
        {
            argument.type = Argument::FloatingPoint;
            return get_number(data, argument.floating_point, path + ".as_float");
        }
        if (type == "as_bool")
        {
            argument.type = Argument::Boolean;
            return get_bool(data, argument.boolean, path + ".as_bool");
        }
        if (type == "as_string")
        {
            argument.type = Argument::String;
            return get_string(data, argument.string, path + ".as_string");
        }
        if (type == "as_ints" || type == "as_floats" || type == "as_bools" || type == "as_strings")
            return decode_scalar_list(type, data, argument, path + "." + type);
        if (type == "as_sym_int")
        {
            argument.type = Argument::SymInteger;
            return decode_sym_argument(data, argument, path + ".as_sym_int");
        }
        if (type == "as_scalar_type" || type == "as_memory_format" || type == "as_layout")
        {
            argument.type = type == "as_scalar_type" ? Argument::ScalarType : type == "as_memory_format" ? Argument::MemoryFormat
                                                                                                         : Argument::Layout;
            return get_int(data, argument.integer, path + "." + type);
        }
        if (type == "as_device")
        {
            argument.type = Argument::DeviceValue;
            return decode_device(data, argument.device, path + ".as_device");
        }

        return fail(path, "unsupported argument variant " + type);
    }

    bool get_number(const JsonValue& value, double& result, const std::string& path)
    {
        if (!value.get_number(result))
            return fail(path, "expected number");
        return true;
    }

    bool decode_scalar_list(const std::string& type, const JsonValue& value, Argument& argument, const std::string& path)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail(path, "expected array");

        argument.type = type == "as_ints" ? Argument::Integers : type == "as_floats" ? Argument::FloatingPoints
                                                             : type == "as_bools"    ? Argument::Booleans
                                                                                     : Argument::Strings;
        for (size_t i = 0; i < array->size(); i++)
        {
            Argument item;
            const std::string item_path = path + "[" + std::to_string(i) + "]";
            if (type == "as_ints")
            {
                item.type = Argument::Integer;
                if (!get_int((*array)[i], item.integer, item_path)) return false;
            }
            else if (type == "as_floats")
            {
                item.type = Argument::FloatingPoint;
                if (!get_number((*array)[i], item.floating_point, item_path)) return false;
            }
            else if (type == "as_bools")
            {
                item.type = Argument::Boolean;
                if (!get_bool((*array)[i], item.boolean, item_path)) return false;
            }
            else
            {
                item.type = Argument::String;
                if (!get_string((*array)[i], item.string, item_path)) return false;
            }
            argument.values.push_back(item);
        }
        return true;
    }

    bool decode_named_reference(const JsonValue& value, std::string& name, const std::string& path)
    {
        const JsonValue* name_value = member(value, "name", path);
        return name_value && get_string(*name_value, name, path + ".name");
    }

    bool decode_reference_list(const JsonValue& value, std::vector<Argument>& arguments, const std::string& path)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail(path, "expected array");
        for (size_t i = 0; i < array->size(); i++)
        {
            Argument argument;
            argument.type = Argument::Tensor;
            if (!decode_named_reference((*array)[i], argument.name, path + "[" + std::to_string(i) + "]"))
                return false;
            arguments.push_back(argument);
        }
        return true;
    }

    bool decode_sym_argument(const JsonValue& value, Argument& argument, const std::string& path)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object || object->size() != 1)
            return fail(path, "expected single symbolic integer variant");
        if (object->begin()->first == "as_name")
            return get_string(object->begin()->second, argument.name, path + ".as_name");
        if (object->begin()->first == "as_int")
            return get_int(object->begin()->second, argument.integer, path + ".as_int");
        return fail(path, "unsupported symbolic integer variant");
    }

    bool decode_tensor_map(const JsonValue& value, std::map<std::string, TensorMeta>& result)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object)
            return fail("graph.tensor_values", "expected object");
        for (std::map<std::string, JsonValue>::const_iterator it = object->begin(); it != object->end(); ++it)
        {
            TensorMeta meta;
            if (!decode_tensor_meta(it->second, meta, "graph.tensor_values." + it->first))
                return false;
            result[it->first] = meta;
        }
        return true;
    }

    bool decode_tensor_meta(const JsonValue& value, TensorMeta& meta, const std::string& path)
    {
        const JsonValue* dtype = member(value, "dtype", path);
        const JsonValue* sizes = member(value, "sizes", path);
        const JsonValue* requires_grad = member(value, "requires_grad", path);
        const JsonValue* device = member(value, "device", path);
        const JsonValue* strides = member(value, "strides", path);
        const JsonValue* storage_offset = member(value, "storage_offset", path);
        const JsonValue* layout = member(value, "layout", path);
        return dtype && sizes && requires_grad && device && strides && storage_offset && layout
               && get_enum(*dtype, meta.scalar_type, path + ".dtype")
               && decode_sym_int_list(*sizes, meta.sizes, path + ".sizes")
               && get_bool(*requires_grad, meta.requires_grad, path + ".requires_grad")
               && decode_device(*device, meta.device, path + ".device")
               && decode_sym_int_list(*strides, meta.strides, path + ".strides")
               && decode_sym_int(*storage_offset, meta.storage_offset, path + ".storage_offset")
               && get_enum(*layout, meta.layout, path + ".layout");
    }

    bool decode_device(const JsonValue& value, Device& device, const std::string& path)
    {
        const JsonValue* type = member(value, "type", path);
        if (!type || !get_string(*type, device.type, path + ".type"))
            return false;
        const JsonValue* index = value.get("index");
        if (index && !index->is_null())
        {
            int64_t integer = 0;
            if (!get_int(*index, integer, path + ".index") || integer < 0 || integer > INT_MAX)
                return error.empty() ? fail(path + ".index", "device index is out of range") : false;
            device.has_index = true;
            device.index = (int)integer;
        }
        return true;
    }

    bool decode_sym_int_list(const JsonValue& value, std::vector<SymInt>& result, const std::string& path)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail(path, "expected array");
        for (size_t i = 0; i < array->size(); i++)
        {
            SymInt item;
            if (!decode_sym_int((*array)[i], item, path + "[" + std::to_string(i) + "]"))
                return false;
            result.push_back(item);
        }
        return true;
    }

    bool decode_sym_int(const JsonValue& value, SymInt& result, const std::string& path)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object || object->size() != 1)
            return fail(path, "expected single symbolic integer variant");
        if (object->begin()->first == "as_int")
            return get_int(object->begin()->second, result.integer, path + ".as_int");
        if (object->begin()->first != "as_expr")
            return fail(path, "unsupported symbolic integer variant");

        result.type = SymInt::Expression;
        const JsonValue* expression = member(object->begin()->second, "expr_str", path + ".as_expr");
        if (!expression || !get_string(*expression, result.expression, path + ".as_expr.expr_str"))
            return false;
        const JsonValue* hint = object->begin()->second.get("hint");
        if (hint && !hint->is_null())
        {
            const JsonValue* integer = member(*hint, "as_int", path + ".as_expr.hint");
            if (!integer || !get_int(*integer, result.hint, path + ".as_expr.hint.as_int"))
                return false;
            result.has_hint = true;
        }
        return true;
    }

    bool decode_sym_int_map(const JsonValue& value, std::map<std::string, SymInt>& result)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object)
            return fail("graph.sym_int_values", "expected object");
        for (std::map<std::string, JsonValue>::const_iterator it = object->begin(); it != object->end(); ++it)
        {
            SymInt sym_int;
            if (!decode_sym_int(it->second, sym_int, "graph.sym_int_values." + it->first))
                return false;
            result[it->first] = sym_int;
        }
        return true;
    }

    bool decode_signature(const JsonValue& value, GraphSignature& signature)
    {
        const JsonValue* inputs = member(value, "input_specs", "graph_signature");
        const JsonValue* outputs = member(value, "output_specs", "graph_signature");
        if (!inputs || !outputs)
            return false;
        return decode_input_specs(*inputs, signature.inputs) && decode_output_specs(*outputs, signature.outputs);
    }

    bool decode_input_specs(const JsonValue& value, std::vector<InputSpec>& result)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail("graph_signature.input_specs", "expected array");
        for (size_t i = 0; i < array->size(); i++)
        {
            const std::string path = "graph_signature.input_specs[" + std::to_string(i) + "]";
            const std::map<std::string, JsonValue>* variant = (*array)[i].get_object();
            if (!variant || variant->size() != 1)
                return fail(path, "expected single input spec variant");

            InputSpec spec;
            const std::string& type = variant->begin()->first;
            const JsonValue& data = variant->begin()->second;
            const JsonValue* arg = member(data, "arg", path + "." + type);
            if (!arg)
                return false;

            if (type == "user_input")
            {
                spec.type = InputSpec::UserInput;
                if (!decode_argument(*arg, spec.argument, path + ".user_input.arg")) return false;
            }
            else if (type == "parameter" || type == "buffer" || type == "tensor_constant")
            {
                spec.type = type == "parameter" ? InputSpec::Parameter : type == "buffer" ? InputSpec::Buffer
                                                                                          : InputSpec::TensorConstant;
                spec.argument.type = Argument::Tensor;
                if (!decode_named_reference(*arg, spec.argument.name, path + "." + type + ".arg")) return false;
                const char* target_name = type == "parameter" ? "parameter_name" : type == "buffer" ? "buffer_name"
                                                                                                    : "tensor_constant_name";
                const JsonValue* target = member(data, target_name, path + "." + type);
                if (!target || !get_string(*target, spec.target, path + "." + type + "." + target_name)) return false;
                if (type == "buffer")
                {
                    const JsonValue* persistent = member(data, "persistent", path + ".buffer");
                    if (!persistent || !get_bool(*persistent, spec.persistent, path + ".buffer.persistent")) return false;
                }
            }
            else
            {
                return fail(path, "unsupported input spec " + type);
            }
            result.push_back(spec);
        }
        return true;
    }

    bool decode_output_specs(const JsonValue& value, std::vector<OutputSpec>& result)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail("graph_signature.output_specs", "expected array");
        for (size_t i = 0; i < array->size(); i++)
        {
            const std::string path = "graph_signature.output_specs[" + std::to_string(i) + "]";
            const std::map<std::string, JsonValue>* variant = (*array)[i].get_object();
            if (!variant || variant->size() != 1 || variant->begin()->first != "user_output")
                return fail(path, "unsupported output spec");
            const JsonValue* arg = member(variant->begin()->second, "arg", path + ".user_output");
            OutputSpec spec;
            if (!arg || !decode_argument(*arg, spec.argument, path + ".user_output.arg"))
                return false;
            result.push_back(spec);
        }
        return true;
    }

    bool decode_int_map(const JsonValue& value, std::map<std::string, int>& result, const std::string& path)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object)
            return fail(path, "expected object");
        for (std::map<std::string, JsonValue>::const_iterator it = object->begin(); it != object->end(); ++it)
        {
            if (!get_enum(it->second, result[it->first], path + "." + it->first))
                return false;
        }
        return true;
    }

    bool decode_string_map(const JsonValue& value, std::map<std::string, std::string>& result, const std::string& path)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object)
            return fail(path, "expected object");
        for (std::map<std::string, JsonValue>::const_iterator it = object->begin(); it != object->end(); ++it)
        {
            if (!get_string(it->second, result[it->first], path + "." + it->first))
                return false;
        }
        return true;
    }

    bool decode_ranges(const JsonValue& value, std::map<std::string, RangeConstraint>& result)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object)
            return fail("range_constraints", "expected object");
        for (std::map<std::string, JsonValue>::const_iterator it = object->begin(); it != object->end(); ++it)
        {
            RangeConstraint range;
            const JsonValue* min = member(it->second, "min_val", "range_constraints." + it->first);
            const JsonValue* max = member(it->second, "max_val", "range_constraints." + it->first);
            if (!min || !max)
                return false;
            if (!min->is_null())
            {
                if (!get_int(*min, range.min, "range_constraints." + it->first + ".min_val")) return false;
                range.has_min = true;
            }
            if (!max->is_null())
            {
                if (!get_int(*max, range.max, "range_constraints." + it->first + ".max_val")) return false;
                range.has_max = true;
            }
            result[it->first] = range;
        }
        return true;
    }

    bool decode_schema_version(const JsonValue& value, SchemaVersion& result)
    {
        const JsonValue* major = member(value, "major", "schema_version");
        const JsonValue* minor = member(value, "minor", "schema_version");
        int64_t major_value = 0;
        int64_t minor_value = 0;
        if (!major || !minor || !get_int(*major, major_value, "schema_version.major") || !get_int(*minor, minor_value, "schema_version.minor") || major_value < 0 || major_value > INT_MAX || minor_value < 0 || minor_value > INT_MAX)
            return error.empty() ? fail("schema_version", "version is out of range") : false;
        result.major = (int)major_value;
        result.minor = (int)minor_value;
        return true;
    }

    bool fail(const std::string& path, const std::string& message)
    {
        if (error.empty())
            error = path + ": " + message;
        return false;
    }

private:
    std::string& error;
};

static std::string common_archive_root(const std::vector<std::string>& names)
{
    if (names.empty()) return std::string();
    const size_t slash = names[0].find('/');
    if (slash == std::string::npos) return std::string();
    const std::string root = names[0].substr(0, slash + 1);
    for (size_t i = 1; i < names.size(); i++)
        if (names[i].compare(0, root.size(), root) != 0) return std::string();
    return root;
}

static bool read_record(StoreZipReader& reader, const std::string& name, std::string& data, std::string& error)
{
    const uint64_t size = reader.get_file_size(name);
    if (size > std::numeric_limits<size_t>::max() || size > 512ull * 1024 * 1024)
    {
        error = name + ": archive record exceeds size limit";
        return false;
    }
    data.resize((size_t)size);
    if (size && reader.read_file(name, &data[0]) != 0)
    {
        error = name + ": failed to read archive record";
        return false;
    }
    return true;
}

static bool parse_payload_config(const std::string& text, std::map<std::string, PayloadMeta>& payloads, const std::string& path, std::string& error)
{
    JsonValue root;
    if (!parse_json(text, root, error))
        return false;
    ExportedProgramDecoder decoder(error);
    return decoder.decode_payload_config(root, payloads, path);
}

static size_t scalar_type_size(int scalar_type)
{
    if (scalar_type == 1 || scalar_type == 2 || scalar_type == 12 || (scalar_type >= 29 && scalar_type <= 33)) return 1;
    if (scalar_type == 3 || scalar_type == 6 || scalar_type == 13 || scalar_type == 28) return 2;
    if (scalar_type == 4 || scalar_type == 7 || scalar_type == 9 || scalar_type == 34) return 4;
    if (scalar_type == 5 || scalar_type == 8 || scalar_type == 10 || scalar_type == 35) return 8;
    if (scalar_type == 11) return 16;
    return 0;
}

static bool concrete_nonnegative(const SymInt& value, uint64_t& result)
{
    if (value.type != SymInt::Integer || value.integer < 0)
        return false;
    result = (uint64_t)value.integer;
    return true;
}

static bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
        return false;
    result = lhs + rhs;
    return true;
}

static bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        return false;
    result = lhs * rhs;
    return true;
}

static bool validate_tensor_storage(const PayloadMeta& payload, uint64_t storage_size, const std::string& name, std::string& error)
{
    if (payload.use_pickle)
    {
        error = name + ": pickled tensor payload is not supported";
        return false;
    }
    if (!payload.has_tensor_meta)
    {
        error = name + ": tensor metadata is missing";
        return false;
    }

    const TensorMeta& meta = payload.tensor_meta;
    if (meta.sizes.size() != meta.strides.size())
    {
        error = name + ": tensor size and stride rank mismatch";
        return false;
    }

    const size_t element_size = scalar_type_size(meta.scalar_type);
    if (element_size == 0)
    {
        error = name + ": unsupported tensor scalar type " + std::to_string(meta.scalar_type);
        return false;
    }

    uint64_t storage_offset = 0;
    if (!concrete_nonnegative(meta.storage_offset, storage_offset))
    {
        error = name + ": storage offset must be a nonnegative integer";
        return false;
    }

    bool empty = false;
    uint64_t maximum_element = storage_offset;
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        uint64_t size = 0;
        uint64_t stride = 0;
        if (!concrete_nonnegative(meta.sizes[i], size) || !concrete_nonnegative(meta.strides[i], stride))
        {
            error = name + ": tensor sizes and strides must be nonnegative integers";
            return false;
        }
        if (size == 0)
        {
            empty = true;
            continue;
        }

        uint64_t extent = 0;
        if (!checked_multiply(size - 1, stride, extent) || !checked_add(maximum_element, extent, maximum_element))
        {
            error = name + ": tensor storage range overflows uint64";
            return false;
        }
    }

    if (empty)
        return true;

    uint64_t required_elements = 0;
    uint64_t required_bytes = 0;
    if (!checked_add(maximum_element, 1, required_elements) || !checked_multiply(required_elements, element_size, required_bytes))
    {
        error = name + ": tensor storage size overflows uint64";
        return false;
    }
    if (required_bytes > storage_size)
    {
        error = name + ": tensor view exceeds storage payload";
        return false;
    }
    return true;
}

static bool load_payloads(StoreZipReader& reader, const std::string& root, const std::string& directory, const std::string& model_name, const std::string& config_suffix, std::map<std::string, PayloadMeta>& payloads, std::map<std::string, std::vector<char> >& storages, std::string& error)
{
    const std::string logical_config = directory + model_name + config_suffix;
    const std::string config_name = root + logical_config;
    const std::vector<std::string> names = reader.get_names();
    if (std::find(names.begin(), names.end(), config_name) == names.end())
    {
        error = logical_config + ": missing payload config";
        return false;
    }

    std::string config;
    if (!read_record(reader, config_name, config, error) || !parse_payload_config(config, payloads, logical_config, error))
        return false;

    for (std::map<std::string, PayloadMeta>::const_iterator it = payloads.begin(); it != payloads.end(); ++it)
    {
        const PayloadMeta& payload = it->second;
        if (payload.path.empty() || payload.path.find('/') != std::string::npos || payload.path.find('\\') != std::string::npos)
        {
            error = it->first + ": invalid payload path";
            return false;
        }

        const std::string logical_storage = directory + payload.path;
        const std::string storage_name = root + logical_storage;
        if (storages.find(logical_storage) == storages.end())
        {
            if (std::find(names.begin(), names.end(), storage_name) == names.end())
            {
                error = logical_storage + ": missing tensor payload";
                return false;
            }

            const uint64_t size = reader.get_file_size(storage_name);
            if (size > std::numeric_limits<size_t>::max())
            {
                error = logical_storage + ": tensor payload is too large";
                return false;
            }
            std::vector<char>& storage = storages[logical_storage];
            storage.resize((size_t)size);
            if (size && reader.read_file(storage_name, storage.data()) != 0)
            {
                error = logical_storage + ": failed to read tensor payload";
                return false;
            }
        }

        if (!validate_tensor_storage(payload, storages[logical_storage].size(), it->first, error))
            return false;
    }
    return true;
}

bool parse_exported_program(const std::string& text, ExportedProgram& program, std::string& error)
{
    error.clear();
    program = ExportedProgram();
    JsonValue root;
    if (!parse_json(text, root, error))
        return false;
    ExportedProgramDecoder decoder(error);
    return decoder.decode(root, program);
}

bool load_exported_program_archive_metadata(const std::string& path, ExportedProgramArchive& archive, std::string& error)
{
    error.clear();
    archive = ExportedProgramArchive();
    if (detect_model_format(path, error) != ModelFormatExportedProgram)
        return false;

    StoreZipReader reader;
    if (reader.open(path) != 0)
    {
        error = "failed to read pt2 archive";
        return false;
    }

    const std::vector<std::string> names = reader.get_names();
    const std::string root = common_archive_root(names);
    std::vector<std::string> models;
    for (size_t i = 0; i < names.size(); i++)
    {
        const std::string logical_name = root.empty() ? names[i] : names[i].substr(root.size());
        if (logical_name.size() > 12 && logical_name.compare(0, 7, "models/") == 0 && logical_name.compare(logical_name.size() - 5, 5, ".json") == 0)
            models.push_back(names[i]);
    }

    if (models.size() != 1)
    {
        error = "pt2 archive must contain exactly one exported program";
        return false;
    }

    const std::string logical_model = root.empty() ? models[0] : models[0].substr(root.size());
    archive.model_name = logical_model.substr(7, logical_model.size() - 12);
    archive.archive_version = 0;

    std::string document;
    if (!read_record(reader, models[0], document, error))
        return false;
    return parse_exported_program(document, archive.program, error);
}

bool load_exported_program_archive(const std::string& path, ExportedProgramArchive& archive, std::string& error)
{
    if (!load_exported_program_archive_metadata(path, archive, error))
        return false;

    StoreZipReader reader;
    if (reader.open(path) != 0)
    {
        error = "failed to read pt2 archive payloads";
        return false;
    }

    const std::string root = common_archive_root(reader.get_names());
    if (!load_payloads(reader, root, "data/weights/", archive.model_name, "_weights_config.json", archive.state_dict, archive.state_dict_storages, error))
        return false;
    if (!load_payloads(reader, root, "data/constants/", archive.model_name, "_constants_config.json", archive.constants, archive.constant_storages, error))
        return false;
    return true;
}

} // namespace pt2
} // namespace pnnx