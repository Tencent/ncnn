// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#if PNNX_TORCH_HAS_PICKLE_LOAD
#include <torch/serialize.h>
#endif

#include "json.h"
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

        return validate_exported_program_version(program, error);
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
        return graph && signature && decode_graph(*graph, program.graph) && decode_signature(*signature, program.signature)
               && validate_call_graph(value, program.signature);
    }

    bool tree_leaf(const JsonValue& value) const
    {
        const JsonValue* type = value.get("type");
        const JsonValue* context = value.get("context");
        const JsonValue* children = value.get("children_spec");
        return type && type->is_null() && context && context->is_null()
               && children && children->get_array() && children->get_array()->empty();
    }

    const std::vector<JsonValue>* tree_children(const JsonValue& value, const char* expected_type, bool empty_dict = false) const
    {
        const JsonValue* type = value.get("type");
        const JsonValue* context = value.get("context");
        const JsonValue* children = value.get("children_spec");
        if (!type || !type->get_string() || *type->get_string() != expected_type
                || !context || !context->get_string() || !children || !children->get_array())
            return 0;

        // Protocol 1 encodes tuple/dict context as a JSON STRING, unlike a
        // leaf's literal null context. Parse it rather than guessing spellings.
        JsonValue decoded_context;
        std::string context_error;
        if (!parse_json(*context->get_string(), decoded_context, context_error))
            return 0;
        if (empty_dict ? (!decoded_context.get_array() || !decoded_context.get_array()->empty()) : !decoded_context.is_null())
            return 0;
        return children->get_array();
    }

    bool parse_tree_spec(const JsonValue& signature, const char* name, JsonValue& document, const std::string& path)
    {
        const JsonValue* value = member(signature, name, path);
        if (!value)
            return false;
        if (!value->get_string())
            return fail(path + "." + name, "expected serialized PyTree TreeSpec string");
        std::string parse_error;
        if (!parse_json(*value->get_string(), document, parse_error))
            return fail(path + "." + name, "invalid PyTree TreeSpec JSON: " + parse_error);
        const std::vector<JsonValue>* spec = document.get_array();
        int64_t protocol = 0;
        if (!spec || spec->size() != 2 || !(*spec)[0].get_int(protocol) || protocol != 1)
            return fail(path + "." + name, "unsupported PyTree TreeSpec protocol; expected protocol 1");
        return true;
    }

    bool validate_call_signature(const JsonValue& value, const GraphSignature& signature, const std::string& path)
    {
        JsonValue input_document;
        JsonValue output_document;
        if (!parse_tree_spec(value, "in_spec", input_document, path) || !parse_tree_spec(value, "out_spec", output_document, path))
            return false;

        const JsonValue& input_tree = (*input_document.get_array())[1];
        const std::vector<JsonValue>* call = tree_children(input_tree, "builtins.tuple");
        if (!call || call->size() != 2)
            return fail(path + ".in_spec", "unsupported PyTree input; expected (positional args, empty kwargs)");
        const std::vector<JsonValue>* args = tree_children((*call)[0], "builtins.tuple");
        const std::vector<JsonValue>* kwargs = tree_children((*call)[1], "builtins.dict", true);
        if (!kwargs || !kwargs->empty())
            return fail(path + ".in_spec", "unsupported PyTree kwargs; only flat positional tensor inputs are supported");
        if (!args)
            return fail(path + ".in_spec", "unsupported PyTree input; expected flat positional tensor inputs");
        for (size_t i = 0; i < args->size(); i++)
            if (!tree_leaf((*args)[i]))
                return fail(path + ".in_spec", "unsupported PyTree nested input; dict/list/namedtuple inputs are not supported");

        size_t input_count = 0;
        for (size_t i = 0; i < signature.inputs.size(); i++)
        {
            const InputSpec& input = signature.inputs[i];
            if (input.type == InputSpec::ConstantInput || (input.type == InputSpec::UserInput && input.argument.type != Argument::Tensor))
                return fail(path + ".in_spec", "unsupported PyTree input leaf; only positional tensors are supported");
            if (input.type == InputSpec::UserInput)
                input_count++;
        }
        if (input_count != args->size())
            return fail(path + ".in_spec", "PyTree input leaf count does not match graph signature");

        const JsonValue& output_tree = (*output_document.get_array())[1];
        size_t output_count = 1;
        if (!tree_leaf(output_tree))
        {
            const std::vector<JsonValue>* outputs = tree_children(output_tree, "builtins.tuple");
            if (!outputs || outputs->empty())
                return fail(path + ".out_spec", "unsupported PyTree output; expected a tensor/scalar or a nonempty flat tensor/scalar tuple");
            for (size_t i = 0; i < outputs->size(); i++)
                if (!tree_leaf((*outputs)[i]))
                    return fail(path + ".out_spec", "unsupported PyTree nested output; dict/list/namedtuple outputs are not supported");
            output_count = outputs->size();
        }
        size_t user_outputs = 0;
        for (size_t i = 0; i < signature.outputs.size(); i++)
        {
            if (signature.outputs[i].type != OutputSpec::UserOutput)
                continue; // Mutations/tokens are rejected by the importer.
            const Argument::Type type = signature.outputs[i].argument.type;
            if (type != Argument::Tensor && type != Argument::Integer && type != Argument::FloatingPoint
                    && type != Argument::Boolean && type != Argument::Complex && type != Argument::SymInteger
                    && type != Argument::SymFloat && type != Argument::SymBoolean)
                return fail(path + ".out_spec", "unsupported PyTree output leaf; expected tensor or numeric scalar");
            user_outputs++;
        }
        if (output_count != user_outputs)
            return fail(path + ".out_spec", "PyTree output leaf count does not match graph signature");
        return true;
    }

    bool validate_call_graph(const JsonValue& module, const GraphSignature& signature)
    {
        const JsonValue* value = module.get("module_call_graph");
        // Schema-only fixtures and older metadata callers may omit this field.
        // If present and nonempty, the root calling convention must be known.
        if (!value)
            return true;
        const std::vector<JsonValue>* calls = value->get_array();
        if (!calls)
            return fail("graph_module.module_call_graph", "expected array");
        if (calls->empty())
            return true;
        bool found_root = false;
        for (size_t i = 0; i < calls->size(); i++)
        {
            const std::string path = "graph_module.module_call_graph[" + std::to_string(i) + "]";
            const JsonValue* fqn = member((*calls)[i], "fqn", path);
            if (!fqn || !fqn->get_string())
                return fail(path + ".fqn", "expected string");
            if (!fqn->get_string()->empty())
                continue; // Submodule trees are not the public flattened boundary.
            if (found_root)
                return fail(path, "duplicate root PyTree call signature");
            found_root = true;
            const JsonValue* call_signature = member((*calls)[i], "signature", path);
            if (!call_signature || !call_signature->get_object())
                return fail(path + ".signature", "missing root PyTree call signature");
            if (!validate_call_signature(*call_signature, signature, path + ".signature"))
                return false;
        }
        return found_root || fail("graph_module.module_call_graph", "missing root PyTree call signature");
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
        if (type == "as_optional_tensor")
        {
            argument.type = Argument::OptionalTensor;
            Argument item;
            if (!decode_optional_tensor(data, item, path + ".as_optional_tensor"))
                return false;
            argument.values.push_back(item);
            return true;
        }
        if (type == "as_optional_tensors")
        {
            argument.type = Argument::OptionalTensors;
            const std::vector<JsonValue>* array = data.get_array();
            if (!array)
                return fail(path + ".as_optional_tensors", "expected array");
            for (size_t i = 0; i < array->size(); i++)
            {
                Argument item;
                if (!decode_optional_tensor((*array)[i], item, path + ".as_optional_tensors[" + std::to_string(i) + "]"))
                    return false;
                argument.values.push_back(item);
            }
            return true;
        }
        if (type == "as_int")
        {
            argument.type = Argument::Integer;
            return get_int(data, argument.integer, path + ".as_int");
        }
        if (type == "as_float")
        {
            argument.type = Argument::FloatingPoint;
            if (get_number(data, argument.floating_point, path + ".as_float", false))
                return true;
            const std::string* value = data.get_string();
            if (value && *value == "Infinity")
                argument.floating_point = std::numeric_limits<double>::infinity();
            else if (value && *value == "-Infinity")
                argument.floating_point = -std::numeric_limits<double>::infinity();
            else if (value && *value == "NaN")
                argument.floating_point = std::numeric_limits<double>::quiet_NaN();
            else
                return fail(path + ".as_float", "expected number or non-finite float string");
            return true;
        }
        if (type == "as_bool")
        {
            argument.type = Argument::Boolean;
            return get_bool(data, argument.boolean, path + ".as_bool");
        }
        if (type == "as_complex")
        {
            argument.type = Argument::Complex;
            const JsonValue* real = member(data, "real", path + ".as_complex");
            const JsonValue* imag = member(data, "imag", path + ".as_complex");
            return real && imag
                   && get_number(*real, argument.complex_real, path + ".as_complex.real")
                   && get_number(*imag, argument.complex_imag, path + ".as_complex.imag");
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
        if (type == "as_sym_ints")
        {
            argument.type = Argument::SymIntegers;
            const std::vector<JsonValue>* array = data.get_array();
            if (!array)
                return fail(path + ".as_sym_ints", "expected array");
            for (size_t i = 0; i < array->size(); i++)
            {
                Argument item;
                item.type = Argument::SymInteger;
                if (!decode_sym_argument((*array)[i], item, path + ".as_sym_ints[" + std::to_string(i) + "]"))
                    return false;
                argument.values.push_back(item);
            }
            return true;
        }
        if (type == "as_sym_bool")
        {
            argument.type = Argument::SymBoolean;
            return decode_sym_argument(data, argument, path + ".as_sym_bool");
        }
        if (type == "as_sym_float")
        {
            argument.type = Argument::SymFloat;
            return decode_sym_argument(data, argument, path + ".as_sym_float");
        }
        if (type == "as_scalar_type" || type == "as_memory_format" || type == "as_layout")
        {
            argument.type = type == "as_scalar_type" ? Argument::ScalarType : type == "as_memory_format" ? Argument::MemoryFormat : Argument::Layout;
            return get_int(data, argument.integer, path + "." + type);
        }
        if (type == "as_device")
        {
            argument.type = Argument::DeviceValue;
            return decode_device(data, argument.device, path + ".as_device");
        }

        return fail(path, "unsupported argument variant " + type);
    }

    bool get_number(const JsonValue& value, double& result, const std::string& path, bool report_error = true)
    {
        if (!value.get_number(result))
        {
            if (report_error)
                return fail(path, "expected number");
            return false;
        }
        return true;
    }

    bool decode_scalar_list(const std::string& type, const JsonValue& value, Argument& argument, const std::string& path)
    {
        const std::vector<JsonValue>* array = value.get_array();
        if (!array)
            return fail(path, "expected array");

        argument.type = type == "as_ints" ? Argument::Integers : type == "as_floats" ? Argument::FloatingPoints : type == "as_bools" ? Argument::Booleans : Argument::Strings;
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

    bool decode_optional_tensor(const JsonValue& value, Argument& argument, const std::string& path)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object || object->size() != 1)
            return fail(path, "expected single optional tensor variant");
        if (object->begin()->first == "as_none")
        {
            argument.type = Argument::None;
            return true;
        }
        if (object->begin()->first == "as_tensor")
        {
            argument.type = Argument::Tensor;
            return decode_named_reference(object->begin()->second, argument.name, path + ".as_tensor");
        }
        return fail(path, "unsupported optional tensor variant");
    }

    bool decode_sym_argument(const JsonValue& value, Argument& argument, const std::string& path)
    {
        const std::map<std::string, JsonValue>* object = value.get_object();
        if (!object || object->size() != 1)
            return fail(path, "expected single symbolic argument variant");
        if (object->begin()->first == "as_name")
            return get_string(object->begin()->second, argument.name, path + ".as_name");
        if (argument.type == Argument::SymInteger && object->begin()->first == "as_int")
            return get_int(object->begin()->second, argument.integer, path + ".as_int");
        if (argument.type == Argument::SymBoolean && object->begin()->first == "as_bool")
            return get_bool(object->begin()->second, argument.boolean, path + ".as_bool");
        if (argument.type == Argument::SymFloat && object->begin()->first == "as_float")
        {
            if (get_number(object->begin()->second, argument.floating_point, path + ".as_float", false))
                return true;
            const std::string* concrete = object->begin()->second.get_string();
            if (concrete && *concrete == "Infinity")
                argument.floating_point = std::numeric_limits<double>::infinity();
            else if (concrete && *concrete == "-Infinity")
                argument.floating_point = -std::numeric_limits<double>::infinity();
            else if (concrete && *concrete == "NaN")
                argument.floating_point = std::numeric_limits<double>::quiet_NaN();
            else
                return fail(path + ".as_float", "expected number or non-finite float string");
            return true;
        }
        return fail(path, "unsupported symbolic argument variant");
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
            if (type == "constant_input")
            {
                spec.type = InputSpec::ConstantInput;
                const JsonValue* name = member(data, "name", path + ".constant_input");
                const JsonValue* value = member(data, "value", path + ".constant_input");
                if (!name || !value || !get_string(*name, spec.target, path + ".constant_input.name") || !decode_argument(*value, spec.argument, path + ".constant_input.value"))
                    return false;
                result.push_back(spec);
                continue;
            }

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
                spec.type = type == "parameter" ? InputSpec::Parameter : type == "buffer" ? InputSpec::Buffer : InputSpec::TensorConstant;
                spec.argument.type = Argument::Tensor;
                if (!decode_named_reference(*arg, spec.argument.name, path + "." + type + ".arg")) return false;
                const char* target_name = type == "parameter" ? "parameter_name" : type == "buffer" ? "buffer_name" : "tensor_constant_name";
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

static const uint64_t PAYLOAD_MEMORY_BUDGET = 512ull * 1024 * 1024;

// Consult the validated index only for records we consume. Unused attachments
// may be compressed; no record-sized allocation or decompression happens here.
static bool check_record_readable(const StoreZipReader& reader, const std::string& name, std::string& error)
{
    const int compression = reader.get_file_compression(name);
    if (compression < 0)
    {
        error = name + ": failed to read archive record";
        return false;
    }
    if (compression != 0)
    {
        error = "pt2 archive: unsupported ZIP compression; only STORE records are supported (checked before payload allocation)";
        return false;
    }
    // Descriptor-mode and UTF-8 names are used by real torch exporters.
    const int flags = reader.get_file_flags(name);
    if (flags < 0 || (flags & ~(0x0008 | 0x0800)))
    {
        error = "pt2 archive: unsupported ZIP record flags";
        return false;
    }
    return true;
}

static bool supported_host_byteorder(std::string& error)
{
    const uint32_t value = 0x01020304;
    const unsigned char* bytes = (const unsigned char*)&value;
    if (bytes[0] == 4 && bytes[1] == 3 && bytes[2] == 2 && bytes[3] == 1)
        return true;
    error = "pt2 byteorder: big-endian or unknown host byte order is not supported";
    return false;
}

static bool read_record(StoreZipReader& reader, const std::string& name, std::string& data, uint64_t physical_size, std::string& error)
{
    if (!check_record_readable(reader, name, error))
        return false;
    const uint64_t size = reader.get_file_size(name);
    if (size > std::numeric_limits<size_t>::max() || size > PAYLOAD_MEMORY_BUDGET || size > physical_size)
    {
        error = name + ": archive record exceeds size limit or physical archive size";
        return false;
    }
    data.resize((size_t)size);
    if (reader.read_file(name, size ? &data[0] : 0) != 0)
    {
        error = name + ": failed to read archive record";
        return false;
    }
    return true;
}

static bool read_archive_marker(StoreZipReader& reader, const std::string& name, const char* label, std::string& value, uint64_t physical_size, std::string& error)
{
    if (!check_record_readable(reader, name, error))
        return false;
    const uint64_t size = reader.get_file_size(name);
    if (size == 0 || size > 64 || !read_record(reader, name, value, physical_size, error))
    {
        error = std::string("failed to read pt2 archive ") + label;
        return false;
    }
    return true;
}

static bool read_byteorder(StoreZipReader& reader, const std::vector<std::string>& names, const std::string& root, ExportedProgramArchive& archive, uint64_t physical_size, std::string& error)
{
    // PyTorchStreamWriter writes "byteorder", NOT ".data/byteorder". The
    // inspected raw serde stores native CPU storage bytes, not a config field.
    // Keep unmarked little-endian fixtures/checkpoints compatible; never guess
    // native byte order on a big-endian host (rejected before opening the ZIP).
    archive.byteorder = "little";
    const std::string name = root + "byteorder";
    if (std::find(names.begin(), names.end(), name) != names.end())
    {
        if (!check_record_readable(reader, name, error))
            return false;
        if (reader.get_file_size(name) > 16)
        {
            error = "byteorder: unknown raw tensor byte order";
            return false;
        }
        if (!read_record(reader, name, archive.byteorder, physical_size, error))
            return false;
    }
    if (archive.byteorder != "little")
    {
        error = archive.byteorder == "big" ? "byteorder: big-endian raw tensor payload is not supported"
                : "byteorder: unknown raw tensor byte order";
        return false;
    }
    return true;
}

static bool reserve_payload_memory(uint64_t size, uint64_t& used, std::string& error)
{
    if (size > PAYLOAD_MEMORY_BUDGET - used)
    {
        error = "tensor payloads exceed aggregate 512 MiB storage/materialization memory budget";
        return false;
    }
    used += size;
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

#if PNNX_TORCH_HAS_PICKLE_LOAD
static int serialized_scalar_type(c10::ScalarType scalar_type)
{
    if (scalar_type == c10::ScalarType::Byte) return 1;
    if (scalar_type == c10::ScalarType::Char) return 2;
    if (scalar_type == c10::ScalarType::Short) return 3;
    if (scalar_type == c10::ScalarType::Int) return 4;
    if (scalar_type == c10::ScalarType::Long) return 5;
    if (scalar_type == c10::ScalarType::Half) return 6;
    if (scalar_type == c10::ScalarType::Float) return 7;
    if (scalar_type == c10::ScalarType::Double) return 8;
    if (scalar_type == c10::ScalarType::ComplexHalf) return 9;
    if (scalar_type == c10::ScalarType::ComplexFloat) return 10;
    if (scalar_type == c10::ScalarType::ComplexDouble) return 11;
    if (scalar_type == c10::ScalarType::Bool) return 12;
    if (scalar_type == c10::ScalarType::BFloat16) return 13;
    return 0;
}

static SymInt concrete_sym_int(int64_t value)
{
    SymInt result;
    result.integer = value;
    return result;
}

static bool load_legacy_tensor_dict(const std::string& data, const std::string& directory, bool is_parameter, std::map<std::string, PayloadMeta>& payloads, std::map<std::string, std::vector<char> >& storages, uint64_t& memory_used, std::string& error)
{
    torch::IValue value;
    try
    {
        value = torch::pickle_load(std::vector<char>(data.begin(), data.end()));
    }
    catch (const c10::Error& e)
    {
        error = directory + ": failed to load legacy tensor dictionary: " + e.what_without_backtrace();
        return false;
    }

    if (!value.isGenericDict())
    {
        error = directory + ": legacy payload is not a dictionary";
        return false;
    }

    const c10::impl::GenericDict dictionary = value.toGenericDict();
    size_t index = 0;
    for (c10::impl::GenericDict::iterator it = dictionary.begin(); it != dictionary.end(); ++it)
    {
        const torch::IValue& key = it->key();
        if (!key.isString())
        {
            error = directory + ": legacy payload key is not a string";
            return false;
        }
        const torch::IValue& item = it->value();
        if (!item.isTensor())
        {
            error = key.toStringRef() + ": unsupported legacy non-tensor payload";
            return false;
        }

        at::Tensor tensor = item.toTensor();
        if (tensor.layout() != c10::kStrided)
        {
            error = key.toStringRef() + ": unsupported legacy tensor layout";
            return false;
        }

        const int scalar_type = serialized_scalar_type(tensor.scalar_type());
        if (scalar_type == 0)
        {
            error = key.toStringRef() + ": unsupported legacy tensor scalar type";
            return false;
        }

        if (tensor.numel() < 0 || (uint64_t)tensor.numel() > PAYLOAD_MEMORY_BUDGET / tensor.element_size())
        {
            error = "legacy tensor exceeds 512 MiB materialization budget";
            return false;
        }
        const uint64_t byte_count = (uint64_t)tensor.numel() * tensor.element_size();
        // Reserve both the retained copy and dense materialization. Pickle's
        // own allocations precede this check; this is not a pickle sandbox.
        if (byte_count > std::numeric_limits<size_t>::max()
                || !reserve_payload_memory(byte_count, memory_used, error)
                || !reserve_payload_memory(byte_count, memory_used, error))
        {
            if (error.empty()) error = "legacy tensor payload is too large";
            return false;
        }
        const bool requires_grad = tensor.requires_grad();
        tensor = tensor.detach().to(c10::kCPU).contiguous();

        PayloadMeta payload;
        payload.path = "legacy_" + std::to_string(index++);
        payload.is_parameter = is_parameter;
        payload.has_tensor_meta = true;
        payload.tensor_meta.scalar_type = scalar_type;
        payload.tensor_meta.requires_grad = requires_grad;
        payload.tensor_meta.device.type = "cpu";
        payload.tensor_meta.layout = 7;
        payload.tensor_meta.storage_offset = concrete_sym_int(0);
        for (size_t i = 0; i < (size_t)tensor.dim(); i++)
        {
            payload.tensor_meta.sizes.push_back(concrete_sym_int(tensor.size(i)));
            payload.tensor_meta.strides.push_back(concrete_sym_int(tensor.stride(i)));
        }

        std::vector<char>& storage = storages[directory + payload.path];
        storage.resize((size_t)byte_count);
        if (byte_count)
            memcpy(storage.data(), tensor.const_data_ptr(), (size_t)byte_count);
        payloads[key.toStringRef()] = payload;
    }

    return true;
}
#endif

static bool load_payload_config(StoreZipReader& reader, const std::string& root, const std::string& directory, const std::string& model_name, const std::string& config_suffix, std::map<std::string, PayloadMeta>& payloads, uint64_t physical_size, std::string& error)
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
    return read_record(reader, config_name, config, physical_size, error) && parse_payload_config(config, payloads, logical_config, error);
}

static bool plan_payloads(StoreZipReader& reader, const std::string& root, const std::string& directory, const std::map<std::string, PayloadMeta>& payloads, std::map<std::string, uint64_t>& storage_sizes, uint64_t physical_size, uint64_t& memory_used, std::string& error)
{
    const std::vector<std::string> names = reader.get_names();
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
        if (std::find(names.begin(), names.end(), storage_name) == names.end())
        {
            error = logical_storage + ": missing tensor payload";
            return false;
        }
        if (!check_record_readable(reader, storage_name, error))
            return false;
        // Validate every view, including aliases of an already loaded storage,
        // before allocating or reading raw tensor data.
        const uint64_t size = reader.get_file_size(storage_name);
        if (!validate_tensor_storage(payload, size, error))
        {
            error = it->first + ": " + error;
            return false;
        }
        if (size > std::numeric_limits<size_t>::max() || size > physical_size)
        {
            error = logical_storage + ": tensor payload exceeds physical archive size or container size limit";
            return false;
        }
        if (storage_sizes.insert(std::make_pair(logical_storage, size)).second)
        {
            if (!reserve_payload_memory(size, memory_used, error))
                return false;
        }
        // validate_tensor_storage already checked dtype, concrete dimensions,
        // overflow and the per-view dense bound. Count each view, even aliases:
        // the importer materializes each attribute independently.
        const unsigned int element_sizes[] = {0, 1, 1, 2, 4, 8, 2, 4, 8, 4, 8, 16, 1, 2};
        uint64_t elements = 1;
        const TensorMeta& meta = payload.tensor_meta;
        for (size_t i = 0; i < meta.sizes.size(); i++)
            if (meta.sizes[i].integer == 0) elements = 0;
        for (size_t i = 0; i < meta.sizes.size(); i++)
            elements *= (uint64_t)meta.sizes[i].integer;
        if (!reserve_payload_memory(elements * element_sizes[meta.scalar_type], memory_used, error))
            return false;
    }
    return true;
}

static bool read_payloads(StoreZipReader& reader, const std::string& root, const std::map<std::string, uint64_t>& storage_sizes, ExportedProgramArchive& archive, std::string& error)
{
    for (std::map<std::string, uint64_t>::const_iterator it = storage_sizes.begin(); it != storage_sizes.end(); ++it)
    {
        std::map<std::string, std::vector<char> >& storages = it->first.compare(0, 13, "data/weights/") == 0
                ? archive.state_dict_storages
                : archive.constant_storages;
        std::vector<char>& storage = storages[it->first];
        storage.resize((size_t)it->second);
        // Compression/flags were checked during planning; even empty records
        // still require read_file's CRC validation.
        if (reader.read_file(root + it->first, storage.empty() ? 0 : storage.data()) != 0)
        {
            error = it->first + ": failed to read tensor payload";
            return false;
        }
    }
    return true;
}

static bool parse_exported_program_impl(const std::string& text, ExportedProgram& program, std::string& error)
{
    error.clear();
    program = ExportedProgram();
    JsonValue root;
    if (!parse_json(text, root, error))
        return false;
    ExportedProgramDecoder decoder(error);
    return decoder.decode(root, program);
}

static bool load_exported_program_archive_metadata_impl(StoreZipReader& reader, ExportedProgramArchive& archive, std::string& error)
{
    error.clear();
    archive = ExportedProgramArchive();
    const uint64_t physical_size = reader.get_archive_size();
    const std::vector<std::string> names = reader.get_names();
    const std::string root = common_archive_root(names);
    // Apply the same PT2 format/version contract as detect_model_format, using
    // the open reader instead of opening and indexing the directory again.
    if (reader.get_file_compression(root + "archive_format") < 0)
    {
        if (reader.get_file_compression(root + "serialized_exported_program.json") < 0
                || reader.get_file_compression(root + "serialized_state_dict.pt") < 0
                || reader.get_file_compression(root + "serialized_constants.pt") < 0
                || reader.get_file_compression(root + "serialized_example_inputs.pt") < 0
                || reader.get_file_compression(root + "version") < 0)
        {
            if (reader.get_file_compression(root + "data.pkl") < 0 || reader.get_file_compression(root + "constants.pkl") < 0)
                error = "unsupported model zip archive";
            return false;
        }
        archive.archive_version = -1;
        archive.model_name = "model";
        std::string document;
        if (!read_record(reader, root + "serialized_exported_program.json", document, physical_size, error))
            return false;
        return parse_exported_program_impl(document, archive.program, error);
    }

    std::string format;
    if (!read_archive_marker(reader, root + "archive_format", "format", format, physical_size, error))
        return false;
    if (format != "pt2")
    {
        error = "unsupported model archive format " + format;
        return false;
    }
    std::vector<std::string> models;
    for (size_t i = 0; i < names.size(); i++)
    {
        const std::string logical_name = root.empty() ? names[i] : names[i].substr(root.size());
        if (logical_name.size() > 12 && logical_name.compare(0, 7, "models/") == 0 && logical_name.compare(logical_name.size() - 5, 5, ".json") == 0)
            models.push_back(names[i]);
    }

    if (reader.get_file_compression(root + "archive_version") < 0 || models.empty())
    {
        error = "incomplete pt2 model archive";
        return false;
    }
    std::string version;
    if (!read_archive_marker(reader, root + "archive_version", "version", version, physical_size, error))
        return false;
    if (version != "0")
    {
        error = "unsupported pt2 archive version " + version;
        return false;
    }
    if (!read_byteorder(reader, names, root, archive, physical_size, error))
        return false;
    if (models.size() != 1)
    {
        error = "pt2 archive must contain exactly one exported program";
        return false;
    }

    const std::string logical_model = root.empty() ? models[0] : models[0].substr(root.size());
    archive.model_name = logical_model.substr(7, logical_model.size() - 12);
    archive.archive_version = 0;

    std::string document;
    if (!read_record(reader, models[0], document, physical_size, error))
        return false;
    return parse_exported_program_impl(document, archive.program, error);
}

static bool load_exported_program_archive_impl(StoreZipReader& reader, ExportedProgramArchive& archive, std::string& error)
{
    if (!load_exported_program_archive_metadata_impl(reader, archive, error))
        return false;

    const uint64_t physical_size = reader.get_archive_size();
    const std::vector<std::string> names = reader.get_names();
    const std::string root = common_archive_root(names);
    uint64_t memory_used = 0;
    if (archive.archive_version == -1)
    {
#if PNNX_TORCH_HAS_PICKLE_LOAD
        std::string state_dict;
        std::string constants;
        if (!check_record_readable(reader, root + "serialized_state_dict.pt", error)
                || !check_record_readable(reader, root + "serialized_constants.pt", error)
                || !reserve_payload_memory(reader.get_file_size(root + "serialized_state_dict.pt"), memory_used, error)
                || !reserve_payload_memory(reader.get_file_size(root + "serialized_constants.pt"), memory_used, error)
                || !read_record(reader, root + "serialized_state_dict.pt", state_dict, physical_size, error)
                || !read_record(reader, root + "serialized_constants.pt", constants, physical_size, error))
            return false;
        if (!load_legacy_tensor_dict(state_dict, "data/weights/", true, archive.state_dict, archive.state_dict_storages, memory_used, error))
            return false;
        return load_legacy_tensor_dict(constants, "data/constants/", false, archive.constants, archive.constant_storages, memory_used, error);
#else
        error = "legacy exported program payloads require LibTorch pickle support";
        return false;
#endif
    }

    if (!load_payload_config(reader, root, "data/weights/", archive.model_name, "_weights_config.json", archive.state_dict, physical_size, error)
            || !load_payload_config(reader, root, "data/constants/", archive.model_name, "_constants_config.json", archive.constants, physical_size, error))
        return false;
    // Plan BOTH dictionaries before reading ANY storage. This also catches
    // many tiny, zero-stride views whose combined materialization is enormous.
    std::map<std::string, uint64_t> storage_sizes;
    if (!plan_payloads(reader, root, "data/weights/", archive.state_dict, storage_sizes, physical_size, memory_used, error)
            || !plan_payloads(reader, root, "data/constants/", archive.constants, storage_sizes, physical_size, memory_used, error))
        return false;
    return read_payloads(reader, root, storage_sizes, archive, error);
}

bool parse_exported_program(const std::string& text, ExportedProgram& program, std::string& error)
{
    program = ExportedProgram();
    error.clear();
    try
    {
        ExportedProgram result;
        if (!parse_exported_program_impl(text, result, error))
            return false;
        program = std::move(result);
        return true;
    }
    catch (const std::bad_alloc&)
    {
        program = ExportedProgram();
        error = "pt2 metadata allocation failed";
    }
    catch (const std::length_error&)
    {
        program = ExportedProgram();
        error = "pt2 metadata length exceeds container limits";
    }
    return false;
}

static bool load_archive_boundary(const std::string& path, ExportedProgramArchive& archive, std::string& error, bool metadata_only)
{
    // Free a previous result first, so it cannot double the loading budget.
    archive = ExportedProgramArchive();
    error.clear();
    try
    {
        if (!supported_host_byteorder(error))
            return false;
        // Keep one validated directory and file handle through metadata and
        // payload planning/reads, including the physical file size bound.
        StoreZipReader reader;
        if (reader.open(path) != 0)
        {
            error = "failed to read pt2 archive";
            return false;
        }
        ExportedProgramArchive result;
        const bool loaded = metadata_only ? load_exported_program_archive_metadata_impl(reader, result, error)
                            : load_exported_program_archive_impl(reader, result, error);
        if (!loaded)
            return false;
        archive = std::move(result);
        return true;
    }
    catch (const std::bad_alloc&)
    {
        archive = ExportedProgramArchive();
        error = "pt2 archive allocation failed";
    }
    catch (const std::length_error&)
    {
        archive = ExportedProgramArchive();
        error = "pt2 archive length exceeds container limits";
    }
    // Do not turn unrelated runtime/programming exceptions into diagnostics.
    return false;
}

bool load_exported_program_archive_metadata(const std::string& path, ExportedProgramArchive& archive, std::string& error)
{
    return load_archive_boundary(path, archive, error, true);
}

bool load_exported_program_archive(const std::string& path, ExportedProgramArchive& archive, std::string& error)
{
    return load_archive_boundary(path, archive, error, false);
}

} // namespace pt2
} // namespace pnnx