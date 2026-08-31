// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_schema.h"
#include "json.hpp"

#include <stdio.h>
#include <string.h>

namespace pnnx {

static bool string_ends_with(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size()
           && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string find_pt2_model_json_entry(const std::vector<std::string>& entry_names)
{
    for (size_t i = 0; i < entry_names.size(); i++)
    {
        if (string_ends_with(entry_names[i], "models/model.json"))
            return entry_names[i];
    }
    return std::string();
}

std::string Pt2Program::weight_entry_path(const std::string& path_name) const
{
    return archive_root + "data/weights/" + path_name;
}

std::string Pt2Program::constant_entry_path(const std::string& path_name) const
{
    return archive_root + "data/constants/" + path_name;
}

const Pt2WeightEntry* Pt2Program::find_weight(const std::string& state_dict_name) const
{
    for (size_t i = 0; i < weights.size(); i++)
    {
        if (weights[i].state_dict_name == state_dict_name)
            return &weights[i];
    }
    return 0;
}

const Pt2WeightEntry* Pt2Program::find_constant(const std::string& state_dict_name) const
{
    for (size_t i = 0; i < constants.size(); i++)
    {
        if (constants[i].state_dict_name == state_dict_name)
            return &constants[i];
    }
    return 0;
}

// ----- argument -----

// as_* 变体 → 参数类型。未知变体返回 NONE 并告警(容忍后续 torch 版本扩展)。
static Pt2Argument::ArgType detect_arg_type(const JsonValue& arg)
{
    if (arg.hasMember("as_tensor"))
        return Pt2Argument::TENSOR;
    if (arg.hasMember("as_tensors"))
        return Pt2Argument::TENSORS;
    if (arg.hasMember("as_int"))
        return Pt2Argument::INT;
    if (arg.hasMember("as_ints"))
        return Pt2Argument::INTS;
    if (arg.hasMember("as_float"))
        return Pt2Argument::FLOAT;
    if (arg.hasMember("as_floats"))
        return Pt2Argument::FLOATS;
    if (arg.hasMember("as_bool"))
        return Pt2Argument::BOOL;
    if (arg.hasMember("as_bools"))
        return Pt2Argument::BOOLS;
    if (arg.hasMember("as_string"))
        return Pt2Argument::STRING;
    if (arg.hasMember("as_strings"))
        return Pt2Argument::STRINGS;
    if (arg.hasMember("as_scalar_type"))
        return Pt2Argument::SCALAR_TYPE;
    if (arg.hasMember("as_device"))
        return Pt2Argument::DEVICE;
    if (arg.hasMember("as_memory_format"))
        return Pt2Argument::MEMORY_FORMAT;
    if (arg.hasMember("as_none"))
        return Pt2Argument::NONE;

    fprintf(stderr, "load_pt2_schema: unknown argument variant:");
    for (std::map<std::string, JsonValue>::const_iterator it = arg.object_value.begin();
         it != arg.object_value.end(); ++it)
    {
        fprintf(stderr, " %s", it->first.c_str());
    }
    fprintf(stderr, "\n");
    return Pt2Argument::NONE;
}

static void collect_tensor_names(const JsonValue& v, std::vector<std::string>& names)
{
    // as_tensor = {"name": x};as_tensors = [{"name": x}, ...]
    if (v.isObject() && v.hasMember("name"))
    {
        names.push_back(v["name"].asString());
        return;
    }
    if (v.isArray())
    {
        for (size_t i = 0; i < v.size(); i++)
        {
            if (v[i].isObject() && v[i].hasMember("name"))
                names.push_back(v[i]["name"].asString());
        }
    }
}

static Pt2Argument parse_argument(const JsonValue& arg, const std::string& name, bool is_kwarg)
{
    Pt2Argument a;
    a.name = name;
    a.is_kwarg = is_kwarg;
    a.type = detect_arg_type(arg);

    switch (a.type)
    {
    case Pt2Argument::TENSOR:
    case Pt2Argument::TENSORS:
        collect_tensor_names(arg[a.type == Pt2Argument::TENSOR ? "as_tensor" : "as_tensors"], a.tensor_names);
        break;
    case Pt2Argument::INT:
        a.int_value = arg["as_int"].asInt();
        break;
    case Pt2Argument::INTS:
        for (size_t i = 0; i < arg["as_ints"].size(); i++)
            a.int_values.push_back(arg["as_ints"][i].asInt());
        break;
    case Pt2Argument::FLOAT:
        a.float_value = arg["as_float"].asDouble();
        break;
    case Pt2Argument::FLOATS:
        for (size_t i = 0; i < arg["as_floats"].size(); i++)
            a.float_values.push_back(arg["as_floats"][i].asDouble());
        break;
    case Pt2Argument::BOOL:
        a.bool_value = arg["as_bool"].asBool();
        break;
    case Pt2Argument::BOOLS:
        for (size_t i = 0; i < arg["as_bools"].size(); i++)
            a.bool_values.push_back(arg["as_bools"][i].asBool());
        break;
    case Pt2Argument::STRING:
        a.string_value = arg["as_string"].asString();
        break;
    case Pt2Argument::STRINGS:
        for (size_t i = 0; i < arg["as_strings"].size(); i++)
            a.string_values.push_back(arg["as_strings"][i].asString());
        break;
    case Pt2Argument::SCALAR_TYPE:
        a.int_value = arg["as_scalar_type"].asInt();
        break;
    case Pt2Argument::DEVICE:
        if (arg["as_device"].isObject())
        {
            a.device_type = arg["as_device"]["type"].asString();
            if (arg["as_device"]["index"].isNumber())
                a.device_index = arg["as_device"]["index"].asInt();
            else
                a.device_index = -1; // null
        }
        break;
    case Pt2Argument::MEMORY_FORMAT:
        a.int_value = arg["as_memory_format"].asInt();
        break;
    case Pt2Argument::NONE:
    default:
        break;
    }

    return a;
}

// ----- node -----

static Pt2Node parse_node(const JsonValue& n)
{
    Pt2Node node;
    node.name = n["name"].asString();
    node.target = n["target"].asString();

    const JsonValue& inputs = n["inputs"];
    for (size_t i = 0; i < inputs.size(); i++)
    {
        Pt2NodeInput input;
        input.name = inputs[i]["name"].asString();
        // kind: 1 = positional, 2 = keyword(实测,其余值按 positional 处理)
        input.arg = parse_argument(inputs[i]["arg"], input.name, inputs[i]["kind"].asInt() == 2);
        node.inputs.push_back(input);
    }

    const JsonValue& outputs = n["outputs"];
    for (size_t i = 0; i < outputs.size(); i++)
    {
        Pt2NodeOutput output;
        // as_tensor(单输出)或 as_tensors(元组输出,如 chunk / unbind)
        if (outputs[i].hasMember("as_tensor"))
            collect_tensor_names(outputs[i]["as_tensor"], output.tensor_names);
        else if (outputs[i].hasMember("as_tensors"))
            collect_tensor_names(outputs[i]["as_tensors"], output.tensor_names);
        node.outputs.push_back(output);
    }

    // metadata 可能整体缺失或为 null
    const JsonValue& metadata = n["metadata"];
    if (metadata.isObject())
    {
        if (metadata["nn_module_stack"].isString())
            node.nn_module_stack = metadata["nn_module_stack"].asString();
        if (metadata["torch_fn"].isString())
            node.torch_fn = metadata["torch_fn"].asString();
        if (metadata["stack_trace"].isString())
            node.stack_trace = metadata["stack_trace"].asString();
    }

    return node;
}

// ----- signature -----

// arg 形态:parameter/buffer/tensor_constant = {"name": x};user_input = {"as_tensor": {"name": x}}
static std::string parse_spec_graph_name(const JsonValue& inner)
{
    if (!inner.hasMember("arg"))
        return std::string();

    const JsonValue& arg = inner["arg"];
    if (arg.hasMember("name"))
        return arg["name"].asString();
    if (arg.hasMember("as_tensor"))
        return arg["as_tensor"]["name"].asString();

    return std::string();
}

static void parse_input_specs(const JsonValue& specs, std::vector<Pt2InputSpec>& out)
{
    for (size_t i = 0; i < specs.size(); i++)
    {
        const JsonValue& spec = specs[i];
        // 实测:每个 spec 对象恰含一个 kind 键(user_input/parameter/buffer/tensor_constant)
        for (std::map<std::string, JsonValue>::const_iterator it = spec.object_value.begin();
             it != spec.object_value.end(); ++it)
        {
            Pt2InputSpec s;
            const std::string& kind_name = it->first;
            const JsonValue& inner = it->second;

            if (kind_name == "user_input")
                s.kind = Pt2InputSpec::USER_INPUT;
            else if (kind_name == "parameter")
                s.kind = Pt2InputSpec::PARAMETER;
            else if (kind_name == "buffer")
                s.kind = Pt2InputSpec::BUFFER;
            else if (kind_name == "tensor_constant")
                s.kind = Pt2InputSpec::TENSOR_CONSTANT;
            else
            {
                fprintf(stderr, "load_pt2_schema: unknown input spec kind %s\n", kind_name.c_str());
                continue;
            }

            s.graph_name = parse_spec_graph_name(inner);
            if (s.kind == Pt2InputSpec::PARAMETER && inner.hasMember("parameter_name"))
                s.state_dict_name = inner["parameter_name"].asString();
            if (s.kind == Pt2InputSpec::BUFFER && inner.hasMember("buffer_name"))
                s.state_dict_name = inner["buffer_name"].asString();
            if (s.kind == Pt2InputSpec::TENSOR_CONSTANT && inner.hasMember("tensor_constant_name"))
                s.state_dict_name = inner["tensor_constant_name"].asString();
            if (inner.hasMember("persistent") && inner["persistent"].isBool())
                s.persistent = inner["persistent"].asBool();

            out.push_back(s);
        }
    }
}

static void parse_output_specs(const JsonValue& specs, std::vector<Pt2OutputSpec>& out)
{
    for (size_t i = 0; i < specs.size(); i++)
    {
        const JsonValue& spec = specs[i];
        // 实测:仅 user_output 一种,{ "arg": { "as_tensor": { "name": x } } }
        for (std::map<std::string, JsonValue>::const_iterator it = spec.object_value.begin();
             it != spec.object_value.end(); ++it)
        {
            Pt2OutputSpec s;
            s.graph_name = parse_spec_graph_name(it->second);
            out.push_back(s);
        }
    }
}

// ----- weights / constants config -----

// sizes/strides 元素是单键对象 {"as_int": 4};storage_offset 同形
static std::vector<long long> parse_int_list_of_objects(const JsonValue& v)
{
    std::vector<long long> out;
    if (!v.isArray())
        return out;
    for (size_t i = 0; i < v.size(); i++)
    {
        if (v[i].isObject() && v[i].hasMember("as_int"))
            out.push_back(v[i]["as_int"].asInt());
    }
    return out;
}

static void parse_weight_config(const JsonValue& config, std::vector<Pt2WeightEntry>& out)
{
    // config: { state_dict_name → {path_name, is_param, use_pickle, tensor_meta{...}} }
    for (std::map<std::string, JsonValue>::const_iterator it = config.object_value.begin();
         it != config.object_value.end(); ++it)
    {
        Pt2WeightEntry e;
        e.state_dict_name = it->first;
        const JsonValue& v = it->second;

        if (v.hasMember("path_name"))
            e.path_name = v["path_name"].asString();
        if (v.hasMember("is_param") && v["is_param"].isBool())
            e.is_param = v["is_param"].asBool();
        if (v.hasMember("use_pickle") && v["use_pickle"].isBool())
            e.use_pickle = v["use_pickle"].asBool();

        const JsonValue& tm = v["tensor_meta"];
        if (tm.isObject())
        {
            if (tm.hasMember("dtype") && tm["dtype"].isNumber())
                e.dtype = tm["dtype"].asInt();
            if (tm.hasMember("sizes"))
                e.sizes = parse_int_list_of_objects(tm["sizes"]);
            if (tm.hasMember("strides"))
                e.strides = parse_int_list_of_objects(tm["strides"]);
            if (tm.hasMember("storage_offset") && tm["storage_offset"].isObject()
                && tm["storage_offset"].hasMember("as_int"))
            {
                e.storage_offset = tm["storage_offset"]["as_int"].asInt();
            }
        }

        out.push_back(e);
    }
}

// ----- top level -----

// ----- graph.tensor_values(张量名 → FakeTensor 元数据,含中间张量) -----
// sizes 元素实测全为 {"as_int": N};符号形状等其他变体出现时按 -1(未知)记录
static void parse_tensor_values(const JsonValue& tv, std::map<std::string, Pt2TensorMeta>& out)
{
    if (!tv.isObject())
        return;

    for (std::map<std::string, JsonValue>::const_iterator it = tv.object_value.begin();
         it != tv.object_value.end(); ++it)
    {
        Pt2TensorMeta meta;

        const JsonValue& m = it->second;
        if (!m.isObject())
            continue;

        if (m.hasMember("dtype") && m["dtype"].isInt())
            meta.dtype = m["dtype"].asInt();

        if (m.hasMember("sizes") && m["sizes"].isArray())
        {
            const JsonValue& sizes = m["sizes"];
            for (size_t i = 0; i < sizes.size(); i++)
            {
                long long dim = -1;
                const JsonValue& d = sizes[i];
                if (d.isObject() && d.hasMember("as_int") && d["as_int"].isInt())
                    dim = d["as_int"].asInt();
                meta.sizes.push_back(dim);
            }
        }

        out[it->first] = meta;
    }
}

int load_pt2_schema(const std::string& ptpath, Pt2Program& program)
{
    try
    {
        StoreZipReader zip;
        if (zip.open(ptpath) != 0)
        {
            fprintf(stderr, "load_pt2_schema: open failed %s\n", ptpath.c_str());
            return -1;
        }

        const std::string model_json_entry = find_pt2_model_json_entry(zip.get_names());
        if (model_json_entry.empty())
        {
            fprintf(stderr, "load_pt2_schema: models/model.json not found in %s\n", ptpath.c_str());
            return -1;
        }

        // archive_root = 条目名去掉 "models/model.json"(含尾部斜杠)
        program.archive_root = model_json_entry.substr(0, model_json_entry.size() - strlen("models/model.json"));

        const uint64_t json_size = zip.get_file_size(model_json_entry);
        std::vector<char> buf((size_t)json_size);
        if (zip.read_file(model_json_entry, buf.data()) != 0)
        {
            fprintf(stderr, "load_pt2_schema: read failed %s\n", model_json_entry.c_str());
            return -1;
        }

        const JsonValue root = parse_json(std::string(buf.data(), (size_t)json_size));

        // header
        const JsonValue& schema_version = root["schema_version"];
        if (schema_version.isObject()) // 实测 {"major": 8, "minor": 20}
        {
            program.schema_version_major = schema_version["major"].asInt();
            program.schema_version_minor = schema_version["minor"].asInt();
        }
        program.torch_version = root["torch_version"].asString();
        const JsonValue& opset = root["opset_version"];
        if (opset.isObject())
        {
            for (std::map<std::string, JsonValue>::const_iterator it = opset.object_value.begin();
                 it != opset.object_value.end(); ++it)
            {
                char buf32[32];
                if (it->second.isInt())
                {
                    snprintf(buf32, sizeof(buf32), "%lld", it->second.asInt());
                    program.opset_version[it->first] = buf32;
                }
                else if (it->second.isString())
                {
                    program.opset_version[it->first] = it->second.asString();
                }
            }
        }

        const JsonValue& graph_module = root["graph_module"];

        // nodes
        const JsonValue& nodes = graph_module["graph"]["nodes"];
        for (size_t i = 0; i < nodes.size(); i++)
        {
            program.nodes.push_back(parse_node(nodes[i]));
        }

        // graph.tensor_values(张量元数据表,含中间张量的形状/dtype)
        const JsonValue& graph = graph_module["graph"];
        if (graph.isObject() && graph.hasMember("tensor_values"))
            parse_tensor_values(graph["tensor_values"], program.tensor_values);

        // signature
        const JsonValue& signature = graph_module["signature"];
        if (signature.isObject())
        {
            if (signature.hasMember("input_specs"))
                parse_input_specs(signature["input_specs"], program.input_specs);
            if (signature.hasMember("output_specs"))
                parse_output_specs(signature["output_specs"], program.output_specs);
        }

        // weights / constants config(close() 只关句柄,filemetas 仍可用)
        const std::string weights_entry = program.archive_root + "data/weights/model_weights_config.json";
        const std::string constants_entry = program.archive_root + "data/constants/model_constants_config.json";

        if (zip.get_file_size(weights_entry) > 0)
        {
            std::vector<char> wbuf((size_t)zip.get_file_size(weights_entry));
            if (zip.read_file(weights_entry, wbuf.data()) == 0)
            {
                const JsonValue wroot = parse_json(std::string(wbuf.data(), wbuf.size()));
                if (wroot.hasMember("config"))
                    parse_weight_config(wroot["config"], program.weights);
            }
        }

        if (zip.get_file_size(constants_entry) > 0)
        {
            std::vector<char> cbuf((size_t)zip.get_file_size(constants_entry));
            if (zip.read_file(constants_entry, cbuf.data()) == 0)
            {
                const JsonValue croot = parse_json(std::string(cbuf.data(), cbuf.size()));
                if (croot.hasMember("config"))
                    parse_weight_config(croot["config"], program.constants);
            }
        }

        zip.close();

        return 0;
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "load_pt2_schema: %s (%s)\n", e.what(), ptpath.c_str());
        return -1;
    }
}

} // namespace pnnx
