// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exportedprogram.h"
#include "load_exportedprogram_legacy.h"
#include "load_pt2_defaults.h"
#include "load_pt2_serde.h"
#include "load_pt2_tensor.h"

#include "pnnx_json.h"
#include "storezip.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <climits>
#include <complex>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

// serde dtype / memory-format / target normalization + tensor_meta readers are
// split out into load_pt2_serde.cpp (see load_pt2_serde.h).

// tensor_meta materialization is split out into load_pt2_tensor.cpp
// (see load_pt2_tensor.h).

// default-kwargs restoration (append_default_kwargs + new_constant helpers)
// is split out into load_pt2_defaults.cpp (see load_pt2_defaults.h).

// recursively build a higher_order subgraph (wrap_with_set_grad_enabled /
// wrap_with_autocast); subgraph nodes are merged into the main graph, subgraph
// inputs reference main-graph operands and subgraph outputs become the
// higher_order node outputs
static int build_subgraph_nodes(Graph& g, const JsonValue& subgraph,
                                std::map<std::string, Operand*>& operands_by_name,
                                int& constant_index, int& subop_index);

// inline a wrap_with_autocast / wrap_with_set_grad_enabled higher-order node:
// the wrapper carries scalar context args, one embedded subgraph, and the
// captured closure tensors (as_tensor inputs). bind those captures to the
// subgraph placeholders in order, build the subgraph body, then map the
// subgraph results onto the wrapper output names so later nodes resolve.
static int inline_wrapper_subgraph(Graph& g, const JsonValue& nd,
                                   std::map<std::string, Operand*>& operands_by_name,
                                   int& constant_index, int& subop_index)
{
    const JsonValue& ho_inputs = nd["inputs"];
    const JsonValue* subgraph = 0;
    std::vector<Operand*> captures;

    for (size_t j = 0; j < ho_inputs.size(); j++)
    {
        const JsonValue& arg = ho_inputs[j]["arg"];
        if (arg.has("as_graph"))
        {
            subgraph = &arg["as_graph"]["graph"];
        }
        else if (arg.has("as_tensor"))
        {
            // captured closure tensor feeding a subgraph placeholder
            std::string name = arg["as_tensor"]["name"].as_string();
            std::map<std::string, Operand*>::iterator it = operands_by_name.find(name);
            if (it == operands_by_name.end())
            {
                fprintf(stderr, "captured operand %s not found for higher_order op\n", name.c_str());
                return -1;
            }
            captures.push_back(it->second);
        }
    }
    if (!subgraph)
        return 0;

    // bind the subgraph placeholders to the captured operands in order
    if (subgraph->has("inputs"))
    {
        const JsonValue& sub_inputs = (*subgraph)["inputs"];
        if (sub_inputs.size() == captures.size())
        {
            for (size_t k = 0; k < sub_inputs.size(); k++)
            {
                if (sub_inputs[k].has("as_tensor"))
                {
                    std::string pname = sub_inputs[k]["as_tensor"]["name"].as_string();
                    if (operands_by_name.find(pname) == operands_by_name.end())
                        operands_by_name[pname] = captures[k];
                }
            }
        }
        else if (!captures.empty())
        {
            // this torch version may name the placeholders identically to the
            // captured operands (binding above is then a no-op), but flag the
            // mismatch so a future naming change does not fail silently
            fprintf(stderr, "warning: higher_order subgraph has %zu inputs but %zu captured operands\n", sub_inputs.size(), captures.size());
        }
    }

    int ret = build_subgraph_nodes(g, *subgraph, operands_by_name, constant_index, subop_index);
    if (ret != 0)
        return ret;

    // map the subgraph results to the wrapper output names
    if (subgraph->has("outputs") && nd.has("outputs"))
    {
        const JsonValue& sub_outs = (*subgraph)["outputs"];
        const JsonValue& wrap_outs = nd["outputs"];
        for (size_t k = 0; k < sub_outs.size() && k < wrap_outs.size(); k++)
        {
            if (!sub_outs[k].has("as_tensor") || !wrap_outs[k].has("as_tensor"))
                continue;
            std::string sname = sub_outs[k]["as_tensor"]["name"].as_string();
            std::string wname = wrap_outs[k]["as_tensor"]["name"].as_string();
            std::map<std::string, Operand*>::iterator it = operands_by_name.find(sname);
            if (it != operands_by_name.end() && operands_by_name.find(wname) == operands_by_name.end())
                operands_by_name[wname] = it->second;
        }
    }

    return 0;
}

static int build_subgraph_nodes(Graph& g, const JsonValue& subgraph,
                                std::map<std::string, Operand*>& operands_by_name,
                                int& constant_index, int& subop_index)
{
    const JsonValue& nodes = subgraph["nodes"];
    const JsonValue& tensor_values = subgraph["tensor_values"];

    for (size_t i = 0; i < nodes.size(); i++)
    {
        const JsonValue& nd = nodes[i];
        std::string target = nd["target"].as_string();
        std::string op_type = normalize_target(target);

        if (op_type == "torchvision::deform_conv2d")
            op_type = "torchvision.ops.DeformConv2d";
        else if (op_type == "torchvision::roi_align")
            op_type = "torchvision.ops.RoIAlign";
        if (op_type == "aten::hann_window")
            op_type = "torch.hann_window";
        else if (op_type == "aten::hamming_window")
            op_type = "torch.hamming_window";

        // dynamo assertion / shape guard ops have no tensor output, skip them
        if (op_type.compare(0, 14, "aten::_assert_") == 0)
            continue;
        if (op_type == "_operator")
            continue;

        // nested higher_order: inline its subgraph (bind captures + outputs)
        if (target.find("higher_order.wrap_with_set_grad_enabled") != std::string::npos
                || target.find("higher_order.wrap_with_autocast") != std::string::npos)
        {
            int ret = inline_wrapper_subgraph(g, nd, operands_by_name, constant_index, subop_index);
            if (ret != 0)
                return ret;
            continue;
        }

        char op_name[32];
        snprintf(op_name, 32, "subop_%d", subop_index++);

        Operator* op = g.new_operator(op_type, op_name);

        // inputs
        const JsonValue& inputs = nd["inputs"];
        std::vector<std::string> inputnames;
        for (size_t j = 0; j < inputs.size(); j++)
        {
            const JsonValue& inp = inputs[j];
            std::string argname = inp["name"].as_string();
            const JsonValue& arg = inp["arg"];

            if (op_type == "torch.hann_window" || op_type == "torch.hamming_window")
            {
                // window function args become params (no constant input); drop pin_memory
                if (argname == "pin_memory")
                    continue;
                if (arg.has("as_int"))
                {
                    op->params[argname] = (int)arg["as_int"].as_int();
                }
                else if (arg.has("as_device"))
                {
                    std::string dev = arg["as_device"]["type"].as_string();
                    if (arg["as_device"].has("index") && !arg["as_device"]["index"].is_null())
                    {
                        char tmp[32];
                        snprintf(tmp, 32, ":%lld", (long long)arg["as_device"]["index"].as_int());
                        dev += tmp;
                    }
                    op->params[argname] = dev;
                }
                else if (arg.has("as_bool"))
                {
                    op->params[argname] = arg["as_bool"].as_bool();
                }
                else if (arg.has("as_scalar_type"))
                {
                    // hann/hamming_window carry a dtype override (e.g.
                    // float64); keep it so the level2 fold can honor it; an
                    // unrepresentable scalar dtype is rejected, not dropped
                    int pnnx_type = 0;
                    if (!scalar_dtype_to_pnnx_type(arg["as_scalar_type"].as_int(), pnnx_type))
                        return -1;
                    op->params[argname] = pnnx_type;
                }
                else if (arg.has("as_float"))
                {
                    // hamming_window's alpha/beta arrive as float scalars
                    op->params[argname] = (float)arg["as_float"].as_double();
                }
                continue;
            }

            inputnames.push_back(argname);

            if (arg.has("as_tensor"))
            {
                std::string name = arg["as_tensor"]["name"].as_string();
                Operand* r = operands_by_name[name];
                if (!r)
                {
                    fprintf(stderr, "operand %s not found for %s\n", name.c_str(), op_type.c_str());
                    return -1;
                }
                r->consumers.push_back(op);
                op->inputs.push_back(r);
            }
            else if (arg.has("as_int"))
            {
                long long iv = arg["as_int"].as_int();
                if (iv == std::numeric_limits<long long>::max())
                    iv = INT_MAX;
                if (iv == std::numeric_limits<long long>::min())
                    iv = INT_MIN;
                new_constant(g, op, (long long)iv, constant_index);
            }
            else if (arg.has("as_ints"))
            {
                std::vector<int> ai;
                for (size_t k = 0; k < arg["as_ints"].size(); k++)
                {
                    long long v = arg["as_ints"][k].as_int();
                    if (v == std::numeric_limits<long long>::max())
                        v = INT_MAX;
                    if (v == std::numeric_limits<long long>::min())
                        v = INT_MIN;
                    ai.push_back((int)v);
                }
                new_constant(g, op, ai, constant_index);
            }
            else if (arg.has("as_float"))
            {
                new_constant(g, op, (float)arg["as_float"].as_double(), constant_index);
            }
            else if (arg.has("as_bool"))
            {
                new_constant(g, op, arg["as_bool"].as_bool(), constant_index);
            }
            else if (arg.has("as_none"))
            {
                new_constant(g, op, Parameter(), constant_index);
            }
            else if (arg.has("as_scalar_type"))
            {
                int dtype_value = 0;
                if (!scalar_dtype_to_pnnx_dtype_value(arg["as_scalar_type"].as_int(), dtype_value))
                    return -1;
                new_constant(g, op, dtype_value, constant_index);
            }
            else if (arg.has("as_tensors"))
            {
                char lc_name[32];
                snprintf(lc_name, 32, "pnnx_list_%d", constant_index++);

                // insert before the consumer so the saved python keeps the list
                // construct ahead of the op that consumes it (the ops follow the
                // json node order and are appended one by one)
                Operator* lc = g.new_operator_before("prim::ListConstruct", lc_name, op);
                Operand* lr = g.new_operand(lc_name);
                lr->producer = lc;
                lc->outputs.push_back(lr);

                for (size_t k = 0; k < arg["as_tensors"].size(); k++)
                {
                    std::string name = arg["as_tensors"][k]["name"].as_string();
                    Operand* r = operands_by_name[name];
                    if (!r)
                    {
                        fprintf(stderr, "operand %s not found for list\n", name.c_str());
                        return -1;
                    }
                    r->consumers.push_back(lc);
                    lc->inputs.push_back(r);
                }

                lr->consumers.push_back(op);
                op->inputs.push_back(lr);
            }
            else if (arg.has("as_device"))
            {
                std::string dev = arg["as_device"]["type"].as_string();
                if (arg["as_device"].has("index") && !arg["as_device"]["index"].is_null())
                {
                    char tmp[32];
                    snprintf(tmp, 32, ":%lld", (long long)arg["as_device"]["index"].as_int());
                    dev += tmp;
                }
                new_constant(g, op, dev, constant_index);
            }
            else if (arg.has("as_string"))
            {
                new_constant(g, op, arg["as_string"].as_string(), constant_index);
            }
            else if (arg.has("as_strings"))
            {
                std::vector<std::string> as;
                for (size_t k = 0; k < arg["as_strings"].size(); k++)
                    as.push_back(arg["as_strings"][k].as_string());
                new_constant(g, op, as, constant_index);
            }
            else if (arg.has("as_floats"))
            {
                std::vector<float> af;
                for (size_t k = 0; k < arg["as_floats"].size(); k++)
                    af.push_back((float)arg["as_floats"][k].as_double());
                new_constant(g, op, af, constant_index);
            }
            else if (arg.has("as_layout"))
            {
                new_constant(g, op, (int)arg["as_layout"].as_int(), constant_index);
            }
            else if (arg.has("as_memory_format"))
            {
                new_constant(g, op, serde_memory_format_to_pnnx(arg["as_memory_format"].as_int()), constant_index);
            }
            else if (arg.has("as_complex"))
            {
                // complex constant {"real": r, "imag": i}
                float real = (float)arg["as_complex"]["real"].as_double();
                float imag = (float)arg["as_complex"]["imag"].as_double();
                new_constant(g, op, std::complex<float>(real, imag), constant_index);
            }
            else
            {
                // unknown scalar arg: fail loudly (mirror the main loader loop)
                // instead of silently skipping it and skewing inputs/inputnames
                fprintf(stderr, "unsupported subgraph arg type for %s arg %s\n", op_type.c_str(), argname.c_str());
                return -1;
            }
        }

        // outputs
        const JsonValue& outputs = nd["outputs"];
        for (size_t j = 0; j < outputs.size(); j++)
        {
            if (outputs[j].has("as_tensor"))
            {
                std::string name = outputs[j]["as_tensor"]["name"].as_string();

                Operand* r = g.new_operand(name);
                r->producer = op;
                op->outputs.push_back(r);

                if (tensor_values.has(name))
                {
                    const JsonValue& meta = tensor_values[name];
                    r->type = read_dtype(meta);
                    read_sizes(meta, r->shape);
                }

                operands_by_name[name] = r;
            }
        }

        if (!inputnames.empty())
            op->inputnames = inputnames;

        append_default_kwargs(g, op, op_type, inputnames, constant_index);
    }

    return 0;
}

int load_exportedprogram(const std::string& pt2path, Graph& g,
                         const std::vector<std::vector<int64_t> >& input_shapes,
                         const std::vector<std::string>& input_types)
{
    StoreZipReader zip;
    if (zip.open(pt2path) != 0)
    {
        fprintf(stderr, "open %s failed\n", pt2path.c_str());
        return -1;
    }

    // locate records
    // container layouts handled here:
    //   2.8+  : {base}/models/model.json graph + {base}/data/weights/*config*.json
    //           + raw byte shards (+ archive_format/archive_version records)
    //   <2.8  : flat serialized_exported_program.json graph + pickled
    //           serialized_state_dict.pt / serialized_constants.pt + version
    std::vector<std::string> names = zip.get_names();
    std::string model_json_name;
    std::string weights_config_name;
    std::string constants_config_name;
    bool is_legacy = false;
    for (size_t i = 0; i < names.size(); i++)
    {
        if (model_json_name.empty() && names[i].find("models/model.json") != std::string::npos)
            model_json_name = names[i];
        if (names[i] == "serialized_exported_program.json")
            is_legacy = true;
        if (weights_config_name.empty() && names[i].find("weights") != std::string::npos && names[i].find("config") != std::string::npos && names[i].size() >= 5 && names[i].compare(names[i].size() - 5, 5, ".json") == 0)
            weights_config_name = names[i];
        if (constants_config_name.empty() && names[i].find("constants") != std::string::npos && names[i].find("config") != std::string::npos && names[i].size() >= 5 && names[i].compare(names[i].size() - 5, 5, ".json") == 0)
            constants_config_name = names[i];
    }
    if (is_legacy && model_json_name.empty())
        model_json_name = "serialized_exported_program.json";

    if (model_json_name.empty())
    {
        fprintf(stderr, "model graph record not found in %s\n", pt2path.c_str());
        return -1;
    }

    // read and parse model json
    JsonValue root;
    {
        uint64_t size = zip.get_file_size(model_json_name);
        std::vector<char> buf((size_t)size + 1);
        if (zip.read_file(model_json_name, buf.data()) != 0)
        {
            fprintf(stderr, "read %s failed\n", model_json_name.c_str());
            return -1;
        }
        buf[size] = 0;

        if (!JsonParser::parse(buf.data(), (size_t)size, root))
        {
            fprintf(stderr, "parse %s failed\n", model_json_name.c_str());
            return -1;
        }
    }

    // weights payload config : fqn -> { path_name, tensor_meta }
    std::map<std::string, std::pair<std::string, JsonValue> > weights;
    if (!weights_config_name.empty())
    {
        JsonValue cfg;
        uint64_t size = zip.get_file_size(weights_config_name);
        std::vector<char> buf((size_t)size + 1);
        if (zip.read_file(weights_config_name, buf.data()) != 0)
        {
            fprintf(stderr, "read %s failed\n", weights_config_name.c_str());
            return -1;
        }
        buf[size] = 0;

        if (!JsonParser::parse(buf.data(), (size_t)size, cfg))
        {
            fprintf(stderr, "parse %s failed\n", weights_config_name.c_str());
            return -1;
        }

        const std::map<std::string, JsonValue>& c = cfg["config"].as_object();
        for (std::map<std::string, JsonValue>::const_iterator it = c.begin(); it != c.end(); ++it)
        {
            std::string path_name = it->second["path_name"].as_string();
            JsonValue meta = it->second["tensor_meta"];
            weights[it->first] = std::make_pair(path_name, meta);
        }
    }

    // constants payload config : fqn -> { path_name, tensor_meta }
    std::map<std::string, std::pair<std::string, JsonValue> > constants;
    // legacy payload raw bytes (filled by pnnx_load_legacy_payloads below)
    std::map<std::string, std::vector<char> > legacy_raw;
    if (!constants_config_name.empty())
    {
        JsonValue cfg;
        uint64_t size = zip.get_file_size(constants_config_name);
        std::vector<char> buf((size_t)size + 1);
        if (zip.read_file(constants_config_name, buf.data()) != 0)
        {
            fprintf(stderr, "read %s failed\n", constants_config_name.c_str());
            return -1;
        }
        buf[size] = 0;

        if (!JsonParser::parse(buf.data(), (size_t)size, cfg))
        {
            fprintf(stderr, "parse %s failed\n", constants_config_name.c_str());
            return -1;
        }

        const std::map<std::string, JsonValue>& c = cfg["config"].as_object();
        for (std::map<std::string, JsonValue>::const_iterator it = c.begin(); it != c.end(); ++it)
        {
            std::string path_name = it->second["path_name"].as_string();
            JsonValue meta = it->second["tensor_meta"];
            constants[it->first] = std::make_pair(path_name, meta);
        }
    }

    if (is_legacy)
    {
        // <2.8 container: there are no *config.json records, so weights and
        // constants above are empty; decode the two pickled state dicts into
        // the same maps and fill legacy_raw (see load_exportedprogram_legacy)
        if (pnnx_load_legacy_payloads(zip, names, root, weights, constants, legacy_raw) != 0)
            return -1;
    }

    const JsonValue& graph = root["graph_module"]["graph"];
    const JsonValue& signature = root["graph_module"]["signature"];

    // tensor_values : name -> { dtype, sizes, ... }
    const JsonValue& tensor_values = graph["tensor_values"];

    // some serde ScalarTypes have no pnnx representation (uint16=28, float8
    // variants, ...): reject them explicitly instead of decoding with the wrong
    // element size (uint16 as 1-byte u8 would silently truncate/misalign every
    // weight/constant using that dtype) or materializing a type-0 attribute
    {
        std::string reject_fqn;
        int64_t reject_dtype = -1;
        for (std::map<std::string, std::pair<std::string, JsonValue> >::const_iterator it = weights.begin(); it != weights.end(); ++it)
        {
            const JsonValue& meta = it->second.second;
            if (meta.is_object() && meta.has("dtype") && serde_dtype_to_pnnx_type(meta["dtype"].as_int()) == 0)
            {
                reject_fqn = it->first;
                reject_dtype = meta["dtype"].as_int();
                break;
            }
        }
        if (reject_fqn.empty())
        {
            for (std::map<std::string, std::pair<std::string, JsonValue> >::const_iterator it = constants.begin(); it != constants.end(); ++it)
            {
                const JsonValue& meta = it->second.second;
                if (meta.is_object() && meta.has("dtype") && serde_dtype_to_pnnx_type(meta["dtype"].as_int()) == 0)
                {
                    reject_fqn = it->first;
                    reject_dtype = meta["dtype"].as_int();
                    break;
                }
            }
        }
        if (!reject_fqn.empty())
        {
            if (reject_dtype == 28)
                fprintf(stderr, "unsupported dtype uint16 for tensor '%s'\n", reject_fqn.c_str());
            else
                fprintf(stderr, "unsupported dtype %lld for tensor '%s'\n", (long long)reject_dtype, reject_fqn.c_str());
            return -1;
        }
    }

    // name -> operand
    std::map<std::string, Operand*> operands_by_name;

    int constant_index = 0;
    int subop_index = 0;

    // pass 1 : build graph inputs
    const JsonValue& input_specs = signature["input_specs"];

    // materialize one attribute: the archive path reads raw bytes from the
    // outer zip; the legacy path reads the pre-decoded bytes from legacy_raw
    // (both then share load_tensor_from_raw for the meta view handling)
    auto load_weight_attr = [&](Attribute& a, const std::string& fqn, const JsonValue& meta, const char* dir, const std::string& path_name) -> int {
        if (is_legacy)
        {
            std::map<std::string, std::vector<char> >::const_iterator lit = legacy_raw.find(fqn);
            if (lit == legacy_raw.end())
            {
                fprintf(stderr, "legacy weight bytes for '%s' not found\n", fqn.c_str());
                return -1;
            }
            load_tensor_from_raw(lit->second, meta, a);
        }
        else
        {
            if (load_tensor_data(zip, names, dir, path_name, meta, a) != 0)
                return -1;
        }
        return 0;
    };

    int user_input_index = 0;
    for (size_t i = 0; i < input_specs.size(); i++)
    {
        const JsonValue& spec = input_specs[i];

        if (spec.has("parameter") || spec.has("buffer"))
        {
            const JsonValue& p = spec.has("parameter") ? spec["parameter"] : spec["buffer"];
            std::string graph_name = p["arg"]["name"].as_string();
            std::string fqn = spec.has("parameter") ? p["parameter_name"].as_string() : p["buffer_name"].as_string();

            Operator* op = g.new_operator("pnnx.Attribute", graph_name);

            Operand* r = g.new_operand(graph_name);
            r->producer = op;
            op->outputs.push_back(r);

            // weight data
            if (weights.find(fqn) != weights.end())
            {
                const std::string& path_name = weights[fqn].first;
                const JsonValue& meta = weights[fqn].second;

                Attribute a;
                a.type = read_dtype(meta);
                read_sizes(meta, a.shape);
                if (load_weight_attr(a, fqn, meta, "weights", path_name) != 0)
                    return -1;

                op->attrs["data"] = a;

                r->type = a.type;
                r->shape = a.shape;
            }
            else if (constants.find(fqn) != constants.end())
            {
                // non-persistent buffer data lives in constants, not weights
                const std::string& path_name = constants[fqn].first;
                const JsonValue& meta = constants[fqn].second;

                Attribute a;
                a.type = read_dtype(meta);
                read_sizes(meta, a.shape);
                if (load_weight_attr(a, fqn, meta, "constants", path_name) != 0)
                    return -1;

                op->attrs["data"] = a;

                r->type = a.type;
                r->shape = a.shape;
            }

            operands_by_name[graph_name] = r;
        }
        else if (spec.has("tensor_constant"))
        {
            const JsonValue& c = spec["tensor_constant"];
            std::string graph_name = c["arg"]["name"].as_string();
            std::string fqn = c["tensor_constant_name"].as_string();

            Operator* op = g.new_operator("pnnx.Attribute", graph_name);

            Operand* r = g.new_operand(graph_name);
            r->producer = op;
            op->outputs.push_back(r);

            if (constants.find(fqn) != constants.end())
            {
                const std::string& path_name = constants[fqn].first;
                const JsonValue& meta = constants[fqn].second;

                Attribute a;
                a.type = read_dtype(meta);
                read_sizes(meta, a.shape);
                if (load_weight_attr(a, fqn, meta, "constants", path_name) != 0)
                    return -1;

                op->attrs["data"] = a;

                r->type = a.type;
                r->shape = a.shape;
            }

            operands_by_name[graph_name] = r;
        }
        else if (spec.has("user_input"))
        {
            const JsonValue& u = spec["user_input"];
            std::string graph_name = u["arg"]["as_tensor"]["name"].as_string();

            Operator* op = g.new_operator("pnnx.Input", graph_name);

            Operand* r = g.new_operand(graph_name);
            r->producer = op;
            op->outputs.push_back(r);

            // shape/type from input_shapes override, or from tensor_values
            if (user_input_index < (int)input_shapes.size())
            {
                r->shape.clear();
                const std::vector<int64_t>& s = input_shapes[user_input_index];
                for (size_t j = 0; j < s.size(); j++)
                    r->shape.push_back((int)s[j]);
                if (user_input_index < (int)input_types.size())
                {
                    const std::string& t = input_types[user_input_index];
                    if (t == "f32")
                        r->type = 1;
                    else if (t == "f64")
                        r->type = 2;
                    else if (t == "f16")
                        r->type = 3;
                    else if (t == "i32")
                        r->type = 4;
                    else if (t == "i64")
                        r->type = 5;
                    else if (t == "i16")
                        r->type = 6;
                    else if (t == "i8")
                        r->type = 7;
                    else if (t == "u8")
                        r->type = 8;
                    else if (t == "bf16")
                        r->type = 13;
                    else if (t == "c32")
                        r->type = 12;
                    else if (t == "c64")
                        r->type = 10;
                    else if (t == "c128")
                        r->type = 11;
                    else if (t == "bool")
                        r->type = 9;
                }
            }
            else if (tensor_values.has(graph_name))
            {
                const JsonValue& meta = tensor_values[graph_name];
                r->type = read_dtype(meta);
                read_sizes(meta, r->shape);

                // without inputshape=, fall back to the static tensor_values
                // shape; if it is dynamic (sym int, recorded as -1) fail with a
                // clear message instead of crashing later
                for (size_t j = 0; j < r->shape.size(); j++)
                {
                    if (r->shape[j] == -1)
                    {
                        fprintf(stderr, "input '%s' has dynamic shape, please specify inputshape= explicitly\n", graph_name.c_str());
                        return -1;
                    }
                }
            }
            else
            {
                // no inputshape= and no static shape in tensor_values for this input
                fprintf(stderr, "input '%s' shape unknown, please specify inputshape= explicitly\n", graph_name.c_str());
                return -1;
            }

            user_input_index++;
            operands_by_name[graph_name] = r;
        }
        // TODO: constant_input / tensor_constant specs
    }

    // pass 2 : build nodes
    const JsonValue& nodes = graph["nodes"];
    for (size_t i = 0; i < nodes.size(); i++)
    {
        const JsonValue& nd = nodes[i];

        std::string target = nd["target"].as_string();
        std::string op_type = normalize_target(target);

        // map torchvision custom ops to pnnx op types (match pass_level1/pass_ncnn)
        if (op_type == "torchvision::deform_conv2d")
            op_type = "torchvision.ops.DeformConv2d";
        else if (op_type == "torchvision::roi_align")
            op_type = "torchvision.ops.RoIAlign";

        // map window function ops to torch API (args become params below, not inputs)
        if (op_type == "aten::hann_window")
            op_type = "torch.hann_window";
        else if (op_type == "aten::hamming_window")
            op_type = "torch.hamming_window";

        // dynamo metadata assertion ops (_assert_tensor_metadata etc.) have no
        // output and are pure noops, skip them
        if (op_type.compare(0, 14, "aten::_assert_") == 0)
            continue;

        // dynamo shape guard ops (_operator.ge/le etc., sym int/bool compares)
        // have no tensor output, skip them
        if (op_type == "_operator")
            continue;

        // higher_order ops (wrap_with_set_grad_enabled / wrap_with_autocast):
        // inline the subgraph, binding its placeholders to the captured closure
        // operands and mapping its results to the wrapper output names
        if (target.find("higher_order.wrap_with_set_grad_enabled") != std::string::npos
                || target.find("higher_order.wrap_with_autocast") != std::string::npos)
        {
            int ret = inline_wrapper_subgraph(g, nd, operands_by_name, constant_index, subop_index);
            if (ret != 0)
                return ret;
            continue;
        }

        char op_name[32];
        snprintf(op_name, 32, "op_%zu", i);

        Operator* op = g.new_operator(op_type, op_name);

        // inputs
        const JsonValue& inputs = nd["inputs"];
        std::vector<std::string> inputnames;

        if (op_type == "torchvision.ops.DeformConv2d" || op_type == "torchvision.ops.RoIAlign")
        {
            // torchvision custom ops: scalar args -> locals, weight/bias -> attrs,
            // the rest of the tensors (input/offset/mask/rois) -> inputs
            std::map<std::string, int> int_params;
            std::map<std::string, float> float_params;
            std::map<std::string, bool> bool_params;

            for (size_t j = 0; j < inputs.size(); j++)
            {
                const JsonValue& inp = inputs[j];
                std::string argname = inp["name"].as_string();
                const JsonValue& arg = inp["arg"];

                if (arg.has("as_int"))
                    int_params[argname] = (int)arg["as_int"].as_int();
                else if (arg.has("as_float"))
                    float_params[argname] = (float)arg["as_float"].as_double();
                else if (arg.has("as_bool"))
                    bool_params[argname] = arg["as_bool"].as_bool();
            }

            bool deform_use_mask = bool_params.count("use_mask") && bool_params["use_mask"];

            for (size_t j = 0; j < inputs.size(); j++)
            {
                const JsonValue& inp = inputs[j];
                std::string argname = inp["name"].as_string();
                const JsonValue& arg = inp["arg"];

                if (!arg.has("as_tensor"))
                    continue;

                std::string name = arg["as_tensor"]["name"].as_string();
                Operand* r = operands_by_name[name];
                if (!r)
                {
                    fprintf(stderr, "operand %s not found for node %s\n", name.c_str(), op_type.c_str());
                    return -1;
                }

                if (argname == "weight" || argname == "bias")
                {
                    // move weight/bias to attrs (their producer is a pnnx.Attribute)
                    if (r->producer && r->producer->type == "pnnx.Attribute" && r->producer->has_attr("data"))
                    {
                        op->attrs[argname] = r->producer->attrs["data"];
                        continue;
                    }
                    // a runtime-produced weight/bias cannot be folded into the
                    // ncnn layer attribute; reject explicitly instead of
                    // dropping the tensor (which would throw on attrs.at() for
                    // weight or silently omit bias and change the result)
                    fprintf(stderr, "unsupported dynamic %s for %s\n", argname.c_str(), op_type.c_str());
                    return -1;
                }

                if (op_type == "torchvision.ops.DeformConv2d" && argname == "mask" && !deform_use_mask)
                {
                    // when use_mask=False the mask is a constant zeros, unused
                    continue;
                }

                r->consumers.push_back(op);
                op->inputs.push_back(r);
                inputnames.push_back(argname);
            }

            if (op_type == "torchvision.ops.DeformConv2d")
            {
                const Attribute& w = op->attrs.at("weight");
                int groups = int_params["groups"];
                op->params["in_channels"] = w.shape[1] * groups;
                op->params["out_channels"] = w.shape[0];
                op->params["kernel_size"] = Parameter{w.shape[2], w.shape[3]};
                op->params["stride"] = Parameter{int_params["stride_h"], int_params["stride_w"]};
                op->params["padding"] = Parameter{int_params["pad_h"], int_params["pad_w"]};
                op->params["dilation"] = Parameter{int_params["dilation_h"], int_params["dilation_w"]};
                op->params["groups"] = groups;
                op->params["bias"] = op->has_attr("bias");
            }
            else
            {
                op->params["output_size"] = Parameter{int_params["pooled_height"], int_params["pooled_width"]};
                op->params["spatial_scale"] = float_params["spatial_scale"];
                op->params["sampling_ratio"] = int_params["sampling_ratio"];
                op->params["aligned"] = bool_params["aligned"];
            }

            if (!inputnames.empty())
                op->inputnames = inputnames;
        }
        else if (op_type == "torch.hann_window" || op_type == "torch.hamming_window")
        {
            // window function args become params directly (no constant input);
            // drop pin_memory (the torch API has no such argument)
            for (size_t j = 0; j < inputs.size(); j++)
            {
                const JsonValue& inp = inputs[j];
                std::string argname = inp["name"].as_string();
                const JsonValue& arg = inp["arg"];

                if (argname == "pin_memory")
                    continue;

                if (arg.has("as_int"))
                {
                    op->params[argname] = (int)arg["as_int"].as_int();
                }
                else if (arg.has("as_device"))
                {
                    std::string dev = arg["as_device"]["type"].as_string();
                    if (arg["as_device"].has("index") && !arg["as_device"]["index"].is_null())
                    {
                        char tmp[32];
                        snprintf(tmp, 32, ":%lld", (long long)arg["as_device"]["index"].as_int());
                        dev += tmp;
                    }
                    op->params[argname] = dev;
                }
                else if (arg.has("as_bool"))
                {
                    op->params[argname] = arg["as_bool"].as_bool();
                }
                else if (arg.has("as_scalar_type"))
                {
                    // hann/hamming_window carry a dtype override (e.g.
                    // float64); keep it so the level2 fold can honor it; an
                    // unrepresentable scalar dtype is rejected, not dropped
                    int pnnx_type = 0;
                    if (!scalar_dtype_to_pnnx_type(arg["as_scalar_type"].as_int(), pnnx_type))
                        return -1;
                    op->params[argname] = pnnx_type;
                }
                else if (arg.has("as_float"))
                {
                    // hamming_window's alpha/beta arrive as float scalars
                    op->params[argname] = (float)arg["as_float"].as_double();
                }
            }
        }
        else
        {
            for (size_t j = 0; j < inputs.size(); j++)
            {
                const JsonValue& inp = inputs[j];
                std::string argname = inp["name"].as_string();

                // dynamo's upsample .vec overload uses output_size/scale_factors;
                // the pnnx pattern uses size/scale_factor
                if (op_type.compare(0, 15, "aten::upsample_") == 0 || op_type.compare(0, 16, "aten::_upsample_") == 0)
                {
                    if (argname == "output_size")
                        argname = "size";
                    else if (argname == "scale_factors")
                        argname = "scale_factor";
                }
                const JsonValue& arg = inp["arg"];

                inputnames.push_back(argname);

                if (arg.has("as_tensor"))
                {
                    std::string name = arg["as_tensor"]["name"].as_string();
                    Operand* r = operands_by_name[name];
                    if (!r)
                    {
                        fprintf(stderr, "operand %s not found for node %s\n", name.c_str(), op_type.c_str());
                        return -1;
                    }
                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                }
                else if (arg.has("as_int"))
                {
                    // INT64_MAX/MIN are dynamo's "to the end" sentinels for slice
                    // etc., map to pnnx INT_MAX/INT_MIN
                    long long iv = arg["as_int"].as_int();
                    if (iv == std::numeric_limits<long long>::max())
                        iv = INT_MAX;
                    if (iv == std::numeric_limits<long long>::min())
                        iv = INT_MIN;
                    if (iv > INT_MAX || iv < INT_MIN)
                    {
                        // pnnx Parameter stores integers as int32; an exported
                        // 64-bit scalar beyond that range (e.g. a torch.full fill
                        // value of 1<<40) would be silently truncated by the
                        // Parameter(long long) narrowing and change the model
                        // result - reject it explicitly instead
                        fprintf(stderr, "unsupported 64-bit integer argument %lld in node %s\n", iv, op_type.c_str());
                        return -1;
                    }
                    new_constant(g, op, (int)iv, constant_index);
                }
                else if (arg.has("as_ints"))
                {
                    std::vector<int> ai;
                    for (size_t k = 0; k < arg["as_ints"].size(); k++)
                    {
                        long long v = arg["as_ints"][k].as_int();
                        if (v == std::numeric_limits<long long>::max())
                            v = INT_MAX;
                        if (v == std::numeric_limits<long long>::min())
                            v = INT_MIN;
                        if (v > INT_MAX || v < INT_MIN)
                        {
                            // see the as_int branch above: reject out-of-range
                            // 64-bit scalars instead of truncating them
                            fprintf(stderr, "unsupported 64-bit integer argument %lld in node %s\n", v, op_type.c_str());
                            return -1;
                        }
                        ai.push_back((int)v);
                    }
                    new_constant(g, op, ai, constant_index);
                }
                else if (arg.has("as_float"))
                {
                    new_constant(g, op, (float)arg["as_float"].as_double(), constant_index);
                }
                else if (arg.has("as_floats"))
                {
                    std::vector<float> af;
                    for (size_t k = 0; k < arg["as_floats"].size(); k++)
                        af.push_back((float)arg["as_floats"][k].as_double());
                    new_constant(g, op, af, constant_index);
                }
                else if (arg.has("as_bool"))
                {
                    new_constant(g, op, arg["as_bool"].as_bool(), constant_index);
                }
                else if (arg.has("as_string"))
                {
                    new_constant(g, op, arg["as_string"].as_string(), constant_index);
                }
                else if (arg.has("as_strings"))
                {
                    std::vector<std::string> as;
                    for (size_t k = 0; k < arg["as_strings"].size(); k++)
                        as.push_back(arg["as_strings"][k].as_string());
                    new_constant(g, op, as, constant_index);
                }
                else if (arg.has("as_none"))
                {
                    new_constant(g, op, Parameter(), constant_index);
                }
                else if (arg.has("as_scalar_type"))
                {
                    // serde ScalarType enum -> pnnx dtype input enum; reject a
                    // dtype with no pnnx representation instead of a bogus -1
                    int dtype_value = 0;
                    if (!scalar_dtype_to_pnnx_dtype_value(arg["as_scalar_type"].as_int(), dtype_value))
                        return -1;
                    new_constant(g, op, dtype_value, constant_index);
                }
                else if (arg.has("as_device"))
                {
                    // {"type":"cpu","index":null} / {"type":"cuda","index":0}
                    std::string dev = arg["as_device"]["type"].as_string();
                    if (arg["as_device"].has("index") && !arg["as_device"]["index"].is_null())
                    {
                        char tmp[32];
                        snprintf(tmp, 32, ":%lld", (long long)arg["as_device"]["index"].as_int());
                        dev += tmp;
                    }
                    new_constant(g, op, dev, constant_index);
                }
                else if (arg.has("as_layout"))
                {
                    new_constant(g, op, (int)arg["as_layout"].as_int(), constant_index);
                }
                else if (arg.has("as_memory_format"))
                {
                    new_constant(g, op, serde_memory_format_to_pnnx(arg["as_memory_format"].as_int()), constant_index);
                }
                else if (arg.has("as_complex"))
                {
                    // complex constant {"real": r, "imag": i}
                    float real = (float)arg["as_complex"]["real"].as_double();
                    float imag = (float)arg["as_complex"]["imag"].as_double();
                    new_constant(g, op, std::complex<float>(real, imag), constant_index);
                }
                else if (arg.has("as_tensors"))
                {
                    // tensor list -> prim::ListConstruct
                    char lc_name[32];
                    snprintf(lc_name, 32, "pnnx_list_%d", constant_index++);

                    // insert before the consumer to keep graph order topological
                    Operator* lc = g.new_operator_before("prim::ListConstruct", lc_name, op);
                    Operand* lr = g.new_operand(lc_name);
                    lr->producer = lc;
                    lc->outputs.push_back(lr);

                    for (size_t k = 0; k < arg["as_tensors"].size(); k++)
                    {
                        std::string name = arg["as_tensors"][k]["name"].as_string();
                        Operand* r = operands_by_name[name];
                        if (!r)
                        {
                            fprintf(stderr, "operand %s not found for list\n", name.c_str());
                            return -1;
                        }
                        r->consumers.push_back(lc);
                        lc->inputs.push_back(r);
                    }

                    lr->consumers.push_back(op);
                    op->inputs.push_back(lr);
                }
                else
                {
                    fprintf(stderr, "unsupported arg type for %s arg %s\n", op_type.c_str(), argname.c_str());
                    return -1;
                }
            }
        }

        // outputs
        const JsonValue& outputs = nd["outputs"];
        for (size_t j = 0; j < outputs.size(); j++)
        {
            if (outputs[j].has("as_tensor"))
            {
                std::string name = outputs[j]["as_tensor"]["name"].as_string();

                Operand* r = g.new_operand(name);
                r->producer = op;
                op->outputs.push_back(r);

                // shape/type from tensor_values
                if (tensor_values.has(name))
                {
                    const JsonValue& meta = tensor_values[name];
                    r->type = read_dtype(meta);
                    read_sizes(meta, r->shape);
                }

                operands_by_name[name] = r;
            }
            else if (outputs[j].has("as_tensors"))
            {
                // multiple outputs: one list output + prim::ListUnpack to split
                // pnnx convention: multi-output ops emit one list first, then
                // fuse_op1ton_unpack expands it
                char list_name[32];
                snprintf(list_name, 32, "%s_list", op_name);

                Operand* list_op = g.new_operand(list_name);
                list_op->producer = op;
                op->outputs.push_back(list_op);

                char lu_name[32];
                snprintf(lu_name, 32, "pnnx_unpack_%zu", i);
                Operator* lu = g.new_operator("prim::ListUnpack", lu_name);

                list_op->consumers.push_back(lu);
                lu->inputs.push_back(list_op);

                for (size_t k = 0; k < outputs[j]["as_tensors"].size(); k++)
                {
                    std::string name = outputs[j]["as_tensors"][k]["name"].as_string();

                    Operand* r = g.new_operand(name);
                    r->producer = lu;
                    lu->outputs.push_back(r);

                    // shape/type from tensor_values
                    if (tensor_values.has(name))
                    {
                        const JsonValue& meta = tensor_values[name];
                        r->type = read_dtype(meta);
                        read_sizes(meta, r->shape);
                    }

                    operands_by_name[name] = r;
                }
            }
        }

        if (!inputnames.empty())
            op->inputnames = inputnames;

        // append the default kwargs omitted by dynamo
        append_default_kwargs(g, op, op_type, inputnames, constant_index);
    }

    // torch.export functionalizes in-place buffer/input mutations by appending
    // the updated tensors to the raw graph outputs and tags them in the graph
    // signature. a static pnnx graph cannot express such runtime state updates,
    // so reject models that carry any non-user (mutation) output explicitly
    // instead of emitting those hidden values as public pnnx outputs.
    if (signature.has("output_specs"))
    {
        const JsonValue& output_specs = signature["output_specs"];
        for (size_t i = 0; i < output_specs.size(); i++)
        {
            if (!output_specs[i].has("user_output"))
            {
                fprintf(stderr, "unsupported exported program with buffer/input mutation outputs\n");
                return -1;
            }
        }
    }

    // pass 3 : build graph outputs
    const JsonValue& outputs = graph["outputs"];
    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (!outputs[i].has("as_tensor"))
        {
            fprintf(stderr, "unsupported exported program graph output %zu (not a plain tensor)\n", i);
            return -1;
        }
        std::string name = outputs[i]["as_tensor"]["name"].as_string();

        char op_name[32];
        snprintf(op_name, 32, "output_%zu", i);

        Operator* op = g.new_operator("pnnx.Output", op_name);

        Operand* r = operands_by_name[name];
        if (!r)
        {
            fprintf(stderr, "output operand %s not found\n", name.c_str());
            return -1;
        }

        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }

    zip.close();

    return 0;
}

} // namespace pnnx
