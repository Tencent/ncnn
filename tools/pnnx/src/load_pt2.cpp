// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"
#include "json.h"
#include "pass_pt2/pt2_operator.h"
#include "pass_pt2/pt2_tensor.h"
#include "pass_pt2/normalize_dtype.h"
#include "pass_pt2/fold_constants.h"
#include "pass_pt2.h"
#include "torch_alias.h"
#include "storezip.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

namespace pnnx {

static bool ends_with(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string direct_symbol(const std::string& expr)
{
    // keep the scale of direct symbolic dimensions for both shape samples
    if (expr.compare(0, 12, "Mul(Integer(") == 0 && ends_with(expr, "))"))
    {
        const size_t end = expr.find("), ", 12);
        if (end == std::string::npos || end == 12)
            return std::string();
        int64_t scale = 0;
        for (size_t i = 12; i < end; i++)
        {
            if (expr[i] < '0' || expr[i] > '9' || scale > INT_MAX / 10)
                return std::string();
            scale = scale * 10 + expr[i] - '0';
        }
        if (scale <= 0 || scale > INT_MAX)
            return std::string();
        const std::string symbol = direct_symbol(expr.substr(end + 3, expr.size() - end - 4));
        if (symbol.empty() || symbol.find('*') != std::string::npos)
            return std::string();
        return std::to_string(scale) + "*" + symbol;
    }
    const std::string prefix = "Symbol('";
    const std::string suffix = "', positive=True, integer=True)";
    if (expr.compare(0, prefix.size(), prefix) != 0 || !ends_with(expr, suffix))
        return std::string();
    std::string name = expr.substr(prefix.size(), expr.size() - prefix.size() - suffix.size());
    if (name.empty())
        return std::string();
    for (char c : name)
    {
        if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') && c != '_')
            return std::string();
    }
    return name;
}

static int read_entry(StoreZipReader& zip, const std::string& name, std::string& data)
{
    if (!zip.has_file(name))
    {
        fprintf(stderr, "pnnx: pt2 load: missing entry %s\n", name.c_str());
        return -1;
    }
    uint64_t size = zip.get_file_size(name);
    // bound metadata separately from tensor storage
    if (size > 256 * 1024 * 1024)
    {
        fprintf(stderr, "pnnx: pt2 load: metadata entry too large: %s\n", name.c_str());
        return -1;
    }
    data.resize((size_t)size);
    if (zip.read_file(name, data.empty() ? 0 : &data[0]) != 0)
    {
        fprintf(stderr, "pnnx: pt2 load: cannot read entry %s\n", name.c_str());
        return -1;
    }
    return 0;
}

static int read_json(StoreZipReader& zip, const std::string& name, JsonValue& value)
{
    std::string data;
    if (read_entry(zip, name, data) != 0)
        return -1;
    if (parse_json(data.data(), data.size(), value) != 0)
    {
        fprintf(stderr, "pnnx: pt2 load: invalid JSON entry %s\n", name.c_str());
        return -1;
    }
    return 0;
}

static void collect_archive_roots(StoreZipReader& zip, std::set<std::string>& roots, bool& legacy)
{
    const std::vector<std::string> names = zip.get_names();
    legacy = false;
    for (const std::string& name : names)
    {
        const size_t slash = name.find('/');
        if (name == "serialized_exported_program.json")
            legacy = true;
        if (slash == std::string::npos || slash == 0)
            continue;
        const std::string tail = name.substr(slash + 1);
        if (tail == "serialized_exported_program.json")
            legacy = true;
        if (tail == "archive_format" || (tail.compare(0, 7, "models/") == 0 && tail.find('/', 7) == std::string::npos && ends_with(tail, ".json")))
            roots.insert(name.substr(0, slash + 1));
    }
}

static int archive_root(StoreZipReader& zip, std::string& root, std::string& model)
{
    std::set<std::string> roots;
    bool legacy;
    collect_archive_roots(zip, roots, legacy);
    const std::vector<std::string> names = zip.get_names();
    if (roots.empty())
    {
        if (legacy)
        {
            fprintf(stderr, "pnnx: pt2 load: legacy pt2 archive is unsupported\n");
            return -1;
        }
        fprintf(stderr, "pnnx: pt2 load: no ExportedProgram archive found\n");
        return -1;
    }
    if (roots.size() != 1)
    {
        fprintf(stderr, "pnnx: pt2 load: conflicting pt2 archive roots\n");
        return -1;
    }
    root = *roots.begin();
    std::string format;
    if (read_entry(zip, root + "archive_format", format) != 0)
        return -1;
    if (format != "pt2")
    {
        fprintf(stderr, "pnnx: pt2 load: unsupported archive_format %s\n", format.c_str());
        return -1;
    }
    std::vector<std::string> models;
    for (const std::string& name : names)
    {
        const std::string prefix = root + "models/";
        if (name.compare(0, prefix.size(), prefix) == 0 && name.find('/', prefix.size()) == std::string::npos && ends_with(name, ".json"))
            models.push_back(name.substr(prefix.size(), name.size() - prefix.size() - 5));
    }
    if (models.size() != 1)
    {
        fprintf(stderr, "pnnx: pt2 load: pt2 requires exactly one ExportedProgram\n");
        return -1;
    }
    model = models[0];
    return 0;
}

static bool same_json(const JsonValue& a, const JsonValue& b)
{
    if (a.type != b.type)
        return false;
    if (a.type == JsonValue::Null)
        return true;
    if (a.type == JsonValue::Bool)
        return a.b == b.b;
    if (a.type == JsonValue::Int)
        return a.i == b.i;
    if (a.type == JsonValue::Float)
        return a.f == b.f;
    if (a.type == JsonValue::String)
        return a.s == b.s;
    if (a.type == JsonValue::Array)
    {
        if (a.a.size() != b.a.size())
            return false;
        for (size_t i = 0; i < a.a.size(); i++)
        {
            if (!same_json(a.a[i], b.a[i]))
                return false;
        }
        return true;
    }
    if (a.o.size() != b.o.size())
        return false;
    for (const auto& item : a.o)
    {
        const JsonValue* value = b.get(item.first);
        if (!value || !same_json(item.second, *value))
            return false;
    }
    return true;
}

static std::string value_name(const JsonValue& value)
{
    if (value.type != JsonValue::Object || value.o.size() != 1)
        return std::string();
    const JsonValue* tensor = value.get("as_tensor");
    const JsonValue* symbol = value.get("as_sym_int");
    if (!symbol)
        symbol = value.get("as_sym_bool");
    if (!symbol)
        symbol = value.get("as_sym_float");
    const JsonValue* name = 0;
    if (tensor)
        name = tensor->get("name");
    else if (symbol)
        name = symbol->get("as_name");
    return name && name->type == JsonValue::String ? name->s : std::string();
}

static void collect_references(const JsonValue& value, std::vector<std::string>& names)
{
    std::string name = value_name(value);
    if (!name.empty())
    {
        names.push_back(name);
        return;
    }
    if (value.type == JsonValue::Object)
    {
        if (value.get("as_graph"))
            return;
        const JsonValue* ref = value.get("as_name");
        if (value.o.size() == 1 && ref && ref->type == JsonValue::String)
        {
            names.push_back(ref->s);
            return;
        }
        const JsonValue* tensors = value.get("as_tensors");
        if (tensors && tensors->type == JsonValue::Array)
        {
            for (const JsonValue& t : tensors->a)
            {
                const JsonValue* n = t.get("name");
                if (n && n->type == JsonValue::String)
                    names.push_back(n->s);
            }
            return;
        }
        for (const auto& item : value.o)
        {
            collect_references(item.second, names);
        }
    }
    else if (value.type == JsonValue::Array)
        for (const JsonValue& item : value.a)
        {
            collect_references(item, names);
        }
}

static bool dense_layout(const Pt2TensorMeta& meta)
{
    if (meta.shape.size() != meta.strides.size())
        return false;
    std::vector<std::pair<int64_t, size_t> > axes;
    for (size_t i = 0; i < meta.shape.size(); i++)
    {
        axes.push_back(std::make_pair(meta.strides[i], i));
    }
    std::sort(axes.begin(), axes.end());
    int64_t count = 1;
    for (const auto& item : axes)
    {
        const size_t axis = item.second;
        const int n = meta.shape[axis];
        if (n <= 0 || (n > 1 && meta.strides[axis] != count) || count > INT64_MAX / n)
            return false;
        count *= n;
    }
    return true;
}

static bool contiguous_layout(const Pt2TensorMeta& meta)
{
    if (meta.storage_offset != 0 || meta.shape.size() != meta.strides.size())
        return false;
    int64_t stride = 1;
    for (size_t i = meta.shape.size(); i > 0; i--)
    {
        const int n = meta.shape[i - 1];
        if (n < 0 || meta.strides[i - 1] != stride || (n > 0 && stride > INT64_MAX / n))
            return false;
        stride *= std::max(n, 1);
    }
    return true;
}

static bool is_functionalizable(const std::string& type)
{
    static const std::set<std::string> functionalizable = {
        "aten::relu_", "aten::relu6_", "aten::hardtanh_", "aten::hardswish_",
        "aten::silu_", "aten::leaky_relu_", "aten::sigmoid_", "aten::hardsigmoid_",
        "aten::elu_", "aten::celu_", "aten::selu_", "aten::tanh_", "aten::threshold_", "aten::mish_",
        "aten::add_", "aten::sub_", "aten::mul_", "aten::div_", "aten::exp_",
        "aten::clamp_", "aten::masked_fill_", "aten::fill_", "aten::copy_", "aten::index_put_",
        "aten::transpose_", "aten::t_", "aten::squeeze_", "aten::unsqueeze_"
    };
    return functionalizable.count(type) != 0;
}

static const Operand* mutation_root(const Operand* r)
{
    while (torch_alias_kind(r->producer) == TorchAliasView || is_functionalizable(r->producer->type))
    {
        r = r->producer->inputs[0];
    }
    return r;
}

class Pt2Loader
{
public:
    struct MetadataVersion
    {
        Operand* value;
        Pt2TensorMeta metadata;
        bool known_layout;
    };

    Pt2Loader(Graph& graph)
        : g(graph), next_id(0), next_storage(0)
    {
    }

    const JsonValue* get_field(const JsonValue& value, const std::string& name, JsonValue::Type type)
    {
        const JsonValue* result = value.get(name);
        if (!result || result->type != type)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: missing or invalid %s\n", context.c_str(), name.c_str());
            return 0;
        }
        return result;
    }

    Operator* new_op(const std::string& type)
    {
        return g.new_operator(type, "pt2_" + std::to_string(next_id++));
    }

    Operand* new_value(Operator* op)
    {
        Operand* r = g.new_operand("v" + std::to_string(next_id++));
        r->producer = op;
        op->outputs.push_back(r);
        return r;
    }

    void add_input(Operator* op, Operand* r)
    {
        op->inputs.push_back(r);
        r->consumers.push_back(op);
    }

    Operand* new_constant(const Parameter& p)
    {
        Operator* op = new_op("prim::Constant");
        op->params["value"] = p;
        return new_value(op);
    }

    Operand* number_as_tensor(Operand* r, const JsonValue& arg)
    {
        Operator* op = new_op("torch.scalar_tensor");
        add_input(op, r);
        op->inputnames.push_back("s");
        Operand* result = new_value(op);
        if (arg.get("as_int") || arg.get("as_sym_int"))
        {
            op->params["dtype"] = "torch.int64";
            result->type = 5;
        }
        else if (arg.get("as_bool") || arg.get("as_sym_bool"))
        {
            op->params["dtype"] = "torch.bool";
            result->type = 9;
        }
        else if (arg.get("as_complex"))
        {
            op->params["dtype"] = "torch.complex64";
            result->type = 10;
        }
        else
        {
            op->params["dtype"] = "torch.float32";
            result->type = 1;
        }
        return result;
    }

    int define_value(const std::string& name, Operand* r, bool tensor)
    {
        if (name.empty() || values.count(name))
        {
            fprintf(stderr, "pnnx: pt2 load: %s: duplicate or empty value definition %s\n", context.c_str(), name.c_str());
            return -1;
        }
        values[name] = r;
        if (tensor)
        {
            auto it = metadata.find(name);
            if (it == metadata.end())
            {
                fprintf(stderr, "pnnx: pt2 load: %s: missing tensor metadata for %s\n", context.c_str(), name.c_str());
                return -1;
            }
            r->type = it->second.type;
            r->shape = it->second.shape;
        }
        else
            symbols.insert(name);
        return 0;
    }

    void set_provenance(const std::string& name, const std::set<int>& storage)
    {
        provenance[name] = storage;
        if (remaining[name] > 0)
            for (int id : storage)
            {
                live_references[id].insert(name);
            }
    }

    void consume_reference(const std::string& name)
    {
        auto use = remaining.find(name);
        if (use == remaining.end() || use->second == 0 || --use->second != 0)
            return;
        auto value = provenance.find(name);
        if (value == provenance.end())
            return;
        for (int id : value->second)
        {
            auto live = live_references.find(id);
            if (live == live_references.end())
                continue;
            live->second.erase(name);
            if (live->second.empty())
                live_references.erase(live);
        }
    }

    int int_parameter(int64_t value, Parameter& p, const std::string& target, const std::string& argname)
    {
        if (value < INT_MIN || value > INT_MAX)
        {
            if ((target == "aten::slice" || target == "aten::slice_scatter") && argname == "end" && value == INT64_MAX)
                value = INT_MAX;
            else
            {
                fprintf(stderr, "pnnx: pt2 load: %s: integer outside pnnx parameter range\n", context.c_str());
                return -1;
            }
        }
        p = (int)value;
        return 0;
    }

    int float_parameter(const JsonValue& value, Parameter& p, bool precise)
    {
        double f;
        if (value.type == JsonValue::Int)
            f = (double)value.i;
        else if (value.type == JsonValue::Float)
            f = value.f;
        else if (value.type == JsonValue::String && value.s == "NaN")
            f = std::numeric_limits<double>::quiet_NaN();
        else if (value.type == JsonValue::String && value.s == "Infinity")
            f = std::numeric_limits<double>::infinity();
        else if (value.type == JsonValue::String && value.s == "-Infinity")
            f = -std::numeric_limits<double>::infinity();
        else
        {
            fprintf(stderr, "pnnx: pt2 load: %s: invalid floating point argument\n", context.c_str());
            return -1;
        }
        float narrowed = (float)f;
        if (std::isfinite(f) && (!std::isfinite(narrowed) || (precise && (double)narrowed != f)))
        {
            fprintf(stderr, "pnnx: pt2 load: %s: floating point parameter cannot be represented faithfully\n", context.c_str());
            return -1;
        }
        p = narrowed;
        return 0;
    }

    Operand* load_argument(const JsonValue& arg, const std::string& target, const std::string& argname, bool precise = false)
    {
        if (arg.type != JsonValue::Object || arg.o.size() != 1)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: invalid Argument union\n", context.c_str());
            return 0;
        }
        const std::string tag = arg.o.begin()->first;
        const JsonValue& v = arg.o.begin()->second;
        std::string name = value_name(arg);
        if (!name.empty())
        {
            auto it = values.find(name);
            if (it == values.end())
            {
                fprintf(stderr, "pnnx: pt2 load: %s: undefined value %s\n", context.c_str(), name.c_str());
                return 0;
            }
            if ((tag == "as_tensor") == (symbols.count(name) != 0))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: value kind mismatch %s\n", context.c_str(), name.c_str());
                return 0;
            }
            return it->second;
        }
        Parameter p;
        if (tag == "as_none" && v.type == JsonValue::Bool && v.b)
            return new_constant(p);
        if (tag == "as_int" && v.type == JsonValue::Int)
        {
            if (int_parameter(v.i, p, target, argname) != 0)
                return 0;
        }
        else if (tag == "as_float")
        {
            if (float_parameter(v, p, precise) != 0)
                return 0;
        }
        else if (tag == "as_complex")
        {
            const JsonValue* real = v.get("real");
            const JsonValue* imag = v.get("imag");
            Parameter re, im;
            if (!real || !imag || float_parameter(*real, re, precise) != 0 || float_parameter(*imag, im, precise) != 0)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid complex argument\n", context.c_str());
                return 0;
            }
            p = std::complex<float>(re.f, im.f);
        }
        else if (tag == "as_bool" && v.type == JsonValue::Bool)
            p = v.b;
        else if (tag == "as_string" && v.type == JsonValue::String)
        {
            // keep strings representable in Python literals
            if (v.s.find_first_of("'\\\n\r") != std::string::npos)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: string is not representable by the Python writer\n", context.c_str());
                return 0;
            }
            p = v.s;
        }
        else if (tag == "as_device")
        {
            const JsonValue* type = get_field(v, "type", JsonValue::String);
            if (!type)
                return 0;
            if (type->s != "cpu")
            {
                fprintf(stderr, "pnnx: pt2 load: %s: only CPU devices are supported\n", context.c_str());
                return 0;
            }
            p = "cpu";
        }
        else if (tag == "as_scalar_type" && v.type == JsonValue::Int)
        {
            if (v.i < 1 || v.i > 13)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: unsupported scalar type\n", context.c_str());
                return 0;
            }
            p = normalize_pt2_scalar_type(pt2_scalar_type(pt2_tensor_type((int)v.i)));
        }
        else if (tag == "as_memory_format" && v.type == JsonValue::Int)
        {
            // serde contiguous/channels_last/channels_last_3d/preserve -> c10 MemoryFormat
            static const int formats[] = {-1, 0, 2, 3, 1};
            if (v.i < 1 || v.i > 4)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: unsupported memory format\n", context.c_str());
                return 0;
            }
            p = formats[v.i];
        }
        else if (tag == "as_layout" && v.type == JsonValue::Int)
        {
            // serde Strided=7, c10 Strided=0
            if (v.i != 7)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: only strided layout is supported\n", context.c_str());
                return 0;
            }
            p = 0;
        }
        else if (tag == "as_sym_int" || tag == "as_sym_bool" || tag == "as_sym_float")
            return load_argument(v, target, argname, precise);
        else if ((tag == "as_ints" || tag == "as_floats" || tag == "as_strings" || tag == "as_bools") && v.type == JsonValue::Array)
        {
            p.type = 5;
            if (tag == "as_floats")
                p.type = 6;
            else if (tag == "as_strings")
                p.type = 7;
            for (const JsonValue& item : v.a)
            {
                Parameter e;
                if (tag == "as_ints")
                {
                    if (item.type != JsonValue::Int || int_parameter(item.i, e, target, argname) != 0)
                    {
                        fprintf(stderr, "pnnx: pt2 load: %s: invalid integer list\n", context.c_str());
                        return 0;
                    }
                    p.ai.push_back(e.i);
                }
                else if (tag == "as_floats")
                {
                    if (float_parameter(item, e, precise) != 0)
                        return 0;
                    p.af.push_back(e.f);
                }
                else if (tag == "as_strings" && item.type == JsonValue::String)
                {
                    if (item.s.find_first_of("'\\\n\r") != std::string::npos)
                    {
                        fprintf(stderr, "pnnx: pt2 load: %s: string list is not representable by the Python writer\n", context.c_str());
                        return 0;
                    }
                    p.as.push_back(item.s);
                }
                else if (tag == "as_bools" && item.type == JsonValue::Bool)
                    p.ai.push_back(item.b ? 1 : 0);
                else
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: invalid list element\n", context.c_str());
                    return 0;
                }
            }
        }
        else if ((tag == "as_tensors" || tag == "as_optional_tensors" || tag == "as_sym_ints") && v.type == JsonValue::Array)
        {
            std::vector<Operand*> list;
            for (const JsonValue& item : v.a)
            {
                JsonValue a;
                a.type = JsonValue::Object;
                if (tag == "as_tensors")
                    a.o["as_tensor"] = item;
                else if (tag == "as_sym_ints")
                    a.o["as_sym_int"] = item;
                else
                    a = item;
                Operand* r = load_argument(a, target, argname, precise);
                if (!r)
                    return 0;
                list.push_back(r);
            }
            Operator* op = new_op("prim::ListConstruct");
            for (Operand* r : list)
            {
                add_input(op, r);
            }
            return new_value(op);
        }
        else if (tag == "as_name" && v.type == JsonValue::String && symbols.count(v.s))
            return values.at(v.s);
        else
        {
            fprintf(stderr, "pnnx: pt2 load: %s: unsupported Argument %s\n", context.c_str(), tag.c_str());
            return 0;
        }
        Operand* r = new_constant(p);
        if (tag == "as_float" && (v.type == JsonValue::Float || v.type == JsonValue::Int))
            floating_constants[r] = v.type == JsonValue::Float ? v.f : (double)v.i;
        return r;
    }

    int treespec_node(const JsonValue& node, size_t& leaves)
    {
        const JsonValue* type = node.get("type");
        const JsonValue* children = get_field(node, "children_spec", JsonValue::Array);
        if (!children)
            return -1;
        if (!type)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: missing treespec type\n", context.c_str());
            return -1;
        }
        if (type->type == JsonValue::Null)
        {
            if (!children->a.empty())
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid treespec leaf\n", context.c_str());
                return -1;
            }
            leaves++;
            return 0;
        }
        if (type->type != JsonValue::String || (type->s != "builtins.tuple" && type->s != "builtins.list" && type->s != "builtins.dict"))
        {
            fprintf(stderr, "pnnx: pt2 load: %s: unsupported custom PyTree\n", context.c_str());
            return -1;
        }
        if (type->s == "builtins.dict")
        {
            const JsonValue* c = get_field(node, "context", JsonValue::String);
            if (!c)
                return -1;
            JsonValue keys;
            if (parse_json(c->s.data(), c->s.size(), keys) != 0 || keys.type != JsonValue::Array || keys.a.size() != children->a.size())
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid dict treespec context\n", context.c_str());
                return -1;
            }
        }
        for (const JsonValue& child : children->a)
        {
            if (treespec_node(child, leaves) != 0)
                return -1;
        }
        return 0;
    }

    int treespec(const std::string& text, size_t expected)
    {
        JsonValue spec;
        if (parse_json(text.data(), text.size(), spec) != 0 || spec.type != JsonValue::Array || spec.a.size() != 2 || spec.a[0].type != JsonValue::Int || spec.a[0].i != 1)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: invalid treespec\n", context.c_str());
            return -1;
        }
        size_t leaves = 0;
        if (treespec_node(spec.a[1], leaves) != 0)
            return -1;
        if (leaves != expected)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: treespec leaf count mismatch\n", context.c_str());
            return -1;
        }
        return 0;
    }

    int specialize_shape(Pt2TensorMeta& meta)
    {
        for (size_t i = 0; i < meta.shape.size(); i++)
        {
            if (meta.shape[i] >= 0 || meta.shape_symbols[i].empty())
                continue;
            std::string symbol = meta.shape_symbols[i];
            int64_t scale = 1;
            const size_t star = symbol.find('*');
            if (star != std::string::npos)
            {
                scale = atoi(symbol.substr(0, star).c_str());
                symbol = symbol.substr(star + 1);
            }
            const auto value = shape_bindings.find(symbol);
            if (value == shape_bindings.end())
                continue;
            if (value->second > INT_MAX / scale)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: specialized tensor dimension exceeds pnnx range\n", context.c_str());
                return -1;
            }
            meta.shape[i] = (int)(value->second * scale);
        }
        return 0;
    }

    int specialize_shapes(const std::vector<std::vector<int64_t> >& supplied)
    {
        for (size_t i = 0; i < runtime_inputs.size(); i++)
        {
            const Pt2TensorMeta& meta = metadata.at(runtime_inputs[i]);
            for (size_t j = 0; j < meta.shape.size(); j++)
            {
                if (meta.shape[j] >= 0)
                    continue;
                const std::string& symbol = meta.shape_symbols[j];
                const size_t star = symbol.find('*');
                if (star == std::string::npos)
                    shape_bindings[symbol] = supplied[i][j];
                else
                    shape_bindings[symbol.substr(star + 1)] = supplied[i][j] / atoi(symbol.substr(0, star).c_str());
            }
        }
        for (auto& item : metadata)
        {
            if (specialize_shape(item.second) != 0)
                return -1;
        }
        for (const std::string& name : runtime_inputs)
        {
            values.at(name)->shape = metadata.at(name).shape;
        }
        return 0;
    }

    int check_input_shapes(const std::vector<std::vector<int64_t> >& supplied, const std::vector<std::string>& types, const JsonValue& ranges, bool required)
    {
        if (supplied.empty() && !required)
            return 0;
        if (supplied.size() != runtime_inputs.size() || types.size() != supplied.size())
        {
            fprintf(stderr, "pnnx: pt2 load: %s: input shape/type count mismatch\n", context.c_str());
            return -1;
        }
        std::map<std::string, int64_t> bound;
        static const char* type_names[] = {"null", "f32", "f64", "f16", "i32", "i64", "i16", "i8", "u8", "bool", "c64", "c128", "c32", "bf16"};
        for (size_t i = 0; i < supplied.size(); i++)
        {
            const Pt2TensorMeta& meta = metadata.at(runtime_inputs[i]);
            if (types[i] != type_names[meta.type] || supplied[i].size() != meta.shape.size())
            {
                fprintf(stderr, "pnnx: pt2 load: %s: input dtype/rank mismatch\n", context.c_str());
                return -1;
            }
            for (size_t j = 0; j < supplied[i].size(); j++)
            {
                int64_t n = supplied[i][j];
                if (n < 0 || n > INT_MAX || (meta.shape[j] >= 0 && n != meta.shape[j]))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: input static dimension mismatch\n", context.c_str());
                    return -1;
                }
                if (meta.shape[j] >= 0)
                    continue;
                const std::string& symbol = meta.shape_symbols[j];
                if (symbol.empty())
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: unsupported complex input shape expression\n", context.c_str());
                    return -1;
                }
                std::string base = symbol;
                int64_t value = n;
                const size_t star = symbol.find('*');
                if (star != std::string::npos)
                {
                    const int scale = atoi(symbol.substr(0, star).c_str());
                    if (n % scale != 0)
                    {
                        fprintf(stderr, "pnnx: pt2 load: %s: input violates derived dimension %s\n", context.c_str(), symbol.c_str());
                        return -1;
                    }
                    base = symbol.substr(star + 1);
                    value = n / scale;
                }
                if (bound.count(base) && bound[base] != value)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: shared input symbol mismatch %s\n", context.c_str(), base.c_str());
                    return -1;
                }
                bound[base] = value;
                const JsonValue* range = ranges.get(symbol);
                if (!range)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: missing range constraint for %s\n", context.c_str(), symbol.c_str());
                    return -1;
                }
                const JsonValue* min = range->get("min_val");
                const JsonValue* max = range->get("max_val");
                if (!min || !max)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: invalid range constraint\n", context.c_str());
                    return -1;
                }
                if (min->type == JsonValue::Int && n < min->i)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: input below range minimum %s\n", context.c_str(), symbol.c_str());
                    return -1;
                }
                if (max->type == JsonValue::Int && n > max->i)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: input above range maximum %s\n", context.c_str(), symbol.c_str());
                    return -1;
                }
                if ((min->type != JsonValue::Int && min->type != JsonValue::Null && !(min->type == JsonValue::String && min->s == "-Infinity"))
                        || (max->type != JsonValue::Int && max->type != JsonValue::Null && !(max->type == JsonValue::String && max->s == "Infinity")))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: unsupported range bound\n", context.c_str());
                    return -1;
                }
                if (base != symbol)
                {
                    const JsonValue* base_range = ranges.get(base);
                    const JsonValue* lower = base_range ? base_range->get("min_val") : 0;
                    const JsonValue* upper = base_range ? base_range->get("max_val") : 0;
                    if (!lower || !upper)
                    {
                        fprintf(stderr, "pnnx: pt2 load: %s: missing range constraint for %s\n", context.c_str(), base.c_str());
                        return -1;
                    }
                    if ((lower->type == JsonValue::Int && value < lower->i) || (upper->type == JsonValue::Int && value > upper->i))
                    {
                        fprintf(stderr, "pnnx: pt2 load: %s: input violates derived symbol range %s\n", context.c_str(), base.c_str());
                        return -1;
                    }
                }
            }
        }
        return 0;
    }

    const MetadataVersion* metadata_version(const Operand* value) const
    {
        if (tensor_objects.count(value))
            value = tensor_objects.at(value);
        const auto version = metadata_versions.find(value);
        return version == metadata_versions.end() ? 0 : &version->second;
    }

    int check_mutation(const Pt2Operator& bound)
    {
        const std::string& type = bound.type;
        const bool metadata_mutation = torch_mutation_kind(type) == TorchMutationMetadata;
        if (!is_functionalizable(type) || bound.writes.size() != 1 || bound.writes[0] != 0 || bound.outputs.size() != 1)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: unsupported mutable operator\n", context.c_str());
            return -1;
        }
        const std::string n = value_name(bound.inputs[bound.writes[0]].value);
        if (n.empty() || provenance[n].empty())
        {
            fprintf(stderr, "pnnx: pt2 load: %s: cannot prove fresh storage for mutation\n", context.c_str());
            return -1;
        }
        const Operand* destination = values.at(n);
        while (is_functionalizable(destination->producer->type))
        {
            destination = destination->producer->inputs[0];
        }
        const std::set<int>& storage = provenance[n];
        for (int id : storage)
        {
            if (external.count(id) && !(metadata_mutation && (torch_alias_kind(destination->producer) == TorchAliasView)))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: mutable operator writes external tensor storage\n", context.c_str());
                return -1;
            }
        }
        const Operand* root = mutation_root(values.at(n));
        // check the destination even at its last use
        std::set<std::string> aliases = {n};
        for (int id : storage)
        {
            auto live = live_references.find(id);
            if (live != live_references.end())
                aliases.insert(live->second.begin(), live->second.end());
        }
        for (const std::string& alias : aliases)
        {
            // write back through supported views or a local dense root
            const Operand* r = mutation_root(values.at(alias));
            if (unsupported_aliases.count(r) || unsupported_aliases.count(root))
            {
                // use a dense local root only when all live aliases are its slice/select views
                if (r != root || !dense_aliases.count(root))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: mutation has an unsupported observable alias\n", context.c_str());
                    return -1;
                }
            }
        }
        return 0;
    }

    int add_node_outputs(Operator* op, const Pt2Operator& bound, TorchAliasKind alias_kind, bool known_layout, bool contiguous_allocation, const Operand* object)
    {
        const std::string& type = bound.type;
        const bool inplace = !bound.writes.empty() && alias_kind != TorchAliasIdentity;
        const bool metadata_mutation = torch_mutation_kind(type) == TorchMutationMetadata;
        for (size_t i = 0; i < bound.outputs.size(); i++)
        {
            const JsonValue& output = bound.outputs[i];
            if (output.get("as_none"))
            {
                // preserve unnamed Tensor return slots in schema order
                new_value(op);
                continue;
            }
            std::set<int> storage;
            // use proven conditional aliases, otherwise retain schema aliases
            const bool classified_alias = torch_alias_kind(type) != TorchAliasNone;
            if (alias_kind == TorchAliasView || alias_kind == TorchAliasUntracked)
                storage = provenance[value_name(bound.inputs[0].value)];
            if (!classified_alias && i < bound.aliases.size())
            {
                for (size_t alias : bound.aliases[i])
                {
                    std::vector<std::string> refs;
                    collect_references(bound.inputs[alias].value, refs);
                    for (const std::string& ref : refs)
                    {
                        storage.insert(provenance[ref].begin(), provenance[ref].end());
                    }
                }
            }
            if (storage.empty())
                storage.insert(next_storage++);
            const JsonValue* list = output.get("as_tensors");
            if (list)
            {
                Operand* container = new_value(op);
                Operator* unpack = new_op("prim::ListUnpack");
                add_input(unpack, container);
                for (const JsonValue& t : list->a)
                {
                    const JsonValue* name_value = get_field(t, "name", JsonValue::String);
                    if (!name_value)
                        return -1;
                    const std::string& name = name_value->s;
                    if (define_value(name, new_value(unpack), true) != 0)
                        return -1;
                    set_provenance(name, storage);
                    if (alias_kind == TorchAliasUntracked || (!classified_alias && i < bound.aliases.size() && !bound.aliases[i].empty()))
                    {
                        unsupported_aliases.insert(values.at(name));
                        if (dense_layout(metadata.at(name)))
                            dense_aliases.insert(values.at(name));
                    }
                }
            }
            else
            {
                const std::string name = value_name(output);
                if (define_value(name, new_value(op), output.get("as_tensor") != 0) != 0)
                    return -1;
                set_provenance(name, storage);
                if (known_layout && output.get("as_tensor"))
                {
                    const Pt2TensorMeta& meta = metadata.at(name);
                    if (meta.storage_offset >= 0
                            && std::find(meta.shape.begin(), meta.shape.end(), -1) == meta.shape.end()
                            && std::find(meta.strides.begin(), meta.strides.end(), -1) == meta.strides.end()
                            && (!contiguous_allocation || contiguous_layout(meta)))
                        known_layouts.insert(name);
                }
                if (inplace)
                {
                    tensor_objects[values.at(name)] = object;
                    if (metadata_mutation)
                        metadata_versions[object] = {values.at(name), metadata.at(name), known_layouts.count(name) != 0};
                }
                if (!inplace && (alias_kind == TorchAliasUntracked || (!classified_alias && i < bound.aliases.size() && !bound.aliases[i].empty())))
                {
                    unsupported_aliases.insert(values.at(name));
                    if (dense_layout(metadata.at(name)))
                        dense_aliases.insert(values.at(name));
                }
            }
        }
        return 0;
    }

    int load_nodes(const JsonValue& nodes, const JsonValue& outputs)
    {
        for (const JsonValue& n : nodes.a)
        {
            const JsonValue* in = get_field(n, "inputs", JsonValue::Array);
            if (!in)
                return -1;
            for (const JsonValue& a : in->a)
            {
                const JsonValue* arg = get_field(a, "arg", JsonValue::Object);
                if (!arg)
                    return -1;
                std::vector<std::string> refs;
                collect_references(*arg, refs);
                for (const std::string& ref : refs)
                {
                    remaining[ref]++;
                }
            }
        }
        for (const JsonValue& out : outputs.a)
        {
            remaining[value_name(out)]++;
        }

        // discard export contract assertions before binding their arguments
        std::vector<bool> skip(nodes.a.size(), false);
        for (size_t i = nodes.a.size(); i > 0; i--)
        {
            const JsonValue& n = nodes.a[i - 1];
            const JsonValue* target = get_field(n, "target", JsonValue::String);
            const JsonValue* out = get_field(n, "outputs", JsonValue::Array);
            if (!target || !out)
                return -1;
            const std::string& type = target->s;
            bool discard = type == "torch.ops.aten._assert_scalar.default"
                           || type == "torch.ops.aten._assert_tensor_metadata.default"
                           || type == "torch.ops.aten._assert_async.default"
                           || type == "torch.ops.aten._assert_async.msg"
                           || type == "torch.ops.aten.sym_constrain_range.default"
                           || type == "torch.ops.aten.sym_constrain_range_for_size.default";
            if (discard && !out->a.empty())
            {
                fprintf(stderr, "pnnx: pt2 load: %s: unexpected assertion outputs\n", context.c_str());
                return -1;
            }
            const bool scalar = type == "_operator.add" || type == "_operator.sub" || type == "_operator.mul" || type == "_operator.floordiv"
                                || type == "_operator.ge" || type == "_operator.gt" || type == "_operator.le" || type == "_operator.lt"
                                || type == "_operator.eq" || type == "_operator.ne" || type == "_operator.or_"
                                || type == "torch.ops.aten.item.default" || type == "torch.ops.aten.sym_size.int"
                                || type == "torch.ops.aten.stride.int" || type == "torch.ops.aten.sym_stride.int"
                                || type == "torch.ops.aten.storage_offset.default" || type == "torch.ops.aten.sym_storage_offset.default";
            if (scalar && out->a.size() == 1 && (out->a[0].get("as_sym_int") || out->a[0].get("as_sym_float") || out->a[0].get("as_sym_bool")))
            {
                const std::string name = value_name(out->a[0]);
                if (!name.empty() && remaining[name] == 0)
                    discard = true;
            }
            if (!discard)
                continue;
            skip[i - 1] = true;
            std::vector<std::string> refs;
            collect_references(*n.get("inputs"), refs);
            for (const std::string& ref : refs)
            {
                remaining[ref]--;
            }
        }
        // refresh references after counting input and capture uses
        for (const auto& value : provenance)
        {
            if (remaining[value.first] > 0)
                for (int id : value.second)
                {
                    live_references[id].insert(value.first);
                }
        }
        for (size_t i = 0; i < nodes.a.size(); i++)
        {
            if (!skip[i] && load_node(nodes.a[i]) != 0)
                return -1;
        }
        return 0;
    }

    int load_node(const JsonValue& serialized)
    {
        const JsonValue* target_value = get_field(serialized, "target", JsonValue::String);
        const JsonValue* name = get_field(serialized, "name", JsonValue::String);
        if (!target_value || !name)
            return -1;
        const std::string& target = target_value->s;
        context = "node " + name->s + " (" + target + ")";
        if (target == "torch.ops.higher_order.wrap_with_set_grad_enabled" || target == "torch.ops.higher_order.wrap_with_autocast")
            return load_inference_subgraph(serialized, target);
        Pt2Operator bound;
        bool shape_comparison = false;
        if (target.compare(0, 10, "_operator.") == 0)
        {
            const std::string kind = target.substr(10);
            shape_comparison = kind == "ge" || kind == "gt" || kind == "le" || kind == "lt" || kind == "eq" || kind == "ne";
            const bool boolean = kind == "or_";
            if (!shape_comparison && !boolean && kind != "add" && kind != "sub" && kind != "mul" && kind != "floordiv")
            {
                fprintf(stderr, "pnnx: pt2 load: %s: unsupported symbolic operator\n", context.c_str());
                return -1;
            }
            if (boolean)
                bound.type = "aten::__or__";
            else if (kind == "floordiv")
                bound.type = "aten::floor_divide";
            else
                bound.type = "aten::" + kind;
            const JsonValue* in = get_field(serialized, "inputs", JsonValue::Array);
            const JsonValue* out = get_field(serialized, "outputs", JsonValue::Array);
            if (!in || !out)
                return -1;
            if (in->a.size() != 2 || out->a.size() != 1 || !out->a[0].get(shape_comparison || boolean ? "as_sym_bool" : "as_sym_int"))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid symbolic operator signature\n", context.c_str());
                return -1;
            }
            for (const JsonValue& a : in->a)
            {
                const JsonValue* name = get_field(a, "name", JsonValue::String);
                const JsonValue* value = get_field(a, "arg", JsonValue::Object);
                if (!name || !value)
                    return -1;
                Pt2Argument arg;
                arg.name = name->s;
                arg.value = *value;
                const std::string n = value_name(arg.value);
                const bool literal = boolean ? arg.value.get("as_bool") != 0
                                     : arg.value.get("as_int") || (shape_comparison && arg.value.get("as_float"));
                const bool symbolic = boolean ? arg.value.get("as_sym_bool") != 0
                                      : arg.value.get("as_sym_int") || (shape_comparison && arg.value.get("as_sym_float"));
                if (!literal && (!symbolic || n.empty() || !symbols.count(n)))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: symbolic operator argument kind mismatch\n", context.c_str());
                    return -1;
                }
                bound.inputs.push_back(arg);
            }
            if (kind == "floordiv")
            {
                const JsonValue* divisor = bound.inputs[1].value.get("as_int");
                if (!divisor || divisor->i <= 0)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: shape floor division requires a positive literal divisor\n", context.c_str());
                    return -1;
                }
            }
            bound.outputs = out->a;
            bound.aliases.resize(1);
        }
        else
        {
            if (bind_pt2_operator(serialized, bound) != 0)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: operator binding failed\n", context.c_str());
                return -1;
            }
        }
        // reject hidden state effects even for immutable schemas
        const std::string& type = bound.type;
        if (type.find("rand") != std::string::npos || type.find("bernoulli") != std::string::npos
                || type.find("normal_") != std::string::npos || type.find("multinomial") != std::string::npos
                || type == "aten::normal" || type == "aten::poisson" || type == "aten::_fused_dropout" || type == "aten::native_dropout"
                || type.find("backward") != std::string::npos)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: random or training operation is unsupported\n", context.c_str());
            return -1;
        }
        for (const Pt2Argument& a : bound.inputs)
        {
            if ((a.name == "training" || a.name == "train") && a.value.get("as_bool") && a.value.get("as_bool")->b)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: training operation is unsupported\n", context.c_str());
                return -1;
            }
            if (a.name == "dropout_p")
            {
                Parameter p;
                const JsonValue* v = a.value.get("as_float");
                if (!v || float_parameter(*v, p, false) != 0 || p.f != 0.f)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: random attention dropout is unsupported\n", context.c_str());
                    return -1;
                }
            }
        }
        if (type == "aten::instance_norm" && bound.inputs[5].value.get("as_bool") && bound.inputs[5].value.get("as_bool")->b
                && (!bound.inputs[3].value.get("as_none") || !bound.inputs[4].value.get("as_none")))
        {
            fprintf(stderr, "pnnx: pt2 load: %s: instance_norm running statistics update is unsupported\n", context.c_str());
            return -1;
        }
        // count references before graph or alias simplification
        for (const Pt2Argument& a : bound.inputs)
        {
            std::vector<std::string> refs;
            collect_references(a.value, refs);
            for (const std::string& ref : refs)
            {
                if (!values.count(ref))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: undefined input %s\n", context.c_str(), ref.c_str());
                    return -1;
                }
                consume_reference(ref);
            }
        }
        if (bound.outputs.empty())
        {
            fprintf(stderr, "pnnx: pt2 load: %s: unsupported operation without outputs\n", context.c_str());
            return -1;
        }
        std::vector<Operand*> inputs;
        // preserve double precision scalar thresholds; tensor factory fills use FP32
        const bool precise = shape_comparison || (bound.outputs.size() == 1 && bound.outputs[0].get("as_sym_bool") && (type == "aten::ge" || type == "aten::gt" || type == "aten::le" || type == "aten::lt" || type == "aten::eq" || type == "aten::ne"));
        for (const Pt2Argument& a : bound.inputs)
        {
            Operand* r = load_argument(a.value, type, a.name, precise);
            if (!r)
                return -1;
            inputs.push_back(r);
        }
        // keep one Tensor operand to preserve bool arithmetic and scalar Tensor returns
        bool has_tensor = false;
        for (size_t i : bound.tensor_inputs)
        {
            if (inputs[i]->type != 0)
                has_tensor = true;
        }
        if (!has_tensor && !bound.tensor_inputs.empty())
        {
            const size_t i = bound.tensor_inputs[0];
            inputs[i] = number_as_tensor(inputs[i], bound.inputs[i].value);
        }
        // copy conversion and broadcasting require a Tensor source
        if ((type == "aten::copy" || type == "aten::copy_") && inputs[1]->type == 0)
            inputs[1] = number_as_tensor(inputs[1], bound.inputs[1].value);
        const Operand* object = inputs.empty() ? 0 : inputs[0];
        if (tensor_objects.count(object))
            object = tensor_objects.at(object);
        const std::string source_name = bound.inputs.empty() ? std::string() : value_name(bound.inputs[0].value);
        const MetadataVersion* version = metadata_version(object);
        const bool layout_query = type == "aten::as_strided" || type == "aten::stride" || type == "aten::sym_stride"
                                  || type == "aten::storage_offset" || type == "aten::sym_storage_offset";
        // use current object geometry; independent views keep their original metadata
        if ((layout_query || type == "aten::size" || type == "aten::sym_size") && version)
            inputs[0] = version->value;
        int output_type = 0;
        const auto output_meta = metadata.find(value_name(bound.outputs[0]));
        if (output_meta != metadata.end())
            output_type = output_meta->second.type;
        const TorchAliasKind alias_kind = torch_alias_kind(type, inputs, output_type);
        const bool identity = alias_kind == TorchAliasIdentity;
        const bool inplace = !bound.writes.empty() && !identity;
        const bool metadata_mutation = torch_mutation_kind(type) == TorchMutationMetadata;
        if (inplace && check_mutation(bound) != 0)
            return -1;
        if (identity)
        {
            if (bound.outputs.size() != 1)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid identity signature\n", context.c_str());
                return -1;
            }
            const std::string result = value_name(bound.outputs[0]);
            if (source_name.empty() || result.empty())
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid identity reference\n", context.c_str());
                return -1;
            }
            if (define_value(result, version ? version->value : values.at(source_name), true) != 0)
                return -1;
            set_provenance(result, provenance[source_name]);
            if (version ? version->known_layout : known_layouts.count(source_name) != 0)
                known_layouts.insert(result);
            return 0;
        }
        std::string normalized = type;
        // preserve cast allocation after folding identities in the original dtype
        if (type == "aten::to" && alias_kind == TorchAliasNone)
            inputs[inputs.size() - 2] = new_constant(true);
        if (type == "aten::type_as" && alias_kind == TorchAliasNone)
        {
            normalized = "aten::to";
            const int dtype = normalize_pt2_scalar_type(pt2_scalar_type(output_type));
            inputs = {inputs[0], new_constant(dtype), new_constant(false), new_constant(true), new_constant(Parameter())};
        }
        if (type == "aten::sym_size")
            normalized = "aten::size";
        if (type == "aten::lift_fresh_copy")
        {
            normalized = "aten::clone";
            inputs.push_back(new_constant(Parameter(1)));
        }
        // input metadata is not a runtime layout contract
        // these allocations are contiguous regardless of input strides
        const bool contiguous_allocation = type == "aten::linear" || type == "aten::mm"
                                           || type == "aten::bmm" || type == "aten::addmm"
                                           || type == "aten::masked_fill"
                                           || type == "aten::empty" || type == "aten::zeros"
                                           || type == "aten::ones" || type == "aten::full"
                                           || type == "aten::new_empty" || type == "aten::new_zeros"
                                           || type == "aten::new_ones" || type == "aten::new_full";
        bool known_layout = contiguous_allocation;
        if (!inputs.empty())
        {
            if (type == "aten::slice" || type == "aten::select" || type == "aten::view" || type == "aten::reshape"
                    || type == "aten::transpose" || type == "aten::permute" || type == "aten::t"
                    || type == "aten::squeeze" || type == "aten::unsqueeze" || type == "aten::expand"
                    || type == "aten::as_strided" || type == "aten::detach" || type == "aten::alias"
                    || normalized == "aten::clone" || metadata_mutation
                    || type == "aten::pad" || type == "aten::constant_pad_nd")
                known_layout = version ? version->known_layout : known_layouts.count(source_name) != 0;
            if (normalized == "aten::clone" && inputs.size() == 2 && inputs[1]->producer->type == "prim::Constant")
            {
                const Parameter& format = inputs[1]->producer->params.at("value");
                if (format.type == 2 && (format.i == 0 || format.i == 2 || format.i == 3))
                    known_layout = true;
            }
            for (size_t i = contiguous_allocation ? 0 : 1; i < inputs.size(); i++)
            {
                if (contiguous_allocation && inputs[i]->type != 0)
                    continue;
                if (inputs[i]->producer->type != "prim::Constant")
                    known_layout = false;
            }
            // data writes preserve layout; metadata writes require constant axes
            if (inplace && !metadata_mutation)
                known_layout = version ? version->known_layout : known_layouts.count(source_name) != 0;
        }
        Operator* op = new_op(normalized);
        targets[op] = target;
        for (Operand* r : inputs)
        {
            add_input(op, r);
        }
        if (add_node_outputs(op, bound, alias_kind, known_layout, contiguous_allocation, object) != 0)
            return -1;
        if (layout_query)
        {
            if (!version && !metadata.count(source_name))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: layout source metadata missing\n", context.c_str());
                return -1;
            }
            if (!(version ? version->known_layout : known_layouts.count(source_name) != 0))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: layout query source is not statically known\n", context.c_str());
                return -1;
            }
            layout_sources[op] = version ? version->metadata : metadata.at(source_name);
        }
        return 0;
    }

    int load_inference_subgraph(const JsonValue& serialized, const std::string& target)
    {
        const JsonValue* args = get_field(serialized, "inputs", JsonValue::Array);
        const JsonValue* returns = get_field(serialized, "outputs", JsonValue::Array);
        if (!args || !returns)
            return -1;
        const size_t graph_index = target == "torch.ops.higher_order.wrap_with_set_grad_enabled" ? 1 : 4;
        const size_t enabled_index = graph_index == 1 ? 0 : 2;
        if (args->a.size() <= graph_index)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: invalid inference wrapper signature\n", context.c_str());
            return -1;
        }
        const JsonValue* enabled_arg = get_field(args->a[enabled_index], "arg", JsonValue::Object);
        if (!enabled_arg)
            return -1;
        const JsonValue* enabled = enabled_arg->get("as_bool");
        if (!enabled || enabled->type != JsonValue::Bool || enabled->b)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: enabled gradient/autocast wrapper is unsupported\n", context.c_str());
            return -1;
        }
        const JsonValue* graph_arg = get_field(args->a[graph_index], "arg", JsonValue::Object);
        if (!graph_arg)
            return -1;
        const JsonValue* as_graph = get_field(*graph_arg, "as_graph", JsonValue::Object);
        if (!as_graph)
            return -1;
        const JsonValue* body = get_field(*as_graph, "graph", JsonValue::Object);
        if (!body)
            return -1;
        const JsonValue* inputs = get_field(*body, "inputs", JsonValue::Array);
        const JsonValue* outputs = get_field(*body, "outputs", JsonValue::Array);
        if (!inputs || !outputs)
            return -1;
        if (args->a.size() != graph_index + 1 + inputs->a.size() || returns->a.size() != outputs->a.size())
        {
            fprintf(stderr, "pnnx: pt2 load: %s: inference wrapper input/output count mismatch\n", context.c_str());
            return -1;
        }

        // import a lexical scope while preserving captured storage identities
        Pt2Loader child(g);
        child.next_id = next_id;
        child.next_storage = next_storage;
        child.context = context;
        child.shape_bindings = shape_bindings;
        child.unsupported_aliases = unsupported_aliases;
        child.dense_aliases = dense_aliases;
        child.tensor_objects = tensor_objects;
        child.metadata_versions = metadata_versions;
        const JsonValue* tensor_values = get_field(*body, "tensor_values", JsonValue::Object);
        if (!tensor_values)
            return -1;
        for (const auto& item : tensor_values->o)
        {
            Pt2TensorMeta meta;
            if (parse_pt2_tensor_meta(item.second, meta, true) != 0)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: tensor %s: invalid metadata\n", context.c_str(), item.first.c_str());
                return -1;
            }
            if (meta.type == 12)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: complex32 tensors are unsupported by the Python writer\n", context.c_str());
                return -1;
            }
            for (std::string& symbol : meta.shape_symbols)
            {
                symbol = direct_symbol(symbol);
            }
            if (child.specialize_shape(meta) != 0)
            {
                return -1;
            }
            child.metadata[item.first] = meta;
        }
        for (size_t i = 0; i < inputs->a.size(); i++)
        {
            const JsonValue* arg = get_field(args->a[graph_index + 1 + i], "arg", JsonValue::Object);
            if (!arg)
                return -1;
            const std::string source = value_name(*arg);
            const std::string name = value_name(inputs->a[i]);
            if (!arg->get("as_tensor") || !inputs->a[i].get("as_tensor") || !values.count(source) || !child.metadata.count(name))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid inference wrapper tensor input\n", context.c_str());
                return -1;
            }
            const MetadataVersion* version = metadata_version(values.at(source));
            Operand* r = version ? version->value : values.at(source);
            const Pt2TensorMeta& meta = child.metadata.at(name);
            if (r->type != meta.type || r->shape != meta.shape)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: inference wrapper input metadata mismatch\n", context.c_str());
                return -1;
            }
            if (child.define_value(name, r, true) != 0)
            {
                return -1;
            }
            child.set_provenance(name, provenance[source]);
            if (version ? version->known_layout : known_layouts.count(source) != 0)
                child.known_layouts.insert(name);
            child.external.insert(provenance[source].begin(), provenance[source].end());
            consume_reference(source);
        }
        const JsonValue* nodes = get_field(*body, "nodes", JsonValue::Array);
        if (!nodes)
            return -1;
        if (child.load_nodes(*nodes, *outputs) != 0)
        {
            return -1;
        }
        for (size_t i = 0; i < returns->a.size(); i++)
        {
            const std::string source = value_name(outputs->a[i]);
            const std::string name = value_name(returns->a[i]);
            if (!returns->a[i].get("as_tensor") || !outputs->a[i].get("as_tensor") || !child.values.count(source) || !metadata.count(name))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid inference wrapper tensor output\n", context.c_str());
                return -1;
            }
            const MetadataVersion* version = child.metadata_version(child.values.at(source));
            Operand* r = version ? version->value : child.values.at(source);
            if (r->type != metadata.at(name).type || r->shape != metadata.at(name).shape)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: inference wrapper output metadata mismatch\n", context.c_str());
                return -1;
            }
            if (define_value(name, r, true) != 0)
                return -1;
            set_provenance(name, child.provenance[source]);
            if (version ? version->known_layout : child.known_layouts.count(source) != 0)
                known_layouts.insert(name);
        }
        next_id = child.next_id;
        next_storage = child.next_storage;
        unsupported_aliases.insert(child.unsupported_aliases.begin(), child.unsupported_aliases.end());
        dense_aliases.insert(child.dense_aliases.begin(), child.dense_aliases.end());
        tensor_objects = child.tensor_objects;
        metadata_versions = child.metadata_versions;
        layout_sources.insert(child.layout_sources.begin(), child.layout_sources.end());
        targets.insert(child.targets.begin(), child.targets.end());
        floating_constants.insert(child.floating_constants.begin(), child.floating_constants.end());
        return 0;
    }

    int load_program(StoreZipReader& zip, const std::string& root, const std::string& model, JsonValue& program)
    {
        context = "archive";
        std::string version, byteorder;
        if (read_entry(zip, root + "archive_version", version) != 0)
            return -1;
        if (version != "0")
        {
            fprintf(stderr, "pnnx: pt2 load: %s: unsupported archive version %s\n", context.c_str(), version.c_str());
            return -1;
        }
        if (read_entry(zip, root + ".data/version", version) != 0)
            return -1;
        if (version != "6\n" && version != "6")
        {
            fprintf(stderr, "pnnx: pt2 load: %s: unsupported ZIP data version %s\n", context.c_str(), version.c_str());
            return -1;
        }
        if (read_entry(zip, root + "byteorder", byteorder) != 0)
            return -1;
        uint16_t endian = 1;
        if (byteorder != "little" || *(const unsigned char*)&endian != 1)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: only little-endian tensors are supported\n", context.c_str());
            return -1;
        }
        if (read_json(zip, root + "models/" + model + ".json", program) != 0)
            return -1;
        const JsonValue* schema = get_field(program, "schema_version", JsonValue::Object);
        if (!schema)
            return -1;
        const JsonValue* schema_major = get_field(*schema, "major", JsonValue::Int);
        const JsonValue* schema_minor = get_field(*schema, "minor", JsonValue::Int);
        if (!schema_major || !schema_minor)
            return -1;
        // accept compatible schema fields and reject unknown variants
        if (schema_major->i != 8 || schema_minor->i < 17 || schema_minor->i > 20)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: unsupported schema version (expected 8.17 through 8.20)\n", context.c_str());
            return -1;
        }
        const JsonValue* opset = get_field(program, "opset_version", JsonValue::Object);
        if (!opset)
            return -1;
        const JsonValue* aten = get_field(*opset, "aten", JsonValue::Int);
        if (!aten)
            return -1;
        if (aten->i != 10 || opset->o.size() != 1)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: unsupported opset (expected aten 10)\n", context.c_str());
            return -1;
        }
        return 0;
    }

    int load_inputs(StoreZipReader& zip, const std::string& root, const std::string& model, const JsonValue& inputs, const JsonValue& specs, size_t& user_leaves, bool& dynamic)
    {
        JsonValue weights, constants;
        std::string weight_path = root + "data/weights/" + model + "_weights_config.json";
        std::string constant_path = root + "data/constants/" + model + "_constants_config.json";
        if (zip.has_file(weight_path) && read_json(zip, weight_path, weights) != 0)
            return -1;
        if (zip.has_file(constant_path) && read_json(zip, constant_path, constants) != 0)
            return -1;
        std::map<std::string, Pt2TensorStorage> storage_cache;
        // release raw storage after its last referenced payload
        for (const JsonValue& spec : specs.a)
        {
            if (spec.type != JsonValue::Object || spec.o.size() != 1)
                continue;
            const std::string& kind = spec.o.begin()->first;
            const JsonValue& detail = spec.o.begin()->second;
            const char* key = kind == "parameter" ? "parameter_name"
                              : kind == "buffer" ? "buffer_name"
                              : kind == "tensor_constant" ? "tensor_constant_name" : 0;
            if (!key)
                continue;
            const JsonValue* state = detail.get(key);
            const JsonValue* persistent = detail.get("persistent");
            const bool is_weight = kind == "parameter" || (kind == "buffer" && persistent && persistent->type == JsonValue::Bool && persistent->b);
            const JsonValue* config = (is_weight ? weights : constants).get("config");
            const JsonValue* payload = config && state && state->type == JsonValue::String ? config->get(state->s) : 0;
            const JsonValue* path = payload ? payload->get("path_name") : 0;
            if (path && path->type == JsonValue::String)
                storage_cache[root + (is_weight ? "data/weights/" : "data/constants/") + path->s].remaining++;
        }
        for (size_t i = 0; i < inputs.a.size(); i++)
        {
            context = "input " + std::to_string(i);
            const JsonValue& spec = specs.a[i];
            if (spec.type != JsonValue::Object || spec.o.size() != 1)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: invalid input signature union\n", context.c_str());
                return -1;
            }
            const std::string& kind = spec.o.begin()->first;
            const JsonValue& detail = spec.o.begin()->second;
            const std::string name = value_name(inputs.a[i]);
            if (kind == "constant_input")
            {
                user_leaves++;
                const JsonValue* value = get_field(detail, "value", JsonValue::Object);
                if (!value)
                    return -1;
                if (!same_json(*value, inputs.a[i]))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: specialized input signature mismatch\n", context.c_str());
                    return -1;
                }
                if (!load_argument(*value, "", ""))
                    return -1;
                continue;
            }
            if (kind == "user_input")
            {
                user_leaves++;
                const JsonValue* arg = get_field(detail, "arg", JsonValue::Object);
                if (!arg)
                    return -1;
                if (!same_json(*arg, inputs.a[i]))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: user input signature mismatch\n", context.c_str());
                    return -1;
                }
                if (name.empty())
                {
                    if (!load_argument(inputs.a[i], "", ""))
                        return -1;
                    continue;
                }
                if (!inputs.a[i].get("as_tensor"))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: only Tensor runtime inputs are supported\n", context.c_str());
                    return -1;
                }
                Operator* op = new_op("pnnx.Input");
                op->name = "pnnx_input_" + std::to_string(runtime_inputs.size());
                if (define_value(name, new_value(op), true) != 0)
                    return -1;
                runtime_inputs.push_back(name);
                const Pt2TensorMeta& meta = metadata.at(name);
                for (size_t d = 0; d < meta.shape.size(); d++)
                {
                    if (meta.shape[d] >= 0)
                        continue;
                    dynamic = true;
                    if (meta.shape_symbols[d].empty())
                    {
                        fprintf(stderr, "pnnx: pt2 load: %s: unsupported complex input shape expression\n", context.c_str());
                        return -1;
                    }
                }
            }
            else if (kind == "parameter" || kind == "buffer" || kind == "tensor_constant")
            {
                if (name.empty())
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: state input signature mismatch\n", context.c_str());
                    return -1;
                }
                const JsonValue* arg = get_field(detail, "arg", JsonValue::Object);
                if (!arg)
                    return -1;
                const JsonValue* arg_name = get_field(*arg, "name", JsonValue::String);
                if (!arg_name)
                    return -1;
                if (arg_name->s != name)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: state input signature mismatch\n", context.c_str());
                    return -1;
                }
                const char* key = "tensor_constant_name";
                if (kind == "parameter")
                    key = "parameter_name";
                else if (kind == "buffer")
                    key = "buffer_name";
                const JsonValue* state_name = get_field(detail, key, JsonValue::String);
                if (!state_name)
                    return -1;
                const std::string& state = state_name->s;
                bool is_weight = kind == "parameter";
                if (kind == "buffer")
                {
                    const JsonValue* persistent = get_field(detail, "persistent", JsonValue::Bool);
                    if (!persistent)
                        return -1;
                    is_weight = persistent->b;
                }
                const JsonValue* config = get_field(is_weight ? weights : constants, "config", JsonValue::Object);
                if (!config)
                    return -1;
                const JsonValue* payload = config->get(state);
                if (!payload)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: missing state %s\n", context.c_str(), state.c_str());
                    return -1;
                }
                const JsonValue* is_param = get_field(*payload, "is_param", JsonValue::Bool);
                if (!is_param)
                    return -1;
                if (is_param->b != (kind == "parameter"))
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: state parameter kind mismatch\n", context.c_str());
                    return -1;
                }
                Attribute attr;
                Pt2TensorMeta physical;
                if (load_pt2_tensor(zip, root + (is_weight ? "data/weights/" : "data/constants/"), *payload, attr, physical, storage_cache) != 0)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: cannot load tensor state %s\n", context.c_str(), state.c_str());
                    return -1;
                }
                auto m = metadata.find(name);
                if (m == metadata.end() || m->second.type != physical.type || m->second.shape != physical.shape || m->second.strides != physical.strides || m->second.storage_offset != physical.storage_offset)
                {
                    fprintf(stderr, "pnnx: pt2 load: %s: payload/graph tensor metadata mismatch\n", context.c_str());
                    return -1;
                }
                Operator* op = new_op("pnnx.Attribute");
                std::swap(op->attrs["data"], attr);
                if (define_value(name, new_value(op), true) != 0)
                    return -1;
                known_layouts.insert(name);
            }
            else
            {
                fprintf(stderr, "pnnx: pt2 load: %s: unsupported input signature %s\n", context.c_str(), kind.c_str());
                return -1;
            }
            set_provenance(name, {next_storage});
            external.insert(next_storage++);
        }
        return 0;
    }

    int load(StoreZipReader& zip, const std::string& root, const std::string& model, const std::vector<std::vector<int64_t> >& input_shapes, const std::vector<std::string>& input_types,
             const std::vector<std::vector<int64_t> >& input_shapes2, const std::vector<std::string>& input_types2)
    {
        JsonValue program;
        if (load_program(zip, root, model, program) != 0)
            return -1;
        const JsonValue* guards = get_field(program, "guards_code", JsonValue::Array);
        const JsonValue* gm = get_field(program, "graph_module", JsonValue::Object);
        if (!guards || !gm)
            return -1;
        const JsonValue* graph = get_field(*gm, "graph", JsonValue::Object);
        const JsonValue* signature = get_field(*gm, "signature", JsonValue::Object);
        if (!graph || !signature)
            return -1;
        const JsonValue* tensor_values = get_field(*graph, "tensor_values", JsonValue::Object);
        if (!tensor_values)
            return -1;
        for (const auto& item : tensor_values->o)
        {
            Pt2TensorMeta meta;
            if (parse_pt2_tensor_meta(item.second, meta, true) != 0)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: tensor %s: invalid metadata\n", context.c_str(), item.first.c_str());
                return -1;
            }
            if (meta.type == 12)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: complex32 tensors are unsupported by the Python writer\n", context.c_str());
                return -1;
            }
            for (std::string& symbol : meta.shape_symbols)
            {
                symbol = direct_symbol(symbol);
            }
            metadata[item.first] = meta;
        }
        const JsonValue* inputs = get_field(*graph, "inputs", JsonValue::Array);
        const JsonValue* specs = get_field(*signature, "input_specs", JsonValue::Array);
        const JsonValue* outputs = get_field(*graph, "outputs", JsonValue::Array);
        const JsonValue* output_specs = get_field(*signature, "output_specs", JsonValue::Array);
        if (!inputs || !specs || !outputs || !output_specs)
            return -1;
        if (inputs->a.size() != specs->a.size() || outputs->a.size() != output_specs->a.size())
        {
            fprintf(stderr, "pnnx: pt2 load: %s: signature input/output count mismatch\n", context.c_str());
            return -1;
        }
        size_t user_leaves = 0;
        bool dynamic = false;
        if (load_inputs(zip, root, model, *inputs, *specs, user_leaves, dynamic) != 0)
            return -1;
        context = "signature";
        for (size_t i = 0; i < outputs->a.size(); i++)
        {
            const JsonValue& spec = output_specs->a[i];
            const JsonValue* user = spec.get("user_output");
            if (spec.type != JsonValue::Object || spec.o.size() != 1 || !user)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: state mutation, loss or backward output is unsupported\n", context.c_str());
                return -1;
            }
            const JsonValue* arg = get_field(*user, "arg", JsonValue::Object);
            if (!arg)
                return -1;
            if (!same_json(*arg, outputs->a[i]))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: output signature mismatch\n", context.c_str());
                return -1;
            }
            if (!outputs->a[i].get("as_tensor"))
            {
                fprintf(stderr, "pnnx: pt2 load: %s: non-Tensor user output is unsupported\n", context.c_str());
                return -1;
            }
        }
        const JsonValue* calls = get_field(*gm, "module_call_graph", JsonValue::Array);
        if (!calls)
            return -1;
        bool root_call = false;
        for (const JsonValue& call : calls->a)
        {
            const JsonValue* fqn = get_field(call, "fqn", JsonValue::String);
            if (!fqn)
                return -1;
            if (!fqn->s.empty())
                continue;
            if (root_call)
            {
                fprintf(stderr, "pnnx: pt2 load: %s: duplicate root module call signature\n", context.c_str());
                return -1;
            }
            root_call = true;
            const JsonValue* sig = get_field(call, "signature", JsonValue::Object);
            if (!sig)
                return -1;
            const JsonValue* in_spec = get_field(*sig, "in_spec", JsonValue::String);
            const JsonValue* out_spec = get_field(*sig, "out_spec", JsonValue::String);
            if (!in_spec || !out_spec)
                return -1;
            if (treespec(in_spec->s, user_leaves) != 0 || treespec(out_spec->s, outputs->a.size()) != 0)
                return -1;
        }
        if (!root_call)
        {
            fprintf(stderr, "pnnx: pt2 load: %s: missing root module call signature\n", context.c_str());
            return -1;
        }
        context = "input shapes";
        const JsonValue* ranges = get_field(program, "range_constraints", JsonValue::Object);
        if (!ranges)
            return -1;
        if (dynamic && input_shapes.empty())
        {
            fprintf(stderr, "pnnx: pt2 load: %s: dynamic inputs require concrete inputshape or input samples\n", context.c_str());
            return -1;
        }
        if (check_input_shapes(input_shapes, input_types, *ranges, dynamic) != 0 || check_input_shapes(input_shapes2, input_types2, *ranges, false) != 0)
            return -1;
        if (!input_shapes.empty() && input_shapes2.empty() && specialize_shapes(input_shapes) != 0)
            return -1;
        const JsonValue* nodes = get_field(*graph, "nodes", JsonValue::Array);
        if (!nodes)
            return -1;
        if (load_nodes(*nodes, *outputs) != 0)
            return -1;
        context = "outputs";
        size_t output_index = 0;
        for (const JsonValue& out : outputs->a)
        {
            Operand* r = load_argument(out, "", "");
            if (!r)
                return -1;
            Operator* op = new_op("pnnx.Output");
            op->name = "pnnx_output_" + std::to_string(output_index++);
            add_input(op, r);
        }
        return 0;
    }

    Graph& g;
    std::string context;
    int next_id;
    int next_storage;
    std::map<std::string, Operand*> values;
    std::map<std::string, Pt2TensorMeta> metadata;
    std::set<std::string> symbols;
    std::set<std::string> known_layouts;
    std::vector<std::string> runtime_inputs;
    std::map<std::string, std::set<int> > provenance;
    std::map<int, std::set<std::string> > live_references;
    std::set<const Operand*> unsupported_aliases;
    std::set<const Operand*> dense_aliases;
    std::set<int> external;
    std::map<std::string, int> remaining;
    std::map<const Operand*, const Operand*> tensor_objects;
    std::map<const Operand*, MetadataVersion> metadata_versions;
    std::map<Operator*, Pt2TensorMeta> layout_sources;
    std::map<Operator*, std::string> targets;
    std::map<const Operand*, double> floating_constants;
    std::map<std::string, int64_t> shape_bindings;
};

bool model_file_maybe_pt2(const std::string& path)
{
    StoreZipReader zip;
    if (zip.open(path) != 0)
        return ends_with(path, ".pt2");
    std::set<std::string> roots;
    bool legacy;
    collect_archive_roots(zip, roots, legacy);
    return legacy || !roots.empty();
}

int load_pt2(const std::string& path, Graph& g, const std::vector<std::vector<int64_t> >& input_shapes, const std::vector<std::string>& input_types,
             const std::vector<std::vector<int64_t> >& input_shapes2, const std::vector<std::string>& input_types2)
{
    StoreZipReader zip;
    if (zip.open(path) != 0)
    {
        fprintf(stderr, "pnnx: pt2 archive: cannot open %s\n", path.c_str());
        return -1;
    }
    std::string root, model;
    if (archive_root(zip, root, model) != 0)
        return -1;
    Graph graph;
    Pt2Loader loader(graph);
    if (loader.load(zip, root, model, input_shapes, input_types, input_shapes2, input_types2) != 0)
        return -1;
    normalize_pt2_tensor_dtype(graph);
    if (fold_pt2_constants(graph, loader.targets, loader.floating_constants) != 0)
        return -1;
    if (pass_pt2(graph, loader.layout_sources) != 0)
        return -1;
    g.ops.swap(graph.ops);
    g.operands.swap(graph.operands);
    return 0;
}

} // namespace pnnx
