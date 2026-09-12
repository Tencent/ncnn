// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"

#include "json.h"
#include "storezip.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace pnnx {

static std::string zip_slash(const std::string& name)
{
    std::string n = name;
    for (size_t i = 0; i < n.size(); i++)
    {
        if (n[i] == '\\')
            n[i] = '/';
    }
    return n;
}

static bool zip_name_is(const std::string& name, const std::string& suffix)
{
    const std::string n = zip_slash(name);
    if (n == suffix)
        return true;
    if (n.size() > suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
        char c = n[n.size() - suffix.size() - 1];
        return c == '/';
    }
    return false;
}

static bool zip_is_model_json(const std::string& name)
{
    const std::string n = zip_slash(name);
    if (n.size() < 6 || n.compare(n.size() - 5, 5, ".json") != 0)
        return false;
    return n.compare(0, 7, "models/") == 0 || n.find("/models/") != std::string::npos;
}

static std::string find_zip_entry(const std::vector<std::string>& names, const std::string& suffix)
{
    for (size_t i = 0; i < names.size(); i++)
    {
        if (zip_name_is(names[i], suffix))
            return names[i];
    }
    return std::string();
}

static std::string read_zip_text(StoreZipReader& zip, const std::string& name)
{
    uint64_t size = zip.get_file_size(name);
    if (size == 0)
        return std::string();
    std::string s;
    s.resize(size);
    zip.read_file(name, &s[0]);
    return s;
}

bool is_pt2_archive(const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
        return false;

    uint32_t signature = 0;
    fread((char*)&signature, sizeof(signature), 1, fp);
    fclose(fp);

    if (signature != 0x04034b50)
        return false;

    StoreZipReader zip;
    if (zip.open(path) != 0)
        return false;

    std::string fmt_name = find_zip_entry(zip.get_names(), "archive_format");
    if (fmt_name.empty())
        return false;

    std::string fmt = read_zip_text(zip, fmt_name);
    while (!fmt.empty() && (fmt.back() == '\n' || fmt.back() == '\r' || fmt.back() == ' '))
        fmt.pop_back();
    return fmt == "pt2";
}

static const JsonValue* find_graph(const JsonValue& v)
{
    if (v.is_object() && v.find("nodes") && v.find("tensor_values"))
        return &v;
    if (v.is_object())
    {
        for (std::map<std::string, JsonValue>::const_iterator it = v.o.begin(); it != v.o.end(); ++it)
        {
            const JsonValue* g = find_graph(it->second);
            if (g)
                return g;
        }
    }
    else if (v.is_array())
    {
        for (size_t i = 0; i < v.a.size(); i++)
        {
            const JsonValue* g = find_graph(v.a[i]);
            if (g)
                return g;
        }
    }
    return 0;
}

static void collect_tensor_names(const JsonValue& arg, std::vector<std::string>& names)
{
    if (!arg.is_object())
        return;
    if (arg.find("as_tensor") && arg.get("as_tensor").is_object())
    {
        std::string n = arg.get("as_tensor").get("name").as_string();
        if (!n.empty())
            names.push_back(n);
        return;
    }
    if (arg.find("as_optional_tensor") && arg.get("as_optional_tensor").is_object())
    {
        const JsonValue& opt = arg.get("as_optional_tensor");
        if (opt.find("as_tensor") && opt.get("as_tensor").is_object())
        {
            std::string n = opt.get("as_tensor").get("name").as_string();
            if (!n.empty())
                names.push_back(n);
        }
        return;
    }
    if (arg.find("as_tensors") && arg.get("as_tensors").is_array())
    {
        const JsonValue& arr = arg.get("as_tensors");
        for (size_t i = 0; i < arr.a.size(); i++)
            collect_tensor_names(arr.a[i], names);
        return;
    }
    // export list items are often {"name": "x"} without as_tensor
    if (arg.find("name") && !arg.find("arg") && !arg.find("as_int") && !arg.find("as_float"))
    {
        std::string n = arg.get("name").as_string();
        if (!n.empty())
            names.push_back(n);
    }
}

static std::string tensor_name_from_argument(const JsonValue& arg)
{
    std::vector<std::string> names;
    collect_tensor_names(arg, names);
    if (names.size() == 1)
        return names[0];
    return std::string();
}

static std::string canonicalize_arg_name(const std::string& name)
{
    if (name == "self")
        return "input";
    if (name == "split_size")
        return "split_size_or_sections";
    if (name == "ord")
        return "p";
    return name;
}

static std::string normalize_aten_target(const std::string& target)
{
    std::string t = target;
    // torch.ops.aten.relu.default  /  aten.relu.default  /  aten.mul.Tensor
    const char* prefix = "torch.ops.";
    if (t.compare(0, 10, prefix) == 0)
        t = t.substr(10);

    std::vector<std::string> parts;
    std::string cur;
    for (size_t i = 0; i <= t.size(); i++)
    {
        if (i == t.size() || t[i] == '.')
        {
            if (!cur.empty())
                parts.push_back(cur);
            cur.clear();
        }
        else
        {
            cur += t[i];
        }
    }

    if (parts.size() >= 2 && parts[0] == "aten")
    {
        std::string op = parts[1];
        std::string overload = parts.size() >= 3 ? parts[2] : std::string();
        // torch.max(a, b) / torch.min(a, b) are elementwise, not reduce
        if (overload == "other")
        {
            if (op == "max")
                return "aten::maximum";
            if (op == "min")
                return "aten::minimum";
        }
        // drop trailing overload: default / Tensor / Scalar / dim / int ...
        return std::string("aten::") + op;
    }

    if (parts.size() >= 1 && t.find("::") != std::string::npos)
        return t;

    return t;
}

static int scalar_type_to_pnnx(int st)
{
    // torch._export.serde.schema ScalarType
    // FLOAT=7 DOUBLE=8 HALF=6 INT=4 LONG=5 BOOL=12 BFLOAT16=13
    if (st == 7) return 1;
    if (st == 8) return 2;
    if (st == 6) return 3;
    if (st == 4) return 4;
    if (st == 5) return 5;
    if (st == 3) return 6;
    if (st == 2) return 7;
    if (st == 1) return 8;
    if (st == 12) return 9;
    if (st == 10) return 10;
    if (st == 11) return 11;
    if (st == 13) return 13;
    return 1;
}

static void apply_tensor_meta(Operand* opnd, const JsonValue& tensor_values, const std::string& name)
{
    const JsonValue* meta = tensor_values.find(name);
    if (!meta || !meta->is_object())
        return;
    opnd->type = scalar_type_to_pnnx(meta->get("dtype").as_int());
    const JsonValue& sizes = meta->get("sizes");
    if (sizes.is_array())
    {
        opnd->shape.clear();
        for (size_t i = 0; i < sizes.a.size(); i++)
        {
            const JsonValue& dim = sizes.a[i];
            int d = 0;
            if (dim.is_object() && dim.find("as_int"))
                d = dim.get("as_int").as_int();
            else if (dim.is_number())
                d = dim.as_int();
            opnd->shape.push_back(d);
        }
    }
}

static Operand* get_or_new_operand(Graph& g, const std::string& name)
{
    Operand* r = g.get_operand(name);
    if (r)
        return r;
    return g.new_operand(name);
}

static int json_as_int(const JsonValue& v)
{
    if (v.is_object() && v.find("as_int"))
        return v.get("as_int").as_int();
    return v.as_int();
}

static void parse_sizes(const JsonValue& sizes, std::vector<int>& shape)
{
    shape.clear();
    if (!sizes.is_array())
        return;
    for (size_t i = 0; i < sizes.a.size(); i++)
        shape.push_back(json_as_int(sizes.a[i]));
}

static bool parse_argument_to_parameter(const JsonValue& argv, Parameter& pv)
{
    if (argv.find("as_int"))
    {
        pv = Parameter(argv.get("as_int").as_int());
        return true;
    }
    if (argv.find("as_float"))
    {
        pv = Parameter((float)argv.get("as_float").as_double());
        return true;
    }
    if (argv.find("as_bool"))
    {
        pv = Parameter(argv.get("as_bool").as_bool());
        return true;
    }
    if (argv.find("as_string"))
    {
        pv = Parameter(argv.get("as_string").as_string());
        return true;
    }
    if (argv.find("as_none"))
    {
        pv = Parameter();
        return true;
    }
    if (argv.find("as_sym_int") && argv.get("as_sym_int").find("as_int"))
    {
        pv = Parameter(argv.get("as_sym_int").get("as_int").as_int());
        return true;
    }
    if (argv.find("as_ints") && argv.get("as_ints").is_array())
    {
        std::vector<int> ai;
        const JsonValue& arr = argv.get("as_ints");
        for (size_t i = 0; i < arr.a.size(); i++)
            ai.push_back(json_as_int(arr.a[i]));
        pv = Parameter(ai);
        return true;
    }
    if (argv.find("as_floats") && argv.get("as_floats").is_array())
    {
        std::vector<float> af;
        const JsonValue& arr = argv.get("as_floats");
        for (size_t i = 0; i < arr.a.size(); i++)
            af.push_back((float)arr.a[i].as_double());
        pv = Parameter(af);
        return true;
    }
    if (argv.find("as_bools") && argv.get("as_bools").is_array())
    {
        std::vector<int> ai;
        const JsonValue& arr = argv.get("as_bools");
        for (size_t i = 0; i < arr.a.size(); i++)
            ai.push_back(arr.a[i].as_bool() ? 1 : 0);
        pv = Parameter(ai);
        return true;
    }
    if (argv.find("as_sym_ints") && argv.get("as_sym_ints").is_array())
    {
        std::vector<int> ai;
        const JsonValue& arr = argv.get("as_sym_ints");
        for (size_t i = 0; i < arr.a.size(); i++)
            ai.push_back(json_as_int(arr.a[i]));
        pv = Parameter(ai);
        return true;
    }
    return false;
}

static bool has_input_name(const Operator* op, const std::string& name)
{
    for (size_t i = 0; i < op->inputnames.size(); i++)
    {
        if (op->inputnames[i] == name)
            return true;
    }
    return false;
}

static void add_constant_input(Graph& g, Operator* op, const std::string& arg_name, const Parameter& pv)
{
    std::string cname = op->name + "_" + arg_name;
    Operator* cop = g.new_operator_before("prim::Constant", cname, op);
    cop->params["value"] = pv;
    Operand* r = get_or_new_operand(g, cname);
    r->producer = cop;
    cop->outputs.push_back(r);
    r->consumers.push_back(op);
    op->inputs.push_back(r);
    op->inputnames.push_back(arg_name);
}

static void ensure_constant_arg(Graph& g, Operator* op, const std::string& arg_name, const Parameter& pv)
{
    if (!has_input_name(op, arg_name))
        add_constant_input(g, op, arg_name, pv);
}

static void add_list_construct_input(Graph& g, Operator* op, const std::string& arg_name, const std::vector<std::string>& tensor_names)
{
    std::string lname = op->name + "_" + arg_name + "_list";
    Operator* lop = g.new_operator_before("prim::ListConstruct", lname, op);
    for (size_t i = 0; i < tensor_names.size(); i++)
    {
        Operand* r = get_or_new_operand(g, tensor_names[i]);
        r->consumers.push_back(lop);
        lop->inputs.push_back(r);
    }
    Operand* out = get_or_new_operand(g, lname);
    out->producer = lop;
    lop->outputs.push_back(out);
    out->consumers.push_back(op);
    op->inputs.push_back(out);
    op->inputnames.push_back(arg_name);
}

static void fill_missing_aten_args(Graph& g, Operator* op)
{
    const std::string& t = op->type;
    const int n = (int)op->inputs.size();

    if (t == "aten::conv1d" || t == "aten::conv2d" || t == "aten::conv3d")
    {
        int nd = 1;
        if (t == "aten::conv2d") nd = 2;
        if (t == "aten::conv3d") nd = 3;
        std::vector<int> one(nd, 1);
        std::vector<int> zero(nd, 0);
        if (n < 3) add_constant_input(g, op, "bias", Parameter());
        if (n < 4) add_constant_input(g, op, "stride", Parameter(one));
        if (n < 5) add_constant_input(g, op, "padding", Parameter(zero));
        if (n < 6) add_constant_input(g, op, "dilation", Parameter(one));
        if (n < 7) add_constant_input(g, op, "groups", Parameter(1));
    }
    else if (t == "aten::conv_transpose1d" || t == "aten::conv_transpose2d" || t == "aten::conv_transpose3d")
    {
        int nd = 1;
        if (t == "aten::conv_transpose2d") nd = 2;
        if (t == "aten::conv_transpose3d") nd = 3;
        std::vector<int> one(nd, 1);
        std::vector<int> zero(nd, 0);
        if (n < 3) add_constant_input(g, op, "bias", Parameter());
        if (n < 4) add_constant_input(g, op, "stride", Parameter(one));
        if (n < 5) add_constant_input(g, op, "padding", Parameter(zero));
        if (n < 6) add_constant_input(g, op, "output_padding", Parameter(zero));
        if (n < 7) add_constant_input(g, op, "groups", Parameter(1));
        if (n < 8) add_constant_input(g, op, "dilation", Parameter(one));
    }
    else if (t == "aten::linear")
    {
        if (n < 3) add_constant_input(g, op, "bias", Parameter());
    }
    else if (t == "aten::softmax" || t == "aten::log_softmax")
    {
        if (n < 3) add_constant_input(g, op, "dtype", Parameter());
    }
    else if (t == "aten::leaky_relu")
    {
        if (n < 2) add_constant_input(g, op, "negative_slope", Parameter(0.01f));
    }
    else if (t == "aten::flatten")
    {
        if (n < 2) add_constant_input(g, op, "start_dim", Parameter(0));
        if (n < 3) add_constant_input(g, op, "end_dim", Parameter(-1));
    }
    else if (t == "aten::gelu")
    {
        if (n < 2) add_constant_input(g, op, "approximate", Parameter(std::string("none")));
    }
    else if (t == "aten::elu")
    {
        if (n < 2) add_constant_input(g, op, "alpha", Parameter(1.f));
        if (n < 3) add_constant_input(g, op, "scale", Parameter(1.f));
        if (n < 4) add_constant_input(g, op, "input_scale", Parameter(1.f));
    }
    else if (t == "aten::celu")
    {
        if (n < 2) add_constant_input(g, op, "alpha", Parameter(1.f));
    }
    else if (t == "aten::hardtanh")
    {
        if (n < 2) add_constant_input(g, op, "min_val", Parameter(-1.f));
        if (n < 3) add_constant_input(g, op, "max_val", Parameter(1.f));
    }
    else if (t == "aten::hardshrink" || t == "aten::softshrink")
    {
        if (n < 2) add_constant_input(g, op, "lambd", Parameter(0.5f));
    }
    else if (t == "aten::softplus")
    {
        if (n < 2) add_constant_input(g, op, "beta", Parameter(1.f));
        if (n < 3) add_constant_input(g, op, "threshold", Parameter(20.f));
    }
    else if (t == "aten::pad")
    {
        if (n < 3) add_constant_input(g, op, "mode", Parameter(std::string("constant")));
        if (n < 4) add_constant_input(g, op, "value", Parameter());
    }
    else if (t == "aten::clamp")
    {
        if (n < 2) add_constant_input(g, op, "min", Parameter());
        if (n < 3) add_constant_input(g, op, "max", Parameter());
    }
    else if (t == "aten::mean" || t == "aten::sum" || t == "aten::prod")
    {
        // mean.dim / sum.dim / prod.dim may omit keepdim and/or dtype.
        // n==2 (input, dim) must not be left as-is: torch_mean_1 would treat dim as dtype.
        if (has_input_name(op, "dim"))
            ensure_constant_arg(g, op, "keepdim", Parameter(false));
        if (n == 1 || has_input_name(op, "dim") || has_input_name(op, "keepdim"))
            ensure_constant_arg(g, op, "dtype", Parameter());
    }
    else if (t == "aten::max" || t == "aten::min")
    {
        // aten.max.dim(self, dim) omits keepdim=False
        if (n == 2)
            add_constant_input(g, op, "keepdim", Parameter(false));
    }
    else if (t == "aten::linalg_vector_norm")
    {
        // schema: (self, ord=2, dim=None, keepdim=False, *, dtype=None)
        ensure_constant_arg(g, op, "p", Parameter(2.f));
        ensure_constant_arg(g, op, "dim", Parameter());
        ensure_constant_arg(g, op, "keepdim", Parameter(false));
        ensure_constant_arg(g, op, "dtype", Parameter());
    }
    else if (t == "aten::layer_norm" || t == "aten::group_norm")
    {
        if (n < 6) add_constant_input(g, op, "cudnn_enabled", Parameter(true));
    }
    else if (t == "aten::batch_norm")
    {
        if (n < 9) add_constant_input(g, op, "cudnn_enabled", Parameter(true));
    }
    else if (t == "aten::instance_norm")
    {
        if (n < 6) add_constant_input(g, op, "use_input_stats", Parameter(true));
        if (n < 7) add_constant_input(g, op, "momentum", Parameter(0.1f));
        if (n < 8) add_constant_input(g, op, "eps", Parameter(1e-5f));
        if (n < 9) add_constant_input(g, op, "cudnn_enabled", Parameter(false));
    }
    else if (t == "aten::cat" || t == "aten::stack")
    {
        if (n < 2) add_constant_input(g, op, "dim", Parameter(0));
    }
    else if (t == "aten::chunk")
    {
        if (n < 3) add_constant_input(g, op, "dim", Parameter(0));
    }
    else if (t == "aten::split" || t == "aten::split_with_sizes")
    {
        if (n < 3) add_constant_input(g, op, "dim", Parameter(0));
    }
    else if (t == "aten::unbind")
    {
        if (n < 2) add_constant_input(g, op, "dim", Parameter(0));
    }
    else if (t == "aten::avg_pool1d" || t == "aten::avg_pool2d" || t == "aten::avg_pool3d")
    {
        int nd = 1;
        if (t == "aten::avg_pool2d") nd = 2;
        if (t == "aten::avg_pool3d") nd = 3;
        std::vector<int> zero(nd, 0);
        if (n < 3) add_constant_input(g, op, "stride", Parameter());
        if (n < 4) add_constant_input(g, op, "padding", Parameter(zero));
        if (n < 5) add_constant_input(g, op, "ceil_mode", Parameter(false));
        if (n < 6) add_constant_input(g, op, "count_include_pad", Parameter(true));
        if (t != "aten::avg_pool1d" && n < 7) add_constant_input(g, op, "divisor_override", Parameter());
    }
    else if (t == "aten::max_pool1d" || t == "aten::max_pool2d" || t == "aten::max_pool3d"
             || t == "aten::max_pool1d_with_indices" || t == "aten::max_pool2d_with_indices"
             || t == "aten::max_pool3d_with_indices")
    {
        int nd = 1;
        if (t.find("2d") != std::string::npos) nd = 2;
        if (t.find("3d") != std::string::npos) nd = 3;
        std::vector<int> zero(nd, 0);
        std::vector<int> one(nd, 1);
        ensure_constant_arg(g, op, "stride", Parameter());
        ensure_constant_arg(g, op, "padding", Parameter(zero));
        ensure_constant_arg(g, op, "dilation", Parameter(one));
        ensure_constant_arg(g, op, "ceil_mode", Parameter(false));
    }
    else if (t == "aten::slice")
    {
        ensure_constant_arg(g, op, "step", Parameter(1));
    }
    else if (t == "aten::scaled_dot_product_attention")
    {
        ensure_constant_arg(g, op, "attn_mask", Parameter());
        ensure_constant_arg(g, op, "dropout_p", Parameter(0.f));
        ensure_constant_arg(g, op, "is_causal", Parameter(false));
    }
    else if (t == "aten::baddbmm")
    {
        ensure_constant_arg(g, op, "beta", Parameter(1.f));
        ensure_constant_arg(g, op, "alpha", Parameter(1.f));
    }
}

struct TensorBlob
{
    std::string zip_path;
    int dtype;
    std::vector<int> shape;
    bool use_pickle;

    TensorBlob()
        : dtype(1), use_pickle(false)
    {
    }
};

static void load_tensor_config(const JsonValue& root, const std::string& config_zip_name,
                               std::map<std::string, TensorBlob>& out)
{
    const JsonValue* cfg = root.find("config");
    if (!cfg || !cfg->is_object())
        return;

    std::string dir = zip_slash(config_zip_name);
    size_t slash = dir.find_last_of('/');
    if (slash != std::string::npos)
        dir = dir.substr(0, slash + 1);
    else
        dir.clear();

    for (std::map<std::string, JsonValue>::const_iterator it = cfg->o.begin(); it != cfg->o.end(); ++it)
    {
        const JsonValue& item = it->second;
        TensorBlob blob;
        blob.zip_path = dir + item.get("path_name").as_string();
        blob.use_pickle = item.get("use_pickle").as_bool();
        blob.dtype = 1;
        const JsonValue& meta = item.get("tensor_meta");
        if (meta.is_object())
        {
            blob.dtype = scalar_type_to_pnnx(meta.get("dtype").as_int());
            parse_sizes(meta.get("sizes"), blob.shape);
        }
        out[it->first] = blob;
    }
}

static const JsonValue* find_signature(const JsonValue& v)
{
    if (v.is_object() && v.find("input_specs") && v.find("output_specs"))
        return &v;
    if (v.is_object())
    {
        for (std::map<std::string, JsonValue>::const_iterator it = v.o.begin(); it != v.o.end(); ++it)
        {
            const JsonValue* s = find_signature(it->second);
            if (s)
                return s;
        }
    }
    else if (v.is_array())
    {
        for (size_t i = 0; i < v.a.size(); i++)
        {
            const JsonValue* s = find_signature(v.a[i]);
            if (s)
                return s;
        }
    }
    return 0;
}

static std::string spec_tensor_name(const JsonValue& spec_inner)
{
    const JsonValue* arg = spec_inner.find("arg");
    if (!arg)
        return std::string();
    if (arg->find("as_tensor"))
        return arg->get("as_tensor").get("name").as_string();
    if (arg->find("name"))
        return arg->get("name").as_string();
    return tensor_name_from_argument(*arg);
}

static void create_input_op(Graph& g, const JsonValue& tensor_values, const std::string& name)
{
    if (name.empty() || g.get_operand(name))
        return;
    Operator* op = g.new_operator("pnnx.Input", name);
    Operand* r = get_or_new_operand(g, name);
    apply_tensor_meta(r, tensor_values, name);
    r->producer = op;
    op->outputs.push_back(r);
}

static void create_attribute_op(Graph& g, StoreZipReader& zip, const JsonValue& tensor_values,
                                const std::string& name, const TensorBlob* blob)
{
    if (name.empty() || g.get_operand(name))
        return;

    Operator* op = g.new_operator("pnnx.Attribute", name);
    Operand* r = get_or_new_operand(g, name);
    apply_tensor_meta(r, tensor_values, name);
    r->producer = op;
    op->outputs.push_back(r);

    Attribute attr;
    attr.type = r->type ? r->type : 1;
    attr.shape = r->shape;
    if (blob)
    {
        attr.type = blob->dtype;
        if (!blob->shape.empty() || r->shape.empty())
            attr.shape = blob->shape;
        if (blob->use_pickle)
        {
            fprintf(stderr, "skip pickle weight %s\n", blob->zip_path.c_str());
        }
        else
        {
            uint64_t size = zip.get_file_size(blob->zip_path);
            attr.data.resize((size_t)size);
            if (size)
                zip.read_file(blob->zip_path, attr.data.data());
        }
    }
    r->type = attr.type;
    r->shape = attr.shape;
    op->attrs["data"] = attr;
}

int load_pt2(const std::string& pt2path, Graph& pnnx_graph)
{
    StoreZipReader zip;
    if (zip.open(pt2path) != 0)
    {
        fprintf(stderr, "open pt2 zip failed %s\n", pt2path.c_str());
        return -1;
    }

    std::vector<std::string> names = zip.get_names();
    std::string fmt_name = find_zip_entry(names, "archive_format");
    if (fmt_name.empty())
    {
        fprintf(stderr, "pt2 missing archive_format\n");
        return -1;
    }

    std::string fmt = read_zip_text(zip, fmt_name);
    while (!fmt.empty() && (fmt.back() == '\n' || fmt.back() == '\r' || fmt.back() == ' '))
        fmt.pop_back();
    if (fmt != "pt2")
    {
        fprintf(stderr, "not a pt2 archive (archive_format=%s)\n", fmt.c_str());
        return -1;
    }

    std::string json_name;
    for (size_t i = 0; i < names.size(); i++)
    {
        if (zip_is_model_json(names[i]))
        {
            json_name = names[i];
            break;
        }
    }
    if (json_name.empty())
    {
        fprintf(stderr, "pt2 missing models/*.json\n");
        return -1;
    }

    std::string json_text = read_zip_text(zip, json_name);
    JsonValue root;
    if (parse_json(json_text.c_str(), json_text.size(), root) != 0)
        return -1;

    const JsonValue* graph = find_graph(root);
    if (!graph)
    {
        fprintf(stderr, "pt2 json has no graph.nodes\n");
        return -1;
    }

    std::map<std::string, TensorBlob> weights;
    std::map<std::string, TensorBlob> constants;
    for (size_t i = 0; i < names.size(); i++)
    {
        const std::string n = zip_slash(names[i]);
        if (n.size() >= 20 && n.compare(n.size() - 20, 20, "_weights_config.json") == 0)
        {
            std::string text = read_zip_text(zip, names[i]);
            JsonValue cfg;
            if (parse_json(text.c_str(), text.size(), cfg) == 0)
                load_tensor_config(cfg, names[i], weights);
        }
        else if (n.size() >= 22 && n.compare(n.size() - 22, 22, "_constants_config.json") == 0)
        {
            std::string text = read_zip_text(zip, names[i]);
            JsonValue cfg;
            if (parse_json(text.c_str(), text.size(), cfg) == 0)
                load_tensor_config(cfg, names[i], constants);
        }
    }

    const JsonValue& nodes = graph->get("nodes");
    const JsonValue& inputs = graph->get("inputs");
    const JsonValue& outputs = graph->get("outputs");
    const JsonValue& tensor_values = graph->get("tensor_values");

    fprintf(stderr, "pt2 graph nodes=%d inputs=%d outputs=%d weights=%d constants=%d\n",
            nodes.is_array() ? (int)nodes.a.size() : 0,
            inputs.is_array() ? (int)inputs.a.size() : 0,
            outputs.is_array() ? (int)outputs.a.size() : 0,
            (int)weights.size(), (int)constants.size());

    std::set<std::string> node_output_names;
    if (nodes.is_array())
    {
        for (size_t i = 0; i < nodes.a.size(); i++)
        {
            const JsonValue& node = nodes.a[i];
            const JsonValue& nouts = node.get("outputs");
            if (!nouts.is_array())
                continue;
            for (size_t j = 0; j < nouts.a.size(); j++)
            {
                std::string tn = tensor_name_from_argument(nouts.a[j]);
                if (!tn.empty())
                    node_output_names.insert(tn);
            }
        }
    }

    const JsonValue* signature = find_signature(root);
    bool used_signature = false;
    if (signature)
    {
        const JsonValue& specs = signature->get("input_specs");
        if (specs.is_array())
        {
            used_signature = true;
            for (size_t i = 0; i < specs.a.size(); i++)
            {
                const JsonValue& spec = specs.a[i];
                if (spec.find("user_input"))
                {
                    std::string name = spec_tensor_name(spec.get("user_input"));
                    if (node_output_names.find(name) != node_output_names.end())
                        continue;
                    create_input_op(pnnx_graph, tensor_values, name);
                }
                else if (spec.find("parameter"))
                {
                    const JsonValue& p = spec.get("parameter");
                    std::string name = spec_tensor_name(p);
                    std::string fqn = p.get("parameter_name").as_string();
                    const TensorBlob* blob = 0;
                    if (weights.find(fqn) != weights.end())
                        blob = &weights[fqn];
                    create_attribute_op(pnnx_graph, zip, tensor_values, name, blob);
                }
                else if (spec.find("buffer"))
                {
                    const JsonValue& p = spec.get("buffer");
                    std::string name = spec_tensor_name(p);
                    std::string fqn = p.get("buffer_name").as_string();
                    const TensorBlob* blob = 0;
                    if (weights.find(fqn) != weights.end())
                        blob = &weights[fqn];
                    else if (constants.find(fqn) != constants.end())
                        blob = &constants[fqn];
                    create_attribute_op(pnnx_graph, zip, tensor_values, name, blob);
                }
                else if (spec.find("tensor_constant"))
                {
                    const JsonValue& p = spec.get("tensor_constant");
                    std::string name = spec_tensor_name(p);
                    std::string fqn = p.get("tensor_constant_name").as_string();
                    if (fqn.empty())
                        fqn = p.get("name").as_string();
                    if (fqn.empty())
                        fqn = name;
                    const TensorBlob* blob = 0;
                    if (constants.find(fqn) != constants.end())
                        blob = &constants[fqn];
                    else if (weights.find(fqn) != weights.end())
                        blob = &weights[fqn];
                    create_attribute_op(pnnx_graph, zip, tensor_values, name, blob);
                }
            }
        }
    }

    if (!used_signature && inputs.is_array())
    {
        for (size_t i = 0; i < inputs.a.size(); i++)
        {
            std::string name = tensor_name_from_argument(inputs.a[i]);
            if (name.empty())
                continue;
            if (node_output_names.find(name) != node_output_names.end())
                continue;
            create_input_op(pnnx_graph, tensor_values, name);
        }
    }

    if (nodes.is_array())
    {
        for (size_t i = 0; i < nodes.a.size(); i++)
        {
            const JsonValue& node = nodes.a[i];
            std::string target = node.get("target").as_string();
            std::string op_type = normalize_aten_target(target);
            std::string op_name = node.get("name").as_string();
            if (op_name.empty())
                op_name = op_type + "_" + std::to_string((int)i);

            // FX placeholder/output/get_attr may appear if json still has an "op" field
            std::string fx_op = node.get("op").as_string();
            if (fx_op == "placeholder")
            {
                std::string name = op_name;
                const JsonValue& nouts = node.get("outputs");
                if (nouts.is_array() && !nouts.a.empty())
                {
                    std::string tn = tensor_name_from_argument(nouts.a[0]);
                    if (!tn.empty())
                        name = tn;
                }
                if (pnnx_graph.get_operand(name))
                    continue;
                Operator* op = pnnx_graph.new_operator("pnnx.Input", name);
                Operand* r = get_or_new_operand(pnnx_graph, name);
                apply_tensor_meta(r, tensor_values, name);
                r->producer = op;
                op->outputs.push_back(r);
                continue;
            }
            if (fx_op == "output")
                continue;
            if (fx_op == "get_attr")
            {
                Operator* op = pnnx_graph.new_operator("pnnx.Attribute", op_name);
                const JsonValue& nouts = node.get("outputs");
                std::string out_name = op_name;
                if (nouts.is_array() && !nouts.a.empty())
                {
                    std::string tn = tensor_name_from_argument(nouts.a[0]);
                    if (!tn.empty())
                        out_name = tn;
                }
                Operand* r = get_or_new_operand(pnnx_graph, out_name);
                apply_tensor_meta(r, tensor_values, out_name);
                r->producer = op;
                op->outputs.push_back(r);
                continue;
            }

            if (op_type.empty())
            {
                fprintf(stderr, "skip node %s with empty target\n", op_name.c_str());
                continue;
            }

            Operator* op = pnnx_graph.new_operator(op_type, op_name);

            const JsonValue& nins = node.get("inputs");
            if (nins.is_array())
            {
                for (size_t j = 0; j < nins.a.size(); j++)
                {
                    const JsonValue& named = nins.a[j];
                    const JsonValue* arg = named.find("arg");
                    const JsonValue& argv = arg ? *arg : named;
                    std::string arg_name = canonicalize_arg_name(named.get("name").as_string());

                    if (argv.find("as_tensors"))
                    {
                        std::vector<std::string> tns;
                        collect_tensor_names(argv, tns);
                        if (arg_name.empty())
                            arg_name = "tensors";
                        add_list_construct_input(pnnx_graph, op, arg_name, tns);
                        continue;
                    }

                    std::string in_name = tensor_name_from_argument(argv);
                    if (!in_name.empty())
                    {
                        Operand* r = get_or_new_operand(pnnx_graph, in_name);
                        r->consumers.push_back(op);
                        op->inputs.push_back(r);
                        op->inputnames.push_back(arg_name);
                        continue;
                    }

                    Parameter pv;
                    if (parse_argument_to_parameter(argv, pv))
                    {
                        if (arg_name.empty())
                            arg_name = "c" + std::to_string((int)j);
                        add_constant_input(pnnx_graph, op, arg_name, pv);
                    }
                }
            }

            fill_missing_aten_args(pnnx_graph, op);

            const JsonValue& nouts = node.get("outputs");
            if (nouts.is_array())
            {
                std::vector<std::string> unpack_names;
                if (nouts.a.size() == 1)
                    collect_tensor_names(nouts.a[0], unpack_names);

                if (unpack_names.size() > 1)
                {
                    std::string list_name = op_name + "_list";
                    Operand* list_r = get_or_new_operand(pnnx_graph, list_name);
                    list_r->producer = op;
                    op->outputs.push_back(list_r);

                    Operator* unpack = pnnx_graph.new_operator_after("prim::ListUnpack", op_name + "_unpack", op);
                    list_r->consumers.push_back(unpack);
                    unpack->inputs.push_back(list_r);
                    for (size_t j = 0; j < unpack_names.size(); j++)
                    {
                        Operand* r = get_or_new_operand(pnnx_graph, unpack_names[j]);
                        apply_tensor_meta(r, tensor_values, unpack_names[j]);
                        r->producer = unpack;
                        unpack->outputs.push_back(r);
                    }
                }
                else
                {
                    for (size_t j = 0; j < nouts.a.size(); j++)
                    {
                        std::string out_name = tensor_name_from_argument(nouts.a[j]);
                        if (out_name.empty())
                            out_name = op_name + "_out" + std::to_string((int)j);
                        Operand* r = get_or_new_operand(pnnx_graph, out_name);
                        apply_tensor_meta(r, tensor_values, out_name);
                        r->producer = op;
                        op->outputs.push_back(r);
                    }
                }
            }

            fprintf(stderr, "  %s  %s\n", op_type.c_str(), op_name.c_str());
        }
    }

    if (outputs.is_array())
    {
        for (size_t i = 0; i < outputs.a.size(); i++)
        {
            std::string name = tensor_name_from_argument(outputs.a[i]);
            if (name.empty())
                continue;
            Operator* op = pnnx_graph.new_operator("pnnx.Output", name);
            Operand* r = pnnx_graph.get_operand(name);
            if (!r)
                r = get_or_new_operand(pnnx_graph, name);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }
    }

    zip.close();
    return 0;
}

} // namespace pnnx
