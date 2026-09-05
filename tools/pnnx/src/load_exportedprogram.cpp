// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exportedprogram.h"

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

// torch serde ScalarType -> pnnx type
//   1=uint8 2=int8 28=uint16 3=int16 4=int32 5=int64 6=float16 7=float32 8=float64
//   9=complex32 10=complex64 11=complex128 12=bool 13=bfloat16
static int serde_dtype_to_pnnx_type(int64_t dtype)
{
    switch (dtype)
    {
    case 1:
        return 8; // uint8 -> u8
    case 2:
        return 7; // int8 -> i8
    case 3:
        return 6; // int16 -> i16
    case 4:
        return 4; // int32 -> i32
    case 5:
        return 5; // int64 -> i64
    case 6:
        return 3; // float16 -> f16
    case 7:
        return 1; // float32 -> f32
    case 8:
        return 2; // float64 -> f64
    case 9:
        return 12; // complex32 -> c32
    case 10:
        return 10; // complex64 -> c64
    case 11:
        return 11; // complex128 -> c128
    case 12:
        return 9; // bool
    case 13:
        return 13; // bfloat16
    default:
        fprintf(stderr, "unsupported serde dtype %lld\n", (long long)dtype);
        return 0;
    }
}

// torch serde ScalarType -> pnnx dtype input enum (prim::Constant value)
//   pnnx dtype enum (see pass_level2/Tensor_to.cpp): 0=uint8 1=int8 2=short 3=int
//   4=long 5=half 6=float 7=double 8=complex32 9=complex64 10=complex128 11=bool 15=bfloat16
static int serde_dtype_to_pnnx_dtype_value(int64_t dtype)
{
    switch (dtype)
    {
    case 1:
        return 0; // uint8
    case 2:
        return 1; // int8
    case 3:
        return 2; // int16 -> short
    case 4:
        return 3; // int32 -> int
    case 5:
        return 4; // int64 -> long
    case 6:
        return 5; // float16 -> half
    case 7:
        return 6; // float32 -> float
    case 8:
        return 7; // float64 -> double
    case 9:
        return 8; // complex32
    case 10:
        return 9; // complex64
    case 11:
        return 10; // complex128
    case 12:
        return 11; // bool
    case 13:
        return 15; // bfloat16
    default:
        fprintf(stderr, "unsupported serde dtype %lld\n", (long long)dtype);
        return -1;
    }
}

// torch serde MemoryFormat -> pnnx memory_format enum
//   serde: 0=none 1=contiguous 2=channels_last 3=channels_last_3d 4=preserve
//   pnnx enum (see pass_level2/Tensor_to.cpp): 0=contiguous 1=preserve 2=channels_last
static int serde_memory_format_to_pnnx(int64_t mf)
{
    switch (mf)
    {
    case 1:
        return 0; // contiguous
    case 2:
        return 2; // channels_last
    case 4:
        return 1; // preserve
    case 0:       // none -> preserve
    case 3:       // channels_last_3d has no counterpart, fall back to preserve
    default:
        return 1;
    }
}

static size_t type_to_elemsize(int type)
{
    if (type == 1) return 4;
    if (type == 2) return 8;
    if (type == 3) return 2;
    if (type == 4) return 4;
    if (type == 5) return 8;
    if (type == 6) return 2;
    if (type == 7) return 1;
    if (type == 8) return 1;
    if (type == 9) return 1;
    if (type == 10) return 8;
    if (type == 11) return 16;
    if (type == 12) return 4;
    if (type == 13) return 2;
    return 0;
}

// torch.ops.aten.conv2d.default -> aten::conv2d
static std::string normalize_target(const std::string& target)
{
    std::string t = target;

    const char* prefix = "torch.ops.";
    if (t.compare(0, strlen(prefix), prefix) == 0)
        t = t.substr(strlen(prefix));

    size_t dot = t.rfind('.');
    if (dot != std::string::npos)
    {
        // keep the non-default overloads only for the arange family (pt2
        // arange.start_step / arange.start variants have different arguments
        // than the overload-less aten::arange and cannot be mixed), return
        // them as "aten::arange.start_step" (overload separated by a dot);
        // arange.default / arange.end are normalized to the suffix-less
        // aten::arange (matches the pass_level2 torch_arange patterns)
        if (t.compare(0, 12, "aten.arange.") == 0)
        {
            std::string body = t.substr(0, dot); // "aten.arange"
            std::string overload = t.substr(dot + 1);
            if (overload == "default" || overload == "end")
            {
                t = body;
            }
            else
            {
                std::string rb;
                for (size_t i = 0; i < body.size(); i++)
                {
                    if (body[i] == '.')
                        rb += "::";
                    else
                        rb += body[i];
                }
                return rb + "." + overload;
            }
        }
        else
        {
            t = t.substr(0, dot);
        }
    }

    std::string r;
    for (size_t i = 0; i < t.size(); i++)
    {
        if (t[i] == '.')
            r += "::";
        else
            r += t[i];
    }

    return r;
}

// read the sizes array (each element is {"as_int": n} or {"as_sym_int": ...})
static void read_sizes(const JsonValue& meta, std::vector<int>& shape)
{
    shape.clear();

    if (!meta.is_object())
        return;

    const JsonValue& sizes = meta["sizes"];
    for (size_t i = 0; i < sizes.size(); i++)
    {
        const JsonValue& s = sizes[i];
        if (s.has("as_int"))
            shape.push_back((int)s["as_int"].as_int());
        else
            shape.push_back(-1); // symbolic dimension, unresolved
    }
}

static int read_dtype(const JsonValue& meta)
{
    if (!meta.is_object() || !meta.has("dtype"))
        return 0;

    return serde_dtype_to_pnnx_type(meta["dtype"].as_int());
}

// read one weight/constant record (raw storage bytes) from the zip into an
// Attribute, materializing the logical row-major tensor described by the
// serialized tensor_meta (sizes / strides / storage_offset) so transposed,
// sliced, or shared-storage views are stored correctly.
static void load_tensor_data(StoreZipReader& zip, const std::vector<std::string>& names,
                             const std::string& dir, const std::string& path_name,
                             const JsonValue& meta, Attribute& a)
{
    std::string record;
    for (size_t j = 0; j < names.size(); j++)
    {
        if (names[j].find(dir + "/" + path_name) != std::string::npos || names[j] == path_name)
        {
            record = names[j];
            break;
        }
    }

    if (record.empty())
    {
        fprintf(stderr, "tensor record %s/%s not found\n", dir.c_str(), path_name.c_str());
        return;
    }

    uint64_t size = zip.get_file_size(record);
    std::vector<char> raw((size_t)size);
    zip.read_file(record, raw.data());

    // parse serialized sizes / strides / storage_offset from tensor_meta
    std::vector<int> sizes;
    std::vector<int64_t> strides;
    int64_t storage_offset = 0;
    if (meta.is_object())
    {
        if (meta.has("sizes"))
        {
            const JsonValue& s = meta["sizes"];
            for (size_t i = 0; i < s.size(); i++)
            {
                if (s[i].has("as_int"))
                    sizes.push_back((int)s[i]["as_int"].as_int());
                else
                    sizes.push_back(-1);
            }
        }
        if (meta.has("strides"))
        {
            const JsonValue& st = meta["strides"];
            for (size_t i = 0; i < st.size(); i++)
            {
                if (st[i].has("as_int"))
                    strides.push_back(st[i]["as_int"].as_int());
                else
                    strides.push_back(0);
            }
        }
        if (meta.has("storage_offset") && meta["storage_offset"].has("as_int"))
            storage_offset = meta["storage_offset"]["as_int"].as_int();
    }

    const int dims = (int)sizes.size();
    if (dims == 0 || strides.size() != sizes.size())
    {
        // no usable tensor_meta: keep the raw storage bytes as-is
        a.data = raw;
        return;
    }

    bool symbolic = false;
    size_t count = 1;
    for (int i = 0; i < dims; i++)
    {
        if (sizes[i] <= 0)
        {
            symbolic = true;
            break;
        }
        if (count > (size_t)-1 / (size_t)sizes[i])
        {
            symbolic = true; // product overflow
            break;
        }
        count *= (size_t)sizes[i];
    }

    if (symbolic)
    {
        // dynamic dimension cannot be materialized; keep raw storage
        a.data = raw;
        return;
    }

    const int elemsize = (int)a.elemsize();
    if (elemsize <= 0)
    {
        // unknown/unsupported dtype maps to type 0 with no element size;
        // dividing by zero below would be UB, keep the raw storage instead
        a.data = raw;
        return;
    }

    // the logical tensor is a view of this storage: materializing must never
    // grow the element count beyond what the raw storage holds (only expand/
    // stride tricks can do that, and those carry no data here). an exaggerated
    // tensor_meta (corrupt / hostile sizes) would otherwise make resize/
    // vector-alloc below explode into gigabytes and OOM the process.
    const size_t raw_elems_total = raw.size() / (size_t)elemsize; // full elems in raw
    if (count > raw_elems_total)
    {
        // a zero-stride (expanded) view legitimately repeats elements, so its
        // logical count may exceed the backing storage; materialization below
        // repeats through stride 0 and bounds-checks every source offset. only
        // reject when the excess is not explained by zero-stride dimensions.
        bool has_zero_stride = false;
        for (int i = 0; i < dims; i++)
        {
            if (strides[i] == 0)
            {
                has_zero_stride = true;
                break;
            }
        }
        if (!has_zero_stride)
        {
            // meta claims more elements than the storage provides: keep raw
            // bytes as-is (bounds-checked readers below tolerate a short buffer)
            a.data = raw;
            return;
        }

        // bound the expansion so a hostile meta cannot force a huge allocation:
        // each zero-stride dim may repeat its elements at most to its declared
        // size; a count beyond that product is not a valid expanded view
        size_t expanded = raw_elems_total;
        for (int i = 0; i < dims; i++)
        {
            if (strides[i] == 0 && sizes[i] > 1)
            {
                if (expanded > (size_t)-1 / (size_t)sizes[i])
                {
                    a.data = raw;
                    return;
                }
                expanded *= (size_t)sizes[i];
            }
        }
        if (count > expanded)
        {
            // still more elements than any zero-stride expansion can explain
            a.data = raw;
            return;
        }
    }

    // already row-major contiguous with zero offset? keep raw
    bool contiguous = storage_offset == 0;
    if (contiguous)
    {
        int64_t expected = 1;
        for (int i = dims - 1; i >= 0; i--)
        {
            if (strides[i] != expected)
            {
                contiguous = false;
                break;
            }
            expected *= sizes[i];
        }
    }
    if (contiguous)
    {
        // clamp the raw storage to the logical tensor size: a contiguous view
        // may share a larger storage, and downstream accessors assume
        // data.size() == elemcount * elemsize
        const size_t expect = count * (size_t)elemsize;
        if (raw.size() > expect)
            raw.resize(expect);
        else if (raw.size() < expect)
            raw.resize(expect, 0);
        a.data = raw;
        return;
    }

    if (elemsize <= 0)
    {
        a.data = raw;
        return;
    }

    // guard against integer overflow when allocating the materialized buffer
    if (count > (size_t)-1 / (size_t)elemsize)
    {
        a.data = raw;
        return;
    }

    // materialize row-major from (sizes, strides, storage_offset), bounds-checking
    // every source offset against the raw storage (values come from the archive).
    // use a division compare so the offset multiply cannot overflow.
    const size_t raw_elems = raw.size() / (size_t)elemsize; // floor: full elems in raw
    std::vector<char> out(count * (size_t)elemsize);
    char* dst = out.data();
    const char* src = raw.data();
    for (size_t n = 0; n < count; n++)
    {
        size_t tmp = n;
        int64_t sto = storage_offset;
        for (int i = dims - 1; i >= 0; i--)
        {
            const int idx = (int)(tmp % (size_t)sizes[i]);
            tmp /= (size_t)sizes[i];
            sto += (int64_t)idx * strides[i];
        }
        if (sto < 0 || (uint64_t)sto >= (uint64_t)raw_elems)
        {
            // out-of-bounds source: this tensor_meta cannot address the raw
            // storage, so the shape/strides are inconsistent with it. keep the
            // raw bytes as-is (never grow it to the claimed size - that is how
            // corrupt meta turns into a multi-gigabyte allocation).
            a.data = raw;
            return;
        }
        memcpy(dst + n * (size_t)elemsize, src + sto * (size_t)elemsize, (size_t)elemsize);
    }
    a.data = out;
}

// create a prim::Constant operator and wire it as an input of the consumer
// note: must be inserted before the consumer so that pass_level3
// fuse_expression (iterating backwards) handles the consumer first while the
// constant is still a prim::Constant and can be inlined correctly; otherwise
// the constant is fused into a pnnx.Expression first and the consumer's expr
// ends up with dangling @N references
static void new_constant(Graph& g, Operator* consumer, const Parameter& value, int& constant_index)
{
    char name[32];
    snprintf(name, 32, "pnnx_constant_%d", constant_index++);

    Operator* op = g.new_operator_before("prim::Constant", name, consumer);
    op->params["value"] = value;

    Operand* r = g.new_operand(name);
    r->producer = op;
    op->outputs.push_back(r);

    r->consumers.push_back(consumer);
    consumer->inputs.push_back(r);
}

// append default scalar inputs for aten operators that omitted default kwargs
// dynamo omits schema default arguments; fill them by parameter name here,
// keeping the input order consistent with the pass_level2 patterns (omitted
// ones are trailing defaults, so appending keeps the order)
static bool has_input_name(const std::vector<std::string>& inputnames, const std::string& name)
{
    for (size_t i = 0; i < inputnames.size(); i++)
        if (inputnames[i] == name)
            return true;
    return false;
}

// find the prim::Constant value of an input by parameter name
// used for defaults that depend on other params, e.g. stride = kernel_size
static Parameter find_input_value(Operator* op, const std::vector<std::string>& inputnames, const std::string& name)
{
    for (size_t i = 0; i < inputnames.size() && i < op->inputs.size(); i++)
    {
        if (inputnames[i] == name)
        {
            Operator* prod = op->inputs[i]->producer;
            if (prod && prod->type == "prim::Constant" && prod->params.find("value") != prod->params.end())
                return prod->params["value"];
        }
    }
    return Parameter();
}

static void append_default_kwargs(Graph& g, Operator* op, const std::string& type, const std::vector<std::string>& inputnames, int& constant_index)
{
    // add a default constant input and keep op->inputnames in sync with op->inputs
    auto add_const = [&](const std::string& name, const Parameter& value) {
        new_constant(g, op, value, constant_index);
        op->inputnames.push_back(name);
    };

    // reorder op inputs/inputnames to a canonical schema order (the exported
    // overload may omit middle defaults, and blindly appending them would
    // misalign the level-2 patterns that match by position)
    auto reorder_inputs = [&](const std::vector<std::string>& order) {
        std::vector<Operand*> new_inputs;
        std::vector<std::string> new_names;
        for (const std::string& nm : order)
        {
            for (size_t k = 0; k < op->inputnames.size(); k++)
            {
                if (op->inputnames[k] == nm)
                {
                    new_inputs.push_back(op->inputs[k]);
                    new_names.push_back(nm);
                    break;
                }
            }
        }
        if (new_names.size() == op->inputnames.size())
        {
            op->inputs = new_inputs;
            op->inputnames = new_names;
        }
    };

    if (type == "aten::conv1d" || type == "aten::conv2d" || type == "aten::conv3d")
    {
        int dim = 2;
        if (type == "aten::conv1d")
            dim = 1;
        else if (type == "aten::conv3d")
            dim = 3;

        std::vector<int> ones(dim, 1);
        std::vector<int> zeros(dim, 0);

        if (!has_input_name(inputnames, "stride"))
            add_const("stride", ones);
        if (!has_input_name(inputnames, "padding"))
            add_const("padding", zeros);
        if (!has_input_name(inputnames, "dilation"))
            add_const("dilation", ones);
        if (!has_input_name(inputnames, "groups"))
            add_const("groups", 1);
    }
    else if (type == "aten::batch_norm")
    {
        if (!has_input_name(inputnames, "training"))
            add_const("training", false);
        if (!has_input_name(inputnames, "momentum"))
            add_const("momentum", 0.1f);
        if (!has_input_name(inputnames, "eps"))
            add_const("eps", 1e-5f);
        if (!has_input_name(inputnames, "cudnn_enabled"))
            add_const("cudnn_enabled", true);
    }
    else if (type == "aten::add")
    {
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1);
    }
    else if (type == "aten::max_pool1d" || type == "aten::max_pool2d" || type == "aten::max_pool3d"
             || type == "aten::max_pool1d_with_indices" || type == "aten::max_pool2d_with_indices" || type == "aten::max_pool3d_with_indices")
    {
        int dim = 2;
        if (type == "aten::max_pool1d" || type == "aten::max_pool1d_with_indices")
            dim = 1;
        else if (type == "aten::max_pool3d" || type == "aten::max_pool3d_with_indices")
            dim = 3;

        // torch max_pool stride defaults to kernel_size
        Parameter kernel = find_input_value(op, inputnames, "kernel_size");

        if (!has_input_name(inputnames, "stride"))
        {
            if (kernel.type == 5)
                add_const("stride", kernel.ai);
            else
                add_const("stride", std::vector<int>(dim, 1));
        }
        if (!has_input_name(inputnames, "padding"))
            add_const("padding", std::vector<int>(dim, 0));
        if (!has_input_name(inputnames, "dilation"))
            add_const("dilation", std::vector<int>(dim, 1));
        if (!has_input_name(inputnames, "ceil_mode"))
            add_const("ceil_mode", false);
    }
    else if (type == "aten::avg_pool1d" || type == "aten::avg_pool2d" || type == "aten::avg_pool3d")
    {
        int dim = 2;
        if (type == "aten::avg_pool1d")
            dim = 1;
        else if (type == "aten::avg_pool3d")
            dim = 3;

        // torch avg_pool stride defaults to kernel_size
        Parameter kernel = find_input_value(op, inputnames, "kernel_size");

        if (!has_input_name(inputnames, "stride"))
        {
            if (kernel.type == 5)
                add_const("stride", kernel.ai);
            else
                add_const("stride", std::vector<int>(dim, 1));
        }
        if (!has_input_name(inputnames, "padding"))
            add_const("padding", std::vector<int>(dim, 0));
        if (!has_input_name(inputnames, "ceil_mode"))
            add_const("ceil_mode", false);
        if (!has_input_name(inputnames, "count_include_pad"))
            add_const("count_include_pad", true);
        if (type != "aten::avg_pool1d")
        {
            if (!has_input_name(inputnames, "divisor_override"))
                add_const("divisor_override", Parameter());
        }
    }
    else if (type == "aten::argmax" || type == "aten::argmin")
    {
        // dim=None means full reduction (torch.argmax(x) without dim)
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
    }
    else if (type == "aten::sum" || type == "aten::mean")
    {
        // mean.default / sum full-reduction versions have no keepdim argument
        if (has_input_name(inputnames, "dim"))
        {
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            if (!has_input_name(inputnames, "dtype"))
                add_const("dtype", Parameter());
            // keepdim sits before dtype in the schema; when dtype was already
            // serialized but keepdim omitted, reorder so the level-2 pattern
            // does not read dtype as keepdim
            reorder_inputs({"self", "dim", "keepdim", "dtype"});
        }
        else
        {
            // full-reduction overload: self (+ optional dtype) is already in order
            if (!has_input_name(inputnames, "dtype"))
                add_const("dtype", Parameter());
        }
    }
    else if (type == "aten::var" || type == "aten::std")
    {
        // aten::var/std.dim overloads serialize an unbiased argument; the
        // .correction overload serializes correction. never mix the two families
        // (a 5-input node matches no level-2 pattern), and keep the defaults in
        // canonical schema order so the torch_std/torch_var rewrites match.
        if (has_input_name(inputnames, "unbiased"))
        {
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            reorder_inputs({"self", "dim", "unbiased", "keepdim"});
        }
        else if (has_input_name(inputnames, "correction"))
        {
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            reorder_inputs({"self", "dim", "correction", "keepdim"});
        }
        else if (has_input_name(inputnames, "dim"))
        {
            // dim overload with the default unbiased/keepdim omitted
            if (!has_input_name(inputnames, "unbiased"))
                add_const("unbiased", true);
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            reorder_inputs({"self", "dim", "unbiased", "keepdim"});
        }
        else
        {
            // reduce-all overload (self only, serialized under .correction)
            if (!has_input_name(inputnames, "correction"))
                add_const("correction", 1);
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
        }
    }
    else if (type == "aten::softmax" || type == "aten::log_softmax")
    {
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
    }
    else if (type == "aten::pad")
    {
        if (!has_input_name(inputnames, "mode"))
            add_const("mode", std::string("constant"));
        if (!has_input_name(inputnames, "value"))
            add_const("value", Parameter());
    }
    else if (type == "aten::to")
    {
        if (!has_input_name(inputnames, "non_blocking"))
            add_const("non_blocking", false);
        if (!has_input_name(inputnames, "copy"))
            add_const("copy", false);
        if (!has_input_name(inputnames, "memory_format"))
            add_const("memory_format", Parameter());
    }
    else if (type == "aten::contiguous")
    {
        // eliminate_contiguous expects 2 inputs (input + memory_format); dynamo omits memory_format
        if (!has_input_name(inputnames, "memory_format"))
            add_const("memory_format", Parameter());
    }
    else if (type == "aten::slice")
    {
        if (!has_input_name(inputnames, "step"))
            add_const("step", 1);
    }
    else if (type == "aten::slice_scatter")
    {
        if (!has_input_name(inputnames, "end"))
            add_const("end", INT_MAX);
        if (!has_input_name(inputnames, "step"))
            add_const("step", 1);
    }
    else if (type == "aten::flatten")
    {
        if (!has_input_name(inputnames, "start_dim"))
            add_const("start_dim", 0);
        if (!has_input_name(inputnames, "end_dim"))
            add_const("end_dim", -1);
    }
    else if (type == "aten::celu")
    {
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1.0f);
    }
    else if (type == "aten::elu")
    {
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1.0f);
        if (!has_input_name(inputnames, "scale"))
            add_const("scale", 1.0f);
        if (!has_input_name(inputnames, "input_scale"))
            add_const("input_scale", 1.0f);
    }
    else if (type == "aten::hardshrink")
    {
        if (!has_input_name(inputnames, "lambd"))
            add_const("lambd", 0.5f);
    }
    else if (type == "aten::hardtanh")
    {
        if (!has_input_name(inputnames, "min_val"))
            add_const("min_val", -1.0f);
        if (!has_input_name(inputnames, "max_val"))
            add_const("max_val", 1.0f);
    }
    else if (type == "aten::leaky_relu")
    {
        if (!has_input_name(inputnames, "negative_slope"))
            add_const("negative_slope", 0.01f);
    }
    else if (type == "aten::softplus")
    {
        if (!has_input_name(inputnames, "beta"))
            add_const("beta", 1.0f);
        if (!has_input_name(inputnames, "threshold"))
            add_const("threshold", 20.0f);
    }
    else if (type == "aten::softshrink")
    {
        if (!has_input_name(inputnames, "lambd"))
            add_const("lambd", 0.5f);
    }
    else if (type == "aten::rrelu")
    {
        if (!has_input_name(inputnames, "lower"))
            add_const("lower", 0.125f);
        if (!has_input_name(inputnames, "upper"))
            add_const("upper", 1.0f / 3.0f);
        if (!has_input_name(inputnames, "training"))
            add_const("training", false);
        if (!has_input_name(inputnames, "generator"))
            add_const("generator", Parameter());
    }
    else if (type == "aten::pairwise_distance")
    {
        if (!has_input_name(inputnames, "p"))
            add_const("p", 2);
        if (!has_input_name(inputnames, "eps"))
            add_const("eps", 1e-6f);
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
    }
    else if (type == "aten::linear")
    {
        if (!has_input_name(inputnames, "bias"))
            add_const("bias", Parameter());
    }
    else if (type == "aten::rms_norm")
    {
        if (!has_input_name(inputnames, "weight"))
            add_const("weight", Parameter());
        if (!has_input_name(inputnames, "eps"))
            add_const("eps", Parameter());
    }
    else if (type == "aten::scaled_dot_product_attention")
    {
        // dynamo omits intermediate default args and enable_gqa moves up to the
        // attn_mask slot; reorder to match the pattern:
        // query key value attn_mask dropout_p is_causal scale enable_gqa
        static const std::vector<std::string> order = {"query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"};

        std::vector<Operand*> old_inputs = op->inputs;
        std::vector<std::string> old_names = op->inputnames;
        op->inputs.clear();
        op->inputnames.clear();

        for (size_t i = 0; i < order.size(); i++)
        {
            const std::string& name = order[i];

            int found = -1;
            for (size_t j = 0; j < old_names.size(); j++)
            {
                if (old_names[j] == name)
                {
                    found = (int)j;
                    break;
                }
            }

            if (found != -1)
            {
                op->inputs.push_back(old_inputs[found]);
                op->inputnames.push_back(name);
            }
            else
            {
                Parameter v;
                if (name == "attn_mask")
                    v = Parameter();
                else if (name == "dropout_p")
                    v = Parameter(0.0f);
                else if (name == "is_causal")
                    v = Parameter(false);
                else if (name == "scale")
                    v = Parameter();
                else if (name == "enable_gqa")
                    v = Parameter(false);

                new_constant(g, op, v, constant_index);
                op->inputnames.push_back(name);
            }
        }
    }
    else if (type == "aten::embedding")
    {
        if (!has_input_name(inputnames, "padding_idx"))
            add_const("padding_idx", -1);
        if (!has_input_name(inputnames, "scale_grad_by_freq"))
            add_const("scale_grad_by_freq", false);
        if (!has_input_name(inputnames, "sparse"))
            add_const("sparse", false);
    }
    else if (type == "aten::glu")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", -1);
    }
    else if (type == "aten::conv_transpose1d" || type == "aten::conv_transpose2d" || type == "aten::conv_transpose3d")
    {
        int dim = 2;
        if (type == "aten::conv_transpose1d")
            dim = 1;
        else if (type == "aten::conv_transpose3d")
            dim = 3;

        if (!has_input_name(inputnames, "stride"))
            add_const("stride", std::vector<int>(dim, 1));
        if (!has_input_name(inputnames, "padding"))
            add_const("padding", std::vector<int>(dim, 0));
        if (!has_input_name(inputnames, "output_padding"))
            add_const("output_padding", std::vector<int>(dim, 0));
        if (!has_input_name(inputnames, "groups"))
            add_const("groups", 1);
        if (!has_input_name(inputnames, "dilation"))
            add_const("dilation", std::vector<int>(dim, 1));
    }
    else if (type == "aten::amax" || type == "aten::amin")
    {
        // dim defaults to None (reduce all); when omitted add the null dim slot
        // so [self, dim, keepdim] matches the level-2 pattern instead of
        // leaving a keepdim-only node that no pattern rewrites
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
        reorder_inputs({"self", "dim", "keepdim"});
    }
    else if (type == "aten::max" || type == "aten::min")
    {
        // max.dim/min.dim omit keepdim when it is False; inputnames has dim but
        // lacks keepdim (max.other/default have no dim and are unaffected)
        if (has_input_name(inputnames, "dim") && !has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
    }
    else if (type == "aten::logsumexp")
    {
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
    }
    else if (type == "aten::prod")
    {
        // only the dim overload (prod(x, dim)) takes keepdim; the full-reduction
        // overload prod(x) has no dim/keepdim inputs, and appending keepdim here
        // would yield [input, keepdim, dtype], which no level-2 pattern matches
        if (has_input_name(inputnames, "dim"))
        {
            if (!has_input_name(inputnames, "keepdim"))
                add_const("keepdim", false);
            if (!has_input_name(inputnames, "dtype"))
                add_const("dtype", Parameter());
            reorder_inputs({"self", "dim", "keepdim", "dtype"});
        }
        else
        {
            // full-reduction prod(x): self (+ optional dtype) is already in order
            if (!has_input_name(inputnames, "dtype"))
                add_const("dtype", Parameter());
        }
    }
    else if (type == "aten::cumsum")
    {
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
    }
    else if (type == "aten::cumprod")
    {
        // cumprod(x, dim) carries an omitted kwonly dtype default; append it so
        // the [input dim dtype] level-2 pattern matches
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        reorder_inputs({"self", "dim", "dtype"});
    }
    else if (type == "aten::roll")
    {
        // roll(x, shifts) with omitted dims=None: restore [input shifts dims]
        if (!has_input_name(inputnames, "dims"))
            add_const("dims", Parameter());
        reorder_inputs({"self", "shifts", "dims"});
    }
    else if (type == "aten::repeat_interleave")
    {
        // repeat_interleave(x, repeats) omits dim/output_size defaults
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
        if (!has_input_name(inputnames, "output_size"))
            add_const("output_size", Parameter());
        reorder_inputs({"self", "repeats", "dim", "output_size"});
    }
    else if (type == "aten::topk")
    {
        // topk(x, k) omits dim=-1/largest=True/sorted=True defaults; torch's
        // sorted default is True so the emitted values are in descending order
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", -1);
        if (!has_input_name(inputnames, "largest"))
            add_const("largest", true);
        if (!has_input_name(inputnames, "sorted"))
            add_const("sorted", true);
        reorder_inputs({"self", "k", "dim", "largest", "sorted"});
    }
    else if (type == "aten::istft")
    {
        // dynamo omits istft trailing defaults (onesided/length/return_complex)
        if (!has_input_name(inputnames, "onesided"))
            add_const("onesided", Parameter());
        if (!has_input_name(inputnames, "length"))
            add_const("length", Parameter());
        if (!has_input_name(inputnames, "return_complex"))
            add_const("return_complex", false);
    }
    else if (type == "aten::cat" || type == "aten::stack")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
    }
    else if (type == "aten::chunk" || type == "aten::unbind")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
    }
    else if (type == "aten::split" || type == "aten::split_with_sizes" || type == "aten::tensor_split")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
    }
    else if (type == "aten::diag")
    {
        if (!has_input_name(inputnames, "diagonal"))
            add_const("diagonal", 0);
    }
    else if (type == "aten::clone")
    {
        // dynamo omits memory_format; torch.clone defaults to preserve_format(=1)
        if (!has_input_name(inputnames, "memory_format"))
            add_const("memory_format", 1);
    }
    else if (type == "aten::addmm")
    {
        // beta/alpha are trailing keyword-only defaults; a call supplying only
        // one of them (e.g. addmm(b, m1, m2, alpha=2)) serializes the omitted
        // one out of place, so restore the canonical [self mat1 mat2 beta alpha]
        // order for the level-2 pattern
        if (!has_input_name(inputnames, "beta"))
            add_const("beta", 1);
        if (!has_input_name(inputnames, "alpha"))
            add_const("alpha", 1);
        reorder_inputs({"self", "mat1", "mat2", "beta", "alpha"});
    }
    else if (type == "aten::linalg_vector_norm")
    {
        // dtype is a keyword-only trailing default; restore the canonical
        // [self ord dim keepdim dtype] order when it was serialized ahead of
        // the omitted ord/dim/keepdim defaults
        if (!has_input_name(inputnames, "ord"))
            add_const("ord", 2.0f);
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", Parameter());
        if (!has_input_name(inputnames, "keepdim"))
            add_const("keepdim", false);
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        reorder_inputs({"self", "ord", "dim", "keepdim", "dtype"});
    }
    else if (type == "aten::_weight_norm")
    {
        if (!has_input_name(inputnames, "dim"))
            add_const("dim", 0);
    }
    else if (type == "aten::clamp")
    {
        if (!has_input_name(inputnames, "min"))
            add_const("min", Parameter());
        if (!has_input_name(inputnames, "max"))
            add_const("max", Parameter());
    }
    else if (type == "aten::zeros" || type == "aten::ones")
    {
        // dynamo omits the dtype default (None) from aten::zeros/ones, leaving
        // [size device pin_memory] which matches neither the level-2 fold nor
        // the torch.zeros rewrite; restore the canonical [size dtype device
        // pin_memory] order so the constant folds to an Attribute
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        reorder_inputs({"self", "size", "dtype", "layout", "device", "pin_memory"});
    }
    else if (type == "aten::full")
    {
        // same dtype-default omission for aten::full; restore
        // [size fill_value dtype device pin_memory] for the level-2 fold
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        reorder_inputs({"self", "size", "fill_value", "dtype", "layout", "device", "pin_memory"});
    }
    else if (type == "aten::new_full")
    {
        // Tensor.new_full(self, size, fill_value) omits the dtype/layout/
        // device defaults; restore the canonical order for the level-2 fold
        if (!has_input_name(inputnames, "dtype"))
            add_const("dtype", Parameter());
        if (!has_input_name(inputnames, "layout"))
            add_const("layout", Parameter());
        if (!has_input_name(inputnames, "device"))
            add_const("device", Parameter());
        reorder_inputs({"self", "size", "fill_value", "dtype", "layout", "device", "pin_memory"});
    }
    else if (type == "aten::new_zeros" || type == "aten::new_ones" || type == "aten::new_empty")
    {
        // dynamo emits Tensor.new_zeros(self, size, pin_memory); the pnnx
        // pass_level2 pattern expects input size dtype layout device pin_memory.
        // GraphRewriter matches constant inputs POSITIONALLY, so the final
        // order must be exactly: input size dtype layout device pin_memory.
        std::vector<Operand*> old_inputs = op->inputs;
        std::vector<std::string> old_names = op->inputnames;

        // detach op from all old inputs; keep the interesting ones
        for (size_t j = 0; j < old_inputs.size(); j++)
        {
            auto& cons = old_inputs[j]->consumers;
            cons.erase(std::find(cons.begin(), cons.end(), op));
        }

        op->inputs.clear();
        op->inputnames.clear();

        Operand* self_op = 0;
        Operand* size_op = 0;
        Operand* dtype_op = 0;
        Operand* layout_op = 0;
        Operand* device_op = 0;
        bool have_pin = false;
        for (size_t j = 0; j < old_names.size(); j++)
        {
            std::string n = old_names[j];
            if (n == "self" || n == "input")
                self_op = old_inputs[j];
            else if (n == "size")
                size_op = old_inputs[j];
            else if (n == "dtype")
                dtype_op = old_inputs[j];
            else if (n == "layout")
                layout_op = old_inputs[j];
            else if (n == "device")
                device_op = old_inputs[j];
            else if (n == "pin_memory")
                have_pin = true;
        }

        // rebuild in pattern order: input size dtype layout device pin_memory
        if (self_op)
        {
            op->inputs.push_back(self_op);
            op->inputnames.push_back("input");
            self_op->consumers.push_back(op);
        }
        if (size_op)
        {
            op->inputs.push_back(size_op);
            op->inputnames.push_back("size");
            size_op->consumers.push_back(op);
        }
        // (size is always present for new_*; if it were missing the pattern
        //  simply will not match and the op stays as-is, which is safe)
        if (dtype_op)
        {
            // keep an explicit dtype constant when dynamo emitted one
            // (e.g. x.new_empty(..., dtype=torch.long)); otherwise null means
            // "inherit self's dtype", matching the pattern default
            op->inputs.push_back(dtype_op);
            op->inputnames.push_back("dtype");
            dtype_op->consumers.push_back(op);
        }
        else
        {
            add_const("dtype", Parameter());
        }
        if (layout_op)
        {
            op->inputs.push_back(layout_op);
            op->inputnames.push_back("layout");
            layout_op->consumers.push_back(op);
        }
        else
        {
            add_const("layout", Parameter());
        }
        if (device_op)
        {
            op->inputs.push_back(device_op);
            op->inputnames.push_back("device");
            device_op->consumers.push_back(op);
        }
        else
        {
            add_const("device", Parameter());
        }
        add_const("pin_memory", have_pin); // fresh constant either way
    }
    else if (type == "aten::ones_like" || type == "aten::zeros_like"
             || type == "aten::rand_like" || type == "aten::randn_like"
             || type == "aten::empty_like" || type == "aten::full_like")
    {
        // dynamo emits input [dtype|fill_value] pin_memory; the pnnx pattern
        // expects input dtype layout device requires_grad memory_format
        // (full_like additionally carries fill_value)
        std::vector<Operand*> old_inputs = op->inputs;
        std::vector<std::string> old_names = op->inputnames;
        for (size_t j = 0; j < old_names.size(); j++)
        {
            if (old_names[j] != "self" && old_names[j] != "input" && old_names[j] != "dtype")
            {
                // drop irrelevant inputs like pin_memory and clean up consumer refs
                auto& cons = old_inputs[j]->consumers;
                cons.erase(std::find(cons.begin(), cons.end(), op));
            }
        }

        op->inputs.clear();
        op->inputnames.clear();

        int found_input = -1;
        for (size_t j = 0; j < old_names.size(); j++)
            if (old_names[j] == "self" || old_names[j] == "input")
            {
                found_input = (int)j;
                break;
            }
        if (found_input != -1)
        {
            op->inputs.push_back(old_inputs[found_input]);
            op->inputnames.push_back("input");
        }

        int found_dtype = -1;
        for (size_t j = 0; j < old_names.size(); j++)
            if (old_names[j] == "dtype")
            {
                found_dtype = (int)j;
                break;
            }
        if (found_dtype != -1)
        {
            op->inputs.push_back(old_inputs[found_dtype]);
            op->inputnames.push_back("dtype");
        }
        else
        {
            add_const("dtype", Parameter());
        }

        // full_like: dynamo passes fill_value as a scalar input; the pnnx
        // pattern wants it as the first input (input fill_value dtype ...)
        if (type == "aten::full_like")
        {
            int found_fill = -1;
            for (size_t j = 0; j < old_names.size(); j++)
                if (old_names[j] == "fill_value")
                {
                    found_fill = (int)j;
                    break;
                }
            if (found_fill != -1)
            {
                Operand* fill = old_inputs[found_fill];
                op->inputs.insert(op->inputs.begin() + 1, fill);
                op->inputnames.insert(op->inputnames.begin() + 1, "fill_value");
            }
            else
            {
                // fill_value missing: synthesize a constant 0 and place it at
                // index 1 (new_constant appends at the end)
                add_const("fill_value", 0);
                Operand* fill = op->inputs.back();
                op->inputs.pop_back();
                op->inputs.insert(op->inputs.begin() + 1, fill);
                op->inputnames.pop_back();
                op->inputnames.insert(op->inputnames.begin() + 1, "fill_value");
            }
        }

        add_const("layout", Parameter());
        add_const("device", Parameter());
        add_const("requires_grad", false);
        add_const("memory_format", Parameter());
    }
}

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
                    // float64); keep it so the level2 fold can honor it
                    int pnnx_type = serde_dtype_to_pnnx_type(arg["as_scalar_type"].as_int());
                    if (pnnx_type > 0)
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
                new_constant(g, op, serde_dtype_to_pnnx_dtype_value(arg["as_scalar_type"].as_int()), constant_index);
            }
            else if (arg.has("as_tensors"))
            {
                char lc_name[32];
                snprintf(lc_name, 32, "pnnx_list_%d", constant_index++);

                Operator* lc = g.new_operator("prim::ListConstruct", lc_name);
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
    std::vector<std::string> names = zip.get_names();
    std::string model_json_name;
    std::string weights_config_name;
    std::string constants_config_name;
    for (size_t i = 0; i < names.size(); i++)
    {
        if (model_json_name.empty() && names[i].find("models/model.json") != std::string::npos)
            model_json_name = names[i];
        if (weights_config_name.empty() && names[i].find("weights") != std::string::npos && names[i].find("config") != std::string::npos && names[i].size() >= 5 && names[i].compare(names[i].size() - 5, 5, ".json") == 0)
            weights_config_name = names[i];
        if (constants_config_name.empty() && names[i].find("constants") != std::string::npos && names[i].find("config") != std::string::npos && names[i].size() >= 5 && names[i].compare(names[i].size() - 5, 5, ".json") == 0)
            constants_config_name = names[i];
    }

    if (model_json_name.empty())
    {
        fprintf(stderr, "models/model.json not found in %s\n", pt2path.c_str());
        return -1;
    }

    // read and parse model json
    JsonValue root;
    {
        uint64_t size = zip.get_file_size(model_json_name);
        std::vector<char> buf((size_t)size + 1);
        zip.read_file(model_json_name, buf.data());
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
        zip.read_file(weights_config_name, buf.data());
        buf[size] = 0;

        if (JsonParser::parse(buf.data(), (size_t)size, cfg))
        {
            const std::map<std::string, JsonValue>& c = cfg["config"].as_object();
            for (std::map<std::string, JsonValue>::const_iterator it = c.begin(); it != c.end(); ++it)
            {
                std::string path_name = it->second["path_name"].as_string();
                JsonValue meta = it->second["tensor_meta"];
                weights[it->first] = std::make_pair(path_name, meta);
            }
        }
    }

    // constants payload config : fqn -> { path_name, tensor_meta }
    std::map<std::string, std::pair<std::string, JsonValue> > constants;
    if (!constants_config_name.empty())
    {
        JsonValue cfg;
        uint64_t size = zip.get_file_size(constants_config_name);
        std::vector<char> buf((size_t)size + 1);
        zip.read_file(constants_config_name, buf.data());
        buf[size] = 0;

        if (JsonParser::parse(buf.data(), (size_t)size, cfg))
        {
            const std::map<std::string, JsonValue>& c = cfg["config"].as_object();
            for (std::map<std::string, JsonValue>::const_iterator it = c.begin(); it != c.end(); ++it)
            {
                std::string path_name = it->second["path_name"].as_string();
                JsonValue meta = it->second["tensor_meta"];
                constants[it->first] = std::make_pair(path_name, meta);
            }
        }
    }

    const JsonValue& graph = root["graph_module"]["graph"];
    const JsonValue& signature = root["graph_module"]["signature"];

    // tensor_values : name -> { dtype, sizes, ... }
    const JsonValue& tensor_values = graph["tensor_values"];

    // uint16 (serde ScalarType 28) has no pnnx representation: reject it
    // explicitly instead of decoding 2-byte elements as 1-byte u8, which would
    // silently truncate or misalign every weight/constant using that dtype
    {
        std::string reject_fqn;
        for (std::map<std::string, std::pair<std::string, JsonValue> >::const_iterator it = weights.begin(); it != weights.end(); ++it)
        {
            const JsonValue& meta = it->second.second;
            if (meta.is_object() && meta.has("dtype") && meta["dtype"].as_int() == 28)
            {
                reject_fqn = it->first;
                break;
            }
        }
        if (reject_fqn.empty())
        {
            for (std::map<std::string, std::pair<std::string, JsonValue> >::const_iterator it = constants.begin(); it != constants.end(); ++it)
            {
                const JsonValue& meta = it->second.second;
                if (meta.is_object() && meta.has("dtype") && meta["dtype"].as_int() == 28)
                {
                    reject_fqn = it->first;
                    break;
                }
            }
        }
        if (!reject_fqn.empty())
        {
            fprintf(stderr, "unsupported dtype uint16 for tensor '%s'\n", reject_fqn.c_str());
            return -1;
        }
    }

    // name -> operand
    std::map<std::string, Operand*> operands_by_name;

    int constant_index = 0;
    int subop_index = 0;

    // pass 1 : build graph inputs
    const JsonValue& input_specs = signature["input_specs"];
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
                load_tensor_data(zip, names, "weights", path_name, meta, a);

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
                load_tensor_data(zip, names, "constants", path_name, meta, a);

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
                load_tensor_data(zip, names, "constants", path_name, meta, a);

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
                    // float64); keep it so the level2 fold can honor it
                    int pnnx_type = serde_dtype_to_pnnx_type(arg["as_scalar_type"].as_int());
                    if (pnnx_type > 0)
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
                    // INT64_MAX is dynamo's "to the end" sentinel for slice etc., map to pnnx INT_MAX
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
                    // serde ScalarType enum -> pnnx dtype input enum
                    new_constant(g, op, serde_dtype_to_pnnx_dtype_value(arg["as_scalar_type"].as_int()), constant_index);
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

                    Operator* lc = g.new_operator("prim::ListConstruct", lc_name);
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
