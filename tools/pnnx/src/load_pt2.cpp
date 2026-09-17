// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"
#include "pt2_schema.h"
#include "aten_defaults_table.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <limits>

namespace pnnx {

static int pt2_dtype_enum_to_pnnx_type(long long dtype)
{
    switch (dtype)
    {
    case 1:
        return 8; // u8
    case 2:
        return 7; // i8
    case 3:
        return 6; // i16
    case 4:
        return 4; // i32
    case 7:
        return 1; // f32
    case 5:
        return 5; // i64
    case 6:
        return 3; // f16
    case 8:
        return 2; // f64
    case 9:
        return 12; // c32
    case 10:
        return 10; // c64
    case 11:
        return 11; // c128
    case 12:
        return 9; // bool
    case 13:
        return 13; // bf16
    default:
        return 0;
    }
}

static int pt2_dtype_to_pnnx_type(long long dtype)
{
    const int type = pt2_dtype_enum_to_pnnx_type(dtype);
    if (type == 0)
        fprintf(stderr, "load_pt2: unsupported weight dtype %lld\n", dtype);
    return type;
}

static bool pt2_scalar_type_to_jit_type(long long scalar_type, long long& jit_type)
{
    switch (scalar_type)
    {
    case 1:
        jit_type = 0;
        return true;
    case 2:
        jit_type = 1;
        return true;
    case 3:
        jit_type = 2;
        return true;
    case 4:
        jit_type = 3;
        return true;
    case 5:
        jit_type = 4;
        return true;
    case 6:
        jit_type = 5;
        return true;
    case 7:
        jit_type = 6;
        return true;
    case 8:
        jit_type = 7;
        return true;
    case 9:
        jit_type = 8;
        return true;
    case 10:
        jit_type = 9;
        return true;
    case 11:
        jit_type = 10;
        return true;
    case 12:
        jit_type = 11;
        return true;
    case 13:
        jit_type = 15;
        return true;
    default:
        return false;
    }
}

static int pnnx_type_from_string(const std::string& t)
{
    if (t == "f32") return 1;
    if (t == "f64") return 2;
    if (t == "f16") return 3;
    if (t == "i32") return 4;
    if (t == "i64") return 5;
    if (t == "i16") return 6;
    if (t == "i8") return 7;
    if (t == "u8") return 8;
    if (t == "bool") return 9;
    if (t == "c64") return 10;
    if (t == "c128") return 11;
    if (t == "c32") return 12;
    if (t == "bf16") return 13;
    return 0;
}

static bool checked_size_add(size_t a, size_t b, size_t& c)
{
    if (b > std::numeric_limits<size_t>::max() - a)
        return false;

    c = a + b;
    return true;
}

static bool checked_size_mul(size_t a, size_t b, size_t& c)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        return false;

    c = a * b;
    return true;
}

static void apply_input_shape(Operand* r, const std::vector<int64_t>& input_shape)
{
    if (!r->shape.empty())
        return;

    for (size_t i = 0; i < input_shape.size(); i++)
        r->shape.push_back((int)input_shape[i]);
}

static bool append_tensor_list_item(Graph& pg, Operator* op_list, const Pt2TensorRef& tensor_ref,
                                    const std::string& node_name, const std::string& input_name,
                                    size_t item_index, int& pnnx_unknown_index)
{
    Operand* r = 0;
    if (tensor_ref.is_none)
    {
        Operator* op_const = pg.new_operator("prim::Constant", "pnnx_" + std::to_string(pnnx_unknown_index++));
        op_const->params["value"] = Parameter();

        r = pg.new_operand(node_name + "." + input_name + "." + std::to_string(item_index));
        r->producer = op_const;
        op_const->outputs.push_back(r);
    }
    else
    {
        r = pg.get_operand(tensor_ref.name);
        if (!r)
        {
            fprintf(stderr, "load_pt2: operand not found %s (node %s)\n", tensor_ref.name.c_str(), node_name.c_str());
            return false;
        }
    }

    r->consumers.push_back(op_list);
    op_list->inputs.push_back(r);
    return true;
}

// Match TorchScript kind display by dropping the PT2 overload suffix.
static std::string map_pt2_target(const std::string& target)
{
    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return target;

    std::string rest = target.substr(prefix.size()); // "aten.conv2d.default"

    const size_t dot1 = rest.find('.');
    const size_t dot2 = rest.find('.', dot1 + 1);
    if (dot1 == std::string::npos)
        return rest;

    if (dot2 == std::string::npos)
        return rest.substr(0, dot1) + "::" + rest.substr(dot1 + 1);

    return rest.substr(0, dot1) + "::" + rest.substr(dot1 + 1, dot2 - dot1 - 1);
}

// Keep the overload for default-table lookup.
static std::string pt2_full_target_name(const std::string& target)
{
    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return target;

    const std::string rest = target.substr(prefix.size()); // "aten.conv2d.default"

    const size_t dot1 = rest.find('.');
    if (dot1 == std::string::npos)
        return rest;

    return rest.substr(0, dot1) + "::" + rest.substr(dot1 + 1);
}

static bool default_value_to_parameter(int type, const char* value, Parameter& p)
{
    switch (type)
    {
    case PT2_D_NONE:
        p = Parameter();
        return true;
    case PT2_D_INT:
        p = Parameter((long long)strtoll(value, 0, 10));
        return true;
    case PT2_D_FLOAT:
        p = Parameter((float)strtod(value, 0));
        return true;
    case PT2_D_BOOL:
        p = Parameter(value[0] == '1');
        return true;
    case PT2_D_STRING:
        p = Parameter(std::string(value));
        return true;
    case PT2_D_INTS:
    {
        if (value[0] == '\0')
        {
            p = Parameter();
            return true;
        }

        std::vector<int> ai;
        const char* pch = value;
        while (*pch != '\0')
        {
            ai.push_back((int)strtoll(pch, 0, 10));
            pch = strchr(pch, ',');
            if (!pch)
                break;
            pch++;
        }
        p = Parameter(ai);
        return true;
    }
    case PT2_D_FLOATS:
    {
        if (value[0] == '\0')
        {
            p = Parameter();
            return true;
        }

        std::vector<float> af;
        const char* pch = value;
        while (*pch != '\0')
        {
            af.push_back((float)strtod(pch, 0));
            pch = strchr(pch, ',');
            if (!pch)
                break;
            pch++;
        }
        p = Parameter(af);
        return true;
    }
    case PT2_D_STRINGS:
    {
        if (value[0] == '\0')
        {
            p = Parameter();
            return true;
        }

        std::vector<std::string> as;
        const char* pch = value;
        while (*pch != '\0')
        {
            const char* comma = strchr(pch, ',');
            const size_t len = comma ? (size_t)(comma - pch) : strlen(pch);
            as.push_back(std::string(pch, len));
            if (!comma)
                break;
            pch = comma + 1;
        }
        p = Parameter(as);
        return true;
    }
    case PT2_D_DEVICE:
        if (value[0] == '\0')
        {
            p = Parameter();
        }
        else
        {
            p = Parameter(std::string(value));
        }
        return true;
    default:
        return false;
    }
}

static bool argument_to_constant(const Pt2Argument& a, Parameter& value)
{
    switch (a.type)
    {
    case Pt2Argument::NONE:
        value = Parameter();
        return true;
    case Pt2Argument::INT:
        value = Parameter((long long)a.int_value);
        return true;
    case Pt2Argument::SCALAR_TYPE:
        // PT2 dtype enums differ from TorchScript scalar types.
    {
        long long jit_type = 0;
        if (!pt2_scalar_type_to_jit_type(a.int_value, jit_type))
            return false;
        value = Parameter(jit_type);
    }
    return true;
    case Pt2Argument::MEMORY_FORMAT:
        value = Parameter((long long)a.int_value);
        return true;
    case Pt2Argument::DEVICE:
        // Preserve the device index; cuda:1 must not collapse to cuda.
        if (a.device_type.empty())
        {
            value = Parameter();
        }
        else
        {
            std::string device = a.device_type;
            if (a.device_index >= 0)
                device += ":" + std::to_string(a.device_index);
            value = Parameter(device);
        }
        return true;
    case Pt2Argument::INTS:
    {
        std::vector<int> ai;
        for (size_t k = 0; k < a.int_values.size(); k++)
            ai.push_back((int)a.int_values[k]);
        value = Parameter(ai);
        return true;
    }
    case Pt2Argument::FLOAT:
        value = Parameter((float)a.float_value);
        return true;
    case Pt2Argument::FLOATS:
        value = Parameter(a.float_values);
        return true;
    case Pt2Argument::BOOL:
        value = Parameter(a.bool_value);
        return true;
    case Pt2Argument::BOOLS:
    {
        std::vector<int> ai;
        for (size_t k = 0; k < a.bool_values.size(); k++)
            ai.push_back(a.bool_values[k] ? 1 : 0);
        value = Parameter(ai);
        return true;
    }
    case Pt2Argument::STRING:
        value = Parameter(a.string_value);
        return true;
    case Pt2Argument::STRINGS:
        value = Parameter(a.string_values);
        return true;
    default:
        fprintf(stderr, "load_pt2: unsupported constant argument %s (%d)\n", a.name.c_str(), a.type);
        return false;
    }
}

static int load_weight_attribute(StoreZipReader& zip, const Pt2Program& program, const Pt2WeightEntry& entry,
                                 bool is_constant, Attribute& attr)
{
    if (entry.use_pickle)
    {
        fprintf(stderr, "load_pt2: use_pickle weights not supported yet (%s)\n", entry.state_dict_name.c_str());
        return -1;
    }

    const std::string entry_path = is_constant ? program.constant_entry_path(entry.path_name)
                                   : program.weight_entry_path(entry.path_name);

    const uint64_t raw_size = zip.get_file_size(entry_path);
    if (raw_size > (uint64_t)std::numeric_limits<size_t>::max())
        return -1;

    attr.type = pt2_dtype_to_pnnx_type(entry.dtype);
    if (attr.type == 0)
        return -1;

    const size_t elemsize = attr.elemsize();
    if (elemsize == 0)
    {
        fprintf(stderr, "load_pt2: unsupported attribute type %d\n", attr.type);
        return -1;
    }

    if (entry.storage_offset < 0)
    {
        fprintf(stderr, "load_pt2: invalid weight storage offset %s\n", entry_path.c_str());
        return -1;
    }
    for (size_t i = 0; i < entry.sizes.size(); i++)
    {
        if (entry.sizes[i] < 0)
        {
            fprintf(stderr, "load_pt2: invalid weight shape %s\n", entry_path.c_str());
            return -1;
        }
        if ((unsigned long long)entry.sizes[i] > (unsigned long long)std::numeric_limits<int>::max())
        {
            fprintf(stderr, "load_pt2: weight dimension out of range %s\n", entry_path.c_str());
            return -1;
        }
        attr.shape.push_back((int)entry.sizes[i]);
    }

    size_t elem_count = 1;
    for (size_t i = 0; i < attr.shape.size(); i++)
    {
        if (!checked_size_mul(elem_count, (size_t)attr.shape[i], elem_count))
        {
            fprintf(stderr, "load_pt2: weight element count overflow %s\n", entry_path.c_str());
            return -1;
        }
    }

    if (entry.strides.size() != entry.sizes.size() && !entry.strides.empty())
    {
        fprintf(stderr, "load_pt2: invalid weight strides %s\n", entry_path.c_str());
        return -1;
    }

    if ((unsigned long long)entry.storage_offset > (unsigned long long)std::numeric_limits<size_t>::max())
    {
        fprintf(stderr, "load_pt2: invalid weight storage offset %s\n", entry_path.c_str());
        return -1;
    }

    std::vector<size_t> strides(entry.sizes.size());
    if (entry.strides.empty())
    {
        size_t stride = 1;
        for (size_t i = entry.sizes.size(); i-- > 0;)
        {
            strides[i] = stride;
            if (!checked_size_mul(stride, (size_t)attr.shape[i], stride))
            {
                fprintf(stderr, "load_pt2: weight stride overflow %s\n", entry_path.c_str());
                return -1;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < entry.strides.size(); i++)
        {
            if (entry.strides[i] < 0 || (unsigned long long)entry.strides[i] > (unsigned long long)std::numeric_limits<size_t>::max())
            {
                fprintf(stderr, "load_pt2: invalid weight shape/strides %s\n", entry_path.c_str());
                return -1;
            }
            strides[i] = (size_t)entry.strides[i];
        }
    }

    size_t storage_count = elem_count;
    {
        size_t max_offset = (size_t)entry.storage_offset;
        for (size_t i = 0; i < entry.sizes.size(); i++)
        {
            if (entry.sizes[i] > 0)
            {
                size_t extent = 0;
                if (!checked_size_mul((size_t)(entry.sizes[i] - 1), strides[i], extent)
                        || !checked_size_add(max_offset, extent, max_offset))
                {
                    fprintf(stderr, "load_pt2: weight storage offset overflow %s\n", entry_path.c_str());
                    return -1;
                }
            }
        }
        if (elem_count != 0 && !checked_size_add(max_offset, 1, storage_count))
        {
            fprintf(stderr, "load_pt2: weight storage size overflow %s\n", entry_path.c_str());
            return -1;
        }
    }

    size_t storage_bytes = 0;
    if (!checked_size_mul(storage_count, elemsize, storage_bytes))
    {
        fprintf(stderr, "load_pt2: weight storage byte size overflow %s\n", entry_path.c_str());
        return -1;
    }
    if (raw_size < storage_bytes)
    {
        fprintf(stderr, "load_pt2: weight storage too small %s: need %zu got %llu\n", entry_path.c_str(),
                storage_bytes, (unsigned long long)raw_size);
        return -1;
    }

    std::vector<char> raw((size_t)raw_size);
    if (zip.read_file(entry_path, raw.data()) != 0)
    {
        fprintf(stderr, "load_pt2: read weight failed %s\n", entry_path.c_str());
        return -1;
    }

    bool contiguous = true;
    size_t expected_stride = 1;
    for (size_t i = entry.sizes.size(); i-- > 0;)
    {
        if (strides[i] != expected_stride)
            contiguous = false;
        if (!checked_size_mul(expected_stride, (size_t)attr.shape[i], expected_stride))
        {
            fprintf(stderr, "load_pt2: weight stride overflow %s\n", entry_path.c_str());
            return -1;
        }
    }

    if (storage_count == elem_count && entry.storage_offset == 0 && contiguous
            && raw.size() == storage_bytes)
    {
        attr.data = raw;
        return 0;
    }

    size_t data_bytes = 0;
    if (!checked_size_mul(elem_count, elemsize, data_bytes))
    {
        fprintf(stderr, "load_pt2: weight byte size overflow %s\n", entry_path.c_str());
        return -1;
    }
    attr.data.resize(data_bytes);
    for (size_t linear = 0; linear < elem_count; linear++)
    {
        size_t remaining = linear;
        size_t storage_index = (size_t)entry.storage_offset;
        for (size_t d = entry.sizes.size(); d-- > 0;)
        {
            const size_t coordinate = remaining % (size_t)entry.sizes[d];
            remaining /= (size_t)entry.sizes[d];
            storage_index += coordinate * strides[d];
        }
        memcpy(attr.data.data() + linear * elemsize, raw.data() + storage_index * elemsize, elemsize);
    }
    return 0;
}

// Constants must precede consumers for the reverse fuse_expression scan.
static void hoist_constants(Graph& pg)
{
    for (size_t i = 0; i < pg.ops.size(); i++)
    {
        Operator* op = pg.ops[i];
        if (op->type != "prim::Constant")
            continue;

        size_t consumer_pos = pg.ops.size();
        for (size_t j = 0; j < op->outputs.size(); j++)
        {
            const std::vector<Operator*>& consumers = op->outputs[j]->consumers;
            for (size_t k = 0; k < consumers.size(); k++)
            {
                size_t pos = std::find(pg.ops.begin(), pg.ops.end(), consumers[k]) - pg.ops.begin();
                if (pos < consumer_pos)
                    consumer_pos = pos;
            }
        }

        if (consumer_pos >= pg.ops.size() || consumer_pos > i)
            continue;

        pg.ops.erase(pg.ops.begin() + i);
        pg.ops.insert(pg.ops.begin() + consumer_pos, op);
        i--; // Compensate for the erased element.
    }
}

// Recreate the ListUnpack shape expected by existing split passes.
static bool pt2_target_unpackable(const std::string& op_type)
{
    return op_type == "aten::unbind" || op_type == "aten::split"
           || op_type == "aten::split_with_sizes" || op_type == "aten::chunk"
           || op_type == "aten::tensor_split";
}

static bool parse_nn_module_stack(const std::string& nms, std::string& short_class, std::string& module_name)
{
    if (nms.empty())
        return false;

    const size_t semi = nms.rfind(';');
    const std::string inner = (semi == std::string::npos) ? nms : nms.substr(semi + 1);

    const size_t c1 = inner.find(',');
    if (c1 == std::string::npos)
        return false;
    const size_t c2 = inner.find(',', c1 + 1);
    if (c2 == std::string::npos)
        return false;

    const std::string cls = inner.substr(c2 + 1);
    const std::string prefix = "torch.nn.modules.";
    if (cls.compare(0, prefix.size(), prefix) != 0)
        return false;

    const size_t dot = cls.rfind('.');
    short_class = (dot == std::string::npos) ? cls : cls.substr(dot + 1);
    module_name = inner.substr(c1 + 1, c2 - c1 - 1);
    return true;
}

static const char* pt2_upsample_mode(const std::string& aten)
{
    if (aten == "aten::upsample_nearest1d" || aten == "aten::upsample_nearest2d" || aten == "aten::upsample_nearest3d")
        return "nearest";
    if (aten == "aten::_upsample_nearest_exact1d" || aten == "aten::_upsample_nearest_exact2d" || aten == "aten::_upsample_nearest_exact3d")
        return "nearest-exact";
    if (aten == "aten::upsample_linear1d")
        return "linear";
    if (aten == "aten::upsample_bilinear2d")
        return "bilinear";
    if (aten == "aten::upsample_bicubic2d")
        return "bicubic";
    if (aten == "aten::upsample_trilinear3d")
        return "trilinear";
    return 0;
}

struct Pt2ModuleFormRule
{
    const char* cls;
    const char* aten;
};

static const Pt2ModuleFormRule pt2_module_form_rules[] = {
    {"ReLU6", "aten::hardtanh"},
    {"Softmax2d", "aten::softmax"},
    {"ChannelShuffle", "aten::channel_shuffle"},
    {"PixelShuffle", "aten::pixel_shuffle"},
    {"MaxPool1d", "aten::max_pool1d"},
    {"MaxPool1d", "aten::max_pool1d_with_indices"},
    {"MaxPool2d", "aten::max_pool2d"},
    {"MaxPool2d", "aten::max_pool2d_with_indices"},
    {"MaxPool3d", "aten::max_pool3d"},
    {"MaxPool3d", "aten::max_pool3d_with_indices"},
    {"AdaptiveAvgPool1d", "aten::adaptive_avg_pool1d"},
    {"AdaptiveAvgPool2d", "aten::adaptive_avg_pool2d"},
    {"AdaptiveAvgPool3d", "aten::adaptive_avg_pool3d"},
    {"ConstantPad1d", "aten::pad"},
    {"ConstantPad2d", "aten::pad"},
    {"ConstantPad3d", "aten::pad"},
    {"ReflectionPad1d", "aten::pad"},
    {"ReflectionPad2d", "aten::pad"},
    {"ReplicationPad1d", "aten::pad"},
    {"ReplicationPad2d", "aten::pad"},
    {"ReplicationPad3d", "aten::pad"},
    {"ZeroPad2d", "aten::pad"},
    {"UpsamplingNearest2d", "aten::upsample_nearest2d"},
    {"UpsamplingBilinear2d", "aten::upsample_bilinear2d"},
    {"LayerNorm", "aten::layer_norm"},
    {"RMSNorm", "aten::rms_norm"},
};

static bool pt2_module_form_allowed(const std::string& cls, const std::string& aten)
{
    if (cls == "Upsample")
        return pt2_upsample_mode(aten) != 0;

    for (size_t i = 0; i < sizeof(pt2_module_form_rules) / sizeof(pt2_module_form_rules[0]); i++)
    {
        if (cls == pt2_module_form_rules[i].cls && aten == pt2_module_form_rules[i].aten)
            return true;
    }
    return false;
}

int load_pt2(const std::string& ptpath, Graph& pg,
             const std::vector<std::vector<int64_t> >& input_shapes,
             const std::vector<std::string>& input_types)
{
    Pt2Program program;
    program.zippath = ptpath;

    int ret = load_pt2_schema(ptpath, program);
    if (ret != 0)
        return ret;

    StoreZipReader zip;
    if (zip.open(ptpath) != 0)
    {
        fprintf(stderr, "load_pt2: open zip failed %s\n", ptpath.c_str());
        return -1;
    }

    fprintf(stderr, "load_pt2: schema_version=%lld.%lld torch_version=%s nodes=%zu params=%zu\n",
            program.schema_version_major, program.schema_version_minor, program.torch_version.c_str(),
            program.nodes.size(), program.weights.size());

    int pnnx_unknown_index = 0;

    {
        int input_index = 0;
        for (size_t i = 0; i < program.input_specs.size(); i++)
        {
            const Pt2InputSpec& spec = program.input_specs[i];
            if (spec.kind != Pt2InputSpec::USER_INPUT)
                continue;

            char name[32];
            snprintf(name, sizeof(name), "pnnx_input_%d", input_index);

            Operator* op = pg.new_operator("pnnx.Input", name);
            Operand* r = pg.new_operand(spec.graph_name);
            r->producer = op;
            op->outputs.push_back(r);

            std::map<std::string, Pt2TensorMeta>::const_iterator it = program.tensor_values.find(spec.graph_name);
            const bool has_tensor_meta = it != program.tensor_values.end();
            if (has_tensor_meta)
            {
                r->type = pt2_dtype_enum_to_pnnx_type(it->second.dtype);
                for (size_t j = 0; j < it->second.sizes.size(); j++)
                    r->shape.push_back((int)it->second.sizes[j]);
            }

            if (has_tensor_meta && r->type == 0)
            {
                fprintf(stderr, "load_pt2: unsupported input dtype enum %lld for %s\n", it->second.dtype,
                        spec.graph_name.c_str());
                return -1;
            }

            if (input_index < (int)input_shapes.size())
            {
                if (!has_tensor_meta && r->type == 0)
                    r->type = pnnx_type_from_string(input_types[input_index]);

                apply_input_shape(r, input_shapes[input_index]);
            }

            input_index++;
        }
    }

    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const Pt2InputSpec& spec = program.input_specs[i];
        if (spec.kind == Pt2InputSpec::USER_INPUT)
            continue;

        const bool is_constant = spec.kind == Pt2InputSpec::TENSOR_CONSTANT
                                 || (spec.kind == Pt2InputSpec::BUFFER && !spec.persistent);
        const Pt2WeightEntry* entry = is_constant ? program.find_constant(spec.state_dict_name)
                                      : program.find_weight(spec.state_dict_name);

        if (!entry)
        {
            fprintf(stderr, "load_pt2: weight entry not found for %s (%s)\n", spec.graph_name.c_str(),
                    spec.state_dict_name.c_str());
            return -1;
        }

        Attribute attr;
        if (load_weight_attribute(zip, program, *entry, is_constant, attr) != 0)
            return -1;

        Operator* op = pg.new_operator("pnnx.Attribute", spec.state_dict_name);
        op->attrs["data"] = attr;

        Operand* r = pg.new_operand(spec.graph_name);
        r->producer = op;
        op->outputs.push_back(r);
        r->type = attr.type;
        r->shape = attr.shape;
    }

    for (size_t i = 0; i < program.nodes.size(); i++)
    {
        const Pt2Node& node = program.nodes[i];
        const std::string aten_type = map_pt2_target(node.target);

        std::string module_class;
        std::string module_name;
        const bool is_module_form = parse_nn_module_stack(node.nn_module_stack, module_class, module_name)
                                    && pt2_module_form_allowed(module_class, aten_type);

        Operator* op = pg.new_operator(aten_type, "pnnx_" + std::to_string(pnnx_unknown_index++));

        if (is_module_form)
        {
            op->params["__pt2_module_class"] = module_class;
            op->params["__pt2_module_name"] = module_name;
        }

        // Mark only PT2-originated materialized None dimensions.
        bool adaptive_pool_has_none = node.adaptive_pool_has_none;
        std::vector<int> adaptive_pool_none_axes = node.adaptive_pool_none_axes;
        if (adaptive_pool_has_none
                && (aten_type == "aten::adaptive_avg_pool1d" || aten_type == "aten::adaptive_avg_pool2d"
                    || aten_type == "aten::adaptive_avg_pool3d" || aten_type == "aten::adaptive_max_pool1d"
                    || aten_type == "aten::adaptive_max_pool2d" || aten_type == "aten::adaptive_max_pool3d"))
        {
            std::string none_axes;
            for (size_t i = 0; i < adaptive_pool_none_axes.size(); i++)
                none_axes += adaptive_pool_none_axes[i] ? '1' : '0';
            Parameter marker;
            marker.type = 4;
            marker.s = none_axes;
            op->params["__pt2_none_axes"] = marker;
        }

        {
            // Fill omitted defaults to match TorchScript parameter arity.
            const std::string full_target = pt2_full_target_name(node.target);
            const Pt2DefaultsEntry* defaults = find_pt2_aten_defaults(full_target.c_str());

            std::vector<const Pt2NodeInput*> ordered_inputs;
            if (defaults)
            {
                std::map<std::string, size_t> table_index;
                for (size_t j = 0; j < defaults->arg_count; j++)
                    table_index[defaults->args[j].name] = j;

                bool table_matches = true;
                for (size_t j = 0; j < node.inputs.size(); j++)
                {
                    if (table_index.find(node.inputs[j].name) == table_index.end())
                    {
                        table_matches = false;
                        break;
                    }
                }

                if (table_matches)
                {
                    ordered_inputs.resize(defaults->arg_count, 0);
                    for (size_t j = 0; j < node.inputs.size(); j++)
                    {
                        ordered_inputs[table_index[node.inputs[j].name]] = &node.inputs[j];
                    }
                }
            }

            if (ordered_inputs.empty())
            {
                if (defaults)
                {
                    fprintf(stderr, "load_pt2: %s node %s: arg names mismatch defaults table, fallback to raw order\n",
                            full_target.c_str(), node.name.c_str());
                }
                else if (strncmp(aten_type.c_str(), "aten::", 6) == 0 && !node.inputs.empty())
                {
                    // Warn when a missing default can prevent a rewrite.
                    fprintf(stderr, "load_pt2: %s node %s: not in defaults table, %d arg(s) emitted as-is\n",
                            full_target.c_str(), node.name.c_str(), (int)node.inputs.size());
                }

                for (size_t j = 0; j < node.inputs.size(); j++)
                {
                    ordered_inputs.push_back(&node.inputs[j]);
                }
            }

            std::vector<std::string> pt2_input_names;
            for (size_t j = 0; j < ordered_inputs.size(); j++)
            {
                const Pt2NodeInput* input = ordered_inputs[j];

                if (input == 0)
                {
                    const Pt2ArgDefault& d = defaults->args[j];

                    Parameter value;
                    if (d.type == PT2_D_NO_DEFAULT || d.type == PT2_D_UNSUPPORTED
                            || !default_value_to_parameter(d.type, d.value, value))
                    {
                        fprintf(stderr, "load_pt2: %s node %s: missing arg %s has no usable default\n",
                                full_target.c_str(), node.name.c_str(), d.name);
                        return -1;
                    }

                    fprintf(stderr, "load_pt2: %s node %s: fill default %s=%s (from defaults table)\n",
                            full_target.c_str(), node.name.c_str(), d.name, d.value);

                    Operator* op_const = pg.new_operator("prim::Constant",
                                                         "pnnx_" + std::to_string(pnnx_unknown_index++));
                    op_const->params["value"] = value;

                    Operand* r = pg.new_operand(node.name + "." + d.name);
                    r->producer = op_const;
                    op_const->outputs.push_back(r);

                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                    pt2_input_names.push_back(d.name);
                    continue;
                }

                const Pt2Argument& arg = input->arg;

                if (arg.type == Pt2Argument::TENSOR)
                {
                    if (arg.tensor_refs.size() != 1 || arg.tensor_refs[0].is_none)
                    {
                        fprintf(stderr, "load_pt2: bad tensor argument %s.%s\n", node.name.c_str(), input->name.c_str());
                        return -1;
                    }

                    Operand* r = pg.get_operand(arg.tensor_refs[0].name);
                    if (!r)
                    {
                        fprintf(stderr, "load_pt2: operand not found %s (node %s)\n", arg.tensor_refs[0].name.c_str(),
                                node.name.c_str());
                        return -1;
                    }

                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                    pt2_input_names.push_back(input->name);
                    continue;
                }

                if (arg.type == Pt2Argument::TENSORS)
                {
                    Operator* op_list = pg.new_operator("prim::ListConstruct",
                                                        "pnnx_" + std::to_string(pnnx_unknown_index++));

                    for (size_t k = 0; k < arg.tensor_refs.size(); k++)
                    {
                        if (!append_tensor_list_item(pg, op_list, arg.tensor_refs[k], node.name, input->name, k,
                                                     pnnx_unknown_index))
                            return -1;
                    }

                    Operand* r = pg.new_operand(node.name + "." + input->name);
                    r->producer = op_list;
                    op_list->outputs.push_back(r);

                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                    pt2_input_names.push_back(input->name);
                    continue;
                }

                Parameter value;
                if (!argument_to_constant(arg, value))
                    return -1;

                Operator* op_const = pg.new_operator("prim::Constant",
                                                     "pnnx_" + std::to_string(pnnx_unknown_index++));
                op_const->params["value"] = value;

                Operand* r = pg.new_operand(node.name + "." + input->name);
                r->producer = op_const;
                op_const->outputs.push_back(r);

                r->consumers.push_back(op);
                op->inputs.push_back(r);
                pt2_input_names.push_back(input->name);
            }

            if (is_module_form)
                op->params["__pt2_module_input_names"] = pt2_input_names;
        }

        std::vector<std::string> out_tensor_names;
        for (size_t j = 0; j < node.outputs.size(); j++)
        {
            for (size_t k = 0; k < node.outputs[j].tensor_refs.size(); k++)
            {
                if (node.outputs[j].tensor_refs[k].is_none)
                {
                    fprintf(stderr, "load_pt2: output tensor cannot be None (node %s)\n", node.name.c_str());
                    return -1;
                }
                out_tensor_names.push_back(node.outputs[j].tensor_refs[k].name);
            }
        }

        if (pt2_target_unpackable(op->type) && out_tensor_names.size() > 1)
        {
            Operand* list_out = pg.new_operand(node.name + ".out");
            list_out->producer = op;
            op->outputs.push_back(list_out);

            Operator* op_unpack = pg.new_operator("prim::ListUnpack",
                                                  "pnnx_" + std::to_string(pnnx_unknown_index++));

            list_out->consumers.push_back(op_unpack);
            op_unpack->inputs.push_back(list_out);

            for (size_t j = 0; j < out_tensor_names.size(); j++)
            {
                Operand* r = pg.new_operand(out_tensor_names[j]);
                r->producer = op_unpack;
                op_unpack->outputs.push_back(r);
            }
        }
        else
        {
            for (size_t j = 0; j < out_tensor_names.size(); j++)
            {
                Operand* r = pg.new_operand(out_tensor_names[j]);
                r->producer = op;
                op->outputs.push_back(r);
            }
        }
    }

    for (size_t i = 0; i < pg.operands.size(); i++)
    {
        Operand* r = pg.operands[i];

        std::map<std::string, Pt2TensorMeta>::const_iterator it = program.tensor_values.find(r->name);
        if (it == program.tensor_values.end())
            continue;

        if (r->type == 0)
            r->type = pt2_dtype_enum_to_pnnx_type(it->second.dtype);

        if (r->shape.empty())
        {
            for (size_t j = 0; j < it->second.sizes.size(); j++)
                r->shape.push_back((int)it->second.sizes[j]);
        }
    }

    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        char name[32];
        snprintf(name, sizeof(name), "pnnx_output_%d", (int)i);

        Operator* op = pg.new_operator("pnnx.Output", name);

        Operand* r = pg.get_operand(program.output_specs[i].graph_name);
        if (!r)
        {
            fprintf(stderr, "load_pt2: output operand not found %s\n", program.output_specs[i].graph_name.c_str());
            return -1;
        }

        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }

    // Keep constants before consumers for the reverse fusion pass.
    hoist_constants(pg);

    return 0;
}

} // namespace pnnx
