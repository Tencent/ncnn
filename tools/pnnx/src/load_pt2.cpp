// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"
#include "pt2_schema.h"
#include "aten_defaults_table.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

namespace pnnx {

// PT2 dtype enum -> pnnx type. Unknown values return 0.
static int pt2_dtype_enum_to_pnnx_type(long long dtype)
{
    switch (dtype)
    {
    case 7:
        return 1; // f32
    case 5:
        return 5; // i64
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

static long long pt2_scalar_type_to_jit_type(long long scalar_type)
{
    switch (scalar_type)
    {
    case 7:
        return 6;
    case 5:
        return 4;
    default:
        return scalar_type;
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
    return 0;
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

// Decode a default-table value into a prim::Constant.
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

// Convert a PT2 argument to a prim::Constant value.
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
        value = Parameter(pt2_scalar_type_to_jit_type(a.int_value));
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
    case Pt2Argument::STRING:
        value = Parameter(a.string_value);
        return true;
    default:
        fprintf(stderr, "load_pt2: unsupported constant argument %s (%d)\n", a.name.c_str(), a.type);
        return false;
    }
}

// Read a raw PT2 tensor entry into a pnnx Attribute.
static int load_weight_attribute(const Pt2Program& program, const Pt2WeightEntry& entry, bool is_constant, Attribute& attr)
{
    if (entry.use_pickle)
    {
        fprintf(stderr, "load_pt2: use_pickle weights not supported yet (%s)\n", entry.state_dict_name.c_str());
        return -1;
    }

    const std::string entry_path = is_constant ? program.constant_entry_path(entry.path_name)
                                   : program.weight_entry_path(entry.path_name);

    StoreZipReader zip;
    if (zip.open(program.zippath) != 0)
    {
        fprintf(stderr, "load_pt2: reopen zip failed %s\n", program.zippath.c_str());
        return -1;
    }

    const uint64_t raw_size = zip.get_file_size(entry_path);
    std::vector<char> raw((size_t)raw_size);
    int ret = zip.read_file(entry_path, raw.data());
    zip.close();
    if (ret != 0)
    {
        fprintf(stderr, "load_pt2: read weight failed %s\n", entry_path.c_str());
        return -1;
    }

    attr.type = pt2_dtype_to_pnnx_type(entry.dtype);
    if (attr.type == 0)
        return -1;

    const int elemsize = (attr.type == 1) ? 4 : ((attr.type == 5) ? 8 : 0);
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
        attr.shape.push_back((int)entry.sizes[i]);
    }

    size_t elem_count = 1;
    for (size_t i = 0; i < attr.shape.size(); i++)
        elem_count *= (size_t)attr.shape[i];

    if (entry.strides.size() != entry.sizes.size() && !entry.strides.empty())
    {
        fprintf(stderr, "load_pt2: invalid weight strides %s\n", entry_path.c_str());
        return -1;
    }

    size_t storage_count = elem_count;
    if (entry.strides.size() == entry.sizes.size())
    {
        size_t max_offset = (size_t)entry.storage_offset;
        for (size_t i = 0; i < entry.sizes.size(); i++)
        {
            if (entry.strides[i] < 0)
            {
                fprintf(stderr, "load_pt2: invalid weight shape/strides %s\n", entry_path.c_str());
                return -1;
            }
            if (entry.sizes[i] > 0)
                max_offset += (size_t)(entry.sizes[i] - 1) * (size_t)entry.strides[i];
        }
        storage_count = elem_count == 0 ? 0 : max_offset + 1;
    }

    if (raw.size() < storage_count * elemsize)
    {
        fprintf(stderr, "load_pt2: weight storage too small %s: need %zu got %llu\n", entry_path.c_str(),
                storage_count * elemsize, (unsigned long long)raw.size());
        return -1;
    }

    bool contiguous = entry.strides.empty();
    if (entry.strides.size() == entry.sizes.size())
    {
        long long expected_stride = 1;
        contiguous = true;
        for (size_t i = entry.sizes.size(); i-- > 0;)
        {
            if (entry.strides[i] != expected_stride)
                contiguous = false;
            expected_stride *= entry.sizes[i];
        }
    }

    if (storage_count == elem_count && entry.storage_offset == 0 && contiguous
            && raw.size() == elem_count * elemsize)
    {
        attr.data = raw;
        return 0;
    }

    attr.data.resize(elem_count * elemsize);
    for (size_t linear = 0; linear < elem_count; linear++)
    {
        size_t remaining = linear;
        size_t storage_index = (size_t)entry.storage_offset;
        for (size_t d = entry.sizes.size(); d-- > 0;)
        {
            const size_t coordinate = remaining % (size_t)entry.sizes[d];
            remaining /= (size_t)entry.sizes[d];
            storage_index += coordinate * (size_t)entry.strides[d];
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

// Decode the innermost nn_module_stack entry from node metadata.
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

// Infer interpolate mode from the exported aten operator.
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

// Infer spatial rank and broadcast scalar module arguments to JIT list shape.
static int pt2_aten_spatial_ndim(const std::string& aten)
{
    const size_t n = aten.size();
    if (n >= 2 && aten[n - 2] == '1' && aten[n - 1] == 'd')
        return 1;
    if (n >= 2 && aten[n - 2] == '2' && aten[n - 1] == 'd')
        return 2;
    if (n >= 2 && aten[n - 2] == '3' && aten[n - 1] == 'd')
        return 3;
    return 0;
}

// Normalize module-form arguments, including scalar broadcasting.
static void fold_module_param(Operator* op, const std::string& key, const Parameter& raw, int nd)
{
    Parameter value = raw;
    if (nd > 0 && value.type == 2)
        value = Parameter(std::vector<int>(nd, (int)value.i));
    if (nd > 0 && value.type == 3)
        value = Parameter(std::vector<float>(nd, value.f));

    op->params[key] = value;
}

// Only exact module/operator pairs use module-form normalization.
static bool pt2_module_form_allowed(const std::string& cls, const std::string& aten)
{
    if (cls == "ReLU6")
        return aten == "aten::hardtanh";
    if (cls == "Softmax2d")
        return aten == "aten::softmax";
    if (cls == "ChannelShuffle")
        return aten == "aten::channel_shuffle";
    if (cls == "PixelShuffle")
        return aten == "aten::pixel_shuffle";
    if (cls == "MaxPool1d")
        return aten == "aten::max_pool1d" || aten == "aten::max_pool1d_with_indices";
    if (cls == "MaxPool2d")
        return aten == "aten::max_pool2d" || aten == "aten::max_pool2d_with_indices";
    if (cls == "MaxPool3d")
        return aten == "aten::max_pool3d" || aten == "aten::max_pool3d_with_indices";
    if (cls == "AdaptiveAvgPool1d")
        return aten == "aten::adaptive_avg_pool1d";
    if (cls == "AdaptiveAvgPool2d")
        return aten == "aten::adaptive_avg_pool2d";
    if (cls == "AdaptiveAvgPool3d")
        return aten == "aten::adaptive_avg_pool3d";
    if (cls == "ConstantPad1d" || cls == "ConstantPad2d" || cls == "ConstantPad3d"
            || cls == "ReflectionPad1d" || cls == "ReflectionPad2d"
            || cls == "ReplicationPad1d" || cls == "ReplicationPad2d" || cls == "ReplicationPad3d"
            || cls == "ZeroPad2d")
        return aten == "aten::pad";
    if (cls == "Upsample")
        return pt2_upsample_mode(aten) != 0;
    if (cls == "UpsamplingNearest2d")
        return aten == "aten::upsample_nearest2d";
    if (cls == "UpsamplingBilinear2d")
        return aten == "aten::upsample_bilinear2d";
    if (cls == "LayerNorm")
        return aten == "aten::layer_norm";
    if (cls == "RMSNorm")
        return aten == "aten::rms_norm";
    return false;
}

// Names mirror the level1 module conversion output exactly.
static std::string pt2_module_param_key(const std::string& cls, const std::string& name)
{
    if (cls == "ReLU6" || cls == "Softmax2d")
        return "";

    if (cls == "ChannelShuffle")
        return name == "groups" ? name : "";

    if (cls == "PixelShuffle")
        return name == "upscale_factor" ? name : "";

    if (cls == "MaxPool1d" || cls == "MaxPool2d" || cls == "MaxPool3d")
    {
        if (name == "kernel_size" || name == "stride" || name == "padding" || name == "dilation"
                || name == "ceil_mode")
            return name;
        return "";
    }

    if (cls == "AdaptiveAvgPool1d" || cls == "AdaptiveAvgPool2d" || cls == "AdaptiveAvgPool3d")
        return name == "output_size" ? name : "";

    if (cls == "ConstantPad1d" || cls == "ConstantPad2d" || cls == "ConstantPad3d")
    {
        if (name == "pad")
            return "padding";
        if (name == "value")
            return "value";
        return ""; // Not folded by level1 module conversion.
    }

    if (cls == "ReflectionPad1d" || cls == "ReflectionPad2d" || cls == "ReplicationPad1d"
            || cls == "ReplicationPad2d" || cls == "ReplicationPad3d" || cls == "ZeroPad2d")
        return name == "pad" ? "padding" : "";

    if (cls == "Upsample")
    {
        if (name == "output_size")
            return "size";
        if (name == "scale_factors")
            return "scale_factor";
        if (name == "align_corners")
            return "align_corners";
        return "";
    }

    if (cls == "UpsamplingNearest2d" || cls == "UpsamplingBilinear2d")
    {
        if (name == "output_size")
            return "size";
        if (name == "scale_factors")
            return "scale_factor";
        return "";
    }

    if (cls == "LayerNorm" || cls == "RMSNorm")
    {
        if (name == "normalized_shape" || name == "eps")
            return name;
        return "";
    }

    return "";
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

    fprintf(stderr, "load_pt2: schema_version=%lld.%lld torch_version=%s nodes=%zu params=%zu\n",
            program.schema_version_major, program.schema_version_minor, program.torch_version.c_str(),
            program.nodes.size(), program.weights.size());

    int pnnx_unknown_index = 0;

    // 1. user_input -> pnnx.Input.
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
            if (it != program.tensor_values.end())
            {
                r->type = pt2_dtype_enum_to_pnnx_type(it->second.dtype);
                for (size_t j = 0; j < it->second.sizes.size(); j++)
                    r->shape.push_back((int)it->second.sizes[j]);
            }

            if (input_index < (int)input_shapes.size())
            {
                if (r->type == 0)
                    r->type = pnnx_type_from_string(input_types[input_index]);

                if (r->shape.empty())
                {
                    for (size_t j = 0; j < input_shapes[input_index].size(); j++)
                        r->shape.push_back((int)input_shapes[input_index][j]);
                }
            }

            input_index++;
        }
    }

    // 2. parameter / buffer / tensor_constant -> pnnx.Attribute.
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
        if (load_weight_attribute(program, *entry, is_constant, attr) != 0)
            return -1;

        Operator* op = pg.new_operator("pnnx.Attribute", spec.state_dict_name);
        op->attrs["data"] = attr;

        Operand* r = pg.new_operand(spec.graph_name);
        r->producer = op;
        op->outputs.push_back(r);
        r->type = attr.type;
        r->shape = attr.shape;
    }

    // 3. Convert graph nodes to Operators.
    for (size_t i = 0; i < program.nodes.size(); i++)
    {
        const Pt2Node& node = program.nodes[i];
        const std::string aten_type = map_pt2_target(node.target);

        std::string module_class;
        std::string module_name;
        const bool is_module_form = parse_nn_module_stack(node.nn_module_stack, module_class, module_name)
                                    && pt2_module_form_allowed(module_class, aten_type);

        Operator* op = pg.new_operator(is_module_form ? ("nn." + module_class) : aten_type,
                                       "pnnx_" + std::to_string(pnnx_unknown_index++));

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

        if (is_module_form)
        {
            for (size_t j = 0; j < node.inputs.size(); j++)
            {
                const Pt2NodeInput& input = node.inputs[j];
                const Pt2Argument& arg = input.arg;

                if (arg.type == Pt2Argument::TENSOR)
                {
                    if (arg.tensor_names.size() != 1)
                    {
                        fprintf(stderr, "load_pt2: bad tensor argument %s.%s\n", node.name.c_str(),
                                input.name.c_str());
                        return -1;
                    }

                    Operand* r = pg.get_operand(arg.tensor_names[0]);
                    if (!r)
                    {
                        fprintf(stderr, "load_pt2: operand not found %s (node %s)\n", arg.tensor_names[0].c_str(),
                                node.name.c_str());
                        return -1;
                    }

                    if ((module_class == "LayerNorm" || module_class == "RMSNorm")
                            && (input.name == "weight" || input.name == "bias") && r->producer
                            && r->producer->type == "pnnx.Attribute")
                    {
                        op->attrs[input.name] = r->producer->attrs["data"];
                        continue;
                    }

                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                    continue;
                }

                if (arg.type == Pt2Argument::TENSORS)
                {
                    Operator* op_list = pg.new_operator("prim::ListConstruct",
                                                        "pnnx_" + std::to_string(pnnx_unknown_index++));

                    for (size_t k = 0; k < arg.tensor_names.size(); k++)
                    {
                        Operand* r = pg.get_operand(arg.tensor_names[k]);
                        if (!r)
                        {
                            fprintf(stderr, "load_pt2: operand not found %s (node %s)\n", arg.tensor_names[k].c_str(),
                                    node.name.c_str());
                            return -1;
                        }
                        r->consumers.push_back(op_list);
                        op_list->inputs.push_back(r);
                    }

                    Operand* r = pg.new_operand(node.name + "." + input.name);
                    r->producer = op_list;
                    op_list->outputs.push_back(r);

                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                    continue;
                }

                const std::string key = pt2_module_param_key(module_class, input.name);
                if (key.empty())
                    continue;

                Parameter value;
                if (!argument_to_constant(arg, value))
                    return -1;

                fold_module_param(op, key, value, pt2_aten_spatial_ndim(aten_type));
            }

            if (module_class == "LayerNorm" && op->attrs.find("weight") != op->attrs.end()
                    && op->attrs.find("bias") == op->attrs.end())
            {
                const Attribute& weight = op->attrs.at("weight");
                Attribute bias;
                bias.type = weight.type;
                bias.shape = weight.shape;
                bias.data.resize(weight.data.size(), 0);
                op->attrs["bias"] = bias;
            }

            const std::string full_target = pt2_full_target_name(node.target);
            const Pt2DefaultsEntry* defaults = find_pt2_aten_defaults(full_target.c_str());
            if (defaults)
            {
                bool table_matches = true;
                for (size_t j = 0; j < node.inputs.size(); j++)
                {
                    bool found = false;
                    for (size_t k = 0; k < defaults->arg_count; k++)
                    {
                        if (defaults->args[k].name == node.inputs[j].name)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        table_matches = false;
                        break;
                    }
                }

                if (table_matches)
                {
                    for (size_t j = 0; j < defaults->arg_count; j++)
                    {
                        const Pt2ArgDefault& d = defaults->args[j];

                        bool provided = false;
                        for (size_t k = 0; k < node.inputs.size(); k++)
                        {
                            if (node.inputs[k].name == d.name)
                            {
                                provided = true;
                                break;
                            }
                        }
                        if (provided)
                            continue;

                        if (d.type == PT2_D_NO_DEFAULT || d.type == PT2_D_UNSUPPORTED)
                            continue;

                        const std::string key = pt2_module_param_key(module_class, d.name);
                        if (key.empty())
                            continue;

                        Parameter value;
                        if (!default_value_to_parameter(d.type, d.value, value))
                            continue;

                        fold_module_param(op, key, value, pt2_aten_spatial_ndim(aten_type));
                    }
                }
            }

            if (module_class == "MaxPool1d" || module_class == "MaxPool2d" || module_class == "MaxPool3d")
            {
                size_t out_count = 0;
                for (size_t j = 0; j < node.outputs.size(); j++)
                    out_count += node.outputs[j].tensor_names.size();
                op->params["return_indices"] = (out_count > 1);
            }

            if (module_class == "Upsample")
                op->params["mode"] = std::string(pt2_upsample_mode(aten_type));

            if (module_class == "LayerNorm" || module_class == "RMSNorm")
            {
                bool has_weight = false;
                for (size_t j = 0; j < node.inputs.size(); j++)
                {
                    if (node.inputs[j].name == "weight" && node.inputs[j].arg.type == Pt2Argument::TENSOR)
                    {
                        has_weight = true;
                        break;
                    }
                }
                op->params["elementwise_affine"] = has_weight;
            }

        }

        if (!is_module_form)
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

                for (size_t j = 0; j < node.inputs.size(); j++)
                {
                    ordered_inputs.push_back(&node.inputs[j]);
                }
            }

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
                        fprintf(stderr, "load_pt2: %s node %s: missing arg %s has no usable default, skipped\n",
                                full_target.c_str(), node.name.c_str(), d.name);
                        continue;
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
                    continue;
                }

                const Pt2Argument& arg = input->arg;

                if (arg.type == Pt2Argument::TENSOR)
                {
                    if (arg.tensor_names.size() != 1)
                    {
                        fprintf(stderr, "load_pt2: bad tensor argument %s.%s\n", node.name.c_str(), input->name.c_str());
                        return -1;
                    }

                    Operand* r = pg.get_operand(arg.tensor_names[0]);
                    if (!r)
                    {
                        fprintf(stderr, "load_pt2: operand not found %s (node %s)\n", arg.tensor_names[0].c_str(),
                                node.name.c_str());
                        return -1;
                    }

                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
                    continue;
                }

                if (arg.type == Pt2Argument::TENSORS)
                {
                    Operator* op_list = pg.new_operator("prim::ListConstruct",
                                                        "pnnx_" + std::to_string(pnnx_unknown_index++));

                    for (size_t k = 0; k < arg.tensor_names.size(); k++)
                    {
                        Operand* r = pg.get_operand(arg.tensor_names[k]);
                        if (!r)
                        {
                            fprintf(stderr, "load_pt2: operand not found %s (node %s)\n", arg.tensor_names[k].c_str(),
                                    node.name.c_str());
                            return -1;
                        }
                        r->consumers.push_back(op_list);
                        op_list->inputs.push_back(r);
                    }

                    Operand* r = pg.new_operand(node.name + "." + input->name);
                    r->producer = op_list;
                    op_list->outputs.push_back(r);

                    r->consumers.push_back(op);
                    op->inputs.push_back(r);
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
            }
        } // if (!is_module_form)

        std::vector<std::string> out_tensor_names;
        for (size_t j = 0; j < node.outputs.size(); j++)
        {
            for (size_t k = 0; k < node.outputs[j].tensor_names.size(); k++)
            {
                out_tensor_names.push_back(node.outputs[j].tensor_names[k]);
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
