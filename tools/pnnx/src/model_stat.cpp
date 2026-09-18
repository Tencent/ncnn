// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "model_stat.h"

#include "ir.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <climits>
#include <map>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace pnnx {

static bool type_in(const std::string& type, std::initializer_list<const char*> types)
{
    for (const char* t : types)
    {
        if (type == t)
            return true;
    }

    return false;
}

static bool string_starts_with(const std::string& s, const char* prefix)
{
    const size_t len = strlen(prefix);
    return s.size() >= len && s.compare(0, len, prefix) == 0;
}

std::string format_model_stat_ops(uint64_t ops)
{
    static const char* unit_names[] = {"", "K", "M", "G", "T", "P"};
    static const uint64_t unit_scales[] = {1ull, 1000ull, 1000000ull, 1000000000ull, 1000000000000ull, 1000000000000000ull};

    int unit_index = 0;
    while (unit_index + 1 < (int)(sizeof(unit_scales) / sizeof(unit_scales[0])) && ops >= unit_scales[unit_index + 1])
    {
        unit_index++;
    }

    if (unit_index == 0)
        return std::to_string(ops);

    const uint64_t scale = unit_scales[unit_index];
    uint64_t integer = ops / scale;
    // Round half up to 3 decimals and carry values such as 1.9995K to 2K.
    uint64_t fraction = ((ops % scale) * 1000 + scale / 2) / scale;

    if (fraction == 1000)
    {
        integer++;
        fraction = 0;
    }

    std::string s = std::to_string(integer);
    if (fraction != 0)
    {
        std::string fraction_string = std::to_string(fraction);
        while (fraction_string.size() < 3)
        {
            fraction_string.insert(fraction_string.begin(), '0');
        }
        while (!fraction_string.empty() && fraction_string[fraction_string.size() - 1] == '0')
        {
            fraction_string.erase(fraction_string.size() - 1);
        }

        s += ".";
        s += fraction_string;
    }

    s += unit_names[unit_index];
    return s;
}

static const char* operand_type_to_string(int type)
{
    if (type == 1) return "f32";
    if (type == 2) return "f64";
    if (type == 3) return "f16";
    if (type == 4) return "i32";
    if (type == 5) return "i64";
    if (type == 6) return "i16";
    if (type == 7) return "i8";
    if (type == 8) return "u8";
    if (type == 9) return "bool";
    if (type == 10) return "c64";
    if (type == 11) return "c128";
    if (type == 12) return "c32";
    if (type == 13) return "bf16";
    return "null";
}

static std::string format_operand_shape(const Operand* operand)
{
    if (!operand)
        return std::string();

    std::string s = "[";
    for (size_t i = 0; i < operand->shape.size(); i++)
    {
        if (operand->shape[i] == -1)
            s += "?";
        else
            s += std::to_string(operand->shape[i]);

        if (i + 1 != operand->shape.size())
            s += ",";
    }
    s += "]";
    s += operand_type_to_string(operand->type);

    return s;
}

std::string format_model_stat_input_shapes(const Graph& graph)
{
    std::string input_shapes;

    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        const Operator* op = graph.ops[i];
        if (op->type != "pnnx.Input")
            continue;

        for (size_t j = 0; j < op->outputs.size(); j++)
        {
            if (!input_shapes.empty())
                input_shapes += ",";

            input_shapes += format_operand_shape(op->outputs[j]);
        }
    }

    return input_shapes;
}

static uint64_t shape_size(const std::vector<int>& shape)
{
    uint64_t size = 1;
    for (size_t i = 0; i < shape.size(); i++)
    {
        if (shape[i] <= 0)
            return 0;

        size *= shape[i];
    }

    return size;
}

static uint64_t operand_size(const Operand* r)
{
    if (!r)
        return 0;

    return shape_size(r->shape);
}

static uint64_t input_size(const Operator* op, size_t i)
{
    if (i >= op->inputs.size())
        return 0;

    return operand_size(op->inputs[i]);
}

static uint64_t output_size(const Operator* op, size_t i)
{
    if (i >= op->outputs.size())
        return 0;

    return operand_size(op->outputs[i]);
}

static uint64_t output_size_all(const Operator* op)
{
    uint64_t size = 0;
    for (size_t i = 0; i < op->outputs.size(); i++)
    {
        size += operand_size(op->outputs[i]);
    }

    return size;
}

static uint64_t input_size_all(const Operator* op)
{
    uint64_t size = 0;
    for (size_t i = 0; i < op->inputs.size(); i++)
    {
        size += operand_size(op->inputs[i]);
    }

    return size;
}

static bool get_bool_param(const Operator* op, const char* key, bool def)
{
    if (!op->has_param(key))
        return def;

    const Parameter& p = op->params.at(key);
    if (p.type == 0)
        return false;
    if (p.type == 1)
        return p.b;
    if (p.type == 2)
        return p.i != 0;
    if (p.type == 4)
        return p.s == "True" || p.s == "true" || p.s == "1";

    return def;
}

static int get_int_param(const Operator* op, const char* key, int def)
{
    if (!op->has_param(key))
        return def;

    const Parameter& p = op->params.at(key);
    if (p.type == 1)
        return p.b ? 1 : 0;
    if (p.type == 2)
        return p.i;

    return def;
}

static float get_float_param(const Operator* op, const char* key, float def)
{
    if (!op->has_param(key))
        return def;

    const Parameter& p = op->params.at(key);
    if (p.type == 1)
        return p.b ? 1.f : 0.f;
    if (p.type == 2)
        return (float)p.i;
    if (p.type == 3)
        return p.f;

    return def;
}

enum NormPKind
{
    NORM_P1,
    NORM_P2,
    NORM_PINF,
    NORM_POTHER
};

static NormPKind norm_p_kind_from_float(float p)
{
    if (p == 1.f)
        return NORM_P1;
    if (p == 2.f)
        return NORM_P2;
    if (std::isinf(p))
        return NORM_PINF;

    return NORM_POTHER;
}

static NormPKind get_norm_p_kind(const Operator* op, const char* key, NormPKind def)
{
    if (!op->has_param(key))
        return def;

    const Parameter& p = op->params.at(key);
    if (p.type == 1)
        return p.b ? NORM_P1 : NORM_POTHER;
    if (p.type == 2)
        return norm_p_kind_from_float((float)p.i);
    if (p.type == 3)
        return norm_p_kind_from_float(p.f);
    if (p.type == 4)
    {
        if (p.s == "fro")
            return NORM_P2;
        if (p.s.find("inf") != std::string::npos || p.s.find("Inf") != std::string::npos)
            return NORM_PINF;

        char* endptr = 0;
        const float v = std::strtof(p.s.c_str(), &endptr);
        if (endptr && endptr != p.s.c_str() && *endptr == '\0')
            return norm_p_kind_from_float(v);
    }

    return def;
}

static uint64_t norm_flops(uint64_t in_elems, uint64_t out_elems, NormPKind p)
{
    if (p == NORM_P1 || p == NORM_PINF)
        return in_elems * 2;
    if (p == NORM_P2)
        return in_elems * 2 + out_elems * 4;

    return in_elems * 6 + out_elems * 4;
}

static std::string get_string_param(const Operator* op, const char* key, const std::string& def)
{
    if (!op->has_param(key))
        return def;

    const Parameter& p = op->params.at(key);
    if (p.type == 4)
        return p.s;

    return def;
}

static std::vector<int> get_int_array_param(const Operator* op, const char* key)
{
    if (!op->has_param(key))
        return std::vector<int>();

    const Parameter& p = op->params.at(key);
    if (p.type == 2)
        return std::vector<int>(1, p.i);
    if (p.type == 5)
        return p.ai;

    return std::vector<int>();
}

static uint64_t product(const std::vector<int>& a)
{
    uint64_t p = 1;
    for (size_t i = 0; i < a.size(); i++)
    {
        if (a[i] <= 0)
            return 0;
        p *= a[i];
    }

    return p;
}

static std::vector<int> get_reduction_dims(const Operator* op, const std::vector<int>& shape, int default_dim)
{
    std::vector<int> dims = get_int_array_param(op, "dim");
    if (dims.empty() && op->has_param("dims"))
        dims = get_int_array_param(op, "dims");

    if (dims.empty() && default_dim != INT_MAX)
        dims.push_back(default_dim);

    const int rank = (int)shape.size();
    for (size_t i = 0; i < dims.size(); i++)
    {
        if (dims[i] < 0)
            dims[i] += rank;
    }

    std::sort(dims.begin(), dims.end());
    dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

    return dims;
}

static uint64_t reduced_elems(const std::vector<int>& shape, const std::vector<int>& dims)
{
    uint64_t elems = 1;
    for (size_t i = 0; i < dims.size(); i++)
    {
        if (dims[i] < 0 || dims[i] >= (int)shape.size() || shape[dims[i]] <= 0)
            return 0;

        elems *= shape[dims[i]];
    }

    return elems;
}

static uint64_t reduction_output_elems(const Operator* op, int default_dim)
{
    if (op->inputs.empty())
        return 0;

    const uint64_t in_elems = input_size(op, 0);
    if (in_elems == 0)
        return 0;

    const std::vector<int> dims = get_reduction_dims(op, op->inputs[0]->shape, default_dim);
    const uint64_t r_elems = reduced_elems(op->inputs[0]->shape, dims);
    if (r_elems == 0)
        return output_size_all(op);

    return in_elems / r_elems;
}

static std::vector<int> kernel_from_weight(const Operand* weight, int dims)
{
    std::vector<int> k;
    if (!weight || (int)weight->shape.size() < dims + 2)
        return k;

    k.insert(k.end(), weight->shape.end() - dims, weight->shape.end());
    return k;
}

static std::vector<int> kernel_size(const Operator* op, int dims)
{
    std::vector<int> k = get_int_array_param(op, "kernel_size");

    if (k.empty() && op->inputs.size() > 1)
        k = kernel_from_weight(op->inputs[1], dims);

    if (k.size() == 1 && dims > 1)
        k.resize(dims, k[0]);

    if ((int)k.size() > dims)
        k.erase(k.begin(), k.end() - dims);

    return k;
}

static uint64_t attr_size(const Operator* op, const char* key)
{
    if (!op->has_attr(key))
        return 0;

    return shape_size(op->attrs.at(key).shape);
}

static uint64_t tensor_or_attr_size(const Operator* op, size_t input_index, const char* attr_key)
{
    uint64_t size = attr_size(op, attr_key);
    if (size != 0)
        return size;

    return input_size(op, input_index);
}

static bool param_is_none(const Operator* op, const char* key)
{
    return op->has_param(key) && op->params.at(key).type == 0;
}

static uint64_t optional_input_size(const Operator* op, const char* key, int input_index)
{
    if (param_is_none(op, key))
        return 0;

    const Operand* r = op->named_input(key);
    if (r)
        return operand_size(r);

    if (input_index >= 0)
        return input_size(op, (size_t)input_index);

    return 0;
}

static uint64_t optional_tensor_size(const Operator* op, const char* key, int input_index)
{
    if (param_is_none(op, key))
        return 0;

    const uint64_t size = attr_size(op, key);
    if (size != 0)
        return size;

    return optional_input_size(op, key, input_index);
}

static bool has_bias(const Operator* op)
{
    if (op->has_param("bias"))
    {
        const Parameter& p = op->params.at("bias");
        if (p.type == 0)
            return false;
        if (p.type == 1)
            return p.b;
        if (p.type == 4 && p.s == "None")
            return false;
        return true;
    }

    if (op->has_attr("bias"))
        return attr_size(op, "bias") != 0;

    return op->has_input("bias");
}

static int op_1d2d3d(const std::string& type)
{
    if (type.find("1d") != std::string::npos || type.find("1D") != std::string::npos)
        return 1;
    if (type.find("2d") != std::string::npos || type.find("2D") != std::string::npos)
        return 2;
    if (type.find("3d") != std::string::npos || type.find("3D") != std::string::npos)
        return 3;

    return 0;
}

static bool convolution_shape(const std::vector<int>& shape, int dims, int& channels)
{
    if ((int)shape.size() == dims + 1)
    {
        channels = shape[0];
        return true;
    }

    if ((int)shape.size() == dims + 2)
    {
        channels = shape[1];
        return true;
    }

    return false;
}

static bool count_convolution(const Operator* op, ModelStat& stat)
{
    const bool conv = type_in(op->type, {"nn.Conv1d", "nn.Conv2d", "nn.Conv3d", "F.conv1d", "F.conv2d", "F.conv3d", "nn.quantized.Conv2d"});
    const bool deconv = type_in(op->type, {"nn.ConvTranspose1d", "nn.ConvTranspose2d", "nn.ConvTranspose3d", "F.conv_transpose1d", "F.conv_transpose2d", "F.conv_transpose3d"});
    if (!conv && !deconv)
        return false;

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const std::vector<int>& in_shape = op->inputs[0]->shape;
    const std::vector<int>& out_shape = op->outputs[0]->shape;

    const int dims = op_1d2d3d(op->type);
    if (dims == 0)
        return true;

    int in_channels_default = 0;
    int out_channels_default = 0;
    if (!convolution_shape(in_shape, dims, in_channels_default) || !convolution_shape(out_shape, dims, out_channels_default))
        return true;

    const uint64_t in_elems = operand_size(op->inputs[0]);
    const uint64_t out_elems = operand_size(op->outputs[0]);
    if (in_elems == 0 || out_elems == 0)
        return true;

    const int in_channels = get_int_param(op, "in_channels", in_channels_default);
    const int out_channels = get_int_param(op, "out_channels", out_channels_default);
    const int groups = std::max(1, get_int_param(op, "groups", 1));
    if (in_channels <= 0 || out_channels <= 0)
        return true;

    const std::vector<int> k = kernel_size(op, dims);
    const uint64_t kernel_elems = product(k);
    if (kernel_elems == 0)
        return true;

    const uint64_t macs = (deconv ? in_elems * out_channels : out_elems * in_channels) / groups * kernel_elems;

    const bool bias = has_bias(op);
    stat.flops += macs * 2;
    if (bias)
        stat.flops += out_elems;

    uint64_t weight_elems = tensor_or_attr_size(op, 1, "weight");
    if (weight_elems == 0)
        weight_elems = (uint64_t)out_channels * (in_channels / groups) * kernel_elems;

    stat.memops += in_elems;
    stat.memops += out_elems;
    stat.memops += weight_elems;
    if (bias && out_channels > 0)
        stat.memops += out_channels;

    return true;
}

static bool count_linear(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"nn.Linear", "F.linear", "nn.quantized.Linear"}))
        return false;

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const uint64_t in_elems = input_size(op, 0);
    const uint64_t out_elems = output_size(op, 0);
    if (in_elems == 0 || out_elems == 0)
        return true;

    int in_features = get_int_param(op, "in_features", 0);
    int out_features = get_int_param(op, "out_features", 0);

    if ((in_features == 0 || out_features == 0) && op->inputs.size() > 1 && op->inputs[1]->shape.size() >= 2)
    {
        out_features = op->inputs[1]->shape[0];
        in_features = op->inputs[1]->shape[1];
    }

    if (in_features == 0 && !op->inputs[0]->shape.empty())
        in_features = op->inputs[0]->shape.back();
    if (out_features == 0 && !op->outputs[0]->shape.empty())
        out_features = op->outputs[0]->shape.back();

    if (in_features <= 0 || out_features <= 0)
        return true;

    const uint64_t batch = in_elems / in_features;
    const bool bias = has_bias(op);

    stat.flops += batch * out_features * in_features * 2;
    if (bias)
        stat.flops += out_elems;

    uint64_t weight_elems = tensor_or_attr_size(op, 1, "weight");
    if (weight_elems == 0)
        weight_elems = (uint64_t)in_features * out_features;

    stat.memops += in_elems;
    stat.memops += out_elems;
    stat.memops += weight_elems;
    if (bias && out_features > 0)
        stat.memops += out_features;

    return true;
}

static uint64_t adaptive_pool_work_elems(const std::vector<int>& in_shape, const std::vector<int>& out_shape, int dims)
{
    if ((int)in_shape.size() < dims || (int)out_shape.size() < dims)
        return 0;

    uint64_t work_elems = 1;
    for (int i = 0; i < (int)out_shape.size() - dims; i++)
    {
        if (out_shape[i] <= 0)
            return 0;

        work_elems *= out_shape[i];
    }

    for (int i = 0; i < dims; i++)
    {
        const int in_dim = in_shape[in_shape.size() - dims + i];
        const int out_dim = out_shape[out_shape.size() - dims + i];
        if (in_dim <= 0 || out_dim <= 0)
            return 0;

        uint64_t dim_work = 0;
        for (int j = 0; j < out_dim; j++)
        {
            const int start = j * in_dim / out_dim;
            const int end = ((j + 1) * in_dim + out_dim - 1) / out_dim;
            dim_work += end - start;
        }

        work_elems *= dim_work;
    }

    return work_elems;
}

static bool count_pooling(const Operator* op, ModelStat& stat)
{
    const bool avgpool = type_in(op->type, {"nn.AvgPool1d", "nn.AvgPool2d", "nn.AvgPool3d", "F.avg_pool1d", "F.avg_pool2d", "F.avg_pool3d"});
    const bool maxpool = type_in(op->type, {"nn.MaxPool1d", "nn.MaxPool2d", "nn.MaxPool3d", "F.max_pool1d", "F.max_pool2d", "F.max_pool3d"});
    const bool adaptive_avgpool = type_in(op->type, {"nn.AdaptiveAvgPool1d", "nn.AdaptiveAvgPool2d", "nn.AdaptiveAvgPool3d", "F.adaptive_avg_pool1d", "F.adaptive_avg_pool2d", "F.adaptive_avg_pool3d"});
    const bool adaptive_maxpool = type_in(op->type, {"nn.AdaptiveMaxPool1d", "nn.AdaptiveMaxPool2d", "nn.AdaptiveMaxPool3d", "F.adaptive_max_pool1d", "F.adaptive_max_pool2d", "F.adaptive_max_pool3d"});

    if (!avgpool && !maxpool && !adaptive_avgpool && !adaptive_maxpool)
        return false;

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const int dims = op_1d2d3d(op->type);
    const uint64_t in_elems = input_size(op, 0);
    const uint64_t out_elems = output_size(op, 0);
    if (dims == 0 || in_elems == 0 || out_elems == 0)
        return true;

    uint64_t kernel_elems = 0;
    uint64_t work_elems = 0;
    if (adaptive_avgpool || adaptive_maxpool)
        work_elems = adaptive_pool_work_elems(op->inputs[0]->shape, op->outputs[0]->shape, dims);
    else
        kernel_elems = product(kernel_size(op, dims));

    if (adaptive_avgpool || adaptive_maxpool)
    {
        if (work_elems == 0)
            work_elems = out_elems;

        if (adaptive_avgpool)
            stat.flops += work_elems;
        else if (work_elems > out_elems)
            stat.flops += work_elems - out_elems;
    }
    else
    {
        if (kernel_elems == 0)
            kernel_elems = 1;

        if (avgpool)
            stat.flops += out_elems * kernel_elems;
        else
            stat.flops += out_elems * (kernel_elems - 1);
    }

    stat.memops += in_elems;
    stat.memops += out_elems;
    if (get_bool_param(op, "return_indices", false) || op->outputs.size() > 1)
        stat.memops += out_elems;

    return true;
}

static bool count_normalization(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"nn.BatchNorm1d", "nn.BatchNorm2d", "nn.BatchNorm3d", "F.batch_norm", "F.batchnorm",
                            "nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d", "F.instance_norm",
                            "nn.LayerNorm", "F.layer_norm", "nn.GroupNorm", "F.group_norm", "nn.RMSNorm", "F.rms_norm"
                           }))
    {
        return false;
    }

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const uint64_t elems = input_size(op, 0);
    if (elems == 0)
        return true;

    const bool batch_norm = type_in(op->type, {"nn.BatchNorm1d", "nn.BatchNorm2d", "nn.BatchNorm3d", "F.batch_norm", "F.batchnorm"});
    const bool instance_norm = type_in(op->type, {"nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d", "F.instance_norm"});
    const bool rms_norm = type_in(op->type, {"nn.RMSNorm", "F.rms_norm"});
    const bool functional = string_starts_with(op->type, "F.");
    const bool instance_norm_running_stats_none = instance_norm && functional && (param_is_none(op, "running_mean") || param_is_none(op, "running_var"));
    const bool instance_norm_compute_stats = instance_norm && (functional ? (instance_norm_running_stats_none || get_bool_param(op, "use_input_stats", true)) : !get_bool_param(op, "track_running_stats", false));

    uint64_t flops_per_elem = 2;
    if (type_in(op->type, {"nn.LayerNorm", "F.layer_norm", "nn.GroupNorm", "F.group_norm"}) || instance_norm_compute_stats)
        flops_per_elem = 7;
    if (rms_norm)
        flops_per_elem = 5;

    uint64_t running_mean_size = 0;
    uint64_t running_var_size = 0;
    uint64_t weight_size = 0;
    uint64_t bias_size = 0;

    if (batch_norm)
    {
        running_mean_size = optional_tensor_size(op, "running_mean", functional ? 1 : -1);
        running_var_size = optional_tensor_size(op, "running_var", functional ? 2 : -1);
        weight_size = optional_tensor_size(op, "weight", functional ? 3 : -1);
        bias_size = optional_tensor_size(op, "bias", functional ? 4 : -1);
    }
    else if (instance_norm)
    {
        const bool running_stats_none = instance_norm_running_stats_none;
        running_mean_size = optional_tensor_size(op, "running_mean", functional && !running_stats_none ? 1 : -1);
        running_var_size = optional_tensor_size(op, "running_var", functional && !running_stats_none ? 2 : -1);
        weight_size = optional_tensor_size(op, "weight", functional && running_stats_none ? 1 : (functional ? 3 : -1));
        bias_size = optional_tensor_size(op, "bias", functional && running_stats_none ? 2 : (functional ? 4 : -1));
    }
    else if (rms_norm)
    {
        weight_size = optional_tensor_size(op, "weight", functional ? 1 : -1);
    }
    else
    {
        weight_size = optional_tensor_size(op, "weight", functional ? 1 : -1);
        bias_size = optional_tensor_size(op, "bias", functional ? 2 : -1);
    }

    uint64_t affine_flops_per_elem = 0;
    if (functional)
    {
        if (weight_size != 0)
            affine_flops_per_elem += 1;
        if (bias_size != 0)
            affine_flops_per_elem += 1;
    }
    else
    {
        const bool affine = get_bool_param(op, "affine", get_bool_param(op, "elementwise_affine", true));
        if (affine)
            affine_flops_per_elem = rms_norm ? 1 : 2;
        else
        {
            weight_size = 0;
            bias_size = 0;
        }
    }

    stat.flops += elems * (flops_per_elem + affine_flops_per_elem);
    stat.memops += elems;
    stat.memops += output_size_all(op);
    stat.memops += running_mean_size;
    stat.memops += running_var_size;
    stat.memops += weight_size;
    stat.memops += bias_size;

    return true;
}

static bool count_normalize(const Operator* op, ModelStat& stat)
{
    if (op->type != "F.normalize")
        return false;

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const uint64_t in_elems = input_size(op, 0);
    const uint64_t out_elems = output_size_all(op);
    if (in_elems == 0 || out_elems == 0)
        return true;

    const uint64_t norm_elems = reduction_output_elems(op, 1);
    const NormPKind p = get_norm_p_kind(op, "p", NORM_P2);

    stat.flops += norm_flops(in_elems, norm_elems, p);
    stat.flops += norm_elems;
    stat.flops += out_elems;

    stat.memops += in_elems * 2;
    stat.memops += norm_elems * 2;
    stat.memops += out_elems * 2;
    return true;
}

static uint64_t activation_flops(const std::string& type)
{
    if (type_in(type, {"nn.ReLU", "F.relu", "nn.ReLU6", "F.relu6", "nn.Hardtanh", "F.hardtanh", "nn.Threshold", "F.threshold"}))
        return 1;
    if (type_in(type, {"nn.LeakyReLU", "F.leaky_relu", "nn.PReLU", "F.prelu", "nn.ELU", "F.elu", "nn.CELU", "F.celu", "nn.SELU", "F.selu"}))
        return 2;
    if (type_in(type, {"nn.Hardsigmoid", "F.hardsigmoid"}))
        return 3;
    if (type_in(type, {"nn.Hardswish", "F.hardswish", "nn.Softsign", "F.softsign"}))
        return 4;
    if (type_in(type, {"nn.Mish", "F.mish", "nn.Softplus", "F.softplus", "nn.Sigmoid", "F.sigmoid", "nn.SiLU", "F.silu", "nn.Tanh", "F.tanh"}))
        return 6;
    if (type_in(type, {"nn.GELU", "F.gelu"}))
        return 10;
    if (type_in(type, {"nn.GLU", "F.glu"}))
        return 3;

    return 0;
}

static bool count_activation(const Operator* op, ModelStat& stat)
{
    if (type_in(op->type, {"nn.Softmax", "nn.Softmax2d", "F.softmax", "F.softmin", "nn.Softmin", "nn.LogSoftmax", "F.log_softmax"}))
    {
        const uint64_t elems = input_size(op, 0);
        stat.flops += elems * 5;
        stat.memops += elems;
        stat.memops += output_size_all(op);
        return true;
    }

    const uint64_t per_elem = activation_flops(op->type);
    if (per_elem == 0)
        return false;

    const uint64_t elems = input_size(op, 0);
    stat.flops += elems * per_elem;
    stat.memops += elems;
    stat.memops += output_size_all(op);

    if (op->type == "nn.PReLU" || op->type == "F.prelu")
        stat.memops += optional_tensor_size(op, "weight", 1);

    return true;
}

static bool count_upsample(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"nn.Upsample", "nn.UpsamplingNearest2d", "nn.UpsamplingBilinear2d",
                            "F.interpolate", "F.upsample", "F.upsample_nearest", "F.upsample_bilinear"
                           }))
    {
        return false;
    }

    const uint64_t in_elems = input_size(op, 0);
    const uint64_t out_elems = output_size(op, 0);
    std::string mode = get_string_param(op, "mode", "nearest");
    if (op->type == "nn.UpsamplingBilinear2d" || op->type == "F.upsample_bilinear")
        mode = "bilinear";
    if (op->type == "nn.UpsamplingNearest2d" || op->type == "F.upsample_nearest")
        mode = "nearest";

    uint64_t per_elem = 0;
    if (mode == "linear")
        per_elem = 5;
    else if (mode == "bilinear")
        per_elem = 11;
    else if (mode == "bicubic")
        per_elem = 35;
    else if (mode == "trilinear")
        per_elem = 31;
    // Nearest-neighbor upsample is modeled as a gather, so it contributes memory ops but no FLOPS.

    stat.flops += out_elems * per_elem;
    stat.memops += in_elems;
    stat.memops += out_elems;
    return true;
}

static bool count_matmul(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"torch.matmul", "torch.mm", "torch.bmm", "torch.mv"}))
        return false;

    if (op->inputs.size() < 2 || op->outputs.empty())
        return true;

    const std::vector<int>& a = op->inputs[0]->shape;
    if (a.empty())
        return true;

    const int k = a.back();
    const uint64_t out_elems = output_size(op, 0);
    if (k <= 0 || out_elems == 0)
        return true;

    stat.flops += out_elems * k * 2;
    stat.memops += input_size(op, 0);
    stat.memops += input_size(op, 1);
    stat.memops += out_elems;
    return true;
}

static bool count_addmm(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"torch.addmm", "torch.baddbmm"}))
        return false;

    if (op->inputs.size() < 3 || op->outputs.empty())
        return true;

    const std::vector<int>& a = op->inputs[1]->shape;
    if (a.empty())
        return true;

    const int k = a.back();
    const uint64_t out_elems = output_size(op, 0);
    if (k <= 0 || out_elems == 0)
        return true;

    const float alpha = get_float_param(op, "alpha", 1.f);
    const float beta = get_float_param(op, "beta", 1.f);

    stat.flops += out_elems * k * 2;
    if (alpha != 1.f)
        stat.flops += out_elems;
    if (beta != 0.f)
    {
        stat.flops += out_elems;
        if (beta != 1.f)
            stat.flops += out_elems;

        stat.memops += input_size(op, 0);
    }
    stat.memops += input_size(op, 1);
    stat.memops += input_size(op, 2);
    stat.memops += out_elems;
    return true;
}

static uint64_t expression_function_cost(const std::string& name)
{
    if (type_in(name, {"add", "sub", "rsub", "mul", "div", "floor_divide", "remainder", "fmod",
                       "maximum", "minimum", "max", "min", "clamp", "clip", "where", "abs", "neg", "square",
                       "reciprocal", "round", "floor", "ceil", "trunc", "sign",
                       "eq", "ne", "gt", "ge", "lt", "le",
                       "logical_and", "logical_or", "logical_xor", "logical_not",
                       "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not"
                      }))
    {
        return 1;
    }

    if (type_in(name, {"pow", "exp", "log", "sqrt", "rsqrt",
                       "log10", "logaddexp", "erf",
                       "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                       "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"
                      }))
    {
        return 4;
    }

    if (type_in(name, {"sigmoid", "silu"}))
        return 6;

    if (name == "int" || name == "float" || name == "size")
        return 0;

    return 0;
}

static bool is_identifier_char(char ch)
{
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_';
}

static uint64_t expression_flops_per_elem(const std::string& expr)
{
    uint64_t flops_per_elem = 0;

    for (size_t i = 0; i < expr.size(); i++)
    {
        if (expr[i] != '(')
            continue;

        size_t name_end = i;
        size_t name_begin = name_end;
        while (name_begin > 0 && is_identifier_char(expr[name_begin - 1]))
        {
            name_begin--;
        }

        if (name_begin == name_end)
            continue;

        flops_per_elem += expression_function_cost(expr.substr(name_begin, name_end - name_begin));
    }

    return flops_per_elem;
}

static uint64_t expression_data_input_size(const Operator* op, const std::string& expr)
{
    std::vector<std::string> scope;
    std::vector<bool> input_used(op->inputs.size(), false);

    for (size_t i = 0; i < expr.size(); i++)
    {
        if (expr[i] == '(')
        {
            size_t name_end = i;
            while (name_end > 0 && expr[name_end - 1] == ' ')
            {
                name_end--;
            }
            size_t name_begin = name_end;
            while (name_begin > 0 && is_identifier_char(expr[name_begin - 1]))
            {
                name_begin--;
            }

            scope.push_back(name_begin == name_end ? std::string() : expr.substr(name_begin, name_end - name_begin));
            continue;
        }

        if (expr[i] == ')' && !scope.empty())
        {
            scope.pop_back();
            continue;
        }

        if (expr[i] != '@')
            continue;

        size_t index_begin = i + 1;
        size_t index_end = index_begin;
        while (index_end < expr.size() && expr[index_end] >= '0' && expr[index_end] <= '9')
        {
            index_end++;
        }

        if (index_begin == index_end)
            continue;

        bool in_size = false;
        for (size_t j = 0; j < scope.size(); j++)
        {
            if (scope[j] == "size")
            {
                in_size = true;
                break;
            }
        }
        if (in_size)
            continue;

        const int input_index = std::atoi(expr.substr(index_begin, index_end - index_begin).c_str());
        if (input_index >= 0 && input_index < (int)input_used.size())
            input_used[input_index] = true;

        i = index_end - 1;
    }

    uint64_t size = 0;
    for (size_t i = 0; i < input_used.size(); i++)
    {
        if (input_used[i])
            size += input_size(op, i);
    }

    return size;
}

static bool count_expression(const Operator* op, ModelStat& stat)
{
    if (op->type != "pnnx.Expression")
        return false;

    const std::string expr = get_string_param(op, "expr", "");
    if (expr.empty())
        return true;

    if (expr[0] == '[' || expr == "%expr_zero_shape")
        return true;

    const uint64_t out_elems = output_size_all(op);
    if (out_elems == 0)
        return true;

    stat.flops += out_elems * expression_flops_per_elem(expr);
    stat.memops += expression_data_input_size(op, expr);
    stat.memops += out_elems;
    return true;
}

static bool count_elementwise(const Operator* op, ModelStat& stat)
{
    uint64_t per_elem = 1;
    if (type_in(op->type, {"torch.pow", "torch.exp", "torch.log", "torch.sqrt", "torch.rsqrt", "torch.sin", "torch.cos", "torch.tan", "torch.asin", "torch.acos", "torch.atan"}))
        per_elem = 4;
    else if (!type_in(op->type, {"torch.add", "torch.sub", "torch.mul", "torch.div", "torch.floor_divide", "torch.remainder",
                                 "torch.maximum", "torch.minimum", "torch.clamp", "torch.clamp_min", "torch.clamp_max", "torch.abs", "torch.neg", "torch.square",
                                 "torch.eq", "torch.ne", "torch.gt", "torch.ge", "torch.lt", "torch.le",
                                 "torch.logical_and", "torch.logical_or", "torch.logical_xor", "torch.logical_not",
                                 "torch.bitwise_and", "torch.bitwise_or", "torch.bitwise_xor", "torch.bitwise_not"
                                }))
    {
        return false;
    }

    const uint64_t out_elems = output_size_all(op);
    stat.flops += out_elems * per_elem;
    stat.memops += input_size_all(op);
    stat.memops += out_elems;
    return true;
}

static bool count_reduction(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"torch.sum", "torch.mean", "torch.prod", "torch.max", "torch.min", "torch.amax", "torch.amin",
                            "torch.norm", "torch.var", "torch.std", "torch.logsumexp", "torch.cumsum", "torch.cumprod"
                           }))
    {
        return false;
    }

    const uint64_t in_elems = input_size(op, 0);
    const uint64_t out_elems = output_size_all(op);
    if (in_elems == 0 || out_elems == 0)
        return true;

    if (op->type == "torch.mean")
        stat.flops += in_elems + out_elems;
    else if (op->type == "torch.norm")
    {
        const NormPKind p = get_norm_p_kind(op, "p", NORM_P2);
        stat.flops += norm_flops(in_elems, out_elems, p);
    }
    else if (op->type == "torch.var")
        stat.flops += in_elems * 4 + out_elems * 2;
    else if (op->type == "torch.std")
        stat.flops += in_elems * 4 + out_elems * 6;
    else if (op->type == "torch.logsumexp")
        stat.flops += in_elems * 5 + out_elems * 4;
    else
        stat.flops += in_elems;

    stat.memops += in_elems;
    stat.memops += out_elems;
    return true;
}

static bool count_pairwise_distance(const Operator* op, ModelStat& stat)
{
    if (op->type != "F.pairwise_distance")
        return false;

    if (op->inputs.size() < 2 || op->outputs.empty())
        return true;

    const uint64_t in_elems = input_size(op, 0);
    const uint64_t out_elems = output_size_all(op);
    if (in_elems == 0 || out_elems == 0)
        return true;

    const NormPKind p = get_norm_p_kind(op, "p", NORM_P2);
    if (p == NORM_P1 || p == NORM_PINF)
        stat.flops += in_elems * 3;
    else if (p == NORM_P2)
        stat.flops += in_elems * 4 + out_elems * 4;
    else
        stat.flops += in_elems * 7 + out_elems * 4;

    stat.memops += input_size_all(op);
    stat.memops += out_elems;
    return true;
}

static bool count_lp_pool(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"F.lp_pool1d", "F.lp_pool2d"}))
        return false;

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const int dims = op_1d2d3d(op->type);
    const uint64_t in_elems = input_size(op, 0);
    const uint64_t out_elems = output_size(op, 0);
    uint64_t kernel_elems = product(kernel_size(op, dims));
    if (kernel_elems == 0)
        kernel_elems = 1;

    stat.flops += in_elems * 4;
    stat.flops += out_elems * (kernel_elems + 9);
    stat.memops += in_elems;
    stat.memops += out_elems * 4;
    return true;
}

static bool count_local_response_norm(const Operator* op, ModelStat& stat)
{
    if (op->type != "F.local_response_norm")
        return false;

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const uint64_t in_elems = input_size(op, 0);
    const uint64_t out_elems = output_size_all(op);
    const int size = std::max(1, get_int_param(op, "size", 1));

    stat.flops += in_elems * (size + 7);
    stat.memops += in_elems * 4;
    stat.memops += out_elems;
    return true;
}

static bool count_fold(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"nn.Fold", "F.fold"}))
        return false;

    const uint64_t in_elems = input_size_all(op);
    const uint64_t out_elems = output_size_all(op);
    if (in_elems > out_elems)
        stat.flops += in_elems - out_elems;

    stat.memops += in_elems;
    stat.memops += out_elems;
    return true;
}

static bool count_scaled_dot_product_attention(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"F.scaled_dot_product_attention"}))
        return false;

    if (op->inputs.size() < 3 || op->outputs.empty())
        return true;

    const std::vector<int>& q = op->inputs[0]->shape;
    const std::vector<int>& k = op->inputs[1]->shape;
    const std::vector<int>& v = op->inputs[2]->shape;
    if (q.size() < 2 || k.size() < 2 || v.size() < 2)
        return true;

    const int qlen = q[q.size() - 2];
    const int klen = k[k.size() - 2];
    const int qdim = q.back();
    const int vdim = v.back();
    if (qlen <= 0 || klen <= 0 || qdim <= 0 || vdim <= 0)
        return true;

    const uint64_t q_elems = input_size(op, 0);
    const uint64_t batch_heads = q_elems / ((uint64_t)qlen * qdim);
    const uint64_t attn_elems = batch_heads * qlen * klen;

    stat.flops += attn_elems * qdim * 2;
    stat.flops += attn_elems * 5;
    stat.flops += attn_elems * vdim * 2;

    const uint64_t mask_elems = operand_size(op->named_input("attn_mask"));
    if (mask_elems != 0)
        stat.flops += attn_elems;
    if (get_bool_param(op, "is_causal", false))
        stat.flops += attn_elems;

    stat.memops += input_size(op, 0);
    stat.memops += input_size(op, 1);
    stat.memops += input_size(op, 2);
    stat.memops += mask_elems;
    stat.memops += output_size(op, 0);
    stat.memops += attn_elems;
    return true;
}

static size_t attention_input_count(const Operator* op)
{
    size_t input_count = op->inputs.size();
    for (size_t i = 0; i < op->inputnames.size() && i < op->inputs.size(); i++)
    {
        if (op->inputnames[i] == "attn_mask")
            input_count -= 1;
    }

    return input_count;
}

static uint64_t attention_mask_input_size(const Operator* op)
{
    uint64_t size = 0;
    for (size_t i = 0; i < op->inputnames.size() && i < op->inputs.size(); i++)
    {
        if (op->inputnames[i] == "attn_mask")
            size += input_size(op, i);
    }

    return size;
}

static bool sequence_batch_shape(const std::vector<int>& shape, bool batch_first, int& batch, int& seq, int& dim)
{
    if (shape.size() == 2)
    {
        batch = 1;
        seq = shape[0];
        dim = shape[1];
        return true;
    }

    if (shape.size() == 3)
    {
        batch = shape[batch_first ? 0 : 1];
        seq = shape[batch_first ? 1 : 0];
        dim = shape[2];
        return true;
    }

    return false;
}

static bool count_multihead_attention(const Operator* op, ModelStat& stat)
{
    if (op->type != "nn.MultiheadAttention")
        return false;

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const bool batch_first = get_bool_param(op, "batch_first", false);
    const std::vector<int>& q_shape = op->inputs[0]->shape;

    const size_t input_count = attention_input_count(op);
    const Operand* k_operand = input_count >= 2 ? op->inputs[1] : op->inputs[0];
    const Operand* v_operand = input_count >= 3 ? op->inputs[2] : k_operand;
    if (!k_operand || !v_operand)
        return true;

    int batch = 0;
    int qlen = 0;
    int embed_dim_default = 0;
    int k_batch = 0;
    int klen = 0;
    int kdim_default = 0;
    int v_batch = 0;
    int vlen = 0;
    int vdim_default = 0;
    if (!sequence_batch_shape(q_shape, batch_first, batch, qlen, embed_dim_default))
        return true;
    if (!sequence_batch_shape(k_operand->shape, batch_first, k_batch, klen, kdim_default))
        return true;
    if (!sequence_batch_shape(v_operand->shape, batch_first, v_batch, vlen, vdim_default))
        return true;

    const int embed_dim = get_int_param(op, "embed_dim", embed_dim_default);
    const int kdim = get_int_param(op, "kdim", kdim_default);
    const int vdim = get_int_param(op, "vdim", vdim_default);
    const int num_heads = std::max(1, get_int_param(op, "num_heads", 1));
    if (batch <= 0 || k_batch != batch || v_batch != batch || qlen <= 0 || klen <= 0 || vlen <= 0 || embed_dim <= 0 || kdim <= 0 || vdim <= 0)
        return true;

    const uint64_t head_dim = embed_dim / num_heads;
    const bool bias = get_bool_param(op, "bias", true);
    int attn_klen = klen;
    int attn_vlen = vlen;
    if (get_bool_param(op, "add_bias_kv", false))
    {
        attn_klen++;
        attn_vlen++;
    }
    if (get_bool_param(op, "add_zero_attn", false))
    {
        attn_klen++;
        attn_vlen++;
    }
    const uint64_t attn_elems = (uint64_t)batch * num_heads * qlen * attn_klen;
    const uint64_t mask_elems = attention_mask_input_size(op);

    stat.flops += (uint64_t)batch * qlen * embed_dim * embed_dim * 2;
    stat.flops += (uint64_t)batch * klen * kdim * embed_dim * 2;
    stat.flops += (uint64_t)batch * vlen * vdim * embed_dim * 2;
    stat.flops += attn_elems * head_dim * 2;
    if (mask_elems != 0)
        stat.flops += attn_elems;
    stat.flops += attn_elems * 5;
    stat.flops += (uint64_t)batch * num_heads * qlen * attn_vlen * head_dim * 2;
    stat.flops += (uint64_t)batch * qlen * embed_dim * embed_dim * 2;

    if (bias)
    {
        stat.flops += (uint64_t)batch * qlen * embed_dim * 2;
        stat.flops += (uint64_t)batch * (klen + vlen) * embed_dim;
    }

    stat.memops += input_size(op, 0);
    stat.memops += operand_size(k_operand);
    stat.memops += operand_size(v_operand);
    stat.memops += output_size_all(op);
    stat.memops += attr_size(op, "in_proj_weight");
    stat.memops += attr_size(op, "q_proj_weight");
    stat.memops += attr_size(op, "k_proj_weight");
    stat.memops += attr_size(op, "v_proj_weight");
    stat.memops += attr_size(op, "out_proj.weight");
    stat.memops += attr_size(op, "in_proj_bias");
    stat.memops += attr_size(op, "out_proj.bias");
    stat.memops += attr_size(op, "bias_k");
    stat.memops += attr_size(op, "bias_v");
    stat.memops += mask_elems;
    stat.memops += attn_elems;
    return true;
}

static bool count_rnn(const Operator* op, ModelStat& stat)
{
    if (!type_in(op->type, {"nn.RNN", "nn.LSTM", "nn.GRU"}))
        return false;

    if (op->inputs.empty() || op->outputs.empty())
        return true;

    const bool batch_first = get_bool_param(op, "batch_first", false);
    int batch = 0;
    int seq = 0;
    int input_dim_default = 0;
    if (!sequence_batch_shape(op->inputs[0]->shape, batch_first, batch, seq, input_dim_default))
        return true;

    const int input_dim = get_int_param(op, "input_size", input_dim_default);
    const int hidden_size = get_int_param(op, "hidden_size", 0);
    const int num_layers = std::max(1, get_int_param(op, "num_layers", 1));
    const int directions = get_bool_param(op, "bidirectional", false) ? 2 : 1;
    const bool bias = get_bool_param(op, "bias", true);
    const int proj_size = op->type == "nn.LSTM" ? get_int_param(op, "proj_size", 0) : 0;
    const int recurrent_size = proj_size > 0 ? proj_size : hidden_size;
    if (batch <= 0 || seq <= 0 || input_dim <= 0 || hidden_size <= 0)
        return true;

    int gate_count = 1;
    uint64_t elementwise_per_hidden = 1;
    if (op->type == "nn.LSTM")
    {
        gate_count = 4;
        elementwise_per_hidden = 10;
    }
    else if (op->type == "nn.GRU")
    {
        gate_count = 3;
        elementwise_per_hidden = 8;
    }

    uint64_t flops_per_step_batch = 0;
    for (int layer = 0; layer < num_layers; layer++)
    {
        const int layer_input_dim = layer == 0 ? input_dim : recurrent_size * directions;
        uint64_t layer_flops = (uint64_t)directions * 2 * gate_count * layer_input_dim * hidden_size;
        layer_flops += (uint64_t)directions * 2 * gate_count * recurrent_size * hidden_size;
        layer_flops += (uint64_t)directions * elementwise_per_hidden * hidden_size;
        if (proj_size > 0)
            layer_flops += (uint64_t)directions * 2 * hidden_size * proj_size;
        if (bias)
            layer_flops += (uint64_t)directions * gate_count * hidden_size * 2;
        flops_per_step_batch += layer_flops;
    }

    stat.flops += flops_per_step_batch * batch * seq;
    stat.memops += input_size(op, 0);

    uint64_t weight_elems = 0;
    for (const auto& it : op->attrs)
    {
        if (string_starts_with(it.first, "weight") || string_starts_with(it.first, "bias"))
            weight_elems += shape_size(it.second.shape);
    }

    if (op->inputs.size() > 1)
    {
        for (size_t i = 1; i < op->inputs.size(); i++)
        {
            stat.memops += input_size(op, i);
        }
    }
    stat.memops += output_size_all(op);

    if (weight_elems == 0)
    {
        if (op->inputs.size() == 1)
        {
            for (int layer = 0; layer < num_layers; layer++)
            {
                const int layer_input_dim = layer == 0 ? input_dim : recurrent_size * directions;
                weight_elems += (uint64_t)directions * gate_count * hidden_size * (layer_input_dim + recurrent_size);
                if (proj_size > 0)
                    weight_elems += (uint64_t)directions * hidden_size * proj_size;
                if (bias)
                    weight_elems += (uint64_t)directions * gate_count * hidden_size * 2;
            }
        }
    }
    stat.memops += weight_elems;
    return true;
}

static bool count_memory_only(const Operator* op, ModelStat& stat)
{
    if (type_in(op->type, {"nn.Embedding", "F.embedding"}))
    {
        const uint64_t out_elems = output_size_all(op);
        stat.memops += input_size(op, 0);
        stat.memops += out_elems;
        stat.memops += out_elems;
        return true;
    }

    if (type_in(op->type, {"nn.Identity", "F.dropout", "nn.Dropout", "nn.Dropout2d", "nn.Dropout3d",
                           "Tensor.size", "Tensor.dim", "Tensor.reshape", "Tensor.reshape_as", "Tensor.view", "Tensor.view_as",
                           "Tensor.permute", "Tensor.transpose", "Tensor.t", "Tensor.squeeze", "Tensor.unsqueeze",
                           "Tensor.flatten", "Tensor.slice", "Tensor.select", "Tensor.unflatten",
                           "torch.flatten", "torch.reshape", "torch.view", "torch.permute", "torch.transpose", "torch.t",
                           "torch.squeeze", "torch.unsqueeze", "torch.split", "torch.chunk", "torch.unbind", "torch.tensor_split",
                           "torch.view_as_real", "torch.view_as_complex"
                          }))
    {
        return true;
    }

    if (type_in(op->type, {"nn.Unfold", "F.unfold",
                           "nn.PixelShuffle", "F.pixel_shuffle", "nn.PixelUnshuffle", "F.pixel_unshuffle",
                           "nn.ChannelShuffle", "F.pad", "nn.ConstantPad1d", "nn.ConstantPad2d", "nn.ConstantPad3d",
                           "nn.ReflectionPad1d", "nn.ReflectionPad2d", "nn.ReplicationPad1d", "nn.ReplicationPad2d", "nn.ReplicationPad3d",
                           "nn.ZeroPad2d", "torch.cat", "torch.stack", "torch.clone", "torch.flip", "torch.roll"
                          }))
    {
        stat.memops += input_size_all(op);
        stat.memops += output_size_all(op);
        return true;
    }

    if (string_starts_with(op->type, "Tensor."))
    {
        stat.memops += input_size_all(op);
        stat.memops += output_size_all(op);
        return true;
    }

    return false;
}

ModelStat get_model_stat(const Graph& graph)
{
    ModelStat stat;

    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        const Operator* op = graph.ops[i];

        if (op->type == "pnnx.Input" || op->type == "pnnx.Output" || op->type == "pnnx.Attribute")
            continue;

        if (count_convolution(op, stat))
            continue;
        if (count_linear(op, stat))
            continue;
        if (count_pooling(op, stat))
            continue;
        if (count_normalization(op, stat))
            continue;
        if (count_normalize(op, stat))
            continue;
        if (count_activation(op, stat))
            continue;
        if (count_upsample(op, stat))
            continue;
        if (count_local_response_norm(op, stat))
            continue;
        if (count_lp_pool(op, stat))
            continue;
        if (count_pairwise_distance(op, stat))
            continue;
        if (count_fold(op, stat))
            continue;
        if (count_scaled_dot_product_attention(op, stat))
            continue;
        if (count_multihead_attention(op, stat))
            continue;
        if (count_rnn(op, stat))
            continue;
        if (count_matmul(op, stat))
            continue;
        if (count_addmm(op, stat))
            continue;
        if (count_expression(op, stat))
            continue;
        if (count_elementwise(op, stat))
            continue;
        if (count_reduction(op, stat))
            continue;
        if (count_memory_only(op, stat))
            continue;

        if (string_starts_with(op->type, "nn.") || string_starts_with(op->type, "F.") || string_starts_with(op->type, "torch."))
        {
            stat.memops += input_size_all(op);
            stat.memops += output_size_all(op);
        }
    }

    return stat;
}

} // namespace pnnx
