// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_SCHEMA_H
#define PNNX_PT2_SCHEMA_H

#include "storezip.h"

#include <stdint.h>
#include <map>
#include <string>
#include <vector>

// PT2 schema mirror. Normalization belongs to pass_level2.

namespace pnnx {

// Node argument, represented by one of the as_* variants.
struct Pt2Argument
{
    enum ArgType
    {
        NONE,
        TENSOR,        // as_tensor
        TENSORS,       // as_tensors
        INT,           // as_int
        INTS,          // as_ints
        FLOAT,         // as_float
        FLOATS,        // as_floats
        BOOL,          // as_bool
        BOOLS,         // as_bools
        STRING,        // as_string
        STRINGS,       // as_strings
        SCALAR_TYPE,   // as_scalar_type
        DEVICE,        // as_device
        MEMORY_FORMAT, // as_memory_format
        SYMBOLIC       // as_sym_int / as_sym_ints
    };

    ArgType type;
    bool is_kwarg;
    std::string name;

    std::vector<std::string> tensor_names;
    long long int_value;
    std::vector<long long> int_values;
    double float_value;
    std::vector<double> float_values;
    bool bool_value;
    std::vector<bool> bool_values;
    std::string string_value;
    std::vector<std::string> string_values;
    std::string device_type;
    long long device_index;

    Pt2Argument()
        : type(NONE), is_kwarg(false), int_value(0), float_value(0), bool_value(false), device_index(-1)
    {
    }
};

struct Pt2NodeInput
{
    std::string name;
    Pt2Argument arg;
};

struct Pt2NodeOutput
{
    std::vector<std::string> tensor_names;
};

struct Pt2Node
{
    std::string name;
    std::string target;
    std::vector<Pt2NodeInput> inputs;
    std::vector<Pt2NodeOutput> outputs;

    std::string nn_module_stack;
    std::string torch_fn;
    std::string stack_trace;
    std::vector<int> adaptive_pool_none_axes;
    bool adaptive_pool_has_none;

    Pt2Node()
        : adaptive_pool_has_none(false)
    {
    }
};

struct Pt2InputSpec
{
    enum SpecKind
    {
        USER_INPUT,
        PARAMETER,
        BUFFER,
        TENSOR_CONSTANT
    };

    SpecKind kind;
    std::string graph_name;
    std::string state_dict_name;
    bool persistent;

    Pt2InputSpec()
        : kind(USER_INPUT), persistent(false)
    {
    }
};

struct Pt2OutputSpec
{
    std::string graph_name;
};

struct Pt2WeightEntry
{
    std::string state_dict_name;
    std::string path_name;
    bool is_param;
    bool use_pickle;
    long long dtype;
    std::vector<long long> sizes;
    std::vector<long long> strides;
    long long storage_offset;

    Pt2WeightEntry()
        : is_param(false), use_pickle(false), dtype(0), storage_offset(0)
    {
    }
};

struct Pt2TensorMeta
{
    long long dtype;
    std::vector<long long> sizes;

    Pt2TensorMeta()
        : dtype(-1)
    {
    }
};

struct Pt2Program
{
    std::string zippath;

    long long schema_version_major;
    long long schema_version_minor;
    std::string torch_version;
    std::map<std::string, std::string> opset_version;
    std::string archive_root;

    Pt2Program()
        : schema_version_major(-1), schema_version_minor(-1)
    {
    }

    std::vector<Pt2Node> nodes;
    std::vector<Pt2InputSpec> input_specs;
    std::vector<Pt2OutputSpec> output_specs;

    std::map<std::string, Pt2TensorMeta> tensor_values;

    std::vector<Pt2WeightEntry> weights;
    std::vector<Pt2WeightEntry> constants;

    std::string weight_entry_path(const std::string& path_name) const;
    std::string constant_entry_path(const std::string& path_name) const;

    const Pt2WeightEntry* find_weight(const std::string& state_dict_name) const;
    const Pt2WeightEntry* find_constant(const std::string& state_dict_name) const;
};

int load_pt2_schema(const std::string& ptpath, Pt2Program& program);

// Locate the model.json entry used to identify a PT2 archive.
std::string find_pt2_model_json_entry(const std::vector<std::string>& entry_names);

} // namespace pnnx

#endif // PNNX_PT2_SCHEMA_H
