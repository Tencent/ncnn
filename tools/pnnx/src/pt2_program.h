// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_PROGRAM_H
#define PNNX_PT2_PROGRAM_H

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace pnnx {

class Pt2ArchiveReader;
struct Pt2JsonValue;

struct Pt2SymInt
{
    Pt2SymInt();

    bool symbolic;
    int64_t value;
    bool has_hint;
    int64_t hint;
    std::string expression;
};

struct Pt2Argument
{
    enum Type
    {
        None,
        Tensor,
        Tensors,
        OptionalTensor,
        OptionalTensors,
        Int,
        Ints,
        Float,
        Floats,
        Complex,
        String,
        Strings,
        Bool,
        Bools,
        ScalarType,
        MemoryFormat,
        Layout,
        Device,
        SymInt,
        SymInts,
        SymBool
    };

    Pt2Argument();

    Type type;
    int64_t i;
    double f;
    bool b;
    std::string s;
    std::vector<int64_t> ai;
    std::vector<double> af;
    std::vector<std::string> as;
    std::vector<unsigned char> ab;
    std::vector<Pt2Argument> args;
};

struct Pt2Tensor
{
    int dtype;
    bool requires_grad;
    std::string device;
    int device_index;
    int layout;
    std::vector<Pt2SymInt> sizes;
    std::vector<Pt2SymInt> strides;
    Pt2SymInt storage_offset;
};

struct Pt2NamedArgument
{
    std::string name;
    Pt2Argument arg;
    int kind;
};

struct Pt2Node
{
    std::string name;
    std::string target;
    std::vector<Pt2NamedArgument> inputs;
    std::vector<Pt2Argument> outputs;
};

struct Pt2InputSpec
{
    enum Kind
    {
        UserInput,
        Parameter,
        Buffer,
        TensorConstant,
        ConstantInput
    };

    Kind kind;
    Pt2Argument arg;
    std::string target;
    bool persistent;
};

struct Pt2OutputSpec
{
    Pt2Argument arg;
};

struct Pt2RangeConstraint
{
    bool has_min;
    bool has_max;
    int64_t min;
    int64_t max;
};

struct Pt2Program
{
    Pt2Program();

    int schema_major;
    int schema_minor;
    std::string torch_version;
    std::map<std::string, int> opset_versions;
    std::vector<Pt2Argument> inputs;
    std::vector<Pt2Argument> outputs;
    std::vector<Pt2Node> nodes;
    std::map<std::string, Pt2Tensor> tensors;
    std::map<std::string, Pt2SymInt> sym_ints;
    std::set<std::string> sym_bools;
    std::vector<Pt2InputSpec> input_specs;
    std::vector<Pt2OutputSpec> output_specs;
    std::map<std::string, Pt2RangeConstraint> range_constraints;
    std::vector<std::string> guards_code;
    std::string error;
};

int parse_pt2_program(const unsigned char* data, size_t size, Pt2Program& program);
int decode_pt2_tensor_meta(const Pt2JsonValue& value, const std::string& path, Pt2Tensor& tensor, std::string& error);
int load_pt2_program(Pt2ArchiveReader& archive, Pt2Program& program);

} // namespace pnnx

#endif // PNNX_PT2_PROGRAM_H
