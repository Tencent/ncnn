// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_H
#define PNNX_EXPORTED_PROGRAM_H

#include <stdint.h>

#include <map>
#include <string>
#include <vector>

namespace pnnx {
namespace pt2 {

struct SymInt
{
    enum Type
    {
        Integer,
        Expression
    };

    SymInt();

    Type type;
    int64_t integer;
    std::string expression;
    bool has_hint;
    int64_t hint;
};

struct Device
{
    Device();

    std::string type;
    bool has_index;
    int index;
};

struct TensorMeta
{
    TensorMeta();

    int scalar_type;
    std::vector<SymInt> sizes;
    bool requires_grad;
    Device device;
    std::vector<SymInt> strides;
    SymInt storage_offset;
    int layout;
};

struct Argument
{
    enum Type
    {
        Unknown,
        None,
        Tensor,
        Tensors,
        Integer,
        Integers,
        FloatingPoint,
        FloatingPoints,
        String,
        Strings,
        SymInteger,
        SymIntegers,
        ScalarType,
        MemoryFormat,
        Layout,
        DeviceValue,
        Boolean,
        Booleans,
        SymBoolean,
        SymBooleans,
        SymFloat,
        SymFloats,
        OptionalTensor,
        OptionalTensors,
        Complex
    };

    Argument();

    Type type;
    std::string name;
    bool boolean;
    int64_t integer;
    double floating_point;
    double complex_real;
    double complex_imag;
    std::string string;
    Device device;
    std::vector<Argument> values;
};

struct NamedArgument
{
    enum Kind
    {
        KindUnknown,
        Positional,
        Keyword
    };

    NamedArgument();

    std::string name;
    Argument argument;
    Kind kind;
};

struct Node
{
    std::string target;
    std::vector<NamedArgument> inputs;
    std::vector<Argument> outputs;
    std::map<std::string, std::string> metadata;
};

struct Graph
{
    Graph();

    std::vector<Argument> inputs;
    std::vector<Argument> outputs;
    std::vector<Node> nodes;
    std::map<std::string, TensorMeta> tensor_values;
    std::map<std::string, SymInt> sym_int_values;
    bool is_single_tensor_return;
};

struct InputSpec
{
    enum Type
    {
        UserInput,
        Parameter,
        Buffer,
        TensorConstant,
        CustomObject,
        Token,
        ConstantInput
    };

    InputSpec();

    Type type;
    Argument argument;
    std::string target;
    bool persistent;
};

struct OutputSpec
{
    enum Type
    {
        UserOutput,
        LossOutput,
        BufferMutation,
        ParameterMutation,
        GradientToParameter,
        GradientToUserInput,
        UserInputMutation,
        Token
    };

    OutputSpec();

    Type type;
    Argument argument;
    std::string target;
};

struct GraphSignature
{
    std::vector<InputSpec> inputs;
    std::vector<OutputSpec> outputs;
};

struct RangeConstraint
{
    RangeConstraint();

    bool has_min;
    int64_t min;
    bool has_max;
    int64_t max;
};

struct SchemaVersion
{
    SchemaVersion();

    int major;
    int minor;
};

struct ExportedProgram
{
    Graph graph;
    GraphSignature signature;
    std::map<std::string, int> opset_version;
    std::map<std::string, RangeConstraint> range_constraints;
    SchemaVersion schema_version;
    std::string torch_version;
};

struct PayloadMeta
{
    PayloadMeta();

    std::string path;
    bool is_parameter;
    bool use_pickle;
    bool has_tensor_meta;
    TensorMeta tensor_meta;
};

struct ExportedProgramArchive
{
    ExportedProgramArchive();

    int archive_version;
    std::string model_name;
    ExportedProgram program;
    std::map<std::string, PayloadMeta> state_dict;
    std::map<std::string, PayloadMeta> constants;
};

} // namespace pt2
} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_H