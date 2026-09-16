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
    std::string name;
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
    // Archive parsing validates protocol-1 root TreeSpecs when supplied:
    // flat positional tensor inputs (no kwargs), and a tensor/numeric scalar
    // or nonempty flat tuple of such outputs. Nested/list/dict/namedtuple call
    // structures are rejected, not silently flattened. Singleton output tuples
    // retain the existing flattened PNNX convention, not Python tuple identity.
    // Missing/empty module_call_graph is allowed for schema-only fixtures.
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
    // Current raw archives: root "byteorder" record, default little if absent.
    // Only little-endian payloads on little-endian hosts are supported.
    std::string byteorder = "little";
    ExportedProgram program;
    std::map<std::string, PayloadMeta> state_dict;
    std::map<std::string, PayloadMeta> constants;
    std::map<std::string, std::vector<char> > state_dict_storages;
    std::map<std::string, std::vector<char> > constant_storages;
};

bool parse_exported_program(const std::string& text, ExportedProgram& program, std::string& error);
// On failure these public loaders clear the result, including partial payloads.
// Both loaders preflight STORE-only records, byteorder and the root call spec.
// The full loader also preflights a combined 512 MiB raw payload budget
// (unique storage bytes + every dense view, across weights/constants).
// This is not a bound on LibTorch's legacy pickle deserializer allocations.
bool load_exported_program_archive_metadata(const std::string& path, ExportedProgramArchive& archive, std::string& error);
bool load_exported_program_archive(const std::string& path, ExportedProgramArchive& archive, std::string& error);

bool validate_exported_program_version(const ExportedProgram& program, std::string& error);
// Shared by the archive reader and the public in-memory importer. Validates
// metadata, source bounds and the dense materialization budget before allocation.
bool validate_tensor_storage(const PayloadMeta& payload, uint64_t storage_size, std::string& error);

} // namespace pt2
} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_H