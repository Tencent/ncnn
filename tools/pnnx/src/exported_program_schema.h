// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_SCHEMA_H
#define PNNX_EXPORTED_PROGRAM_SCHEMA_H

#include "json_reader.h"

#include <stdint.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pnnx {

struct ExportedProgramHeader
{
    ExportedProgramHeader();

    int schema_major;
    int schema_minor;
    std::string torch_version;
    std::map<std::string, int64_t> opset_version;
};

struct ExportedTensorMeta
{
    ExportedTensorMeta();

    int64_t dtype;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int64_t storage_offset;
    int64_t layout;
    bool requires_grad;
    std::string device_type;
    int64_t device_index;
    bool has_device_index;
};

struct ExportedPayloadEntry
{
    ExportedPayloadEntry();

    std::string path_name;
    bool is_param;
    bool use_pickle;
    bool has_tensor_meta;
    ExportedTensorMeta tensor_meta;
};

struct ExportedPayloadConfig
{
    std::map<std::string, ExportedPayloadEntry> entries;
};

struct ExportedGraph;

enum ExportedArgumentType
{
    EXPORTED_ARGUMENT_NONE,
    EXPORTED_ARGUMENT_TENSOR,
    EXPORTED_ARGUMENT_TENSOR_LIST,
    EXPORTED_ARGUMENT_INT,
    EXPORTED_ARGUMENT_INT_LIST,
    EXPORTED_ARGUMENT_FLOAT,
    EXPORTED_ARGUMENT_FLOAT_LIST,
    EXPORTED_ARGUMENT_COMPLEX,
    EXPORTED_ARGUMENT_BOOL,
    EXPORTED_ARGUMENT_BOOL_LIST,
    EXPORTED_ARGUMENT_STRING,
    EXPORTED_ARGUMENT_STRING_LIST,
    EXPORTED_ARGUMENT_SCALAR_TYPE,
    EXPORTED_ARGUMENT_MEMORY_FORMAT,
    EXPORTED_ARGUMENT_LAYOUT,
    EXPORTED_ARGUMENT_DEVICE,
    EXPORTED_ARGUMENT_GRAPH,
    EXPORTED_ARGUMENT_UNSUPPORTED
};

struct ExportedDevice
{
    ExportedDevice();

    std::string type;
    int64_t index;
    bool has_index;
};

struct ExportedArgument
{
    ExportedArgument();

    ExportedArgumentType type;
    std::string name;
    int64_t int_value;
    double float_value;
    double complex_real_value;
    double complex_imag_value;
    bool bool_value;
    std::string string_value;
    std::vector<std::string> tensor_names;
    std::vector<int64_t> int_values;
    std::vector<double> float_values;
    std::vector<bool> bool_values;
    std::vector<std::string> string_values;
    int64_t enum_value;
    ExportedDevice device_value;
    std::string graph_name;
    std::shared_ptr<ExportedGraph> graph_value;
    std::string unsupported_tag;
};

enum ExportedArgumentKind
{
    EXPORTED_ARGUMENT_KIND_MISSING = -1,
    EXPORTED_ARGUMENT_KIND_UNKNOWN = 0,
    EXPORTED_ARGUMENT_KIND_POSITIONAL = 1,
    EXPORTED_ARGUMENT_KIND_KEYWORD = 2
};

struct ExportedNamedArgument
{
    ExportedNamedArgument();

    std::string name;
    ExportedArgument arg;
    ExportedArgumentKind kind;
};

struct ExportedNode
{
    ExportedNode();

    std::string name;
    bool has_name;
    std::string target;
    std::vector<ExportedNamedArgument> inputs;
    std::vector<ExportedArgument> outputs;
};

enum ExportedInputKind
{
    EXPORTED_USER_INPUT,
    EXPORTED_PARAMETER,
    EXPORTED_BUFFER,
    EXPORTED_TENSOR_CONSTANT,
    EXPORTED_CONSTANT_INPUT,
    EXPORTED_CUSTOM_OBJ,
    EXPORTED_TOKEN
};

enum ExportedOutputKind
{
    EXPORTED_USER_OUTPUT,
    EXPORTED_LOSS_OUTPUT,
    EXPORTED_BUFFER_MUTATION,
    EXPORTED_PARAMETER_MUTATION,
    EXPORTED_GRADIENT_TO_PARAMETER,
    EXPORTED_GRADIENT_TO_USER_INPUT,
    EXPORTED_USER_INPUT_MUTATION,
    EXPORTED_OUTPUT_TOKEN
};

struct ExportedInputSpec
{
    ExportedInputSpec();

    ExportedInputKind kind;
    ExportedArgument arg;
    std::string target;
    bool persistent;
};

struct ExportedOutputSpec
{
    ExportedOutputSpec();

    ExportedOutputKind kind;
    ExportedArgument arg;
    std::string target;
};

enum ExportedTreeSpecType
{
    EXPORTED_TREE_SPEC_NONE,
    EXPORTED_TREE_SPEC_LEAF,
    EXPORTED_TREE_SPEC_TUPLE,
    EXPORTED_TREE_SPEC_LIST
};

struct ExportedTreeSpec
{
    ExportedTreeSpec();

    ExportedTreeSpecType type;
    std::vector<ExportedTreeSpec> children;
};

struct ExportedGraph
{
    ExportedGraph();

    std::vector<ExportedArgument> inputs;
    std::vector<ExportedNode> nodes;
    std::vector<ExportedArgument> outputs;
    std::map<std::string, ExportedTensorMeta> tensor_values;
    std::map<std::string, ExportedArgument> custom_obj_values;
    bool is_single_tensor_return;
};

struct ExportedProgram
{
    ExportedProgramHeader header;
    ExportedGraph graph;
    std::vector<ExportedInputSpec> input_specs;
    std::vector<ExportedOutputSpec> output_specs;
    ExportedTreeSpec output_tree_spec;
};

struct ExportedSchemaError
{
    std::string path;
    std::string message;
};

int parse_exported_program_header(const JsonValue& value, ExportedProgramHeader& header, ExportedSchemaError& error);
int parse_exported_tensor_meta(const JsonValue& value, ExportedTensorMeta& tensor_meta, ExportedSchemaError& error, const std::string& path);
int parse_exported_payload_config(const JsonValue& value, ExportedPayloadConfig& payload_config, ExportedSchemaError& error);
int parse_exported_program(const JsonValue& value, ExportedProgram& program, ExportedSchemaError& error);

} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_SCHEMA_H
