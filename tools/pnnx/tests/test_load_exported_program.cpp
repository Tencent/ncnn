// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <string.h>

#include <limits>
#include <set>

#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/library.h>

#include "exported_program_defaults.h"
#include "load_exported_program.h"

static int test_failures = 0;

static void expect_true(bool value, const std::string& message)
{
    if (value)
        return;
    fprintf(stderr, "FAILED: %s\n", message.c_str());
    test_failures++;
}

static pnnx::pt2::ExportedProgram make_program()
{
    pnnx::pt2::ExportedProgram program;
    program.schema_version.major = 8;
    program.schema_version.minor = 20;
    program.opset_version["aten"] = (int)torch::jit::getMaxOperatorVersion();
    return program;
}

static pnnx::pt2::SymInt dimension(int64_t value)
{
    pnnx::pt2::SymInt result;
    result.integer = value;
    return result;
}

static pnnx::pt2::ExportedProgramArchive make_archive()
{
    pnnx::pt2::ExportedProgramArchive archive;
    archive.program = make_program();

    pnnx::pt2::TensorMeta input_meta;
    input_meta.scalar_type = 7;
    input_meta.sizes.push_back(dimension(2));
    input_meta.sizes.push_back(dimension(3));
    archive.program.graph.tensor_values["x"] = input_meta;

    pnnx::pt2::TensorMeta output_meta = input_meta;
    output_meta.sizes[1] = dimension(2);
    archive.program.graph.tensor_values["linear"] = output_meta;

    pnnx::pt2::InputSpec weight;
    weight.type = pnnx::pt2::InputSpec::Parameter;
    weight.argument.type = pnnx::pt2::Argument::Tensor;
    weight.argument.name = "p_weight";
    weight.target = "linear.weight";
    archive.program.signature.inputs.push_back(weight);
    archive.program.graph.inputs.push_back(weight.argument);

    pnnx::pt2::InputSpec input;
    input.type = pnnx::pt2::InputSpec::UserInput;
    input.argument.type = pnnx::pt2::Argument::Tensor;
    input.argument.name = "x";
    archive.program.signature.inputs.push_back(input);
    archive.program.graph.inputs.push_back(input.argument);

    pnnx::pt2::PayloadMeta payload;
    payload.path = "weight_0";
    payload.is_parameter = true;
    payload.has_tensor_meta = true;
    payload.tensor_meta.scalar_type = 7;
    payload.tensor_meta.device.type = "cpu";
    payload.tensor_meta.layout = 7;
    payload.tensor_meta.sizes.push_back(dimension(2));
    payload.tensor_meta.sizes.push_back(dimension(2));
    payload.tensor_meta.strides.push_back(dimension(3));
    payload.tensor_meta.strides.push_back(dimension(1));
    payload.tensor_meta.storage_offset = dimension(1);
    archive.state_dict["linear.weight"] = payload;

    std::vector<char>& storage = archive.state_dict_storages["data/weights/weight_0"];
    const float values[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    storage.resize(sizeof(values));
    memcpy(storage.data(), values, sizeof(values));

    pnnx::pt2::Node node;
    node.name = "linear";
    node.target = "torch.ops.aten.linear.default";
    pnnx::pt2::NamedArgument node_input;
    node_input.name = "input";
    node_input.argument.type = pnnx::pt2::Argument::Tensor;
    node_input.argument.name = "x";
    node.inputs.push_back(node_input);
    pnnx::pt2::NamedArgument node_weight;
    node_weight.name = "weight";
    node_weight.argument.type = pnnx::pt2::Argument::Tensor;
    node_weight.argument.name = "p_weight";
    node.inputs.push_back(node_weight);
    pnnx::pt2::NamedArgument node_bias;
    node_bias.name = "bias";
    node_bias.argument.type = pnnx::pt2::Argument::None;
    node.inputs.push_back(node_bias);
    pnnx::pt2::Argument output;
    output.type = pnnx::pt2::Argument::Tensor;
    output.name = "linear";
    node.outputs.push_back(output);
    archive.program.graph.nodes.push_back(node);

    archive.program.graph.outputs.push_back(output);
    archive.program.graph.outputs.push_back(output);
    pnnx::pt2::OutputSpec output_spec;
    output_spec.type = pnnx::pt2::OutputSpec::UserOutput;
    output_spec.argument = output;
    archive.program.signature.outputs.push_back(output_spec);
    archive.program.signature.outputs.push_back(output_spec);
    return archive;
}

static bool import_constant(const pnnx::pt2::Argument& value, pnnx::Parameter& result, std::string& error)
{
    pnnx::pt2::ExportedProgram program = make_program();
    program.graph.outputs.push_back(value);
    pnnx::pt2::OutputSpec spec;
    spec.argument = value;
    program.signature.outputs.push_back(spec);
    pnnx::Graph graph;
    if (pnnx::import_exported_program_outputs(program, graph, error) != 0)
        return false;
    result = graph.ops.front()->params["value"];
    return true;
}

static void test_scalar_contract()
{
    using namespace pnnx::pt2;
    Argument value;
    pnnx::Parameter result;
    std::string error;
    value.type = Argument::ScalarType;
    const int expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15};
    for (int dtype = 1; dtype <= 13; dtype++)
    {
        value.integer = dtype;
        expect_true(import_constant(value, result, error), error);
        expect_true(result.type == 2 && result.i == expected[dtype - 1], "explicit serde dtype to JIT mapping (including bf16=15)");
    }
    const int unsupported[] = {-1, 0, 14, 15, 28, 29, 35, 999};
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++)
    {
        value.integer = unsupported[i];
        expect_true(!import_constant(value, result, error) && error.find("unsupported serde scalar type") != std::string::npos, "unknown/unrepresentable dtype rejected");
    }
    value.type = Argument::MemoryFormat;
    const int formats[] = {0, 2, 3, 1};
    for (int format = 1; format <= 4; format++)
    {
        value.integer = format;
        expect_true(import_constant(value, result, error) && result.i == formats[format - 1], "serde memory format maps explicitly");
    }
    value.integer = 0;
    expect_true(!import_constant(value, result, error), "unknown memory format rejected");
    value.integer = 5;
    expect_true(!import_constant(value, result, error), "future memory format rejected");
    value.type = Argument::Layout;
    value.integer = 7;
    expect_true(import_constant(value, result, error) && result.i == 0, "serde Strided maps to c10 Strided");
    value.integer = 0;
    expect_true(!import_constant(value, result, error), "unknown layout rejected");
    value.integer = 1;
    expect_true(!import_constant(value, result, error), "unsupported sparse layout rejected");

    value.type = Argument::SymBoolean;
    value.boolean = true;
    expect_true(import_constant(value, result, error) && result.type == 1 && result.b, "concrete SymBool is boolean");
    value.boolean = false;
    expect_true(import_constant(value, result, error) && result.type == 1 && !result.b, "concrete false SymBool is boolean");
    value.type = Argument::SymFloat;
    value.floating_point = 1.5;
    expect_true(import_constant(value, result, error) && result.type == 3 && result.f == 1.5f, "concrete SymFloat is float");
    value.floating_point = -std::numeric_limits<double>::infinity();
    expect_true(import_constant(value, result, error) && result.f == -std::numeric_limits<float>::infinity(), "concrete infinite SymFloat");
    value.type = Argument::SymInteger;
    const int64_t valid[] = {INT_MIN, 0, INT_MAX};
    for (size_t i = 0; i < 3; i++)
    {
        value.integer = valid[i];
        expect_true(import_constant(value, result, error) && result.i == valid[i], "concrete SymInt boundary");
    }
    const int64_t invalid[] = {(int64_t)INT_MIN - 1, (int64_t)INT_MAX + 1, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    for (size_t i = 0; i < 4; i++)
    {
        value.integer = invalid[i];
        expect_true(!import_constant(value, result, error) && error.find("out of pnnx range") != std::string::npos, "SymInt narrowing is checked, not truncated or sentinel-clamped");
    }
}

static void expect_inputs_rejected(const pnnx::pt2::ExportedProgramArchive& archive, const char* diagnostic)
{
    pnnx::Graph graph;
    std::string error;
    expect_true(pnnx::import_exported_program_inputs(archive, graph, error) != 0, "hostile in-memory input rejected without reader");
    expect_true(error.find(diagnostic) != std::string::npos, error);
    for (size_t i = 0; i < graph.ops.size(); i++)
        if (graph.ops[i]->type == "pnnx.Attribute") expect_true(graph.ops[i]->attrs["data"].data.empty(), "rejected payload is not allocated");
}

static void test_import_bypass_storage()
{
    using namespace pnnx::pt2;
    const ExportedProgramArchive baseline = make_archive();
    ExportedProgramArchive archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.strides.pop_back();
    expect_inputs_rejected(archive, "rank mismatch");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.storage_offset = dimension(-1);
    expect_inputs_rejected(archive, "storage offset");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.storage_offset = dimension(2);
    expect_inputs_rejected(archive, "exceeds storage");
    archive = baseline;
    archive.state_dict_storages["data/weights/weight_0"].resize(3);
    expect_inputs_rejected(archive, "exceeds storage");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.strides[0] = dimension(-1);
    expect_inputs_rejected(archive, "nonnegative integers");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.strides[0].type = SymInt::Expression;
    expect_inputs_rejected(archive, "nonnegative integers");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.sizes[0] = dimension(std::numeric_limits<int64_t>::max());
    archive.state_dict["linear.weight"].tensor_meta.strides[0] = dimension(std::numeric_limits<int64_t>::max());
    expect_inputs_rejected(archive, "overflows uint64");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.sizes.assign(65, dimension(1));
    archive.state_dict["linear.weight"].tensor_meta.strides.assign(65, dimension(0));
    expect_inputs_rejected(archive, "rank exceeds");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.sizes.assign(1, dimension(INT_MAX));
    archive.state_dict["linear.weight"].tensor_meta.strides.assign(1, dimension(0));
    expect_inputs_rejected(archive, "materialization budget");
    archive = baseline;
    archive.state_dict["linear.weight"].use_pickle = true;
    expect_inputs_rejected(archive, "pickled");
    archive = baseline;
    archive.state_dict["linear.weight"].has_tensor_meta = false;
    expect_inputs_rejected(archive, "metadata is missing");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.layout = 1;
    expect_inputs_rejected(archive, "Strided layout");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.device.type = "cuda";
    expect_inputs_rejected(archive, "CPU device");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.scalar_type = 29;
    expect_inputs_rejected(archive, "scalar type 29");

    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.scalar_type = 13;
    pnnx::Graph bf16_graph;
    std::string error;
    expect_true(pnnx::import_exported_program_inputs(archive, bf16_graph, error) == 0, error);
    expect_true(!bf16_graph.ops.empty() && bf16_graph.ops[0]->attrs["data"].type == 13 && bf16_graph.ops[0]->attrs["data"].data.size() == 8, "bf16 raw attribute stays bf16");
    archive = baseline;
    archive.state_dict["linear.weight"].tensor_meta.sizes.assign(1, dimension(4));
    archive.state_dict["linear.weight"].tensor_meta.strides.assign(1, dimension(0));
    pnnx::Graph broadcast_graph;
    expect_true(pnnx::import_exported_program_inputs(archive, broadcast_graph, error) == 0, "small zero-stride view remains supported");
    if (!broadcast_graph.ops.empty() && broadcast_graph.ops[0]->attrs["data"].data.size() == 16)
    {
        float values[4];
        memcpy(values, broadcast_graph.ops[0]->attrs["data"].data.data(), sizeof(values));
        expect_true(values[0] == 1.f && values[3] == 1.f, "zero-stride materialization repeats value");
    }
}

static pnnx::pt2::Node unary_node(const char* target)
{
    pnnx::pt2::Node node;
    node.name = "contract_node";
    node.target = target;
    pnnx::pt2::NamedArgument input;
    input.name = "self";
    input.argument.type = pnnx::pt2::Argument::Tensor;
    input.argument.name = "x";
    node.inputs.push_back(input);
    pnnx::pt2::Argument output = input.argument;
    output.name = "contract_output";
    node.outputs.push_back(output);
    return node;
}

static void expect_node_rejected(const pnnx::pt2::Node& node, const char* diagnostic, const char* importer_diagnostic = 0)
{
    pnnx::pt2::Node normalized = node;
    std::string normalization_error;
    expect_true(!pnnx::pt2::normalize_exported_program_node(normalized, normalization_error), "normalizer itself rejects the malformed/unsafe node");
    expect_true(normalization_error.find(diagnostic) != std::string::npos && normalization_error.find(node.target) != std::string::npos, normalization_error);
    pnnx::pt2::ExportedProgram program = make_program();
    program.graph.nodes.push_back(node);
    pnnx::Graph graph;
    std::string error;
    expect_true(pnnx::import_exported_program_nodes(program, graph, error) != 0, "public node import rejects unsafe schema");
    if (importer_diagnostic)
        expect_true(error == importer_diagnostic, error);
    else
        expect_true(error.find(diagnostic) != std::string::npos && error.find(node.target) != std::string::npos, error);
    expect_true(graph.ops.empty(), "unsafe node rejected before graph construction");
}

// Schemas only: never execute custom code. A dispatcher registration is not
// evidence that an external operator has no side effects.
TORCH_LIBRARY(pnnx_pt2_contract, m)
{
    m.def("write(Tensor(a!) self) -> Tensor(a!)");
    m.def("external(Tensor self) -> Tensor");
    m.def(torch::schema("conservative(Tensor self) -> Tensor", c10::AliasAnalysisKind::CONSERVATIVE));
}

static void test_node_safety()
{
    using namespace pnnx::pt2;
    Node node = unary_node("torch.ops.aten.add_.Tensor");
    NamedArgument other = node.inputs[0];
    other.name = "other";
    node.inputs.push_back(other);
    expect_node_rejected(node, "alias write/mutation of argument self");
    node = unary_node("torch.ops.aten.add.out");
    node.inputs.push_back(other);
    NamedArgument out = node.inputs[0];
    out.name = "out";
    out.argument.name = "out_tensor";
    node.inputs.push_back(out);
    expect_node_rejected(node, "alias write/mutation of argument out");
    node = unary_node("torch.ops.pnnx_pt2_contract.write.default");
    expect_node_rejected(node, "alias write");
    node = unary_node("torch.ops.pnnx_pt2_contract.external.default");
    expect_node_rejected(node, "external side effects");
    node = unary_node("torch.ops.pnnx_pt2_contract.conservative.default");
    expect_node_rejected(node, "external side effects");
    node = unary_node("_operator.setitem");
    expect_node_rejected(node, "validated pure schema");
    node = unary_node("torch.ops.aten.relu.default");
    node.outputs[0] = Argument();
    node.outputs[0].type = Argument::None;
    expect_node_rejected(node, "does not match schema return type Tensor");
    node.outputs.clear();
    expect_node_rejected(node, "output count 0 does not match schema return count 1");
    node = unary_node("torch.ops.aten._assert_async.msg");
    expect_node_rejected(node, "guard");
    node = unary_node("torch.ops.aten._functional_assert_async.msg");
    expect_node_rejected(node, "guard");
    node = unary_node("torch.ops.aten.sym_constrain_range.default");
    expect_node_rejected(node, "guard");
    node = unary_node("torch.ops.aten._test_check_tensor.default");
    expect_node_rejected(node, "runtime guard evaluation is not implemented");
    const char* random_factories[] = {"torch.ops.aten.rand_like.default", "torch.ops.aten.randn_like.default"};
    for (size_t i = 0; i < 2; i++)
        expect_node_rejected(unary_node(random_factories[i]), "RNG state effect");

    node = unary_node("torch.ops.aten._assert_scalar.default");
    node.inputs[0].argument = Argument();
    node.inputs[0].argument.type = Argument::SymBoolean;
    node.inputs[0].argument.boolean = false;
    NamedArgument message;
    message.name = "assert_msg";
    message.argument.type = Argument::String;
    message.argument.string = "required predicate";
    node.inputs.push_back(message);
    node.outputs[0] = Argument();
    node.outputs[0].type = Argument::None;
    expect_node_rejected(node, "scalar guard is false: required predicate");
    node.inputs[0].argument.name = "symbolic_predicate";
    expect_node_rejected(node, "runtime guards are not implemented");
    node.inputs[0].argument.name.clear();
    node.inputs[0].argument.boolean = true;
    ExportedProgram guard_program = make_program();
    guard_program.graph.nodes.push_back(node);
    pnnx::Graph guard_graph;
    std::string error;
    expect_true(pnnx::import_exported_program_nodes(guard_program, guard_graph, error) == 0 && guard_graph.ops.empty(), "proven true concrete guard is safely discharged");
    guard_program.graph.nodes[0].outputs.clear();
    expect_true(pnnx::import_exported_program_nodes(guard_program, guard_graph, error) == 0, "zero-return true guard is evaluated too");
    guard_program.graph.nodes[0].inputs[0].argument.type = Argument::Boolean;
    expect_true(pnnx::import_exported_program_nodes(guard_program, guard_graph, error) == 0 && guard_graph.ops.empty(), "concrete Boolean guard with [] is safely discharged too");
    guard_program.graph.nodes[0].inputs[0].argument.boolean = false;
    expect_node_rejected(guard_program.graph.nodes[0], "scalar guard is false: required predicate");
    Node malformed_guard = node;
    malformed_guard.outputs.push_back(node.outputs[0]);
    expect_node_rejected(malformed_guard, "output count 2 does not match schema return count 0");
    malformed_guard.outputs.assign(1, unary_node("torch.ops.aten.relu.default").outputs[0]);
    expect_node_rejected(malformed_guard, "output count 1 does not match schema return count 0");
    malformed_guard = node;
    malformed_guard.inputs[0].argument.type = Argument::Boolean;
    malformed_guard.inputs[0].argument.name = "unresolved_predicate";
    expect_node_rejected(malformed_guard, "reference name unresolved_predicate");

    node = unary_node("torch.ops.aten.relu.default");
    node.inputs[0].argument.type = Argument::Integer;
    node.inputs[0].argument.name.clear();
    expect_node_rejected(node, "does not match schema type Tensor");
    node = unary_node("torch.ops.aten.mul.Tensor");
    NamedArgument scalar;
    scalar.name = "other";
    scalar.argument.type = Argument::Integer;
    scalar.argument.integer = 2;
    node.inputs.push_back(scalar);
    std::string scalar_error;
    expect_true(normalize_exported_program_node(node, scalar_error), scalar_error);
    expect_true(node.target == "torch.ops.aten.mul.Scalar", "serde scalar argument selects the dispatcher Scalar overload");
    node = unary_node("torch.ops.aten.clone.default");
    NamedArgument wrong;
    wrong.name = "memory_format";
    wrong.argument.type = Argument::String;
    node.inputs.push_back(wrong);
    expect_node_rejected(node, "memory_format");

    const char* pure[] = {"torch.ops.aten.relu.default", "torch.ops.aten.alias.default", "torch.ops.aten.clone.default"};
    for (size_t i = 0; i < 3; i++)
    {
        ExportedProgramArchive archive = make_archive();
        archive.program.graph.nodes.assign(1, unary_node(pure[i]));
        pnnx::Graph graph;
        expect_true(pnnx::import_exported_program_inputs(archive, graph, error) == 0, error);
        expect_true(pnnx::import_exported_program_nodes(archive.program, graph, error) == 0, "ordinary pure operators and non-writing aliases remain supported");
    }

    ExportedProgramArchive local_write = make_archive();
    local_write.program.graph.nodes.assign(1, unary_node("torch.ops.aten.clone.default"));
    Node inplace = unary_node("torch.ops.aten.relu_.default");
    inplace.name = "local_write";
    inplace.inputs[0].argument.name = "contract_output";
    inplace.outputs[0].name = "local_write_output";
    pnnx::Graph local_graph;
    expect_true(pnnx::import_exported_program_inputs(local_write, local_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(local_write.program, local_graph, error) == 0, error);
    local_write.program.graph.nodes.assign(1, inplace);
    const size_t local_op_count = local_graph.ops.size();
    expect_true(pnnx::import_exported_program_nodes(local_write.program, local_graph, error) != 0 && error.find("unsupported alias write/mutation of argument self") != std::string::npos, "a preexisting IR target does not prove whole-program ownership for relu_");
    expect_true(local_graph.ops.size() == local_op_count, "rejected local write does not alter the existing graph");

    // A real factory schema covers SymInt[], Scalar, optional enum and Device
    // arguments, including dispatcher defaults interleaved with explicit kwargs.
    ExportedProgram factory = make_program();
    node = unary_node("torch.ops.aten.full.default");
    node.inputs.clear();
    NamedArgument size;
    size.name = "size";
    size.argument.type = Argument::SymIntegers;
    Argument extent;
    extent.type = Argument::SymInteger;
    extent.integer = 2;
    size.argument.values.push_back(extent);
    node.inputs.push_back(size);
    NamedArgument fill;
    fill.name = "fill_value";
    fill.argument.type = Argument::SymFloat;
    fill.argument.floating_point = 1.5;
    node.inputs.push_back(fill);
    NamedArgument dtype;
    dtype.name = "dtype";
    dtype.argument.type = Argument::ScalarType;
    dtype.argument.integer = 13;
    node.inputs.push_back(dtype);
    NamedArgument device;
    device.name = "device";
    device.argument.type = Argument::DeviceValue;
    device.argument.device.type = "cpu";
    node.inputs.push_back(device);
    factory.graph.nodes.push_back(node);
    expect_true(append_default_arguments(factory, error), error);
    pnnx::Graph factory_graph;
    expect_true(pnnx::import_exported_program_nodes(factory, factory_graph, error) == 0, error);
    const pnnx::Operand* dtype_operand = factory_graph.get_operand("contract_node_arg_2");
    expect_true(dtype_operand && dtype_operand->producer->params["value"].i == 15, "BF16 survives real factory schema and repeated default normalization");

    ExportedProgramArchive indexed = make_archive();
    node = unary_node("torch.ops.aten.index.Tensor");
    NamedArgument indices;
    indices.name = "indices";
    indices.argument.type = Argument::OptionalTensors;
    Argument none;
    none.type = Argument::None;
    indices.argument.values.push_back(none);
    indices.argument.values.push_back(node.inputs[0].argument);
    node.inputs.push_back(indices);
    indexed.program.graph.nodes.assign(1, node);
    pnnx::Graph index_graph;
    expect_true(pnnx::import_exported_program_inputs(indexed, index_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(indexed.program, index_graph, error) == 0, "Tensor?[] accepts both None and tensor references");
    indexed.program.graph.nodes[0].inputs[1].argument.values[0].type = Argument::Integer;
    expect_node_rejected(indexed.program.graph.nodes[0], "does not match schema type");
}

static void expect_program_rejected(const pnnx::pt2::ExportedProgram& program, const char* diagnostic)
{
    pnnx::pt2::ExportedProgram normalized = program;
    std::string error;
    expect_true(!pnnx::pt2::append_default_arguments(normalized, error), "whole-program normalizer rejects unsafe program");
    expect_true(error.find(diagnostic) != std::string::npos, error);
    expect_true(normalized.graph.nodes.size() == program.graph.nodes.size(), "failed normalization preserves node count");
    for (size_t i = 0; i < program.graph.nodes.size() && i < normalized.graph.nodes.size(); i++)
        expect_true(normalized.graph.nodes[i].target == program.graph.nodes[i].target && normalized.graph.nodes[i].inputs.size() == program.graph.nodes[i].inputs.size(), "failed normalization does not commit partial rewrites/defaults");
    pnnx::Graph graph;
    expect_true(pnnx::import_exported_program_nodes(program, graph, error) != 0, "public importer enforces whole-program checks without archive reader");
    expect_true(error.find(diagnostic) != std::string::npos, error);
    expect_true(graph.ops.empty() && graph.operands.empty(), "unsafe program rejected before IR construction");
}

static pnnx::pt2::NamedArgument integer_list(const char* name, int64_t first, int64_t second)
{
    using namespace pnnx::pt2;
    NamedArgument result;
    result.name = name;
    result.argument.type = Argument::SymIntegers;
    Argument item;
    item.type = Argument::SymInteger;
    item.integer = first;
    result.argument.values.push_back(item);
    item.integer = second;
    result.argument.values.push_back(item);
    return result;
}

static pnnx::pt2::ExportedProgram factory_dtype_program(bool like, int scalar_type, bool none_dtype)
{
    using namespace pnnx::pt2;
    ExportedProgram program = make_program();
    Node node = unary_node(like ? "torch.ops.aten.full_like.default" : "torch.ops.aten.full.default");
    node.name = "factory";
    node.outputs[0].name = "factory_output";
    const NamedArgument self = node.inputs[0];
    node.inputs.clear();
    // Deliberately not schema order: an inserted dtype must still precede
    // layout/device, without changing the explicit fill or device argument.
    NamedArgument fill;
    fill.name = "fill_value";
    fill.argument.type = Argument::FloatingPoint;
    fill.argument.floating_point = 3.0;
    node.inputs.push_back(fill);
    NamedArgument device;
    device.name = "device";
    device.kind = NamedArgument::Keyword;
    device.argument.type = Argument::DeviceValue;
    device.argument.device.type = "cpu";
    node.inputs.push_back(device);
    node.inputs.push_back(like ? self : integer_list("size", 2, 3));
    if (none_dtype)
    {
        NamedArgument dtype;
        dtype.name = "dtype";
        dtype.kind = NamedArgument::Keyword;
        dtype.argument.type = Argument::None;
        node.inputs.push_back(dtype);
    }
    TensorMeta meta;
    meta.scalar_type = scalar_type;
    meta.sizes.push_back(dimension(2));
    meta.sizes.push_back(dimension(3));
    program.graph.tensor_values["factory_output"] = meta;
    if (like)
    {
        program.graph.inputs.push_back(self.argument);
        // The result, not self or the scalar's serialized variant, is the
        // authority for a missing dtype. Keep self deliberately different.
        meta.scalar_type = 7;
        program.graph.tensor_values["x"] = meta;
    }
    program.graph.nodes.push_back(node);
    return program;
}

static void expect_factory_dtype(const pnnx::pt2::ExportedProgram& program, int dtype, int output_type, bool scalar = false)
{
    const bool like = program.graph.nodes[0].target == "torch.ops.aten.full_like.default";
    pnnx::Graph graph;
    if (like)
    {
        pnnx::Operand* input = graph.new_operand("x");
        input->type = 1;
        input->shape = std::vector<int> {2, 3};
    }
    std::string error;
    const int result = pnnx::import_exported_program_nodes(program, graph, error);
    expect_true(result == 0, "factory dtype import: " + error);
    if (result != 0 || graph.ops.empty()) return;
    const pnnx::Operator* op = graph.ops.back();
    expect_true(op->type == (scalar ? "aten::scalar_tensor" : like ? "aten::full_like" : "torch.ops.aten.full.default"), "factory target and rank-zero full rewrite are preserved");
    const char* arguments[] = {like ? "self" : "size", "fill_value", "dtype", "layout", "device", "pin_memory", "memory_format"};
    const size_t offset = scalar ? 1 : 0;
    const size_t count = like ? 7 : 6 - offset;
    expect_true(op->inputs.size() == count && op->inputnames.size() == count, "factory defaults retain schema arity");
    if (op->inputs.size() != count || op->inputnames.size() != count) return;
    for (size_t i = 0; i < count; i++)
        expect_true(op->inputnames[i] == arguments[i + offset], "factory arguments retain schema order");
    const pnnx::Parameter& fill = op->inputs[1 - offset]->producer->params.at("value");
    const pnnx::Parameter& actual_dtype = op->inputs[2 - offset]->producer->params.at("value");
    expect_true(fill.type == 3 && fill.f == 3.f, "integer-valued as_float stays floating point even for int64 output");
    expect_true(dtype < 0 ? actual_dtype.type == 0 : actual_dtype.type == 2 && actual_dtype.i == dtype, "factory dtype uses the existing serde-to-c10 mapping, or retains None without metadata");
    expect_true(op->inputs[3 - offset]->producer->params.at("value").type == 0, "omitted layout remains None");
    expect_true(op->inputs[4 - offset]->producer->params.at("value").s == "cpu", "explicit device stays in its schema position");
    expect_true(op->outputs.size() == 1 && op->outputs[0]->type == output_type, "output dtype metadata is not rewritten");
    if (scalar && op->outputs.size() == 1)
        expect_true(op->outputs[0]->shape.empty() && !graph.get_operand("factory_arg_0"), "zero-dimensional full still omits the size input");
    std::set<std::string> operators;
    std::set<std::string> operands;
    for (size_t i = 0; i < graph.ops.size(); i++)
        expect_true(operators.insert(graph.ops[i]->name).second, "restored factory dtype does not duplicate operator names");
    for (size_t i = 0; i < graph.operands.size(); i++)
        expect_true(operands.insert(graph.operands[i]->name).second, "restored factory dtype does not duplicate operand names");
}

static void test_factory_dtype_contract()
{
    using namespace pnnx::pt2;
    const int c10_types[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15};
    const int pnnx_types[] = {8, 7, 6, 4, 5, 3, 1, 2, 12, 10, 11, 9, 13};
    for (int like = 0; like < 2; like++)
    {
        for (int mode = 0; mode < 3; mode++)
        {
            for (int scalar_type = 1; scalar_type <= 13; scalar_type++)
            {
                ExportedProgram program = factory_dtype_program(like != 0, scalar_type, mode == 1);
                std::string error;
                if (mode == 2)
                    expect_true(append_default_arguments(program, error), "already-normalized factory: " + error);
                const Node original = program.graph.nodes[0];
                expect_factory_dtype(program, c10_types[scalar_type - 1], pnnx_types[scalar_type - 1]);
                const Node& unchanged = program.graph.nodes[0];
                expect_true(unchanged.inputs.size() == original.inputs.size() && unchanged.target == original.target, "factory restoration only mutates the importer's private copy");
                for (size_t i = 0; i < original.inputs.size() && i < unchanged.inputs.size(); i++)
                    expect_true(unchanged.inputs[i].name == original.inputs[i].name && unchanged.inputs[i].kind == original.inputs[i].kind
                                && unchanged.inputs[i].argument.type == original.inputs[i].argument.type
                                && unchanged.inputs[i].argument.integer == original.inputs[i].argument.integer
                                && unchanged.inputs[i].argument.floating_point == original.inputs[i].argument.floating_point, "caller arguments retain their original order, values and dtype variant");
            }
        }
        for (int none_dtype = 0; none_dtype < 2; none_dtype++)
        {
            ExportedProgram program = factory_dtype_program(like != 0, 5, none_dtype != 0);
            program.graph.tensor_values.erase("factory_output");
            expect_factory_dtype(program, -1, 0);
        }
        // A supplied dtype is not overwritten even if result metadata differs.
        // as_int is already a c10 enum; it must not be decoded as serde.
        ExportedProgram program = factory_dtype_program(like != 0, 5, true);
        Argument& dtype = program.graph.nodes[0].inputs.back().argument;
        dtype.type = Argument::ScalarType;
        dtype.integer = 13;
        expect_factory_dtype(program, 15, 5);
        dtype.type = Argument::Integer;
        dtype.integer = 7;
        expect_factory_dtype(program, 7, 5);

        dtype.type = Argument::ScalarType;
        dtype.integer = 999;
        pnnx::Graph graph;
        if (like) graph.new_operand("x");
        std::string error;
        expect_true(pnnx::import_exported_program_nodes(program, graph, error) == -1
                    && error.find("unsupported serde scalar type 999") != std::string::npos, "unsupported explicit dtype is still validated, never replaced by metadata");
    }
    const int scalar_types[] = {5, 8};
    for (size_t i = 0; i < 2; i++)
    {
        ExportedProgram program = factory_dtype_program(false, scalar_types[i], false);
        program.graph.nodes[0].inputs[2].argument.values.clear();
        program.graph.tensor_values["factory_output"].sizes.clear();
        expect_factory_dtype(program, c10_types[scalar_types[i] - 1], pnnx_types[scalar_types[i] - 1], true);
    }
}

static void test_factory_dtype_validation()
{
    using namespace pnnx::pt2;
    const int unsupported[] = {-1, 0, 14, 29, 999};
    for (int like = 0; like < 2; like++)
    {
        for (int none_dtype = 0; none_dtype < 2; none_dtype++)
        {
            for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++)
            {
                ExportedProgram program = factory_dtype_program(like != 0, unsupported[i], none_dtype != 0);
                pnnx::Graph graph;
                std::string error;
                expect_true(pnnx::import_exported_program_nodes(program, graph, error) == -1
                            && error.find(".dtype: unsupported serde scalar type " + std::to_string(unsupported[i])) != std::string::npos
                            && error.find("output tensor metadata for factory_output") != std::string::npos, "unsupported inferred dtype is rejected with output metadata context");
                expect_true(graph.ops.empty() && graph.operands.empty(), "unsupported factory metadata fails before IR construction");
            }
        }
        ExportedProgram program = factory_dtype_program(like != 0, 5, true);
        program.graph.nodes[0].inputs.push_back(program.graph.nodes[0].inputs.back());
        expect_program_rejected(program, "duplicate argument dtype");
        program = factory_dtype_program(like != 0, 5, true);
        program.graph.nodes[0].inputs.back().argument.name = "hidden_reference";
        expect_program_rejected(program, "reference name hidden_reference is not valid");
        program = factory_dtype_program(like != 0, 5, true);
        Argument hidden;
        hidden.type = Argument::Tensor;
        hidden.name = "hidden_reference";
        program.graph.nodes[0].inputs.back().argument.values.push_back(hidden);
        expect_program_rejected(program, "non-container argument contains nested values/references");
        program = factory_dtype_program(like != 0, 5, false);
        program.graph.nodes[0].outputs.clear();
        expect_program_rejected(program, "output count");
        program = factory_dtype_program(like != 0, 5, false);
        program.graph.nodes[0].outputs[0].type = Argument::SymInteger;
        expect_program_rejected(program, "does not match schema return type");
    }
}

static void test_factory_dtype_name_collisions()
{
    using namespace pnnx::pt2;
    for (int none_dtype = 0; none_dtype < 2; none_dtype++)
    {
        for (int kind = 0; kind < 4; kind++)
        {
            ExportedProgram program = factory_dtype_program(false, 5, none_dtype != 0);
            pnnx::Graph graph;
            pnnx::Operand* existing = 0;
            if (kind == 0) existing = graph.new_operand("factory_arg_2");
            if (kind == 1) program.graph.tensor_values["factory_arg_2"] = program.graph.tensor_values.at("factory_output");
            if (kind >= 2)
            {
                Node later = unary_node("torch.ops.aten.relu.default");
                later.inputs[0].argument.name = "factory_output";
                if (kind == 2) later.outputs[0].name = "factory_arg_2";
                if (kind == 3) later.name = "factory_arg_2";
                program.graph.nodes.push_back(later);
            }
            std::string error;
            const std::string diagnostic = kind == 3
                                              ? "factory.dtype: operator name collision: 'factory_arg_2' conflicts with 'factory_arg_2' after identifier sanitization to 'factory_arg_2'"
                                              : "factory.dtype: generated value name collision: 'factory_arg_2' is already defined or referenced";
            expect_true(pnnx::import_exported_program_nodes(program, graph, error) == -1 && error == diagnostic, error);
            expect_true(graph.get_operand("factory_arg_2") == existing, "inferred dtype does not shadow existing/future values");
            for (size_t i = 0; i < graph.ops.size(); i++)
                expect_true(graph.ops[i]->name != "factory_arg_2", "inferred dtype does not bypass reserved operator names");
        }
    }
}

static pnnx::pt2::Node metadata_guard_node()
{
    using namespace pnnx::pt2;
    Node node = unary_node("torch.ops.aten._assert_tensor_metadata.default");
    node.inputs[0].name = "a";
    node.outputs[0] = Argument();
    node.outputs[0].type = Argument::None;
    NamedArgument dtype;
    dtype.name = "dtype";
    dtype.argument.type = Argument::ScalarType;
    dtype.argument.integer = 7;
    node.inputs.push_back(dtype);
    return node;
}

static pnnx::pt2::ExportedProgramArchive metadata_guard_archive()
{
    pnnx::pt2::ExportedProgramArchive archive = make_archive();
    archive.program.graph.tensor_values["x"].device.type = "cpu";
    archive.program.graph.tensor_values["x"].layout = 7;
    archive.program.graph.tensor_values["x"].strides = {dimension(3), dimension(1)};
    archive.program.graph.nodes.assign(1, metadata_guard_node());
    return archive;
}

static void test_metadata_guards()
{
    using namespace pnnx::pt2;
    std::string error;
    ExportedProgramArchive archive = metadata_guard_archive();
    const ExportedProgram baseline = archive.program;
    Node node = metadata_guard_node();
    expect_true(!normalize_exported_program_node(node, error) && error.find("requires graph metadata context") != std::string::npos, "standalone metadata guard is not a blanket pure-op exception");
    expect_true(normalize_exported_program_node(node, error, &baseline.graph), error);
    expect_true(node.inputs.size() >= 4 && node.inputs[1].name == "size" && node.inputs[1].argument.type == Argument::None
                && node.inputs[2].name == "stride" && node.inputs[2].argument.type == Argument::None
                && node.inputs[3].name == "dtype" && node.inputs[3].argument.type == Argument::ScalarType, "omitted metadata checks use dispatcher None defaults without reinterpreting serde dtype");
    expect_true(normalize_exported_program_node(node, error, &baseline.graph), "metadata normalization is idempotent");
    pnnx::Graph graph;
    expect_true(pnnx::import_exported_program_inputs(archive, graph, error) == 0, error);
    const size_t before = graph.ops.size();
    expect_true(pnnx::import_exported_program_nodes(baseline, graph, error) == 0 && graph.ops.size() == before, "proven metadata guard with [None] creates no IR operator/constant");
    ExportedProgram program = baseline;
    program.graph.nodes[0].outputs.clear();
    expect_true(pnnx::import_exported_program_nodes(program, graph, error) == 0 && graph.ops.size() == before, "metadata guard with [] creates no IR values");
    program.graph.nodes[0].inputs.resize(1);
    expect_true(append_default_arguments(program, error), "all omitted optional fields are no checks, not inferred checks");
    expect_true(pnnx::import_exported_program_nodes(program, graph, error) == 0 && graph.ops.size() == before, "default-only guard still resolves its tensor");

    program = baseline;
    Node& full = program.graph.nodes[0];
    full.inputs.push_back(integer_list("stride", 3, 1));
    full.inputs.push_back(integer_list("size", 2, 3));
    NamedArgument device;
    device.name = "device";
    device.argument.type = Argument::DeviceValue;
    device.argument.device.type = "cpu";
    full.inputs.push_back(device);
    NamedArgument layout;
    layout.name = "layout";
    layout.argument.type = Argument::Layout;
    layout.argument.integer = 7;
    full.inputs.push_back(layout);
    const ExportedProgram all_fields = program;
    expect_true(append_default_arguments(program, error), error);
    expect_true(pnnx::import_exported_program_nodes(program, graph, error) == 0 && graph.ops.size() == before, "all five concrete metadata checks agree, despite serialized argument order");
    // Test every predicate independently; no guard may be discarded just
    // because a different predicate agrees with the recorded tensor metadata.
    for (size_t i = 1; i < all_fields.graph.nodes[0].inputs.size(); i++)
    {
        program = all_fields;
        Argument& value = program.graph.nodes[0].inputs[i].argument;
        if (i == 1) value.integer = 8; // dtype
        if (i == 2 || i == 3) value.values[0].integer++;
        if (i == 4) value.device.type = "cuda";
        if (i == 5) value.integer = 1; // sparse layout
        expect_program_rejected(program, "metadata guard is false or unsupported");
    }
    program = all_fields;
    program.graph.nodes[0].inputs[1].argument.type = Argument::Integer;
    program.graph.nodes[0].inputs[1].argument.integer = 6; // c10 Float
    program.graph.nodes[0].inputs[5].argument.type = Argument::Integer;
    program.graph.nodes[0].inputs[5].argument.integer = 0; // c10 Strided
    expect_true(append_default_arguments(program, error), "c10 integer enum arguments are not serde enums");
    program.graph.nodes[0].inputs[3].argument.integer = 7; // ordered dtype, c10 Double
    expect_program_rejected(program, "metadata guard is false or unsupported");
    program = baseline;
    program.graph.nodes[0].inputs[1].argument.integer = 0;
    expect_program_rejected(program, "metadata guard is false or unsupported");
    program.graph.nodes[0].inputs[1].argument.integer = std::numeric_limits<int64_t>::max();
    expect_program_rejected(program, "metadata guard is false or unsupported");

    // Known intermediate dtypes include BF16; only direct guarded external
    // inputs are float32-only. This is enum validation, not native dtype support.
    Graph metadata;
    metadata.tensor_values["x"] = baseline.graph.tensor_values.at("x");
    metadata.tensor_values["x"].scalar_type = 13;
    node = metadata_guard_node();
    node.inputs[1].argument.integer = 13;
    expect_true(normalize_exported_program_node(node, error, &metadata), "serde BF16 metadata matches serde dtype 13");
    node = metadata_guard_node();
    node.inputs[1].argument.type = Argument::Integer;
    node.inputs[1].argument.integer = 15;
    expect_true(normalize_exported_program_node(node, error, &metadata), "c10 BF16 dtype is 15");
    node = metadata_guard_node();
    node.inputs[1].argument.type = Argument::Integer;
    node.inputs[1].argument.integer = 6;
    expect_true(!normalize_exported_program_node(node, error, &metadata), "BF16 metadata does not match c10 Float");
    program = baseline;
    program.graph.tensor_values["x"].scalar_type = 13;
    program.graph.nodes[0].inputs[1].argument.integer = 13;
    expect_program_rejected(program, "guarded graph inputs require static float32");

    // Like dtype queries on factory/to results, intermediate guards can check
    // a known non-float dtype without broadening the external input contract.
    const int intermediate_dtypes[] = {5, 13}; // serde Long and BFloat16
    for (size_t i = 0; i < 2; i++)
    {
        program = baseline;
        Node converted = unary_node("torch.ops.aten._to_copy.default");
        converted.name = "converted";
        converted.outputs[0].name = "converted";
        NamedArgument dtype = metadata_guard_node().inputs[1];
        dtype.argument.integer = intermediate_dtypes[i];
        converted.inputs.push_back(dtype);
        program.graph.nodes[0].inputs[0].argument.name = "converted";
        program.graph.nodes[0].inputs[1] = dtype;
        program.graph.nodes.insert(program.graph.nodes.begin(), converted);
        program.graph.tensor_values["converted"] = program.graph.tensor_values["x"];
        program.graph.tensor_values["converted"].scalar_type = intermediate_dtypes[i];
        pnnx::Graph intermediate;
        expect_true(pnnx::import_exported_program_inputs(archive, intermediate, error) == 0, error);
        expect_true(pnnx::import_exported_program_nodes(program, intermediate, error) == 0, error);
        expect_true(!intermediate.ops.empty() && intermediate.ops.back()->type == "aten::_to_copy", "known intermediate dtype guard is checked and creates no operator");
    }

    program = all_fields;
    program.graph.nodes[0].inputs[2].argument.values[0].name = "runtime_stride";
    expect_program_rejected(program, "runtime-dependent metadata guard argument");
    program = all_fields;
    program.graph.nodes[0].inputs[3].argument.values[0].name = "runtime_size";
    expect_program_rejected(program, "runtime-dependent metadata guard argument");
    program = all_fields;
    program.graph.nodes[0].inputs[3].argument.values.pop_back();
    expect_program_rejected(program, "metadata guard is false or unsupported");
    program = all_fields;
    program.graph.nodes[0].inputs[4].argument.device.has_index = true;
    program.graph.nodes[0].inputs[4].argument.device.index = 0;
    expect_program_rejected(program, "metadata guard is false or unsupported");
    for (int i = 0; i < 6; i++)
    {
        program = baseline;
        TensorMeta& meta = program.graph.tensor_values["x"];
        if (i == 0 || i == 1)
        {
            SymInt& value = i == 0 ? meta.sizes[0] : meta.strides[0];
            value.type = SymInt::Expression;
            value.expression = "s0";
            value.has_hint = true;
            value.hint = value.integer;
        }
        if (i == 2) meta.strides.pop_back();
        if (i == 3) meta.layout = 0;
        if (i == 4) meta.device.type = "cuda";
        if (i == 5) meta.scalar_type = 29;
        expect_program_rejected(program, "metadata guard requires known dtype, CPU/Strided and static sizes/strides");
    }
    program = baseline;
    program.graph.tensor_values.erase("x");
    expect_program_rejected(program, "tensor metadata is missing for x");
    program = baseline;
    program.graph.inputs.clear(); // Metadata alone is not a tensor definition.
    expect_program_rejected(program, "input x is not defined before metadata guard");
    Node late = unary_node("torch.ops.aten.clone.default");
    late.inputs[0].argument.name = "other";
    late.outputs[0].name = "x";
    program.graph.nodes.push_back(late);
    expect_program_rejected(program, "input x is not defined before metadata guard");
    pnnx::Graph absent_input;
    expect_true(pnnx::import_exported_program_nodes(baseline, absent_input, error) != 0 && error.find("input x is not defined") != std::string::npos
                && absent_input.ops.empty(), "declared input and metadata do not bypass IR reference lookup");
    for (int i = 0; i < 2; i++)
    {
        pnnx::Graph inconsistent;
        expect_true(pnnx::import_exported_program_inputs(archive, inconsistent, error) == 0, error);
        pnnx::Operand* input = inconsistent.get_operand("x");
        if (input)
        {
            if (i == 0) input->type = 5;
            else input->shape.assign(1, 6);
        }
        const size_t old_count = inconsistent.ops.size();
        expect_true(pnnx::import_exported_program_nodes(baseline, inconsistent, error) != 0 && error.find("imported tensor dtype/shape disagrees") != std::string::npos
                    && inconsistent.ops.size() == old_count, "public importer cross-checks recorded guard metadata with the actual imported operand");
    }
    program = baseline;
    program.graph.nodes[0].outputs.assign(1, unary_node("torch.ops.aten.relu.default").outputs[0]);
    expect_program_rejected(program, "output count 1 does not match schema return count 0");
    program.graph.nodes[0].outputs.assign(2, baseline.graph.nodes[0].outputs[0]);
    expect_program_rejected(program, "output count 2 does not match schema return count 0");
    program = all_fields;
    program.graph.tensor_values["x"].sizes.clear();
    program.graph.tensor_values["x"].strides.clear();
    program.graph.nodes[0].inputs[2].argument.values.clear();
    program.graph.nodes[0].inputs[3].argument.values.clear();
    expect_true(append_default_arguments(program, error), "empty size/stride lists mean rank zero, not omitted checks");
}

static pnnx::pt2::ExportedProgramArchive local_fill_archive(bool tensor_value)
{
    using namespace pnnx::pt2;
    ExportedProgramArchive archive = metadata_guard_archive();
    archive.program.graph.nodes.clear();
    Node slice = unary_node("torch.ops.aten.slice.Tensor");
    slice.name = "slice";
    slice.outputs[0].name = "slice";
    NamedArgument end;
    end.name = "end";
    end.argument.type = Argument::Integer;
    end.argument.integer = 1;
    slice.inputs.push_back(end);
    archive.program.graph.nodes.push_back(slice);
    Node clone = unary_node("torch.ops.aten.clone.default");
    clone.name = "clone";
    clone.inputs[0].argument.name = "slice";
    clone.outputs[0].name = "local";
    archive.program.graph.nodes.push_back(clone);
    Node fill = unary_node(tensor_value ? "torch.ops.aten.fill_.Tensor" : "torch.ops.aten.fill_.Scalar");
    fill.name = "fill";
    fill.inputs[0].argument.name = "local";
    fill.outputs[0].name = "filled";
    NamedArgument value;
    value.name = "value";
    value.argument.type = tensor_value ? Argument::Tensor : Argument::Integer;
    if (tensor_value)
    {
        value.argument.name = "fill_value";
        InputSpec spec;
        spec.type = InputSpec::UserInput;
        spec.argument = value.argument;
        archive.program.graph.inputs.push_back(value.argument);
        archive.program.signature.inputs.push_back(spec);
        TensorMeta scalar = archive.program.graph.tensor_values["x"];
        scalar.sizes.clear();
        scalar.strides.clear();
        archive.program.graph.tensor_values["fill_value"] = scalar;
    }
    else
        value.argument.integer = 7;
    fill.inputs.push_back(value);
    archive.program.graph.nodes.push_back(fill);
    TensorMeta result = archive.program.graph.tensor_values["x"];
    result.sizes[0] = dimension(1);
    archive.program.graph.tensor_values["slice"] = result;
    archive.program.graph.tensor_values["local"] = result;
    archive.program.graph.tensor_values["filled"] = result;
    archive.program.graph.outputs.assign(1, fill.outputs[0]);
    OutputSpec output;
    output.type = OutputSpec::UserOutput;
    output.argument = fill.outputs[0];
    archive.program.signature.outputs.assign(1, output);
    return archive;
}

static void test_local_fill_contract()
{
    using namespace pnnx::pt2;
    std::string error;
    for (int tensor_value = 0; tensor_value < 2; tensor_value++)
    {
        ExportedProgramArchive archive = local_fill_archive(tensor_value != 0);
        Node standalone = archive.program.graph.nodes[2];
        expect_node_rejected(standalone, "alias write/mutation of argument self");
        expect_true(!normalize_exported_program_node(standalone, error, &archive.program.graph), "metadata context alone never authorizes writes");
        pnnx::Graph graph;
        expect_true(pnnx::import_exported_program_inputs(archive, graph, error) == 0, error);
        expect_true(pnnx::import_exported_program_nodes(archive.program, graph, error) == 0, error);
        expect_true(pnnx::import_exported_program_outputs(archive.program, graph, error) == 0, error);
        const pnnx::Operand* filled = graph.get_operand("filled");
        const pnnx::Operand* local = graph.get_operand("local");
        expect_true(filled && local && filled->producer->type == "aten::fill" && filled->producer->inputs[0] == local
                    && local->consumers.size() == 1 && filled->consumers.size() == 1 && filled->consumers[0]->type == "pnnx.Output", "single-use clone fill imports out-of-place and consumers use the mutation result");
        expect_true(graph.get_operand("x") && graph.get_operand("x")->producer->type == "pnnx.Input", "input is not replaced by the local fill result");
        ExportedProgram normalized = archive.program;
        expect_true(append_default_arguments(normalized, error), error);
        expect_true(normalized.graph.nodes[2].target == (tensor_value ? "torch.ops.aten.fill.Tensor" : "torch.ops.aten.fill.Scalar"), "only proven fill_ target is rewritten");
        expect_true(append_default_arguments(normalized, error), "local fill normalization is idempotent");
        pnnx::Graph repeated;
        expect_true(pnnx::import_exported_program_inputs(archive, repeated, error) == 0 && pnnx::import_exported_program_nodes(normalized, repeated, error) == 0, "public importer also accepts the already normalized program");
    }
    const ExportedProgram baseline = local_fill_archive(false).program;
    ExportedProgram program;
    const char* external_targets[] = {"x", "p_weight", "slice", "missing"};
    for (size_t i = 0; i < 4; i++)
    {
        program = baseline;
        program.graph.nodes[2].inputs[0].argument.name = external_targets[i];
        expect_program_rejected(program, "single-use unaliased local target");
    }
    // Schema alias sets, not an operator-name freshness list, propagate views.
    const char* aliases[] = {"torch.ops.aten.alias.default", "torch.ops.aten.detach.default", "torch.ops.aten.t.default"};
    for (size_t i = 0; i < 3; i++)
    {
        program = baseline;
        program.graph.nodes[1].target = aliases[i];
        expect_program_rejected(program, "single-use unaliased local target");
    }
    program = baseline;
    program.graph.nodes[1].target = "torch.ops.aten._unsafe_view.default";
    program.graph.nodes[1].inputs.push_back(integer_list("size", 1, 3));
    expect_program_rejected(program, "allocated by a known allocator");
    program = baseline;
    program.graph.nodes[1].target = "torch.ops.aten.neg.default";
    expect_program_rejected(program, "allocated by a known allocator");
    program = baseline;
    Node split = unary_node("torch.ops.aten.unbind.int");
    split.name = "split_alias";
    Argument item = split.outputs[0];
    item.name = "local";
    split.outputs[0] = Argument();
    split.outputs[0].type = Argument::Tensors;
    split.outputs[0].values.push_back(item);
    item.name = "other_alias";
    split.outputs[0].values.push_back(item);
    program.graph.nodes[1] = split;
    expect_program_rejected(program, "single-use unaliased local target");
    program = baseline;
    program.graph.nodes[1].target = "_operator.getitem";
    NamedArgument index;
    index.name = "index";
    index.argument.type = Argument::Integer;
    program.graph.nodes[1].inputs.push_back(index);
    expect_program_rejected(program, "single-use unaliased local target");

    // Every graph input tensor is external, even with no signature entry and
    // when recursively nested inside optional/list serialized arguments.
    program = baseline;
    program.graph.nodes.assign(1, baseline.graph.nodes[2]);
    program.graph.inputs.clear();
    program.signature.inputs.clear();
    Argument optional;
    optional.type = Argument::OptionalTensor;
    optional.values.push_back(program.graph.nodes[0].inputs[0].argument);
    Argument nested;
    nested.type = Argument::OptionalTensors;
    nested.values.push_back(optional);
    program.graph.inputs.push_back(nested);
    expect_program_rejected(program, "single-use unaliased local target");
    program.graph.inputs.clear();
    program.graph.nodes[0].metadata["local"] = "true";
    expect_program_rejected(program, "single-use unaliased local target");
    pnnx::Graph preexisting;
    pnnx::Operator* fake_clone = preexisting.new_operator("aten::clone", "not_in_program");
    pnnx::Operand* fake_local = preexisting.new_operand("local");
    fake_local->producer = fake_clone;
    fake_clone->outputs.push_back(fake_local);
    expect_true(pnnx::import_exported_program_nodes(program, preexisting, error) != 0 && error.find("single-use unaliased local target") != std::string::npos
                && preexisting.ops.size() == 1 && preexisting.operands.size() == 1, "an existing IR clone operand cannot forge whole-program write authorization");

    for (int before = 0; before < 2; before++)
    {
        program = baseline;
        Node use = unary_node("torch.ops.aten.relu.default");
        use.name = "use_old_local";
        use.inputs[0].argument.name = "local";
        program.graph.nodes.insert(program.graph.nodes.begin() + (before ? 2 : 3), use);
        expect_program_rejected(program, "single-use unaliased local target");
    }
    program = baseline;
    program.graph.outputs.push_back(program.graph.nodes[2].inputs[0].argument);
    expect_program_rejected(program, "single-use unaliased local target");
    program = baseline;
    program.signature.outputs[0].argument = program.graph.nodes[2].inputs[0].argument;
    expect_program_rejected(program, "single-use unaliased local target");
    program = baseline;
    Node view = unary_node("torch.ops.aten.alias.default");
    view.name = "local_alias";
    view.inputs[0].argument.name = "local";
    view.outputs[0].name = "alias_local";
    program.graph.nodes.insert(program.graph.nodes.begin() + 2, view);
    program.graph.nodes[3].inputs[0].argument.name = "alias_local";
    expect_program_rejected(program, "single-use unaliased local target");
    program = baseline;
    Node cat = unary_node("torch.ops.aten.cat.default");
    cat.inputs[0].name = "tensors";
    cat.inputs[0].argument = Argument();
    cat.inputs[0].argument.type = Argument::Tensors;
    cat.inputs[0].argument.values.push_back(program.graph.nodes[2].inputs[0].argument);
    program.graph.nodes.push_back(cat);
    expect_program_rejected(program, "single-use unaliased local target");
    program = local_fill_archive(true).program;
    program.graph.nodes[2].inputs[1].argument.name = "local";
    expect_program_rejected(program, "single-use unaliased local target");
    program = baseline;
    program.graph.nodes[2].outputs.clear();
    expect_program_rejected(program, "output count 0 does not match schema return count 1");
    program = baseline;
    program.graph.nodes[2].outputs[0] = Argument();
    program.graph.nodes[2].outputs[0].type = Argument::None;
    expect_program_rejected(program, "does not match schema return type Tensor");
    program = baseline;
    program.graph.nodes[2].outputs[0].name = "local";
    expect_program_rejected(program, "already defined");
    program = baseline;
    program.graph.nodes[2].target = "torch.ops.aten.copy_.default";
    program.graph.nodes[2].inputs[1].name = "src";
    program.graph.nodes[2].inputs[1].argument = program.graph.inputs[1];
    expect_program_rejected(program, "alias write/mutation of argument self");
}

static pnnx::pt2::ExportedProgramArchive local_pointwise_archive(const char* target)
{
    using namespace pnnx::pt2;
    ExportedProgramArchive archive = metadata_guard_archive();
    Node add = unary_node("torch.ops.aten.add.Tensor");
    add.name = "allocate";
    add.outputs[0].name = "local";
    NamedArgument other;
    other.name = "other";
    other.argument.type = Argument::Integer;
    other.argument.integer = 3;
    // Serde Tensor + scalar and non-schema input order must normalize first.
    add.inputs.insert(add.inputs.begin(), other);
    archive.program.graph.nodes.assign(1, add);
    Node inplace = unary_node(target);
    inplace.name = "pointwise";
    inplace.inputs[0].argument.name = "local";
    inplace.outputs[0].name = "updated";
    inplace.metadata["contract"] = "preserve";
    NamedArgument minimum;
    minimum.argument.type = Argument::Integer;
    minimum.argument.integer = 0;
    NamedArgument maximum = minimum;
    maximum.argument.integer = 6;
    if (inplace.target == "torch.ops.aten.hardtanh_.default" || inplace.target == "torch.ops.aten.clamp_.default")
    {
        minimum.name = inplace.target == "torch.ops.aten.hardtanh_.default" ? "min_val" : "min";
        maximum.name = inplace.target == "torch.ops.aten.hardtanh_.default" ? "max_val" : "max";
        inplace.inputs.insert(inplace.inputs.begin(), maximum);
        inplace.inputs.push_back(minimum);
    }
    if (inplace.target == "torch.ops.aten.clamp_.Tensor")
    {
        minimum.name = "min";
        minimum.argument = add.inputs[1].argument; // x, same shape as local
        inplace.inputs.insert(inplace.inputs.begin(), minimum);
    }
    if (inplace.target == "torch.ops.aten.fill_.Scalar")
    {
        maximum.name = "value";
        inplace.inputs.push_back(maximum);
    }
    archive.program.graph.nodes.push_back(inplace);
    archive.program.graph.tensor_values["local"] = archive.program.graph.tensor_values["x"];
    archive.program.graph.tensor_values["updated"] = archive.program.graph.tensor_values["x"];
    archive.program.graph.outputs.assign(1, inplace.outputs[0]);
    OutputSpec output;
    output.type = OutputSpec::UserOutput;
    output.argument = inplace.outputs[0];
    archive.program.signature.outputs.assign(1, output);
    return archive;
}

static void expect_local_pointwise_import(const pnnx::pt2::ExportedProgramArchive& archive, const char* functional_target, const char* ir_type)
{
    using namespace pnnx::pt2;
    std::string error;
    ExportedProgram normalized = archive.program;
    const bool ok = append_default_arguments(normalized, error);
    expect_true(ok, error);
    if (!ok) return;
    const Node& node = normalized.graph.nodes[1];
    expect_true(node.target == functional_target && node.inputs[0].name == "self" && node.inputs[0].argument.name == "local"
                && node.name == "pointwise" && node.outputs.size() == 1 && node.outputs[0].name == "updated"
                && node.metadata.at("contract") == "preserve", "explicit functional mapping preserves node/result identity and metadata after ordering self");
    expect_true(normalized.graph.outputs[0].name == "updated" && normalized.signature.outputs[0].argument.name == "updated", "only mutation result is returned; no output rewiring");
    if (node.target == "torch.ops.aten.hardtanh.default" || node.target == "torch.ops.aten.clamp.default")
        expect_true(node.inputs.size() == 3 && node.inputs[1].argument.integer == 0 && node.inputs[2].argument.integer == 6, "non-default [0, 6] bounds survive schema reordering and rewrite");
    expect_true(append_default_arguments(normalized, error), "pointwise normalization is idempotent");

    pnnx::Graph graph;
    const bool imported = pnnx::import_exported_program_inputs(archive, graph, error) == 0
                          && pnnx::import_exported_program_nodes(archive.program, graph, error) == 0
                          && pnnx::import_exported_program_outputs(archive.program, graph, error) == 0;
    expect_true(imported, error);
    if (!imported) return;
    const pnnx::Operand* local = graph.get_operand("local");
    const pnnx::Operand* updated = graph.get_operand("updated");
    expect_true(local && updated && updated->producer->type == ir_type && updated->producer->inputs[0] == local
                && local->consumers.size() == 1 && updated->consumers.size() == 1 && updated->consumers[0]->type == "pnnx.Output"
                && updated->type == local->type && updated->shape == local->shape, "functional IR keeps single-use target, result consumers and return dtype/shape");
    expect_true(graph.get_operand("x") && graph.get_operand("x")->producer->type == "pnnx.Input", "functional result never replaces the external input");
    pnnx::Graph repeated;
    expect_true(pnnx::import_exported_program_inputs(archive, repeated, error) == 0
                && pnnx::import_exported_program_nodes(normalized, repeated, error) == 0, "already normalized pointwise program imports too");
}

static void test_local_pointwise_contract()
{
    using namespace pnnx::pt2;
    // Actual ATen schemas: no guessed overloads or synthetic mutable operators.
    const char* pairs[][3] = {
        {"torch.ops.aten.relu_.default", "torch.ops.aten.relu.default", "aten::relu"},
        {"torch.ops.aten.relu6_.default", "torch.ops.aten.relu6.default", "aten::relu6"},
        {"torch.ops.aten.hardtanh_.default", "torch.ops.aten.hardtanh.default", "aten::hardtanh"},
        {"torch.ops.aten.hardsigmoid_.default", "torch.ops.aten.hardsigmoid.default", "aten::hardsigmoid"},
        {"torch.ops.aten.hardswish_.default", "torch.ops.aten.hardswish.default", "aten::hardswish"},
        {"torch.ops.aten.silu_.default", "torch.ops.aten.silu.default", "aten::silu"},
        {"torch.ops.aten.sigmoid_.default", "torch.ops.aten.sigmoid.default", "aten::sigmoid"},
        {"torch.ops.aten.tanh_.default", "torch.ops.aten.tanh.default", "aten::tanh"},
        {"torch.ops.aten.clamp_.default", "torch.ops.aten.clamp.default", "aten::clamp"},
        {"torch.ops.aten.clamp_.Tensor", "torch.ops.aten.clamp.Tensor", "aten::clamp"},
        {"torch.ops.aten.leaky_relu_.default", "torch.ops.aten.leaky_relu.default", "aten::leaky_relu"},
        {"torch.ops.aten.elu_.default", "torch.ops.aten.elu.default", "aten::elu"},
        {"torch.ops.aten.celu_.default", "torch.ops.aten.celu.default", "aten::celu"}
    };
    std::string error;
    const char* mutation_error = "unsupported alias write/mutation of argument self";
    for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); i++)
    {
        ExportedProgramArchive archive = local_pointwise_archive(pairs[i][0]);
        expect_local_pointwise_import(archive, pairs[i][1], pairs[i][2]);
        ExportedProgram normalized = archive.program;
        expect_true(append_default_arguments(normalized, error) && normalized.graph.nodes[0].target == "torch.ops.aten.add.Scalar"
                    && normalized.graph.nodes[0].inputs[0].name == "self" && normalized.graph.nodes[0].inputs[1].argument.integer == 3, "x + 3 establishes ownership after real Scalar overload selection");
        Node standalone = archive.program.graph.nodes[1];
        expect_node_rejected(standalone, mutation_error);
        expect_true(!normalize_exported_program_node(standalone, error, &archive.program.graph), "tensor metadata alone cannot authorize a pointwise mutation");
        // Keep the schema-valid scalar/tensor bounds while targeting an input.
        for (size_t j = 0; j < archive.program.graph.nodes[1].inputs.size(); j++)
            if (archive.program.graph.nodes[1].inputs[j].name == "self") archive.program.graph.nodes[1].inputs[j].argument.name = "x";
        expect_program_rejected(archive.program, mutation_error);
    }

    const ExportedProgram baseline = local_pointwise_archive("torch.ops.aten.relu_.default").program;
    ExportedProgramArchive cloned = local_pointwise_archive("torch.ops.aten.relu_.default");
    Node clone = unary_node("torch.ops.aten.clone.default");
    clone.name = "allocate";
    clone.outputs[0].name = "local";
    cloned.program.graph.nodes[0] = clone;
    expect_local_pointwise_import(cloned, "torch.ops.aten.relu.default", "aten::relu");

    const char* allocators[] = {
        "torch.ops.aten.add.Tensor", "torch.ops.aten.add.Scalar",
        "torch.ops.aten.mul.Tensor", "torch.ops.aten.mul.Scalar",
        "torch.ops.aten.sub.Tensor", "torch.ops.aten.sub.Scalar",
        "torch.ops.aten.div.Tensor", "torch.ops.aten.div.Scalar",
        "torch.ops.aten.clone.default", "torch.ops.aten.empty.memory_format",
        "torch.ops.aten.zeros.default", "torch.ops.aten.ones.default",
        "torch.ops.aten.new_empty.default", "torch.ops.aten.new_zeros.default", "torch.ops.aten.new_ones.default"
    };
    for (size_t i = 0; i < sizeof(allocators) / sizeof(allocators[0]); i++)
    {
        // Fill also exercises every allocator without reading empty storage.
        ExportedProgramArchive archive = local_pointwise_archive("torch.ops.aten.fill_.Scalar");
        Node producer = unary_node(allocators[i]);
        producer.name = "allocate";
        producer.outputs[0].name = "local";
        if (i < 8)
        {
            NamedArgument other;
            other.name = "other";
            other.argument.type = i % 2 == 0 ? Argument::Tensor : Argument::Integer;
            if (i % 2 == 0) other.argument.name = "x";
            else other.argument.integer = 3;
            producer.inputs.push_back(other);
        }
        if (i >= 9)
        {
            if (i < 12) producer.inputs.clear(); // non-new factories have no self
            producer.inputs.push_back(integer_list("size", 2, 3));
            NamedArgument dtype;
            dtype.name = "dtype";
            dtype.argument.type = Argument::ScalarType;
            dtype.argument.integer = 7;
            producer.inputs.push_back(dtype);
        }
        archive.program.graph.nodes[0] = producer;
        expect_local_pointwise_import(archive, "torch.ops.aten.fill.Scalar", "aten::fill");
    }

    ExportedProgram program;
    const char* external[] = {"x", "p_weight", "missing"};
    for (size_t i = 0; i < 3; i++)
    {
        program = baseline;
        program.graph.nodes[1].inputs[0].argument.name = external[i];
        expect_program_rejected(program, mutation_error);
    }
    const char* views[] = {
        "torch.ops.aten.alias.default", "torch.ops.aten.detach.default", "torch.ops.aten.t.default",
        "torch.ops.aten.contiguous.default", "torch.ops.aten._unsafe_view.default", "torch.ops.aten.slice.Tensor"
    };
    for (size_t i = 0; i < sizeof(views) / sizeof(views[0]); i++)
    {
        Node view = unary_node(views[i]);
        view.name = "view";
        view.outputs[0].name = "local";
        if (i == 4) view.inputs.push_back(integer_list("size", 2, 3));
        program = baseline;
        program.graph.nodes[0] = view;
        expect_program_rejected(program, mutation_error);
        program = cloned.program;
        view.inputs[0].argument.name = "local";
        view.outputs[0].name = "alias_local";
        program.graph.nodes.insert(program.graph.nodes.begin() + 1, view);
        program.graph.nodes[2].inputs[0].argument.name = "alias_local";
        expect_program_rejected(program, mutation_error); // no local view writes
        program.graph.nodes[2].inputs[0].argument.name = "local";
        expect_program_rejected(program, mutation_error); // even an unused alias blocks the root
    }
    // An unannotated unknown return must not lose possible preceding roots,
    // even when the target itself occurs only once in node inputs.
    program = baseline;
    Node unknown = unary_node("torch.ops.aten._unsafe_view.default");
    unknown.name = "unknown_alias";
    unknown.outputs[0].name = "unknown_alias";
    unknown.inputs.push_back(integer_list("size", 2, 3));
    program.graph.nodes.insert(program.graph.nodes.begin() + 1, unknown);
    expect_program_rejected(program, mutation_error);
    program = baseline;
    program.graph.nodes[0] = unary_node("torch.ops.aten.neg.default");
    program.graph.nodes[0].outputs[0].name = "local";
    expect_program_rejected(program, mutation_error); // pure does not mean whitelisted allocator

    for (int before = 0; before < 2; before++)
    {
        program = baseline;
        Node use = unary_node("torch.ops.aten.relu.default");
        use.name = "use_original";
        use.inputs[0].argument.name = "local";
        program.graph.nodes.insert(program.graph.nodes.begin() + (before ? 1 : 2), use);
        expect_program_rejected(program, mutation_error);
    }
    program = baseline;
    Node repeated = program.graph.nodes[1];
    repeated.name = "repeat_write";
    repeated.outputs[0].name = "repeat_result";
    program.graph.nodes.push_back(repeated);
    expect_program_rejected(program, mutation_error);
    program = baseline;
    repeated.inputs[0].argument.name = "x";
    program.graph.nodes.push_back(repeated);
    expect_program_rejected(program, mutation_error); // a later failure must not commit the first rewrite
    program = baseline;
    program.graph.outputs.push_back(program.graph.nodes[1].inputs[0].argument);
    expect_program_rejected(program, mutation_error);
    program = baseline;
    program.signature.outputs[0].argument = program.graph.nodes[1].inputs[0].argument;
    expect_program_rejected(program, mutation_error);
    program = baseline;
    Node cat = unary_node("torch.ops.aten.cat.default");
    cat.inputs[0].name = "tensors";
    cat.inputs[0].argument = Argument();
    cat.inputs[0].argument.type = Argument::Tensors;
    cat.inputs[0].argument.values.push_back(program.graph.nodes[1].inputs[0].argument);
    program.graph.nodes.push_back(cat);
    expect_program_rejected(program, mutation_error);
    program = local_pointwise_archive("torch.ops.aten.clamp_.Tensor").program;
    program.graph.nodes[1].inputs[0].argument.name = "local"; // min and self reference the same root
    expect_program_rejected(program, mutation_error);

    program = baseline;
    program.graph.nodes[1].target = "torch.ops.aten.zero_.default";
    expect_program_rejected(program, mutation_error);
    program.graph.nodes[1].target = "torch.ops.aten.rrelu_.default";
    expect_program_rejected(program, mutation_error);
    const char* random_factories[] = {"torch.ops.aten.rand_like.default", "torch.ops.aten.randn_like.default"};
    for (size_t i = 0; i < 2; i++)
    {
        program = baseline;
        program.graph.nodes[0] = unary_node(random_factories[i]);
        program.graph.nodes[0].outputs[0].name = "local";
        expect_program_rejected(program, "RNG state effect");
    }
    program = baseline;
    program.graph.nodes[0] = unary_node("torch.ops.aten.add.out");
    NamedArgument other = program.graph.nodes[0].inputs[0];
    other.name = "other";
    program.graph.nodes[0].inputs.push_back(other);
    other.name = "out";
    program.graph.nodes[0].inputs.push_back(other);
    program.graph.nodes[0].outputs[0].name = "local";
    expect_program_rejected(program, "unsupported alias write/mutation of argument out");

    program = baseline;
    program.graph.nodes[1].outputs.clear();
    expect_program_rejected(program, "output count 0 does not match schema return count 1");
    program = baseline;
    program.graph.nodes[1].outputs[0] = Argument();
    program.graph.nodes[1].outputs[0].type = Argument::None;
    expect_program_rejected(program, "does not match schema return type Tensor");
    program = baseline;
    program.graph.nodes[1].outputs[0].name = "local";
    expect_program_rejected(program, "already defined");
    for (int i = 0; i < 5; i++)
    {
        program = baseline;
        TensorMeta& result = program.graph.tensor_values["updated"];
        if (i == 0) result.scalar_type = 5;
        if (i == 1) result.sizes[0] = dimension(1);
        if (i == 2) result.strides[0] = dimension(4);
        if (i == 3) result.storage_offset = dimension(1);
        if (i == 4)
        {
            result.sizes.push_back(dimension(1));
            result.strides.push_back(dimension(1));
        }
        expect_program_rejected(program, "in-place return metadata disagrees with target");
    }
    ExportedProgramArchive dynamic = local_pointwise_archive("torch.ops.aten.relu_.default");
    SymInt symbol;
    symbol.type = SymInt::Expression;
    symbol.expression = "s0";
    const char* tensors[] = {"x", "local", "updated"};
    for (size_t i = 0; i < 3; i++) dynamic.program.graph.tensor_values[tensors[i]].sizes[0] = symbol;
    expect_local_pointwise_import(dynamic, "torch.ops.aten.relu.default", "aten::relu");
    program = dynamic.program;
    expect_true(append_default_arguments(program, error) && program.graph.tensor_values["updated"].sizes[0].expression == "s0"
                && !program.graph.tensor_values["updated"].sizes[0].has_hint, "rewrite preserves symbolic return metadata without adding hints or runtime shape evaluation");
    program = dynamic.program;
    program.graph.tensor_values["local"].sizes[0].has_hint = true;
    program.graph.tensor_values["local"].sizes[0].hint = 2;
    program.graph.tensor_values["updated"].sizes[0].has_hint = true;
    program.graph.tensor_values["updated"].sizes[0].hint = 2;
    program.graph.tensor_values["updated"].sizes[0].expression = "s1";
    expect_program_rejected(program, "in-place return metadata disagrees with target");
    program = baseline;
    program.graph.outputs.clear();
    program.signature.outputs.clear();
    expect_true(append_default_arguments(program, error) && program.graph.nodes.size() == 2
                && program.graph.nodes[1].target == "torch.ops.aten.relu.default" && program.graph.nodes[1].outputs[0].name == "updated", "an unused result is still validated and retained, not treated as a discardable None-only effect");
}

static void test_node_return_contract()
{
    using namespace pnnx::pt2;
    std::string error;
    Node node = unary_node("torch.ops.aten.relu.default");
    expect_true(normalize_exported_program_node(node, error), error);
    Argument second = node.outputs[0];
    second.name = "second_output";
    node.outputs.push_back(second);
    expect_node_rejected(node, "output count 2 does not match schema return count 1");
    node.outputs.resize(1);
    node.outputs[0].type = Argument::SymInteger;
    expect_node_rejected(node, "outputs[0]: output type");
    node.outputs[0] = Argument();
    node.outputs[0].type = Argument::Integer;
    expect_node_rejected(node, "does not match schema return type Tensor");
    node.outputs[0].type = Argument::Tensors;
    node.outputs[0].values.push_back(second);
    expect_node_rejected(node, "does not match schema return type Tensor");

    // A real tuple-return schema uses two top-level arguments, not as_tensors.
    Node tuple = unary_node("torch.ops.aten.max.dim");
    NamedArgument dim;
    dim.name = "dim";
    dim.argument.type = Argument::Integer;
    dim.argument.integer = 1;
    tuple.inputs.push_back(dim);
    tuple.outputs.push_back(second);
    expect_true(normalize_exported_program_node(tuple, error), error);
    ExportedProgramArchive tuple_archive = make_archive();
    tuple_archive.program.graph.nodes.assign(1, tuple);
    TensorMeta values_meta;
    values_meta.scalar_type = 7;
    values_meta.sizes.push_back(dimension(2));
    tuple_archive.program.graph.tensor_values[tuple.outputs[0].name] = values_meta;
    values_meta.scalar_type = 5;
    tuple_archive.program.graph.tensor_values[tuple.outputs[1].name] = values_meta;
    pnnx::Graph tuple_graph;
    expect_true(pnnx::import_exported_program_inputs(tuple_archive, tuple_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(tuple_archive.program, tuple_graph, error) == 0, error);
    const pnnx::Operand* indices = tuple_graph.get_operand(second.name);
    expect_true(indices && indices->type == 5 && indices->producer->outputs.size() == 2 && indices->producer->outputs[1] == indices, "tuple return order and indices tensor metadata are preserved");
    node = tuple;
    node.outputs.pop_back();
    expect_node_rejected(node, "output count 1 does not match schema return count 2");
    node = tuple;
    node.outputs[1].type = Argument::SymInteger;
    expect_node_rejected(node, "outputs[1]: output type");
    Argument tensor_list;
    tensor_list.type = Argument::Tensors;
    tensor_list.values = tuple.outputs;
    node = tuple;
    node.outputs.assign(1, tensor_list);
    expect_node_rejected(node, "output count 1 does not match schema return count 2");

    // Tensor[] is one serialized argument regardless of the list length.
    Node list = unary_node("torch.ops.aten.unbind.int");
    list.outputs.assign(1, tensor_list);
    expect_true(normalize_exported_program_node(list, error), error);
    ExportedProgramArchive list_archive = make_archive();
    list_archive.program.graph.nodes.assign(1, list);
    pnnx::Graph list_graph;
    expect_true(pnnx::import_exported_program_inputs(list_archive, list_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(list_archive.program, list_graph, error) == 0, error);
    const pnnx::Operand* list_item = list_graph.get_operand(second.name);
    expect_true(list_item && list_item->producer->type == "torch.unbind" && list_item->producer->outputs.size() == 2, "one as_tensors return expands to two tensor operands");
    node = list;
    node.outputs = tensor_list.values;
    expect_node_rejected(node, "output count 2 does not match schema return count 1");
    node.outputs.assign(1, second);
    expect_node_rejected(node, "outputs[0]: output type");
    node = list;
    node.outputs[0].values[1].type = Argument::SymInteger;
    expect_node_rejected(node, "outputs[0]: output type");
    node = list;
    node.outputs[0].values[1] = Argument();
    expect_node_rejected(node, "unknown serialized argument type");
    node = list;
    node.outputs[0].values[1].name.clear();
    expect_node_rejected(node, "reference name is empty", "graph.nodes[0].outputs[0].values[1].name: expected nonempty ASCII identifier [A-Za-z_][A-Za-z0-9_]*");
    node = list;
    node.outputs[0] = Argument();
    node.outputs[0].type = Argument::Integers;
    expect_node_rejected(node, "outputs[0]: output type");
    node = list;
    node.outputs[0].values.resize(1);
    expect_true(normalize_exported_program_node(node, error), "single-element tensor list is still one list return");
    node.outputs[0].values.clear();
    expect_true(normalize_exported_program_node(node, error), "empty tensor list is not a void return");
    list_archive.program.graph.tensor_values["x"].sizes[0] = dimension(0);
    list_archive.program.graph.nodes.assign(1, node);
    pnnx::Graph empty_list_graph;
    expect_true(pnnx::import_exported_program_inputs(list_archive, empty_list_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(list_archive.program, empty_list_graph, error) == 0, error);
    expect_true(!empty_list_graph.ops.empty() && empty_list_graph.ops.back()->type == "torch.unbind" && empty_list_graph.ops.back()->outputs.empty(), "zero-length unbind has a real empty Tensor[] return");

    // Scalar results use named as_sym_* variants, not tensor references or
    // input-style numeric coercions (float returns cannot be serialized as int).
    const char* scalar_targets[] = {"torch.ops.aten.sym_numel.default", "torch.ops.aten.is_floating_point.default", "torch.ops.aten.sym_is_contiguous.default", "torch.ops.aten.q_scale.default"};
    const Argument::Type scalar_types[] = {Argument::SymInteger, Argument::SymBoolean, Argument::SymBoolean, Argument::SymFloat};
    for (size_t i = 0; i < 4; i++)
    {
        node = unary_node(scalar_targets[i]);
        node.outputs[0].type = scalar_types[i];
        expect_true(normalize_exported_program_node(node, error), error);
        node.outputs[0].type = scalar_types[i] == Argument::SymInteger ? Argument::SymBoolean : Argument::SymInteger;
        expect_node_rejected(node, "outputs[0]: output type");
        node.outputs[0].type = Argument::Tensor;
        expect_node_rejected(node, "outputs[0]: output type");
    }
    const Argument::Type number_types[] = {Argument::SymInteger, Argument::SymFloat, Argument::SymBoolean};
    for (size_t i = 0; i < 3; i++)
    {
        node = unary_node("torch.ops.aten._local_scalar_dense.default");
        node.outputs[0].type = number_types[i];
        expect_true(normalize_exported_program_node(node, error), "Scalar schema accepts named integer, floating and boolean results");
    }
    node.outputs[0].name.clear();
    expect_node_rejected(node, "reference name is empty", "graph.nodes[0].outputs[0].name: expected nonempty ASCII identifier [A-Za-z_][A-Za-z0-9_]*");

    ExportedProgramArchive dynamic = make_archive();
    SymInt symbol;
    symbol.type = SymInt::Expression;
    symbol.expression = "s0";
    dynamic.program.graph.tensor_values["x"].sizes[0] = symbol;
    node = unary_node("torch.ops.aten.relu.default");
    dynamic.program.graph.nodes.assign(1, node);
    dynamic.program.graph.tensor_values[node.outputs[0].name] = dynamic.program.graph.tensor_values["x"];
    node = unary_node("torch.ops.aten.sym_numel.default");
    node.name = "numel";
    node.inputs[0].argument.name = "contract_output";
    node.outputs[0].type = Argument::SymInteger;
    node.outputs[0].name = "numel";
    dynamic.program.graph.nodes.push_back(node);
    symbol.expression = "3*s0";
    dynamic.program.graph.sym_int_values["numel"] = symbol;
    Node scalar_tensor = unary_node("torch.ops.aten.scalar_tensor.default");
    scalar_tensor.name = "numel_tensor";
    scalar_tensor.inputs[0].name = "s";
    scalar_tensor.inputs[0].argument = node.outputs[0];
    scalar_tensor.outputs[0].name = "numel_tensor";
    dynamic.program.graph.nodes.push_back(scalar_tensor);
    pnnx::Graph dynamic_graph;
    expect_true(pnnx::import_exported_program_inputs(dynamic, dynamic_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(dynamic.program, dynamic_graph, error) == 0, error);
    const pnnx::Operand* dynamic_output = dynamic_graph.get_operand("contract_output");
    expect_true(dynamic_output && dynamic_output->shape.size() == 2 && dynamic_output->shape[0] == -233 && dynamic_output->params.at("__shape_expr__0").s == "s0", "Tensor returns retain legitimate symbolic tensor metadata without a concrete hint");
    const pnnx::Operand* numel_tensor = dynamic_graph.get_operand("numel_tensor");
    expect_true(numel_tensor && numel_tensor->producer->inputs[0] == dynamic_graph.get_operand("numel"), "symbolic scalar result is resolved as an operand, not a metadata hint or constant");
}

static void test_node_reference_contract()
{
    using namespace pnnx::pt2;
    const Argument::Type types[] = {Argument::Tensor, Argument::SymInteger, Argument::SymFloat, Argument::SymBoolean};
    for (size_t i = 0; i < 4; i++)
    {
        Node node = unary_node(i == 0 ? "torch.ops.aten.relu.default" : "torch.ops.aten.scalar_tensor.default");
        node.inputs[0].name = i == 0 ? "self" : "s";
        node.inputs[0].argument.type = types[i];
        node.inputs[0].argument.name = "missing_reference";
        // Nonzero fields must not turn an unresolved reference into a constant.
        node.inputs[0].argument.integer = 7;
        node.inputs[0].argument.floating_point = 1.5;
        node.inputs[0].argument.boolean = true;
        std::string error;
        expect_true(normalize_exported_program_node(node, error), error);
        ExportedProgram program = make_program();
        program.graph.nodes.push_back(node);
        pnnx::Graph graph;
        expect_true(pnnx::import_exported_program_nodes(program, graph, error) != 0 && error.find("input missing_reference is not defined") != std::string::npos, "undefined tensor/symbolic references must not fall back to constants");
        expect_true(graph.ops.empty(), "undefined reference creates no constant or operator");
    }

    const Argument::Type constants[] = {Argument::Integer, Argument::FloatingPoint, Argument::Boolean};
    for (size_t i = 0; i < 3; i++)
    {
        Node node = unary_node("torch.ops.aten.scalar_tensor.default");
        node.inputs[0].name = "s";
        node.inputs[0].argument.type = constants[i];
        node.inputs[0].argument.name = "missing_reference";
        expect_node_rejected(node, "reference name missing_reference");
    }
    Node node = unary_node("torch.ops.aten.relu.default");
    node.inputs[0].argument.type = Argument::Unknown;
    expect_node_rejected(node, "unknown serialized argument type");
    node = unary_node("_operator.neg");
    node.inputs[0].argument.type = Argument::Integer;
    expect_node_rejected(node, "reference name x");
    node.inputs[0].argument.name.clear();
    node.outputs[0].type = Argument::Unknown;
    expect_node_rejected(node, "unknown serialized argument type");

    node = unary_node("torch.ops.aten.view.default");
    NamedArgument size;
    size.name = "size";
    size.argument.type = Argument::SymIntegers;
    Argument extent;
    extent.type = Argument::Integer;
    extent.name = "missing_extent";
    size.argument.values.push_back(extent);
    node.inputs.push_back(size);
    expect_node_rejected(node, "reference name missing_extent");
    node.inputs[1].argument.values[0].type = Argument::Unknown;
    expect_node_rejected(node, "unknown serialized argument type");

    node = unary_node("torch.ops.aten.scalar_tensor.default");
    node.inputs[0].name = "s";
    node.inputs[0].argument = Argument();
    node.inputs[0].argument.type = Argument::Integer;
    extent.type = Argument::SymInteger;
    node.inputs[0].argument.values.push_back(extent);
    expect_node_rejected(node, "non-container argument contains nested values/references");
    node = unary_node("torch.ops.aten.relu.default");
    node.outputs[0].values.push_back(extent);
    expect_node_rejected(node, "non-container argument contains nested values/references");
}

static void test_signature_and_shape_contract()
{
    using namespace pnnx::pt2;
    ExportedProgramArchive archive = make_archive();
    archive.program.graph.inputs[1].name = "other";
    expect_inputs_rejected(archive, "does not match graph signature");
    archive = make_archive();
    archive.program.graph.inputs[1] = archive.program.graph.inputs[0];
    archive.program.signature.inputs[1].argument = archive.program.graph.inputs[0];
    expect_inputs_rejected(archive, "duplicate input name");
    archive = make_archive();
    archive.program.graph.inputs[1].name.clear();
    archive.program.signature.inputs[1].argument.name.clear();
    expect_inputs_rejected(archive, "graph.inputs[1].name: expected nonempty ASCII identifier");
    archive = make_archive();
    archive.program.opset_version["aten"]++;
    expect_inputs_rejected(archive, "opset_version.aten");

    archive = make_archive();
    SymInt& bare = archive.program.graph.tensor_values["x"].sizes[0];
    bare.type = SymInt::Expression;
    bare.expression = "s0";
    bare.has_hint = false;
    pnnx::Graph bare_graph;
    std::string bare_error;
    expect_true(pnnx::import_exported_program_inputs(archive, bare_graph, bare_error) == 0, "bare input symbol needs no hint");
    archive.program.range_constraints["2*s0"] = RangeConstraint();
    expect_inputs_rejected(archive, "unvalidated derived range constraint");

    archive = make_archive();
    archive.program.signature.outputs[0].type = OutputSpec::UserInputMutation;
    archive.program.signature.outputs[0].target = "x";
    pnnx::Graph mutation_graph;
    std::string mutation_error;
    expect_true(pnnx::import_exported_program_outputs(archive.program, mutation_graph, mutation_error) != 0 && mutation_error.find("mutation") != std::string::npos, "signature mutation is explicitly rejected");

    const Argument::Type types[] = {Argument::Integer, Argument::Boolean, Argument::FloatingPoint, Argument::String, Argument::SymInteger, Argument::SymBoolean, Argument::SymFloat};
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++)
    {
        ExportedProgram program = make_program();
        Argument value;
        value.type = types[i];
        program.graph.outputs.push_back(value);
        OutputSpec spec;
        spec.argument = value;
        spec.argument.integer = 1;
        spec.argument.boolean = true;
        spec.argument.floating_point = 1.5;
        spec.argument.string = "different";
        program.signature.outputs.push_back(spec);
        pnnx::Graph graph;
        std::string error;
        expect_true(pnnx::import_exported_program_outputs(program, graph, error) != 0 && error.find("constant value") != std::string::npos, "constant output must agree with signature value, not just type");
        expect_true(graph.ops.empty(), "mismatched constant rejected before graph construction");
    }

    ExportedProgram list_program = make_program();
    Argument list;
    list.type = Argument::Integers;
    Argument item;
    item.type = Argument::Integer;
    list.values.push_back(item);
    list_program.graph.outputs.push_back(list);
    OutputSpec list_spec;
    list_spec.argument = list;
    list_spec.argument.values[0].integer = 1;
    list_program.signature.outputs.push_back(list_spec);
    pnnx::Graph list_graph;
    std::string list_error;
    expect_true(pnnx::import_exported_program_outputs(list_program, list_graph, list_error) != 0, "nested constant output values must agree");

    const char* derived[] = {"Add(s0, 1)", "2*s0", "Symbol('s0', integer=True) + 1", "FloorDiv(Symbol('s0'), Integer(2))"};
    for (size_t i = 0; i < 4; i++)
    {
        archive = make_archive();
        SymInt& size = archive.program.graph.tensor_values["x"].sizes[0];
        size.type = SymInt::Expression;
        size.expression = derived[i];
        size.has_hint = true;
        size.hint = 2;
        expect_inputs_rejected(archive, "unvalidated derived input expression");
        std::string error;
        std::vector<std::vector<int64_t> > shapes(1, std::vector<int64_t> {2, 3});
        expect_true(!pnnx::validate_exported_program_input_shapes(archive.program, shapes, error), "sample matching a hint does not prove a derived constraint");
    }
}

static void expect_names_rejected(const pnnx::pt2::ExportedProgram& program, const std::string& diagnostic)
{
    // Exercise all three public entry points without going through the reader.
    for (int boundary = 0; boundary < 3; boundary++)
    {
        pnnx::Graph graph;
        std::string error = "stale diagnostic";
        int result;
        if (boundary == 0)
        {
            pnnx::pt2::ExportedProgramArchive archive;
            archive.program = program;
            result = pnnx::import_exported_program_inputs(archive, graph, error);
        }
        else if (boundary == 1)
            result = pnnx::import_exported_program_nodes(program, graph, error);
        else
            result = pnnx::import_exported_program_outputs(program, graph, error);
        expect_true(result == -1 && error == diagnostic, "name boundary " + std::to_string(boundary) + ": " + error);
        expect_true(graph.ops.empty() && graph.operands.empty(), "invalid names rejected before IR construction");
    }
}

static void test_identifier_contract()
{
    using namespace pnnx::pt2;
    const std::string suffix = ": expected nonempty ASCII identifier [A-Za-z_][A-Za-z0-9_]*";
    const std::string invalid[] = {"", "0value", "x.y", "x:y", "x/y", "x\\y", "x-y", "x y", "x\nnext", "x\rnext", "x\t", "x\v", "x\f", "x'", "x\"", "x;pass", "x#", "x[0]", "x=1", "x\x7f", "x\xc3\xa9", std::string("x\0tail", 6)};
    const Argument::Type references[] = {Argument::Tensor, Argument::SymInteger, Argument::SymBoolean, Argument::SymFloat};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
    {
        ExportedProgram program = make_program();
        program.graph.tensor_values[invalid[i]] = TensorMeta();
        expect_names_rejected(program, "graph.tensor_values[0].name" + suffix);
        program = make_program();
        program.graph.sym_int_values[invalid[i]] = SymInt();
        expect_names_rejected(program, "graph.sym_int_values[0].name" + suffix);

        for (size_t j = 0; j < 4; j++)
        {
            Node node = unary_node("torch.ops.aten.relu.default");
            node.outputs[0].type = references[j];
            node.outputs[0].name = invalid[i];
            program = make_program();
            program.graph.nodes.push_back(node);
            expect_names_rejected(program, "graph.nodes[0].outputs[0].name" + suffix);
            // Empty input-style Sym* values are concrete literals, not names.
            if (j != 0 && invalid[i].empty()) continue;
            Argument reference;
            reference.type = references[j];
            reference.name = invalid[i];
            program = make_program();
            program.graph.inputs.push_back(reference);
            expect_names_rejected(program, "graph.inputs[0].name" + suffix);
            program.graph.inputs.clear();
            InputSpec input;
            input.argument = reference;
            program.signature.inputs.push_back(input);
            expect_names_rejected(program, "signature.inputs[0].argument.name" + suffix);
            program = make_program();
            node = unary_node("torch.ops.aten.relu.default");
            node.inputs[0].argument = reference;
            program.graph.nodes.push_back(node);
            expect_names_rejected(program, "graph.nodes[0].inputs[0].argument.name" + suffix);
            program = make_program();
            program.graph.outputs.push_back(reference);
            expect_names_rejected(program, "graph.outputs[0].name" + suffix);
            program.graph.outputs.clear();
            OutputSpec output;
            output.argument = reference;
            program.signature.outputs.push_back(output);
            expect_names_rejected(program, "signature.outputs[0].argument.name" + suffix);
        }

        program = make_program();
        Node node = unary_node("torch.ops.aten.relu.default");
        node.inputs[0].name = invalid[i];
        program.graph.nodes.push_back(node);
        expect_names_rejected(program, "graph.nodes[0].inputs[0].name" + suffix);
        if (!invalid[i].empty())
        {
            program.graph.nodes[0] = unary_node("torch.ops.aten.relu.default");
            program.graph.nodes[0].name = invalid[i];
            expect_names_rejected(program, "graph.nodes[0].name" + suffix);
        }

        // Validate recursively even when the hostile reference is not top-level.
        Argument tensor;
        tensor.type = Argument::Tensor;
        tensor.name = invalid[i];
        Argument optional;
        optional.type = Argument::OptionalTensor;
        optional.values.push_back(tensor);
        Argument list;
        list.type = Argument::OptionalTensors;
        list.values.push_back(optional);
        program = make_program();
        program.graph.inputs.push_back(list);
        expect_names_rejected(program, "graph.inputs[0].values[0].values[0].name" + suffix);
        program = make_program();
        node = unary_node("torch.ops.aten.unbind.int");
        node.outputs[0] = Argument();
        node.outputs[0].type = Argument::Tensors;
        node.outputs[0].values.push_back(tensor);
        program.graph.nodes.push_back(node);
        expect_names_rejected(program, "graph.nodes[0].outputs[0].values[0].name" + suffix);
    }

    ExportedProgramArchive archive = make_archive();
    archive.program.graph.nodes[0].name.clear();
    pnnx::Graph unnamed;
    std::string error;
    expect_true(pnnx::import_exported_program_inputs(archive, unnamed, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(archive.program, unnamed, error) == 0, error);
    expect_true(!unnamed.ops.empty() && unnamed.ops.back()->name == "pnnx_0" && unnamed.get_operand("pnnx_0_arg_2"), "empty node names retain deterministic unnamed generation");
}

static void test_target_name_contract()
{
    using namespace pnnx::pt2;
    const std::string invalid[] = {"", "aten::relu", "torch.ops.aten.relu", "torch.ops.aten..default", "torch.ops..relu.default", "torch.ops.aten.relu.", "torch.ops.aten.relu.default.extra", "torch.ops.aten.relu.default\n", "torch.ops.aten.relu.def'ault", "torch.ops.at/en.relu.default", "torch.ops.aten.re:lu.default", "torch.ops.aten.relu.0", "_operator.neg()", "operator.neg;pass", "operator.neg\r", "operator.neg\"", "torch.relu", std::string("torch.ops.aten.relu.default\0tail", 31)};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
    {
        ExportedProgram program = make_program();
        Node node = unary_node("torch.ops.aten.relu.default");
        node.target = invalid[i];
        program.graph.nodes.push_back(node);
        expect_names_rejected(program, "graph.nodes[0].target: expected torch.ops.<namespace>.<operator>.<overload> or operator.<function> / _operator.<function> with ASCII identifiers");
    }
    // Correct spelling alone must not bypass the actual schema/purity checks.
    expect_node_rejected(unary_node("torch.ops.aten.pnnx_nonexistent_name_test.default"), "operator schema was not found");
    expect_node_rejected(unary_node("operator.setitem"), "validated pure schema");
    const char* pure[] = {"_operator.neg", "operator.neg"};
    for (size_t i = 0; i < 2; i++)
    {
        ExportedProgram program = make_program();
        Node node = unary_node(pure[i]);
        node.inputs[0].argument = Argument();
        node.inputs[0].argument.type = Argument::Integer;
        node.inputs[0].argument.integer = 2;
        node.outputs[0].type = Argument::SymInteger;
        program.graph.nodes.push_back(node);
        pnnx::Graph graph;
        std::string error;
        expect_true(pnnx::import_exported_program_nodes(program, graph, error) == 0, error);
        expect_true(!graph.ops.empty() && graph.ops.back()->type == "operator.neg", "supported pure Python operator remains accepted");
    }
}

static void set_attribute_target(pnnx::pt2::ExportedProgramArchive& archive, const std::string& target)
{
    const pnnx::pt2::PayloadMeta payload = archive.state_dict.at(archive.program.signature.inputs[0].target);
    archive.state_dict.erase(archive.program.signature.inputs[0].target);
    archive.state_dict[target] = payload;
    archive.program.signature.inputs[0].target = target;
}

static void test_attribute_name_contract()
{
    using namespace pnnx::pt2;
    const std::string invalid[] = {"", ".weight", "layer.", "layer..weight", "layer:weight", "layer/weight", "layer\\weight", "layer.0x.weight", "0layer.weight", "layer.0.weight\n", "layer.0.weight\r", "layer.0.weight\t", "layer.'weight", "layer.\"weight", "layer.wei ght", "layer.\xc3\xa9", std::string("layer.\0weight", 13)};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
    {
        ExportedProgram program = make_archive().program;
        program.signature.inputs[0].target = invalid[i];
        expect_names_rejected(program, "signature.inputs[0].target: expected ASCII FQN with nonempty identifier or numeric segments");
    }
    const char* valid[] = {"linear.weight", "encoder.layers.0.weight", "blocks.12.3._weight", "_lifted_tensor_constant0", "_root._0.007.weight", "0.weight", "123"};
    const char* emitted[] = {"linear.weight", "encoder.layers.0.weight", "blocks.12.3._weight", "_lifted_tensor_constant0", "_root._0.007.weight", "_0.weight", "_123"};
    for (size_t i = 0; i < sizeof(valid) / sizeof(valid[0]); i++)
    {
        ExportedProgramArchive archive = make_archive();
        set_attribute_target(archive, valid[i]);
        // The same name rules cover state parameters, buffers and constants.
        archive.program.signature.inputs[0].type = i % 3 == 0 ? InputSpec::Parameter : i % 3 == 1 ? InputSpec::Buffer : InputSpec::TensorConstant;
        pnnx::Graph graph;
        std::string error;
        expect_true(pnnx::import_exported_program_inputs(archive, graph, error) == 0, error);
        expect_true(pnnx::import_exported_program_nodes(archive.program, graph, error) == 0, error);
        expect_true(pnnx::import_exported_program_outputs(archive.program, graph, error) == 0, error);
        expect_true(!graph.ops.empty() && graph.ops[0]->name == emitted[i] && graph.ops[0]->attrs["data"].data.size() == 16, "valid dotted/numeric FQN keeps its payload and has an identifier-safe emitted root");
    }

    for (int kind = 0; kind < 3; kind++)
    {
        ExportedProgramArchive archive = make_archive();
        const std::string first = kind == 2 ? "0.weight" : kind == 1 ? "a_b" : "a.b";
        const std::string second = kind == 2 ? "_0.weight" : kind == 1 ? "a.b" : "a_b";
        set_attribute_target(archive, first);
        InputSpec other = archive.program.signature.inputs[0];
        other.target = second;
        other.argument.name = "p_other";
        archive.program.signature.inputs.insert(archive.program.signature.inputs.begin() + 1, other);
        archive.program.graph.inputs.insert(archive.program.graph.inputs.begin() + 1, other.argument);
        archive.state_dict[second] = archive.state_dict.at(first);
        pnnx::Graph graph;
        std::string error;
        expect_true(pnnx::import_exported_program_inputs(archive, graph, error) == -1, "colliding attribute FQNs rejected");
        const std::string diagnostic = kind == 2 ? "operator name collision: '_0.weight' conflicts with '_0.weight' after identifier sanitization to '_0_weight'"
                               : "operator name collision: '" + second + "' conflicts with '" + first + "' after identifier sanitization to 'a_b'";
        expect_true(error == diagnostic, error);
        expect_true(graph.ops.empty() && graph.operands.empty(), "FQN collision is checked before payload materialization");
    }

    ExportedProgramArchive archive = make_archive();
    set_attribute_target(archive, "a.b");
    archive.program.graph.nodes[0].name = "a_b";
    pnnx::Graph collision;
    std::string error;
    expect_true(pnnx::import_exported_program_inputs(archive, collision, error) == 0, error);
    const size_t old_ops = collision.ops.size();
    const size_t old_values = collision.operands.size();
    expect_true(pnnx::import_exported_program_nodes(archive.program, collision, error) == -1
                && error == "operator name collision: 'a_b' conflicts with 'a.b' after identifier sanitization to 'a_b'", error);
    expect_true(collision.ops.size() == old_ops && collision.operands.size() == old_values, "attribute/node collision does not change imported inputs");

    // An attribute's self.linear_weight and a tensor's v_linear_weight are
    // distinct namespaces; node/result names such as linear are distinct too.
    archive = make_archive();
    archive.program.graph.inputs[1].name = "linear_weight";
    archive.program.signature.inputs[1].argument.name = "linear_weight";
    archive.program.graph.tensor_values["linear_weight"] = archive.program.graph.tensor_values.at("x");
    archive.program.graph.tensor_values.erase("x");
    archive.program.graph.nodes[0].inputs[0].argument.name = "linear_weight";
    pnnx::Graph separate;
    expect_true(pnnx::import_exported_program_inputs(archive, separate, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(archive.program, separate, error) == 0, error);
    expect_true(pnnx::import_exported_program_outputs(archive.program, separate, error) == 0, "operator and operand names must not be conflated");

    archive = make_archive();
    set_attribute_target(archive, "pnnx.input.0");
    pnnx::Graph input_collision;
    expect_true(pnnx::import_exported_program_inputs(archive, input_collision, error) == -1
                && error == "operator name collision: 'pnnx_input_0' conflicts with 'pnnx.input.0' after identifier sanitization to 'pnnx_input_0'", error);
    expect_true(input_collision.ops.empty() && input_collision.operands.empty(), "generated input operators cannot collide with FQNs");

    archive = make_archive();
    archive.program.graph.nodes[0].name = "pnnx_output_0";
    pnnx::Graph output_collision;
    expect_true(pnnx::import_exported_program_inputs(archive, output_collision, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(archive.program, output_collision, error) == 0, error);
    const size_t output_old_ops = output_collision.ops.size();
    expect_true(pnnx::import_exported_program_outputs(archive.program, output_collision, error) == -1
                && error == "operator name collision: 'pnnx_output_0' conflicts with 'pnnx_output_0' after identifier sanitization to 'pnnx_output_0'", error);
    const pnnx::Operand* output = output_collision.get_operand("linear");
    expect_true(output_collision.ops.size() == output_old_ops && output && output->consumers.empty(), "generated output collision does not attach consumers");
}

static void test_generated_name_collisions()
{
    using namespace pnnx::pt2;
    const std::string bias_error = "linear.bias: generated value name collision: 'linear_arg_2' is already defined or referenced";
    for (int kind = 0; kind < 5; kind++)
    {
        ExportedProgramArchive archive = make_archive();
        if (kind == 1)
            archive.program.graph.nodes[0].outputs[0].name = "linear_arg_2";
        if (kind == 2)
        {
            Node later = unary_node("torch.ops.aten.relu.default");
            later.outputs[0].name = "linear_arg_2";
            archive.program.graph.nodes.push_back(later);
        }
        if (kind == 3)
        {
            Node later = unary_node("torch.ops.aten.scalar_tensor.default");
            later.inputs[0].name = "s";
            later.inputs[0].argument.type = Argument::SymInteger;
            later.inputs[0].argument.name = "linear_arg_2";
            archive.program.graph.nodes.push_back(later);
        }
        if (kind == 4)
            archive.program.graph.sym_int_values["linear_arg_2"] = SymInt();
        pnnx::Graph graph;
        std::string error;
        expect_true(pnnx::import_exported_program_inputs(archive, graph, error) == 0, error);
        pnnx::Operand* existing = 0;
        if (kind == 0)
        {
            pnnx::Operator* producer = graph.new_operator("prim::Constant", "existing_value");
            producer->params["value"] = 17;
            existing = graph.new_operand("linear_arg_2");
            existing->producer = producer;
            producer->outputs.push_back(existing);
        }
        const size_t old_ops = graph.ops.size();
        const size_t old_values = graph.operands.size();
        expect_true(pnnx::import_exported_program_nodes(archive.program, graph, error) == -1 && error == bias_error, error);
        expect_true(graph.ops.size() == old_ops && graph.operands.size() == old_values, "constant collision rejects existing/future definitions, references and metadata without changing IR");
        if (existing)
            expect_true(graph.get_operand("linear_arg_2") == existing && existing->producer->params.at("value").i == 17, "existing output value is not overwritten or shadowed");
    }

    // A generated constant operator also must not steal an attribute's name.
    ExportedProgramArchive archive = make_archive();
    set_attribute_target(archive, "linear.arg.2");
    pnnx::Graph attribute_collision;
    std::string error;
    expect_true(pnnx::import_exported_program_inputs(archive, attribute_collision, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(archive.program, attribute_collision, error) == -1
                && error == "linear.bias: operator name collision: 'linear_arg_2' conflicts with 'linear.arg.2' after identifier sanitization to 'linear_arg_2'", error);
    expect_true(attribute_collision.ops.size() == 2 && attribute_collision.operands.size() == 2, "generated constant cannot collide with an attribute FQN");

    // Reserve later nodes even though their operators do not exist yet.
    archive = make_archive();
    Node later = unary_node("torch.ops.aten.relu.default");
    later.name = "linear_arg_2";
    archive.program.graph.nodes.push_back(later);
    pnnx::Graph later_graph;
    expect_true(pnnx::import_exported_program_inputs(archive, later_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(archive.program, later_graph, error) == -1
                && error == "linear.bias: operator name collision: 'linear_arg_2' conflicts with 'linear_arg_2' after identifier sanitization to 'linear_arg_2'", error);
    expect_true(later_graph.ops.size() == 2 && later_graph.operands.size() == 2, "later node names are reserved before constants");

    const char* list_names[] = {"factory_arg_0", "factory_arg_0_item_0"};
    for (size_t i = 0; i < 2; i++)
    {
        ExportedProgram program = make_program();
        Node node = unary_node("torch.ops.aten.full.default");
        node.name = "factory";
        node.inputs.clear();
        node.inputs.push_back(integer_list("size", 2, 3));
        NamedArgument fill;
        fill.name = "fill_value";
        fill.argument.type = Argument::Integer;
        fill.argument.integer = 1;
        node.inputs.push_back(fill);
        program.graph.nodes.push_back(node);
        pnnx::Graph graph;
        pnnx::Operand* existing = graph.new_operand(list_names[i]);
        expect_true(pnnx::import_exported_program_nodes(program, graph, error) == -1
                    && error == std::string("factory.size: generated value name collision: '") + list_names[i] + "' is already defined or referenced", error);
        expect_true(graph.ops.empty() && graph.operands.size() == 1 && graph.get_operand(list_names[i]) == existing, "list and nested item generated names are protected");
    }

    const char* output_names[] = {"pnnx_output_value_0", "pnnx_output_value_0_item_0"};
    for (int kind = 0; kind < 3; kind++)
    {
        ExportedProgram program = make_program();
        Argument value;
        value.type = Argument::Integer;
        value.integer = 7;
        if (kind != 0)
        {
            Argument list;
            list.type = Argument::Integers;
            list.values.push_back(value);
            value = list;
        }
        program.graph.outputs.push_back(value);
        OutputSpec spec;
        spec.argument = value;
        program.signature.outputs.push_back(spec);
        pnnx::Graph graph;
        const std::string name = output_names[kind == 2 ? 1 : 0];
        pnnx::Operand* existing = graph.new_operand(name);
        expect_true(pnnx::import_exported_program_outputs(program, graph, error) == -1
                    && error == "graph output 0: generated value name collision: '" + name + "' is already defined or referenced", error);
        expect_true(graph.ops.empty() && graph.operands.size() == 1 && graph.get_operand(name) == existing, "output constants/lists cannot overwrite existing results");
    }

    archive = make_archive();
    archive.program.graph.nodes[0].name.clear();
    later = unary_node("torch.ops.aten.relu.default");
    later.name = "pnnx_0";
    archive.program.graph.nodes.push_back(later);
    pnnx::Graph unnamed;
    expect_true(pnnx::import_exported_program_inputs(archive, unnamed, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(archive.program, unnamed, error) == -1
                && error == "operator name collision: 'pnnx_0' conflicts with 'pnnx_0' after identifier sanitization to 'pnnx_0'", error);
    expect_true(unnamed.ops.size() == 2 && unnamed.operands.size() == 2, "generated unnamed node collision is rejected before nodes");
}

int main()
{
    test_identifier_contract();
    test_target_name_contract();
    test_attribute_name_contract();
    test_generated_name_collisions();
    test_scalar_contract();
    test_import_bypass_storage();
    test_node_safety();
    test_factory_dtype_contract();
    test_factory_dtype_validation();
    test_factory_dtype_name_collisions();
    test_metadata_guards();
    test_local_fill_contract();
    test_local_pointwise_contract();
    test_node_return_contract();
    test_node_reference_contract();
    test_signature_and_shape_contract();
    pnnx::pt2::ExportedProgram defaults_program = make_program();
    pnnx::pt2::Node defaults_node;
    defaults_node.name = "scaled_dot_product_attention";
    defaults_node.target = "torch.ops.aten.scaled_dot_product_attention.default";
    const char* explicit_arguments[] = {"query", "key", "value", "enable_gqa"};
    for (size_t i = 0; i < 4; i++)
    {
        pnnx::pt2::NamedArgument argument;
        argument.name = explicit_arguments[i];
        argument.argument.type = i == 3 ? pnnx::pt2::Argument::Boolean : pnnx::pt2::Argument::Tensor;
        if (i != 3)
            argument.argument.name = explicit_arguments[i];
        argument.argument.boolean = true;
        defaults_node.inputs.push_back(argument);
    }
    pnnx::pt2::Argument defaults_output;
    defaults_output.type = pnnx::pt2::Argument::Tensor;
    defaults_output.name = "attention_output";
    defaults_node.outputs.push_back(defaults_output);
    defaults_program.graph.nodes.push_back(defaults_node);
    std::string error;
    expect_true(pnnx::pt2::append_default_arguments(defaults_program, error), error);
    const pnnx::pt2::Node& ordered_node = defaults_program.graph.nodes[0];
    expect_true(ordered_node.inputs.size() == 8, "dispatcher defaults are appended");
    if (ordered_node.inputs.size() == 8)
    {
        expect_true(ordered_node.inputs[3].name == "attn_mask" && ordered_node.inputs[3].argument.type == pnnx::pt2::Argument::None, "default arguments precede later explicit arguments");
        expect_true(ordered_node.inputs[7].name == "enable_gqa" && ordered_node.inputs[7].argument.boolean, "explicit keyword argument retains schema position");
    }

    pnnx::pt2::ExportedProgramArchive archive = make_archive();
    pnnx::Graph graph;
    expect_true(pnnx::import_exported_program_inputs(archive, graph, error) == 0, error);
    expect_true(graph.ops.size() == 2, "attribute and user input operators");
    expect_true(graph.ops[0]->type == "pnnx.Attribute" && graph.ops[0]->name == "linear.weight", "parameter attribute operator");
    expect_true(graph.ops[1]->type == "pnnx.Input" && graph.ops[1]->name == "pnnx_input_0", "user input operator");
    expect_true(graph.operands[0]->name == "p_weight" && graph.operands[0]->shape[0] == 2, "parameter operand metadata");
    expect_true(graph.operands[1]->name == "x" && graph.operands[1]->shape[1] == 3, "input operand metadata");

    const pnnx::Attribute& attribute = graph.ops[0]->attrs["data"];
    const float* data = (const float*)attribute.data.data();
    expect_true(attribute.shape.size() == 2 && attribute.shape[0] == 2 && attribute.shape[1] == 2, "attribute shape");
    expect_true(data[0] == 1.f && data[1] == 2.f && data[2] == 4.f && data[3] == 5.f, "strided attribute is materialized contiguously");

    expect_true(pnnx::import_exported_program_nodes(archive.program, graph, error) == 0, error);
    expect_true(graph.ops.size() == 4, "attribute, input, aten and constant operators");
    expect_true(graph.ops[2]->type == "prim::Constant" && graph.ops[2]->params["value"].type == 0, "none argument becomes constant");
    expect_true(graph.ops[3]->type == "aten::linear", "aten target is normalized");
    expect_true(graph.ops[3]->inputs.size() == 3 && graph.ops[3]->inputnames[2] == "bias", "named arguments are preserved");
    expect_true(graph.get_operand("linear")->shape[1] == 2, "node output tensor metadata");

    expect_true(pnnx::import_exported_program_outputs(archive.program, graph, error) == 0, error);
    expect_true(graph.ops.size() == 6, "two graph outputs are imported");
    expect_true(graph.ops[4]->type == "pnnx.Output" && graph.ops[5]->type == "pnnx.Output", "pnnx output operators");
    expect_true(graph.get_operand("linear")->consumers.size() == 2, "tuple may return the same tensor twice");

    pnnx::pt2::ExportedProgram argument_program = make_program();
    pnnx::pt2::Argument tensor_list;
    tensor_list.type = pnnx::pt2::Argument::Tensors;
    pnnx::pt2::Argument tensor_reference;
    tensor_reference.type = pnnx::pt2::Argument::Tensor;
    tensor_reference.name = "x";
    tensor_list.values.push_back(tensor_reference);
    argument_program.graph.outputs.push_back(tensor_list);
    pnnx::pt2::Argument bool_list;
    bool_list.type = pnnx::pt2::Argument::Booleans;
    pnnx::pt2::Argument flag;
    flag.type = pnnx::pt2::Argument::Boolean;
    flag.boolean = true;
    bool_list.values.push_back(flag);
    argument_program.graph.outputs.push_back(bool_list);
    pnnx::pt2::Argument dtype;
    dtype.type = pnnx::pt2::Argument::ScalarType;
    dtype.integer = 7;
    argument_program.graph.outputs.push_back(dtype);
    pnnx::pt2::Argument device;
    device.type = pnnx::pt2::Argument::DeviceValue;
    device.device.type = "cpu";
    argument_program.graph.outputs.push_back(device);
    pnnx::pt2::Argument optional;
    optional.type = pnnx::pt2::Argument::OptionalTensor;
    optional.values.push_back(tensor_reference);
    argument_program.graph.outputs.push_back(optional);
    for (size_t i = 0; i < argument_program.graph.outputs.size(); i++)
    {
        pnnx::pt2::OutputSpec spec;
        spec.argument = argument_program.graph.outputs[i];
        argument_program.signature.outputs.push_back(spec);
    }
    // Independent programs must not append a second pnnx_output_0 to one IR.
    pnnx::Graph argument_graph;
    expect_true(pnnx::import_exported_program_inputs(archive, argument_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_outputs(argument_program, argument_graph, error) == 0, error);
    expect_true(argument_graph.get_operand("pnnx_output_value_1") && argument_graph.get_operand("pnnx_output_value_1")->producer->type == "prim::ListConstruct", "bool list construct");
    expect_true(argument_graph.get_operand("pnnx_output_value_2") && argument_graph.get_operand("pnnx_output_value_2")->producer->params["value"].i == 6, "pt2 dtype maps to c10 scalar type");
    expect_true(argument_graph.get_operand("pnnx_output_value_3") && argument_graph.get_operand("pnnx_output_value_3")->producer->params["value"].s == "cpu", "device argument constant");

    pnnx::pt2::ExportedProgram split_program = make_program();
    pnnx::pt2::Node split_node;
    split_node.name = "split";
    split_node.target = "torch.ops.aten.split.Tensor";
    pnnx::pt2::NamedArgument split_input;
    split_input.name = "self";
    split_input.argument = tensor_reference;
    split_node.inputs.push_back(split_input);
    pnnx::pt2::NamedArgument split_size;
    split_size.name = "split_size";
    split_size.argument.type = pnnx::pt2::Argument::Integer;
    split_size.argument.integer = 1;
    split_node.inputs.push_back(split_size);
    pnnx::pt2::NamedArgument split_dim;
    split_dim.name = "dim";
    split_dim.argument.type = pnnx::pt2::Argument::Integer;
    split_dim.argument.integer = 0;
    split_node.inputs.push_back(split_dim);
    pnnx::pt2::Argument split_output = tensor_reference;
    split_output.name = "split_output";
    pnnx::pt2::Argument split_output_list;
    split_output_list.type = pnnx::pt2::Argument::Tensors;
    split_output_list.values.push_back(split_output);
    pnnx::pt2::Argument split_output_second = split_output;
    split_output_second.name = "split_output_second";
    split_output_list.values.push_back(split_output_second);
    split_node.outputs.push_back(split_output_list);
    split_program.graph.nodes.push_back(split_node);
    pnnx::pt2::TensorMeta split_meta = archive.program.graph.tensor_values["x"];
    split_meta.sizes[0] = dimension(1);
    split_program.graph.tensor_values["split_output"] = split_meta;
    split_program.graph.tensor_values["split_output_second"] = split_meta;

    const size_t split_old_op_count = graph.ops.size();
    expect_true(pnnx::import_exported_program_nodes(split_program, graph, error) == 0, error);
    const pnnx::Operator* canonical_split = graph.ops.back();
    expect_true(graph.ops.size() == split_old_op_count + 3, "split constants and operator");
    expect_true(canonical_split->type == "torch.split", "exported split target is canonicalized");
    expect_true(canonical_split->outputs.size() == 2 && canonical_split->outputs[0]->name == "split_output" && canonical_split->outputs[1]->name == "split_output_second", "split.Tensor uses one as_tensors return with both split pieces");
    expect_true(canonical_split->inputnames.size() == 3 && canonical_split->inputnames[0] == "tensor" && canonical_split->inputnames[1] == "split_size_or_sections", "exported split argument names are canonicalized");

    pnnx::pt2::ExportedProgram list_program = make_program();
    pnnx::pt2::Node list_node;
    list_node.name = "split";
    list_node.target = "torch.ops.aten.split_with_sizes.default";
    list_node.inputs.push_back(split_input);
    pnnx::pt2::NamedArgument split_sizes;
    split_sizes.name = "split_sizes";
    split_sizes.argument.type = pnnx::pt2::Argument::Integers;
    split_sizes.argument.values.push_back(split_size.argument);
    split_sizes.argument.values.push_back(split_size.argument);
    list_node.inputs.push_back(split_sizes);
    pnnx::pt2::Argument tensor_list_output;
    tensor_list_output.type = pnnx::pt2::Argument::Tensors;
    pnnx::pt2::Argument first = tensor_reference;
    first.name = "split_0";
    pnnx::pt2::Argument second = tensor_reference;
    second.name = "split_1";
    tensor_list_output.values.push_back(first);
    tensor_list_output.values.push_back(second);
    list_node.outputs.push_back(tensor_list_output);
    list_program.graph.nodes.push_back(list_node);
    list_program.graph.tensor_values["split_0"] = split_meta;
    list_program.graph.tensor_values["split_1"] = split_meta;
    list_program.graph.outputs.push_back(tensor_list_output);
    pnnx::pt2::OutputSpec list_output_spec;
    list_output_spec.argument = tensor_list_output;
    list_program.signature.outputs.push_back(list_output_spec);
    pnnx::pt2::Argument none_output;
    none_output.type = pnnx::pt2::Argument::None;
    list_program.graph.outputs.push_back(none_output);
    pnnx::pt2::OutputSpec none_output_spec;
    none_output_spec.argument = none_output;
    list_program.signature.outputs.push_back(none_output_spec);

    // This separate split fixture intentionally uses the same node name.
    pnnx::Graph tensor_list_graph;
    expect_true(pnnx::import_exported_program_inputs(archive, tensor_list_graph, error) == 0, error);
    expect_true(pnnx::import_exported_program_nodes(list_program, tensor_list_graph, error) == 0, error);
    expect_true(!tensor_list_graph.ops.empty() && tensor_list_graph.ops.back()->outputs.size() == 2, "tensor-list node has multiple outputs");
    expect_true(pnnx::import_exported_program_outputs(list_program, tensor_list_graph, error) == 0, error);
    expect_true(!tensor_list_graph.ops.empty() && tensor_list_graph.ops.back()->type == "pnnx.Output", "mixed constant output is imported");

    pnnx::pt2::ExportedProgramArchive dynamic_archive = make_archive();
    pnnx::pt2::SymInt dynamic_dimension;
    dynamic_dimension.type = pnnx::pt2::SymInt::Expression;
    dynamic_dimension.expression = "Symbol('s0', integer=True)";
    dynamic_dimension.has_hint = true;
    dynamic_dimension.hint = 3;
    dynamic_archive.program.graph.tensor_values["x"].sizes[0] = dynamic_dimension;
    pnnx::Graph dynamic_graph;
    expect_true(pnnx::import_exported_program_inputs(dynamic_archive, dynamic_graph, error) == 0, error);
    const pnnx::Operand* dynamic_input = dynamic_graph.get_operand("x");
    expect_true(dynamic_input->shape[0] == -233, "symbolic dimension marker");
    expect_true(dynamic_input->params.at("__shape__0").s == "Symbol_s0_integer_True_", "symbolic dimension key");
    expect_true(dynamic_input->params.at("__shape_expr__0").s == "Symbol('s0', integer=True)", "symbolic expression metadata");
    expect_true(dynamic_input->params.at("__shape_hint__0").i == 3, "symbolic hint metadata");

    const char* dynamic_param = "test_load_exported_program_dynamic.param";
    const char* dynamic_bin = "test_load_exported_program_dynamic.bin";
    expect_true(dynamic_graph.save(dynamic_param, dynamic_bin) == 0, "save symbolic graph");
    pnnx::Graph loaded_dynamic_graph;
    expect_true(loaded_dynamic_graph.load(dynamic_param, dynamic_bin) == 0, "load symbolic graph");
    const pnnx::Operand* loaded_dynamic_input = loaded_dynamic_graph.get_operand("x");
    expect_true(loaded_dynamic_input && loaded_dynamic_input->shape[0] == -233, "symbolic dimension round trip");
    expect_true(loaded_dynamic_input && loaded_dynamic_input->params.at("__shape__0").s == "Symbol_s0_integer_True_", "symbolic key round trip");
    remove(dynamic_param);
    remove(dynamic_bin);

    pnnx::pt2::ExportedProgram shape_program = make_program();
    pnnx::pt2::SymInt shared;
    shared.type = pnnx::pt2::SymInt::Expression;
    shared.expression = "Symbol('s17', positive=True, integer=True)";
    shared.has_hint = true;
    shared.hint = 3;
    pnnx::pt2::TensorMeta shared_meta;
    shared_meta.sizes.push_back(shared);
    shared_meta.sizes.push_back(dimension(4));
    shape_program.graph.tensor_values["x"] = shared_meta;
    shape_program.graph.tensor_values["y"] = shared_meta;
    pnnx::pt2::InputSpec shared_input;
    shared_input.type = pnnx::pt2::InputSpec::UserInput;
    shared_input.argument.type = pnnx::pt2::Argument::Tensor;
    shared_input.argument.name = "x";
    shape_program.signature.inputs.push_back(shared_input);
    shared_input.argument.name = "y";
    shape_program.signature.inputs.push_back(shared_input);
    pnnx::pt2::RangeConstraint range;
    range.has_min = true;
    range.min = 2;
    range.has_max = true;
    range.max = 8;
    shape_program.range_constraints["s17"] = range;

    std::vector<std::vector<int64_t> > valid_shapes;
    valid_shapes.push_back(std::vector<int64_t> {5, 4});
    valid_shapes.push_back(std::vector<int64_t> {5, 4});
    expect_true(pnnx::validate_exported_program_input_shapes(shape_program, valid_shapes, error), error);
    std::vector<std::vector<int64_t> > static_mismatch = valid_shapes;
    static_mismatch[0][1] = 3;
    expect_true(!pnnx::validate_exported_program_input_shapes(shape_program, static_mismatch, error) && error.find("expected 4") != std::string::npos, "static dimension mismatch");
    std::vector<std::vector<int64_t> > range_mismatch = valid_shapes;
    range_mismatch[0][0] = range_mismatch[1][0] = 9;
    expect_true(!pnnx::validate_exported_program_input_shapes(shape_program, range_mismatch, error) && error.find("[2, 8]") != std::string::npos, "range constraint mismatch");
    std::vector<std::vector<int64_t> > shared_mismatch = valid_shapes;
    shared_mismatch[1][0] = 6;
    expect_true(!pnnx::validate_exported_program_input_shapes(shape_program, shared_mismatch, error) && error.find("shared symbol") != std::string::npos, "shared symbol mismatch");

    pnnx::pt2::ExportedProgramArchive unsupported_archive = make_archive();
    pnnx::pt2::SymInt unsupported;
    unsupported.type = pnnx::pt2::SymInt::Expression;
    unsupported.expression = "FloorDiv(Symbol('s0', integer=True), Integer(2))";
    unsupported_archive.program.graph.tensor_values["x"].sizes[0] = unsupported;
    pnnx::Graph unsupported_graph;
    expect_true(pnnx::import_exported_program_inputs(unsupported_archive, unsupported_graph, error) != 0, "unsupported expression without hint is rejected");
    expect_true(error.find("FloorDiv") != std::string::npos, "unsupported expression error includes expression");

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program input test(s) failed\n", test_failures);
        return 1;
    }
    return 0;
}