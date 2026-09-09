// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <string.h>

#include <limits>

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

static void expect_node_rejected(const pnnx::pt2::Node& node, const char* diagnostic)
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
    expect_true(pnnx::import_exported_program_nodes(local_write.program, local_graph, error) != 0 && error.find("including local temporaries (alias/liveness analysis is not implemented)") != std::string::npos, "local writes report the conservative implementation limit, not a proven external mutation");
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
    expect_node_rejected(node, "reference name is empty");
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
    expect_node_rejected(node, "reference name is empty");

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
    expect_inputs_rejected(archive, "empty or duplicate input name");
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

int main()
{
    test_scalar_contract();
    test_import_bypass_storage();
    test_node_safety();
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
    expect_true(pnnx::import_exported_program_outputs(argument_program, graph, error) == 0, error);
    expect_true(graph.get_operand("pnnx_output_value_1") && graph.get_operand("pnnx_output_value_1")->producer->type == "prim::ListConstruct", "bool list construct");
    expect_true(graph.get_operand("pnnx_output_value_2") && graph.get_operand("pnnx_output_value_2")->producer->params["value"].i == 6, "pt2 dtype maps to c10 scalar type");
    expect_true(graph.get_operand("pnnx_output_value_3") && graph.get_operand("pnnx_output_value_3")->producer->params["value"].s == "cpu", "device argument constant");

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

    expect_true(pnnx::import_exported_program_nodes(list_program, graph, error) == 0, error);
    expect_true(graph.ops.back()->outputs.size() == 2, "tensor-list node has multiple outputs");
    expect_true(pnnx::import_exported_program_outputs(list_program, graph, error) == 0, error);
    expect_true(graph.ops[graph.ops.size() - 1]->type == "pnnx.Output", "mixed constant output is imported");

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