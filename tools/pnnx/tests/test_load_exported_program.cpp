// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>

#include "load_exported_program.h"

static int test_failures = 0;

static void expect_true(bool value, const char* message)
{
    if (value)
        return;
    fprintf(stderr, "FAILED: %s\n", message);
    test_failures++;
}

static pnnx::pt2::SymInt dimension(int value)
{
    pnnx::pt2::SymInt result;
    result.integer = value;
    return result;
}

static pnnx::pt2::ExportedProgramArchive make_archive()
{
    pnnx::pt2::ExportedProgramArchive archive;

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

int main()
{
    pnnx::pt2::ExportedProgramArchive archive = make_archive();
    pnnx::Graph graph;
    std::string error;
    expect_true(pnnx::import_exported_program_inputs(archive, graph, error) == 0, error.c_str());
    expect_true(graph.ops.size() == 2, "attribute and user input operators");
    expect_true(graph.ops[0]->type == "pnnx.Attribute" && graph.ops[0]->name == "linear.weight", "parameter attribute operator");
    expect_true(graph.ops[1]->type == "pnnx.Input" && graph.ops[1]->name == "pnnx_input_0", "user input operator");
    expect_true(graph.operands[0]->name == "p_weight" && graph.operands[0]->shape[0] == 2, "parameter operand metadata");
    expect_true(graph.operands[1]->name == "x" && graph.operands[1]->shape[1] == 3, "input operand metadata");

    const pnnx::Attribute& attribute = graph.ops[0]->attrs["data"];
    const float* data = (const float*)attribute.data.data();
    expect_true(attribute.shape.size() == 2 && attribute.shape[0] == 2 && attribute.shape[1] == 2, "attribute shape");
    expect_true(data[0] == 1.f && data[1] == 2.f && data[2] == 4.f && data[3] == 5.f, "strided attribute is materialized contiguously");

    expect_true(pnnx::import_exported_program_nodes(archive.program, graph, error) == 0, error.c_str());
    expect_true(graph.ops.size() == 4, "attribute, input, aten and constant operators");
    expect_true(graph.ops[2]->type == "prim::Constant" && graph.ops[2]->params["value"].type == 0, "none argument becomes constant");
    expect_true(graph.ops[3]->type == "aten::linear", "aten target is normalized");
    expect_true(graph.ops[3]->inputs.size() == 3 && graph.ops[3]->inputnames[2] == "bias", "named arguments are preserved");
    expect_true(graph.get_operand("linear")->shape[1] == 2, "node output tensor metadata");

    expect_true(pnnx::import_exported_program_outputs(archive.program, graph, error) == 0, error.c_str());
    expect_true(graph.ops.size() == 6, "two graph outputs are imported");
    expect_true(graph.ops[4]->type == "pnnx.Output" && graph.ops[5]->type == "pnnx.Output", "pnnx output operators");
    expect_true(graph.get_operand("linear")->consumers.size() == 2, "tuple may return the same tensor twice");

    pnnx::pt2::ExportedProgram argument_program;
    pnnx::pt2::Node argument_node;
    argument_node.name = "argument_test";
    argument_node.target = "torch.ops.aten.argument_test.default";
    pnnx::pt2::NamedArgument tensor_list;
    tensor_list.name = "tensors";
    tensor_list.argument.type = pnnx::pt2::Argument::Tensors;
    pnnx::pt2::Argument tensor_reference;
    tensor_reference.type = pnnx::pt2::Argument::Tensor;
    tensor_reference.name = "x";
    tensor_list.argument.values.push_back(tensor_reference);
    argument_node.inputs.push_back(tensor_list);
    pnnx::pt2::NamedArgument bool_list;
    bool_list.name = "flags";
    bool_list.argument.type = pnnx::pt2::Argument::Booleans;
    pnnx::pt2::Argument flag;
    flag.type = pnnx::pt2::Argument::Boolean;
    flag.boolean = true;
    bool_list.argument.values.push_back(flag);
    argument_node.inputs.push_back(bool_list);
    pnnx::pt2::NamedArgument dtype;
    dtype.name = "dtype";
    dtype.argument.type = pnnx::pt2::Argument::ScalarType;
    dtype.argument.integer = 7;
    argument_node.inputs.push_back(dtype);
    pnnx::pt2::NamedArgument device;
    device.name = "device";
    device.argument.type = pnnx::pt2::Argument::DeviceValue;
    device.argument.device.type = "cpu";
    argument_node.inputs.push_back(device);
    pnnx::pt2::NamedArgument optional;
    optional.name = "optional";
    optional.argument.type = pnnx::pt2::Argument::OptionalTensor;
    optional.argument.values.push_back(tensor_reference);
    argument_node.inputs.push_back(optional);
    pnnx::pt2::Argument argument_output;
    argument_output.type = pnnx::pt2::Argument::Tensor;
    argument_output.name = "argument_output";
    argument_node.outputs.push_back(argument_output);
    argument_program.graph.nodes.push_back(argument_node);
    argument_program.graph.tensor_values["argument_output"] = archive.program.graph.tensor_values["x"];

    const size_t old_op_count = graph.ops.size();
    expect_true(pnnx::import_exported_program_nodes(argument_program, graph, error) == 0, error.c_str());
    expect_true(graph.ops[old_op_count]->type == "prim::ListConstruct", "tensor list construct");
    expect_true(graph.ops[old_op_count + 2]->type == "prim::ListConstruct", "bool list construct");
    expect_true(graph.ops[old_op_count + 3]->params["value"].i == 6, "pt2 dtype maps to c10 scalar type");
    expect_true(graph.ops[old_op_count + 4]->params["value"].s == "cpu", "device argument constant");

    pnnx::pt2::ExportedProgram list_program;
    pnnx::pt2::Node list_node;
    list_node.name = "split";
    list_node.target = "torch.ops.aten.split_with_sizes.default";
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
    list_program.graph.tensor_values["split_0"] = archive.program.graph.tensor_values["x"];
    list_program.graph.tensor_values["split_1"] = archive.program.graph.tensor_values["x"];
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

    expect_true(pnnx::import_exported_program_nodes(list_program, graph, error) == 0, error.c_str());
    expect_true(graph.ops.back()->outputs.size() == 2, "tensor-list node has multiple outputs");
    expect_true(pnnx::import_exported_program_outputs(list_program, graph, error) == 0, error.c_str());
    expect_true(graph.ops[graph.ops.size() - 1]->type == "pnnx.Output", "mixed constant output is imported");

    pnnx::pt2::ExportedProgramArchive dynamic_archive = make_archive();
    pnnx::pt2::SymInt dynamic_dimension;
    dynamic_dimension.type = pnnx::pt2::SymInt::Expression;
    dynamic_dimension.expression = "Add(s0, 1)";
    dynamic_dimension.has_hint = true;
    dynamic_dimension.hint = 3;
    dynamic_archive.program.graph.tensor_values["x"].sizes[0] = dynamic_dimension;
    pnnx::Graph dynamic_graph;
    expect_true(pnnx::import_exported_program_inputs(dynamic_archive, dynamic_graph, error) == 0, error.c_str());
    const pnnx::Operand* dynamic_input = dynamic_graph.get_operand("x");
    expect_true(dynamic_input->shape[0] == -233, "symbolic dimension marker");
    expect_true(dynamic_input->params.at("__shape__0").s == "Add_s0_1_", "symbolic dimension key");
    expect_true(dynamic_input->params.at("__shape_expr__0").s == "Add(s0, 1)", "symbolic expression metadata");
    expect_true(dynamic_input->params.at("__shape_hint__0").i == 3, "symbolic hint metadata");

    const char* dynamic_param = "test_load_exported_program_dynamic.param";
    const char* dynamic_bin = "test_load_exported_program_dynamic.bin";
    expect_true(dynamic_graph.save(dynamic_param, dynamic_bin) == 0, "save symbolic graph");
    pnnx::Graph loaded_dynamic_graph;
    expect_true(loaded_dynamic_graph.load(dynamic_param, dynamic_bin) == 0, "load symbolic graph");
    const pnnx::Operand* loaded_dynamic_input = loaded_dynamic_graph.get_operand("x");
    expect_true(loaded_dynamic_input && loaded_dynamic_input->shape[0] == -233, "symbolic dimension round trip");
    expect_true(loaded_dynamic_input && loaded_dynamic_input->params.at("__shape__0").s == "Add_s0_1_", "symbolic key round trip");
    remove(dynamic_param);
    remove(dynamic_bin);

    pnnx::pt2::ExportedProgram shape_program;
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
    valid_shapes.push_back(std::vector<int64_t>{5, 4});
    valid_shapes.push_back(std::vector<int64_t>{5, 4});
    expect_true(pnnx::validate_exported_program_input_shapes(shape_program, valid_shapes, error), error.c_str());
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