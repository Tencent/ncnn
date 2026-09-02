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

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program input test(s) failed\n", test_failures);
        return 1;
    }
    return 0;
}