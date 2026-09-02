// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>

#include <string>

#include "exported_program.h"
#include "storezip.h"

static int test_failures = 0;

static void expect_true(bool value, const char* message)
{
    if (value)
        return;

    fprintf(stderr, "FAILED: %s\n", message);
    test_failures++;
}

static void test_defaults()
{
    pnnx::pt2::ExportedProgramArchive archive;
    expect_true(archive.archive_version == 0, "default archive version");
    expect_true(archive.program.graph.inputs.empty(), "default graph inputs");
    expect_true(!archive.program.graph.is_single_tensor_return, "default multiple return flag");

    pnnx::pt2::TensorMeta tensor_meta;
    expect_true(tensor_meta.scalar_type == 0, "default scalar type");
    expect_true(!tensor_meta.requires_grad, "default requires grad");
    expect_true(!tensor_meta.device.has_index, "default device index");
}

static void test_minimal_archive()
{
    pnnx::pt2::ExportedProgramArchive archive;
    archive.model_name = "model";
    archive.program.schema_version.major = 8;
    archive.program.schema_version.minor = 14;
    archive.program.opset_version["aten"] = 10;

    pnnx::pt2::SymInt batch;
    batch.type = pnnx::pt2::SymInt::Expression;
    batch.expression = "s0";
    batch.has_hint = true;
    batch.hint = 2;

    pnnx::pt2::SymInt features;
    features.integer = 16;

    pnnx::pt2::TensorMeta input_meta;
    input_meta.scalar_type = 7;
    input_meta.sizes.push_back(batch);
    input_meta.sizes.push_back(features);
    archive.program.graph.tensor_values["input"] = input_meta;

    pnnx::pt2::InputSpec input;
    input.type = pnnx::pt2::InputSpec::UserInput;
    input.argument.type = pnnx::pt2::Argument::Tensor;
    input.argument.name = "input";
    archive.program.signature.inputs.push_back(input);

    pnnx::pt2::InputSpec weight;
    weight.type = pnnx::pt2::InputSpec::Parameter;
    weight.argument.type = pnnx::pt2::Argument::Tensor;
    weight.argument.name = "weight";
    weight.target = "linear.weight";
    archive.program.signature.inputs.push_back(weight);

    pnnx::pt2::RangeConstraint constraint;
    constraint.has_min = true;
    constraint.min = 1;
    constraint.has_max = true;
    constraint.max = 8;
    archive.program.range_constraints["s0"] = constraint;

    pnnx::pt2::PayloadMeta payload;
    payload.path = "weight_0";
    payload.is_parameter = true;
    payload.has_tensor_meta = true;
    archive.state_dict["linear.weight"] = payload;

    expect_true(archive.program.graph.tensor_values["input"].sizes[0].hint == 2, "symbolic dimension hint");
    expect_true(archive.program.signature.inputs[1].target == "linear.weight", "parameter target");
    expect_true(archive.program.range_constraints["s0"].max == 8, "range constraint");
    expect_true(archive.state_dict["linear.weight"].is_parameter, "parameter payload");
}

static const char* exported_program_json = R"json({
    "graph_module": {
        "graph": {
            "inputs": [{"as_tensor":{"name":"p_linear_weight"}},{"as_tensor":{"name":"x"}}],
            "outputs": [{"as_tensor":{"name":"linear"}}],
            "nodes": [{
                "name": "linear",
                "target": "torch.ops.aten.linear.default",
                "inputs": [
                    {"name":"input","arg":{"as_tensor":{"name":"x"}},"kind":1},
                    {"name":"weight","arg":{"as_tensor":{"name":"p_linear_weight"}},"kind":1}
                ],
                "outputs": [{"as_tensor":{"name":"linear"}}],
                "metadata": {"torch_fn":"linear"},
                "future_node_field": true
            }],
            "tensor_values": {
                "p_linear_weight":{"dtype":7,"sizes":[{"as_int":4},{"as_int":3}],"requires_grad":true,"device":{"type":"cpu","index":null},"strides":[{"as_int":3},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7},
                "x":{"dtype":7,"sizes":[{"as_expr":{"expr_str":"s0","hint":{"as_int":2}}},{"as_int":3}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":3},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7},
                "linear":{"dtype":7,"sizes":[{"as_expr":{"expr_str":"s0","hint":{"as_int":2}}},{"as_int":4}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":4},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}
            },
            "sym_int_values": {},
            "sym_bool_values": {},
            "sym_float_values": {},
            "is_single_tensor_return": true
        },
        "signature": {
            "input_specs": [
                {"parameter":{"arg":{"name":"p_linear_weight"},"parameter_name":"linear.weight"}},
                {"user_input":{"arg":{"as_tensor":{"name":"x"}}}}
            ],
            "output_specs": [{"user_output":{"arg":{"as_tensor":{"name":"linear"}}}}]
        },
        "module_call_graph": [],
        "future_graph_module_field": "ignored"
    },
    "opset_version": {"aten": 10},
    "range_constraints": {"s0":{"min_val":1,"max_val":null}},
    "schema_version": {"major":8,"minor":20},
    "torch_version": "2.12.0",
    "future_exported_program_field": {"ignored":true}
})json";

static void test_parse_exported_program()
{
    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_true(pnnx::pt2::parse_exported_program(exported_program_json, program, error), error.c_str());
    expect_true(program.schema_version.major == 8 && program.schema_version.minor == 20, "schema version");
    expect_true(program.graph.nodes.size() == 1 && program.graph.nodes[0].name == "linear", "graph node");
    expect_true(program.graph.nodes[0].inputs.size() == 2 && program.graph.nodes[0].inputs[0].kind == pnnx::pt2::NamedArgument::Positional, "named positional arguments");
    expect_true(program.signature.inputs.size() == 2 && program.signature.inputs[0].target == "linear.weight", "graph signature");
    expect_true(program.graph.tensor_values["x"].sizes[0].has_hint && program.graph.tensor_values["x"].sizes[0].hint == 2, "symbolic shape hint");
    expect_true(program.range_constraints["s0"].has_min && !program.range_constraints["s0"].has_max, "unbounded range constraint");
}

static void test_load_archive_metadata()
{
    const char* path = "test_exported_program_metadata.pt2";
    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    writer.close();

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_true(pnnx::pt2::load_exported_program_archive_metadata(path, archive, error), error.c_str());
    expect_true(archive.model_name == "model", "archive model name");
    expect_true(archive.program.graph.nodes[0].target == "torch.ops.aten.linear.default", "archive graph target");
    remove(path);
}

static void test_invalid_schema()
{
    std::string document(exported_program_json);
    const size_t position = document.find("\"major\":8");
    document.replace(position, 9, "\"major\":9");

    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_true(!pnnx::pt2::parse_exported_program(document, program, error), "unsupported schema is rejected");
    expect_true(error.find("schema_version") != std::string::npos, "schema error has field path");
}

static void test_multiple_models_are_rejected()
{
    const char* path = "test_exported_program_multiple_models.pt2";
    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    writer.write_file("package/models/another.json", exported_program_json, std::string(exported_program_json).size());
    writer.close();

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_true(!pnnx::pt2::load_exported_program_archive_metadata(path, archive, error), "multiple models are rejected");
    expect_true(error.find("exactly one") != std::string::npos, "multiple model error is explicit");
    remove(path);
}

int main()
{
    test_defaults();
    test_minimal_archive();
    test_parse_exported_program();
    test_load_archive_metadata();
    test_invalid_schema();
    test_multiple_models_are_rejected();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program test(s) failed\n", test_failures);
        return 1;
    }

    return 0;
}