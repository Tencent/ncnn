// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>

#include "exported_program.h"

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

int main()
{
    test_defaults();
    test_minimal_archive();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program test(s) failed\n", test_failures);
        return 1;
    }

    return 0;
}