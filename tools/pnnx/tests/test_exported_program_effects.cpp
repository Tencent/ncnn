// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <string.h>

#include <torch/csrc/jit/operator_upgraders/utils.h>

#include "exported_program_defaults.h"
#include "load_exported_program.h"

// Standalone target: use the same loader sources/LibTorch linkage as
// test_pnnx_load_exported_program; do not add this main to that executable.
using namespace pnnx::pt2;

static int failures = 0;

static void expect(bool ok, const std::string& message)
{
    if (!ok)
    {
        fprintf(stderr, "FAILED: %s\n", message.c_str());
        failures++;
    }
}

static Argument tensor(const char* name)
{
    Argument a;
    a.type = Argument::Tensor;
    a.name = name;
    return a;
}

static Argument integer(int64_t value)
{
    Argument a;
    a.type = Argument::Integer;
    a.integer = value;
    return a;
}

static NamedArgument named(const char* name, const Argument& value)
{
    NamedArgument a;
    a.name = name;
    a.argument = value;
    return a;
}

static TensorMeta metadata(int dtype = 7)
{
    TensorMeta m;
    m.scalar_type = dtype;
    m.requires_grad = false;
    m.device.type = "cpu";
    m.layout = 7;
    m.sizes.resize(1);
    m.sizes[0].integer = 2;
    m.strides.resize(1);
    m.strides[0].integer = 1;
    return m;
}

static ExportedProgram program()
{
    ExportedProgram p;
    p.schema_version.major = 8;
    p.schema_version.minor = 20;
    p.opset_version["aten"] = (int)torch::jit::getMaxOperatorVersion();
    return p;
}

static Node unary(const char* target, const char* self, const char* output)
{
    Node n;
    n.name = output;
    n.target = std::string("torch.ops.aten.") + target;
    n.inputs.push_back(named("self", tensor(self)));
    n.outputs.push_back(tensor(output));
    return n;
}

static void output(ExportedProgram& p, const char* name)
{
    p.graph.outputs.assign(1, tensor(name));
    OutputSpec spec;
    spec.argument = tensor(name);
    p.signature.outputs.assign(1, spec);
}

static ExportedProgram local(const char* operation, int dtype = 7, bool tensor_other = false)
{
    ExportedProgram p = program();
    Node allocate;
    allocate.name = "allocate";
    allocate.target = "torch.ops.aten.zeros.default";
    Argument size;
    size.type = Argument::Integers;
    size.values.push_back(integer(2));
    allocate.inputs.push_back(named("size", size));
    Argument type = integer(dtype);
    type.type = Argument::ScalarType;
    allocate.inputs.push_back(named("dtype", type));
    allocate.outputs.push_back(tensor("local"));
    p.graph.nodes.push_back(allocate);
    p.graph.tensor_values["local"] = metadata(dtype);
    if (tensor_other)
    {
        allocate.name = "allocate_other";
        allocate.outputs[0] = tensor("other");
        p.graph.nodes.push_back(allocate);
        p.graph.tensor_values["other"] = metadata(dtype);
    }
    Node n = unary(operation, "local", "result");
    // Deliberately unordered; serde also uses Tensor overloads for scalars.
    n.inputs.insert(n.inputs.begin(), named("other", tensor_other ? tensor("other") : integer(3)));
    p.graph.nodes.push_back(n);
    p.graph.tensor_values["result"] = metadata(dtype);
    output(p, "result");
    return p;
}

static void rejected(const ExportedProgram& p, const char* diagnostic)
{
    ExportedProgram normalized = p;
    std::string error;
    const bool ok = append_default_arguments(normalized, error);
    expect(!ok && error.find(diagnostic) != std::string::npos, std::string(diagnostic) + ": " + error);
    for (size_t i = 0; i < p.graph.nodes.size(); i++)
        expect(normalized.graph.nodes[i].target == p.graph.nodes[i].target && normalized.graph.nodes[i].inputs.size() == p.graph.nodes[i].inputs.size(), "failed proof must not commit partial normalization");
    pnnx::Graph ir;
    const int imported = pnnx::import_exported_program_nodes(p, ir, error);
    expect(imported != 0 && error.find(diagnostic) != std::string::npos && ir.ops.empty() && ir.operands.empty(), "public importer must reject before constructing IR: " + error);
}

static void test_arithmetic()
{
    const char* inplace[] = {"add_.Tensor", "sub_.Tensor", "mul_.Tensor", "div_.Tensor"};
    const char* functional[] = {"add", "sub", "mul", "div"};
    const int dtypes[] = {6, 7, 8, 13}; // Half, Float, Double, BFloat16 (serde).
    std::string error;
    for (int d = 0; d < 4; d++)
        for (int i = 0; i < 4; i++)
            for (int t = 0; t < 2; t++)
            {
                ExportedProgram p = local(inplace[i], dtypes[d], t != 0);
                if (!t)
                {
                    p.graph.nodes.back().inputs[0].argument.type = Argument::FloatingPoint;
                    p.graph.nodes.back().inputs[0].argument.floating_point = 1.5;
                }
                Node standalone = p.graph.nodes.back();
                expect(!normalize_exported_program_node(standalone, error, &p.graph), "metadata alone never authorizes arithmetic writes");
                const bool ok = append_default_arguments(p, error);
                expect(ok, error);
                expect(p.graph.nodes.back().target == std::string("torch.ops.aten.") + functional[i] + (t ? ".Tensor" : ".Scalar"), "exact arithmetic counterpart and overload");
                expect(append_default_arguments(p, error), "arithmetic normalization is idempotent");
                pnnx::Graph ir;
                const bool imported = pnnx::import_exported_program_nodes(p, ir, error) == 0 && pnnx::import_exported_program_outputs(p, ir, error) == 0;
                expect(imported, error);
                const pnnx::Operand* result = ir.get_operand("result");
                const pnnx::Operand* self = ir.get_operand("local");
                expect(result && self && result->type == self->type && result->producer->type == std::string("aten::") + functional[i], "functional IR preserves dtype and result identity");
            }

    // Fresh results remain allocators across a hardsigmoid-style chain.
    ExportedProgram p = local("add_.Tensor");
    Node clamp = unary("clamp_.default", "result", "clamped");
    clamp.inputs.push_back(named("min", integer(0)));
    clamp.inputs.push_back(named("max", integer(6)));
    p.graph.nodes.push_back(clamp);
    Node divide = unary("div_.Scalar", "clamped", "divided");
    divide.inputs.push_back(named("other", integer(6)));
    p.graph.nodes.push_back(divide);
    Node multiply = unary("mul_.Scalar", "divided", "scaled");
    multiply.inputs.push_back(named("other", integer(2)));
    p.graph.nodes.push_back(multiply);
    p.graph.tensor_values["clamped"] = p.graph.tensor_values["divided"] = p.graph.tensor_values["scaled"] = metadata();
    output(p, "scaled");
    expect(append_default_arguments(p, error), error);
    expect(p.graph.nodes[2].target == "torch.ops.aten.clamp.default" && p.graph.nodes[3].target == "torch.ops.aten.div.Scalar" && p.graph.nodes[4].target == "torch.ops.aten.mul.Scalar", "all chain writes become fresh functional results");
    expect(append_default_arguments(p, error), "chain normalization is idempotent");
    pnnx::Graph chain;
    expect(pnnx::import_exported_program_nodes(p, chain, error) == 0, error);

    p = local("mul_.Tensor", 7, true);
    p.graph.nodes[1].inputs[0].argument.values.clear();
    p.graph.tensor_values["other"].sizes.clear();
    p.graph.tensor_values["other"].strides.clear();
    expect(append_default_arguments(p, error), "same-dtype scalar tensor broadcasts without promotion");

    rejected(local("div_.Tensor", 5), "dtype promotion requires floating self");
    p = local("add_.Tensor", 7, true);
    p.graph.tensor_values["other"].scalar_type = 8;
    p.graph.nodes[1].inputs[1].argument.integer = 8;
    rejected(p, "static same-dtype other tensor");
    p = local("mul_.Tensor", 7, true);
    p.graph.tensor_values["other"].sizes[0].integer = 3;
    p.graph.nodes[1].inputs[0].argument.values[0].integer = 3;
    rejected(p, "broadcast without changing self shape");
    for (int i = 0; i < 3; i++)
    {
        p = local("add_.Tensor", 7, true);
        p.graph.tensor_values.erase(i == 0 ? "local" : i == 1 ? "result" : "other");
        rejected(p, i == 2 ? "static same-dtype other tensor" : "static input/output tensor metadata");
    }
    p = local("add_.Scalar");
    p.graph.tensor_values["result"].scalar_type = 8;
    rejected(p, "in-place return metadata disagrees");
    p = local("mul_.Scalar");
    p.graph.tensor_values["local"].sizes[0].type = p.graph.tensor_values["result"].sizes[0].type = SymInt::Expression;
    p.graph.tensor_values["local"].sizes[0].expression = p.graph.tensor_values["result"].sizes[0].expression = "s0";
    rejected(p, "static input/output tensor metadata");
    p = local("add_.Scalar");
    p.graph.nodes.back().inputs[0].argument.type = Argument::Complex;
    rejected(p, "concrete real scalar");
    p = local("add_.Scalar");
    p.graph.outputs.push_back(tensor("local"));
    rejected(p, "single-use unaliased local target");
    p = local("add_.Scalar");
    p.signature.outputs[0].argument = tensor("local");
    rejected(p, "single-use unaliased local target");
    p = local("add_.Scalar");
    p.graph.nodes.insert(p.graph.nodes.begin() + 1, unary("detach.default", "local", "alias"));
    p.graph.tensor_values["alias"] = metadata();
    p.graph.nodes.back().inputs[1].argument = tensor("alias");
    rejected(p, "single-use unaliased local target");
    p = local("add_.Scalar");
    p.graph.nodes.erase(p.graph.nodes.begin());
    p.graph.inputs.push_back(tensor("local"));
    InputSpec input;
    input.argument = tensor("local");
    p.signature.inputs.push_back(input);
    rejected(p, "single-use unaliased local target");
    p = local("add_.Scalar");
    p.graph.nodes.insert(p.graph.nodes.begin() + 1, unary("clone.default", "local", "cloned"));
    p.graph.tensor_values["cloned"] = metadata();
    p.graph.nodes.back() = unary("index_put_.default", "cloned", "result");
    Argument indices;
    indices.type = Argument::OptionalTensors;
    Argument none;
    none.type = Argument::None;
    indices.values.push_back(none);
    p.graph.nodes.back().inputs.push_back(named("indices", indices));
    p.graph.nodes.back().inputs.push_back(named("values", tensor("local")));
    p.graph.outputs.push_back(tensor("cloned"));
    rejected(p, "alias write/mutation of argument self");
}

static ExportedProgramArchive constant_archive()
{
    ExportedProgramArchive a;
    a.archive_version = 1;
    a.program = program();
    InputSpec spec;
    spec.type = InputSpec::TensorConstant;
    spec.argument = tensor("constant");
    spec.target = "lifted_tensor_0";
    a.program.signature.inputs.push_back(spec);
    a.program.graph.inputs.push_back(spec.argument);
    a.program.graph.tensor_values["constant"] = a.program.graph.tensor_values["detached"] = metadata(5);
    a.program.graph.nodes.push_back(unary("detach_.default", "constant", "detached"));
    output(a.program, "detached");
    PayloadMeta payload;
    payload.path = "constant_0";
    payload.has_tensor_meta = true;
    payload.tensor_meta = metadata(5);
    a.constants[spec.target] = payload;
    const int64_t values[] = {2, 1};
    std::vector<char>& storage = a.constant_storages["data/constants/constant_0"];
    storage.resize(sizeof(values));
    memcpy(storage.data(), values, sizeof(values));
    return a;
}

static void test_detach_and_guards()
{
    ExportedProgramArchive archive = constant_archive();
    ExportedProgram p = archive.program;
    Node guard;
    guard.name = "constant_metadata";
    guard.target = "torch.ops.aten._assert_tensor_metadata.default";
    guard.inputs.push_back(named("a", tensor("constant")));
    Argument dtype = integer(5);
    dtype.type = Argument::ScalarType;
    guard.inputs.push_back(named("dtype", dtype));
    p.graph.nodes.insert(p.graph.nodes.begin(), guard);
    std::string error;
    Node standalone = p.graph.nodes.back();
    expect(!normalize_exported_program_node(standalone, error, &p.graph), "standalone detach_ stays forbidden");
    expect(append_default_arguments(p, error), error);
    expect(p.graph.nodes.back().target == "torch.ops.aten.detach.default", "constant detach_ becomes an alias, not an allocator");
    expect(append_default_arguments(p, error), "constant detach and integer guard are idempotent");
    pnnx::Graph ir;
    const bool imported = pnnx::import_exported_program_inputs(archive, ir, error) == 0 && pnnx::import_exported_program_nodes(p, ir, error) == 0 && pnnx::import_exported_program_outputs(p, ir, error) == 0;
    expect(imported, error);
    const pnnx::Operand* detached = ir.get_operand("detached");
    expect(detached && detached->producer->type == "aten::detach" && detached->producer->inputs[0] == ir.get_operand("constant"), "integer lifted constant imports with explicit detach alias");

    const InputSpec::Type denied[] = {InputSpec::UserInput, InputSpec::Parameter, InputSpec::Buffer};
    for (int i = 0; i < 3; i++)
        for (int grad = 0; grad < 2; grad++)
        {
            p = archive.program;
            p.signature.inputs[0].type = denied[i];
            p.graph.tensor_values["constant"].requires_grad = grad != 0;
            rejected(p, "rooted only in a lifted TensorConstant");
        }
    for (int i = 0; i < 3; i++)
    {
        p = archive.program;
        if (i == 2) p.graph.tensor_values.erase("constant");
        else p.graph.tensor_values[i == 0 ? "constant" : "detached"].requires_grad = true;
        rejected(p, "requires_grad=false");
    }
    p = archive.program;
    p.graph.nodes.insert(p.graph.nodes.begin(), unary("alias.default", "constant", "view"));
    p.graph.tensor_values["view"] = metadata(5);
    p.graph.nodes.back().inputs[0].argument = tensor("view");
    rejected(p, "non-view alias");
    p.graph.nodes.front().target = "torch.ops.aten.detach.default";
    expect(append_default_arguments(p, error), "known non-view detach aliases retain constant roots");
    Node fill = unary("fill_.Scalar", "detached", "filled");
    fill.inputs.push_back(named("value", integer(0)));
    p.graph.nodes.push_back(fill);
    p.graph.tensor_values["filled"] = metadata(5);
    output(p, "filled");
    rejected(p, "single-use unaliased local target");

    p = archive.program;
    p.graph.nodes.assign(1, guard);
    output(p, "constant");
    p.graph.nodes[0].inputs[1].argument.integer = 7;
    rejected(p, "metadata guard is false or unsupported");
    p.graph.nodes[0].inputs[1].argument.integer = 5;
    p.signature.inputs[0].type = InputSpec::UserInput;
    rejected(p, "guarded graph inputs require static float32");
    p.signature.inputs[0].type = InputSpec::Parameter;
    expect(append_default_arguments(p, error), "loaded integer state metadata is not a runtime UserInput restriction");
}

static ExportedProgramArchive lifted_constant_archive()
{
    ExportedProgramArchive a = constant_archive();
    ExportedProgram& p = a.program;
    p.signature.inputs[0].argument = tensor("c_lifted_tensor_0");
    p.graph.inputs[0] = p.signature.inputs[0].argument;
    InputSpec input;
    input.argument = tensor("x");
    p.signature.inputs.push_back(input);
    p.graph.inputs.push_back(input.argument);

    // Torch 2.13 repeat_interleave: int64 [3], no-grad CPU strided constant
    // -> lift_fresh_copy -> detach_ -> repeat_interleave.self_Tensor.
    TensorMeta meta = metadata(5);
    meta.sizes[0].integer = 3;
    p.graph.tensor_values.clear();
    p.graph.tensor_values["c_lifted_tensor_0"] = p.graph.tensor_values["lift_fresh_copy"] = p.graph.tensor_values["detach_"] = meta;
    a.constants[p.signature.inputs[0].target].tensor_meta = meta;
    const int64_t values[] = {1, 2, 3};
    std::vector<char>& storage = a.constant_storages["data/constants/constant_0"];
    storage.resize(sizeof(values));
    memcpy(storage.data(), values, sizeof(values));
    meta.scalar_type = 7;
    p.graph.tensor_values["x"] = meta;
    meta.sizes[0].integer = 6;
    p.graph.tensor_values["repeated"] = meta;

    p.graph.nodes.clear();
    p.graph.nodes.push_back(unary("lift_fresh_copy.default", "c_lifted_tensor_0", "lift_fresh_copy"));
    p.graph.nodes.push_back(unary("detach_.default", "lift_fresh_copy", "detach_"));
    Node repeat = unary("repeat_interleave.self_Tensor", "x", "repeated");
    repeat.inputs.push_back(named("repeats", tensor("detach_")));
    repeat.inputs.push_back(named("dim", integer(0)));
    p.graph.nodes.push_back(repeat);
    output(p, "repeated");
    return a;
}

static void test_constant_lift_fresh_copy()
{
    const ExportedProgramArchive archive = lifted_constant_archive();
    ExportedProgram p = archive.program;
    std::string error;
    Node standalone = p.graph.nodes[1];
    expect(!normalize_exported_program_node(standalone, error, &p.graph), "lift metadata alone never authorizes standalone detach_");
    expect(append_default_arguments(p, error), error);
    expect(p.graph.nodes[0].target == "torch.ops.aten.lift_fresh_copy.default"
           && p.graph.nodes[1].target == "torch.ops.aten.detach.default"
           && p.graph.nodes[2].target == "torch.ops.aten.repeat_interleave.self_Tensor", "only the proven constant-derived detach_ is rewritten");
    expect(append_default_arguments(p, error), "constant lift/detach sequence is idempotent");
    pnnx::Graph ir;
    const bool imported = pnnx::import_exported_program_inputs(archive, ir, error) == 0
                          && pnnx::import_exported_program_nodes(archive.program, ir, error) == 0
                          && pnnx::import_exported_program_outputs(archive.program, ir, error) == 0;
    expect(imported, error);
    const pnnx::Operand* constant = ir.get_operand("c_lifted_tensor_0");
    const pnnx::Operand* lifted = ir.get_operand("lift_fresh_copy");
    const pnnx::Operand* detached = ir.get_operand("detach_");
    const pnnx::Operand* repeated = ir.get_operand("repeated");
    expect(imported && constant && lifted && detached && repeated
           && lifted->producer->type == "Tensor.clone" && lifted->producer->inputs[0] == constant
           && detached->producer->type == "aten::detach" && detached->producer->inputs[0] == lifted
           && repeated->producer->type == "aten::repeat_interleave" && repeated->producer->inputs[1] == detached
           && lifted->type == constant->type && detached->type == constant->type
           && lifted->shape == constant->shape && detached->shape == constant->shape, "public importer preserves constant-copy-detach-repeat wiring and int64 metadata");

    // Previously authorized non-view detach aliases and copied roots may feed
    // another exact lift. Neither alias roots alone nor fresh ownership suffice.
    p = archive.program;
    p.graph.nodes.insert(p.graph.nodes.begin(), unary("detach.default", "c_lifted_tensor_0", "source_alias"));
    p.graph.tensor_values["source_alias"] = p.graph.tensor_values["c_lifted_tensor_0"];
    p.graph.nodes[1].inputs[0].argument = tensor("source_alias");
    expect(append_default_arguments(p, error), "non-view no-grad constant aliases may be lifted");
    p = archive.program;
    p.graph.nodes.insert(p.graph.nodes.begin() + 1, unary("lift_fresh_copy.default", "lift_fresh_copy", "lifted_again"));
    p.graph.tensor_values["lifted_again"] = p.graph.tensor_values["lift_fresh_copy"];
    p.graph.nodes[2].inputs[0].argument = tensor("lifted_again");
    expect(append_default_arguments(p, error), "proven constant-derived copies establish new authorized roots");

    const InputSpec::Type denied[] = {InputSpec::UserInput, InputSpec::Parameter, InputSpec::Buffer};
    const char* copies[] = {"torch.ops.aten.lift_fresh_copy.default", "torch.ops.aten.clone.default"};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
        {
            p = archive.program;
            p.signature.inputs[0].type = denied[i];
            p.graph.nodes[0].target = copies[j];
            rejected(p, "rooted only in a lifted TensorConstant");
            p.graph.nodes.resize(1);
            output(p, "lift_fresh_copy");
            expect(append_default_arguments(p, error), "pure copies of nonconstants remain allowed without detach_");
        }
    const char* fresh[] = {"torch.ops.aten.clone.default", "torch.ops.aten.zeros_like.default", "torch.ops.aten.neg.default"};
    for (int i = 0; i < 3; i++)
    {
        p = archive.program;
        p.graph.nodes[0].target = fresh[i];
        rejected(p, "rooted only in a lifted TensorConstant");
    }
    const char* tensors[] = {"c_lifted_tensor_0", "lift_fresh_copy", "detach_"};
    for (int i = 0; i < 3; i++)
    {
        p = archive.program;
        p.graph.tensor_values[tensors[i]].requires_grad = true;
        rejected(p, i == 2 ? "requires_grad=false" : "rooted only in a lifted TensorConstant");
        p = archive.program;
        p.graph.tensor_values.erase(tensors[i]);
        rejected(p, i == 2 ? "requires_grad=false" : "rooted only in a lifted TensorConstant");
    }
    p = archive.program;
    p.graph.nodes.insert(p.graph.nodes.begin(), unary("detach.default", "c_lifted_tensor_0", "source_alias"));
    p.graph.tensor_values["source_alias"] = p.graph.tensor_values["c_lifted_tensor_0"];
    p.graph.nodes[1].inputs[0].argument = tensor("source_alias");
    p.graph.tensor_values["c_lifted_tensor_0"].requires_grad = true;
    rejected(p, "rooted only in a lifted TensorConstant");

    for (int i = 0; i < 6; i++)
    {
        p = archive.program;
        TensorMeta& meta = p.graph.tensor_values["lift_fresh_copy"];
        if (i == 0) meta.sizes[0].integer = 4;
        if (i == 1) meta.scalar_type = 7;
        if (i == 2) meta.device.type = "cuda";
        if (i == 3) meta.device.has_index = true;
        if (i == 4) meta.layout = 0;
        if (i == 5)
        {
            meta.sizes[0].type = SymInt::Expression;
            meta.sizes[0].expression = "s0";
        }
        // Keep detach's own metadata equal so only the lift proof can reject.
        p.graph.tensor_values["detach_"] = meta;
        rejected(p, "rooted only in a lifted TensorConstant");
    }
    for (int i = 0; i < 2; i++)
    {
        p = archive.program;
        p.graph.nodes[0].outputs.clear();
        if (i)
        {
            p.graph.nodes[0].outputs.push_back(tensor("lift_fresh_copy"));
            p.graph.nodes[0].outputs.push_back(tensor("extra"));
            p.graph.tensor_values["extra"] = p.graph.tensor_values["lift_fresh_copy"];
        }
        rejected(p, "output count");
    }
    for (int i = 0; i < 2; i++)
    {
        p = archive.program;
        Node view = unary("view.default", i ? "lift_fresh_copy" : "c_lifted_tensor_0", "view");
        Argument size;
        size.type = Argument::Integers;
        size.values.push_back(integer(3));
        view.inputs.push_back(named("size", size));
        p.graph.nodes.insert(p.graph.nodes.begin() + i, view);
        p.graph.tensor_values["view"] = p.graph.tensor_values["lift_fresh_copy"];
        p.graph.nodes[i + 1].inputs[0].argument = tensor("view");
        rejected(p, "non-view alias");
    }
    const char* write_targets[] = {"c_lifted_tensor_0", "lift_fresh_copy", "detach_"};
    for (int i = 0; i < 3; i++)
    {
        p = archive.program;
        p.graph.nodes.resize(i == 2 ? 2 : 1);
        Node fill = unary("fill_.Scalar", write_targets[i], "filled");
        fill.inputs.push_back(named("value", integer(0)));
        p.graph.nodes.push_back(fill);
        p.graph.tensor_values["filled"] = p.graph.tensor_values[write_targets[i]];
        output(p, "filled");
        rejected(p, "single-use unaliased local target");
    }
}

int main()
{
    test_arithmetic();
    test_detach_and_guards();
    test_constant_lift_fresh_copy();
    return failures ? 1 : 0;
}