// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ir.h"
#include "pt2_archive.h"
#include "pt2_graph_lowering.h"
#include "pt2_program.h"
#include "pt2_weights.h"

static const pnnx::Operator* find_operator(const pnnx::Graph& graph, const std::string& name)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        if (graph.ops[i]->name == name)
            return graph.ops[i];
    }
    return 0;
}

static int check_constant(const pnnx::Operand* operand, int type, int value)
{
    if (!operand || !operand->producer || operand->producer->type != "prim::Constant")
        return -1;
    std::map<std::string, pnnx::Parameter>::const_iterator it = operand->producer->params.find("value");
    if (it == operand->producer->params.end() || it->second.type != type)
        return -1;
    return type == 0 || (type == 1 ? it->second.b == (value != 0) : it->second.i == value) ? 0 : -1;
}

static int check_topology(const pnnx::Graph& graph)
{
    std::map<const pnnx::Operator*, size_t> positions;
    std::set<std::string> operator_names;
    std::set<std::string> operand_names;
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        positions[graph.ops[i]] = i;
        if (!operator_names.insert(graph.ops[i]->name).second)
            return -1;
    }
    for (size_t i = 0; i < graph.operands.size(); i++)
    {
        const pnnx::Operand* operand = graph.operands[i];
        if (!operand_names.insert(operand->name).second || !operand->producer)
            return -1;
        for (size_t j = 0; j < operand->consumers.size(); j++)
        {
            if (positions[operand->producer] >= positions[operand->consumers[j]])
                return -1;
        }
    }
    return 0;
}

static int check_real_graph(const pnnx::Graph& graph, const std::string& name)
{
    if (check_topology(graph) != 0)
        return -1;

    if (name == "state_and_constants")
    {
        const pnnx::Operator* linear = find_operator(graph, "linear");
        const pnnx::Operand* output = graph.get_operand("add_2");
        if (!linear || linear->type != "aten::linear" || linear->inputnames != std::vector<std::string>{"input", "weight", "bias"} ||
            !output || output->type != 1 || output->shape != std::vector<int>({2, 4}))
            return -1;
        const char* attributes[] = {"weight", "bias", "persistent_buffer", "non_persistent_buffer", "tensor_constant"};
        for (size_t i = 0; i < sizeof(attributes) / sizeof(attributes[0]); i++)
        {
            const pnnx::Operator* op = find_operator(graph, attributes[i]);
            if (!op || op->type != "pnnx.Attribute" || op->attrs.find("data") == op->attrs.end())
                return -1;
        }
        const char* adds[] = {"add", "add_1", "add_2"};
        for (size_t i = 0; i < sizeof(adds) / sizeof(adds[0]); i++)
        {
            const pnnx::Operator* op = find_operator(graph, adds[i]);
            if (!op || op->type != "aten::add" || op->inputs.size() != 3 || check_constant(op->inputs[2], 2, 1) != 0)
                return -1;
        }
        return 0;
    }

    if (name == "strided_tensors")
    {
        const pnnx::Operator* matmul = find_operator(graph, "matmul");
        const pnnx::Operand* weight = graph.get_operand("p_weight");
        return matmul && matmul->type == "aten::matmul" && matmul->inputs.size() == 2 && weight && weight->shape == std::vector<int>({6, 5}) ? 0 : -1;
    }

    if (name == "structured_io")
    {
        const pnnx::Operator* add = find_operator(graph, "add");
        const pnnx::Operator* mul = find_operator(graph, "mul");
        const pnnx::Operator* mean = find_operator(graph, "mean");
        const pnnx::Operator* sum = find_operator(graph, "sum_1");
        if (!add || add->inputs.size() != 3 || check_constant(add->inputs[2], 2, 1) != 0 ||
            !mul || mul->inputs.size() != 2 || check_constant(mul->inputs[1], 2, 3) != 0 ||
            !mean || mean->inputs.size() != 2 || check_constant(mean->inputs[1], 0, 0) != 0 ||
            !sum || sum->inputs.size() != 4 || check_constant(sum->inputs[2], 1, 0) != 0 || check_constant(sum->inputs[3], 0, 0) != 0)
            return -1;
        const char* output_names[] = {"mul", "mean", "sum_1"};
        int outputs = 0;
        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            if (graph.ops[i]->type != "pnnx.Output")
                continue;
            if (outputs == 3 || graph.ops[i]->inputs.size() != 1 || graph.ops[i]->inputs[0]->name != output_names[outputs])
                return -1;
            outputs++;
        }
        return outputs == 3 ? 0 : -1;
    }
    return -1;
}

static int lower_archive(const char* path, pnnx::Graph& graph, std::string& error)
{
    pnnx::Pt2ArchiveReader archive;
    pnnx::Pt2Program program;
    pnnx::Pt2Weights weights;
    if (archive.open(path) != 0)
    {
        error = archive.error;
        return -1;
    }
    if (pnnx::load_pt2_program(archive, program) != 0)
    {
        error = program.error;
        return -1;
    }
    if (pnnx::load_pt2_weights(archive, program, weights) != 0)
    {
        error = weights.error;
        return -1;
    }
    return pnnx::lower_pt2_graph(program, weights, graph, error);
}

static pnnx::Pt2Argument tensor_argument(const std::string& name)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::Tensor;
    arg.s = name;
    return arg;
}

static pnnx::Pt2Argument optional_tensor_argument()
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::OptionalTensor;
    return arg;
}

static pnnx::Pt2Argument optional_tensors_argument(const std::vector<std::string>& values)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::OptionalTensors;
    arg.as = values;
    return arg;
}

static pnnx::Pt2Argument tensors_argument(const std::vector<std::string>& values)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::Tensors;
    arg.as = values;
    return arg;
}

static pnnx::Pt2Argument ints_argument(const std::vector<int64_t>& values)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::Ints;
    arg.ai = values;
    return arg;
}

static pnnx::Pt2Argument bool_argument(bool value)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::Bool;
    arg.b = value;
    return arg;
}

static pnnx::Pt2Argument int_argument(int64_t value)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::Int;
    arg.i = value;
    return arg;
}

static pnnx::Pt2Argument enum_argument(pnnx::Pt2Argument::Type type, int64_t value)
{
    pnnx::Pt2Argument arg;
    arg.type = type;
    arg.i = value;
    return arg;
}

static pnnx::Pt2Argument device_argument()
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::Device;
    arg.i = -1;
    arg.s = "cpu";
    return arg;
}

static pnnx::Pt2Argument sym_int_argument(const std::string& name)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::SymInt;
    arg.s = name;
    return arg;
}

static pnnx::Pt2Argument sym_float_argument(const std::string& name)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::SymFloat;
    arg.s = name;
    return arg;
}

static pnnx::Pt2Argument sym_bool_argument(const std::string& name)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::SymBool;
    arg.s = name;
    return arg;
}

static pnnx::Pt2Argument string_argument(const std::string& value)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::String;
    arg.s = value;
    return arg;
}

static pnnx::Pt2Argument sym_ints_argument(const std::vector<int64_t>& values)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::SymInts;
    for (size_t i = 0; i < values.size(); i++)
    {
        pnnx::Pt2Argument item;
        item.type = pnnx::Pt2Argument::SymInt;
        item.i = values[i];
        arg.args.push_back(item);
    }
    return arg;
}

static pnnx::Pt2Argument float_argument(double value)
{
    pnnx::Pt2Argument arg;
    arg.type = pnnx::Pt2Argument::Float;
    arg.f = value;
    return arg;
}

static pnnx::Pt2Tensor tensor_meta(const std::vector<int64_t>& shape)
{
    pnnx::Pt2Tensor tensor;
    tensor.dtype = 7;
    tensor.requires_grad = false;
    tensor.device = "cpu";
    tensor.device_index = -1;
    tensor.layout = 7;
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;)
    {
        pnnx::Pt2SymInt size;
        pnnx::Pt2SymInt step;
        size.value = shape[i];
        step.value = stride;
        tensor.sizes.insert(tensor.sizes.begin(), size);
        tensor.strides.insert(tensor.strides.begin(), step);
        stride *= shape[i];
    }
    return tensor;
}

static pnnx::Pt2NamedArgument named_argument(const char* name, const pnnx::Pt2Argument& arg)
{
    pnnx::Pt2NamedArgument named;
    named.name = name;
    named.arg = arg;
    return named;
}

static void add_user_input(pnnx::Pt2Program& program, const char* name, const std::vector<int64_t>& shape)
{
    pnnx::Pt2Argument arg = tensor_argument(name);
    program.inputs.push_back(arg);
    pnnx::Pt2InputSpec spec;
    spec.kind = pnnx::Pt2InputSpec::UserInput;
    spec.arg = arg;
    program.input_specs.push_back(spec);
    program.tensors[name] = tensor_meta(shape);
}

static pnnx::Pt2Program pilot_program()
{
    pnnx::Pt2Program program;
    program.schema_major = 8;
    program.schema_minor = 20;
    program.opset_versions["aten"] = 10;
    add_user_input(program, "x", std::vector<int64_t>{1, 3, 8, 8});
    add_user_input(program, "weight", std::vector<int64_t>{4, 3, 3, 3});

    pnnx::Pt2Node convolution;
    convolution.name = "convolution";
    convolution.target = "torch.ops.aten.convolution.default";
    convolution.inputs.push_back(named_argument("input", tensor_argument("x")));
    convolution.inputs.push_back(named_argument("weight", tensor_argument("weight")));
    convolution.inputs.push_back(named_argument("bias", optional_tensor_argument()));
    convolution.inputs.push_back(named_argument("stride", ints_argument(std::vector<int64_t>{1, 1})));
    convolution.inputs.push_back(named_argument("padding", ints_argument(std::vector<int64_t>{1, 1})));
    convolution.inputs.push_back(named_argument("dilation", ints_argument(std::vector<int64_t>{1, 1})));
    convolution.inputs.push_back(named_argument("transposed", bool_argument(false)));
    convolution.inputs.push_back(named_argument("output_padding", ints_argument(std::vector<int64_t>{0, 0})));
    convolution.inputs.push_back(named_argument("groups", int_argument(1)));
    convolution.outputs.push_back(tensor_argument("convolution_out"));
    program.nodes.push_back(convolution);
    program.tensors["convolution_out"] = tensor_meta(std::vector<int64_t>{1, 4, 8, 8});

    pnnx::Pt2Node norm;
    norm.name = "native_layer_norm";
    norm.target = "torch.ops.aten.native_layer_norm.default";
    norm.inputs.push_back(named_argument("input", tensor_argument("convolution_out")));
    norm.inputs.push_back(named_argument("normalized_shape", ints_argument(std::vector<int64_t>{4, 8, 8})));
    norm.inputs.push_back(named_argument("weight", optional_tensor_argument()));
    norm.inputs.push_back(named_argument("bias", optional_tensor_argument()));
    norm.inputs.push_back(named_argument("eps", float_argument(1e-5)));
    norm.outputs.push_back(tensor_argument("norm_out"));
    norm.outputs.push_back(tensor_argument("norm_mean"));
    norm.outputs.push_back(tensor_argument("norm_rstd"));
    program.nodes.push_back(norm);
    program.tensors["norm_out"] = tensor_meta(std::vector<int64_t>{1, 4, 8, 8});
    program.tensors["norm_mean"] = tensor_meta(std::vector<int64_t>{1, 1, 1, 1});
    program.tensors["norm_rstd"] = tensor_meta(std::vector<int64_t>{1, 1, 1, 1});

    pnnx::Pt2Node view;
    view.name = "view";
    view.target = "torch.ops.aten.view.default";
    view.inputs.push_back(named_argument("self", tensor_argument("norm_out")));
    view.inputs.push_back(named_argument("size", ints_argument(std::vector<int64_t>{1, 256})));
    view.outputs.push_back(tensor_argument("view_out"));
    program.nodes.push_back(view);
    program.tensors["view_out"] = tensor_meta(std::vector<int64_t>{1, 256});

    pnnx::Pt2Node item;
    item.name = "item";
    item.target = "torch.ops.aten.item.default";
    item.inputs.push_back(named_argument("self", tensor_argument("view_out")));
    item.outputs.push_back(sym_int_argument("item_out"));
    program.nodes.push_back(item);

    pnnx::Pt2Node index;
    index.name = "index";
    index.target = "torch.ops.aten.index.Tensor";
    index.inputs.push_back(named_argument("self", tensor_argument("view_out")));
    index.inputs.push_back(named_argument("indices", optional_tensors_argument(std::vector<std::string>{std::string()})));
    index.outputs.push_back(tensor_argument("index_out"));
    program.nodes.push_back(index);
    program.tensors["index_out"] = tensor_meta(std::vector<int64_t>{1, 256});

    program.outputs.push_back(tensor_argument("index_out"));
    program.output_specs.push_back(tensor_argument("index_out"));
    return program;
}

static int check_pilot()
{
    pnnx::Pt2Program program = pilot_program();
    pnnx::Pt2Weights weights;
    pnnx::Graph graph;
    std::string error;
    if (pnnx::lower_pt2_graph(program, weights, graph, error) != 0)
    {
        fprintf(stderr, "%s\n", error.c_str());
        return -1;
    }
    const pnnx::Operator* convolution = find_operator(graph, "convolution");
    const pnnx::Operator* norm = find_operator(graph, "native_layer_norm");
    const pnnx::Operator* view = find_operator(graph, "view");
    const pnnx::Operator* item = find_operator(graph, "item");
    const pnnx::Operator* index = find_operator(graph, "index");
    if (!convolution || convolution->type != "aten::convolution" || convolution->inputs.size() != 9 ||
        !norm || norm->type != "aten::native_layer_norm" || norm->inputs.size() != 5 || norm->outputs.size() != 3 ||
        !view || view->type != "aten::view" || view->inputs.size() != 2 ||
        !item || item->type != "aten::item" || item->inputs.size() != 1 ||
        !index || index->type != "aten::index" || index->inputs.size() != 2 || index->inputs[1]->producer->type != "prim::ListConstruct" || check_topology(graph) != 0)
        return -1;

    pnnx::Pt2Program generated_name_collision = pilot_program();
    generated_name_collision.nodes[1].name = "pnnx_0";
    pnnx::Graph generated_name_collision_graph;
    if (pnnx::lower_pt2_graph(generated_name_collision, weights, generated_name_collision_graph, error) != 0 ||
        check_topology(generated_name_collision_graph) != 0)
        return -1;

    pnnx::Pt2Program input_output_name_collision = pilot_program();
    input_output_name_collision.nodes[0].name = "pnnx_input_0";
    input_output_name_collision.nodes[1].name = "pnnx_output_0";
    pnnx::Graph input_output_name_collision_graph;
    if (pnnx::lower_pt2_graph(input_output_name_collision, weights, input_output_name_collision_graph, error) != 0 ||
        check_topology(input_output_name_collision_graph) != 0)
        return -1;

    pnnx::Pt2Program attribute_name_collision = pilot_program();
    attribute_name_collision.input_specs[1].kind = pnnx::Pt2InputSpec::Parameter;
    attribute_name_collision.input_specs[1].target = "convolution";
    pnnx::Pt2Weights collision_weights;
    collision_weights.values["convolution"] = pnnx::Attribute({4, 3, 3, 3}, std::vector<float>(4 * 3 * 3 * 3));
    pnnx::Graph attribute_name_collision_graph;
    if (pnnx::lower_pt2_graph(attribute_name_collision, collision_weights, attribute_name_collision_graph, error) != 0 ||
        check_topology(attribute_name_collision_graph) != 0)
        return -1;

    pnnx::Pt2Program unnamed_list_output = pilot_program();
    pnnx::Pt2Node chunk;
    chunk.target = "torch.ops.aten.chunk.default";
    chunk.inputs.push_back(named_argument("self", tensor_argument("index_out")));
    chunk.inputs.push_back(named_argument("chunks", int_argument(2)));
    chunk.inputs.push_back(named_argument("dim", int_argument(1)));
    chunk.outputs.push_back(tensors_argument(std::vector<std::string>{"chunk_0", "chunk_1"}));
    unnamed_list_output.nodes.push_back(chunk);
    unnamed_list_output.tensors["chunk_0"] = tensor_meta(std::vector<int64_t>{1, 128});
    unnamed_list_output.tensors["chunk_1"] = tensor_meta(std::vector<int64_t>{1, 128});
    unnamed_list_output.outputs[0] = tensor_argument("chunk_0");
    unnamed_list_output.output_specs[0] = tensor_argument("chunk_0");
    pnnx::Graph unnamed_list_output_graph;
    if (pnnx::lower_pt2_graph(unnamed_list_output, weights, unnamed_list_output_graph, error) != 0 ||
        check_topology(unnamed_list_output_graph) != 0 || find_operator(unnamed_list_output_graph, "") != 0)
        return -1;

    pnnx::Pt2Program mismatched_output = pilot_program();
    mismatched_output.nodes[0].outputs[0] = tensors_argument(std::vector<std::string>{"convolution_out"});
    pnnx::Graph mismatched_output_graph;
    if (pnnx::lower_pt2_graph(mismatched_output, weights, mismatched_output_graph, error) == 0 ||
        error.find("output type does not match dispatcher schema") == std::string::npos)
        return -1;

    pnnx::Pt2Program mismatched_input = pilot_program();
    mismatched_input.nodes[2].inputs[0].arg = int_argument(1);
    pnnx::Graph mismatched_input_graph;
    if (pnnx::lower_pt2_graph(mismatched_input, weights, mismatched_input_graph, error) == 0 ||
        error.find("argument type does not match dispatcher schema for self") == std::string::npos)
        return -1;

    pnnx::Pt2Program unknown = pilot_program();
    unknown.nodes[0].target = "torch.ops.aten.pnnx_missing.default";
    pnnx::Graph unknown_graph;
    if (pnnx::lower_pt2_graph(unknown, weights, unknown_graph, error) == 0 || error.find("node 0") == std::string::npos ||
        error.find("torch.ops.aten.pnnx_missing.default") == std::string::npos || error.find("dispatcher schema") == std::string::npos)
        return -1;

    pnnx::Pt2Program unknown_no_output = pilot_program();
    unknown_no_output.nodes[0].target = "torch.ops.aten.pnnx_missing.default";
    unknown_no_output.nodes[0].outputs.clear();
    pnnx::Graph unknown_no_output_graph;
    if (pnnx::lower_pt2_graph(unknown_no_output, weights, unknown_no_output_graph, error) == 0 ||
        error.find("operator without a supported output is unsupported") == std::string::npos)
        return -1;

    pnnx::Pt2Program oversized_integer = pilot_program();
    oversized_integer.nodes[0].inputs[8].arg = int_argument(INT64_MAX / 2);
    pnnx::Graph oversized_integer_graph;
    if (pnnx::lower_pt2_graph(oversized_integer, weights, oversized_integer_graph, error) == 0 ||
        error.find("integer argument is outside the pnnx parameter range") == std::string::npos)
        return -1;

    pnnx::Pt2Program metadata_assertion = pilot_program();
    pnnx::Pt2Node assertion;
    assertion.name = "assert_tensor_metadata";
    assertion.target = "torch.ops.aten._assert_tensor_metadata.default";
    assertion.inputs.push_back(named_argument("a", tensor_argument("index_out")));
    assertion.inputs.push_back(named_argument("size", ints_argument(std::vector<int64_t>{1, 256})));
    pnnx::Pt2Argument stride = sym_ints_argument(std::vector<int64_t>{256, 1});
    stride.args[0] = sym_int_argument("assert_stride");
    assertion.inputs.push_back(named_argument("stride", stride));
    assertion.inputs.push_back(named_argument("dtype", enum_argument(pnnx::Pt2Argument::ScalarType, 7)));
    assertion.inputs.push_back(named_argument("device", device_argument()));
    assertion.inputs.push_back(named_argument("layout", enum_argument(pnnx::Pt2Argument::Layout, 7)));
    metadata_assertion.tensors["index_out"].strides[0].symbolic = true;
    metadata_assertion.tensors["index_out"].strides[0].expression = "s0";
    metadata_assertion.sym_ints["assert_stride"] = metadata_assertion.tensors["index_out"].strides[0];
    metadata_assertion.nodes.push_back(assertion);
    pnnx::Graph metadata_assertion_graph;
    if (pnnx::lower_pt2_graph(metadata_assertion, weights, metadata_assertion_graph, error) != 0)
        return -1;

    pnnx::Pt2Program duplicate_metadata_assertion = metadata_assertion;
    duplicate_metadata_assertion.nodes.back().inputs.push_back(named_argument("dtype", enum_argument(pnnx::Pt2Argument::ScalarType, 7)));
    pnnx::Graph duplicate_metadata_assertion_graph;
    if (pnnx::lower_pt2_graph(duplicate_metadata_assertion, weights, duplicate_metadata_assertion_graph, error) == 0 ||
        error.find("duplicate argument dtype") == std::string::npos)
        return -1;

    pnnx::Pt2Program invalid_size_assertion = metadata_assertion;
    invalid_size_assertion.nodes.back().inputs[1].arg.ai[0] = 2;
    pnnx::Graph invalid_size_assertion_graph;
    if (pnnx::lower_pt2_graph(invalid_size_assertion, weights, invalid_size_assertion_graph, error) == 0 ||
        error.find("tensor metadata size assertion mismatch") == std::string::npos)
        return -1;

    pnnx::Pt2Program invalid_stride_assertion = metadata_assertion;
    invalid_stride_assertion.sym_ints["assert_stride"].expression = "s1";
    pnnx::Graph invalid_stride_assertion_graph;
    if (pnnx::lower_pt2_graph(invalid_stride_assertion, weights, invalid_stride_assertion_graph, error) == 0 ||
        error.find("tensor metadata stride assertion mismatch") == std::string::npos)
        return -1;

    pnnx::Pt2Program invalid_dtype_assertion = metadata_assertion;
    invalid_dtype_assertion.nodes.back().inputs[3].arg.i = 5;
    pnnx::Graph invalid_dtype_assertion_graph;
    if (pnnx::lower_pt2_graph(invalid_dtype_assertion, weights, invalid_dtype_assertion_graph, error) == 0 ||
        error.find("tensor metadata dtype assertion mismatch") == std::string::npos)
        return -1;

    pnnx::Pt2Program scalar_assertion = pilot_program();
    scalar_assertion.nodes[3].outputs[0] = sym_float_argument("item_start");
    scalar_assertion.sym_floats["item_start"] = "Symbol('zuf0')";
    pnnx::Pt2Node item_end = scalar_assertion.nodes[3];
    item_end.name = "item_end";
    item_end.outputs[0] = sym_float_argument("item_end");
    scalar_assertion.sym_floats["item_end"] = "Symbol('zuf1')";
    scalar_assertion.nodes.push_back(item_end);
    pnnx::Pt2Node compare;
    compare.name = "le";
    compare.target = "_operator.le";
    compare.inputs.push_back(named_argument("a", sym_float_argument("item_end")));
    compare.inputs.push_back(named_argument("b", sym_float_argument("item_start")));
    compare.outputs.push_back(sym_bool_argument("le_out"));
    scalar_assertion.sym_bools.insert("le_out");
    scalar_assertion.nodes.push_back(compare);

    pnnx::Pt2Node scalar_assert;
    scalar_assert.name = "assert_scalar";
    scalar_assert.target = "torch.ops.aten._assert_scalar.default";
    scalar_assert.inputs.push_back(named_argument("self", sym_bool_argument("le_out")));
    scalar_assert.inputs.push_back(named_argument("assert_msg", string_argument("item must be positive")));
    scalar_assertion.nodes.push_back(scalar_assert);

    pnnx::Pt2Node arange;
    arange.name = "arange";
    arange.target = "torch.ops.aten.arange.start_step";
    arange.inputs.push_back(named_argument("start", sym_float_argument("item_start")));
    arange.inputs.push_back(named_argument("end", sym_float_argument("item_end")));
    arange.inputs.push_back(named_argument("step", int_argument(-1)));
    arange.inputs.push_back(named_argument("dtype", enum_argument(pnnx::Pt2Argument::ScalarType, 5)));
    arange.inputs.push_back(named_argument("device", device_argument()));
    arange.inputs.push_back(named_argument("pin_memory", bool_argument(false)));
    arange.outputs.push_back(tensor_argument("arange_out"));
    scalar_assertion.nodes.push_back(arange);
    scalar_assertion.tensors["arange_out"] = tensor_meta(std::vector<int64_t>{1});

    pnnx::Graph scalar_assertion_graph;
    if (pnnx::lower_pt2_graph(scalar_assertion, weights, scalar_assertion_graph, error) != 0 ||
        !find_operator(scalar_assertion_graph, "le") || !find_operator(scalar_assertion_graph, "assert_scalar") ||
        find_operator(scalar_assertion_graph, "assert_scalar")->type != "pnnx.Assert" || !find_operator(scalar_assertion_graph, "arange") ||
        check_topology(scalar_assertion_graph) != 0)
        return -1;

    pnnx::Pt2Program symbolic = scalar_assertion;
    const auto add_symbolic = [&symbolic](const char* target, const char* name, const std::vector<pnnx::Pt2Argument>& inputs, const pnnx::Pt2Argument& output) {
        pnnx::Pt2Node node;
        node.target = target;
        node.name = name;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            pnnx::Pt2NamedArgument input;
            input.name.assign(1, 'a' + i);
            input.arg = inputs[i];
            node.inputs.push_back(input);
        }
        node.outputs.push_back(output);
        symbolic.nodes.push_back(node);
        if (output.type == pnnx::Pt2Argument::SymInt)
            symbolic.sym_ints[output.s] = pnnx::Pt2SymInt();
        if (output.type == pnnx::Pt2Argument::SymBool)
            symbolic.sym_bools.insert(output.s);
        if (output.type == pnnx::Pt2Argument::SymFloat)
            symbolic.sym_floats[output.s] = "Symbol('zuf2')";
    };
    add_symbolic("_operator.sub", "sym_sub", {sym_float_argument("item_start"), sym_float_argument("item_end")}, sym_float_argument("sym_sub"));
    add_symbolic("_operator.truediv", "sym_div", {sym_float_argument("item_start"), float_argument(2.)}, sym_float_argument("sym_div"));
    add_symbolic("_operator.floordiv", "sym_floor_div", {sym_float_argument("item_start"), float_argument(2.)}, sym_float_argument("sym_floor_div"));
    add_symbolic("_operator.mod", "sym_mod", {sym_float_argument("item_start"), float_argument(2.)}, sym_float_argument("sym_mod"));
    add_symbolic("_operator.pow", "sym_pow", {sym_float_argument("item_start"), float_argument(2.)}, sym_float_argument("sym_pow"));
    add_symbolic("_operator.neg", "sym_neg", {sym_float_argument("item_start")}, sym_float_argument("sym_neg"));
    add_symbolic("_operator.pos", "sym_pos", {sym_float_argument("item_start")}, sym_float_argument("sym_pos"));
    add_symbolic("_operator.ge", "sym_ge", {sym_float_argument("item_start"), sym_float_argument("item_end")}, sym_bool_argument("sym_ge"));
    add_symbolic("_operator.eq", "sym_eq", {sym_bool_argument("le_out"), sym_bool_argument("sym_ge")}, sym_bool_argument("sym_eq"));
    add_symbolic("_operator.ne", "sym_ne", {sym_float_argument("item_start"), sym_float_argument("item_end")}, sym_bool_argument("sym_ne"));
    add_symbolic("_operator.lt", "sym_lt", {sym_float_argument("item_start"), sym_float_argument("item_end")}, sym_bool_argument("sym_lt"));
    add_symbolic("_operator.gt", "sym_gt", {sym_float_argument("item_start"), sym_float_argument("item_end")}, sym_bool_argument("sym_gt"));
    add_symbolic("torch.sym_not", "sym_not", {sym_bool_argument("sym_eq")}, sym_bool_argument("sym_not"));
    add_symbolic("math.trunc", "sym_trunc", {sym_float_argument("item_start")}, sym_int_argument("sym_trunc"));
    add_symbolic("torch.sym_int", "sym_int", {sym_float_argument("item_start")}, sym_int_argument("sym_int"));
    add_symbolic("torch.sym_float", "sym_float", {sym_int_argument("sym_trunc")}, sym_float_argument("sym_float"));
    add_symbolic("torch.sym_ite", "sym_ite", {sym_bool_argument("sym_not"), sym_float_argument("item_start"), sym_float_argument("item_end")}, sym_float_argument("sym_ite"));
    add_symbolic("torch.sym_min", "sym_min", {sym_float_argument("item_start"), sym_float_argument("item_end")}, sym_float_argument("sym_min"));
    add_symbolic("torch.sym_max", "sym_max", {sym_float_argument("item_start"), sym_float_argument("item_end")}, sym_float_argument("sym_max"));
    add_symbolic("torch._sym_sqrt", "sym_sqrt", {sym_float_argument("item_start")}, sym_float_argument("sym_sqrt"));
    add_symbolic("_operator.and_", "sym_and", {sym_bool_argument("le_out"), sym_bool_argument("sym_ge")}, sym_bool_argument("sym_and"));
    add_symbolic("_operator.or_", "sym_or", {sym_bool_argument("le_out"), sym_bool_argument("sym_ge")}, sym_bool_argument("sym_or"));
    add_symbolic("_operator.lshift", "sym_lshift", {sym_int_argument("sym_trunc"), int_argument(1)}, sym_int_argument("sym_lshift"));
    add_symbolic("_operator.rshift", "sym_rshift", {sym_int_argument("sym_lshift"), int_argument(1)}, sym_int_argument("sym_rshift"));

    pnnx::Graph symbolic_graph;
    if (pnnx::lower_pt2_graph(symbolic, weights, symbolic_graph, error) != 0 || check_topology(symbolic_graph) != 0)
        return -1;
    const char* symbolic_names[] = {"sym_sub", "sym_div", "sym_floor_div", "sym_mod", "sym_pow", "sym_neg", "sym_pos", "sym_ge", "sym_eq", "sym_ne", "sym_lt", "sym_gt", "sym_not", "sym_trunc", "sym_int", "sym_float", "sym_ite", "sym_min", "sym_max", "sym_sqrt", "sym_and", "sym_or", "sym_lshift", "sym_rshift"};
    for (size_t i = 0; i < sizeof(symbolic_names) / sizeof(symbolic_names[0]); i++)
    {
        const pnnx::Operator* op = find_operator(symbolic_graph, symbolic_names[i]);
        if (!op)
            return -1;
    }

    pnnx::Pt2Program unsupported_assertion = pilot_program();
    unsupported_assertion.nodes.push_back(scalar_assert);
    pnnx::Graph unsupported_assertion_graph;
    if (pnnx::lower_pt2_graph(unsupported_assertion, weights, unsupported_assertion_graph, error) == 0 ||
        error.find("unknown symbolic value") == std::string::npos)
        return -1;

    pnnx::Pt2Program malformed = pilot_program();
    malformed.nodes[0].inputs.push_back(named_argument("unknown", int_argument(1)));
    pnnx::Graph malformed_graph;
    if (pnnx::lower_pt2_graph(malformed, weights, malformed_graph, error) == 0 || error.find("node 0") == std::string::npos || error.find("schema aten::convolution") == std::string::npos)
        return -1;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--pilot")
        return check_pilot() == 0 ? 0 : 1;
    if (argc != 3)
        return 1;

    std::string error;
    pnnx::Graph first;
    if (lower_archive(argv[1], first, error) != 0 || check_real_graph(first, argv[2]) != 0)
    {
        fprintf(stderr, "%s%s\n", error.c_str(), error.empty() ? "PT2 graph differs from expected structure" : "");
        return 1;
    }
    return 0;
}
