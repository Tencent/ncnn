// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Regression harness for PT2 passes. It has no libtorch dependency.

#include "load_pt2.cpp"
#include "pt2_schema.cpp"
#include "pass_level2/F_pt2.cpp"

#include <stdio.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

using namespace pnnx;

static int g_failed = 0;

#define CHECK(cond, msg)                \
    do                                  \
    {                                   \
        if (cond)                       \
            printf("ok   %s\n", (msg)); \
        else                            \
        {                               \
            printf("FAIL %s\n", (msg)); \
            g_failed++;                 \
        }                               \
    } while (0)

static Operator* find_op(Graph& g, const char* type)
{
    for (size_t i = 0; i < g.ops.size(); i++)
    {
        if (g.ops[i]->type == type)
            return g.ops[i];
    }
    return 0;
}

static std::string load_fixture(const char* path)
{
    FILE* fp = fopen(path, "rb");
    if (!fp)
        return "";

    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    std::string data;
    if (size > 0)
    {
        data.resize(size);
        if (fread(&data[0], size, 1, fp) != 1)
            data.clear();
    }

    fclose(fp);
    return data;
}

static void build_ones_like_graph(Graph& g, bool string_other)
{
    g.parse(
        "7767517\n"
        "6 5\n"
        "pnnx.Input              input_0     0 1 input\n"
        "torch.ones_like         op_0        1 1 input ones_out dtype=0\n"
        "prim::Constant          op_c        0 1 other value=0.5\n"
        "prim::Constant          op_a        0 1 alpha value=1\n"
        "aten::add               op_1        3 1 ones_out other alpha out\n"
        "pnnx.Output             output      1 0 out\n");

    if (string_other)
    {
        for (size_t i = 0; i < g.ops.size(); i++)
        {
            if (g.ops[i]->name == "op_c")
                g.ops[i]->params["value"] = Parameter("abc");
        }
    }
}

static void run_ones_like_pass(Graph& g)
{
    F_pt2_fold_ones_like pass;
    int opindex = 0;
    pnnx_graph_rewrite(g, &pass, opindex);
}

static void test_ones_like_fold()
{
    {
        Graph g;
        build_ones_like_graph(g, false);
        Operator* add = find_op(g, "aten::add");
        add->outputs[0]->type = 1; // f32
        add->outputs[0]->shape.push_back(2);
        add->outputs[0]->shape.push_back(3);

        run_ones_like_pass(g);

        const Operator* fold = find_op(g, "pnnx.Attribute");
        CHECK(fold != 0, "ones_like: f32 static shape folded to pnnx.Attribute");
        CHECK(find_op(g, "torch.ones_like") == 0 && find_op(g, "aten::add") == 0,
              "ones_like: matched subgraph consumed");
        if (fold != 0)
        {
            std::map<std::string, Attribute>::const_iterator it = fold->attrs.find("data");
            CHECK(it != fold->attrs.end() && it->second.data.size() == 6 * sizeof(float),
                  "ones_like: attr data = 6 floats");
            CHECK(it != fold->attrs.end() && it->second.shape.size() == 2 && it->second.shape[0] == 2
                  && it->second.shape[1] == 3,
                  "ones_like: attr shape = (2,3)");
            float v0 = 0.f;
            if (it != fold->attrs.end() && it->second.data.size() >= 4)
                memcpy(&v0, it->second.data.data(), 4);
            CHECK(v0 == 1.5f, "ones_like: folded value = 1+1*0.5 = 1.5");
        }
    }

    {
        Graph g;
        build_ones_like_graph(g, false);
        find_op(g, "aten::add")->outputs[0]->type = 5; // i64
        run_ones_like_pass(g);
        CHECK(find_op(g, "pnnx.Attribute") == 0 && find_op(g, "aten::add") != 0 && g.ops.size() == 6,
              "ones_like: non-f32 output keeps original graph");
    }
    {
        Graph g;
        build_ones_like_graph(g, false);
        find_op(g, "aten::add")->outputs[0]->type = 1; // Missing or dynamic shape.
        run_ones_like_pass(g);
        CHECK(find_op(g, "pnnx.Attribute") == 0 && find_op(g, "aten::add") != 0 && g.ops.size() == 6,
              "ones_like: missing shape keeps original graph");
    }
    {
        Graph g;
        build_ones_like_graph(g, false);
        Operator* add = find_op(g, "aten::add");
        add->outputs[0]->type = 1;
        add->outputs[0]->shape.push_back(0); // Non-positive dimension.
        add->outputs[0]->shape.push_back(3);
        run_ones_like_pass(g);
        CHECK(find_op(g, "pnnx.Attribute") == 0 && find_op(g, "aten::add") != 0 && g.ops.size() == 6,
              "ones_like: non-positive dim keeps original graph");
    }
    {
        Graph g;
        build_ones_like_graph(g, true); // Non-scalar other.
        Operator* add = find_op(g, "aten::add");
        add->outputs[0]->type = 1;
        add->outputs[0]->shape.push_back(2);
        add->outputs[0]->shape.push_back(3);
        run_ones_like_pass(g);
        CHECK(find_op(g, "pnnx.Attribute") == 0 && find_op(g, "aten::add") != 0 && g.ops.size() == 6,
              "ones_like: non-scalar other keeps original graph");
    }
}

static void test_device_argument()
{
    Parameter v;
    Pt2Argument a;
    a.type = Pt2Argument::DEVICE;

    a.device_type = "cuda";
    a.device_index = 1;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda:1",
          "device: cuda:1 encoded as cuda:1");

    a.device_index = 0;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda:0",
          "device: cuda:0 encoded as cuda:0");

    a.device_index = -1;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda",
          "device: cuda with null index encoded as cuda");

    a.device_type = "cpu";
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cpu",
          "device: cpu with null index encoded as cpu");

    a.device_type = "";
    CHECK(argument_to_constant(a, v) && v.type == 0,
          "device: empty device encoded as None");
}

static void test_scalar_type_argument()
{
    Parameter v;
    Pt2Argument a;
    a.type = Pt2Argument::SCALAR_TYPE;

    const int expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15};
    for (int i = 1; i <= 13; i++)
    {
        a.int_value = i;
        CHECK(argument_to_constant(a, v) && v.type == 2 && v.i == expected[i - 1],
              "scalar_type: serialized enum maps to JIT enum");
    }

    a.int_value = 99;
    CHECK(!argument_to_constant(a, v), "scalar_type: unknown enum is rejected");
}

static void test_tensor_list_null_slot()
{
    const std::string json = load_fixture("pt2_tensor_list.json");
    CHECK(!json.empty(), "tensor list: fixture is readable");
    if (json.empty())
        return;

    const JsonValue tensors = parse_json(json);
    std::vector<Pt2TensorRef> refs;
    collect_tensor_refs(tensors, refs);
    CHECK(refs.size() == 2 && refs[0].is_none && !refs[1].is_none && refs[1].name == "index",
          "tensor list: null slot is preserved during schema parsing");

    Graph g;
    Operator* input = g.new_operator("pnnx.Input", "input");
    Operand* index = g.new_operand("index");
    index->producer = input;
    input->outputs.push_back(index);

    Operator* list = g.new_operator("prim::ListConstruct", "list");
    int pnnx_unknown_index = 0;
    CHECK(append_tensor_list_item(g, list, refs[0], "index", "indices", 0, pnnx_unknown_index)
          && append_tensor_list_item(g, list, refs[1], "index", "indices", 1, pnnx_unknown_index),
          "tensor list: None and tensor operands are created");
    CHECK(list->inputs.size() == 2 && list->inputs[0]->producer
          && list->inputs[0]->producer->type == "prim::Constant"
          && list->inputs[0]->producer->params.at("value").type == 0 && list->inputs[1] == index,
          "tensor list: None remains before its indexed tensor");
}

static void test_input_shape_override()
{
    Graph g;
    Operand* dynamic_input = g.new_operand("dynamic_input");
    dynamic_input->shape = std::vector<int> {-1, 3, -1, 8};
    apply_input_shape(dynamic_input, std::vector<int64_t> {2, 3, 11, 8});
    CHECK(dynamic_input->shape == std::vector<int>({-1, 3, -1, 8}),
          "input shape: exported symbolic dimensions remain authoritative");

    Operand* mismatched_input = g.new_operand("mismatched_input");
    mismatched_input->shape = std::vector<int> {-1, 3, -1};
    apply_input_shape(mismatched_input, std::vector<int64_t> {2, 3});
    CHECK(mismatched_input->shape == std::vector<int>({-1, 3, -1}),
          "input shape: mismatched rank leaves exported dimensions unchanged");
}

static void test_input_dtype_mapping()
{
    const int expected[] = {8, 7, 6, 4, 5, 3, 1, 2, 12, 10, 11, 9, 13};
    for (long long dtype = 1; dtype <= 13; dtype++)
        CHECK(pt2_dtype_enum_to_pnnx_type(dtype) == expected[dtype - 1], "input dtype: PT2 enum mapping");
    CHECK(pt2_dtype_enum_to_pnnx_type(0) == 0 && pt2_dtype_enum_to_pnnx_type(99) == 0,
          "input dtype: unknown enum is rejected");
}

static void build_adaptive_pool_graph(Graph& g)
{
    g.parse(
        "7767517\n"
        "5 4\n"
        "pnnx.Input              input_0     0 1 input\n"
        "prim::Constant          op_sz       0 1 output_size value=(8,8)\n"
        "aten::adaptive_avg_pool2d op_0      2 1 input output_size out\n"
        "pnnx.Output             output      1 0 out\n");

    Operator* pool = find_op(g, "aten::adaptive_avg_pool2d");
    pool->inputs[0]->shape = std::vector<int> {1, 3, 8, 8};
}

static void run_adaptive_pool_pass(Graph& g)
{
    F_pt2_adaptive_avg_pool2d pass;
    int opindex = 0;
    pnnx_graph_rewrite(g, &pass, opindex);
}

static void test_adaptive_pool_source_guard()
{
    {
        Graph g;
        build_adaptive_pool_graph(g);
        run_adaptive_pool_pass(g);
        const Operator* sz = find_op(g, "prim::Constant");
        CHECK(sz != 0 && sz->params.at("value").ai.size() == 2 && sz->params.at("value").ai[0] == 8
              && sz->params.at("value").ai[1] == 8,
              "adaptive_pool: explicit size equal to input is preserved");
    }

    {
        Graph g;
        build_adaptive_pool_graph(g);
        Parameter marker;
        marker.type = 4;
        marker.s = "11";
        find_op(g, "aten::adaptive_avg_pool2d")->params["__pt2_none_axes"] = marker;
        run_adaptive_pool_pass(g);
        const Operator* sz = find_op(g, "prim::Constant");
        CHECK(sz != 0 && sz->params.at("value").ai.size() == 2 && sz->params.at("value").ai[0] == 0
              && sz->params.at("value").ai[1] == 0,
              "adaptive_pool: PT2 marker permits None restoration");
    }

    {
        Graph g;
        build_adaptive_pool_graph(g);
        Parameter marker;
        marker.type = 4;
        marker.s = "10";
        find_op(g, "aten::adaptive_avg_pool2d")->params["__pt2_none_axes"] = marker;
        run_adaptive_pool_pass(g);
        const Operator* sz = find_op(g, "prim::Constant");
        CHECK(sz != 0 && sz->params.at("value").ai.size() == 2 && sz->params.at("value").ai[0] == 0
              && sz->params.at("value").ai[1] == 8,
              "adaptive_pool: per-axis None mask is preserved");
    }
}

static void test_adaptive_pool_module_source_guard()
{
    Graph g;
    g.parse(
        "7767517\n"
        "3 2\n"
        "pnnx.Input              input_0     0 1 input\n"
        "nn.AdaptiveAvgPool2d   op_0        1 1 input out output_size=(8,8)\n"
        "pnnx.Output             output      1 0 out\n");
    find_op(g, "nn.AdaptiveAvgPool2d")->inputs[0]->shape = std::vector<int> {1, 3, 8, 8};

    F_pt2_nn_adaptive_avg_pool2d pass;
    int opindex = 0;
    pnnx_graph_rewrite(g, &pass, opindex);
    CHECK(find_op(g, "nn.AdaptiveAvgPool2d")->params.at("output_size").ai[0] == 8
          && find_op(g, "nn.AdaptiveAvgPool2d")->params.at("output_size").ai[1] == 8,
          "adaptive_pool module: explicit size is preserved");

    Graph pt2;
    pt2.parse(
        "7767517\n"
        "3 2\n"
        "pnnx.Input              input_0     0 1 input\n"
        "nn.AdaptiveAvgPool2d   op_0        1 1 input out output_size=(8,8)\n"
        "pnnx.Output             output      1 0 out\n");
    Operator* pool = find_op(pt2, "nn.AdaptiveAvgPool2d");
    pool->inputs[0]->shape = std::vector<int> {1, 3, 8, 8};
    Parameter marker;
    marker.type = 4;
    marker.s = "10";
    pool->params["__pt2_none_axes"] = marker;
    opindex = 0;
    pnnx_graph_rewrite(pt2, &pass, opindex);
    CHECK(find_op(pt2, "nn.AdaptiveAvgPool2d")->params.at("output_size").ai[0] == 0
          && find_op(pt2, "nn.AdaptiveAvgPool2d")->params.at("output_size").ai[1] == 8,
          "adaptive_pool module: per-axis None mask is preserved");
}

static void test_storezip_zip64_roundtrip()
{
    const char* path = "test_pt2_storezip_regress.zip";
    const char payload[] = "pt2 zip64 regression";

    StoreZipWriter writer;
    CHECK(writer.open(path) == 0, "storezip: writer opens regression archive");
    CHECK(writer.write_file("payload.txt", payload, sizeof(payload) - 1) == 0,
          "storezip: writer writes regression payload");
    CHECK(writer.close() == 0, "storezip: writer closes Zip64 archive");

    StoreZipReader reader;
    CHECK(reader.open(path) == 0, "storezip: reader opens writer Zip64 archive");
    CHECK(reader.get_file_size("payload.txt") == sizeof(payload) - 1,
          "storezip: reader sees payload size");
    char loaded[sizeof(payload)] = {0};
    CHECK(reader.read_file("payload.txt", loaded) == 0 && memcmp(loaded, payload, sizeof(payload) - 1) == 0,
          "storezip: reader round-trips payload");
    reader.close();
    remove(path);

    const char* empty_path = "test_pt2_storezip_empty_regress.zip";
    StoreZipWriter empty_writer;
    CHECK(empty_writer.open(empty_path) == 0, "storezip: empty writer opens archive");
    CHECK(empty_writer.close() == 0, "storezip: empty writer closes Zip64 archive");
    StoreZipReader empty_reader;
    CHECK(empty_reader.open(empty_path) == 0 && empty_reader.get_names().empty(),
          "storezip: empty Zip64 archive is accepted");
    empty_reader.close();
    remove(empty_path);
}

static void test_storezip_eocd_validation()
{
    const char* path = "test_pt2_storezip_comment_regress.zip";
    std::vector<unsigned char> archive(22 + 28, 0);
    archive[0] = 0x50;
    archive[1] = 0x4b;
    archive[2] = 0x05;
    archive[3] = 0x06;
    archive[20] = 28;
    archive[22] = 0x50;
    archive[23] = 0x4b;
    archive[24] = 0x05;
    archive[25] = 0x06;
    archive[42] = 1;
    FILE* fp = fopen(path, "wb");
    CHECK(fp != 0 && fwrite(archive.data(), archive.size(), 1, fp) == 1,
          "storezip: writes EOCD comment regression archive");
    if (fp)
        fclose(fp);

    StoreZipReader reader;
    CHECK(reader.open(path) == 0 && reader.get_names().empty(),
          "storezip: ignores EOCD signature inside comment");
    reader.close();
    remove(path);
}

static void test_storezip_long_comment_zip64()
{
    const char* path = "test_pt2_storezip_long_comment.zip";
    const char payload[] = "zip64 long comment";
    StoreZipWriter writer;
    CHECK(writer.open(path) == 0 && writer.write_file("payload.txt", payload, sizeof(payload) - 1) == 0
          && writer.close() == 0,
          "storezip: writes Zip64 archive for long comment regression");

    FILE* fp = fopen(path, "rb");
    long size = 0;
    std::vector<unsigned char> archive;
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        archive.resize(size);
    }
    CHECK(fp != 0 && !archive.empty() && fread(archive.data(), archive.size(), 1, fp) == 1,
          "storezip: reads Zip64 archive for long comment regression");
    if (fp)
        fclose(fp);
    if (archive.empty())
    {
        remove(path);
        return;
    }

    long eocd = -1;
    for (long i = size - 22; i >= 0; i--)
    {
        if (archive[i] == 0x50 && archive[i + 1] == 0x4b && archive[i + 2] == 0x05 && archive[i + 3] == 0x06)
        {
            eocd = i;
            break;
        }
    }
    CHECK(eocd >= 0, "storezip: finds EOCD in Zip64 archive");
    if (eocd < 0)
    {
        remove(path);
        return;
    }
    const uint16_t comment_length = 65516;
    archive[eocd + 20] = (unsigned char)(comment_length & 0xff);
    archive[eocd + 21] = (unsigned char)(comment_length >> 8);
    archive.resize(archive.size() + comment_length, 0);
    fp = fopen(path, "wb");
    CHECK(fp != 0 && fwrite(archive.data(), archive.size(), 1, fp) == 1,
          "storezip: appends maximum scan-boundary comment");
    if (fp)
        fclose(fp);

    StoreZipReader reader;
    CHECK(reader.open(path) == 0 && reader.get_file_size("payload.txt") == sizeof(payload) - 1,
          "storezip: reads Zip64 locator before scan buffer");
    reader.close();
    remove(path);
}

static void test_output_spec_filter()
{
    const std::string json = load_fixture("pt2_output_specs_with_mutation.json");
    CHECK(!json.empty(), "signature: fixture is readable");
    if (json.empty())
        return;

    const JsonValue specs = parse_json(json);
    Pt2Program program;
    CHECK(parse_output_specs(specs, program.output_specs) != 0,
          "signature: mutation specs are rejected explicitly");
}

static void test_module_form_normalization()
{
    Graph graph;
    graph.parse(
        "7767517\n"
        "4 3\n"
        "pnnx.Input              input       0 1 input\n"
        "prim::Constant          kernel      0 1 kernel value=3\n"
        "aten::max_pool2d        op          2 1 input kernel out\n"
        "pnnx.Output             output      1 0 out\n");

    Operator* op = find_op(graph, "aten::max_pool2d");
    CHECK(op != 0, "module-form: finds raw aten operator before normalization");
    if (!op)
        return;

    op->params["__pt2_module_class"] = "MaxPool2d";
    op->params["__pt2_module_input_names"] = std::vector<std::string> {"input", "kernel_size"};
    normalize_pt2_module_forms(graph);

    CHECK(op->type == "nn.MaxPool2d" && op->inputs.size() == 1,
          "module-form: moves operator type and removes folded constant input");
    CHECK(op->params.find("kernel_size") != op->params.end() && op->params.at("kernel_size").type == 5
          && op->params.at("kernel_size").ai.size() == 2 && op->params.at("kernel_size").ai[0] == 3
          && op->params.at("kernel_size").ai[1] == 3,
          "module-form: folds scalar parameter in pass_level2");
}

static void test_module_form_maxpool_default_stride()
{
    Graph graph;
    graph.parse(
        "7767517\n"
        "5 4\n"
        "pnnx.Input              input       0 1 input\n"
        "prim::Constant          kernel      0 1 kernel value=3\n"
        "prim::Constant          stride      0 1 stride value=None\n"
        "aten::max_pool2d        op          3 1 input kernel stride out\n"
        "pnnx.Output             output      1 0 out\n");

    Operator* op = find_op(graph, "aten::max_pool2d");
    CHECK(op != 0, "module-form: finds max-pool operator");
    if (!op)
        return;

    op->params["__pt2_module_class"] = "MaxPool2d";
    op->params["__pt2_module_input_names"] = std::vector<std::string> {"input", "kernel_size", "stride"};
    normalize_pt2_module_forms(graph);

    CHECK(op->type == "nn.MaxPool2d" && op->inputs.size() == 1,
          "module-form: removes default max-pool stride input");
    CHECK(op->params.find("stride") != op->params.end() && op->params.at("stride").type == 5
          && op->params.at("stride").ai == op->params.at("kernel_size").ai,
          "module-form: default max-pool stride matches kernel_size");
}

static void test_window_function_fold()
{
    Graph graph;
    graph.parse(
        "7767517\n"
        "7 6\n"
        "prim::Constant          length      0 1 length value=4\n"
        "prim::Constant          dtype       0 1 dtype value=None\n"
        "prim::Constant          layout      0 1 layout value=None\n"
        "prim::Constant          device      0 1 device value=cpu\n"
        "prim::Constant          pin_memory  0 1 pin_memory value=False\n"
        "aten::hann_window       op          5 1 length dtype layout device pin_memory out\n"
        "pnnx.Output             output      1 0 out\n");

    fold_pt2_window_functions(graph);
    Operator* attr = find_op(graph, "pnnx.Attribute");
    CHECK(attr != 0 && attr->inputs.empty(), "window: static hann_window folds to pnnx.Attribute");
    if (attr)
    {
        const Attribute& data = attr->attrs.at("data");
        const std::vector<float> values = data.get_float32_data();
        CHECK(data.shape == std::vector<int>({4}) && values.size() == 4 && values[0] == 0.f
              && values[1] == 0.5f && values[2] == 1.f && values[3] == 0.5f,
              "window: folded hann_window has periodic f32 values");
    }
}

static void test_window_function_one_element_fold()
{
    Graph graph;
    graph.parse(
        "7767517\n"
        "9 8\n"
        "prim::Constant          length      0 1 length value=1\n"
        "prim::Constant          dtype       0 1 dtype value=None\n"
        "prim::Constant          layout      0 1 layout value=None\n"
        "prim::Constant          device      0 1 device value=cpu\n"
        "prim::Constant          pin_memory  0 1 pin_memory value=False\n"
        "aten::hann_window       op0         5 1 length dtype layout device pin_memory out0\n"
        "aten::hamming_window    op1         5 1 length dtype layout device pin_memory out1\n"
        "pnnx.Output             output      2 0 out0 out1\n");

    fold_pt2_window_functions(graph);
    int folded_one = 0;
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        if (graph.ops[i]->type != "pnnx.Attribute")
            continue;
        const Attribute& data = graph.ops[i]->attrs.at("data");
        const std::vector<float> values = data.get_float32_data();
        if (data.shape == std::vector<int>({1}) && values.size() == 1 && values[0] == 1.f)
            folded_one++;
    }
    CHECK(folded_one == 2, "window: one-element hann/hamming fold to {1} like PyTorch");
}

static void test_window_function_periodic_fold()
{
    Graph graph;
    graph.parse(
        "7767517\n"
        "8 7\n"
        "prim::Constant          length      0 1 length value=8\n"
        "prim::Constant          periodic    0 1 periodic value=False\n"
        "prim::Constant          dtype       0 1 dtype value=None\n"
        "prim::Constant          layout      0 1 layout value=None\n"
        "prim::Constant          device      0 1 device value=cpu\n"
        "prim::Constant          pin_memory  0 1 pin_memory value=False\n"
        "aten::hann_window       op          6 1 length periodic dtype layout device pin_memory out\n"
        "pnnx.Output             output      1 0 out\n");

    fold_pt2_window_functions(graph);
    Operator* attr = find_op(graph, "pnnx.Attribute");
    CHECK(attr != 0 && attr->inputs.empty(), "window: periodic overload folds to pnnx.Attribute");
    if (attr)
    {
        const Attribute& data = attr->attrs.at("data");
        const std::vector<float> values = data.get_float32_data();
        CHECK(data.shape == std::vector<int>({8}) && values.size() == 8
              && values[0] == 0.f && values[7] == 0.f
              && values[1] > 0.1882f && values[1] < 0.1883f,
              "window: non-periodic hann_window uses symmetric n-1 formula");
    }
}

int main()
{
    test_ones_like_fold();
    test_device_argument();
    test_scalar_type_argument();
    test_tensor_list_null_slot();
    test_input_shape_override();
    test_input_dtype_mapping();
    test_adaptive_pool_source_guard();
    test_adaptive_pool_module_source_guard();
    test_storezip_zip64_roundtrip();
    test_storezip_eocd_validation();
    test_storezip_long_comment_zip64();
    test_output_spec_filter();
    test_module_form_normalization();
    test_module_form_maxpool_default_stride();
    test_window_function_fold();
    test_window_function_one_element_fold();
    test_window_function_periodic_fold();

    if (g_failed == 0)
    {
        printf("RESULT: all pass\n");
        return 0;
    }
    printf("RESULT: %d failed\n", g_failed);
    return 1;
}
