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

static void build_ones_like_graph(Graph& g, bool string_other)
{
    // clang-format off
    g.parse(R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
torch.ones_like         op_0        1 1 input ones_out dtype=0
prim::Constant          op_c        0 1 other value=0.5
prim::Constant          op_a        0 1 alpha value=1
aten::add               op_1        3 1 ones_out other alpha out
pnnx.Output             output      1 0 out
)PNNXIR");
        // clang-format on

        if (string_other)
    {
    // Non-scalar other must not match.
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
    // Valid f32 shape folds to an attribute with data.
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

    // Invalid shapes remain unchanged.
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

    // cuda:1 must retain its index.
    a.device_type = "cuda";
    a.device_index = 1;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda:1",
          "device: cuda:1 encoded as cuda:1");

    // cuda:0 remains distinct from an unindexed device.
    a.device_index = 0;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda:0",
          "device: cuda:0 encoded as cuda:0");

    // An unindexed device uses the bare type.
    a.device_index = -1;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda",
          "device: cuda with null index encoded as cuda");

    a.device_type = "cpu";
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cpu",
          "device: cpu with null index encoded as cpu");

    // An empty device type encodes None.
    a.device_type = "";
    CHECK(argument_to_constant(a, v) && v.type == 0,
          "device: empty device encoded as None");
}

static void test_scalar_type_argument()
{
    Parameter v;
    Pt2Argument a;
    a.type = Pt2Argument::SCALAR_TYPE;

    a.int_value = 7;
    CHECK(argument_to_constant(a, v) && v.type == 2 && v.i == 6,
          "scalar_type: serialized float32 maps to JIT float32");

    a.int_value = 5;
    CHECK(argument_to_constant(a, v) && v.type == 2 && v.i == 4,
          "scalar_type: serialized int64 maps to JIT int64");
}

static void build_adaptive_pool_graph(Graph& g)
{
    // clang-format off
    g.parse(R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(8,8)
aten::adaptive_avg_pool2d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR");
        // clang-format on

        Operator* pool
        = find_op(g, "aten::adaptive_avg_pool2d");
    pool->inputs[0]->shape = std::vector<int>{1, 3, 8, 8};
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
    g.parse(R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool2d    op_0        1 1 input out output_size=(8,8)
pnnx.Output             output      1 0 out
)PNNXIR");
    find_op(g, "nn.AdaptiveAvgPool2d")->inputs[0]->shape = std::vector<int>{1, 3, 8, 8};

    F_pt2_nn_adaptive_avg_pool2d pass;
    int opindex = 0;
    pnnx_graph_rewrite(g, &pass, opindex);
    CHECK(find_op(g, "nn.AdaptiveAvgPool2d")->params.at("output_size").ai[0] == 8
              && find_op(g, "nn.AdaptiveAvgPool2d")->params.at("output_size").ai[1] == 8,
          "adaptive_pool module: explicit size is preserved");

    Graph pt2;
    pt2.parse(R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool2d    op_0        1 1 input out output_size=(8,8)
pnnx.Output             output      1 0 out
)PNNXIR");
    Operator* pool = find_op(pt2, "nn.AdaptiveAvgPool2d");
    pool->inputs[0]->shape = std::vector<int>{1, 3, 8, 8};
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

static void test_output_spec_filter()
{
    const JsonValue specs = parse_json(R"JSON([
        {"user_output":{"arg":{"as_tensor":{"name":"out"}}}},
        {"buffer_mutation":{"arg":{"as_tensor":{"name":"mut"}}}}
    ])JSON");
    Pt2Program program;
    CHECK(parse_output_specs(specs, program.output_specs) != 0,
          "signature: mutation specs are rejected explicitly");
}

int main()
{
    test_ones_like_fold();
    test_device_argument();
    test_scalar_type_argument();
    test_adaptive_pool_source_guard();
    test_adaptive_pool_module_source_guard();
    test_storezip_zip64_roundtrip();
    test_output_spec_filter();

    if (g_failed == 0)
    {
        printf("RESULT: all pass\n");
        return 0;
    }
    printf("RESULT: %d failed\n", g_failed);
    return 1;
}
