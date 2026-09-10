// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <limits.h>
#include <string.h>

#include "datareader.h"
#include "layer_type.h"
#include "net.h"
#include "paramdict.h"

struct LayerState
{
    LayerState()
        : created(0), destroyed(0), pipeline_created(0), pipeline_destroyed(0), invalid_pipeline_destroyed(0), load_param_ret(0), cpu_load_param_ret(0), one_blob_only(false), support_vulkan(false)
    {
    }

    int created;
    int destroyed;
    int pipeline_created;
    int pipeline_destroyed;
    int invalid_pipeline_destroyed;
    int load_param_ret;
    int cpu_load_param_ret;
    bool one_blob_only;
    bool support_vulkan;
};

class TestLayer : public ncnn::Layer
{
public:
    TestLayer(LayerState* _state)
        : state(_state), param_loaded(false)
    {
        one_blob_only = state->one_blob_only;
        support_vulkan = state->support_vulkan && state->created == 1;
    }

    virtual int load_param(const ncnn::ParamDict&)
    {
        support_vulkan = false;
        if (state->created == 2 && state->cpu_load_param_ret != 0)
            return state->cpu_load_param_ret;
        param_loaded = state->load_param_ret == 0;
        return state->load_param_ret;
    }

    virtual int create_pipeline(const ncnn::Option&)
    {
        state->pipeline_created++;
        return 0;
    }

    virtual int destroy_pipeline(const ncnn::Option&)
    {
        state->pipeline_destroyed++;
        if (!param_loaded)
            state->invalid_pipeline_destroyed++;
        return 0;
    }

private:
    LayerState* state;
    bool param_loaded;
};

static ncnn::Layer* create_test_layer(void* userdata)
{
    LayerState* state = (LayerState*)userdata;
    state->created++;
    return new TestLayer(state);
}

static void destroy_test_layer(ncnn::Layer* layer, void* userdata)
{
    LayerState* state = (LayerState*)userdata;
    state->destroyed++;
    delete layer;
}

static ncnn::Layer* create_null_layer(void*)
{
    return 0;
}

static bool empty_net(const ncnn::Net& net)
{
    return net.layers().empty() && net.blobs().empty() && net.input_indexes().empty() && net.output_indexes().empty()
           && net.input_names().empty() && net.output_names().empty();
}

// memory readers without a length cannot detect truncated binary input
class BoundedNetReader : public ncnn::DataReader
{
public:
    BoundedNetReader(const unsigned char* p, size_t size)
        : data(p), remaining(size)
    {
    }

    virtual size_t read(void* buf, size_t size) const
    {
        if (size > remaining) size = remaining;
        if (size != 0)
        {
            memcpy(buf, data, size);
            data += size;
            remaining -= size;
        }
        return size;
    }

private:
    mutable const unsigned char* data;
    mutable size_t remaining;
};

static int load_binary(ncnn::Net& net, const unsigned char* data, size_t size)
{
    BoundedNetReader dr(data, size);
    return net.load_param_bin(dr);
}

static void append_int(std::vector<unsigned char>& data, int value)
{
    const unsigned int v = (unsigned int)value;
    for (int i = 0; i < 4; i++)
        data.push_back((unsigned char)(v >> (i * 8)));
}

static std::vector<unsigned char> binary_header(int layers, int blobs, int type, int bottoms, int tops)
{
    std::vector<unsigned char> data;
    append_int(data, 7767517);
    append_int(data, layers);
    append_int(data, blobs);
    append_int(data, type);
    append_int(data, bottoms);
    append_int(data, tops);
    return data;
}

static int check_text(const char* text, int expected_ret)
{
    LayerState state;
    ncnn::Net net;
    net.register_custom_layer("Test", create_test_layer, destroy_test_layer, &state);
    net.register_custom_layer("Input", create_test_layer, destroy_test_layer, &state);

    int ret = net.load_param_mem(text);
    if (ret != expected_ret || (ret != 0 && !empty_net(net)))
    {
        fprintf(stderr, "test_net text failed ret=%d expected=%d\n%s\n", ret, expected_ret, text);
        return -1;
    }

    net.clear();
    if (!empty_net(net) || state.created != state.destroyed || state.invalid_pipeline_destroyed != 0
        || state.pipeline_destroyed != (expected_ret == 0 ? state.created : 0))
    {
        fprintf(stderr, "test_net text cleanup failed created=%d destroyed=%d pipelines=%d invalid_pipelines=%d\n%s\n", state.created, state.destroyed, state.pipeline_destroyed, state.invalid_pipeline_destroyed, text);
        return -1;
    }

    return 0;
}

static int check_binary(const char* name, const std::vector<unsigned char>& data, size_t size, int expected_ret)
{
    LayerState state;
    ncnn::Net net;
    net.register_custom_layer("Test", create_test_layer, destroy_test_layer, &state);
    net.register_custom_layer("Input", create_test_layer, destroy_test_layer, &state);

    int ret = load_binary(net, &data[0], size);
    if (ret != expected_ret || (ret != 0 && !empty_net(net)))
    {
        fprintf(stderr, "test_net binary %s failed size=%zu ret=%d expected=%d\n", name, size, ret, expected_ret);
        return -1;
    }

    net.clear();
    if (!empty_net(net) || state.created != state.destroyed || state.invalid_pipeline_destroyed != 0
        || state.pipeline_destroyed != (expected_ret == 0 ? state.created : 0))
    {
        fprintf(stderr, "test_net binary %s cleanup failed created=%d destroyed=%d pipelines=%d invalid_pipelines=%d\n", name, state.created, state.destroyed, state.pipeline_destroyed, state.invalid_pipeline_destroyed);
        return -1;
    }

    return 0;
}

static int test_text_errors()
{
    const char* cases[] = {
        "", "7767517", "7767517\n0 1\n", "7767517\n1 -1\n",
        "7767517\n2147483647 1\n", "7767517\n1 2147483647\n",
        "7767517\n1000001 1\n", "7767517\n1 1000001\n",
        "7767517\n1 1\nTest t 1000001 0\n", "7767517\n1 1\nTest t 0 1000001\n",
        "7767517\n4294967297 1\nTest t 0 1 out\n",
        "7767517\n1 1\nTest t -1 1 out\n", "7767517\n1 1\nTest t 0 -1\n",
        "7767517\n1 1\nTest t 2147483647 1\n", "7767517\n1 1\nTest t 0 2147483647\n",
        "7767517\n1 1\nTest t 0 4294967297 out\n",
        "7767517\n1x 1\nTest t 0 1 out\n",
        "7767517\n1 1\nTest t 0 1x out\n",
        "7767517\n1 1\nTest t 1 1 in out\n",
        "7767517\n1 1\nTest t 2 0 in in2\n",
        "7767517\n1 1\nTest t 0 2 out out2\n",
        "7767517\n1 1\nTest t 1 0", "7767517\n1 1\nTest t 0 1",
        "7767517\n2 1\nTest t 0 1 out\n",
        "7767517\n1 1\nTest t 0 1 out 0=1,,2\n",
        "7767517\n1 1\nTest t 0 1 out 31=1.0\n",
        "7767517\n1 1\nTest t 0 1 out 31=1,2\n",
        "7767517\n1 1\nInput t 1 0",
        "7767517\n1 1\nInput t 0 1 out 0=1,,2\n"};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        if (check_text(cases[i], -1))
            return -1;
    }

    return 0;
}

static int test_binary_errors()
{
    const int custom = ncnn::LayerType::CustomBit;
    for (int builtin = 0; builtin < 2; builtin++)
    {
        const int type = builtin ? ncnn::LayerType::Input : custom;
        const char* layer_name = builtin ? "overwritten layer" : "custom layer";
        std::vector<unsigned char> valid = binary_header(2, 2, type, 0, 1);
        append_int(valid, 0);
        append_int(valid, -233);
        append_int(valid, type);
        append_int(valid, 1);
        append_int(valid, 1);
        append_int(valid, 0);
        append_int(valid, 1);
        append_int(valid, -233);
        for (size_t n = 0; n < valid.size(); n++)
        {
            if (check_binary(layer_name, valid, n, -1))
                return -1;
        }
        if (check_binary(layer_name, valid, valid.size(), 0))
            return -1;

        const int bad[] = {-1, INT_MIN, 2, INT_MAX};
        for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
        {
            for (int bottom = 0; bottom < 2; bottom++)
            {
                std::vector<unsigned char> data = binary_header(1, 2, type, bottom, 1 - bottom);
                append_int(data, bad[i]);
                append_int(data, -233);
                if (check_binary(layer_name, data, data.size(), -1))
                {
                    fprintf(stderr, "test_net blob index failed bottom=%d index=%d\n", bottom, bad[i]);
                    return -1;
                }
            }
        }
    }
    const int bad_counts[] = {-1, INT_MIN, 1000001, INT_MAX};
    for (size_t i = 0; i < sizeof(bad_counts) / sizeof(bad_counts[0]); i++)
    {
        for (int field = 0; field < 4; field++)
        {
            int counts[] = {1, 1, 0, 1};
            counts[field] = bad_counts[i];
            std::vector<unsigned char> data = binary_header(counts[0], counts[1], custom, counts[2], counts[3]);
            if (check_binary("header count", data, data.size(), -1))
            {
                fprintf(stderr, "test_net header count failed field=%d value=%d\n", field, bad_counts[i]);
                return -1;
            }
        }
    }
    std::vector<unsigned char> data = binary_header(1, 1, -1, 0, 1);
    append_int(data, 0);
    append_int(data, -233);
    if (check_binary("negative layer type", data, data.size(), -1))
        return -1;

    return 0;
}

static int test_layer_load_param_error()
{
    const char* layer_types[] = {"Test", "Input"};
    const int type_indexes[] = {ncnn::LayerType::CustomBit, ncnn::LayerType::Input};
    for (int i = 0; i < 2; i++)
    {
        for (int binary = 0; binary < 2; binary++)
        {
            LayerState state;
            state.load_param_ret = -1;
            ncnn::Net net;
            net.register_custom_layer(layer_types[i], create_test_layer, destroy_test_layer, &state);

            char text[128];
            snprintf(text, sizeof(text), "7767517\n1 1\n%s t 0 1 out\n", layer_types[i]);
            std::vector<unsigned char> data = binary_header(1, 1, type_indexes[i], 0, 1);
            append_int(data, 0);
            append_int(data, -233);

            int ret = binary ? load_binary(net, &data[0], data.size()) : net.load_param_mem(text);
            if (ret != -1 || !empty_net(net) || state.created != 1 || state.destroyed != 1 || state.pipeline_destroyed != 0)
            {
                fprintf(stderr, "test_net load_param error cleanup failed type=%s binary=%d ret=%d created=%d destroyed=%d pipelines=%d\n", layer_types[i], binary, ret, state.created, state.destroyed, state.pipeline_destroyed);
                return -1;
            }
        }
    }

    return 0;
}

static int test_one_blob_only()
{
    for (int binary = 0; binary < 2; binary++)
    {
        LayerState state;
        state.one_blob_only = true;
        ncnn::Net net;
        net.register_custom_layer("Test", create_test_layer, destroy_test_layer, &state);

        std::vector<unsigned char> data = binary_header(1, 1, ncnn::LayerType::CustomBit, 0, 1);
        append_int(data, 0);
        append_int(data, -233);
        int ret = binary ? load_binary(net, &data[0], data.size()) : net.load_param_mem("7767517\n1 1\nTest t 0 1 out\n");
        if (ret != -1 || !empty_net(net) || state.created != 1 || state.destroyed != 1 || state.pipeline_destroyed != 0)
        {
            fprintf(stderr, "test_net one_blob_only failed binary=%d ret=%d created=%d destroyed=%d pipelines=%d\n", binary, ret, state.created, state.destroyed, state.pipeline_destroyed);
            return -1;
        }
    }

    return 0;
}

static int test_shape_hints()
{
    struct Case
    {
        const char* text;
        int count;
        int values[10];
        int tops;
        int expected_ret;
    };
    const Case cases[] = {
        {"4,1,7,0,0", 4, {1, 7, 0, 0}, 1, 0},
        {"4,2,7,6,0", 4, {2, 7, 6, 0}, 1, 0},
        {"4,3,7,6,5", 4, {3, 7, 6, 5}, 1, 0},
        {"5,3,7,6,1,5", 5, {3, 7, 6, 1, 5}, 1, 0},
        {"5,4,7,6,5,4", 5, {4, 7, 6, 5, 4}, 1, 0},
        {"8,1,7,0,0,2,6,5,0", 8, {1, 7, 0, 0, 2, 6, 5, 0}, 2, 0},
        {"4,0,0,0,0", 4, {0, 0, 0, 0}, 1, 0},
        {"4,1,0,0,0", 4, {1, 0, 0, 0}, 1, 0},
        {"1,4", 1, {4}, 0, 0},
        {"1,4", 1, {4}, 1, -1},
        {"3,3,7,6", 3, {3, 7, 6}, 1, -1},
        {"4,4,7,6,5", 4, {4, 7, 6, 5}, 1, -1},
        {"5,3,7,6,1,5", 5, {3, 7, 6, 1, 5}, 2, -1},
        {"4,5,7,6,5", 4, {5, 7, 6, 5}, 1, -1},
        {"4,-1,7,6,5", 4, {-1, 7, 6, 5}, 1, -1},
        {"4,3,-1,6,5", 4, {3, -1, 6, 5}, 1, -1},
        {"5,4,2147483647,2147483647,2147483647,1", 5, {4, INT_MAX, INT_MAX, INT_MAX, 1}, 1, -1},
        {"4,3,2147483647,2147483647,2147483647", 4, {3, INT_MAX, INT_MAX, INT_MAX}, 1, -1}};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const Case& c = cases[i];
        char text[512];
        const char* top_names[] = {"", " out", " out out2"};
        snprintf(text, sizeof(text), "7767517\n1 2\nTest t 0 %d%s -23330=%s\n", c.tops, top_names[c.tops], c.text);
        if (check_text(text, c.expected_ret))
            return -1;
        std::vector<unsigned char> data = binary_header(1, 2, ncnn::LayerType::CustomBit, 0, c.tops);
        for (int j = 0; j < c.tops; j++) append_int(data, j);
        append_int(data, -23330);
        append_int(data, c.count);
        for (int j = 0; j < c.count; j++) append_int(data, c.values[j]);
        append_int(data, -233);
        if (check_binary(c.text, data, data.size(), c.expected_ret))
            return -1;
    }
    if (check_text("7767517\n1 1\nTest t 0 1 out 30=4\n", -1)
        || check_text("7767517\n1 1\nTest t 0 1 out 30=3.0,7.0,6.0,5.0\n", -1)
        || check_text("7767517\n1 1\nTest t 0 1 out 30=3,7,6,5\n", 0))
        return -1;

    return 0;
}

static int test_reload()
{
    LayerState state;
    ncnn::Net net;
    net.register_custom_layer("Input", create_test_layer, destroy_test_layer, &state);

    const char* text = "7767517\n1 1\nInput t 0 1 out\n";
    int ret = net.load_param_mem(text);
    if (ret != 0 || net.input_names().size() != 1 || net.output_names().size() != 1)
    {
        fprintf(stderr, "test_net initial load failed ret=%d inputs=%zu outputs=%zu\n", ret, net.input_names().size(), net.output_names().size());
        return -1;
    }

    ret = net.load_param_mem(text);
    if (ret != 0 || state.created != 2 || state.destroyed != 1)
    {
        fprintf(stderr, "test_net reload failed ret=%d created=%d destroyed=%d\n", ret, state.created, state.destroyed);
        return -1;
    }

    ret = net.load_param_mem("7767517\n1 1\nInput t 1 0");
    if (ret != -1 || !empty_net(net) || state.created != state.destroyed)
    {
        fprintf(stderr, "test_net failed reload cleanup failed ret=%d created=%d destroyed=%d\n", ret, state.created, state.destroyed);
        return -1;
    }

    ret = net.load_param_mem(text);
    if (ret != 0)
    {
        fprintf(stderr, "test_net reload after error failed ret=%d\n", ret);
        return -1;
    }

    net.clear();
    if (!empty_net(net) || state.created != state.destroyed)
    {
        fprintf(stderr, "test_net clear after reload failed created=%d destroyed=%d\n", state.created, state.destroyed);
        return -1;
    }

    return 0;
}

static int test_name_length()
{
    char name[257];
    memset(name, 'a', 256);
    name[255] = 0;
    char text[1024];
    snprintf(text, sizeof(text), "7767517\n1 1\nTest %s 0 1 %s\n", name, name);
    if (check_text(text, 0))
        return -1;
    name[255] = 'a';
    name[256] = 0;
    snprintf(text, sizeof(text), "7767517\n1 1\nTest %s 0 1 out\n", name);
    if (check_text(text, -1))
        return -1;
    snprintf(text, sizeof(text), "7767517\n1 1\nTest t 0 1 %s\n", name);
    if (check_text(text, -1))
        return -1;

    return 0;
}

static int test_repeated_blob_references()
{
    // duplicate references are legal even when the count exceeds blob_count
    if (check_text("7767517\n1 1\nTest t 2 0 in in\n", 0))
        return -1;
    if (check_text("7767517\n1 2\nTest t 3 0 a a b\n", 0)
        || check_text("7767517\n1 2\nTest t 3 0 a b a\n", 0)
        || check_text("7767517\n1 1\nTest t 3 0 a a b\n", -1))
        return -1;

    std::vector<unsigned char> data = binary_header(1, 1, ncnn::LayerType::CustomBit, 2, 0);
    append_int(data, 0);
    append_int(data, 0);
    append_int(data, -233);
    if (check_binary("repeated blob references", data, data.size(), 0))
        return -1;

    return 0;
}

static int test_null_creators()
{
    ncnn::Net net;
    int ret = net.register_custom_layer(-1, create_null_layer);
    if (ret != -1)
    {
        fprintf(stderr, "test_net negative custom layer index failed ret=%d\n", ret);
        return -1;
    }

    net.register_custom_layer("Test", create_null_layer);
    net.register_custom_layer("Input", create_null_layer);
    ret = net.load_param_mem("7767517\n1 1\nTest t 0 1 out\n");
    if (ret != -1 || !empty_net(net))
    {
        fprintf(stderr, "test_net null custom creator failed ret=%d\n", ret);
        return -1;
    }

    ret = net.load_param_mem("7767517\n1 1\nInput t 0 1 out\n");
    if (ret != 0)
    {
        fprintf(stderr, "test_net null overwritten creator fallback failed ret=%d\n", ret);
        return -1;
    }

    return 0;
}

static int test_shape_layout()
{
    const char* text = "7767517\n2 2\nInput in 0 1 in -23330=5,4,7,6,5,4\nTest t 1 1 in out -23330=4,3,3,2,1\n";
    std::vector<unsigned char> data = binary_header(2, 2, ncnn::LayerType::Input, 0, 1);
    const int values[] = {0, -23330, 5, 4, 7, 6, 5, 4, -233, ncnn::LayerType::CustomBit, 1, 1, 0, 1, -23330, 4, 3, 3, 2, 1, -233};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) append_int(data, values[i]);
    for (int binary = 0; binary < 2; binary++)
    {
        LayerState state;
        ncnn::Net net;
        net.register_custom_layer("Test", create_test_layer, destroy_test_layer, &state);
        int ret = binary ? load_binary(net, &data[0], data.size()) : net.load_param_mem(text);
        if (ret != 0)
        {
            fprintf(stderr, "test_net shape layout load failed binary=%d ret=%d\n", binary, ret);
            return -1;
        }

        const ncnn::Mat& in = net.layers()[1]->bottom_shapes[0];
        const ncnn::Mat& out = net.layers()[1]->top_shapes[0];
        if (in.dims != 4 || in.w != 7 || in.h != 6 || in.d != 5 || in.c != 4
            || out.dims != 3 || out.w != 3 || out.h != 2 || out.d != 1 || out.c != 1)
        {
            fprintf(stderr, "test_net shape layout binary=%d failed\n", binary);
            return -1;
        }
    }
    return 0;
}

static int test_recreate_layer()
{
    for (int binary = 0; binary < 2; binary++)
    {
        for (int builtin = 0; builtin < 2; builtin++)
        {
            for (int fail = 0; fail < 2; fail++)
            {
                LayerState state;
                state.support_vulkan = true;
                state.cpu_load_param_ret = fail ? -1 : 0;
                ncnn::Net net;
                net.register_custom_layer(builtin ? "Input" : "Test", create_test_layer, destroy_test_layer, &state);
                std::vector<unsigned char> data = binary_header(1, 1, builtin ? ncnn::LayerType::Input : ncnn::LayerType::CustomBit, 0, 1);
                append_int(data, 0);
                append_int(data, -233);
                const char* text = builtin ? "7767517\n1 1\nInput t 0 1 out\n" : "7767517\n1 1\nTest t 0 1 out\n";
                int ret = binary ? load_binary(net, &data[0], data.size()) : net.load_param_mem(text);
                if (ret != state.cpu_load_param_ret || state.created != 2 || state.destroyed != (fail ? 2 : 1) || state.pipeline_destroyed != 0 || (fail && !empty_net(net)))
                {
                    fprintf(stderr, "test_net recreate binary=%d builtin=%d fail=%d ret=%d created=%d destroyed=%d pipelines=%d\n", binary, builtin, fail, ret, state.created, state.destroyed, state.pipeline_destroyed);
                    return -1;
                }
                net.clear();
                if (state.destroyed != 2 || state.pipeline_destroyed != (fail ? 0 : 1) || state.invalid_pipeline_destroyed != 0)
                {
                    fprintf(stderr, "test_net recreate cleanup failed binary=%d builtin=%d fail=%d destroyed=%d pipelines=%d invalid_pipelines=%d\n", binary, builtin, fail, state.destroyed, state.pipeline_destroyed, state.invalid_pipeline_destroyed);
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int test_external_input()
{
    LayerState state;
    ncnn::Net net;
    net.register_custom_layer("Test", create_test_layer, destroy_test_layer, &state);
    if (net.load_param_mem("7767517\n1 1\nTest t 1 0 in\n") != 0)
    {
        fprintf(stderr, "test_net external input load failed\n");
        return -1;
    }

    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat out;
    int ret = ex.extract(0, out);
    if (ret != -1)
    {
        fprintf(stderr, "test_net missing input ret=%d\n", ret);
        return -1;
    }
    ncnn::Mat in(1);
    in[0] = 42.f;
    int input_ret = ex.input(0, in);
    ret = ex.extract(0, out);
    if (input_ret || ret || out.empty() || out[0] != 42.f)
    {
        fprintf(stderr, "test_net external input input_ret=%d ret=%d dims=%d value=%f\n", input_ret, ret, out.dims, out.empty() ? 0.f : out[0]);
        return -1;
    }
    return 0;
}

static int test_binary_layer_types()
{
    LayerState state;
    ncnn::Net net;
    net.register_custom_layer(ncnn::LayerType::CustomBit | 512, create_test_layer, destroy_test_layer, &state);
    std::vector<unsigned char> data = binary_header(1, 1, 512, 0, 1);
    append_int(data, 0);
    append_int(data, -233);
    int ret = load_binary(net, &data[0], data.size());
    if (ret != -1 || state.created != 0 || !empty_net(net))
    {
        fprintf(stderr, "test_net untagged custom layer failed ret=%d created=%d\n", ret, state.created);
        return -1;
    }

    data = binary_header(1, 1, ncnn::LayerType::CustomBit | 512, 0, 1);
    append_int(data, 0);
    append_int(data, -233);
    ret = load_binary(net, &data[0], data.size());
    if (ret != 0 || state.created != 1)
    {
        fprintf(stderr, "test_net tagged custom layer failed ret=%d created=%d\n", ret, state.created);
        return -1;
    }

    net.clear();
    if (state.destroyed != 1 || !empty_net(net))
    {
        fprintf(stderr, "test_net tagged custom layer cleanup failed destroyed=%d\n", state.destroyed);
        return -1;
    }

    return 0;
}

static int test_binary_param_types()
{
    const int ids[] = {30, -23331};
    for (int i = 0; i < 2; i++)
    {
        std::vector<unsigned char> data = binary_header(1, 1, ncnn::LayerType::CustomBit, 0, 1);
        append_int(data, 0);
        append_int(data, ids[i]);
        append_int(data, 1);
        if (i == 1) append_int(data, 1);
        append_int(data, -233);
        if (check_binary("parameter type", data, data.size(), -1))
        {
            fprintf(stderr, "test_net binary parameter type failed id=%d\n", ids[i]);
            return -1;
        }
    }

    return 0;
}

static int test_magic_mismatch()
{
    for (int binary = 0; binary < 2; binary++)
    {
        LayerState state;
        ncnn::Net net;
        net.register_custom_layer("Input", create_test_layer, destroy_test_layer, &state);

        std::vector<unsigned char> data = binary_header(1, 1, ncnn::LayerType::Input, 0, 1);
        append_int(data, 0);
        append_int(data, -233);
        int ret = binary ? load_binary(net, &data[0], data.size()) : net.load_param_mem("7767517\n1 1\nInput t 0 1 out\n");
        if (ret != 0 || net.input_indexes().size() != 1 || net.output_indexes().size() != 1)
        {
            fprintf(stderr, "test_net magic mismatch setup failed binary=%d ret=%d\n", binary, ret);
            return -1;
        }

        data[0] = 0;
        ret = binary ? load_binary(net, &data[0], data.size()) : net.load_param_mem("0\n1 1\nInput t 0 1 out\n");
        if (ret != -1 || !empty_net(net) || state.created != 1 || state.destroyed != 1 || state.pipeline_destroyed != 1 || state.invalid_pipeline_destroyed != 0)
        {
            fprintf(stderr, "test_net magic mismatch failed binary=%d ret=%d created=%d destroyed=%d pipelines=%d\n", binary, ret, state.created, state.destroyed, state.pipeline_destroyed);
            return -1;
        }
    }

    return 0;
}

static int test_feature_mask()
{
    const int masks[] = {0, 1, 2, 17, 128, 255};
    for (size_t i = 0; i < sizeof(masks) / sizeof(masks[0]); i++)
    {
        for (int binary = 0; binary < 2; binary++)
        {
            LayerState state;
            ncnn::Net net;
            net.register_custom_layer("Test", create_test_layer, destroy_test_layer, &state);

            char text[128];
            snprintf(text, sizeof(text), "7767517\n1 1\nTest t 0 1 out 31=%d\n", masks[i]);
            std::vector<unsigned char> data = binary_header(1, 1, ncnn::LayerType::CustomBit, 0, 1);
            append_int(data, 0);
            append_int(data, 31);
            append_int(data, masks[i]);
            append_int(data, -233);
            int ret = binary ? load_binary(net, &data[0], data.size()) : net.load_param_mem(text);
            if (ret != 0 || net.layers().size() != 1 || net.layers()[0]->featmask != masks[i])
            {
                fprintf(stderr, "test_net feature mask failed binary=%d mask=%d ret=%d\n", binary, masks[i], ret);
                return -1;
            }
        }
    }

    return 0;
}

static int test_pipeline_lifecycle()
{
    for (int binary = 0; binary < 2; binary++)
    {
        LayerState state;
        ncnn::Net net;
        net.register_custom_layer("Test", create_test_layer, destroy_test_layer, &state);

        std::vector<unsigned char> data = binary_header(1, 1, ncnn::LayerType::CustomBit, 0, 1);
        append_int(data, 0);
        append_int(data, -233);
        int ret = binary ? load_binary(net, &data[0], data.size()) : net.load_param_mem("7767517\n1 1\nTest t 0 1 out\n");
        if (ret != 0 || state.pipeline_created != 0 || state.pipeline_destroyed != 0)
        {
            fprintf(stderr, "test_net pipeline load_param failed binary=%d ret=%d\n", binary, ret);
            return -1;
        }

        BoundedNetReader dr(&data[0], 0);
        ret = net.load_model(dr);
        if (ret != 0 || state.pipeline_created != 1 || state.pipeline_destroyed != 0)
        {
            fprintf(stderr, "test_net pipeline load_model failed binary=%d ret=%d created=%d destroyed=%d\n", binary, ret, state.pipeline_created, state.pipeline_destroyed);
            return -1;
        }

        net.clear();
        if (!empty_net(net) || state.destroyed != 1 || state.pipeline_destroyed != 1 || state.invalid_pipeline_destroyed != 0)
        {
            fprintf(stderr, "test_net pipeline clear failed binary=%d destroyed=%d pipelines=%d invalid_pipelines=%d\n", binary, state.destroyed, state.pipeline_destroyed, state.invalid_pipeline_destroyed);
            return -1;
        }
    }

    return 0;
}

int main()
{
    return 0
           || test_text_errors()
           || test_binary_errors()
           || test_shape_hints()
           || test_layer_load_param_error()
           || test_one_blob_only()
           || test_reload()
           || test_name_length()
           || test_repeated_blob_references()
           || test_null_creators()
           || test_shape_layout()
           || test_recreate_layer()
           || test_external_input()
           || test_binary_layer_types()
           || test_binary_param_types()
           || test_magic_mismatch()
           || test_feature_mask()
           || test_pipeline_lifecycle();
}
