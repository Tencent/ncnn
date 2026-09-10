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
        : created(0), destroyed(0), fail(false), single(false), recreate(false), fail_cpu(false)
    {
    }

    int created;
    int destroyed;
    bool fail;
    bool single;
    bool recreate;
    bool fail_cpu;
};

class TestLayer : public ncnn::Layer
{
public:
    TestLayer(LayerState* _state)
        : state(_state)
    {
        one_blob_only = state->single;
        support_vulkan = state->recreate && state->created == 1;
    }

    virtual int load_param(const ncnn::ParamDict&)
    {
        support_vulkan = false;
        return state->fail || (state->fail_cpu && state->created == 2) ? -1 : 0;
    }

private:
    LayerState* state;
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

static void register_test_layer(ncnn::Net& net, LayerState& state, bool builtin = false)
{
    net.register_custom_layer(builtin ? "Input" : "Test", create_test_layer, destroy_test_layer, &state);
}

static int check_text(const char* text, bool valid, bool builtin = false, bool fail = false, bool single = false)
{
    LayerState state;
    state.fail = fail;
    state.single = single;
    ncnn::Net net;
    register_test_layer(net, state, builtin);
    int ret = net.load_param_mem(text);
    if ((valid ? ret != 0 : ret != -1) || (!valid && !empty_net(net)))
    {
        fprintf(stderr, "test_net text ret=%d valid=%d\n%s\n", ret, valid, text);
        return -1;
    }
    net.clear();
    if (!empty_net(net) || state.created != state.destroyed)
    {
        fprintf(stderr, "test_net text cleanup created=%d destroyed=%d\n%s\n", state.created, state.destroyed, text);
        return -1;
    }
    return 0;
}

static int check_binary(const std::vector<unsigned char>& data, size_t size, bool valid, bool builtin = false, bool fail = false, bool single = false)
{
    LayerState state;
    state.fail = fail;
    state.single = single;
    ncnn::Net net;
    register_test_layer(net, state, builtin);
    int ret = load_binary(net, &data[0], size);
    if ((valid ? ret != 0 : ret != -1) || (!valid && !empty_net(net)))
    {
        fprintf(stderr, "test_net binary size=%zu ret=%d valid=%d\n", size, ret, valid);
        return -1;
    }
    net.clear();
    if (!empty_net(net) || state.created != state.destroyed)
    {
        fprintf(stderr, "test_net binary cleanup created=%d destroyed=%d\n", state.created, state.destroyed);
        return -1;
    }
    return 0;
}

static int test_text_errors()
{
    const char* cases[] =
    {
        "", "7767517", "7767517\n0 1\n", "7767517\n1 -1\n",
        "7767517\n2147483647 1\n", "7767517\n1 2147483647\n",
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
        "7767517\n1 1\nInput t 0 1 out 0=1,,2\n"
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        if (check_text(cases[i], false, strstr(cases[i], "Input") != 0))
            return -1;
    }
    if (check_text("7767517\n1 1\nTest t 0 1 out\n", false, false, true)
        || check_text("7767517\n1 1\nInput t 0 1 out\n", false, true, true)
        || check_text("7767517\n1 1\nTest t 0 1 out\n", false, false, false, true))
        return -1;

    return 0;
}

static int test_binary_errors()
{
    const int custom = ncnn::LayerType::CustomBit;
    for (int builtin = 0; builtin < 2; builtin++)
    {
        int type = builtin ? ncnn::LayerType::Input : custom;
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
            if (check_binary(valid, n, false, builtin))
                return -1;
        }
        if (check_binary(valid, valid.size(), true, builtin)
            || check_binary(valid, valid.size(), false, builtin, true))
            return -1;

        const int bad[] = {-1, INT_MIN, 2, INT_MAX};
        for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
        {
            for (int bottom = 0; bottom < 2; bottom++)
            {
                std::vector<unsigned char> data = binary_header(1, 2, type, bottom, 1 - bottom);
                append_int(data, bad[i]);
                append_int(data, -233);
                if (check_binary(data, data.size(), false, builtin))
                    return -1;
            }
        }
    }
    const int bad_counts[] = {-1, INT_MIN, INT_MAX};
    for (size_t i = 0; i < sizeof(bad_counts) / sizeof(bad_counts[0]); i++)
    {
        for (int field = 0; field < 4; field++)
        {
            int counts[] = {1, 1, 0, 1};
            counts[field] = bad_counts[i];
            std::vector<unsigned char> data = binary_header(counts[0], counts[1], custom, counts[2], counts[3]);
            if (check_binary(data, data.size(), false))
                return -1;
        }
    }
    std::vector<unsigned char> data = binary_header(1, 1, -1, 0, 1);
    append_int(data, 0);
    append_int(data, -233);
    if (check_binary(data, data.size(), false))
        return -1;

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
        bool valid;
    };
    const Case cases[] =
    {
        {"4,1,7,0,0", 4, {1, 7, 0, 0}, 1, true},
        {"4,2,7,6,0", 4, {2, 7, 6, 0}, 1, true},
        {"4,3,7,6,5", 4, {3, 7, 6, 5}, 1, true},
        {"5,3,7,6,1,5", 5, {3, 7, 6, 1, 5}, 1, true},
        {"5,4,7,6,5,4", 5, {4, 7, 6, 5, 4}, 1, true},
        {"8,1,7,0,0,2,6,5,0", 8, {1, 7, 0, 0, 2, 6, 5, 0}, 2, true},
        {"4,0,0,0,0", 4, {0, 0, 0, 0}, 1, true},
        {"4,1,0,0,0", 4, {1, 0, 0, 0}, 1, true},
        {"1,4", 1, {4}, 0, true},
        {"1,4", 1, {4}, 1, false},
        {"3,3,7,6", 3, {3, 7, 6}, 1, false},
        {"4,4,7,6,5", 4, {4, 7, 6, 5}, 1, false},
        {"5,3,7,6,1,5", 5, {3, 7, 6, 1, 5}, 2, false},
        {"4,5,7,6,5", 4, {5, 7, 6, 5}, 1, false},
        {"4,-1,7,6,5", 4, {-1, 7, 6, 5}, 1, false},
        {"4,3,-1,6,5", 4, {3, -1, 6, 5}, 1, false},
        {"5,4,2147483647,2147483647,2147483647,1", 5, {4, INT_MAX, INT_MAX, INT_MAX, 1}, 1, false},
        {"4,3,2147483647,2147483647,2147483647", 4, {3, INT_MAX, INT_MAX, INT_MAX}, 1, false}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const Case& c = cases[i];
        char text[512];
        const char* top_names[] = {"", " out", " out out2"};
        snprintf(text, sizeof(text), "7767517\n1 2\nTest t 0 %d%s -23330=%s\n", c.tops, top_names[c.tops], c.text);
        if (check_text(text, c.valid))
            return -1;
        std::vector<unsigned char> data = binary_header(1, 2, ncnn::LayerType::CustomBit, 0, c.tops);
        for (int j = 0; j < c.tops; j++) append_int(data, j);
        append_int(data, -23330);
        append_int(data, c.count);
        for (int j = 0; j < c.count; j++) append_int(data, c.values[j]);
        append_int(data, -233);
        if (check_binary(data, data.size(), c.valid))
            return -1;
    }
    if (check_text("7767517\n1 1\nTest t 0 1 out 30=4\n", false)
        || check_text("7767517\n1 1\nTest t 0 1 out 30=3.0,7.0,6.0,5.0\n", false)
        || check_text("7767517\n1 1\nTest t 0 1 out 30=3,7,6,5\n", true))
        return -1;

    return 0;
}

static int test_reload_and_names()
{
    LayerState state;
    ncnn::Net net;
    register_test_layer(net, state, true);
    const char* valid = "7767517\n1 1\nInput t 0 1 out\n";
    if (net.load_param_mem(valid) || net.input_names().size() != 1 || net.output_names().size() != 1)
        return -1;
    if (net.load_param_mem(valid) || state.created != 2 || state.destroyed != 1)
        return -1;
    if (net.load_param_mem("7767517\n1 1\nInput t 1 0") != -1 || !empty_net(net) || state.created != state.destroyed)
        return -1;
    if (net.load_param_mem(valid))
        return -1;
    net.clear();
    if (!empty_net(net) || state.created != state.destroyed)
        return -1;

    char name[257];
    memset(name, 'a', 256);
    name[255] = 0;
    char text[1024];
    snprintf(text, sizeof(text), "7767517\n1 1\nTest %s 0 1 %s\n", name, name);
    if (check_text(text, true))
        return -1;
    name[255] = 'a';
    name[256] = 0;
    snprintf(text, sizeof(text), "7767517\n1 1\nTest %s 0 1 out\n", name);
    if (check_text(text, false))
        return -1;
    snprintf(text, sizeof(text), "7767517\n1 1\nTest t 0 1 %s\n", name);
    if (check_text(text, false))
        return -1;

    // duplicate references are legal even when the count exceeds blob_count
    if (check_text("7767517\n1 1\nTest t 2 0 in in\n", true))
        return -1;
    std::vector<unsigned char> data = binary_header(1, 1, ncnn::LayerType::CustomBit, 2, 0);
    append_int(data, 0);
    append_int(data, 0);
    append_int(data, -233);
    if (check_binary(data, data.size(), true))
        return -1;

    return 0;
}

static int test_null_creators()
{
    ncnn::Net net;
    if (net.register_custom_layer(-1, create_null_layer) != -1)
        return -1;
    net.register_custom_layer("Test", create_null_layer);
    net.register_custom_layer("Input", create_null_layer);
    if (net.load_param_mem("7767517\n1 1\nTest t 0 1 out\n") != -1 || !empty_net(net))
        return -1;
    if (net.load_param_mem("7767517\n1 1\nInput t 0 1 out\n") != 0)
        return -1;
    return 0;
}

static int test_shape_layout()
{
    const char* text = "7767517\n2 2\nInput in 0 1 in -23330=5,4,7,6,5,4\nTest t 1 1 in out -23330=4,3,3,2,1\n";
    std::vector<unsigned char> data = binary_header(2, 2, ncnn::LayerType::Input, 0, 1);
    const int values[] = {0, -23330, 5, 4, 7, 6, 5, 4, -233, ncnn::LayerType::CustomBit, 1, 1, 0, 1, -23330, 4, 3, 3, 2, 1, -233};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) append_int(data, values[i]);
    for (int mode = 0; mode < 2; mode++)
    {
        LayerState state;
        ncnn::Net net;
        register_test_layer(net, state);
        int ret = mode == 0 ? net.load_param_mem(text) : load_binary(net, &data[0], data.size());
        if (ret != 0)
            return -1;
        const ncnn::Mat& in = net.layers()[1]->bottom_shapes[0];
        const ncnn::Mat& out = net.layers()[1]->top_shapes[0];
        if (in.dims != 4 || in.w != 7 || in.h != 6 || in.d != 5 || in.c != 4
            || out.dims != 3 || out.w != 3 || out.h != 2 || out.d != 1 || out.c != 1)
        {
            fprintf(stderr, "test_net shape layout mode=%d failed\n", mode);
            return -1;
        }
    }
    return 0;
}

static int test_recreate_layer()
{
    for (int mode = 0; mode < 2; mode++)
    {
        for (int builtin = 0; builtin < 2; builtin++)
        {
            for (int fail = 0; fail < 2; fail++)
            {
                LayerState state;
                state.recreate = true;
                state.fail_cpu = fail;
                ncnn::Net net;
                register_test_layer(net, state, builtin);
                std::vector<unsigned char> data = binary_header(1, 1, builtin ? ncnn::LayerType::Input : ncnn::LayerType::CustomBit, 0, 1);
                append_int(data, 0);
                append_int(data, -233);
                const char* text = builtin ? "7767517\n1 1\nInput t 0 1 out\n" : "7767517\n1 1\nTest t 0 1 out\n";
                int ret = mode == 0 ? net.load_param_mem(text) : load_binary(net, &data[0], data.size());
                if (ret != (fail ? -1 : 0) || state.created != 2 || state.destroyed != (fail ? 2 : 1) || (fail && !empty_net(net)))
                {
                    fprintf(stderr, "test_net recreate mode=%d builtin=%d fail=%d ret=%d created=%d destroyed=%d\n", mode, builtin, fail, ret, state.created, state.destroyed);
                    return -1;
                }
                net.clear();
                if (state.destroyed != 2)
                    return -1;
            }
        }
    }
    return 0;
}

static int test_external_input()
{
    LayerState state;
    ncnn::Net net;
    register_test_layer(net, state);
    if (net.load_param_mem("7767517\n1 1\nTest t 1 0 in\n"))
        return -1;
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

static int test_binary_types()
{
    LayerState state;
    ncnn::Net net;
    net.register_custom_layer(ncnn::LayerType::CustomBit | 512, create_test_layer, destroy_test_layer, &state);
    std::vector<unsigned char> data = binary_header(1, 1, 512, 0, 1);
    append_int(data, 0);
    append_int(data, -233);
    if (load_binary(net, &data[0], data.size()) != -1 || state.created != 0 || !empty_net(net))
        return -1;

    data = binary_header(1, 1, ncnn::LayerType::CustomBit | 512, 0, 1);
    append_int(data, 0);
    append_int(data, -233);
    if (load_binary(net, &data[0], data.size()) != 0 || state.created != 1)
        return -1;
    net.clear();
    if (state.destroyed != 1 || !empty_net(net))
        return -1;

    const int ids[] = {30, -23331};
    for (int i = 0; i < 2; i++)
    {
        data = binary_header(1, 1, ncnn::LayerType::CustomBit, 0, 1);
        append_int(data, 0);
        append_int(data, ids[i]);
        append_int(data, 1);
        if (i == 1) append_int(data, 1);
        append_int(data, -233);
        if (check_binary(data, data.size(), false))
            return -1;
    }

    return 0;
}

int main()
{
    return 0
           || test_text_errors()
           || test_binary_errors()
           || test_shape_hints()
           || test_reload_and_names()
           || test_null_creators()
           || test_shape_layout()
           || test_recreate_layer()
           || test_external_input()
           || test_binary_types();
}
