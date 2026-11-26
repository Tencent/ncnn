// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_WEIGHT_QUANT
#include "datareader.h"
#include "net.h"
#endif

static int test_embed(int words, int num_output, int input_dim, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, input_dim);
    pd.set(2, bias);
    pd.set(3, num_output * input_dim);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(num_output * input_dim);
    if (bias)
        weights[1] = RandomMat(num_output);

    ncnn::Mat a(words);
    RandomizeInt(a, 0, input_dim);

    int ret = test_layer("Embed", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_embed failed words=%d num_output=%d input_dim=%d bias=%d\n", words, num_output, input_dim, bias);
    }

    return ret;
}

static int test_embed_0()
{
    return 0
           || test_embed(128, 128, 128, 0)
           || test_embed(128, 128, 128, 1)
           || test_embed(127, 127, 127, 0)
           || test_embed(127, 127, 127, 1)
           || test_embed(124, 124, 124, 0)
           || test_embed(124, 124, 124, 1);
}

#if NCNN_INT8
static int test_embed_int8(int words, int num_output, int input_dim, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, input_dim);
    pd.set(2, bias);
    pd.set(3, num_output * input_dim);
    pd.set(18, 2);

    std::vector<ncnn::Mat> weights(bias ? 3 : 2);
    weights[0] = RandomS8Mat(num_output * input_dim);
    if (bias)
    {
        weights[1] = RandomMat(num_output);
        weights[2] = RandomMat(1, 100.f, 200.f);
    }
    else
    {
        weights[1] = RandomMat(1, 100.f, 200.f);
    }

    ncnn::Mat a(words);
    RandomizeInt(a, 0, input_dim);

    int ret = test_layer("Embed", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_embed_int8 failed words=%d num_output=%d input_dim=%d bias=%d\n", words, num_output, input_dim, bias);
    }

    return ret;
}

static int test_embed_1()
{
    return 0
           || test_embed_int8(128, 128, 128, 0)
           || test_embed_int8(128, 128, 128, 1)
           || test_embed_int8(127, 127, 127, 0)
           || test_embed_int8(127, 127, 127, 1)
           || test_embed_int8(124, 124, 124, 0)
           || test_embed_int8(124, 124, 124, 1);
}
#endif // NCNN_INT8

#if NCNN_WEIGHT_QUANT
static int test_embed_q8_0()
{
    const char* param = "7767517\n2 2\nInput in0 0 1 in0\nEmbed embed 1 1 in0 out0 0=32 1=3 3=96 18=900\n";
    unsigned char bin[104];
    for (int i = 0; i < 96; i++)
        bin[i] = 128 + i;

    // little-endian half scales 0.5, 2, and zero, followed by two padding bytes
    const unsigned char scales[8] = {0, 0x38, 0, 0x40, 0, 0, 0, 0};
    for (int i = 0; i < 8; i++)
        bin[96 + i] = scales[i];

    ncnn::Net model;
    model.opt.num_threads = 2;

    const unsigned char* weights = bin;
    ncnn::DataReaderFromMemory reader(weights, sizeof(bin));
    if (model.load_param_mem(param) != 0 || model.load_model(reader) != 0 || weights != bin + sizeof(bin))
        return -1;

    ncnn::Mat input(4);
    int* tokens = input;
    tokens[0] = 1;
    tokens[1] = -1;
    tokens[2] = 3;
    tokens[3] = 1;
    ncnn::Extractor ex = model.create_extractor();
    ncnn::Mat output;
    if (ex.input("in0", input) != 0 || ex.extract("out0", output) != 0 || output.w != 32 || output.h != 4)
        return -1;

    const int rows[4] = {1, 0, 2, 1};
    const float d[3] = {0.5f, 2.f, 0.f};
    for (int j = 0; j < 4; j++)
    {
        const int row = rows[j];
        for (int i = 0; i < 32; i++)
        {
            if (output.row(j)[i] != (row * 32 + i - 128) * d[row])
                return -1;
        }
    }

    ncnn::Net truncated;
    weights = bin;
    ncnn::DataReaderFromMemory short_reader(weights, sizeof(bin) - 3);
    if (truncated.load_param_mem(param) != 0 || truncated.load_model(short_reader) == 0)
        return -1;

    return 0;
}
#endif // NCNN_WEIGHT_QUANT

int main()
{
    SRAND(7767517);

    return 0
           || test_embed_0()
#if NCNN_INT8
           || test_embed_1()
#endif
#if NCNN_WEIGHT_QUANT
           || test_embed_q8_0()
#endif
           ;
}
