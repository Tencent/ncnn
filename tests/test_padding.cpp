// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_padding(const ncnn::Mat& a, int top, int bottom, int left, int right, int front, int behind, int type, float value, int per_channel_pad_data_size)
{
    ncnn::ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, value);
    pd.set(6, per_channel_pad_data_size);
    pd.set(7, front);
    pd.set(8, behind);

    std::vector<ncnn::Mat> weights(per_channel_pad_data_size ? 1 : 0);
    if (per_channel_pad_data_size)
        weights[0] = RandomMat(per_channel_pad_data_size);

    int ret = test_layer("Padding", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_padding failed a.dims=%d a=(%d %d %d %d) top=%d bottom=%d left=%d right=%d front=%d behind=%d type=%d value=%f per_channel_pad_data_size=%d\n", a.dims, a.w, a.h, a.d, a.c, top, bottom, left, right, front, behind, type, value, per_channel_pad_data_size);
    }

    return ret;
}

static int test_padding_0()
{
    ncnn::Mat a = RandomMat(9, 10, 11, 96);
    ncnn::Mat b = RandomMat(10, 12, 13, 44);
    ncnn::Mat c = RandomMat(8, 7, 9, 13);

    return 0
           || test_padding(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test_padding(a, 2, 2, 2, 2, 2, 2, 0, 1.f, 0)
           || test_padding(b, 2, 2, 2, 2, 2, 2, 0, 2.f, 0)
           || test_padding(c, 2, 2, 2, 2, 2, 2, 0, -3.f, 0)

           || test_padding(a, 2, 1, 2, 1, 2, 1, 0, 0.f, a.c)
           || test_padding(b, 2, 1, 2, 1, 2, 1, 0, 0.f, b.c)
           || test_padding(c, 2, 1, 2, 1, 2, 1, 0, 0.f, c.c)

           || test_padding(a, 0, 1, 0, 1, 0, 1, 0, 0.f, 0)
           || test_padding(b, 0, 1, 0, 1, 0, 1, 0, 0.f, 0)
           || test_padding(c, 0, 1, 0, 1, 0, 1, 0, 0.f, 0)

           || test_padding(a, 1, 2, 3, 4, 5, 6, 0, 0.f, 0)
           || test_padding(b, 1, 2, 3, 4, 5, 6, 0, 0.f, 0)
           || test_padding(c, 1, 2, 3, 4, 5, 6, 0, 0.f, 0)

           || test_padding(a, 2, 3, 2, 3, 2, 3, 0, 0.f, 0)
           || test_padding(b, 2, 3, 2, 3, 2, 3, 0, 0.f, 0)
           || test_padding(c, 2, 3, 2, 3, 2, 3, 0, 0.f, 0)

           || test_padding(a, 1, 1, 1, 1, 1, 1, 0, -1.f, 0)
           || test_padding(b, 1, 1, 1, 1, 1, 1, 0, -2.f, 0)
           || test_padding(c, 1, 1, 1, 1, 1, 1, 0, 3.f, 0);
}

static int test_padding_1()
{
    ncnn::Mat a = RandomMat(9, 11, 96);
    ncnn::Mat b = RandomMat(10, 13, 44);
    ncnn::Mat c = RandomMat(8, 9, 13);

    return 0
           || test_padding(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test_padding(a, 2, 2, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding(b, 2, 2, 2, 2, 0, 0, 0, 2.f, 0)
           || test_padding(c, 2, 2, 2, 2, 0, 0, 0, -3.f, 0)

           || test_padding(a, 2, 1, 2, 1, 0, 0, 0, 0.f, a.c)
           || test_padding(b, 2, 1, 2, 1, 0, 0, 0, 0.f, b.c)
           || test_padding(c, 2, 1, 2, 1, 0, 0, 0, 0.f, c.c)

           || test_padding(a, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding(b, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding(c, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)

           || test_padding(a, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding(b, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding(c, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)

           || test_padding(a, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding(b, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding(c, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)

           || test_padding(a, 1, 1, 1, 1, 1, 1, 0, -1.f, 0)
           || test_padding(b, 1, 1, 1, 1, 1, 1, 0, -2.f, 0)
           || test_padding(c, 1, 1, 1, 1, 1, 1, 0, 3.f, 0)

           || test_padding(a, 2, 1, 0, 0, 2, 3, 0, 0.f, a.c + 5)
           || test_padding(b, 2, 1, 0, 0, 2, 3, 0, 0.f, b.c + 5)
           || test_padding(c, 2, 1, 0, 0, 2, 3, 0, 0.f, c.c + 5)

           || test_padding(a, 1, 2, 3, 4, 32, 16, 0, 0.f, a.c + 48)
           || test_padding(b, 1, 2, 3, 4, 32, 16, 0, 0.f, b.c + 48)
           || test_padding(c, 1, 2, 3, 4, 32, 16, 0, 0.f, c.c + 48)

           || test_padding(a, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)
           || test_padding(b, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)
           || test_padding(c, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)

           || test_padding(a, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)
           || test_padding(b, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)
           || test_padding(c, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)

           || test_padding(a, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)
           || test_padding(b, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)
           || test_padding(c, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)

           || test_padding(a, 4, 2, 1, 3, 3, 5, 2, 0.f, 0)
           || test_padding(b, 4, 2, 1, 3, 3, 5, 2, 0.f, 0)
           || test_padding(c, 4, 2, 1, 3, 3, 5, 2, 0.f, 0);
}

static int test_padding_2()
{
    ncnn::Mat a = RandomMat(15, 96);
    ncnn::Mat b = RandomMat(19, 44);
    ncnn::Mat c = RandomMat(17, 15);

    return 0
           || test_padding(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test_padding(a, 0, 0, 1, 1, 0, 0, 0, 1.f, 0)
           || test_padding(b, 0, 0, 1, 1, 0, 0, 0, 2.f, 0)
           || test_padding(c, 0, 0, 1, 1, 0, 0, 0, -3.f, 0)

           || test_padding(a, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding(b, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding(c, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)

           || test_padding(a, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)
           || test_padding(b, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)
           || test_padding(c, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)

           || test_padding(a, 2, 2, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding(b, 2, 2, 2, 2, 0, 0, 0, 2.f, 0)
           || test_padding(c, 2, 2, 2, 2, 0, 0, 0, -3.f, 0)

           || test_padding(a, 16, 16, 2, 5, 0, 0, 0, -1.f, 0)
           || test_padding(b, 16, 16, 2, 5, 0, 0, 0, -2.f, 0)
           || test_padding(c, 16, 16, 2, 5, 0, 0, 0, 3.f, 0)

           || test_padding(a, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)
           || test_padding(b, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)
           || test_padding(c, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)

           || test_padding(a, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding(b, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding(c, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)

           || test_padding(a, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding(b, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding(c, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)

           || test_padding(a, 2, 6, 1, 0, 0, 0, 2, 0.f, 0)
           || test_padding(b, 2, 6, 1, 0, 0, 0, 2, 0.f, 0)
           || test_padding(c, 2, 6, 1, 0, 0, 0, 2, 0.f, 0);
}

static int test_padding_3()
{
    ncnn::Mat a = RandomMat(128);
    ncnn::Mat b = RandomMat(124);
    ncnn::Mat c = RandomMat(127);

    return 0
           || test_padding(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test_padding(a, 0, 0, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding(b, 0, 0, 2, 2, 0, 0, 0, 2.f, 0)
           || test_padding(c, 0, 0, 2, 2, 0, 0, 0, -3.f, 0)

           || test_padding(a, 0, 0, 32, 16, 0, 0, 0, -1.f, 0)
           || test_padding(b, 0, 0, 32, 16, 0, 0, 0, -2.f, 0)
           || test_padding(c, 0, 0, 32, 16, 0, 0, 0, 3.f, 0)

           || test_padding(a, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding(b, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding(c, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)

           || test_padding(a, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)
           || test_padding(b, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)
           || test_padding(c, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)

           || test_padding(a, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding(b, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding(c, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)

           || test_padding(a, 0, 0, 10, 6, 0, 0, 2, 0.f, 0)
           || test_padding(b, 0, 0, 10, 6, 0, 0, 2, 0.f, 0)
           || test_padding(c, 0, 0, 10, 6, 0, 0, 2, 0.f, 0);
}

static int test_padding_int8(const ncnn::Mat& a, int top, int bottom, int left, int right, int front, int behind, int type, float value, int per_channel_pad_data_size)
{
    // TODO enable padding int8 with per_channel_pad_data_size
    per_channel_pad_data_size = 0;

    ncnn::ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, value);
    pd.set(6, per_channel_pad_data_size);
    pd.set(7, front);
    pd.set(8, behind);

    std::vector<ncnn::Mat> weights(per_channel_pad_data_size ? 1 : 0);
    if (per_channel_pad_data_size)
        weights[0] = RandomMat(per_channel_pad_data_size);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING | TEST_LAYER_DISABLE_GPU_TESTING;
    int ret = test_layer("Padding", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_padding_int8 failed a.dims=%d a=(%d %d %d %d) top=%d bottom=%d left=%d right=%d front=%d behind=%d type=%d value=%f per_channel_pad_data_size=%d\n", a.dims, a.w, a.h, a.d, a.c, top, bottom, left, right, front, behind, type, value, per_channel_pad_data_size);
    }

    return ret;
}

static int test_padding_4()
{
    ncnn::Mat a = RandomS8Mat(9, 10, 11, 96);
    ncnn::Mat b = RandomS8Mat(10, 12, 13, 44);
    ncnn::Mat c = RandomS8Mat(8, 7, 9, 13);

    return 0
           || test_padding_int8(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding_int8(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding_int8(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test_padding_int8(a, 2, 2, 2, 2, 1, 1, 0, 1.f, 0)
           || test_padding_int8(b, 2, 2, 2, 2, 1, 1, 0, 2.f, 0)
           || test_padding_int8(c, 2, 2, 2, 2, 1, 1, 0, -3.f, 0)

           || test_padding_int8(a, 2, 1, 2, 1, 2, 1, 0, 0.f, a.c)
           || test_padding_int8(b, 2, 1, 2, 1, 2, 1, 0, 0.f, b.c)
           || test_padding_int8(c, 2, 1, 2, 1, 2, 1, 0, 0.f, c.c)

           || test_padding_int8(a, 0, 1, 0, 1, 0, 1, 0, 0.f, 0)
           || test_padding_int8(b, 0, 1, 0, 1, 0, 1, 0, 0.f, 0)
           || test_padding_int8(c, 0, 1, 0, 1, 0, 1, 0, 0.f, 0)

           || test_padding_int8(a, 1, 2, 3, 4, 5, 6, 0, 0.f, 0)
           || test_padding_int8(b, 1, 2, 3, 4, 5, 6, 0, 0.f, 0)
           || test_padding_int8(c, 1, 2, 3, 4, 5, 6, 0, 0.f, 0)

           || test_padding_int8(a, 2, 3, 2, 3, 2, 3, 0, 0.f, 0)
           || test_padding_int8(b, 2, 3, 2, 3, 2, 3, 0, 0.f, 0)
           || test_padding_int8(c, 2, 3, 2, 3, 2, 3, 0, 0.f, 0)

           || test_padding_int8(a, 1, 1, 1, 1, 1, 1, 0, -1.f, 0)
           || test_padding_int8(b, 1, 1, 1, 1, 1, 1, 0, -2.f, 0)
           || test_padding_int8(c, 1, 1, 1, 1, 1, 1, 0, 3.f, 0);
}

static int test_padding_5()
{
    ncnn::Mat a = RandomS8Mat(9, 11, 96);
    ncnn::Mat b = RandomS8Mat(10, 13, 44);
    ncnn::Mat c = RandomS8Mat(8, 9, 13);

    return 0
           || test_padding_int8(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding_int8(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding_int8(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test_padding_int8(a, 2, 2, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding_int8(b, 2, 2, 2, 2, 0, 0, 0, 2.f, 0)
           || test_padding_int8(c, 2, 2, 2, 2, 0, 0, 0, -3.f, 0)

           || test_padding_int8(a, 2, 1, 2, 1, 0, 0, 0, 0.f, a.c)
           || test_padding_int8(b, 2, 1, 2, 1, 0, 0, 0, 0.f, b.c)
           || test_padding_int8(c, 2, 1, 2, 1, 0, 0, 0, 0.f, c.c)

           || test_padding_int8(a, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding_int8(b, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding_int8(c, 0, 1, 0, 1, 0, 0, 1, 0.f, 0)

           || test_padding_int8(a, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding_int8(b, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding_int8(c, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)

           || test_padding_int8(a, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding_int8(b, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding_int8(c, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)

           || test_padding_int8(a, 1, 1, 1, 1, 1, 1, 0, -1.f, 0)
           || test_padding_int8(b, 1, 1, 1, 1, 1, 1, 0, -2.f, 0)
           || test_padding_int8(c, 1, 1, 1, 1, 1, 1, 0, 3.f, 0)

           || test_padding_int8(a, 2, 1, 0, 0, 2, 3, 0, 0.f, a.c + 5)
           || test_padding_int8(b, 2, 1, 0, 0, 2, 3, 0, 0.f, b.c + 5)
           || test_padding_int8(c, 2, 1, 0, 0, 2, 3, 0, 0.f, c.c + 5)

           || test_padding_int8(a, 1, 2, 3, 4, 8, 4, 0, 0.f, a.c + 12)
           || test_padding_int8(b, 1, 2, 3, 4, 8, 4, 0, 0.f, b.c + 12)
           || test_padding_int8(c, 1, 2, 3, 4, 8, 4, 0, 0.f, c.c + 12)

           || test_padding_int8(a, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)
           || test_padding_int8(b, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)
           || test_padding_int8(c, 0, 0, 0, 0, 3, 1, 1, 0.f, 0)

           || test_padding_int8(a, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)
           || test_padding_int8(b, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)
           || test_padding_int8(c, 2, 0, 1, 0, 4, 4, 1, 0.f, 0)

           || test_padding_int8(a, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)
           || test_padding_int8(b, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)
           || test_padding_int8(c, 2, 0, 2, 0, 0, 2, 2, 0.f, 0)

           || test_padding_int8(a, 4, 2, 1, 3, 3, 5, 2, 0.f, 0)
           || test_padding_int8(b, 4, 2, 1, 3, 3, 5, 2, 0.f, 0)
           || test_padding_int8(c, 4, 2, 1, 3, 3, 5, 2, 0.f, 0);
}

static int test_padding_6()
{
    ncnn::Mat a = RandomS8Mat(15, 96);
    ncnn::Mat b = RandomS8Mat(19, 44);
    ncnn::Mat c = RandomS8Mat(17, 15);

    return 0
           || test_padding_int8(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding_int8(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding_int8(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test_padding_int8(a, 0, 0, 1, 1, 0, 0, 0, 1.f, 0)
           || test_padding_int8(b, 0, 0, 1, 1, 0, 0, 0, 2.f, 0)
           || test_padding_int8(c, 0, 0, 1, 1, 0, 0, 0, -3.f, 0)

           || test_padding_int8(a, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding_int8(b, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding_int8(c, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)

           || test_padding_int8(a, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)
           || test_padding_int8(b, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)
           || test_padding_int8(c, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)

           || test_padding_int8(a, 2, 2, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding_int8(b, 2, 2, 2, 2, 0, 0, 0, 2.f, 0)
           || test_padding_int8(c, 2, 2, 2, 2, 0, 0, 0, -3.f, 0)

           || test_padding_int8(a, 8, 8, 2, 5, 0, 0, 0, -1.f, 0)
           || test_padding_int8(b, 8, 8, 2, 5, 0, 0, 0, -2.f, 0)
           || test_padding_int8(c, 8, 8, 2, 5, 0, 0, 0, 3.f, 0)

           || test_padding_int8(a, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)
           || test_padding_int8(b, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)
           || test_padding_int8(c, 3, 1, 3, 1, 0, 0, 1, 0.f, 0)

           || test_padding_int8(a, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding_int8(b, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding_int8(c, 4, 4, 0, 1, 0, 0, 1, 0.f, 0)

           || test_padding_int8(a, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding_int8(b, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding_int8(c, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)

           || test_padding_int8(a, 2, 6, 1, 0, 0, 0, 2, 0.f, 0)
           || test_padding_int8(b, 2, 6, 1, 0, 0, 0, 2, 0.f, 0)
           || test_padding_int8(c, 2, 6, 1, 0, 0, 0, 2, 0.f, 0);
}

static int test_padding_7()
{
    ncnn::Mat a = RandomS8Mat(128);
    ncnn::Mat b = RandomS8Mat(124);
    ncnn::Mat c = RandomS8Mat(127);

    return 0
           || test_padding_int8(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding_int8(b, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding_int8(c, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)

           || test_padding_int8(a, 0, 0, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding_int8(b, 0, 0, 2, 2, 0, 0, 0, 2.f, 0)
           || test_padding_int8(c, 0, 0, 2, 2, 0, 0, 0, -3.f, 0)

           || test_padding_int8(a, 0, 0, 16, 8, 0, 0, 0, -1.f, 0)
           || test_padding_int8(b, 0, 0, 16, 8, 0, 0, 0, -2.f, 0)
           || test_padding_int8(c, 0, 0, 16, 8, 0, 0, 0, 3.f, 0)

           || test_padding_int8(a, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding_int8(b, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding_int8(c, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)

           || test_padding_int8(a, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)
           || test_padding_int8(b, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)
           || test_padding_int8(c, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)

           || test_padding_int8(a, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding_int8(b, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding_int8(c, 0, 0, 2, 3, 0, 0, 2, 0.f, 0)

           || test_padding_int8(a, 0, 0, 10, 6, 0, 0, 2, 0.f, 0)
           || test_padding_int8(b, 0, 0, 10, 6, 0, 0, 2, 0.f, 0)
           || test_padding_int8(c, 0, 0, 10, 6, 0, 0, 2, 0.f, 0);
}

int main()
{
    SRAND(7767517);

    return test_padding_0()
           || test_padding_1()
           || test_padding_2()
           || test_padding_3()
           || test_padding_4()
           || test_padding_5()
           || test_padding_6()
           || test_padding_7();
}
