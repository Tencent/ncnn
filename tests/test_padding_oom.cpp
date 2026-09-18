// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_padding_oom(const ncnn::Mat& a, int top, int bottom, int left, int right, int front, int behind, int type, float value, int per_channel_pad_data_size)
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

    int ret = test_layer_oom("Padding", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_padding_oom failed a.dims=%d a=(%d %d %d %d) top=%d bottom=%d left=%d right=%d front=%d behind=%d type=%d value=%f per_channel_pad_data_size=%d\n", a.dims, a.w, a.h, a.d, a.c, top, bottom, left, right, front, behind, type, value, per_channel_pad_data_size);
    }

    return ret;
}

static int test_padding_0()
{
    ncnn::Mat a = RandomMat(9, 10, 11, 16);

    return 0
           || test_padding_oom(a, 2, 2, 2, 2, 2, 2, 0, 1.f, 0)
           || test_padding_oom(a, 2, 1, 2, 1, 2, 1, 0, 0.f, a.c)
           || test_padding_oom(a, 1, 2, 3, 4, 5, 6, 0, 0.f, 0)
           || test_padding_oom(a, 1, 2, 3, 1, 2, 1, 1, 0.f, 0)
           || test_padding_oom(a, 2, 1, 1, 2, 1, 2, 2, 0.f, 0);
}

static int test_padding_1()
{
    ncnn::Mat a = RandomMat(9, 11, 16);

    return 0
           || test_padding_oom(a, 2, 2, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding_oom(a, 2, 1, 2, 1, 0, 0, 0, 0.f, a.c)
           || test_padding_oom(a, 1, 2, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding_oom(a, 2, 3, 2, 3, 0, 0, 2, 0.f, 0)
           || test_padding_oom(a, 2, 1, 0, 0, 2, 3, 0, 0.f, a.c + 5);
}

static int test_padding_2()
{
    ncnn::Mat a = RandomMat(15, 16);

    return 0
           || test_padding_oom(a, 0, 0, 1, 1, 0, 0, 0, 1.f, 0)
           || test_padding_oom(a, 0, 0, 3, 4, 0, 0, 1, 0.f, 0)
           || test_padding_oom(a, 0, 0, 3, 2, 0, 0, 2, 0.f, 0)
           || test_padding_oom(a, 2, 2, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding_oom(a, 3, 1, 3, 1, 0, 0, 1, 0.f, 0);
}

static int test_padding_3()
{
    ncnn::Mat a = RandomMat(128);

    return 0
           || test_padding_oom(a, 0, 0, 2, 2, 0, 0, 0, 1.f, 0)
           || test_padding_oom(a, 0, 0, 0, 1, 0, 0, 1, 0.f, 0)
           || test_padding_oom(a, 0, 0, 4, 12, 0, 0, 1, 0.f, 0)
           || test_padding_oom(a, 0, 0, 2, 3, 0, 0, 2, 0.f, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_padding_0()
           || test_padding_1()
           || test_padding_2()
           || test_padding_3();
}
