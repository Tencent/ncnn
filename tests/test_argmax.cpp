// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_argmax(const ncnn::Mat& a, int out_max_val, int topk)
{
    ncnn::ParamDict pd;
    pd.set(0, out_max_val);
    pd.set(1, topk);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ArgMax", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_argmax failed a.dims=%d a=(%d %d %d %d) out_max_val=%d topk=%d\n", a.dims, a.w, a.h, a.d, a.c, out_max_val, topk);
    }

    return ret;
}

static int test_argmax_0()
{
    return 0
           || test_argmax(RandomMat(5, 6, 7, 24), 0, 1)
           || test_argmax(RandomMat(5, 6, 7, 24), 1, 1)
           || test_argmax(RandomMat(7, 8, 9, 12), 0, 1)
           || test_argmax(RandomMat(7, 8, 9, 12), 1, 1)
           || test_argmax(RandomMat(3, 4, 5, 13), 0, 3)
           || test_argmax(RandomMat(3, 4, 5, 13), 1, 3);
}

static int test_argmax_1()
{
    return 0
           || test_argmax(RandomMat(5, 7, 24), 0, 1)
           || test_argmax(RandomMat(5, 7, 24), 1, 1)
           || test_argmax(RandomMat(7, 9, 12), 0, 2)
           || test_argmax(RandomMat(7, 9, 12), 1, 2)
           || test_argmax(RandomMat(3, 5, 13), 0, 5)
           || test_argmax(RandomMat(3, 5, 13), 1, 5);
}

static int test_argmax_2()
{
    return 0
           || test_argmax(RandomMat(15, 24), 0, 1)
           || test_argmax(RandomMat(15, 24), 1, 1)
           || test_argmax(RandomMat(17, 12), 0, 3)
           || test_argmax(RandomMat(17, 12), 1, 3)
           || test_argmax(RandomMat(19, 15), 0, 5)
           || test_argmax(RandomMat(19, 15), 1, 5);
}

static int test_argmax_3()
{
    return 0
           || test_argmax(RandomMat(128), 0, 1)
           || test_argmax(RandomMat(128), 1, 1)
           || test_argmax(RandomMat(124), 0, 3)
           || test_argmax(RandomMat(124), 1, 3)
           || test_argmax(RandomMat(127), 0, 5)
           || test_argmax(RandomMat(127), 1, 5);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_argmax_0()
           || test_argmax_1()
           || test_argmax_2()
           || test_argmax_3();
}
