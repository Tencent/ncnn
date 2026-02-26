// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_topk(const ncnn::Mat& a, int axis, int k, int largest, int sorted)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, largest);
    pd.set(2, sorted);
    pd.set(3, k);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    int ret = test_layer("TopK", pd, weights, a0, 2, 0.01f, TEST_LAYER_DISABLE_AUTO_INPUT_CASTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk failed a.dims=%d a=(%d %d %d %d) axis=%d k=%d largest=%d sorted=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, k, largest, sorted);
    }

    return ret;
}

static int test_topk_0()
{
    ncnn::Mat a = RandomMat(13);

    return 0
           || test_topk(a, 0, 1, 1, 1)
           || test_topk(a, 0, 5, 1, 1)
           || test_topk(a, -1, 7, 0, 1)
           || test_topk(a, 0, 9, 1, 1);
}

static int test_topk_1()
{
    ncnn::Mat a = RandomMat(12, 17);

    return 0
           || test_topk(a, 0, 1, 1, 1)
           || test_topk(a, 0, 5, 1, 1)
           || test_topk(a, 1, 3, 1, 1)
           || test_topk(a, -1, 8, 0, 1)
           || test_topk(a, -2, 7, 1, 1);
}

static int test_topk_2()
{
    ncnn::Mat a = RandomMat(8, 9, 11);

    return 0
           || test_topk(a, 0, 3, 1, 1)
           || test_topk(a, 1, 4, 1, 1)
           || test_topk(a, 2, 2, 0, 1)
           || test_topk(a, -1, 6, 1, 1)
           || test_topk(a, -2, 5, 0, 1)
           || test_topk(a, -3, 7, 1, 1);
}

static int test_topk_3()
{
    ncnn::Mat a = RandomMat(5, 7, 9, 10);

    return 0
           || test_topk(a, 0, 2, 1, 1)
           || test_topk(a, 1, 3, 0, 1)
           || test_topk(a, 2, 4, 1, 1)
           || test_topk(a, 3, 5, 1, 1)
           || test_topk(a, -1, 6, 0, 1)
           || test_topk(a, -2, 3, 1, 1)
           || test_topk(a, -3, 4, 0, 1)
           || test_topk(a, -4, 2, 1, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_topk_0()
           || test_topk_1()
           || test_topk_2()
           || test_topk_3();
}
