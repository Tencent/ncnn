// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_celu(const ncnn::Mat& a, float alpha)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("CELU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_celu failed a.dims=%d a=(%d %d %d %d) alpha=%f\n", a.dims, a.w, a.h, a.d, a.c, alpha);
    }

    return ret;
}

static int test_celu_0()
{
    return 0
           || test_celu(RandomMat(3, 8, 12, 18), 1.f)
           || test_celu(RandomMat(4, 7, 9, 16), 0.1f)
           || test_celu(RandomMat(3, 5, 12, 16), 1.f)
           || test_celu(RandomMat(9, 6, 7, 14), 0.1f)
           || test_celu(RandomMat(5, 6, 9, 10), 1.f)
           || test_celu(RandomMat(6, 8, 2, 15), 0.1f);
}

static int test_celu_1()
{
    return 0
           || test_celu(RandomMat(7, 6, 18), 1.f)
           || test_celu(RandomMat(9, 6, 15), 0.1f)
           || test_celu(RandomMat(9, 7, 16), 1.f)
           || test_celu(RandomMat(6, 10, 15), 0.1f)
           || test_celu(RandomMat(2, 7, 11), 1.f)
           || test_celu(RandomMat(6, 10, 7), 0.1f);
}

static int test_celu_2()
{
    return 0
           || test_celu(RandomMat(12, 18), 1.f)
           || test_celu(RandomMat(18, 12), 0.1f)
           || test_celu(RandomMat(23, 27), 1.f)
           || test_celu(RandomMat(18, 16), 0.1f)
           || test_celu(RandomMat(18, 16), 1.f)
           || test_celu(RandomMat(20, 16), 0.1f);
}

static int test_celu_3()
{
    return 0
           || test_celu(RandomMat(256), 1.f)
           || test_celu(RandomMat(64), 0.1f)
           || test_celu(RandomMat(128), 1.f)
           || test_celu(RandomMat(96), 0.1f)
           || test_celu(RandomMat(128), 1.f)
           || test_celu(RandomMat(128), 0.1f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_celu_0()
           || test_celu_1()
           || test_celu_2()
           || test_celu_3();
}
