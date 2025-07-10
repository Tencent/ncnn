// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_dropout(const ncnn::Mat& a, float scale)
{
    ncnn::ParamDict pd;
    pd.set(0, scale);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Dropout", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_dropout failed a.dims=%d a=(%d %d %d) scale=%f\n", a.dims, a.w, a.h, a.c, scale);
    }

    return ret;
}

static int test_dropout_0()
{
    return 0
           || test_dropout(RandomMat(5, 7, 24), 1.f)
           || test_dropout(RandomMat(5, 7, 24), 0.2f)
           || test_dropout(RandomMat(7, 9, 12), 1.f)
           || test_dropout(RandomMat(7, 9, 12), 0.3f)
           || test_dropout(RandomMat(3, 5, 13), 1.f)
           || test_dropout(RandomMat(3, 5, 13), 0.5f);
}

static int test_dropout_1()
{
    return 0
           || test_dropout(RandomMat(15, 24), 1.f)
           || test_dropout(RandomMat(15, 24), 0.6f)
           || test_dropout(RandomMat(19, 12), 1.f)
           || test_dropout(RandomMat(19, 12), 0.4f)
           || test_dropout(RandomMat(17, 15), 1.f)
           || test_dropout(RandomMat(17, 15), 0.7f);
}

static int test_dropout_2()
{
    return 0
           || test_dropout(RandomMat(128), 1.f)
           || test_dropout(RandomMat(128), 0.4f)
           || test_dropout(RandomMat(124), 1.f)
           || test_dropout(RandomMat(124), 0.1f)
           || test_dropout(RandomMat(127), 1.f)
           || test_dropout(RandomMat(127), 0.5f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_dropout_0()
           || test_dropout_1()
           || test_dropout_2();
}
