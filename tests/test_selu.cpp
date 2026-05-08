// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_selu(const ncnn::Mat& a, float alpha, float lambda)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, lambda);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("SELU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_selu failed a.dims=%d a=(%d %d %d %d) alpha=%f lambda=%f\n", a.dims, a.w, a.h, a.d, a.c, alpha, lambda);
    }

    return ret;
}

static int test_selu_0()
{
    return 0
           || test_selu(RandomMat(7, 6, 5, 32), 1.673264f, 1.050700f)
           || test_selu(RandomMat(5, 6, 7, 24), 1.673264f, 1.050700f)
           || test_selu(RandomMat(7, 8, 9, 12), 1.673264f, 1.050700f)
           || test_selu(RandomMat(3, 4, 5, 13), 1.673264f, 1.050700f);
}

static int test_selu_1()
{
    return 0
           || test_selu(RandomMat(4, 7, 32), 1.673264f, 1.050700f)
           || test_selu(RandomMat(5, 7, 24), 1.673264f, 1.050700f)
           || test_selu(RandomMat(7, 9, 12), 1.673264f, 1.050700f)
           || test_selu(RandomMat(3, 5, 13), 1.673264f, 1.050700f);
}

static int test_selu_2()
{
    return 0
           || test_selu(RandomMat(13, 32), 1.673264f, 1.050700f)
           || test_selu(RandomMat(15, 24), 1.673264f, 1.050700f)
           || test_selu(RandomMat(17, 12), 1.673264f, 1.050700f)
           || test_selu(RandomMat(19, 15), 1.673264f, 1.050700f);
}

static int test_selu_3()
{
    return 0
           || test_selu(RandomMat(128), 1.673264f, 1.050700f)
           || test_selu(RandomMat(124), 1.673264f, 1.050700f)
           || test_selu(RandomMat(127), 1.673264f, 1.050700f)
           || test_selu(RandomMat(120), 1.673264f, 1.050700f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_selu_0()
           || test_selu_1()
           || test_selu_2()
           || test_selu_3();
}
