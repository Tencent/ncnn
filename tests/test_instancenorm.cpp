// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_instancenorm(const ncnn::Mat& a, float eps, int affine)
{
    int channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, affine ? channels : 0);
    pd.set(1, eps);
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);

    int ret = test_layer("InstanceNorm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_instancenorm failed a.dims=%d a=(%d %d %d) eps=%f affine=%d\n", a.dims, a.w, a.h, a.c, eps, affine);
    }

    return ret;
}

static int test_instancenorm_0()
{
    return 0
           || test_instancenorm(RandomMat(6, 4, 2), 0.01f, 0)
           || test_instancenorm(RandomMat(3, 3, 12), 0.002f, 0)
           || test_instancenorm(RandomMat(5, 7, 16), 0.02f, 0)
           || test_instancenorm(RandomMat(6, 4, 2), 0.01f, 1)
           || test_instancenorm(RandomMat(3, 3, 12), 0.002f, 1)
           || test_instancenorm(RandomMat(5, 7, 16), 0.02f, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_instancenorm_0();
}
