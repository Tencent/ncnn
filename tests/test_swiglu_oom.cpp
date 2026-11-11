// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_swiglu_oom(const ncnn::Mat& a, float sigmoid_scale, float up_offset, float activation_limit)
{
    ncnn::ParamDict pd;
    pd.set(0, sigmoid_scale);
    pd.set(1, up_offset);
    pd.set(2, activation_limit);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = a;

    int ret = test_layer_oom("SwiGLU", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_swiglu_oom failed a.dims=%d a=(%d %d %d %d) scale=%f offset=%f limit=%f\n", a.dims, a.w, a.h, a.d, a.c, sigmoid_scale, up_offset, activation_limit);
    }

    return ret;
}

static int test_swiglu_oom_0()
{
    return 0
           || test_swiglu_oom(RandomMat(7, 9, 12), 1.f, 0.f, 0.f)
           || test_swiglu_oom(RandomMat(17, 15), 1.702f, 1.f, 7.f)
           || test_swiglu_oom(RandomMat(127), 1.f, 0.f, 10.f);
}

int main()
{
    SRAND(7767517);

    return test_swiglu_oom_0();
}
