// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_lrn(const ncnn::Mat& a, int region_type, int local_size, float alpha, float beta, float bias)
{
    ncnn::ParamDict pd;
    pd.set(0, region_type);
    pd.set(1, local_size);
    pd.set(2, alpha);
    pd.set(3, beta);
    pd.set(4, bias);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("LRN", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_lrn failed a.dims=%d a=(%d %d %d) region_type=%d local_size=%d alpha=%f beta=%f bias=%f\n", a.dims, a.w, a.h, a.c, region_type, local_size, alpha, beta, bias);
    }

    return ret;
}

static int test_lrn_0()
{
    ncnn::Mat a = RandomMat(11, 7, 12);

    return 0
           || test_lrn(a, 0, 1, 1.f, 0.75f, 1.f)
           || test_lrn(a, 0, 5, 2.f, 0.12f, 1.33f)
           || test_lrn(a, 1, 1, 0.6f, 0.4f, 2.4f)
           || test_lrn(a, 1, 3, 1.f, 0.75f, 0.5f);
}

static int test_lrn_1()
{
    ncnn::Mat a = RandomMat(10, 8, 16);

    return 0
           || test_lrn(a, 0, 1, 1.f, 0.75f, 1.f)
           || test_lrn(a, 0, 5, 2.f, 0.12f, 1.33f)
           || test_lrn(a, 1, 1, 0.6f, 0.4f, 2.4f)
           || test_lrn(a, 1, 3, 1.f, 0.75f, 0.5f);
}

static int test_lrn_2()
{
    ncnn::Mat a = RandomMat(12, 10, 9);

    return 0
           || test_lrn(a, 0, 1, 1.f, 0.75f, 1.f)
           || test_lrn(a, 0, 5, 2.f, 0.12f, 1.33f)
           || test_lrn(a, 1, 1, 0.6f, 0.4f, 2.4f)
           || test_lrn(a, 1, 3, 1.f, 0.75f, 0.5f);
}

static int test_lrn_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::LRN);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        fprintf(stderr, "test_lrn_load_param failed ret=%d expected=%d region_type=%d\n", ret, valid ? 0 : -1, pd.get(0, 0));
        return -1;
    }

    return 0;
}

static int test_lrn_load_param()
{
    ncnn::ParamDict base;
    if (test_lrn_load_param_case(base, true) != 0)
        return -1;

    for (int i = 0; i <= 1; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(0, i);
        if (test_lrn_load_param_case(pd, true) != 0)
            return -1;
    }

    const int invalid[] = {-1, 2, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(0, invalid[i]);
        if (test_lrn_load_param_case(pd, false) != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_lrn_0()
           || test_lrn_1()
           || test_lrn_2()
           || test_lrn_load_param();
}
