// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_spp_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::SPP);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        fprintf(stderr, "test_spp_load_param failed ret=%d expected=%d pooling_type=%d pyramid_height=%d\n", ret, valid ? 0 : -1, pd.get(0, 0), pd.get(1, 1));
        return -1;
    }

    return 0;
}

static int test_spp_load_param()
{
    ncnn::ParamDict base;
    if (test_spp_load_param_case(base, true) != 0)
        return -1;

    for (int i = 0; i <= 1; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(0, i);
        if (test_spp_load_param_case(pd, true) != 0)
            return -1;
    }

    const int invalid[] = {-1, 2, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(0, invalid[i]);
        if (test_spp_load_param_case(pd, false) != 0)
            return -1;
    }

    for (int i = 1; i <= 15; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(1, i);
        if (test_spp_load_param_case(pd, true) != 0)
            return -1;
    }

    const int invalid_height[] = {-1, 0, 16, INT_MIN, INT_MAX};
    for (int i = 0; i < 5; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(1, invalid_height[i]);
        if (test_spp_load_param_case(pd, false) != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_spp_load_param();
}
