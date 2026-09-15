// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_spp_load_param()
{
    ncnn::ParamDict base;
    if (test_layer_param(ncnn::LayerType::SPP, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 1; i++)
    {
        if (test_layer_param(ncnn::LayerType::SPP, base, 0, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 2, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::SPP, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    for (int i = 1; i <= 15; i++)
    {
        if (test_layer_param(ncnn::LayerType::SPP, base, 1, i, 0) != 0)
            return -1;
    }

    const int invalid_height[] = {-1, 0, 16, INT_MIN, INT_MAX};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::SPP, base, 1, invalid_height[i], -1) != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_spp_load_param();
}
