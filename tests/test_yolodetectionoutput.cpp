// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_yolodetectionoutput_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::YoloDetectionOutput);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if ((ret == 0) != valid)
    {
        fprintf(stderr, "YoloDetectionOutput load_param returned %d, expected %s\n", ret, valid ? "success" : "failure");
        return -1;
    }

    return 0;
}

static int test_yolodetectionoutput_load_param()
{
    ncnn::ParamDict pd;
    pd.set(0, 20);
    pd.set(1, 1);
    ncnn::Mat biases(2);
    biases.fill(1.f);
    pd.set(4, biases);
    if (test_yolodetectionoutput_load_param_case(pd, true) != 0)
        return -1;

    pd.set(4, biases.range(0, 1));
    if (test_yolodetectionoutput_load_param_case(pd, false) != 0)
        return -1;
    return 0;
}

int main()
{
    SRAND(7767517);

    return test_yolodetectionoutput_load_param();
}
