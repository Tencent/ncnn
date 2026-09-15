// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

static int test_yolodetectionoutput_load_param()
{
    ncnn::ParamDict pd;
    pd.set(0, 20);
    pd.set(1, 1);
    ncnn::Mat biases(2);
    biases.fill(1.f);
    pd.set(4, biases);
    if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, pd, 0) != 0)
        return -1;

    const ncnn::ParamDict base = pd;

    if (test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, 4, ncnn::Mat(0), -1) != 0)
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    return 0
           || test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, 4, missing_data, -1)
           || test_layer_param(ncnn::LayerType::YoloDetectionOutput, base, 4, biases.range(0, 1), -1);
}

int main()
{
    SRAND(7767517);

    return test_yolodetectionoutput_load_param();
}
