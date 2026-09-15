// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

static int test_yolodetectionoutput_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::YoloDetectionOutput);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        const int num_class = pd.get(0, 20);
        const int num_box = pd.get(1, 5);

        fprintf(stderr, "test_yolodetectionoutput_load_param failed ret=%d expected=%d num_class=%d num_box=%d\n", ret, valid ? 0 : -1, num_class, num_box);

        const ncnn::Mat biases = pd.get(4, ncnn::Mat());
        fprintf(stderr, "biases type=%d dims=%d w=%d elemsize=%zu elempack=%d\n", pd.type(4), biases.dims, biases.w, biases.elemsize, biases.elempack);
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
