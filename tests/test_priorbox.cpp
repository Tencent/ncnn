// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

static int test_priorbox_caffe()
{
    ncnn::Mat min_sizes(1);
    min_sizes[0] = 105.f;

    ncnn::Mat max_sizes(1);
    max_sizes[0] = 150.f;

    ncnn::Mat aspect_ratios(2);
    aspect_ratios[0] = 2.f;
    aspect_ratios[1] = 3.f;

    ncnn::ParamDict pd;
    pd.set(0, min_sizes);
    pd.set(1, max_sizes);
    pd.set(2, aspect_ratios);
    pd.set(3, 0.1f);    // variances[0]
    pd.set(4, 0.1f);    // variances[1]
    pd.set(5, 0.2f);    // variances[2]
    pd.set(6, 0.2f);    // variances[3]
    pd.set(7, 1);       // flip
    pd.set(8, 0);       // clip
    pd.set(9, -233);    // image_width
    pd.set(10, -233);   // image_height
    pd.set(11, -233.f); // step_width
    pd.set(12, -233.f); // step_height
    pd.set(13, 0.f);    // offset
    pd.set(14, 0.f);    // step_mmdetection
    pd.set(15, 0.f);    // center_mmdetection

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = RandomMat(72, 72, 1);
    as[1] = RandomMat(512, 512, 1);

    int ret = test_layer("PriorBox", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_priorbox_caffe failed\n");
    }

    return ret;
}

static int test_priorbox_mxnet()
{
    ncnn::Mat min_sizes(2);
    min_sizes[0] = 0.15f;
    min_sizes[1] = 0.2121f;

    ncnn::Mat max_sizes(0);

    ncnn::Mat aspect_ratios(5);
    aspect_ratios[0] = 1.f;
    aspect_ratios[1] = 2.f;
    aspect_ratios[2] = 0.5f;
    aspect_ratios[3] = 3.f;
    aspect_ratios[4] = 0.333333;

    ncnn::ParamDict pd;
    pd.set(0, min_sizes);
    pd.set(1, max_sizes);
    pd.set(2, aspect_ratios);
    pd.set(3, 0.1f);    // variances[0]
    pd.set(4, 0.1f);    // variances[1]
    pd.set(5, 0.2f);    // variances[2]
    pd.set(6, 0.2f);    // variances[3]
    pd.set(7, 0);       // flip
    pd.set(8, 0);       // clip
    pd.set(9, -233);    // image_width
    pd.set(10, -233);   // image_height
    pd.set(11, -233.f); // step_width
    pd.set(12, -233.f); // step_height
    pd.set(13, 0.5f);   // offset
    pd.set(14, 0.f);    // step_mmdetection
    pd.set(15, 0.f);    // center_mmdetection

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(1);
    as[0] = RandomMat(72, 72, 1);

    int ret = test_layer("PriorBox", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_priorbox_mxnet failed\n");
    }

    return ret;
}

static int test_priorbox_load_param()
{
    ncnn::ParamDict pd;
    ncnn::Mat sizes(2);
    sizes.fill(1.f);
    pd.set(0, sizes);
    if (test_layer_param(ncnn::LayerType::PriorBox, pd, 0) != 0)
        return -1;

    const ncnn::ParamDict base = pd;

    if (test_layer_param(ncnn::LayerType::PriorBox, base, 1, ncnn::Mat(0), 0)
        || test_layer_param(ncnn::LayerType::PriorBox, base, 2, ncnn::Mat(0), 0))
        return -1;

    {
        ncnn::ParamDict pd = base;
        pd.set(2, ncnn::Mat(0));
        pd.set(1, ncnn::Mat(0));
        if (test_layer_param(ncnn::LayerType::PriorBox, pd, 0) != 0)
            return -1;
    }

    {
        ncnn::ParamDict pd = base;
        pd.set(2, ncnn::Mat(0));
        pd.set(1, ncnn::Mat(0));
        pd.set(0, ncnn::Mat(0));
        if (test_layer_param(ncnn::LayerType::PriorBox, pd, -1) != 0)
            return -1;
    }

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    return 0
           || test_layer_param(ncnn::LayerType::PriorBox, base, 0, missing_data, -1)
           || test_layer_param(ncnn::LayerType::PriorBox, base, 1, missing_data, -1)
           || test_layer_param(ncnn::LayerType::PriorBox, base, 2, missing_data, -1)
           || test_layer_param(ncnn::LayerType::PriorBox, base, 1, sizes.range(0, 1), -1);
}

static int test_priorbox_load_param_serialized()
{
    TestParamDict pd;
#if NCNN_STRING
    if (pd.load_param("-23300=1,1.0 -23301=0 -23302=0") != 0)
        return -1;

    if (test_layer_param(ncnn::LayerType::PriorBox, pd, 0) != 0)
        return -1;
#endif

    // binary parameters use little-endian byte order
    const unsigned char binary[] = {
        0xfc, 0xa4, 0xff, 0xff,
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x3f,
        0xfb, 0xa4, 0xff, 0xff,
        0x00, 0x00, 0x00, 0x00,
        0xfa, 0xa4, 0xff, 0xff,
        0x00, 0x00, 0x00, 0x00,
        0x17, 0xff, 0xff, 0xff
    };
    if (pd.load_param_bin(binary) != 0)
        return -1;

    return test_layer_param(ncnn::LayerType::PriorBox, pd, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_priorbox_caffe()
           || test_priorbox_mxnet()
           || test_priorbox_load_param()
           || test_priorbox_load_param_serialized();
}
