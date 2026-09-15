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

static int test_priorbox_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::PriorBox);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        fprintf(stderr, "test_priorbox_load_param failed ret=%d expected=%d\n", ret, valid ? 0 : -1);

        const ncnn::Mat min_sizes = pd.get(0, ncnn::Mat());
        fprintf(stderr, "min_sizes type=%d dims=%d w=%d elemsize=%zu elempack=%d\n", pd.type(0), min_sizes.dims, min_sizes.w, min_sizes.elemsize, min_sizes.elempack);

        const ncnn::Mat max_sizes = pd.get(1, ncnn::Mat());
        fprintf(stderr, "max_sizes type=%d dims=%d w=%d elemsize=%zu elempack=%d\n", pd.type(1), max_sizes.dims, max_sizes.w, max_sizes.elemsize, max_sizes.elempack);

        const ncnn::Mat aspect_ratios = pd.get(2, ncnn::Mat());
        fprintf(stderr, "aspect_ratios type=%d dims=%d w=%d elemsize=%zu elempack=%d\n", pd.type(2), aspect_ratios.dims, aspect_ratios.w, aspect_ratios.elemsize, aspect_ratios.elempack);
        return -1;
    }

    return 0;
}

static int test_priorbox_load_param()
{
    ncnn::ParamDict pd;
    ncnn::Mat sizes(2);
    sizes.fill(1.f);
    pd.set(0, sizes);
    if (test_priorbox_load_param_case(pd, true) != 0)
        return -1;

    pd.set(1, sizes.range(0, 1));
    if (test_priorbox_load_param_case(pd, false) != 0)
        return -1;

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_priorbox_caffe()
           || test_priorbox_mxnet()
           || test_priorbox_load_param();
}
