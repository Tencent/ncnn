// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer.h"
#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_roialign(int w, int h, int c, int pooled_width, int pooled_height, float spatial_scale, int sampling_ratio, bool aligned, int version)
{
    std::vector<ncnn::Mat> a;
    a.push_back(RandomMat(w, h, c));
    ncnn::Mat b(4);
    b[0] = RandomFloat(0.001, w - 2.001);        //roi_x1
    b[2] = RandomFloat(b[0] + 1.001, w - 1.001); //roi_x2
    b[1] = RandomFloat(0.001, h - 2.001);        //roi_y1
    b[3] = RandomFloat(b[2] + 1.001, h - 1.001); //roi_y2
    a.push_back(b);

    ncnn::ParamDict pd;
    pd.set(0, pooled_width);   // pooled_width
    pd.set(1, pooled_height);  // pooled_height
    pd.set(2, spatial_scale);  // spatial_scale
    pd.set(3, sampling_ratio); // sampling_ratio
    pd.set(4, aligned);        // aligned
    pd.set(5, version);        // version

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ROIAlign", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_roialign failed base_w=%d base_h=%d base_c=%d pooled_width=%d pooled_height=%d spatial_scale=%4f.3\n", w, h, c, pooled_width, pooled_height, spatial_scale);
    }

    return ret;
}

static int test_roialign_0()
{
    return 0
           || test_roialign(56, 56, 1, 28, 28, 0.50000, 0, 0, 0)
           || test_roialign(28, 28, 3, 14, 14, 0.25000, 1, 1, 1)
           || test_roialign(14, 14, 4, 7, 7, 0.12500, 2, 0, 1)
           || test_roialign(14, 14, 8, 7, 7, 0.06250, 3, 1, 0)
           || test_roialign(7, 7, 12, 3, 3, 0.03125, 4, 0, 0)
           || test_roialign(7, 7, 16, 3, 3, 0.03125, 4, 1, 1);
}

static int test_roialign_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::ROIAlign);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if ((ret == 0) != valid)
    {
        fprintf(stderr, "ROIAlign load_param returned %d, expected %s\n", ret, valid ? "success" : "failure");
        return -1;
    }

    return 0;
}

static int test_roialign_load_param()
{
    ncnn::ParamDict pd;
    pd.set(0, 2);
    pd.set(1, 2);

    const int sampling_ratios[] = {0, -1, -2, INT_MIN};
    for (int version = 0; version < 2; version++)
    {
        pd.set(5, version);
        for (int i = 0; i < 4; i++)
        {
            pd.set(3, sampling_ratios[i]);
            if (test_roialign_load_param_case(pd, true) != 0)
                return -1;
        }
    }

    pd.set(5, 2);
    if (test_roialign_load_param_case(pd, false) != 0)
        return -1;
    return 0;
}

static int test_roialign_adaptive(int version, bool aligned)
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(8, 8, 3);
    a[1].create(4);
    a[1][0] = 1.25f;
    a[1][1] = 1.5f;
    a[1][2] = 6.75f;
    a[1][3] = 5.5f;

    ncnn::ParamDict pd;
    pd.set(0, 3);
    pd.set(1, 2);
    pd.set(4, aligned);
    pd.set(5, version);

    std::vector<ncnn::Mat> weights(0);
    std::vector<ncnn::Mat> reference;
    int ret = test_layer_naive(ncnn::LayerType::ROIAlign, pd, weights, a, 1, reference, 0);
    if (ret != 0)
    {
        fprintf(stderr, "test_roialign_adaptive reference failed version=%d aligned=%d ret=%d\n", version, aligned, ret);
        return ret;
    }

    const int sampling_ratios[] = {-1, -2, INT_MIN};
    for (int i = 0; i < 3; i++)
    {
        pd.set(3, sampling_ratios[i]);
        ret = test_layer("ROIAlign", pd, weights, a);
        if (ret != 0)
        {
            fprintf(stderr, "test_roialign_adaptive failed version=%d aligned=%d sampling_ratio=%d ret=%d\n", version, aligned, sampling_ratios[i], ret);
            return ret;
        }

        std::vector<ncnn::Mat> output;
        ret = test_layer_naive(ncnn::LayerType::ROIAlign, pd, weights, a, 1, output, 0);
        if (ret == 0)
            ret = CompareMat(reference, output, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_roialign_adaptive output mismatch version=%d aligned=%d sampling_ratio=%d ret=%d\n", version, aligned, sampling_ratios[i], ret);
            return ret;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_roialign_0()
           || test_roialign_load_param()
           || test_roialign_adaptive(0, false)
           || test_roialign_adaptive(0, true)
           || test_roialign_adaptive(1, false)
           || test_roialign_adaptive(1, true);
}
