// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer.h"
#include "testutil.h"

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

int main()
{
    SRAND(7767517);

    return 0
           || test_roialign_0();
}
