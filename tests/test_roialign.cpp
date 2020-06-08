// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "testutil.h"

#include "layer.h"
#include "layer/roialign.h"


static int test_roialign(int w, int h, int c,
    int pooled_width, int pooled_height, float spatial_scale,
    int sampling_ratio, bool aligned, int version)
{
    std::vector<ncnn::Mat> a;
    a.push_back(RandomMat(w, h, c));
    ncnn::Mat b(4);
    b[0] = RandomFloat(0.001     , w-2.001);    //roi_x1
    b[2] = RandomFloat(b[0]+1.001, w-1.001);    //roi_x2
    b[1] = RandomFloat(0.001     , h-2.001);    //roi_y1
    b[3] = RandomFloat(b[2]+1.001, h-1.001);    //roi_y2
    a.push_back(b);

    ncnn::ParamDict pd;
    pd.set(0, pooled_width);    // pooled_width
    pd.set(1, pooled_height);   // pooled_height
    pd.set(2, spatial_scale);   // spatial_scale
    pd.set(3, sampling_ratio);  // sampling_ratio
    pd.set(4, aligned);         // aligned
    pd.set(5, version);         // version

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::ROIAlign>("ROIAlign", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_roialign failed base_w=%d base_h=%d base_c=%d pooled_width=%d pooled_height=%d spatial_scale=%4f.3\n", w, h, c, pooled_width, pooled_height, spatial_scale);
    }

    return ret;
}

static int test_roialign_0()
{
    int ret = 0;
    for (int sampling_ratio = 0; sampling_ratio <= 4 && ret == 0; ++sampling_ratio)
    {
        for (int aligned_i = 0; aligned_i < 2 && ret == 0; ++aligned_i)
        {
            for (int version = 0; version <= 1; ++version)
            {
              bool aligned = aligned_i;
              int lret = 0
                  || test_roialign(112, 112, 16 , 56, 56, 0.50000, sampling_ratio, aligned, version)
                  || test_roialign(56 , 56 , 32 , 28, 28, 0.25000, sampling_ratio, aligned, version)
                  || test_roialign(28 , 28 , 64 , 14, 14, 0.12500, sampling_ratio, aligned, version)
                  || test_roialign(14 , 14 , 128, 27, 17, 0.06250, sampling_ratio, aligned, version)
                  || test_roialign(7  , 7  , 256,  3,  3, 0.03125, sampling_ratio, aligned, version)
                  ;
              ret |= lret;
            }
        }
    }

    if (ret != 0)
        return -1;

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_roialign_0()
        ;
}
