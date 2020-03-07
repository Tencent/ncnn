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
#include "layer/roipooling.h"

static int RandomInt(float a = -2.f, float b = 2.f)
{
    return RandomFloat( a, b ) / 1;
}

static int test_roipooling(int w, int h, int c, int pooled_width, int pooled_height, float spatial_scale)
{
    std::vector<ncnn::Mat> a;
    int num_rois = RandomInt(2, 1000);
    for(int i=0; i<num_rois; i++){
        a.push_back(RandomMat(w + RandomInt(0, 15), h + RandomInt(0, 13), c));
    }

    ncnn::ParamDict pd;
    pd.set(0, pooled_width);    // pooled_width
    pd.set(1, pooled_height);   // pooled_height
    pd.set(2, spatial_scale);   // spatial_scale

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    int ret = test_layer<ncnn::ROIPooling>("ROIPooling", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_roipooling failed base_w=%d base_h=%d base_c=%d num_rois=%d pooled_width=%d pooled_height=%d spatial_scale=%4f.3\n", w, h, c, num_rois, pooled_width, pooled_height, spatial_scale);
    }

    return ret;
}

static int test_roipooling_0()
{
    for (int i=0; i<11; i++)
    {
        int ret = 0
            || test_roipooling(2, 2, 81, 4, 4, 0.0156)
            || test_roipooling(4, 4, 27, 7, 7, 0.0312)
            || test_roipooling(9, 9, 9, 14, 14, 0.0625)
            || test_roipooling(15, 21, 81, 8, 6, 0.0123)
            || test_roipooling(17, 23, 27, 27, 17, 0.0370)
            || test_roipooling(19, 25, 9, 80, 53, 0.1111)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_roipooling_1()
{
    for (int i=0; i<11; i++)
    {
        int ret = 0
            || test_roipooling(2, 2, 27, 4, 4, 0.0156)
            || test_roipooling(4, 4, 9, 7, 7, 0.0312)
            || test_roipooling(9, 9, 3, 14, 14, 0.0625)
            || test_roipooling(4, 21, 27, 12, 9, 0.0123)
            || test_roipooling(17, 23, 9, 38, 29, 0.0370)
            || test_roipooling(105, 65, 3, 113, 85, 0.1111)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_roipooling_2()
{
    for (int i=0; i<11; i++)
    {
        int ret = 0
            || test_roipooling(4, 7, 1, 4, 5, 0.0156)
            || test_roipooling(6, 11, 2, 6, 7, 0.0312)
            || test_roipooling(9, 17, 3, 14, 17, 0.0625)
            || test_roipooling(13, 19, 4, 15, 18, 0.1250)
            || test_roipooling(15, 21, 7, 18, 21, 0.0123)
            || test_roipooling(17, 23, 8, 21, 25, 0.0370)
            || test_roipooling(19, 25, 15, 21, 26, 0.1111)
            || test_roipooling(23, 31, 16, 25, 34, 0.3333)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_roipooling_0()
        || test_roipooling_1()
        || test_roipooling_2()
        ;
}
