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

#include "layer/interp.h"
#include "testutil.h"

static int test_interp(const ncnn::Mat& a, int resize_type, float height_scale, float width_scale, int output_height, int output_width)
{
    ncnn::ParamDict pd;
    pd.set(0, resize_type);
    pd.set(1, height_scale);
    pd.set(2, width_scale);
    pd.set(3, output_height);
    pd.set(4, output_width);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::Interp>("Interp", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_interp failed a.dims=%d a=(%d %d %d) resize_type=%d height_scale=%f width_scale=%f output_height=%d output_width=%d\n", a.dims, a.w, a.h, a.c, resize_type, height_scale, width_scale, output_height, output_width);
    }

    return ret;
}

static int test_interp_0()
{
    ncnn::Mat a = RandomMat(5, 6, 7);
    ncnn::Mat b = RandomMat(4, 7, 16);

    return 0
           || test_interp(a, 1, 2.f, 2.f, 0, 0)
           || test_interp(a, 1, 4.f, 0.5f, 0, 0)
           || test_interp(a, 1, 1.f, 1.f, 10, 12)
           || test_interp(a, 1, 1.f, 1.f, 2, 2)

           || test_interp(b, 1, 2.f, 2.f, 0, 0)
           || test_interp(b, 1, 4.f, 0.5f, 0, 0)
           || test_interp(b, 1, 1.f, 1.f, 10, 12)
           || test_interp(b, 1, 1.f, 1.f, 2, 2);
}

static int test_interp_1()
{
    ncnn::Mat a = RandomMat(5, 6, 7);
    ncnn::Mat b = RandomMat(4, 7, 16);

    return 0
           || test_interp(a, 2, 2.f, 2.f, 0, 0)
           || test_interp(a, 2, 4.f, 0.5f, 0, 0)
           || test_interp(a, 2, 1.f, 1.f, 10, 12)
           || test_interp(a, 2, 1.f, 1.f, 2, 2)

           || test_interp(b, 2, 2.f, 2.f, 0, 0)
           || test_interp(b, 2, 4.f, 0.5f, 0, 0)
           || test_interp(b, 2, 1.f, 1.f, 10, 12)
           || test_interp(b, 2, 1.f, 1.f, 2, 2);
}

static int test_interp_2()
{
    ncnn::Mat a = RandomMat(6, 7, 13);
    ncnn::Mat b = RandomMat(8, 9, 16);

    return 0
           || test_interp(a, 3, 2.f, 2.f, 0, 0)
           || test_interp(a, 3, 4.f, 0.5f, 0, 0)
           || test_interp(a, 3, 1.f, 1.f, 10, 12)
           || test_interp(a, 3, 1.f, 1.f, 2, 2)

           || test_interp(b, 3, 2.f, 2.f, 0, 0)
           || test_interp(b, 3, 4.f, 0.5f, 0, 0)
           || test_interp(b, 3, 1.f, 1.f, 10, 12)
           || test_interp(b, 3, 1.f, 1.f, 2, 2);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_interp_0()
           || test_interp_1()
           || test_interp_2();
}
