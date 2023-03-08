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

    int ret = test_layer<ncnn::Interp>("Interp", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_interp failed a.dims=%d a=(%d %d %d) resize_type=%d height_scale=%f width_scale=%f output_height=%d output_width=%d\n", a.dims, a.w, a.h, a.c, resize_type, height_scale, width_scale, output_height, output_width);
    }

    return ret;
}

static int test_interp_ref(const ncnn::Mat& a, int resize_type, int output_height, int output_width)
{
    ncnn::ParamDict pd;
    pd.set(0, resize_type);
    pd.set(5, 1);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = ncnn::Mat(output_width, output_height, 1);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Interp>("Interp", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_interp_ref failed a.dims=%d a=(%d %d %d) resize_type=%d output_height=%d output_width=%d\n", a.dims, a.w, a.h, a.c, resize_type, output_height, output_width);
    }

    return ret;
}

static int test_interp_align_corner(const ncnn::Mat& a, int resize_type, float height_scale, float width_scale, int output_height, int output_width, int align_corner)
{
    ncnn::ParamDict pd;
    pd.set(0, resize_type);
    pd.set(1, height_scale);
    pd.set(2, width_scale);
    pd.set(3, output_height);
    pd.set(4, output_width);
    pd.set(6, align_corner);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Interp>("Interp", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_interp failed a.dims=%d a=(%d %d %d) resize_type=%d height_scale=%f width_scale=%f output_height=%d output_width=%d align_corner=%d\n", a.dims, a.w, a.h, a.c, resize_type, height_scale, width_scale, output_height, output_width, align_corner);
    }

    return ret;
}

static int test_interp(const ncnn::Mat& a, int resize_type, float width_scale, int output_width)
{
    ncnn::ParamDict pd;
    pd.set(0, resize_type);
    pd.set(1, 1.f);
    pd.set(2, width_scale);
    pd.set(3, 0);
    pd.set(4, output_width);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Interp>("Interp", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_interp failed a.dims=%d a=(%d %d %d) resize_type=%d width_scale=%f output_width=%d\n", a.dims, a.w, a.h, a.c, resize_type, width_scale, output_width);
    }

    return ret;
}

static int test_interp_ref(const ncnn::Mat& a, int resize_type, int output_width)
{
    ncnn::ParamDict pd;
    pd.set(0, resize_type);
    pd.set(5, 1);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = ncnn::Mat(output_width, 1);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Interp>("Interp", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_interp_ref failed a.dims=%d a=(%d %d %d) resize_type=%d output_width=%d\n", a.dims, a.w, a.h, a.c, resize_type, output_width);
    }

    return ret;
}

static int test_interp_align_corner(const ncnn::Mat& a, int resize_type, float width_scale, int output_width, int align_corner)
{
    ncnn::ParamDict pd;
    pd.set(0, resize_type);
    pd.set(1, 1.f);
    pd.set(2, width_scale);
    pd.set(3, 0);
    pd.set(4, output_width);
    pd.set(6, align_corner);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Interp>("Interp", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_interp failed a.dims=%d a=(%d %d %d) resize_type=%d width_scale=%f output_width=%d align_corner=%d\n", a.dims, a.w, a.h, a.c, resize_type, width_scale, output_width, align_corner);
    }

    return ret;
}

static int test_interp_0()
{
    ncnn::Mat a = RandomMat(15, 16, 7);
    ncnn::Mat b = RandomMat(14, 17, 12);
    ncnn::Mat c = RandomMat(13, 14, 32);

    return 0
           || test_interp(a, 1, 2.f, 2.f, 0, 0)
           || test_interp(a, 1, 4.f, 0.5f, 0, 0)
           || test_interp(a, 1, 1.2f, 1.2f, 0, 0)
           || test_interp(a, 1, 0.8f, 0.8f, 0, 0)
           || test_interp(a, 1, 1.f, 1.f, 10, 12)
           || test_interp(a, 1, 1.f, 1.f, 2, 2)
           || test_interp(a, 1, 1.f, 1.f, 15, 16)
           || test_interp_ref(a, 1, 10, 12)
           || test_interp_ref(a, 1, 2, 2)
           || test_interp_ref(a, 1, 15, 16)

           || test_interp(b, 1, 2.f, 2.f, 0, 0)
           || test_interp(b, 1, 4.f, 0.5f, 0, 0)
           || test_interp(b, 1, 1.2f, 1.2f, 0, 0)
           || test_interp(b, 1, 0.8f, 0.8f, 0, 0)
           || test_interp(b, 1, 1.f, 1.f, 10, 12)
           || test_interp(b, 1, 1.f, 1.f, 2, 2)
           || test_interp(b, 1, 1.f, 1.f, 14, 17)
           || test_interp_ref(b, 1, 10, 12)
           || test_interp_ref(b, 1, 2, 2)
           || test_interp_ref(b, 1, 14, 17)

           || test_interp(c, 1, 2.f, 2.f, 0, 0)
           || test_interp(c, 1, 4.f, 0.5f, 0, 0)
           || test_interp(c, 1, 1.2f, 1.2f, 0, 0)
           || test_interp(c, 1, 0.8f, 0.8f, 0, 0)
           || test_interp(c, 1, 1.f, 1.f, 10, 12)
           || test_interp(c, 1, 1.f, 1.f, 2, 2)
           || test_interp(c, 1, 1.f, 1.f, 14, 17)
           || test_interp_ref(c, 1, 10, 12)
           || test_interp_ref(c, 1, 2, 2)
           || test_interp_ref(c, 1, 14, 17);
}

static int test_interp_1()
{
    ncnn::Mat a = RandomMat(15, 16, 7);
    ncnn::Mat b = RandomMat(14, 17, 12);
    ncnn::Mat c = RandomMat(13, 14, 32);

    return 0
           || test_interp(a, 2, 2.f, 2.f, 0, 0)
           || test_interp(a, 2, 4.f, 0.5f, 0, 0)
           || test_interp(a, 2, 1.2f, 1.2f, 0, 0)
           || test_interp(a, 2, 0.8f, 0.8f, 0, 0)
           || test_interp(a, 2, 1.f, 1.f, 10, 12)
           || test_interp(a, 2, 1.f, 1.f, 2, 2)
           || test_interp(a, 2, 1.f, 1.f, 15, 16)
           || test_interp_align_corner(a, 2, 2.f, 2.f, 0, 0, 1)
           || test_interp_align_corner(a, 2, 4.f, 0.5f, 0, 0, 1)
           || test_interp_align_corner(a, 2, 1.2f, 1.2f, 0, 0, 1)
           || test_interp_align_corner(a, 2, 0.8f, 0.8f, 0, 0, 1)
           || test_interp_align_corner(a, 2, 1.f, 1.f, 10, 12, 1)
           || test_interp_align_corner(a, 2, 1.f, 1.f, 2, 2, 1)
           || test_interp_align_corner(a, 2, 1.f, 1.f, 15, 16, 1)
           || test_interp_ref(a, 2, 10, 12)
           || test_interp_ref(a, 2, 2, 2)
           || test_interp_ref(a, 2, 15, 16)

           || test_interp(b, 2, 2.f, 2.f, 0, 0)
           || test_interp(b, 2, 4.f, 0.5f, 0, 0)
           || test_interp(b, 2, 1.2f, 1.2f, 0, 0)
           || test_interp(b, 2, 0.8f, 0.8f, 0, 0)
           || test_interp(b, 2, 1.f, 1.f, 10, 12)
           || test_interp(b, 2, 1.f, 1.f, 2, 2)
           || test_interp(b, 2, 1.f, 1.f, 14, 17)
           || test_interp_align_corner(b, 2, 2.f, 2.f, 0, 0, 1)
           || test_interp_align_corner(b, 2, 4.f, 0.5f, 0, 0, 1)
           || test_interp_align_corner(b, 2, 1.2f, 1.2f, 0, 0, 1)
           || test_interp_align_corner(b, 2, 0.8f, 0.8f, 0, 0, 1)
           || test_interp_align_corner(b, 2, 1.f, 1.f, 10, 12, 1)
           || test_interp_align_corner(b, 2, 1.f, 1.f, 2, 2, 1)
           || test_interp_align_corner(b, 2, 1.f, 1.f, 14, 17, 1)
           || test_interp_ref(b, 2, 10, 12)
           || test_interp_ref(b, 2, 2, 2)
           || test_interp_ref(b, 2, 14, 17)

           || test_interp(c, 2, 2.f, 2.f, 0, 0)
           || test_interp(c, 2, 4.f, 0.5f, 0, 0)
           || test_interp(c, 2, 1.2f, 1.2f, 0, 0)
           || test_interp(c, 2, 0.8f, 0.8f, 0, 0)
           || test_interp(c, 2, 1.f, 1.f, 10, 12)
           || test_interp(c, 2, 1.f, 1.f, 2, 2)
           || test_interp(c, 2, 1.f, 1.f, 14, 17)
           || test_interp_align_corner(c, 2, 2.f, 2.f, 0, 0, 1)
           || test_interp_align_corner(c, 2, 4.f, 0.5f, 0, 0, 1)
           || test_interp_align_corner(c, 2, 1.2f, 1.2f, 0, 0, 1)
           || test_interp_align_corner(c, 2, 0.8f, 0.8f, 0, 0, 1)
           || test_interp_align_corner(c, 2, 1.f, 1.f, 10, 12, 1)
           || test_interp_align_corner(c, 2, 1.f, 1.f, 2, 2, 1)
           || test_interp_align_corner(c, 2, 1.f, 1.f, 14, 17, 1)
           || test_interp_ref(c, 2, 10, 12)
           || test_interp_ref(c, 2, 2, 2)
           || test_interp_ref(c, 2, 14, 17);
}

static int test_interp_2()
{
    ncnn::Mat a = RandomMat(16, 17, 13);
    ncnn::Mat b = RandomMat(18, 19, 12);
    ncnn::Mat c = RandomMat(13, 14, 32);

    return 0
           || test_interp(a, 3, 2.f, 2.f, 0, 0)
           || test_interp(a, 3, 4.f, 0.5f, 0, 0)
           || test_interp(a, 3, 1.2f, 1.2f, 0, 0)
           || test_interp(a, 3, 0.8f, 0.8f, 0, 0)
           || test_interp(a, 3, 1.f, 1.f, 10, 12)
           || test_interp(a, 3, 1.f, 1.f, 2, 2)
           || test_interp(a, 3, 1.f, 1.f, 6, 7)
           || test_interp(a, 3, 1.f, 1.f, 16, 17)
           || test_interp_align_corner(a, 3, 2.f, 2.f, 0, 0, 1)
           || test_interp_align_corner(a, 3, 4.f, 0.5f, 0, 0, 1)
           || test_interp_align_corner(a, 3, 1.2f, 1.2f, 0, 0, 1)
           || test_interp_align_corner(a, 3, 0.8f, 0.8f, 0, 0, 1)
           || test_interp_align_corner(a, 3, 1.f, 1.f, 10, 12, 1)
           || test_interp_align_corner(a, 3, 1.f, 1.f, 2, 2, 1)
           || test_interp_align_corner(a, 3, 1.f, 1.f, 6, 7, 1)
           || test_interp_align_corner(a, 3, 1.f, 1.f, 16, 17, 1)
           || test_interp_ref(a, 3, 2, 2)
           || test_interp_ref(a, 3, 6, 7)
           || test_interp_ref(a, 3, 16, 17)

           || test_interp(b, 3, 2.f, 2.f, 0, 0)
           || test_interp(b, 3, 4.f, 0.5f, 0, 0)
           || test_interp(b, 3, 1.2f, 1.2f, 0, 0)
           || test_interp(b, 3, 0.8f, 0.8f, 0, 0)
           || test_interp(b, 3, 1.f, 1.f, 10, 12)
           || test_interp(b, 3, 1.f, 1.f, 2, 2)
           || test_interp(b, 3, 1.f, 1.f, 6, 7)
           || test_interp(b, 3, 1.f, 1.f, 18, 19)
           || test_interp_align_corner(b, 3, 2.f, 2.f, 0, 0, 1)
           || test_interp_align_corner(b, 3, 4.f, 0.5f, 0, 0, 1)
           || test_interp_align_corner(b, 3, 1.2f, 1.2f, 0, 0, 1)
           || test_interp_align_corner(b, 3, 0.8f, 0.8f, 0, 0, 1)
           || test_interp_align_corner(b, 3, 1.f, 1.f, 10, 12, 1)
           || test_interp_align_corner(b, 3, 1.f, 1.f, 2, 2, 1)
           || test_interp_align_corner(b, 3, 1.f, 1.f, 6, 7, 1)
           || test_interp_align_corner(b, 3, 1.f, 1.f, 18, 19, 1)
           || test_interp_ref(b, 3, 2, 2)
           || test_interp_ref(b, 3, 6, 7)
           || test_interp_ref(b, 3, 18, 19)

           || test_interp(c, 3, 2.f, 2.f, 0, 0)
           || test_interp(c, 3, 4.f, 0.5f, 0, 0)
           || test_interp(c, 3, 1.2f, 1.2f, 0, 0)
           || test_interp(c, 3, 0.8f, 0.8f, 0, 0)
           || test_interp(c, 3, 1.f, 1.f, 10, 12)
           || test_interp(c, 3, 1.f, 1.f, 2, 2)
           || test_interp(c, 3, 1.f, 1.f, 6, 7)
           || test_interp(c, 3, 1.f, 1.f, 18, 19)
           || test_interp_align_corner(c, 3, 2.f, 2.f, 0, 0, 1)
           || test_interp_align_corner(c, 3, 4.f, 0.5f, 0, 0, 1)
           || test_interp_align_corner(c, 3, 1.2f, 1.2f, 0, 0, 1)
           || test_interp_align_corner(c, 3, 0.8f, 0.8f, 0, 0, 1)
           || test_interp_align_corner(c, 3, 1.f, 1.f, 10, 12, 1)
           || test_interp_align_corner(c, 3, 1.f, 1.f, 2, 2, 1)
           || test_interp_align_corner(c, 3, 1.f, 1.f, 6, 7, 1)
           || test_interp_align_corner(c, 3, 1.f, 1.f, 18, 19, 1)
           || test_interp_ref(c, 3, 2, 2)
           || test_interp_ref(c, 3, 6, 7)
           || test_interp_ref(c, 3, 18, 19);
}

static int test_interp_3()
{
    ncnn::Mat a = RandomMat(15, 7);
    ncnn::Mat b = RandomMat(14, 12);
    ncnn::Mat c = RandomMat(13, 32);

    return 0
           || test_interp(a, 1, 2.f, 0)
           || test_interp(a, 1, 0.5f, 0)
           || test_interp(a, 1, 1.2f, 0)
           || test_interp(a, 1, 0.8f, 0)
           || test_interp(a, 1, 1.f, 12)
           || test_interp(a, 1, 1.f, 2)
           || test_interp(a, 1, 1.f, 16)
           || test_interp_ref(a, 1, 12)
           || test_interp_ref(a, 1, 2)
           || test_interp_ref(a, 1, 16)

           || test_interp(b, 1, 2.f, 0)
           || test_interp(b, 1, 0.5f, 0)
           || test_interp(b, 1, 1.2f, 0)
           || test_interp(b, 1, 0.8f, 0)
           || test_interp(b, 1, 1.f, 12)
           || test_interp(b, 1, 1.f, 2)
           || test_interp(b, 1, 1.f, 17)
           || test_interp_ref(b, 1, 12)
           || test_interp_ref(b, 1, 2)
           || test_interp_ref(b, 1, 17)

           || test_interp(c, 1, 2.f, 0)
           || test_interp(c, 1, 0.5f, 0)
           || test_interp(c, 1, 1.2f, 0)
           || test_interp(c, 1, 0.8f, 0)
           || test_interp(c, 1, 1.f, 12)
           || test_interp(c, 1, 1.f, 2)
           || test_interp(c, 1, 1.f, 17)
           || test_interp_ref(c, 1, 12)
           || test_interp_ref(c, 1, 2)
           || test_interp_ref(c, 1, 17);
}

static int test_interp_4()
{
    ncnn::Mat a = RandomMat(15, 7);
    ncnn::Mat b = RandomMat(14, 12);
    ncnn::Mat c = RandomMat(13, 32);

    return 0
           || test_interp(a, 2, 2.f, 0)
           || test_interp(a, 2, 0.5f, 0)
           || test_interp(a, 2, 1.2f, 0)
           || test_interp(a, 2, 0.8f, 0)
           || test_interp(a, 2, 1.f, 12)
           || test_interp(a, 2, 1.f, 2)
           || test_interp(a, 2, 1.f, 16)
           || test_interp_align_corner(a, 2, 2.f, 0, 1)
           || test_interp_align_corner(a, 2, 0.5f, 0, 1)
           || test_interp_align_corner(a, 2, 1.2f, 0, 1)
           || test_interp_align_corner(a, 2, 0.8f, 0, 1)
           || test_interp_align_corner(a, 2, 1.f, 12, 1)
           || test_interp_align_corner(a, 2, 1.f, 2, 1)
           || test_interp_align_corner(a, 2, 1.f, 16, 1)
           || test_interp_ref(a, 2, 12)
           || test_interp_ref(a, 2, 2)
           || test_interp_ref(a, 2, 16)

           || test_interp(b, 2, 2.f, 0)
           || test_interp(b, 2, 0.5f, 0)
           || test_interp(b, 2, 1.2f, 0)
           || test_interp(b, 2, 0.8f, 0)
           || test_interp(b, 2, 1.f, 12)
           || test_interp(b, 2, 1.f, 2)
           || test_interp(b, 2, 1.f, 17)
           || test_interp_align_corner(b, 2, 2.f, 0, 1)
           || test_interp_align_corner(b, 2, 0.5f, 0, 1)
           || test_interp_align_corner(b, 2, 1.2f, 0, 1)
           || test_interp_align_corner(b, 2, 0.8f, 0, 1)
           || test_interp_align_corner(b, 2, 1.f, 12, 1)
           || test_interp_align_corner(b, 2, 1.f, 2, 1)
           || test_interp_align_corner(b, 2, 1.f, 17, 1)
           || test_interp_ref(b, 2, 12)
           || test_interp_ref(b, 2, 2)
           || test_interp_ref(b, 2, 17)

           || test_interp(c, 2, 2.f, 0)
           || test_interp(c, 2, 0.5f, 0)
           || test_interp(c, 2, 1.2f, 0)
           || test_interp(c, 2, 0.8f, 0)
           || test_interp(c, 2, 1.f, 12)
           || test_interp(c, 2, 1.f, 2)
           || test_interp(c, 2, 1.f, 17)
           || test_interp_align_corner(c, 2, 2.f, 0, 1)
           || test_interp_align_corner(c, 2, 0.5f, 0, 1)
           || test_interp_align_corner(c, 2, 1.2f, 0, 1)
           || test_interp_align_corner(c, 2, 0.8f, 0, 1)
           || test_interp_align_corner(c, 2, 1.f, 12, 1)
           || test_interp_align_corner(c, 2, 1.f, 2, 1)
           || test_interp_align_corner(c, 2, 1.f, 17, 1)
           || test_interp_ref(c, 2, 12)
           || test_interp_ref(c, 2, 2)
           || test_interp_ref(c, 2, 17);
}

static int test_interp_5()
{
    ncnn::Mat a = RandomMat(16, 13);
    ncnn::Mat b = RandomMat(18, 12);
    ncnn::Mat c = RandomMat(13, 32);

    return 0
           || test_interp(a, 3, 2.f, 0)
           || test_interp(a, 3, 0.5f, 0)
           || test_interp(a, 3, 1.2f, 0)
           || test_interp(a, 3, 0.8f, 0)
           || test_interp(a, 3, 1.f, 12)
           || test_interp(a, 3, 1.f, 2)
           || test_interp(a, 3, 1.f, 7)
           || test_interp(a, 3, 1.f, 17)
           || test_interp_align_corner(a, 3, 2.f, 0, 1)
           || test_interp_align_corner(a, 3, 0.5f, 0, 1)
           || test_interp_align_corner(a, 3, 1.2f, 0, 1)
           || test_interp_align_corner(a, 3, 0.8f, 0, 1)
           || test_interp_align_corner(a, 3, 1.f, 12, 1)
           || test_interp_align_corner(a, 3, 1.f, 2, 1)
           || test_interp_align_corner(a, 3, 1.f, 7, 1)
           || test_interp_align_corner(a, 3, 1.f, 17, 1)
           || test_interp_ref(a, 3, 2)
           || test_interp_ref(a, 3, 7)
           || test_interp_ref(a, 3, 17)

           || test_interp(b, 3, 2.f, 0)
           || test_interp(b, 3, 0.5f, 0)
           || test_interp(b, 3, 1.2f, 0)
           || test_interp(b, 3, 0.8f, 0)
           || test_interp(b, 3, 1.f, 12)
           || test_interp(b, 3, 1.f, 2)
           || test_interp(b, 3, 1.f, 7)
           || test_interp(b, 3, 1.f, 19)
           || test_interp_align_corner(b, 3, 2.f, 0, 1)
           || test_interp_align_corner(b, 3, 0.5f, 0, 1)
           || test_interp_align_corner(b, 3, 1.2f, 0, 1)
           || test_interp_align_corner(b, 3, 0.8f, 0, 1)
           || test_interp_align_corner(b, 3, 1.f, 12, 1)
           || test_interp_align_corner(b, 3, 1.f, 2, 1)
           || test_interp_align_corner(b, 3, 1.f, 7, 1)
           || test_interp_align_corner(b, 3, 1.f, 19, 1)
           || test_interp_ref(b, 3, 2)
           || test_interp_ref(b, 3, 7)
           || test_interp_ref(b, 3, 19)

           || test_interp(c, 3, 2.f, 0)
           || test_interp(c, 3, 0.5f, 0)
           || test_interp(c, 3, 1.2f, 0)
           || test_interp(c, 3, 0.8f, 0)
           || test_interp(c, 3, 1.f, 12)
           || test_interp(c, 3, 1.f, 2)
           || test_interp(c, 3, 1.f, 7)
           || test_interp(c, 3, 1.f, 19)
           || test_interp_align_corner(c, 3, 2.f, 0, 1)
           || test_interp_align_corner(c, 3, 0.5f, 0, 1)
           || test_interp_align_corner(c, 3, 1.2f, 0, 1)
           || test_interp_align_corner(c, 3, 0.8f, 0, 1)
           || test_interp_align_corner(c, 3, 1.f, 12, 1)
           || test_interp_align_corner(c, 3, 1.f, 2, 1)
           || test_interp_align_corner(c, 3, 1.f, 7, 1)
           || test_interp_align_corner(c, 3, 1.f, 19, 1)
           || test_interp_ref(c, 3, 2)
           || test_interp_ref(c, 3, 7)
           || test_interp_ref(c, 3, 19);
}

static int test_interp_6()
{
    ncnn::Mat a = RandomMat(17);
    ncnn::Mat b = RandomMat(20);
    ncnn::Mat c = RandomMat(48);

    return 0
           || test_interp(a, 1, 2.f, 3.f, 0, 0)
           || test_interp(a, 1, 1.f, 1.f, 10, 12)
           || test_interp(a, 1, 1.f, 1.f, 15, 16)
           || test_interp_ref(a, 1, 10, 12)
           || test_interp_ref(a, 1, 4, 4)
           || test_interp_ref(a, 1, 15, 16)

           || test_interp(b, 1, 4.f, 5.f, 0, 0)
           || test_interp(b, 1, 1.f, 1.f, 10, 12)
           || test_interp(b, 1, 1.f, 1.f, 14, 17)
           || test_interp_ref(b, 1, 5, 5)
           || test_interp_ref(b, 1, 14, 17)

           || test_interp(c, 1, 6.f, 7.f, 0, 0)
           || test_interp(c, 1, 1.f, 1.f, 10, 12)
           || test_interp(c, 1, 1.f, 1.f, 14, 17)
           || test_interp_ref(c, 1, 6, 6)
           || test_interp_ref(c, 1, 14, 17);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_interp_0()
           || test_interp_1()
           || test_interp_2()
           || test_interp_3()
           || test_interp_4()
           || test_interp_5()
           || test_interp_6();
}
