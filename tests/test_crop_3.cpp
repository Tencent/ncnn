// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_crop(const ncnn::Mat& a, const char* starts_expr, const char* ends_expr, const char* axes_expr)
{
    ncnn::ParamDict pd;
    pd.set(19, std::string(starts_expr));
    pd.set(20, std::string(ends_expr));
    pd.set(21, std::string(axes_expr));

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Crop", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d) starts_expr=%s ends_expr=%s axes_expr=%s\n", a.dims, a.w, a.h, a.d, a.c, starts_expr, ends_expr, axes_expr);
    }

    return ret;
}

static int test_crop(const std::vector<ncnn::Mat>& as, const char* starts_expr, const char* ends_expr, const char* axes_expr)
{
    ncnn::ParamDict pd;
    pd.set(19, std::string(starts_expr));
    pd.set(20, std::string(ends_expr));
    pd.set(21, std::string(axes_expr));

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Crop", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d) starts_expr=%s ends_expr=%s axes_expr=%s\n", as[0].dims, as[0].w, as[0].h, as[0].d, as[0].c, starts_expr, ends_expr, axes_expr);
    }

    return ret;
}

static int test_crop_0()
{
    ncnn::Mat a = RandomMat(13, 12, 25, 32);
    ncnn::Mat b = RandomMat(13, 12, 32);

    return 0
           || test_crop(a, "2", "-2", "0")
           || test_crop(b, "2", "-2", "0")
           || test_crop(a, "10", "11", "1")
           || test_crop(a, "-(0w,3),0h//2,floor(*(0c,0.3))", "-1,0h,ceil(*(0c,0.9))", "3,2,0");
}

static int test_crop_1()
{
    std::vector<ncnn::Mat> as(2);
    as[0] = RandomMat(14, 15, 16);
    as[1] = RandomMat(28, 45, 48);

    std::vector<ncnn::Mat> bs(2);
    bs[0] = RandomMat(4, 5, 3, 16);
    bs[1] = RandomMat(8, 5, 3, 4);

    return 0
           || test_crop(as, "-(1w,20)", "-2", "0")
           || test_crop(bs, "-(1w,4)", "neg(1h,3)", "0")
           || test_crop(as, "//(1h,15)", "neg(//(1w,7))", "2")
           || test_crop(as, "//(100,0h),round(fmod(100,0c))", "-233,min(1c,0c)", "1,0");
}

int main()
{
    SRAND(776757);

    return 0
           || test_crop_0()
           || test_crop_1();
}
