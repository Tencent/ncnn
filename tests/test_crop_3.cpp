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
    ncnn::Mat a = RandomMat(13, 12, 25, 48);
    ncnn::Mat b = RandomMat(13, 12, 48);
    ncnn::Mat c = RandomMat(13, 48);
    ncnn::Mat d = RandomMat(128);

    return 0
           || test_crop(a, "2", "-2", "0")
           || test_crop(b, "2", "-2", "0")
           || test_crop(c, "2", "-2", "0")
           || test_crop(d, "2", "-2", "0")
           || test_crop(a, "16", "32", "-4")
           || test_crop(b, "16", "32", "-3")
           || test_crop(c, "16", "32", "-2")
           || test_crop(d, "16", "32", "-1")
           || test_crop(a, "16,//(0d,4),2,3", "32,-1,-2,-3", "0,1,2,3")
           || test_crop(b, "16,//(0h,4),2", "32,-1,-(0w,2)", "0,1,2")
           || test_crop(c, "16,//(0w,4)", "32,-2", "0,1")
           || test_crop(a, "10", "11", "1")
           || test_crop(b, "1,1", "-(0c,15),-(0w,5)", "0,2")
           || test_crop(a, "-(0w,3),0h//2,floor(*(0c,0.3))", "-1,0h,ceil(*(0c,0.9))", "3,2,0")
           || test_crop(b, "-(0w,3),0h//2,floor(*(0c,0.3))", "-1,0h,ceil(*(0c,0.9))", "2,1,0")
           || test_crop(c, "-(0w,3),floor(*(0h,0.3))", "-1,ceil(*(0h,0.9))", "1,0")
           || test_crop(d, "floor(*(0w,0.3))", "ceil(*(0w,0.9))", "0");
}

static int test_crop_1()
{
    std::vector<ncnn::Mat> as(2);
    as[0] = RandomMat(14, 15, 13, 48);
    as[1] = RandomMat(8, 5, 3, 4);

    std::vector<ncnn::Mat> bs(2);
    bs[0] = RandomMat(14, 15, 48);
    bs[1] = RandomMat(28, 45, 16);

    std::vector<ncnn::Mat> cs(2);
    cs[0] = RandomMat(24, 48);
    cs[1] = RandomMat(28, 6);

    std::vector<ncnn::Mat> ds(3);
    ds[0] = RandomMat(128);
    ds[1] = RandomMat(16);
    ds[2] = RandomMat(64);

    return 0
           || test_crop(as, "*(1c,4)", "*(1c,8)", "-4")
           || test_crop(bs, "1c", "-(0c,1c)", "-3")
           || test_crop(cs, "+(1h,10)", "-(1h,22)", "-2")
           || test_crop(ds, "1w", "2w", "-1")
           || test_crop(as, "16,//(min(0w,1d),4),2,3", "32,-1,-2,-3", "0,1,2,3")
           || test_crop(bs, "16,//(min(0w,1h),4),2", "32,-1,-(0w,2)", "0,1,2")
           || test_crop(cs, "16,//(min(0w,1w),4)", "32,-2", "0,1")
           || test_crop(bs, "1,//(1w,7)", "+(1c,1),-(0w,2)", "0,2")
           || test_crop(as, "-(1w,4)", "neg(1h,3)", "0")
           || test_crop(bs, "-(1w,20)", "-2", "0")
           || test_crop(bs, "//(1h,15)", "neg(//(1w,7))", "2")
           || test_crop(bs, "//(100,0h),round(fmod(100,0c))", "-233,min(1c,0c)", "1,0");
}

int main()
{
    SRAND(776757);

    return 0
           || test_crop_0()
           || test_crop_1();
}
