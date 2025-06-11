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

static int test_crop_oom(const ncnn::Mat& a, int woffset, int hoffset, int doffset, int coffset, int outw, int outh, int outd, int outc, int woffset2, int hoffset2, int doffset2, int coffset2)
{
    ncnn::ParamDict pd;
    pd.set(0, woffset);
    pd.set(1, hoffset);
    pd.set(13, doffset);
    pd.set(2, coffset);
    pd.set(3, outw);
    pd.set(4, outh);
    pd.set(14, outd);
    pd.set(5, outc);
    pd.set(6, woffset2);
    pd.set(7, hoffset2);
    pd.set(15, doffset2);
    pd.set(8, coffset2);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Crop", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop_oom failed a.dims=%d a=(%d %d %d %d) woffset=%d hoffset=%d doffset=%d coffset=%d outw=%d outh=%d outd=%d outc=%d woffset2=%d hoffset2=%d doffset2=%d coffset2=%d\n", a.dims, a.w, a.h, a.d, a.c, woffset, hoffset, doffset, coffset, outw, outh, outd, outc, woffset2, hoffset2, doffset2, coffset2);
    }

    return ret;
}

static int test_crop_oom(const ncnn::Mat& a, const char* starts_expr, const char* ends_expr, const char* axes_expr)
{
    ncnn::ParamDict pd;
    pd.set(19, std::string(starts_expr));
    pd.set(20, std::string(ends_expr));
    pd.set(21, std::string(axes_expr));

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Crop", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop_oom failed a.dims=%d a=(%d %d %d %d) starts_expr=%s ends_expr=%s axes_expr=%s\n", a.dims, a.w, a.h, a.d, a.c, starts_expr, ends_expr, axes_expr);
    }

    return ret;
}

static int test_crop_oom(const std::vector<ncnn::Mat>& as, const char* starts_expr, const char* ends_expr, const char* axes_expr)
{
    ncnn::ParamDict pd;
    pd.set(19, std::string(starts_expr));
    pd.set(20, std::string(ends_expr));
    pd.set(21, std::string(axes_expr));

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Crop", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop_oom failed a.dims=%d a=(%d %d %d %d) starts_expr=%s ends_expr=%s axes_expr=%s\n", as[0].dims, as[0].w, as[0].h, as[0].d, as[0].c, starts_expr, ends_expr, axes_expr);
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
           || test_crop_oom(a, 1, 1, 1, 1, -233, -233, -233, -233, 1, 1, 1, 1)
           || test_crop_oom(b, 1, 1, 0, 1, -233, -233, 0, -233, 1, 1, 0, 1)
           || test_crop_oom(c, 1, 1, 0, 0, -233, -233, 0, 0, 1, 1, 0, 0)
           || test_crop_oom(d, 1, 0, 0, 0, -233, 0, 0, 0, 1, 0, 0, 0)
           || test_crop_oom(a, 2, 2, 2, 2, 6, 6, 6, 16, 0, 0, 0, 0)
           || test_crop_oom(b, 2, 2, 0, 2, 6, 6, 0, 16, 0, 0, 0, 0)
           || test_crop_oom(c, 2, 2, 0, 0, 6, 16, 0, 0, 0, 0, 0, 0)
           || test_crop_oom(d, 2, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0)
           || test_crop_oom(a, 3, 3, 3, 16, 3, 4, 5, 16, 0, 0, 0, 0)
           || test_crop_oom(b, 3, 3, 0, 16, 3, 4, 0, 16, 0, 0, 0, 0)
           || test_crop_oom(c, 3, 16, 0, 0, 3, 16, 0, 0, 0, 0, 0, 0)
           || test_crop_oom(d, 16, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0);
}

static int test_crop_1()
{
    ncnn::Mat a = RandomMat(13, 12, 25, 47);
    ncnn::Mat b = RandomMat(13, 12, 47);
    ncnn::Mat c = RandomMat(13, 47);
    ncnn::Mat d = RandomMat(129);

    return 0
           || test_crop_oom(a, 1, 1, 1, 1, -233, -233, -233, -233, 1, 1, 1, 1)
           || test_crop_oom(b, 1, 1, 0, 1, -233, -233, 0, -233, 1, 1, 0, 1)
           || test_crop_oom(c, 1, 1, 0, 0, -233, -233, 0, 0, 1, 1, 0, 0)
           || test_crop_oom(d, 1, 0, 0, 0, -233, 0, 0, 0, 1, 0, 0, 0)
           || test_crop_oom(a, 2, 2, 2, 2, 6, 6, 6, 16, 0, 0, 0, 0)
           || test_crop_oom(b, 2, 2, 0, 2, 6, 6, 0, 16, 0, 0, 0, 0)
           || test_crop_oom(c, 2, 2, 0, 0, 6, 16, 0, 0, 0, 0, 0, 0)
           || test_crop_oom(d, 2, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0)
           || test_crop_oom(a, 3, 3, 3, 16, 6, 6, 6, 16, 0, 0, 0, 0)
           || test_crop_oom(b, 3, 3, 0, 16, 6, 6, 0, 16, 0, 0, 0, 0)
           || test_crop_oom(c, 3, 16, 0, 0, 6, 16, 0, 0, 0, 0, 0, 0)
           || test_crop_oom(d, 16, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0);
}

static int test_crop_2()
{
    ncnn::Mat a = RandomMat(13, 12, 25, 48);
    ncnn::Mat b = RandomMat(13, 12, 48);
    ncnn::Mat c = RandomMat(13, 48);
    ncnn::Mat d = RandomMat(128);

    return 0
           || test_crop_oom(a, "2", "-2", "0")
           || test_crop_oom(b, "2", "-2", "0")
           || test_crop_oom(c, "2", "-2", "0")
           || test_crop_oom(d, "2", "-2", "0")
           || test_crop_oom(a, "16", "32", "-4")
           || test_crop_oom(b, "16", "32", "-3")
           || test_crop_oom(c, "16", "32", "-2")
           || test_crop_oom(d, "16", "32", "-1")
           || test_crop_oom(a, "16,//(0d,4),2,1", "32,-1,-2,-3", "0,1,2,3")
           || test_crop_oom(b, "16,//(0h,4),2", "32,-1,-(0w,2)", "0,1,2")
           || test_crop_oom(c, "16,//(0w,4)", "32,-2", "0,1")
           || test_crop_oom(a, "10", "11", "1")
           || test_crop_oom(b, "1,1", "-(0c,15),-(0w,5)", "0,2")
           || test_crop_oom(a, "-(0w,3),0h//2,floor(*(0c,0.3))", "-1,0h,ceil(*(0c,0.9))", "3,2,0")
           || test_crop_oom(b, "-(0w,3),0h//2,floor(*(0c,0.3))", "-1,0h,ceil(*(0c,0.9))", "2,1,0")
           || test_crop_oom(c, "-(0w,3),floor(*(0h,0.3))", "-1,ceil(*(0h,0.9))", "1,0")
           || test_crop_oom(d, "floor(*(0w,0.3))", "ceil(*(0w,0.9))", "0");
}

static int test_crop_3()
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
           || test_crop_oom(as, "*(1c,4)", "*(1c,8)", "-4")
           || test_crop_oom(bs, "1c", "-(0c,1c)", "-3")
           || test_crop_oom(cs, "+(1h,10)", "-(1h,22)", "-2")
           || test_crop_oom(ds, "1w", "2w", "-1")
           || test_crop_oom(as, "16,//(min(0w,1d),4),2,3", "32,-1,-2,-3", "0,1,2,3")
           || test_crop_oom(bs, "16,//(min(0w,1h),4),2", "32,-1,-(0w,2)", "0,1,2")
           || test_crop_oom(cs, "16,//(min(0w,1w),4)", "32,-2", "0,1")
           || test_crop_oom(bs, "1,//(1w,7)", "+(1c,1),-(0w,2)", "0,2")
           || test_crop_oom(as, "-(1w,4)", "neg(1h,3)", "0")
           || test_crop_oom(bs, "-(1w,20)", "-2", "0")
           || test_crop_oom(bs, "//(1h,15)", "neg(//(1w,7))", "2")
           || test_crop_oom(bs, "//(100,0h),round(fmod(100,0c))", "-233,min(1c,0c)", "1,0");
}

int main()
{
    SRAND(776757);

    return 0
           || test_crop_0()
           || test_crop_1()
           || test_crop_2()
           || test_crop_3();
}
