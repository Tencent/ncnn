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

static int test_reshape(const ncnn::Mat& a, const char* shape_expr)
{
    ncnn::ParamDict pd;
    pd.set(6, std::string(shape_expr));

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Reshape", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reshape failed a.dims=%d a=(%d %d %d %d) shape_expr=%s\n", a.dims, a.w, a.h, a.d, a.c, shape_expr);
    }

    return ret;
}

static int test_reshape(const std::vector<ncnn::Mat>& as, const char* shape_expr)
{
    ncnn::ParamDict pd;
    pd.set(6, std::string(shape_expr));

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Reshape", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_reshape failed a.dims=%d a=(%d %d %d %d) shape_expr=%s\n", as[0].dims, as[0].w, as[0].h, as[0].d, as[0].c, shape_expr);
    }

    return ret;
}

static int test_reshape_0()
{
    ncnn::Mat a = RandomMat(3, 2, 25, 32);

    return 0
           || test_reshape(a, "0w,0h,*(0d,2),-1")
           || test_reshape(a, "-1,0w,5,4")
           || test_reshape(a, "-1,square(neg(abs(max(min(//(*(-(+(0w,1),-1),4),4),100),-4))))")
           || test_reshape(a, "-1,square(neg(abs(max(min(//(*(-(+(0w,1.0),-1.0),2.0),2.0),100.3),-2.2)))")
           || test_reshape(a, "-1,trunc(*(round(-(floor(+(ceil(*(0w,1.2)),0.7)),-0.4)),1.0001))")
           || test_reshape(a, "-1,ceil(sqrt(square(*(asinh(sinh(atanh(tanh(atan(tan(acos(cos(asin(sin(/(0w,2))))))))))),16))))");
}

static int test_reshape_1()
{
    std::vector<ncnn::Mat> as(2);
    as[0] = RandomMat(14, 15, 16);
    as[1] = RandomMat(28, 45, 48);

    return 0
           || test_reshape(as, "*(1w,0.5),/(1h,3),-(1c,32)")
           || test_reshape(as, "*(0w,0h),-(-(1c,0c),16)");
}

static int test_reshape_2()
{
    ncnn::Mat a = RandomMat(14, 15, 16);

    return 0
           || test_reshape(a, "*(0w,0.5),/(0h,3),-1")
           || test_reshape(a, "-1")
           || test_reshape(a, "*(0w,0h),0c");
}

int main()
{
    SRAND(7767517);

    return 0
           || test_reshape_0()
           || test_reshape_1()
           || test_reshape_2();
}
