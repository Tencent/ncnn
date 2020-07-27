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

#include "layer/permute.h"
#include "testutil.h"

static int test_permute(const ncnn::Mat& a, int order_type)
{
    ncnn::ParamDict pd;
    pd.set(0, order_type);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Permute>("Permute", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_permute failed a.dims=%d a=(%d %d %d) order_type=%d\n", a.dims, a.w, a.h, a.c, order_type);
    }

    return ret;
}

static int test_permute_0()
{
    ncnn::Mat a = RandomMat(128);
    ncnn::Mat b = RandomMat(127);

    return 0
           || test_permute(a, 0)
           || test_permute(b, 0);
}

static int test_permute_1()
{
    ncnn::Mat a = RandomMat(12, 32);
    ncnn::Mat b = RandomMat(8, 15);
    ncnn::Mat c = RandomMat(11, 16);
    ncnn::Mat d = RandomMat(7, 9);

    for (int order_type = 0; order_type < 2; order_type++)
    {
        int ret = 0
                  || test_permute(a, order_type)
                  || test_permute(b, order_type)
                  || test_permute(c, order_type)
                  || test_permute(d, order_type);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_permute_2()
{
    ncnn::Mat a = RandomMat(8, 16, 32);
    ncnn::Mat b = RandomMat(12, 8, 16);
    ncnn::Mat c = RandomMat(7, 14, 12);
    ncnn::Mat d = RandomMat(4, 4, 13);
    ncnn::Mat e = RandomMat(1, 2, 7);
    ncnn::Mat f = RandomMat(8, 5, 6);

    for (int order_type = 0; order_type < 6; order_type++)
    {
        int ret = 0
                  || test_permute(a, order_type)
                  || test_permute(b, order_type)
                  || test_permute(c, order_type)
                  || test_permute(d, order_type)
                  || test_permute(e, order_type)
                  || test_permute(f, order_type);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_permute_0()
           || test_permute_1()
           || test_permute_2();
}
