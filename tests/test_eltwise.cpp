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

#include "layer/eltwise.h"
#include "testutil.h"

static void print_float_array(const ncnn::Mat& a)
{
    fprintf(stderr, "[");
    for (int i = 0; i < a.w; i++)
    {
        fprintf(stderr, " %f", a[i]);
    }
    fprintf(stderr, " ]");
}

static int test_eltwise(const std::vector<ncnn::Mat>& a, int op_type, const ncnn::Mat& coeffs)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, coeffs);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::Eltwise>("Eltwise", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_eltwise failed a[0].dims=%d a[0]=(%d %d %d) op_type=%d", a[0].dims, a[0].w, a[0].h, a[0].c, op_type);
        fprintf(stderr, " coeffs=");
        print_float_array(coeffs);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_eltwise_0()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(16, 12, 8);
    a[1] = RandomMat(16, 12, 8);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(2))
           || test_eltwise(a, 1, RandomMat(2))
           || test_eltwise(a, 2, RandomMat(2));
}

static int test_eltwise_1()
{
    std::vector<ncnn::Mat> a(4);
    a[0] = RandomMat(7, 3, 5);
    a[1] = RandomMat(7, 3, 5);
    a[2] = RandomMat(7, 3, 5);
    a[3] = RandomMat(7, 3, 5);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(4))
           || test_eltwise(a, 1, RandomMat(4))
           || test_eltwise(a, 2, RandomMat(4));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_eltwise_0()
           || test_eltwise_1();
}
