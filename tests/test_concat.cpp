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

#include "layer/concat.h"
#include "testutil.h"

static int test_concat(const std::vector<ncnn::Mat>& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); //axis

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::Concat>("Concat", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_concat failed a[0].dims=%d a[0]=(%d %d %d) axis=%d\n", a[0].dims, a[0].w, a[0].h, a[0].c, axis);
    }

    return ret;
}

static int test_concat_0()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(16, 12, 8);
    a[1] = RandomMat(16, 12, 8);
    a[2] = RandomMat(16, 12, 8);

    return 0
           || test_concat(a, 0)
           || test_concat(a, 1)
           || test_concat(a, 2);
}

static int test_concat_1()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(7, 3, 3);
    a[1] = RandomMat(7, 3, 8);
    a[2] = RandomMat(7, 3, 5);

    return test_concat(a, 0);
}

static int test_concat_2()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(7, 3, 2);
    a[1] = RandomMat(7, 8, 2);
    a[2] = RandomMat(7, 5, 2);

    return test_concat(a, 1);
}

static int test_concat_3()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(7, 3, 8);
    a[1] = RandomMat(7, 8, 8);
    a[2] = RandomMat(7, 5, 8);

    return test_concat(a, 1);
}

static int test_concat_4()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(3, 7, 2);
    a[1] = RandomMat(8, 7, 2);
    a[2] = RandomMat(5, 7, 2);

    return test_concat(a, 2);
}

static int test_concat_5()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(3, 7, 8);
    a[1] = RandomMat(8, 7, 8);
    a[2] = RandomMat(5, 7, 8);

    return test_concat(a, 2);
}

static int test_concat_6()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(7, 3);
    a[1] = RandomMat(7, 8);
    a[2] = RandomMat(7, 5);

    return test_concat(a, 0);
}

static int test_concat_7()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(3, 2);
    a[1] = RandomMat(8, 2);
    a[2] = RandomMat(5, 2);

    return test_concat(a, 1);
}

static int test_concat_8()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(3, 8);
    a[1] = RandomMat(8, 8);
    a[2] = RandomMat(5, 8);

    return test_concat(a, 1);
}

static int test_concat_9()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(3);
    a[1] = RandomMat(8);
    a[2] = RandomMat(5);

    return test_concat(a, 0);
}

static int test_concat_10()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(4);
    a[1] = RandomMat(8);
    a[2] = RandomMat(12);

    return test_concat(a, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_concat_0()
           || test_concat_1()
           || test_concat_2()
           || test_concat_3()
           || test_concat_4()
           || test_concat_5()
           || test_concat_6()
           || test_concat_7()
           || test_concat_8()
           || test_concat_9()
           || test_concat_10();
}
