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

#include "layer.h"
#include "testutil.h"

// 为兼容低于c++11弃用如下实现
// ncnn::Mat axis_mat(axis.size());
// for (size_t i = 0; i < axis.size(); i++)
// {
//     axis_mat[i] = axis[i];
// }
static ncnn::Mat IntArrayMat(int a0)
{
    ncnn::Mat m(1);
    int* p = m;
    p[0] = a0;
    return m;
}

static ncnn::Mat IntArrayMat(int a0, int a1)
{
    ncnn::Mat m(2);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    return m;
}

static ncnn::Mat IntArrayMat(int a0, int a1, int a2)
{
    ncnn::Mat m(3);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    return m;
}

static ncnn::Mat IntArrayMat(int a0, int a1, int a2, int a3)
{
    ncnn::Mat m(4);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    p[3] = a3;
    return m;
}

static int test_flip(const ncnn::Mat& a, const ncnn::Mat& axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Flip", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_flip failed a.dims=%d a=(%d %d %d) axis_w=%d", a.dims, a.w, a.h, a.c, axis.w);
    }

    return ret;
}

static int test_flip_0()
{
    return 0
           || test_flip(RandomMat(2, 3, 4, 5), IntArrayMat(0))
           || test_flip(RandomMat(3, 2, 4, 5), IntArrayMat(1))
           || test_flip(RandomMat(4, 3, 2, 5), IntArrayMat(2))
           || test_flip(RandomMat(2, 3, 1, 5), IntArrayMat(3))
           || test_flip(RandomMat(6, 3, 4, 5), IntArrayMat(0, 1))
           || test_flip(RandomMat(2, 3, 1, 6), IntArrayMat(0, 2))
           || test_flip(RandomMat(5, 1, 2, 5), IntArrayMat(0, 3))
           || test_flip(RandomMat(5, 2, 1, 5), IntArrayMat(1, 2))
           || test_flip(RandomMat(4, 5, 2, 3), IntArrayMat(1, 3))
           || test_flip(RandomMat(2, 6, 4, 5), IntArrayMat(2, 3))
           || test_flip(RandomMat(6, 1, 4, 5), IntArrayMat(0, 1, 2))
           || test_flip(RandomMat(5, 2, 1, 5), IntArrayMat(0, 1, 3))
           || test_flip(RandomMat(4, 3, 3, 5), IntArrayMat(0, 2, 3))
           || test_flip(RandomMat(4, 3, 4, 5), IntArrayMat(1, 2, 3))
           || test_flip(RandomMat(6, 3, 3, 2), IntArrayMat(0, 1, 2, 3));
}

static int test_flip_1()
{
    return 0
           || test_flip(RandomMat(2, 3, 5), IntArrayMat(0))
           || test_flip(RandomMat(3, 3, 5), IntArrayMat(1))
           || test_flip(RandomMat(4, 3, 5), IntArrayMat(2))
           || test_flip(RandomMat(3, 1, 5), IntArrayMat(0, 1))
           || test_flip(RandomMat(3, 2, 5), IntArrayMat(0, 2))
           || test_flip(RandomMat(3, 3, 4), IntArrayMat(1, 2))
           || test_flip(RandomMat(4, 3, 2), IntArrayMat(0, 1, 2));
}

static int test_flip_2()
{
    return 0
           || test_flip(RandomMat(8, 2), IntArrayMat(-2))
           || test_flip(RandomMat(16, 3), IntArrayMat(-1))
           || test_flip(RandomMat(7, 2), IntArrayMat(-2, -1));
}

static int test_flip_3()
{
    return 0
           || test_flip(RandomMat(18), IntArrayMat(-1));
}

int main()
{
    SRAND(7767517);
    return 0
           || test_flip_0()
           || test_flip_1()
           || test_flip_2()
           || test_flip_3();
}