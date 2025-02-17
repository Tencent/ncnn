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

#include "testutil.h"

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

static int test_index_select(const ncnn::Mat& a, const ncnn::Mat& index, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(2);
    a0[0] = a;
    a0[1] = index;

    int ret = test_layer("IndexSelect", pd, weights, a0);
    if (ret != 0)
    {
        fprintf(stderr, "test_index_select failed a.dims=%d a=(%d %d %d %d) index.w=%d axis=%d\n", a.dims, a.w, a.h, a.d, a.c, index.w, axis);
    }

    return ret;
}

static int test_index_select_0()
{
    return 0
           || test_index_select(RandomMat(3, 4, 5, 6), IntArrayMat(1, 0), 0)
           || test_index_select(RandomMat(4, 4, 5, 6), IntArrayMat(1, 0), 1)
           || test_index_select(RandomMat(3, 4, 7, 6), IntArrayMat(1, 0), 2)
           || test_index_select(RandomMat(7, 2, 5, 6), IntArrayMat(1, 0, 3), 3);
}

static int test_index_select_1()
{
    return 0
           || test_index_select(RandomMat(2, 3, 5), IntArrayMat(1, 0), -3)
           || test_index_select(RandomMat(4, 3, 5), IntArrayMat(0, 1), -2)
           || test_index_select(RandomMat(6, 4, 5), IntArrayMat(2, 0), -1);
}

static int test_index_select_2()
{
    return 0
           || test_index_select(RandomMat(8, 6), IntArrayMat(1, 4, 3), 0)
           || test_index_select(RandomMat(8, 7), IntArrayMat(3, 5), 1);
}

static int test_index_select_3()
{
    return 0
           || test_index_select(RandomMat(18), IntArrayMat(1, 7, 9, 15), -1);
}

int main()
{
    SRAND(7767517);
    return 0
           || test_index_select_0()
           || test_index_select_1()
           || test_index_select_2()
           || test_index_select_3();
}