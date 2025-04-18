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

static std::vector<int> IntArray(int a0)
{
    std::vector<int> m(1);
    m[0] = a0;
    return m;
}

static std::vector<int> IntArray(int a0, int a1)
{
    std::vector<int> m(2);
    m[0] = a0;
    m[1] = a1;
    return m;
}

static std::vector<int> IntArray(int a0, int a1, int a2)
{
    std::vector<int> m(3);
    m[0] = a0;
    m[1] = a1;
    m[2] = a2;
    return m;
}

static std::vector<int> IntArray(int a0, int a1, int a2, int a3)
{
    std::vector<int> m(4);
    m[0] = a0;
    m[1] = a1;
    m[2] = a2;
    m[3] = a3;
    return m;
}

static void print_int_array(const std::vector<int>& a)
{
    fprintf(stderr, "[");
    for (size_t i = 0; i < a.size(); i++)
    {
        fprintf(stderr, " %d", a[i]);
    }
    fprintf(stderr, " ]");
}

static int test_slice(const ncnn::Mat& a, const std::vector<int>& slices_array, int axis)
{
    ncnn::Mat slices(slices_array.size());
    {
        int* p = slices;
        for (size_t i = 0; i < slices_array.size(); i++)
        {
            p[i] = slices_array[i];
        }
    }

    ncnn::ParamDict pd;
    pd.set(0, slices);
    pd.set(1, axis);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    int ret = test_layer("Slice", pd, weights, a0, slices.w);
    if (ret != 0)
    {
        fprintf(stderr, "test_slice failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " slices=");
        print_int_array(slices_array);
        fprintf(stderr, " axis=%d\n", axis);
    }

    return ret;
}

static int test_slice_indices(const ncnn::Mat& a, const std::vector<int>& indices_array, int axis)
{
    ncnn::Mat indices(indices_array.size());
    {
        int* p = indices;
        for (size_t i = 0; i < indices_array.size(); i++)
        {
            p[i] = indices_array[i];
        }
    }

    ncnn::ParamDict pd;
    pd.set(1, axis);
    pd.set(2, indices);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    int ret = test_layer("Slice", pd, weights, a0, indices.w);
    if (ret != 0)
    {
        fprintf(stderr, "test_slice_indices failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " indices=");
        print_int_array(indices_array);
        fprintf(stderr, " axis=%d\n", axis);
    }

    return ret;
}

static int test_slice_0()
{
    ncnn::Mat a[] = {
        RandomMat(30, 32, 36, 48),
        RandomMat(36, 30, 32, 51),
        RandomMat(30, 32, 36, 60)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        int ret = 0
                  || test_slice(a[i], IntArray(-233, -233, -233), 0)
                  || test_slice(a[i], IntArray(-233, -233, -233), 1)
                  || test_slice(a[i], IntArray(-233, -233, -233), -2)
                  || test_slice(a[i], IntArray(-233, -233, -233), 3)
                  || test_slice(a[i], IntArray(3, 12, 16, -233), 0)
                  || test_slice(a[i], IntArray(12, 16, -233), 0)
                  || test_slice(a[i], IntArray(32, 8, -233), 0)
                  || test_slice(a[i], IntArray(2, 12, 16, -233), 1)
                  || test_slice(a[i], IntArray(16, 4, 5, -233), -2)
                  || test_slice(a[i], IntArray(8, 2, 16, -233), 3)
                  || test_slice_indices(a[i], IntArray(2, -24, -8), 0)
                  || test_slice_indices(a[i], IntArray(4, 20, 4), 1)
                  || test_slice_indices(a[i], IntArray(16, -16), -2)
                  || test_slice_indices(a[i], IntArray(1, -12), 3);

        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_slice_1()
{
    ncnn::Mat a[] = {
        RandomMat(51, 36, 48),
        RandomMat(48, 48, 51),
        RandomMat(51, 36, 60)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        int ret = 0
                  || test_slice(a[i], IntArray(-233, -233, -233), 0)
                  || test_slice(a[i], IntArray(-233, -233, -233), 1)
                  || test_slice(a[i], IntArray(-233, -233, -233), -1)
                  || test_slice(a[i], IntArray(3, 12, 16, -233), 0)
                  || test_slice(a[i], IntArray(12, 16, -233), 0)
                  || test_slice(a[i], IntArray(32, 8, -233), 0)
                  || test_slice(a[i], IntArray(2, 12, 16, -233), 1)
                  || test_slice(a[i], IntArray(16, 4, 5, -233), -1)
                  || test_slice_indices(a[i], IntArray(2, -24, -8), 0)
                  || test_slice_indices(a[i], IntArray(4, 20, 4), 1)
                  || test_slice_indices(a[i], IntArray(1, -12), 2);

        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_slice_2()
{
    ncnn::Mat a[] = {
        RandomMat(36, 48),
        RandomMat(48, 51),
        RandomMat(36, 60)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        int ret = 0
                  || test_slice(a[i], IntArray(-233, -233, -233), 0)
                  || test_slice(a[i], IntArray(-233, -233, -233), -1)
                  || test_slice(a[i], IntArray(3, 12, 16, -233), 0)
                  || test_slice(a[i], IntArray(12, 16, -233), 0)
                  || test_slice(a[i], IntArray(32, 8, -233), -2)
                  || test_slice(a[i], IntArray(2, 12, 16, -233), -1)
                  || test_slice_indices(a[i], IntArray(2, -24, -8), 0)
                  || test_slice_indices(a[i], IntArray(1, -12), 1);

        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_slice_3()
{
    ncnn::Mat a[] = {
        RandomMat(48),
        RandomMat(51),
        RandomMat(60)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        int ret = 0
                  || test_slice(a[i], IntArray(-233, -233, -233), 0)
                  || test_slice(a[i], IntArray(3, 12, 16, -233), 0)
                  || test_slice(a[i], IntArray(12, 16, -233), 0)
                  || test_slice(a[i], IntArray(32, 8, -233), -1)
                  || test_slice_indices(a[i], IntArray(2, -24, -8), 0);

        if (ret != 0)
            return ret;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_slice_0()
           || test_slice_1()
           || test_slice_2()
           || test_slice_3();
}
