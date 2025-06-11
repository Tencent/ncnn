// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_copyto(const ncnn::Mat& self, const ncnn::Mat& src, const std::vector<int>& starts_array, const std::vector<int>& axes_array)
{
    ncnn::Mat starts(starts_array.size());
    {
        int* p = starts;
        for (size_t i = 0; i < starts_array.size(); i++)
        {
            p[i] = starts_array[i];
        }
    }

    ncnn::Mat axes(axes_array.size());
    {
        int* p = axes;
        for (size_t i = 0; i < axes_array.size(); i++)
        {
            p[i] = axes_array[i];
        }
    }

    ncnn::ParamDict pd;
    pd.set(9, starts); // starts
    pd.set(11, axes);  // axes

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = self;
    as[1] = src;

    int ret = test_layer("CopyTo", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_copyto failed self.dims=%d self=(%d %d %d %d) src.dims=%d src=(%d %d %d %d)", self.dims, self.w, self.h, self.d, self.c, src.dims, src.w, src.h, src.d, src.c);
        fprintf(stderr, " starts=");
        print_int_array(starts_array);
        fprintf(stderr, " axes=");
        print_int_array(axes_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_copyto_0()
{
    ncnn::Mat a[] = {
        RandomMat(112),
        RandomMat(126),
        RandomMat(127)
    };
    ncnn::Mat b[] = {
        RandomMat(33),
        RandomMat(36),
        RandomMat(64)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        for (int j = 0; j < sizeof(b) / sizeof(b[0]); j++)
        {
            const ncnn::Mat& self = a[i];
            const ncnn::Mat& src = b[j];

            int ret = 0
                      || test_copyto(self, src, IntArray(0), IntArray(0))
                      || test_copyto(self, src, IntArray(13), IntArray(-1))
                      || test_copyto(self, src, IntArray(28), IntArray(0))
                      || test_copyto(self, src, IntArray(32), std::vector<int>());

            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

static int test_copyto_1()
{
    ncnn::Mat a[] = {
        RandomMat(72, 112),
        RandomMat(87, 126),
        RandomMat(64, 127)
    };
    ncnn::Mat b[] = {
        RandomMat(14, 33),
        RandomMat(24, 36),
        RandomMat(16, 64),
        RandomMat(14),
        RandomMat(24),
        RandomMat(16)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        for (int j = 0; j < sizeof(b) / sizeof(b[0]); j++)
        {
            const ncnn::Mat& self = a[i];
            const ncnn::Mat& src = b[j];

            int ret = 0
                      || test_copyto(self, src, IntArray(0, 0), IntArray(0, 1))
                      || test_copyto(self, src, IntArray(13, 1), IntArray(-2, -1))
                      || test_copyto(self, src, IntArray(28, 3), IntArray(0, 1))
                      || test_copyto(self, src, IntArray(32, 10), IntArray(0, 1));

            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

static int test_copyto_2()
{
    ncnn::Mat a[] = {
        RandomMat(32, 42, 81),
        RandomMat(33, 57, 80),
        RandomMat(36, 34, 88)
    };
    ncnn::Mat b[] = {
        RandomMat(1, 14, 23),
        RandomMat(12, 1, 28),
        RandomMat(11, 8, 32),
        RandomMat(1, 14),
        RandomMat(12, 1),
        RandomMat(11, 8),
        RandomMat(1),
        RandomMat(12),
        RandomMat(11)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        for (int j = 0; j < sizeof(b) / sizeof(b[0]); j++)
        {
            const ncnn::Mat& self = a[i];
            const ncnn::Mat& src = b[j];

            int ret = 0
                      || test_copyto(self, src, IntArray(0, 0, 0), IntArray(0, 1, 2))
                      || test_copyto(self, src, IntArray(13, 1, 0), IntArray(-3, -2, -1))
                      || test_copyto(self, src, IntArray(28, 3, 4), IntArray(0, 1, 2))
                      || test_copyto(self, src, IntArray(32, 0, 5), IntArray(0, 1, 2));

            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

static int test_copyto_3()
{
    ncnn::Mat a[] = {
        RandomMat(12, 42, 7, 81),
        RandomMat(13, 57, 5, 80),
        RandomMat(16, 34, 6, 88)
    };
    ncnn::Mat b[] = {
        RandomMat(1, 14, 2, 23),
        RandomMat(12, 1, 3, 28),
        RandomMat(11, 8, 1, 32),
        RandomMat(1, 14, 2),
        RandomMat(12, 1, 3),
        RandomMat(11, 8, 1),
        RandomMat(1, 14),
        RandomMat(12, 1),
        RandomMat(11, 8),
        RandomMat(1),
        RandomMat(12),
        RandomMat(11)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        for (int j = 0; j < sizeof(b) / sizeof(b[0]); j++)
        {
            const ncnn::Mat& self = a[i];
            const ncnn::Mat& src = b[j];

            int ret = 0
                      || test_copyto(self, src, IntArray(0, 0, 0, 0), IntArray(0, 1, 2, 3))
                      || test_copyto(self, src, IntArray(13, 1, 1, 0), IntArray(-4, -3, 2, 3))
                      || test_copyto(self, src, IntArray(28, 0, 3, 4), IntArray(0, 1, 2, 3))
                      || test_copyto(self, src, IntArray(32, 2, 0, 5), IntArray(0, 1, 2, 3));

            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

int main()
{
    SRAND(776757);

    return 0
           || test_copyto_0()
           || test_copyto_1()
           || test_copyto_2()
           || test_copyto_3();
}
