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

static int test_copyto(const ncnn::Mat& self, const ncnn::Mat& src, int woffset, int hoffset, int doffset, int coffset)
{
    ncnn::ParamDict pd;
    pd.set(0, woffset);  // woffset
    pd.set(1, hoffset);  // hoffset
    pd.set(13, doffset); // doffset
    pd.set(2, coffset);  // coffset

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = self;
    as[1] = src;

    int ret = test_layer("CopyTo", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_copyto failed self.dims=%d self=(%d %d %d %d) src.dims=%d src=(%d %d %d %d) woffset=%d hoffset=%d doffset=%d coffset=%d\n", self.dims, self.w, self.h, self.d, self.c, src.dims, src.w, src.h, src.d, src.c, woffset, hoffset, doffset, coffset);
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
                      || test_copyto(self, src, 0, 0, 0, 0)
                      || test_copyto(self, src, 13, 0, 0, 0)
                      || test_copyto(self, src, 28, 0, 0, 0)
                      || test_copyto(self, src, 32, 0, 0, 0);

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
                      || test_copyto(self, src, 0, 0, 0, 0)
                      || test_copyto(self, src, 1, 13, 0, 0)
                      || test_copyto(self, src, 3, 28, 0, 0)
                      || test_copyto(self, src, 10, 32, 0, 0);

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
                      || test_copyto(self, src, 0, 0, 0, 0)
                      || test_copyto(self, src, 0, 1, 0, 13)
                      || test_copyto(self, src, 4, 3, 0, 28)
                      || test_copyto(self, src, 5, 0, 0, 32);

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
                      || test_copyto(self, src, 0, 0, 0, 0)
                      || test_copyto(self, src, 0, 1, 1, 13)
                      || test_copyto(self, src, 4, 3, 0, 28)
                      || test_copyto(self, src, 5, 0, 2, 32);

            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

static int test_copyto_4()
{
    return 0
           || test_copyto(RandomMat(15), RandomMat(15), 0, 0, 0, 0)
           || test_copyto(RandomMat(11, 15), RandomMat(11, 15), 0, 0, 0, 0)
           || test_copyto(RandomMat(4, 5, 16), RandomMat(4, 5, 16), 0, 0, 0, 0)
           || test_copyto(RandomMat(3, 4, 5, 16), RandomMat(3, 4, 5, 16), 0, 0, 0, 0);
}

int main()
{
    SRAND(776757);

    return 0
           || test_copyto_0()
           || test_copyto_1()
           || test_copyto_2()
           || test_copyto_3()
           || test_copyto_4();
}
