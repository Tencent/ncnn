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

static int test_crop(const ncnn::Mat& a, int woffset, int hoffset, int doffset, int coffset, int outw, int outh, int outd, int outc, int woffset2, int hoffset2, int doffset2, int coffset2)
{
    ncnn::ParamDict pd;
    pd.set(0, woffset);   // woffset
    pd.set(1, hoffset);   // hoffset
    pd.set(13, doffset);  // doffset
    pd.set(2, coffset);   // coffset
    pd.set(3, outw);      // outw
    pd.set(4, outh);      // outh
    pd.set(14, outd);     // outd
    pd.set(5, outc);      // outc
    pd.set(6, woffset2);  // woffset2
    pd.set(7, hoffset2);  // hoffset2
    pd.set(15, doffset2); // doffset2
    pd.set(8, coffset2);  // coffset2

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Crop", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d) woffset=%d hoffset=%d doffset=%d coffset=%d outw=%d outh=%d outd=%d outc=%d woffset2=%d hoffset2=%d doffset2=%d coffset2=%d\n", a.dims, a.w, a.h, a.d, a.c, woffset, hoffset, doffset, coffset, outw, outh, outd, outc, woffset2, hoffset2, doffset2, coffset2);
    }

    return ret;
}

static int test_crop_0(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 0, -233, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 0, -233, 0, 0, 0, 12, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, -233, 0, 0, 0, 16, 0, 0, 0)
           || test_crop(a, 16, 0, 0, 0, -233, 0, 0, 0, 7, 0, 0, 0);
}

static int test_crop_3(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 5, 0, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 11, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 6, 12, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 4, 8, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, 5, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, 6, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, 4, -233, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, 12, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, 16, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, 7, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 5, 11, 0, 0, 4, 16, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 6, 12, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 4, 8, 0, 0, 6, 12, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, -233, -233, 0, 0, 5, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, 0, 0, 6, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, -233, -233, 0, 0, 4, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, -233, 0, 0, 0, 12, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, -233, 0, 0, 0, 16, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, -233, 0, 0, 0, 7, 0, 0)

           || test_crop(a, 5, 11, 0, 0, -233, -233, 0, 0, 4, 16, 0, 0)
           || test_crop(a, 6, 12, 0, 0, -233, -233, 0, 0, 5, 7, 0, 0)
           || test_crop(a, 4, 8, 0, 0, -233, -233, 0, 0, 6, 12, 0, 0);
}

static int test_crop_6(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 5, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 5, 5, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 4, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 5, 0, 0, 11, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 8, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 6, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, 6, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, 4, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, 5, -233, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 0, -233, 4, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 0, -233, 5, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, 6, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, 0, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 8, -233, -233, 0, 16, 0, 0, 0, 0)

           || test_crop(a, 5, 5, 0, 0, 4, 4, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 6, 0, 0, 6, 6, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 4, 0, 0, 5, 5, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 11, -233, 6, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 12, -233, 4, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 8, -233, 5, 0, 7, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 11, 4, -233, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 12, 5, -233, 0, 7, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 8, 4, -233, 0, 12, 0, 0, 0, 0)

           || test_crop(a, 5, 3, 0, 11, 6, 5, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 6, 4, 0, 12, 4, 4, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 4, 5, 0, 8, 5, 3, 0, 7, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, -233, -233, 0, -233, 4, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, 0, -233, 5, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, -233, -233, 0, -233, 6, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 0, -233, -233, 0, -233, 0, 5, 0, 0)
           || test_crop(a, 0, 6, 0, 0, -233, -233, 0, -233, 0, 6, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, -233, 0, -233, 0, 4, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, 0, -233, 0, 0, 0, 12)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, -233, 0, 0, 0, 16)
           || test_crop(a, 0, 0, 0, 8, -233, -233, 0, -233, 0, 0, 0, 7)

           || test_crop(a, 5, 4, 0, 0, -233, -233, 0, -233, 4, 2, 0, 0)
           || test_crop(a, 6, 3, 0, 0, -233, -233, 0, -233, 5, 3, 0, 0)
           || test_crop(a, 4, 2, 0, 0, -233, -233, 0, -233, 6, 4, 0, 0)

           || test_crop(a, 0, 5, 0, 11, -233, -233, 0, -233, 0, 5, 0, 7)
           || test_crop(a, 0, 6, 0, 12, -233, -233, 0, -233, 0, 6, 0, 12)
           || test_crop(a, 0, 4, 0, 8, -233, -233, 0, -233, 0, 4, 0, 16)

           || test_crop(a, 5, 0, 0, 11, -233, -233, 0, -233, 6, 0, 0, 12)
           || test_crop(a, 6, 0, 0, 12, -233, -233, 0, -233, 4, 0, 0, 16)
           || test_crop(a, 4, 0, 0, 8, -233, -233, 0, -233, 5, 0, 0, 7)

           || test_crop(a, 5, 2, 0, 11, -233, -233, 0, -233, 4, 3, 0, 16)
           || test_crop(a, 6, 3, 0, 12, -233, -233, 0, -233, 5, 4, 0, 7)
           || test_crop(a, 4, 4, 0, 8, -233, -233, 0, -233, 6, 2, 0, 12);
}

static int test_crop_9(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 5, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 5, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 4, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 5, 5, 5, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 4, 4, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 6, 12, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 5, 0, 5, 11, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 4, 8, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 6, 6, 12, -233, -233, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, 6, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, 5, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, 4, -233, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 0, -233, 4, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 0, -233, 5, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, 6, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 5, 0, -233, -233, 6, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 6, 0, -233, -233, 4, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 4, 0, -233, -233, 5, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, -233, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 8, -233, -233, -233, 16, 0, 0, 0, 0)

           || test_crop(a, 5, 2, 0, 0, 6, 5, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 3, 0, 0, 5, 6, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 4, 0, 0, 4, 4, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 3, 0, 2, -233, 5, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 2, 0, 3, -233, 3, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 4, 0, 4, -233, 4, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 11, -233, 1, -233, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 12, -233, 2, -233, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 8, -233, 3, -233, 7, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 4, 11, -233, -233, 3, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 5, 12, -233, -233, 2, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 6, 8, -233, -233, 1, 16, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 5, 11, 1, -233, 2, 16, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 6, 12, 2, -233, 3, 7, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 4, 8, 3, -233, 1, 16, 0, 0, 0, 0)

           || test_crop(a, 4, 6, 3, 11, 2, 3, 4, 12, 0, 0, 0, 0)
           || test_crop(a, 5, 5, 4, 12, 3, 4, 5, 16, 0, 0, 0, 0)
           || test_crop(a, 6, 4, 2, 8, 4, 5, 6, 7, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, -233, -233, -233, -233, 2, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, -233, -233, 3, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, -233, -233, -233, -233, 4, 0, 0, 0)

           || test_crop(a, 0, 4, 0, 0, -233, -233, -233, -233, 0, 4, 0, 0)
           || test_crop(a, 0, 3, 0, 0, -233, -233, -233, -233, 0, 3, 0, 0)
           || test_crop(a, 0, 2, 0, 0, -233, -233, -233, -233, 0, 2, 0, 0)

           || test_crop(a, 0, 0, 4, 0, -233, -233, -233, -233, 0, 0, 4, 0)
           || test_crop(a, 0, 0, 5, 0, -233, -233, -233, -233, 0, 0, 2, 0)
           || test_crop(a, 0, 0, 6, 0, -233, -233, -233, -233, 0, 0, 3, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, -233, -233, 0, 0, 0, 12)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, -233, 0, 0, 0, 16)
           || test_crop(a, 0, 0, 0, 8, -233, -233, -233, -233, 0, 0, 0, 7)

           || test_crop(a, 5, 3, 0, 0, -233, -233, -233, -233, 5, 2, 0, 0)
           || test_crop(a, 6, 4, 0, 0, -233, -233, -233, -233, 3, 3, 0, 0)
           || test_crop(a, 4, 4, 0, 0, -233, -233, -233, -233, 2, 5, 0, 0)

           || test_crop(a, 0, 4, 0, 11, -233, -233, -233, -233, 0, 3, 0, 7)
           || test_crop(a, 0, 3, 0, 12, -233, -233, -233, -233, 0, 4, 0, 12)
           || test_crop(a, 0, 2, 0, 8, -233, -233, -233, -233, 0, 5, 0, 16)

           || test_crop(a, 0, 4, 4, 0, -233, -233, -233, -233, 0, 4, 1, 0)
           || test_crop(a, 0, 5, 5, 0, -233, -233, -233, -233, 0, 2, 2, 0)
           || test_crop(a, 0, 2, 6, 0, -233, -233, -233, -233, 0, 1, 3, 0)

           || test_crop(a, 3, 0, 0, 11, -233, -233, -233, -233, 3, 0, 0, 12)
           || test_crop(a, 4, 0, 0, 12, -233, -233, -233, -233, 4, 0, 0, 16)
           || test_crop(a, 5, 0, 0, 8, -233, -233, -233, -233, 2, 0, 0, 7)

           || test_crop(a, 0, 4, 4, 11, -233, -233, -233, -233, 0, 4, 4, 12)
           || test_crop(a, 0, 5, 5, 12, -233, -233, -233, -233, 0, 4, 4, 16)
           || test_crop(a, 0, 6, 6, 8, -233, -233, -233, -233, 0, 3, 3, 7)

           || test_crop(a, 1, 1, 1, 11, -233, -233, -233, -233, 1, 1, 1, 16)
           || test_crop(a, 2, 2, 2, 12, -233, -233, -233, -233, 2, 2, 2, 7)
           || test_crop(a, 3, 3, 3, 8, -233, -233, -233, -233, 3, 3, 3, 12);
}

int main()
{
    SRAND(776757);

    return 0
           || test_crop_0(RandomMat(112))
           || test_crop_0(RandomMat(126))
           || test_crop_0(RandomMat(127))
           || test_crop_3(RandomMat(20, 48))
           || test_crop_3(RandomMat(15, 36))
           || test_crop_3(RandomMat(16, 33))
           || test_crop_6(RandomMat(20, 20, 48))
           || test_crop_6(RandomMat(15, 15, 36))
           || test_crop_6(RandomMat(16, 16, 33))
           || test_crop_9(RandomMat(20, 20, 20, 48))
           || test_crop_9(RandomMat(15, 15, 15, 36))
           || test_crop_9(RandomMat(16, 16, 16, 33));
}
