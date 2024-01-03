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

static int test_crop(const ncnn::Mat& a, int woffset, int hoffset, int doffset, int coffset, const ncnn::Mat& ref)
{
    ncnn::ParamDict pd;
    pd.set(0, woffset);
    pd.set(1, hoffset);
    pd.set(13, doffset);
    pd.set(2, coffset);
    pd.set(3, 0);  // outw
    pd.set(4, 0);  // outh
    pd.set(14, 0); // outd
    pd.set(5, 0);  // outc
    pd.set(6, 0);  // woffset2
    pd.set(7, 0);  // hoffset2
    pd.set(15, 0); // doffset2
    pd.set(8, 0);  // coffset2

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = ref;

    int ret = test_layer("Crop", pd, weights, ab);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d) woffset=%d hoffset=%d doffset=%d coffset=%d ref.dims=%d ref=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c, woffset, hoffset, doffset, coffset, ref.dims, ref.w, ref.h, ref.d, ref.c);
    }

    return ret;
}

static int test_crop_2(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, a)
           || test_crop(a, 0, 0, 0, 0, ncnn::Mat(27))

           || test_crop(a, 11, 0, 0, 0, ncnn::Mat(7))
           || test_crop(a, 12, 0, 0, 0, ncnn::Mat(12))
           || test_crop(a, 16, 0, 0, 0, ncnn::Mat(16));
}

static int test_crop_5(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, a)

           || test_crop(a, 0, 12, 0, 0, ncnn::Mat(8, 7))
           || test_crop(a, 5, 0, 0, 0, ncnn::Mat(7, 27))

           || test_crop(a, 5, 11, 0, 0, ncnn::Mat(5, 12))
           || test_crop(a, 6, 12, 0, 0, ncnn::Mat(4, 16))
           || test_crop(a, 4, 8, 0, 0, ncnn::Mat(6, 7));
}

static int test_crop_8(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, a)

           || test_crop(a, 0, 5, 0, 0, ncnn::Mat(6, 6))
           || test_crop(a, 6, 0, 0, 0, ncnn::Mat(8, 8))
           || test_crop(a, 5, 2, 0, 0, ncnn::Mat(6, 3))
           || test_crop(a, 6, 3, 0, 0, ncnn::Mat(8, 4))
           || test_crop(a, 4, 4, 0, 0, ncnn::Mat(7, 5))

           || test_crop(a, 5, 3, 0, 11, ncnn::Mat(7, 3, 7))
           || test_crop(a, 6, 4, 0, 12, ncnn::Mat(6, 4, 12))
           || test_crop(a, 4, 2, 0, 8, ncnn::Mat(5, 5, 16));
}

static int test_crop_11(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, a)

           || test_crop(a, 0, 5, 0, 0, ncnn::Mat(6, 6, 6))
           || test_crop(a, 6, 0, 0, 0, ncnn::Mat(8, 8, 8))
           || test_crop(a, 5, 5, 5, 0, ncnn::Mat(6, 6, 6))
           || test_crop(a, 6, 6, 6, 0, ncnn::Mat(8, 8, 8))
           || test_crop(a, 4, 4, 4, 0, ncnn::Mat(5, 5, 5))

           || test_crop(a, 3, 3, 3, 11, ncnn::Mat(3, 3, 3, 7))
           || test_crop(a, 4, 4, 4, 12, ncnn::Mat(6, 6, 6, 12))
           || test_crop(a, 5, 5, 5, 8, ncnn::Mat(8, 8, 8, 16));
}

int main()
{
    SRAND(776757);

    return 0
           || test_crop_2(RandomMat(112))
           || test_crop_2(RandomMat(126))
           || test_crop_2(RandomMat(127))
           || test_crop_5(RandomMat(20, 48))
           || test_crop_5(RandomMat(15, 36))
           || test_crop_5(RandomMat(16, 33))
           || test_crop_8(RandomMat(20, 20, 48))
           || test_crop_8(RandomMat(15, 15, 36))
           || test_crop_8(RandomMat(16, 16, 33))
           || test_crop_11(RandomMat(20, 20, 20, 48))
           || test_crop_11(RandomMat(15, 15, 15, 36))
           || test_crop_11(RandomMat(16, 16, 16, 33));
}
