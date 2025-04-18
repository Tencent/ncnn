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

static int test_gridsample(const ncnn::Mat& a, const ncnn::Mat& grid, int sample_type, int padding_mode, int align_corner, int permute_fusion)
{
    ncnn::ParamDict pd;
    pd.set(0, sample_type);
    pd.set(1, padding_mode);
    pd.set(2, align_corner);
    pd.set(3, permute_fusion);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = grid;

    int ret = test_layer("GridSample", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_gridsample failed a.dims=%d a=(%d %d %d %d) grid.dims=%d grid=(%d %d %d %d) sample_type=%d padding_mode=%d align_corner=%d permute_fusion=%d",
                a.dims, a.w, a.h, a.d, a.c, grid.dims, grid.w, grid.h, grid.d, grid.c,
                sample_type, padding_mode, align_corner, permute_fusion);
    }

    return ret;
}

static int test_gridsample_0()
{
    return 0
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 1, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 1, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 2, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 2, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 3, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 3, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 1, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 1, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 2, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 2, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 3, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 3, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 1, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 1, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 2, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 2, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 3, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 3, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 1, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 1, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 2, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 2, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 3, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 3, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 1, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 1, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 2, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 2, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 3, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 3, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 1, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 1, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 2, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 2, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 3, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 3, 1, 1);
}

static int test_gridsample_1()
{
    return 0
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 1, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 1, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 2, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 2, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 3, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 3, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 1, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 1, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 2, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 2, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 3, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 3, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 1, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 1, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 2, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 2, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 3, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 3, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 1, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 1, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 2, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 2, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 3, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 3, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 1, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 1, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 2, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 2, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 3, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 3, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 1, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 1, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 2, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 2, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 3, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 3, 1, 1);
}

static int test_gridsample_2()
{
    return 0
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 1, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 1, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 2, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 2, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 3, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 3, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 1, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 1, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 2, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 2, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 3, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 3, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 1, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 1, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 2, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 2, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 3, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 3, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 1, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 1, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 2, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 2, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 3, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 3, 1, 1);
}

static int test_gridsample_3()
{
    return 0
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 1, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 1, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 2, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 2, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 3, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 3, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 1, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 1, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 2, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 2, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 3, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 3, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 1, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 1, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 2, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 2, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 3, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 3, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 1, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 1, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 2, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 2, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 3, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 3, 1, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gridsample_0()
           || test_gridsample_1()
           || test_gridsample_2()
           || test_gridsample_3();
}
