// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/gridsample.h"
#include "testutil.h"

static int test_gridsample(const ncnn::Mat& a, const ncnn::Mat& grid, int sample_type, int padding_mode, int align_corner)
{
    ncnn::ParamDict pd;
    pd.set(0, sample_type);
    pd.set(1, padding_mode);
    pd.set(2, align_corner);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = grid;

    int ret = test_layer<ncnn::GridSample>("GridSample", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_gridsample failed a.dims=%d a=(%d %d %d %d) grid.dims=%d grid=(%d %d %d %d) sample_type=%d padding_mode=%d align_corner=%d",
                a.dims, a.w, a.h, a.d, a.c, grid.dims, grid.w, grid.h, grid.d, grid.c,
                sample_type, padding_mode, align_corner);
    }

    return ret;
}

static int test_gridsample_0()
{
    return 0
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 1, 1, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 1, 1, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 1, 2, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 1, 2, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 1, 3, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 1, 3, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 2, 1, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 2, 1, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 2, 2, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 2, 2, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 2, 3, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 2, 3, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 3, 1, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 3, 1, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 3, 2, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 3, 2, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 3, 3, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 16, 12), 3, 3, 1);
}

static int test_gridsample_1()
{
    return 0
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 1, 1, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 1, 1, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 1, 2, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 1, 2, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 1, 3, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 1, 3, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 2, 1, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 2, 1, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 2, 2, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 2, 2, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 2, 3, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 2, 3, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 3, 1, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 3, 1, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 3, 2, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 3, 2, 1)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 3, 3, 0)
           || test_gridsample(RandomMat(16, 12, 3), RandomMat(2, 27, 21), 3, 3, 1);
}

static int test_gridsample_2()
{
    return 0
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 1, 1, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 1, 1, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 1, 2, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 1, 2, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 1, 3, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 1, 3, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 2, 1, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 2, 1, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 2, 2, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 2, 2, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 2, 3, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 27, 21, 10), 2, 3, 1);
}

static int test_gridsample_3()
{
    return 0
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 1, 1, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 1, 1, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 1, 2, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 1, 2, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 1, 3, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 1, 3, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 2, 1, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 2, 1, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 2, 2, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 2, 2, 1)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 2, 3, 0)
           || test_gridsample(RandomMat(16, 12, 10, 5), RandomMat(3, 16, 12, 10), 2, 3, 1);
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
