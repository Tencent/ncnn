// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/pooling1d.h"
#include "testutil.h"

static int test_pooling1d(int w, int h, int pooling_type, int kernel, int stride, int pad, int global_pooling, int pad_mode, int avgpool_count_include_pad, int adaptive_pooling, int out_w)
{
    ncnn::Mat a = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(0, pooling_type);              // pooling_type
    pd.set(1, kernel);                    // kernel_w
    pd.set(2, stride);                    // stride_w
    pd.set(3, pad);                       // pad_w
    pd.set(4, global_pooling);            // global_pooling
    pd.set(5, pad_mode);                  // pad_mode
    pd.set(6, avgpool_count_include_pad); // avgpool_count_include_pad
    pd.set(7, adaptive_pooling);          // adaptive_pooling
    pd.set(8, out_w);                     // out_w

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Pooling1D>("Pooling1D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_pooling1d failed w=%d h=%d pooling_type=%d kernel=%d stride=%d pad=%d global_pooling=%d pad_mode=%d avgpool_count_include_pad=%d adaptive_pooling=%d out_w=%d\n", w, h, pooling_type, kernel, stride, pad, global_pooling, pad_mode, avgpool_count_include_pad, adaptive_pooling, out_w);
    }

    return ret;
}

static int test_pooling1d_0()
{
    static const int ksp[11][3] = {
        {2, 1, 0},
        {2, 2, 0},
        {3, 1, 0},
        {3, 2, 1},
        {4, 1, 0},
        {4, 2, 1},
        {5, 1, 0},
        {5, 2, 2},
        {7, 1, 0},
        {7, 2, 1},
        {7, 3, 2},
    };

    for (int i = 0; i < 11; i++)
    {
        int ret = 0
                  || test_pooling1d(9, 1, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 2, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 3, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 0, 0, 0)
                  || test_pooling1d(9, 4, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 0, 0, 0)
                  || test_pooling1d(9, 7, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 8, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 15, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 0, 0, 0)
                  || test_pooling1d(9, 16, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 0, 0, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_pooling1d_1()
{
    static const int ksp[11][3] = {
        {2, 1, 0},
        {2, 2, 0},
        {3, 1, 0},
        {3, 2, 1},
        {4, 1, 0},
        {4, 2, 1},
        {5, 1, 0},
        {5, 2, 2},
        {7, 1, 0},
        {7, 2, 1},
        {7, 3, 2},
    };

    for (int i = 0; i < 11; i++)
    {
        int ret = 0
                  || test_pooling1d(9, 1, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 2, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 3, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 1, 0, 0)
                  || test_pooling1d(9, 4, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 7, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 8, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 1, 0, 0)
                  || test_pooling1d(9, 12, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 1, 0, 0)
                  || test_pooling1d(9, 15, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 16, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 64, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 1, 0, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_pooling1d_2()
{
    return 0
           || test_pooling1d(2, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(6, 3, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(7, 8, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(48, 4, 0, 2, 2, 0, 0, 0, 0, 0, 0)
           || test_pooling1d(48, 15, 0, 2, 2, 1, 0, 0, 0, 0, 0);
}

// adaptive avg pool
static int test_pooling1d_3()
{
    return 0
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 1)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 3)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 5)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 7)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 9)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 11)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 13)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 2)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 4)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 6)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 8)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 10)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 12);
}

// adaptive max pool
static int test_pooling1d_4()
{
    return 0
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 1)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 3)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 5)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 7)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 9)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 11)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 13)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 2)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 4)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 6)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 8)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 10)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 12);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_pooling1d_0()
           || test_pooling1d_1()
           || test_pooling1d_2()
           || test_pooling1d_3()
           || test_pooling1d_4();
}
