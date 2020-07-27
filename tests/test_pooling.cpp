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

#include "layer/pooling.h"
#include "testutil.h"

static int test_pooling(int w, int h, int c, int pooling_type, int kernel, int stride, int pad, int global_pooling, int pad_mode, int avgpool_count_include_pad)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, pooling_type);              // pooling_type
    pd.set(1, kernel);                    // kernel_w
    pd.set(2, stride);                    // stride_w
    pd.set(3, pad);                       // pad_w
    pd.set(4, global_pooling);            // global_pooling
    pd.set(5, pad_mode);                  // pad_mode
    pd.set(6, avgpool_count_include_pad); // avgpool_count_include_pad

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Pooling>("Pooling", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_pooling failed w=%d h=%d c=%d pooling_type=%d kernel=%d stride=%d pad=%d global_pooling=%d pad_mode=%d avgpool_count_include_pad=%d\n", w, h, c, pooling_type, kernel, stride, pad, global_pooling, pad_mode, avgpool_count_include_pad);
    }

    return ret;
}

static int test_pooling_0()
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
                  || test_pooling(9, 7, 1, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0)
                  || test_pooling(9, 7, 2, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0)
                  || test_pooling(9, 7, 3, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 0)
                  || test_pooling(9, 7, 4, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 0)
                  || test_pooling(9, 7, 7, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0)
                  || test_pooling(9, 7, 8, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0)
                  || test_pooling(9, 7, 15, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 0)
                  || test_pooling(9, 7, 16, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_pooling_1()
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
                  || test_pooling(9, 7, 1, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0)
                  || test_pooling(9, 7, 2, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0)
                  || test_pooling(9, 7, 3, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 1)
                  || test_pooling(9, 7, 4, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0)
                  || test_pooling(9, 7, 7, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0)
                  || test_pooling(9, 7, 8, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 1)
                  || test_pooling(9, 7, 15, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0)
                  || test_pooling(9, 7, 16, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_pooling_2()
{
    return 0
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 1, 0, 0)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 1, 0, 0)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 1, 0, 0)
           || test_pooling(6, 3, 3, 1, 1, 1, 0, 1, 0, 0)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 1, 0, 0)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 1, 0, 0)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 1, 0, 0)
           || test_pooling(7, 8, 8, 1, 1, 1, 0, 1, 0, 0)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 1, 0, 0)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 1, 0, 0)
           || test_pooling(48, 48, 4, 0, 2, 2, 0, 0, 0, 0)
           || test_pooling(48, 48, 15, 0, 2, 2, 1, 0, 0, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_pooling_0()
           || test_pooling_1()
           || test_pooling_2();
}
