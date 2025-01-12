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

#include "layer.h"
#include "testutil.h"

static int test_flip(const ncnn::Mat& a, std::vector<int> axis)
{
    ncnn::Mat axis_mat(axis.size());
    for (size_t i = 0; i < axis.size(); i++)
    {
        axis_mat[i] = axis[i];
    }
    ncnn::ParamDict pd;
    pd.set(0, axis_mat); // axis

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Flip", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_flip failed a.dims=%d a=(%d %d %d) axis=", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_flip_0()
{
    return 0
           || test_flip(RandomMat(3, 2, 6, 7), {0})
           || test_flip(RandomMat(3, 2, 6, 7), {0, 1})
           || test_flip(RandomMat(3, 2, 6, 7), {0, 2})
           || test_flip(RandomMat(3, 2, 6, 7), {0, 3});
}

static int test_flip_1()
{
    return 0
           || test_flip(RandomMat(2, 3, 5), {0})
           || test_flip(RandomMat(4, 2, 5), {0, 1})
           || test_flip(RandomMat(3, 4, 2), {0, 1, 2});
}

static int test_flip_2()
{
    return 0
           || test_flip(RandomMat(8, 2), {-2})
           || test_flip(RandomMat(16, 3), {-2, -1});
}

static int test_flip_3()
{
    return 0
           || test_flip(RandomMat(16), {-1})
           || test_flip(RandomMat(32), {0});
}

int main()
{
    SRAND(7767517);

    return 0
           || test_flip_0()
           || test_flip_1()
           || test_flip_2()
           || test_flip_3();
}