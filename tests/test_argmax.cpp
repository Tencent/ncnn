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

static int test_argmax(const ncnn::Mat& a, int dim, int keepdim)
{
    ncnn::ParamDict pd;
    pd.set(0, dim);
    pd.set(1, keepdim);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ArgMax", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_argmax failed a.dims=%d a=(%d %d %d) dim=%d keepdim=%d\n", a.dims, a.w, a.h, a.c, dim, keepdim);
    }

    return ret;
}

static int test_argmax_0()
{
    return 0
           || test_argmax(RandomMat(3, 2, 6, 7), 0, 0)
           || test_argmax(RandomMat(3, 4, 6, 8), 1, 1)
           || test_argmax(RandomMat(3, 4, 6, 5), 2, 0)
           || test_argmax(RandomMat(4, 2, 6, 5), 3, 1);
}

static int test_argmax_1()
{
    return 0
           || test_argmax(RandomMat(2, 3, 5), 0, 0)
           || test_argmax(RandomMat(4, 3, 5), 1, 1)
           || test_argmax(RandomMat(6, 3, 5), 2, 0);
}

static int test_argmax_2()
{
    return 0
           || test_argmax(RandomMat(8, 2), -2, 0)
           || test_argmax(RandomMat(16, 3), -1, 1);
}

static int test_argmax_3()
{
    return 0
           || test_argmax(RandomMat(16), -1, 1)
           || test_argmax(RandomMat(32), 0, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_argmax_0()
           || test_argmax_1()
           || test_argmax_2()
           || test_argmax_3();
}