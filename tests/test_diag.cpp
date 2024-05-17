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

static int test_diag(const ncnn::Mat& a, int diagonal)
{
    ncnn::ParamDict pd;
    pd.set(0, diagonal);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Diag", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_diag failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_diag_0()
{
    return 0
           || test_diag(RandomMat(5, 24), 3)
           || test_diag(RandomMat(7, 12), 0)
           || test_diag(RandomMat(6, 6), -4)
           || test_diag(RandomMat(3, 4), -6);
}

static int test_diag_1()
{
    return 0
           || test_diag(RandomMat(5), -1)
           || test_diag(RandomMat(7), 0)
           || test_diag(RandomMat(3), 2);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_diag_0()
           || test_diag_1();
}
