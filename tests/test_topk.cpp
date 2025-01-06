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

static int test_topk(const ncnn::Mat& a, int k, int axis, int largest, int sorted)
{
    ncnn::ParamDict pd;
    pd.set(0, k);       // k
    pd.set(1, axis);    // axis
    pd.set(2, largest); // largest
    pd.set(3, sorted);  // sorted

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    int ret = test_layer("TopK", pd, weights, a0, 2);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk failed a.dims=%d a=(%d %d %d) k=%d axis=%d largest=%d sorted=%d\n", a.dims, a.w, a.h, a.c, k, axis, largest, sorted);
    }

    return ret;
}

static int test_topk_0()
{
    return 0
           //    || test_topk(RandomMat(3, 2, 6, 7), 1, 0, 1, 1) // axis=0暂未实现
           //    || test_topk(RandomMat(3, 4, 2, 5), 2, 1, 0, 1) // axis=1暂未实现
           || test_topk(RandomMat(3, 6, 4, 2), 2, 2, 1, 0)
           || test_topk(RandomMat(5, 3, 5, 3), 1, 3, 1, 1);
}

static int test_topk_1()
{
    return 0
           || test_topk(RandomMat(2, 3, 5), 1, 0, 1, 1)
           || test_topk(RandomMat(4, 2, 5), 1, 1, 0, 1)
           || test_topk(RandomMat(3, 4, 2), 3, 2, 1, 0);
}

static int test_topk_2()
{
    return 0
           || test_topk(RandomMat(8, 2), 2, 0, 1, 1)
           || test_topk(RandomMat(16, 3), 5, 1, 0, 1);
}

static int test_topk_3()
{
    return 0
           || test_topk(RandomMat(16), 5, 0, 1, 1)
           || test_topk(RandomMat(32), 10, 0, 0, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_topk_0()
           || test_topk_1()
           || test_topk_2()
           || test_topk_3();
}