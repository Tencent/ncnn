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

#include "layer/reorg.h"
#include "testutil.h"

static int test_reorg(const ncnn::Mat& a, int stride)
{
    ncnn::ParamDict pd;
    pd.set(0, stride); //stride

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Reorg>("Reorg", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reorg failed a.dims=%d a=(%d %d %d) stride=%d\n", a.dims, a.w, a.h, a.c, stride);
    }

    return ret;
}

static int test_reorg_0()
{
    return 0
           || test_reorg(RandomMat(6, 7, 1), 1)
           || test_reorg(RandomMat(6, 6, 2), 2)
           || test_reorg(RandomMat(6, 8, 3), 2)
           || test_reorg(RandomMat(4, 4, 4), 4)
           || test_reorg(RandomMat(8, 8, 8), 2)
           || test_reorg(RandomMat(10, 10, 12), 2)
           || test_reorg(RandomMat(9, 9, 16), 3);
}

int main()
{
    SRAND(7767517);

    return test_reorg_0();
}
