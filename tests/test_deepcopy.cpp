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

#include "layer/deepcopy.h"
#include "testutil.h"

static int test_deepcopy(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::DeepCopy>("DeepCopy", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_deepcopy failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_deepcopy_0()
{
    return 0
           || test_deepcopy(RandomMat(5, 7, 24))
           || test_deepcopy(RandomMat(7, 9, 12))
           || test_deepcopy(RandomMat(3, 5, 13));
}

static int test_deepcopy_1()
{
    return 0
           || test_deepcopy(RandomMat(15, 24))
           || test_deepcopy(RandomMat(17, 12))
           || test_deepcopy(RandomMat(19, 15));
}

static int test_deepcopy_2()
{
    return 0
           || test_deepcopy(RandomMat(128))
           || test_deepcopy(RandomMat(124))
           || test_deepcopy(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_deepcopy_0()
           || test_deepcopy_1()
           || test_deepcopy_2();
}
