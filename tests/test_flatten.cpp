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

#include "layer/flatten.h"
#include "testutil.h"

static int test_flatten(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Flatten>("Flatten", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_flatten failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_flatten_0()
{
    return 0
           || test_flatten(RandomMat(2, 4, 4))
           || test_flatten(RandomMat(3, 5, 8))
           || test_flatten(RandomMat(1, 1, 16))
           || test_flatten(RandomMat(9, 10, 16))
           || test_flatten(RandomMat(1, 7, 1))
           || test_flatten(RandomMat(6, 6, 15))
           || test_flatten(RandomMat(13, 13))
           || test_flatten(RandomMat(16, 16))
           || test_flatten(RandomMat(8, 12))
           || test_flatten(RandomMat(8, 2))
           || test_flatten(RandomMat(32))
           || test_flatten(RandomMat(17));
}

int main()
{
    SRAND(7767517);

    return test_flatten_0();
}
