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

static int test_flatten_int8(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING | TEST_LAYER_DISABLE_GPU_TESTING;
    int ret = test_layer<ncnn::Flatten>("Flatten", pd, weights, a, 0.001, 0, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_flatten_int8 failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_flatten_1()
{
    return 0
           || test_flatten_int8(RandomS8Mat(2, 4, 16))
           || test_flatten_int8(RandomS8Mat(3, 5, 32))
           || test_flatten_int8(RandomS8Mat(1, 1, 64))
           || test_flatten_int8(RandomS8Mat(9, 10, 64))
           || test_flatten_int8(RandomS8Mat(1, 7, 4))
           || test_flatten_int8(RandomS8Mat(6, 6, 70))
           || test_flatten_int8(RandomS8Mat(13, 52))
           || test_flatten_int8(RandomS8Mat(16, 64))
           || test_flatten_int8(RandomS8Mat(8, 48))
           || test_flatten_int8(RandomS8Mat(8, 8))
           || test_flatten_int8(RandomS8Mat(128))
           || test_flatten_int8(RandomS8Mat(127));
}

int main()
{
    SRAND(7767517);

    return test_flatten_0() || test_flatten_1();
}
