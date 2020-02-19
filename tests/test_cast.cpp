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

#include "testutil.h"

#include "layer/cast.h"

static int test_cast(const ncnn::Mat& a, int type_from, int type_to)
{
    ncnn::ParamDict pd;
    pd.set(0, type_from);
    pd.set(1, type_to);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    int ret = test_layer<ncnn::Cast>("Cast", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_cast failed a.dims=%d a=(%d %d %d) type_from=%d type_to=%d\n", a.dims, a.w, a.h, a.c, type_from, type_to);
    }

    return ret;
}

static int test_cast_0()
{
    return 0
        || test_cast(RandomMat(6, 7, 16), 1, 2)
        || test_cast(RandomMat(3, 5, 13), 1, 2)
        ;
}

static int test_cast_1()
{
    return 0
        || test_cast(RandomMat(6, 16), 1, 2)
        || test_cast(RandomMat(7, 15), 1, 2)
        ;
}

static int test_cast_2()
{
    return 0
        || test_cast(RandomMat(128), 1, 2)
        || test_cast(RandomMat(127), 1, 2)
        ;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_cast_0()
        || test_cast_1()
        || test_cast_2()
        ;
}
