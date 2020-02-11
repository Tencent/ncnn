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

#include "layer/sigmoid.h"

static int test_sigmoid(const ncnn::Mat& a, bool use_packing_layout)
{
    ncnn::ParamDict pd;

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
    opt.use_packing_layout = use_packing_layout;

    int ret = test_layer<ncnn::Sigmoid>("Sigmoid", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_sigmoid failed a.dims=%d a=(%d %d %d) use_packing_layout=%d\n", a.dims, a.w, a.h, a.c, use_packing_layout);
    }

    return ret;
}

static int test_sigmoid_0()
{
    return 0
        || test_sigmoid(RandomMat(6, 7, 16), false)
        || test_sigmoid(RandomMat(6, 7, 16), false)
        || test_sigmoid(RandomMat(3, 5, 13), false)
        || test_sigmoid(RandomMat(3, 5, 13), false)

        || test_sigmoid(RandomMat(6, 7, 16), true)
        || test_sigmoid(RandomMat(6, 7, 16), true)
        || test_sigmoid(RandomMat(3, 5, 13), true)
        || test_sigmoid(RandomMat(3, 5, 13), true)
        ;
}

static int test_sigmoid_1()
{
    return 0
        || test_sigmoid(RandomMat(6, 16), false)
        || test_sigmoid(RandomMat(6, 16), false)
        || test_sigmoid(RandomMat(7, 15), false)
        || test_sigmoid(RandomMat(7, 15), false)

        || test_sigmoid(RandomMat(6, 16), true)
        || test_sigmoid(RandomMat(6, 16), true)
        || test_sigmoid(RandomMat(7, 15), true)
        || test_sigmoid(RandomMat(7, 15), true)
        ;
}

static int test_sigmoid_2()
{
    return 0
        || test_sigmoid(RandomMat(128), false)
        || test_sigmoid(RandomMat(128), false)
        || test_sigmoid(RandomMat(127), false)
        || test_sigmoid(RandomMat(127), false)

        || test_sigmoid(RandomMat(128), true)
        || test_sigmoid(RandomMat(128), true)
        || test_sigmoid(RandomMat(127), true)
        || test_sigmoid(RandomMat(127), true)
        ;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_sigmoid_0()
        || test_sigmoid_1()
        || test_sigmoid_2()
        ;
}
