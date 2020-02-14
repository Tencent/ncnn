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

#include "layer/unaryop.h"

#define OP_TYPE_MAX 17

static int test_unaryop(const ncnn::Mat& _a, int op_type)
{
    ncnn::Mat a = _a;
    if (op_type == 5 || op_type == 6 || op_type == 8)
    {
        // value must be positive for sqrt rsqrt log
        Randomize(a, 0.001f, 2.f);
    }
    if (op_type == 11 || op_type == 12 || op_type == 13)
    {
        // smaller range for tan asin acos
        Randomize(a, -1.f, 1.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);

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

    int ret = test_layer<ncnn::UnaryOp>("UnaryOp", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_unaryop failed a.dims=%d a=(%d %d %d) op_type=%d\n", a.dims, a.w, a.h, a.c, op_type);
    }

    return ret;
}

static int test_unaryop_0()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_unaryop(RandomMat(6, 7, 16), op_type)
            || test_unaryop(RandomMat(3, 5, 13), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_unaryop_1()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_unaryop(RandomMat(6, 16), op_type)
            || test_unaryop(RandomMat(7, 15), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_unaryop_2()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_unaryop(RandomMat(128), op_type)
            || test_unaryop(RandomMat(127), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_unaryop_0()
        || test_unaryop_1()
        || test_unaryop_2()
        ;
}
