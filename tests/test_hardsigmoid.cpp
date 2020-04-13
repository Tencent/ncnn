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

#include "layer/hardsigmoid.h"

static int test_hardsigmoid(const ncnn::Mat& a, float alpha, float beta)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(0, beta);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::HardSigmoid>("HardSigmoid", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_hardsigmoid failed a.dims=%d a=(%d %d %d) alpha=%f beta=%f\n", a.dims, a.w, a.h, a.c, alpha, beta);
    }

    return ret;
}

static int test_hardsigmoid_0()
{
    return 0
        || test_hardsigmoid(RandomMat(6, 7, 16), 0.5f, 0.5f)
        || test_hardsigmoid(RandomMat(3, 5, 13), 0.5f, 0.5f)
        ;
}

static int test_hardsigmoid_1()
{
    return 0
        || test_hardsigmoid(RandomMat(6, 16), 0.5f, 0.5f)
        || test_hardsigmoid(RandomMat(7, 15), 0.5f, 0.5f)
        ;
}

static int test_hardsigmoid_2()
{
    return 0
        || test_hardsigmoid(RandomMat(128), 0.5f, 0.5f)
        || test_hardsigmoid(RandomMat(127), 0.5f, 0.5f)
        ;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_hardsigmoid_0()
        || test_hardsigmoid_1()
        || test_hardsigmoid_2()
        ;
}
