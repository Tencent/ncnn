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

#include "layer/batchnorm.h"
#include "testutil.h"

static int test_batchnorm(const ncnn::Mat& a, float eps)
{
    int channels;
    if (a.dims == 1) channels = a.w;
    if (a.dims == 2) channels = a.h;
    if (a.dims == 3) channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, channels); // channels
    pd.set(1, eps);      // eps

    std::vector<ncnn::Mat> weights(4);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);
    weights[2] = RandomMat(channels);
    weights[3] = RandomMat(channels);

    // var must be positive
    Randomize(weights[2], 0.001f, 2.f);

    int ret = test_layer<ncnn::BatchNorm>("BatchNorm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_batchnorm failed a.dims=%d a=(%d %d %d) eps=%f\n", a.dims, a.w, a.h, a.c, eps);
    }

    return ret;
}

static int test_batchnorm_0()
{
    return 0
           || test_batchnorm(RandomMat(5, 7, 24), 0.f)
           || test_batchnorm(RandomMat(5, 7, 24), 0.01f)
           || test_batchnorm(RandomMat(7, 9, 12), 0.f)
           || test_batchnorm(RandomMat(7, 9, 12), 0.001f)
           || test_batchnorm(RandomMat(3, 5, 13), 0.f)
           || test_batchnorm(RandomMat(3, 5, 13), 0.001f);
}

static int test_batchnorm_1()
{
    return 0
           || test_batchnorm(RandomMat(15, 24), 0.f)
           || test_batchnorm(RandomMat(15, 24), 0.01f)
           || test_batchnorm(RandomMat(17, 12), 0.f)
           || test_batchnorm(RandomMat(17, 12), 0.001f)
           || test_batchnorm(RandomMat(19, 15), 0.f)
           || test_batchnorm(RandomMat(19, 15), 0.001f);
}

static int test_batchnorm_2()
{
    return 0
           || test_batchnorm(RandomMat(128), 0.f)
           || test_batchnorm(RandomMat(128), 0.001f)
           || test_batchnorm(RandomMat(124), 0.f)
           || test_batchnorm(RandomMat(124), 0.1f)
           || test_batchnorm(RandomMat(127), 0.f)
           || test_batchnorm(RandomMat(127), 0.1f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_batchnorm_0()
           || test_batchnorm_1()
           || test_batchnorm_2();
}
