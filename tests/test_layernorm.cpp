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

#include "layer/layernorm.h"
#include "testutil.h"

static int test_layernorm(const ncnn::Mat& a, float eps, int affine)
{
    int affine_size = a.dims == 2 ? a.w : a.w * a.h;

    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, eps);
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(affine_size);
    weights[1] = RandomMat(affine_size);

    int ret = test_layer<ncnn::LayerNorm>("LayerNorm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_layernorm failed a.dims=%d a=(%d %d %d) eps=%f affine=%d\n", a.dims, a.w, a.h, a.c, eps, affine);
    }

    return ret;
}

static int test_layernorm_0()
{
    return 0
           || test_layernorm(RandomMat(6, 4, 2), 0.01f, 0)
           || test_layernorm(RandomMat(4, 5, 6), 0.01f, 0)
           || test_layernorm(RandomMat(3, 3, 8), 0.002f, 0)
           || test_layernorm(RandomMat(5, 6, 12), 0.02f, 0)
           || test_layernorm(RandomMat(6, 7, 24), 0.001f, 0)
           || test_layernorm(RandomMat(6, 4, 2), 0.01f, 1)
           || test_layernorm(RandomMat(4, 5, 6), 0.01f, 1)
           || test_layernorm(RandomMat(3, 3, 8), 0.002f, 1)
           || test_layernorm(RandomMat(5, 6, 12), 0.02f, 1)
           || test_layernorm(RandomMat(6, 7, 24), 0.001f, 1);
}

static int test_layernorm_1()
{
    return 0
           || test_layernorm(RandomMat(4, 2), 0.01f, 0)
           || test_layernorm(RandomMat(5, 6), 0.01f, 0)
           || test_layernorm(RandomMat(3, 8), 0.002f, 0)
           || test_layernorm(RandomMat(6, 12), 0.02f, 0)
           || test_layernorm(RandomMat(7, 24), 0.001f, 0)
           || test_layernorm(RandomMat(4, 2), 0.01f, 1)
           || test_layernorm(RandomMat(5, 6), 0.01f, 1)
           || test_layernorm(RandomMat(3, 8), 0.002f, 1)
           || test_layernorm(RandomMat(6, 12), 0.02f, 1)
           || test_layernorm(RandomMat(7, 24), 0.001f, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_layernorm_0()
           || test_layernorm_1();
}
