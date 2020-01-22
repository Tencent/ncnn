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

#include "layer/batchnorm.h"

static int test_batchnorm(const ncnn::Mat& a, int channels, float eps, bool use_packing_layout)
{
    ncnn::ParamDict pd;
    pd.set(0, channels);// channels
    pd.set(1, eps);// eps

    std::vector<ncnn::Mat> weights(4);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);
    weights[2] = RandomMat(channels);
    {
        // var must be positive
        for (int i=0; i<channels; i++)
        {
            float w = weights[2][i];
            if (w == 0.f) weights[2][i] = 0.001;
            if (w < 0.f) weights[2][i] = -w;
        }
    }
    weights[3] = RandomMat(channels);
    ncnn::ModelBinFromMatArray mb(weights.data());

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

    int ret = test_layer<ncnn::BatchNorm>("BatchNorm", pd, mb, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_batchnorm failed a.dims=%d a=(%d %d %d) channels=%d eps=%f use_packing_layout=%d\n", a.dims, a.w, a.h, a.c, channels, eps, use_packing_layout);
    }

    return ret;
}

static int test_batchnorm_0()
{
    return 0
        || test_batchnorm(RandomMat(6, 7, 16), 16, 0.f, false)
        || test_batchnorm(RandomMat(6, 7, 16), 16, 0.01f, false)
        || test_batchnorm(RandomMat(3, 5, 13), 13, 0.f, false)
        || test_batchnorm(RandomMat(3, 5, 13), 13, 0.001f, false)

        || test_batchnorm(RandomMat(6, 7, 16), 16, 0.f, true)
        || test_batchnorm(RandomMat(6, 7, 16), 16, 0.01f, true)
        || test_batchnorm(RandomMat(3, 5, 13), 13, 0.f, true)
        || test_batchnorm(RandomMat(3, 5, 13), 13, 0.001f, true)
        ;
}

static int test_batchnorm_1()
{
    return 0
        || test_batchnorm(RandomMat(6, 16), 16, 0.f, false)
        || test_batchnorm(RandomMat(6, 16), 16, 0.01f, false)
        || test_batchnorm(RandomMat(7, 15), 15, 0.f, false)
        || test_batchnorm(RandomMat(7, 15), 15, 0.001f, false)

        || test_batchnorm(RandomMat(6, 16), 16, 0.f, true)
        || test_batchnorm(RandomMat(6, 16), 16, 0.01f, true)
        || test_batchnorm(RandomMat(7, 15), 15, 0.f, true)
        || test_batchnorm(RandomMat(7, 15), 15, 0.001f, true)
        ;
}

static int test_batchnorm_2()
{
    return 0
        || test_batchnorm(RandomMat(128), 128, 0.f, false)
        || test_batchnorm(RandomMat(128), 128, 0.001f, false)
        || test_batchnorm(RandomMat(127), 127, 0.f, false)
        || test_batchnorm(RandomMat(127), 127, 0.1f, false)

        || test_batchnorm(RandomMat(128), 128, 0.f, true)
        || test_batchnorm(RandomMat(128), 128, 0.001f, true)
        || test_batchnorm(RandomMat(127), 127, 0.f, true)
        || test_batchnorm(RandomMat(127), 127, 0.1f, true)
        ;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_batchnorm_0()
        || test_batchnorm_1()
        || test_batchnorm_2()
        ;
}
