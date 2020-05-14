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

#include "layer/instancenorm.h"

static int test_instancenorm(const ncnn::Mat& a, float eps)
{
    int channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, channels);
    pd.set(1, eps);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::InstanceNorm>("InstanceNorm", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_instancenorm failed a.dims=%d a=(%d %d %d) eps=%f\n", a.dims, a.w, a.h, a.c, eps);
    }

    return ret;
}

static int test_instancenorm_0()
{
    return 0
        || test_instancenorm(RandomMat(6, 4, 2), 0.01f)
        || test_instancenorm(RandomMat(3, 3, 8), 0.002f)
        ;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_instancenorm_0()
        ;
}
