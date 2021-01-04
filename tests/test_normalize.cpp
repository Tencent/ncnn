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

#include "layer/normalize.h"
#include "testutil.h"

static int test_normalize(const ncnn::Mat& a, int across_spatial, int across_channel, int channel_shared, float eps, int eps_mode)
{
    int scale_data_size = channel_shared ? 1 : a.c;

    ncnn::ParamDict pd;
    pd.set(0, across_spatial);
    pd.set(4, across_channel);
    pd.set(1, channel_shared);
    pd.set(2, eps);
    pd.set(3, scale_data_size);
    pd.set(9, eps_mode);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(scale_data_size);

    int ret = test_layer<ncnn::Normalize>("Normalize", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_normalize failed a.dims=%d a=(%d %d %d) across_spatial=%d across_channel=%d channel_shared=%d eps=%f eps_mode=%d\n", a.dims, a.w, a.h, a.c, across_spatial, across_channel, channel_shared, eps, eps_mode);
    }

    return ret;
}

static int test_normalize_0()
{
    ncnn::Mat a = RandomMat(6, 4, 2);
    ncnn::Mat b = RandomMat(5, 7, 8);

    return 0
           || test_normalize(a, 1, 0, 0, 0.01f, 0)
           || test_normalize(a, 1, 0, 0, 0.001f, 1)
           || test_normalize(a, 1, 0, 0, 0.002f, 2)
           || test_normalize(a, 1, 0, 1, 0.01f, 0)
           || test_normalize(a, 1, 0, 1, 0.001f, 1)
           || test_normalize(a, 1, 0, 1, 0.002f, 2)
           || test_normalize(b, 1, 0, 0, 0.01f, 0)
           || test_normalize(b, 1, 0, 0, 0.001f, 1)
           || test_normalize(b, 1, 0, 0, 0.002f, 2)
           || test_normalize(b, 1, 0, 1, 0.01f, 0)
           || test_normalize(b, 1, 0, 1, 0.001f, 1)
           || test_normalize(b, 1, 0, 1, 0.002f, 2);
}

static int test_normalize_1()
{
    ncnn::Mat a = RandomMat(5, 6, 3);
    ncnn::Mat b = RandomMat(3, 4, 8);

    return 0
           || test_normalize(a, 0, 1, 0, 0.01f, 0)
           || test_normalize(a, 0, 1, 0, 0.001f, 1)
           || test_normalize(a, 0, 1, 0, 0.002f, 2)
           || test_normalize(a, 0, 1, 1, 0.01f, 0)
           || test_normalize(a, 0, 1, 1, 0.001f, 1)
           || test_normalize(a, 0, 1, 1, 0.002f, 2)
           || test_normalize(b, 0, 1, 0, 0.01f, 0)
           || test_normalize(b, 0, 1, 0, 0.001f, 1)
           || test_normalize(b, 0, 1, 0, 0.002f, 2)
           || test_normalize(b, 0, 1, 1, 0.01f, 0)
           || test_normalize(b, 0, 1, 1, 0.001f, 1)
           || test_normalize(b, 0, 1, 1, 0.002f, 2);
}

static int test_normalize_2()
{
    ncnn::Mat a = RandomMat(2, 3, 5);
    ncnn::Mat b = RandomMat(4, 6, 8);

    return 0
           || test_normalize(a, 1, 1, 0, 0.01f, 0)
           || test_normalize(a, 1, 1, 0, 0.001f, 1)
           || test_normalize(a, 1, 1, 0, 0.002f, 2)
           || test_normalize(a, 1, 1, 1, 0.01f, 0)
           || test_normalize(a, 1, 1, 1, 0.001f, 1)
           || test_normalize(a, 1, 1, 1, 0.002f, 2)
           || test_normalize(b, 1, 1, 0, 0.01f, 0)
           || test_normalize(b, 1, 1, 0, 0.001f, 1)
           || test_normalize(b, 1, 1, 0, 0.002f, 2)
           || test_normalize(b, 1, 1, 1, 0.01f, 0)
           || test_normalize(b, 1, 1, 1, 0.001f, 1)
           || test_normalize(b, 1, 1, 1, 0.002f, 2);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_normalize_0()
           || test_normalize_1()
           || test_normalize_2();
}
