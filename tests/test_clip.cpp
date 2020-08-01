// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/clip.h"
#include "testutil.h"

static int test_clip(const ncnn::Mat& a, float min, float max)
{
    ncnn::ParamDict pd;
    pd.set(0, min);
    pd.set(1, max);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Clip>("Clip", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_clip failed a.dims=%d a=(%d,%d,%d) min=%f max=%f\n", a.dims, a.w, a.h, a.c, min, max);
    }

    return ret;
}

static int test_clip_0()
{
    return 0
           || test_clip(RandomMat(5, 7, 24), -1.f, 1.f)
           || test_clip(RandomMat(7, 9, 12), -1.f, 1.f)
           || test_clip(RandomMat(3, 5, 13), -1.f, 1.f);
}

static int test_clip_1()
{
    return 0
           || test_clip(RandomMat(15, 24), -1.f, 1.f)
           || test_clip(RandomMat(17, 12), -1.f, 1.f)
           || test_clip(RandomMat(19, 15), -1.f, 1.f);
}

static int test_clip_2()
{
    return 0
           || test_clip(RandomMat(128), -1.f, 1.f)
           || test_clip(RandomMat(124), -1.f, 1.f)
           || test_clip(RandomMat(127), -1.f, 1.f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_clip_0()
           || test_clip_1()
           || test_clip_2();
}
