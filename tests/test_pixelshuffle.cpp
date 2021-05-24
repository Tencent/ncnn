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

#include "layer/pixelshuffle.h"
#include "testutil.h"

static int test_pixelshuffle(const ncnn::Mat& a, int upscale_factor, int mode)
{
    ncnn::ParamDict pd;
    pd.set(0, upscale_factor);
    pd.set(1, mode);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::PixelShuffle>("PixelShuffle", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_pixelshuffle failed a.dims=%d a=(%d %d %d) upscale_factor=%d mode=%d\n", a.dims, a.w, a.h, a.c, upscale_factor, mode);
    }

    return ret;
}

static int test_pixelshuffle_0()
{
    return 0
           || test_pixelshuffle(RandomMat(7, 7, 1), 1, 0)
           || test_pixelshuffle(RandomMat(7, 7, 8), 2, 0)
           || test_pixelshuffle(RandomMat(7, 7, 12), 2, 0)
           || test_pixelshuffle(RandomMat(7, 7, 64), 4, 0)
           || test_pixelshuffle(RandomMat(7, 7, 32), 2, 0)
           || test_pixelshuffle(RandomMat(7, 7, 48), 2, 0)
           || test_pixelshuffle(RandomMat(7, 7, 36), 3, 0)
           || test_pixelshuffle(RandomMat(7, 7, 72), 3, 0)
           || test_pixelshuffle(RandomMat(7, 7, 90), 3, 0);
}

static int test_pixelshuffle_1()
{
    return 0
           || test_pixelshuffle(RandomMat(7, 7, 1), 1, 1)
           || test_pixelshuffle(RandomMat(7, 7, 8), 2, 1)
           || test_pixelshuffle(RandomMat(7, 7, 12), 2, 1)
           || test_pixelshuffle(RandomMat(7, 7, 64), 4, 1)
           || test_pixelshuffle(RandomMat(7, 7, 32), 2, 1)
           || test_pixelshuffle(RandomMat(7, 7, 48), 2, 1)
           || test_pixelshuffle(RandomMat(7, 7, 36), 3, 1)
           || test_pixelshuffle(RandomMat(7, 7, 90), 3, 1);
}

int main()
{
    SRAND(7767517);

    return test_pixelshuffle_0() || test_pixelshuffle_1();
}
