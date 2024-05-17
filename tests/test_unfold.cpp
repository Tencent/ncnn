// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_unfold(int w, int h, int c, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_w, int pad_h, float pad_value)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(1, kernel_w);
    pd.set(11, kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_w);
    pd.set(14, pad_h);
    pd.set(18, pad_value);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Unfold", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_unfold failed w=%d h=%d c=%d kernel=%d,%d dilation=%d,%d stride=%d,%d pad=%d,%d pad_value=%f\n", w, h, c, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_w, pad_h, pad_value);
    }

    return ret;
}

static int test_unfold_0()
{
    return 0
           || test_unfold(32, 32, 11, 3, 3, 1, 1, 1, 1, 0, 0, 0.f)
           || test_unfold(32, 32, 12, 4, 2, 1, 1, 1, 2, 2, 2, -0.5f)
           || test_unfold(32, 32, 16, 3, 2, 2, 1, 1, 1, 4, 2, 2.f);
}

static int test_unfold_1()
{
    return 0
           || test_unfold(32, 32, 11, 3, 3, 1, 1, 1, 1, -233, -233, -0.5f)
           || test_unfold(32, 32, 12, 4, 2, 1, 1, 1, 2, -234, -234, 0.f)
           || test_unfold(32, 32, 16, 3, 2, 2, 1, 1, 1, -233, -233, 1.f);
}

int main()
{
    SRAND(7767517);

    return test_unfold_0() || test_unfold_1();
}
