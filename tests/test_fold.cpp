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

static int test_fold(int w, int h, int outw, int outh, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_w, int pad_h)
{
    ncnn::Mat a = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(1, kernel_w);
    pd.set(11, kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_w);
    pd.set(14, pad_h);
    pd.set(20, outw);
    pd.set(21, outh);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Fold", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_fold failed w=%d h=%d outw=%d outh=%d kernel=%d,%d dilation=%d,%d stride=%d,%d pad=%d,%d\n", w, h, outw, outh, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_w, pad_h);
    }

    return ret;
}

static int test_fold_0()
{
    return 0
           || test_fold(400, 108, 22, 22, 3, 3, 1, 1, 1, 1, 0, 0)
           || test_fold(190, 96, 18, 17, 4, 2, 1, 1, 1, 2, 2, 2)
           || test_fold(120, 36, 11, 5, 3, 2, 2, 1, 1, 1, 4, 2);
}

int main()
{
    SRAND(7767517);

    return test_fold_0();
}
