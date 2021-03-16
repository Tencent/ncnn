// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/dequantize.h"
#include "testutil.h"

static int test_dequantize(const ncnn::Mat& a, float scale, int bias)
{
    int bias_data_size;
    if (a.dims == 1) bias_data_size = a.w;
    if (a.dims == 2) bias_data_size = a.h;
    if (a.dims == 3) bias_data_size = a.c;

    ncnn::ParamDict pd;
    pd.set(0, scale);
    pd.set(1, bias);
    pd.set(2, bias_data_size);

    std::vector<ncnn::Mat> weights(bias ? 1 : 0);
    if (bias)
        weights[0] = RandomMat(bias_data_size);

    int ret = test_layer<ncnn::Dequantize>("Dequantize", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_dequantize failed a.dims=%d a=(%d %d %d) scale=%f bias=%d\n", a.dims, a.w, a.h, a.c, scale, bias);
    }

    return ret;
}

static int test_dequantize_0()
{
    return 0
           || test_dequantize(RandomMat(5, 7, 24), 1.0f, 1)
           || test_dequantize(RandomMat(5, 7, 24), 2.4f, 0)
           || test_dequantize(RandomMat(7, 9, 12), 1.0f, 1)
           || test_dequantize(RandomMat(7, 9, 12), 2.4f, 0)
           || test_dequantize(RandomMat(3, 5, 13), 1.0f, 1)
           || test_dequantize(RandomMat(3, 5, 13), 2.4f, 0);
}

static int test_dequantize_1()
{
    return 0
           || test_dequantize(RandomMat(15, 24), 1.0f, 1)
           || test_dequantize(RandomMat(15, 24), 2.4f, 0)
           || test_dequantize(RandomMat(17, 12), 1.0f, 1)
           || test_dequantize(RandomMat(17, 12), 2.4f, 0)
           || test_dequantize(RandomMat(19, 15), 1.0f, 1)
           || test_dequantize(RandomMat(19, 15), 2.4f, 0);
}

static int test_dequantize_2()
{
    return 0
           || test_dequantize(RandomMat(128), 1.0f, 1)
           || test_dequantize(RandomMat(128), 2.4f, 0)
           || test_dequantize(RandomMat(124), 1.0f, 1)
           || test_dequantize(RandomMat(124), 2.4f, 0)
           || test_dequantize(RandomMat(127), 1.0f, 1)
           || test_dequantize(RandomMat(127), 2.4f, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_dequantize_0()
           || test_dequantize_1()
           || test_dequantize_2();
}
