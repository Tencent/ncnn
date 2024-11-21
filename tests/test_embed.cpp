// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_embed(int words, int num_output, int input_dim, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, input_dim);
    pd.set(2, bias);
    pd.set(3, num_output * input_dim);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(num_output * input_dim);
    if (bias)
        weights[1] = RandomMat(num_output);

    ncnn::Mat a(words);
    RandomizeInt(a, 0, input_dim);

    int ret = test_layer("Embed", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_embed failed words=%d num_output=%d input_dim=%d bias=%d\n", words, num_output, input_dim, bias);
    }

    return ret;
}

static int test_embed_0()
{
    return 0
           || test_embed(128, 128, 128, 0)
           || test_embed(128, 128, 128, 1)
           || test_embed(127, 127, 127, 0)
           || test_embed(127, 127, 127, 1)
           || test_embed(124, 124, 124, 0)
           || test_embed(124, 124, 124, 1);
}

#if NCNN_INT8
static int test_embed_int8(int words, int num_output, int input_dim, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, input_dim);
    pd.set(2, bias);
    pd.set(3, num_output * input_dim);
    pd.set(18, 2);

    std::vector<ncnn::Mat> weights(bias ? 3 : 2);
    weights[0] = RandomS8Mat(num_output * input_dim);
    if (bias)
    {
        weights[1] = RandomMat(num_output);
        weights[2] = RandomMat(1, 100.f, 200.f);
    }
    else
    {
        weights[1] = RandomMat(1, 100.f, 200.f);
    }

    ncnn::Mat a(words);
    RandomizeInt(a, 0, input_dim);

    int ret = test_layer("Embed", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_embed_int8 failed words=%d num_output=%d input_dim=%d bias=%d\n", words, num_output, input_dim, bias);
    }

    return ret;
}

static int test_embed_1()
{
    return 0
           || test_embed_int8(128, 128, 128, 0)
           || test_embed_int8(128, 128, 128, 1)
           || test_embed_int8(127, 127, 127, 0)
           || test_embed_int8(127, 127, 127, 1)
           || test_embed_int8(124, 124, 124, 0)
           || test_embed_int8(124, 124, 124, 1);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return test_embed_0() || test_embed_1();
#else
    return test_embed_0();
#endif
}
