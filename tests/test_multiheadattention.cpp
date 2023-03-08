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

#include "layer/multiheadattention.h"
#include "testutil.h"

static int test_multiheadattention(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int num_heads, int kdim, int vdim)
{
    int embed_dim = q.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * embed_dim);
    pd.set(3, kdim);
    pd.set(4, vdim);

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * embed_dim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * kdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * vdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(embed_dim * embed_dim);
    weights[7] = RandomMat(embed_dim);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    float epsilon = 0.005;

    int ret = test_layer<ncnn::MultiHeadAttention>("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention failed q=(%d %d) k=(%d %d) v=(%d %d)\n", q.w, q.h, k.w, k.h, v.w, v.h);
    }

    return ret;
}

static int test_multiheadattention_samekv(const ncnn::Mat& q, const ncnn::Mat& kv, int num_heads, int kvdim)
{
    int embed_dim = q.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * embed_dim);
    pd.set(3, kvdim);
    pd.set(4, kvdim);

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * embed_dim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * kvdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * kvdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(embed_dim * embed_dim);
    weights[7] = RandomMat(embed_dim);

    std::vector<ncnn::Mat> as(2);
    as[0] = q;
    as[1] = kv;

    float epsilon = 0.005;

    int ret = test_layer<ncnn::MultiHeadAttention>("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_samekv failed q=(%d %d) kv=(%d %d)\n", q.w, q.h, kv.w, kv.h);
    }

    return ret;
}

static int test_multiheadattention_sameqkv(const ncnn::Mat& a, int num_heads)
{
    int embed_dim = a.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * embed_dim);

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * embed_dim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * embed_dim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * embed_dim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(embed_dim * embed_dim);
    weights[7] = RandomMat(embed_dim);

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    float epsilon = 0.005;

    int ret = test_layer<ncnn::MultiHeadAttention>("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_sameqkv failed a=(%d %d)\n", a.w, a.h);
    }

    return ret;
}

static int test_multiheadattention_0()
{
    return 0
           || test_multiheadattention(RandomMat(64, 128), RandomMat(64, 128), RandomMat(64, 128), 4, 64, 64)
           || test_multiheadattention(RandomMat(64, 127), RandomMat(64, 127), RandomMat(64, 127), 16, 64, 64)
           || test_multiheadattention(RandomMat(16, 128), RandomMat(44, 128), RandomMat(55, 128), 2, 44, 55)
           || test_multiheadattention(RandomMat(16, 128), RandomMat(44, 127), RandomMat(55, 127), 4, 44, 55)
           || test_multiheadattention(RandomMat(12, 17), RandomMat(28, 127), RandomMat(32, 127), 3, 28, 32)
           || test_multiheadattention(RandomMat(12, 17), RandomMat(28, 32), RandomMat(11, 32), 3, 28, 11);
}

static int test_multiheadattention_1()
{
    return 0
           || test_multiheadattention_samekv(RandomMat(64, 128), RandomMat(64, 128), 4, 64)
           || test_multiheadattention_samekv(RandomMat(64, 127), RandomMat(64, 127), 16, 64)
           || test_multiheadattention_samekv(RandomMat(16, 128), RandomMat(44, 128), 2, 44)
           || test_multiheadattention_samekv(RandomMat(16, 128), RandomMat(22, 127), 4, 22)
           || test_multiheadattention_samekv(RandomMat(12, 17), RandomMat(28, 127), 3, 28)
           || test_multiheadattention_samekv(RandomMat(12, 17), RandomMat(11, 32), 3, 11);
}

static int test_multiheadattention_2()
{
    return 0
           || test_multiheadattention_sameqkv(RandomMat(64, 128), 8)
           || test_multiheadattention_sameqkv(RandomMat(64, 127), 32);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_multiheadattention_0()
           || test_multiheadattention_1()
           || test_multiheadattention_2();
}
