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

#include "testutil.h"

static int test_multiheadattention(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int num_heads, int kdim, int vdim, int attn_mask)
{
    int embed_dim = q.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * embed_dim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);

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

    if (attn_mask)
    {
        as.push_back(RandomMat(k.h, q.h));
    }

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention failed q=(%d %d) k=(%d %d) v=(%d %d) num_heads=%d kdim=%d vdim=%d attn_mask=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, num_heads, kdim, vdim, attn_mask);
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

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_samekv failed q=(%d %d) kv=(%d %d) num_heads=%d kvdim=%d\n", q.w, q.h, kv.w, kv.h, num_heads, kvdim);
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

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_sameqkv failed a=(%d %d) num_heads=%d\n", a.w, a.h, num_heads);
    }

    return ret;
}

static int test_multiheadattention_0()
{
    return 0
           || test_multiheadattention(RandomMat(62, 66), RandomMat(32, 66), RandomMat(20, 66), 2, 32, 20, 0)
           || test_multiheadattention(RandomMat(26, 64), RandomMat(32, 64), RandomMat(18, 64), 2, 32, 18, 1)
           || test_multiheadattention(RandomMat(64, 128), RandomMat(64, 128), RandomMat(64, 128), 4, 64, 64, 0)
           || test_multiheadattention(RandomMat(64, 127), RandomMat(64, 127), RandomMat(64, 127), 16, 64, 64, 1)
           || test_multiheadattention(RandomMat(16, 128), RandomMat(44, 128), RandomMat(55, 128), 2, 44, 55, 0)
           || test_multiheadattention(RandomMat(16, 128), RandomMat(44, 127), RandomMat(55, 127), 4, 44, 55, 1)
           || test_multiheadattention(RandomMat(12, 17), RandomMat(28, 127), RandomMat(32, 127), 3, 28, 32, 0)
           || test_multiheadattention(RandomMat(12, 17), RandomMat(28, 32), RandomMat(11, 32), 3, 28, 11, 1);
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
           || test_multiheadattention_sameqkv(RandomMat(64, 128), 4)
           || test_multiheadattention_sameqkv(RandomMat(64, 127), 8);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_multiheadattention_0()
           || test_multiheadattention_1()
           || test_multiheadattention_2();
}
