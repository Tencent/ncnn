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

#include "layer/gemm.h"
#include "testutil.h"

static int test_gemm(int M, int N, int K, float alpha, int transA, int transB)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, 1.f); // beta
    pd.set(2, transA);
    pd.set(3, transB);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a(2);
    a[0] = transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    a[1] = transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K);

    Randomize(a[0]);
    Randomize(a[1]);

    int ret = test_layer<ncnn::Gemm>("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm failed M=%d N=%d K=%d alpha=%f transA=%d transB=%d\n", M, N, K, alpha, transA, transB);
    }

    return ret;
}

static int test_gemm_bias(int M, int N, int K, const ncnn::Mat& C, float alpha, float beta, int transA, int transB)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, beta);
    pd.set(2, transA);
    pd.set(3, transB);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a(3);
    a[0] = transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    a[1] = transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K);
    a[2] = C;

    Randomize(a[0]);
    Randomize(a[1]);

    int ret = test_layer<ncnn::Gemm>("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_bias failed M=%d N=%d K=%d C.dims=%d C=(%d %d %d) alpha=%f transA=%d transB=%d\n", M, N, K, C.dims, C.w, C.h, C.c, alpha, transA, transB);
    }

    return ret;
}

static int test_gemm_0()
{
    return 0
           || test_gemm(13, 14, 15, 0.1f, 0, 0)
           || test_gemm(13, 14, 15, 0.3f, 1, 0)
           || test_gemm(13, 14, 15, -0.4f, 0, 1)
           || test_gemm(13, 14, 15, 1.7f, 1, 1)
           || test_gemm(16, 24, 15, 0.1f, 0, 0)
           || test_gemm(16, 24, 15, 0.3f, 1, 0)
           || test_gemm(16, 24, 15, -0.4f, 0, 1)
           || test_gemm(16, 24, 15, 1.7f, 1, 1);
}

static int test_gemm_1()
{
    return 0
           || test_gemm_bias(13, 14, 15, RandomMat(1), 0.1f, 0.2f, 0, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(1), 0.4f, -1.2f, 1, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(1), -0.3f, 3.f, 0, 1)
           || test_gemm_bias(13, 14, 15, RandomMat(1), 1.7f, 1.f, 1, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(1), 0.1f, 0.2f, 0, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(1), 0.4f, -1.2f, 1, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(1), -0.3f, 3.f, 0, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(1), 1.7f, 1.f, 1, 1);
}

static int test_gemm_2()
{
    return 0
           || test_gemm_bias(13, 14, 15, RandomMat(13), 0.1f, 1.f, 0, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(13), 0.4f, 2.f, 1, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(13), -0.3f, 0.11f, 0, 1)
           || test_gemm_bias(13, 14, 15, RandomMat(13), 1.7f, -20.f, 1, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(13), 0.1f, 1.f, 0, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(13), 0.4f, 2.f, 1, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(13), -0.3f, 0.11f, 0, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(13), 1.7f, -20.f, 1, 1);
}

static int test_gemm_3()
{
    return 0
           || test_gemm_bias(13, 14, 15, RandomMat(13, 1), 0.1f, 4.f, 0, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(13, 1), 0.4f, 1.f, 1, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(13, 1), -0.3f, -0.01f, 0, 1)
           || test_gemm_bias(13, 14, 15, RandomMat(13, 1), 1.7f, 0.3f, 1, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(13, 1), 0.1f, 4.f, 0, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(13, 1), 0.4f, 1.f, 1, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(13, 1), -0.3f, -0.01f, 0, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(13, 1), 1.7f, 0.3f, 1, 1);
}

static int test_gemm_4()
{
    return 0
           || test_gemm_bias(13, 14, 15, RandomMat(13, 14), 0.1f, 6.f, 0, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(13, 14), 0.4f, 1.22f, 1, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(13, 14), -0.3f, 1.01f, 0, 1)
           || test_gemm_bias(13, 14, 15, RandomMat(13, 14), 1.7f, 0.3f, 1, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(13, 14), 0.1f, 6.f, 0, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(13, 14), 0.4f, 1.22f, 1, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(13, 14), -0.3f, 1.01f, 0, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(13, 14), 1.7f, 0.3f, 1, 1);
}

static int test_gemm_5()
{
    return 0
           || test_gemm_bias(13, 14, 15, RandomMat(1, 14), 0.1f, 0.4f, 0, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(1, 14), 0.4f, -1.f, 1, 0)
           || test_gemm_bias(13, 14, 15, RandomMat(1, 14), -0.3f, -0.21f, 0, 1)
           || test_gemm_bias(13, 14, 15, RandomMat(1, 14), 1.7f, 1.3f, 1, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(1, 14), 0.1f, 0.4f, 0, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(1, 14), 0.4f, -1.f, 1, 0)
           || test_gemm_bias(16, 24, 15, RandomMat(1, 14), -0.3f, -0.21f, 0, 1)
           || test_gemm_bias(16, 24, 15, RandomMat(1, 14), 1.7f, 1.3f, 1, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gemm_0()
           || test_gemm_1()
           || test_gemm_2()
           || test_gemm_3()
           || test_gemm_4()
           || test_gemm_5();
}
