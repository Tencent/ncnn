// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gemm_oom(int M, int N, int K, int transA, int transB, int output_transpose, int constantA, int constantB, int output_N1M)
{
    ncnn::ParamDict pd;
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(11, output_N1M);
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? (output_N1M ? ncnn::Mat(M, 1, K) : ncnn::Mat(M, K)) : (output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M)));
    if (constantB) weights.push_back(transB ? (output_N1M ? ncnn::Mat(K, 1, N) : ncnn::Mat(K, N)) : (output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K)));

    std::vector<ncnn::Mat> a;
    if (!constantA) a.push_back(transA ? (output_N1M ? ncnn::Mat(M, 1, K) : ncnn::Mat(M, K)) : (output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M)));
    if (!constantB) a.push_back(transB ? (output_N1M ? ncnn::Mat(K, 1, N) : ncnn::Mat(K, N)) : (output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K)));

    for (size_t i = 0; i < weights.size(); i++)
    {
        Randomize(weights[i]);
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i]);
    }

    int ret = test_layer_oom("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_oom failed M=%d N=%d K=%d transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d output_N1M=%d\n", M, N, K, transA, transB, output_transpose, constantA, constantB, output_N1M);
    }

    return ret;
}

static int test_gemm_bias_oom(int M, int N, int K, const ncnn::Mat& C, float alpha, float beta, int transA, int transB, int output_transpose, int constantA, int constantB, int constantC)
{
    int broadcast_type_C = 0;
    if (C.dims == 1 && C.w == 1)
    {
        // scalar
        broadcast_type_C = 0;
    }
    if (C.dims == 1 && C.w == M)
    {
        // M
        // auto broadcast from h to w is the ncnn-style convention
        broadcast_type_C = 1;
    }
    if (C.dims == 1 && C.w == N)
    {
        // N
        broadcast_type_C = 4;
    }
    if (C.dims == 2 && C.w == 1 && C.h == M)
    {
        // Mx1
        broadcast_type_C = 2;
    }
    if (C.dims == 2 && C.w == N && C.h == M)
    {
        // MxN
        broadcast_type_C = 3;
    }
    if (C.dims == 2 && C.w == N && C.h == 1)
    {
        // 1xN
        broadcast_type_C = 4;
    }

    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, beta);
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, constantC);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, broadcast_type_C);
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
    if (constantB) weights.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));
    if (constantC) weights.push_back(C);

    std::vector<ncnn::Mat> a;
    if (!constantA) a.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
    if (!constantB) a.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));
    if (!constantC) a.push_back(C);

    for (size_t i = 0; i < weights.size(); i++)
    {
        Randomize(weights[i]);
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i]);
    }

    int ret = test_layer_oom("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_bias_oom failed M=%d N=%d K=%d C.dims=%d C=(%d %d %d) alpha=%f beta=%f transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d constantC=%d\n", M, N, K, C.dims, C.w, C.h, C.c, alpha, beta, transA, transB, output_transpose, constantA, constantB, constantC);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K)
{
    return 0
           || test_gemm_oom(M, N, K, 0, 0, 0, 0, 0, 0)
           || test_gemm_oom(M, N, K, 1, 0, 1, 0, 0, 0)
           || test_gemm_oom(M, N, K, 0, 1, 1, 1, 0, 0)
           || test_gemm_oom(M, N, K, 1, 1, 0, 1, 0, 1)
           || test_gemm_oom(M, N, K, 0, 0, 0, 0, 1, 0)
           || test_gemm_oom(M, N, K, 0, 1, 1, 0, 1, 0)
           || test_gemm_oom(M, N, K, 1, 0, 1, 1, 1, 0)
           || test_gemm_oom(M, N, K, 1, 1, 0, 1, 1, 1);
}

static int test_gemm_1(int M, int N, int K)
{
    return 0
           || test_gemm_bias_oom(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 0)
           || test_gemm_bias_oom(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 0, 1, 0, 0)
           || test_gemm_bias_oom(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 1, 0, 1, 0, 1, 0)
           || test_gemm_bias_oom(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 1, 1, 1, 1, 0)
           || test_gemm_bias_oom(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 1)
           || test_gemm_bias_oom(M, N, K, RandomMat(N), 3.1f, 0.6f, 0, 1, 0, 1, 0, 1)
           || test_gemm_bias_oom(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 0, 0, 1, 1)
           || test_gemm_bias_oom(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 1, 1, 1, 1, 1);
}

#if NCNN_INT8
static int test_gemm_int8_oom(int M, int N, int K, int transA, int transB, int output_elemtype, int output_transpose, int constantA, int constantB, int output_N1M)
{
    ncnn::ParamDict pd;
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(11, output_N1M);
    pd.set(13, output_elemtype);
    pd.set(14, output_transpose);
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? RandomS8Mat(M, K) : RandomS8Mat(K, M));
    if (constantB) weights.push_back(transB ? RandomS8Mat(K, N) : RandomS8Mat(N, K));
    if (constantA) weights.push_back(RandomMat(M, 10.f, 20.f));
    if (constantB) weights.push_back(RandomMat(1, 10.f, 20.f));

    std::vector<ncnn::Mat> a;
    if (!constantA)
    {
        a.push_back(transA ? (output_N1M ? ncnn::Mat(M, 1, K) : ncnn::Mat(M, K)) : (output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M)));
    }
    if (!constantB)
    {
        a.push_back(transB ? (output_N1M ? ncnn::Mat(K, 1, N) : ncnn::Mat(K, N)) : (output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K)));
    }

    int ret = test_layer_oom("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_int8_oom failed M=%d N=%d K=%d transA=%d transB=%d output_elemtype=%d output_transpose=%d constantA=%d constantB=%d output_N1M=%d\n", M, N, K, transA, transB, output_elemtype, output_transpose, constantA, constantB, output_N1M);
    }

    return ret;
}

static int test_gemm_int8_bias_oom(int M, int N, int K, const ncnn::Mat& C, float alpha, float beta, int transA, int transB, int output_elemtype, int output_transpose, int constantA, int constantB, int constantC)
{
    int broadcast_type_C = 0;
    if (C.dims == 1 && C.w == 1)
    {
        // scalar
        broadcast_type_C = 0;
    }
    if (C.dims == 1 && C.w == M)
    {
        // M
        // auto broadcast from h to w is the ncnn-style convention
        broadcast_type_C = 1;
    }
    if (C.dims == 1 && C.w == N)
    {
        // N
        broadcast_type_C = 4;
    }
    if (C.dims == 2 && C.w == 1 && C.h == M)
    {
        // Mx1
        broadcast_type_C = 2;
    }
    if (C.dims == 2 && C.w == N && C.h == M)
    {
        // MxN
        broadcast_type_C = 3;
    }
    if (C.dims == 2 && C.w == N && C.h == 1)
    {
        // 1xN
        broadcast_type_C = 4;
    }

    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, beta);
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, constantC);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, broadcast_type_C);
    pd.set(13, output_elemtype);
    pd.set(14, output_transpose);
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? RandomS8Mat(M, K) : RandomS8Mat(K, M));
    if (constantB) weights.push_back(transB ? RandomS8Mat(K, N) : RandomS8Mat(N, K));
    if (constantC) weights.push_back(C);
    if (constantA) weights.push_back(RandomMat(M, 10.f, 20.f));
    if (constantB) weights.push_back(RandomMat(1, 10.f, 20.f));

    std::vector<ncnn::Mat> a;
    if (!constantA)
    {
        a.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
    }
    if (!constantB)
    {
        a.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));
    }
    if (!constantC) a.push_back(C);

    int ret = test_layer_oom("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_int8_bias_oom failed M=%d N=%d K=%d C.dims=%d C=(%d %d %d) alpha=%f beta=%f transA=%d transB=%d output_elemtype=%d output_transpose=%d constantA=%d constantB=%d constantC=%d\n", M, N, K, C.dims, C.w, C.h, C.c, alpha, beta, transA, transB, output_elemtype, output_transpose, constantA, constantB, constantC);
    }

    return ret;
}

static int test_gemm_int8_fp16s_oom(int M, int N, int K, int transA, int transB, int output_elemtype, int output_transpose, int constantA, int constantB, int output_N1M)
{
    ncnn::ParamDict pd;
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(11, output_N1M);
    pd.set(13, output_elemtype);
    pd.set(14, output_transpose);
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? RandomS8Mat(M, K) : RandomS8Mat(K, M));
    if (constantB) weights.push_back(transB ? RandomS8Mat(K, N) : RandomS8Mat(N, K));
    if (constantA) weights.push_back(RandomMat(M, 10.f, 20.f));
    if (constantB) weights.push_back(RandomMat(1, 10.f, 20.f));

    std::vector<ncnn::Mat> a;
    if (!constantA)
    {
        a.push_back(transA ? (output_N1M ? ncnn::Mat(M, 1, K) : ncnn::Mat(M, K)) : (output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M)));
    }
    if (!constantB)
    {
        a.push_back(transB ? (output_N1M ? ncnn::Mat(K, 1, N) : ncnn::Mat(K, N)) : (output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K)));
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    float epsilon = 0.001;

    int ret = test_layer_oom_opt("Gemm", pd, weights, opt, a, 1, epsilon);
    if (ret != 233 && ret != 0)
    {
        fprintf(stderr, "test_gemm_int8_fp16s_oom failed M=%d N=%d K=%d transA=%d transB=%d output_elemtype=%d output_transpose=%d constantA=%d constantB=%d output_N1M=%d\n", M, N, K, transA, transB, output_elemtype, output_transpose, constantA, constantB, output_N1M);
        return ret;
    }

    return 0;
}

static int test_gemm_2(int M, int N, int K)
{
    return 0
           || test_gemm_int8_oom(M, N, K, 0, 0, 0, 0, 0, 0, 0)
           || test_gemm_int8_oom(M, N, K, 1, 0, 0, 1, 0, 0, 0)
           || test_gemm_int8_oom(M, N, K, 0, 1, 1, 1, 1, 0, 0)
           || test_gemm_int8_oom(M, N, K, 1, 1, 0, 0, 1, 0, 1)
           || test_gemm_int8_oom(M, N, K, 0, 0, 2, 0, 0, 1, 0)
           || test_gemm_int8_oom(M, N, K, 0, 1, 0, 1, 0, 1, 0)
           || test_gemm_int8_oom(M, N, K, 1, 0, 0, 1, 1, 1, 0)
           || test_gemm_int8_oom(M, N, K, 1, 1, 3, 0, 1, 1, 1);
}

static int test_gemm_3(int M, int N, int K)
{
    return 0
           || test_gemm_int8_bias_oom(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 0, 0)
           || test_gemm_int8_bias_oom(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 0, 0, 1, 0, 0)
           || test_gemm_int8_bias_oom(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 1, 0, 1, 1, 0, 1, 0)
           || test_gemm_int8_bias_oom(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 1, 0, 1, 1, 1, 0)
           || test_gemm_int8_bias_oom(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 0, 0, 3, 0, 0, 0, 1)
           || test_gemm_int8_bias_oom(M, N, K, RandomMat(N), 3.1f, 0.6f, 0, 1, 0, 0, 1, 0, 1)
           || test_gemm_int8_bias_oom(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 0, 0, 0, 1, 1)
           || test_gemm_int8_bias_oom(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 1, 2, 1, 1, 1, 1);
}

static int test_gemm_4(int M, int N, int K)
{
    return 0
           || test_gemm_int8_fp16s_oom(M, N, K, 0, 0, 0, 0, 0, 0, 0)
           || test_gemm_int8_fp16s_oom(M, N, K, 1, 0, 0, 1, 0, 0, 0)
           || test_gemm_int8_fp16s_oom(M, N, K, 0, 1, 3, 1, 1, 0, 0)
           || test_gemm_int8_fp16s_oom(M, N, K, 1, 1, 0, 0, 1, 0, 1)
           || test_gemm_int8_fp16s_oom(M, N, K, 0, 0, 2, 0, 0, 1, 0)
           || test_gemm_int8_fp16s_oom(M, N, K, 0, 1, 0, 1, 0, 1, 0)
           || test_gemm_int8_fp16s_oom(M, N, K, 1, 0, 0, 1, 1, 1, 0)
           || test_gemm_int8_fp16s_oom(M, N, K, 1, 1, 1, 0, 1, 1, 1);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

    int mnk[][3] = {
        {11, 12, 13},
        {1, 2, 3},
        {4, 1, 6},
        {9, 8, 7}
    };

    int mnk_count = sizeof(mnk) / sizeof(int) / 3;

    for (int i = 0; i < mnk_count; i++)
    {
        int M = mnk[i][0];
        int N = mnk[i][1];
        int K = mnk[i][2];

        int ret = test_gemm_0(M, N, K) || test_gemm_1(M, N, K);
        if (ret != 0)
            return ret;

#if NCNN_INT8
        int ret2 = test_gemm_2(M, N, K) || test_gemm_3(M, N, K) || test_gemm_4(M, N, K);
        if (ret2 != 0)
            return ret2;
#endif
    }

    return 0;
}
