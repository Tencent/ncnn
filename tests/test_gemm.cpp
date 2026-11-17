// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <math.h>

static int test_gemm_bf16_coopmat(int M, int N, int K, int transA, int transB, int output_transpose, int constantA, int constantB, int constantC, int broadcast_type_C, int data_type)
{
    ncnn::Mat A = transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    ncnn::Mat B = transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K);

    // All inputs are exactly representable in bf16. The ordinary cases require
    // scaling either A or B into fp16 while their products stay near one.
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {
            float v;
            if (data_type == 2)
            {
                // Neither row fits in fp16 with one scale. Both reciprocal
                // products contribute one; dropping small inputs gives zero.
                v = k == 0 ? ldexpf(1.f, 40) : k == 1 ? ldexpf(1.f, -40) : 0.f;
            }
            else if (data_type == 3)
            {
                // AB alone overflows, but accumulating from C stays finite.
                v = k < 2 ? ldexpf(1.f, 127) : 0.f;
            }
            else
            {
                v = ldexpf((128 + (i * 11 + k * 3) % 64) / 128.f, (data_type == 0 ? 20 : -20) + i % 5 - 2);
            }
            if (i == M - 1) v = 0.f;

            if (transA)
                A.row(k)[i] = v;
            else
                A.row(i)[k] = v;
        }
    }

    for (int j = 0; j < N; j++)
    {
        for (int k = 0; k < K; k++)
        {
            float v;
            if (data_type == 2)
            {
                v = k == 0 ? ldexpf(1.f, -40) : k == 1 ? ldexpf(1.f, 40) : 0.f;
            }
            else if (data_type == 3)
            {
                v = k < 2 ? 1.f : 0.f;
            }
            else
            {
                v = ldexpf((128 + (j * 13 + k * 5) % 64) / 128.f, (data_type == 0 ? -20 : 20) + j % 5 - 2);
                if (data_type == 1) v = -v;
            }
            if (j == N - 1) v = 0.f;

            if (transB)
                B.row(j)[k] = v;
            else
                B.row(k)[j] = v;
        }
    }

    ncnn::Mat C;
    if (broadcast_type_C == 0) C.create(1);
    if (broadcast_type_C == 1) C.create(M);
    if (broadcast_type_C == 2) C.create(1, M);
    if (broadcast_type_C == 3) C.create(N, M);
    if (broadcast_type_C == 4) C.create(N, 1);
    for (int i = 0; i < C.w * C.h; i++)
    {
        ((float*)C)[i] = data_type == 3 ? -ldexpf(1.f, 127) : (data_type == 1 ? -1.f : 1.f) * (8 + i % 13);
    }

    ncnn::ParamDict pd;
    pd.set(0, data_type >= 2 ? 1.f : 1.5f);
    pd.set(1, data_type == 3 ? 1.f : 0.75f);
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
    if (constantA) weights.push_back(A);
    if (constantB) weights.push_back(B);
    if (constantC && !C.empty()) weights.push_back(C);

    std::vector<ncnn::Mat> a;
    if (!constantA) a.push_back(A);
    if (!constantB) a.push_back(B);
    if (!constantC && !C.empty()) a.push_back(C);

    ncnn::Option opt;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_uniform = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = true;
    opt.use_bf16_storage = false;
    opt.use_cooperative_matrix = true;

    // Bypass test_layer_opt's widened bf16 tolerance. The reciprocal case has
    // exact zero/two outputs, and the other cases only need final bf16 rounding.
    const float epsilon = data_type >= 2 ? 0.000001f : 0.008f;
    int ret = 0;
    if (data_type == 3)
    {
#if NCNN_VULKAN
        // The generic reference computes AB before adding C. Check the existing
        // Vulkan C-first result directly, which stays finite for these inputs.
        std::vector<ncnn::Mat> expected(1);
        expected[0].create(output_transpose ? M : N, output_transpose ? N : M);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                const float v = ldexpf(i == M - 1 || j == N - 1 ? -1.f : 1.f, 127);
                if (output_transpose)
                    expected[0].row(j)[i] = v;
                else
                    expected[0].row(i)[j] = v;
            }
        }

        std::vector<ncnn::Mat> actual;
        ret = test_layer_gpu(ncnn::LayerType::Gemm, pd, weights, opt, a, 1, actual, expected, 0);
        if (ret == 233)
            ret = 0;
        else if (ret == 0)
            ret = CompareMat(expected, actual, epsilon);
#endif // NCNN_VULKAN
    }
    else
    {
        ret = test_layer(ncnn::LayerType::Gemm, pd, weights, opt, a, 1, std::vector<ncnn::Mat>(), epsilon);
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_bf16_coopmat failed M=%d N=%d K=%d transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d constantC=%d broadcast_type_C=%d data_type=%d\n", M, N, K, transA, transB, output_transpose, constantA, constantB, constantC, broadcast_type_C, data_type);
    }

    return ret;
}

static int test_gemm_bf16_coopmat()
{
    return 0
           || test_gemm_bf16_coopmat(16, 20, 17, 0, 0, 0, 0, 0, 0, 0, 0)
           || test_gemm_bf16_coopmat(13, 11, 19, 0, 1, 0, 1, 0, 1, 1, 0)
           || test_gemm_bf16_coopmat(12, 9, 20, 1, 0, 1, 0, 1, 0, 2, 1)
           || test_gemm_bf16_coopmat(17, 12, 23, 1, 1, 0, 1, 1, 0, 3, 0)
           || test_gemm_bf16_coopmat(20, 13, 21, 0, 1, 1, 0, 1, 1, 4, 1)
           || test_gemm_bf16_coopmat(9, 16, 13, 1, 0, 0, 1, 0, 0, 3, 1)
           || test_gemm_bf16_coopmat(16, 12, 19, 0, 1, 0, 0, 0, 1, -1, 2)
           || test_gemm_bf16_coopmat(13, 11, 17, 1, 0, 1, 1, 0, 1, -1, 2)
           || test_gemm_bf16_coopmat(12, 9, 20, 0, 0, 0, 0, 1, 1, -1, 2)
           || test_gemm_bf16_coopmat(16, 12, 19, 0, 1, 0, 1, 0, 1, 0, 3);
}

#if NCNN_VALIDATION
static int test_gemm_load_param()
{
    ncnn::ParamDict base;
    base.set(4, 1);
    base.set(7, 8);
    base.set(9, 8);
    if (test_layer_param(ncnn::LayerType::Gemm, base, 0) != 0)
        return -1;

    return test_layer_param(ncnn::LayerType::Gemm, base, 7, -1, -1);
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_gemm_bf16_coopmat()
#if NCNN_VALIDATION
           || test_gemm_load_param()
#endif // NCNN_VALIDATION
           ;
}
