// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gemm(int M, int N, int K, int output_transpose, int output_N1M = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, 1.f); // alpha
    pd.set(1, 1.f); // beta
    pd.set(2, 0);   // transA
    pd.set(3, 0);   // transB
    pd.set(4, 0);   // constantA
    pd.set(5, 0);   // constantB
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(11, output_N1M);
    pd.set(13, 1); // output_elemtype = fp32
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;

    std::vector<ncnn::Mat> a;
    a.push_back(output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M));
    a.push_back(output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K));

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i]);
    }

    int ret = test_layer("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm failed M=%d N=%d K=%d output_transpose=%d output_N1M=%d\n", M, N, K, output_transpose, output_N1M);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K)
{
    return 0
           || test_gemm(M, N, K, 0, 0)
           || test_gemm(M, N, K, 0, 1)
           || test_gemm(M, N, K, 1, 0)
           || test_gemm(M, N, K, 1, 1);
}

// Test with forced output_elempack, fp32 paths only.
// Uses test_layer_opt to avoid running bf16 paths where this elempack may be invalid.
static int test_gemm_ep(int M, int N, int K, int output_elempack, int output_transpose, int output_N1M = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, 1.f); // alpha
    pd.set(1, 1.f); // beta
    pd.set(2, 0);   // transA
    pd.set(3, 0);   // transB
    pd.set(4, 0);   // constantA
    pd.set(5, 0);   // constantB
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(11, output_N1M);
    pd.set(12, output_elempack);
    pd.set(13, 1); // output_elemtype = fp32
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;

    std::vector<ncnn::Mat> a;
    a.push_back(output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M));
    a.push_back(output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K));

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i]);
    }

    // Only run fp32-safe option combos (use_bf16_storage=0)
    // pack fp16p fp16s fp16a bf16
    const int options[][2] = {
        {0, 0},
        {1, 0},
        {1, 1},
    };

    for (int i = 0; i < 3; i++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = options[i][0];
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = options[i][1];
        opt.use_bf16_storage = options[i][1];

        int ret = test_layer_opt("Gemm", pd, weights, opt, a, 1, 0.001, TEST_LAYER_DISABLE_GPU_TESTING);
        if (ret != 0)
        {
            fprintf(stderr, "test_gemm_ep failed M=%d N=%d K=%d output_elempack=%d output_transpose=%d output_N1M=%d\n", M, N, K, output_elempack, output_transpose, output_N1M);
            return ret;
        }
    }

    return 0;
}

// Test with forced output_elempack, bf16 paths only.
// Uses test_layer_opt to avoid running fp32 paths where this elempack may be invalid.
static int test_gemm_ep_bf16(int M, int N, int K, int output_elempack, int output_transpose)
{
    ncnn::ParamDict pd;
    pd.set(0, 1.f); // alpha
    pd.set(1, 1.f); // beta
    pd.set(2, 0);   // transA
    pd.set(3, 0);   // transB
    pd.set(4, 0);   // constantA
    pd.set(5, 0);   // constantB
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(12, output_elempack);
    // pd.set(13, ...); // output_elemtype = 0 (bf16, default)
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;

    std::vector<ncnn::Mat> a;
    a.push_back(ncnn::Mat(K, M));
    a.push_back(ncnn::Mat(N, K));

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i]);
    }

    // Only run bf16 option combos
    // pack bf16
    const int options[][2] = {
        {0, 1},
        {1, 1},
    };

    for (int i = 0; i < 2; i++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = options[i][0];
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = options[i][1];
        opt.use_bf16_storage = options[i][1];

        int ret = test_layer_opt("Gemm", pd, weights, opt, a, 1, 0.001, TEST_LAYER_DISABLE_GPU_TESTING);
        if (ret != 0)
        {
            fprintf(stderr, "test_gemm_ep_bf16 failed M=%d N=%d K=%d output_elempack=%d output_transpose=%d\n", M, N, K, output_elempack, output_transpose);
            return ret;
        }
    }

    return 0;
}

static int test_gemm_1(int M, int N, int K, int fp32_min_elempack, int fp32_max_elempack, int bf16_min_elempack, int bf16_max_elempack)
{
    const int elempacks[] = {1, 4, 8, 16};

    for (int ei = 0; ei < 4; ei++)
    {
        int ep = elempacks[ei];

        for (int output_transpose = 0; output_transpose < 2; output_transpose++)
        {
            int outh = output_transpose ? N : M;
            if (outh % ep != 0)
                continue;

            // fp32 path
            if (ep <= fp32_max_elempack && ep % fp32_min_elempack == 0)
            {
                for (int output_N1M = 0; output_N1M < 2; output_N1M++)
                {
                    int ret = test_gemm_ep(M, N, K, ep, output_transpose, output_N1M);
                    if (ret != 0)
                        return ret;
                }
            }

            // bf16 path (only when bf16 supports larger elempack than fp32)
            if (ep <= bf16_max_elempack && ep % fp32_min_elempack == 0 && ep > fp32_max_elempack)
            {
                int ret = test_gemm_ep_bf16(M, N, K, ep, output_transpose);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    int mnk_scalar[][3] = {
        {1, 1, 1},
        {2, 2, 2},
        {3, 3, 3},
        {4, 4, 4},
        {5, 5, 5},
        {7, 7, 7},
        {8, 8, 8},
        {15, 15, 15},
        {16, 16, 16},
        {24, 24, 24},
        {31, 32, 31},
        {32, 32, 32},
        {47, 48, 47},
        {64, 64, 64},
    };

    for (int i = 0; i < 14; i++)
    {
        int ret = test_gemm_0(mnk_scalar[i][0], mnk_scalar[i][1], mnk_scalar[i][2]);
        if (ret != 0)
            return ret;
    }

    int fp32_min_elempack = 1;
    int fp32_max_elempack = 1;
#if __SSE2__ || __ARM_NEON
    fp32_min_elempack = 4;
    fp32_max_elempack = 4;
#endif

#if NCNN_AVX
    if (ncnn::cpu_support_x86_avx())
        fp32_max_elempack = 8;
#if NCNN_AVX512
    if (ncnn::cpu_support_x86_avx512())
        fp32_max_elempack = 16;
#endif
#endif

#if NCNN_RVV || NCNN_XTHEADVECTOR
    if (ncnn::cpu_support_riscv_v() || ncnn::cpu_support_riscv_xtheadvector())
        fp32_max_elempack = ncnn::cpu_riscv_vlenb() / 4;
#endif

    int bf16_min_elempack = fp32_min_elempack;
    int bf16_max_elempack = fp32_max_elempack;

    for (int i = 0; i < 14; i++)
    {
        int ret = test_gemm_1(mnk_scalar[i][0], mnk_scalar[i][1], mnk_scalar[i][2], fp32_min_elempack, fp32_max_elempack, bf16_min_elempack, bf16_max_elempack);
        if (ret != 0)
            return ret;
    }

    // Asymmetric M/N to cover output_transpose paths with various ii/jj
    // remainder blocks. In unpack_output_tile_fp32_to_bf16:
    //   ii iterates M dimension, jj iterates N dimension.
    int mnk_asym[][3] = {
        {1, 16, 4},
        {2, 16, 4},
        {3, 16, 4},
        {5, 32, 4},
        {1, 8, 4},
        {2, 8, 4},
        {3, 8, 4},
        {5, 8, 4},
        {16, 1, 4},
        {16, 3, 4},
        {16, 5, 4},
        {17, 16, 4},
        {33, 17, 4},
        {8, 4, 4},
        {8, 2, 4},
        {8, 1, 4},
        {4, 2, 4},
        {4, 1, 4},
        {2, 32, 4},
        {16, 8, 4},
        {4, 16, 4},
        {4, 8, 4},
    };

    int num_asym = sizeof(mnk_asym) / sizeof(mnk_asym[0]);
    for (int i = 0; i < num_asym; i++)
    {
        int ret = test_gemm_1(mnk_asym[i][0], mnk_asym[i][1], mnk_asym[i][2], fp32_min_elempack, fp32_max_elempack, bf16_min_elempack, bf16_max_elempack);
        if (ret != 0)
            return ret;
    }

    if (bf16_max_elempack >= 4 && 4 % bf16_min_elempack == 0)
    {
        // bf16 output (output_elemtype=0) with out_elempack=4, output_transpose=1
        // to cover the bf16 store paths in unpack_output_tile_fp32_to_bf16
        int ret = 0
                  || test_gemm_ep_bf16(4, 16, 4, 4, 1)
                  || test_gemm_ep_bf16(4, 8, 4, 4, 1)
                  || test_gemm_ep_bf16(2, 16, 4, 4, 1)
                  || test_gemm_ep_bf16(1, 16, 4, 4, 1)
                  || test_gemm_ep_bf16(3, 16, 4, 4, 1)
                  || test_gemm_ep_bf16(8, 16, 4, 4, 1);
        if (ret != 0)
            return ret;
    }

    return 0;
}
