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
static int test_gemm_ep(int M, int N, int K, int output_elempack, int output_transpose, int output_N1M = 0, int flag = 0)
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

        int ret = test_layer_opt("Gemm", pd, weights, opt, a, 1, 0.001, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_gemm_ep failed M=%d N=%d K=%d output_elempack=%d output_transpose=%d output_N1M=%d\n", M, N, K, output_elempack, output_transpose, output_N1M);
            return ret;
        }
    }

    return 0;
}

// Test with forced output_elempack, bf16 paths only.
static int test_gemm_ep_bf16(int M, int N, int K, int output_elempack, int output_transpose, int flag = 0)
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

        int ret = test_layer_opt("Gemm", pd, weights, opt, a, 1, 0.001, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_gemm_ep_bf16 failed M=%d N=%d K=%d output_elempack=%d output_transpose=%d\n", M, N, K, output_elempack, output_transpose);
            return ret;
        }
    }

    return 0;
}

static int test_gemm_1(int M, int N, int K, int fp32_min_elempack, int fp32_max_elempack, int bf16_min_elempack, int bf16_max_elempack, int ep, int flag = 0)
{
    for (int output_transpose = 0; output_transpose < 2; output_transpose++)
    {
        int outh = output_transpose ? N : M;
        if (outh % ep != 0)
            continue;

        // fp32 path
        if (ep == 1 || (ep <= fp32_max_elempack && ep % fp32_min_elempack == 0))
        {
            for (int output_N1M = 0; output_N1M < 2; output_N1M++)
            {
                int ret = test_gemm_ep(M, N, K, ep, output_transpose, output_N1M, flag);
                if (ret != 0)
                    return ret;
            }
        }

        // bf16 path (only when bf16 supports larger elempack than fp32)
        if ((ep == 1 || (ep <= bf16_max_elempack && ep % bf16_min_elempack == 0)) && ep > fp32_max_elempack)
        {
            int ret = test_gemm_ep_bf16(M, N, K, ep, output_transpose, flag);
            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

static int test_gemm_1(int M, int N, int K, int fp32_min_elempack, int fp32_max_elempack, int bf16_min_elempack, int bf16_max_elempack)
{
    return 0
           || test_gemm_1(M, N, K, fp32_min_elempack, fp32_max_elempack, bf16_min_elempack, bf16_max_elempack, 1)
           || test_gemm_1(M, N, K, fp32_min_elempack, fp32_max_elempack, bf16_min_elempack, bf16_max_elempack, 4)
           || test_gemm_1(M, N, K, fp32_min_elempack, fp32_max_elempack, bf16_min_elempack, bf16_max_elempack, 8, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_1(M, N, K, fp32_min_elempack, fp32_max_elempack, bf16_min_elempack, bf16_max_elempack, 16, TEST_LAYER_DISABLE_GPU_TESTING);
}

#if NCNN_BF16
static int test_gemm_bf16_storage(int fp32_packn, int packn, int use_packing_layout, int output_elemtype, int output_transpose, int output_N1M, int fp32_C = 0)
{
    ncnn::Layer* op = ncnn::create_layer_cpu("Gemm");
    const bool supported = op->support_bf16_storage && (!use_packing_layout || op->support_packing);
    delete op;
    if (!supported)
        return 0;

    const int M = packn;
    const int N = packn * 2;
    const int K = packn * 2 + 3;

    ncnn::ParamDict pd;
    pd.set(0, 1.5f);
    pd.set(1, 0.25f);
    pd.set(3, 1); // transB
    pd.set(11, output_N1M);
    pd.set(13, output_elemtype);
    pd.set(14, output_transpose);
    pd.set(20, packn);
    pd.set(21, 8);
    pd.set(22, packn);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = use_packing_layout;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = true;

    // exact bf16 values keep the fp32 reference independent of Cast and its tolerance
    std::vector<ncnn::Mat> a(3);
    a[0].create(K, M, (size_t)2u);
    a[1].create(K, N, (size_t)2u);
    a[2].create(N, M, fp32_C ? (size_t)4u : (size_t)2u);
    for (int i = 0; i < M; i++)
    {
        unsigned short* p = a[0].row<unsigned short>(i);
        for (int k = 0; k < K; k++)
            p[k] = ncnn::float32_to_bfloat16(((i * 3 + k) % 11 - 5) * 0.125f);
    }
    for (int j = 0; j < N; j++)
    {
        unsigned short* p = a[1].row<unsigned short>(j);
        for (int k = 0; k < K; k++)
            p[k] = ncnn::float32_to_bfloat16(((k + j * 2) % 13 - 6) * 0.0625f);
    }
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            const float v = ((i - j) % 7) * 0.25f;
            if (fp32_C)
                a[2].row(i)[j] = v;
            else
                a[2].row<unsigned short>(i)[j] = ncnn::float32_to_bfloat16(v);
        }
    }
    if (use_packing_layout)
    {
        for (int i = 0; i < 3; i++)
        {
            const int elempack = i == 2 && fp32_C ? fp32_packn : packn;
            ncnn::Mat packed;
            ncnn::convert_packing(a[i], packed, elempack, opt);
            if (packed.empty() || packed.elempack != elempack)
                return -1;
            a[i] = packed;
        }
    }

    op = ncnn::create_layer_cpu("Gemm");

    std::vector<ncnn::Mat> weights;
    ncnn::ModelBinFromMatArray mb(weights.data());
    std::vector<ncnn::Mat> b(1);
    int ret = op->load_param(pd);
    if (ret == 0) ret = op->load_model(mb);
    if (ret == 0) ret = op->create_pipeline(opt);
    if (ret == 0) ret = op->forward(a, b, opt);
    op->destroy_pipeline(opt);
    delete op;
    if (ret != 0)
        return ret;

    const int out_elempack = use_packing_layout ? (output_elemtype == 1 ? fp32_packn : packn) : 1;
    const int outw = output_transpose ? M : N;
    const int outh = output_transpose ? N : M;
    if (b[0].elembits() != (output_elemtype == 1 ? 32 : 16) || b[0].elempack != out_elempack
            || b[0].dims != (output_N1M ? 3 : 2) || b[0].w != outw
            || (output_N1M ? b[0].c : b[0].h) * out_elempack != outh)
    {
        fprintf(stderr, "test_gemm_bf16_storage output layout failed pack=%d output_elemtype=%d output_transpose=%d output_N1M=%d fp32_C=%d\n", use_packing_layout, output_elemtype, output_transpose, output_N1M, fp32_C);
        return -1;
    }

    ncnn::Mat unpacked;
    ncnn::convert_packing(b[0], unpacked, 1, opt);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // ncnn Gemm applies alpha to both AB and beta * C
            float expected = 0.25f * (((i - j) % 7) * 0.25f);
            for (int k = 0; k < K; k++)
                expected += (((i * 3 + k) % 11 - 5) * 0.125f) * (((k + j * 2) % 13 - 6) * 0.0625f);
            expected *= 1.5f;
            float expected_rne = expected;
            if (output_elemtype != 1)
            {
                // native RNE and ncnn rounding may choose different halfway results
                unsigned int bits;
                memcpy(&bits, &expected, sizeof(bits));
                expected = ncnn::bfloat16_to_float32((unsigned short)((bits + 0x8000u) >> 16));
                expected_rne = expected;
                if ((bits & 0xffffu) == 0x8000u)
                    expected_rne = ncnn::bfloat16_to_float32((unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1)) >> 16));
            }

            const int row = output_transpose ? j : i;
            const int col = output_transpose ? i : j;
            const ncnn::Mat outrow = output_N1M ? unpacked.channel(row) : unpacked.row_range(row, 1);
            const float actual = output_elemtype == 1 ? ((const float*)outrow)[col] : ncnn::bfloat16_to_float32(((const unsigned short*)outrow)[col]);
            if (actual != expected && actual != expected_rne)
            {
                fprintf(stderr, "test_gemm_bf16_storage value failed i=%d j=%d expected=%f actual=%f pack=%d output_elemtype=%d output_transpose=%d output_N1M=%d fp32_C=%d\n", i, j, expected, actual, use_packing_layout, output_elemtype, output_transpose, output_N1M, fp32_C);
                return -1;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

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
    {
        fp32_min_elempack = ncnn::cpu_riscv_vlenb() / 4;
        fp32_max_elempack = ncnn::cpu_riscv_vlenb() / 4;
    }
#endif

#if NCNN_MSA
    if (ncnn::cpu_support_mips_msa())
    {
        fp32_min_elempack = 4;
        fp32_max_elempack = 4;
    }
#endif
#if NCNN_LSX
    if (ncnn::cpu_support_loongarch_lsx())
    {
        fp32_min_elempack = 4;
        fp32_max_elempack = 4;
    }
#endif
#if NCNN_LASX
    if (ncnn::cpu_support_loongarch_lasx())
    {
        fp32_min_elempack = 4;
        fp32_max_elempack = 8;
    }
#endif

    int bf16_min_elempack = fp32_min_elempack;
    int bf16_max_elempack = fp32_max_elempack;

#if NCNN_BF16 && (NCNN_RVV || NCNN_XTHEADVECTOR)
    if (ncnn::cpu_support_riscv_v() || ncnn::cpu_support_riscv_xtheadvector())
    {
        bf16_min_elempack = ncnn::cpu_riscv_vlenb() / 2;
        bf16_max_elempack = ncnn::cpu_riscv_vlenb() / 2;
    }
#endif

#if NCNN_MSA
    if (ncnn::cpu_support_mips_msa())
        bf16_max_elempack = 8;
#endif
#if NCNN_LSX
    if (ncnn::cpu_support_loongarch_lsx())
        bf16_max_elempack = 8;
#endif

#if NCNN_BF16
    for (int packing = 0; packing < 2; packing++)
        for (int output_elemtype = 0; output_elemtype < 2; output_elemtype++)
            for (int output_transpose = 0; output_transpose < 2; output_transpose++)
                for (int output_N1M = 0; output_N1M < 2; output_N1M++)
                    if (test_gemm_bf16_storage(fp32_max_elempack, bf16_max_elempack, packing, output_elemtype, output_transpose, output_N1M) != 0)
                        return -1;

#if NCNN_RVV || NCNN_XTHEADVECTOR || NCNN_MSA || NCNN_LSX || NCNN_LASX
    // these backends also accept dynamic fp32 C alongside bf16 A and B
    if (test_gemm_bf16_storage(fp32_max_elempack, bf16_max_elempack, 1, 1, 0, 0, 1) != 0
            || test_gemm_bf16_storage(fp32_max_elempack, bf16_max_elempack, 1, 1, 1, 1, 1) != 0)
        return -1;
#endif
#endif

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
