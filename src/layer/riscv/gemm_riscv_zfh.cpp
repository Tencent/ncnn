// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_riscv.h"
#include <cstdio>
#include <cstdlib>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

#include "cpu.h"
#include <stdlib.h>

namespace ncnn {

#if NCNN_ZFH
#include "gemm_bf16s_fp16s.h"
#include "gemm_fp16s.h"
#include "gemm_fp16sa.h"
#if NCNN_WEIGHT_QUANT
#include "gemm_wq_int8.h"
#include "gemm_wq_int8_fp16s.h"

void Gemm_riscv::quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    ncnn::quantize_A_tile_wq_int8_fp16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void Gemm_riscv::transpose_quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    ncnn::transpose_quantize_A_tile_wq_int8_fp16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void Gemm_riscv::unpack_output_tile_wq_int8_fp16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose)
{
    ncnn::unpack_output_tile_wq_int8(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, 2, output_transpose);
}
#endif // NCNN_WEIGHT_QUANT

static int gemm_riscv_fp16s(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16s_fp16s(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    // pack B
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        if (transB)
        {
            pack_B_tile_bf16_fp16(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile_bf16_fp16(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    if (nT > nn_M)
    {
        Mat AT(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, opt.workspace_allocator);
        if (AT.empty())
            return -100;

        const int nn_MK = nn_M * nn_K;

        // pack A
        #pragma omp parallel for num_threads(nT)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            if (transA)
            {
                transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
            }
        }

        const int nn_MN = nn_M * nn_N;

        #pragma omp parallel for num_threads(nT)
        for (int ppij = 0; ppij < nn_MN; ppij++)
        {
            const int ppi = ppij / nn_N;
            const int ppj = ppij % nn_N;

            const int i = ppi * TILE_M;
            const int j = ppj * TILE_N;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_jj = std::min((N - j), TILE_N);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            if (broadcast_type_C == 3)
            {
                pack_A_tile_fp32(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;
            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;
                float _alpha = k + TILE_K >= K ? alpha : 1.f;

                gemm_transB_packed_tile_fp16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_fp32_to_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }
    else
    {
        Mat ATX(TILE_K * TILE_M, nn_K, nT, 2u, opt.workspace_allocator);
        if (ATX.empty())
            return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppi = 0; ppi < nn_M; ppi++)
        {
            const int i = ppi * TILE_M;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            for (int j = 0; j < N; j += TILE_N)
            {
                const int max_jj = std::min((N - j), TILE_N);
                if (broadcast_type_C == 3)
                {
                    pack_A_tile_fp32(C, topT_tile, i, max_ii, j, max_jj);
                }

                const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;
                for (int k = 0; k < K; k += TILE_K)
                {
                    const int max_kk = std::min((K - k), TILE_K);

                    // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                    Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);

                    Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                    if (j == 0)
                    {
                        if (transA)
                        {
                            transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                        }
                        else
                        {
                            pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                        }
                    }

                    bool k_end = !output_transpose && k + TILE_K >= K;
                    float _alpha = k + TILE_K >= K ? alpha : 1.f;

                    gemm_transB_packed_tile_fp16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
                }

                if (output_transpose)
                {
                    transpose_unpack_output_tile_fp32_to_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
                }
            }
        }
    }

    return 0;
}

static int gemm_AT_riscv_fp16s(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16s_fp16s(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    // pack B
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        if (transB)
        {
            pack_B_tile_bf16_fp16(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile_bf16_fp16(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const int nn_MN = nn_M * nn_N;
    #pragma omp parallel for num_threads(nT)
    for (int ppij = 0; ppij < nn_MN; ppij++)
    {
        const int ppi = ppij / nn_N;
        const int ppj = ppij % nn_N;

        const int i = ppi * TILE_M;
        const int j = ppj * TILE_N;

        const int max_ii = std::min((M - i), TILE_M);
        const int max_jj = std::min((N - j), TILE_N);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        if (broadcast_type_C == 3)
        {
            pack_A_tile_fp32(C, topT_tile, i, max_ii, j, max_jj);
        }

        const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

            bool k_end = !output_transpose && k + TILE_K >= K;
            float _alpha = k + TILE_K >= K ? alpha : 1.f;

            gemm_transB_packed_tile_fp16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
        }

        if (output_transpose)
        {
            transpose_unpack_output_tile_fp32_to_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_BT_riscv_fp16s(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16s_fp16s(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    if (nT > nn_M)
    {
        Mat AT(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, opt.workspace_allocator);
        if (AT.empty())
            return -100;

        const int nn_MK = nn_M * nn_K;

        // pack A
        #pragma omp parallel for num_threads(nT)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            if (transA)
            {
                transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
            }
        }

        const int nn_MN = nn_M * nn_N;

        #pragma omp parallel for num_threads(nT)
        for (int ppij = 0; ppij < nn_MN; ppij++)
        {
            const int ppi = ppij / nn_N;
            const int ppj = ppij % nn_N;

            const int i = ppi * TILE_M;
            const int j = ppj * TILE_N;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_jj = std::min((N - j), TILE_N);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            if (broadcast_type_C == 3)
            {
                pack_A_tile_fp32(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;
                float _alpha = k + TILE_K >= K ? alpha : 1.f;

                gemm_transB_packed_tile_fp16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_fp32_to_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }
    else
    {
        Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 2u, opt.workspace_allocator);
        if (ATX.empty())
            return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppi = 0; ppi < nn_M; ppi++)
        {
            const int i = ppi * TILE_M;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            for (int j = 0; j < N; j += TILE_N)
            {
                const int max_jj = std::min((N - j), TILE_N);

                if (broadcast_type_C == 3)
                {
                    pack_A_tile_fp32(C, topT_tile, i, max_ii, j, max_jj);
                }

                const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

                for (int k = 0; k < K; k += TILE_K)
                {
                    const int max_kk = std::min((K - k), TILE_K);

                    // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                    Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);

                    Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                    if (j == 0)
                    {
                        if (transA)
                        {
                            transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                        }
                        else
                        {
                            pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                        }
                    }

                    bool k_end = !output_transpose && k + TILE_K >= K;
                    float _alpha = k + TILE_K >= K ? alpha : 1.f;

                    gemm_transB_packed_tile_fp16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
                }

                if (output_transpose)
                {
                    transpose_unpack_output_tile_fp32_to_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
                }
            }
        }
    }

    return 0;
}

static int gemm_AT_BT_riscv_fp16s(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16s_fp16s(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const int nn_MN = nn_M * nn_N;
    #pragma omp parallel for num_threads(nT)
    for (int ppij = 0; ppij < nn_MN; ppij++)
    {
        const int ppi = ppij / nn_N;
        const int ppj = ppij % nn_N;

        const int i = ppi * TILE_M;
        const int j = ppj * TILE_N;

        const int max_ii = std::min((M - i), TILE_M);
        const int max_jj = std::min((N - j), TILE_N);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        if (broadcast_type_C == 3)
        {
            pack_A_tile_fp32(C, topT_tile, i, max_ii, j, max_jj);
        }

        const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

            bool k_end = !output_transpose && k + TILE_K >= K;
            float _alpha = k + TILE_K >= K ? alpha : 1.f;

            gemm_transB_packed_tile_fp16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
        }

        if (output_transpose)
        {
            transpose_unpack_output_tile_fp32_to_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_riscv_fp16sa(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_fp16sa(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    // pack B
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        if (transB)
        {
            pack_B_tile_fp16sa(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile_fp16sa(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    if (nT > nn_M)
    {
        Mat AT(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, opt.workspace_allocator);
        if (AT.empty())
            return -100;

        const int nn_MK = nn_M * nn_K;

        // pack A
        #pragma omp parallel for num_threads(nT)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            if (transA)
            {
                transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
            }
        }

        const int nn_MN = nn_M * nn_N;

        #pragma omp parallel for num_threads(nT)
        for (int ppij = 0; ppij < nn_MN; ppij++)
        {
            const int ppi = ppij / nn_N;
            const int ppj = ppij % nn_N;

            const int i = ppi * TILE_M;
            const int j = ppj * TILE_N;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_jj = std::min((N - j), TILE_N);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            if (broadcast_type_C == 3)
            {
                pack_A_tile_bf16_fp16(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;
            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;
                float _alpha = k + TILE_K >= K ? alpha : 1.f;

                gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_fp16sa(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }
    else
    {
        Mat ATX(TILE_K * TILE_M, nn_K, nT, 2u, opt.workspace_allocator);
        if (ATX.empty())
            return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppi = 0; ppi < nn_M; ppi++)
        {
            const int i = ppi * TILE_M;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            for (int j = 0; j < N; j += TILE_N)
            {
                const int max_jj = std::min((N - j), TILE_N);
                if (broadcast_type_C == 3)
                {
                    pack_A_tile_bf16_fp16(C, topT_tile, i, max_ii, j, max_jj);
                }

                const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;
                for (int k = 0; k < K; k += TILE_K)
                {
                    const int max_kk = std::min((K - k), TILE_K);

                    // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                    Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);

                    Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                    if (j == 0)
                    {
                        if (transA)
                        {
                            transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                        }
                        else
                        {
                            pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                        }
                    }

                    bool k_end = !output_transpose && k + TILE_K >= K;
                    float _alpha = k + TILE_K >= K ? alpha : 1.f;

                    gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
                }

                if (output_transpose)
                {
                    transpose_unpack_output_tile_fp16sa(topT_tile, top_blob, i, max_ii, j, max_jj);
                }
            }
        }
    }

    return 0;
}

static int gemm_AT_riscv_fp16sa(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_fp16sa(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    // pack B
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        if (transB)
        {
            pack_B_tile_fp16sa(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile_fp16sa(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const int nn_MN = nn_M * nn_N;
    #pragma omp parallel for num_threads(nT)
    for (int ppij = 0; ppij < nn_MN; ppij++)
    {
        const int ppi = ppij / nn_N;
        const int ppj = ppij % nn_N;

        const int i = ppi * TILE_M;
        const int j = ppj * TILE_N;

        const int max_ii = std::min((M - i), TILE_M);
        const int max_jj = std::min((N - j), TILE_N);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        if (broadcast_type_C == 3)
        {
            pack_A_tile_bf16_fp16(C, topT_tile, i, max_ii, j, max_jj);
        }

        const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

            bool k_end = !output_transpose && k + TILE_K >= K;
            float _alpha = k + TILE_K >= K ? alpha : 1.f;

            gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
        }

        if (output_transpose)
        {
            transpose_unpack_output_tile_fp16sa(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_BT_riscv_fp16sa(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_fp16sa(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    if (nT > nn_M)
    {
        Mat AT(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, opt.workspace_allocator);
        if (AT.empty())
            return -100;

        const int nn_MK = nn_M * nn_K;

        // pack A
        #pragma omp parallel for num_threads(nT)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            if (transA)
            {
                transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
            }
        }

        const int nn_MN = nn_M * nn_N;

        #pragma omp parallel for num_threads(nT)
        for (int ppij = 0; ppij < nn_MN; ppij++)
        {
            const int ppi = ppij / nn_N;
            const int ppj = ppij % nn_N;

            const int i = ppi * TILE_M;
            const int j = ppj * TILE_N;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_jj = std::min((N - j), TILE_N);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            if (broadcast_type_C == 3)
            {
                pack_A_tile_bf16_fp16(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;
                float _alpha = k + TILE_K >= K ? alpha : 1.f;

                gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_fp16sa(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }
    else
    {
        Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 2u, opt.workspace_allocator);
        if (ATX.empty())
            return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppi = 0; ppi < nn_M; ppi++)
        {
            const int i = ppi * TILE_M;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            for (int j = 0; j < N; j += TILE_N)
            {
                const int max_jj = std::min((N - j), TILE_N);

                if (broadcast_type_C == 3)
                {
                    pack_A_tile_bf16_fp16(C, topT_tile, i, max_ii, j, max_jj);
                }

                const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

                for (int k = 0; k < K; k += TILE_K)
                {
                    const int max_kk = std::min((K - k), TILE_K);

                    // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                    Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);

                    Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                    if (j == 0)
                    {
                        if (transA)
                        {
                            transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                        }
                        else
                        {
                            pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                        }
                    }

                    bool k_end = !output_transpose && k + TILE_K >= K;
                    float _alpha = k + TILE_K >= K ? alpha : 1.f;

                    gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
                }

                if (output_transpose)
                {
                    transpose_unpack_output_tile_fp16sa(topT_tile, top_blob, i, max_ii, j, max_jj);
                }
            }
        }
    }

    return 0;
}

static int gemm_AT_BT_riscv_fp16sa(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_fp16sa(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const int nn_MN = nn_M * nn_N;
    #pragma omp parallel for num_threads(nT)
    for (int ppij = 0; ppij < nn_MN; ppij++)
    {
        const int ppi = ppij / nn_N;
        const int ppj = ppij % nn_N;

        const int i = ppi * TILE_M;
        const int j = ppj * TILE_N;

        const int max_ii = std::min((M - i), TILE_M);
        const int max_jj = std::min((N - j), TILE_N);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        if (broadcast_type_C == 3)
        {
            pack_A_tile_bf16_fp16(C, topT_tile, i, max_ii, j, max_jj);
        }

        const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

            bool k_end = !output_transpose && k + TILE_K >= K;
            float _alpha = k + TILE_K >= K ? alpha : 1.f;

            gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
        }

        if (output_transpose)
        {
            transpose_unpack_output_tile_fp16sa(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

int Gemm_riscv::create_pipeline_fp16sa(const Option& opt)
{
    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_fp16sa(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_M = (M + TILE_M - 1) / TILE_M;

        AT_data.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, (Allocator*)0);
        if (AT_data.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppj = 0; ppj < nn_M; ppj++)
        {
            const int i = ppj * TILE_M;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_ii = std::min((M - i), TILE_M);
                const int max_kk = std::min((K - k), TILE_K);

                Mat AT_tile = AT_data.channel(i / TILE_M).row_range(k / TILE_K, 1);

                if (transA)
                {
                    transpose_pack_A_tile_fp32_to_fp16(A_data, AT_tile, i, max_ii, k, max_kk);
                }
                else
                {
                    pack_A_tile_fp32_to_fp16(A_data, AT_tile, i, max_ii, k, max_kk);
                }
            }
        }

        if (opt.lightmode)
            A_data.release();
    }

    if (constantB)
    {
        const int N = constantN;
        const int K = constantK;

        Mat B_data_fp16;
        cast_float32_to_float16(B_data, B_data_fp16, opt);
        if (B_data_fp16.empty())
            return -100;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_fp16sa(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_N = (N + TILE_N - 1) / TILE_N;

        BT_data.create(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, (Allocator*)0);
        if (BT_data.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppj = 0; ppj < nn_N; ppj++)
        {
            const int j = ppj * TILE_N;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_jj = std::min((N - j), TILE_N);
                const int max_kk = std::min((K - k), TILE_K);

                Mat BT_tile = BT_data.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (transB)
                {
                    pack_B_tile_fp16sa(B_data_fp16, BT_tile, j, max_jj, k, max_kk);
                }
                else
                {
                    transpose_pack_B_tile_fp16sa(B_data_fp16, BT_tile, j, max_jj, k, max_kk);
                }
            }
        }

        if (opt.lightmode)
            B_data.release();
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        cast_float32_to_float16(C_data, CT_data, opt);
        if (CT_data.empty())
            return -100;

#if __riscv_vector
        const int packn = csrr_vlenb() / 2;

        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
            int C_elempack = constantM % packn == 0 ? packn : 1;
            Mat tmp;
            convert_packing(CT_data, tmp, C_elempack, opt);
            CT_data = tmp;
            if (CT_data.empty())
                return -100;
        }
#endif // __riscv_vector

        // pre-multiply C with beta
        if (beta != 1.f)
        {
            const int size = CT_data.total() * CT_data.elempack;
            __fp16* ptr = CT_data;
            for (int i = 0; i < size; i++)
            {
                ptr[i] *= (__fp16)beta;
            }
        }

        if (opt.lightmode)
            C_data.release();
    }

    if (constantA || constantB || constantC)
    {
        nT = opt.num_threads;
    }

    return 0;
}

int Gemm_riscv::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int M;
    int N;
    if (constantA && constantB)
    {
        M = constantM;
        N = constantN;
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        M = constantM;
        N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        N = constantN;
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;
    }

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = CT_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB)
        {
            C = bottom_blobs.size() == 1 ? bottom_blobs[0] : Mat();
        }
        else if (constantA)
        {
            C = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
        }
        else if (constantB)
        {
            C = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
        }
        else
        {
            C = bottom_blobs.size() == 3 ? bottom_blobs[2] : Mat();
        }

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w * C.elempack == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w * C.elempack == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h * C.elempack == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h * C.elempack == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h * C.elempack == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }

            if (beta != 1.f)
            {
                Mat CT_data;
                CT_data.create_like(C, opt.workspace_allocator);
                if (CT_data.empty())
                    return -100;

                const int size = C.total() * C.elempack;
                const __fp16* ptr = C;
                __fp16* outptr = CT_data;
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] * (__fp16)beta;
                }

                C = CT_data;
            }
        }
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector
    if (output_elempack)
        out_elempack = output_elempack;
    size_t out_elemsize = 2u * out_elempack;

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(M, N / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(N, M / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    int _nT = nT ? nT : opt.num_threads;
    if (nT != 0 && opt.num_threads != nT)
    {
        // force num_threads the same as in create_pipeline
        // so we could use pre-packed A/B from the same tile config
        NCNN_LOGE("opt.num_threads %d changed, gemm will use load-time value %d", opt.num_threads, nT);
    }

    int ret = 0;
    if (constantA && constantB)
    {
        ret = gemm_AT_BT_riscv_fp16sa(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_riscv_fp16sa(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_riscv_fp16sa(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_riscv_fp16sa(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    if (ret != 0)
        return ret;

    return 0;
}
int Gemm_riscv::create_pipeline_fp16s(const Option& opt)
{
    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_bf16s_fp16s(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_M = (M + TILE_M - 1) / TILE_M;

        AT_data.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, (Allocator*)0);
        if (AT_data.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppj = 0; ppj < nn_M; ppj++)
        {
            const int i = ppj * TILE_M;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_ii = std::min((M - i), TILE_M);
                const int max_kk = std::min((K - k), TILE_K);

                Mat AT_tile = AT_data.channel(i / TILE_M).row_range(k / TILE_K, 1);

                if (transA)
                {
                    transpose_pack_A_tile_fp32_to_fp16(A_data, AT_tile, i, max_ii, k, max_kk);
                }
                else
                {
                    pack_A_tile_fp32_to_fp16(A_data, AT_tile, i, max_ii, k, max_kk);
                }
            }
        }

        if (opt.lightmode)
            A_data.release();
    }

    if (constantB)
    {
        const int N = constantN;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_bf16s_fp16s(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_N = (N + TILE_N - 1) / TILE_N;

        BT_data.create(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, (Allocator*)0);
        if (BT_data.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppj = 0; ppj < nn_N; ppj++)
        {
            const int j = ppj * TILE_N;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_jj = std::min((N - j), TILE_N);
                const int max_kk = std::min((K - k), TILE_K);

                Mat BT_tile = BT_data.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (transB)
                {
                    pack_B_tile_fp32_to_fp16(B_data, BT_tile, j, max_jj, k, max_kk);
                }
                else
                {
                    transpose_pack_B_tile_fp32_to_fp16(B_data, BT_tile, j, max_jj, k, max_kk);
                }
            }
        }

        if (opt.lightmode)
            B_data.release();
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        CT_data = C_data;

#if __riscv_vector
        const int packn = csrr_vlenb() / 2;

        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
            int C_elempack = constantM % packn == 0 ? packn : 1;
            convert_packing(C_data, CT_data, C_elempack, opt);
            if (CT_data.empty())
                return -100;
        }
#endif // __riscv_vector

        // pre-multiply C with beta
        if (beta != 1.f)
        {
            Mat C2;
            C2.create_like(CT_data);
            if (C2.empty())
                return -100;

            const int size = CT_data.total() * CT_data.elempack;
            for (int i = 0; i < size; i++)
            {
                C2[i] = CT_data[i] * beta;
            }

            CT_data = C2;
        }

        if (opt.lightmode)
            C_data.release();
    }

    if (constantA || constantB || constantC)
    {
        nT = opt.num_threads;
    }

    return 0;
}

int Gemm_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int M;
    int N;
    if (constantA && constantB)
    {
        M = constantM;
        N = constantN;
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        M = constantM;
        N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        N = constantN;
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;
    }

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = CT_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB)
        {
            C = bottom_blobs.size() == 1 ? bottom_blobs[0] : Mat();
        }
        else if (constantA)
        {
            C = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
        }
        else if (constantB)
        {
            C = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
        }
        else
        {
            C = bottom_blobs.size() == 3 ? bottom_blobs[2] : Mat();
        }

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w * C.elempack == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w * C.elempack == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h * C.elempack == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h * C.elempack == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h * C.elempack == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }

            // cast to fp32
            {
                Mat CT_data;
                cast_float16_to_float32(C, CT_data);
                C = CT_data;
                if (C.empty())
                    return -100;
            }

            // pre-multiply C with beta
            if (beta != 1.f)
            {
                Mat CT_data;
                CT_data.create_like(C, opt.workspace_allocator);
                if (CT_data.empty())
                    return -100;

                const int size = C.total() * C.elempack;
                for (int i = 0; i < size; i++)
                {
                    CT_data[i] = C[i] * beta;
                }

                C = CT_data;
            }
        }
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector
    if (output_elempack)
        out_elempack = output_elempack;
    size_t out_elemsize = 2u * out_elempack;

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(M, N / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(N, M / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    int _nT = nT ? nT : opt.num_threads;
    if (nT != 0 && opt.num_threads != nT)
    {
        // force num_threads the same as in create_pipeline
        // so we could use pre-packed A/B from the same tile config
        NCNN_LOGE("opt.num_threads %d changed, gemm will use load-time value %d", opt.num_threads, nT);
    }

    int ret = 0;
    if (constantA && constantB)
    {
        ret = gemm_AT_BT_riscv_fp16s(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_riscv_fp16s(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_riscv_fp16s(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_riscv_fp16s(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }

    return ret;
}
#endif // NCNN_ZFH

// ================== SpacemiT K3 A100 IME2 (smt.vfwmadot) ==================
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
#if __riscv_v
#include "gemm_riscv_ime2.h"
#endif

int Gemm_riscv::create_pipeline_ime2(const Option& opt)
{
#if !__riscv_v
    (void)opt;
    return -99;
#else
    // 先探测：当前核能否执行 smt.vfwmadot（X100 上是 SIGILL）。
    // 探测失败 -> 返回非 0，调用方回退到 fp16s 路径。
    // 用与 forward 相同的团队大小探测工人：forward 的并行区都带
    // num_threads(opt.num_threads)，更大的团队会增派未探测过的工人。
    if (!ncnn_ime2::ime2_probe(opt.num_threads))
        return -99;

    /* 权重值域预检：|w| > 65504 的有限值打包成 ±inf 后整条 IME2 路径（含
     * VLEN 无关回落，它也读 tile）失真。不武装 IME2，返回 -99 让调用方建
     * 上游管线（fp16s/fp32 语义照旧）。 */
    if (!ncnn_ime2::ime2_weights_fit_fp16(B_data))
        return -99;

    // 把常量 B（[N,K] 权重）打包为 IME2 的 8x8 连续 tile 布局。
    // 线程数无关的布局：forward 阶段可以任意 num_threads。
    const int N = constantN;
    const int K = constantK;

    int ret = ncnn_ime2::ime2_pack_B(B_data, BT_data_ime2, N, K);
    if (ret != 0)
        return ret;

    /* int8 权重模式（可选精度模式，NCNN_IME2_INT8=1）：列主序 int8 面板 + 每列一个 fp16 scale，
     * 直接从原始权重量化。解码走整数域点积（vwadd 加宽 + vwmacc 累加），实测解码 +28~31%。
     * 第一版"tile 布局 int8 + 逐 8 元素转浮点"的实现已删除（微基准慢 600 倍，见 REPORT 7.40/7.41）。 */
    if (getenv("NCNN_IME2_INT8"))
    {
        ncnn_ime2::ime2_quantize_B_int8_col(B_data, BT_i8c_ime2, BT_wscale_ime2, N, K);
    }

    use_ime2_path = 1;
    nT = opt.num_threads;

    return 0;
#endif
}

int Gemm_riscv::forward_ime2_fallback(const Mat& A, int M, int N, int K, float alpha, int out_elempack, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if !__riscv_v
    (void)A;
    (void)M;
    (void)N;
    (void)K;
    (void)alpha;
    (void)out_elempack;
    (void)top_blobs;
    (void)opt;
    return -1;
#else

    const bool c_never_exists = (constantC && constant_broadcast_type_C == -1);
    if (!c_never_exists)
        return -99; /* 非 takeover：fp16s 管线在（fp16 输入由调度方落过去）；
                           * fp32 输入由调度方继续通用路径（与无 IME2 构建同语义） */

    /* full-takeover：唯一与 VLEN 无关的权重表示是 BT_data_ime2（固定 8x8
         * tile 布局，行距填充恒定）。朴素 fp32 累加直接从它算，任何核对上都
         * 正确。这是降级模式：先求正确，再求快。 */
    Option opt_tmp = opt;
    opt_tmp.blob_allocator = opt.workspace_allocator;

    Mat A32;
    if (A.elembits() == 16)
        cast_float16_to_float32(A, A32, opt_tmp);
    else
        A32 = A; /* fp32 共享，零拷贝 */
    if (A32.empty())
        return -100;
    /* 任何打包（fp16 packn 或 fp32 packn）对这里的逐行索引都非法：解到 pack1 */
    if (A32.elempack != 1)
    {
        Mat A32_p1;
        convert_packing(A32, A32_p1, 1, opt_tmp);
        if (A32_p1.empty())
            return -100;
        A32 = A32_p1;
    }

    /* BT_data_ime2 布局：tile (i,j) 在 (size_t)i*ldt + j*64，元素 (r,c) 在
         * t[r*8+c] = B[n=i*8+r][k=j*8+c]；ldt = kt*64 + IME2_TILE_PAD。 */
    const int kt = (K + 7) / 8;
    const size_t ldt = IME2_TILE_LD(kt);
    const __fp16* bt = (const __fp16*)BT_data_ime2;

    Mat C32(N, M, 4u, opt_tmp.blob_allocator); /* [w=N, h=M] fp32 pack1 */
    if (C32.empty())
        return -100;

    /* 每线程一份 K 行缓冲（B 行从 tile 布局解出），然后 RVV fp32 点积。 */
    Mat kbuf_mat(K, opt.num_threads, 4u, opt_tmp.blob_allocator);
    if (kbuf_mat.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int m = 0; m < M; m++)
    {
        const float* arow = A32.dims == 3 ? A32.channel(m).row(0) : (A32.dims == 2 ? A32.row(m) : (const float*)A32);
        float* crow = C32.row(m);
        float* kbuf = kbuf_mat.row(get_omp_thread_num());
        for (int n = 0; n < N; n++)
        {
            const __fp16* btile = bt + (size_t)(n / 8) * ldt + (n % 8) * 8;
            for (int k = 0; k < K; k++)
                kbuf[k] = (float)btile[(k / 8) * 64 + (k % 8)];

            float acc = 0.f;
#if __riscv_vector
            /* VLmax 全宽清零、VLmax 全宽归约（逐块 _tu 累加，末块余数 vl
                 * 归约会丢高位 lane 的贡献） */
            const size_t vlmax = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
            int k = 0;
            while (k < K)
            {
                const size_t vl = __riscv_vsetvl_e32m1(K - k);
                vacc = __riscv_vfmacc_vv_f32m1_tu(vacc, __riscv_vle32_v_f32m1(arow + k, vl), __riscv_vle32_v_f32m1(kbuf + k, vl), vl);
                k += vl;
            }
            acc = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vacc, __riscv_vfmv_s_f_f32m1(0.f, vlmax), vlmax));
#else
            for (int k = 0; k < K; k++)
                acc += arow[k] * kbuf[k];
#endif
            crow[n] = alpha * acc;
        }
    }

    Mat& top_blob = top_blobs[0];
    if (A.elembits() == 16)
    {
        /* fp16 输入 → 转回 fp16 存储并重打包到本层正常布局 */
        if (out_elempack == 1)
        {
            /* pack1：无需重打包，直接转出（blob 分配器），任何时刻只有一份。 */
            cast_float32_to_float16(C32, top_blob, opt);
            if (top_blob.empty())
                return -100;
            return 0;
        }
        /* 需要重打包时，pack1 的 fp16 中间量放 workspace（与最终输出同在
             * blob 池会瞬时占两份同尺寸；blob 池按一个返回输出估算会 -100）。 */
        Mat top16_tmp;
        cast_float32_to_float16(C32, top16_tmp, opt_tmp);
        if (top16_tmp.empty())
            return -100;
        convert_packing(top16_tmp, top_blob, out_elempack, opt);
        if (top_blob.empty())
            return -100;
        return 0;
    }
    /* fp32 输入 → fp32 输出（通用路径的输出布局：按其自身的打包决策；
         * C32 已是 pack1，重打包到 fp32 路径应有的 elempack）。 */
    int out32_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out32_elempack = (M % packn == 0) ? packn : 1;
    }
#endif
    /* output_elempack 是**元素数**而非字节数，通用 fp32 路径原样采用
         * （gemm_riscv.cpp 不折算）：此处同样保持，不能减半。 */
    if (output_elempack)
        out32_elempack = output_elempack;
    if (out32_elempack != 1)
    {
        /* 最终输出必须用 blob 分配器（C32 是 workspace 临时量，赋给
             * top_blob 会把工作区内存带出本层，后续层会覆写）。 */
        convert_packing(C32, top_blob, out32_elempack, opt);
        if (top_blob.empty())
            return -100;
        return 0;
    }
    /* pack1：同上，拷到 blob 分配器再交出。 */
    top_blob.create(N, M, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    memcpy(top_blob.data, C32.data, (size_t)N * M * 4u);
    return 0;
    return 0;
#endif
}

int Gemm_riscv::forward_ime2(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if !__riscv_v
    (void)bottom_blobs;
    (void)top_blobs;
    (void)opt;
    return -1;
#else
    const Mat& A = bottom_blobs[0];

    const int M = (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int N = constantN;
    const int K = constantK;

    /* 内核实际以 _nT（建管线时存的 nT 优先）起团队：校验必须覆盖它，
     * 而不是 forward 时的 opt.num_threads（调用方减线程后两者不同）。 */
    int _nT = nT ? nT : opt.num_threads;

    // 与 forward_fp16s 相同的输出打包决策（fp16: packn = vlenb/2）。
    // 在团队校验之前算好：fp16 回落分支与 IME2 主路径都用它。
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 2;
        out_elempack = (M % packn == 0) ? packn : 1;
    }
#endif
    if (output_elempack)
        out_elempack = output_elempack;

    /* 团队校验对所有位宽生效：fp32 输入也不例外——通用 fp32 面板的 tile 是建
     * 管线时按当时 vlenb 打包的，团队离开 A100 后通用路径会按新 vlenb 重算
     * tile 去索引旧面板 → 错位。所以先验团队，再按位宽分流。 */
    /* 队首校验：失败走 VLEN 无关回落（fp16/fp32 输入都接）。 */
    if (!ncnn_ime2::ime2_team_ok(_nT))
        return forward_ime2_fallback(A, M, N, K, alpha, out_elempack, top_blobs, opt);

    if (A.elembits() != 16)
        /* 团队在 A100 上、输入 fp32：dispatch 不应把 fp32 送到这里；防御。 */
        return -1;

    Mat& top_blob = top_blobs[0];
    top_blob.create(N, M / out_elempack, (size_t)2u * out_elempack, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    /* -99 = 内核执行区域内的队级校验失败（守卫与内核是两个 omp region，
     * 运行时可换工人）：与队首失败走同一条回落。top_blob 已建，回落会重建。 */
    const int ret = ncnn_ime2::gemm_ime2_fp16(A, BT_data_ime2, top_blob, M, N, K, alpha, out_elempack, _nT, opt, BT_i8c_ime2, BT_wscale_ime2);
    if (ret == -99)
        return forward_ime2_fallback(A, M, N, K, alpha, out_elempack, top_blobs, opt);
    return ret;
#endif
}
#endif // NCNN_ZFH

} //namespace ncnn
