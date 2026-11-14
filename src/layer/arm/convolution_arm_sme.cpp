// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// 1x1 stride1 fp32 convolution implemented with arm sme
// C[num_output][pixels] = weight[num_output][num_input] * bottom[num_input][pixels]
//
// this translation unit is the only one built with sme enabled
// both kernels run in streaming mode and are entered with pstate.sm and pstate.za already set
//
// they must keep the attributes they are compiled with out of the files that call them: a
// caller that can see them books the vector length sized register state in its own prologue
// with cntd or rdsvl, and apple silicon implements no non-streaming sve, so the transition
// never happens and the call traps. convolution_arm_sme.h enters streaming mode in plain asm
// around an attribute free declaration and saves the registers the kernel is allowed to discard
//
// the channel packed kernel uses st4, an sme2 instruction that runs in streaming mode, so only
// it is compiled for sme2 and its entry is called on cpus that report sme2.

#if __ARM_FEATURE_SME

#include <arm_sme.h>
#include <stddef.h>

// za provides four tiles of svcntw() rows by svcntw() columns of f32
#define NCNN_SME_M_TILES 4

// write the rows of za tile T as output channels of the channel major fp32 layout
// inactive rows are skipped, inactive columns are masked by pg_n
// the kernel keeps the four tiles in za between the two calls, so neither one
// has to read the tile back from memory
template <int T>
static __attribute__((always_inline)) void ncnn_sme_store_plain(const float* bias, float* outptr, size_t out_cstep, int m0, int svl, int m_end, int n0, svbool_t pg_n) __arm_preserves("za") __arm_streaming
{
    const svbool_t ptrue = svptrue_b32();
    const int trow = m0 + T * svl;

    for (int r = 0; r < svl; r++)
    {
        const int m = trow + r;
        if (m >= m_end)
            break;

        svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), ptrue, T, r);

        if (bias)
            v = svadd_f32_x(ptrue, v, svdup_f32(bias[m]));

        svst1_f32(pg_n, outptr + (size_t)m * out_cstep + n0, v);
    }
}

// write the rows of za tile T into the 4 channel packed fp32 layout
// four neighbouring channel rows are stored with one st4, the instruction
// interleaves them so that every pixel receives its four channels contiguously
// st4 is not part of the streaming sve subset of sme1, the entry that selects
// this epilogue is compiled for sme2 and only called when the cpu reports sme2
// no scratch buffer is used, the predicate selects the pixels that are present
// so that partial n tiles store straight into the destination
template <int T>
static __attribute__((target("sme,sme2"), always_inline)) void ncnn_sme_store_pack4(const float* bias, float* outptr, size_t out_cstep, int m0, int svl, int m_end, int N, int n0) __arm_preserves("za") __arm_streaming
{
    const svbool_t ptrue = svptrue_b32();
    const svbool_t pg_n = svwhilelt_b32(n0, N);
    const int trow = m0 + T * svl;

    for (int g = 0; g < svl / 4; g++)
    {
        const int m = trow + g * 4;
        if (m >= m_end)
            break;

        svfloat32_t v0 = svread_hor_za32_f32_m(svundef_f32(), ptrue, T, g * 4 + 0);
        svfloat32_t v1 = svread_hor_za32_f32_m(svundef_f32(), ptrue, T, g * 4 + 1);
        svfloat32_t v2 = svread_hor_za32_f32_m(svundef_f32(), ptrue, T, g * 4 + 2);
        svfloat32_t v3 = svread_hor_za32_f32_m(svundef_f32(), ptrue, T, g * 4 + 3);

        if (bias)
        {
            v0 = svadd_f32_x(ptrue, v0, svdup_f32(bias[m + 0]));
            v1 = svadd_f32_x(ptrue, v1, svdup_f32(bias[m + 1]));
            v2 = svadd_f32_x(ptrue, v2, svdup_f32(bias[m + 2]));
            v3 = svadd_f32_x(ptrue, v3, svdup_f32(bias[m + 3]));
        }

        float* dst = outptr + (size_t)(m / 4) * out_cstep + (size_t)n0 * 4;

        svst4_f32(pg_n, dst, svcreate4_f32(v0, v1, v2, v3));
    }
}

// accumulate A[K][m0, m0 + 4 svl)^T * B[K][n0, n0 + svl) into the four za tiles
// the caller zeroes za and writes the tiles back with one of the store helpers
static __attribute__((always_inline)) void ncnn_sme_mma_tile(const float* A, int lda, const float* B, int ldb, int m0, int n0, int N, int m_end, int K, int svl) __arm_preserves("za") __arm_streaming
{
    const svbool_t pg_m0 = svwhilelt_b32(m0, m_end);
    const svbool_t pg_m1 = svwhilelt_b32(m0 + svl, m_end);
    const svbool_t pg_m2 = svwhilelt_b32(m0 + svl * 2, m_end);
    const svbool_t pg_m3 = svwhilelt_b32(m0 + svl * 3, m_end);
    const svbool_t pg_n = svwhilelt_b32(n0, N);

    svzero_za();

    const float* bp = B + n0;
    const float* ap = A + m0;

    for (int k = 0; k < K; k++)
    {
        const svfloat32_t zn = svld1_f32(pg_n, bp);
        const svfloat32_t zm0 = svld1_f32(pg_m0, ap);
        const svfloat32_t zm1 = svld1_f32(pg_m1, ap + svl);
        const svfloat32_t zm2 = svld1_f32(pg_m2, ap + svl * 2);
        const svfloat32_t zm3 = svld1_f32(pg_m3, ap + svl * 3);

        svmopa_za32_f32_m(0, pg_m0, pg_n, zm0, zn);
        svmopa_za32_f32_m(1, pg_m1, pg_n, zm1, zn);
        svmopa_za32_f32_m(2, pg_m2, pg_n, zm2, zn);
        svmopa_za32_f32_m(3, pg_m3, pg_n, zm3, zn);

        bp += ldb;
        ap += lda;
    }
}

// C[m_begin, m_end) = A[K][M]^T * B[K][N] over the channel major fp32 layout
// out_cstep = N
// entered in streaming mode with za enabled, see ncnn_sme_streaming_guard in the header
extern "C" __arm_new("za") void ncnn_sme_conv1x1s1_f32(const float* A, int lda, const float* B, int ldb, float* outptr, size_t out_cstep, int m_begin, int m_end, int N, int K, const float* bias) __arm_streaming
{
    const int svl = (int)svcntw();
    const int mb = svl * NCNN_SME_M_TILES;

    for (int m0 = m_begin; m0 < m_end; m0 += mb)
    {
        for (int n0 = 0; n0 < N; n0 += svl)
        {
            const svbool_t pg_n = svwhilelt_b32(n0, N);

            ncnn_sme_mma_tile(A, lda, B, ldb, m0, n0, N, m_end, K, svl);

            ncnn_sme_store_plain<0>(bias, outptr, out_cstep, m0, svl, m_end, n0, pg_n);
            ncnn_sme_store_plain<1>(bias, outptr, out_cstep, m0, svl, m_end, n0, pg_n);
            ncnn_sme_store_plain<2>(bias, outptr, out_cstep, m0, svl, m_end, n0, pg_n);
            ncnn_sme_store_plain<3>(bias, outptr, out_cstep, m0, svl, m_end, n0, pg_n);
        }
    }
}

// same kernel over the 4 channel packed fp32 layout, out_cstep = N * 4
// split from the plain entry so that only this one is compiled for sme2
extern "C" __attribute__((target("sme,sme2"))) __arm_new("za") void ncnn_sme_conv1x1s1_pack4_f32(const float* A, int lda, const float* B, int ldb, float* outptr, size_t out_cstep, int m_begin, int m_end, int N, int K, const float* bias) __arm_streaming
{
    const int svl = (int)svcntw();
    const int mb = svl * NCNN_SME_M_TILES;

    for (int m0 = m_begin; m0 < m_end; m0 += mb)
    {
        for (int n0 = 0; n0 < N; n0 += svl)
        {
            ncnn_sme_mma_tile(A, lda, B, ldb, m0, n0, N, m_end, K, svl);

            ncnn_sme_store_pack4<0>(bias, outptr, out_cstep, m0, svl, m_end, N, n0);
            ncnn_sme_store_pack4<1>(bias, outptr, out_cstep, m0, svl, m_end, N, n0);
            ncnn_sme_store_pack4<2>(bias, outptr, out_cstep, m0, svl, m_end, N, n0);
            ncnn_sme_store_pack4<3>(bias, outptr, out_cstep, m0, svl, m_end, N, n0);
        }
    }
}

#endif // __ARM_FEATURE_SME
