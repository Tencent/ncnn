// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_loongarch.h"

#if __loongarch_sx
#include "loongarch_usability.h"
#endif // __loongarch_sx

#include "cpu.h"

namespace ncnn {

Gemm_loongarch::Gemm_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx

    nT = 0;
}

static int unpack_or_cast_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    if (src.empty())
    {
        dst = src;
        return 0;
    }

    Mat unpacked = src;
    if (src.elempack != 1)
    {
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;

        convert_packing(src, unpacked, 1, opt_unpack);
        if (unpacked.empty())
            return -100;
    }

#if NCNN_BF16
    if (unpacked.elembits() == 16)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = opt.workspace_allocator;

        cast_bfloat16_to_float32(unpacked, dst, opt_cast);
        if (dst.empty())
            return -100;
        return 0;
    }
#endif

    dst = unpacked;
    return 0;
}

static int assign_or_copy_top_blob(const Mat& src, Mat& dst)
{
    if (dst.empty())
    {
        dst = src;
        return 0;
    }

    if (dst.dims != src.dims || dst.w != src.w || dst.h != src.h || dst.d != src.d || dst.c != src.c || dst.elemsize != src.elemsize || dst.elempack != src.elempack)
        return -100;

    memcpy((void*)dst, (const void*)src, src.total() * src.elemsize);
    return 0;
}

static int resolve_broadcast_type_C(const Mat& C, int M, int N)
{
    int broadcast_type_C = 0;

    if (C.dims == 1 && C.w == 1)
        broadcast_type_C = 0;
    if (C.dims == 1 && C.w * C.elempack == M)
        broadcast_type_C = 1;
    if (C.dims == 1 && C.w * C.elempack == N)
        broadcast_type_C = 4;
    if (C.dims == 2 && C.w == 1 && C.h * C.elempack == M)
        broadcast_type_C = 2;
    if (C.dims == 2 && C.w == N && C.h * C.elempack == M)
        broadcast_type_C = 3;
    if (C.dims == 2 && C.w == N && C.h * C.elempack == 1)
        broadcast_type_C = 4;

    return broadcast_type_C;
}

static int prepare_C_fp32(const Mat& C_src, Mat& C, int& broadcast_type_C, int M, int N, float beta, const Option& opt)
{
    C = C_src;
    broadcast_type_C = 0;

    if (C.empty())
        return 0;

    if (C.elembits() != 32)
        return -1;

    broadcast_type_C = resolve_broadcast_type_C(C, M, N);

    if (beta != 1.f)
    {
        Option opt_c = opt;
        opt_c.blob_allocator = opt.workspace_allocator;

        Mat C2;
        C2.create_like(C, opt_c.blob_allocator);
        if (C2.empty())
            return -100;

        const int size = C.total() * C.elempack;
        for (int i = 0; i < size; i++)
        {
            C2[i] = C[i] * beta;
        }

        C = C2;
    }

    return 0;
}

static void add_matrix_C_fp32(const Mat& C, Mat& top_blob, int output_transpose)
{
    const int M = C.h;
    const int N = C.w;
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    if (output_transpose)
    {
        if (top_blob.dims == 3)
        {
            for (int j = 0; j < N; j++)
            {
                float* outptr = top_blob.channel(j / out_elempack);
                const int lane = j % out_elempack;
                for (int i = 0; i < M; i++)
                {
                    outptr[i * out_elempack + lane] += C.row(i)[j];
                }
            }
        }
        else
        {
            for (int j = 0; j < N; j++)
            {
                float* outptr = (float*)top_blob + (j / out_elempack) * out_hstep * out_elempack;
                const int lane = j % out_elempack;
                for (int i = 0; i < M; i++)
                {
                    outptr[i * out_elempack + lane] += C.row(i)[j];
                }
            }
        }
    }
    else
    {
        if (top_blob.dims == 3)
        {
            for (int i = 0; i < M; i++)
            {
                float* outptr = top_blob.channel(i / out_elempack);
                const int lane = i % out_elempack;
                const float* cptr = C.row(i);
                for (int j = 0; j < N; j++)
                {
                    outptr[j * out_elempack + lane] += cptr[j];
                }
            }
        }
        else
        {
            for (int i = 0; i < M; i++)
            {
                float* outptr = (float*)top_blob + (i / out_elempack) * out_hstep * out_elempack;
                const int lane = i % out_elempack;
                const float* cptr = C.row(i);
                for (int j = 0; j < N; j++)
                {
                    outptr[j * out_elempack + lane] += cptr[j];
                }
            }
        }
    }
}

#if __loongarch_sx
static NCNN_FORCEINLINE void transpose4x4_ps(__m128& _r0, __m128& _r1, __m128& _r2, __m128& _r3)
{
    __m128i _r01r = __lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
    __m128i _r01l = __lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
    __m128i _r23r = __lsx_vilvl_w((__m128i)_r3, (__m128i)_r2);
    __m128i _r23l = __lsx_vilvh_w((__m128i)_r3, (__m128i)_r2);
    _r0 = (__m128)__lsx_vilvl_d((__m128i)_r23r, (__m128i)_r01r);
    _r1 = (__m128)__lsx_vilvh_d((__m128i)_r23r, (__m128i)_r01r);
    _r2 = (__m128)__lsx_vilvl_d((__m128i)_r23l, (__m128i)_r01l);
    _r3 = (__m128)__lsx_vilvh_d((__m128i)_r23l, (__m128i)_r01l);
}

static NCNN_FORCEINLINE float get_packed_matrix_element(const Mat& m, int row, int col)
{
    const int elempack = m.elempack;
    const size_t hstep = m.dims == 3 ? m.cstep : (size_t)m.w;

    return ((const float*)m)[(size_t)(row / elempack) * hstep * elempack + (size_t)col * elempack + row % elempack];
}

static NCNN_FORCEINLINE void set_packed_output_element(Mat& top_blob, int row, int col, float v, bool output_transpose)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    size_t offset;
    if (output_transpose)
        offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
    else
        offset = (size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack;

    ((float*)top_blob)[offset] = v;
}

static NCNN_FORCEINLINE void store_output_block(Mat& top_blob, int i, int rows, int j, int cols, const float* block, bool output_transpose, bool rowmajor)
{
    if (rowmajor)
    {
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                set_packed_output_element(top_blob, i + r, j + c, block[r * cols + c], output_transpose);
            }
        }
    }
    else
    {
        for (int c = 0; c < cols; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                set_packed_output_element(top_blob, i + r, j + c, block[c * rows + r], output_transpose);
            }
        }
    }
}

static NCNN_FORCEINLINE bool use_8row_packed_kernel(int out_elempack)
{
#if __loongarch_asx
    (void)out_elempack;
    return false;
#else
    (void)out_elempack;
    return true;
#endif
}

static void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, i + ii, k + kk);
                pp[1] = get_packed_matrix_element(A, i + ii + 1, k + kk);
                pp[2] = get_packed_matrix_element(A, i + ii + 2, k + kk);
                pp[3] = get_packed_matrix_element(A, i + ii + 3, k + kk);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    pp += 4;
                    p0 += 4;
                }
            }
            else
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
                const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
                const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
                const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p1[0];
                    pp[2] = p2[0];
                    pp[3] = p3[0];
                    pp += 4;
                    p0++;
                    p1++;
                    p2++;
                    p3++;
                }
            }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __lsx_vst(__lsx_vld(p0, 0), pp, 0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int q = 0; q < 8; q++)
                {
                    pp[q * 4] = get_packed_matrix_element(A, k + kk + q, i + ii);
                    pp[q * 4 + 1] = get_packed_matrix_element(A, k + kk + q, i + ii + 1);
                    pp[q * 4 + 2] = get_packed_matrix_element(A, k + kk + q, i + ii + 2);
                    pp[q * 4 + 3] = get_packed_matrix_element(A, k + kk + q, i + ii + 3);
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, k + kk, i + ii);
                pp[1] = get_packed_matrix_element(A, k + kk, i + ii + 1);
                pp[2] = get_packed_matrix_element(A, k + kk, i + ii + 2);
                pp[3] = get_packed_matrix_element(A, k + kk, i + ii + 3);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p0 + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
                    pp += 16;
                    p0 += A_hstep * 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[4];
                    pp[2] = p0[8];
                    pp[3] = p0[12];
                    pp += 4;
                    p0++;
                }
            }
            else
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii);

                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp += 4;
                    p0 += A_hstep;
                }
            }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                float tmp0[4];
                float tmp1[4];
                __lsx_vst((__m128i)_r0, tmp0, 0);
                __lsx_vst((__m128i)_r1, tmp1, 0);
                pp[0] = tmp0[0];
                pp[1] = tmp1[0];
                pp[2] = tmp0[1];
                pp[3] = tmp1[1];
                pp[4] = tmp0[2];
                pp[5] = tmp1[2];
                pp[6] = tmp0[3];
                pp[7] = tmp1[3];
                pp += 8;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp += 2;
                p0++;
            }
        }
        else
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += A_hstep;
            }
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                pp += 4;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
        else
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void pack_A_tile_8row(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, i + ii, k + kk);
                pp[1] = get_packed_matrix_element(A, i + ii + 1, k + kk);
                pp[2] = get_packed_matrix_element(A, i + ii + 2, k + kk);
                pp[3] = get_packed_matrix_element(A, i + ii + 3, k + kk);
                pp[4] = get_packed_matrix_element(A, i + ii + 4, k + kk);
                pp[5] = get_packed_matrix_element(A, i + ii + 5, k + kk);
                pp[6] = get_packed_matrix_element(A, i + ii + 6, k + kk);
                pp[7] = get_packed_matrix_element(A, i + ii + 7, k + kk);
                pp += 8;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
                const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    __lsx_vst(__lsx_vld(p1, 0), pp + 4, 0);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
            }
            else
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
                const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
                const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
                const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
                const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
                const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
                const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
                const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __m128 _r4 = (__m128)__lsx_vld(p4, 0);
                    __m128 _r5 = (__m128)__lsx_vld(p5, 0);
                    __m128 _r6 = (__m128)__lsx_vld(p6, 0);
                    __m128 _r7 = (__m128)__lsx_vld(p7, 0);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r4, pp + 4, 0);
                    __lsx_vst((__m128i)_r1, pp + 8, 0);
                    __lsx_vst((__m128i)_r5, pp + 12, 0);
                    __lsx_vst((__m128i)_r2, pp + 16, 0);
                    __lsx_vst((__m128i)_r6, pp + 20, 0);
                    __lsx_vst((__m128i)_r3, pp + 24, 0);
                    __lsx_vst((__m128i)_r7, pp + 28, 0);
                    pp += 32;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                    p4 += 4;
                    p5 += 4;
                    p6 += 4;
                    p7 += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p1[0];
                    pp[2] = p2[0];
                    pp[3] = p3[0];
                    pp[4] = p4[0];
                    pp[5] = p5[0];
                    pp[6] = p6[0];
                    pp[7] = p7[0];
                    pp += 8;
                    p0++;
                    p1++;
                    p2++;
                    p3++;
                    p4++;
                    p5++;
                    p6++;
                    p7++;
                }
            }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, i + ii, k + kk);
                pp[1] = get_packed_matrix_element(A, i + ii + 1, k + kk);
                pp[2] = get_packed_matrix_element(A, i + ii + 2, k + kk);
                pp[3] = get_packed_matrix_element(A, i + ii + 3, k + kk);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    pp += 4;
                    p0 += 4;
                }
            }
            else
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
                const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
                const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
                const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p1[0];
                    pp[2] = p2[0];
                    pp[3] = p3[0];
                    pp += 4;
                    p0++;
                    p1++;
                    p2++;
                    p3++;
                }
            }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __lsx_vst(__lsx_vld(p0, 0), pp, 0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_8row(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, k + kk, i + ii);
                pp[1] = get_packed_matrix_element(A, k + kk, i + ii + 1);
                pp[2] = get_packed_matrix_element(A, k + kk, i + ii + 2);
                pp[3] = get_packed_matrix_element(A, k + kk, i + ii + 3);
                pp[4] = get_packed_matrix_element(A, k + kk, i + ii + 4);
                pp[5] = get_packed_matrix_element(A, k + kk, i + ii + 5);
                pp[6] = get_packed_matrix_element(A, k + kk, i + ii + 6);
                pp[7] = get_packed_matrix_element(A, k + kk, i + ii + 7);
                pp += 8;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p0 + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __m128 _r4 = (__m128)__lsx_vld(p0 + 16, 0);
                    __m128 _r5 = (__m128)__lsx_vld(p0 + 20, 0);
                    __m128 _r6 = (__m128)__lsx_vld(p0 + 24, 0);
                    __m128 _r7 = (__m128)__lsx_vld(p0 + 28, 0);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r4, pp + 4, 0);
                    __lsx_vst((__m128i)_r1, pp + 8, 0);
                    __lsx_vst((__m128i)_r5, pp + 12, 0);
                    __lsx_vst((__m128i)_r2, pp + 16, 0);
                    __lsx_vst((__m128i)_r6, pp + 20, 0);
                    __lsx_vst((__m128i)_r3, pp + 24, 0);
                    __lsx_vst((__m128i)_r7, pp + 28, 0);
                    pp += 32;
                    p0 += A_hstep * 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[4];
                    pp[2] = p0[8];
                    pp[3] = p0[12];
                    pp[4] = p0[16];
                    pp[5] = p0[20];
                    pp[6] = p0[24];
                    pp[7] = p0[28];
                    pp += 8;
                    p0++;
                }
            }
            else
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii);

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    __lsx_vst(__lsx_vld(p0 + 4, 0), pp + 4, 0);
                    pp += 8;
                    p0 += A_hstep;
                }
            }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int q = 0; q < 8; q++)
                {
                    pp[q * 4] = get_packed_matrix_element(A, k + kk + q, i + ii);
                    pp[q * 4 + 1] = get_packed_matrix_element(A, k + kk + q, i + ii + 1);
                    pp[q * 4 + 2] = get_packed_matrix_element(A, k + kk + q, i + ii + 2);
                    pp[q * 4 + 3] = get_packed_matrix_element(A, k + kk + q, i + ii + 3);
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, k + kk, i + ii);
                pp[1] = get_packed_matrix_element(A, k + kk, i + ii + 1);
                pp[2] = get_packed_matrix_element(A, k + kk, i + ii + 2);
                pp[3] = get_packed_matrix_element(A, k + kk, i + ii + 3);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p0 + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
                    pp += 16;
                    p0 += A_hstep * 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[4];
                    pp[2] = p0[8];
                    pp[3] = p0[12];
                    pp += 4;
                    p0++;
                }
            }
            else
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii);

                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp += 4;
                    p0 += A_hstep;
                }
            }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                float tmp0[4];
                float tmp1[4];
                __lsx_vst((__m128i)_r0, tmp0, 0);
                __lsx_vst((__m128i)_r1, tmp1, 0);
                pp[0] = tmp0[0];
                pp[1] = tmp1[0];
                pp[2] = tmp0[1];
                pp[3] = tmp1[1];
                pp[4] = tmp0[2];
                pp[5] = tmp1[2];
                pp[6] = tmp0[3];
                pp[7] = tmp1[3];
                pp += 8;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp += 2;
                p0++;
            }
        }
        else
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += A_hstep;
            }
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                pp += 4;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
        else
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    float* pp = BT;

    int jj = 0;
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, j + jj, k + kk);
            pp[1] = get_packed_matrix_element(B, j + jj + 1, k + kk);
            pp[2] = get_packed_matrix_element(B, j + jj + 2, k + kk);
            pp[3] = get_packed_matrix_element(B, j + jj + 3, k + kk);
            pp[4] = get_packed_matrix_element(B, j + jj + 4, k + kk);
            pp[5] = get_packed_matrix_element(B, j + jj + 5, k + kk);
            pp[6] = get_packed_matrix_element(B, j + jj + 6, k + kk);
            pp[7] = get_packed_matrix_element(B, j + jj + 7, k + kk);
            pp[8] = get_packed_matrix_element(B, j + jj + 8, k + kk);
            pp[9] = get_packed_matrix_element(B, j + jj + 9, k + kk);
            pp[10] = get_packed_matrix_element(B, j + jj + 10, k + kk);
            pp[11] = get_packed_matrix_element(B, j + jj + 11, k + kk);
            pp[12] = get_packed_matrix_element(B, j + jj + 12, k + kk);
            pp[13] = get_packed_matrix_element(B, j + jj + 13, k + kk);
            pp[14] = get_packed_matrix_element(B, j + jj + 14, k + kk);
            pp[15] = get_packed_matrix_element(B, j + jj + 15, k + kk);
            pp += 16;
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, j + jj, k + kk);
            pp[1] = get_packed_matrix_element(B, j + jj + 1, k + kk);
            pp[2] = get_packed_matrix_element(B, j + jj + 2, k + kk);
            pp[3] = get_packed_matrix_element(B, j + jj + 3, k + kk);
            pp[4] = get_packed_matrix_element(B, j + jj + 4, k + kk);
            pp[5] = get_packed_matrix_element(B, j + jj + 5, k + kk);
            pp[6] = get_packed_matrix_element(B, j + jj + 6, k + kk);
            pp[7] = get_packed_matrix_element(B, j + jj + 7, k + kk);
            pp += 8;
        }
    }
#endif
    for (; jj + 7 < max_jj; jj += 8)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, j + jj, k + kk);
            pp[1] = get_packed_matrix_element(B, j + jj + 1, k + kk);
            pp[2] = get_packed_matrix_element(B, j + jj + 2, k + kk);
            pp[3] = get_packed_matrix_element(B, j + jj + 3, k + kk);
            pp[4] = get_packed_matrix_element(B, j + jj + 4, k + kk);
            pp[5] = get_packed_matrix_element(B, j + jj + 5, k + kk);
            pp[6] = get_packed_matrix_element(B, j + jj + 6, k + kk);
            pp[7] = get_packed_matrix_element(B, j + jj + 7, k + kk);
            pp += 8;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(B, j + jj, k + kk);
                pp[1] = get_packed_matrix_element(B, j + jj + 1, k + kk);
                pp[2] = get_packed_matrix_element(B, j + jj + 2, k + kk);
                pp[3] = get_packed_matrix_element(B, j + jj + 3, k + kk);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    pp += 4;
                    p0 += 4;
                }
            }
            else
            {
                const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
                const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
                const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
                const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p1[0];
                    pp[2] = p2[0];
                    pp[3] = p3[0];
                    pp += 4;
                    p0++;
                    p1++;
                    p2++;
                    p3++;
                }
            }
    }

    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }

    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __lsx_vst(__lsx_vld(p0, 0), pp, 0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    float* pp = BT;

    int jj = 0;
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, k + kk, j + jj);
            pp[1] = get_packed_matrix_element(B, k + kk, j + jj + 1);
            pp[2] = get_packed_matrix_element(B, k + kk, j + jj + 2);
            pp[3] = get_packed_matrix_element(B, k + kk, j + jj + 3);
            pp[4] = get_packed_matrix_element(B, k + kk, j + jj + 4);
            pp[5] = get_packed_matrix_element(B, k + kk, j + jj + 5);
            pp[6] = get_packed_matrix_element(B, k + kk, j + jj + 6);
            pp[7] = get_packed_matrix_element(B, k + kk, j + jj + 7);
            pp[8] = get_packed_matrix_element(B, k + kk, j + jj + 8);
            pp[9] = get_packed_matrix_element(B, k + kk, j + jj + 9);
            pp[10] = get_packed_matrix_element(B, k + kk, j + jj + 10);
            pp[11] = get_packed_matrix_element(B, k + kk, j + jj + 11);
            pp[12] = get_packed_matrix_element(B, k + kk, j + jj + 12);
            pp[13] = get_packed_matrix_element(B, k + kk, j + jj + 13);
            pp[14] = get_packed_matrix_element(B, k + kk, j + jj + 14);
            pp[15] = get_packed_matrix_element(B, k + kk, j + jj + 15);
            pp += 16;
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, k + kk, j + jj);
            pp[1] = get_packed_matrix_element(B, k + kk, j + jj + 1);
            pp[2] = get_packed_matrix_element(B, k + kk, j + jj + 2);
            pp[3] = get_packed_matrix_element(B, k + kk, j + jj + 3);
            pp[4] = get_packed_matrix_element(B, k + kk, j + jj + 4);
            pp[5] = get_packed_matrix_element(B, k + kk, j + jj + 5);
            pp[6] = get_packed_matrix_element(B, k + kk, j + jj + 6);
            pp[7] = get_packed_matrix_element(B, k + kk, j + jj + 7);
            pp += 8;
        }
    }
#endif
    for (; jj + 7 < max_jj; jj += 8)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, k + kk, j + jj);
            pp[1] = get_packed_matrix_element(B, k + kk, j + jj + 1);
            pp[2] = get_packed_matrix_element(B, k + kk, j + jj + 2);
            pp[3] = get_packed_matrix_element(B, k + kk, j + jj + 3);
            pp[4] = get_packed_matrix_element(B, k + kk, j + jj + 4);
            pp[5] = get_packed_matrix_element(B, k + kk, j + jj + 5);
            pp[6] = get_packed_matrix_element(B, k + kk, j + jj + 6);
            pp[7] = get_packed_matrix_element(B, k + kk, j + jj + 7);
            pp += 8;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(B, k + kk, j + jj);
                pp[1] = get_packed_matrix_element(B, k + kk, j + jj + 1);
                pp[2] = get_packed_matrix_element(B, k + kk, j + jj + 2);
                pp[3] = get_packed_matrix_element(B, k + kk, j + jj + 3);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p0 + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
                    pp += 16;
                    p0 += B_hstep * 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[4];
                    pp[2] = p0[8];
                    pp[3] = p0[12];
                    pp += 4;
                    p0++;
                }
            }
            else
            {
                const float* p0 = (const float*)B + k * B_hstep + (j + jj);

                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp += 4;
                    p0 += B_hstep;
                }
            }
    }

    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                float tmp0[4];
                float tmp1[4];
                __lsx_vst((__m128i)_r0, tmp0, 0);
                __lsx_vst((__m128i)_r1, tmp1, 0);
                pp[0] = tmp0[0];
                pp[1] = tmp1[0];
                pp[2] = tmp0[1];
                pp[3] = tmp1[1];
                pp[4] = tmp0[2];
                pp[5] = tmp1[2];
                pp[6] = tmp0[3];
                pp[7] = tmp1[3];
                pp += 8;
                p0 += B_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp += 2;
                p0++;
            }
        }
        else
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += B_hstep;
            }
        }
    }

    for (; jj < max_jj; jj += 1)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                pp += 4;
                p0 += B_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
        else
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void transpose_unpack_output_tile(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pp = topT;

    int ii = 0;
    if (use_8row_packed_kernel(out_elempack))
    {
        for (; ii + 7 < max_ii; ii += 8)
        {
            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                float block[64];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 8, j + jj, 8, block, true, false);
                pp += 64;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                float block[32];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 8, j + jj, 4, block, true, false);
                pp += 32;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                float block[16];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 8, j + jj, 2, block, true, false);
                pp += 16;
            }
            for (; jj < max_jj; jj++)
            {
                float block[8];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 8, j + jj, 1, block, true, false);
                pp += 8;
            }
        }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 8)
        {
            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 4, j + jj, 16, pp, true, true);
                pp += 64;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 4, j + jj, 8, pp, true, true);
                pp += 32;
            }
#endif
            for (; jj + 3 < max_jj; jj += 4)
            {
                store_output_block(top_blob, i + ii, 4, j + jj, 4, pp, true, false);
                pp += 16;
            }
            for (; jj < max_jj; jj++)
            {
                store_output_block(top_blob, i + ii, 4, j + jj, 1, pp, true, false);
                pp += 4;
            }
        }
        else if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 4, j + jj, 16, pp, true, true);
                pp += 64;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 4, j + jj, 8, pp, true, true);
                pp += 32;
                p0 += out_hstep * 8;
            }
#endif
            for (; jj + 7 < max_jj; jj += 8)
            {
                float block[32];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 4, j + jj, 8, block, true, false);
                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(pp, 0);
                __m128 _r1 = (__m128)__lsx_vld(pp + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(pp + 8, 0);
                __m128 _r3 = (__m128)__lsx_vld(pp + 12, 0);
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                __lsx_vst((__m128i)_r0, p0, 0);
                __lsx_vst((__m128i)_r1, p0 + 4, 0);
                __lsx_vst((__m128i)_r2, p0 + 8, 0);
                __lsx_vst((__m128i)_r3, p0 + 12, 0);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        else
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

#if __loongarch_asx
            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 4, j + jj, 16, pp, true, true);
                pp += 64;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 4, j + jj, 8, pp, true, true);
                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                p0[2] = pp[2];
                p0[3] = pp[3];
                pp += 4;
                p0 += out_hstep;
            }
#else
            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                float block[32];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 4, j + jj, 8, block, true, false);
                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                p0[2] = pp[2];
                p0[3] = pp[3];
                pp += 4;
                p0 += out_hstep;
            }
#endif
        }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        if (out_elempack == 8)
        {
            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 16, pp, true, true);
                pp += 32;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 8, pp, true, true);
                pp += 16;
            }
#endif
            for (; jj + 3 < max_jj; jj += 4)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 4, pp, true, false);
                pp += 8;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 2, pp, true, false);
                pp += 4;
            }
            for (; jj < max_jj; jj++)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 1, pp, true, false);
                pp += 2;
            }
        }
        else if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 16, pp, true, true);
                pp += 32;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 8, pp, true, true);
                pp += 16;
                p0 += out_hstep * 8;
            }
#endif
            for (; jj + 7 < max_jj; jj += 8)
            {
                float block[16];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 2, j + jj, 8, block, true, false);
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[1];
                p0[5] = pp[3];
                p0[6] = pp[5];
                p0[7] = pp[7];
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
        else
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

#if __loongarch_asx
            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 16, pp, true, true);
                pp += 32;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 2, j + jj, 8, pp, true, true);
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
#else
            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                float block[16];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 2, j + jj, 8, block, true, false);
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
#endif
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        if (out_elempack == 8)
        {
            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 16, pp, true, true);
                pp += 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 8, pp, true, true);
                pp += 8;
            }
#endif
            for (; jj + 3 < max_jj; jj += 4)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 4, pp, true, false);
                pp += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 2, pp, true, false);
                pp += 2;
            }
            for (; jj < max_jj; jj++)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 1, pp, true, false);
                pp += 1;
            }
        }
        else if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 16, pp, true, true);
                pp += 16;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 8, pp, true, true);
                pp += 8;
                p0 += out_hstep * 8;
            }
#endif
            for (; jj + 7 < max_jj; jj += 8)
            {
                float block[8];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 1, j + jj, 8, block, true, false);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __lsx_vst(__lsx_vld(pp, 0), p0, 0);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
        else
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

#if __loongarch_asx
            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 16, pp, true, true);
                pp += 16;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 8, pp, true, true);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
#else
            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                float block[8];
                memcpy(block, pp, sizeof(block));
                store_output_block(top_blob, i + ii, 1, j + jj, 8, block, true, false);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
#endif
        }
    }
}

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
    if (use_8row_packed_kernel(out_elempack))
    {
        for (; ii + 7 < max_ii; ii += 8)
        {
            float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

            const float* pB = pBT;
            const float* pCi = pC;

            if (pCi)
            {
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    pCi = (const float*)CT_tile + i + ii;
                if (broadcast_type_C == 4)
                    pCi = (const float*)CT_tile + j;
            }

            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m128 _sum0[8];
                __m128 _sum1[8];

                if (k == 0)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        _sum0[c] = (__m128)__lsx_vreplgr2vr_w(0);
                        _sum1[c] = (__m128)__lsx_vreplgr2vr_w(0);
                    }

                    if (pCi)
                    {
                        if (broadcast_type_C == 0)
                        {
                            __m128 _c0 = __lsx_vreplfr2vr_s(pCi[0]);
                            for (int c = 0; c < 8; c++)
                            {
                                _sum0[c] = _c0;
                                _sum1[c] = _c0;
                            }
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            __m128 _c0 = (__m128)__lsx_vld(pCi, 0);
                            __m128 _c1 = (__m128)__lsx_vld(pCi + 4, 0);
                            for (int c = 0; c < 8; c++)
                            {
                                _sum0[c] = _c0;
                                _sum1[c] = _c1;
                            }
                        }
                        if (broadcast_type_C == 3)
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                _sum0[c] = (__m128)__lsx_vld(pCi + c * 8, 0);
                                _sum1[c] = (__m128)__lsx_vld(pCi + c * 8 + 4, 0);
                            }
                            pCi += 64;
                        }
                        if (broadcast_type_C == 4)
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                __m128 _c0 = __lsx_vreplfr2vr_s(pCi[c]);
                                _sum0[c] = _c0;
                                _sum1[c] = _c0;
                            }
                            pCi += 8;
                        }
                    }
                }
                else
                {
                    for (int c = 0; c < 8; c++)
                    {
                        _sum0[c] = (__m128)__lsx_vld(outptr + c * 8, 0);
                        _sum1[c] = (__m128)__lsx_vld(outptr + c * 8 + 4, 0);
                    }
                }

                const float* pA = pAT;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    for (int c = 0; c < 8; c++)
                    {
                        __m128 _pB = __lsx_vreplfr2vr_s(pB[c]);
                        _sum0[c] = __lsx_vfmadd_s(_pB, _pA0, _sum0[c]);
                        _sum1[c] = __lsx_vfmadd_s(_pB, _pA1, _sum1[c]);
                    }
                    pA += 8;
                    pB += 8;
                }

                if (k_end)
                {
                    float block[64];
                    for (int c = 0; c < 8; c++)
                    {
                        __lsx_vst((__m128i)_sum0[c], block + c * 8, 0);
                        __lsx_vst((__m128i)_sum1[c], block + c * 8 + 4, 0);
                    }
                    store_output_block(top_blob, i + ii, 8, j + jj, 8, block, false, false);
                    outptr0 += 8 * out_elempack;
                }
                else
                {
                    for (int c = 0; c < 8; c++)
                    {
                        __lsx_vst((__m128i)_sum0[c], outptr + c * 8, 0);
                        __lsx_vst((__m128i)_sum1[c], outptr + c * 8 + 4, 0);
                    }
                }

                outptr += 64;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m128 _sum00;
                __m128 _sum01;
                __m128 _sum10;
                __m128 _sum11;
                __m128 _sum20;
                __m128 _sum21;
                __m128 _sum30;
                __m128 _sum31;

                if (k == 0)
                {
                    _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum01 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum10 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum11 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum20 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum21 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum30 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum31 = (__m128)__lsx_vreplgr2vr_w(0);

                    if (pCi)
                    {
                        if (broadcast_type_C == 0)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum01 = _sum00;
                            _sum10 = _sum00;
                            _sum11 = _sum00;
                            _sum20 = _sum00;
                            _sum21 = _sum00;
                            _sum30 = _sum00;
                            _sum31 = _sum00;
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            _sum10 = _sum00;
                            _sum11 = _sum01;
                            _sum20 = _sum00;
                            _sum21 = _sum01;
                            _sum30 = _sum00;
                            _sum31 = _sum01;
                        }
                        if (broadcast_type_C == 3)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            _sum10 = (__m128)__lsx_vld(pCi + 8, 0);
                            _sum11 = (__m128)__lsx_vld(pCi + 12, 0);
                            _sum20 = (__m128)__lsx_vld(pCi + 16, 0);
                            _sum21 = (__m128)__lsx_vld(pCi + 20, 0);
                            _sum30 = (__m128)__lsx_vld(pCi + 24, 0);
                            _sum31 = (__m128)__lsx_vld(pCi + 28, 0);
                            pCi += 32;
                        }
                        if (broadcast_type_C == 4)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum10 = __lsx_vreplfr2vr_s(pCi[1]);
                            _sum20 = __lsx_vreplfr2vr_s(pCi[2]);
                            _sum30 = __lsx_vreplfr2vr_s(pCi[3]);
                            _sum01 = _sum00;
                            _sum11 = _sum10;
                            _sum21 = _sum20;
                            _sum31 = _sum30;
                            pCi += 4;
                        }
                    }
                }
                else
                {
                    _sum00 = (__m128)__lsx_vld(outptr, 0);
                    _sum01 = (__m128)__lsx_vld(outptr + 4, 0);
                    _sum10 = (__m128)__lsx_vld(outptr + 8, 0);
                    _sum11 = (__m128)__lsx_vld(outptr + 12, 0);
                    _sum20 = (__m128)__lsx_vld(outptr + 16, 0);
                    _sum21 = (__m128)__lsx_vld(outptr + 20, 0);
                    _sum30 = (__m128)__lsx_vld(outptr + 24, 0);
                    _sum31 = (__m128)__lsx_vld(outptr + 28, 0);
                }

                const float* pA = pAT;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    _sum00 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA0, _sum00);
                    _sum01 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA1, _sum01);
                    _sum10 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA0, _sum10);
                    _sum11 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA1, _sum11);
                    _sum20 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[2]), _pA0, _sum20);
                    _sum21 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[2]), _pA1, _sum21);
                    _sum30 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[3]), _pA0, _sum30);
                    _sum31 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[3]), _pA1, _sum31);
                    pA += 8;
                    pB += 4;
                }

                if (k_end)
                {
                    if (out_elempack == 4)
                    {
                        float* outptr1 = outptr0 + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, outptr0, 0);
                        __lsx_vst((__m128i)_sum10, outptr0 + 4, 0);
                        __lsx_vst((__m128i)_sum20, outptr0 + 8, 0);
                        __lsx_vst((__m128i)_sum30, outptr0 + 12, 0);
                        __lsx_vst((__m128i)_sum01, outptr1, 0);
                        __lsx_vst((__m128i)_sum11, outptr1 + 4, 0);
                        __lsx_vst((__m128i)_sum21, outptr1 + 8, 0);
                        __lsx_vst((__m128i)_sum31, outptr1 + 12, 0);
                        outptr0 += 16;
                    }
                    else
                    {
                        __m128 _r0 = _sum00;
                        __m128 _r1 = _sum10;
                        __m128 _r2 = _sum20;
                        __m128 _r3 = _sum30;
                        transpose4x4_ps(_r0, _r1, _r2, _r3);
                        __lsx_vst((__m128i)_r0, outptr0, 0);
                        __lsx_vst((__m128i)_r1, outptr0 + out_hstep, 0);
                        __lsx_vst((__m128i)_r2, outptr0 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_r3, outptr0 + out_hstep * 3, 0);
                        __m128 _r4 = _sum01;
                        __m128 _r5 = _sum11;
                        __m128 _r6 = _sum21;
                        __m128 _r7 = _sum31;
                        transpose4x4_ps(_r4, _r5, _r6, _r7);
                        __lsx_vst((__m128i)_r4, outptr0 + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_r5, outptr0 + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_r6, outptr0 + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_r7, outptr0 + out_hstep * 7, 0);
                        outptr0 += 4;
                    }
                }
                else
                {
                    __lsx_vst((__m128i)_sum00, outptr, 0);
                    __lsx_vst((__m128i)_sum01, outptr + 4, 0);
                    __lsx_vst((__m128i)_sum10, outptr + 8, 0);
                    __lsx_vst((__m128i)_sum11, outptr + 12, 0);
                    __lsx_vst((__m128i)_sum20, outptr + 16, 0);
                    __lsx_vst((__m128i)_sum21, outptr + 20, 0);
                    __lsx_vst((__m128i)_sum30, outptr + 24, 0);
                    __lsx_vst((__m128i)_sum31, outptr + 28, 0);
                }

                outptr += 32;
            }

            for (; jj + 1 < max_jj; jj += 2)
            {
                __m128 _sum00;
                __m128 _sum01;
                __m128 _sum10;
                __m128 _sum11;

                if (k == 0)
                {
                    _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum01 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum10 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum11 = (__m128)__lsx_vreplgr2vr_w(0);

                    if (pCi)
                    {
                        if (broadcast_type_C == 0)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum01 = _sum00;
                            _sum10 = _sum00;
                            _sum11 = _sum00;
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            _sum10 = _sum00;
                            _sum11 = _sum01;
                        }
                        if (broadcast_type_C == 3)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            _sum10 = (__m128)__lsx_vld(pCi + 8, 0);
                            _sum11 = (__m128)__lsx_vld(pCi + 12, 0);
                            pCi += 16;
                        }
                        if (broadcast_type_C == 4)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum10 = __lsx_vreplfr2vr_s(pCi[1]);
                            _sum01 = _sum00;
                            _sum11 = _sum10;
                            pCi += 2;
                        }
                    }
                }
                else
                {
                    _sum00 = (__m128)__lsx_vld(outptr, 0);
                    _sum01 = (__m128)__lsx_vld(outptr + 4, 0);
                    _sum10 = (__m128)__lsx_vld(outptr + 8, 0);
                    _sum11 = (__m128)__lsx_vld(outptr + 12, 0);
                }

                const float* pA = pAT;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    _sum00 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA0, _sum00);
                    _sum01 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA1, _sum01);
                    _sum10 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA0, _sum10);
                    _sum11 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA1, _sum11);
                    pA += 8;
                    pB += 2;
                }

                if (k_end)
                {
                    if (out_elempack == 4)
                    {
                        float* outptr1 = outptr0 + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, outptr0, 0);
                        __lsx_vst((__m128i)_sum10, outptr0 + 4, 0);
                        __lsx_vst((__m128i)_sum01, outptr1, 0);
                        __lsx_vst((__m128i)_sum11, outptr1 + 4, 0);
                        outptr0 += 8;
                    }
                    else
                    {
                        float tmp0[4];
                        float tmp1[4];
                        float tmp2[4];
                        float tmp3[4];
                        __lsx_vst((__m128i)_sum00, tmp0, 0);
                        __lsx_vst((__m128i)_sum10, tmp1, 0);
                        __lsx_vst((__m128i)_sum01, tmp2, 0);
                        __lsx_vst((__m128i)_sum11, tmp3, 0);
                        for (int r = 0; r < 4; r++)
                        {
                            outptr0[out_hstep * r] = tmp0[r];
                            outptr0[out_hstep * r + 1] = tmp1[r];
                            outptr0[out_hstep * (r + 4)] = tmp2[r];
                            outptr0[out_hstep * (r + 4) + 1] = tmp3[r];
                        }
                        outptr0 += 2;
                    }
                }
                else
                {
                    __lsx_vst((__m128i)_sum00, outptr, 0);
                    __lsx_vst((__m128i)_sum01, outptr + 4, 0);
                    __lsx_vst((__m128i)_sum10, outptr + 8, 0);
                    __lsx_vst((__m128i)_sum11, outptr + 12, 0);
                }

                outptr += 16;
            }

            for (; jj < max_jj; jj += 1)
            {
                __m128 _sum00;
                __m128 _sum01;

                if (k == 0)
                {
                    _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum01 = (__m128)__lsx_vreplgr2vr_w(0);

                    if (pCi)
                    {
                        if (broadcast_type_C == 0)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum01 = _sum00;
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                        }
                        if (broadcast_type_C == 3)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            pCi += 8;
                        }
                        if (broadcast_type_C == 4)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum01 = _sum00;
                            pCi += 1;
                        }
                    }
                }
                else
                {
                    _sum00 = (__m128)__lsx_vld(outptr, 0);
                    _sum01 = (__m128)__lsx_vld(outptr + 4, 0);
                }

                const float* pA = pAT;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    _sum00 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA0, _sum00);
                    _sum01 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA1, _sum01);
                    pA += 8;
                    pB += 1;
                }

                if (k_end)
                {
                    if (out_elempack == 4)
                    {
                        float* outptr1 = outptr0 + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, outptr0, 0);
                        __lsx_vst((__m128i)_sum01, outptr1, 0);
                        outptr0 += 4;
                    }
                    else
                    {
                        float tmp0[4];
                        float tmp1[4];
                        __lsx_vst((__m128i)_sum00, tmp0, 0);
                        __lsx_vst((__m128i)_sum01, tmp1, 0);
                        for (int r = 0; r < 4; r++)
                        {
                            outptr0[out_hstep * r] = tmp0[r];
                            outptr0[out_hstep * (r + 4)] = tmp1[r];
                        }
                        outptr0 += 1;
                    }
                }
                else
                {
                    __lsx_vst((__m128i)_sum00, outptr, 0);
                    __lsx_vst((__m128i)_sum01, outptr + 4, 0);
                }

                outptr += 8;
            }

            pAT += max_kk * 8;
        }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;
        const float* pCi = pC;

        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)CT_tile + i + ii;
            if (broadcast_type_C == 4)
                pCi = (const float*)CT_tile + j;
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum00;
            __m256 _sum01;
            __m256 _sum10;
            __m256 _sum11;
            __m256 _sum20;
            __m256 _sum21;
            __m256 _sum30;
            __m256 _sum31;

            if (k == 0)
            {
                _sum00 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum01 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum10 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum11 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum20 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum21 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum30 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum31 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum00 = _c;
                        _sum01 = _c;
                        _sum10 = _c;
                        _sum11 = _c;
                        _sum20 = _c;
                        _sum21 = _c;
                        _sum30 = _c;
                        _sum31 = _c;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum01 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum10 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                        _sum11 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                        _sum20 = (__m256)__lasx_xvreplfr2vr_s(pCi[2]);
                        _sum21 = (__m256)__lasx_xvreplfr2vr_s(pCi[2]);
                        _sum30 = (__m256)__lasx_xvreplfr2vr_s(pCi[3]);
                        _sum31 = (__m256)__lasx_xvreplfr2vr_s(pCi[3]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __attribute__((aligned(32))) float ctmp[64];
                        for (int r = 0; r < 4; r++)
                        {
                            for (int c = 0; c < 16; c++)
                            {
                                ctmp[r * 16 + c] = pCi[c * 4 + r];
                            }
                        }
                        _sum00 = (__m256)__lasx_xvld(ctmp, 0);
                        _sum01 = (__m256)__lasx_xvld(ctmp + 8, 0);
                        _sum10 = (__m256)__lasx_xvld(ctmp + 16, 0);
                        _sum11 = (__m256)__lasx_xvld(ctmp + 24, 0);
                        _sum20 = (__m256)__lasx_xvld(ctmp + 32, 0);
                        _sum21 = (__m256)__lasx_xvld(ctmp + 40, 0);
                        _sum30 = (__m256)__lasx_xvld(ctmp + 48, 0);
                        _sum31 = (__m256)__lasx_xvld(ctmp + 56, 0);
                        pCi += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m256 _c0 = (__m256)__lasx_xvld(pCi, 0);
                        __m256 _c1 = (__m256)__lasx_xvld(pCi + 8, 0);
                        _sum00 = _c0;
                        _sum01 = _c1;
                        _sum10 = _c0;
                        _sum11 = _c1;
                        _sum20 = _c0;
                        _sum21 = _c1;
                        _sum30 = _c0;
                        _sum31 = _c1;
                        pCi += 16;
                    }
                }
            }
            else
            {
                _sum00 = (__m256)__lasx_xvld(outptr, 0);
                _sum01 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum10 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum11 = (__m256)__lasx_xvld(outptr + 24, 0);
                _sum20 = (__m256)__lasx_xvld(outptr + 32, 0);
                _sum21 = (__m256)__lasx_xvld(outptr + 40, 0);
                _sum30 = (__m256)__lasx_xvld(outptr + 48, 0);
                _sum31 = (__m256)__lasx_xvld(outptr + 56, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _a0 = (__m256)__lasx_xvreplfr2vr_s(pA[0]);
                __m256 _a1 = (__m256)__lasx_xvreplfr2vr_s(pA[1]);
                __m256 _a2 = (__m256)__lasx_xvreplfr2vr_s(pA[2]);
                __m256 _a3 = (__m256)__lasx_xvreplfr2vr_s(pA[3]);
                _sum00 = __lasx_xvfmadd_s(_a0, _pB0, _sum00);
                _sum01 = __lasx_xvfmadd_s(_a0, _pB1, _sum01);
                _sum10 = __lasx_xvfmadd_s(_a1, _pB0, _sum10);
                _sum11 = __lasx_xvfmadd_s(_a1, _pB1, _sum11);
                _sum20 = __lasx_xvfmadd_s(_a2, _pB0, _sum20);
                _sum21 = __lasx_xvfmadd_s(_a2, _pB1, _sum21);
                _sum30 = __lasx_xvfmadd_s(_a3, _pB0, _sum30);
                _sum31 = __lasx_xvfmadd_s(_a3, _pB1, _sum31);
                pA += 4;
                pB += 16;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[64];
                __lasx_xvst(_sum00, tmp, 0);
                __lasx_xvst(_sum01, tmp + 8, 0);
                __lasx_xvst(_sum10, tmp + 16, 0);
                __lasx_xvst(_sum11, tmp + 24, 0);
                __lasx_xvst(_sum20, tmp + 32, 0);
                __lasx_xvst(_sum21, tmp + 40, 0);
                __lasx_xvst(_sum30, tmp + 48, 0);
                __lasx_xvst(_sum31, tmp + 56, 0);
                store_output_block(top_blob, i + ii, 4, j + jj, 16, tmp, false, true);
                outptr0 += 16 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum00, outptr, 0);
                __lasx_xvst(_sum01, outptr + 8, 0);
                __lasx_xvst(_sum10, outptr + 16, 0);
                __lasx_xvst(_sum11, outptr + 24, 0);
                __lasx_xvst(_sum20, outptr + 32, 0);
                __lasx_xvst(_sum21, outptr + 40, 0);
                __lasx_xvst(_sum30, outptr + 48, 0);
                __lasx_xvst(_sum31, outptr + 56, 0);
            }

            outptr += 64;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum3 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum0 = _c;
                        _sum1 = _c;
                        _sum2 = _c;
                        _sum3 = _c;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum1 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                        _sum2 = (__m256)__lasx_xvreplfr2vr_s(pCi[2]);
                        _sum3 = (__m256)__lasx_xvreplfr2vr_s(pCi[3]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __attribute__((aligned(32))) float ctmp[32];
                        for (int r = 0; r < 4; r++)
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                ctmp[r * 8 + c] = pCi[c * 4 + r];
                            }
                        }
                        _sum0 = (__m256)__lasx_xvld(ctmp, 0);
                        _sum1 = (__m256)__lasx_xvld(ctmp + 8, 0);
                        _sum2 = (__m256)__lasx_xvld(ctmp + 16, 0);
                        _sum3 = (__m256)__lasx_xvld(ctmp + 24, 0);
                        pCi += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m256 _c = (__m256)__lasx_xvld(pCi, 0);
                        _sum0 = _c;
                        _sum1 = _c;
                        _sum2 = _c;
                        _sum3 = _c;
                        pCi += 8;
                    }
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB = (__m256)__lasx_xvld(pB, 0);
                _sum0 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[0]), _pB, _sum0);
                _sum1 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[1]), _pB, _sum1);
                _sum2 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[2]), _pB, _sum2);
                _sum3 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[3]), _pB, _sum3);
                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[32];
                __lasx_xvst(_sum0, tmp, 0);
                __lasx_xvst(_sum1, tmp + 8, 0);
                __lasx_xvst(_sum2, tmp + 16, 0);
                __lasx_xvst(_sum3, tmp + 24, 0);
                store_output_block(top_blob, i + ii, 4, j + jj, 8, tmp, false, true);
                outptr0 += 8 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum0, outptr, 0);
                __lasx_xvst(_sum1, outptr + 8, 0);
                __lasx_xvst(_sum2, outptr + 16, 0);
                __lasx_xvst(_sum3, outptr + 24, 0);
            }

            outptr += 32;
        }
#endif
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum[8];

            if (k == 0)
            {
                for (int c = 0; c < 8; c++)
                {
                    _sum[c] = (__m128)__lsx_vreplgr2vr_w(0);
                }

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m128 _c0 = __lsx_vreplfr2vr_s(pCi[0]);
                        for (int c = 0; c < 8; c++)
                            _sum[c] = _c0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        __m128 _c0 = (__m128)__lsx_vld(pCi, 0);
                        for (int c = 0; c < 8; c++)
                            _sum[c] = _c0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        for (int c = 0; c < 8; c++)
                            _sum[c] = (__m128)__lsx_vld(pCi + c * 4, 0);
                        pCi += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        for (int c = 0; c < 8; c++)
                            _sum[c] = __lsx_vreplfr2vr_s(pCi[c]);
                        pCi += 8;
                    }
                }
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    _sum[c] = (__m128)__lsx_vld(outptr + c * 4, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                for (int c = 0; c < 8; c++)
                {
                    _sum[c] = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[c]), _pA, _sum[c]);
                }
                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                float block[32];
                for (int c = 0; c < 8; c++)
                    __lsx_vst((__m128i)_sum[c], block + c * 4, 0);
                store_output_block(top_blob, i + ii, 4, j + jj, 8, block, false, false);
                outptr0 += 8 * out_elempack;
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    __lsx_vst((__m128i)_sum[c], outptr + c * 4, 0);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        _sum1 = (__m128)__lsx_vld(pCi + 4, 0);
                        _sum2 = (__m128)__lsx_vld(pCi + 8, 0);
                        _sum3 = (__m128)__lsx_vld(pCi + 12, 0);
                        pCi += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        _sum1 = __lsx_vreplfr2vr_s(pCi[1]);
                        _sum2 = __lsx_vreplfr2vr_s(pCi[2]);
                        _sum3 = __lsx_vreplfr2vr_s(pCi[3]);
                        pCi += 4;
                    }
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                _sum0 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA, _sum0);
                _sum1 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA, _sum1);
                _sum2 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[2]), _pA, _sum2);
                _sum3 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[3]), _pA, _sum3);
                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    float tmp[16];
                    __lsx_vst((__m128i)_sum0, tmp, 0);
                    __lsx_vst((__m128i)_sum1, tmp + 4, 0);
                    __lsx_vst((__m128i)_sum2, tmp + 8, 0);
                    __lsx_vst((__m128i)_sum3, tmp + 12, 0);
                    store_output_block(top_blob, i + ii, 4, j + jj, 4, tmp, false, false);
                    outptr0 += 4;
                }
                else if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + 8, 0);
                    __lsx_vst((__m128i)_sum3, outptr0 + 12, 0);
                    outptr0 += 16;
                }
                else
                {
                    __m128 _r0 = _sum0;
                    __m128 _r1 = _sum1;
                    __m128 _r2 = _sum2;
                    __m128 _r3 = _sum3;
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, outptr0, 0);
                    __lsx_vst((__m128i)_r1, outptr0 + out_hstep, 0);
                    __lsx_vst((__m128i)_r2, outptr0 + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_r3, outptr0 + out_hstep * 3, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 8, 0);
                __lsx_vst((__m128i)_sum3, outptr + 12, 0);
            }

            outptr += 16;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        _sum1 = (__m128)__lsx_vld(pCi + 4, 0);
                        pCi += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        _sum1 = __lsx_vreplfr2vr_s(pCi[1]);
                        pCi += 2;
                    }
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                _sum0 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA, _sum0);
                _sum1 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA, _sum1);
                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    float tmp[8];
                    __lsx_vst((__m128i)_sum0, tmp, 0);
                    __lsx_vst((__m128i)_sum1, tmp + 4, 0);
                    store_output_block(top_blob, i + ii, 4, j + jj, 2, tmp, false, false);
                    outptr0 += 2;
                }
                else if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    outptr0 += 8;
                }
                else
                {
                    float tmp0[4];
                    float tmp1[4];
                    __lsx_vst((__m128i)_sum0, tmp0, 0);
                    __lsx_vst((__m128i)_sum1, tmp1, 0);
                    for (int r = 0; r < 4; r++)
                    {
                        outptr0[out_hstep * r] = tmp0[r];
                        outptr0[out_hstep * r + 1] = tmp1[r];
                    }
                    outptr0 += 2;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            }

            outptr += 8;
        }

        for (; jj < max_jj; jj += 1)
        {
            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        pCi += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        pCi += 1;
                    }
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                _sum0 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA, _sum0);
                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    float tmp[4];
                    __lsx_vst((__m128i)_sum0, tmp, 0);
                    store_output_block(top_blob, i + ii, 4, j + jj, 1, tmp, false, false);
                    outptr0 += 1;
                }
                else if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    outptr0 += 4;
                }
                else
                {
                    float tmp0[4];
                    __lsx_vst((__m128i)_sum0, tmp0, 0);
                    for (int r = 0; r < 4; r++)
                    {
                        outptr0[out_hstep * r] = tmp0[r];
                    }
                    outptr0 += 1;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;
        const float* pCi = pC;

        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)CT_tile + i + ii;
            if (broadcast_type_C == 4)
                pCi = (const float*)CT_tile + j;
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum00;
            __m256 _sum01;
            __m256 _sum10;
            __m256 _sum11;

            if (k == 0)
            {
                _sum00 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum01 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum10 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum11 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum00 = _c;
                        _sum01 = _c;
                        _sum10 = _c;
                        _sum11 = _c;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum01 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum10 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                        _sum11 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __attribute__((aligned(32))) float ctmp[32];
                        for (int r = 0; r < 2; r++)
                        {
                            for (int c = 0; c < 16; c++)
                            {
                                ctmp[r * 16 + c] = pCi[c * 2 + r];
                            }
                        }
                        _sum00 = (__m256)__lasx_xvld(ctmp, 0);
                        _sum01 = (__m256)__lasx_xvld(ctmp + 8, 0);
                        _sum10 = (__m256)__lasx_xvld(ctmp + 16, 0);
                        _sum11 = (__m256)__lasx_xvld(ctmp + 24, 0);
                        pCi += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m256 _c0 = (__m256)__lasx_xvld(pCi, 0);
                        __m256 _c1 = (__m256)__lasx_xvld(pCi + 8, 0);
                        _sum00 = _c0;
                        _sum01 = _c1;
                        _sum10 = _c0;
                        _sum11 = _c1;
                        pCi += 16;
                    }
                }
            }
            else
            {
                _sum00 = (__m256)__lasx_xvld(outptr, 0);
                _sum01 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum10 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum11 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _a0 = (__m256)__lasx_xvreplfr2vr_s(pA[0]);
                __m256 _a1 = (__m256)__lasx_xvreplfr2vr_s(pA[1]);
                _sum00 = __lasx_xvfmadd_s(_a0, _pB0, _sum00);
                _sum01 = __lasx_xvfmadd_s(_a0, _pB1, _sum01);
                _sum10 = __lasx_xvfmadd_s(_a1, _pB0, _sum10);
                _sum11 = __lasx_xvfmadd_s(_a1, _pB1, _sum11);
                pA += 2;
                pB += 16;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[32];
                __lasx_xvst(_sum00, tmp, 0);
                __lasx_xvst(_sum01, tmp + 8, 0);
                __lasx_xvst(_sum10, tmp + 16, 0);
                __lasx_xvst(_sum11, tmp + 24, 0);
                store_output_block(top_blob, i + ii, 2, j + jj, 16, tmp, false, true);
                outptr0 += 16 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum00, outptr, 0);
                __lasx_xvst(_sum01, outptr + 8, 0);
                __lasx_xvst(_sum10, outptr + 16, 0);
                __lasx_xvst(_sum11, outptr + 24, 0);
            }

            outptr += 32;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum0 = _c;
                        _sum1 = _c;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum1 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __attribute__((aligned(32))) float ctmp[16];
                        for (int r = 0; r < 2; r++)
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                ctmp[r * 8 + c] = pCi[c * 2 + r];
                            }
                        }
                        _sum0 = (__m256)__lasx_xvld(ctmp, 0);
                        _sum1 = (__m256)__lasx_xvld(ctmp + 8, 0);
                        pCi += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m256 _c = (__m256)__lasx_xvld(pCi, 0);
                        _sum0 = _c;
                        _sum1 = _c;
                        pCi += 8;
                    }
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB = (__m256)__lasx_xvld(pB, 0);
                _sum0 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[0]), _pB, _sum0);
                _sum1 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[1]), _pB, _sum1);
                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[16];
                __lasx_xvst(_sum0, tmp, 0);
                __lasx_xvst(_sum1, tmp + 8, 0);
                store_output_block(top_blob, i + ii, 2, j + jj, 8, tmp, false, true);
                outptr0 += 8 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum0, outptr, 0);
                __lasx_xvst(_sum1, outptr + 8, 0);
            }

            outptr += 16;
        }
#endif
        for (; jj + 7 < max_jj; jj += 8)
        {
            float sum0[8];
            float sum1[8];

            if (k == 0)
            {
                for (int c = 0; c < 8; c++)
                {
                    sum0[c] = 0.f;
                    sum1[c] = 0.f;
                }

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            sum0[c] = pCi[0];
                            sum1[c] = pCi[0];
                        }
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            sum0[c] = pCi[0];
                            sum1[c] = pCi[1];
                        }
                    }
                    if (broadcast_type_C == 3)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            sum0[c] = pCi[c * 2];
                            sum1[c] = pCi[c * 2 + 1];
                        }
                        pCi += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            sum0[c] = pCi[c];
                            sum1[c] = pCi[c];
                        }
                        pCi += 8;
                    }
                }
            }
            else
            {
                for (int c = 0; c < 8; c++)
                {
                    sum0[c] = outptr[c * 2];
                    sum1[c] = outptr[c * 2 + 1];
                }
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                const float a1 = pA[1];
                for (int c = 0; c < 8; c++)
                {
                    sum0[c] += a0 * pB[c];
                    sum1[c] += a1 * pB[c];
                }
                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                float block[16];
                for (int c = 0; c < 8; c++)
                {
                    block[c * 2] = sum0[c];
                    block[c * 2 + 1] = sum1[c];
                }
                store_output_block(top_blob, i + ii, 2, j + jj, 8, block, false, false);
                outptr0 += 8;
            }
            else
            {
                for (int c = 0; c < 8; c++)
                {
                    outptr[c * 2] = sum0[c];
                    outptr[c * 2 + 1] = sum1[c];
                }
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;
            float sum20;
            float sum21;
            float sum30;
            float sum31;

            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;
                sum20 = 0.f;
                sum21 = 0.f;
                sum30 = 0.f;
                sum31 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[0];
                        sum10 = sum00;
                        sum11 = sum00;
                        sum20 = sum00;
                        sum21 = sum00;
                        sum30 = sum00;
                        sum31 = sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[1];
                        sum10 = sum00;
                        sum11 = sum01;
                        sum20 = sum00;
                        sum21 = sum01;
                        sum30 = sum00;
                        sum31 = sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[1];
                        sum10 = pCi[2];
                        sum11 = pCi[3];
                        sum20 = pCi[4];
                        sum21 = pCi[5];
                        sum30 = pCi[6];
                        sum31 = pCi[7];
                        pCi += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pCi[0];
                        sum10 = pCi[1];
                        sum20 = pCi[2];
                        sum30 = pCi[3];
                        sum01 = sum00;
                        sum11 = sum10;
                        sum21 = sum20;
                        sum31 = sum30;
                        pCi += 4;
                    }
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
                sum20 = outptr[4];
                sum21 = outptr[5];
                sum30 = outptr[6];
                sum31 = outptr[7];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                const float a1 = pA[1];
                sum00 += a0 * pB[0];
                sum01 += a1 * pB[0];
                sum10 += a0 * pB[1];
                sum11 += a1 * pB[1];
                sum20 += a0 * pB[2];
                sum21 += a1 * pB[2];
                sum30 += a0 * pB[3];
                sum31 += a1 * pB[3];
                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr0[1] = sum10;
                outptr0[2] = sum20;
                outptr0[3] = sum30;
                outptr0[out_hstep] = sum01;
                outptr0[out_hstep + 1] = sum11;
                outptr0[out_hstep + 2] = sum21;
                outptr0[out_hstep + 3] = sum31;
                outptr0 += 4;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
                outptr[4] = sum20;
                outptr[5] = sum21;
                outptr[6] = sum30;
                outptr[7] = sum31;
            }

            outptr += 8;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[0];
                        sum10 = sum00;
                        sum11 = sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[1];
                        sum10 = sum00;
                        sum11 = sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[1];
                        sum10 = pCi[2];
                        sum11 = pCi[3];
                        pCi += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pCi[0];
                        sum10 = pCi[1];
                        sum01 = sum00;
                        sum11 = sum10;
                        pCi += 2;
                    }
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                const float a1 = pA[1];
                sum00 += a0 * pB[0];
                sum01 += a1 * pB[0];
                sum10 += a0 * pB[1];
                sum11 += a1 * pB[1];
                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr0[1] = sum10;
                outptr0[out_hstep] = sum01;
                outptr0[out_hstep + 1] = sum11;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
            }

            outptr += 4;
        }

        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pCi[0];
                        sum1 = broadcast_type_C == 0 ? pCi[0] : pCi[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[1];
                        pCi += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[0];
                        pCi += 1;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                const float a1 = pA[1];
                sum0 += a0 * pB[0];
                sum1 += a1 * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_hstep] = sum1;
                outptr0 += 1;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }

    for (; ii < max_ii; ii += 1)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;
        const float* pCi = pC;

        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)CT_tile + i + ii;
            if (broadcast_type_C == 4)
                pCi = (const float*)CT_tile + j;
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum0 = _c;
                        _sum1 = _c;
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = (__m256)__lasx_xvld(pCi, 0);
                        _sum1 = (__m256)__lasx_xvld(pCi + 8, 0);
                        pCi += 16;
                    }
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _a0 = (__m256)__lasx_xvreplfr2vr_s(pA[0]);
                _sum0 = __lasx_xvfmadd_s(_a0, _pB0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_a0, _pB1, _sum1);
                pA += 1;
                pB += 16;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[16];
                __lasx_xvst(_sum0, tmp, 0);
                __lasx_xvst(_sum1, tmp + 8, 0);
                store_output_block(top_blob, i + ii, 1, j + jj, 16, tmp, false, true);
                outptr0 += 16 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum0, outptr, 0);
                __lasx_xvst(_sum1, outptr + 8, 0);
            }

            outptr += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = (__m256)__lasx_xvld(pCi, 0);
                        pCi += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = (__m256)__lasx_xvld(pCi, 0);
                        pCi += 8;
                    }
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB = (__m256)__lasx_xvld(pB, 0);
                _sum0 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[0]), _pB, _sum0);
                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[8];
                __lasx_xvst(_sum0, tmp, 0);
                store_output_block(top_blob, i + ii, 1, j + jj, 8, tmp, false, true);
                outptr0 += 8 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum0, outptr, 0);
            }

            outptr += 8;
        }
#endif
        for (; jj + 7 < max_jj; jj += 8)
        {
            float sum[8];

            if (k == 0)
            {
                for (int c = 0; c < 8; c++)
                    sum[c] = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        for (int c = 0; c < 8; c++)
                            sum[c] = pCi[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        for (int c = 0; c < 8; c++)
                            sum[c] = pCi[c];
                        pCi += 8;
                    }
                }
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    sum[c] = outptr[c];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                for (int c = 0; c < 8; c++)
                    sum[c] += a0 * pB[c];
                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                store_output_block(top_blob, i + ii, 1, j + jj, 8, sum, false, false);
                outptr0 += 8;
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    outptr[c] = sum[c];
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0;
            float sum1;
            float sum2;
            float sum3;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;
                sum2 = 0.f;
                sum3 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[0];
                        sum2 = pCi[0];
                        sum3 = pCi[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[1];
                        sum2 = pCi[2];
                        sum3 = pCi[3];
                        pCi += 4;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                sum0 += a0 * pB[0];
                sum1 += a0 * pB[1];
                sum2 += a0 * pB[2];
                sum3 += a0 * pB[3];
                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0[2] = sum2;
                outptr0[3] = sum3;
                outptr0 += 4;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr[2] = sum2;
                outptr[3] = sum3;
            }

            outptr += 4;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[1];
                        pCi += 2;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                sum0 += a0 * pB[0];
                sum1 += a0 * pB[1];
                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        for (; jj < max_jj; jj += 1)
        {
            float sum0;

            if (k == 0)
            {
                sum0 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                        sum0 = pCi[0];
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pCi[0];
                        pCi += 1;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                sum0 += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0 += 1;
            }
            else
            {
                outptr[0] = sum0;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void get_optimal_tile_mnk(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / 3 / sizeof(float));

    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(float) / TILE_K);
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
    }

    if (nT > 1)
    {
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
    }

    if (constant_TILE_M > 0)
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
    if (constant_TILE_N > 0)
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
    if (constant_TILE_K > 0)
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
}

static bool support_fp32_tiled_gemm(const Mat& A, const Mat& B, int output_elempack)
{
    if (A.elembits() != 32 || B.elembits() != 32)
        return false;

    if (A.elempack != 1 && A.elempack != 4
#if __loongarch_asx
            && A.elempack != 8
#endif
       )
        return false;
    if (B.elempack != 1 && B.elempack != 4
#if __loongarch_asx
            && B.elempack != 8
#endif
       )
        return false;

    if (output_elempack != 0 && output_elempack != 1 && output_elempack != 4
#if __loongarch_asx
            && output_elempack != 8
#endif
       )
        return false;

    return true;
}

static int gemm_loongarch(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    int TILE_M;
    int TILE_N;
    int TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat ATX(TILE_K * TILE_M, nn_K, nT, 4u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;
    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 4u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min(N - j, TILE_N);
        const int max_kk = std::min(K - k, TILE_K);

        Mat BT_tile = BT.channel(ppj).row_range(ppk, 1);

        if (transB)
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        else
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const bool use_8row_kernel = use_8row_packed_kernel(top_blob.elempack);

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int M_local = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K_local = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
        const int max_ii = std::min(M_local - i, TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
                if (use_8row_kernel)
                    pack_A_tile_8row(C, topT_tile, i, max_ii, j, max_jj);
                else
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K_local; k += TILE_K)
            {
                const int max_kk = std::min(K_local - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (transA)
                    {
                        if (use_8row_kernel)
                            transpose_pack_A_tile_8row(A, AT_tile, i, max_ii, k, max_kk);
                        else
                            transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        if (use_8row_kernel)
                            pack_A_tile_8row(A, AT_tile, i, max_ii, k, max_kk);
                        else
                            pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                const bool k_end = !output_transpose && k + TILE_K >= K_local;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_AT_mips(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    int TILE_M;
    int TILE_N;
    int TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 4u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min(N - j, TILE_N);
        const int max_kk = std::min(K - k, TILE_K);

        Mat BT_tile = BT.channel(ppj).row_range(ppk, 1);

        if (transB)
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        else
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const bool use_8row_kernel = use_8row_packed_kernel(top_blob.elempack);

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
                if (use_8row_kernel)
                    pack_A_tile_8row(C, topT_tile, i, max_ii, j, max_jj);
                else
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                const bool k_end = !output_transpose && k + TILE_K >= K;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_BT_mips(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    int TILE_M;
    int TILE_N;
    int TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 4u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const bool use_8row_kernel = use_8row_packed_kernel(top_blob.elempack);

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int M_local = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K_local = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
        const int max_ii = std::min(M_local - i, TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
                if (use_8row_kernel)
                    pack_A_tile_8row(C, topT_tile, i, max_ii, j, max_jj);
                else
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K_local; k += TILE_K)
            {
                const int max_kk = std::min(K_local - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (transA)
                    {
                        if (use_8row_kernel)
                            transpose_pack_A_tile_8row(A, AT_tile, i, max_ii, k, max_kk);
                        else
                            transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        if (use_8row_kernel)
                            pack_A_tile_8row(A, AT_tile, i, max_ii, k, max_kk);
                        else
                            pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                const bool k_end = !output_transpose && k + TILE_K >= K_local;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_AT_BT_mips(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    int TILE_M;
    int TILE_N;
    int TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const bool use_8row_kernel = use_8row_packed_kernel(top_blob.elempack);

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
                if (use_8row_kernel)
                    pack_A_tile_8row(C, topT_tile, i, max_ii, j, max_jj);
                else
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                const bool k_end = !output_transpose && k + TILE_K >= K;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}
#endif // __loongarch_sx

int Gemm_loongarch::create_pipeline(const Option& opt)
{
#if __loongarch_sx
    AT_data.release();
    BT_data.release();
    CT_data.release();
    nT = 0;

    if (int8_scale_term)
        return 0;

    if (constantA && A_data.elembits() == 32 && support_fp32_tiled_gemm(A_data, A_data, output_elempack))
    {
        const int M = constantM;
        const int K = constantK;
        const bool use_8row_kernel = use_8row_packed_kernel(output_elempack);

        int TILE_M;
        int TILE_N;
        int TILE_K;
        get_optimal_tile_mnk(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_M = (M + TILE_M - 1) / TILE_M;
        const int nn_K = (K + TILE_K - 1) / TILE_K;

        AT_data.create(TILE_K * TILE_M, nn_K, nn_M, 4u, (Allocator*)0);
        if (AT_data.empty())
            return -100;

        const int nn_MK = nn_M * nn_K;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min(M - i, TILE_M);
            const int max_kk = std::min(K - k, TILE_K);

            Mat AT_tile = AT_data.channel(ppi).row_range(ppk, 1);

            if (transA)
            {
                if (use_8row_kernel)
                    transpose_pack_A_tile_8row(A_data, AT_tile, i, max_ii, k, max_kk);
                else
                    transpose_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                if (use_8row_kernel)
                    pack_A_tile_8row(A_data, AT_tile, i, max_ii, k, max_kk);
                else
                    pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
            }
        }
    }

    if (constantB && B_data.elembits() == 32 && support_fp32_tiled_gemm(B_data, B_data, output_elempack))
    {
        const int N = constantN;
        const int K = constantK;

        int TILE_M;
        int TILE_N;
        int TILE_K;
        get_optimal_tile_mnk(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_N = (N + TILE_N - 1) / TILE_N;
        const int nn_K = (K + TILE_K - 1) / TILE_K;

        BT_data.create(TILE_K * TILE_N, nn_K, nn_N, 4u, (Allocator*)0);
        if (BT_data.empty())
            return -100;

        const int nn_NK = nn_N * nn_K;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min(N - j, TILE_N);
            const int max_kk = std::min(K - k, TILE_K);

            Mat BT_tile = BT_data.channel(ppj).row_range(ppk, 1);

            if (transB)
                pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk);
            else
                transpose_pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk);
        }
    }

    if (constantC && constant_broadcast_type_C != -1 && C_data.elembits() == 32 && (C_data.elempack == 1 || C_data.elempack == 4))
    {
        CT_data = C_data;

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
    }

    if (!AT_data.empty() || !BT_data.empty() || !CT_data.empty())
        nT = opt.num_threads;
#endif // __loongarch_sx

    return 0;
}

int Gemm_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            int ret = unpack_or_cast_to_float32(bottom_blobs[i], bottom_blobs_unpacked[i], opt);
            if (ret != 0)
                return ret;
        }

        Mat A_data_unpacked = A_data;
        Mat B_data_unpacked = B_data;
        Mat C_data_unpacked = C_data;

        if (constantA)
        {
            int ret = unpack_or_cast_to_float32(A_data, A_data_unpacked, opt);
            if (ret != 0)
                return ret;
        }
        if (constantB)
        {
            int ret = unpack_or_cast_to_float32(B_data, B_data_unpacked, opt);
            if (ret != 0)
                return ret;
        }
        if (constantC && constant_broadcast_type_C != -1)
        {
            int ret = unpack_or_cast_to_float32(C_data, C_data_unpacked, opt);
            if (ret != 0)
                return ret;
        }

        Gemm gemm;
        gemm.alpha = alpha;
        gemm.beta = beta;
        gemm.transA = transA;
        gemm.transB = transB;
        gemm.constantA = constantA;
        gemm.constantB = constantB;
        gemm.constantC = constantC;
        gemm.constantM = constantM;
        gemm.constantN = constantN;
        gemm.constantK = constantK;
        gemm.constant_broadcast_type_C = constant_broadcast_type_C;
        gemm.output_N1M = output_N1M;
        gemm.output_elempack = 1;
        gemm.output_elemtype = output_elemtype;
        gemm.output_transpose = output_transpose;
        gemm.int8_scale_term = int8_scale_term;
        gemm.constant_TILE_M = constant_TILE_M;
        gemm.constant_TILE_N = constant_TILE_N;
        gemm.constant_TILE_K = constant_TILE_K;
        gemm.A_data = A_data_unpacked;
        gemm.B_data = B_data_unpacked;
        gemm.C_data = C_data_unpacked;
#if NCNN_INT8
        gemm.A_data_int8_scales = A_data_int8_scales;
        gemm.B_data_int8_scale = B_data_int8_scale;
#endif

        Option opt_int8 = opt;
        opt_int8.use_packing_layout = false;

        std::vector<Mat> top_blobs_unpacked(1);
        int ret = gemm.forward(bottom_blobs_unpacked, top_blobs_unpacked, opt_int8);
        if (ret != 0)
            return ret;

        return assign_or_copy_top_blob(top_blobs_unpacked[0], top_blobs[0]);
    }
#endif

#if __loongarch_sx
    if (output_elemtype == 0 || output_elemtype == 1)
    {
        Mat A_fp32 = constantA ? A_data : bottom_blobs[0];
        Mat B_fp32 = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

        if (support_fp32_tiled_gemm(A_fp32, B_fp32, output_elempack))
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
            Mat C_matrix_post;
            int broadcast_type_C = 0;
            if (constantC)
            {
                const bool need_matrix_post = constant_broadcast_type_C == 3 && output_transpose;
                const Mat& C_src = !need_matrix_post && !CT_data.empty() ? CT_data : C_data;
                if (!C_src.empty() && (C_src.elembits() != 32 || (C_src.elempack != 1 && C_src.elempack != 4)))
                    goto fallback_wrapper;

                if (CT_data.empty() || need_matrix_post)
                {
                    int ret = prepare_C_fp32(C_src, C, broadcast_type_C, M, N, beta, opt);
                    if (ret != 0)
                    {
                        if (ret == -100)
                            return -100;
                        goto fallback_wrapper;
                    }
                }
                else
                {
                    C = C_src;
                    broadcast_type_C = constant_broadcast_type_C;
                }
            }
            else
            {
                Mat C_src;
                if (constantA && constantB)
                    C_src = bottom_blobs.size() == 1 ? bottom_blobs[0] : Mat();
                else if (constantA || constantB)
                    C_src = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
                else
                    C_src = bottom_blobs.size() == 3 ? bottom_blobs[2] : Mat();

                if (!C_src.empty() && (C_src.elembits() != 32 || (C_src.elempack != 1 && C_src.elempack != 4)))
                    goto fallback_wrapper;

                int ret = prepare_C_fp32(C_src, C, broadcast_type_C, M, N, beta, opt);
                if (ret != 0)
                {
                    if (ret == -100)
                        return -100;
                    goto fallback_wrapper;
                }
            }

            if (broadcast_type_C == 3)
            {
                if (C.elempack != 1 || C.dims != 2)
                    goto fallback_wrapper;

                C_matrix_post = C;
                C = Mat();
                broadcast_type_C = 0;
            }

            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
                const int outh = output_transpose ? N : M;
                out_elempack =
#if __loongarch_asx
                    outh % 8 == 0 ? 8 :
#endif
                    outh % 4 == 0 ? 4 : 1;
            }
            if (output_elempack)
                out_elempack = output_elempack;

            Mat& top_blob = top_blobs[0];
            const size_t out_elemsize = 4u * out_elempack;
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

            const int _nT = nT ? nT : opt.num_threads;
            if (nT != 0 && opt.num_threads != nT)
            {
                NCNN_LOGE("opt.num_threads %d changed, gemm will use load-time value %d", opt.num_threads, nT);
            }

            int ret = 0;
            if (constantA && constantB)
            {
                if (AT_data.empty() || BT_data.empty())
                    goto fallback_wrapper;
                ret = gemm_AT_BT_mips(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
            }
            else if (constantA)
            {
                if (AT_data.empty())
                    goto fallback_wrapper;
                ret = gemm_AT_mips(AT_data, bottom_blobs[0], C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
            }
            else if (constantB)
            {
                if (BT_data.empty())
                    goto fallback_wrapper;
                ret = gemm_BT_mips(bottom_blobs[0], BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
            }
            else
            {
                ret = gemm_loongarch(bottom_blobs[0], bottom_blobs[1], C, top_blob, broadcast_type_C, transA, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
            }
            if (ret != 0)
                return ret;

            if (!C_matrix_post.empty())
            {
                add_matrix_C_fp32(C_matrix_post, top_blob, output_transpose);
            }

            if (alpha != 1.f)
            {
                const int size = top_blob.total() * out_elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < size; i++)
                {
                    top_blob[i] *= alpha;
                }
            }

            return 0;
        }
    }
#endif // __loongarch_sx

fallback_wrapper:
    std::vector<Mat> bottom_blobs_fp32(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        int ret = unpack_or_cast_to_float32(bottom_blobs[i], bottom_blobs_fp32[i], opt);
        if (ret != 0)
            return ret;
    }

    Mat A_data_fp32 = A_data;
    Mat B_data_fp32 = B_data;
    Mat C_data_fp32 = C_data;

    if (constantA)
    {
        int ret = unpack_or_cast_to_float32(A_data, A_data_fp32, opt);
        if (ret != 0)
            return ret;
    }
    if (constantB)
    {
        int ret = unpack_or_cast_to_float32(B_data, B_data_fp32, opt);
        if (ret != 0)
            return ret;
    }
    if (constantC && constant_broadcast_type_C != -1)
    {
        int ret = unpack_or_cast_to_float32(C_data, C_data_fp32, opt);
        if (ret != 0)
            return ret;
    }

    const Mat& A0 = constantA ? A_data_fp32 : bottom_blobs_fp32[0];
    const Mat& B0 = constantB ? B_data_fp32 : constantA ? bottom_blobs_fp32[0] : bottom_blobs_fp32[1];

    const int M = transA == 0 ? (A0.dims == 3 ? A0.c : A0.h) : A0.w;
    const int N = transB == 0 ? B0.w : (B0.dims == 3 ? B0.c : B0.h);

    int out_elempack = 1;
#if __loongarch_sx
    if (output_elempack == 0 && opt.use_packing_layout)
    {
        const int outh = output_transpose ? N : M;
        out_elempack = outh % 4 == 0 ? 4 : 1;
    }
    if (output_elempack == 4)
        out_elempack = 4;
#endif // __loongarch_sx

    Gemm gemm;
    gemm.alpha = alpha;
    gemm.beta = beta;
    gemm.transA = transA;
    gemm.transB = transB;
    gemm.constantA = constantA;
    gemm.constantB = constantB;
    gemm.constantC = constantC;
    gemm.constantM = constantM;
    gemm.constantN = constantN;
    gemm.constantK = constantK;
    gemm.constant_broadcast_type_C = constant_broadcast_type_C;
    gemm.output_N1M = output_N1M;
    gemm.output_elempack = 1;
    gemm.output_elemtype = 1;
    gemm.output_transpose = output_transpose;
    gemm.int8_scale_term = 0;
    gemm.constant_TILE_M = constant_TILE_M;
    gemm.constant_TILE_N = constant_TILE_N;
    gemm.constant_TILE_K = constant_TILE_K;
    gemm.A_data = A_data_fp32;
    gemm.B_data = B_data_fp32;
    gemm.C_data = C_data_fp32;

    std::vector<Mat> top_blobs_unpacked(1);
    int ret = gemm.forward(bottom_blobs_fp32, top_blobs_unpacked, opt);
    if (ret != 0)
        return ret;

    Mat top_blob_final;
    if (out_elempack == 4)
    {
        convert_packing(top_blobs_unpacked[0], top_blob_final, 4, opt);
        if (top_blob_final.empty())
            return -100;
    }
    else
    {
        top_blob_final = top_blobs_unpacked[0];
    }

    return assign_or_copy_top_blob(top_blob_final, top_blobs[0]);
}

} // namespace ncnn
