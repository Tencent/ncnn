// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

Gemm_riscv::Gemm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
    one_blob_only = false;
    support_inplace = false;

    nT = 0;
}

static void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif

    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    // NCNN_LOGE("pack_A_tile %d", elempack);

    float* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (elempack == packn)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * packn;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vle32_v_f32m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vlse32_v_f32m1(p0, A_hstep * sizeof(float), vl), vl);
                pp += packn;
                p0++;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat32m1_t v0 = __riscv_vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = __riscv_vle32_v_f32m1(p1, vl);
                __riscv_vsseg2e32_v_f32m1x2(pp, __riscv_vcreate_v_f32m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += packn;
                p1 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vle32_v_f32m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif

    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    // NCNN_LOGE("transpose_pack_A_tile %d", elempack);

    float* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (elempack == packn)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                // transposeNxN
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vse32_v_f32m1(pp, __riscv_vlse32_v_f32m1(p0 + l, packn * sizeof(float), vl), vl);
                    pp += packn;
                }
                p0 += A_hstep * packn;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vle32_v_f32m1(p0, vl), vl);
                pp += packn;
                p0 += A_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat32m1_t v0 = __riscv_vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = __riscv_vle32_v_f32m1(p0 + packn, vl);
                __riscv_vsseg2e32_v_f32m1x2(pp, __riscv_vcreate_v_f32m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += A_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
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
#if __riscv_vector
        if (elempack == packn)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vle32_v_f32m1(p0, vl), vl);
                pp += packn;
                p0 += A_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
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
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif

    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    // NCNN_LOGE("pack_B_tile %d", elempack);

    float* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + (packn - 1) < max_jj; jj += packn)
    {
        if (elempack == packn)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * packn;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vle32_v_f32m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vlse32_v_f32m1(p0, B_hstep * sizeof(float), vl), vl);
                pp += packn;
                p0++;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat32m1_t v0 = __riscv_vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = __riscv_vle32_v_f32m1(p1, vl);
                __riscv_vsseg2e32_v_f32m1x2(pp, __riscv_vcreate_v_f32m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += packn;
                p1 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vle32_v_f32m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif

    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    // NCNN_LOGE("transpose_pack_B_tile %d", elempack);

    float* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + (packn - 1) < max_jj; jj += packn)
    {
        if (elempack == packn)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                // transposeNxN
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vse32_v_f32m1(pp, __riscv_vlse32_v_f32m1(p0 + l, packn * sizeof(float), vl), vl);
                    pp += packn;
                }
                p0 += B_hstep * packn;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vle32_v_f32m1(p0, vl), vl);
                pp += packn;
                p0 += B_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat32m1_t v0 = __riscv_vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = __riscv_vle32_v_f32m1(p0 + packn, vl);
                __riscv_vsseg2e32_v_f32m1x2(pp, __riscv_vcreate_v_f32m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += B_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
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
#if __riscv_vector
        if (elempack == packn)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse32_v_f32m1(pp, __riscv_vle32_v_f32m1(p0, vl), vl);
                pp += packn;
                p0 += B_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
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
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif

    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    // NCNN_LOGE("transpose_unpack_output_tile %d", out_elempack);

    const float* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (out_elempack == packn)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                // transposeNxN
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vsse32_v_f32m1(p0 + l, packn * sizeof(float), __riscv_vle32_v_f32m1(pp, vl), vl);
                    pp += packn;
                }
                p0 += out_hstep * packn;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(pp, vl);
                __riscv_vse32_v_f32m1(p0, _r0, vl);
                pp += packn;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (out_elempack == packn)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                vfloat32m1x2_t _s0 = __riscv_vlseg2e32_v_f32m1x2(pp, vl);
                __riscv_vse32_v_f32m1(p0, __riscv_vget_v_f32m1x2_f32m1(_s0, 0), vl);
                __riscv_vse32_v_f32m1(p0 + packn, __riscv_vget_v_f32m1x2_f32m1(_s0, 1), vl);
                pp += packn * 2;
                p0 += out_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __riscv_vector
        if (out_elempack == packn)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(pp, vl);
                __riscv_vse32_v_f32m1(p0, _r0, vl);
                pp += packn;
                p0 += out_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif

    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + (packn - 1) < max_jj; jj += packn)
        {
            if (packn == 8)
            {
                vfloat32m1_t _sum0;
                vfloat32m1_t _sum1;
                vfloat32m1_t _sum2;
                vfloat32m1_t _sum3;
                vfloat32m1_t _sum4;
                vfloat32m1_t _sum5;
                vfloat32m1_t _sum6;
                vfloat32m1_t _sum7;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum5 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum6 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum7 = __riscv_vfmv_v_f_f32m1(0.f, vl);

                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                        {
                            _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                            _sum1 = _sum0;
                            _sum2 = _sum0;
                            _sum3 = _sum0;
                            _sum4 = _sum0;
                            _sum5 = _sum0;
                            _sum6 = _sum0;
                            _sum7 = _sum0;
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                            _sum1 = _sum0;
                            _sum2 = _sum0;
                            _sum3 = _sum0;
                            _sum4 = _sum0;
                            _sum5 = _sum0;
                            _sum6 = _sum0;
                            _sum7 = _sum0;
                        }
                        if (broadcast_type_C == 3)
                        {
                            _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                            _sum1 = __riscv_vle32_v_f32m1(pC + packn, vl);
                            _sum2 = __riscv_vle32_v_f32m1(pC + packn * 2, vl);
                            _sum3 = __riscv_vle32_v_f32m1(pC + packn * 3, vl);
                            _sum4 = __riscv_vle32_v_f32m1(pC + packn * 4, vl);
                            _sum5 = __riscv_vle32_v_f32m1(pC + packn * 5, vl);
                            _sum6 = __riscv_vle32_v_f32m1(pC + packn * 6, vl);
                            _sum7 = __riscv_vle32_v_f32m1(pC + packn * 7, vl);
                            pC += packn * 8;
                        }
                        if (broadcast_type_C == 4)
                        {
                            _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                            _sum1 = __riscv_vfmv_v_f_f32m1(pC[1], vl);
                            _sum2 = __riscv_vfmv_v_f_f32m1(pC[2], vl);
                            _sum3 = __riscv_vfmv_v_f_f32m1(pC[3], vl);
                            _sum4 = __riscv_vfmv_v_f_f32m1(pC[4], vl);
                            _sum5 = __riscv_vfmv_v_f_f32m1(pC[5], vl);
                            _sum6 = __riscv_vfmv_v_f_f32m1(pC[6], vl);
                            _sum7 = __riscv_vfmv_v_f_f32m1(pC[7], vl);
                            pC += 8;
                        }
                    }
                }
                else
                {
                    _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
                    _sum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
                    _sum2 = __riscv_vle32_v_f32m1(outptr + packn * 2, vl);
                    _sum3 = __riscv_vle32_v_f32m1(outptr + packn * 3, vl);
                    _sum4 = __riscv_vle32_v_f32m1(outptr + packn * 4, vl);
                    _sum5 = __riscv_vle32_v_f32m1(outptr + packn * 5, vl);
                    _sum6 = __riscv_vle32_v_f32m1(outptr + packn * 6, vl);
                    _sum7 = __riscv_vle32_v_f32m1(outptr + packn * 7, vl);
                }

                const float* pA = pAT;
                int kk = 0;
                for (; kk < max_kk; kk += 1)
                {
                    vfloat32m1_t _pA = __riscv_vle32_v_f32m1(pA, vl);
                    _sum0 = __riscv_vfmadd_vf_f32m1(_pA, pB[0], _sum0, vl);
                    _sum1 = __riscv_vfmadd_vf_f32m1(_pA, pB[1], _sum1, vl);
                    _sum2 = __riscv_vfmadd_vf_f32m1(_pA, pB[2], _sum2, vl);
                    _sum3 = __riscv_vfmadd_vf_f32m1(_pA, pB[3], _sum3, vl);
                    _sum4 = __riscv_vfmadd_vf_f32m1(_pA, pB[4], _sum4, vl);
                    _sum5 = __riscv_vfmadd_vf_f32m1(_pA, pB[5], _sum5, vl);
                    _sum6 = __riscv_vfmadd_vf_f32m1(_pA, pB[6], _sum6, vl);
                    _sum7 = __riscv_vfmadd_vf_f32m1(_pA, pB[7], _sum7, vl);
                    pA += packn;
                    pB += 8;
                }

                if (k_end)
                {
                    if (out_elempack == packn)
                    {
                        __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn * 4, _sum4, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn * 5, _sum5, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn * 6, _sum6, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn * 7, _sum7, vl);
                        outptr0 += packn * 8;
                    }
                    if (out_elempack == 1)
                    {
                        vfloat32m1x8_t _sum = __riscv_vcreate_v_f32m1x8(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        __riscv_vssseg8e32_v_f32m1x8(outptr0, out_hstep * sizeof(float), _sum, vl);
                        outptr0 += 8;
                    }
                }
                else
                {
                    __riscv_vse32_v_f32m1(outptr, _sum0, vl);
                    __riscv_vse32_v_f32m1(outptr + packn, _sum1, vl);
                    __riscv_vse32_v_f32m1(outptr + packn * 2, _sum2, vl);
                    __riscv_vse32_v_f32m1(outptr + packn * 3, _sum3, vl);
                    __riscv_vse32_v_f32m1(outptr + packn * 4, _sum4, vl);
                    __riscv_vse32_v_f32m1(outptr + packn * 5, _sum5, vl);
                    __riscv_vse32_v_f32m1(outptr + packn * 6, _sum6, vl);
                    __riscv_vse32_v_f32m1(outptr + packn * 7, _sum7, vl);
                }

                outptr += packn * 8;
            }
            else if (packn == 4)
            {
                vfloat32m1_t _sum0;
                vfloat32m1_t _sum1;
                vfloat32m1_t _sum2;
                vfloat32m1_t _sum3;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    _sum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);

                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                        {
                            _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                            _sum1 = _sum0;
                            _sum2 = _sum0;
                            _sum3 = _sum0;
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                            _sum1 = _sum0;
                            _sum2 = _sum0;
                            _sum3 = _sum0;
                        }
                        if (broadcast_type_C == 3)
                        {
                            _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                            _sum1 = __riscv_vle32_v_f32m1(pC + packn, vl);
                            _sum2 = __riscv_vle32_v_f32m1(pC + packn * 2, vl);
                            _sum3 = __riscv_vle32_v_f32m1(pC + packn * 3, vl);
                            pC += packn * 4;
                        }
                        if (broadcast_type_C == 4)
                        {
                            _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                            _sum1 = __riscv_vfmv_v_f_f32m1(pC[1], vl);
                            _sum2 = __riscv_vfmv_v_f_f32m1(pC[2], vl);
                            _sum3 = __riscv_vfmv_v_f_f32m1(pC[3], vl);
                            pC += 4;
                        }
                    }
                }
                else
                {
                    _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
                    _sum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
                    _sum2 = __riscv_vle32_v_f32m1(outptr + packn * 2, vl);
                    _sum3 = __riscv_vle32_v_f32m1(outptr + packn * 3, vl);
                }

                const float* pA = pAT;
                int kk = 0;
                for (; kk < max_kk; kk += 1)
                {
                    vfloat32m1_t _pA = __riscv_vle32_v_f32m1(pA, vl);
                    _sum0 = __riscv_vfmadd_vf_f32m1(_pA, pB[0], _sum0, vl);
                    _sum1 = __riscv_vfmadd_vf_f32m1(_pA, pB[1], _sum1, vl);
                    _sum2 = __riscv_vfmadd_vf_f32m1(_pA, pB[2], _sum2, vl);
                    _sum3 = __riscv_vfmadd_vf_f32m1(_pA, pB[3], _sum3, vl);
                    pA += packn;
                    pB += 4;
                }

                if (k_end)
                {
                    if (out_elempack == packn)
                    {
                        __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
                        __riscv_vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);
                        outptr0 += packn * 4;
                    }
                    if (out_elempack == 1)
                    {
                        vfloat32m1x4_t _sum = __riscv_vcreate_v_f32m1x4(_sum0, _sum1, _sum2, _sum3);
                        __riscv_vssseg4e32_v_f32m1x4(outptr0, out_hstep * sizeof(float), _sum, vl);
                        outptr0 += 4;
                    }
                }
                else
                {
                    __riscv_vse32_v_f32m1(outptr, _sum0, vl);
                    __riscv_vse32_v_f32m1(outptr + packn, _sum1, vl);
                    __riscv_vse32_v_f32m1(outptr + packn * 2, _sum2, vl);
                    __riscv_vse32_v_f32m1(outptr + packn * 3, _sum3, vl);
                }

                outptr += packn * 4;
            }
            else
            {
                NCNN_LOGE("unsupported vector length");
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                        _sum1 = __riscv_vle32_v_f32m1(pC + packn, vl);
                        pC += packn * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f32m1(pC[1], vl);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _sum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = __riscv_vle32_v_f32m1(pA, vl);

                _sum0 = __riscv_vfmadd_vf_f32m1(_pA, pB[0], _sum0, vl);
                _sum1 = __riscv_vfmadd_vf_f32m1(_pA, pB[1], _sum1, vl);

                pA += packn;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
                    outptr0 += packn * 2;
                }
                if (out_elempack == 1)
                {
                    vfloat32m1x2_t _sum = __riscv_vcreate_v_f32m1x2(_sum0, _sum1);
                    __riscv_vssseg2e32_v_f32m1x2(outptr0, out_hstep * sizeof(float), _sum, vl);
                    outptr0 += 2;
                }
            }
            else
            {
                __riscv_vse32_v_f32m1(outptr, _sum0, vl);
                __riscv_vse32_v_f32m1(outptr + packn, _sum1, vl);
            }

            outptr += packn * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat32m1_t _sum0;

            if (k == 0)
            {
                _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                        pC += packn;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = __riscv_vle32_v_f32m1(pA, vl);
                vfloat32m1_t _pB = __riscv_vfmv_v_f_f32m1(pB[0], vl);

                _sum0 = __riscv_vfmadd_vv_f32m1(_pA, _pB, _sum0, vl);

                pA += packn;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                    outptr0 += packn;
                }
                if (out_elempack == 1)
                {
                    __riscv_vsse32_v_f32m1(outptr0, out_hstep * sizeof(float), _sum0, vl);
                    outptr0++;
                }
            }
            else
            {
                __riscv_vse32_v_f32m1(outptr, _sum0, vl);
            }

            outptr += packn;
        }

        pAT += max_kk * packn;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + (packn - 1) < max_jj; jj += packn)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f32m1(pC[1], vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m1x2_t _s0 = __riscv_vlseg2e32_v_f32m1x2(pC, vl);
                        _sum0 = __riscv_vget_v_f32m1x2_f32m1(_s0, 0);
                        _sum1 = __riscv_vget_v_f32m1x2_f32m1(_s0, 1);
                        pC += packn * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                        _sum1 = _sum0;
                        pC += packn;
                    }
                }
            }
            else
            {
                vfloat32m1x2_t _s0 = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _sum0 = __riscv_vget_v_f32m1x2_f32m1(_s0, 0);
                _sum1 = __riscv_vget_v_f32m1x2_f32m1(_s0, 1);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB = __riscv_vle32_v_f32m1(pB, vl);

                _sum0 = __riscv_vfmadd_vf_f32m1(_pB, pA[0], _sum0, vl);
                _sum1 = __riscv_vfmadd_vf_f32m1(_pB, pA[1], _sum1, vl);

                pA += 2;
                pB += packn;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                    __riscv_vse32_v_f32m1(outptr0 + out_hstep, _sum1, vl);
                    outptr0 += packn;
                }
            }
            else
            {
                __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_sum0, _sum1), vl);
            }

            outptr += packn * 2;
        }
#endif // __riscv_vector
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

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[0];
                        sum11 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[0];
                        sum11 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[2];
                        sum11 = pC[3];
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[1];
                        sum11 = pC[1];
                        pC += 2;
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
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum10;
                    outptr0[out_hstep] = sum01;
                    outptr0[out_hstep + 1] = sum11;
                    outptr0 += 2;
                }
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

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
                    outptr0++;
                }
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

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + (packn - 1) < max_jj; jj += packn)
        {
            vfloat32m1_t _sum;

            if (k == 0)
            {
                _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = __riscv_vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = __riscv_vle32_v_f32m1(pC, vl);
                        pC += packn;
                    }
                }
            }
            else
            {
                _sum = __riscv_vle32_v_f32m1(outptr, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB = __riscv_vle32_v_f32m1(pB, vl);
                vfloat32m1_t _pA = __riscv_vfmv_v_f_f32m1(pA[0], vl);

                _sum = __riscv_vfmadd_vv_f32m1(_pA, _pB, _sum, vl);

                pA += 1;
                pB += packn;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum, vl);
                    outptr0 += packn;
                }
            }
            else
            {
                __riscv_vse32_v_f32m1(outptr, _sum, vl);
            }

            outptr += packn;
        }
#endif // __riscv_vector
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
                    outptr0 += 2;
                }
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
            float sum;

            if (k == 0)
            {
                sum = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum = outptr[0];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void get_optimal_tile_mnk(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / 3 / sizeof(float));

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#else
    const int packn = 4;
#endif

    TILE_M = std::max(packn, tile_size / packn * packn);
    TILE_N = std::max(packn, tile_size / packn * packn);
    TILE_K = std::max(8, tile_size / 8 * 8);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(float) / TILE_K);
            TILE_M = std::max(packn, tile_size / packn * packn);
            TILE_N = std::max(packn, tile_size / packn * packn);
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + (packn - 1)) / packn * packn);
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + (packn - 1)) / packn * packn);
    }

    if (nT > 1)
    {
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + (packn - 1)) / packn * packn);
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
        TILE_M = (constant_TILE_M + (packn - 1)) / packn * packn;
    }

    if (constant_TILE_N > 0)
    {
        TILE_N = (constant_TILE_N + (packn - 1)) / packn * packn;
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
    }
}

static int gemm_riscv(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;

    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);
    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 4u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;
    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    // pack B
#if TIME_TEST
    gettimeofday(&start_time, NULL);
#endif
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
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

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
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
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
                        transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                bool k_end = !output_transpose && k + TILE_K >= K;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_AT_riscv(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);
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
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());
        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);
                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_BT_riscv(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

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
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
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
                        transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                bool k_end = !output_transpose && k + TILE_K >= K;

                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_AT_BT_riscv(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;

                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

int Gemm_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1, bottom_blob);
    std::vector<Mat> top_blobs(1, top_blob);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Gemm_riscv::create_pipeline(const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        support_packing = false;
        return 0;
    }
#endif

    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_M = (M + TILE_M - 1) / TILE_M;

        AT_data.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 4u, (Allocator*)0);
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
                    transpose_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
                }
                else
                {
                    pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
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
        get_optimal_tile_mnk(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_N = (N + TILE_N - 1) / TILE_N;

        BT_data.create(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, (Allocator*)0);
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
                    pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk);
                }
                else
                {
                    transpose_pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk);
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
        const int packn = csrr_vlenb() / 4;

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

int Gemm_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return Gemm::forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif

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
    const int packn = csrr_vlenb() / 4;
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
    size_t out_elemsize = 4u * out_elempack;

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
        ret = gemm_AT_BT_riscv(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_riscv(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_riscv(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_riscv(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    if (ret != 0)
        return ret;

    // multiply top_blob with alpha
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

} // namespace ncnn
