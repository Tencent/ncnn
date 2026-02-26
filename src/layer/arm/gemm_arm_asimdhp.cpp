// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

#include "gemm_bf16s_fp16s.h"
#include "gemm_fp16s.h"

#if NCNN_INT8
#include "gemm_int8_fp16s.h"
#endif

static void gemm_transB_packed_tile_fp16sa(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const __fp16* pAT = AT_tile;
    const __fp16* pBT = BT_tile;
    const __fp16* pC = CT_tile;

    __fp16* outptr = topT_tile;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const __fp16*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const __fp16*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            float16x8_t _sum0;
            float16x8_t _sum1;
            float16x8_t _sum2;
            float16x8_t _sum3;
            float16x8_t _sum4;
            float16x8_t _sum5;
            float16x8_t _sum6;
            float16x8_t _sum7;
            float16x8_t _sum8;
            float16x8_t _sum9;
            float16x8_t _suma;
            float16x8_t _sumb;

            if (k == 0)
            {
                _sum0 = vdupq_n_f16(0.f);
                _sum1 = vdupq_n_f16(0.f);
                _sum2 = vdupq_n_f16(0.f);
                _sum3 = vdupq_n_f16(0.f);
                _sum4 = vdupq_n_f16(0.f);
                _sum5 = vdupq_n_f16(0.f);
                _sum6 = vdupq_n_f16(0.f);
                _sum7 = vdupq_n_f16(0.f);
                _sum8 = vdupq_n_f16(0.f);
                _sum9 = vdupq_n_f16(0.f);
                _suma = vdupq_n_f16(0.f);
                _sumb = vdupq_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdupq_n_f16(pC[0]);
                        _sum2 = vdupq_n_f16(pC[0]);
                        _sum3 = vdupq_n_f16(pC[0]);
                        _sum4 = vdupq_n_f16(pC[0]);
                        _sum5 = vdupq_n_f16(pC[0]);
                        _sum6 = vdupq_n_f16(pC[0]);
                        _sum7 = vdupq_n_f16(pC[0]);
                        _sum8 = vdupq_n_f16(pC[0]);
                        _sum9 = vdupq_n_f16(pC[0]);
                        _suma = vdupq_n_f16(pC[0]);
                        _sumb = vdupq_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f16(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                        _sum8 = _sum0;
                        _sum9 = _sum0;
                        _suma = _sum0;
                        _sumb = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f16(pC);
                        _sum1 = vld1q_f16(pC + 8);
                        _sum2 = vld1q_f16(pC + 8 * 2);
                        _sum3 = vld1q_f16(pC + 8 * 3);
                        _sum4 = vld1q_f16(pC + 8 * 4);
                        _sum5 = vld1q_f16(pC + 8 * 5);
                        _sum6 = vld1q_f16(pC + 8 * 6);
                        _sum7 = vld1q_f16(pC + 8 * 7);
                        _sum8 = vld1q_f16(pC + 8 * 8);
                        _sum9 = vld1q_f16(pC + 8 * 9);
                        _suma = vld1q_f16(pC + 8 * 10);
                        _sumb = vld1q_f16(pC + 8 * 11);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdupq_n_f16(pC[1]);
                        _sum2 = vdupq_n_f16(pC[2]);
                        _sum3 = vdupq_n_f16(pC[3]);
                        _sum4 = vdupq_n_f16(pC[4]);
                        _sum5 = vdupq_n_f16(pC[5]);
                        _sum6 = vdupq_n_f16(pC[6]);
                        _sum7 = vdupq_n_f16(pC[7]);
                        _sum8 = vdupq_n_f16(pC[8]);
                        _sum9 = vdupq_n_f16(pC[9]);
                        _suma = vdupq_n_f16(pC[10]);
                        _sumb = vdupq_n_f16(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1q_f16(outptr + 8 * 1);
                _sum2 = vld1q_f16(outptr + 8 * 2);
                _sum3 = vld1q_f16(outptr + 8 * 3);
                _sum4 = vld1q_f16(outptr + 8 * 4);
                _sum5 = vld1q_f16(outptr + 8 * 5);
                _sum6 = vld1q_f16(outptr + 8 * 6);
                _sum7 = vld1q_f16(outptr + 8 * 7);
                _sum8 = vld1q_f16(outptr + 8 * 8);
                _sum9 = vld1q_f16(outptr + 8 * 9);
                _suma = vld1q_f16(outptr + 8 * 10);
                _sumb = vld1q_f16(outptr + 8 * 11);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "ld1    {v3.8h}, [%0], #16      \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%1], #24 \n"
                    "fmla   %2.8h, v3.8h, v0.h[0]   \n"
                    "fmla   %3.8h, v3.8h, v0.h[1]   \n"
                    "fmla   %4.8h, v3.8h, v0.h[2]   \n"
                    "fmla   %5.8h, v3.8h, v0.h[3]   \n"
                    "fmla   %6.8h, v3.8h, v1.h[0]   \n"
                    "fmla   %7.8h, v3.8h, v1.h[1]   \n"
                    "fmla   %8.8h, v3.8h, v1.h[2]   \n"
                    "fmla   %9.8h, v3.8h, v1.h[3]   \n"
                    "fmla   %10.8h, v3.8h, v2.h[0]  \n"
                    "fmla   %11.8h, v3.8h, v2.h[1]  \n"
                    "fmla   %12.8h, v3.8h, v2.h[2]  \n"
                    "fmla   %13.8h, v3.8h, v2.h[3]  \n"
                    : "=r"(pA),
                    "=r"(pB),
                    "=w"(_sum0),
                    "=w"(_sum1),
                    "=w"(_sum2),
                    "=w"(_sum3),
                    "=w"(_sum4),
                    "=w"(_sum5),
                    "=w"(_sum6),
                    "=w"(_sum7),
                    "=w"(_sum8),
                    "=w"(_sum9),
                    "=w"(_suma),
                    "=w"(_sumb)
                    : "0"(pA),
                    "1"(pB),
                    "2"(_sum0),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3),
                    "6"(_sum4),
                    "7"(_sum5),
                    "8"(_sum6),
                    "9"(_sum7),
                    "10"(_sum8),
                    "11"(_sum9),
                    "12"(_suma),
                    "13"(_sumb)
                    : "memory", "v0", "v1", "v2", "v3");
#else
                float16x8_t _pA = vld1q_f16(pA);

                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);
                float16x4_t _pB2 = vld1_f16(pB + 8);

                _sum0 = vfmaq_lane_f16(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_lane_f16(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_lane_f16(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_lane_f16(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_lane_f16(_sum7, _pA, _pB1, 3);
                _sum8 = vfmaq_lane_f16(_sum8, _pA, _pB2, 0);
                _sum9 = vfmaq_lane_f16(_sum9, _pA, _pB2, 1);
                _suma = vfmaq_lane_f16(_suma, _pA, _pB2, 2);
                _sumb = vfmaq_lane_f16(_sumb, _pA, _pB2, 3);

                pA += 8;
                pB += 12;
#endif
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8 * 1, _sum1);
                    vst1q_f16(outptr0 + 8 * 2, _sum2);
                    vst1q_f16(outptr0 + 8 * 3, _sum3);
                    vst1q_f16(outptr0 + 8 * 4, _sum4);
                    vst1q_f16(outptr0 + 8 * 5, _sum5);
                    vst1q_f16(outptr0 + 8 * 6, _sum6);
                    vst1q_f16(outptr0 + 8 * 7, _sum7);
                    vst1q_f16(outptr0 + 8 * 8, _sum8);
                    vst1q_f16(outptr0 + 8 * 9, _sum9);
                    vst1q_f16(outptr0 + 8 * 10, _suma);
                    vst1q_f16(outptr0 + 8 * 11, _sumb);
                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, vget_low_f16(_sum0));
                    vst1_f16(outptr0 + 4, vget_low_f16(_sum1));
                    vst1_f16(outptr0 + 4 * 2, vget_low_f16(_sum2));
                    vst1_f16(outptr0 + 4 * 3, vget_low_f16(_sum3));
                    vst1_f16(outptr0 + 4 * 4, vget_low_f16(_sum4));
                    vst1_f16(outptr0 + 4 * 5, vget_low_f16(_sum5));
                    vst1_f16(outptr0 + 4 * 6, vget_low_f16(_sum6));
                    vst1_f16(outptr0 + 4 * 7, vget_low_f16(_sum7));
                    vst1_f16(outptr0 + 4 * 8, vget_low_f16(_sum8));
                    vst1_f16(outptr0 + 4 * 9, vget_low_f16(_sum9));
                    vst1_f16(outptr0 + 4 * 10, vget_low_f16(_suma));
                    vst1_f16(outptr0 + 4 * 11, vget_low_f16(_sumb));

                    vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum1));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 2, vget_high_f16(_sum2));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 3, vget_high_f16(_sum3));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 4, vget_high_f16(_sum4));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 5, vget_high_f16(_sum5));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 6, vget_high_f16(_sum6));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 7, vget_high_f16(_sum7));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 8, vget_high_f16(_sum8));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 9, vget_high_f16(_sum9));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 10, vget_high_f16(_suma));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 11, vget_high_f16(_sumb));

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + out_hstep * 1, _sum1);
                    vst1q_f16(outptr0 + out_hstep * 2, _sum2);
                    vst1q_f16(outptr0 + out_hstep * 3, _sum3);
                    vst1q_f16(outptr0 + out_hstep * 4, _sum4);
                    vst1q_f16(outptr0 + out_hstep * 5, _sum5);
                    vst1q_f16(outptr0 + out_hstep * 6, _sum6);
                    vst1q_f16(outptr0 + out_hstep * 7, _sum7);

                    transpose8x4_ph(_sum8, _sum9, _suma, _sumb);

                    vst1_f16(outptr0 + 8, vget_low_f16(_sum8));
                    vst1_f16(outptr0 + out_hstep * 1 + 8, vget_high_f16(_sum8));
                    vst1_f16(outptr0 + out_hstep * 2 + 8, vget_low_f16(_sum9));
                    vst1_f16(outptr0 + out_hstep * 3 + 8, vget_high_f16(_sum9));
                    vst1_f16(outptr0 + out_hstep * 4 + 8, vget_low_f16(_suma));
                    vst1_f16(outptr0 + out_hstep * 5 + 8, vget_high_f16(_suma));
                    vst1_f16(outptr0 + out_hstep * 6 + 8, vget_low_f16(_sumb));
                    vst1_f16(outptr0 + out_hstep * 7 + 8, vget_high_f16(_sumb));

                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f16(outptr, _sum0);
                vst1q_f16(outptr + 8 * 1, _sum1);
                vst1q_f16(outptr + 8 * 2, _sum2);
                vst1q_f16(outptr + 8 * 3, _sum3);
                vst1q_f16(outptr + 8 * 4, _sum4);
                vst1q_f16(outptr + 8 * 5, _sum5);
                vst1q_f16(outptr + 8 * 6, _sum6);
                vst1q_f16(outptr + 8 * 7, _sum7);
                vst1q_f16(outptr + 8 * 8, _sum8);
                vst1q_f16(outptr + 8 * 9, _sum9);
                vst1q_f16(outptr + 8 * 10, _suma);
                vst1q_f16(outptr + 8 * 11, _sumb);
            }

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float16x8_t _sum0;
            float16x8_t _sum1;
            float16x8_t _sum2;
            float16x8_t _sum3;
            float16x8_t _sum4;
            float16x8_t _sum5;
            float16x8_t _sum6;
            float16x8_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_f16(0.f);
                _sum1 = vdupq_n_f16(0.f);
                _sum2 = vdupq_n_f16(0.f);
                _sum3 = vdupq_n_f16(0.f);
                _sum4 = vdupq_n_f16(0.f);
                _sum5 = vdupq_n_f16(0.f);
                _sum6 = vdupq_n_f16(0.f);
                _sum7 = vdupq_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdupq_n_f16(pC[0]);
                        _sum2 = vdupq_n_f16(pC[0]);
                        _sum3 = vdupq_n_f16(pC[0]);
                        _sum4 = vdupq_n_f16(pC[0]);
                        _sum5 = vdupq_n_f16(pC[0]);
                        _sum6 = vdupq_n_f16(pC[0]);
                        _sum7 = vdupq_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f16(pC);
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
                        _sum0 = vld1q_f16(pC);
                        _sum1 = vld1q_f16(pC + 8);
                        _sum2 = vld1q_f16(pC + 8 * 2);
                        _sum3 = vld1q_f16(pC + 8 * 3);
                        _sum4 = vld1q_f16(pC + 8 * 4);
                        _sum5 = vld1q_f16(pC + 8 * 5);
                        _sum6 = vld1q_f16(pC + 8 * 6);
                        _sum7 = vld1q_f16(pC + 8 * 7);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdupq_n_f16(pC[1]);
                        _sum2 = vdupq_n_f16(pC[2]);
                        _sum3 = vdupq_n_f16(pC[3]);
                        _sum4 = vdupq_n_f16(pC[4]);
                        _sum5 = vdupq_n_f16(pC[5]);
                        _sum6 = vdupq_n_f16(pC[6]);
                        _sum7 = vdupq_n_f16(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1q_f16(outptr + 8 * 1);
                _sum2 = vld1q_f16(outptr + 8 * 2);
                _sum3 = vld1q_f16(outptr + 8 * 3);
                _sum4 = vld1q_f16(outptr + 8 * 4);
                _sum5 = vld1q_f16(outptr + 8 * 5);
                _sum6 = vld1q_f16(outptr + 8 * 6);
                _sum7 = vld1q_f16(outptr + 8 * 7);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pA = vld1q_f16(pA);

                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);

                _sum0 = vfmaq_lane_f16(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_lane_f16(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_lane_f16(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_lane_f16(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_lane_f16(_sum7, _pA, _pB1, 3);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8 * 1, _sum1);
                    vst1q_f16(outptr0 + 8 * 2, _sum2);
                    vst1q_f16(outptr0 + 8 * 3, _sum3);
                    vst1q_f16(outptr0 + 8 * 4, _sum4);
                    vst1q_f16(outptr0 + 8 * 5, _sum5);
                    vst1q_f16(outptr0 + 8 * 6, _sum6);
                    vst1q_f16(outptr0 + 8 * 7, _sum7);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, vget_low_f16(_sum0));
                    vst1_f16(outptr0 + 4, vget_low_f16(_sum1));
                    vst1_f16(outptr0 + 4 * 2, vget_low_f16(_sum2));
                    vst1_f16(outptr0 + 4 * 3, vget_low_f16(_sum3));
                    vst1_f16(outptr0 + 4 * 4, vget_low_f16(_sum4));
                    vst1_f16(outptr0 + 4 * 5, vget_low_f16(_sum5));
                    vst1_f16(outptr0 + 4 * 6, vget_low_f16(_sum6));
                    vst1_f16(outptr0 + 4 * 7, vget_low_f16(_sum7));

                    vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum1));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 2, vget_high_f16(_sum2));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 3, vget_high_f16(_sum3));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 4, vget_high_f16(_sum4));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 5, vget_high_f16(_sum5));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 6, vget_high_f16(_sum6));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 7, vget_high_f16(_sum7));

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + out_hstep * 1, _sum1);
                    vst1q_f16(outptr0 + out_hstep * 2, _sum2);
                    vst1q_f16(outptr0 + out_hstep * 3, _sum3);
                    vst1q_f16(outptr0 + out_hstep * 4, _sum4);
                    vst1q_f16(outptr0 + out_hstep * 5, _sum5);
                    vst1q_f16(outptr0 + out_hstep * 6, _sum6);
                    vst1q_f16(outptr0 + out_hstep * 7, _sum7);

                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f16(outptr, _sum0);
                vst1q_f16(outptr + 8 * 1, _sum1);
                vst1q_f16(outptr + 8 * 2, _sum2);
                vst1q_f16(outptr + 8 * 3, _sum3);
                vst1q_f16(outptr + 8 * 4, _sum4);
                vst1q_f16(outptr + 8 * 5, _sum5);
                vst1q_f16(outptr + 8 * 6, _sum6);
                vst1q_f16(outptr + 8 * 7, _sum7);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float16x8_t _sum0;
            float16x8_t _sum1;
            float16x8_t _sum2;
            float16x8_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_f16(0.f);
                _sum1 = vdupq_n_f16(0.f);
                _sum2 = vdupq_n_f16(0.f);
                _sum3 = vdupq_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdupq_n_f16(pC[0]);
                        _sum2 = vdupq_n_f16(pC[0]);
                        _sum3 = vdupq_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f16(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f16(pC);
                        _sum1 = vld1q_f16(pC + 8);
                        _sum2 = vld1q_f16(pC + 8 * 2);
                        _sum3 = vld1q_f16(pC + 8 * 3);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdupq_n_f16(pC[1]);
                        _sum2 = vdupq_n_f16(pC[2]);
                        _sum3 = vdupq_n_f16(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1q_f16(outptr + 8 * 1);
                _sum2 = vld1q_f16(outptr + 8 * 2);
                _sum3 = vld1q_f16(outptr + 8 * 3);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pA = vld1q_f16(pA);

                float16x4_t _pB0 = vld1_f16(pB);

                _sum0 = vfmaq_lane_f16(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _pA, _pB0, 3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8 * 1, _sum1);
                    vst1q_f16(outptr0 + 8 * 2, _sum2);
                    vst1q_f16(outptr0 + 8 * 3, _sum3);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, vget_low_f16(_sum0));
                    vst1_f16(outptr0 + 4, vget_low_f16(_sum1));
                    vst1_f16(outptr0 + 4 * 2, vget_low_f16(_sum2));
                    vst1_f16(outptr0 + 4 * 3, vget_low_f16(_sum3));

                    vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum1));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 2, vget_high_f16(_sum2));
                    vst1_f16(outptr0 + out_hstep * 4 + 4 * 3, vget_high_f16(_sum3));

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_ph(_sum0, _sum1, _sum2, _sum3);

                    vst1_f16(outptr0, vget_low_f16(_sum0));
                    vst1_f16(outptr0 + out_hstep * 1, vget_high_f16(_sum0));
                    vst1_f16(outptr0 + out_hstep * 2, vget_low_f16(_sum1));
                    vst1_f16(outptr0 + out_hstep * 3, vget_high_f16(_sum1));
                    vst1_f16(outptr0 + out_hstep * 4, vget_low_f16(_sum2));
                    vst1_f16(outptr0 + out_hstep * 5, vget_high_f16(_sum2));
                    vst1_f16(outptr0 + out_hstep * 6, vget_low_f16(_sum3));
                    vst1_f16(outptr0 + out_hstep * 7, vget_high_f16(_sum3));

                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f16(outptr, _sum0);
                vst1q_f16(outptr + 8 * 1, _sum1);
                vst1q_f16(outptr + 8 * 2, _sum2);
                vst1q_f16(outptr + 8 * 3, _sum3);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float16x8_t _sum0;
            float16x8_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f16(0.f);
                _sum1 = vdupq_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdupq_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f16(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f16(pC);
                        _sum1 = vld1q_f16(pC + 8);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdupq_n_f16(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1q_f16(outptr + 8);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pA = vld1q_f16(pA);

                float16x8_t _pB0 = vdupq_n_f16(pB[0]);
                float16x8_t _pB1 = vdupq_n_f16(pB[1]);

                _sum0 = vfmaq_f16(_sum0, _pA, _pB0);
                _sum1 = vfmaq_f16(_sum1, _pA, _pB1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8, _sum1);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, vget_low_f16(_sum0));
                    vst1_f16(outptr0 + 4, vget_low_f16(_sum1));

                    vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum1));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    __fp16 sum0[8];
                    __fp16 sum1[8];
                    vst1q_f16(sum0, _sum0);
                    vst1q_f16(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_f16(outptr, _sum0);
                vst1q_f16(outptr + 8, _sum1);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            float16x8_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f16(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f16(pC);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pA = vld1q_f16(pA);

                float16x8_t _pB = vdupq_n_f16(pB[0]);

                _sum0 = vfmaq_f16(_sum0, _pA, _pB);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _sum0);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, vget_low_f16(_sum0));
                    vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    __fp16 sum0[8];
                    vst1q_f16(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f16(outptr, _sum0);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const __fp16*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const __fp16*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            float16x4_t _sum0;
            float16x4_t _sum1;
            float16x4_t _sum2;
            float16x4_t _sum3;
            float16x4_t _sum4;
            float16x4_t _sum5;
            float16x4_t _sum6;
            float16x4_t _sum7;
            float16x4_t _sum8;
            float16x4_t _sum9;
            float16x4_t _suma;
            float16x4_t _sumb;

            if (k == 0)
            {
                _sum0 = vdup_n_f16(0.f);
                _sum1 = vdup_n_f16(0.f);
                _sum2 = vdup_n_f16(0.f);
                _sum3 = vdup_n_f16(0.f);
                _sum4 = vdup_n_f16(0.f);
                _sum5 = vdup_n_f16(0.f);
                _sum6 = vdup_n_f16(0.f);
                _sum7 = vdup_n_f16(0.f);
                _sum8 = vdup_n_f16(0.f);
                _sum9 = vdup_n_f16(0.f);
                _suma = vdup_n_f16(0.f);
                _sumb = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[0]);
                        _sum2 = vdup_n_f16(pC[0]);
                        _sum3 = vdup_n_f16(pC[0]);
                        _sum4 = vdup_n_f16(pC[0]);
                        _sum5 = vdup_n_f16(pC[0]);
                        _sum6 = vdup_n_f16(pC[0]);
                        _sum7 = vdup_n_f16(pC[0]);
                        _sum8 = vdup_n_f16(pC[0]);
                        _sum9 = vdup_n_f16(pC[0]);
                        _suma = vdup_n_f16(pC[0]);
                        _sumb = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                        _sum8 = _sum0;
                        _sum9 = _sum0;
                        _suma = _sum0;
                        _sumb = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = vld1_f16(pC + 4);
                        _sum2 = vld1_f16(pC + 8);
                        _sum3 = vld1_f16(pC + 12);
                        _sum4 = vld1_f16(pC + 16);
                        _sum5 = vld1_f16(pC + 20);
                        _sum6 = vld1_f16(pC + 24);
                        _sum7 = vld1_f16(pC + 28);
                        _sum8 = vld1_f16(pC + 32);
                        _sum9 = vld1_f16(pC + 36);
                        _suma = vld1_f16(pC + 40);
                        _sumb = vld1_f16(pC + 44);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[1]);
                        _sum2 = vdup_n_f16(pC[2]);
                        _sum3 = vdup_n_f16(pC[3]);
                        _sum4 = vdup_n_f16(pC[4]);
                        _sum5 = vdup_n_f16(pC[5]);
                        _sum6 = vdup_n_f16(pC[6]);
                        _sum7 = vdup_n_f16(pC[7]);
                        _sum8 = vdup_n_f16(pC[8]);
                        _sum9 = vdup_n_f16(pC[9]);
                        _suma = vdup_n_f16(pC[10]);
                        _sumb = vdup_n_f16(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vld1_f16(outptr);
                _sum1 = vld1_f16(outptr + 4 * 1);
                _sum2 = vld1_f16(outptr + 4 * 2);
                _sum3 = vld1_f16(outptr + 4 * 3);
                _sum4 = vld1_f16(outptr + 4 * 4);
                _sum5 = vld1_f16(outptr + 4 * 5);
                _sum6 = vld1_f16(outptr + 4 * 6);
                _sum7 = vld1_f16(outptr + 4 * 7);
                _sum8 = vld1_f16(outptr + 4 * 8);
                _sum9 = vld1_f16(outptr + 4 * 9);
                _suma = vld1_f16(outptr + 4 * 10);
                _sumb = vld1_f16(outptr + 4 * 11);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pA = vld1_f16(pA);
                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);
                float16x4_t _pB2 = vld1_f16(pB + 8);

                _sum0 = vfma_lane_f16(_sum0, _pA, _pB0, 0);
                _sum1 = vfma_lane_f16(_sum1, _pA, _pB0, 1);
                _sum2 = vfma_lane_f16(_sum2, _pA, _pB0, 2);
                _sum3 = vfma_lane_f16(_sum3, _pA, _pB0, 3);
                _sum4 = vfma_lane_f16(_sum4, _pA, _pB1, 0);
                _sum5 = vfma_lane_f16(_sum5, _pA, _pB1, 1);
                _sum6 = vfma_lane_f16(_sum6, _pA, _pB1, 2);
                _sum7 = vfma_lane_f16(_sum7, _pA, _pB1, 3);
                _sum8 = vfma_lane_f16(_sum8, _pA, _pB2, 0);
                _sum9 = vfma_lane_f16(_sum9, _pA, _pB2, 1);
                _suma = vfma_lane_f16(_suma, _pA, _pB2, 2);
                _sumb = vfma_lane_f16(_sumb, _pA, _pB2, 3);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 4, _sum1);
                    vst1_f16(outptr0 + 4 * 2, _sum2);
                    vst1_f16(outptr0 + 4 * 3, _sum3);
                    vst1_f16(outptr0 + 4 * 4, _sum4);
                    vst1_f16(outptr0 + 4 * 5, _sum5);
                    vst1_f16(outptr0 + 4 * 6, _sum6);
                    vst1_f16(outptr0 + 4 * 7, _sum7);
                    vst1_f16(outptr0 + 4 * 8, _sum8);
                    vst1_f16(outptr0 + 4 * 9, _sum9);
                    vst1_f16(outptr0 + 4 * 10, _suma);
                    vst1_f16(outptr0 + 4 * 11, _sumb);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose4x12_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 4, _sum1);
                    vst1_f16(outptr0 + 8, _sum2);
                    vst1_f16(outptr0 + out_hstep, _sum3);
                    vst1_f16(outptr0 + out_hstep + 4, _sum4);
                    vst1_f16(outptr0 + out_hstep + 8, _sum5);
                    vst1_f16(outptr0 + out_hstep * 2, _sum6);
                    vst1_f16(outptr0 + out_hstep * 2 + 4, _sum7);
                    vst1_f16(outptr0 + out_hstep * 2 + 8, _sum8);
                    vst1_f16(outptr0 + out_hstep * 3, _sum9);
                    vst1_f16(outptr0 + out_hstep * 3 + 4, _suma);
                    vst1_f16(outptr0 + out_hstep * 3 + 8, _sumb);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
                vst1_f16(outptr + 4 * 2, _sum2);
                vst1_f16(outptr + 4 * 3, _sum3);
                vst1_f16(outptr + 4 * 4, _sum4);
                vst1_f16(outptr + 4 * 5, _sum5);
                vst1_f16(outptr + 4 * 6, _sum6);
                vst1_f16(outptr + 4 * 7, _sum7);
                vst1_f16(outptr + 4 * 8, _sum8);
                vst1_f16(outptr + 4 * 9, _sum9);
                vst1_f16(outptr + 4 * 10, _suma);
                vst1_f16(outptr + 4 * 11, _sumb);
            }

            outptr += 48;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float16x4_t _sum0;
            float16x4_t _sum1;
            float16x4_t _sum2;
            float16x4_t _sum3;
            float16x4_t _sum4;
            float16x4_t _sum5;
            float16x4_t _sum6;
            float16x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdup_n_f16(0.f);
                _sum1 = vdup_n_f16(0.f);
                _sum2 = vdup_n_f16(0.f);
                _sum3 = vdup_n_f16(0.f);
                _sum4 = vdup_n_f16(0.f);
                _sum5 = vdup_n_f16(0.f);
                _sum6 = vdup_n_f16(0.f);
                _sum7 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[0]);
                        _sum2 = vdup_n_f16(pC[0]);
                        _sum3 = vdup_n_f16(pC[0]);
                        _sum4 = vdup_n_f16(pC[0]);
                        _sum5 = vdup_n_f16(pC[0]);
                        _sum6 = vdup_n_f16(pC[0]);
                        _sum7 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1_f16(pC);
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
                        _sum0 = vld1_f16(pC);
                        _sum1 = vld1_f16(pC + 4);
                        _sum2 = vld1_f16(pC + 8);
                        _sum3 = vld1_f16(pC + 12);
                        _sum4 = vld1_f16(pC + 16);
                        _sum5 = vld1_f16(pC + 20);
                        _sum6 = vld1_f16(pC + 24);
                        _sum7 = vld1_f16(pC + 28);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[1]);
                        _sum2 = vdup_n_f16(pC[2]);
                        _sum3 = vdup_n_f16(pC[3]);
                        _sum4 = vdup_n_f16(pC[4]);
                        _sum5 = vdup_n_f16(pC[5]);
                        _sum6 = vdup_n_f16(pC[6]);
                        _sum7 = vdup_n_f16(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vld1_f16(outptr);
                _sum1 = vld1_f16(outptr + 4 * 1);
                _sum2 = vld1_f16(outptr + 4 * 2);
                _sum3 = vld1_f16(outptr + 4 * 3);
                _sum4 = vld1_f16(outptr + 4 * 4);
                _sum5 = vld1_f16(outptr + 4 * 5);
                _sum6 = vld1_f16(outptr + 4 * 6);
                _sum7 = vld1_f16(outptr + 4 * 7);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pA = vld1_f16(pA);
                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);

                _sum0 = vfma_lane_f16(_sum0, _pA, _pB0, 0);
                _sum1 = vfma_lane_f16(_sum1, _pA, _pB0, 1);
                _sum2 = vfma_lane_f16(_sum2, _pA, _pB0, 2);
                _sum3 = vfma_lane_f16(_sum3, _pA, _pB0, 3);
                _sum4 = vfma_lane_f16(_sum4, _pA, _pB1, 0);
                _sum5 = vfma_lane_f16(_sum5, _pA, _pB1, 1);
                _sum6 = vfma_lane_f16(_sum6, _pA, _pB1, 2);
                _sum7 = vfma_lane_f16(_sum7, _pA, _pB1, 3);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 4, _sum1);
                    vst1_f16(outptr0 + 4 * 2, _sum2);
                    vst1_f16(outptr0 + 4 * 3, _sum3);
                    vst1_f16(outptr0 + 4 * 4, _sum4);
                    vst1_f16(outptr0 + 4 * 5, _sum5);
                    vst1_f16(outptr0 + 4 * 6, _sum6);
                    vst1_f16(outptr0 + 4 * 7, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x8_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 4, _sum1);
                    vst1_f16(outptr0 + out_hstep, _sum2);
                    vst1_f16(outptr0 + out_hstep + 4, _sum3);
                    vst1_f16(outptr0 + out_hstep * 2, _sum4);
                    vst1_f16(outptr0 + out_hstep * 2 + 4, _sum5);
                    vst1_f16(outptr0 + out_hstep * 3, _sum6);
                    vst1_f16(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
                vst1_f16(outptr + 4 * 2, _sum2);
                vst1_f16(outptr + 4 * 3, _sum3);
                vst1_f16(outptr + 4 * 4, _sum4);
                vst1_f16(outptr + 4 * 5, _sum5);
                vst1_f16(outptr + 4 * 6, _sum6);
                vst1_f16(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float16x4_t _sum0;
            float16x4_t _sum1;
            float16x4_t _sum2;
            float16x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdup_n_f16(0.f);
                _sum1 = vdup_n_f16(0.f);
                _sum2 = vdup_n_f16(0.f);
                _sum3 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[0]);
                        _sum2 = vdup_n_f16(pC[0]);
                        _sum3 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = vld1_f16(pC + 4);
                        _sum2 = vld1_f16(pC + 8);
                        _sum3 = vld1_f16(pC + 12);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[1]);
                        _sum2 = vdup_n_f16(pC[2]);
                        _sum3 = vdup_n_f16(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = vld1_f16(outptr);
                _sum1 = vld1_f16(outptr + 4 * 1);
                _sum2 = vld1_f16(outptr + 4 * 2);
                _sum3 = vld1_f16(outptr + 4 * 3);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pA = vld1_f16(pA);
                float16x4_t _pB = vld1_f16(pB);

                _sum0 = vfma_lane_f16(_sum0, _pA, _pB, 0);
                _sum1 = vfma_lane_f16(_sum1, _pA, _pB, 1);
                _sum2 = vfma_lane_f16(_sum2, _pA, _pB, 2);
                _sum3 = vfma_lane_f16(_sum3, _pA, _pB, 3);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 4, _sum1);
                    vst1_f16(outptr0 + 4 * 2, _sum2);
                    vst1_f16(outptr0 + 4 * 3, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ph(_sum0, _sum1, _sum2, _sum3);

                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + out_hstep * 1, _sum1);
                    vst1_f16(outptr0 + out_hstep * 2, _sum2);
                    vst1_f16(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
                vst1_f16(outptr + 4 * 2, _sum2);
                vst1_f16(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float16x4_t _sum0;
            float16x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdup_n_f16(0.f);
                _sum1 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = vld1_f16(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = vld1_f16(outptr);
                _sum1 = vld1_f16(outptr + 4);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pA = vld1_f16(pA);
                float16x4_t _pB0 = vdup_n_f16(pB[0]);
                float16x4_t _pB1 = vdup_n_f16(pB[1]);

                _sum0 = vfma_f16(_sum0, _pA, _pB0);
                _sum1 = vfma_f16(_sum1, _pA, _pB1);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    __fp16 sum0[4];
                    __fp16 sum1[4];
                    vst1_f16(sum0, _sum0);
                    vst1_f16(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float16x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1_f16(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1_f16(pC);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = vld1_f16(outptr);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pA = vld1_f16(pA);
                float16x4_t _pB = vdup_n_f16(pB[0]);

                _sum0 = vfma_f16(_sum0, _pA, _pB);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    __fp16 sum0[4];
                    vst1_f16(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                vst1_f16(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const __fp16*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const __fp16*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            float16x4_t _sum00;
            float16x4_t _sum01;
            float16x4_t _sum02;
            float16x4_t _sum10;
            float16x4_t _sum11;
            float16x4_t _sum12;

            if (k == 0)
            {
                _sum00 = vdup_n_f16(0.f);
                _sum01 = vdup_n_f16(0.f);
                _sum02 = vdup_n_f16(0.f);
                _sum10 = vdup_n_f16(0.f);
                _sum11 = vdup_n_f16(0.f);
                _sum12 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdup_n_f16(pC[0]);
                        _sum01 = _sum00;
                        _sum02 = _sum00;
                        _sum10 = _sum00;
                        _sum11 = _sum00;
                        _sum12 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdup_n_f16(pC[0]);
                        _sum01 = _sum00;
                        _sum02 = _sum00;
                        _sum10 = vdup_n_f16(pC[1]);
                        _sum11 = _sum10;
                        _sum12 = _sum10;
                    }
                    if (broadcast_type_C == 3)
                    {
                        float16x4x2_t _tmp01 = vld2_f16(pC);
                        float16x4x2_t _tmp23 = vld2_f16(pC + 8);
                        float16x4x2_t _tmp45 = vld2_f16(pC + 16);
                        _sum00 = _tmp01.val[0];
                        _sum01 = _tmp23.val[0];
                        _sum02 = _tmp45.val[0];
                        _sum10 = _tmp01.val[1];
                        _sum11 = _tmp23.val[1];
                        _sum12 = _tmp45.val[1];
                        pC += 24;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vld1_f16(pC);
                        _sum01 = vld1_f16(pC + 4);
                        _sum02 = vld1_f16(pC + 8);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum12 = _sum02;
                        pC += 12;
                    }
                }
            }
            else
            {
                float16x4x2_t _tmp01 = vld2_f16(outptr);
                float16x4x2_t _tmp23 = vld2_f16(outptr + 8);
                float16x4x2_t _tmp45 = vld2_f16(outptr + 16);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum02 = _tmp45.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
                _sum12 = _tmp45.val[1];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);
                float16x4_t _pB2 = vld1_f16(pB + 8);

                float16x4_t _pA0 = vdup_n_f16(pA[0]);
                float16x4_t _pA1 = vdup_n_f16(pA[1]);

                _sum00 = vfma_f16(_sum00, _pB0, _pA0);
                _sum01 = vfma_f16(_sum01, _pB1, _pA0);
                _sum02 = vfma_f16(_sum02, _pB2, _pA0);
                _sum10 = vfma_f16(_sum10, _pB0, _pA1);
                _sum11 = vfma_f16(_sum11, _pB1, _pA1);
                _sum12 = vfma_f16(_sum12, _pB2, _pA1);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_f16(outptr0, _sum00);
                    vst1_f16(outptr0 + 4, _sum01);
                    vst1_f16(outptr0 + 8, _sum02);
                    vst1_f16(outptr0 + out_hstep, _sum10);
                    vst1_f16(outptr0 + out_hstep + 4, _sum11);
                    vst1_f16(outptr0 + out_hstep + 8, _sum12);
                    outptr0 += 12;
                }
            }
            else
            {
                float16x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float16x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                float16x4x2_t _tmp45;
                _tmp45.val[0] = _sum02;
                _tmp45.val[1] = _sum12;
                vst2_f16(outptr, _tmp01);
                vst2_f16(outptr + 8, _tmp23);
                vst2_f16(outptr + 16, _tmp45);
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float16x4_t _sum00;
            float16x4_t _sum01;
            float16x4_t _sum10;
            float16x4_t _sum11;

            if (k == 0)
            {
                _sum00 = vdup_n_f16(0.f);
                _sum01 = vdup_n_f16(0.f);
                _sum10 = vdup_n_f16(0.f);
                _sum11 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdup_n_f16(pC[0]);
                        _sum01 = vdup_n_f16(pC[0]);
                        _sum10 = vdup_n_f16(pC[0]);
                        _sum11 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdup_n_f16(pC[0]);
                        _sum01 = vdup_n_f16(pC[0]);
                        _sum10 = vdup_n_f16(pC[1]);
                        _sum11 = vdup_n_f16(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float16x4x2_t _tmp01 = vld2_f16(pC);
                        float16x4x2_t _tmp23 = vld2_f16(pC + 8);
                        _sum00 = _tmp01.val[0];
                        _sum01 = _tmp23.val[0];
                        _sum10 = _tmp01.val[1];
                        _sum11 = _tmp23.val[1];
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vld1_f16(pC);
                        _sum01 = vld1_f16(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        pC += 8;
                    }
                }
            }
            else
            {
                float16x4x2_t _tmp01 = vld2_f16(outptr);
                float16x4x2_t _tmp23 = vld2_f16(outptr + 8);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);

                float16x4_t _pA0 = vdup_n_f16(pA[0]);
                float16x4_t _pA1 = vdup_n_f16(pA[1]);

                _sum00 = vfma_f16(_sum00, _pB0, _pA0);
                _sum01 = vfma_f16(_sum01, _pB1, _pA0);
                _sum10 = vfma_f16(_sum10, _pB0, _pA1);
                _sum11 = vfma_f16(_sum11, _pB1, _pA1);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_f16(outptr0, _sum00);
                    vst1_f16(outptr0 + 4, _sum01);
                    vst1_f16(outptr0 + out_hstep, _sum10);
                    vst1_f16(outptr0 + out_hstep + 4, _sum11);
                    outptr0 += 8;
                }
            }
            else
            {
                float16x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float16x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                vst2_f16(outptr, _tmp01);
                vst2_f16(outptr + 8, _tmp23);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float16x4_t _sum0;
            float16x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdup_n_f16(0.f);
                _sum1 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float16x4x2_t _tmp01 = vld2_f16(pC);
                        _sum0 = _tmp01.val[0];
                        _sum1 = _tmp01.val[1];
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
            }
            else
            {
                float16x4x2_t _tmp01 = vld2_f16(outptr);
                _sum0 = _tmp01.val[0];
                _sum1 = _tmp01.val[1];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pB = vld1_f16(pB);

                float16x4_t _pA0 = vdup_n_f16(pA[0]);
                float16x4_t _pA1 = vdup_n_f16(pA[1]);

                _sum0 = vfma_f16(_sum0, _pB, _pA0);
                _sum1 = vfma_f16(_sum1, _pB, _pA1);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + out_hstep, _sum1);
                    outptr0 += 4;
                }
            }
            else
            {
                float16x4x2_t _tmp01;
                _tmp01.val[0] = _sum0;
                _tmp01.val[1] = _sum1;
                vst2_f16(outptr, _tmp01);
            }

            outptr += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __fp16 sum00;
            __fp16 sum01;
            __fp16 sum10;
            __fp16 sum11;

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

            const __fp16* pA = pAT;
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
            __fp16 sum0;
            __fp16 sum1;

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

            const __fp16* pA = pAT;
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
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const __fp16*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const __fp16*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            float16x4_t _sum0;
            float16x4_t _sum1;
            float16x4_t _sum2;

            if (k == 0)
            {
                _sum0 = vdup_n_f16(0.f);
                _sum1 = vdup_n_f16(0.f);
                _sum2 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[0]);
                        _sum2 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = vld1_f16(pC + 4);
                        _sum2 = vld1_f16(pC + 8);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vld1_f16(outptr);
                _sum1 = vld1_f16(outptr + 4);
                _sum2 = vld1_f16(outptr + 8);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);
                float16x4_t _pB2 = vld1_f16(pB + 8);

                float16x4_t _pA0 = vdup_n_f16(pA[0]);

                _sum0 = vfma_f16(_sum0, _pA0, _pB0);
                _sum1 = vfma_f16(_sum1, _pA0, _pB1);
                _sum2 = vfma_f16(_sum2, _pA0, _pB2);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 4, _sum1);
                    vst1_f16(outptr0 + 8, _sum2);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
                vst1_f16(outptr + 8, _sum2);
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float16x4_t _sum0;
            float16x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdup_n_f16(0.f);
                _sum1 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdup_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1_f16(pC);
                        _sum1 = vld1_f16(pC + 4);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vld1_f16(outptr);
                _sum1 = vld1_f16(outptr + 4);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);

                float16x4_t _pA0 = vdup_n_f16(pA[0]);

                _sum0 = vfma_f16(_sum0, _pA0, _pB0);
                _sum1 = vfma_f16(_sum1, _pA0, _pB1);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float16x4_t _sum;

            if (k == 0)
            {
                _sum = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = vld1_f16(pC);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum = vld1_f16(outptr);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x4_t _pB = vld1_f16(pB);
                float16x4_t _pA = vdup_n_f16(pA[0]);

                _sum = vfma_f16(_sum, _pA, _pB);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_f16(outptr0, _sum);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1_f16(outptr, _sum);
            }

            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __fp16 sum0;
            __fp16 sum1;

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

            const __fp16* pA = pAT;
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
            __fp16 sum;

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

            const __fp16* pA = pAT;
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

static void get_optimal_tile_mnk_fp16sa(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / 3 / sizeof(__fp16));

    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(8, tile_size / 8 * 8);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(__fp16) / TILE_K);

            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(4, tile_size / 4 * 4);
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
    }

    if (nT > 1)
    {
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
    }

    if (constant_TILE_N > 0)
    {
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
    }
}

static int gemm_arm_fp16sa(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
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

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 2u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;
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
        topT.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
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

                gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_bf16_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_AT_arm_fp16sa(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
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
        topT.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
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

                gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_bf16_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_BT_arm_fp16sa(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_fp16sa(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 2u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
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

                gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_bf16_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_AT_BT_arm_fp16sa(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_fp16sa(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
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

                gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_bf16_fp16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

int Gemm_arm::create_pipeline_fp16sa(const Option& opt)
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
        cast_float32_to_float16(C_data, CT_data, opt);
        if (CT_data.empty())
            return -100;

        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
            int C_elempack = constantM % 8 == 0 ? 8 : constantM % 4 == 0 ? 4 : 1;
            Mat tmp;
            convert_packing(CT_data, tmp, C_elempack, opt);
            CT_data = tmp;
            if (CT_data.empty())
                return -100;
        }

        // pre-multiply C with beta
        if (beta != 1.f)
        {
            const int size = CT_data.total() * CT_data.elempack;
            __fp16* ptr = CT_data;
            for (int i = 0; i < size; i++)
            {
                ptr[i] *= beta;
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

int Gemm_arm::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

            // pre-multiply C with beta
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

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
    }
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
        ret = gemm_AT_BT_arm_fp16sa(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_arm_fp16sa(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_arm_fp16sa(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_arm_fp16sa(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    if (ret != 0)
        return ret;

    // multiply top_blob with alpha
    if (alpha != 1.f)
    {
        const int size = top_blob.total() * out_elempack;
        __fp16* ptr = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            ptr[i] *= alpha;
        }
    }

    return 0;
}

#if NCNN_INT8
void compute_A_tile_fp16_int8_scales_asimdhp(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    compute_A_tile_fp16_int8_scales(A, scales, B_scale, out_descales, i, max_ii);
}

void transpose_compute_A_tile_fp16_int8_scales_asimdhp(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    transpose_compute_A_tile_fp16_int8_scales(A, scales, B_scale, out_descales, i, max_ii);
}

void compute_B_fp16_int8_scale_asimdhp(const Mat& B, float& scale)
{
    compute_B_fp16_int8_scale(B, scale);
}
#endif // NCNN_INT8

} // namespace ncnn
