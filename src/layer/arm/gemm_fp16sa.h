// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void gemm_transB_packed_tile_fp16sa_asimdhp(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end);
#endif

static void gemm_transB_packed_tile_fp16sa(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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
            float16x8_t _sum0;
            float16x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f16(0.f);
                _sum1 = vdup_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f16(pC[0]);
                        _sum1 = vdup_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f16(pC);
                        _sum1 = vld1_f16(pC + 8);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1_f16(outptr + 8);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            float16x8_t _sum00 = vdupq_n_f16(0.f);
            float16x8_t _sum01 = vdupq_n_f16(0.f);
            float16x8_t _sum02 = vdupq_n_f16(0.f);
            float16x8_t _sum03 = vdupq_n_f16(0.f);
            float16x8_t _sum04 = vdupq_n_f16(0.f);
            float16x8_t _sum05 = vdupq_n_f16(0.f);
            float16x8_t _sum06 = vdupq_n_f16(0.f);
            float16x8_t _sum07 = vdupq_n_f16(0.f);
            float16x4_t _sum10 = vdup_n_f16(0.f);
            float16x4_t _sum11 = vdup_n_f16(0.f);
            float16x4_t _sum12 = vdup_n_f16(0.f);
            float16x4_t _sum13 = vdup_n_f16(0.f);
            float16x4_t _sum14 = vdup_n_f16(0.f);
            float16x4_t _sum15 = vdup_n_f16(0.f);
            float16x4_t _sum16 = vdup_n_f16(0.f);
            float16x4_t _sum17 = vdup_n_f16(0.f);
            for (; kk + 7 < max_kk; kk += 8)
            {
                float16x8_t _pA = vld1q_f16(pA);
                float16x4_t _pA0123 = vget_low_f16(_pA);
                float16x4_t _pA4567 = vget_high_f16(_pA);

                float16x8_t _pB0 = vld1q_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 8);
                _sum00 = vfmaq_laneq_f16(_sum00, _pB0, _pA, 0);
                _sum10 = vfma_lane_f16(_sum10, _pB1, _pA0123, 0);
                float16x8_t _pB2 = vld1q_f16(pB + 12);
                float16x4_t _pB3 = vld1_f16(pB + 20);
                _sum01 = vfmaq_laneq_f16(_sum01, _pB2, _pA, 1);
                _sum11 = vfma_lane_f16(_sum11, _pB3, _pA0123, 1);
                float16x8_t _pB4 = vld1q_f16(pB + 24);
                float16x4_t _pB5 = vld1_f16(pB + 32);
                _sum02 = vfmaq_laneq_f16(_sum02, _pB4, _pA, 2);
                _sum12 = vfma_lane_f16(_sum12, _pB5, _pA0123, 2);
                float16x8_t _pB6 = vld1q_f16(pB + 36);
                float16x4_t _pB7 = vld1_f16(pB + 44);
                _sum03 = vfmaq_laneq_f16(_sum03, _pB6, _pA, 3);
                _sum13 = vfma_lane_f16(_sum13, _pB7, _pA0123, 3);
                float16x8_t _pB8 = vld1q_f16(pB + 48);
                float16x4_t _pB9 = vld1_f16(pB + 56);
                _sum04 = vfmaq_laneq_f16(_sum04, _pB8, _pA, 4);
                _sum14 = vfma_lane_f16(_sum14, _pB9, _pA4567, 0);
                float16x8_t _pBa = vld1q_f16(pB + 60);
                float16x4_t _pBb = vld1_f16(pB + 68);
                _sum05 = vfmaq_laneq_f16(_sum05, _pBa, _pA, 5);
                _sum15 = vfma_lane_f16(_sum15, _pBb, _pA4567, 1);
                float16x8_t _pBc = vld1q_f16(pB + 72);
                float16x4_t _pBd = vld1_f16(pB + 80);
                _sum06 = vfmaq_laneq_f16(_sum06, _pBc, _pA, 6);
                _sum16 = vfma_lane_f16(_sum16, _pBd, _pA4567, 2);
                float16x8_t _pBe = vld1q_f16(pB + 84);
                float16x4_t _pBf = vld1_f16(pB + 92);
                _sum07 = vfmaq_laneq_f16(_sum07, _pBe, _pA, 7);
                _sum17 = vfma_lane_f16(_sum17, _pBf, _pA4567, 3);

                pA += 8;
                pB += 96;
            }
            _sum00 = vaddq_f16(_sum00, _sum01);
            _sum02 = vaddq_f16(_sum02, _sum03);
            _sum04 = vaddq_f16(_sum04, _sum05);
            _sum06 = vaddq_f16(_sum06, _sum07);
            _sum10 = vadd_f16(_sum10, _sum11);
            _sum12 = vadd_f16(_sum12, _sum13);
            _sum14 = vadd_f16(_sum14, _sum15);
            _sum16 = vadd_f16(_sum16, _sum17);
            _sum00 = vaddq_f16(_sum00, _sum02);
            _sum04 = vaddq_f16(_sum04, _sum06);
            _sum10 = vadd_f16(_sum10, _sum12);
            _sum14 = vadd_f16(_sum14, _sum16);
            _sum0 = vaddq_f16(_sum0, _sum00);
            _sum0 = vaddq_f16(_sum0, _sum04);
            _sum1 = vadd_f16(_sum1, _sum10);
            _sum1 = vadd_f16(_sum1, _sum14);
            for (; kk + 3 < max_kk; kk += 4)
            {
                float16x4_t _pA = vld1_f16(pA);
                float16x8_t _pB0 = vld1q_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 8);
                _sum0 = vfmaq_lane_f16(_sum0, _pB0, _pA, 0);
                _sum1 = vfma_lane_f16(_sum1, _pB1, _pA, 0);
                float16x8_t _pB2 = vld1q_f16(pB + 12);
                float16x4_t _pB3 = vld1_f16(pB + 20);
                _sum0 = vfmaq_lane_f16(_sum0, _pB2, _pA, 1);
                _sum1 = vfma_lane_f16(_sum1, _pB3, _pA, 1);
                float16x8_t _pB4 = vld1q_f16(pB + 24);
                float16x4_t _pB5 = vld1_f16(pB + 32);
                _sum0 = vfmaq_lane_f16(_sum0, _pB4, _pA, 2);
                _sum1 = vfma_lane_f16(_sum1, _pB5, _pA, 2);
                float16x8_t _pB6 = vld1q_f16(pB + 36);
                float16x4_t _pB7 = vld1_f16(pB + 44);
                _sum0 = vfmaq_lane_f16(_sum0, _pB6, _pA, 3);
                _sum1 = vfma_lane_f16(_sum1, _pB7, _pA, 3);

                pA += 4;
                pB += 48;
            }
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pB0 = vld1q_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 8);

                float16x8_t _pA0 = vdupq_n_f16(pA[0]);

                _sum0 = vfmaq_f16(_sum0, _pA0, _pB0);
                _sum1 = vfma_f16(_sum1, vget_low_f16(_pA0), _pB1);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f16(outptr0, _sum0);
                    vst1_f16(outptr0 + 8, _sum1);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f16(outptr, _sum0);
                vst1_f16(outptr + 8, _sum1);
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float16x8_t _sum;

            if (k == 0)
            {
                _sum = vdupq_n_f16(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = vdupq_n_f16(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = vld1q_f16(pC);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum = vld1q_f16(outptr);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            float16x8_t _sum0 = vdupq_n_f16(0.f);
            float16x8_t _sum1 = vdupq_n_f16(0.f);
            float16x8_t _sum2 = vdupq_n_f16(0.f);
            float16x8_t _sum3 = vdupq_n_f16(0.f);
            float16x8_t _sum4 = vdupq_n_f16(0.f);
            float16x8_t _sum5 = vdupq_n_f16(0.f);
            float16x8_t _sum6 = vdupq_n_f16(0.f);
            float16x8_t _sum7 = vdupq_n_f16(0.f);
            for (; kk + 7 < max_kk; kk += 8)
            {
                float16x8_t _pA = vld1q_f16(pA);
                float16x8_t _pB0 = vld1q_f16(pB);
                _sum0 = vfmaq_laneq_f16(_sum0, _pB0, _pA, 0);
                float16x8_t _pB1 = vld1q_f16(pB + 8);
                _sum1 = vfmaq_laneq_f16(_sum1, _pB1, _pA, 1);
                float16x8_t _pB2 = vld1q_f16(pB + 16);
                _sum2 = vfmaq_laneq_f16(_sum2, _pB2, _pA, 2);
                float16x8_t _pB3 = vld1q_f16(pB + 24);
                _sum3 = vfmaq_laneq_f16(_sum3, _pB3, _pA, 3);
                float16x8_t _pB4 = vld1q_f16(pB + 32);
                _sum4 = vfmaq_laneq_f16(_sum4, _pB4, _pA, 4);
                float16x8_t _pB5 = vld1q_f16(pB + 40);
                _sum5 = vfmaq_laneq_f16(_sum5, _pB5, _pA, 5);
                float16x8_t _pB6 = vld1q_f16(pB + 48);
                _sum6 = vfmaq_laneq_f16(_sum6, _pB6, _pA, 6);
                float16x8_t _pB7 = vld1q_f16(pB + 56);
                _sum7 = vfmaq_laneq_f16(_sum7, _pB7, _pA, 7);

                pA += 8;
                pB += 64;
            }
            _sum0 = vaddq_f16(_sum0, _sum1);
            _sum2 = vaddq_f16(_sum2, _sum3);
            _sum4 = vaddq_f16(_sum4, _sum5);
            _sum6 = vaddq_f16(_sum6, _sum7);
            _sum0 = vaddq_f16(_sum0, _sum2);
            _sum4 = vaddq_f16(_sum4, _sum6);
            _sum = vaddq_f16(_sum, _sum0);
            _sum = vaddq_f16(_sum, _sum4);
            for (; kk + 3 < max_kk; kk += 4)
            {
                float16x4_t _pA = vld1_f16(pA);
                float16x8_t _pB0 = vld1q_f16(pB);
                _sum = vfmaq_lane_f16(_sum, _pB0, _pA, 0);
                float16x8_t _pB1 = vld1q_f16(pB + 8);
                _sum = vfmaq_lane_f16(_sum, _pB1, _pA, 1);
                float16x8_t _pB2 = vld1q_f16(pB + 16);
                _sum = vfmaq_lane_f16(_sum, _pB2, _pA, 2);
                float16x8_t _pB3 = vld1q_f16(pB + 24);
                _sum = vfmaq_lane_f16(_sum, _pB3, _pA, 3);

                pA += 4;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pB0 = vld1q_f16(pB);

                float16x8_t _pA = vdupq_n_f16(pA[0]);

                _sum = vfmaq_f16(_sum, _pA, _pB0);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f16(outptr0, _sum);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f16(outptr, _sum);
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
            float16x4_t _sum0 = vdup_n_f16(0.f);
            float16x4_t _sum1 = vdup_n_f16(0.f);
            float16x4_t _sum2 = vdup_n_f16(0.f);
            float16x4_t _sum3 = vdup_n_f16(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                float16x4_t _pA = vld1_f16(pA);
                float16x4_t _pB0 = vld1_f16(pB);
                _sum0 = vfma_lane_f16(_sum0, _pB0, _pA, 0);
                float16x4_t _pB1 = vld1_f16(pB + 4);
                _sum1 = vfma_lane_f16(_sum1, _pB1, _pA, 1);
                float16x4_t _pB2 = vld1_f16(pB + 8);
                _sum2 = vfma_lane_f16(_sum2, _pB2, _pA, 2);
                float16x4_t _pB3 = vld1_f16(pB + 12);
                _sum3 = vfma_lane_f16(_sum3, _pB3, _pA, 3);

                pA += 4;
                pB += 16;
            }
            _sum0 = vadd_f16(_sum0, _sum1);
            _sum2 = vadd_f16(_sum2, _sum3);
            _sum0 = vadd_f16(_sum0, _sum2);
            _sum = vadd_f16(_sum, _sum0);
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
            __fp16 sum00 = 0.f;
            __fp16 sum01 = 0.f;
            __fp16 sum02 = 0.f;
            __fp16 sum03 = 0.f;
            __fp16 sum10 = 0.f;
            __fp16 sum11 = 0.f;
            __fp16 sum12 = 0.f;
            __fp16 sum13 = 0.f;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum00 += pA[0] * pB[0];
                sum10 += pA[0] * pB[1];
                sum01 += pA[1] * pB[2];
                sum11 += pA[1] * pB[3];
                sum02 += pA[2] * pB[4];
                sum12 += pA[2] * pB[5];
                sum03 += pA[3] * pB[6];
                sum13 += pA[3] * pB[7];

                pA += 4;
                pB += 8;
            }
            sum00 += sum01;
            sum02 += sum03;
            sum10 += sum11;
            sum12 += sum13;
            sum0 += sum00 + sum02;
            sum1 += sum10 + sum12;
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
            __fp16 sum0 = 0.f;
            __fp16 sum1 = 0.f;
            __fp16 sum2 = 0.f;
            __fp16 sum3 = 0.f;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[1];
                sum2 += pA[2] * pB[2];
                sum3 += pA[3] * pB[3];

                pA += 4;
                pB += 4;
            }
            sum0 += sum1;
            sum2 += sum3;
            sum += sum0 + sum2;
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
#elif NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__
    gemm_transB_packed_tile_fp16sa_asimdhp(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
#else
    (void)AT_tile;
    (void)BT_tile;
    (void)CT_tile;
    (void)topT_tile;
    (void)top_blob;
    (void)broadcast_type_C;
    (void)i;
    (void)max_ii;
    (void)j;
    (void)max_jj;
    (void)k;
    (void)max_kk;
    (void)k_end;
#endif
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

