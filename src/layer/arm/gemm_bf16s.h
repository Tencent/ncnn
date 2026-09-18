// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
void pack_A_tile_bf16_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_bf16_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_bf16_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_bf16_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_bf16_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_fp32_to_bf16_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_fp32_to_bf16_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_fp32_to_bf16_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void unpack_output_tile_fp32_to_bf16_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose, int output_elemtype);
void gemm_transB_packed_tile_bf16s_bf16(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk);
#endif

static void pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        pack_A_tile_bf16_bf16(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _k0 = vld1q_u16(p0);
                uint16x8_t _k1 = vld1q_u16(p0 + 8);
                uint16x8_t _k2 = vld1q_u16(p0 + 16);
                uint16x8_t _k3 = vld1q_u16(p0 + 24);

                uint16x4_t _r0 = vget_low_u16(_k0);
                uint16x4_t _r1 = vget_low_u16(_k1);
                uint16x4_t _r2 = vget_low_u16(_k2);
                uint16x4_t _r3 = vget_low_u16(_k3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));

                _r0 = vget_high_u16(_k0);
                _r1 = vget_high_u16(_k1);
                _r2 = vget_high_u16(_k2);
                _r3 = vget_high_u16(_k3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp + 16, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 24, vcombine_u16(_r2, _r3));

                pp += 32;
                p0 += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _k0 = vld1q_u16(p0);
                uint16x8_t _k1 = vld1q_u16(p0 + 8);
                uint16x8x2_t _r01 = vzipq_u16(_k0, _k1);
                vst1q_u16(pp, _r01.val[0]);
                vst1q_u16(pp + 8, _r01.val[1]);
                pp += 16;
                p0 += 16;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0.val[2], _r0.val[3]));

                _r0 = vld4_u16(p1);
                vst1q_u16(pp + 16, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 24, vcombine_u16(_r0.val[2], _r0.val[3]));

                pp += 32;
                p0 += 16;
                p1 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vld1_u16(p0);
                uint16x4_t _k1 = vld1_u16(p0 + 4);
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
                _k0 = vld1_u16(p1);
                _k1 = vld1_u16(p1 + 4);
                _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp + 8, vcombine_u16(_r01.val[0], _r01.val[1]));
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                uint16x8_t _r0 = vcombine_u16(vld1_u16(p0), vld1_u16(p1));
                vst1q_u16(pp, _r0);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;
            const unsigned short* p4 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k;
            const unsigned short* p5 = (const unsigned short*)A + (i + ii + 5) * A_hstep + k;
            const unsigned short* p6 = (const unsigned short*)A + (i + ii + 6) * A_hstep + k;
            const unsigned short* p7 = (const unsigned short*)A + (i + ii + 7) * A_hstep + k;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                uint16x4_t _r4 = vld1_u16(p4);
                uint16x4_t _r5 = vld1_u16(p5);
                uint16x4_t _r6 = vld1_u16(p6);
                uint16x4_t _r7 = vld1_u16(p7);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
                vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _r0 = vdup_n_u32(0);
                uint32x2_t _r1 = vdup_n_u32(0);
                uint32x2_t _r2 = vdup_n_u32(0);
                uint32x2_t _r3 = vdup_n_u32(0);
                _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                _r0 = vld1_lane_u32((const uint32_t*)p1, _r0, 1);
                _r1 = vld1_lane_u32((const uint32_t*)p2, _r1, 0);
                _r1 = vld1_lane_u32((const uint32_t*)p3, _r1, 1);
                _r2 = vld1_lane_u32((const uint32_t*)p4, _r2, 0);
                _r2 = vld1_lane_u32((const uint32_t*)p5, _r2, 1);
                _r3 = vld1_lane_u32((const uint32_t*)p6, _r3, 0);
                _r3 = vld1_lane_u32((const uint32_t*)p7, _r3, 1);
                vst1_u16(pp, vreinterpret_u16_u32(_r0));
                vst1_u16(pp + 4, vreinterpret_u16_u32(_r1));
                vst1_u16(pp + 8, vreinterpret_u16_u32(_r2));
                vst1_u16(pp + 12, vreinterpret_u16_u32(_r3));
                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p1);
                uint16x8_t _r2 = vld1q_u16(p2);
                uint16x8_t _r3 = vld1q_u16(p3);
                uint16x8_t _r4 = vld1q_u16(p4);
                uint16x8_t _r5 = vld1q_u16(p5);
                uint16x8_t _r6 = vld1q_u16(p6);
                uint16x8_t _r7 = vld1q_u16(p7);
                transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                pp += 64;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
                p4 += 8;
                p5 += 8;
                p6 += 8;
                p7 += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0.val[2], _r0.val[3]));
                pp += 16;
                p0 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vld1_u16(p0);
                uint16x4_t _k1 = vld1_u16(p0 + 4);
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
                pp += 8;
                p0 += 8;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 1 < max_kk; kk += 2)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _r0 = vdup_n_u32(0);
                uint32x2_t _r1 = vdup_n_u32(0);
                _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                _r0 = vld1_lane_u32((const uint32_t*)p1, _r0, 1);
                _r1 = vld1_lane_u32((const uint32_t*)p2, _r1, 0);
                _r1 = vld1_lane_u32((const uint32_t*)p3, _r1, 1);
                vst1_u16(pp, vreinterpret_u16_u32(_r0));
                vst1_u16(pp + 4, vreinterpret_u16_u32(_r1));
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123;
                _r0123.val[0] = vld1q_u16(p0);
                _r0123.val[1] = vld1q_u16(p1);
                _r0123.val[2] = vld1q_u16(p2);
                _r0123.val[3] = vld1q_u16(p3);
                vst4q_u16(pp, _r0123);
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0123;
                _r0123.val[0] = vld1_u16(p0);
                _r0123.val[1] = vld1_u16(p1);
                _r0123.val[2] = vld1_u16(p2);
                _r0123.val[3] = vld1_u16(p3);
                vst4_u16(pp, _r0123);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _r0 = vdup_n_u32(0);
                _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                _r0 = vld1_lane_u32((const uint32_t*)p1, _r0, 1);
                vst1_u16(pp, vreinterpret_u16_u32(_r0));
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x2_t _r01;
                _r01.val[0] = vld1q_u16(p0);
                _r01.val[1] = vld1q_u16(p1);
                vst2q_u16(pp, _r01);
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x2_t _r01;
                _r01.val[0] = vld1_u16(p0);
                _r01.val[1] = vld1_u16(p1);
                vst2_u16(pp, _r01);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __ARM_NEON
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = (unsigned short)p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        transpose_pack_A_tile_bf16_bf16(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);
                vst1q_u16(pp, _r04.val[0]);
                vst1q_u16(pp + 8, _r15.val[0]);
                vst1q_u16(pp + 16, _r26.val[0]);
                vst1q_u16(pp + 24, _r37.val[0]);
                vst1q_u16(pp + 32, _r04.val[1]);
                vst1q_u16(pp + 40, _r15.val[1]);
                vst1q_u16(pp + 48, _r26.val[1]);
                vst1q_u16(pp + 56, _r37.val[1]);
                pp += 64;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                vst1q_u16(pp + 8, vld1q_u16(p0 + 8));
                vst1q_u16(pp + 16, vld1q_u16(p0 + 16));
                vst1q_u16(pp + 24, vld1q_u16(p0 + 24));
                pp += 32;
                p0 += A_hstep * 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                vst1q_u16(pp, _r0123.val[0]);
                vst1q_u16(pp + 8, _r0123.val[1]);
                vst1q_u16(pp + 16, _r0123.val[2]);
                vst1q_u16(pp + 24, _r0123.val[3]);
                pp += 32;
                p0 += A_hstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _k0 = vld1q_u16(p0);
                uint16x8_t _k1 = vld1q_u16(p0 + A_hstep);
                uint16x8_t _k2 = vld1q_u16(p0 + A_hstep * 2);
                uint16x8_t _k3 = vld1q_u16(p0 + A_hstep * 3);

                uint16x4_t _r0 = vget_low_u16(_k0);
                uint16x4_t _r1 = vget_low_u16(_k1);
                uint16x4_t _r2 = vget_low_u16(_k2);
                uint16x4_t _r3 = vget_low_u16(_k3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));

                _r0 = vget_high_u16(_k0);
                _r1 = vget_high_u16(_k1);
                _r2 = vget_high_u16(_k2);
                _r3 = vget_high_u16(_k3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp + 16, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 24, vcombine_u16(_r2, _r3));

                pp += 32;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _k0 = vld1q_u16(p0);
                uint16x8_t _k1 = vld1q_u16(p0 + A_hstep);
                uint16x8x2_t _r01 = vzipq_u16(_k0, _k1);
                vst1q_u16(pp, _r01.val[0]);
                vst1q_u16(pp + 8, _r01.val[1]);
                pp += 16;
                p0 += A_hstep * 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123;
                _r0123.val[0] = vld1q_u16(p0);
                _r0123.val[1] = vld1q_u16(p0 + 8);
                _r0123.val[2] = vld1q_u16(p0 + 16);
                _r0123.val[3] = vld1q_u16(p0 + 24);
                vst4q_u16(pp, _r0123);
                pp += 32;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                vst1q_u16(pp + 8, vld1q_u16(p0 + 8));
                pp += 16;
                p0 += A_hstep * 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0123 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0123.val[0], _r0123.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0123.val[2], _r0123.val[3]));
                pp += 16;
                p0 += A_hstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p0 + A_hstep);
                uint16x4_t _r2 = vld1_u16(p0 + A_hstep * 2);
                uint16x4_t _r3 = vld1_u16(p0 + A_hstep * 3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                pp += 16;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vld1_u16(p0);
                uint16x4_t _k1 = vld1_u16(p0 + A_hstep);
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
                pp += 8;
                p0 += A_hstep * 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x2_t _r01;
                _r01.val[0] = vld1q_u16(p0);
                _r01.val[1] = vld1q_u16(p0 + 8);
                vst2q_u16(pp, _r01);
                pp += 16;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_u16(pp, vcombine_u16(vld1_u16(p0), vld1_u16(p0 + 4)));
                pp += 8;
                p0 += A_hstep * 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x2_t _r01;
                _r01.val[0] = vld1_u16(p0);
                _r01.val[1] = vld1_u16(p0 + 4);
                vst2_u16(pp, _r01);
                pp += 8;
                p0 += A_hstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)p0));
                uint16x4_t _r1 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)(p0 + A_hstep)));
                uint16x4_t _r2 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)(p0 + A_hstep * 2)));
                uint16x4_t _r3 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)(p0 + A_hstep * 3)));
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                pp += 8;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)p0));
                uint16x4_t _k1 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)(p0 + A_hstep)));
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1_u16(pp, _r01.val[0]);
                pp += 4;
                p0 += A_hstep * 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
#if __ARM_NEON
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

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

static void pack_B_tile_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        pack_B_tile_bf16_bf16(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1q_u16(pp, vld1q_u16(p0));
                    vst1_u16(pp + 8, vld1_u16(p1));
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1_u16(pp, vld1_u16(p0 + 4));
                    vst1q_u16(pp + 4, vld1q_u16(p1));
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0.val[2], _r0.val[3]));

                _r0 = vld4_u16(p1);
                vst1q_u16(pp + 16, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 24, vcombine_u16(_r0.val[2], _r0.val[3]));

                _r0 = vld4_u16(p2);
                vst1q_u16(pp + 32, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 40, vcombine_u16(_r0.val[2], _r0.val[3]));

                pp += 48;
                p0 += 16;
                p1 += 16;
                p2 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vld1_u16(p0);
                uint16x4_t _k1 = vld1_u16(p0 + 4);
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
                _k0 = vld1_u16(p1);
                _k1 = vld1_u16(p1 + 4);
                _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp + 8, vcombine_u16(_r01.val[0], _r01.val[1]));
                _k0 = vld1_u16(p2);
                _k1 = vld1_u16(p2 + 4);
                _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp + 16, vcombine_u16(_r01.val[0], _r01.val[1]));
                pp += 24;
                p0 += 8;
                p1 += 8;
                p2 += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
                vst1_u16(pp + 4, vld1_u16(p1));
                vst1_u16(pp + 8, vld1_u16(p2));
                pp += 12;
                p0 += 4;
                p1 += 4;
                p2 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;
            const unsigned short* p4 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k;
            const unsigned short* p5 = (const unsigned short*)B + (j + jj + 5) * B_hstep + k;
            const unsigned short* p6 = (const unsigned short*)B + (j + jj + 6) * B_hstep + k;
            const unsigned short* p7 = (const unsigned short*)B + (j + jj + 7) * B_hstep + k;
            const unsigned short* p8 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k;
            const unsigned short* p9 = (const unsigned short*)B + (j + jj + 9) * B_hstep + k;
            const unsigned short* pa = (const unsigned short*)B + (j + jj + 10) * B_hstep + k;
            const unsigned short* pb = (const unsigned short*)B + (j + jj + 11) * B_hstep + k;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                uint16x4_t _r4 = vld1_u16(p4);
                uint16x4_t _r5 = vld1_u16(p5);
                uint16x4_t _r6 = vld1_u16(p6);
                uint16x4_t _r7 = vld1_u16(p7);
                uint16x4_t _r8 = vld1_u16(p8);
                uint16x4_t _r9 = vld1_u16(p9);
                uint16x4_t _ra = vld1_u16(pa);
                uint16x4_t _rb = vld1_u16(pb);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
                vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
                vst1q_u16(pp + 32, vcombine_u16(_r8, _r9));
                vst1q_u16(pp + 40, vcombine_u16(_ra, _rb));
                pp += 48;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                p4 += 4;
                p5 += 4;
                p6 += 4;
                p7 += 4;
                p8 += 4;
                p9 += 4;
                pa += 4;
                pb += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _r0 = vdup_n_u32(0);
                uint32x2_t _r1 = vdup_n_u32(0);
                uint32x2_t _r2 = vdup_n_u32(0);
                uint32x2_t _r3 = vdup_n_u32(0);
                uint32x2_t _r4 = vdup_n_u32(0);
                uint32x2_t _r5 = vdup_n_u32(0);
                _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                _r0 = vld1_lane_u32((const uint32_t*)p1, _r0, 1);
                _r1 = vld1_lane_u32((const uint32_t*)p2, _r1, 0);
                _r1 = vld1_lane_u32((const uint32_t*)p3, _r1, 1);
                _r2 = vld1_lane_u32((const uint32_t*)p4, _r2, 0);
                _r2 = vld1_lane_u32((const uint32_t*)p5, _r2, 1);
                _r3 = vld1_lane_u32((const uint32_t*)p6, _r3, 0);
                _r3 = vld1_lane_u32((const uint32_t*)p7, _r3, 1);
                _r4 = vld1_lane_u32((const uint32_t*)p8, _r4, 0);
                _r4 = vld1_lane_u32((const uint32_t*)p9, _r4, 1);
                _r5 = vld1_lane_u32((const uint32_t*)pa, _r5, 0);
                _r5 = vld1_lane_u32((const uint32_t*)pb, _r5, 1);
                vst1_u16(pp, vreinterpret_u16_u32(_r0));
                vst1_u16(pp + 4, vreinterpret_u16_u32(_r1));
                vst1_u16(pp + 8, vreinterpret_u16_u32(_r2));
                vst1_u16(pp + 12, vreinterpret_u16_u32(_r3));
                vst1_u16(pp + 16, vreinterpret_u16_u32(_r4));
                vst1_u16(pp + 20, vreinterpret_u16_u32(_r5));
                pp += 24;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
                p8 += 2;
                p9 += 2;
                pa += 2;
                pb += 2;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                uint16x4_t _r4 = vld1_u16(p4);
                uint16x4_t _r5 = vld1_u16(p5);
                uint16x4_t _r6 = vld1_u16(p6);
                uint16x4_t _r7 = vld1_u16(p7);
                uint16x4_t _r8 = vld1_u16(p8);
                uint16x4_t _r9 = vld1_u16(p9);
                uint16x4_t _ra = vld1_u16(pa);
                uint16x4_t _rb = vld1_u16(pb);

                transpose4x4_u16(_r0, _r1, _r2, _r3);
                transpose4x4_u16(_r4, _r5, _r6, _r7);
                transpose4x4_u16(_r8, _r9, _ra, _rb);

                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r4);
                vst1_u16(pp + 4 * 2, _r8);
                vst1_u16(pp + 4 * 3, _r1);
                vst1_u16(pp + 4 * 4, _r5);
                vst1_u16(pp + 4 * 5, _r9);
                vst1_u16(pp + 4 * 6, _r2);
                vst1_u16(pp + 4 * 7, _r6);
                vst1_u16(pp + 4 * 8, _ra);
                vst1_u16(pp + 4 * 9, _r3);
                vst1_u16(pp + 4 * 10, _r7);
                vst1_u16(pp + 4 * 11, _rb);
                pp += 48;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                p4 += 4;
                p5 += 4;
                p6 += 4;
                p7 += 4;
                p8 += 4;
                p9 += 4;
                pa += 4;
                pb += 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
                pp[8] = p8[0];
                pp[9] = p9[0];
                pp[10] = pa[0];
                pp[11] = pb[0];
                pp += 12;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
                p8++;
                p9++;
                pa++;
                pb++;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                for (; kk + 3 < max_kk; kk += 4)
                {
                    uint16x8_t _k0 = vld1q_u16(p0);
                    uint16x8_t _k1 = vld1q_u16(p0 + 8);
                    uint16x8_t _k2 = vld1q_u16(p0 + 16);
                    uint16x8_t _k3 = vld1q_u16(p0 + 24);

                    uint16x4_t _r0 = vget_low_u16(_k0);
                    uint16x4_t _r1 = vget_low_u16(_k1);
                    uint16x4_t _r2 = vget_low_u16(_k2);
                    uint16x4_t _r3 = vget_low_u16(_k3);
                    transpose4x4_u16(_r0, _r1, _r2, _r3);
                    vst1q_u16(pp, vcombine_u16(_r0, _r1));
                    vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));

                    _r0 = vget_high_u16(_k0);
                    _r1 = vget_high_u16(_k1);
                    _r2 = vget_high_u16(_k2);
                    _r3 = vget_high_u16(_k3);
                    transpose4x4_u16(_r0, _r1, _r2, _r3);
                    vst1q_u16(pp + 16, vcombine_u16(_r0, _r1));
                    vst1q_u16(pp + 24, vcombine_u16(_r2, _r3));

                    pp += 32;
                    p0 += 32;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    uint16x8_t _k0 = vld1q_u16(p0);
                    uint16x8_t _k1 = vld1q_u16(p0 + 8);
                    uint16x8x2_t _r01 = vzipq_u16(_k0, _k1);
                    vst1q_u16(pp, _r01.val[0]);
                    vst1q_u16(pp + 8, _r01.val[1]);
                    pp += 16;
                    p0 += 16;
                }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                for (; kk < max_kk; kk++)
                {
                    vst1q_u16(pp, vld1q_u16(p0));
                    pp += 8;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1q_u16(pp, vcombine_u16(vld1_u16(p0 + 4), vld1_u16(p1)));
                    pp += 8;
                    p0 += 8;
                    p1 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0.val[2], _r0.val[3]));

                _r0 = vld4_u16(p1);
                vst1q_u16(pp + 16, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 24, vcombine_u16(_r0.val[2], _r0.val[3]));

                pp += 32;
                p0 += 16;
                p1 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vld1_u16(p0);
                uint16x4_t _k1 = vld1_u16(p0 + 4);
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
                _k0 = vld1_u16(p1);
                _k1 = vld1_u16(p1 + 4);
                _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp + 8, vcombine_u16(_r01.val[0], _r01.val[1]));
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                uint16x8_t _r0 = vcombine_u16(vld1_u16(p0), vld1_u16(p1));
                vst1q_u16(pp, _r0);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;
            const unsigned short* p4 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k;
            const unsigned short* p5 = (const unsigned short*)B + (j + jj + 5) * B_hstep + k;
            const unsigned short* p6 = (const unsigned short*)B + (j + jj + 6) * B_hstep + k;
            const unsigned short* p7 = (const unsigned short*)B + (j + jj + 7) * B_hstep + k;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                uint16x4_t _r4 = vld1_u16(p4);
                uint16x4_t _r5 = vld1_u16(p5);
                uint16x4_t _r6 = vld1_u16(p6);
                uint16x4_t _r7 = vld1_u16(p7);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
                vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _r0 = vdup_n_u32(0);
                uint32x2_t _r1 = vdup_n_u32(0);
                uint32x2_t _r2 = vdup_n_u32(0);
                uint32x2_t _r3 = vdup_n_u32(0);
                _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                _r0 = vld1_lane_u32((const uint32_t*)p1, _r0, 1);
                _r1 = vld1_lane_u32((const uint32_t*)p2, _r1, 0);
                _r1 = vld1_lane_u32((const uint32_t*)p3, _r1, 1);
                _r2 = vld1_lane_u32((const uint32_t*)p4, _r2, 0);
                _r2 = vld1_lane_u32((const uint32_t*)p5, _r2, 1);
                _r3 = vld1_lane_u32((const uint32_t*)p6, _r3, 0);
                _r3 = vld1_lane_u32((const uint32_t*)p7, _r3, 1);
                vst1_u16(pp, vreinterpret_u16_u32(_r0));
                vst1_u16(pp + 4, vreinterpret_u16_u32(_r1));
                vst1_u16(pp + 8, vreinterpret_u16_u32(_r2));
                vst1_u16(pp + 12, vreinterpret_u16_u32(_r3));
                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p1);
                uint16x8_t _r2 = vld1q_u16(p2);
                uint16x8_t _r3 = vld1q_u16(p3);
                uint16x8_t _r4 = vld1q_u16(p4);
                uint16x8_t _r5 = vld1q_u16(p5);
                uint16x8_t _r6 = vld1q_u16(p6);
                uint16x8_t _r7 = vld1q_u16(p7);
                transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                pp += 64;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
                p4 += 8;
                p5 += 8;
                p6 += 8;
                p7 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                uint16x4_t _r4 = vld1_u16(p4);
                uint16x4_t _r5 = vld1_u16(p5);
                uint16x4_t _r6 = vld1_u16(p6);
                uint16x4_t _r7 = vld1_u16(p7);

                transpose4x4_u16(_r0, _r1, _r2, _r3);
                transpose4x4_u16(_r4, _r5, _r6, _r7);

                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r4);
                vst1_u16(pp + 4 * 2, _r1);
                vst1_u16(pp + 4 * 3, _r5);
                vst1_u16(pp + 4 * 4, _r2);
                vst1_u16(pp + 4 * 5, _r6);
                vst1_u16(pp + 4 * 6, _r3);
                vst1_u16(pp + 4 * 7, _r7);
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
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1_u16(pp, vld1_u16(p0));
                    pp += 4;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1_u16(pp, vld1_u16(p0 + 4));
                    pp += 4;
                    p0 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0.val[0], _r0.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0.val[2], _r0.val[3]));
                pp += 16;
                p0 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vld1_u16(p0);
                uint16x4_t _k1 = vld1_u16(p0 + 4);
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
                pp += 8;
                p0 += 8;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 1 < max_kk; kk += 2)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _r0 = vdup_n_u32(0);
                uint32x2_t _r1 = vdup_n_u32(0);
                _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                _r0 = vld1_lane_u32((const uint32_t*)p1, _r0, 1);
                _r1 = vld1_lane_u32((const uint32_t*)p2, _r1, 0);
                _r1 = vld1_lane_u32((const uint32_t*)p3, _r1, 1);
                vst1_u16(pp, vreinterpret_u16_u32(_r0));
                vst1_u16(pp + 4, vreinterpret_u16_u32(_r1));
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123;
                _r0123.val[0] = vld1q_u16(p0);
                _r0123.val[1] = vld1q_u16(p1);
                _r0123.val[2] = vld1q_u16(p2);
                _r0123.val[3] = vld1q_u16(p3);
                vst4q_u16(pp, _r0123);
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0123;
                _r0123.val[0] = vld1_u16(p0);
                _r0123.val[1] = vld1_u16(p1);
                _r0123.val[2] = vld1_u16(p2);
                _r0123.val[3] = vld1_u16(p3);
                vst4_u16(pp, _r0123);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _r0 = vdup_n_u32(0);
                _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                _r0 = vld1_lane_u32((const uint32_t*)p1, _r0, 1);
                vst1_u16(pp, vreinterpret_u16_u32(_r0));
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x2_t _r01;
                _r01.val[0] = vld1q_u16(p0);
                _r01.val[1] = vld1q_u16(p1);
                vst2q_u16(pp, _r01);
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x2_t _r01;
                _r01.val[0] = vld1_u16(p0);
                _r01.val[1] = vld1_u16(p1);
                vst2_u16(pp, _r01);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __ARM_NEON
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        transpose_pack_B_tile_bf16_bf16(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                uint16x8x4_t _r89ab = vld4q_u16(p0 + 64);
                uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);
                uint16x4x2_t _r04_1 = vuzp_u16(vget_low_u16(_r89ab.val[0]), vget_high_u16(_r89ab.val[0]));
                uint16x4x2_t _r15_1 = vuzp_u16(vget_low_u16(_r89ab.val[1]), vget_high_u16(_r89ab.val[1]));
                uint16x4x2_t _r26_1 = vuzp_u16(vget_low_u16(_r89ab.val[2]), vget_high_u16(_r89ab.val[2]));
                uint16x4x2_t _r37_1 = vuzp_u16(vget_low_u16(_r89ab.val[3]), vget_high_u16(_r89ab.val[3]));
                vst1q_u16(pp, _r04.val[0]);
                vst1_u16(pp + 8, _r04_1.val[0]);
                vst1q_u16(pp + 12, _r15.val[0]);
                vst1_u16(pp + 20, _r15_1.val[0]);
                vst1q_u16(pp + 24, _r26.val[0]);
                vst1_u16(pp + 32, _r26_1.val[0]);
                vst1q_u16(pp + 36, _r37.val[0]);
                vst1_u16(pp + 44, _r37_1.val[0]);
                vst1q_u16(pp + 48, _r04.val[1]);
                vst1_u16(pp + 56, _r04_1.val[1]);
                vst1q_u16(pp + 60, _r15.val[1]);
                vst1_u16(pp + 68, _r15_1.val[1]);
                vst1q_u16(pp + 72, _r26.val[1]);
                vst1_u16(pp + 80, _r26_1.val[1]);
                vst1q_u16(pp + 84, _r37.val[1]);
                vst1_u16(pp + 92, _r37_1.val[1]);
                pp += 96;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                vst1q_u16(pp + 8, vld1q_u16(p0 + 8));
                vst1q_u16(pp + 16, vld1q_u16(p0 + 16));
                vst1q_u16(pp + 24, vld1q_u16(p0 + 24));
                vst1q_u16(pp + 32, vld1q_u16(p0 + 32));
                vst1q_u16(pp + 40, vld1q_u16(p0 + 40));
                pp += 48;
                p0 += B_hstep * 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x4x4_t _r89ab = vld4_u16(p0 + 32);
                vst1q_u16(pp, _r0123.val[0]);
                vst1_u16(pp + 8, _r89ab.val[0]);
                vst1q_u16(pp + 12, _r0123.val[1]);
                vst1_u16(pp + 20, _r89ab.val[1]);
                vst1q_u16(pp + 24, _r0123.val[2]);
                vst1_u16(pp + 32, _r89ab.val[2]);
                vst1q_u16(pp + 36, _r0123.val[3]);
                vst1_u16(pp + 44, _r89ab.val[3]);
                pp += 48;
                p0 += B_hstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p0 + 4);
                uint16x4_t _r2 = vld1_u16(p0 + 8);
                uint16x4_t _r3 = vld1_u16(p0 + B_hstep);
                uint16x4_t _r4 = vld1_u16(p0 + B_hstep + 4);
                uint16x4_t _r5 = vld1_u16(p0 + B_hstep + 8);
                uint16x4_t _r6 = vld1_u16(p0 + B_hstep * 2);
                uint16x4_t _r7 = vld1_u16(p0 + B_hstep * 2 + 4);
                uint16x4_t _r8 = vld1_u16(p0 + B_hstep * 2 + 8);
                uint16x4_t _r9 = vld1_u16(p0 + B_hstep * 3);
                uint16x4_t _ra = vld1_u16(p0 + B_hstep * 3 + 4);
                uint16x4_t _rb = vld1_u16(p0 + B_hstep * 3 + 8);

                transpose4x4_u16(_r0, _r3, _r6, _r9);
                transpose4x4_u16(_r1, _r4, _r7, _ra);
                transpose4x4_u16(_r2, _r5, _r8, _rb);

                vst1q_u16(pp, vcombine_u16(_r0, _r3));
                vst1q_u16(pp + 8, vcombine_u16(_r6, _r9));
                vst1q_u16(pp + 16, vcombine_u16(_r1, _r4));
                vst1q_u16(pp + 24, vcombine_u16(_r7, _ra));
                vst1q_u16(pp + 32, vcombine_u16(_r2, _r5));
                vst1q_u16(pp + 40, vcombine_u16(_r8, _rb));
                pp += 48;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vld1_u16(p0);
                uint16x4_t _k1 = vld1_u16(p0 + B_hstep);
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
                _k0 = vld1_u16(p0 + 4);
                _k1 = vld1_u16(p0 + B_hstep + 4);
                _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp + 8, vcombine_u16(_r01.val[0], _r01.val[1]));
                _k0 = vld1_u16(p0 + 8);
                _k1 = vld1_u16(p0 + B_hstep + 8);
                _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp + 16, vcombine_u16(_r01.val[0], _r01.val[1]));
                pp += 24;
                p0 += B_hstep * 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                vst1_u16(pp + 8, vld1_u16(p0 + 8));
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);
                vst1q_u16(pp, _r04.val[0]);
                vst1q_u16(pp + 8, _r15.val[0]);
                vst1q_u16(pp + 16, _r26.val[0]);
                vst1q_u16(pp + 24, _r37.val[0]);
                vst1q_u16(pp + 32, _r04.val[1]);
                vst1q_u16(pp + 40, _r15.val[1]);
                vst1q_u16(pp + 48, _r26.val[1]);
                vst1q_u16(pp + 56, _r37.val[1]);
                pp += 64;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                vst1q_u16(pp + 8, vld1q_u16(p0 + 8));
                vst1q_u16(pp + 16, vld1q_u16(p0 + 16));
                vst1q_u16(pp + 24, vld1q_u16(p0 + 24));
                pp += 32;
                p0 += B_hstep * 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                vst1q_u16(pp, _r0123.val[0]);
                vst1q_u16(pp + 8, _r0123.val[1]);
                vst1q_u16(pp + 16, _r0123.val[2]);
                vst1q_u16(pp + 24, _r0123.val[3]);
                pp += 32;
                p0 += B_hstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _k0 = vld1q_u16(p0);
                uint16x8_t _k1 = vld1q_u16(p0 + B_hstep);
                uint16x8_t _k2 = vld1q_u16(p0 + B_hstep * 2);
                uint16x8_t _k3 = vld1q_u16(p0 + B_hstep * 3);

                uint16x4_t _r0 = vget_low_u16(_k0);
                uint16x4_t _r1 = vget_low_u16(_k1);
                uint16x4_t _r2 = vget_low_u16(_k2);
                uint16x4_t _r3 = vget_low_u16(_k3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));

                _r0 = vget_high_u16(_k0);
                _r1 = vget_high_u16(_k1);
                _r2 = vget_high_u16(_k2);
                _r3 = vget_high_u16(_k3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp + 16, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 24, vcombine_u16(_r2, _r3));

                pp += 32;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _k0 = vld1q_u16(p0);
                uint16x8_t _k1 = vld1q_u16(p0 + B_hstep);
                uint16x8x2_t _r01 = vzipq_u16(_k0, _k1);
                vst1q_u16(pp, _r01.val[0]);
                vst1q_u16(pp + 8, _r01.val[1]);
                pp += 16;
                p0 += B_hstep * 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123;
                _r0123.val[0] = vld1q_u16(p0);
                _r0123.val[1] = vld1q_u16(p0 + 8);
                _r0123.val[2] = vld1q_u16(p0 + 16);
                _r0123.val[3] = vld1q_u16(p0 + 24);
                vst4q_u16(pp, _r0123);
                pp += 32;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                vst1q_u16(pp + 8, vld1q_u16(p0 + 8));
                pp += 16;
                p0 += B_hstep * 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x4_t _r0123 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0123.val[0], _r0123.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0123.val[2], _r0123.val[3]));
                pp += 16;
                p0 += B_hstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p0 + B_hstep);
                uint16x4_t _r2 = vld1_u16(p0 + B_hstep * 2);
                uint16x4_t _r3 = vld1_u16(p0 + B_hstep * 3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                pp += 16;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vld1_u16(p0);
                uint16x4_t _k1 = vld1_u16(p0 + B_hstep);
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
                pp += 8;
                p0 += B_hstep * 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __ARM_NEON
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x2_t _r01;
                _r01.val[0] = vld1q_u16(p0);
                _r01.val[1] = vld1q_u16(p0 + 8);
                vst2q_u16(pp, _r01);
                pp += 16;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_u16(pp, vcombine_u16(vld1_u16(p0), vld1_u16(p0 + 4)));
                pp += 8;
                p0 += B_hstep * 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4x2_t _r01;
                _r01.val[0] = vld1_u16(p0);
                _r01.val[1] = vld1_u16(p0 + 4);
                vst2_u16(pp, _r01);
                pp += 8;
                p0 += B_hstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)p0));
                uint16x4_t _r1 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)(p0 + B_hstep)));
                uint16x4_t _r2 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)(p0 + B_hstep * 2)));
                uint16x4_t _r3 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)(p0 + B_hstep * 3)));
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                pp += 8;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _k0 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)p0));
                uint16x4_t _k1 = vreinterpret_u16_u32(vld1_dup_u32((const uint32_t*)(p0 + B_hstep)));
                uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
                vst1_u16(pp, _r01.val[0]);
                pp += 4;
                p0 += B_hstep * 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
#if __ARM_NEON
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

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

static void pack_A_tile_fp32_to_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        pack_A_tile_fp32_to_bf16_bf16(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
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
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p3));
            uint16x4_t _r4 = float2bfloat(vld1q_f32(p4));
            uint16x4_t _r5 = float2bfloat(vld1q_f32(p5));
            uint16x4_t _r6 = float2bfloat(vld1q_f32(p6));
            uint16x4_t _r7 = float2bfloat(vld1q_f32(p7));
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
            vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
            vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
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
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _r0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p1)));
            uint16x4_t _r1 = float2bfloat(vcombine_f32(vld1_f32(p2), vld1_f32(p3)));
            uint16x4_t _r2 = float2bfloat(vcombine_f32(vld1_f32(p4), vld1_f32(p5)));
            uint16x4_t _r3 = float2bfloat(vcombine_f32(vld1_f32(p6), vld1_f32(p7)));
            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r1);
            vst1_u16(pp + 8, _r2);
            vst1_u16(pp + 12, _r3);
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
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
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            uint16x8_t _r1 = vcombine_u16(float2bfloat(vld1q_f32(p1)), float2bfloat(vld1q_f32(p1 + 4)));
            uint16x8_t _r2 = vcombine_u16(float2bfloat(vld1q_f32(p2)), float2bfloat(vld1q_f32(p2 + 4)));
            uint16x8_t _r3 = vcombine_u16(float2bfloat(vld1q_f32(p3)), float2bfloat(vld1q_f32(p3 + 4)));
            uint16x8_t _r4 = vcombine_u16(float2bfloat(vld1q_f32(p4)), float2bfloat(vld1q_f32(p4 + 4)));
            uint16x8_t _r5 = vcombine_u16(float2bfloat(vld1q_f32(p5)), float2bfloat(vld1q_f32(p5 + 4)));
            uint16x8_t _r6 = vcombine_u16(float2bfloat(vld1q_f32(p6)), float2bfloat(vld1q_f32(p6 + 4)));
            uint16x8_t _r7 = vcombine_u16(float2bfloat(vld1q_f32(p7)), float2bfloat(vld1q_f32(p7 + 4)));
            transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            vst1q_u16(pp, _r0);
            vst1q_u16(pp + 8, _r1);
            vst1q_u16(pp + 16, _r2);
            vst1q_u16(pp + 24, _r3);
            vst1q_u16(pp + 32, _r4);
            vst1q_u16(pp + 40, _r5);
            vst1q_u16(pp + 48, _r6);
            vst1q_u16(pp + 56, _r7);
            pp += 64;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
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
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p3));
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _r0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p1)));
            uint16x4_t _r1 = float2bfloat(vcombine_f32(vld1_f32(p2), vld1_f32(p3)));
            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r1);
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x4_t _r0123;
            _r0123.val[0] = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            _r0123.val[1] = vcombine_u16(float2bfloat(vld1q_f32(p1)), float2bfloat(vld1q_f32(p1 + 4)));
            _r0123.val[2] = vcombine_u16(float2bfloat(vld1q_f32(p2)), float2bfloat(vld1q_f32(p2 + 4)));
            _r0123.val[3] = vcombine_u16(float2bfloat(vld1q_f32(p3)), float2bfloat(vld1q_f32(p3 + 4)));
            vst4q_u16(pp, _r0123);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x4_t _r0123;
            _r0123.val[0] = float2bfloat(vld1q_f32(p0));
            _r0123.val[1] = float2bfloat(vld1q_f32(p1));
            _r0123.val[2] = float2bfloat(vld1q_f32(p2));
            _r0123.val[3] = float2bfloat(vld1q_f32(p3));
            vst4_u16(pp, _r0123);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _r0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p1)));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __ARM_NEON
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x2_t _r01;
            _r01.val[0] = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            _r01.val[1] = vcombine_u16(float2bfloat(vld1q_f32(p1)), float2bfloat(vld1q_f32(p1 + 4)));
            vst2q_u16(pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x2_t _r01;
            _r01.val[0] = float2bfloat(vld1q_f32(p0));
            _r01.val[1] = float2bfloat(vld1q_f32(p1));
            vst2_u16(pp, _r01);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#endif // __ARM_NEON
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __ARM_NEON
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += 4;
        }
#endif // __ARM_NEON
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        transpose_pack_A_tile_fp32_to_bf16_bf16(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x8_t _k0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            uint16x8_t _k1 = vcombine_u16(float2bfloat(vld1q_f32(p0 + A_hstep)), float2bfloat(vld1q_f32(p0 + A_hstep + 4)));
            uint16x8_t _k2 = vcombine_u16(float2bfloat(vld1q_f32(p0 + A_hstep * 2)), float2bfloat(vld1q_f32(p0 + A_hstep * 2 + 4)));
            uint16x8_t _k3 = vcombine_u16(float2bfloat(vld1q_f32(p0 + A_hstep * 3)), float2bfloat(vld1q_f32(p0 + A_hstep * 3 + 4)));

            uint16x4_t _r0 = vget_low_u16(_k0);
            uint16x4_t _r1 = vget_low_u16(_k1);
            uint16x4_t _r2 = vget_low_u16(_k2);
            uint16x4_t _r3 = vget_low_u16(_k3);
            transpose4x4_u16(_r0, _r1, _r2, _r3);
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));

            _r0 = vget_high_u16(_k0);
            _r1 = vget_high_u16(_k1);
            _r2 = vget_high_u16(_k2);
            _r3 = vget_high_u16(_k3);
            transpose4x4_u16(_r0, _r1, _r2, _r3);
            vst1q_u16(pp + 16, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 24, vcombine_u16(_r2, _r3));

            pp += 32;
            p0 += A_hstep * 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x8_t _k0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            uint16x8_t _k1 = vcombine_u16(float2bfloat(vld1q_f32(p0 + A_hstep)), float2bfloat(vld1q_f32(p0 + A_hstep + 4)));
            uint16x8x2_t _r01 = vzipq_u16(_k0, _k1);
            vst1q_u16(pp, _r01.val[0]);
            vst1q_u16(pp + 8, _r01.val[1]);
            pp += 16;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            uint16x8_t _r0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += A_hstep;
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            uint16x8_t _r0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += A_hstep;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p0 + A_hstep));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p0 + A_hstep * 2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p0 + A_hstep * 3));
            transpose4x4_u16(_r0, _r1, _r2, _r3);
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
            pp += 16;
            p0 += A_hstep * 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _k0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _k1 = float2bfloat(vld1q_f32(p0 + A_hstep));
            uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
            vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
            pp += 8;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += A_hstep;
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += A_hstep;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _k0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p0 + A_hstep)));
            uint16x4_t _k1 = float2bfloat(vcombine_f32(vld1_f32(p0 + A_hstep * 2), vld1_f32(p0 + A_hstep * 3)));
            uint16x4x2_t _r01 = vuzp_u16(_k0, _k1);
            vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
            pp += 8;
            p0 += A_hstep * 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[A_hstep]);
            pp[2] = float32_to_bfloat16(p0[1]);
            pp[3] = float32_to_bfloat16(p0[A_hstep + 1]);
            pp += 4;
            p0 += A_hstep * 2;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_fp32_to_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        pack_B_tile_fp32_to_bf16_bf16(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
        const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
        const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
        const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
        const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;
        const float* p8 = (const float*)B + (j + jj + 8) * B_hstep + k;
        const float* p9 = (const float*)B + (j + jj + 9) * B_hstep + k;
        const float* pa = (const float*)B + (j + jj + 10) * B_hstep + k;
        const float* pb = (const float*)B + (j + jj + 11) * B_hstep + k;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p3));
            uint16x4_t _r4 = float2bfloat(vld1q_f32(p4));
            uint16x4_t _r5 = float2bfloat(vld1q_f32(p5));
            uint16x4_t _r6 = float2bfloat(vld1q_f32(p6));
            uint16x4_t _r7 = float2bfloat(vld1q_f32(p7));
            uint16x4_t _r8 = float2bfloat(vld1q_f32(p8));
            uint16x4_t _r9 = float2bfloat(vld1q_f32(p9));
            uint16x4_t _ra = float2bfloat(vld1q_f32(pa));
            uint16x4_t _rb = float2bfloat(vld1q_f32(pb));
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
            vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
            vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
            vst1q_u16(pp + 32, vcombine_u16(_r8, _r9));
            vst1q_u16(pp + 40, vcombine_u16(_ra, _rb));
            pp += 48;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
            p8 += 4;
            p9 += 4;
            pa += 4;
            pb += 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _r0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p1)));
            uint16x4_t _r1 = float2bfloat(vcombine_f32(vld1_f32(p2), vld1_f32(p3)));
            uint16x4_t _r2 = float2bfloat(vcombine_f32(vld1_f32(p4), vld1_f32(p5)));
            uint16x4_t _r3 = float2bfloat(vcombine_f32(vld1_f32(p6), vld1_f32(p7)));
            uint16x4_t _r4 = float2bfloat(vcombine_f32(vld1_f32(p8), vld1_f32(p9)));
            uint16x4_t _r5 = float2bfloat(vcombine_f32(vld1_f32(pa), vld1_f32(pb)));
            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r1);
            vst1_u16(pp + 8, _r2);
            vst1_u16(pp + 12, _r3);
            vst1_u16(pp + 16, _r4);
            vst1_u16(pp + 20, _r5);
            pp += 24;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
            p8 += 2;
            p9 += 2;
            pa += 2;
            pb += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
            pp[8] = float32_to_bfloat16(p8[0]);
            pp[9] = float32_to_bfloat16(p9[0]);
            pp[10] = float32_to_bfloat16(pa[0]);
            pp[11] = float32_to_bfloat16(pb[0]);
            pp += 12;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
            p8++;
            p9++;
            pa++;
            pb++;
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p3));
            uint16x4_t _r4 = float2bfloat(vld1q_f32(p4));
            uint16x4_t _r5 = float2bfloat(vld1q_f32(p5));
            uint16x4_t _r6 = float2bfloat(vld1q_f32(p6));
            uint16x4_t _r7 = float2bfloat(vld1q_f32(p7));
            uint16x4_t _r8 = float2bfloat(vld1q_f32(p8));
            uint16x4_t _r9 = float2bfloat(vld1q_f32(p9));
            uint16x4_t _ra = float2bfloat(vld1q_f32(pa));
            uint16x4_t _rb = float2bfloat(vld1q_f32(pb));

            transpose4x4_u16(_r0, _r1, _r2, _r3);
            transpose4x4_u16(_r4, _r5, _r6, _r7);
            transpose4x4_u16(_r8, _r9, _ra, _rb);

            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r4);
            vst1_u16(pp + 8, _r8);
            vst1_u16(pp + 12, _r1);
            vst1_u16(pp + 16, _r5);
            vst1_u16(pp + 20, _r9);
            vst1_u16(pp + 24, _r2);
            vst1_u16(pp + 28, _r6);
            vst1_u16(pp + 32, _ra);
            vst1_u16(pp + 36, _r3);
            vst1_u16(pp + 40, _r7);
            vst1_u16(pp + 44, _rb);
            pp += 48;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
            p8 += 4;
            p9 += 4;
            pa += 4;
            pb += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
            pp[8] = float32_to_bfloat16(p8[0]);
            pp[9] = float32_to_bfloat16(p9[0]);
            pp[10] = float32_to_bfloat16(pa[0]);
            pp[11] = float32_to_bfloat16(pb[0]);
            pp += 12;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
            p8++;
            p9++;
            pa++;
            pb++;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
        const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
        const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
        const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
        const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p3));
            uint16x4_t _r4 = float2bfloat(vld1q_f32(p4));
            uint16x4_t _r5 = float2bfloat(vld1q_f32(p5));
            uint16x4_t _r6 = float2bfloat(vld1q_f32(p6));
            uint16x4_t _r7 = float2bfloat(vld1q_f32(p7));
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
            vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
            vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
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
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _r0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p1)));
            uint16x4_t _r1 = float2bfloat(vcombine_f32(vld1_f32(p2), vld1_f32(p3)));
            uint16x4_t _r2 = float2bfloat(vcombine_f32(vld1_f32(p4), vld1_f32(p5)));
            uint16x4_t _r3 = float2bfloat(vcombine_f32(vld1_f32(p6), vld1_f32(p7)));
            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r1);
            vst1_u16(pp + 8, _r2);
            vst1_u16(pp + 12, _r3);
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
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
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            uint16x8_t _r1 = vcombine_u16(float2bfloat(vld1q_f32(p1)), float2bfloat(vld1q_f32(p1 + 4)));
            uint16x8_t _r2 = vcombine_u16(float2bfloat(vld1q_f32(p2)), float2bfloat(vld1q_f32(p2 + 4)));
            uint16x8_t _r3 = vcombine_u16(float2bfloat(vld1q_f32(p3)), float2bfloat(vld1q_f32(p3 + 4)));
            uint16x8_t _r4 = vcombine_u16(float2bfloat(vld1q_f32(p4)), float2bfloat(vld1q_f32(p4 + 4)));
            uint16x8_t _r5 = vcombine_u16(float2bfloat(vld1q_f32(p5)), float2bfloat(vld1q_f32(p5 + 4)));
            uint16x8_t _r6 = vcombine_u16(float2bfloat(vld1q_f32(p6)), float2bfloat(vld1q_f32(p6 + 4)));
            uint16x8_t _r7 = vcombine_u16(float2bfloat(vld1q_f32(p7)), float2bfloat(vld1q_f32(p7 + 4)));
            transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            vst1q_u16(pp, _r0);
            vst1q_u16(pp + 8, _r1);
            vst1q_u16(pp + 16, _r2);
            vst1q_u16(pp + 24, _r3);
            vst1q_u16(pp + 32, _r4);
            vst1q_u16(pp + 40, _r5);
            vst1q_u16(pp + 48, _r6);
            vst1q_u16(pp + 56, _r7);
            pp += 64;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p3));
            uint16x4_t _r4 = float2bfloat(vld1q_f32(p4));
            uint16x4_t _r5 = float2bfloat(vld1q_f32(p5));
            uint16x4_t _r6 = float2bfloat(vld1q_f32(p6));
            uint16x4_t _r7 = float2bfloat(vld1q_f32(p7));

            transpose4x4_u16(_r0, _r1, _r2, _r3);
            transpose4x4_u16(_r4, _r5, _r6, _r7);

            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r4);
            vst1_u16(pp + 8, _r1);
            vst1_u16(pp + 12, _r5);
            vst1_u16(pp + 16, _r2);
            vst1_u16(pp + 20, _r6);
            vst1_u16(pp + 24, _r3);
            vst1_u16(pp + 28, _r7);
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
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
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
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p3));
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _r0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p1)));
            uint16x4_t _r1 = float2bfloat(vcombine_f32(vld1_f32(p2), vld1_f32(p3)));
            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r1);
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x4_t _r0123;
            _r0123.val[0] = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            _r0123.val[1] = vcombine_u16(float2bfloat(vld1q_f32(p1)), float2bfloat(vld1q_f32(p1 + 4)));
            _r0123.val[2] = vcombine_u16(float2bfloat(vld1q_f32(p2)), float2bfloat(vld1q_f32(p2 + 4)));
            _r0123.val[3] = vcombine_u16(float2bfloat(vld1q_f32(p3)), float2bfloat(vld1q_f32(p3 + 4)));
            vst4q_u16(pp, _r0123);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x4_t _r0123;
            _r0123.val[0] = float2bfloat(vld1q_f32(p0));
            _r0123.val[1] = float2bfloat(vld1q_f32(p1));
            _r0123.val[2] = float2bfloat(vld1q_f32(p2));
            _r0123.val[3] = float2bfloat(vld1q_f32(p3));
            vst4_u16(pp, _r0123);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p1));
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _r0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p1)));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __ARM_NEON
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x2_t _r01;
            _r01.val[0] = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            _r01.val[1] = vcombine_u16(float2bfloat(vld1q_f32(p1)), float2bfloat(vld1q_f32(p1 + 4)));
            vst2q_u16(pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x2_t _r01;
            _r01.val[0] = float2bfloat(vld1q_f32(p0));
            _r01.val[1] = float2bfloat(vld1q_f32(p1));
            vst2_u16(pp, _r01);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#endif // __ARM_NEON
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
#if __ARM_NEON
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += 4;
        }
#endif // __ARM_NEON
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        transpose_pack_B_tile_fp32_to_bf16_bf16(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p0 + 4));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p0 + 8));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p0 + B_hstep));
            uint16x4_t _r4 = float2bfloat(vld1q_f32(p0 + B_hstep + 4));
            uint16x4_t _r5 = float2bfloat(vld1q_f32(p0 + B_hstep + 8));
            uint16x4_t _r6 = float2bfloat(vld1q_f32(p0 + B_hstep * 2));
            uint16x4_t _r7 = float2bfloat(vld1q_f32(p0 + B_hstep * 2 + 4));
            uint16x4_t _r8 = float2bfloat(vld1q_f32(p0 + B_hstep * 2 + 8));
            uint16x4_t _r9 = float2bfloat(vld1q_f32(p0 + B_hstep * 3));
            uint16x4_t _ra = float2bfloat(vld1q_f32(p0 + B_hstep * 3 + 4));
            uint16x4_t _rb = float2bfloat(vld1q_f32(p0 + B_hstep * 3 + 8));

            transpose4x4_u16(_r0, _r3, _r6, _r9);
            transpose4x4_u16(_r1, _r4, _r7, _ra);
            transpose4x4_u16(_r2, _r5, _r8, _rb);

            vst1q_u16(pp, vcombine_u16(_r0, _r3));
            vst1q_u16(pp + 8, vcombine_u16(_r6, _r9));
            vst1q_u16(pp + 16, vcombine_u16(_r1, _r4));
            vst1q_u16(pp + 24, vcombine_u16(_r7, _ra));
            vst1q_u16(pp + 32, vcombine_u16(_r2, _r5));
            vst1q_u16(pp + 40, vcombine_u16(_r8, _rb));
            pp += 48;
            p0 += B_hstep * 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _k0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _k1 = float2bfloat(vld1q_f32(p0 + B_hstep));
            uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
            vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
            _k0 = float2bfloat(vld1q_f32(p0 + 4));
            _k1 = float2bfloat(vld1q_f32(p0 + B_hstep + 4));
            _r01 = vzip_u16(_k0, _k1);
            vst1q_u16(pp + 8, vcombine_u16(_r01.val[0], _r01.val[1]));
            _k0 = float2bfloat(vld1q_f32(p0 + 8));
            _k1 = float2bfloat(vld1q_f32(p0 + B_hstep + 8));
            _r01 = vzip_u16(_k0, _k1);
            vst1q_u16(pp + 16, vcombine_u16(_r01.val[0], _r01.val[1]));
            pp += 24;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            vst1_u16(pp, float2bfloat(vld1q_f32(p0)));
            vst1_u16(pp + 4, float2bfloat(vld1q_f32(p0 + 4)));
            vst1_u16(pp + 8, float2bfloat(vld1q_f32(p0 + 8)));
            pp += 12;
            p0 += B_hstep;
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            vst1_u16(pp, float2bfloat(vld1q_f32(p0)));
            vst1_u16(pp + 4, float2bfloat(vld1q_f32(p0 + 4)));
            vst1_u16(pp + 8, float2bfloat(vld1q_f32(p0 + 8)));
            pp += 12;
            p0 += B_hstep;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x8_t _k0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            uint16x8_t _k1 = vcombine_u16(float2bfloat(vld1q_f32(p0 + B_hstep)), float2bfloat(vld1q_f32(p0 + B_hstep + 4)));
            uint16x8_t _k2 = vcombine_u16(float2bfloat(vld1q_f32(p0 + B_hstep * 2)), float2bfloat(vld1q_f32(p0 + B_hstep * 2 + 4)));
            uint16x8_t _k3 = vcombine_u16(float2bfloat(vld1q_f32(p0 + B_hstep * 3)), float2bfloat(vld1q_f32(p0 + B_hstep * 3 + 4)));

            uint16x4_t _r0 = vget_low_u16(_k0);
            uint16x4_t _r1 = vget_low_u16(_k1);
            uint16x4_t _r2 = vget_low_u16(_k2);
            uint16x4_t _r3 = vget_low_u16(_k3);
            transpose4x4_u16(_r0, _r1, _r2, _r3);
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));

            _r0 = vget_high_u16(_k0);
            _r1 = vget_high_u16(_k1);
            _r2 = vget_high_u16(_k2);
            _r3 = vget_high_u16(_k3);
            transpose4x4_u16(_r0, _r1, _r2, _r3);
            vst1q_u16(pp + 16, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 24, vcombine_u16(_r2, _r3));

            pp += 32;
            p0 += B_hstep * 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _k0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _k1 = float2bfloat(vld1q_f32(p0 + B_hstep));
            uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
            vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
            _k0 = float2bfloat(vld1q_f32(p0 + 4));
            _k1 = float2bfloat(vld1q_f32(p0 + B_hstep + 4));
            _r01 = vzip_u16(_k0, _k1);
            vst1q_u16(pp + 8, vcombine_u16(_r01.val[0], _r01.val[1]));
            pp += 16;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            uint16x8_t _r0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += B_hstep;
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            uint16x8_t _r0 = vcombine_u16(float2bfloat(vld1q_f32(p0)), float2bfloat(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += B_hstep;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _r1 = float2bfloat(vld1q_f32(p0 + B_hstep));
            uint16x4_t _r2 = float2bfloat(vld1q_f32(p0 + B_hstep * 2));
            uint16x4_t _r3 = float2bfloat(vld1q_f32(p0 + B_hstep * 3));
            transpose4x4_u16(_r0, _r1, _r2, _r3);
            vst1q_u16(pp, vcombine_u16(_r0, _r1));
            vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
            pp += 16;
            p0 += B_hstep * 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            uint16x4_t _k0 = float2bfloat(vld1q_f32(p0));
            uint16x4_t _k1 = float2bfloat(vld1q_f32(p0 + B_hstep));
            uint16x4x2_t _r01 = vzip_u16(_k0, _k1);
            vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
            pp += 8;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += B_hstep;
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            uint16x4_t _r0 = float2bfloat(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += B_hstep;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _k0 = float2bfloat(vcombine_f32(vld1_f32(p0), vld1_f32(p0 + B_hstep)));
            uint16x4_t _k1 = float2bfloat(vcombine_f32(vld1_f32(p0 + B_hstep * 2), vld1_f32(p0 + B_hstep * 3)));
            uint16x4x2_t _r01 = vuzp_u16(_k0, _k1);
            vst1q_u16(pp, vcombine_u16(_r01.val[0], _r01.val[1]));
            pp += 8;
            p0 += B_hstep * 4;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[B_hstep]);
            pp[2] = float32_to_bfloat16(p0[1]);
            pp[3] = float32_to_bfloat16(p0[B_hstep + 1]);
            pp += 4;
            p0 += B_hstep * 2;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void unpack_output_tile_fp32_to_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose, int output_elemtype)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        unpack_output_tile_fp32_to_bf16_bf16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    const float* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0;
        float* p0f;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC = (const float*)C + i + ii;
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _f00 = vld1q_f32(pp + 0);
            float32x4_t _f01 = vld1q_f32(pp + 4);
            float32x4_t _f10 = vld1q_f32(pp + 8);
            float32x4_t _f11 = vld1q_f32(pp + 12);
            float32x4_t _f20 = vld1q_f32(pp + 16);
            float32x4_t _f21 = vld1q_f32(pp + 20);
            float32x4_t _f30 = vld1q_f32(pp + 24);
            float32x4_t _f31 = vld1q_f32(pp + 28);
            float32x4_t _f40 = vld1q_f32(pp + 32);
            float32x4_t _f41 = vld1q_f32(pp + 36);
            float32x4_t _f50 = vld1q_f32(pp + 40);
            float32x4_t _f51 = vld1q_f32(pp + 44);
            float32x4_t _f60 = vld1q_f32(pp + 48);
            float32x4_t _f61 = vld1q_f32(pp + 52);
            float32x4_t _f70 = vld1q_f32(pp + 56);
            float32x4_t _f71 = vld1q_f32(pp + 60);
            float32x4_t _f80 = vld1q_f32(pp + 64);
            float32x4_t _f81 = vld1q_f32(pp + 68);
            float32x4_t _f90 = vld1q_f32(pp + 72);
            float32x4_t _f91 = vld1q_f32(pp + 76);
            float32x4_t _fa0 = vld1q_f32(pp + 80);
            float32x4_t _fa1 = vld1q_f32(pp + 84);
            float32x4_t _fb0 = vld1q_f32(pp + 88);
            float32x4_t _fb1 = vld1q_f32(pp + 92);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4x2_t _r0 = vuzpq_f32(_f00, _f01);
            float32x4x2_t _r1 = vuzpq_f32(_f10, _f11);
            _f00 = _r0.val[0];
            _f10 = _r0.val[1];
            _f01 = _r1.val[0];
            _f11 = _r1.val[1];
            _r0 = vuzpq_f32(_f20, _f21);
            _r1 = vuzpq_f32(_f30, _f31);
            _f20 = _r0.val[0];
            _f30 = _r0.val[1];
            _f21 = _r1.val[0];
            _f31 = _r1.val[1];
            _r0 = vuzpq_f32(_f40, _f41);
            _r1 = vuzpq_f32(_f50, _f51);
            _f40 = _r0.val[0];
            _f50 = _r0.val[1];
            _f41 = _r1.val[0];
            _f51 = _r1.val[1];
            _r0 = vuzpq_f32(_f60, _f61);
            _r1 = vuzpq_f32(_f70, _f71);
            _f60 = _r0.val[0];
            _f70 = _r0.val[1];
            _f61 = _r1.val[0];
            _f71 = _r1.val[1];
            _r0 = vuzpq_f32(_f80, _f81);
            _r1 = vuzpq_f32(_f90, _f91);
            _f80 = _r0.val[0];
            _f90 = _r0.val[1];
            _f81 = _r1.val[0];
            _f91 = _r1.val[1];
            _r0 = vuzpq_f32(_fa0, _fa1);
            _r1 = vuzpq_f32(_fb0, _fb1);
            _fa0 = _r0.val[0];
            _fb0 = _r0.val[1];
            _fa1 = _r1.val[0];
            _fb1 = _r1.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c);
                    _f01 = vaddq_f32(_f01, _c);
                    _f10 = vaddq_f32(_f10, _c);
                    _f11 = vaddq_f32(_f11, _c);
                    _f20 = vaddq_f32(_f20, _c);
                    _f21 = vaddq_f32(_f21, _c);
                    _f30 = vaddq_f32(_f30, _c);
                    _f31 = vaddq_f32(_f31, _c);
                    _f40 = vaddq_f32(_f40, _c);
                    _f41 = vaddq_f32(_f41, _c);
                    _f50 = vaddq_f32(_f50, _c);
                    _f51 = vaddq_f32(_f51, _c);
                    _f60 = vaddq_f32(_f60, _c);
                    _f61 = vaddq_f32(_f61, _c);
                    _f70 = vaddq_f32(_f70, _c);
                    _f71 = vaddq_f32(_f71, _c);
                    _f80 = vaddq_f32(_f80, _c);
                    _f81 = vaddq_f32(_f81, _c);
                    _f90 = vaddq_f32(_f90, _c);
                    _f91 = vaddq_f32(_f91, _c);
                    _fa0 = vaddq_f32(_fa0, _c);
                    _fa1 = vaddq_f32(_fa1, _c);
                    _fb0 = vaddq_f32(_fb0, _c);
                    _fb1 = vaddq_f32(_fb1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c1);
                    _f10 = vaddq_f32(_f10, _c0);
                    _f11 = vaddq_f32(_f11, _c1);
                    _f20 = vaddq_f32(_f20, _c0);
                    _f21 = vaddq_f32(_f21, _c1);
                    _f30 = vaddq_f32(_f30, _c0);
                    _f31 = vaddq_f32(_f31, _c1);
                    _f40 = vaddq_f32(_f40, _c0);
                    _f41 = vaddq_f32(_f41, _c1);
                    _f50 = vaddq_f32(_f50, _c0);
                    _f51 = vaddq_f32(_f51, _c1);
                    _f60 = vaddq_f32(_f60, _c0);
                    _f61 = vaddq_f32(_f61, _c1);
                    _f70 = vaddq_f32(_f70, _c0);
                    _f71 = vaddq_f32(_f71, _c1);
                    _f80 = vaddq_f32(_f80, _c0);
                    _f81 = vaddq_f32(_f81, _c1);
                    _f90 = vaddq_f32(_f90, _c0);
                    _f91 = vaddq_f32(_f91, _c1);
                    _fa0 = vaddq_f32(_fa0, _c0);
                    _fa1 = vaddq_f32(_fa1, _c1);
                    _fb0 = vaddq_f32(_fb0, _c0);
                    _fb1 = vaddq_f32(_fb1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c00 = vld1q_f32(pC);
                        float32x4_t _c01 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c10 = vld1q_f32(pC + 4);
                        float32x4_t _c11 = vld1q_f32(pC + c_hstep * 4 + 4);
                        float32x4_t _c20 = vld1q_f32(pC + 8);
                        float32x4_t _c21 = vld1q_f32(pC + c_hstep * 4 + 8);
                        float32x4_t _c30 = vld1q_f32(pC + 12);
                        float32x4_t _c31 = vld1q_f32(pC + c_hstep * 4 + 12);
                        float32x4_t _c40 = vld1q_f32(pC + 16);
                        float32x4_t _c41 = vld1q_f32(pC + c_hstep * 4 + 16);
                        float32x4_t _c50 = vld1q_f32(pC + 20);
                        float32x4_t _c51 = vld1q_f32(pC + c_hstep * 4 + 20);
                        float32x4_t _c60 = vld1q_f32(pC + 24);
                        float32x4_t _c61 = vld1q_f32(pC + c_hstep * 4 + 24);
                        float32x4_t _c70 = vld1q_f32(pC + 28);
                        float32x4_t _c71 = vld1q_f32(pC + c_hstep * 4 + 28);
                        float32x4_t _c80 = vld1q_f32(pC + 32);
                        float32x4_t _c81 = vld1q_f32(pC + c_hstep * 4 + 32);
                        float32x4_t _c90 = vld1q_f32(pC + 36);
                        float32x4_t _c91 = vld1q_f32(pC + c_hstep * 4 + 36);
                        float32x4_t _ca0 = vld1q_f32(pC + 40);
                        float32x4_t _ca1 = vld1q_f32(pC + c_hstep * 4 + 40);
                        float32x4_t _cb0 = vld1q_f32(pC + 44);
                        float32x4_t _cb1 = vld1q_f32(pC + c_hstep * 4 + 44);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c00);
                            _f01 = vaddq_f32(_f01, _c01);
                            _f10 = vaddq_f32(_f10, _c10);
                            _f11 = vaddq_f32(_f11, _c11);
                            _f20 = vaddq_f32(_f20, _c20);
                            _f21 = vaddq_f32(_f21, _c21);
                            _f30 = vaddq_f32(_f30, _c30);
                            _f31 = vaddq_f32(_f31, _c31);
                            _f40 = vaddq_f32(_f40, _c40);
                            _f41 = vaddq_f32(_f41, _c41);
                            _f50 = vaddq_f32(_f50, _c50);
                            _f51 = vaddq_f32(_f51, _c51);
                            _f60 = vaddq_f32(_f60, _c60);
                            _f61 = vaddq_f32(_f61, _c61);
                            _f70 = vaddq_f32(_f70, _c70);
                            _f71 = vaddq_f32(_f71, _c71);
                            _f80 = vaddq_f32(_f80, _c80);
                            _f81 = vaddq_f32(_f81, _c81);
                            _f90 = vaddq_f32(_f90, _c90);
                            _f91 = vaddq_f32(_f91, _c91);
                            _fa0 = vaddq_f32(_fa0, _ca0);
                            _fa1 = vaddq_f32(_fa1, _ca1);
                            _fb0 = vaddq_f32(_fb0, _cb0);
                            _fb1 = vaddq_f32(_fb1, _cb1);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c00, _beta);
                            _f01 = vmlaq_f32(_f01, _c01, _beta);
                            _f10 = vmlaq_f32(_f10, _c10, _beta);
                            _f11 = vmlaq_f32(_f11, _c11, _beta);
                            _f20 = vmlaq_f32(_f20, _c20, _beta);
                            _f21 = vmlaq_f32(_f21, _c21, _beta);
                            _f30 = vmlaq_f32(_f30, _c30, _beta);
                            _f31 = vmlaq_f32(_f31, _c31, _beta);
                            _f40 = vmlaq_f32(_f40, _c40, _beta);
                            _f41 = vmlaq_f32(_f41, _c41, _beta);
                            _f50 = vmlaq_f32(_f50, _c50, _beta);
                            _f51 = vmlaq_f32(_f51, _c51, _beta);
                            _f60 = vmlaq_f32(_f60, _c60, _beta);
                            _f61 = vmlaq_f32(_f61, _c61, _beta);
                            _f70 = vmlaq_f32(_f70, _c70, _beta);
                            _f71 = vmlaq_f32(_f71, _c71, _beta);
                            _f80 = vmlaq_f32(_f80, _c80, _beta);
                            _f81 = vmlaq_f32(_f81, _c81, _beta);
                            _f90 = vmlaq_f32(_f90, _c90, _beta);
                            _f91 = vmlaq_f32(_f91, _c91, _beta);
                            _fa0 = vmlaq_f32(_fa0, _ca0, _beta);
                            _fa1 = vmlaq_f32(_fa1, _ca1, _beta);
                            _fb0 = vmlaq_f32(_fb0, _cb0, _beta);
                            _fb1 = vmlaq_f32(_fb1, _cb1, _beta);
                        }
                        pC += 12 * c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c000 = vld1q_f32(pC);
                        float32x4_t _c001 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c002 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c003 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c000, _c001, _c002, _c003);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c000);
                            _f10 = vaddq_f32(_f10, _c001);
                            _f20 = vaddq_f32(_f20, _c002);
                            _f30 = vaddq_f32(_f30, _c003);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c000, _beta);
                            _f10 = vmlaq_f32(_f10, _c001, _beta);
                            _f20 = vmlaq_f32(_f20, _c002, _beta);
                            _f30 = vmlaq_f32(_f30, _c003, _beta);
                        }
                        float32x4_t _c010 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c011 = vld1q_f32(pC + c_hstep * 5);
                        float32x4_t _c012 = vld1q_f32(pC + c_hstep * 6);
                        float32x4_t _c013 = vld1q_f32(pC + c_hstep * 7);
                        transpose4x4_ps(_c010, _c011, _c012, _c013);
                        if (beta == 1.f)
                        {
                            _f01 = vaddq_f32(_f01, _c010);
                            _f11 = vaddq_f32(_f11, _c011);
                            _f21 = vaddq_f32(_f21, _c012);
                            _f31 = vaddq_f32(_f31, _c013);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f01 = vmlaq_f32(_f01, _c010, _beta);
                            _f11 = vmlaq_f32(_f11, _c011, _beta);
                            _f21 = vmlaq_f32(_f21, _c012, _beta);
                            _f31 = vmlaq_f32(_f31, _c013, _beta);
                        }
                        float32x4_t _c400 = vld1q_f32(pC + 4);
                        float32x4_t _c401 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c402 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c403 = vld1q_f32(pC + c_hstep * 3 + 4);
                        transpose4x4_ps(_c400, _c401, _c402, _c403);
                        if (beta == 1.f)
                        {
                            _f40 = vaddq_f32(_f40, _c400);
                            _f50 = vaddq_f32(_f50, _c401);
                            _f60 = vaddq_f32(_f60, _c402);
                            _f70 = vaddq_f32(_f70, _c403);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f40 = vmlaq_f32(_f40, _c400, _beta);
                            _f50 = vmlaq_f32(_f50, _c401, _beta);
                            _f60 = vmlaq_f32(_f60, _c402, _beta);
                            _f70 = vmlaq_f32(_f70, _c403, _beta);
                        }
                        float32x4_t _c410 = vld1q_f32(pC + c_hstep * 4 + 4);
                        float32x4_t _c411 = vld1q_f32(pC + c_hstep * 5 + 4);
                        float32x4_t _c412 = vld1q_f32(pC + c_hstep * 6 + 4);
                        float32x4_t _c413 = vld1q_f32(pC + c_hstep * 7 + 4);
                        transpose4x4_ps(_c410, _c411, _c412, _c413);
                        if (beta == 1.f)
                        {
                            _f41 = vaddq_f32(_f41, _c410);
                            _f51 = vaddq_f32(_f51, _c411);
                            _f61 = vaddq_f32(_f61, _c412);
                            _f71 = vaddq_f32(_f71, _c413);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f41 = vmlaq_f32(_f41, _c410, _beta);
                            _f51 = vmlaq_f32(_f51, _c411, _beta);
                            _f61 = vmlaq_f32(_f61, _c412, _beta);
                            _f71 = vmlaq_f32(_f71, _c413, _beta);
                        }
                        float32x4_t _c800 = vld1q_f32(pC + 8);
                        float32x4_t _c801 = vld1q_f32(pC + c_hstep + 8);
                        float32x4_t _c802 = vld1q_f32(pC + c_hstep * 2 + 8);
                        float32x4_t _c803 = vld1q_f32(pC + c_hstep * 3 + 8);
                        transpose4x4_ps(_c800, _c801, _c802, _c803);
                        if (beta == 1.f)
                        {
                            _f80 = vaddq_f32(_f80, _c800);
                            _f90 = vaddq_f32(_f90, _c801);
                            _fa0 = vaddq_f32(_fa0, _c802);
                            _fb0 = vaddq_f32(_fb0, _c803);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f80 = vmlaq_f32(_f80, _c800, _beta);
                            _f90 = vmlaq_f32(_f90, _c801, _beta);
                            _fa0 = vmlaq_f32(_fa0, _c802, _beta);
                            _fb0 = vmlaq_f32(_fb0, _c803, _beta);
                        }
                        float32x4_t _c810 = vld1q_f32(pC + c_hstep * 4 + 8);
                        float32x4_t _c811 = vld1q_f32(pC + c_hstep * 5 + 8);
                        float32x4_t _c812 = vld1q_f32(pC + c_hstep * 6 + 8);
                        float32x4_t _c813 = vld1q_f32(pC + c_hstep * 7 + 8);
                        transpose4x4_ps(_c810, _c811, _c812, _c813);
                        if (beta == 1.f)
                        {
                            _f81 = vaddq_f32(_f81, _c810);
                            _f91 = vaddq_f32(_f91, _c811);
                            _fa1 = vaddq_f32(_fa1, _c812);
                            _fb1 = vaddq_f32(_fb1, _c813);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f81 = vmlaq_f32(_f81, _c810, _beta);
                            _f91 = vmlaq_f32(_f91, _c811, _beta);
                            _fa1 = vmlaq_f32(_fa1, _c812, _beta);
                            _fb1 = vmlaq_f32(_fb1, _c813, _beta);
                        }
                        pC += 12;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c0);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f10 = vaddq_f32(_f10, _c1);
                    _f11 = vaddq_f32(_f11, _c1);
                    float32x4_t _c2 = vdupq_n_f32(pC[2] * beta);
                    _f20 = vaddq_f32(_f20, _c2);
                    _f21 = vaddq_f32(_f21, _c2);
                    float32x4_t _c3 = vdupq_n_f32(pC[3] * beta);
                    _f30 = vaddq_f32(_f30, _c3);
                    _f31 = vaddq_f32(_f31, _c3);
                    float32x4_t _c4 = vdupq_n_f32(pC[4] * beta);
                    _f40 = vaddq_f32(_f40, _c4);
                    _f41 = vaddq_f32(_f41, _c4);
                    float32x4_t _c5 = vdupq_n_f32(pC[5] * beta);
                    _f50 = vaddq_f32(_f50, _c5);
                    _f51 = vaddq_f32(_f51, _c5);
                    float32x4_t _c6 = vdupq_n_f32(pC[6] * beta);
                    _f60 = vaddq_f32(_f60, _c6);
                    _f61 = vaddq_f32(_f61, _c6);
                    float32x4_t _c7 = vdupq_n_f32(pC[7] * beta);
                    _f70 = vaddq_f32(_f70, _c7);
                    _f71 = vaddq_f32(_f71, _c7);
                    float32x4_t _c8 = vdupq_n_f32(pC[8] * beta);
                    _f80 = vaddq_f32(_f80, _c8);
                    _f81 = vaddq_f32(_f81, _c8);
                    float32x4_t _c9 = vdupq_n_f32(pC[9] * beta);
                    _f90 = vaddq_f32(_f90, _c9);
                    _f91 = vaddq_f32(_f91, _c9);
                    float32x4_t _ca = vdupq_n_f32(pC[10] * beta);
                    _fa0 = vaddq_f32(_fa0, _ca);
                    _fa1 = vaddq_f32(_fa1, _ca);
                    float32x4_t _cb = vdupq_n_f32(pC[11] * beta);
                    _fb0 = vaddq_f32(_fb0, _cb);
                    _fb1 = vaddq_f32(_fb1, _cb);
                    pC += 12;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f00 = vmulq_f32(_f00, _alpha);
                _f01 = vmulq_f32(_f01, _alpha);
                _f10 = vmulq_f32(_f10, _alpha);
                _f11 = vmulq_f32(_f11, _alpha);
                _f20 = vmulq_f32(_f20, _alpha);
                _f21 = vmulq_f32(_f21, _alpha);
                _f30 = vmulq_f32(_f30, _alpha);
                _f31 = vmulq_f32(_f31, _alpha);
                _f40 = vmulq_f32(_f40, _alpha);
                _f41 = vmulq_f32(_f41, _alpha);
                _f50 = vmulq_f32(_f50, _alpha);
                _f51 = vmulq_f32(_f51, _alpha);
                _f60 = vmulq_f32(_f60, _alpha);
                _f61 = vmulq_f32(_f61, _alpha);
                _f70 = vmulq_f32(_f70, _alpha);
                _f71 = vmulq_f32(_f71, _alpha);
                _f80 = vmulq_f32(_f80, _alpha);
                _f81 = vmulq_f32(_f81, _alpha);
                _f90 = vmulq_f32(_f90, _alpha);
                _f91 = vmulq_f32(_f91, _alpha);
                _fa0 = vmulq_f32(_fa0, _alpha);
                _fa1 = vmulq_f32(_fa1, _alpha);
                _fb0 = vmulq_f32(_fb0, _alpha);
                _fb1 = vmulq_f32(_fb1, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float32x4x4_t _r0;
                        _r0.val[0] = _f00;
                        _r0.val[1] = _f10;
                        _r0.val[2] = _f20;
                        _r0.val[3] = _f30;
                        vst4q_f32(p0f, _r0);
                        float32x4x4_t _r1;
                        _r1.val[0] = _f01;
                        _r1.val[1] = _f11;
                        _r1.val[2] = _f21;
                        _r1.val[3] = _f31;
                        vst4q_f32(p0f + 16, _r1);
                        float32x4x4_t _r2;
                        _r2.val[0] = _f40;
                        _r2.val[1] = _f50;
                        _r2.val[2] = _f60;
                        _r2.val[3] = _f70;
                        vst4q_f32(p0f + out_hstep * 4, _r2);
                        float32x4x4_t _r3;
                        _r3.val[0] = _f41;
                        _r3.val[1] = _f51;
                        _r3.val[2] = _f61;
                        _r3.val[3] = _f71;
                        vst4q_f32(p0f + out_hstep * 4 + 16, _r3);
                        float32x4x4_t _r4;
                        _r4.val[0] = _f80;
                        _r4.val[1] = _f90;
                        _r4.val[2] = _fa0;
                        _r4.val[3] = _fb0;
                        vst4q_f32(p0f + out_hstep * 8, _r4);
                        float32x4x4_t _r5;
                        _r5.val[0] = _f81;
                        _r5.val[1] = _f91;
                        _r5.val[2] = _fa1;
                        _r5.val[3] = _fb1;
                        vst4q_f32(p0f + out_hstep * 8 + 16, _r5);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + 4, _f01);
                        vst1q_f32(p0f + out_hstep, _f10);
                        vst1q_f32(p0f + out_hstep + 4, _f11);
                        vst1q_f32(p0f + out_hstep * 2, _f20);
                        vst1q_f32(p0f + out_hstep * 2 + 4, _f21);
                        vst1q_f32(p0f + out_hstep * 3, _f30);
                        vst1q_f32(p0f + out_hstep * 3 + 4, _f31);
                        vst1q_f32(p0f + out_hstep * 4, _f40);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f41);
                        vst1q_f32(p0f + out_hstep * 5, _f50);
                        vst1q_f32(p0f + out_hstep * 5 + 4, _f51);
                        vst1q_f32(p0f + out_hstep * 6, _f60);
                        vst1q_f32(p0f + out_hstep * 6 + 4, _f61);
                        vst1q_f32(p0f + out_hstep * 7, _f70);
                        vst1q_f32(p0f + out_hstep * 7 + 4, _f71);
                        vst1q_f32(p0f + out_hstep * 8, _f80);
                        vst1q_f32(p0f + out_hstep * 8 + 4, _f81);
                        vst1q_f32(p0f + out_hstep * 9, _f90);
                        vst1q_f32(p0f + out_hstep * 9 + 4, _f91);
                        vst1q_f32(p0f + out_hstep * 10, _fa0);
                        vst1q_f32(p0f + out_hstep * 10 + 4, _fa1);
                        vst1q_f32(p0f + out_hstep * 11, _fb0);
                        vst1q_f32(p0f + out_hstep * 11 + 4, _fb1);
                    }
                    p0f += out_hstep * 12;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + out_hstep * 4, _f01);
                        vst1q_f32(p0f + 4, _f10);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f11);
                        vst1q_f32(p0f + 8, _f20);
                        vst1q_f32(p0f + out_hstep * 4 + 8, _f21);
                        vst1q_f32(p0f + 12, _f30);
                        vst1q_f32(p0f + out_hstep * 4 + 12, _f31);
                        vst1q_f32(p0f + 16, _f40);
                        vst1q_f32(p0f + out_hstep * 4 + 16, _f41);
                        vst1q_f32(p0f + 20, _f50);
                        vst1q_f32(p0f + out_hstep * 4 + 20, _f51);
                        vst1q_f32(p0f + 24, _f60);
                        vst1q_f32(p0f + out_hstep * 4 + 24, _f61);
                        vst1q_f32(p0f + 28, _f70);
                        vst1q_f32(p0f + out_hstep * 4 + 28, _f71);
                        vst1q_f32(p0f + 32, _f80);
                        vst1q_f32(p0f + out_hstep * 4 + 32, _f81);
                        vst1q_f32(p0f + 36, _f90);
                        vst1q_f32(p0f + out_hstep * 4 + 36, _f91);
                        vst1q_f32(p0f + 40, _fa0);
                        vst1q_f32(p0f + out_hstep * 4 + 40, _fa1);
                        vst1q_f32(p0f + 44, _fb0);
                        vst1q_f32(p0f + out_hstep * 4 + 44, _fb1);
                        p0f += 48;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_f00, _f10, _f20, _f30);
                        transpose4x4_ps(_f40, _f50, _f60, _f70);
                        transpose4x4_ps(_f80, _f90, _fa0, _fb0);
                        transpose4x4_ps(_f01, _f11, _f21, _f31);
                        transpose4x4_ps(_f41, _f51, _f61, _f71);
                        transpose4x4_ps(_f81, _f91, _fa1, _fb1);
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + 4, _f40);
                        vst1q_f32(p0f + 8, _f80);
                        vst1q_f32(p0f + out_hstep, _f10);
                        vst1q_f32(p0f + out_hstep + 4, _f50);
                        vst1q_f32(p0f + out_hstep + 8, _f90);
                        vst1q_f32(p0f + out_hstep * 2, _f20);
                        vst1q_f32(p0f + out_hstep * 2 + 4, _f60);
                        vst1q_f32(p0f + out_hstep * 2 + 8, _fa0);
                        vst1q_f32(p0f + out_hstep * 3, _f30);
                        vst1q_f32(p0f + out_hstep * 3 + 4, _f70);
                        vst1q_f32(p0f + out_hstep * 3 + 8, _fb0);
                        vst1q_f32(p0f + out_hstep * 4, _f01);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f41);
                        vst1q_f32(p0f + out_hstep * 4 + 8, _f81);
                        vst1q_f32(p0f + out_hstep * 5, _f11);
                        vst1q_f32(p0f + out_hstep * 5 + 4, _f51);
                        vst1q_f32(p0f + out_hstep * 5 + 8, _f91);
                        vst1q_f32(p0f + out_hstep * 6, _f21);
                        vst1q_f32(p0f + out_hstep * 6 + 4, _f61);
                        vst1q_f32(p0f + out_hstep * 6 + 8, _fa1);
                        vst1q_f32(p0f + out_hstep * 7, _f31);
                        vst1q_f32(p0f + out_hstep * 7 + 4, _f71);
                        vst1q_f32(p0f + out_hstep * 7 + 8, _fb1);
                        p0f += 12;
                    }
                }
            }
            else
            {
                uint16x4_t _bf00 = float2bfloat(_f00);
                uint16x4_t _bf01 = float2bfloat(_f01);
                uint16x4_t _bf10 = float2bfloat(_f10);
                uint16x4_t _bf11 = float2bfloat(_f11);
                uint16x4_t _bf20 = float2bfloat(_f20);
                uint16x4_t _bf21 = float2bfloat(_f21);
                uint16x4_t _bf30 = float2bfloat(_f30);
                uint16x4_t _bf31 = float2bfloat(_f31);
                uint16x4_t _bf40 = float2bfloat(_f40);
                uint16x4_t _bf41 = float2bfloat(_f41);
                uint16x4_t _bf50 = float2bfloat(_f50);
                uint16x4_t _bf51 = float2bfloat(_f51);
                uint16x4_t _bf60 = float2bfloat(_f60);
                uint16x4_t _bf61 = float2bfloat(_f61);
                uint16x4_t _bf70 = float2bfloat(_f70);
                uint16x4_t _bf71 = float2bfloat(_f71);
                uint16x4_t _bf80 = float2bfloat(_f80);
                uint16x4_t _bf81 = float2bfloat(_f81);
                uint16x4_t _bf90 = float2bfloat(_f90);
                uint16x4_t _bf91 = float2bfloat(_f91);
                uint16x4_t _bfa0 = float2bfloat(_fa0);
                uint16x4_t _bfa1 = float2bfloat(_fa1);
                uint16x4_t _bfb0 = float2bfloat(_fb0);
                uint16x4_t _bfb1 = float2bfloat(_fb1);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        uint16x8x4_t _r0;
                        _r0.val[0] = vcombine_u16(_bf00, _bf01);
                        _r0.val[1] = vcombine_u16(_bf10, _bf11);
                        _r0.val[2] = vcombine_u16(_bf20, _bf21);
                        _r0.val[3] = vcombine_u16(_bf30, _bf31);
                        vst4q_u16(p0, _r0);
                        uint16x8x4_t _r1;
                        _r1.val[0] = vcombine_u16(_bf40, _bf41);
                        _r1.val[1] = vcombine_u16(_bf50, _bf51);
                        _r1.val[2] = vcombine_u16(_bf60, _bf61);
                        _r1.val[3] = vcombine_u16(_bf70, _bf71);
                        vst4q_u16(p0 + out_hstep * 4, _r1);
                        uint16x8x4_t _r2;
                        _r2.val[0] = vcombine_u16(_bf80, _bf81);
                        _r2.val[1] = vcombine_u16(_bf90, _bf91);
                        _r2.val[2] = vcombine_u16(_bfa0, _bfa1);
                        _r2.val[3] = vcombine_u16(_bfb0, _bfb1);
                        vst4q_u16(p0 + out_hstep * 8, _r2);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf00, _bf01));
                        vst1q_u16(p0 + out_hstep, vcombine_u16(_bf10, _bf11));
                        vst1q_u16(p0 + out_hstep * 2, vcombine_u16(_bf20, _bf21));
                        vst1q_u16(p0 + out_hstep * 3, vcombine_u16(_bf30, _bf31));
                        vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf40, _bf41));
                        vst1q_u16(p0 + out_hstep * 5, vcombine_u16(_bf50, _bf51));
                        vst1q_u16(p0 + out_hstep * 6, vcombine_u16(_bf60, _bf61));
                        vst1q_u16(p0 + out_hstep * 7, vcombine_u16(_bf70, _bf71));
                        vst1q_u16(p0 + out_hstep * 8, vcombine_u16(_bf80, _bf81));
                        vst1q_u16(p0 + out_hstep * 9, vcombine_u16(_bf90, _bf91));
                        vst1q_u16(p0 + out_hstep * 10, vcombine_u16(_bfa0, _bfa1));
                        vst1q_u16(p0 + out_hstep * 11, vcombine_u16(_bfb0, _bfb1));
                    }
                    p0 += out_hstep * 12;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf00, _bf10));
                        vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf01, _bf11));
                        vst1q_u16(p0 + 8, vcombine_u16(_bf20, _bf30));
                        vst1q_u16(p0 + out_hstep * 4 + 8, vcombine_u16(_bf21, _bf31));
                        vst1q_u16(p0 + 16, vcombine_u16(_bf40, _bf50));
                        vst1q_u16(p0 + out_hstep * 4 + 16, vcombine_u16(_bf41, _bf51));
                        vst1q_u16(p0 + 24, vcombine_u16(_bf60, _bf70));
                        vst1q_u16(p0 + out_hstep * 4 + 24, vcombine_u16(_bf61, _bf71));
                        vst1q_u16(p0 + 32, vcombine_u16(_bf80, _bf90));
                        vst1q_u16(p0 + out_hstep * 4 + 32, vcombine_u16(_bf81, _bf91));
                        vst1q_u16(p0 + 40, vcombine_u16(_bfa0, _bfb0));
                        vst1q_u16(p0 + out_hstep * 4 + 40, vcombine_u16(_bfa1, _bfb1));
                        p0 += 48;
                    }
                    if (out_elempack == 1)
                    {
                        uint16x8_t _r0 = vcombine_u16(_bf00, _bf01);
                        uint16x8_t _r1 = vcombine_u16(_bf10, _bf11);
                        uint16x8_t _r2 = vcombine_u16(_bf20, _bf21);
                        uint16x8_t _r3 = vcombine_u16(_bf30, _bf31);
                        uint16x8_t _r4 = vcombine_u16(_bf40, _bf41);
                        uint16x8_t _r5 = vcombine_u16(_bf50, _bf51);
                        uint16x8_t _r6 = vcombine_u16(_bf60, _bf61);
                        uint16x8_t _r7 = vcombine_u16(_bf70, _bf71);
                        uint16x8_t _r8 = vcombine_u16(_bf80, _bf81);
                        uint16x8_t _r9 = vcombine_u16(_bf90, _bf91);
                        uint16x8_t _ra = vcombine_u16(_bfa0, _bfa1);
                        uint16x8_t _rb = vcombine_u16(_bfb0, _bfb1);
                        transpose8x12_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);

                        vst1q_u16(p0, _r0);
                        vst1_u16(p0 + 8, vget_low_u16(_r1));
                        vst1q_u16(p0 + out_hstep, vcombine_u16(vget_high_u16(_r1), vget_low_u16(_r2)));
                        vst1_u16(p0 + out_hstep + 8, vget_high_u16(_r2));
                        vst1q_u16(p0 + out_hstep * 2, _r3);
                        vst1_u16(p0 + out_hstep * 2 + 8, vget_low_u16(_r4));
                        vst1q_u16(p0 + out_hstep * 3, vcombine_u16(vget_high_u16(_r4), vget_low_u16(_r5)));
                        vst1_u16(p0 + out_hstep * 3 + 8, vget_high_u16(_r5));
                        vst1q_u16(p0 + out_hstep * 4, _r6);
                        vst1_u16(p0 + out_hstep * 4 + 8, vget_low_u16(_r7));
                        vst1q_u16(p0 + out_hstep * 5, vcombine_u16(vget_high_u16(_r7), vget_low_u16(_r8)));
                        vst1_u16(p0 + out_hstep * 5 + 8, vget_high_u16(_r8));
                        vst1q_u16(p0 + out_hstep * 6, _r9);
                        vst1_u16(p0 + out_hstep * 6 + 8, vget_low_u16(_ra));
                        vst1q_u16(p0 + out_hstep * 7, vcombine_u16(vget_high_u16(_ra), vget_low_u16(_rb)));
                        vst1_u16(p0 + out_hstep * 7 + 8, vget_high_u16(_rb));
                        p0 += 12;
                    }
                }
            }
            pp += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _f00 = vld1q_f32(pp + 0);
            float32x4_t _f01 = vld1q_f32(pp + 4);
            float32x4_t _f10 = vld1q_f32(pp + 8);
            float32x4_t _f11 = vld1q_f32(pp + 12);
            float32x4_t _f20 = vld1q_f32(pp + 16);
            float32x4_t _f21 = vld1q_f32(pp + 20);
            float32x4_t _f30 = vld1q_f32(pp + 24);
            float32x4_t _f31 = vld1q_f32(pp + 28);
            float32x4_t _f40 = vld1q_f32(pp + 32);
            float32x4_t _f41 = vld1q_f32(pp + 36);
            float32x4_t _f50 = vld1q_f32(pp + 40);
            float32x4_t _f51 = vld1q_f32(pp + 44);
            float32x4_t _f60 = vld1q_f32(pp + 48);
            float32x4_t _f61 = vld1q_f32(pp + 52);
            float32x4_t _f70 = vld1q_f32(pp + 56);
            float32x4_t _f71 = vld1q_f32(pp + 60);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4x2_t _r0 = vuzpq_f32(_f00, _f01);
            float32x4x2_t _r1 = vuzpq_f32(_f10, _f11);
            _f00 = _r0.val[0];
            _f10 = _r0.val[1];
            _f01 = _r1.val[0];
            _f11 = _r1.val[1];
            _r0 = vuzpq_f32(_f20, _f21);
            _r1 = vuzpq_f32(_f30, _f31);
            _f20 = _r0.val[0];
            _f30 = _r0.val[1];
            _f21 = _r1.val[0];
            _f31 = _r1.val[1];
            _r0 = vuzpq_f32(_f40, _f41);
            _r1 = vuzpq_f32(_f50, _f51);
            _f40 = _r0.val[0];
            _f50 = _r0.val[1];
            _f41 = _r1.val[0];
            _f51 = _r1.val[1];
            _r0 = vuzpq_f32(_f60, _f61);
            _r1 = vuzpq_f32(_f70, _f71);
            _f60 = _r0.val[0];
            _f70 = _r0.val[1];
            _f61 = _r1.val[0];
            _f71 = _r1.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c);
                    _f01 = vaddq_f32(_f01, _c);
                    _f10 = vaddq_f32(_f10, _c);
                    _f11 = vaddq_f32(_f11, _c);
                    _f20 = vaddq_f32(_f20, _c);
                    _f21 = vaddq_f32(_f21, _c);
                    _f30 = vaddq_f32(_f30, _c);
                    _f31 = vaddq_f32(_f31, _c);
                    _f40 = vaddq_f32(_f40, _c);
                    _f41 = vaddq_f32(_f41, _c);
                    _f50 = vaddq_f32(_f50, _c);
                    _f51 = vaddq_f32(_f51, _c);
                    _f60 = vaddq_f32(_f60, _c);
                    _f61 = vaddq_f32(_f61, _c);
                    _f70 = vaddq_f32(_f70, _c);
                    _f71 = vaddq_f32(_f71, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c1);
                    _f10 = vaddq_f32(_f10, _c0);
                    _f11 = vaddq_f32(_f11, _c1);
                    _f20 = vaddq_f32(_f20, _c0);
                    _f21 = vaddq_f32(_f21, _c1);
                    _f30 = vaddq_f32(_f30, _c0);
                    _f31 = vaddq_f32(_f31, _c1);
                    _f40 = vaddq_f32(_f40, _c0);
                    _f41 = vaddq_f32(_f41, _c1);
                    _f50 = vaddq_f32(_f50, _c0);
                    _f51 = vaddq_f32(_f51, _c1);
                    _f60 = vaddq_f32(_f60, _c0);
                    _f61 = vaddq_f32(_f61, _c1);
                    _f70 = vaddq_f32(_f70, _c0);
                    _f71 = vaddq_f32(_f71, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c00 = vld1q_f32(pC);
                        float32x4_t _c01 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c10 = vld1q_f32(pC + 4);
                        float32x4_t _c11 = vld1q_f32(pC + c_hstep * 4 + 4);
                        float32x4_t _c20 = vld1q_f32(pC + 8);
                        float32x4_t _c21 = vld1q_f32(pC + c_hstep * 4 + 8);
                        float32x4_t _c30 = vld1q_f32(pC + 12);
                        float32x4_t _c31 = vld1q_f32(pC + c_hstep * 4 + 12);
                        float32x4_t _c40 = vld1q_f32(pC + 16);
                        float32x4_t _c41 = vld1q_f32(pC + c_hstep * 4 + 16);
                        float32x4_t _c50 = vld1q_f32(pC + 20);
                        float32x4_t _c51 = vld1q_f32(pC + c_hstep * 4 + 20);
                        float32x4_t _c60 = vld1q_f32(pC + 24);
                        float32x4_t _c61 = vld1q_f32(pC + c_hstep * 4 + 24);
                        float32x4_t _c70 = vld1q_f32(pC + 28);
                        float32x4_t _c71 = vld1q_f32(pC + c_hstep * 4 + 28);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c00);
                            _f01 = vaddq_f32(_f01, _c01);
                            _f10 = vaddq_f32(_f10, _c10);
                            _f11 = vaddq_f32(_f11, _c11);
                            _f20 = vaddq_f32(_f20, _c20);
                            _f21 = vaddq_f32(_f21, _c21);
                            _f30 = vaddq_f32(_f30, _c30);
                            _f31 = vaddq_f32(_f31, _c31);
                            _f40 = vaddq_f32(_f40, _c40);
                            _f41 = vaddq_f32(_f41, _c41);
                            _f50 = vaddq_f32(_f50, _c50);
                            _f51 = vaddq_f32(_f51, _c51);
                            _f60 = vaddq_f32(_f60, _c60);
                            _f61 = vaddq_f32(_f61, _c61);
                            _f70 = vaddq_f32(_f70, _c70);
                            _f71 = vaddq_f32(_f71, _c71);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c00, _beta);
                            _f01 = vmlaq_f32(_f01, _c01, _beta);
                            _f10 = vmlaq_f32(_f10, _c10, _beta);
                            _f11 = vmlaq_f32(_f11, _c11, _beta);
                            _f20 = vmlaq_f32(_f20, _c20, _beta);
                            _f21 = vmlaq_f32(_f21, _c21, _beta);
                            _f30 = vmlaq_f32(_f30, _c30, _beta);
                            _f31 = vmlaq_f32(_f31, _c31, _beta);
                            _f40 = vmlaq_f32(_f40, _c40, _beta);
                            _f41 = vmlaq_f32(_f41, _c41, _beta);
                            _f50 = vmlaq_f32(_f50, _c50, _beta);
                            _f51 = vmlaq_f32(_f51, _c51, _beta);
                            _f60 = vmlaq_f32(_f60, _c60, _beta);
                            _f61 = vmlaq_f32(_f61, _c61, _beta);
                            _f70 = vmlaq_f32(_f70, _c70, _beta);
                            _f71 = vmlaq_f32(_f71, _c71, _beta);
                        }
                        pC += 8 * c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c000 = vld1q_f32(pC);
                        float32x4_t _c001 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c002 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c003 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c000, _c001, _c002, _c003);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c000);
                            _f10 = vaddq_f32(_f10, _c001);
                            _f20 = vaddq_f32(_f20, _c002);
                            _f30 = vaddq_f32(_f30, _c003);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c000, _beta);
                            _f10 = vmlaq_f32(_f10, _c001, _beta);
                            _f20 = vmlaq_f32(_f20, _c002, _beta);
                            _f30 = vmlaq_f32(_f30, _c003, _beta);
                        }
                        float32x4_t _c010 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c011 = vld1q_f32(pC + c_hstep * 5);
                        float32x4_t _c012 = vld1q_f32(pC + c_hstep * 6);
                        float32x4_t _c013 = vld1q_f32(pC + c_hstep * 7);
                        transpose4x4_ps(_c010, _c011, _c012, _c013);
                        if (beta == 1.f)
                        {
                            _f01 = vaddq_f32(_f01, _c010);
                            _f11 = vaddq_f32(_f11, _c011);
                            _f21 = vaddq_f32(_f21, _c012);
                            _f31 = vaddq_f32(_f31, _c013);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f01 = vmlaq_f32(_f01, _c010, _beta);
                            _f11 = vmlaq_f32(_f11, _c011, _beta);
                            _f21 = vmlaq_f32(_f21, _c012, _beta);
                            _f31 = vmlaq_f32(_f31, _c013, _beta);
                        }
                        float32x4_t _c400 = vld1q_f32(pC + 4);
                        float32x4_t _c401 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c402 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c403 = vld1q_f32(pC + c_hstep * 3 + 4);
                        transpose4x4_ps(_c400, _c401, _c402, _c403);
                        if (beta == 1.f)
                        {
                            _f40 = vaddq_f32(_f40, _c400);
                            _f50 = vaddq_f32(_f50, _c401);
                            _f60 = vaddq_f32(_f60, _c402);
                            _f70 = vaddq_f32(_f70, _c403);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f40 = vmlaq_f32(_f40, _c400, _beta);
                            _f50 = vmlaq_f32(_f50, _c401, _beta);
                            _f60 = vmlaq_f32(_f60, _c402, _beta);
                            _f70 = vmlaq_f32(_f70, _c403, _beta);
                        }
                        float32x4_t _c410 = vld1q_f32(pC + c_hstep * 4 + 4);
                        float32x4_t _c411 = vld1q_f32(pC + c_hstep * 5 + 4);
                        float32x4_t _c412 = vld1q_f32(pC + c_hstep * 6 + 4);
                        float32x4_t _c413 = vld1q_f32(pC + c_hstep * 7 + 4);
                        transpose4x4_ps(_c410, _c411, _c412, _c413);
                        if (beta == 1.f)
                        {
                            _f41 = vaddq_f32(_f41, _c410);
                            _f51 = vaddq_f32(_f51, _c411);
                            _f61 = vaddq_f32(_f61, _c412);
                            _f71 = vaddq_f32(_f71, _c413);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f41 = vmlaq_f32(_f41, _c410, _beta);
                            _f51 = vmlaq_f32(_f51, _c411, _beta);
                            _f61 = vmlaq_f32(_f61, _c412, _beta);
                            _f71 = vmlaq_f32(_f71, _c413, _beta);
                        }
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c0);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f10 = vaddq_f32(_f10, _c1);
                    _f11 = vaddq_f32(_f11, _c1);
                    float32x4_t _c2 = vdupq_n_f32(pC[2] * beta);
                    _f20 = vaddq_f32(_f20, _c2);
                    _f21 = vaddq_f32(_f21, _c2);
                    float32x4_t _c3 = vdupq_n_f32(pC[3] * beta);
                    _f30 = vaddq_f32(_f30, _c3);
                    _f31 = vaddq_f32(_f31, _c3);
                    float32x4_t _c4 = vdupq_n_f32(pC[4] * beta);
                    _f40 = vaddq_f32(_f40, _c4);
                    _f41 = vaddq_f32(_f41, _c4);
                    float32x4_t _c5 = vdupq_n_f32(pC[5] * beta);
                    _f50 = vaddq_f32(_f50, _c5);
                    _f51 = vaddq_f32(_f51, _c5);
                    float32x4_t _c6 = vdupq_n_f32(pC[6] * beta);
                    _f60 = vaddq_f32(_f60, _c6);
                    _f61 = vaddq_f32(_f61, _c6);
                    float32x4_t _c7 = vdupq_n_f32(pC[7] * beta);
                    _f70 = vaddq_f32(_f70, _c7);
                    _f71 = vaddq_f32(_f71, _c7);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f00 = vmulq_f32(_f00, _alpha);
                _f01 = vmulq_f32(_f01, _alpha);
                _f10 = vmulq_f32(_f10, _alpha);
                _f11 = vmulq_f32(_f11, _alpha);
                _f20 = vmulq_f32(_f20, _alpha);
                _f21 = vmulq_f32(_f21, _alpha);
                _f30 = vmulq_f32(_f30, _alpha);
                _f31 = vmulq_f32(_f31, _alpha);
                _f40 = vmulq_f32(_f40, _alpha);
                _f41 = vmulq_f32(_f41, _alpha);
                _f50 = vmulq_f32(_f50, _alpha);
                _f51 = vmulq_f32(_f51, _alpha);
                _f60 = vmulq_f32(_f60, _alpha);
                _f61 = vmulq_f32(_f61, _alpha);
                _f70 = vmulq_f32(_f70, _alpha);
                _f71 = vmulq_f32(_f71, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float32x4x4_t _r0;
                        _r0.val[0] = _f00;
                        _r0.val[1] = _f10;
                        _r0.val[2] = _f20;
                        _r0.val[3] = _f30;
                        vst4q_f32(p0f, _r0);
                        float32x4x4_t _r1;
                        _r1.val[0] = _f01;
                        _r1.val[1] = _f11;
                        _r1.val[2] = _f21;
                        _r1.val[3] = _f31;
                        vst4q_f32(p0f + 16, _r1);
                        float32x4x4_t _r2;
                        _r2.val[0] = _f40;
                        _r2.val[1] = _f50;
                        _r2.val[2] = _f60;
                        _r2.val[3] = _f70;
                        vst4q_f32(p0f + out_hstep * 4, _r2);
                        float32x4x4_t _r3;
                        _r3.val[0] = _f41;
                        _r3.val[1] = _f51;
                        _r3.val[2] = _f61;
                        _r3.val[3] = _f71;
                        vst4q_f32(p0f + out_hstep * 4 + 16, _r3);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + 4, _f01);
                        vst1q_f32(p0f + out_hstep, _f10);
                        vst1q_f32(p0f + out_hstep + 4, _f11);
                        vst1q_f32(p0f + out_hstep * 2, _f20);
                        vst1q_f32(p0f + out_hstep * 2 + 4, _f21);
                        vst1q_f32(p0f + out_hstep * 3, _f30);
                        vst1q_f32(p0f + out_hstep * 3 + 4, _f31);
                        vst1q_f32(p0f + out_hstep * 4, _f40);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f41);
                        vst1q_f32(p0f + out_hstep * 5, _f50);
                        vst1q_f32(p0f + out_hstep * 5 + 4, _f51);
                        vst1q_f32(p0f + out_hstep * 6, _f60);
                        vst1q_f32(p0f + out_hstep * 6 + 4, _f61);
                        vst1q_f32(p0f + out_hstep * 7, _f70);
                        vst1q_f32(p0f + out_hstep * 7 + 4, _f71);
                    }
                    p0f += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + out_hstep * 4, _f01);
                        vst1q_f32(p0f + 4, _f10);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f11);
                        vst1q_f32(p0f + 8, _f20);
                        vst1q_f32(p0f + out_hstep * 4 + 8, _f21);
                        vst1q_f32(p0f + 12, _f30);
                        vst1q_f32(p0f + out_hstep * 4 + 12, _f31);
                        vst1q_f32(p0f + 16, _f40);
                        vst1q_f32(p0f + out_hstep * 4 + 16, _f41);
                        vst1q_f32(p0f + 20, _f50);
                        vst1q_f32(p0f + out_hstep * 4 + 20, _f51);
                        vst1q_f32(p0f + 24, _f60);
                        vst1q_f32(p0f + out_hstep * 4 + 24, _f61);
                        vst1q_f32(p0f + 28, _f70);
                        vst1q_f32(p0f + out_hstep * 4 + 28, _f71);
                        p0f += 32;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_f00, _f10, _f20, _f30);
                        transpose4x4_ps(_f40, _f50, _f60, _f70);
                        transpose4x4_ps(_f01, _f11, _f21, _f31);
                        transpose4x4_ps(_f41, _f51, _f61, _f71);
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + 4, _f40);
                        vst1q_f32(p0f + out_hstep, _f10);
                        vst1q_f32(p0f + out_hstep + 4, _f50);
                        vst1q_f32(p0f + out_hstep * 2, _f20);
                        vst1q_f32(p0f + out_hstep * 2 + 4, _f60);
                        vst1q_f32(p0f + out_hstep * 3, _f30);
                        vst1q_f32(p0f + out_hstep * 3 + 4, _f70);
                        vst1q_f32(p0f + out_hstep * 4, _f01);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f41);
                        vst1q_f32(p0f + out_hstep * 5, _f11);
                        vst1q_f32(p0f + out_hstep * 5 + 4, _f51);
                        vst1q_f32(p0f + out_hstep * 6, _f21);
                        vst1q_f32(p0f + out_hstep * 6 + 4, _f61);
                        vst1q_f32(p0f + out_hstep * 7, _f31);
                        vst1q_f32(p0f + out_hstep * 7 + 4, _f71);
                        p0f += 8;
                    }
                }
            }
            else
            {
                uint16x4_t _bf00 = float2bfloat(_f00);
                uint16x4_t _bf01 = float2bfloat(_f01);
                uint16x4_t _bf10 = float2bfloat(_f10);
                uint16x4_t _bf11 = float2bfloat(_f11);
                uint16x4_t _bf20 = float2bfloat(_f20);
                uint16x4_t _bf21 = float2bfloat(_f21);
                uint16x4_t _bf30 = float2bfloat(_f30);
                uint16x4_t _bf31 = float2bfloat(_f31);
                uint16x4_t _bf40 = float2bfloat(_f40);
                uint16x4_t _bf41 = float2bfloat(_f41);
                uint16x4_t _bf50 = float2bfloat(_f50);
                uint16x4_t _bf51 = float2bfloat(_f51);
                uint16x4_t _bf60 = float2bfloat(_f60);
                uint16x4_t _bf61 = float2bfloat(_f61);
                uint16x4_t _bf70 = float2bfloat(_f70);
                uint16x4_t _bf71 = float2bfloat(_f71);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        uint16x8x4_t _r0;
                        _r0.val[0] = vcombine_u16(_bf00, _bf01);
                        _r0.val[1] = vcombine_u16(_bf10, _bf11);
                        _r0.val[2] = vcombine_u16(_bf20, _bf21);
                        _r0.val[3] = vcombine_u16(_bf30, _bf31);
                        vst4q_u16(p0, _r0);
                        uint16x8x4_t _r1;
                        _r1.val[0] = vcombine_u16(_bf40, _bf41);
                        _r1.val[1] = vcombine_u16(_bf50, _bf51);
                        _r1.val[2] = vcombine_u16(_bf60, _bf61);
                        _r1.val[3] = vcombine_u16(_bf70, _bf71);
                        vst4q_u16(p0 + out_hstep * 4, _r1);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf00, _bf01));
                        vst1q_u16(p0 + out_hstep, vcombine_u16(_bf10, _bf11));
                        vst1q_u16(p0 + out_hstep * 2, vcombine_u16(_bf20, _bf21));
                        vst1q_u16(p0 + out_hstep * 3, vcombine_u16(_bf30, _bf31));
                        vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf40, _bf41));
                        vst1q_u16(p0 + out_hstep * 5, vcombine_u16(_bf50, _bf51));
                        vst1q_u16(p0 + out_hstep * 6, vcombine_u16(_bf60, _bf61));
                        vst1q_u16(p0 + out_hstep * 7, vcombine_u16(_bf70, _bf71));
                    }
                    p0 += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf00, _bf10));
                        vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf01, _bf11));
                        vst1q_u16(p0 + 8, vcombine_u16(_bf20, _bf30));
                        vst1q_u16(p0 + out_hstep * 4 + 8, vcombine_u16(_bf21, _bf31));
                        vst1q_u16(p0 + 16, vcombine_u16(_bf40, _bf50));
                        vst1q_u16(p0 + out_hstep * 4 + 16, vcombine_u16(_bf41, _bf51));
                        vst1q_u16(p0 + 24, vcombine_u16(_bf60, _bf70));
                        vst1q_u16(p0 + out_hstep * 4 + 24, vcombine_u16(_bf61, _bf71));
                        p0 += 32;
                    }
                    if (out_elempack == 1)
                    {
                        uint16x8_t _r0 = vcombine_u16(_bf00, _bf01);
                        uint16x8_t _r1 = vcombine_u16(_bf10, _bf11);
                        uint16x8_t _r2 = vcombine_u16(_bf20, _bf21);
                        uint16x8_t _r3 = vcombine_u16(_bf30, _bf31);
                        uint16x8_t _r4 = vcombine_u16(_bf40, _bf41);
                        uint16x8_t _r5 = vcombine_u16(_bf50, _bf51);
                        uint16x8_t _r6 = vcombine_u16(_bf60, _bf61);
                        uint16x8_t _r7 = vcombine_u16(_bf70, _bf71);
                        transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                        vst1q_u16(p0, _r0);
                        vst1q_u16(p0 + out_hstep, _r1);
                        vst1q_u16(p0 + out_hstep * 2, _r2);
                        vst1q_u16(p0 + out_hstep * 3, _r3);
                        vst1q_u16(p0 + out_hstep * 4, _r4);
                        vst1q_u16(p0 + out_hstep * 5, _r5);
                        vst1q_u16(p0 + out_hstep * 6, _r6);
                        vst1q_u16(p0 + out_hstep * 7, _r7);
                        p0 += 8;
                    }
                }
            }
            pp += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _f00 = vld1q_f32(pp + 0);
            float32x4_t _f01 = vld1q_f32(pp + 4);
            float32x4_t _f10 = vld1q_f32(pp + 8);
            float32x4_t _f11 = vld1q_f32(pp + 12);
            float32x4_t _f20 = vld1q_f32(pp + 16);
            float32x4_t _f21 = vld1q_f32(pp + 20);
            float32x4_t _f30 = vld1q_f32(pp + 24);
            float32x4_t _f31 = vld1q_f32(pp + 28);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4x2_t _r0 = vuzpq_f32(_f00, _f01);
            float32x4x2_t _r1 = vuzpq_f32(_f10, _f11);
            _f00 = _r0.val[0];
            _f10 = _r0.val[1];
            _f01 = _r1.val[0];
            _f11 = _r1.val[1];
            _r0 = vuzpq_f32(_f20, _f21);
            _r1 = vuzpq_f32(_f30, _f31);
            _f20 = _r0.val[0];
            _f30 = _r0.val[1];
            _f21 = _r1.val[0];
            _f31 = _r1.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c);
                    _f01 = vaddq_f32(_f01, _c);
                    _f10 = vaddq_f32(_f10, _c);
                    _f11 = vaddq_f32(_f11, _c);
                    _f20 = vaddq_f32(_f20, _c);
                    _f21 = vaddq_f32(_f21, _c);
                    _f30 = vaddq_f32(_f30, _c);
                    _f31 = vaddq_f32(_f31, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c1);
                    _f10 = vaddq_f32(_f10, _c0);
                    _f11 = vaddq_f32(_f11, _c1);
                    _f20 = vaddq_f32(_f20, _c0);
                    _f21 = vaddq_f32(_f21, _c1);
                    _f30 = vaddq_f32(_f30, _c0);
                    _f31 = vaddq_f32(_f31, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c00 = vld1q_f32(pC);
                        float32x4_t _c01 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c10 = vld1q_f32(pC + 4);
                        float32x4_t _c11 = vld1q_f32(pC + c_hstep * 4 + 4);
                        float32x4_t _c20 = vld1q_f32(pC + 8);
                        float32x4_t _c21 = vld1q_f32(pC + c_hstep * 4 + 8);
                        float32x4_t _c30 = vld1q_f32(pC + 12);
                        float32x4_t _c31 = vld1q_f32(pC + c_hstep * 4 + 12);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c00);
                            _f01 = vaddq_f32(_f01, _c01);
                            _f10 = vaddq_f32(_f10, _c10);
                            _f11 = vaddq_f32(_f11, _c11);
                            _f20 = vaddq_f32(_f20, _c20);
                            _f21 = vaddq_f32(_f21, _c21);
                            _f30 = vaddq_f32(_f30, _c30);
                            _f31 = vaddq_f32(_f31, _c31);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c00, _beta);
                            _f01 = vmlaq_f32(_f01, _c01, _beta);
                            _f10 = vmlaq_f32(_f10, _c10, _beta);
                            _f11 = vmlaq_f32(_f11, _c11, _beta);
                            _f20 = vmlaq_f32(_f20, _c20, _beta);
                            _f21 = vmlaq_f32(_f21, _c21, _beta);
                            _f30 = vmlaq_f32(_f30, _c30, _beta);
                            _f31 = vmlaq_f32(_f31, _c31, _beta);
                        }
                        pC += 4 * c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c000 = vld1q_f32(pC);
                        float32x4_t _c001 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c002 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c003 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c000, _c001, _c002, _c003);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c000);
                            _f10 = vaddq_f32(_f10, _c001);
                            _f20 = vaddq_f32(_f20, _c002);
                            _f30 = vaddq_f32(_f30, _c003);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c000, _beta);
                            _f10 = vmlaq_f32(_f10, _c001, _beta);
                            _f20 = vmlaq_f32(_f20, _c002, _beta);
                            _f30 = vmlaq_f32(_f30, _c003, _beta);
                        }
                        float32x4_t _c010 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c011 = vld1q_f32(pC + c_hstep * 5);
                        float32x4_t _c012 = vld1q_f32(pC + c_hstep * 6);
                        float32x4_t _c013 = vld1q_f32(pC + c_hstep * 7);
                        transpose4x4_ps(_c010, _c011, _c012, _c013);
                        if (beta == 1.f)
                        {
                            _f01 = vaddq_f32(_f01, _c010);
                            _f11 = vaddq_f32(_f11, _c011);
                            _f21 = vaddq_f32(_f21, _c012);
                            _f31 = vaddq_f32(_f31, _c013);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f01 = vmlaq_f32(_f01, _c010, _beta);
                            _f11 = vmlaq_f32(_f11, _c011, _beta);
                            _f21 = vmlaq_f32(_f21, _c012, _beta);
                            _f31 = vmlaq_f32(_f31, _c013, _beta);
                        }
                        pC += 4;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c0);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f10 = vaddq_f32(_f10, _c1);
                    _f11 = vaddq_f32(_f11, _c1);
                    float32x4_t _c2 = vdupq_n_f32(pC[2] * beta);
                    _f20 = vaddq_f32(_f20, _c2);
                    _f21 = vaddq_f32(_f21, _c2);
                    float32x4_t _c3 = vdupq_n_f32(pC[3] * beta);
                    _f30 = vaddq_f32(_f30, _c3);
                    _f31 = vaddq_f32(_f31, _c3);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f00 = vmulq_f32(_f00, _alpha);
                _f01 = vmulq_f32(_f01, _alpha);
                _f10 = vmulq_f32(_f10, _alpha);
                _f11 = vmulq_f32(_f11, _alpha);
                _f20 = vmulq_f32(_f20, _alpha);
                _f21 = vmulq_f32(_f21, _alpha);
                _f30 = vmulq_f32(_f30, _alpha);
                _f31 = vmulq_f32(_f31, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float32x4x4_t _r0;
                        _r0.val[0] = _f00;
                        _r0.val[1] = _f10;
                        _r0.val[2] = _f20;
                        _r0.val[3] = _f30;
                        vst4q_f32(p0f, _r0);
                        float32x4x4_t _r1;
                        _r1.val[0] = _f01;
                        _r1.val[1] = _f11;
                        _r1.val[2] = _f21;
                        _r1.val[3] = _f31;
                        vst4q_f32(p0f + 16, _r1);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + 4, _f01);
                        vst1q_f32(p0f + out_hstep, _f10);
                        vst1q_f32(p0f + out_hstep + 4, _f11);
                        vst1q_f32(p0f + out_hstep * 2, _f20);
                        vst1q_f32(p0f + out_hstep * 2 + 4, _f21);
                        vst1q_f32(p0f + out_hstep * 3, _f30);
                        vst1q_f32(p0f + out_hstep * 3 + 4, _f31);
                    }
                    p0f += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + out_hstep * 4, _f01);
                        vst1q_f32(p0f + 4, _f10);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f11);
                        vst1q_f32(p0f + 8, _f20);
                        vst1q_f32(p0f + out_hstep * 4 + 8, _f21);
                        vst1q_f32(p0f + 12, _f30);
                        vst1q_f32(p0f + out_hstep * 4 + 12, _f31);
                        p0f += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_f00, _f10, _f20, _f30);
                        transpose4x4_ps(_f01, _f11, _f21, _f31);
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + out_hstep, _f10);
                        vst1q_f32(p0f + out_hstep * 2, _f20);
                        vst1q_f32(p0f + out_hstep * 3, _f30);
                        vst1q_f32(p0f + out_hstep * 4, _f01);
                        vst1q_f32(p0f + out_hstep * 5, _f11);
                        vst1q_f32(p0f + out_hstep * 6, _f21);
                        vst1q_f32(p0f + out_hstep * 7, _f31);
                        p0f += 4;
                    }
                }
            }
            else
            {
                uint16x4_t _bf00 = float2bfloat(_f00);
                uint16x4_t _bf01 = float2bfloat(_f01);
                uint16x4_t _bf10 = float2bfloat(_f10);
                uint16x4_t _bf11 = float2bfloat(_f11);
                uint16x4_t _bf20 = float2bfloat(_f20);
                uint16x4_t _bf21 = float2bfloat(_f21);
                uint16x4_t _bf30 = float2bfloat(_f30);
                uint16x4_t _bf31 = float2bfloat(_f31);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        uint16x8x4_t _r0;
                        _r0.val[0] = vcombine_u16(_bf00, _bf01);
                        _r0.val[1] = vcombine_u16(_bf10, _bf11);
                        _r0.val[2] = vcombine_u16(_bf20, _bf21);
                        _r0.val[3] = vcombine_u16(_bf30, _bf31);
                        vst4q_u16(p0, _r0);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf00, _bf01));
                        vst1q_u16(p0 + out_hstep, vcombine_u16(_bf10, _bf11));
                        vst1q_u16(p0 + out_hstep * 2, vcombine_u16(_bf20, _bf21));
                        vst1q_u16(p0 + out_hstep * 3, vcombine_u16(_bf30, _bf31));
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf00, _bf10));
                        vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf01, _bf11));
                        vst1q_u16(p0 + 8, vcombine_u16(_bf20, _bf30));
                        vst1q_u16(p0 + out_hstep * 4 + 8, vcombine_u16(_bf21, _bf31));
                        p0 += 16;
                    }
                    if (out_elempack == 1)
                    {
                        uint16x8_t _r0 = vcombine_u16(_bf00, _bf01);
                        uint16x8_t _r1 = vcombine_u16(_bf10, _bf11);
                        uint16x8_t _r2 = vcombine_u16(_bf20, _bf21);
                        uint16x8_t _r3 = vcombine_u16(_bf30, _bf31);
                        transpose8x4_u16(_r0, _r1, _r2, _r3);

                        vst1_u16(p0, vget_low_u16(_r0));
                        vst1_u16(p0 + out_hstep, vget_high_u16(_r0));
                        vst1_u16(p0 + out_hstep * 2, vget_low_u16(_r1));
                        vst1_u16(p0 + out_hstep * 3, vget_high_u16(_r1));
                        vst1_u16(p0 + out_hstep * 4, vget_low_u16(_r2));
                        vst1_u16(p0 + out_hstep * 5, vget_high_u16(_r2));
                        vst1_u16(p0 + out_hstep * 6, vget_low_u16(_r3));
                        vst1_u16(p0 + out_hstep * 7, vget_high_u16(_r3));
                        p0 += 4;
                    }
                }
            }
            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _f00 = vld1q_f32(pp + 0);
            float32x4_t _f01 = vld1q_f32(pp + 4);
            float32x4_t _f10 = vld1q_f32(pp + 8);
            float32x4_t _f11 = vld1q_f32(pp + 12);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4x2_t _r0 = vuzpq_f32(_f00, _f01);
            float32x4x2_t _r1 = vuzpq_f32(_f10, _f11);
            _f00 = _r0.val[0];
            _f10 = _r0.val[1];
            _f01 = _r1.val[0];
            _f11 = _r1.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c);
                    _f01 = vaddq_f32(_f01, _c);
                    _f10 = vaddq_f32(_f10, _c);
                    _f11 = vaddq_f32(_f11, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c1);
                    _f10 = vaddq_f32(_f10, _c0);
                    _f11 = vaddq_f32(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c00 = vld1q_f32(pC);
                        float32x4_t _c01 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c10 = vld1q_f32(pC + 4);
                        float32x4_t _c11 = vld1q_f32(pC + c_hstep * 4 + 4);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c00);
                            _f01 = vaddq_f32(_f01, _c01);
                            _f10 = vaddq_f32(_f10, _c10);
                            _f11 = vaddq_f32(_f11, _c11);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c00, _beta);
                            _f01 = vmlaq_f32(_f01, _c01, _beta);
                            _f10 = vmlaq_f32(_f10, _c10, _beta);
                            _f11 = vmlaq_f32(_f11, _c11, _beta);
                        }
                        pC += 2 * c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x2_t _c0 = vld1_f32(pC);
                        float32x2_t _c1 = vld1_f32(pC + c_hstep);
                        float32x2_t _c2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _c3 = vld1_f32(pC + c_hstep * 3);
                        float32x2x2_t _c01x = vtrn_f32(_c0, _c1);
                        float32x2x2_t _c23x = vtrn_f32(_c2, _c3);
                        float32x4_t _c00 = vcombine_f32(_c01x.val[0], _c23x.val[0]);
                        float32x4_t _c10 = vcombine_f32(_c01x.val[1], _c23x.val[1]);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c00);
                            _f10 = vaddq_f32(_f10, _c10);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c00, _beta);
                            _f10 = vmlaq_f32(_f10, _c10, _beta);
                        }
                        float32x2_t _c4 = vld1_f32(pC + c_hstep * 4);
                        float32x2_t _c5 = vld1_f32(pC + c_hstep * 5);
                        float32x2_t _c6 = vld1_f32(pC + c_hstep * 6);
                        float32x2_t _c7 = vld1_f32(pC + c_hstep * 7);
                        float32x2x2_t _c45 = vtrn_f32(_c4, _c5);
                        float32x2x2_t _c67 = vtrn_f32(_c6, _c7);
                        float32x4_t _c01 = vcombine_f32(_c45.val[0], _c67.val[0]);
                        float32x4_t _c11 = vcombine_f32(_c45.val[1], _c67.val[1]);
                        if (beta == 1.f)
                        {
                            _f01 = vaddq_f32(_f01, _c01);
                            _f11 = vaddq_f32(_f11, _c11);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f01 = vmlaq_f32(_f01, _c01, _beta);
                            _f11 = vmlaq_f32(_f11, _c11, _beta);
                        }
                        pC += 2;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c0);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f10 = vaddq_f32(_f10, _c1);
                    _f11 = vaddq_f32(_f11, _c1);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f00 = vmulq_f32(_f00, _alpha);
                _f01 = vmulq_f32(_f01, _alpha);
                _f10 = vmulq_f32(_f10, _alpha);
                _f11 = vmulq_f32(_f11, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + 4, _f01);
                        vst1q_f32(p0f + out_hstep, _f10);
                        vst1q_f32(p0f + out_hstep + 4, _f11);
                        p0f += out_hstep * 2;
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + out_hstep * 4, _f01);
                        vst1q_f32(p0f + 4, _f10);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f11);
                        p0f += 8;
                    }
                    if (out_elempack == 1)
                    {
                        float32x4x2_t _r0 = vzipq_f32(_f00, _f10);
                        float32x4x2_t _r1 = vzipq_f32(_f01, _f11);
                        vst1_f32(p0f, vget_low_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep, vget_high_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep * 2, vget_low_f32(_r0.val[1]));
                        vst1_f32(p0f + out_hstep * 3, vget_high_f32(_r0.val[1]));
                        vst1_f32(p0f + out_hstep * 4, vget_low_f32(_r1.val[0]));
                        vst1_f32(p0f + out_hstep * 5, vget_high_f32(_r1.val[0]));
                        vst1_f32(p0f + out_hstep * 6, vget_low_f32(_r1.val[1]));
                        vst1_f32(p0f + out_hstep * 7, vget_high_f32(_r1.val[1]));
                        p0f += 2;
                    }
                }
            }
            else
            {
                uint16x4_t _bf00 = float2bfloat(_f00);
                uint16x4_t _bf01 = float2bfloat(_f01);
                uint16x4_t _bf10 = float2bfloat(_f10);
                uint16x4_t _bf11 = float2bfloat(_f11);
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1_u16(p0, _bf00);
                        vst1_u16(p0 + 4, _bf01);
                        vst1_u16(p0 + out_hstep, _bf10);
                        vst1_u16(p0 + out_hstep + 4, _bf11);
                        p0 += out_hstep * 2;
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf00, _bf10));
                        vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf01, _bf11));
                        p0 += 8;
                    }
                    if (out_elempack == 1)
                    {
                        uint16x4x2_t _r0;
                        _r0.val[0] = _bf00;
                        _r0.val[1] = _bf10;
                        vst2_lane_u16(p0, _r0, 0);
                        vst2_lane_u16(p0 + out_hstep, _r0, 1);
                        vst2_lane_u16(p0 + out_hstep * 2, _r0, 2);
                        vst2_lane_u16(p0 + out_hstep * 3, _r0, 3);
                        uint16x4x2_t _r1;
                        _r1.val[0] = _bf01;
                        _r1.val[1] = _bf11;
                        vst2_lane_u16(p0 + out_hstep * 4, _r1, 0);
                        vst2_lane_u16(p0 + out_hstep * 5, _r1, 1);
                        vst2_lane_u16(p0 + out_hstep * 6, _r1, 2);
                        vst2_lane_u16(p0 + out_hstep * 7, _r1, 3);
                        p0 += 2;
                    }
                }
            }
            pp += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _f00 = vld1q_f32(pp + 0);
            float32x4_t _f01 = vld1q_f32(pp + 4);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c);
                    _f01 = vaddq_f32(_f01, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c00 = vld1q_f32(pC);
                        float32x4_t _c01 = vld1q_f32(pC + c_hstep * 4);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c00);
                            _f01 = vaddq_f32(_f01, _c01);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c00, _beta);
                            _f01 = vmlaq_f32(_f01, _c01, _beta);
                        }
                        pC += c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c00 = vdupq_n_f32(0.f);
                        _c00 = vld1q_lane_f32(pC, _c00, 0);
                        _c00 = vld1q_lane_f32(pC + c_hstep, _c00, 1);
                        _c00 = vld1q_lane_f32(pC + c_hstep * 2, _c00, 2);
                        _c00 = vld1q_lane_f32(pC + c_hstep * 3, _c00, 3);
                        if (beta == 1.f)
                        {
                            _f00 = vaddq_f32(_f00, _c00);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f00 = vmlaq_f32(_f00, _c00, _beta);
                        }
                        float32x4_t _c01 = vdupq_n_f32(0.f);
                        _c01 = vld1q_lane_f32(pC + c_hstep * 4, _c01, 0);
                        _c01 = vld1q_lane_f32(pC + c_hstep * 5, _c01, 1);
                        _c01 = vld1q_lane_f32(pC + c_hstep * 6, _c01, 2);
                        _c01 = vld1q_lane_f32(pC + c_hstep * 7, _c01, 3);
                        if (beta == 1.f)
                        {
                            _f01 = vaddq_f32(_f01, _c01);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f01 = vmlaq_f32(_f01, _c01, _beta);
                        }
                        pC += 1;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f00 = vaddq_f32(_f00, _c0);
                    _f01 = vaddq_f32(_f01, _c0);
                    pC += 1;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f00 = vmulq_f32(_f00, _alpha);
                _f01 = vmulq_f32(_f01, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + 4, _f01);
                        p0f += out_hstep;
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f00);
                        vst1q_f32(p0f + out_hstep * 4, _f01);
                        p0f += 4;
                    }
                    if (out_elempack == 1)
                    {
                        p0f[0] = vgetq_lane_f32(_f00, 0);
                        p0f[out_hstep] = vgetq_lane_f32(_f00, 1);
                        p0f[out_hstep * 2] = vgetq_lane_f32(_f00, 2);
                        p0f[out_hstep * 3] = vgetq_lane_f32(_f00, 3);
                        p0f[out_hstep * 4] = vgetq_lane_f32(_f01, 0);
                        p0f[out_hstep * 5] = vgetq_lane_f32(_f01, 1);
                        p0f[out_hstep * 6] = vgetq_lane_f32(_f01, 2);
                        p0f[out_hstep * 7] = vgetq_lane_f32(_f01, 3);
                        p0f += 1;
                    }
                }
            }
            else
            {
                uint16x4_t _bf00 = float2bfloat(_f00);
                uint16x4_t _bf01 = float2bfloat(_f01);
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1_u16(p0, _bf00);
                        vst1_u16(p0 + 4, _bf01);
                        p0 += out_hstep;
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0, _bf00);
                        vst1_u16(p0 + out_hstep * 4, _bf01);
                        p0 += 4;
                    }
                    if (out_elempack == 1)
                    {
                        p0[0] = vget_lane_u16(_bf00, 0);
                        p0[out_hstep] = vget_lane_u16(_bf00, 1);
                        p0[out_hstep * 2] = vget_lane_u16(_bf00, 2);
                        p0[out_hstep * 3] = vget_lane_u16(_bf00, 3);
                        p0[out_hstep * 4] = vget_lane_u16(_bf01, 0);
                        p0[out_hstep * 5] = vget_lane_u16(_bf01, 1);
                        p0[out_hstep * 6] = vget_lane_u16(_bf01, 2);
                        p0[out_hstep * 7] = vget_lane_u16(_bf01, 3);
                        p0 += 1;
                    }
                }
            }
            pp += 8;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0;
        float* p0f;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC = (const float*)C + i + ii;
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _f0 = vld1q_f32(pp + 0);
            float32x4_t _f1 = vld1q_f32(pp + 4);
            float32x4_t _f2 = vld1q_f32(pp + 8);
            float32x4_t _f3 = vld1q_f32(pp + 12);
            float32x4_t _f4 = vld1q_f32(pp + 16);
            float32x4_t _f5 = vld1q_f32(pp + 20);
            float32x4_t _f6 = vld1q_f32(pp + 24);
            float32x4_t _f7 = vld1q_f32(pp + 28);
            float32x4_t _f8 = vld1q_f32(pp + 32);
            float32x4_t _f9 = vld1q_f32(pp + 36);
            float32x4_t _fa = vld1q_f32(pp + 40);
            float32x4_t _fb = vld1q_f32(pp + 44);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4x2_t _r0 = vuzpq_f32(_f0, _f1);
            _f0 = _r0.val[0];
            _f1 = _r0.val[1];
            _r0 = vuzpq_f32(_f2, _f3);
            _f2 = _r0.val[0];
            _f3 = _r0.val[1];
            _r0 = vuzpq_f32(_f4, _f5);
            _f4 = _r0.val[0];
            _f5 = _r0.val[1];
            _r0 = vuzpq_f32(_f6, _f7);
            _f6 = _r0.val[0];
            _f7 = _r0.val[1];
            _r0 = vuzpq_f32(_f8, _f9);
            _f8 = _r0.val[0];
            _f9 = _r0.val[1];
            _r0 = vuzpq_f32(_fa, _fb);
            _fa = _r0.val[0];
            _fb = _r0.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                    _f2 = vaddq_f32(_f2, _c);
                    _f3 = vaddq_f32(_f3, _c);
                    _f4 = vaddq_f32(_f4, _c);
                    _f5 = vaddq_f32(_f5, _c);
                    _f6 = vaddq_f32(_f6, _c);
                    _f7 = vaddq_f32(_f7, _c);
                    _f8 = vaddq_f32(_f8, _c);
                    _f9 = vaddq_f32(_f9, _c);
                    _fa = vaddq_f32(_fa, _c);
                    _fb = vaddq_f32(_fb, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c0 = vmulq_n_f32(_c0, beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                    _f8 = vaddq_f32(_f8, _c0);
                    _f9 = vaddq_f32(_f9, _c0);
                    _fa = vaddq_f32(_fa, _c0);
                    _fb = vaddq_f32(_fb, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        float32x4_t _c4 = vld1q_f32(pC + 16);
                        float32x4_t _c5 = vld1q_f32(pC + 20);
                        float32x4_t _c6 = vld1q_f32(pC + 24);
                        float32x4_t _c7 = vld1q_f32(pC + 28);
                        float32x4_t _c8 = vld1q_f32(pC + 32);
                        float32x4_t _c9 = vld1q_f32(pC + 36);
                        float32x4_t _ca = vld1q_f32(pC + 40);
                        float32x4_t _cb = vld1q_f32(pC + 44);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                            _f4 = vaddq_f32(_f4, _c4);
                            _f5 = vaddq_f32(_f5, _c5);
                            _f6 = vaddq_f32(_f6, _c6);
                            _f7 = vaddq_f32(_f7, _c7);
                            _f8 = vaddq_f32(_f8, _c8);
                            _f9 = vaddq_f32(_f9, _c9);
                            _fa = vaddq_f32(_fa, _ca);
                            _fb = vaddq_f32(_fb, _cb);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                            _f4 = vmlaq_f32(_f4, _c4, _beta);
                            _f5 = vmlaq_f32(_f5, _c5, _beta);
                            _f6 = vmlaq_f32(_f6, _c6, _beta);
                            _f7 = vmlaq_f32(_f7, _c7, _beta);
                            _f8 = vmlaq_f32(_f8, _c8, _beta);
                            _f9 = vmlaq_f32(_f9, _c9, _beta);
                            _fa = vmlaq_f32(_fa, _ca, _beta);
                            _fb = vmlaq_f32(_fb, _cb, _beta);
                        }
                        pC += 12 * c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c000 = vld1q_f32(pC);
                        float32x4_t _c001 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c002 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c003 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c000, _c001, _c002, _c003);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c000);
                            _f1 = vaddq_f32(_f1, _c001);
                            _f2 = vaddq_f32(_f2, _c002);
                            _f3 = vaddq_f32(_f3, _c003);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c000, _beta);
                            _f1 = vmlaq_f32(_f1, _c001, _beta);
                            _f2 = vmlaq_f32(_f2, _c002, _beta);
                            _f3 = vmlaq_f32(_f3, _c003, _beta);
                        }
                        float32x4_t _c400 = vld1q_f32(pC + 4);
                        float32x4_t _c401 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c402 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c403 = vld1q_f32(pC + c_hstep * 3 + 4);
                        transpose4x4_ps(_c400, _c401, _c402, _c403);
                        if (beta == 1.f)
                        {
                            _f4 = vaddq_f32(_f4, _c400);
                            _f5 = vaddq_f32(_f5, _c401);
                            _f6 = vaddq_f32(_f6, _c402);
                            _f7 = vaddq_f32(_f7, _c403);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f4 = vmlaq_f32(_f4, _c400, _beta);
                            _f5 = vmlaq_f32(_f5, _c401, _beta);
                            _f6 = vmlaq_f32(_f6, _c402, _beta);
                            _f7 = vmlaq_f32(_f7, _c403, _beta);
                        }
                        float32x4_t _c800 = vld1q_f32(pC + 8);
                        float32x4_t _c801 = vld1q_f32(pC + c_hstep + 8);
                        float32x4_t _c802 = vld1q_f32(pC + c_hstep * 2 + 8);
                        float32x4_t _c803 = vld1q_f32(pC + c_hstep * 3 + 8);
                        transpose4x4_ps(_c800, _c801, _c802, _c803);
                        if (beta == 1.f)
                        {
                            _f8 = vaddq_f32(_f8, _c800);
                            _f9 = vaddq_f32(_f9, _c801);
                            _fa = vaddq_f32(_fa, _c802);
                            _fb = vaddq_f32(_fb, _c803);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f8 = vmlaq_f32(_f8, _c800, _beta);
                            _f9 = vmlaq_f32(_f9, _c801, _beta);
                            _fa = vmlaq_f32(_fa, _c802, _beta);
                            _fb = vmlaq_f32(_fb, _c803, _beta);
                        }
                        pC += 12;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f1 = vaddq_f32(_f1, _c1);
                    float32x4_t _c2 = vdupq_n_f32(pC[2] * beta);
                    _f2 = vaddq_f32(_f2, _c2);
                    float32x4_t _c3 = vdupq_n_f32(pC[3] * beta);
                    _f3 = vaddq_f32(_f3, _c3);
                    float32x4_t _c4 = vdupq_n_f32(pC[4] * beta);
                    _f4 = vaddq_f32(_f4, _c4);
                    float32x4_t _c5 = vdupq_n_f32(pC[5] * beta);
                    _f5 = vaddq_f32(_f5, _c5);
                    float32x4_t _c6 = vdupq_n_f32(pC[6] * beta);
                    _f6 = vaddq_f32(_f6, _c6);
                    float32x4_t _c7 = vdupq_n_f32(pC[7] * beta);
                    _f7 = vaddq_f32(_f7, _c7);
                    float32x4_t _c8 = vdupq_n_f32(pC[8] * beta);
                    _f8 = vaddq_f32(_f8, _c8);
                    float32x4_t _c9 = vdupq_n_f32(pC[9] * beta);
                    _f9 = vaddq_f32(_f9, _c9);
                    float32x4_t _ca = vdupq_n_f32(pC[10] * beta);
                    _fa = vaddq_f32(_fa, _ca);
                    float32x4_t _cb = vdupq_n_f32(pC[11] * beta);
                    _fb = vaddq_f32(_fb, _cb);
                    pC += 12;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
                _f8 = vmulq_f32(_f8, _alpha);
                _f9 = vmulq_f32(_f9, _alpha);
                _fa = vmulq_f32(_fa, _alpha);
                _fb = vmulq_f32(_fb, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float32x4x4_t _r0;
                        _r0.val[0] = _f0;
                        _r0.val[1] = _f1;
                        _r0.val[2] = _f2;
                        _r0.val[3] = _f3;
                        vst4q_f32(p0f, _r0);
                        float32x4x4_t _r1;
                        _r1.val[0] = _f4;
                        _r1.val[1] = _f5;
                        _r1.val[2] = _f6;
                        _r1.val[3] = _f7;
                        vst4q_f32(p0f + out_hstep * 4, _r1);
                        float32x4x4_t _r2;
                        _r2.val[0] = _f8;
                        _r2.val[1] = _f9;
                        _r2.val[2] = _fa;
                        _r2.val[3] = _fb;
                        vst4q_f32(p0f + out_hstep * 8, _r2);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + out_hstep, _f1);
                        vst1q_f32(p0f + out_hstep * 2, _f2);
                        vst1q_f32(p0f + out_hstep * 3, _f3);
                        vst1q_f32(p0f + out_hstep * 4, _f4);
                        vst1q_f32(p0f + out_hstep * 5, _f5);
                        vst1q_f32(p0f + out_hstep * 6, _f6);
                        vst1q_f32(p0f + out_hstep * 7, _f7);
                        vst1q_f32(p0f + out_hstep * 8, _f8);
                        vst1q_f32(p0f + out_hstep * 9, _f9);
                        vst1q_f32(p0f + out_hstep * 10, _fa);
                        vst1q_f32(p0f + out_hstep * 11, _fb);
                    }
                    p0f += out_hstep * 12;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                        vst1q_f32(p0f + 8, _f2);
                        vst1q_f32(p0f + 12, _f3);
                        vst1q_f32(p0f + 16, _f4);
                        vst1q_f32(p0f + 20, _f5);
                        vst1q_f32(p0f + 24, _f6);
                        vst1q_f32(p0f + 28, _f7);
                        vst1q_f32(p0f + 32, _f8);
                        vst1q_f32(p0f + 36, _f9);
                        vst1q_f32(p0f + 40, _fa);
                        vst1q_f32(p0f + 44, _fb);
                        p0f += 48;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_f0, _f1, _f2, _f3);
                        transpose4x4_ps(_f4, _f5, _f6, _f7);
                        transpose4x4_ps(_f8, _f9, _fa, _fb);
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f4);
                        vst1q_f32(p0f + 8, _f8);
                        vst1q_f32(p0f + out_hstep, _f1);
                        vst1q_f32(p0f + out_hstep + 4, _f5);
                        vst1q_f32(p0f + out_hstep + 8, _f9);
                        vst1q_f32(p0f + out_hstep * 2, _f2);
                        vst1q_f32(p0f + out_hstep * 2 + 4, _f6);
                        vst1q_f32(p0f + out_hstep * 2 + 8, _fa);
                        vst1q_f32(p0f + out_hstep * 3, _f3);
                        vst1q_f32(p0f + out_hstep * 3 + 4, _f7);
                        vst1q_f32(p0f + out_hstep * 3 + 8, _fb);
                        p0f += 12;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                uint16x4_t _bf2 = float2bfloat(_f2);
                uint16x4_t _bf3 = float2bfloat(_f3);
                uint16x4_t _bf4 = float2bfloat(_f4);
                uint16x4_t _bf5 = float2bfloat(_f5);
                uint16x4_t _bf6 = float2bfloat(_f6);
                uint16x4_t _bf7 = float2bfloat(_f7);
                uint16x4_t _bf8 = float2bfloat(_f8);
                uint16x4_t _bf9 = float2bfloat(_f9);
                uint16x4_t _bfa = float2bfloat(_fa);
                uint16x4_t _bfb = float2bfloat(_fb);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        uint16x4x4_t _r0;
                        _r0.val[0] = _bf0;
                        _r0.val[1] = _bf1;
                        _r0.val[2] = _bf2;
                        _r0.val[3] = _bf3;
                        vst4_u16(p0, _r0);
                        uint16x4x4_t _r1;
                        _r1.val[0] = _bf4;
                        _r1.val[1] = _bf5;
                        _r1.val[2] = _bf6;
                        _r1.val[3] = _bf7;
                        vst4_u16(p0 + out_hstep * 4, _r1);
                        uint16x4x4_t _r2;
                        _r2.val[0] = _bf8;
                        _r2.val[1] = _bf9;
                        _r2.val[2] = _bfa;
                        _r2.val[3] = _bfb;
                        vst4_u16(p0 + out_hstep * 8, _r2);
                    }
                    if (out_elempack == 1)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + out_hstep, _bf1);
                        vst1_u16(p0 + out_hstep * 2, _bf2);
                        vst1_u16(p0 + out_hstep * 3, _bf3);
                        vst1_u16(p0 + out_hstep * 4, _bf4);
                        vst1_u16(p0 + out_hstep * 5, _bf5);
                        vst1_u16(p0 + out_hstep * 6, _bf6);
                        vst1_u16(p0 + out_hstep * 7, _bf7);
                        vst1_u16(p0 + out_hstep * 8, _bf8);
                        vst1_u16(p0 + out_hstep * 9, _bf9);
                        vst1_u16(p0 + out_hstep * 10, _bfa);
                        vst1_u16(p0 + out_hstep * 11, _bfb);
                    }
                    p0 += out_hstep * 12;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                        vst1q_u16(p0 + 8, vcombine_u16(_bf2, _bf3));
                        vst1q_u16(p0 + 16, vcombine_u16(_bf4, _bf5));
                        vst1q_u16(p0 + 24, vcombine_u16(_bf6, _bf7));
                        vst1q_u16(p0 + 32, vcombine_u16(_bf8, _bf9));
                        vst1q_u16(p0 + 40, vcombine_u16(_bfa, _bfb));
                        p0 += 48;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x12_u16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7, _bf8, _bf9, _bfa, _bfb);

                        vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                        vst1_u16(p0 + 8, _bf2);
                        vst1q_u16(p0 + out_hstep, vcombine_u16(_bf3, _bf4));
                        vst1_u16(p0 + out_hstep + 8, _bf5);
                        vst1q_u16(p0 + out_hstep * 2, vcombine_u16(_bf6, _bf7));
                        vst1_u16(p0 + out_hstep * 2 + 8, _bf8);
                        vst1q_u16(p0 + out_hstep * 3, vcombine_u16(_bf9, _bfa));
                        vst1_u16(p0 + out_hstep * 3 + 8, _bfb);
                        p0 += 12;
                    }
                }
            }
            pp += 48;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _f0 = vld1q_f32(pp + 0);
            float32x4_t _f1 = vld1q_f32(pp + 4);
            float32x4_t _f2 = vld1q_f32(pp + 8);
            float32x4_t _f3 = vld1q_f32(pp + 12);
            float32x4_t _f4 = vld1q_f32(pp + 16);
            float32x4_t _f5 = vld1q_f32(pp + 20);
            float32x4_t _f6 = vld1q_f32(pp + 24);
            float32x4_t _f7 = vld1q_f32(pp + 28);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4x2_t _r0 = vuzpq_f32(_f0, _f1);
            _f0 = _r0.val[0];
            _f1 = _r0.val[1];
            _r0 = vuzpq_f32(_f2, _f3);
            _f2 = _r0.val[0];
            _f3 = _r0.val[1];
            _r0 = vuzpq_f32(_f4, _f5);
            _f4 = _r0.val[0];
            _f5 = _r0.val[1];
            _r0 = vuzpq_f32(_f6, _f7);
            _f6 = _r0.val[0];
            _f7 = _r0.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                    _f2 = vaddq_f32(_f2, _c);
                    _f3 = vaddq_f32(_f3, _c);
                    _f4 = vaddq_f32(_f4, _c);
                    _f5 = vaddq_f32(_f5, _c);
                    _f6 = vaddq_f32(_f6, _c);
                    _f7 = vaddq_f32(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c0 = vmulq_n_f32(_c0, beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        float32x4_t _c4 = vld1q_f32(pC + 16);
                        float32x4_t _c5 = vld1q_f32(pC + 20);
                        float32x4_t _c6 = vld1q_f32(pC + 24);
                        float32x4_t _c7 = vld1q_f32(pC + 28);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                            _f4 = vaddq_f32(_f4, _c4);
                            _f5 = vaddq_f32(_f5, _c5);
                            _f6 = vaddq_f32(_f6, _c6);
                            _f7 = vaddq_f32(_f7, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                            _f4 = vmlaq_f32(_f4, _c4, _beta);
                            _f5 = vmlaq_f32(_f5, _c5, _beta);
                            _f6 = vmlaq_f32(_f6, _c6, _beta);
                            _f7 = vmlaq_f32(_f7, _c7, _beta);
                        }
                        pC += 8 * c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c000 = vld1q_f32(pC);
                        float32x4_t _c001 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c002 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c003 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c000, _c001, _c002, _c003);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c000);
                            _f1 = vaddq_f32(_f1, _c001);
                            _f2 = vaddq_f32(_f2, _c002);
                            _f3 = vaddq_f32(_f3, _c003);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c000, _beta);
                            _f1 = vmlaq_f32(_f1, _c001, _beta);
                            _f2 = vmlaq_f32(_f2, _c002, _beta);
                            _f3 = vmlaq_f32(_f3, _c003, _beta);
                        }
                        float32x4_t _c400 = vld1q_f32(pC + 4);
                        float32x4_t _c401 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c402 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c403 = vld1q_f32(pC + c_hstep * 3 + 4);
                        transpose4x4_ps(_c400, _c401, _c402, _c403);
                        if (beta == 1.f)
                        {
                            _f4 = vaddq_f32(_f4, _c400);
                            _f5 = vaddq_f32(_f5, _c401);
                            _f6 = vaddq_f32(_f6, _c402);
                            _f7 = vaddq_f32(_f7, _c403);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f4 = vmlaq_f32(_f4, _c400, _beta);
                            _f5 = vmlaq_f32(_f5, _c401, _beta);
                            _f6 = vmlaq_f32(_f6, _c402, _beta);
                            _f7 = vmlaq_f32(_f7, _c403, _beta);
                        }
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f1 = vaddq_f32(_f1, _c1);
                    float32x4_t _c2 = vdupq_n_f32(pC[2] * beta);
                    _f2 = vaddq_f32(_f2, _c2);
                    float32x4_t _c3 = vdupq_n_f32(pC[3] * beta);
                    _f3 = vaddq_f32(_f3, _c3);
                    float32x4_t _c4 = vdupq_n_f32(pC[4] * beta);
                    _f4 = vaddq_f32(_f4, _c4);
                    float32x4_t _c5 = vdupq_n_f32(pC[5] * beta);
                    _f5 = vaddq_f32(_f5, _c5);
                    float32x4_t _c6 = vdupq_n_f32(pC[6] * beta);
                    _f6 = vaddq_f32(_f6, _c6);
                    float32x4_t _c7 = vdupq_n_f32(pC[7] * beta);
                    _f7 = vaddq_f32(_f7, _c7);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float32x4x4_t _r0;
                        _r0.val[0] = _f0;
                        _r0.val[1] = _f1;
                        _r0.val[2] = _f2;
                        _r0.val[3] = _f3;
                        vst4q_f32(p0f, _r0);
                        float32x4x4_t _r1;
                        _r1.val[0] = _f4;
                        _r1.val[1] = _f5;
                        _r1.val[2] = _f6;
                        _r1.val[3] = _f7;
                        vst4q_f32(p0f + out_hstep * 4, _r1);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + out_hstep, _f1);
                        vst1q_f32(p0f + out_hstep * 2, _f2);
                        vst1q_f32(p0f + out_hstep * 3, _f3);
                        vst1q_f32(p0f + out_hstep * 4, _f4);
                        vst1q_f32(p0f + out_hstep * 5, _f5);
                        vst1q_f32(p0f + out_hstep * 6, _f6);
                        vst1q_f32(p0f + out_hstep * 7, _f7);
                    }
                    p0f += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                        vst1q_f32(p0f + 8, _f2);
                        vst1q_f32(p0f + 12, _f3);
                        vst1q_f32(p0f + 16, _f4);
                        vst1q_f32(p0f + 20, _f5);
                        vst1q_f32(p0f + 24, _f6);
                        vst1q_f32(p0f + 28, _f7);
                        p0f += 32;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_f0, _f1, _f2, _f3);
                        transpose4x4_ps(_f4, _f5, _f6, _f7);
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f4);
                        vst1q_f32(p0f + out_hstep, _f1);
                        vst1q_f32(p0f + out_hstep + 4, _f5);
                        vst1q_f32(p0f + out_hstep * 2, _f2);
                        vst1q_f32(p0f + out_hstep * 2 + 4, _f6);
                        vst1q_f32(p0f + out_hstep * 3, _f3);
                        vst1q_f32(p0f + out_hstep * 3 + 4, _f7);
                        p0f += 8;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                uint16x4_t _bf2 = float2bfloat(_f2);
                uint16x4_t _bf3 = float2bfloat(_f3);
                uint16x4_t _bf4 = float2bfloat(_f4);
                uint16x4_t _bf5 = float2bfloat(_f5);
                uint16x4_t _bf6 = float2bfloat(_f6);
                uint16x4_t _bf7 = float2bfloat(_f7);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        uint16x4x4_t _r0;
                        _r0.val[0] = _bf0;
                        _r0.val[1] = _bf1;
                        _r0.val[2] = _bf2;
                        _r0.val[3] = _bf3;
                        vst4_u16(p0, _r0);
                        uint16x4x4_t _r1;
                        _r1.val[0] = _bf4;
                        _r1.val[1] = _bf5;
                        _r1.val[2] = _bf6;
                        _r1.val[3] = _bf7;
                        vst4_u16(p0 + out_hstep * 4, _r1);
                    }
                    if (out_elempack == 1)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + out_hstep, _bf1);
                        vst1_u16(p0 + out_hstep * 2, _bf2);
                        vst1_u16(p0 + out_hstep * 3, _bf3);
                        vst1_u16(p0 + out_hstep * 4, _bf4);
                        vst1_u16(p0 + out_hstep * 5, _bf5);
                        vst1_u16(p0 + out_hstep * 6, _bf6);
                        vst1_u16(p0 + out_hstep * 7, _bf7);
                    }
                    p0 += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                        vst1q_u16(p0 + 8, vcombine_u16(_bf2, _bf3));
                        vst1q_u16(p0 + 16, vcombine_u16(_bf4, _bf5));
                        vst1q_u16(p0 + 24, vcombine_u16(_bf6, _bf7));
                        p0 += 32;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x8_u16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);

                        vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                        vst1q_u16(p0 + out_hstep, vcombine_u16(_bf2, _bf3));
                        vst1q_u16(p0 + out_hstep * 2, vcombine_u16(_bf4, _bf5));
                        vst1q_u16(p0 + out_hstep * 3, vcombine_u16(_bf6, _bf7));
                        p0 += 8;
                    }
                }
            }
            pp += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _f0 = vld1q_f32(pp + 0);
            float32x4_t _f1 = vld1q_f32(pp + 4);
            float32x4_t _f2 = vld1q_f32(pp + 8);
            float32x4_t _f3 = vld1q_f32(pp + 12);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4x2_t _r0 = vuzpq_f32(_f0, _f1);
            _f0 = _r0.val[0];
            _f1 = _r0.val[1];
            _r0 = vuzpq_f32(_f2, _f3);
            _f2 = _r0.val[0];
            _f3 = _r0.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                    _f2 = vaddq_f32(_f2, _c);
                    _f3 = vaddq_f32(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c0 = vmulq_n_f32(_c0, beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                        }
                        pC += 4 * c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c000 = vld1q_f32(pC);
                        float32x4_t _c001 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c002 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c003 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c000, _c001, _c002, _c003);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c000);
                            _f1 = vaddq_f32(_f1, _c001);
                            _f2 = vaddq_f32(_f2, _c002);
                            _f3 = vaddq_f32(_f3, _c003);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c000, _beta);
                            _f1 = vmlaq_f32(_f1, _c001, _beta);
                            _f2 = vmlaq_f32(_f2, _c002, _beta);
                            _f3 = vmlaq_f32(_f3, _c003, _beta);
                        }
                        pC += 4;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f1 = vaddq_f32(_f1, _c1);
                    float32x4_t _c2 = vdupq_n_f32(pC[2] * beta);
                    _f2 = vaddq_f32(_f2, _c2);
                    float32x4_t _c3 = vdupq_n_f32(pC[3] * beta);
                    _f3 = vaddq_f32(_f3, _c3);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float32x4x4_t _r0;
                        _r0.val[0] = _f0;
                        _r0.val[1] = _f1;
                        _r0.val[2] = _f2;
                        _r0.val[3] = _f3;
                        vst4q_f32(p0f, _r0);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + out_hstep, _f1);
                        vst1q_f32(p0f + out_hstep * 2, _f2);
                        vst1q_f32(p0f + out_hstep * 3, _f3);
                    }
                    p0f += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                        vst1q_f32(p0f + 8, _f2);
                        vst1q_f32(p0f + 12, _f3);
                        p0f += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_f0, _f1, _f2, _f3);
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + out_hstep, _f1);
                        vst1q_f32(p0f + out_hstep * 2, _f2);
                        vst1q_f32(p0f + out_hstep * 3, _f3);
                        p0f += 4;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                uint16x4_t _bf2 = float2bfloat(_f2);
                uint16x4_t _bf3 = float2bfloat(_f3);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        uint16x4x4_t _r0;
                        _r0.val[0] = _bf0;
                        _r0.val[1] = _bf1;
                        _r0.val[2] = _bf2;
                        _r0.val[3] = _bf3;
                        vst4_u16(p0, _r0);
                    }
                    if (out_elempack == 1)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + out_hstep, _bf1);
                        vst1_u16(p0 + out_hstep * 2, _bf2);
                        vst1_u16(p0 + out_hstep * 3, _bf3);
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                        vst1q_u16(p0 + 8, vcombine_u16(_bf2, _bf3));
                        p0 += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_u16(_bf0, _bf1, _bf2, _bf3);

                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + out_hstep, _bf1);
                        vst1_u16(p0 + out_hstep * 2, _bf2);
                        vst1_u16(p0 + out_hstep * 3, _bf3);
                        p0 += 4;
                    }
                }
            }
            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _f0 = vld1q_f32(pp + 0);
            float32x4_t _f1 = vld1q_f32(pp + 4);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4x2_t _r0 = vuzpq_f32(_f0, _f1);
            _f0 = _r0.val[0];
            _f1 = _r0.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c0 = vmulq_n_f32(_c0, beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                        }
                        pC += 2 * c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x2_t _c0 = vld1_f32(pC);
                        float32x2_t _c1 = vld1_f32(pC + c_hstep);
                        float32x2_t _c2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _c3 = vld1_f32(pC + c_hstep * 3);
                        float32x2x2_t _c01 = vtrn_f32(_c0, _c1);
                        float32x2x2_t _c23 = vtrn_f32(_c2, _c3);
                        float32x4_t _c00 = vcombine_f32(_c01.val[0], _c23.val[0]);
                        float32x4_t _c10 = vcombine_f32(_c01.val[1], _c23.val[1]);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c00);
                            _f1 = vaddq_f32(_f1, _c10);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c00, _beta);
                            _f1 = vmlaq_f32(_f1, _c10, _beta);
                        }
                        pC += 2;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f1 = vaddq_f32(_f1, _c1);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + out_hstep, _f1);
                        p0f += out_hstep * 2;
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                        p0f += 8;
                    }
                    if (out_elempack == 1)
                    {
                        float32x4x2_t _r0 = vzipq_f32(_f0, _f1);
                        vst1_f32(p0f, vget_low_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep, vget_high_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep * 2, vget_low_f32(_r0.val[1]));
                        vst1_f32(p0f + out_hstep * 3, vget_high_f32(_r0.val[1]));
                        p0f += 2;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + out_hstep, _bf1);
                        p0 += out_hstep * 2;
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                        p0 += 8;
                    }
                    if (out_elempack == 1)
                    {
                        uint16x4x2_t _r0;
                        _r0.val[0] = _bf0;
                        _r0.val[1] = _bf1;
                        vst2_lane_u16(p0, _r0, 0);
                        vst2_lane_u16(p0 + out_hstep, _r0, 1);
                        vst2_lane_u16(p0 + out_hstep * 2, _r0, 2);
                        vst2_lane_u16(p0 + out_hstep * 3, _r0, 3);
                        p0 += 2;
                    }
                }
            }
            pp += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _f0 = vld1q_f32(pp + 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c0 = vmulq_n_f32(_c0, beta);
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                        }
                        pC += c_elempack;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c00 = vdupq_n_f32(0.f);
                        _c00 = vld1q_lane_f32(pC, _c00, 0);
                        _c00 = vld1q_lane_f32(pC + c_hstep, _c00, 1);
                        _c00 = vld1q_lane_f32(pC + c_hstep * 2, _c00, 2);
                        _c00 = vld1q_lane_f32(pC + c_hstep * 3, _c00, 3);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c00);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c00, _beta);
                        }
                        pC += 1;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    pC += 1;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        p0f += out_hstep;
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        p0f += 4;
                    }
                    if (out_elempack == 1)
                    {
                        p0f[0] = vgetq_lane_f32(_f0, 0);
                        p0f[out_hstep] = vgetq_lane_f32(_f0, 1);
                        p0f[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                        p0f[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                        p0f += 1;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1_u16(p0, _bf0);
                        p0 += out_hstep;
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0, _bf0);
                        p0 += 4;
                    }
                    if (out_elempack == 1)
                    {
                        p0[0] = vget_lane_u16(_bf0, 0);
                        p0[out_hstep] = vget_lane_u16(_bf0, 1);
                        p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                        p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                        p0 += 1;
                    }
                }
            }
            pp += 4;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* p0;
        float* p0f;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;
            p0f = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC = (const float*)C + i + ii;
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _s0 = vld1q_f32(pp);
            float32x4_t _s1 = vld1q_f32(pp + 4);
            float32x4_t _s2 = vld1q_f32(pp + 8);
            float32x4_t _s3 = vld1q_f32(pp + 12);
            float32x4_t _s4 = vld1q_f32(pp + 16);
            float32x4_t _s5 = vld1q_f32(pp + 20);
            float32x4_t _f0 = vcombine_f32(vget_low_f32(_s0), vget_low_f32(_s1));
            float32x4_t _f1 = vcombine_f32(vget_high_f32(_s0), vget_high_f32(_s1));
            float32x4_t _f2 = vcombine_f32(vget_low_f32(_s2), vget_low_f32(_s3));
            float32x4_t _f3 = vcombine_f32(vget_high_f32(_s2), vget_high_f32(_s3));
            float32x4_t _f4 = vcombine_f32(vget_low_f32(_s4), vget_low_f32(_s5));
            float32x4_t _f5 = vcombine_f32(vget_high_f32(_s4), vget_high_f32(_s5));
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                    _f2 = vaddq_f32(_f2, _c);
                    _f3 = vaddq_f32(_f3, _c);
                    _f4 = vaddq_f32(_f4, _c);
                    _f5 = vaddq_f32(_f5, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c1);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    float32x4_t _c2 = vld1q_f32(pC + 4);
                    float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                    if (beta == 1.f)
                    {
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                    float32x4_t _c4 = vld1q_f32(pC + 8);
                    float32x4_t _c5 = vld1q_f32(pC + c_hstep + 8);
                    if (beta == 1.f)
                    {
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f4 = vmlaq_f32(_f4, _c4, _beta);
                        _f5 = vmlaq_f32(_f5, _c5, _beta);
                    }
                    pC += 12;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + 8);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c1);
                        _f3 = vaddq_f32(_f3, _c1);
                        _f4 = vaddq_f32(_f4, _c2);
                        _f5 = vaddq_f32(_f5, _c2);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c0, _beta);
                        _f2 = vmlaq_f32(_f2, _c1, _beta);
                        _f3 = vmlaq_f32(_f3, _c1, _beta);
                        _f4 = vmlaq_f32(_f4, _c2, _beta);
                        _f5 = vmlaq_f32(_f5, _c2, _beta);
                    }
                    pC += 12;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                        vst1q_f32(p0f + out_hstep * 4, _f2);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f3);
                        vst1q_f32(p0f + out_hstep * 8, _f4);
                        vst1q_f32(p0f + out_hstep * 8 + 4, _f5);
                    }
                    if (out_elempack == 1)
                    {
                        float32x4x2_t _r0 = vzipq_f32(_f0, _f1);
                        float32x4x2_t _r1 = vzipq_f32(_f2, _f3);
                        float32x4x2_t _r2 = vzipq_f32(_f4, _f5);
                        vst1_f32(p0f, vget_low_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep, vget_high_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep * 2, vget_low_f32(_r0.val[1]));
                        vst1_f32(p0f + out_hstep * 3, vget_high_f32(_r0.val[1]));
                        vst1_f32(p0f + out_hstep * 4, vget_low_f32(_r1.val[0]));
                        vst1_f32(p0f + out_hstep * 5, vget_high_f32(_r1.val[0]));
                        vst1_f32(p0f + out_hstep * 6, vget_low_f32(_r1.val[1]));
                        vst1_f32(p0f + out_hstep * 7, vget_high_f32(_r1.val[1]));
                        vst1_f32(p0f + out_hstep * 8, vget_low_f32(_r2.val[0]));
                        vst1_f32(p0f + out_hstep * 9, vget_high_f32(_r2.val[0]));
                        vst1_f32(p0f + out_hstep * 10, vget_low_f32(_r2.val[1]));
                        vst1_f32(p0f + out_hstep * 11, vget_high_f32(_r2.val[1]));
                    }
                    p0f += out_hstep * 12;
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f2);
                        vst1q_f32(p0f + 8, _f4);
                        vst1q_f32(p0f + out_hstep, _f1);
                        vst1q_f32(p0f + out_hstep + 4, _f3);
                        vst1q_f32(p0f + out_hstep + 8, _f5);
                        p0f += 12;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                uint16x4_t _bf2 = float2bfloat(_f2);
                uint16x4_t _bf3 = float2bfloat(_f3);
                uint16x4_t _bf4 = float2bfloat(_f4);
                uint16x4_t _bf5 = float2bfloat(_f5);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + 4, _bf1);
                        vst1_u16(p0 + out_hstep * 4, _bf2);
                        vst1_u16(p0 + out_hstep * 4 + 4, _bf3);
                        vst1_u16(p0 + out_hstep * 8, _bf4);
                        vst1_u16(p0 + out_hstep * 8 + 4, _bf5);
                    }
                    if (out_elempack == 1)
                    {
                        uint16x4x2_t _r0;
                        _r0.val[0] = _bf0;
                        _r0.val[1] = _bf1;
                        vst2_lane_u16(p0, _r0, 0);
                        vst2_lane_u16(p0 + out_hstep, _r0, 1);
                        vst2_lane_u16(p0 + out_hstep * 2, _r0, 2);
                        vst2_lane_u16(p0 + out_hstep * 3, _r0, 3);
                        uint16x4x2_t _r1;
                        _r1.val[0] = _bf2;
                        _r1.val[1] = _bf3;
                        vst2_lane_u16(p0 + out_hstep * 4, _r1, 0);
                        vst2_lane_u16(p0 + out_hstep * 5, _r1, 1);
                        vst2_lane_u16(p0 + out_hstep * 6, _r1, 2);
                        vst2_lane_u16(p0 + out_hstep * 7, _r1, 3);
                        uint16x4x2_t _r2;
                        _r2.val[0] = _bf4;
                        _r2.val[1] = _bf5;
                        vst2_lane_u16(p0 + out_hstep * 8, _r2, 0);
                        vst2_lane_u16(p0 + out_hstep * 9, _r2, 1);
                        vst2_lane_u16(p0 + out_hstep * 10, _r2, 2);
                        vst2_lane_u16(p0 + out_hstep * 11, _r2, 3);
                    }
                    p0 += out_hstep * 12;
                }
                else
                {
                    vst1q_u16(p0, vcombine_u16(_bf0, _bf2));
                    vst1_u16(p0 + 8, _bf4);
                    vst1q_u16(p0 + out_hstep, vcombine_u16(_bf1, _bf3));
                    vst1_u16(p0 + out_hstep + 8, _bf5);
                    p0 += 12;
                }
            }
            pp += 24;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _s0 = vld1q_f32(pp);
            float32x4_t _s1 = vld1q_f32(pp + 4);
            float32x4_t _s2 = vld1q_f32(pp + 8);
            float32x4_t _s3 = vld1q_f32(pp + 12);
            float32x4_t _f0 = vcombine_f32(vget_low_f32(_s0), vget_low_f32(_s1));
            float32x4_t _f1 = vcombine_f32(vget_high_f32(_s0), vget_high_f32(_s1));
            float32x4_t _f2 = vcombine_f32(vget_low_f32(_s2), vget_low_f32(_s3));
            float32x4_t _f3 = vcombine_f32(vget_high_f32(_s2), vget_high_f32(_s3));
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                    _f2 = vaddq_f32(_f2, _c);
                    _f3 = vaddq_f32(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    float32x4_t _c2 = vld1q_f32(pC + 4);
                    float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                    if (beta == 1.f)
                    {
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                        _f2 = vaddq_f32(_f2, _c1);
                        _f3 = vaddq_f32(_f3, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c0, _beta);
                        _f2 = vmlaq_f32(_f2, _c1, _beta);
                        _f3 = vmlaq_f32(_f3, _c1, _beta);
                    }
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                        vst1q_f32(p0f + out_hstep * 4, _f2);
                        vst1q_f32(p0f + out_hstep * 4 + 4, _f3);
                    }
                    if (out_elempack == 1)
                    {
                        float32x4x2_t _r0 = vzipq_f32(_f0, _f1);
                        float32x4x2_t _r1 = vzipq_f32(_f2, _f3);
                        vst1_f32(p0f, vget_low_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep, vget_high_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep * 2, vget_low_f32(_r0.val[1]));
                        vst1_f32(p0f + out_hstep * 3, vget_high_f32(_r0.val[1]));
                        vst1_f32(p0f + out_hstep * 4, vget_low_f32(_r1.val[0]));
                        vst1_f32(p0f + out_hstep * 5, vget_high_f32(_r1.val[0]));
                        vst1_f32(p0f + out_hstep * 6, vget_low_f32(_r1.val[1]));
                        vst1_f32(p0f + out_hstep * 7, vget_high_f32(_r1.val[1]));
                    }
                    p0f += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f2);
                        vst1q_f32(p0f + out_hstep, _f1);
                        vst1q_f32(p0f + out_hstep + 4, _f3);
                        p0f += 8;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                uint16x4_t _bf2 = float2bfloat(_f2);
                uint16x4_t _bf3 = float2bfloat(_f3);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + 4, _bf1);
                        vst1_u16(p0 + out_hstep * 4, _bf2);
                        vst1_u16(p0 + out_hstep * 4 + 4, _bf3);
                    }
                    if (out_elempack == 1)
                    {
                        uint16x4x2_t _r0;
                        _r0.val[0] = _bf0;
                        _r0.val[1] = _bf1;
                        vst2_lane_u16(p0, _r0, 0);
                        vst2_lane_u16(p0 + out_hstep, _r0, 1);
                        vst2_lane_u16(p0 + out_hstep * 2, _r0, 2);
                        vst2_lane_u16(p0 + out_hstep * 3, _r0, 3);
                        uint16x4x2_t _r1;
                        _r1.val[0] = _bf2;
                        _r1.val[1] = _bf3;
                        vst2_lane_u16(p0 + out_hstep * 4, _r1, 0);
                        vst2_lane_u16(p0 + out_hstep * 5, _r1, 1);
                        vst2_lane_u16(p0 + out_hstep * 6, _r1, 2);
                        vst2_lane_u16(p0 + out_hstep * 7, _r1, 3);
                    }
                    p0 += out_hstep * 8;
                }
                else
                {
                    vst1q_u16(p0, vcombine_u16(_bf0, _bf2));
                    vst1q_u16(p0 + out_hstep, vcombine_u16(_bf1, _bf3));
                    p0 += 8;
                }
            }
            pp += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _s0 = vld1q_f32(pp);
            float32x4_t _s1 = vld1q_f32(pp + 4);
            float32x4_t _f0 = vcombine_f32(vget_low_f32(_s0), vget_low_f32(_s1));
            float32x4_t _f1 = vcombine_f32(vget_high_f32(_s0), vget_high_f32(_s1));
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c0);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c0, _beta);
                    }
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                    }
                    if (out_elempack == 1)
                    {
                        float32x4x2_t _r0 = vzipq_f32(_f0, _f1);
                        vst1_f32(p0f, vget_low_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep, vget_high_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep * 2, vget_low_f32(_r0.val[1]));
                        vst1_f32(p0f + out_hstep * 3, vget_high_f32(_r0.val[1]));
                    }
                    p0f += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + out_hstep, _f1);
                        p0f += 4;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + 4, _bf1);
                    }
                    if (out_elempack == 1)
                    {
                        uint16x4x2_t _r0;
                        _r0.val[0] = _bf0;
                        _r0.val[1] = _bf1;
                        vst2_lane_u16(p0, _r0, 0);
                        vst2_lane_u16(p0 + out_hstep, _r0, 1);
                        vst2_lane_u16(p0 + out_hstep * 2, _r0, 2);
                        vst2_lane_u16(p0 + out_hstep * 3, _r0, 3);
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    vst1_u16(p0, _bf0);
                    vst1_u16(p0 + out_hstep, _bf1);
                    p0 += 4;
                }
            }
            pp += 8;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_NEON
            float32x4_t _f0 = vld1q_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    float32x4_t _c1 = vdupq_n_f32(pC[1] * beta);
                    _f0 = vaddq_f32(_f0, vcombine_f32(vget_low_f32(_c0), vget_low_f32(_c1)));
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x4_t _c = vcombine_f32(vld1_f32(pC), vld1_f32(pC + c_hstep));
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c, _beta);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c0 = vld1_f32(pC);
                    float32x4_t _c = vcombine_f32(_c0, _c0);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c, _beta);
                    }
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        float32x4x2_t _r0 = vuzpq_f32(_f0, _f0);
                        vst1_f32(p0f, vget_low_f32(_r0.val[0]));
                        vst1_f32(p0f + out_hstep, vget_low_f32(_r0.val[1]));
                        p0f += out_hstep * 2;
                    }
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        vst1_f32(p0f, vget_low_f32(_f0));
                        vst1_f32(p0f + out_hstep, vget_high_f32(_f0));
                        p0f += 2;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint32x2_t _bf0_32 = vreinterpret_u32_u16(_bf0);

                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        uint16x4x2_t _r0 = vuzp_u16(_bf0, _bf0);
                        vst1_lane_u32((uint32_t*)p0, vreinterpret_u32_u16(_r0.val[0]), 0);
                        vst1_lane_u32((uint32_t*)(p0 + out_hstep), vreinterpret_u32_u16(_r0.val[1]), 0);
                        p0 += out_hstep * 2;
                    }
                }
                else
                {
                    vst1_lane_u32((uint32_t*)p0, _bf0_32, 0);
                    vst1_lane_u32((uint32_t*)(p0 + out_hstep), _bf0_32, 1);
                    p0 += 2;
                }
            }
            pp += 4;
#else  // __ARM_NEON
            float sum00 = pp[0];
            float sum01 = pp[1];
            float sum10 = pp[2];
            float sum11 = pp[3];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float c0 = pC[0] * beta;
                    sum00 += c0;
                    sum10 += c0;
                    sum01 += c0;
                    sum11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float c0 = pC[0] * beta;
                    float c1 = pC[1] * beta;
                    sum00 += c0;
                    sum10 += c1;
                    sum01 += c0;
                    sum11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    sum00 += pC[0] * beta;
                    sum10 += pC[c_hstep] * beta;
                    sum01 += pC[1] * beta;
                    sum11 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float c0 = pC[0] * beta;
                    float c1 = pC[1] * beta;
                    sum00 += c0;
                    sum10 += c0;
                    sum01 += c1;
                    sum11 += c1;
                    pC += 2;
                }
            }

            sum00 *= alpha;
            sum10 *= alpha;
            sum01 *= alpha;
            sum11 *= alpha;

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0f[0] = sum00;
                        p0f[1] = sum10;
                        p0f[out_hstep] = sum01;
                        p0f[out_hstep + 1] = sum11;
                        p0f += out_hstep * 2;
                    }
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        p0f[0] = sum00;
                        p0f[1] = sum01;
                        p0f[out_hstep] = sum10;
                        p0f[out_hstep + 1] = sum11;
                        p0f += 2;
                    }
                }
            }
            else
            {
                unsigned short bf00 = float32_to_bfloat16(sum00);
                unsigned short bf10 = float32_to_bfloat16(sum10);
                unsigned short bf01 = float32_to_bfloat16(sum01);
                unsigned short bf11 = float32_to_bfloat16(sum11);
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0[0] = bf00;
                        p0[1] = bf10;
                        p0[out_hstep] = bf01;
                        p0[out_hstep + 1] = bf11;
                        p0 += out_hstep * 2;
                    }
                }
                else
                {
                    p0[0] = bf00;
                    p0[1] = bf01;
                    p0[out_hstep] = bf10;
                    p0[out_hstep + 1] = bf11;
                    p0 += 2;
                }
            }
            pp += 4;
#endif // __ARM_NEON
        }
        for (; jj < max_jj; jj += 1)
        {
#if __ARM_NEON
            float32x2_t _f0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x2_t _c = vdup_n_f32(pC[0] * beta);
                    _f0 = vadd_f32(_f0, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    _f0 = vadd_f32(_f0, _c);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x2_t _c = vld1_dup_f32(pC);
                    _c = vld1_lane_f32(pC + c_hstep, _c, 1);
                    if (beta == 1.f)
                    {
                        _f0 = vadd_f32(_f0, _c);
                    }
                    else
                    {
                        float32x2_t _beta = vdup_n_f32(beta);
                        _f0 = vmla_f32(_f0, _c, _beta);
                    }
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vdup_n_f32(pC[0]);
                    if (beta == 1.f)
                    {
                        _f0 = vadd_f32(_f0, _c);
                    }
                    else
                    {
                        float32x2_t _beta = vdup_n_f32(beta);
                        _f0 = vmla_f32(_f0, _c, _beta);
                    }
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                float32x2_t _alpha = vdup_n_f32(alpha);
                _f0 = vmul_f32(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1_f32(p0f, _f0);
                        p0f += out_hstep;
                    }
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        p0f[0] = vget_lane_f32(_f0, 0);
                        p0f[out_hstep] = vget_lane_f32(_f0, 1);
                        p0f += 1;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(vcombine_f32(_f0, _f0));
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1_lane_u32((uint32_t*)p0, vreinterpret_u32_u16(_bf0), 0);
                        p0 += out_hstep;
                    }
                }
                else
                {
                    vst1_lane_u16(p0, _bf0, 0);
                    vst1_lane_u16(p0 + out_hstep, _bf0, 1);
                    p0 += 1;
                }
            }
            pp += 2;
#else  // __ARM_NEON
            float sum0 = pp[0];
            float sum1 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[0] * beta;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    sum0 += pC[0] * beta;
                    sum1 += pC[c_hstep] * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[0] * beta;
                    pC += 1;
                }
            }

            sum0 *= alpha;
            sum1 *= alpha;

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0f[0] = sum0;
                        p0f[1] = sum1;
                        p0f += out_hstep;
                    }
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        p0f[0] = sum0;
                        p0f[out_hstep] = sum1;
                        p0f += 1;
                    }
                }
            }
            else
            {
                unsigned short bf0 = float32_to_bfloat16(sum0);
                unsigned short bf1 = float32_to_bfloat16(sum1);
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0[0] = bf0;
                        p0[1] = bf1;
                        p0 += out_hstep;
                    }
                }
                else
                {
                    p0[0] = bf0;
                    p0[out_hstep] = bf1;
                    p0 += 1;
                }
            }
            pp += 2;
#endif // __ARM_NEON
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        unsigned short* p0;
        float* p0f;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;
            p0f = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC = (const float*)C + i + ii;
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _f0 = vld1q_f32(pp + 0);
            float32x4_t _f1 = vld1q_f32(pp + 4);
            float32x4_t _f2 = vld1q_f32(pp + 8);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                    _f2 = vaddq_f32(_f2, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                    }
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta == 1.f)
                    {
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    float32x4_t _c2 = vld1q_f32(pC + 8);
                    if (beta == 1.f)
                    {
                        _f2 = vaddq_f32(_f2, _c2);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                    }
                    pC += 12;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + 8);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                    }
                    pC += 12;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + out_hstep * 4, _f1);
                        vst1q_f32(p0f + out_hstep * 8, _f2);
                    }
                    if (out_elempack == 1)
                    {
                        p0f[0] = vgetq_lane_f32(_f0, 0);
                        p0f[out_hstep] = vgetq_lane_f32(_f0, 1);
                        p0f[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                        p0f[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                        p0f[out_hstep * 4] = vgetq_lane_f32(_f1, 0);
                        p0f[out_hstep * 5] = vgetq_lane_f32(_f1, 1);
                        p0f[out_hstep * 6] = vgetq_lane_f32(_f1, 2);
                        p0f[out_hstep * 7] = vgetq_lane_f32(_f1, 3);
                        p0f[out_hstep * 8] = vgetq_lane_f32(_f2, 0);
                        p0f[out_hstep * 9] = vgetq_lane_f32(_f2, 1);
                        p0f[out_hstep * 10] = vgetq_lane_f32(_f2, 2);
                        p0f[out_hstep * 11] = vgetq_lane_f32(_f2, 3);
                    }
                    p0f += out_hstep * 12;
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                        vst1q_f32(p0f + 8, _f2);
                        p0f += 12;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                uint16x4_t _bf2 = float2bfloat(_f2);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + out_hstep * 4, _bf1);
                        vst1_u16(p0 + out_hstep * 8, _bf2);
                    }
                    if (out_elempack == 1)
                    {
                        p0[0] = vget_lane_u16(_bf0, 0);
                        p0[out_hstep] = vget_lane_u16(_bf0, 1);
                        p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                        p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                        p0[out_hstep * 4] = vget_lane_u16(_bf1, 0);
                        p0[out_hstep * 5] = vget_lane_u16(_bf1, 1);
                        p0[out_hstep * 6] = vget_lane_u16(_bf1, 2);
                        p0[out_hstep * 7] = vget_lane_u16(_bf1, 3);
                        p0[out_hstep * 8] = vget_lane_u16(_bf2, 0);
                        p0[out_hstep * 9] = vget_lane_u16(_bf2, 1);
                        p0[out_hstep * 10] = vget_lane_u16(_bf2, 2);
                        p0[out_hstep * 11] = vget_lane_u16(_bf2, 3);
                    }
                    p0 += out_hstep * 12;
                }
                else
                {
                    vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                    vst1_u16(p0 + 8, _bf2);
                    p0 += 12;
                }
            }
            pp += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _f0 = vld1q_f32(pp + 0);
            float32x4_t _f1 = vld1q_f32(pp + 4);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                    _f1 = vaddq_f32(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                    }
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta == 1.f)
                    {
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + out_hstep * 4, _f1);
                    }
                    if (out_elempack == 1)
                    {
                        p0f[0] = vgetq_lane_f32(_f0, 0);
                        p0f[out_hstep] = vgetq_lane_f32(_f0, 1);
                        p0f[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                        p0f[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                        p0f[out_hstep * 4] = vgetq_lane_f32(_f1, 0);
                        p0f[out_hstep * 5] = vgetq_lane_f32(_f1, 1);
                        p0f[out_hstep * 6] = vgetq_lane_f32(_f1, 2);
                        p0f[out_hstep * 7] = vgetq_lane_f32(_f1, 3);
                    }
                    p0f += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        vst1q_f32(p0f + 4, _f1);
                        p0f += 8;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint16x4_t _bf1 = float2bfloat(_f1);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0, _bf0);
                        vst1_u16(p0 + out_hstep * 4, _bf1);
                    }
                    if (out_elempack == 1)
                    {
                        p0[0] = vget_lane_u16(_bf0, 0);
                        p0[out_hstep] = vget_lane_u16(_bf0, 1);
                        p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                        p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                        p0[out_hstep * 4] = vget_lane_u16(_bf1, 0);
                        p0[out_hstep * 5] = vget_lane_u16(_bf1, 1);
                        p0[out_hstep * 6] = vget_lane_u16(_bf1, 2);
                        p0[out_hstep * 7] = vget_lane_u16(_bf1, 3);
                    }
                    p0 += out_hstep * 8;
                }
                else
                {
                    vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                    p0 += 8;
                }
            }
            pp += 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _f0 = vld1q_f32(pp + 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                    }
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
            }
            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0f, _f0);
                    }
                    if (out_elempack == 1)
                    {
                        p0f[0] = vgetq_lane_f32(_f0, 0);
                        p0f[out_hstep] = vgetq_lane_f32(_f0, 1);
                        p0f[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                        p0f[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                    }
                    p0f += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0f, _f0);
                        p0f += 4;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0, _bf0);
                    }
                    if (out_elempack == 1)
                    {
                        p0[0] = vget_lane_u16(_bf0, 0);
                        p0[out_hstep] = vget_lane_u16(_bf0, 1);
                        p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                        p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    vst1_u16(p0, _bf0);
                    p0 += 4;
                }
            }
            pp += 4;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_NEON
            float32x4_t _f0 = vcombine_f32(vld1_f32(pp), vdup_n_f32(0.f));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x4_t _c = vcombine_f32(vld1_f32(pC), vdup_n_f32(0.f));
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c, _beta);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vcombine_f32(vld1_f32(pC), vdup_n_f32(0.f));
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c, _beta);
                    }
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0f[0] = vgetq_lane_f32(_f0, 0);
                        p0f[out_hstep] = vgetq_lane_f32(_f0, 1);
                        p0f += out_hstep * 2;
                    }
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        vst1_f32(p0f, vget_low_f32(_f0));
                        p0f += 2;
                    }
                }
            }
            else
            {
                uint16x4_t _bf0 = float2bfloat(_f0);
                uint32x2_t _bf0_32 = vreinterpret_u32_u16(_bf0);

                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        vst1_lane_u16(p0, _bf0, 0);
                        vst1_lane_u16(p0 + out_hstep, _bf0, 1);
                        p0 += out_hstep * 2;
                    }
                }
                else
                {
                    vst1_lane_u32((uint32_t*)p0, _bf0_32, 0);
                    p0 += 2;
                }
            }
            pp += 2;
#else  // __ARM_NEON
            float sum0 = pp[0];
            float sum1 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    float c0 = pC[0] * beta;
                    sum0 += c0;
                    sum1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float c0 = pC[0] * beta;
                    sum0 += c0;
                    sum1 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                    pC += 2;
                }
            }

            sum0 *= alpha;
            sum1 *= alpha;

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0f[0] = sum0;
                        p0f[out_hstep] = sum1;
                        p0f += out_hstep * 2;
                    }
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        p0f[0] = sum0;
                        p0f[1] = sum1;
                        p0f += 2;
                    }
                }
            }
            else
            {
                unsigned short bf0 = float32_to_bfloat16(sum0);
                unsigned short bf1 = float32_to_bfloat16(sum1);
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0[0] = bf0;
                        p0[out_hstep] = bf1;
                        p0 += out_hstep * 2;
                    }
                }
                else
                {
                    p0[0] = bf0;
                    p0[1] = bf1;
                    p0 += 2;
                }
            }
            pp += 2;
#endif // __ARM_NEON
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum = pp[0];

            if (pC)
            {
                if (broadcast_type_C == 0)
                    sum += pC[0] * beta;
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    sum += pC[0] * beta;
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    sum += pC[0] * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    sum += pC[0] * beta;
                    pC += 1;
                }
            }

            sum *= alpha;

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0f[0] = sum;
                        p0f += out_hstep;
                    }
                }
                else
                {
                    if (out_elempack == 1)
                    {
                        p0f[0] = sum;
                        p0f += 1;
                    }
                }
            }
            else
            {
                unsigned short bf = float32_to_bfloat16(sum);
                if (output_transpose)
                {
                    // if (out_elempack == 1)
                    {
                        p0[0] = bf;
                        p0 += out_hstep;
                    }
                }
                else
                {
                    p0[0] = bf;
                    p0 += 1;
                }
            }
            pp += 1;
        }
    }
}

static void gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        gemm_transB_packed_tile_bf16s_bf16(AT_tile, BT_tile, topT_tile, max_ii, max_jj, k, max_kk);
        return;
    }
#endif

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;
            float32x4_t _sum80;
            float32x4_t _sum81;
            float32x4_t _sum90;
            float32x4_t _sum91;
            float32x4_t _suma0;
            float32x4_t _suma1;
            float32x4_t _sumb0;
            float32x4_t _sumb1;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);
                _sum40 = vdupq_n_f32(0.f);
                _sum41 = vdupq_n_f32(0.f);
                _sum50 = vdupq_n_f32(0.f);
                _sum51 = vdupq_n_f32(0.f);
                _sum60 = vdupq_n_f32(0.f);
                _sum61 = vdupq_n_f32(0.f);
                _sum70 = vdupq_n_f32(0.f);
                _sum71 = vdupq_n_f32(0.f);
                _sum80 = vdupq_n_f32(0.f);
                _sum81 = vdupq_n_f32(0.f);
                _sum90 = vdupq_n_f32(0.f);
                _sum91 = vdupq_n_f32(0.f);
                _suma0 = vdupq_n_f32(0.f);
                _suma1 = vdupq_n_f32(0.f);
                _sumb0 = vdupq_n_f32(0.f);
                _sumb1 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4);
                _sum10 = vld1q_f32(outptr + 8);
                _sum11 = vld1q_f32(outptr + 12);
                _sum20 = vld1q_f32(outptr + 16);
                _sum21 = vld1q_f32(outptr + 20);
                _sum30 = vld1q_f32(outptr + 24);
                _sum31 = vld1q_f32(outptr + 28);
                _sum40 = vld1q_f32(outptr + 32);
                _sum41 = vld1q_f32(outptr + 36);
                _sum50 = vld1q_f32(outptr + 40);
                _sum51 = vld1q_f32(outptr + 44);
                _sum60 = vld1q_f32(outptr + 48);
                _sum61 = vld1q_f32(outptr + 52);
                _sum70 = vld1q_f32(outptr + 56);
                _sum71 = vld1q_f32(outptr + 60);
                _sum80 = vld1q_f32(outptr + 64);
                _sum81 = vld1q_f32(outptr + 68);
                _sum90 = vld1q_f32(outptr + 72);
                _sum91 = vld1q_f32(outptr + 76);
                _suma0 = vld1q_f32(outptr + 80);
                _suma1 = vld1q_f32(outptr + 84);
                _sumb0 = vld1q_f32(outptr + 88);
                _sumb1 = vld1q_f32(outptr + 92);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint16x8_t _pA2 = vld1q_u16(pA + 16);
                uint16x8_t _pA3 = vld1q_u16(pA + 24);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                uint16x8_t _pB4 = vld1q_u16(pB + 32);
                uint16x8_t _pB5 = vld1q_u16(pB + 40);
                _sum00 = vbfmmlaq_f32(_sum00, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB0);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB0);
                _sum10 = vbfmmlaq_f32(_sum10, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB0);
                _sum11 = vbfmmlaq_f32(_sum11, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB0);
                _sum20 = vbfmmlaq_f32(_sum20, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB1);
                _sum21 = vbfmmlaq_f32(_sum21, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB1);
                _sum30 = vbfmmlaq_f32(_sum30, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB1);
                _sum31 = vbfmmlaq_f32(_sum31, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB1);
                _sum40 = vbfmmlaq_f32(_sum40, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB2);
                _sum41 = vbfmmlaq_f32(_sum41, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB2);
                _sum50 = vbfmmlaq_f32(_sum50, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB2);
                _sum51 = vbfmmlaq_f32(_sum51, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB2);
                _sum60 = vbfmmlaq_f32(_sum60, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB3);
                _sum61 = vbfmmlaq_f32(_sum61, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB3);
                _sum70 = vbfmmlaq_f32(_sum70, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB3);
                _sum71 = vbfmmlaq_f32(_sum71, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB3);
                _sum80 = vbfmmlaq_f32(_sum80, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB4);
                _sum81 = vbfmmlaq_f32(_sum81, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB4);
                _sum90 = vbfmmlaq_f32(_sum90, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB4);
                _sum91 = vbfmmlaq_f32(_sum91, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB4);
                _suma0 = vbfmmlaq_f32(_suma0, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB5);
                _suma1 = vbfmmlaq_f32(_suma1, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB5);
                _sumb0 = vbfmmlaq_f32(_sumb0, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB5);
                _sumb1 = vbfmmlaq_f32(_sumb1, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB5);

                pA += 32;
                pB += 48;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pA1 = vld1_u16(pA + 4);
                uint16x4_t _pA2 = vld1_u16(pA + 8);
                uint16x4_t _pA3 = vld1_u16(pA + 12);
                uint32x2x2_t _pA0_32x2 = vzip_u32(vreinterpret_u32_u16(_pA0), vreinterpret_u32_u16(_pA0));
                uint32x2x2_t _pA1_32x2 = vzip_u32(vreinterpret_u32_u16(_pA1), vreinterpret_u32_u16(_pA1));
                uint32x2x2_t _pA2_32x2 = vzip_u32(vreinterpret_u32_u16(_pA2), vreinterpret_u32_u16(_pA2));
                uint32x2x2_t _pA3_32x2 = vzip_u32(vreinterpret_u32_u16(_pA3), vreinterpret_u32_u16(_pA3));
                uint16x8_t _pA00 = vreinterpretq_u16_u32(vcombine_u32(_pA0_32x2.val[0], _pA0_32x2.val[1]));
                uint16x8_t _pA11 = vreinterpretq_u16_u32(vcombine_u32(_pA1_32x2.val[0], _pA1_32x2.val[1]));
                uint16x8_t _pA22 = vreinterpretq_u16_u32(vcombine_u32(_pA2_32x2.val[0], _pA2_32x2.val[1]));
                uint16x8_t _pA33 = vreinterpretq_u16_u32(vcombine_u32(_pA3_32x2.val[0], _pA3_32x2.val[1]));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x4_t _pB2 = vld1_u16(pB + 8);
                uint16x4_t _pB3 = vld1_u16(pB + 12);
                uint16x4_t _pB4 = vld1_u16(pB + 16);
                uint16x4_t _pB5 = vld1_u16(pB + 20);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                uint16x8_t _pB22 = vcombine_u16(_pB2, _pB2);
                uint16x8_t _pB33 = vcombine_u16(_pB3, _pB3);
                uint16x8_t _pB44 = vcombine_u16(_pB4, _pB4);
                uint16x8_t _pB55 = vcombine_u16(_pB5, _pB5);
                _sum00 = vbfdotq_f32(_sum00, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB00);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB00);
                _sum10 = vbfdotq_f32(_sum10, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB00);
                _sum11 = vbfdotq_f32(_sum11, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB00);
                _sum20 = vbfdotq_f32(_sum20, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB11);
                _sum21 = vbfdotq_f32(_sum21, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB11);
                _sum30 = vbfdotq_f32(_sum30, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB11);
                _sum31 = vbfdotq_f32(_sum31, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB11);
                _sum40 = vbfdotq_f32(_sum40, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB22);
                _sum41 = vbfdotq_f32(_sum41, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB22);
                _sum50 = vbfdotq_f32(_sum50, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB22);
                _sum51 = vbfdotq_f32(_sum51, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB22);
                _sum60 = vbfdotq_f32(_sum60, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB33);
                _sum61 = vbfdotq_f32(_sum61, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB33);
                _sum70 = vbfdotq_f32(_sum70, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB33);
                _sum71 = vbfdotq_f32(_sum71, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB33);
                _sum80 = vbfdotq_f32(_sum80, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB44);
                _sum81 = vbfdotq_f32(_sum81, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB44);
                _sum90 = vbfdotq_f32(_sum90, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB44);
                _sum91 = vbfdotq_f32(_sum91, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB44);
                _suma0 = vbfdotq_f32(_suma0, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB55);
                _suma1 = vbfdotq_f32(_suma1, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB55);
                _sumb0 = vbfdotq_f32(_sumb0, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB55);
                _sumb1 = vbfdotq_f32(_sumb1, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB55);

                pA += 16;
                pB += 24;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x4x2_t _pA0123 = vzip_u16(vget_low_u16(_pA), vget_low_u16(_pA));
                uint16x4x2_t _pA4567 = vzip_u16(vget_high_u16(_pA), vget_high_u16(_pA));
                float32x4_t _pA00 = bfloat2float(_pA0123.val[0]);
                float32x4_t _pA11 = bfloat2float(_pA0123.val[1]);
                float32x4_t _pA22 = bfloat2float(_pA4567.val[0]);
                float32x4_t _pA33 = bfloat2float(_pA4567.val[1]);
                uint16x4_t _pB0123 = vld1_u16(pB);
                uint16x4_t _pB4567 = vld1_u16(pB + 4);
                uint16x4_t _pB89ab = vld1_u16(pB + 8);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                uint32x2x2_t _pB4567_32x2 = vzip_u32(vreinterpret_u32_u16(_pB4567), vreinterpret_u32_u16(_pB4567));
                uint32x2x2_t _pB89ab_32x2 = vzip_u32(vreinterpret_u32_u16(_pB89ab), vreinterpret_u32_u16(_pB89ab));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                float32x4_t _pB2 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[0]));
                float32x4_t _pB3 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[1]));
                float32x4_t _pB4 = bfloat2float(vreinterpret_u16_u32(_pB89ab_32x2.val[0]));
                float32x4_t _pB5 = bfloat2float(vreinterpret_u16_u32(_pB89ab_32x2.val[1]));
                _sum00 = vfmaq_f32(_sum00, _pA00, _pB0);
                _sum01 = vfmaq_f32(_sum01, _pA11, _pB0);
                _sum10 = vfmaq_f32(_sum10, _pA22, _pB0);
                _sum11 = vfmaq_f32(_sum11, _pA33, _pB0);
                _sum20 = vfmaq_f32(_sum20, _pA00, _pB1);
                _sum21 = vfmaq_f32(_sum21, _pA11, _pB1);
                _sum30 = vfmaq_f32(_sum30, _pA22, _pB1);
                _sum31 = vfmaq_f32(_sum31, _pA33, _pB1);
                _sum40 = vfmaq_f32(_sum40, _pA00, _pB2);
                _sum41 = vfmaq_f32(_sum41, _pA11, _pB2);
                _sum50 = vfmaq_f32(_sum50, _pA22, _pB2);
                _sum51 = vfmaq_f32(_sum51, _pA33, _pB2);
                _sum60 = vfmaq_f32(_sum60, _pA00, _pB3);
                _sum61 = vfmaq_f32(_sum61, _pA11, _pB3);
                _sum70 = vfmaq_f32(_sum70, _pA22, _pB3);
                _sum71 = vfmaq_f32(_sum71, _pA33, _pB3);
                _sum80 = vfmaq_f32(_sum80, _pA00, _pB4);
                _sum81 = vfmaq_f32(_sum81, _pA11, _pB4);
                _sum90 = vfmaq_f32(_sum90, _pA22, _pB4);
                _sum91 = vfmaq_f32(_sum91, _pA33, _pB4);
                _suma0 = vfmaq_f32(_suma0, _pA00, _pB5);
                _suma1 = vfmaq_f32(_suma1, _pA11, _pB5);
                _sumb0 = vfmaq_f32(_sumb0, _pA22, _pB5);
                _sumb1 = vfmaq_f32(_sumb1, _pA33, _pB5);

                pA += 8;
                pB += 12;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = bfloat2float(vget_low_u16(_pA));
                float32x4_t _pA1 = bfloat2float(vget_high_u16(_pA));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));
                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum00);
            vst1q_f32(outptr + 4, _sum01);
            vst1q_f32(outptr + 8, _sum10);
            vst1q_f32(outptr + 12, _sum11);
            vst1q_f32(outptr + 16, _sum20);
            vst1q_f32(outptr + 20, _sum21);
            vst1q_f32(outptr + 24, _sum30);
            vst1q_f32(outptr + 28, _sum31);
            vst1q_f32(outptr + 32, _sum40);
            vst1q_f32(outptr + 36, _sum41);
            vst1q_f32(outptr + 40, _sum50);
            vst1q_f32(outptr + 44, _sum51);
            vst1q_f32(outptr + 48, _sum60);
            vst1q_f32(outptr + 52, _sum61);
            vst1q_f32(outptr + 56, _sum70);
            vst1q_f32(outptr + 60, _sum71);
            vst1q_f32(outptr + 64, _sum80);
            vst1q_f32(outptr + 68, _sum81);
            vst1q_f32(outptr + 72, _sum90);
            vst1q_f32(outptr + 76, _sum91);
            vst1q_f32(outptr + 80, _suma0);
            vst1q_f32(outptr + 84, _suma1);
            vst1q_f32(outptr + 88, _sumb0);
            vst1q_f32(outptr + 92, _sumb1);

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);
                _sum40 = vdupq_n_f32(0.f);
                _sum41 = vdupq_n_f32(0.f);
                _sum50 = vdupq_n_f32(0.f);
                _sum51 = vdupq_n_f32(0.f);
                _sum60 = vdupq_n_f32(0.f);
                _sum61 = vdupq_n_f32(0.f);
                _sum70 = vdupq_n_f32(0.f);
                _sum71 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4);
                _sum10 = vld1q_f32(outptr + 8);
                _sum11 = vld1q_f32(outptr + 12);
                _sum20 = vld1q_f32(outptr + 16);
                _sum21 = vld1q_f32(outptr + 20);
                _sum30 = vld1q_f32(outptr + 24);
                _sum31 = vld1q_f32(outptr + 28);
                _sum40 = vld1q_f32(outptr + 32);
                _sum41 = vld1q_f32(outptr + 36);
                _sum50 = vld1q_f32(outptr + 40);
                _sum51 = vld1q_f32(outptr + 44);
                _sum60 = vld1q_f32(outptr + 48);
                _sum61 = vld1q_f32(outptr + 52);
                _sum70 = vld1q_f32(outptr + 56);
                _sum71 = vld1q_f32(outptr + 60);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint16x8_t _pA2 = vld1q_u16(pA + 16);
                uint16x8_t _pA3 = vld1q_u16(pA + 24);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                _sum00 = vbfmmlaq_f32(_sum00, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB0);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB0);
                _sum10 = vbfmmlaq_f32(_sum10, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB0);
                _sum11 = vbfmmlaq_f32(_sum11, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB0);
                _sum20 = vbfmmlaq_f32(_sum20, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB1);
                _sum21 = vbfmmlaq_f32(_sum21, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB1);
                _sum30 = vbfmmlaq_f32(_sum30, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB1);
                _sum31 = vbfmmlaq_f32(_sum31, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB1);
                _sum40 = vbfmmlaq_f32(_sum40, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB2);
                _sum41 = vbfmmlaq_f32(_sum41, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB2);
                _sum50 = vbfmmlaq_f32(_sum50, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB2);
                _sum51 = vbfmmlaq_f32(_sum51, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB2);
                _sum60 = vbfmmlaq_f32(_sum60, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB3);
                _sum61 = vbfmmlaq_f32(_sum61, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB3);
                _sum70 = vbfmmlaq_f32(_sum70, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB3);
                _sum71 = vbfmmlaq_f32(_sum71, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB3);

                pA += 32;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pA1 = vld1_u16(pA + 4);
                uint16x4_t _pA2 = vld1_u16(pA + 8);
                uint16x4_t _pA3 = vld1_u16(pA + 12);
                uint32x2x2_t _pA0_32x2 = vzip_u32(vreinterpret_u32_u16(_pA0), vreinterpret_u32_u16(_pA0));
                uint32x2x2_t _pA1_32x2 = vzip_u32(vreinterpret_u32_u16(_pA1), vreinterpret_u32_u16(_pA1));
                uint32x2x2_t _pA2_32x2 = vzip_u32(vreinterpret_u32_u16(_pA2), vreinterpret_u32_u16(_pA2));
                uint32x2x2_t _pA3_32x2 = vzip_u32(vreinterpret_u32_u16(_pA3), vreinterpret_u32_u16(_pA3));
                uint16x8_t _pA00 = vreinterpretq_u16_u32(vcombine_u32(_pA0_32x2.val[0], _pA0_32x2.val[1]));
                uint16x8_t _pA11 = vreinterpretq_u16_u32(vcombine_u32(_pA1_32x2.val[0], _pA1_32x2.val[1]));
                uint16x8_t _pA22 = vreinterpretq_u16_u32(vcombine_u32(_pA2_32x2.val[0], _pA2_32x2.val[1]));
                uint16x8_t _pA33 = vreinterpretq_u16_u32(vcombine_u32(_pA3_32x2.val[0], _pA3_32x2.val[1]));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x4_t _pB2 = vld1_u16(pB + 8);
                uint16x4_t _pB3 = vld1_u16(pB + 12);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                uint16x8_t _pB22 = vcombine_u16(_pB2, _pB2);
                uint16x8_t _pB33 = vcombine_u16(_pB3, _pB3);
                _sum00 = vbfdotq_f32(_sum00, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB00);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB00);
                _sum10 = vbfdotq_f32(_sum10, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB00);
                _sum11 = vbfdotq_f32(_sum11, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB00);
                _sum20 = vbfdotq_f32(_sum20, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB11);
                _sum21 = vbfdotq_f32(_sum21, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB11);
                _sum30 = vbfdotq_f32(_sum30, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB11);
                _sum31 = vbfdotq_f32(_sum31, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB11);
                _sum40 = vbfdotq_f32(_sum40, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB22);
                _sum41 = vbfdotq_f32(_sum41, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB22);
                _sum50 = vbfdotq_f32(_sum50, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB22);
                _sum51 = vbfdotq_f32(_sum51, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB22);
                _sum60 = vbfdotq_f32(_sum60, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB33);
                _sum61 = vbfdotq_f32(_sum61, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB33);
                _sum70 = vbfdotq_f32(_sum70, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB33);
                _sum71 = vbfdotq_f32(_sum71, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB33);

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x4x2_t _pA0123 = vzip_u16(vget_low_u16(_pA), vget_low_u16(_pA));
                uint16x4x2_t _pA4567 = vzip_u16(vget_high_u16(_pA), vget_high_u16(_pA));
                float32x4_t _pA00 = bfloat2float(_pA0123.val[0]);
                float32x4_t _pA11 = bfloat2float(_pA0123.val[1]);
                float32x4_t _pA22 = bfloat2float(_pA4567.val[0]);
                float32x4_t _pA33 = bfloat2float(_pA4567.val[1]);
                uint16x4_t _pB0123 = vld1_u16(pB);
                uint16x4_t _pB4567 = vld1_u16(pB + 4);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                uint32x2x2_t _pB4567_32x2 = vzip_u32(vreinterpret_u32_u16(_pB4567), vreinterpret_u32_u16(_pB4567));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                float32x4_t _pB2 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[0]));
                float32x4_t _pB3 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[1]));
                _sum00 = vfmaq_f32(_sum00, _pA00, _pB0);
                _sum01 = vfmaq_f32(_sum01, _pA11, _pB0);
                _sum10 = vfmaq_f32(_sum10, _pA22, _pB0);
                _sum11 = vfmaq_f32(_sum11, _pA33, _pB0);
                _sum20 = vfmaq_f32(_sum20, _pA00, _pB1);
                _sum21 = vfmaq_f32(_sum21, _pA11, _pB1);
                _sum30 = vfmaq_f32(_sum30, _pA22, _pB1);
                _sum31 = vfmaq_f32(_sum31, _pA33, _pB1);
                _sum40 = vfmaq_f32(_sum40, _pA00, _pB2);
                _sum41 = vfmaq_f32(_sum41, _pA11, _pB2);
                _sum50 = vfmaq_f32(_sum50, _pA22, _pB2);
                _sum51 = vfmaq_f32(_sum51, _pA33, _pB2);
                _sum60 = vfmaq_f32(_sum60, _pA00, _pB3);
                _sum61 = vfmaq_f32(_sum61, _pA11, _pB3);
                _sum70 = vfmaq_f32(_sum70, _pA22, _pB3);
                _sum71 = vfmaq_f32(_sum71, _pA33, _pB3);

                pA += 8;
                pB += 8;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = bfloat2float(vget_low_u16(_pA));
                float32x4_t _pA1 = bfloat2float(vget_high_u16(_pA));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);

                pA += 8;
                pB += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum00);
            vst1q_f32(outptr + 4, _sum01);
            vst1q_f32(outptr + 8, _sum10);
            vst1q_f32(outptr + 12, _sum11);
            vst1q_f32(outptr + 16, _sum20);
            vst1q_f32(outptr + 20, _sum21);
            vst1q_f32(outptr + 24, _sum30);
            vst1q_f32(outptr + 28, _sum31);
            vst1q_f32(outptr + 32, _sum40);
            vst1q_f32(outptr + 36, _sum41);
            vst1q_f32(outptr + 40, _sum50);
            vst1q_f32(outptr + 44, _sum51);
            vst1q_f32(outptr + 48, _sum60);
            vst1q_f32(outptr + 52, _sum61);
            vst1q_f32(outptr + 56, _sum70);
            vst1q_f32(outptr + 60, _sum71);

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4);
                _sum10 = vld1q_f32(outptr + 8);
                _sum11 = vld1q_f32(outptr + 12);
                _sum20 = vld1q_f32(outptr + 16);
                _sum21 = vld1q_f32(outptr + 20);
                _sum30 = vld1q_f32(outptr + 24);
                _sum31 = vld1q_f32(outptr + 28);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint16x8_t _pA2 = vld1q_u16(pA + 16);
                uint16x8_t _pA3 = vld1q_u16(pA + 24);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                _sum00 = vbfmmlaq_f32(_sum00, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB0);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB0);
                _sum10 = vbfmmlaq_f32(_sum10, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB0);
                _sum11 = vbfmmlaq_f32(_sum11, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB0);
                _sum20 = vbfmmlaq_f32(_sum20, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB1);
                _sum21 = vbfmmlaq_f32(_sum21, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB1);
                _sum30 = vbfmmlaq_f32(_sum30, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB1);
                _sum31 = vbfmmlaq_f32(_sum31, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB1);

                pA += 32;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pA1 = vld1_u16(pA + 4);
                uint16x4_t _pA2 = vld1_u16(pA + 8);
                uint16x4_t _pA3 = vld1_u16(pA + 12);
                uint32x2x2_t _pA0_32x2 = vzip_u32(vreinterpret_u32_u16(_pA0), vreinterpret_u32_u16(_pA0));
                uint32x2x2_t _pA1_32x2 = vzip_u32(vreinterpret_u32_u16(_pA1), vreinterpret_u32_u16(_pA1));
                uint32x2x2_t _pA2_32x2 = vzip_u32(vreinterpret_u32_u16(_pA2), vreinterpret_u32_u16(_pA2));
                uint32x2x2_t _pA3_32x2 = vzip_u32(vreinterpret_u32_u16(_pA3), vreinterpret_u32_u16(_pA3));
                uint16x8_t _pA00 = vreinterpretq_u16_u32(vcombine_u32(_pA0_32x2.val[0], _pA0_32x2.val[1]));
                uint16x8_t _pA11 = vreinterpretq_u16_u32(vcombine_u32(_pA1_32x2.val[0], _pA1_32x2.val[1]));
                uint16x8_t _pA22 = vreinterpretq_u16_u32(vcombine_u32(_pA2_32x2.val[0], _pA2_32x2.val[1]));
                uint16x8_t _pA33 = vreinterpretq_u16_u32(vcombine_u32(_pA3_32x2.val[0], _pA3_32x2.val[1]));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                _sum00 = vbfdotq_f32(_sum00, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB00);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB00);
                _sum10 = vbfdotq_f32(_sum10, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB00);
                _sum11 = vbfdotq_f32(_sum11, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB00);
                _sum20 = vbfdotq_f32(_sum20, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB11);
                _sum21 = vbfdotq_f32(_sum21, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB11);
                _sum30 = vbfdotq_f32(_sum30, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB11);
                _sum31 = vbfdotq_f32(_sum31, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB11);

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x4x2_t _pA0123 = vzip_u16(vget_low_u16(_pA), vget_low_u16(_pA));
                uint16x4x2_t _pA4567 = vzip_u16(vget_high_u16(_pA), vget_high_u16(_pA));
                float32x4_t _pA00 = bfloat2float(_pA0123.val[0]);
                float32x4_t _pA11 = bfloat2float(_pA0123.val[1]);
                float32x4_t _pA22 = bfloat2float(_pA4567.val[0]);
                float32x4_t _pA33 = bfloat2float(_pA4567.val[1]);
                uint16x4_t _pB0123 = vld1_u16(pB);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                _sum00 = vfmaq_f32(_sum00, _pA00, _pB0);
                _sum01 = vfmaq_f32(_sum01, _pA11, _pB0);
                _sum10 = vfmaq_f32(_sum10, _pA22, _pB0);
                _sum11 = vfmaq_f32(_sum11, _pA33, _pB0);
                _sum20 = vfmaq_f32(_sum20, _pA00, _pB1);
                _sum21 = vfmaq_f32(_sum21, _pA11, _pB1);
                _sum30 = vfmaq_f32(_sum30, _pA22, _pB1);
                _sum31 = vfmaq_f32(_sum31, _pA33, _pB1);

                pA += 8;
                pB += 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = bfloat2float(vget_low_u16(_pA));
                float32x4_t _pA1 = bfloat2float(vget_high_u16(_pA));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);

                pA += 8;
                pB += 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum00);
            vst1q_f32(outptr + 4, _sum01);
            vst1q_f32(outptr + 8, _sum10);
            vst1q_f32(outptr + 12, _sum11);
            vst1q_f32(outptr + 16, _sum20);
            vst1q_f32(outptr + 20, _sum21);
            vst1q_f32(outptr + 24, _sum30);
            vst1q_f32(outptr + 28, _sum31);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4);
                _sum10 = vld1q_f32(outptr + 8);
                _sum11 = vld1q_f32(outptr + 12);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint16x8_t _pA2 = vld1q_u16(pA + 16);
                uint16x8_t _pA3 = vld1q_u16(pA + 24);
                uint16x8_t _pB = vld1q_u16(pB);
                _sum00 = vbfmmlaq_f32(_sum00, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB);
                _sum10 = vbfmmlaq_f32(_sum10, (bfloat16x8_t)_pA2, (bfloat16x8_t)_pB);
                _sum11 = vbfmmlaq_f32(_sum11, (bfloat16x8_t)_pA3, (bfloat16x8_t)_pB);

                pA += 32;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pA1 = vld1_u16(pA + 4);
                uint16x4_t _pA2 = vld1_u16(pA + 8);
                uint16x4_t _pA3 = vld1_u16(pA + 12);
                uint32x2x2_t _pA0_32x2 = vzip_u32(vreinterpret_u32_u16(_pA0), vreinterpret_u32_u16(_pA0));
                uint32x2x2_t _pA1_32x2 = vzip_u32(vreinterpret_u32_u16(_pA1), vreinterpret_u32_u16(_pA1));
                uint32x2x2_t _pA2_32x2 = vzip_u32(vreinterpret_u32_u16(_pA2), vreinterpret_u32_u16(_pA2));
                uint32x2x2_t _pA3_32x2 = vzip_u32(vreinterpret_u32_u16(_pA3), vreinterpret_u32_u16(_pA3));
                uint16x8_t _pA00 = vreinterpretq_u16_u32(vcombine_u32(_pA0_32x2.val[0], _pA0_32x2.val[1]));
                uint16x8_t _pA11 = vreinterpretq_u16_u32(vcombine_u32(_pA1_32x2.val[0], _pA1_32x2.val[1]));
                uint16x8_t _pA22 = vreinterpretq_u16_u32(vcombine_u32(_pA2_32x2.val[0], _pA2_32x2.val[1]));
                uint16x8_t _pA33 = vreinterpretq_u16_u32(vcombine_u32(_pA3_32x2.val[0], _pA3_32x2.val[1]));
                uint16x4_t _pB = vld1_u16(pB);
                uint16x8_t _pB01 = vcombine_u16(_pB, _pB);
                _sum00 = vbfdotq_f32(_sum00, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB01);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB01);
                _sum10 = vbfdotq_f32(_sum10, (bfloat16x8_t)_pA22, (bfloat16x8_t)_pB01);
                _sum11 = vbfdotq_f32(_sum11, (bfloat16x8_t)_pA33, (bfloat16x8_t)_pB01);

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x4x2_t _pA0123 = vzip_u16(vget_low_u16(_pA), vget_low_u16(_pA));
                uint16x4x2_t _pA4567 = vzip_u16(vget_high_u16(_pA), vget_high_u16(_pA));
                float32x4_t _pA00 = bfloat2float(_pA0123.val[0]);
                float32x4_t _pA11 = bfloat2float(_pA0123.val[1]);
                float32x4_t _pA22 = bfloat2float(_pA4567.val[0]);
                float32x4_t _pA33 = bfloat2float(_pA4567.val[1]);
                uint16x4_t _pB01 = vld1_dup_u16(pB);
                _pB01 = vld1_lane_u16(pB + 1, _pB01, 1);
                _pB01 = vld1_lane_u16(pB + 1, _pB01, 3);
                float32x4_t _pB = bfloat2float(_pB01);
                _sum00 = vfmaq_f32(_sum00, _pA00, _pB);
                _sum01 = vfmaq_f32(_sum01, _pA11, _pB);
                _sum10 = vfmaq_f32(_sum10, _pA22, _pB);
                _sum11 = vfmaq_f32(_sum11, _pA33, _pB);

                pA += 8;
                pB += 2;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = bfloat2float(vget_low_u16(_pA));
                float32x4_t _pA1 = bfloat2float(vget_high_u16(_pA));

                float32x4_t _pB0 = bfloat2float(vdup_n_u16(pB[0]));
                float32x4_t _pB1 = bfloat2float(vdup_n_u16(pB[1]));
                _sum00 = vfmaq_f32(_sum00, _pA0, _pB0);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB0);
                _sum10 = vfmaq_f32(_sum10, _pA0, _pB1);
                _sum11 = vfmaq_f32(_sum11, _pA1, _pB1);

                pA += 8;
                pB += 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum00);
            vst1q_f32(outptr + 4, _sum01);
            vst1q_f32(outptr + 8, _sum10);
            vst1q_f32(outptr + 12, _sum11);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _s01 = vdupq_n_f32(0.f);
            float32x4_t _s23 = vdupq_n_f32(0.f);
            float32x4_t _s45 = vdupq_n_f32(0.f);
            float32x4_t _s67 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA01 = vld1q_u16(pA);
                uint16x8_t _pA23 = vld1q_u16(pA + 8);
                uint16x8_t _pA45 = vld1q_u16(pA + 16);
                uint16x8_t _pA67 = vld1q_u16(pA + 24);
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x8_t _pB = vcombine_u16(_pB0, _pB0);

                _s01 = vbfdotq_f32(_s01, (bfloat16x8_t)_pA01, (bfloat16x8_t)_pB);
                _s23 = vbfdotq_f32(_s23, (bfloat16x8_t)_pA23, (bfloat16x8_t)_pB);
                _s45 = vbfdotq_f32(_s45, (bfloat16x8_t)_pA45, (bfloat16x8_t)_pB);
                _s67 = vbfdotq_f32(_s67, (bfloat16x8_t)_pA67, (bfloat16x8_t)_pB);

                pA += 32;
                pB += 4;
            }
            float32x2_t _s01p = vpadd_f32(vget_low_f32(_s01), vget_high_f32(_s01));
            float32x2_t _s23p = vpadd_f32(vget_low_f32(_s23), vget_high_f32(_s23));
            float32x2_t _s45p = vpadd_f32(vget_low_f32(_s45), vget_high_f32(_s45));
            float32x2_t _s67p = vpadd_f32(vget_low_f32(_s67), vget_high_f32(_s67));
            _sum00 = vaddq_f32(_sum00, vcombine_f32(_s01p, _s23p));
            _sum01 = vaddq_f32(_sum01, vcombine_f32(_s45p, _s67p));
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint32x2_t _pB01 = vld1_dup_u32((const uint32_t*)pB);
                uint16x8_t _pB = vreinterpretq_u16_u32(vcombine_u32(_pB01, _pB01));
                _sum00 = vbfdotq_f32(_sum00, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB);

                pA += 16;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = bfloat2float(vdup_n_u16(pB[0]));
                _sum00 = vfmaq_f32(_sum00, bfloat2float(vld1_u16(pA)), _pB);
                _sum01 = vfmaq_f32(_sum01, bfloat2float(vld1_u16(pA + 4)), _pB);

                pA += 8;
                pB += 1;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = bfloat2float(vget_low_u16(_pA));
                float32x4_t _pA1 = bfloat2float(vget_high_u16(_pA));

                float32x4_t _pB = bfloat2float(vld1_dup_u16(pB));
                _sum00 = vfmaq_f32(_sum00, _pA0, _pB);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB);

                pA += 8;
                pB += 1;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum00);
            vst1q_f32(outptr + 4, _sum01);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;
            float32x4_t _sum8;
            float32x4_t _sum9;
            float32x4_t _suma;
            float32x4_t _sumb;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
                _sum4 = vdupq_n_f32(0.f);
                _sum5 = vdupq_n_f32(0.f);
                _sum6 = vdupq_n_f32(0.f);
                _sum7 = vdupq_n_f32(0.f);
                _sum8 = vdupq_n_f32(0.f);
                _sum9 = vdupq_n_f32(0.f);
                _suma = vdupq_n_f32(0.f);
                _sumb = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
                _sum3 = vld1q_f32(outptr + 12);
                _sum4 = vld1q_f32(outptr + 16);
                _sum5 = vld1q_f32(outptr + 20);
                _sum6 = vld1q_f32(outptr + 24);
                _sum7 = vld1q_f32(outptr + 28);
                _sum8 = vld1q_f32(outptr + 32);
                _sum9 = vld1q_f32(outptr + 36);
                _suma = vld1q_f32(outptr + 40);
                _sumb = vld1q_f32(outptr + 44);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                uint16x8_t _pB4 = vld1q_u16(pB + 32);
                uint16x8_t _pB5 = vld1q_u16(pB + 40);
                _sum0 = vbfmmlaq_f32(_sum0, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB0);
                _sum1 = vbfmmlaq_f32(_sum1, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB0);
                _sum2 = vbfmmlaq_f32(_sum2, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB1);
                _sum3 = vbfmmlaq_f32(_sum3, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB1);
                _sum4 = vbfmmlaq_f32(_sum4, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB2);
                _sum5 = vbfmmlaq_f32(_sum5, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB2);
                _sum6 = vbfmmlaq_f32(_sum6, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB3);
                _sum7 = vbfmmlaq_f32(_sum7, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB3);
                _sum8 = vbfmmlaq_f32(_sum8, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB4);
                _sum9 = vbfmmlaq_f32(_sum9, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB4);
                _suma = vbfmmlaq_f32(_suma, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB5);
                _sumb = vbfmmlaq_f32(_sumb, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB5);

                pA += 16;
                pB += 48;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pA1 = vld1_u16(pA + 4);
                uint32x2x2_t _pA0_32x2 = vzip_u32(vreinterpret_u32_u16(_pA0), vreinterpret_u32_u16(_pA0));
                uint32x2x2_t _pA1_32x2 = vzip_u32(vreinterpret_u32_u16(_pA1), vreinterpret_u32_u16(_pA1));
                uint16x8_t _pA00 = vreinterpretq_u16_u32(vcombine_u32(_pA0_32x2.val[0], _pA0_32x2.val[1]));
                uint16x8_t _pA11 = vreinterpretq_u16_u32(vcombine_u32(_pA1_32x2.val[0], _pA1_32x2.val[1]));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x4_t _pB2 = vld1_u16(pB + 8);
                uint16x4_t _pB3 = vld1_u16(pB + 12);
                uint16x4_t _pB4 = vld1_u16(pB + 16);
                uint16x4_t _pB5 = vld1_u16(pB + 20);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                uint16x8_t _pB22 = vcombine_u16(_pB2, _pB2);
                uint16x8_t _pB33 = vcombine_u16(_pB3, _pB3);
                uint16x8_t _pB44 = vcombine_u16(_pB4, _pB4);
                uint16x8_t _pB55 = vcombine_u16(_pB5, _pB5);
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB00);
                _sum1 = vbfdotq_f32(_sum1, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB00);
                _sum2 = vbfdotq_f32(_sum2, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB11);
                _sum3 = vbfdotq_f32(_sum3, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB11);
                _sum4 = vbfdotq_f32(_sum4, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB22);
                _sum5 = vbfdotq_f32(_sum5, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB22);
                _sum6 = vbfdotq_f32(_sum6, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB33);
                _sum7 = vbfdotq_f32(_sum7, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB33);
                _sum8 = vbfdotq_f32(_sum8, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB44);
                _sum9 = vbfdotq_f32(_sum9, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB44);
                _suma = vbfdotq_f32(_suma, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB55);
                _sumb = vbfdotq_f32(_sumb, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB55);

                pA += 8;
                pB += 24;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA = vld1_u16(pA);
                uint16x4x2_t _pA01 = vzip_u16(_pA, _pA);
                float32x4_t _pA00 = bfloat2float(_pA01.val[0]);
                float32x4_t _pA11 = bfloat2float(_pA01.val[1]);
                uint16x4_t _pB0123 = vld1_u16(pB);
                uint16x4_t _pB4567 = vld1_u16(pB + 4);
                uint16x4_t _pB89ab = vld1_u16(pB + 8);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                uint32x2x2_t _pB4567_32x2 = vzip_u32(vreinterpret_u32_u16(_pB4567), vreinterpret_u32_u16(_pB4567));
                uint32x2x2_t _pB89ab_32x2 = vzip_u32(vreinterpret_u32_u16(_pB89ab), vreinterpret_u32_u16(_pB89ab));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                float32x4_t _pB2 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[0]));
                float32x4_t _pB3 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[1]));
                float32x4_t _pB4 = bfloat2float(vreinterpret_u16_u32(_pB89ab_32x2.val[0]));
                float32x4_t _pB5 = bfloat2float(vreinterpret_u16_u32(_pB89ab_32x2.val[1]));
                _sum0 = vfmaq_f32(_sum0, _pA00, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA11, _pB0);
                _sum2 = vfmaq_f32(_sum2, _pA00, _pB1);
                _sum3 = vfmaq_f32(_sum3, _pA11, _pB1);
                _sum4 = vfmaq_f32(_sum4, _pA00, _pB2);
                _sum5 = vfmaq_f32(_sum5, _pA11, _pB2);
                _sum6 = vfmaq_f32(_sum6, _pA00, _pB3);
                _sum7 = vfmaq_f32(_sum7, _pA11, _pB3);
                _sum8 = vfmaq_f32(_sum8, _pA00, _pB4);
                _sum9 = vfmaq_f32(_sum9, _pA11, _pB4);
                _suma = vfmaq_f32(_suma, _pA00, _pB5);
                _sumb = vfmaq_f32(_sumb, _pA11, _pB5);

                pA += 4;
                pB += 12;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
                _sum8 = vfmaq_laneq_f32(_sum8, _pA, _pB2, 0);
                _sum9 = vfmaq_laneq_f32(_sum9, _pA, _pB2, 1);
                _suma = vfmaq_laneq_f32(_suma, _pA, _pB2, 2);
                _sumb = vfmaq_laneq_f32(_sumb, _pA, _pB2, 3);

                pA += 4;
                pB += 12;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);
            vst1q_f32(outptr + 8, _sum2);
            vst1q_f32(outptr + 12, _sum3);
            vst1q_f32(outptr + 16, _sum4);
            vst1q_f32(outptr + 20, _sum5);
            vst1q_f32(outptr + 24, _sum6);
            vst1q_f32(outptr + 28, _sum7);
            vst1q_f32(outptr + 32, _sum8);
            vst1q_f32(outptr + 36, _sum9);
            vst1q_f32(outptr + 40, _suma);
            vst1q_f32(outptr + 44, _sumb);

            outptr += 48;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
                _sum4 = vdupq_n_f32(0.f);
                _sum5 = vdupq_n_f32(0.f);
                _sum6 = vdupq_n_f32(0.f);
                _sum7 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
                _sum3 = vld1q_f32(outptr + 12);
                _sum4 = vld1q_f32(outptr + 16);
                _sum5 = vld1q_f32(outptr + 20);
                _sum6 = vld1q_f32(outptr + 24);
                _sum7 = vld1q_f32(outptr + 28);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                _sum0 = vbfmmlaq_f32(_sum0, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB0);
                _sum1 = vbfmmlaq_f32(_sum1, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB0);
                _sum2 = vbfmmlaq_f32(_sum2, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB1);
                _sum3 = vbfmmlaq_f32(_sum3, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB1);
                _sum4 = vbfmmlaq_f32(_sum4, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB2);
                _sum5 = vbfmmlaq_f32(_sum5, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB2);
                _sum6 = vbfmmlaq_f32(_sum6, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB3);
                _sum7 = vbfmmlaq_f32(_sum7, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB3);

                pA += 16;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pA1 = vld1_u16(pA + 4);
                uint32x2x2_t _pA0_32x2 = vzip_u32(vreinterpret_u32_u16(_pA0), vreinterpret_u32_u16(_pA0));
                uint32x2x2_t _pA1_32x2 = vzip_u32(vreinterpret_u32_u16(_pA1), vreinterpret_u32_u16(_pA1));
                uint16x8_t _pA00 = vreinterpretq_u16_u32(vcombine_u32(_pA0_32x2.val[0], _pA0_32x2.val[1]));
                uint16x8_t _pA11 = vreinterpretq_u16_u32(vcombine_u32(_pA1_32x2.val[0], _pA1_32x2.val[1]));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x4_t _pB2 = vld1_u16(pB + 8);
                uint16x4_t _pB3 = vld1_u16(pB + 12);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                uint16x8_t _pB22 = vcombine_u16(_pB2, _pB2);
                uint16x8_t _pB33 = vcombine_u16(_pB3, _pB3);
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB00);
                _sum1 = vbfdotq_f32(_sum1, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB00);
                _sum2 = vbfdotq_f32(_sum2, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB11);
                _sum3 = vbfdotq_f32(_sum3, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB11);
                _sum4 = vbfdotq_f32(_sum4, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB22);
                _sum5 = vbfdotq_f32(_sum5, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB22);
                _sum6 = vbfdotq_f32(_sum6, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB33);
                _sum7 = vbfdotq_f32(_sum7, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB33);

                pA += 8;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA = vld1_u16(pA);
                uint16x4x2_t _pA01 = vzip_u16(_pA, _pA);
                float32x4_t _pA00 = bfloat2float(_pA01.val[0]);
                float32x4_t _pA11 = bfloat2float(_pA01.val[1]);
                uint16x4_t _pB0123 = vld1_u16(pB);
                uint16x4_t _pB4567 = vld1_u16(pB + 4);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                uint32x2x2_t _pB4567_32x2 = vzip_u32(vreinterpret_u32_u16(_pB4567), vreinterpret_u32_u16(_pB4567));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                float32x4_t _pB2 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[0]));
                float32x4_t _pB3 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[1]));
                _sum0 = vfmaq_f32(_sum0, _pA00, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA11, _pB0);
                _sum2 = vfmaq_f32(_sum2, _pA00, _pB1);
                _sum3 = vfmaq_f32(_sum3, _pA11, _pB1);
                _sum4 = vfmaq_f32(_sum4, _pA00, _pB2);
                _sum5 = vfmaq_f32(_sum5, _pA11, _pB2);
                _sum6 = vfmaq_f32(_sum6, _pA00, _pB3);
                _sum7 = vfmaq_f32(_sum7, _pA11, _pB3);

                pA += 4;
                pB += 8;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _pA, vget_low_f32(_pB1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _pA, vget_low_f32(_pB1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _pA, vget_high_f32(_pB1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _pA, vget_high_f32(_pB1), 1);
#endif

                pA += 4;
                pB += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);
            vst1q_f32(outptr + 8, _sum2);
            vst1q_f32(outptr + 12, _sum3);
            vst1q_f32(outptr + 16, _sum4);
            vst1q_f32(outptr + 20, _sum5);
            vst1q_f32(outptr + 24, _sum6);
            vst1q_f32(outptr + 28, _sum7);

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
                _sum3 = vld1q_f32(outptr + 12);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                _sum0 = vbfmmlaq_f32(_sum0, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB0);
                _sum1 = vbfmmlaq_f32(_sum1, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB0);
                _sum2 = vbfmmlaq_f32(_sum2, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB1);
                _sum3 = vbfmmlaq_f32(_sum3, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB1);

                pA += 16;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pA1 = vld1_u16(pA + 4);
                uint32x2x2_t _pA0_32x2 = vzip_u32(vreinterpret_u32_u16(_pA0), vreinterpret_u32_u16(_pA0));
                uint32x2x2_t _pA1_32x2 = vzip_u32(vreinterpret_u32_u16(_pA1), vreinterpret_u32_u16(_pA1));
                uint16x8_t _pA00 = vreinterpretq_u16_u32(vcombine_u32(_pA0_32x2.val[0], _pA0_32x2.val[1]));
                uint16x8_t _pA11 = vreinterpretq_u16_u32(vcombine_u32(_pA1_32x2.val[0], _pA1_32x2.val[1]));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB00);
                _sum1 = vbfdotq_f32(_sum1, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB00);
                _sum2 = vbfdotq_f32(_sum2, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB11);
                _sum3 = vbfdotq_f32(_sum3, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB11);

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA = vld1_u16(pA);
                uint16x4x2_t _pA01 = vzip_u16(_pA, _pA);
                float32x4_t _pA00 = bfloat2float(_pA01.val[0]);
                float32x4_t _pA11 = bfloat2float(_pA01.val[1]);
                uint16x4_t _pB0123 = vld1_u16(pB);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                _sum0 = vfmaq_f32(_sum0, _pA00, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA11, _pB0);
                _sum2 = vfmaq_f32(_sum2, _pA00, _pB1);
                _sum3 = vfmaq_f32(_sum3, _pA11, _pB1);

                pA += 4;
                pB += 4;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB = bfloat2float(vld1_u16(pB));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB), 1);
#endif

                pA += 4;
                pB += 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);
            vst1q_f32(outptr + 8, _sum2);
            vst1q_f32(outptr + 12, _sum3);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA0 = vld1q_u16(pA);
                uint16x8_t _pA1 = vld1q_u16(pA + 8);
                uint16x8_t _pB = vld1q_u16(pB);
                _sum0 = vbfmmlaq_f32(_sum0, (bfloat16x8_t)_pA0, (bfloat16x8_t)_pB);
                _sum1 = vbfmmlaq_f32(_sum1, (bfloat16x8_t)_pA1, (bfloat16x8_t)_pB);

                pA += 16;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pA1 = vld1_u16(pA + 4);
                uint32x2x2_t _pA0_32x2 = vzip_u32(vreinterpret_u32_u16(_pA0), vreinterpret_u32_u16(_pA0));
                uint32x2x2_t _pA1_32x2 = vzip_u32(vreinterpret_u32_u16(_pA1), vreinterpret_u32_u16(_pA1));
                uint16x8_t _pA00 = vreinterpretq_u16_u32(vcombine_u32(_pA0_32x2.val[0], _pA0_32x2.val[1]));
                uint16x8_t _pA11 = vreinterpretq_u16_u32(vcombine_u32(_pA1_32x2.val[0], _pA1_32x2.val[1]));
                uint16x4_t _pB = vld1_u16(pB);
                uint16x8_t _pB01 = vcombine_u16(_pB, _pB);
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA00, (bfloat16x8_t)_pB01);
                _sum1 = vbfdotq_f32(_sum1, (bfloat16x8_t)_pA11, (bfloat16x8_t)_pB01);

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA = vld1_u16(pA);
                uint16x4x2_t _pA01 = vzip_u16(_pA, _pA);
                float32x4_t _pA00 = bfloat2float(_pA01.val[0]);
                float32x4_t _pA11 = bfloat2float(_pA01.val[1]);
                uint16x4_t _pB01 = vld1_dup_u16(pB);
                _pB01 = vld1_lane_u16(pB + 1, _pB01, 1);
                _pB01 = vld1_lane_u16(pB + 1, _pB01, 3);
                float32x4_t _pB = bfloat2float(_pB01);
                _sum0 = vfmaq_f32(_sum0, _pA00, _pB);
                _sum1 = vfmaq_f32(_sum1, _pA11, _pB);

                pA += 4;
                pB += 2;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0 = bfloat2float(vdup_n_u16(pB[0]));
                float32x4_t _pB1 = bfloat2float(vdup_n_u16(pB[1]));

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA, _pB1);
#else
                _sum0 = vmlaq_f32(_sum0, _pA, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA, _pB1);
#endif

                pA += 4;
                pB += 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _s01 = vdupq_n_f32(0.f);
            float32x4_t _s23 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA01 = vld1q_u16(pA);
                uint16x8_t _pA23 = vld1q_u16(pA + 8);
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x8_t _pB = vcombine_u16(_pB0, _pB0);

                _s01 = vbfdotq_f32(_s01, (bfloat16x8_t)_pA01, (bfloat16x8_t)_pB);
                _s23 = vbfdotq_f32(_s23, (bfloat16x8_t)_pA23, (bfloat16x8_t)_pB);

                pA += 16;
                pB += 4;
            }
            float32x2_t _s01p = vpadd_f32(vget_low_f32(_s01), vget_high_f32(_s01));
            float32x2_t _s23p = vpadd_f32(vget_low_f32(_s23), vget_high_f32(_s23));
            _sum0 = vaddq_f32(_sum0, vcombine_f32(_s01p, _s23p));
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint32x2_t _pB01 = vld1_dup_u32((const uint32_t*)pB);
                uint16x8_t _pB = vreinterpretq_u16_u32(vcombine_u32(_pB01, _pB01));
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB);

                pA += 8;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = bfloat2float(vdup_n_u16(pB[0]));
                _sum0 = vfmaq_f32(_sum0, bfloat2float(vld1_u16(pA)), _pB);

                pA += 4;
                pB += 1;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB = bfloat2float(vdup_n_u16(pB[0]));

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, _pB);
#else
                _sum0 = vmlaq_f32(_sum0, _pA, _pB);
#endif

                pA += 4;
                pB += 1;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
                _sum4 = vdupq_n_f32(0.f);
                _sum5 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
                _sum3 = vld1q_f32(outptr + 12);
                _sum4 = vld1q_f32(outptr + 16);
                _sum5 = vld1q_f32(outptr + 20);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                uint16x8_t _pB4 = vld1q_u16(pB + 32);
                uint16x8_t _pB5 = vld1q_u16(pB + 40);
                _sum0 = vbfmmlaq_f32(_sum0, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB0);
                _sum1 = vbfmmlaq_f32(_sum1, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB1);
                _sum2 = vbfmmlaq_f32(_sum2, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB2);
                _sum3 = vbfmmlaq_f32(_sum3, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB3);
                _sum4 = vbfmmlaq_f32(_sum4, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB4);
                _sum5 = vbfmmlaq_f32(_sum5, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB5);

                pA += 8;
                pB += 48;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA01 = vld1_u16(pA);
                uint32x2x2_t _pA01_32x2 = vzip_u32(vreinterpret_u32_u16(_pA01), vreinterpret_u32_u16(_pA01));
                uint16x8_t _pA0011 = vreinterpretq_u16_u32(vcombine_u32(_pA01_32x2.val[0], _pA01_32x2.val[1]));

                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x4_t _pB2 = vld1_u16(pB + 8);
                uint16x4_t _pB3 = vld1_u16(pB + 12);
                uint16x4_t _pB4 = vld1_u16(pB + 16);
                uint16x4_t _pB5 = vld1_u16(pB + 20);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                uint16x8_t _pB22 = vcombine_u16(_pB2, _pB2);
                uint16x8_t _pB33 = vcombine_u16(_pB3, _pB3);
                uint16x8_t _pB44 = vcombine_u16(_pB4, _pB4);
                uint16x8_t _pB55 = vcombine_u16(_pB5, _pB5);
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB00);
                _sum1 = vbfdotq_f32(_sum1, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB11);
                _sum2 = vbfdotq_f32(_sum2, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB22);
                _sum3 = vbfdotq_f32(_sum3, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB33);
                _sum4 = vbfdotq_f32(_sum4, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB44);
                _sum5 = vbfdotq_f32(_sum5, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB55);

                pA += 4;
                pB += 24;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA01 = vld1_dup_u16(pA);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 2);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 3);
                float32x4_t _pA = bfloat2float(_pA01);

                uint16x4_t _pB0123 = vld1_u16(pB);
                uint16x4_t _pB4567 = vld1_u16(pB + 4);
                uint16x4_t _pB89ab = vld1_u16(pB + 8);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                uint32x2x2_t _pB4567_32x2 = vzip_u32(vreinterpret_u32_u16(_pB4567), vreinterpret_u32_u16(_pB4567));
                uint32x2x2_t _pB89ab_32x2 = vzip_u32(vreinterpret_u32_u16(_pB89ab), vreinterpret_u32_u16(_pB89ab));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                float32x4_t _pB2 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[0]));
                float32x4_t _pB3 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[1]));
                float32x4_t _pB4 = bfloat2float(vreinterpret_u16_u32(_pB89ab_32x2.val[0]));
                float32x4_t _pB5 = bfloat2float(vreinterpret_u16_u32(_pB89ab_32x2.val[1]));
                _sum0 = vfmaq_f32(_sum0, _pA, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA, _pB2);
                _sum3 = vfmaq_f32(_sum3, _pA, _pB3);
                _sum4 = vfmaq_f32(_sum4, _pA, _pB4);
                _sum5 = vfmaq_f32(_sum5, _pA, _pB5);

                pA += 2;
                pB += 12;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA01 = vld1_dup_u16(pA);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 2);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 3);
                float32x4_t _pA = bfloat2float(_pA01);

                float32x4_t _pB0123 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB4567 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB89ab = bfloat2float(vld1_u16(pB + 8));
                _sum0 = vfmaq_f32(_sum0, _pA, vcombine_f32(vget_low_f32(_pB0123), vget_low_f32(_pB0123)));
                _sum1 = vfmaq_f32(_sum1, _pA, vcombine_f32(vget_high_f32(_pB0123), vget_high_f32(_pB0123)));
                _sum2 = vfmaq_f32(_sum2, _pA, vcombine_f32(vget_low_f32(_pB4567), vget_low_f32(_pB4567)));
                _sum3 = vfmaq_f32(_sum3, _pA, vcombine_f32(vget_high_f32(_pB4567), vget_high_f32(_pB4567)));
                _sum4 = vfmaq_f32(_sum4, _pA, vcombine_f32(vget_low_f32(_pB89ab), vget_low_f32(_pB89ab)));
                _sum5 = vfmaq_f32(_sum5, _pA, vcombine_f32(vget_high_f32(_pB89ab), vget_high_f32(_pB89ab)));

                pA += 2;
                pB += 12;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);
            vst1q_f32(outptr + 8, _sum2);
            vst1q_f32(outptr + 12, _sum3);
            vst1q_f32(outptr + 16, _sum4);
            vst1q_f32(outptr + 20, _sum5);

            outptr += 24;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
                _sum3 = vld1q_f32(outptr + 12);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                _sum0 = vbfmmlaq_f32(_sum0, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB0);
                _sum1 = vbfmmlaq_f32(_sum1, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB1);
                _sum2 = vbfmmlaq_f32(_sum2, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB2);
                _sum3 = vbfmmlaq_f32(_sum3, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB3);

                pA += 8;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA01 = vld1_u16(pA);
                uint32x2x2_t _pA01_32x2 = vzip_u32(vreinterpret_u32_u16(_pA01), vreinterpret_u32_u16(_pA01));
                uint16x8_t _pA0011 = vreinterpretq_u16_u32(vcombine_u32(_pA01_32x2.val[0], _pA01_32x2.val[1]));

                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x4_t _pB2 = vld1_u16(pB + 8);
                uint16x4_t _pB3 = vld1_u16(pB + 12);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                uint16x8_t _pB22 = vcombine_u16(_pB2, _pB2);
                uint16x8_t _pB33 = vcombine_u16(_pB3, _pB3);
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB00);
                _sum1 = vbfdotq_f32(_sum1, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB11);
                _sum2 = vbfdotq_f32(_sum2, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB22);
                _sum3 = vbfdotq_f32(_sum3, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB33);

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA01 = vld1_dup_u16(pA);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 2);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 3);
                float32x4_t _pA = bfloat2float(_pA01);

                uint16x4_t _pB0123 = vld1_u16(pB);
                uint16x4_t _pB4567 = vld1_u16(pB + 4);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                uint32x2x2_t _pB4567_32x2 = vzip_u32(vreinterpret_u32_u16(_pB4567), vreinterpret_u32_u16(_pB4567));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                float32x4_t _pB2 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[0]));
                float32x4_t _pB3 = bfloat2float(vreinterpret_u16_u32(_pB4567_32x2.val[1]));
                _sum0 = vfmaq_f32(_sum0, _pA, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA, _pB2);
                _sum3 = vfmaq_f32(_sum3, _pA, _pB3);

                pA += 2;
                pB += 8;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA01 = vld1_dup_u16(pA);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 2);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 3);
                float32x4_t _pA = bfloat2float(_pA01);

                float32x4_t _pB0123 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB4567 = bfloat2float(vld1_u16(pB + 4));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, vcombine_f32(vget_low_f32(_pB0123), vget_low_f32(_pB0123)));
                _sum1 = vfmaq_f32(_sum1, _pA, vcombine_f32(vget_high_f32(_pB0123), vget_high_f32(_pB0123)));
                _sum2 = vfmaq_f32(_sum2, _pA, vcombine_f32(vget_low_f32(_pB4567), vget_low_f32(_pB4567)));
                _sum3 = vfmaq_f32(_sum3, _pA, vcombine_f32(vget_high_f32(_pB4567), vget_high_f32(_pB4567)));
#else
                _sum0 = vmlaq_f32(_sum0, _pA, vcombine_f32(vget_low_f32(_pB0123), vget_low_f32(_pB0123)));
                _sum1 = vmlaq_f32(_sum1, _pA, vcombine_f32(vget_high_f32(_pB0123), vget_high_f32(_pB0123)));
                _sum2 = vmlaq_f32(_sum2, _pA, vcombine_f32(vget_low_f32(_pB4567), vget_low_f32(_pB4567)));
                _sum3 = vmlaq_f32(_sum3, _pA, vcombine_f32(vget_high_f32(_pB4567), vget_high_f32(_pB4567)));
#endif

                pA += 2;
                pB += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);
            vst1q_f32(outptr + 8, _sum2);
            vst1q_f32(outptr + 12, _sum3);

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                _sum0 = vbfmmlaq_f32(_sum0, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB0);
                _sum1 = vbfmmlaq_f32(_sum1, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB1);

                pA += 8;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA01 = vld1_u16(pA);
                uint32x2x2_t _pA01_32x2 = vzip_u32(vreinterpret_u32_u16(_pA01), vreinterpret_u32_u16(_pA01));
                uint16x8_t _pA0011 = vreinterpretq_u16_u32(vcombine_u32(_pA01_32x2.val[0], _pA01_32x2.val[1]));

                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB00);
                _sum1 = vbfdotq_f32(_sum1, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB11);

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA01 = vld1_dup_u16(pA);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 2);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 3);
                float32x4_t _pA = bfloat2float(_pA01);

                uint16x4_t _pB0123 = vld1_u16(pB);
                uint32x2x2_t _pB0123_32x2 = vzip_u32(vreinterpret_u32_u16(_pB0123), vreinterpret_u32_u16(_pB0123));
                float32x4_t _pB0 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[0]));
                float32x4_t _pB1 = bfloat2float(vreinterpret_u16_u32(_pB0123_32x2.val[1]));
                _sum0 = vfmaq_f32(_sum0, _pA, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA, _pB1);

                pA += 2;
                pB += 4;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA01 = vld1_dup_u16(pA);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 2);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 3);
                float32x4_t _pA = bfloat2float(_pA01);

                float32x4_t _pB0123 = bfloat2float(vld1_u16(pB));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, vcombine_f32(vget_low_f32(_pB0123), vget_low_f32(_pB0123)));
                _sum1 = vfmaq_f32(_sum1, _pA, vcombine_f32(vget_high_f32(_pB0123), vget_high_f32(_pB0123)));
#else
                _sum0 = vmlaq_f32(_sum0, _pA, vcombine_f32(vget_low_f32(_pB0123), vget_low_f32(_pB0123)));
                _sum1 = vmlaq_f32(_sum1, _pA, vcombine_f32(vget_high_f32(_pB0123), vget_high_f32(_pB0123)));
#endif

                pA += 2;
                pB += 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);

            outptr += 8;
        }
#endif // __ARM_NEON
#if __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum;

            if (k == 0)
                _sum = vdupq_n_f32(0.f);
            else
                _sum = vld1q_f32(outptr);

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x8_t _pB = vld1q_u16(pB);
                _sum = vbfmmlaq_f32(_sum, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB);

                pA += 8;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA01 = vld1_u16(pA);
                uint32x2x2_t _pA01_32x2 = vzip_u32(vreinterpret_u32_u16(_pA01), vreinterpret_u32_u16(_pA01));
                uint16x8_t _pA0011 = vreinterpretq_u16_u32(vcombine_u32(_pA01_32x2.val[0], _pA01_32x2.val[1]));
                uint16x4_t _pB = vld1_u16(pB);
                uint16x8_t _pB01 = vcombine_u16(_pB, _pB);
                _sum = vbfdotq_f32(_sum, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB01);

                pA += 4;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA01 = vld1_dup_u16(pA);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 2);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 3);
                float32x4_t _pA = bfloat2float(_pA01);

                uint16x4_t _pB01 = vld1_dup_u16(pB);
                _pB01 = vld1_lane_u16(pB + 1, _pB01, 1);
                _pB01 = vld1_lane_u16(pB + 1, _pB01, 3);
                float32x4_t _pB = bfloat2float(_pB01);
                _sum = vfmaq_f32(_sum, _pA, _pB);

                pA += 2;
                pB += 2;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            // clang 15.0.1 on aarch64 auto vectorization produces wrong result on this loop
            // we have to teach it a bit  :$   --- nihui
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _pA0123 = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0123 = bfloat2float(vld1_u16(pB));

                float32x4x2_t _pB0213 = vtrnq_f32(_pB0123, _pB0123);

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0123, _pB0213.val[0]);
                _sum1 = vfmaq_f32(_sum1, _pA0123, _pB0213.val[1]);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0123, _pB0213.val[0]);
                _sum1 = vmlaq_f32(_sum1, _pA0123, _pB0213.val[1]);
#endif

                pA += 4;
                pB += 4;
            }

            float32x2_t _s0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
            float32x2_t _s1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2x2_t _ss = vzip_f32(_s0, _s1);
            _sum = vaddq_f32(_sum, vcombine_f32(_ss.val[0], _ss.val[1]));
            for (; kk < max_kk; kk += 1)
            {
                uint16x4_t _pA01 = vld1_dup_u16(pA);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 2);
                _pA01 = vld1_lane_u16(pA + 1, _pA01, 3);
                float32x4_t _pA = bfloat2float(_pA01);

                uint16x4_t _pB01 = vld1_dup_u16(pB);
                _pB01 = vld1_lane_u16(pB + 1, _pB01, 1);
                _pB01 = vld1_lane_u16(pB + 1, _pB01, 3);
                float32x4_t _pB = bfloat2float(_pB01);

#if __aarch64__
                _sum = vfmaq_f32(_sum, _pA, _pB);
#else
                _sum = vmlaq_f32(_sum, _pA, _pB);
#endif

                pA += 2;
                pB += 2;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum);

            outptr += 4;
        }
#else  // __ARM_NEON
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
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float pA0 = bfloat16_to_float32(pA[0]);
                float pA1 = bfloat16_to_float32(pA[1]);
                float pB0 = bfloat16_to_float32(pB[0]);
                float pB1 = bfloat16_to_float32(pB[1]);

                sum00 += pA0 * pB0;
                sum01 += pA1 * pB0;
                sum10 += pA0 * pB1;
                sum11 += pA1 * pB1;

                pA += 2;
                pB += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;

            outptr += 4;
        }
#endif // __ARM_NEON
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum01 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8_t _pA = vld1q_u16(pA);
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x8_t _pB = vcombine_u16(_pB0, _pB0);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB);

                pA += 8;
                pB += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint16x4_t _pA01 = vld1_u16(pA);
                uint32x2x2_t _pA01_32x2 = vzip_u32(vreinterpret_u32_u16(_pA01), vreinterpret_u32_u16(_pA01));
                uint16x8_t _pA0011 = vreinterpretq_u16_u32(vcombine_u32(_pA01_32x2.val[0], _pA01_32x2.val[1]));
                uint32x2_t _pB01 = vld1_dup_u32((const uint32_t*)pB);
                uint16x8_t _pB = vreinterpretq_u16_u32(vcombine_u32(_pB01, _pB01));
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA0011, (bfloat16x8_t)_pB);

                pA += 4;
                pB += 2;
            }
            sum0 += vgetq_lane_f32(_sum01, 0);
            sum1 += vgetq_lane_f32(_sum01, 2);
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk += 1)
            {
                float pA0 = bfloat16_to_float32(pA[0]);
                float pA1 = bfloat16_to_float32(pA[1]);
                float pB0 = bfloat16_to_float32(pB[0]);

                sum0 += pA0 * pB0;
                sum1 += pA1 * pB0;

                pA += 2;
                pB += 1;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum01 = vdupq_n_f32(0.f);
            float32x4_t _sum23 = vdupq_n_f32(0.f);
            float32x4_t _sum45 = vdupq_n_f32(0.f);
            float32x4_t _sum67 = vdupq_n_f32(0.f);
            float32x4_t _sum89 = vdupq_n_f32(0.f);
            float32x4_t _sumab = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x8_t _pA = vcombine_u16(_pA0, _pA0);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                uint16x8_t _pB4 = vld1q_u16(pB + 32);
                uint16x8_t _pB5 = vld1q_u16(pB + 40);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB0);
                _sum23 = vbfmmlaq_f32(_sum23, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB1);
                _sum45 = vbfmmlaq_f32(_sum45, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB2);
                _sum67 = vbfmmlaq_f32(_sum67, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB3);
                _sum89 = vbfmmlaq_f32(_sum89, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB4);
                _sumab = vbfmmlaq_f32(_sumab, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB5);

                pA += 4;
                pB += 48;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _pA01 = vld1_dup_u32((const uint32_t*)pA);
                uint16x8_t _pA = vreinterpretq_u16_u32(vcombine_u32(_pA01, _pA01));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x4_t _pB2 = vld1_u16(pB + 8);
                uint16x4_t _pB3 = vld1_u16(pB + 12);
                uint16x4_t _pB4 = vld1_u16(pB + 16);
                uint16x4_t _pB5 = vld1_u16(pB + 20);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                uint16x8_t _pB22 = vcombine_u16(_pB2, _pB2);
                uint16x8_t _pB33 = vcombine_u16(_pB3, _pB3);
                uint16x8_t _pB44 = vcombine_u16(_pB4, _pB4);
                uint16x8_t _pB55 = vcombine_u16(_pB5, _pB5);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB00);
                _sum23 = vbfdotq_f32(_sum23, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB11);
                _sum45 = vbfdotq_f32(_sum45, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB22);
                _sum67 = vbfdotq_f32(_sum67, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB33);
                _sum89 = vbfdotq_f32(_sum89, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB44);
                _sumab = vbfdotq_f32(_sumab, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB55);

                pA += 2;
                pB += 24;
            }
            _sum0 = vaddq_f32(_sum0, vcombine_f32(vget_low_f32(_sum01), vget_low_f32(_sum23)));
            _sum1 = vaddq_f32(_sum1, vcombine_f32(vget_low_f32(_sum45), vget_low_f32(_sum67)));
            _sum2 = vaddq_f32(_sum2, vcombine_f32(vget_low_f32(_sum89), vget_low_f32(_sumab)));
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA0, _pB2);

                pA += 1;
                pB += 12;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum00 = vdupq_n_f32(0.f);
            float32x4_t _sum01 = vdupq_n_f32(0.f);
            float32x4_t _sum02 = vdupq_n_f32(0.f);
            float32x4_t _sum03 = vdupq_n_f32(0.f);
            float32x4_t _sum10 = vdupq_n_f32(0.f);
            float32x4_t _sum11 = vdupq_n_f32(0.f);
            float32x4_t _sum12 = vdupq_n_f32(0.f);
            float32x4_t _sum13 = vdupq_n_f32(0.f);
            float32x4_t _sum20 = vdupq_n_f32(0.f);
            float32x4_t _sum21 = vdupq_n_f32(0.f);
            float32x4_t _sum22 = vdupq_n_f32(0.f);
            float32x4_t _sum23 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                uint16x8_t _pB0 = vld1q_u16(pB);
                float32x4_t _pB00 = bfloat2float(vget_low_u16(_pB0));
                float32x4_t _pB01 = bfloat2float(vget_high_u16(_pB0));
                float32x4_t _pB02 = bfloat2float(vld1_u16(pB + 8));
                _sum00 = vfmaq_laneq_f32(_sum00, _pB00, _pA, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pB01, _pA, 0);
                _sum20 = vfmaq_laneq_f32(_sum20, _pB02, _pA, 0);
                uint16x8_t _pB1 = vld1q_u16(pB + 12);
                float32x4_t _pB10 = bfloat2float(vget_low_u16(_pB1));
                float32x4_t _pB11 = bfloat2float(vget_high_u16(_pB1));
                float32x4_t _pB12 = bfloat2float(vld1_u16(pB + 20));
                _sum01 = vfmaq_laneq_f32(_sum01, _pB10, _pA, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pB11, _pA, 1);
                _sum21 = vfmaq_laneq_f32(_sum21, _pB12, _pA, 1);
                uint16x8_t _pB2 = vld1q_u16(pB + 24);
                float32x4_t _pB20 = bfloat2float(vget_low_u16(_pB2));
                float32x4_t _pB21 = bfloat2float(vget_high_u16(_pB2));
                float32x4_t _pB22 = bfloat2float(vld1_u16(pB + 32));
                _sum02 = vfmaq_laneq_f32(_sum02, _pB20, _pA, 2);
                _sum12 = vfmaq_laneq_f32(_sum12, _pB21, _pA, 2);
                _sum22 = vfmaq_laneq_f32(_sum22, _pB22, _pA, 2);
                uint16x8_t _pB3 = vld1q_u16(pB + 36);
                float32x4_t _pB30 = bfloat2float(vget_low_u16(_pB3));
                float32x4_t _pB31 = bfloat2float(vget_high_u16(_pB3));
                float32x4_t _pB32 = bfloat2float(vld1_u16(pB + 44));
                _sum03 = vfmaq_laneq_f32(_sum03, _pB30, _pA, 3);
                _sum13 = vfmaq_laneq_f32(_sum13, _pB31, _pA, 3);
                _sum23 = vfmaq_laneq_f32(_sum23, _pB32, _pA, 3);

                pA += 4;
                pB += 48;
            }
            _sum00 = vaddq_f32(_sum00, _sum01);
            _sum02 = vaddq_f32(_sum02, _sum03);
            _sum10 = vaddq_f32(_sum10, _sum11);
            _sum12 = vaddq_f32(_sum12, _sum13);
            _sum20 = vaddq_f32(_sum20, _sum21);
            _sum22 = vaddq_f32(_sum22, _sum23);
            _sum00 = vaddq_f32(_sum00, _sum02);
            _sum10 = vaddq_f32(_sum10, _sum12);
            _sum20 = vaddq_f32(_sum20, _sum22);
            _sum0 = vaddq_f32(_sum0, _sum00);
            _sum1 = vaddq_f32(_sum1, _sum10);
            _sum2 = vaddq_f32(_sum2, _sum20);
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pB = vld1q_u16(pB);
                float32x4_t _pB0 = bfloat2float(vget_low_u16(_pB));
                float32x4_t _pB1 = bfloat2float(vget_high_u16(_pB));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));

                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA0, _pB2);

                pA += 1;
                pB += 12;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);
            vst1q_f32(outptr + 8, _sum2);

            outptr += 12;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum01 = vdupq_n_f32(0.f);
            float32x4_t _sum23 = vdupq_n_f32(0.f);
            float32x4_t _sum45 = vdupq_n_f32(0.f);
            float32x4_t _sum67 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x8_t _pA = vcombine_u16(_pA0, _pA0);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB0);
                _sum23 = vbfmmlaq_f32(_sum23, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB1);
                _sum45 = vbfmmlaq_f32(_sum45, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB2);
                _sum67 = vbfmmlaq_f32(_sum67, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB3);

                pA += 4;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _pA01 = vld1_dup_u32((const uint32_t*)pA);
                uint16x8_t _pA = vreinterpretq_u16_u32(vcombine_u32(_pA01, _pA01));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x4_t _pB2 = vld1_u16(pB + 8);
                uint16x4_t _pB3 = vld1_u16(pB + 12);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                uint16x8_t _pB22 = vcombine_u16(_pB2, _pB2);
                uint16x8_t _pB33 = vcombine_u16(_pB3, _pB3);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB00);
                _sum23 = vbfdotq_f32(_sum23, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB11);
                _sum45 = vbfdotq_f32(_sum45, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB22);
                _sum67 = vbfdotq_f32(_sum67, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB33);

                pA += 2;
                pB += 16;
            }
            _sum0 = vaddq_f32(_sum0, vcombine_f32(vget_low_f32(_sum01), vget_low_f32(_sum23)));
            _sum1 = vaddq_f32(_sum1, vcombine_f32(vget_low_f32(_sum45), vget_low_f32(_sum67)));
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);

                pA += 1;
                pB += 8;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum00 = vdupq_n_f32(0.f);
            float32x4_t _sum01 = vdupq_n_f32(0.f);
            float32x4_t _sum02 = vdupq_n_f32(0.f);
            float32x4_t _sum03 = vdupq_n_f32(0.f);
            float32x4_t _sum10 = vdupq_n_f32(0.f);
            float32x4_t _sum11 = vdupq_n_f32(0.f);
            float32x4_t _sum12 = vdupq_n_f32(0.f);
            float32x4_t _sum13 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                uint16x8_t _pB0 = vld1q_u16(pB);
                float32x4_t _pB00 = bfloat2float(vget_low_u16(_pB0));
                float32x4_t _pB01 = bfloat2float(vget_high_u16(_pB0));

#if __aarch64__
                _sum00 = vfmaq_laneq_f32(_sum00, _pB00, _pA, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pB01, _pA, 0);
#else
                float32x2_t _pA01 = vget_low_f32(_pA);
                float32x2_t _pA23 = vget_high_f32(_pA);
                _sum00 = vmlaq_lane_f32(_sum00, _pB00, _pA01, 0);
                _sum10 = vmlaq_lane_f32(_sum10, _pB01, _pA01, 0);
#endif

                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                float32x4_t _pB10 = bfloat2float(vget_low_u16(_pB1));
                float32x4_t _pB11 = bfloat2float(vget_high_u16(_pB1));

#if __aarch64__
                _sum01 = vfmaq_laneq_f32(_sum01, _pB10, _pA, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pB11, _pA, 1);
#else
                _sum01 = vmlaq_lane_f32(_sum01, _pB10, _pA01, 1);
                _sum11 = vmlaq_lane_f32(_sum11, _pB11, _pA01, 1);
#endif

                uint16x8_t _pB2 = vld1q_u16(pB + 16);
                float32x4_t _pB20 = bfloat2float(vget_low_u16(_pB2));
                float32x4_t _pB21 = bfloat2float(vget_high_u16(_pB2));

#if __aarch64__
                _sum02 = vfmaq_laneq_f32(_sum02, _pB20, _pA, 2);
                _sum12 = vfmaq_laneq_f32(_sum12, _pB21, _pA, 2);
#else
                _sum02 = vmlaq_lane_f32(_sum02, _pB20, _pA23, 0);
                _sum12 = vmlaq_lane_f32(_sum12, _pB21, _pA23, 0);
#endif

                uint16x8_t _pB3 = vld1q_u16(pB + 24);
                float32x4_t _pB30 = bfloat2float(vget_low_u16(_pB3));
                float32x4_t _pB31 = bfloat2float(vget_high_u16(_pB3));

#if __aarch64__
                _sum03 = vfmaq_laneq_f32(_sum03, _pB30, _pA, 3);
                _sum13 = vfmaq_laneq_f32(_sum13, _pB31, _pA, 3);
#else
                _sum03 = vmlaq_lane_f32(_sum03, _pB30, _pA23, 1);
                _sum13 = vmlaq_lane_f32(_sum13, _pB31, _pA23, 1);
#endif

                pA += 4;
                pB += 32;
            }
            _sum00 = vaddq_f32(_sum00, _sum01);
            _sum02 = vaddq_f32(_sum02, _sum03);
            _sum10 = vaddq_f32(_sum10, _sum11);
            _sum12 = vaddq_f32(_sum12, _sum13);
            _sum00 = vaddq_f32(_sum00, _sum02);
            _sum10 = vaddq_f32(_sum10, _sum12);
            _sum0 = vaddq_f32(_sum0, _sum00);
            _sum1 = vaddq_f32(_sum1, _sum10);
            for (; kk < max_kk; kk += 1)
            {
                uint16x8_t _pB = vld1q_u16(pB);
                float32x4_t _pB0 = bfloat2float(vget_low_u16(_pB));
                float32x4_t _pB1 = bfloat2float(vget_high_u16(_pB));

                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA0, _pB1);
#endif

                pA += 1;
                pB += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum;

            if (k == 0)
            {
                _sum = vdupq_n_f32(0.f);
            }
            else
            {
                _sum = vld1q_f32(outptr);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum01 = vdupq_n_f32(0.f);
            float32x4_t _sum23 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x8_t _pA = vcombine_u16(_pA0, _pA0);
                uint16x8_t _pB0 = vld1q_u16(pB);
                uint16x8_t _pB1 = vld1q_u16(pB + 8);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB0);
                _sum23 = vbfmmlaq_f32(_sum23, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB1);

                pA += 4;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _pA01 = vld1_dup_u32((const uint32_t*)pA);
                uint16x8_t _pA = vreinterpretq_u16_u32(vcombine_u32(_pA01, _pA01));
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x4_t _pB1 = vld1_u16(pB + 4);
                uint16x8_t _pB00 = vcombine_u16(_pB0, _pB0);
                uint16x8_t _pB11 = vcombine_u16(_pB1, _pB1);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB00);
                _sum23 = vbfdotq_f32(_sum23, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB11);

                pA += 2;
                pB += 8;
            }
            _sum = vaddq_f32(_sum, vcombine_f32(vget_low_f32(_sum01), vget_low_f32(_sum23)));
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));
                float32x4_t _pB = bfloat2float(vld1_u16(pB));
                _sum = vfmaq_f32(_sum, _pA0, _pB);

                pA += 1;
                pB += 4;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pB0, _pA, 0);
#else
                float32x2_t _pA01 = vget_low_f32(_pA);
                float32x2_t _pA23 = vget_high_f32(_pA);
                _sum0 = vmlaq_lane_f32(_sum0, _pB0, _pA01, 0);
#endif

                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
#if __aarch64__
                _sum1 = vfmaq_laneq_f32(_sum1, _pB1, _pA, 1);
#else
                _sum1 = vmlaq_lane_f32(_sum1, _pB1, _pA01, 1);
#endif

                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));
#if __aarch64__
                _sum2 = vfmaq_laneq_f32(_sum2, _pB2, _pA, 2);
#else
                _sum2 = vmlaq_lane_f32(_sum2, _pB2, _pA23, 0);
#endif

                float32x4_t _pB3 = bfloat2float(vld1_u16(pB + 12));
#if __aarch64__
                _sum3 = vfmaq_laneq_f32(_sum3, _pB3, _pA, 3);
#else
                _sum3 = vmlaq_lane_f32(_sum3, _pB3, _pA23, 1);
#endif

                pA += 4;
                pB += 16;
            }
            _sum0 = vaddq_f32(_sum0, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _sum0 = vaddq_f32(_sum0, _sum2);
            _sum = vaddq_f32(_sum, _sum0);
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = bfloat2float(vld1_u16(pB));
                float32x4_t _pA = bfloat2float(vdup_n_u16(pA[0]));
#if __aarch64__
                _sum = vfmaq_f32(_sum, _pA, _pB);
#else
                _sum = vmlaq_f32(_sum, _pA, _pB);
#endif

                pA += 1;
                pB += 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            vst1q_f32(outptr, _sum);

            outptr += 4;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum01 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x8_t _pA = vcombine_u16(_pA0, _pA0);
                uint16x8_t _pB = vld1q_u16(pB);
                _sum01 = vbfmmlaq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB);

                pA += 4;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _pA01 = vld1_dup_u32((const uint32_t*)pA);
                uint16x8_t _pA = vreinterpretq_u16_u32(vcombine_u32(_pA01, _pA01));
                uint16x4_t _pB = vld1_u16(pB);
                uint16x8_t _pB01 = vcombine_u16(_pB, _pB);
                _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB01);

                pA += 2;
                pB += 4;
            }
            sum0 += vgetq_lane_f32(_sum01, 0);
            sum1 += vgetq_lane_f32(_sum01, 1);
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float sum00 = 0.f;
            float sum01 = 0.f;
            float sum02 = 0.f;
            float sum03 = 0.f;
            float sum10 = 0.f;
            float sum11 = 0.f;
            float sum12 = 0.f;
            float sum13 = 0.f;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float pA0 = bfloat16_to_float32(pA[0]);
                float pA1 = bfloat16_to_float32(pA[1]);
                float pA2 = bfloat16_to_float32(pA[2]);
                float pA3 = bfloat16_to_float32(pA[3]);

                sum00 += pA0 * bfloat16_to_float32(pB[0]);
                sum10 += pA0 * bfloat16_to_float32(pB[1]);
                sum01 += pA1 * bfloat16_to_float32(pB[2]);
                sum11 += pA1 * bfloat16_to_float32(pB[3]);
                sum02 += pA2 * bfloat16_to_float32(pB[4]);
                sum12 += pA2 * bfloat16_to_float32(pB[5]);
                sum03 += pA3 * bfloat16_to_float32(pB[6]);
                sum13 += pA3 * bfloat16_to_float32(pB[7]);

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
                float pA0 = bfloat16_to_float32(pA[0]);
                float pB0 = bfloat16_to_float32(pB[0]);
                float pB1 = bfloat16_to_float32(pB[1]);

                sum0 += pA0 * pB0;
                sum1 += pA0 * pB1;

                pA += 1;
                pB += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                sum = 0.f;
            }
            else
            {
                sum = outptr[0];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _pA0 = vld1_u16(pA);
                uint16x4_t _pB0 = vld1_u16(pB);
                uint16x8_t _pA = vcombine_u16(_pA0, _pA0);
                uint16x8_t _pB = vcombine_u16(_pB0, _pB0);
                _sum0 = vbfmmlaq_f32(_sum0, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB);

                pA += 4;
                pB += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                uint32x2_t _pA01 = vld1_dup_u32((const uint32_t*)pA);
                uint32x2_t _pB01 = vld1_dup_u32((const uint32_t*)pB);
                uint16x8_t _pA = vreinterpretq_u16_u32(vcombine_u32(_pA01, _pA01));
                uint16x8_t _pB = vreinterpretq_u16_u32(vcombine_u32(_pB01, _pB01));
                _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_pA, (bfloat16x8_t)_pB);

                pA += 2;
                pB += 2;
            }
            sum += vgetq_lane_f32(_sum0, 0);
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[1]);
                sum2 += bfloat16_to_float32(pA[2]) * bfloat16_to_float32(pB[2]);
                sum3 += bfloat16_to_float32(pA[3]) * bfloat16_to_float32(pB[3]);

                pA += 4;
                pB += 4;
            }
            sum0 += sum1;
            sum2 += sum3;
            sum += sum0 + sum2;
            for (; kk < max_kk; kk += 1)
            {
                float pA0 = bfloat16_to_float32(pA[0]);
                float pB0 = bfloat16_to_float32(pB[0]);

                sum += pA0 * pB0;
                pA += 1;
                pB += 1;
            }

            outptr[0] = sum;

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void get_optimal_tile_mnk_bf16s(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(unsigned short) + sizeof(float)));

    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(8, tile_size / 8 * 8);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(unsigned short) / TILE_K);

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
