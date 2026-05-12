// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
void convolution_im2col_gemm_transform_kernel_bf16s_bf16(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt);
int convolution_im2col_gemm_bf16s_bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt);
#endif

static void convolution_im2col_pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __ARM_NEON
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
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#endif // __ARM_NEON
        for (; kk < max_kk; kk++)
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

static void convolution_im2col_input_tile_conv1x1s1d1_bf16(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    unsigned short* pp = B;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk / 4; kk++)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p0 + 4);
                uint16x4_t _r2 = vld1_u16(p0 + 8);
                uint16x4_t _r3 = vld1_u16(p0 + 12);
                uint16x4_t _r4 = vld1_u16(p0 + 16);
                uint16x4_t _r5 = vld1_u16(p0 + 20);
                uint16x4_t _r6 = vld1_u16(p0 + 24);
                uint16x4_t _r7 = vld1_u16(p0 + 28);
                uint16x4_t _r8 = vld1_u16(p0 + 32);
                uint16x4_t _r9 = vld1_u16(p0 + 36);
                uint16x4_t _ra = vld1_u16(p0 + 40);
                uint16x4_t _rb = vld1_u16(p0 + 44);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
                vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
                vst1q_u16(pp + 32, vcombine_u16(_r8, _r9));
                vst1q_u16(pp + 40, vcombine_u16(_ra, _rb));
                pp += 48;
                p0 += bottom_blob.cstep * 4;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x12
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #256]   \n"
                    "ld4    {v4.4h, v5.4h, v6.4h, v7.4h}, [%0]      \n"
                    "st1    {v0.8h}, [%1], #16          \n"
                    "st1    {v4.4h}, [%1], #8           \n"
                    "st1    {v1.8h}, [%1], #16          \n"
                    "st1    {v5.4h}, [%1], #8           \n"
                    "sub    %0, %0, #64                 \n"
                    "st1    {v2.8h}, [%1], #16          \n"
                    "st1    {v6.4h}, [%1], #8           \n"
                    "st1    {v3.8h}, [%1], #16          \n"
                    "st1    {v7.4h}, [%1], #8           \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // NCNN_GNU_INLINE_ASM
                uint16x4x4_t _r0 = vld4_u16(p0);
                uint16x4x4_t _r1 = vld4_u16(p0 + 16);
                uint16x4x4_t _r2 = vld4_u16(p0 + 32);
                vst1_u16(pp, _r0.val[0]);
                vst1_u16(pp + 4, _r1.val[0]);
                vst1_u16(pp + 4 * 2, _r2.val[0]);
                vst1_u16(pp + 4 * 3, _r0.val[1]);
                vst1_u16(pp + 4 * 4, _r1.val[1]);
                vst1_u16(pp + 4 * 5, _r2.val[1]);
                vst1_u16(pp + 4 * 6, _r0.val[2]);
                vst1_u16(pp + 4 * 7, _r1.val[2]);
                vst1_u16(pp + 4 * 8, _r2.val[2]);
                vst1_u16(pp + 4 * 9, _r0.val[3]);
                vst1_u16(pp + 4 * 10, _r1.val[3]);
                vst1_u16(pp + 4 * 11, _r2.val[3]);
                pp += 48;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }

        if (elempack == 1)
        {
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k + kk + 1) + (j + jj);
                const unsigned short* p2 = (const unsigned short*)bottom_blob.channel(k + kk + 2) + (j + jj);
                const unsigned short* p3 = (const unsigned short*)bottom_blob.channel(k + kk + 3) + (j + jj);

                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));

                _r0 = vld1_u16(p0 + 4);
                _r1 = vld1_u16(p1 + 4);
                _r2 = vld1_u16(p2 + 4);
                _r3 = vld1_u16(p3 + 4);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp + 16, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 24, vcombine_u16(_r2, _r3));

                _r0 = vld1_u16(p0 + 8);
                _r1 = vld1_u16(p1 + 8);
                _r2 = vld1_u16(p2 + 8);
                _r3 = vld1_u16(p3 + 8);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp + 32, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 40, vcombine_u16(_r2, _r3));

                pp += 48;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k + kk + 1) + (j + jj);

                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p1);
                uint16x8x2_t _r01 = vzipq_u16(_r0, _r1);
                vst1q_u16(pp, _r01.val[0]);
                vst1q_u16(pp + 8, _r01.val[1]);

                uint16x4x2_t _r23 = vzip_u16(vld1_u16(p0 + 8), vld1_u16(p1 + 8));
                vst1_u16(pp + 16, _r23.val[0]);
                vst1_u16(pp + 20, _r23.val[1]);
                pp += 24;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                uint16x8_t _r01 = vld1q_u16(p0);
                uint16x4_t _r2 = vld1_u16(p0 + 8);
                vst1q_u16(pp, _r01);
                vst1_u16(pp + 8, _r2);
                pp += 12;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk / 4; kk++)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p0 + 4);
                uint16x4_t _r2 = vld1_u16(p0 + 8);
                uint16x4_t _r3 = vld1_u16(p0 + 12);
                uint16x4_t _r4 = vld1_u16(p0 + 16);
                uint16x4_t _r5 = vld1_u16(p0 + 20);
                uint16x4_t _r6 = vld1_u16(p0 + 24);
                uint16x4_t _r7 = vld1_u16(p0 + 28);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
                vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
                pp += 32;
                p0 += bottom_blob.cstep * 4;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x8
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld4.u16   {d0,d2,d4,d6}, [%0 :64]! \n"
                    "pld        [%0, #256]          \n"
                    "vld4.u16   {d1,d3,d5,d7}, [%0 :64] \n"
                    "sub        %0, %0, #32         \n"
                    "vstm       %1!, {d0-d7}        \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                uint16x8x4_t _r0 = vld4q_u16(p0);
                vst1q_u16(pp, _r0.val[0]);
                vst1q_u16(pp + 8, _r0.val[1]);
                vst1q_u16(pp + 16, _r0.val[2]);
                vst1q_u16(pp + 24, _r0.val[3]);
                pp += 32;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }

        if (elempack == 1)
        {
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k + kk + 1) + (j + jj);
                const unsigned short* p2 = (const unsigned short*)bottom_blob.channel(k + kk + 2) + (j + jj);
                const unsigned short* p3 = (const unsigned short*)bottom_blob.channel(k + kk + 3) + (j + jj);

                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));

                _r0 = vld1_u16(p0 + 4);
                _r1 = vld1_u16(p1 + 4);
                _r2 = vld1_u16(p2 + 4);
                _r3 = vld1_u16(p3 + 4);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp + 16, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 24, vcombine_u16(_r2, _r3));

                pp += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k + kk + 1) + (j + jj);

                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p1);
                uint16x8x2_t _r01 = vzipq_u16(_r0, _r1);
                vst1q_u16(pp, _r01.val[0]);
                vst1q_u16(pp + 8, _r01.val[1]);
                pp += 16;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                uint16x8_t _r0 = vld1q_u16(p0);
                vst1q_u16(pp, _r0);
                pp += 8;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk / 4; kk++)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p0 + 4);
                uint16x4_t _r2 = vld1_u16(p0 + 8);
                uint16x4_t _r3 = vld1_u16(p0 + 12);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                pp += 16;
                p0 += bottom_blob.cstep * 4;
            }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x4
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0] \n"
                    "st4    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.u16   {d0-d3}, [%0 :64]   \n"
                    "vst4.u16   {d0-d3}, [%1 :64]!  \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "q0", "q1");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                uint16x4x4_t _r0;
                _r0.val[0] = vld1_u16(p0);
                _r0.val[1] = vld1_u16(p0 + 4);
                _r0.val[2] = vld1_u16(p0 + 4 * 2);
                _r0.val[3] = vld1_u16(p0 + 4 * 3);
                vst4_u16(pp, _r0);
                pp += 16;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }

        if (elempack == 1)
        {
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k + kk + 1) + (j + jj);
                const unsigned short* p2 = (const unsigned short*)bottom_blob.channel(k + kk + 2) + (j + jj);
                const unsigned short* p3 = (const unsigned short*)bottom_blob.channel(k + kk + 3) + (j + jj);

                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                pp += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k + kk + 1) + (j + jj);

                uint16x4x2_t _r01 = vzip_u16(vld1_u16(p0), vld1_u16(p1));
                vst1_u16(pp, _r01.val[0]);
                vst1_u16(pp + 4, _r01.val[1]);
                pp += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                uint16x4_t _r0 = vld1_u16(p0);
                vst1_u16(pp, _r0);
                pp += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __ARM_NEON

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk / 4; kk++)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p0 + 4);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                pp += 8;
                p0 += bottom_blob.cstep * 4;
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x2
                uint16x4x2_t _r0;
                _r0.val[0] = vld1_u16(p0);
                _r0.val[1] = vld1_u16(p0 + 4);
                vst2_u16(pp, _r0);
                pp += 8;
                p0 += bottom_blob.cstep * 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk + 3 < max_kk; kk += 4)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k + kk + 1) + (j + jj);
                const unsigned short* p2 = (const unsigned short*)bottom_blob.channel(k + kk + 2) + (j + jj);
                const unsigned short* p3 = (const unsigned short*)bottom_blob.channel(k + kk + 3) + (j + jj);

                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p0[1];
                pp[5] = p1[1];
                pp[6] = p2[1];
                pp[7] = p3[1];
                pp += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k + kk + 1) + (j + jj);

                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p0[1];
                pp[3] = p1[1];
                pp += 4;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; kk < max_kk; kk++)
            {
                const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k + kk) + (j + jj);
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __ARM_NEON

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += bottom_blob.cstep;
            }
        }
    }
}

static void convolution_im2col_input_tile_bf16(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_bf16(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    unsigned short* pp = B;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dy8 = (j + jj + 8) / outw;
        int dy9 = (j + jj + 9) / outw;
        int dya = (j + jj + 10) / outw;
        int dyb = (j + jj + 11) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;
        int dx8 = (j + jj + 8) % outw;
        int dx9 = (j + jj + 9) % outw;
        int dxa = (j + jj + 10) % outw;
        int dxb = (j + jj + 11) % outw;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        if (elempack == 4)
        {
            for (; kk < max_kk / 4; kk++)
            {
                int p = (k / 4 + kk) / maxk;
                int uv = (k / 4 + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int x8 = stride_w * dx8 + dilation_w * v;
                int x9 = stride_w * dx9 + dilation_w * v;
                int xa = stride_w * dxa + dilation_w * v;
                int xb = stride_w * dxb + dilation_w * v;

                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;
                int y8 = stride_h * dy8 + dilation_h * u;
                int y9 = stride_h * dy9 + dilation_h * u;
                int ya = stride_h * dya + dilation_h * u;
                int yb = stride_h * dyb + dilation_h * u;

                const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * 4;
                const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * 4;
                const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * 4;
                const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * 4;
                const unsigned short* sptr4 = img.row<const unsigned short>(y4) + x4 * 4;
                const unsigned short* sptr5 = img.row<const unsigned short>(y5) + x5 * 4;
                const unsigned short* sptr6 = img.row<const unsigned short>(y6) + x6 * 4;
                const unsigned short* sptr7 = img.row<const unsigned short>(y7) + x7 * 4;
                const unsigned short* sptr8 = img.row<const unsigned short>(y8) + x8 * 4;
                const unsigned short* sptr9 = img.row<const unsigned short>(y9) + x9 * 4;
                const unsigned short* sptra = img.row<const unsigned short>(ya) + xa * 4;
                const unsigned short* sptrb = img.row<const unsigned short>(yb) + xb * 4;

                uint16x4_t _r0 = vld1_u16(sptr0);
                uint16x4_t _r1 = vld1_u16(sptr1);
                uint16x4_t _r2 = vld1_u16(sptr2);
                uint16x4_t _r3 = vld1_u16(sptr3);
                uint16x4_t _r4 = vld1_u16(sptr4);
                uint16x4_t _r5 = vld1_u16(sptr5);
                uint16x4_t _r6 = vld1_u16(sptr6);
                uint16x4_t _r7 = vld1_u16(sptr7);
                uint16x4_t _r8 = vld1_u16(sptr8);
                uint16x4_t _r9 = vld1_u16(sptr9);
                uint16x4_t _ra = vld1_u16(sptra);
                uint16x4_t _rb = vld1_u16(sptrb);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
                vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
                vst1q_u16(pp + 32, vcombine_u16(_r8, _r9));
                vst1q_u16(pp + 40, vcombine_u16(_ra, _rb));
                pp += 48;
            }
        }
        if (elempack == 1)
        {
            for (; kk + 3 < max_kk; kk += 4)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int p2 = (k + kk + 2) / maxk;
                int p3 = (k + kk + 3) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int uv2 = (k + kk + 2) % maxk;
                int uv3 = (k + kk + 3) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int u2 = uv2 / kernel_w;
                int u3 = uv3 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;
                int v2 = uv2 % kernel_w;
                int v3 = uv3 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);
                const Mat img2 = bottom_blob.channel(p2);
                const Mat img3 = bottom_blob.channel(p3);

                pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                pp[2] = img2.row<const unsigned short>(stride_h * dy0 + dilation_h * u2)[stride_w * dx0 + dilation_w * v2];
                pp[3] = img3.row<const unsigned short>(stride_h * dy0 + dilation_h * u3)[stride_w * dx0 + dilation_w * v3];
                pp[4] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                pp[5] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                pp[6] = img2.row<const unsigned short>(stride_h * dy1 + dilation_h * u2)[stride_w * dx1 + dilation_w * v2];
                pp[7] = img3.row<const unsigned short>(stride_h * dy1 + dilation_h * u3)[stride_w * dx1 + dilation_w * v3];
                pp[8] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                pp[9] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                pp[10] = img2.row<const unsigned short>(stride_h * dy2 + dilation_h * u2)[stride_w * dx2 + dilation_w * v2];
                pp[11] = img3.row<const unsigned short>(stride_h * dy2 + dilation_h * u3)[stride_w * dx2 + dilation_w * v3];
                pp[12] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                pp[13] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                pp[14] = img2.row<const unsigned short>(stride_h * dy3 + dilation_h * u2)[stride_w * dx3 + dilation_w * v2];
                pp[15] = img3.row<const unsigned short>(stride_h * dy3 + dilation_h * u3)[stride_w * dx3 + dilation_w * v3];
                pp[16] = img0.row<const unsigned short>(stride_h * dy4 + dilation_h * u0)[stride_w * dx4 + dilation_w * v0];
                pp[17] = img1.row<const unsigned short>(stride_h * dy4 + dilation_h * u1)[stride_w * dx4 + dilation_w * v1];
                pp[18] = img2.row<const unsigned short>(stride_h * dy4 + dilation_h * u2)[stride_w * dx4 + dilation_w * v2];
                pp[19] = img3.row<const unsigned short>(stride_h * dy4 + dilation_h * u3)[stride_w * dx4 + dilation_w * v3];
                pp[20] = img0.row<const unsigned short>(stride_h * dy5 + dilation_h * u0)[stride_w * dx5 + dilation_w * v0];
                pp[21] = img1.row<const unsigned short>(stride_h * dy5 + dilation_h * u1)[stride_w * dx5 + dilation_w * v1];
                pp[22] = img2.row<const unsigned short>(stride_h * dy5 + dilation_h * u2)[stride_w * dx5 + dilation_w * v2];
                pp[23] = img3.row<const unsigned short>(stride_h * dy5 + dilation_h * u3)[stride_w * dx5 + dilation_w * v3];
                pp[24] = img0.row<const unsigned short>(stride_h * dy6 + dilation_h * u0)[stride_w * dx6 + dilation_w * v0];
                pp[25] = img1.row<const unsigned short>(stride_h * dy6 + dilation_h * u1)[stride_w * dx6 + dilation_w * v1];
                pp[26] = img2.row<const unsigned short>(stride_h * dy6 + dilation_h * u2)[stride_w * dx6 + dilation_w * v2];
                pp[27] = img3.row<const unsigned short>(stride_h * dy6 + dilation_h * u3)[stride_w * dx6 + dilation_w * v3];
                pp[28] = img0.row<const unsigned short>(stride_h * dy7 + dilation_h * u0)[stride_w * dx7 + dilation_w * v0];
                pp[29] = img1.row<const unsigned short>(stride_h * dy7 + dilation_h * u1)[stride_w * dx7 + dilation_w * v1];
                pp[30] = img2.row<const unsigned short>(stride_h * dy7 + dilation_h * u2)[stride_w * dx7 + dilation_w * v2];
                pp[31] = img3.row<const unsigned short>(stride_h * dy7 + dilation_h * u3)[stride_w * dx7 + dilation_w * v3];
                pp[32] = img0.row<const unsigned short>(stride_h * dy8 + dilation_h * u0)[stride_w * dx8 + dilation_w * v0];
                pp[33] = img1.row<const unsigned short>(stride_h * dy8 + dilation_h * u1)[stride_w * dx8 + dilation_w * v1];
                pp[34] = img2.row<const unsigned short>(stride_h * dy8 + dilation_h * u2)[stride_w * dx8 + dilation_w * v2];
                pp[35] = img3.row<const unsigned short>(stride_h * dy8 + dilation_h * u3)[stride_w * dx8 + dilation_w * v3];
                pp[36] = img0.row<const unsigned short>(stride_h * dy9 + dilation_h * u0)[stride_w * dx9 + dilation_w * v0];
                pp[37] = img1.row<const unsigned short>(stride_h * dy9 + dilation_h * u1)[stride_w * dx9 + dilation_w * v1];
                pp[38] = img2.row<const unsigned short>(stride_h * dy9 + dilation_h * u2)[stride_w * dx9 + dilation_w * v2];
                pp[39] = img3.row<const unsigned short>(stride_h * dy9 + dilation_h * u3)[stride_w * dx9 + dilation_w * v3];
                pp[40] = img0.row<const unsigned short>(stride_h * dya + dilation_h * u0)[stride_w * dxa + dilation_w * v0];
                pp[41] = img1.row<const unsigned short>(stride_h * dya + dilation_h * u1)[stride_w * dxa + dilation_w * v1];
                pp[42] = img2.row<const unsigned short>(stride_h * dya + dilation_h * u2)[stride_w * dxa + dilation_w * v2];
                pp[43] = img3.row<const unsigned short>(stride_h * dya + dilation_h * u3)[stride_w * dxa + dilation_w * v3];
                pp[44] = img0.row<const unsigned short>(stride_h * dyb + dilation_h * u0)[stride_w * dxb + dilation_w * v0];
                pp[45] = img1.row<const unsigned short>(stride_h * dyb + dilation_h * u1)[stride_w * dxb + dilation_w * v1];
                pp[46] = img2.row<const unsigned short>(stride_h * dyb + dilation_h * u2)[stride_w * dxb + dilation_w * v2];
                pp[47] = img3.row<const unsigned short>(stride_h * dyb + dilation_h * u3)[stride_w * dxb + dilation_w * v3];
                pp += 48;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);

                pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                pp[2] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                pp[3] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                pp[4] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                pp[5] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                pp[6] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                pp[7] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                pp[8] = img0.row<const unsigned short>(stride_h * dy4 + dilation_h * u0)[stride_w * dx4 + dilation_w * v0];
                pp[9] = img1.row<const unsigned short>(stride_h * dy4 + dilation_h * u1)[stride_w * dx4 + dilation_w * v1];
                pp[10] = img0.row<const unsigned short>(stride_h * dy5 + dilation_h * u0)[stride_w * dx5 + dilation_w * v0];
                pp[11] = img1.row<const unsigned short>(stride_h * dy5 + dilation_h * u1)[stride_w * dx5 + dilation_w * v1];
                pp[12] = img0.row<const unsigned short>(stride_h * dy6 + dilation_h * u0)[stride_w * dx6 + dilation_w * v0];
                pp[13] = img1.row<const unsigned short>(stride_h * dy6 + dilation_h * u1)[stride_w * dx6 + dilation_w * v1];
                pp[14] = img0.row<const unsigned short>(stride_h * dy7 + dilation_h * u0)[stride_w * dx7 + dilation_w * v0];
                pp[15] = img1.row<const unsigned short>(stride_h * dy7 + dilation_h * u1)[stride_w * dx7 + dilation_w * v1];
                pp[16] = img0.row<const unsigned short>(stride_h * dy8 + dilation_h * u0)[stride_w * dx8 + dilation_w * v0];
                pp[17] = img1.row<const unsigned short>(stride_h * dy8 + dilation_h * u1)[stride_w * dx8 + dilation_w * v1];
                pp[18] = img0.row<const unsigned short>(stride_h * dy9 + dilation_h * u0)[stride_w * dx9 + dilation_w * v0];
                pp[19] = img1.row<const unsigned short>(stride_h * dy9 + dilation_h * u1)[stride_w * dx9 + dilation_w * v1];
                pp[20] = img0.row<const unsigned short>(stride_h * dya + dilation_h * u0)[stride_w * dxa + dilation_w * v0];
                pp[21] = img1.row<const unsigned short>(stride_h * dya + dilation_h * u1)[stride_w * dxa + dilation_w * v1];
                pp[22] = img0.row<const unsigned short>(stride_h * dyb + dilation_h * u0)[stride_w * dxb + dilation_w * v0];
                pp[23] = img1.row<const unsigned short>(stride_h * dyb + dilation_h * u1)[stride_w * dxb + dilation_w * v1];
                pp += 24;
            }
            for (; kk < max_kk; kk++)
            {
                int p = (k + kk) / maxk;
                int uv = (k + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                pp[0] = img.row<const unsigned short>(stride_h * dy0 + dilation_h * u)[stride_w * dx0 + dilation_w * v];
                pp[1] = img.row<const unsigned short>(stride_h * dy1 + dilation_h * u)[stride_w * dx1 + dilation_w * v];
                pp[2] = img.row<const unsigned short>(stride_h * dy2 + dilation_h * u)[stride_w * dx2 + dilation_w * v];
                pp[3] = img.row<const unsigned short>(stride_h * dy3 + dilation_h * u)[stride_w * dx3 + dilation_w * v];
                pp[4] = img.row<const unsigned short>(stride_h * dy4 + dilation_h * u)[stride_w * dx4 + dilation_w * v];
                pp[5] = img.row<const unsigned short>(stride_h * dy5 + dilation_h * u)[stride_w * dx5 + dilation_w * v];
                pp[6] = img.row<const unsigned short>(stride_h * dy6 + dilation_h * u)[stride_w * dx6 + dilation_w * v];
                pp[7] = img.row<const unsigned short>(stride_h * dy7 + dilation_h * u)[stride_w * dx7 + dilation_w * v];
                pp[8] = img.row<const unsigned short>(stride_h * dy8 + dilation_h * u)[stride_w * dx8 + dilation_w * v];
                pp[9] = img.row<const unsigned short>(stride_h * dy9 + dilation_h * u)[stride_w * dx9 + dilation_w * v];
                pp[10] = img.row<const unsigned short>(stride_h * dya + dilation_h * u)[stride_w * dxa + dilation_w * v];
                pp[11] = img.row<const unsigned short>(stride_h * dyb + dilation_h * u)[stride_w * dxb + dilation_w * v];
                pp += 12;
            }
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int x2 = stride_w * dx2 + dilation_w * v;
            int x3 = stride_w * dx3 + dilation_w * v;
            int x4 = stride_w * dx4 + dilation_w * v;
            int x5 = stride_w * dx5 + dilation_w * v;
            int x6 = stride_w * dx6 + dilation_w * v;
            int x7 = stride_w * dx7 + dilation_w * v;
            int x8 = stride_w * dx8 + dilation_w * v;
            int x9 = stride_w * dx9 + dilation_w * v;
            int xa = stride_w * dxa + dilation_w * v;
            int xb = stride_w * dxb + dilation_w * v;

            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;
            int y2 = stride_h * dy2 + dilation_h * u;
            int y3 = stride_h * dy3 + dilation_h * u;
            int y4 = stride_h * dy4 + dilation_h * u;
            int y5 = stride_h * dy5 + dilation_h * u;
            int y6 = stride_h * dy6 + dilation_h * u;
            int y7 = stride_h * dy7 + dilation_h * u;
            int y8 = stride_h * dy8 + dilation_h * u;
            int y9 = stride_h * dy9 + dilation_h * u;
            int ya = stride_h * dya + dilation_h * u;
            int yb = stride_h * dyb + dilation_h * u;

            const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
            const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
            const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
            const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;
            const unsigned short* sptr4 = img.row<const unsigned short>(y4) + x4 * elempack;
            const unsigned short* sptr5 = img.row<const unsigned short>(y5) + x5 * elempack;
            const unsigned short* sptr6 = img.row<const unsigned short>(y6) + x6 * elempack;
            const unsigned short* sptr7 = img.row<const unsigned short>(y7) + x7 * elempack;
            const unsigned short* sptr8 = img.row<const unsigned short>(y8) + x8 * elempack;
            const unsigned short* sptr9 = img.row<const unsigned short>(y9) + x9 * elempack;
            const unsigned short* sptra = img.row<const unsigned short>(ya) + xa * elempack;
            const unsigned short* sptrb = img.row<const unsigned short>(yb) + xb * elempack;

            if (elempack == 4)
            {
                uint16x4_t _r0 = vld1_u16(sptr0);
                uint16x4_t _r1 = vld1_u16(sptr1);
                uint16x4_t _r2 = vld1_u16(sptr2);
                uint16x4_t _r3 = vld1_u16(sptr3);
                uint16x4_t _r4 = vld1_u16(sptr4);
                uint16x4_t _r5 = vld1_u16(sptr5);
                uint16x4_t _r6 = vld1_u16(sptr6);
                uint16x4_t _r7 = vld1_u16(sptr7);
                uint16x4_t _r8 = vld1_u16(sptr8);
                uint16x4_t _r9 = vld1_u16(sptr9);
                uint16x4_t _ra = vld1_u16(sptra);
                uint16x4_t _rb = vld1_u16(sptrb);
                transpose4x12_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r1);
                vst1_u16(pp + 4 * 2, _r2);
                vst1_u16(pp + 4 * 3, _r3);
                vst1_u16(pp + 4 * 4, _r4);
                vst1_u16(pp + 4 * 5, _r5);
                vst1_u16(pp + 4 * 6, _r6);
                vst1_u16(pp + 4 * 7, _r7);
                vst1_u16(pp + 4 * 8, _r8);
                vst1_u16(pp + 4 * 9, _r9);
                vst1_u16(pp + 4 * 10, _ra);
                vst1_u16(pp + 4 * 11, _rb);
                pp += 48;
            }
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr2[0];
                pp[3] = sptr3[0];
                pp[4] = sptr4[0];
                pp[5] = sptr5[0];
                pp[6] = sptr6[0];
                pp[7] = sptr7[0];
                pp[8] = sptr8[0];
                pp[9] = sptr9[0];
                pp[10] = sptra[0];
                pp[11] = sptrb[0];
                pp += 12;
            }
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        if (elempack == 4)
        {
            for (; kk < max_kk / 4; kk++)
            {
                int p = (k / 4 + kk) / maxk;
                int uv = (k / 4 + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;

                const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * 4;
                const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * 4;
                const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * 4;
                const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * 4;
                const unsigned short* sptr4 = img.row<const unsigned short>(y4) + x4 * 4;
                const unsigned short* sptr5 = img.row<const unsigned short>(y5) + x5 * 4;
                const unsigned short* sptr6 = img.row<const unsigned short>(y6) + x6 * 4;
                const unsigned short* sptr7 = img.row<const unsigned short>(y7) + x7 * 4;

                uint16x4_t _r0 = vld1_u16(sptr0);
                uint16x4_t _r1 = vld1_u16(sptr1);
                uint16x4_t _r2 = vld1_u16(sptr2);
                uint16x4_t _r3 = vld1_u16(sptr3);
                uint16x4_t _r4 = vld1_u16(sptr4);
                uint16x4_t _r5 = vld1_u16(sptr5);
                uint16x4_t _r6 = vld1_u16(sptr6);
                uint16x4_t _r7 = vld1_u16(sptr7);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                vst1q_u16(pp + 16, vcombine_u16(_r4, _r5));
                vst1q_u16(pp + 24, vcombine_u16(_r6, _r7));
                pp += 32;
            }
        }
        if (elempack == 1)
        {
            for (; kk + 3 < max_kk; kk += 4)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int p2 = (k + kk + 2) / maxk;
                int p3 = (k + kk + 3) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int uv2 = (k + kk + 2) % maxk;
                int uv3 = (k + kk + 3) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int u2 = uv2 / kernel_w;
                int u3 = uv3 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;
                int v2 = uv2 % kernel_w;
                int v3 = uv3 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);
                const Mat img2 = bottom_blob.channel(p2);
                const Mat img3 = bottom_blob.channel(p3);

                pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                pp[2] = img2.row<const unsigned short>(stride_h * dy0 + dilation_h * u2)[stride_w * dx0 + dilation_w * v2];
                pp[3] = img3.row<const unsigned short>(stride_h * dy0 + dilation_h * u3)[stride_w * dx0 + dilation_w * v3];
                pp[4] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                pp[5] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                pp[6] = img2.row<const unsigned short>(stride_h * dy1 + dilation_h * u2)[stride_w * dx1 + dilation_w * v2];
                pp[7] = img3.row<const unsigned short>(stride_h * dy1 + dilation_h * u3)[stride_w * dx1 + dilation_w * v3];
                pp[8] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                pp[9] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                pp[10] = img2.row<const unsigned short>(stride_h * dy2 + dilation_h * u2)[stride_w * dx2 + dilation_w * v2];
                pp[11] = img3.row<const unsigned short>(stride_h * dy2 + dilation_h * u3)[stride_w * dx2 + dilation_w * v3];
                pp[12] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                pp[13] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                pp[14] = img2.row<const unsigned short>(stride_h * dy3 + dilation_h * u2)[stride_w * dx3 + dilation_w * v2];
                pp[15] = img3.row<const unsigned short>(stride_h * dy3 + dilation_h * u3)[stride_w * dx3 + dilation_w * v3];
                pp[16] = img0.row<const unsigned short>(stride_h * dy4 + dilation_h * u0)[stride_w * dx4 + dilation_w * v0];
                pp[17] = img1.row<const unsigned short>(stride_h * dy4 + dilation_h * u1)[stride_w * dx4 + dilation_w * v1];
                pp[18] = img2.row<const unsigned short>(stride_h * dy4 + dilation_h * u2)[stride_w * dx4 + dilation_w * v2];
                pp[19] = img3.row<const unsigned short>(stride_h * dy4 + dilation_h * u3)[stride_w * dx4 + dilation_w * v3];
                pp[20] = img0.row<const unsigned short>(stride_h * dy5 + dilation_h * u0)[stride_w * dx5 + dilation_w * v0];
                pp[21] = img1.row<const unsigned short>(stride_h * dy5 + dilation_h * u1)[stride_w * dx5 + dilation_w * v1];
                pp[22] = img2.row<const unsigned short>(stride_h * dy5 + dilation_h * u2)[stride_w * dx5 + dilation_w * v2];
                pp[23] = img3.row<const unsigned short>(stride_h * dy5 + dilation_h * u3)[stride_w * dx5 + dilation_w * v3];
                pp[24] = img0.row<const unsigned short>(stride_h * dy6 + dilation_h * u0)[stride_w * dx6 + dilation_w * v0];
                pp[25] = img1.row<const unsigned short>(stride_h * dy6 + dilation_h * u1)[stride_w * dx6 + dilation_w * v1];
                pp[26] = img2.row<const unsigned short>(stride_h * dy6 + dilation_h * u2)[stride_w * dx6 + dilation_w * v2];
                pp[27] = img3.row<const unsigned short>(stride_h * dy6 + dilation_h * u3)[stride_w * dx6 + dilation_w * v3];
                pp[28] = img0.row<const unsigned short>(stride_h * dy7 + dilation_h * u0)[stride_w * dx7 + dilation_w * v0];
                pp[29] = img1.row<const unsigned short>(stride_h * dy7 + dilation_h * u1)[stride_w * dx7 + dilation_w * v1];
                pp[30] = img2.row<const unsigned short>(stride_h * dy7 + dilation_h * u2)[stride_w * dx7 + dilation_w * v2];
                pp[31] = img3.row<const unsigned short>(stride_h * dy7 + dilation_h * u3)[stride_w * dx7 + dilation_w * v3];
                pp += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);

                pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                pp[2] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                pp[3] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                pp[4] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                pp[5] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                pp[6] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                pp[7] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                pp[8] = img0.row<const unsigned short>(stride_h * dy4 + dilation_h * u0)[stride_w * dx4 + dilation_w * v0];
                pp[9] = img1.row<const unsigned short>(stride_h * dy4 + dilation_h * u1)[stride_w * dx4 + dilation_w * v1];
                pp[10] = img0.row<const unsigned short>(stride_h * dy5 + dilation_h * u0)[stride_w * dx5 + dilation_w * v0];
                pp[11] = img1.row<const unsigned short>(stride_h * dy5 + dilation_h * u1)[stride_w * dx5 + dilation_w * v1];
                pp[12] = img0.row<const unsigned short>(stride_h * dy6 + dilation_h * u0)[stride_w * dx6 + dilation_w * v0];
                pp[13] = img1.row<const unsigned short>(stride_h * dy6 + dilation_h * u1)[stride_w * dx6 + dilation_w * v1];
                pp[14] = img0.row<const unsigned short>(stride_h * dy7 + dilation_h * u0)[stride_w * dx7 + dilation_w * v0];
                pp[15] = img1.row<const unsigned short>(stride_h * dy7 + dilation_h * u1)[stride_w * dx7 + dilation_w * v1];
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                int p = (k + kk) / maxk;
                int uv = (k + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                pp[0] = img.row<const unsigned short>(stride_h * dy0 + dilation_h * u)[stride_w * dx0 + dilation_w * v];
                pp[1] = img.row<const unsigned short>(stride_h * dy1 + dilation_h * u)[stride_w * dx1 + dilation_w * v];
                pp[2] = img.row<const unsigned short>(stride_h * dy2 + dilation_h * u)[stride_w * dx2 + dilation_w * v];
                pp[3] = img.row<const unsigned short>(stride_h * dy3 + dilation_h * u)[stride_w * dx3 + dilation_w * v];
                pp[4] = img.row<const unsigned short>(stride_h * dy4 + dilation_h * u)[stride_w * dx4 + dilation_w * v];
                pp[5] = img.row<const unsigned short>(stride_h * dy5 + dilation_h * u)[stride_w * dx5 + dilation_w * v];
                pp[6] = img.row<const unsigned short>(stride_h * dy6 + dilation_h * u)[stride_w * dx6 + dilation_w * v];
                pp[7] = img.row<const unsigned short>(stride_h * dy7 + dilation_h * u)[stride_w * dx7 + dilation_w * v];
                pp += 8;
            }
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int x2 = stride_w * dx2 + dilation_w * v;
            int x3 = stride_w * dx3 + dilation_w * v;
            int x4 = stride_w * dx4 + dilation_w * v;
            int x5 = stride_w * dx5 + dilation_w * v;
            int x6 = stride_w * dx6 + dilation_w * v;
            int x7 = stride_w * dx7 + dilation_w * v;
            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;
            int y2 = stride_h * dy2 + dilation_h * u;
            int y3 = stride_h * dy3 + dilation_h * u;
            int y4 = stride_h * dy4 + dilation_h * u;
            int y5 = stride_h * dy5 + dilation_h * u;
            int y6 = stride_h * dy6 + dilation_h * u;
            int y7 = stride_h * dy7 + dilation_h * u;

            const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
            const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
            const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
            const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;
            const unsigned short* sptr4 = img.row<const unsigned short>(y4) + x4 * elempack;
            const unsigned short* sptr5 = img.row<const unsigned short>(y5) + x5 * elempack;
            const unsigned short* sptr6 = img.row<const unsigned short>(y6) + x6 * elempack;
            const unsigned short* sptr7 = img.row<const unsigned short>(y7) + x7 * elempack;

            if (elempack == 4)
            {
                uint16x4_t _r0 = vld1_u16(sptr0);
                uint16x4_t _r1 = vld1_u16(sptr1);
                uint16x4_t _r2 = vld1_u16(sptr2);
                uint16x4_t _r3 = vld1_u16(sptr3);
                uint16x4_t _r4 = vld1_u16(sptr4);
                uint16x4_t _r5 = vld1_u16(sptr5);
                uint16x4_t _r6 = vld1_u16(sptr6);
                uint16x4_t _r7 = vld1_u16(sptr7);
                transpose4x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r1);
                vst1_u16(pp + 4 * 2, _r2);
                vst1_u16(pp + 4 * 3, _r3);
                vst1_u16(pp + 4 * 4, _r4);
                vst1_u16(pp + 4 * 5, _r5);
                vst1_u16(pp + 4 * 6, _r6);
                vst1_u16(pp + 4 * 7, _r7);
                pp += 32;
            }
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr2[0];
                pp[3] = sptr3[0];
                pp[4] = sptr4[0];
                pp[5] = sptr5[0];
                pp[6] = sptr6[0];
                pp[7] = sptr7[0];
                pp += 8;
            }
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        if (elempack == 4)
        {
            for (; kk < max_kk / 4; kk++)
            {
                int p = (k / 4 + kk) / maxk;
                int uv = (k / 4 + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;

                const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * 4;
                const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * 4;
                const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * 4;
                const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * 4;

                uint16x4_t _r0 = vld1_u16(sptr0);
                uint16x4_t _r1 = vld1_u16(sptr1);
                uint16x4_t _r2 = vld1_u16(sptr2);
                uint16x4_t _r3 = vld1_u16(sptr3);
                vst1q_u16(pp, vcombine_u16(_r0, _r1));
                vst1q_u16(pp + 8, vcombine_u16(_r2, _r3));
                pp += 16;
            }
        }
        if (elempack == 1)
        {
            for (; kk + 3 < max_kk; kk += 4)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int p2 = (k + kk + 2) / maxk;
                int p3 = (k + kk + 3) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int uv2 = (k + kk + 2) % maxk;
                int uv3 = (k + kk + 3) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int u2 = uv2 / kernel_w;
                int u3 = uv3 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;
                int v2 = uv2 % kernel_w;
                int v3 = uv3 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);
                const Mat img2 = bottom_blob.channel(p2);
                const Mat img3 = bottom_blob.channel(p3);

                pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                pp[2] = img2.row<const unsigned short>(stride_h * dy0 + dilation_h * u2)[stride_w * dx0 + dilation_w * v2];
                pp[3] = img3.row<const unsigned short>(stride_h * dy0 + dilation_h * u3)[stride_w * dx0 + dilation_w * v3];
                pp[4] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                pp[5] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                pp[6] = img2.row<const unsigned short>(stride_h * dy1 + dilation_h * u2)[stride_w * dx1 + dilation_w * v2];
                pp[7] = img3.row<const unsigned short>(stride_h * dy1 + dilation_h * u3)[stride_w * dx1 + dilation_w * v3];
                pp[8] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                pp[9] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                pp[10] = img2.row<const unsigned short>(stride_h * dy2 + dilation_h * u2)[stride_w * dx2 + dilation_w * v2];
                pp[11] = img3.row<const unsigned short>(stride_h * dy2 + dilation_h * u3)[stride_w * dx2 + dilation_w * v3];
                pp[12] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                pp[13] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                pp[14] = img2.row<const unsigned short>(stride_h * dy3 + dilation_h * u2)[stride_w * dx3 + dilation_w * v2];
                pp[15] = img3.row<const unsigned short>(stride_h * dy3 + dilation_h * u3)[stride_w * dx3 + dilation_w * v3];
                pp += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);

                pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                pp[2] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                pp[3] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                pp[4] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                pp[5] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                pp[6] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                pp[7] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                int p = (k + kk) / maxk;
                int uv = (k + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                pp[0] = img.row<const unsigned short>(stride_h * dy0 + dilation_h * u)[stride_w * dx0 + dilation_w * v];
                pp[1] = img.row<const unsigned short>(stride_h * dy1 + dilation_h * u)[stride_w * dx1 + dilation_w * v];
                pp[2] = img.row<const unsigned short>(stride_h * dy2 + dilation_h * u)[stride_w * dx2 + dilation_w * v];
                pp[3] = img.row<const unsigned short>(stride_h * dy3 + dilation_h * u)[stride_w * dx3 + dilation_w * v];
                pp += 4;
            }
        }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int x2 = stride_w * dx2 + dilation_w * v;
            int x3 = stride_w * dx3 + dilation_w * v;
            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;
            int y2 = stride_h * dy2 + dilation_h * u;
            int y3 = stride_h * dy3 + dilation_h * u;

            const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
            const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
            const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
            const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;

            if (elempack == 4)
            {
                uint16x4x4_t _r0;
                _r0.val[0] = vld1_u16(sptr0);
                _r0.val[1] = vld1_u16(sptr1);
                _r0.val[2] = vld1_u16(sptr2);
                _r0.val[3] = vld1_u16(sptr3);
                vst4_u16(pp, _r0);
                pp += 16;
            }
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr2[0];
                pp[3] = sptr3[0];
                pp += 4;
            }
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        if (elempack == 4)
        {
            for (; kk < max_kk / 4; kk++)
            {
                int p = (k / 4 + kk) / maxk;
                int uv = (k / 4 + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;

                const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * 4;
                const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * 4;

                vst1q_u16(pp, vcombine_u16(vld1_u16(sptr0), vld1_u16(sptr1)));
                pp += 8;
            }
        }
        if (elempack == 1)
        {
            for (; kk + 3 < max_kk; kk += 4)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int p2 = (k + kk + 2) / maxk;
                int p3 = (k + kk + 3) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int uv2 = (k + kk + 2) % maxk;
                int uv3 = (k + kk + 3) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int u2 = uv2 / kernel_w;
                int u3 = uv3 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;
                int v2 = uv2 % kernel_w;
                int v3 = uv3 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);
                const Mat img2 = bottom_blob.channel(p2);
                const Mat img3 = bottom_blob.channel(p3);

                pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                pp[2] = img2.row<const unsigned short>(stride_h * dy0 + dilation_h * u2)[stride_w * dx0 + dilation_w * v2];
                pp[3] = img3.row<const unsigned short>(stride_h * dy0 + dilation_h * u3)[stride_w * dx0 + dilation_w * v3];
                pp[4] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                pp[5] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                pp[6] = img2.row<const unsigned short>(stride_h * dy1 + dilation_h * u2)[stride_w * dx1 + dilation_w * v2];
                pp[7] = img3.row<const unsigned short>(stride_h * dy1 + dilation_h * u3)[stride_w * dx1 + dilation_w * v3];
                pp += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);

                pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                pp[2] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                pp[3] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                int p = (k + kk) / maxk;
                int uv = (k + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                pp[0] = img.row<const unsigned short>(stride_h * dy0 + dilation_h * u)[stride_w * dx0 + dilation_w * v];
                pp[1] = img.row<const unsigned short>(stride_h * dy1 + dilation_h * u)[stride_w * dx1 + dilation_w * v];
                pp += 2;
            }
        }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int x1 = stride_w * dx1 + dilation_w * v;
            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;

            const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
            const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;

#if __ARM_NEON
            if (elempack == 4)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr0[1];
                pp[3] = sptr1[1];
                pp[4] = sptr0[2];
                pp[5] = sptr1[2];
                pp[6] = sptr0[3];
                pp[7] = sptr1[3];
                pp += 8;
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp += 2;
            }
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        if (elempack == 4)
        {
            for (; kk < max_kk / 4; kk++)
            {
                int p = (k / 4 + kk) / maxk;
                int uv = (k / 4 + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x = stride_w * dx + dilation_w * v;
                int y = stride_h * dy + dilation_h * u;

                const unsigned short* sptr = img.row<const unsigned short>(y) + x * 4;

                vst1_u16(pp, vld1_u16(sptr));
                pp += 4;
            }
        }
        if (elempack == 1)
        {
            for (; kk < max_kk; kk++)
            {
                int p = (k + kk) / maxk;
                int uv = (k + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x = stride_w * dx + dilation_w * v;
                int y = stride_h * dy + dilation_h * u;

                pp[0] = img.row<const unsigned short>(y)[x];
                pp += 1;
            }
        }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const unsigned short* sptr = img.row<const unsigned short>(y) + x * elempack;

#if __ARM_NEON
            if (elempack == 4)
            {
                pp[0] = sptr[0];
                pp[1] = sptr[1];
                pp[2] = sptr[2];
                pp[3] = sptr[3];
                pp += 4;
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    }
}

static void convolution_gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end, int use_a53_a55_optimized_kernel)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_bf16s %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (use_a53_a55_optimized_kernel && cpu_support_arm_asimdhp())
            {
                // a55
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #320                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v8.4s}, [%8]               \n"
                    "ld1    {v20.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v20.16b, v20.16b, v20.16b   \n"

                    "2:                                 \n"
                    "mov    v9.16b, v8.16b              \n"
                    "mov    v10.16b, v8.16b             \n"
                    "mov    v11.16b, v8.16b             \n"
                    "mov    v12.16b, v8.16b             \n"
                    "mov    v13.16b, v8.16b             \n"
                    "mov    v14.16b, v8.16b             \n"
                    "mov    v15.16b, v8.16b             \n"
                    "mov    v16.16b, v8.16b             \n"
                    "mov    v17.16b, v8.16b             \n"
                    "mov    v18.16b, v8.16b             \n"
                    "mov    v19.16b, v8.16b             \n"

                    "mov    v21.16b, v20.16b            \n"
                    "mov    v22.16b, v20.16b            \n"
                    "mov    v23.16b, v20.16b            \n"
                    "mov    v24.16b, v20.16b            \n"
                    "mov    v25.16b, v20.16b            \n"
                    "mov    v26.16b, v20.16b            \n"
                    "mov    v27.16b, v20.16b            \n"
                    "mov    v28.16b, v20.16b            \n"
                    "mov    v29.16b, v20.16b            \n"
                    "mov    v30.16b, v20.16b            \n"
                    "mov    v31.16b, v20.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v0.4s, v0.4h, #16           \n"

                    ".align 4                           \n"
                    "4:                                 \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "ldr    d6, [%1], #8                \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "ldr    d7, [%1], #8                \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "ldr    d4, [%1], #8                \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"
                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                    "ldr    d5, [%1], #8                \n"
                    "fmla   v10.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v3.s[3]      \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v3.s[3]      \n"
                    "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v13.4s, v6.4s, v0.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v0.s[3]      \n"
                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v16.4s, v6.4s, v1.s[0]      \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v1.s[3]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "ldr    d6, [%1], #8                \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "ldr    d7, [%1], #8                \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v2.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v3.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v3.s[3]      \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "ldr    d4, [%1], #8                \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                    "ldr    d5, [%1], #8                 \n"
                    "fmla   v10.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v1.s[3]      \n"
                    "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                    "ldr    d1, [%2], #8                 \n"
                    "fmla   v13.4s, v6.4s, v2.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v2.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v2.s[3]      \n"
                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v16.4s, v6.4s, v3.s[0]      \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v3.s[3]      \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v28.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v3.s[3]      \n"
                    "bne    4b                          \n"

                    "sub    %1, %1, #16                 \n"
                    "sub    %2, %2, #24                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"

                    "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v8.4s, #16           \n"
                    "shrn2  v0.8h, v9.4s, #16           \n"
                    "shrn   v1.4h, v10.4s, #16          \n"
                    "shrn2  v1.8h, v11.4s, #16          \n"
                    "shrn   v2.4h, v12.4s, #16          \n"
                    "shrn2  v2.8h, v13.4s, #16          \n"
                    "shrn   v3.4h, v14.4s, #16          \n"
                    "shrn2  v3.8h, v15.4s, #16          \n"
                    "shrn   v4.4h, v16.4s, #16          \n"
                    "shrn2  v4.8h, v17.4s, #16          \n"
                    "shrn   v5.4h, v18.4s, #16          \n"
                    "shrn2  v5.8h, v19.4s, #16          \n"
                    "shrn   v6.4h, v20.4s, #16          \n"
                    "shrn2  v6.8h, v21.4s, #16          \n"
                    "shrn   v7.4h, v22.4s, #16          \n"
                    "shrn2  v7.8h, v23.4s, #16          \n"
                    "shrn   v8.4h, v24.4s, #16          \n"
                    "shrn2  v8.8h, v25.4s, #16          \n"
                    "shrn   v9.4h, v26.4s, #16          \n"
                    "shrn2  v9.8h, v27.4s, #16          \n"
                    "shrn   v10.4h, v28.4s, #16         \n"
                    "shrn2  v10.8h, v29.4s, #16         \n"
                    "shrn   v11.4h, v30.4s, #16         \n"
                    "shrn2  v11.8h, v31.4s, #16         \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h}, [%3], #32 \n"
                    "st1    {v6.8h, v7.8h, v8.8h, v9.8h}, [x4], #64 \n"
                    "st1    {v10.8h, v11.8h}, [x4]      \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x12
                    "uzp1   v20.8h, v0.8h, v1.8h        \n"
                    "uzp2   v21.8h, v0.8h, v1.8h        \n"
                    "uzp1   v22.8h, v2.8h, v3.8h        \n"
                    "uzp2   v23.8h, v2.8h, v3.8h        \n"
                    "uzp1   v24.8h, v4.8h, v5.8h        \n"
                    "uzp2   v25.8h, v4.8h, v5.8h        \n"
                    "uzp1   v26.8h, v6.8h, v7.8h        \n"
                    "uzp2   v27.8h, v6.8h, v7.8h        \n"
                    "uzp1   v28.8h, v8.8h, v9.8h        \n"
                    "uzp2   v29.8h, v8.8h, v9.8h        \n"
                    "uzp1   v30.8h, v10.8h, v11.8h      \n"
                    "uzp2   v31.8h, v10.8h, v11.8h      \n"

                    "uzp1   v0.8h, v20.8h, v22.8h       \n"
                    "uzp2   v6.8h, v20.8h, v22.8h       \n"
                    "uzp1   v3.8h, v21.8h, v23.8h       \n"
                    "uzp2   v9.8h, v21.8h, v23.8h       \n"
                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"
                    "mov    v4.d[0], v3.d[1]            \n"
                    "mov    v10.d[0], v9.d[1]           \n"
                    "uzp1   v2.8h, v24.8h, v24.8h       \n"
                    "uzp2   v8.8h, v24.8h, v24.8h       \n"
                    "uzp1   v5.8h, v25.8h, v25.8h       \n"
                    "uzp2   v11.8h, v25.8h, v25.8h      \n"

                    "uzp1   v12.8h, v26.8h, v28.8h      \n"
                    "uzp2   v18.8h, v26.8h, v28.8h      \n"
                    "uzp1   v15.8h, v27.8h, v29.8h      \n"
                    "uzp2   v21.8h, v27.8h, v29.8h      \n"
                    "mov    v13.d[0], v12.d[1]          \n"
                    "mov    v19.d[0], v18.d[1]          \n"
                    "mov    v16.d[0], v15.d[1]          \n"
                    "mov    v22.d[0], v21.d[1]          \n"
                    "uzp1   v14.8h, v30.8h, v30.8h      \n"
                    "uzp2   v20.8h, v30.8h, v30.8h      \n"
                    "uzp1   v17.8h, v31.8h, v31.8h      \n"
                    "uzp2   v23.8h, v31.8h, v31.8h      \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                    "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v12.4h, v13.4h, v14.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v15.4h, v16.4h, v17.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v18.4h, v19.4h, v20.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v21.4h, v22.4h, v23.4h}, [x4] \n"

                    "9:                                 \n"
                    "add    %0, %0, #384                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else if (use_a53_a55_optimized_kernel && !cpu_support_arm_asimdhp())
            {
                // a53
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #320                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v8.4s}, [%8]               \n"
                    "ld1    {v20.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v20.16b, v20.16b, v20.16b   \n"

                    "2:                                 \n"
                    "mov    v9.16b, v8.16b              \n"
                    "mov    v10.16b, v8.16b             \n"
                    "mov    v11.16b, v8.16b             \n"
                    "mov    v12.16b, v8.16b             \n"
                    "mov    v13.16b, v8.16b             \n"
                    "mov    v14.16b, v8.16b             \n"
                    "mov    v15.16b, v8.16b             \n"
                    "mov    v16.16b, v8.16b             \n"
                    "mov    v17.16b, v8.16b             \n"
                    "mov    v18.16b, v8.16b             \n"
                    "mov    v19.16b, v8.16b             \n"

                    "mov    v21.16b, v20.16b            \n"
                    "mov    v22.16b, v20.16b            \n"
                    "mov    v23.16b, v20.16b            \n"
                    "mov    v24.16b, v20.16b            \n"
                    "mov    v25.16b, v20.16b            \n"
                    "mov    v26.16b, v20.16b            \n"
                    "mov    v27.16b, v20.16b            \n"
                    "mov    v28.16b, v20.16b            \n"
                    "mov    v29.16b, v20.16b            \n"
                    "mov    v30.16b, v20.16b            \n"
                    "mov    v31.16b, v20.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v4.4h}, [%1], #8           \n"

                    "prfm   pldl1keep, [%2, #384]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "ldr    x25, [%1]                   \n"
                    "add    %1, %1, #8                  \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v0.4s, v0.4h, #16           \n"

                    ".align 4                           \n"
                    "4:                                 \n"

                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "ldr    x26, [%1]                   \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"

                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "ldr    x23, [%2]                   \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"

                    "nop                                \n"
                    "ins    v5.d[0], x25                \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "ldr    x20, [%2]                   \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"

                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "ldr    x21, [%2]                   \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "ins    v6.d[0], x26                \n"
                    "ins    v3.d[0], x23                \n"
                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "prfm   pldl1keep, [%2, #384]       \n" // NOTE PRELOAD
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "nop                                \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"

                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "ldr    x27, [%1]                   \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"

                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "prfm   pldl1keep, [%1, #256]       \n" // NOTE PRELOAD
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "nop                                \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"

                    "ins    v0.d[0], x20                \n"
                    "ins    v1.d[0], x21                \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "nop                                \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "nop                                \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                    "ldr    x24, [%1]                   \n"
                    "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v10.4s, v6.4s, v3.s[2]      \n"

                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v11.4s, v6.4s, v3.s[3]      \n"
                    "ldr    x22, [%2]                   \n"
                    "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v13.4s, v6.4s, v0.s[1]      \n"

                    "nop                                \n"
                    "ins    v7.d[0], x27                \n"
                    "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                    "ldr    x23, [%2]                   \n"
                    "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v16.4s, v6.4s, v1.s[0]      \n"

                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                    "ldr    x20, [%2]                   \n"
                    "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v19.4s, v6.4s, v1.s[3]      \n"

                    "ins    v4.d[0], x24                \n"
                    "ins    v2.d[0], x22                \n"
                    "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                    "nop                                \n"
                    "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                    "nop                                \n"
                    "fmla   v22.4s, v7.4s, v3.s[2]      \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v23.4s, v7.4s, v3.s[3]      \n"
                    "ldr    x25, [%1]                   \n"
                    "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v25.4s, v7.4s, v0.s[1]      \n"

                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                    "nop                                \n"
                    "fmla   v27.4s, v7.4s, v0.s[3]      \n"
                    "nop                                \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"

                    "ins    v3.d[0], x23                \n"
                    "ins    v0.d[0], x20                \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "nop                                \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "nop                                \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "ldr    x26, [%1]                   \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                    "ldr    x21, [%2]                   \n"
                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"

                    "nop                                \n"
                    "ins    v5.d[0], x25                \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "ldr    x22, [%2]                   \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"

                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "ldr    x23, [%2]                   \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"

                    "ins    v6.d[0], x26                \n"
                    "ins    v1.d[0], x21                \n"
                    "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                    "prfm   pldl1keep, [%2, #384]       \n" // NOTE PRELOAD
                    "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                    "nop                                \n"
                    "fmla   v22.4s, v5.4s, v2.s[2]      \n"

                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v23.4s, v5.4s, v2.s[3]      \n"
                    "ldr    x27, [%1]                   \n"
                    "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v25.4s, v5.4s, v3.s[1]      \n"

                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                    "prfm   pldl1keep, [%1, #256]       \n" // NOTE PRELOAD
                    "fmla   v27.4s, v5.4s, v3.s[3]      \n"
                    "nop                                \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"

                    "ins    v2.d[0], x22                \n"
                    "ins    v3.d[0], x23                \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "nop                                \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "nop                                \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                    "ldr    x24, [%1]                   \n"
                    "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v10.4s, v6.4s, v1.s[2]      \n"

                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                    "ldr    x20, [%2]                   \n"
                    "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v13.4s, v6.4s, v2.s[1]      \n"

                    "nop                                \n"
                    "ins    v7.d[0], x27                \n"
                    "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                    "ldr    x21, [%2]                   \n"
                    "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v16.4s, v6.4s, v3.s[0]      \n"

                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                    "ldr    x22, [%2]                   \n"
                    "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v19.4s, v6.4s, v3.s[3]      \n"

                    "ins    v4.d[0], x24                \n"
                    "ins    v0.d[0], x20                \n"
                    "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                    "nop                                \n"
                    "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                    "nop                                \n"
                    "fmla   v22.4s, v7.4s, v1.s[2]      \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v23.4s, v7.4s, v1.s[3]      \n"
                    "ldr    x25, [%1]                   \n"
                    "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v25.4s, v7.4s, v2.s[1]      \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                    "nop                                \n"
                    "fmla   v27.4s, v7.4s, v2.s[3]      \n"
                    "nop                                \n"
                    "fmla   v28.4s, v7.4s, v3.s[0]      \n"

                    "ins    v1.d[0], x21                \n"
                    "ins    v2.d[0], x22                \n"
                    "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                    "nop                                \n"
                    "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                    "bne    4b                          \n"

                    "sub    %1, %1, #16                 \n"
                    "sub    %2, %2, #24                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"

                    "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v8.4s, #16           \n"
                    "shrn2  v0.8h, v9.4s, #16           \n"
                    "shrn   v1.4h, v10.4s, #16          \n"
                    "shrn2  v1.8h, v11.4s, #16          \n"
                    "shrn   v2.4h, v12.4s, #16          \n"
                    "shrn2  v2.8h, v13.4s, #16          \n"
                    "shrn   v3.4h, v14.4s, #16          \n"
                    "shrn2  v3.8h, v15.4s, #16          \n"
                    "shrn   v4.4h, v16.4s, #16          \n"
                    "shrn2  v4.8h, v17.4s, #16          \n"
                    "shrn   v5.4h, v18.4s, #16          \n"
                    "shrn2  v5.8h, v19.4s, #16          \n"
                    "shrn   v6.4h, v20.4s, #16          \n"
                    "shrn2  v6.8h, v21.4s, #16          \n"
                    "shrn   v7.4h, v22.4s, #16          \n"
                    "shrn2  v7.8h, v23.4s, #16          \n"
                    "shrn   v8.4h, v24.4s, #16          \n"
                    "shrn2  v8.8h, v25.4s, #16          \n"
                    "shrn   v9.4h, v26.4s, #16          \n"
                    "shrn2  v9.8h, v27.4s, #16          \n"
                    "shrn   v10.4h, v28.4s, #16         \n"
                    "shrn2  v10.8h, v29.4s, #16         \n"
                    "shrn   v11.4h, v30.4s, #16         \n"
                    "shrn2  v11.8h, v31.4s, #16         \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h}, [%3], #32 \n"
                    "st1    {v6.8h, v7.8h, v8.8h, v9.8h}, [x4], #64 \n"
                    "st1    {v10.8h, v11.8h}, [x4]      \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x12
                    "uzp1   v20.8h, v0.8h, v1.8h        \n"
                    "uzp2   v21.8h, v0.8h, v1.8h        \n"
                    "uzp1   v22.8h, v2.8h, v3.8h        \n"
                    "uzp2   v23.8h, v2.8h, v3.8h        \n"
                    "uzp1   v24.8h, v4.8h, v5.8h        \n"
                    "uzp2   v25.8h, v4.8h, v5.8h        \n"
                    "uzp1   v26.8h, v6.8h, v7.8h        \n"
                    "uzp2   v27.8h, v6.8h, v7.8h        \n"
                    "uzp1   v28.8h, v8.8h, v9.8h        \n"
                    "uzp2   v29.8h, v8.8h, v9.8h        \n"
                    "uzp1   v30.8h, v10.8h, v11.8h      \n"
                    "uzp2   v31.8h, v10.8h, v11.8h      \n"

                    "uzp1   v0.8h, v20.8h, v22.8h       \n"
                    "uzp2   v6.8h, v20.8h, v22.8h       \n"
                    "uzp1   v3.8h, v21.8h, v23.8h       \n"
                    "uzp2   v9.8h, v21.8h, v23.8h       \n"
                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"
                    "mov    v4.d[0], v3.d[1]            \n"
                    "mov    v10.d[0], v9.d[1]           \n"
                    "uzp1   v2.8h, v24.8h, v24.8h       \n"
                    "uzp2   v8.8h, v24.8h, v24.8h       \n"
                    "uzp1   v5.8h, v25.8h, v25.8h       \n"
                    "uzp2   v11.8h, v25.8h, v25.8h      \n"

                    "uzp1   v12.8h, v26.8h, v28.8h      \n"
                    "uzp2   v18.8h, v26.8h, v28.8h      \n"
                    "uzp1   v15.8h, v27.8h, v29.8h      \n"
                    "uzp2   v21.8h, v27.8h, v29.8h      \n"
                    "mov    v13.d[0], v12.d[1]          \n"
                    "mov    v19.d[0], v18.d[1]          \n"
                    "mov    v16.d[0], v15.d[1]          \n"
                    "mov    v22.d[0], v21.d[1]          \n"
                    "uzp1   v14.8h, v30.8h, v30.8h      \n"
                    "uzp2   v20.8h, v30.8h, v30.8h      \n"
                    "uzp1   v17.8h, v31.8h, v31.8h      \n"
                    "uzp2   v23.8h, v31.8h, v31.8h      \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                    "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v12.4h, v13.4h, v14.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v15.4h, v16.4h, v17.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v18.4h, v19.4h, v20.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v21.4h, v22.4h, v23.4h}, [x4] \n"

                    "9:                                 \n"
                    "add    %0, %0, #384                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #320                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v8.4s}, [%8]               \n"
                    "ld1    {v20.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v20.16b, v20.16b, v20.16b   \n"

                    "2:                                 \n"
                    "mov    v9.16b, v8.16b              \n"
                    "mov    v10.16b, v8.16b             \n"
                    "mov    v11.16b, v8.16b             \n"
                    "mov    v12.16b, v8.16b             \n"
                    "mov    v13.16b, v8.16b             \n"
                    "mov    v14.16b, v8.16b             \n"
                    "mov    v15.16b, v8.16b             \n"
                    "mov    v16.16b, v8.16b             \n"
                    "mov    v17.16b, v8.16b             \n"
                    "mov    v18.16b, v8.16b             \n"
                    "mov    v19.16b, v8.16b             \n"

                    "mov    v21.16b, v20.16b            \n"
                    "mov    v22.16b, v20.16b            \n"
                    "mov    v23.16b, v20.16b            \n"
                    "mov    v24.16b, v20.16b            \n"
                    "mov    v25.16b, v20.16b            \n"
                    "mov    v26.16b, v20.16b            \n"
                    "mov    v27.16b, v20.16b            \n"
                    "mov    v28.16b, v20.16b            \n"
                    "mov    v29.16b, v20.16b            \n"
                    "mov    v30.16b, v20.16b            \n"
                    "mov    v31.16b, v20.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%1], #32 \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "shll   v7.4s, v7.4h, #16           \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "shll   v3.4s, v3.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"

                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"

                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v3.s[3]      \n"
                    "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "shll   v3.4s, v3.4h, #16           \n"

                    "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v0.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v0.s[3]      \n"

                    "fmla   v16.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%1], #32 \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "shll   v7.4s, v7.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                    "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v2.s[3]      \n"

                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v3.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "shll   v3.4s, v3.4h, #16           \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v1.s[3]      \n"

                    "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v2.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                    "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v2.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v2.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v16.4s, v6.4s, v3.s[0]      \n"
                    "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v3.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                    "bne    4b                          \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"

                    "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v8.4s, #16           \n"
                    "shrn2  v0.8h, v9.4s, #16           \n"
                    "shrn   v1.4h, v10.4s, #16          \n"
                    "shrn2  v1.8h, v11.4s, #16          \n"
                    "shrn   v2.4h, v12.4s, #16          \n"
                    "shrn2  v2.8h, v13.4s, #16          \n"
                    "shrn   v3.4h, v14.4s, #16          \n"
                    "shrn2  v3.8h, v15.4s, #16          \n"
                    "shrn   v4.4h, v16.4s, #16          \n"
                    "shrn2  v4.8h, v17.4s, #16          \n"
                    "shrn   v5.4h, v18.4s, #16          \n"
                    "shrn2  v5.8h, v19.4s, #16          \n"
                    "shrn   v6.4h, v20.4s, #16          \n"
                    "shrn2  v6.8h, v21.4s, #16          \n"
                    "shrn   v7.4h, v22.4s, #16          \n"
                    "shrn2  v7.8h, v23.4s, #16          \n"
                    "shrn   v8.4h, v24.4s, #16          \n"
                    "shrn2  v8.8h, v25.4s, #16          \n"
                    "shrn   v9.4h, v26.4s, #16          \n"
                    "shrn2  v9.8h, v27.4s, #16          \n"
                    "shrn   v10.4h, v28.4s, #16         \n"
                    "shrn2  v10.8h, v29.4s, #16         \n"
                    "shrn   v11.4h, v30.4s, #16         \n"
                    "shrn2  v11.8h, v31.4s, #16         \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h}, [%3], #32 \n"
                    "st1    {v6.8h, v7.8h, v8.8h, v9.8h}, [x4], #64 \n"
                    "st1    {v10.8h, v11.8h}, [x4]      \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x12
                    "uzp1   v20.8h, v0.8h, v1.8h        \n"
                    "uzp2   v21.8h, v0.8h, v1.8h        \n"
                    "uzp1   v22.8h, v2.8h, v3.8h        \n"
                    "uzp2   v23.8h, v2.8h, v3.8h        \n"
                    "uzp1   v24.8h, v4.8h, v5.8h        \n"
                    "uzp2   v25.8h, v4.8h, v5.8h        \n"
                    "uzp1   v26.8h, v6.8h, v7.8h        \n"
                    "uzp2   v27.8h, v6.8h, v7.8h        \n"
                    "uzp1   v28.8h, v8.8h, v9.8h        \n"
                    "uzp2   v29.8h, v8.8h, v9.8h        \n"
                    "uzp1   v30.8h, v10.8h, v11.8h      \n"
                    "uzp2   v31.8h, v10.8h, v11.8h      \n"

                    "uzp1   v0.8h, v20.8h, v22.8h       \n"
                    "uzp2   v6.8h, v20.8h, v22.8h       \n"
                    "uzp1   v3.8h, v21.8h, v23.8h       \n"
                    "uzp2   v9.8h, v21.8h, v23.8h       \n"
                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"
                    "mov    v4.d[0], v3.d[1]            \n"
                    "mov    v10.d[0], v9.d[1]           \n"
                    "uzp1   v2.8h, v24.8h, v24.8h       \n"
                    "uzp2   v8.8h, v24.8h, v24.8h       \n"
                    "uzp1   v5.8h, v25.8h, v25.8h       \n"
                    "uzp2   v11.8h, v25.8h, v25.8h      \n"

                    "uzp1   v12.8h, v26.8h, v28.8h      \n"
                    "uzp2   v18.8h, v26.8h, v28.8h      \n"
                    "uzp1   v15.8h, v27.8h, v29.8h      \n"
                    "uzp2   v21.8h, v27.8h, v29.8h      \n"
                    "mov    v13.d[0], v12.d[1]          \n"
                    "mov    v19.d[0], v18.d[1]          \n"
                    "mov    v16.d[0], v15.d[1]          \n"
                    "mov    v22.d[0], v21.d[1]          \n"
                    "uzp1   v14.8h, v30.8h, v30.8h      \n"
                    "uzp2   v20.8h, v30.8h, v30.8h      \n"
                    "uzp1   v17.8h, v31.8h, v31.8h      \n"
                    "uzp2   v23.8h, v31.8h, v31.8h      \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                    "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v12.4h, v13.4h, v14.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v15.4h, v16.4h, v17.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v18.4h, v19.4h, v20.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v21.4h, v22.4h, v23.4h}, [x4] \n"

                    "9:                                 \n"
                    "add    %0, %0, #384                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
                if (pC)
                {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4x2_t _cc0 = vzipq_f32(_c0, _c0);
                    float32x4x2_t _cc1 = vzipq_f32(_c1, _c1);
                    _sum00 = _cc0.val[0];
                    _sum01 = _cc0.val[1];
                    _sum10 = _cc1.val[0];
                    _sum11 = _cc1.val[1];
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum10;
                    _sum31 = _sum11;
                    _sum40 = _sum00;
                    _sum41 = _sum01;
                    _sum50 = _sum10;
                    _sum51 = _sum11;
                    _sum60 = _sum00;
                    _sum61 = _sum01;
                    _sum70 = _sum10;
                    _sum71 = _sum11;
                    _sum80 = _sum00;
                    _sum81 = _sum01;
                    _sum90 = _sum10;
                    _sum91 = _sum11;
                    _suma0 = _sum00;
                    _suma1 = _sum01;
                    _sumb0 = _sum10;
                    _sumb1 = _sum11;
#else
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
                    _sum40 = _sum00;
                    _sum41 = _sum01;
                    _sum50 = _sum00;
                    _sum51 = _sum01;
                    _sum60 = _sum00;
                    _sum61 = _sum01;
                    _sum70 = _sum00;
                    _sum71 = _sum01;
                    _sum80 = _sum00;
                    _sum81 = _sum01;
                    _sum90 = _sum00;
                    _sum91 = _sum01;
                    _suma0 = _sum00;
                    _suma1 = _sum01;
                    _sumb0 = _sum00;
                    _sumb1 = _sum01;
#endif
                }
                else
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
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
                _sum80 = vld1q_f32(outptr + 4 * 16);
                _sum81 = vld1q_f32(outptr + 4 * 17);
                _sum90 = vld1q_f32(outptr + 4 * 18);
                _sum91 = vld1q_f32(outptr + 4 * 19);
                _suma0 = vld1q_f32(outptr + 4 * 20);
                _suma1 = vld1q_f32(outptr + 4 * 21);
                _sumb0 = vld1q_f32(outptr + 4 * 22);
                _sumb1 = vld1q_f32(outptr + 4 * 23);
            }

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

            if (k_end)
            {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4x2_t _r0 = vuzpq_f32(_sum00, _sum01);
                float32x4x2_t _r1 = vuzpq_f32(_sum10, _sum11);
                _sum00 = _r0.val[0];
                _sum10 = _r0.val[1];
                _sum01 = _r1.val[0];
                _sum11 = _r1.val[1];
                _r0 = vuzpq_f32(_sum20, _sum21);
                _r1 = vuzpq_f32(_sum30, _sum31);
                _sum20 = _r0.val[0];
                _sum30 = _r0.val[1];
                _sum21 = _r1.val[0];
                _sum31 = _r1.val[1];
                _r0 = vuzpq_f32(_sum40, _sum41);
                _r1 = vuzpq_f32(_sum50, _sum51);
                _sum40 = _r0.val[0];
                _sum50 = _r0.val[1];
                _sum41 = _r1.val[0];
                _sum51 = _r1.val[1];
                _r0 = vuzpq_f32(_sum60, _sum61);
                _r1 = vuzpq_f32(_sum70, _sum71);
                _sum60 = _r0.val[0];
                _sum70 = _r0.val[1];
                _sum61 = _r1.val[0];
                _sum71 = _r1.val[1];
                _r0 = vuzpq_f32(_sum80, _sum81);
                _r1 = vuzpq_f32(_sum90, _sum91);
                _sum80 = _r0.val[0];
                _sum90 = _r0.val[1];
                _sum81 = _r1.val[0];
                _sum91 = _r1.val[1];
                _r0 = vuzpq_f32(_suma0, _suma1);
                _r1 = vuzpq_f32(_sumb0, _sumb1);
                _suma0 = _r0.val[0];
                _sumb0 = _r0.val[1];
                _suma1 = _r1.val[0];
                _sumb1 = _r1.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum40));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum50));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum60));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum70));
                    vst1_u16(outptr0 + 4 * 8, float2bfloat(_sum80));
                    vst1_u16(outptr0 + 4 * 9, float2bfloat(_sum90));
                    vst1_u16(outptr0 + 4 * 10, float2bfloat(_suma0));
                    vst1_u16(outptr0 + 4 * 11, float2bfloat(_sumb0));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 4, float2bfloat(_sum41));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 5, float2bfloat(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 6, float2bfloat(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 7, float2bfloat(_sum71));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 8, float2bfloat(_sum81));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 9, float2bfloat(_sum91));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 10, float2bfloat(_suma1));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 11, float2bfloat(_sumb1));

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    uint16x8_t _t4 = vcombine_u16(float2bfloat(_sum40), float2bfloat(_sum41));
                    uint16x8_t _t5 = vcombine_u16(float2bfloat(_sum50), float2bfloat(_sum51));
                    uint16x8_t _t6 = vcombine_u16(float2bfloat(_sum60), float2bfloat(_sum61));
                    uint16x8_t _t7 = vcombine_u16(float2bfloat(_sum70), float2bfloat(_sum71));
                    uint16x8_t _t8 = vcombine_u16(float2bfloat(_sum80), float2bfloat(_sum81));
                    uint16x8_t _t9 = vcombine_u16(float2bfloat(_sum90), float2bfloat(_sum91));
                    uint16x8_t _ta = vcombine_u16(float2bfloat(_suma0), float2bfloat(_suma1));
                    uint16x8_t _tb = vcombine_u16(float2bfloat(_sumb0), float2bfloat(_sumb1));
                    transpose8x12_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _ta, _tb);

                    vst1_u16(outptr0, vget_low_u16(_t0));
                    vst1_u16(outptr0 + 4, vget_high_u16(_t0));
                    vst1_u16(outptr0 + 8, vget_low_u16(_t1));
                    vst1_u16(outptr0 + out_hstep, vget_high_u16(_t1));
                    vst1_u16(outptr0 + out_hstep + 4, vget_low_u16(_t2));
                    vst1_u16(outptr0 + out_hstep + 8, vget_high_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 2, vget_low_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 2 + 4, vget_high_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 2 + 8, vget_low_u16(_t4));
                    vst1_u16(outptr0 + out_hstep * 3, vget_high_u16(_t4));
                    vst1_u16(outptr0 + out_hstep * 3 + 4, vget_low_u16(_t5));
                    vst1_u16(outptr0 + out_hstep * 3 + 8, vget_high_u16(_t5));
                    vst1_u16(outptr0 + out_hstep * 4, vget_low_u16(_t6));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, vget_high_u16(_t6));
                    vst1_u16(outptr0 + out_hstep * 4 + 8, vget_low_u16(_t7));
                    vst1_u16(outptr0 + out_hstep * 5, vget_high_u16(_t7));
                    vst1_u16(outptr0 + out_hstep * 5 + 4, vget_low_u16(_t8));
                    vst1_u16(outptr0 + out_hstep * 5 + 8, vget_high_u16(_t8));
                    vst1_u16(outptr0 + out_hstep * 6, vget_low_u16(_t9));
                    vst1_u16(outptr0 + out_hstep * 6 + 4, vget_high_u16(_t9));
                    vst1_u16(outptr0 + out_hstep * 6 + 8, vget_low_u16(_ta));
                    vst1_u16(outptr0 + out_hstep * 7, vget_high_u16(_ta));
                    vst1_u16(outptr0 + out_hstep * 7 + 4, vget_low_u16(_tb));
                    vst1_u16(outptr0 + out_hstep * 7 + 8, vget_high_u16(_tb));

                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
                vst1q_f32(outptr + 4 * 16, _sum80);
                vst1q_f32(outptr + 4 * 17, _sum81);
                vst1q_f32(outptr + 4 * 18, _sum90);
                vst1q_f32(outptr + 4 * 19, _sum91);
                vst1q_f32(outptr + 4 * 20, _suma0);
                vst1q_f32(outptr + 4 * 21, _suma1);
                vst1q_f32(outptr + 4 * 22, _sumb0);
                vst1q_f32(outptr + 4 * 23, _sumb1);
            }

            outptr += 96;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (use_a53_a55_optimized_kernel && cpu_support_arm_asimdhp())
            {
                // a55
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #192                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v16.4s}, [%8]              \n"
                    "ld1    {v24.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"

                    "2:                                 \n"
                    "mov    v17.16b, v16.16b            \n"
                    "mov    v18.16b, v16.16b            \n"
                    "mov    v19.16b, v16.16b            \n"
                    "mov    v20.16b, v16.16b            \n"
                    "mov    v21.16b, v16.16b            \n"
                    "mov    v22.16b, v16.16b            \n"
                    "mov    v23.16b, v16.16b            \n"

                    "mov    v25.16b, v24.16b            \n"
                    "mov    v26.16b, v24.16b            \n"
                    "mov    v27.16b, v24.16b            \n"
                    "mov    v28.16b, v24.16b            \n"
                    "mov    v29.16b, v24.16b            \n"
                    "mov    v30.16b, v24.16b            \n"
                    "mov    v31.16b, v24.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v8.4h, v9.4h}, [%1], #16   \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v8.4s, v8.4h, #16           \n"

                    ".align 4                           \n"
                    "4:                                 \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "ldr    d10, [%1], #8               \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                    "ldr    d11, [%1], #8               \n"
                    "fmla   v19.4s, v8.4s, v0.s[3]      \n"
                    "shll   v9.4s, v9.4h, #16           \n"
                    "fmla   v20.4s, v8.4s, v1.s[0]      \n"
                    "ldr    d4, [%2], #8                \n"
                    "fmla   v21.4s, v8.4s, v1.s[1]      \n"
                    "ldr    d12, [%1], #8               \n"
                    "fmla   v22.4s, v8.4s, v1.s[2]      \n"
                    "ldr    d5, [%2], #8                \n"
                    "fmla   v23.4s, v8.4s, v1.s[3]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v24.4s, v9.4s, v0.s[0]      \n"
                    "ldr    d13, [%1], #8               \n"
                    "fmla   v25.4s, v9.4s, v0.s[1]      \n"
                    "ldr    d6, [%2], #8                \n"
                    "fmla   v26.4s, v9.4s, v0.s[2]      \n"
                    "ldr    d14, [%1], #8               \n"
                    "fmla   v27.4s, v9.4s, v0.s[3]      \n"
                    "shll   v10.4s, v10.4h, #16         \n"
                    "fmla   v28.4s, v9.4s, v1.s[0]      \n"
                    "ldr    d7, [%2], #8                \n"
                    "fmla   v29.4s, v9.4s, v1.s[1]      \n"
                    "ldr    d15, [%1], #8               \n"
                    "fmla   v30.4s, v9.4s, v1.s[2]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                    "fmla   v31.4s, v9.4s, v1.s[3]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                    "ldr    d8, [%1], #8                \n"
                    "fmla   v17.4s, v10.4s, v2.s[1]     \n"
                    "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                    "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v19.4s, v10.4s, v2.s[3]     \n"
                    "shll   v11.4s, v11.4h, #16         \n"
                    "fmla   v20.4s, v10.4s, v3.s[0]     \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v21.4s, v10.4s, v3.s[1]     \n"
                    "ldr    d9, [%1], #8                \n"
                    "fmla   v22.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v23.4s, v10.4s, v3.s[3]     \n"
                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v24.4s, v11.4s, v2.s[0]     \n"
                    "fmla   v25.4s, v11.4s, v2.s[1]     \n"
                    "fmla   v26.4s, v11.4s, v2.s[2]     \n"
                    "fmla   v27.4s, v11.4s, v2.s[3]     \n"
                    "shll   v12.4s, v12.4h, #16         \n"
                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v16.4s, v12.4s, v4.s[0]     \n"
                    "fmla   v17.4s, v12.4s, v4.s[1]     \n"
                    "fmla   v18.4s, v12.4s, v4.s[2]     \n"
                    "fmla   v19.4s, v12.4s, v4.s[3]     \n"
                    "shll   v13.4s, v13.4h, #16         \n"
                    "fmla   v20.4s, v12.4s, v5.s[0]     \n"
                    "fmla   v21.4s, v12.4s, v5.s[1]     \n"
                    "fmla   v22.4s, v12.4s, v5.s[2]     \n"
                    "fmla   v23.4s, v12.4s, v5.s[3]     \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v24.4s, v13.4s, v4.s[0]     \n"
                    "fmla   v25.4s, v13.4s, v4.s[1]     \n"
                    "fmla   v26.4s, v13.4s, v4.s[2]     \n"
                    "fmla   v27.4s, v13.4s, v4.s[3]     \n"
                    "shll   v14.4s, v14.4h, #16         \n"
                    "fmla   v28.4s, v13.4s, v5.s[0]     \n"
                    "fmla   v29.4s, v13.4s, v5.s[1]     \n"
                    "fmla   v30.4s, v13.4s, v5.s[2]     \n"
                    "fmla   v31.4s, v13.4s, v5.s[3]     \n"
                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                    "fmla   v17.4s, v14.4s, v6.s[1]     \n"
                    "fmla   v18.4s, v14.4s, v6.s[2]     \n"
                    "fmla   v19.4s, v14.4s, v6.s[3]     \n"
                    "shll   v15.4s, v15.4h, #16         \n"
                    "fmla   v20.4s, v14.4s, v7.s[0]     \n"
                    "fmla   v21.4s, v14.4s, v7.s[1]     \n"
                    "fmla   v22.4s, v14.4s, v7.s[2]     \n"
                    "fmla   v23.4s, v14.4s, v7.s[3]     \n"
                    "shll   v8.4s, v8.4h, #16           \n"
                    "fmla   v24.4s, v15.4s, v6.s[0]     \n"
                    "fmla   v25.4s, v15.4s, v6.s[1]     \n"
                    "fmla   v26.4s, v15.4s, v6.s[2]     \n"
                    "fmla   v27.4s, v15.4s, v6.s[3]     \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v28.4s, v15.4s, v7.s[0]     \n"
                    "fmla   v29.4s, v15.4s, v7.s[1]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v15.4s, v7.s[2]     \n"
                    "fmla   v31.4s, v15.4s, v7.s[3]     \n"
                    "bne    4b                          \n"

                    "sub    %1, %1, #16                 \n"
                    "sub    %2, %2, #24                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v2.8h}, [%2], #16          \n"
                    "shll   v0.4s, v2.4h, #16           \n"
                    "shll2  v1.4s, v2.8h, #16           \n"
                    "ld1    {v3.8h}, [%1], #16          \n"
                    "shll   v4.4s, v3.4h, #16           \n"
                    "shll2  v5.4s, v3.8h, #16           \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v4.4s, v1.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v24.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v1.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v16.4s, #16          \n"
                    "shrn2  v0.8h, v17.4s, #16          \n"
                    "shrn   v1.4h, v18.4s, #16          \n"
                    "shrn2  v1.8h, v19.4s, #16          \n"
                    "shrn   v2.4h, v20.4s, #16          \n"
                    "shrn2  v2.8h, v21.4s, #16          \n"
                    "shrn   v3.4h, v22.4s, #16          \n"
                    "shrn2  v3.8h, v23.4s, #16          \n"
                    "shrn   v4.4h, v24.4s, #16          \n"
                    "shrn2  v4.8h, v25.4s, #16          \n"
                    "shrn   v5.4h, v26.4s, #16          \n"
                    "shrn2  v5.8h, v27.4s, #16          \n"
                    "shrn   v6.4h, v28.4s, #16          \n"
                    "shrn2  v6.8h, v29.4s, #16          \n"
                    "shrn   v7.4h, v30.4s, #16          \n"
                    "shrn2  v7.8h, v31.4s, #16          \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h, v6.8h, v7.8h}, [x4] \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x8
                    "uzp1   v24.8h, v0.8h, v1.8h        \n"
                    "uzp2   v25.8h, v0.8h, v1.8h        \n"
                    "uzp1   v26.8h, v2.8h, v3.8h        \n"
                    "uzp2   v27.8h, v2.8h, v3.8h        \n"
                    "uzp1   v28.8h, v4.8h, v5.8h        \n"
                    "uzp2   v29.8h, v4.8h, v5.8h        \n"
                    "uzp1   v30.8h, v6.8h, v7.8h        \n"
                    "uzp2   v31.8h, v6.8h, v7.8h        \n"

                    "uzp1   v0.8h, v24.8h, v26.8h       \n"
                    "uzp2   v2.8h, v24.8h, v26.8h       \n"
                    "uzp1   v1.8h, v25.8h, v27.8h       \n"
                    "uzp2   v3.8h, v25.8h, v27.8h       \n"

                    "uzp1   v4.8h, v28.8h, v30.8h       \n"
                    "uzp2   v6.8h, v28.8h, v30.8h       \n"
                    "uzp1   v5.8h, v29.8h, v31.8h       \n"
                    "uzp2   v7.8h, v29.8h, v31.8h       \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.8h}, [%3], #16          \n"
                    "st1    {v1.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v2.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v4.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v5.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v7.8h}, [x4]               \n"

                    "9:                                 \n"
                    "add    %0, %0, #256                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #192                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v16.4s}, [%8]              \n"
                    "ld1    {v24.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"

                    "2:                                 \n"
                    "mov    v17.16b, v16.16b            \n"
                    "mov    v18.16b, v16.16b            \n"
                    "mov    v19.16b, v16.16b            \n"
                    "mov    v20.16b, v16.16b            \n"
                    "mov    v21.16b, v16.16b            \n"
                    "mov    v22.16b, v16.16b            \n"
                    "mov    v23.16b, v16.16b            \n"

                    "mov    v25.16b, v24.16b            \n"
                    "mov    v26.16b, v24.16b            \n"
                    "mov    v27.16b, v24.16b            \n"
                    "mov    v28.16b, v24.16b            \n"
                    "mov    v29.16b, v24.16b            \n"
                    "mov    v30.16b, v24.16b            \n"
                    "mov    v31.16b, v24.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n"
                    "shll   v0.4s, v4.4h, #16           \n"
                    "shll2  v1.4s, v4.8h, #16           \n"
                    "shll   v2.4s, v5.4h, #16           \n"
                    "shll2  v3.4s, v5.8h, #16           \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                    "shll   v8.4s, v12.4h, #16          \n"
                    "shll2  v9.4s, v12.8h, #16          \n"
                    "shll   v10.4s, v13.4h, #16         \n"
                    "shll2  v11.4s, v13.8h, #16         \n"
                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v8.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v8.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v8.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v8.4s, v1.s[3]      \n"
                    "fmla   v24.4s, v9.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v9.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v9.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v9.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v9.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v9.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v9.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v9.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                    "fmla   v17.4s, v10.4s, v2.s[1]     \n"
                    "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v10.4s, v2.s[3]     \n"
                    "fmla   v20.4s, v10.4s, v3.s[0]     \n"
                    "fmla   v21.4s, v10.4s, v3.s[1]     \n"
                    "fmla   v22.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v23.4s, v10.4s, v3.s[3]     \n"
                    "fmla   v24.4s, v11.4s, v2.s[0]     \n"
                    "fmla   v25.4s, v11.4s, v2.s[1]     \n"
                    "fmla   v26.4s, v11.4s, v2.s[2]     \n"
                    "fmla   v27.4s, v11.4s, v2.s[3]     \n"
                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "shll   v4.4s, v6.4h, #16           \n"
                    "shll2  v5.4s, v6.8h, #16           \n"
                    "shll   v6.4s, v7.4h, #16           \n"
                    "shll2  v7.4s, v7.8h, #16           \n"
                    "shll   v12.4s, v14.4h, #16         \n"
                    "shll2  v13.4s, v14.8h, #16         \n"
                    "shll   v14.4s, v15.4h, #16         \n"
                    "shll2  v15.4s, v15.8h, #16         \n"
                    "fmla   v16.4s, v12.4s, v4.s[0]     \n"
                    "fmla   v17.4s, v12.4s, v4.s[1]     \n"
                    "fmla   v18.4s, v12.4s, v4.s[2]     \n"
                    "fmla   v19.4s, v12.4s, v4.s[3]     \n"
                    "fmla   v20.4s, v12.4s, v5.s[0]     \n"
                    "fmla   v21.4s, v12.4s, v5.s[1]     \n"
                    "fmla   v22.4s, v12.4s, v5.s[2]     \n"
                    "fmla   v23.4s, v12.4s, v5.s[3]     \n"
                    "fmla   v24.4s, v13.4s, v4.s[0]     \n"
                    "fmla   v25.4s, v13.4s, v4.s[1]     \n"
                    "fmla   v26.4s, v13.4s, v4.s[2]     \n"
                    "fmla   v27.4s, v13.4s, v4.s[3]     \n"
                    "fmla   v28.4s, v13.4s, v5.s[0]     \n"
                    "fmla   v29.4s, v13.4s, v5.s[1]     \n"
                    "fmla   v30.4s, v13.4s, v5.s[2]     \n"
                    "fmla   v31.4s, v13.4s, v5.s[3]     \n"
                    "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                    "fmla   v17.4s, v14.4s, v6.s[1]     \n"
                    "fmla   v18.4s, v14.4s, v6.s[2]     \n"
                    "fmla   v19.4s, v14.4s, v6.s[3]     \n"
                    "fmla   v20.4s, v14.4s, v7.s[0]     \n"
                    "fmla   v21.4s, v14.4s, v7.s[1]     \n"
                    "fmla   v22.4s, v14.4s, v7.s[2]     \n"
                    "fmla   v23.4s, v14.4s, v7.s[3]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v24.4s, v15.4s, v6.s[0]     \n"
                    "fmla   v25.4s, v15.4s, v6.s[1]     \n"
                    "fmla   v26.4s, v15.4s, v6.s[2]     \n"
                    "fmla   v27.4s, v15.4s, v6.s[3]     \n"
                    "fmla   v28.4s, v15.4s, v7.s[0]     \n"
                    "fmla   v29.4s, v15.4s, v7.s[1]     \n"
                    "fmla   v30.4s, v15.4s, v7.s[2]     \n"
                    "fmla   v31.4s, v15.4s, v7.s[3]     \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v2.8h}, [%2], #16          \n"
                    "shll   v0.4s, v2.4h, #16           \n"
                    "shll2  v1.4s, v2.8h, #16           \n"
                    "ld1    {v3.8h}, [%1], #16          \n"
                    "shll   v4.4s, v3.4h, #16           \n"
                    "shll2  v5.4s, v3.8h, #16           \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v4.4s, v1.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v24.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v1.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v16.4s, #16          \n"
                    "shrn2  v0.8h, v17.4s, #16          \n"
                    "shrn   v1.4h, v18.4s, #16          \n"
                    "shrn2  v1.8h, v19.4s, #16          \n"
                    "shrn   v2.4h, v20.4s, #16          \n"
                    "shrn2  v2.8h, v21.4s, #16          \n"
                    "shrn   v3.4h, v22.4s, #16          \n"
                    "shrn2  v3.8h, v23.4s, #16          \n"
                    "shrn   v4.4h, v24.4s, #16          \n"
                    "shrn2  v4.8h, v25.4s, #16          \n"
                    "shrn   v5.4h, v26.4s, #16          \n"
                    "shrn2  v5.8h, v27.4s, #16          \n"
                    "shrn   v6.4h, v28.4s, #16          \n"
                    "shrn2  v6.8h, v29.4s, #16          \n"
                    "shrn   v7.4h, v30.4s, #16          \n"
                    "shrn2  v7.8h, v31.4s, #16          \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h, v6.8h, v7.8h}, [x4] \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x8
                    "uzp1   v24.8h, v0.8h, v1.8h        \n"
                    "uzp2   v25.8h, v0.8h, v1.8h        \n"
                    "uzp1   v26.8h, v2.8h, v3.8h        \n"
                    "uzp2   v27.8h, v2.8h, v3.8h        \n"
                    "uzp1   v28.8h, v4.8h, v5.8h        \n"
                    "uzp2   v29.8h, v4.8h, v5.8h        \n"
                    "uzp1   v30.8h, v6.8h, v7.8h        \n"
                    "uzp2   v31.8h, v6.8h, v7.8h        \n"

                    "uzp1   v0.8h, v24.8h, v26.8h       \n"
                    "uzp2   v2.8h, v24.8h, v26.8h       \n"
                    "uzp1   v1.8h, v25.8h, v27.8h       \n"
                    "uzp2   v3.8h, v25.8h, v27.8h       \n"

                    "uzp1   v4.8h, v28.8h, v30.8h       \n"
                    "uzp2   v6.8h, v28.8h, v30.8h       \n"
                    "uzp1   v5.8h, v29.8h, v31.8h       \n"
                    "uzp2   v7.8h, v29.8h, v31.8h       \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.8h}, [%3], #16          \n"
                    "st1    {v1.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v2.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v4.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v5.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v7.8h}, [x4]               \n"

                    "9:                                 \n"
                    "add    %0, %0, #256                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
                if (pC)
                {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4x2_t _cc0 = vzipq_f32(_c0, _c0);
                    float32x4x2_t _cc1 = vzipq_f32(_c1, _c1);
                    _sum00 = _cc0.val[0];
                    _sum01 = _cc0.val[1];
                    _sum10 = _cc1.val[0];
                    _sum11 = _cc1.val[1];
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum10;
                    _sum31 = _sum11;
                    _sum40 = _sum00;
                    _sum41 = _sum01;
                    _sum50 = _sum10;
                    _sum51 = _sum11;
                    _sum60 = _sum00;
                    _sum61 = _sum01;
                    _sum70 = _sum10;
                    _sum71 = _sum11;
#else
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
                    _sum40 = _sum00;
                    _sum41 = _sum01;
                    _sum50 = _sum00;
                    _sum51 = _sum01;
                    _sum60 = _sum00;
                    _sum61 = _sum01;
                    _sum70 = _sum00;
                    _sum71 = _sum01;
#endif
                }
                else
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
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
            }

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

            if (k_end)
            {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4x2_t _r0 = vuzpq_f32(_sum00, _sum01);
                float32x4x2_t _r1 = vuzpq_f32(_sum10, _sum11);
                _sum00 = _r0.val[0];
                _sum10 = _r0.val[1];
                _sum01 = _r1.val[0];
                _sum11 = _r1.val[1];
                _r0 = vuzpq_f32(_sum20, _sum21);
                _r1 = vuzpq_f32(_sum30, _sum31);
                _sum20 = _r0.val[0];
                _sum30 = _r0.val[1];
                _sum21 = _r1.val[0];
                _sum31 = _r1.val[1];
                _r0 = vuzpq_f32(_sum40, _sum41);
                _r1 = vuzpq_f32(_sum50, _sum51);
                _sum40 = _r0.val[0];
                _sum50 = _r0.val[1];
                _sum41 = _r1.val[0];
                _sum51 = _r1.val[1];
                _r0 = vuzpq_f32(_sum60, _sum61);
                _r1 = vuzpq_f32(_sum70, _sum71);
                _sum60 = _r0.val[0];
                _sum70 = _r0.val[1];
                _sum61 = _r1.val[0];
                _sum71 = _r1.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum40));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum50));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum60));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum70));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 4, float2bfloat(_sum41));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 5, float2bfloat(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 6, float2bfloat(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 7, float2bfloat(_sum71));

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    uint16x8_t _t4 = vcombine_u16(float2bfloat(_sum40), float2bfloat(_sum41));
                    uint16x8_t _t5 = vcombine_u16(float2bfloat(_sum50), float2bfloat(_sum51));
                    uint16x8_t _t6 = vcombine_u16(float2bfloat(_sum60), float2bfloat(_sum61));
                    uint16x8_t _t7 = vcombine_u16(float2bfloat(_sum70), float2bfloat(_sum71));
                    transpose8x8_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7);

                    vst1q_u16(outptr0, _t0);
                    vst1q_u16(outptr0 + out_hstep, _t1);
                    vst1q_u16(outptr0 + out_hstep * 2, _t2);
                    vst1q_u16(outptr0 + out_hstep * 3, _t3);
                    vst1q_u16(outptr0 + out_hstep * 4, _t4);
                    vst1q_u16(outptr0 + out_hstep * 5, _t5);
                    vst1q_u16(outptr0 + out_hstep * 6, _t6);
                    vst1q_u16(outptr0 + out_hstep * 7, _t7);

                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
            }

            outptr += 64;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            if (use_a53_a55_optimized_kernel && cpu_support_arm_asimdhp())
            {
                // a55
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #64                 \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v24.4s}, [%8]              \n"
                    "ld1    {v28.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"

                    "2:                                 \n"
                    "mov    v25.16b, v24.16b            \n"
                    "mov    v26.16b, v24.16b            \n"
                    "mov    v27.16b, v24.16b            \n"

                    "mov    v29.16b, v28.16b            \n"
                    "mov    v30.16b, v28.16b            \n"
                    "mov    v31.16b, v28.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.4h, v5.4h, v6.4h}, [%1], #24 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4h, v1.4h}, [%2], #16   \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v4.4s, v4.4h, #16           \n"

                    ".align 4                           \n"
                    "4:                                 \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "ldr    d7, [%1], #8                \n"
                    "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "ldr    d8, [%1], #8                \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "ldr    d9, [%1], #8                \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v24.4s, v6.4s, v1.s[0]      \n"
                    "ldr    d10, [%1], #8               \n"
                    "fmla   v25.4s, v6.4s, v1.s[1]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v26.4s, v6.4s, v1.s[2]      \n"
                    "ldr    d11, [%1], #8               \n"
                    "fmla   v27.4s, v6.4s, v1.s[3]      \n"
                    "shll   v8.4s, v8.4h, #16           \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "ldr    d4, [%1], #8                \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "prfm   pldl1keep, [%2, #256]       \n" // NOTE PRELOAD
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                    "shll   v9.4s, v9.4h, #16           \n"
                    "fmla   v24.4s, v8.4s, v2.s[0]      \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v25.4s, v8.4s, v2.s[1]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v26.4s, v8.4s, v2.s[2]      \n"
                    "ldr    d5, [%1], #8                \n"
                    "fmla   v27.4s, v8.4s, v2.s[3]      \n"
                    "shll   v10.4s, v10.4h, #16         \n"
                    "fmla   v28.4s, v9.4s, v2.s[0]      \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v29.4s, v9.4s, v2.s[1]      \n"
                    "ldr    d6, [%1], #8                \n"
                    "fmla   v30.4s, v9.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v9.4s, v2.s[3]      \n"
                    "shll   v11.4s, v11.4h, #16         \n"
                    "fmla   v24.4s, v10.4s, v3.s[0]     \n"
                    "fmla   v25.4s, v10.4s, v3.s[1]     \n"
                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v26.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v27.4s, v10.4s, v3.s[3]     \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "bne    4b                          \n"

                    "sub    %1, %1, #24                 \n"
                    "sub    %2, %2, #16                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h}, [%2], #8           \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "ld1    {v3.8h}, [%1], #16          \n"
                    "shll   v4.4s, v3.4h, #16           \n"
                    "shll2  v5.4s, v3.8h, #16           \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v24.4s, #16          \n"
                    "shrn2  v0.8h, v25.4s, #16          \n"
                    "shrn   v1.4h, v26.4s, #16          \n"
                    "shrn2  v1.8h, v27.4s, #16          \n"
                    "shrn   v2.4h, v28.4s, #16          \n"
                    "shrn2  v2.8h, v29.4s, #16          \n"
                    "shrn   v3.4h, v30.4s, #16          \n"
                    "shrn2  v3.8h, v31.4s, #16          \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h}, [%3], #32   \n"
                    "st1    {v2.8h, v3.8h}, [x4]        \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x4
                    "uzp1   v28.8h, v0.8h, v1.8h        \n"
                    "uzp2   v29.8h, v0.8h, v1.8h        \n"
                    "uzp1   v30.8h, v2.8h, v3.8h        \n"
                    "uzp2   v31.8h, v2.8h, v3.8h        \n"

                    "uzp1   v0.8h, v28.8h, v29.8h       \n"
                    "uzp2   v2.8h, v28.8h, v29.8h       \n"
                    "uzp1   v4.8h, v30.8h, v31.8h       \n"
                    "uzp2   v6.8h, v30.8h, v31.8h       \n"

                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v3.d[0], v2.d[1]            \n"
                    "mov    v5.d[0], v4.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h}, [%3], #8           \n"
                    "st1    {v1.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v2.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v4.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v5.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v7.4h}, [x4]               \n"

                    "9:                                 \n"
                    "add    %0, %0, #128                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #64                 \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v24.4s}, [%8]              \n"
                    "ld1    {v28.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"

                    "2:                                 \n"
                    "mov    v25.16b, v24.16b            \n"
                    "mov    v26.16b, v24.16b            \n"
                    "mov    v27.16b, v24.16b            \n"

                    "mov    v29.16b, v28.16b            \n"
                    "mov    v30.16b, v28.16b            \n"
                    "mov    v31.16b, v28.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                    "shll   v4.4s, v12.4h, #16          \n"
                    "shll2  v5.4s, v12.8h, #16          \n"
                    "shll   v6.4s, v13.4h, #16          \n"
                    "shll2  v7.4s, v13.8h, #16          \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                    "shll   v8.4s, v14.4h, #16          \n"
                    "shll2  v9.4s, v14.8h, #16          \n"
                    "shll   v10.4s, v15.4h, #16         \n"
                    "shll2  v11.4s, v15.8h, #16         \n"
                    "fmla   v24.4s, v8.4s, v2.s[0]      \n"
                    "fmla   v25.4s, v8.4s, v2.s[1]      \n"
                    "fmla   v26.4s, v8.4s, v2.s[2]      \n"
                    "fmla   v27.4s, v8.4s, v2.s[3]      \n"
                    "fmla   v28.4s, v9.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v9.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v9.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v9.4s, v2.s[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v24.4s, v10.4s, v3.s[0]     \n"
                    "fmla   v25.4s, v10.4s, v3.s[1]     \n"
                    "fmla   v26.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v27.4s, v10.4s, v3.s[3]     \n"
                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h}, [%2], #8           \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "ld1    {v3.8h}, [%1], #16          \n"
                    "shll   v4.4s, v3.4h, #16           \n"
                    "shll2  v5.4s, v3.8h, #16           \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v24.4s, #16          \n"
                    "shrn2  v0.8h, v25.4s, #16          \n"
                    "shrn   v1.4h, v26.4s, #16          \n"
                    "shrn2  v1.8h, v27.4s, #16          \n"
                    "shrn   v2.4h, v28.4s, #16          \n"
                    "shrn2  v2.8h, v29.4s, #16          \n"
                    "shrn   v3.4h, v30.4s, #16          \n"
                    "shrn2  v3.8h, v31.4s, #16          \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h}, [%3], #32   \n"
                    "st1    {v2.8h, v3.8h}, [x4]        \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x4
                    "uzp1   v28.8h, v0.8h, v1.8h        \n"
                    "uzp2   v29.8h, v0.8h, v1.8h        \n"
                    "uzp1   v30.8h, v2.8h, v3.8h        \n"
                    "uzp2   v31.8h, v2.8h, v3.8h        \n"

                    "uzp1   v0.8h, v28.8h, v29.8h       \n"
                    "uzp2   v2.8h, v28.8h, v29.8h       \n"
                    "uzp1   v4.8h, v30.8h, v31.8h       \n"
                    "uzp2   v6.8h, v30.8h, v31.8h       \n"

                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v3.d[0], v2.d[1]            \n"
                    "mov    v5.d[0], v4.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h}, [%3], #8           \n"
                    "st1    {v1.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v2.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v4.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v5.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v7.4h}, [x4]               \n"

                    "9:                                 \n"
                    "add    %0, %0, #128                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
                if (pC)
                {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4x2_t _cc0 = vzipq_f32(_c0, _c0);
                    float32x4x2_t _cc1 = vzipq_f32(_c1, _c1);
                    _sum00 = _cc0.val[0];
                    _sum01 = _cc0.val[1];
                    _sum10 = _cc1.val[0];
                    _sum11 = _cc1.val[1];
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum10;
                    _sum31 = _sum11;
#else
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
#endif
                }
                else
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
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
            }

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

            if (k_end)
            {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4x2_t _r0 = vuzpq_f32(_sum00, _sum01);
                float32x4x2_t _r1 = vuzpq_f32(_sum10, _sum11);
                _sum00 = _r0.val[0];
                _sum10 = _r0.val[1];
                _sum01 = _r1.val[0];
                _sum11 = _r1.val[1];
                _r0 = vuzpq_f32(_sum20, _sum21);
                _r1 = vuzpq_f32(_sum30, _sum31);
                _sum20 = _r0.val[0];
                _sum30 = _r0.val[1];
                _sum21 = _r1.val[0];
                _sum31 = _r1.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    transpose8x4_u16(_t0, _t1, _t2, _t3);

                    vst1_u16(outptr0, vget_low_u16(_t0));
                    vst1_u16(outptr0 + out_hstep * 1, vget_high_u16(_t0));
                    vst1_u16(outptr0 + out_hstep * 2, vget_low_u16(_t1));
                    vst1_u16(outptr0 + out_hstep * 3, vget_high_u16(_t1));
                    vst1_u16(outptr0 + out_hstep * 4, vget_low_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 5, vget_high_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 6, vget_low_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 7, vget_high_u16(_t3));

                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "add    x4, %8, #16                 \n"
                "ld1    {v28.4s}, [%8]              \n"
                "ld1    {v30.4s}, [x4]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"

                "2:                                 \n"
                "mov    v29.16b, v28.16b            \n"
                "mov    v31.16b, v30.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #128]       \n"
                "ld1    {v0.4h, v1.4h}, [%2], #16   \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                "shll   v4.4s, v12.4h, #16          \n"
                "shll2  v5.4s, v12.8h, #16          \n"
                "shll   v6.4s, v13.4h, #16          \n"
                "shll2  v7.4s, v13.8h, #16          \n"
                "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                "fmla   v29.4s, v4.4s, v0.s[1]      \n"
                "fmla   v30.4s, v5.4s, v0.s[0]      \n"
                "fmla   v31.4s, v5.4s, v0.s[1]      \n"
                "fmla   v28.4s, v6.4s, v0.s[2]      \n"
                "fmla   v29.4s, v6.4s, v0.s[3]      \n"
                "fmla   v30.4s, v7.4s, v0.s[2]      \n"
                "fmla   v31.4s, v7.4s, v0.s[3]      \n"
                "shll   v8.4s, v14.4h, #16          \n"
                "shll2  v9.4s, v14.8h, #16          \n"
                "shll   v10.4s, v15.4h, #16         \n"
                "shll2  v11.4s, v15.8h, #16         \n"
                "fmla   v28.4s, v8.4s, v1.s[0]      \n"
                "fmla   v29.4s, v8.4s, v1.s[1]      \n"
                "fmla   v30.4s, v9.4s, v1.s[0]      \n"
                "fmla   v31.4s, v9.4s, v1.s[1]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v10.4s, v1.s[2]     \n"
                "fmla   v29.4s, v10.4s, v1.s[3]     \n"
                "fmla   v30.4s, v11.4s, v1.s[2]     \n"
                "fmla   v31.4s, v11.4s, v1.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.s}[0], [%2], #4         \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v3.8h}, [%1], #16          \n"
                "shll   v4.4s, v3.4h, #16           \n"
                "shll2  v5.4s, v3.8h, #16           \n"
                "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                "fmla   v29.4s, v4.4s, v0.s[1]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v5.4s, v0.s[0]      \n"
                "fmla   v31.4s, v5.4s, v0.s[1]      \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v28.4s, #16          \n"
                "shrn2  v0.8h, v29.4s, #16          \n"
                "shrn   v1.4h, v30.4s, #16          \n"
                "shrn2  v1.8h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 1          \n"
                "st1    {v0.8h}, [%3], #16          \n"
                "st1    {v1.8h}, [x4]               \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose8x2
                "uzp1   v2.8h, v0.8h, v1.8h         \n"
                "uzp2   v3.8h, v0.8h, v1.8h         \n"
                "uzp1   v0.8h, v2.8h, v3.8h         \n"
                "uzp2   v1.8h, v2.8h, v3.8h         \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.s}[0], [%3], #4         \n"
                "st1    {v0.s}[2], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.s}[0], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.s}[2], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v0.s}[1], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v0.s}[3], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.s}[1], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.s}[3], [x4]             \n"

                "9:                                 \n"
                "add    %0, %0, #64                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v28", "v29", "v30", "v31");
#else // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                if (pC)
                {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4x2_t _cc0 = vzipq_f32(_c0, _c0);
                    float32x4x2_t _cc1 = vzipq_f32(_c1, _c1);
                    _sum00 = _cc0.val[0];
                    _sum01 = _cc0.val[1];
                    _sum10 = _cc1.val[0];
                    _sum11 = _cc1.val[1];
#else
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
#endif
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
            }

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

            if (k_end)
            {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4x2_t _r0 = vuzpq_f32(_sum00, _sum01);
                float32x4x2_t _r1 = vuzpq_f32(_sum10, _sum11);
                _sum00 = _r0.val[0];
                _sum10 = _r0.val[1];
                _sum01 = _r1.val[0];
                _sum11 = _r1.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    unsigned short sum1[8];
                    vst1_u16(sum0, float2bfloat(_sum00));
                    vst1_u16(sum0 + 4, float2bfloat(_sum01));
                    vst1_u16(sum1, float2bfloat(_sum10));
                    vst1_u16(sum1 + 4, float2bfloat(_sum11));

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
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
            }

            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v30.4s, v31.4s}, [%0]      \n"
                "b      2f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v30.4s, v31.4s}, [%8]      \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"

                "2:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "3:                                 \n"
                "prfm   pldl1keep, [%2, #64]        \n"
                "ld1    {v0.4h}, [%2], #8           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                "shll   v4.4s, v12.4h, #16          \n"
                "shll2  v5.4s, v12.8h, #16          \n"
                "shll   v6.4s, v13.4h, #16          \n"
                "shll2  v7.4s, v13.8h, #16          \n"
                "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                "fmla   v29.4s, v5.4s, v0.s[0]      \n"
                "fmla   v30.4s, v6.4s, v0.s[1]      \n"
                "fmla   v31.4s, v7.4s, v0.s[1]      \n"
                "shll   v8.4s, v14.4h, #16          \n"
                "shll2  v9.4s, v14.8h, #16          \n"
                "shll   v10.4s, v15.4h, #16         \n"
                "shll2  v11.4s, v15.8h, #16         \n"
                "fmla   v28.4s, v8.4s, v0.s[2]      \n"
                "fmla   v29.4s, v9.4s, v0.s[2]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v10.4s, v0.s[3]     \n"
                "fmla   v31.4s, v11.4s, v0.s[3]     \n"
                "bne    3b                          \n"
                "fadd   v30.4s, v30.4s, v28.4s      \n"
                "fadd   v31.4s, v31.4s, v29.4s      \n"

                "4:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    6f                          \n"

                "5:                                 \n"
                "ld1r   {v0.4h}, [%2], #2           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v3.8h}, [%1], #16          \n"
                "shll   v4.4s, v3.4h, #16           \n"
                "shll2  v5.4s, v3.8h, #16           \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v4.4s, v0.4s        \n"
                "fmla   v31.4s, v5.4s, v0.4s        \n"
                "bne    5b                          \n"

                "6:                                 \n"
                "shrn   v30.4h, v30.4s, #16         \n"
                "shrn   v31.4h, v31.4s, #16         \n"
                "tst    %w11, #255                  \n"
                "beq    9f                          \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    7f                          \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 1          \n"
                "st1    {v30.4h}, [%3], #8          \n"
                "st1    {v31.4h}, [x4]              \n"
                "b      8f                          \n"

                // if out_elempack == 1
                "7:                                 \n"
                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v30.h}[0], [%3], #2        \n"
                "st1    {v30.h}[1], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v30.h}[2], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v30.h}[3], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.h}[0], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.h}[1], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.h}[2], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.h}[3], [x4]            \n"

                "8:                                 \n"
                "add    %0, %0, #32                 \n"
                "b      10f                         \n"

                "9:                                 \n"
                "st1    {v30.4s, v31.4s}, [%0], #32 \n"

                "10:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v28", "v29", "v30", "v31");
#else // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum00;
            float32x4_t _sum01;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
            }

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

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    vst1_u16(sum0, float2bfloat(_sum00));
                    vst1_u16(sum0 + 4, float2bfloat(_sum01));

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
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
            }

            outptr += 8;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }

        pAT += max_kk * 8;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "subs   %0, %0, #128                \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v20.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"

                "2:                                 \n"
                "mov    v21.16b, v20.16b            \n"
                "mov    v22.16b, v20.16b            \n"
                "mov    v23.16b, v20.16b            \n"
                "mov    v24.16b, v20.16b            \n"
                "mov    v25.16b, v20.16b            \n"
                "mov    v26.16b, v20.16b            \n"
                "mov    v27.16b, v20.16b            \n"
                "mov    v28.16b, v20.16b            \n"
                "mov    v29.16b, v20.16b            \n"
                "mov    v30.16b, v20.16b            \n"
                "mov    v31.16b, v20.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n"
                "shll   v0.4s, v4.4h, #16           \n"
                "shll2  v1.4s, v4.8h, #16           \n"
                "shll   v2.4s, v5.4h, #16           \n"
                "shll2  v3.4s, v5.8h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                "fmla   v21.4s, v16.4s, v0.s[1]     \n"
                "fmla   v22.4s, v16.4s, v0.s[2]     \n"
                "fmla   v23.4s, v16.4s, v0.s[3]     \n"
                "fmla   v24.4s, v16.4s, v1.s[0]     \n"
                "fmla   v25.4s, v16.4s, v1.s[1]     \n"
                "fmla   v26.4s, v16.4s, v1.s[2]     \n"
                "fmla   v27.4s, v16.4s, v1.s[3]     \n"
                "fmla   v28.4s, v16.4s, v2.s[0]     \n"
                "fmla   v29.4s, v16.4s, v2.s[1]     \n"
                "fmla   v30.4s, v16.4s, v2.s[2]     \n"
                "fmla   v31.4s, v16.4s, v2.s[3]     \n"
                "shll   v4.4s, v6.4h, #16           \n"
                "shll2  v5.4s, v6.8h, #16           \n"
                "shll   v6.4s, v7.4h, #16           \n"
                "shll2  v7.4s, v7.8h, #16           \n"
                "fmla   v20.4s, v17.4s, v3.s[0]     \n"
                "fmla   v21.4s, v17.4s, v3.s[1]     \n"
                "fmla   v22.4s, v17.4s, v3.s[2]     \n"
                "fmla   v23.4s, v17.4s, v3.s[3]     \n"
                "fmla   v24.4s, v17.4s, v4.s[0]     \n"
                "fmla   v25.4s, v17.4s, v4.s[1]     \n"
                "fmla   v26.4s, v17.4s, v4.s[2]     \n"
                "fmla   v27.4s, v17.4s, v4.s[3]     \n"
                "fmla   v28.4s, v17.4s, v5.s[0]     \n"
                "fmla   v29.4s, v17.4s, v5.s[1]     \n"
                "fmla   v30.4s, v17.4s, v5.s[2]     \n"
                "fmla   v31.4s, v17.4s, v5.s[3]     \n"
                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"
                "fmla   v20.4s, v18.4s, v6.s[0]     \n"
                "fmla   v21.4s, v18.4s, v6.s[1]     \n"
                "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                "fmla   v23.4s, v18.4s, v6.s[3]     \n"
                "fmla   v24.4s, v18.4s, v7.s[0]     \n"
                "fmla   v25.4s, v18.4s, v7.s[1]     \n"
                "fmla   v26.4s, v18.4s, v7.s[2]     \n"
                "fmla   v27.4s, v18.4s, v7.s[3]     \n"
                "fmla   v28.4s, v18.4s, v0.s[0]     \n"
                "fmla   v29.4s, v18.4s, v0.s[1]     \n"
                "fmla   v30.4s, v18.4s, v0.s[2]     \n"
                "fmla   v31.4s, v18.4s, v0.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v20.4s, v19.4s, v1.s[0]     \n"
                "fmla   v21.4s, v19.4s, v1.s[1]     \n"
                "fmla   v22.4s, v19.4s, v1.s[2]     \n"
                "fmla   v23.4s, v19.4s, v1.s[3]     \n"
                "fmla   v24.4s, v19.4s, v2.s[0]     \n"
                "fmla   v25.4s, v19.4s, v2.s[1]     \n"
                "fmla   v26.4s, v19.4s, v2.s[2]     \n"
                "fmla   v27.4s, v19.4s, v2.s[3]     \n"
                "fmla   v28.4s, v19.4s, v3.s[0]     \n"
                "fmla   v29.4s, v19.4s, v3.s[1]     \n"
                "fmla   v30.4s, v19.4s, v3.s[2]     \n"
                "fmla   v31.4s, v19.4s, v3.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                "fmla   v21.4s, v16.4s, v0.s[1]     \n"
                "fmla   v22.4s, v16.4s, v0.s[2]     \n"
                "fmla   v23.4s, v16.4s, v0.s[3]     \n"
                "fmla   v24.4s, v16.4s, v1.s[0]     \n"
                "fmla   v25.4s, v16.4s, v1.s[1]     \n"
                "fmla   v26.4s, v16.4s, v1.s[2]     \n"
                "fmla   v27.4s, v16.4s, v1.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v16.4s, v2.s[0]     \n"
                "fmla   v29.4s, v16.4s, v2.s[1]     \n"
                "fmla   v30.4s, v16.4s, v2.s[2]     \n"
                "fmla   v31.4s, v16.4s, v2.s[3]     \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v20.4s, #16          \n"
                "shrn2  v0.8h, v21.4s, #16          \n"
                "shrn   v1.4h, v22.4s, #16          \n"
                "shrn2  v1.8h, v23.4s, #16          \n"
                "shrn   v2.4h, v24.4s, #16          \n"
                "shrn2  v2.8h, v25.4s, #16          \n"
                "shrn   v3.4h, v26.4s, #16          \n"
                "shrn2  v3.8h, v27.4s, #16          \n"
                "shrn   v4.4h, v28.4s, #16          \n"
                "shrn2  v4.8h, v29.4s, #16          \n"
                "shrn   v5.4h, v30.4s, #16          \n"
                "shrn2  v5.8h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                "st1    {v4.8h, v5.8h}, [%3], #32   \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose4x12
                "uzp1   v20.8h, v0.8h, v1.8h        \n"
                "uzp2   v21.8h, v0.8h, v1.8h        \n"
                "uzp1   v22.8h, v2.8h, v3.8h        \n"
                "uzp2   v23.8h, v2.8h, v3.8h        \n"
                "uzp1   v24.8h, v4.8h, v5.8h        \n"
                "uzp2   v25.8h, v4.8h, v5.8h        \n"

                "uzp1   v0.8h, v20.8h, v21.8h       \n"
                "uzp2   v6.8h, v20.8h, v21.8h       \n"
                "uzp1   v1.8h, v22.8h, v23.8h       \n"
                "uzp2   v7.8h, v22.8h, v23.8h       \n"
                "uzp1   v2.8h, v24.8h, v25.8h       \n"
                "uzp2   v8.8h, v24.8h, v25.8h       \n"

                "mov    v3.d[0], v0.d[1]            \n"
                "mov    v4.d[0], v1.d[1]            \n"
                "mov    v5.d[0], v2.d[1]            \n"
                "mov    v9.d[0], v6.d[1]            \n"
                "mov    v10.d[0], v7.d[1]           \n"
                "mov    v11.d[0], v8.d[1]           \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"

                "9:                                 \n"
                "add    %0, %0, #192                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
                if (pC)
                {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                    float32x4_t _c = vld1q_f32(pC);
                    float32x4x2_t _cc = vzipq_f32(_c, _c);
                    _sum0 = _cc.val[0];
                    _sum1 = _cc.val[1];
                    _sum2 = _sum0;
                    _sum3 = _sum1;
                    _sum4 = _sum0;
                    _sum5 = _sum1;
                    _sum6 = _sum0;
                    _sum7 = _sum1;
                    _sum8 = _sum0;
                    _sum9 = _sum1;
                    _suma = _sum0;
                    _sumb = _sum1;
#else
                    _sum0 = vld1q_f32(pC);
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
#endif
                }
                else
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
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
                _sum8 = vld1q_f32(outptr + 4 * 8);
                _sum9 = vld1q_f32(outptr + 4 * 9);
                _suma = vld1q_f32(outptr + 4 * 10);
                _sumb = vld1q_f32(outptr + 4 * 11);
            }

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

            if (k_end)
            {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4x2_t _r0 = vuzpq_f32(_sum0, _sum1);
                _sum0 = _r0.val[0];
                _sum1 = _r0.val[1];
                _r0 = vuzpq_f32(_sum2, _sum3);
                _sum2 = _r0.val[0];
                _sum3 = _r0.val[1];
                _r0 = vuzpq_f32(_sum4, _sum5);
                _sum4 = _r0.val[0];
                _sum5 = _r0.val[1];
                _r0 = vuzpq_f32(_sum6, _sum7);
                _sum6 = _r0.val[0];
                _sum7 = _r0.val[1];
                _r0 = vuzpq_f32(_sum8, _sum9);
                _sum8 = _r0.val[0];
                _sum9 = _r0.val[1];
                _r0 = vuzpq_f32(_suma, _sumb);
                _suma = _r0.val[0];
                _sumb = _r0.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum4));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum5));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum6));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum7));
                    vst1_u16(outptr0 + 4 * 8, float2bfloat(_sum8));
                    vst1_u16(outptr0 + 4 * 9, float2bfloat(_sum9));
                    vst1_u16(outptr0 + 4 * 10, float2bfloat(_suma));
                    vst1_u16(outptr0 + 4 * 11, float2bfloat(_sumb));
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    uint16x4_t _t4 = float2bfloat(_sum4);
                    uint16x4_t _t5 = float2bfloat(_sum5);
                    uint16x4_t _t6 = float2bfloat(_sum6);
                    uint16x4_t _t7 = float2bfloat(_sum7);
                    uint16x4_t _t8 = float2bfloat(_sum8);
                    uint16x4_t _t9 = float2bfloat(_sum9);
                    uint16x4_t _ta = float2bfloat(_suma);
                    uint16x4_t _tb = float2bfloat(_sumb);
                    transpose4x12_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _ta, _tb);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + 4, _t1);
                    vst1_u16(outptr0 + 8, _t2);
                    vst1_u16(outptr0 + out_hstep, _t3);
                    vst1_u16(outptr0 + out_hstep + 4, _t4);
                    vst1_u16(outptr0 + out_hstep + 8, _t5);
                    vst1_u16(outptr0 + out_hstep * 2, _t6);
                    vst1_u16(outptr0 + out_hstep * 2 + 4, _t7);
                    vst1_u16(outptr0 + out_hstep * 2 + 8, _t8);
                    vst1_u16(outptr0 + out_hstep * 3, _t9);
                    vst1_u16(outptr0 + out_hstep * 3 + 4, _ta);
                    vst1_u16(outptr0 + out_hstep * 3 + 8, _tb);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
                vst1q_f32(outptr + 4 * 8, _sum8);
                vst1q_f32(outptr + 4 * 9, _sum9);
                vst1q_f32(outptr + 4 * 10, _suma);
                vst1q_f32(outptr + 4 * 11, _sumb);
            }

            outptr += 48;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __aarch64__
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "subs   %0, %0, #64                 \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v24.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"

                "2:                                 \n"
                "mov    v25.16b, v24.16b            \n"
                "mov    v26.16b, v24.16b            \n"
                "mov    v27.16b, v24.16b            \n"
                "mov    v28.16b, v24.16b            \n"
                "mov    v29.16b, v24.16b            \n"
                "mov    v30.16b, v24.16b            \n"
                "mov    v31.16b, v24.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n"
                "shll   v0.4s, v4.4h, #16           \n"
                "shll2  v1.4s, v4.8h, #16           \n"
                "shll   v2.4s, v5.4h, #16           \n"
                "shll2  v3.4s, v5.8h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v24.4s, v16.4s, v0.s[0]     \n"
                "fmla   v25.4s, v16.4s, v0.s[1]     \n"
                "fmla   v26.4s, v16.4s, v0.s[2]     \n"
                "fmla   v27.4s, v16.4s, v0.s[3]     \n"
                "fmla   v28.4s, v16.4s, v1.s[0]     \n"
                "fmla   v29.4s, v16.4s, v1.s[1]     \n"
                "fmla   v30.4s, v16.4s, v1.s[2]     \n"
                "fmla   v31.4s, v16.4s, v1.s[3]     \n"
                "fmla   v24.4s, v17.4s, v2.s[0]     \n"
                "fmla   v25.4s, v17.4s, v2.s[1]     \n"
                "fmla   v26.4s, v17.4s, v2.s[2]     \n"
                "fmla   v27.4s, v17.4s, v2.s[3]     \n"
                "fmla   v28.4s, v17.4s, v3.s[0]     \n"
                "fmla   v29.4s, v17.4s, v3.s[1]     \n"
                "fmla   v30.4s, v17.4s, v3.s[2]     \n"
                "fmla   v31.4s, v17.4s, v3.s[3]     \n"
                "shll   v4.4s, v6.4h, #16           \n"
                "shll2  v5.4s, v6.8h, #16           \n"
                "shll   v6.4s, v7.4h, #16           \n"
                "shll2  v7.4s, v7.8h, #16           \n"
                "fmla   v24.4s, v18.4s, v4.s[0]     \n"
                "fmla   v25.4s, v18.4s, v4.s[1]     \n"
                "fmla   v26.4s, v18.4s, v4.s[2]     \n"
                "fmla   v27.4s, v18.4s, v4.s[3]     \n"
                "fmla   v28.4s, v18.4s, v5.s[0]     \n"
                "fmla   v29.4s, v18.4s, v5.s[1]     \n"
                "fmla   v30.4s, v18.4s, v5.s[2]     \n"
                "fmla   v31.4s, v18.4s, v5.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v24.4s, v19.4s, v6.s[0]     \n"
                "fmla   v25.4s, v19.4s, v6.s[1]     \n"
                "fmla   v26.4s, v19.4s, v6.s[2]     \n"
                "fmla   v27.4s, v19.4s, v6.s[3]     \n"
                "fmla   v28.4s, v19.4s, v7.s[0]     \n"
                "fmla   v29.4s, v19.4s, v7.s[1]     \n"
                "fmla   v30.4s, v19.4s, v7.s[2]     \n"
                "fmla   v31.4s, v19.4s, v7.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4h, v1.4h}, [%2], #16   \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "fmla   v24.4s, v16.4s, v0.s[0]     \n"
                "fmla   v25.4s, v16.4s, v0.s[1]     \n"
                "fmla   v26.4s, v16.4s, v0.s[2]     \n"
                "fmla   v27.4s, v16.4s, v0.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v16.4s, v1.s[0]     \n"
                "fmla   v29.4s, v16.4s, v1.s[1]     \n"
                "fmla   v30.4s, v16.4s, v1.s[2]     \n"
                "fmla   v31.4s, v16.4s, v1.s[3]     \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v24.4s, #16          \n"
                "shrn2  v0.8h, v25.4s, #16          \n"
                "shrn   v1.4h, v26.4s, #16          \n"
                "shrn2  v1.8h, v27.4s, #16          \n"
                "shrn   v2.4h, v28.4s, #16          \n"
                "shrn2  v2.8h, v29.4s, #16          \n"
                "shrn   v3.4h, v30.4s, #16          \n"
                "shrn2  v3.8h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose4x8
                "uzp1   v20.8h, v0.8h, v1.8h        \n"
                "uzp2   v21.8h, v0.8h, v1.8h        \n"
                "uzp1   v22.8h, v2.8h, v3.8h        \n"
                "uzp2   v23.8h, v2.8h, v3.8h        \n"

                "uzp1   v0.8h, v20.8h, v22.8h       \n"
                "uzp2   v2.8h, v20.8h, v22.8h       \n"
                "uzp1   v1.8h, v21.8h, v23.8h       \n"
                "uzp2   v3.8h, v21.8h, v23.8h       \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.8h}, [%3], #16          \n"
                "st1    {v1.8h}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v2.8h}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v3.8h}, [x4]               \n"

                "9:                                 \n"
                "add    %0, %0, #128                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %10, #0             \n"
                "beq        0f                  \n"

                "vldm       %0!, {d16-d23}      \n"
                "vldm       %0, {d24-d31}       \n"
                "sub        %0, %0, #64         \n"
                "b          3f                  \n"

                "0:                             \n"
                // if pC
                "cmp        %8, #0              \n"
                "beq        1f                  \n"

                "vld1.f32   {d16-d17}, [%8]     \n"
                "b          2f                  \n"

                // else
                "1:                             \n"
                "veor       q8, q8              \n"

                "2:                             \n"
                "vmov       q9, q8              \n"
                "vmov       q10, q8             \n"
                "vmov       q11, q8             \n"
                "vmov       q12, q8             \n"
                "vmov       q13, q8             \n"
                "vmov       q14, q8             \n"
                "vmov       q15, q8             \n"

                "3:                             \n"
                "lsr        r4, %9, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                "4:                             \n"
                "pld        [%2, #256]          \n"
                "vld1.u16   {d4-d7}, [%2 :64]!  \n"
                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"
                "vshll.u16  q0, d4, #16         \n"
                "vshll.u16  q1, d5, #16         \n"
                "vshll.u16  q2, d6, #16         \n"
                "vshll.u16  q3, d7, #16         \n"
                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"
                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q4, d0[1]       \n"
                "vmla.f32   q10, q4, d1[0]      \n"
                "vmla.f32   q11, q4, d1[1]      \n"
                "vmla.f32   q12, q4, d2[0]      \n"
                "vmla.f32   q13, q4, d2[1]      \n"
                "vmla.f32   q14, q4, d3[0]      \n"
                "vmla.f32   q15, q4, d3[1]      \n"
                "vmla.f32   q8, q5, d4[0]       \n"
                "vmla.f32   q9, q5, d4[1]       \n"
                "vmla.f32   q10, q5, d5[0]      \n"
                "vmla.f32   q11, q5, d5[1]      \n"
                "vmla.f32   q12, q5, d6[0]      \n"
                "vmla.f32   q13, q5, d6[1]      \n"
                "vmla.f32   q14, q5, d7[0]      \n"
                "vmla.f32   q15, q5, d7[1]      \n"
                "pld        [%2, #256]          \n"
                "vld1.u16   {d4-d7}, [%2 :64]!  \n"
                "vshll.u16  q0, d4, #16         \n"
                "vshll.u16  q1, d5, #16         \n"
                "vshll.u16  q2, d6, #16         \n"
                "vshll.u16  q3, d7, #16         \n"
                "vmla.f32   q8, q6, d0[0]       \n"
                "vmla.f32   q9, q6, d0[1]       \n"
                "vmla.f32   q10, q6, d1[0]      \n"
                "vmla.f32   q11, q6, d1[1]      \n"
                "vmla.f32   q12, q6, d2[0]      \n"
                "vmla.f32   q13, q6, d2[1]      \n"
                "vmla.f32   q14, q6, d3[0]      \n"
                "vmla.f32   q15, q6, d3[1]      \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q8, q7, d4[0]       \n"
                "vmla.f32   q9, q7, d4[1]       \n"
                "vmla.f32   q10, q7, d5[0]      \n"
                "vmla.f32   q11, q7, d5[1]      \n"
                "vmla.f32   q12, q7, d6[0]      \n"
                "vmla.f32   q13, q7, d6[1]      \n"
                "vmla.f32   q14, q7, d7[0]      \n"
                "vmla.f32   q15, q7, d7[1]      \n"
                "bne        4b                  \n"

                "5:                             \n"
                "and        r4, %9, #3          \n" // r4 = remain = max_kk & 3
                "cmp        r4, #0              \n"
                "beq        7f                  \n"

                "6:                             \n"
                "vld1.u16   {d2-d3}, [%2 :64]!  \n"
                "vshll.u16  q0, d2, #16         \n"
                "vshll.u16  q1, d3, #16         \n"
                "vld1.u16   {d9}, [%1 :64]!     \n"
                "vshll.u16  q4, d9, #16         \n"
                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q4, d0[1]       \n"
                "vmla.f32   q10, q4, d1[0]      \n"
                "vmla.f32   q11, q4, d1[1]      \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q12, q4, d2[0]      \n"
                "vmla.f32   q13, q4, d2[1]      \n"
                "vmla.f32   q14, q4, d3[0]      \n"
                "vmla.f32   q15, q4, d3[1]      \n"
                "bne        6b                  \n"

                "7:                             \n"
                "vshrn.u32  d16, q8, #16        \n"
                "vshrn.u32  d17, q9, #16        \n"
                "vshrn.u32  d18, q10, #16       \n"
                "vshrn.u32  d19, q11, #16       \n"
                "vshrn.u32  d20, q12, #16       \n"
                "vshrn.u32  d21, q13, #16       \n"
                "vshrn.u32  d22, q14, #16       \n"
                "vshrn.u32  d23, q15, #16       \n"
                "cmp        %11, #0             \n"
                "beq        10f                 \n"

                // if out_elempack == 4
                "cmp        %12, #4             \n"
                "bne        8f                  \n"

                "vstm       %3!, {d16-d23}      \n"
                "b          9f                  \n"

                // if out_elempack == 1
                "8:                             \n"
                // transpose4x8
                "vuzp.16    q8, q9              \n"
                "vuzp.16    q10, q11            \n"
                "vuzp.16    q8, q10             \n"
                "vuzp.16    q9, q11             \n"

                "add        r4, %3, %13, lsl #1 \n"
                "vst1.u16   {d16-d17}, [%3 :64]! \n"
                "vst1.u16   {d18-d19}, [r4 :64] \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d20-d21}, [r4 :64] \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d22-d23}, [r4 :64] \n"

                "9:                             \n"
                "add        %0, %0, #128        \n"
                "b          11f                 \n"

                "10:                            \n"
                "vstm       %0!, {d16-d23}      \n"
                "vstm       %0!, {d24-d31}      \n"

                "11:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
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
                if (pC)
                {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                    float32x4_t _c = vld1q_f32(pC);
                    float32x4x2_t _cc = vzipq_f32(_c, _c);
                    _sum0 = _cc.val[0];
                    _sum1 = _cc.val[1];
                    _sum2 = _sum0;
                    _sum3 = _sum1;
                    _sum4 = _sum0;
                    _sum5 = _sum1;
                    _sum6 = _sum0;
                    _sum7 = _sum1;
#else
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
#endif
                }
                else
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
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
            }

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

            if (k_end)
            {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4x2_t _r0 = vuzpq_f32(_sum0, _sum1);
                _sum0 = _r0.val[0];
                _sum1 = _r0.val[1];
                _r0 = vuzpq_f32(_sum2, _sum3);
                _sum2 = _r0.val[0];
                _sum3 = _r0.val[1];
                _r0 = vuzpq_f32(_sum4, _sum5);
                _sum4 = _r0.val[0];
                _sum5 = _r0.val[1];
                _r0 = vuzpq_f32(_sum6, _sum7);
                _sum6 = _r0.val[0];
                _sum7 = _r0.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum4));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum5));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum6));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum7));
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    uint16x4_t _t4 = float2bfloat(_sum4);
                    uint16x4_t _t5 = float2bfloat(_sum5);
                    uint16x4_t _t6 = float2bfloat(_sum6);
                    uint16x4_t _t7 = float2bfloat(_sum7);
                    transpose4x8_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + 4, _t1);
                    vst1_u16(outptr0 + out_hstep, _t2);
                    vst1_u16(outptr0 + out_hstep + 4, _t3);
                    vst1_u16(outptr0 + out_hstep * 2, _t4);
                    vst1_u16(outptr0 + out_hstep * 2 + 4, _t5);
                    vst1_u16(outptr0 + out_hstep * 3, _t6);
                    vst1_u16(outptr0 + out_hstep * 3 + 4, _t7);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __aarch64__
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v28.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"

                "2:                                 \n"
                "mov    v29.16b, v28.16b            \n"
                "mov    v30.16b, v28.16b            \n"
                "mov    v31.16b, v28.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                "fmla   v29.4s, v16.4s, v0.s[1]     \n"
                "fmla   v30.4s, v16.4s, v0.s[2]     \n"
                "fmla   v31.4s, v16.4s, v0.s[3]     \n"
                "fmla   v28.4s, v17.4s, v1.s[0]     \n"
                "fmla   v29.4s, v17.4s, v1.s[1]     \n"
                "fmla   v30.4s, v17.4s, v1.s[2]     \n"
                "fmla   v31.4s, v17.4s, v1.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v18.4s, v2.s[0]     \n"
                "fmla   v29.4s, v18.4s, v2.s[1]     \n"
                "fmla   v30.4s, v18.4s, v2.s[2]     \n"
                "fmla   v31.4s, v18.4s, v2.s[3]     \n"
                "fmla   v28.4s, v19.4s, v3.s[0]     \n"
                "fmla   v29.4s, v19.4s, v3.s[1]     \n"
                "fmla   v30.4s, v19.4s, v3.s[2]     \n"
                "fmla   v31.4s, v19.4s, v3.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4h}, [%2], #8           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                "fmla   v29.4s, v16.4s, v0.s[1]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v16.4s, v0.s[2]     \n"
                "fmla   v31.4s, v16.4s, v0.s[3]     \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v28.4s, #16          \n"
                "shrn2  v0.8h, v29.4s, #16          \n"
                "shrn   v1.4h, v30.4s, #16          \n"
                "shrn2  v1.8h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "st1    {v0.8h, v1.8h}, [%3], #32   \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose4x4
                "uzp1   v20.8h, v0.8h, v1.8h        \n"
                "uzp2   v21.8h, v0.8h, v1.8h        \n"

                "uzp1   v0.8h, v20.8h, v21.8h       \n"
                "uzp2   v1.8h, v20.8h, v21.8h       \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.d}[0], [%3], #8         \n"
                "st1    {v0.d}[1], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.d}[0], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.d}[1], [x4]             \n"

                "9:                                 \n"
                "add    %0, %0, #64                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %10, #0             \n"
                "beq        0f                  \n"

                "vldm       %0, {d24-d31}       \n"
                "b          3f                  \n"

                "0:                             \n"
                // if pC
                "cmp        %8, #0              \n"
                "beq        1f                  \n"

                "vld1.f32   {d24-d25}, [%8]     \n"
                "b          2f                  \n"

                // else
                "1:                             \n"
                "veor       q12, q12            \n"

                "2:                             \n"
                "vmov       q13, q12            \n"
                "vmov       q14, q12            \n"
                "vmov       q15, q12            \n"

                "3:                             \n"
                "lsr        r4, %9, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                "4:                             \n"
                "pld        [%2, #256]          \n"
                "vld1.u16   {d4-d7}, [%2 :64]!  \n"
                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"
                "vshll.u16  q0, d4, #16         \n"
                "vshll.u16  q1, d5, #16         \n"
                "vshll.u16  q2, d6, #16         \n"
                "vshll.u16  q3, d7, #16         \n"
                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"
                "vmla.f32   q12, q4, d0[0]      \n"
                "vmla.f32   q13, q4, d0[1]      \n"
                "vmla.f32   q14, q4, d1[0]      \n"
                "vmla.f32   q15, q4, d1[1]      \n"
                "vmla.f32   q12, q5, d2[0]      \n"
                "vmla.f32   q13, q5, d2[1]      \n"
                "vmla.f32   q14, q5, d3[0]      \n"
                "vmla.f32   q15, q5, d3[1]      \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q12, q6, d4[0]      \n"
                "vmla.f32   q13, q6, d4[1]      \n"
                "vmla.f32   q14, q6, d5[0]      \n"
                "vmla.f32   q15, q6, d5[1]      \n"
                "vmla.f32   q12, q7, d6[0]      \n"
                "vmla.f32   q13, q7, d6[1]      \n"
                "vmla.f32   q14, q7, d7[0]      \n"
                "vmla.f32   q15, q7, d7[1]      \n"
                "bne        4b                  \n"

                "5:                             \n"
                "and        r4, %9, #3          \n" // r4 = remain = max_kk & 3
                "cmp        r4, #0              \n"
                "beq        7f                  \n"

                "6:                             \n"
                "vld1.u16   {d0}, [%2 :64]!     \n"
                "vshll.u16  q0, d0, #16         \n"
                "vld1.u16   {d8}, [%1 :64]!     \n"
                "vshll.u16  q4, d8, #16         \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q12, q4, d0[0]      \n"
                "vmla.f32   q13, q4, d0[1]      \n"
                "vmla.f32   q14, q4, d1[0]      \n"
                "vmla.f32   q15, q4, d1[1]      \n"
                "bne        6b                  \n"

                "7:                             \n"
                "vshrn.u32  d24, q12, #16       \n"
                "vshrn.u32  d25, q13, #16       \n"
                "vshrn.u32  d26, q14, #16       \n"
                "vshrn.u32  d27, q15, #16       \n"
                "cmp        %11, #0             \n"
                "beq        10f                 \n"

                // if out_elempack == 4
                "cmp        %12, #4             \n"
                "bne        8f                  \n"

                "vst1.u16   {d24-d27}, [%3]!    \n"
                "b          9f                  \n"

                // if out_elempack == 1
                "8:                             \n"
                // transpose4x4
                "vuzp.16    q12, q13            \n"
                "vuzp.16    q12, q13            \n"

                "add        r4, %3, %13, lsl #1 \n"
                "vst1.u16   {d24}, [%3]!        \n"
                "vst1.u16   {d25}, [r4]         \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d26}, [r4]         \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d27}, [r4]         \n"

                "9:                             \n"
                "add        %0, %0, #64         \n"
                "b          11f                 \n"

                "10:                            \n"
                "vstm       %0!, {d24-d31}      \n"

                "11:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                    float32x4_t _c = vld1q_f32(pC);
                    float32x4x2_t _cc = vzipq_f32(_c, _c);
                    _sum0 = _cc.val[0];
                    _sum1 = _cc.val[1];
                    _sum2 = _sum0;
                    _sum3 = _sum1;
#else
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
#endif
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
            }

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

            if (k_end)
            {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4x2_t _r0 = vuzpq_f32(_sum0, _sum1);
                _sum0 = _r0.val[0];
                _sum1 = _r0.val[1];
                _r0 = vuzpq_f32(_sum2, _sum3);
                _sum2 = _r0.val[0];
                _sum3 = _r0.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    transpose4x4_u16(_t0, _t1, _t2, _t3);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + out_hstep * 1, _t1);
                    vst1_u16(outptr0 + out_hstep * 2, _t2);
                    vst1_u16(outptr0 + out_hstep * 3, _t3);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __aarch64__
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v30.4s, v31.4s}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v30.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"

                "2:                                 \n"
                "mov    v31.16b, v30.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "4:                                 \n"
                "prfm   pldl1keep, [%2, #128]       \n"
                "ld1    {v0.4h, v1.4h}, [%2], #16   \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                "fmla   v29.4s, v16.4s, v0.s[1]     \n"
                "fmla   v30.4s, v17.4s, v0.s[2]     \n"
                "fmla   v31.4s, v17.4s, v0.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v18.4s, v1.s[0]     \n"
                "fmla   v29.4s, v18.4s, v1.s[1]     \n"
                "fmla   v30.4s, v19.4s, v1.s[2]     \n"
                "fmla   v31.4s, v19.4s, v1.s[3]     \n"
                "bne    4b                          \n"
                "fadd   v30.4s, v30.4s, v28.4s      \n"
                "fadd   v31.4s, v31.4s, v29.4s      \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.s}[0], [%2], #4         \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v16.4s, v0.s[0]     \n"
                "fmla   v31.4s, v16.4s, v0.s[1]     \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v30.4s, #16          \n"
                "shrn   v1.4h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "st1    {v0.4h, v1.4h}, [%3], #16   \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose4x2
                "zip1   v30.4h, v0.4h, v1.4h        \n"
                "zip2   v31.4h, v0.4h, v1.4h        \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v30.s}[0], [%3], #4        \n"
                "st1    {v30.s}[1], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.s}[0], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.s}[1], [x4]            \n"

                "9:                                 \n"
                "add    %0, %0, #32                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v30.4s, v31.4s}, [%0], #32 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %10, #0             \n"
                "beq        0f                  \n"

                "vld1.f32   {d28-d31}, [%0 :128] \n"
                "b          3f                  \n"

                "0:                             \n"
                // if pC
                "cmp        %8, #0              \n"
                "beq        1f                  \n"

                "vld1.f32   {d28-d29}, [%8]     \n"
                "b          2f                  \n"

                // else
                "1:                             \n"
                "veor       q14, q14            \n"

                "2:                             \n"
                "vmov       q15, q14            \n"

                "3:                             \n"
                "lsr        r4, %9, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                "veor       q12, q12            \n"
                "veor       q13, q13            \n"
                "4:                             \n"
                "pld        [%2, #128]          \n"
                "vld1.u16   {d2-d3}, [%2 :64]!  \n"
                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"
                "vshll.u16  q0, d2, #16         \n"
                "vshll.u16  q1, d3, #16         \n"
                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"
                "vmla.f32   q12, q4, d0[0]      \n"
                "vmla.f32   q13, q4, d0[1]      \n"
                "vmla.f32   q14, q5, d1[0]      \n"
                "vmla.f32   q15, q5, d1[1]      \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q12, q6, d2[0]      \n"
                "vmla.f32   q13, q6, d2[1]      \n"
                "vmla.f32   q14, q7, d3[0]      \n"
                "vmla.f32   q15, q7, d3[1]      \n"
                "bne        4b                  \n"
                "vadd.f32   q14, q14, q12       \n"
                "vadd.f32   q15, q15, q13       \n"

                "5:                             \n"
                "and        r4, %9, #3          \n" // r4 = remain = max_kk & 3
                "cmp        r4, #0              \n"
                "beq        7f                  \n"

                "6:                             \n"
                "vld1.u32   {d0[0]}, [%2]!      \n"
                "vshll.u16  q0, d0, #16         \n"
                "vld1.u16   {d8}, [%1 :64]!     \n"
                "vshll.u16  q4, d8, #16         \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q14, q4, d0[0]      \n"
                "vmla.f32   q15, q4, d0[1]      \n"
                "bne        6b                  \n"

                "7:                             \n"
                "vshrn.u32  d28, q14, #16       \n"
                "vshrn.u32  d29, q15, #16       \n"
                "cmp        %11, #0             \n"
                "beq        10f                 \n"

                // if out_elempack == 4
                "cmp        %12, #4             \n"
                "bne        8f                  \n"

                "vst1.u16   {d28-d29}, [%3]!    \n"
                "b          9f                  \n"

                // if out_elempack == 1
                "8:                             \n"
                // transpose4x2
                "vzip.16    d28, d29            \n"

                "add        r4, %3, %13, lsl #1 \n"
                "vst1.u32   {d28[0]}, [%3]!     \n"
                "vst1.u32   {d28[1]}, [r4]      \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u32   {d29[0]}, [r4]      \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u32   {d29[1]}, [r4]      \n"

                "9:                             \n"
                "add        %0, %0, #32         \n"
                "b          11f                 \n"

                "10:                            \n"
                "vst1.f32   {d28-d31}, [%0 :128]! \n"

                "11:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                    float32x4_t _c = vld1q_f32(pC);
                    float32x4x2_t _cc = vzipq_f32(_c, _c);
                    _sum0 = _cc.val[0];
                    _sum1 = _cc.val[1];
#else
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
#endif
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

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

            if (k_end)
            {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4x2_t _r0 = vuzpq_f32(_sum0, _sum1);
                _sum0 = _r0.val[0];
                _sum1 = _r0.val[1];
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    unsigned short sum1[4];
                    vst1_u16(sum0, float2bfloat(_sum0));
                    vst1_u16(sum1, float2bfloat(_sum1));

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
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __aarch64__
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v31.4s}, [%0]              \n"
                "b      2f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v31.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"

                "2:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "3:                                 \n"
                "prfm   pldl1keep, [%2, #64]        \n"
                "ld1    {v0.4h}, [%2], #8           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                "fmla   v29.4s, v17.4s, v0.s[1]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v18.4s, v0.s[2]     \n"
                "fmla   v31.4s, v19.4s, v0.s[3]     \n"
                "bne    3b                          \n"
                "fadd   v30.4s, v30.4s, v28.4s      \n"
                "fadd   v31.4s, v31.4s, v29.4s      \n"
                "fadd   v31.4s, v31.4s, v30.4s      \n"

                "4:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    6f                          \n"

                "5:                                 \n"
                "ld1r   {v0.4h}, [%2], #2           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v31.4s, v16.4s, v0.4s       \n"
                "bne    5b                          \n"

                "6:                                 \n"
                "shrn   v0.4h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    9f                          \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    7f                          \n"

                "st1    {v0.4h}, [%3], #8           \n"
                "b      8f                          \n"

                // if out_elempack == 1
                "7:                                 \n"
                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.h}[0], [%3], #2         \n"
                "st1    {v0.h}[1], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v0.h}[2], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v0.h}[3], [x4]             \n"

                "8:                                 \n"
                "add    %0, %0, #16                 \n"
                "b      10f                         \n"

                "9:                                 \n"
                "st1    {v31.4s}, [%0], #16         \n"

                "10:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %10, #0             \n"
                "beq        0f                  \n"

                "vld1.f32   {d30-d31}, [%0 :64] \n"
                "b          2f                  \n"

                "0:                             \n"
                // if pC
                "cmp        %8, #0              \n"
                "beq        1f                  \n"

                "vld1.f32   {d30-d31}, [%8]     \n"
                "b          2f                  \n"

                // else
                "1:                             \n"
                "veor       q15, q15            \n"

                "2:                             \n"
                "lsr        r4, %9, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        4f                  \n"

                "veor       q12, q12            \n"
                "veor       q13, q13            \n"
                "veor       q14, q14            \n"
                "3:                             \n"
                "pld        [%2, #64]           \n"
                "vld1.u16   {d1}, [%2]!         \n"
                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"
                "vshll.u16  q0, d1, #16         \n"
                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"
                "vmla.f32   q12, q4, d0[0]      \n"
                "vmla.f32   q13, q5, d0[1]      \n"
                "vmla.f32   q14, q6, d1[0]      \n"
                "vmla.f32   q15, q7, d1[1]      \n"
                "subs       r4, r4, #1          \n"
                "bne        3b                  \n"
                "vadd.f32   q14, q14, q12       \n"
                "vadd.f32   q15, q15, q13       \n"
                "vadd.f32   q15, q15, q14       \n"

                "4:                             \n"
                "and        r4, %9, #3          \n" // r4 = remain = max_kk & 3
                "cmp        r4, #0              \n"
                "beq        6f                  \n"

                "5:                             \n"
                "vld1.u16   {d0[]}, [%2]!       \n"
                "vshll.u16  q0, d0, #16         \n"
                "vld1.u16   {d8}, [%1 :64]!     \n"
                "vshll.u16  q4, d8, #16         \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q15, q4, q0         \n"
                "bne        5b                  \n"

                "6:                             \n"
                "vshrn.u32  d30, q15, #16       \n"
                "cmp        %11, #0             \n"
                "beq        9f                  \n"

                // if out_elempack == 4
                "cmp        %12, #4             \n"
                "bne        7f                  \n"

                "vst1.u16   {d30}, [%3]!        \n"
                "b          8f                  \n"

                // if out_elempack == 1
                "7:                             \n"

                "add        r4, %3, %13, lsl #1 \n"
                "vst1.u16   {d30[0]}, [%3]!     \n"
                "vst1.u16   {d30[1]}, [r4]      \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d30[2]}, [r4]      \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d30[3]}, [r4]      \n"

                "8:                             \n"
                "add        %0, %0, #16         \n"
                "b          10f                 \n"

                "9:                             \n"
                "vst1.f32   {d30-d31}, [%0 :64]! \n"

                "10:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
            }

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

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    vst1_u16(sum0, float2bfloat(_sum0));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
            }

            outptr += 4;
#endif // NCNN_GNU_INLINE_ASM && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

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
                if (pC)
                {
                    float32x2_t _c = vld1_f32(pC);
                    float32x2x2_t _cc = vzip_f32(_c, _c);
                    float32x4_t _bias = vcombine_f32(_cc.val[0], _cc.val[1]);
                    _sum0 = _bias;
                    _sum1 = _bias;
                    _sum2 = _bias;
                    _sum3 = _bias;
                    _sum4 = _bias;
                    _sum5 = _bias;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                    _sum4 = vdupq_n_f32(0.f);
                    _sum5 = vdupq_n_f32(0.f);
                }
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    uint16x4_t _bf0 = float2bfloat(vcombine_f32(vget_low_f32(_sum0), vget_low_f32(_sum1)));
                    uint16x4_t _bf1 = float2bfloat(vcombine_f32(vget_high_f32(_sum0), vget_high_f32(_sum1)));
                    uint16x4_t _bf2 = float2bfloat(vcombine_f32(vget_low_f32(_sum2), vget_low_f32(_sum3)));
                    uint16x4_t _bf3 = float2bfloat(vcombine_f32(vget_high_f32(_sum2), vget_high_f32(_sum3)));
                    uint16x4_t _bf4 = float2bfloat(vcombine_f32(vget_low_f32(_sum4), vget_low_f32(_sum5)));
                    uint16x4_t _bf5 = float2bfloat(vcombine_f32(vget_high_f32(_sum4), vget_high_f32(_sum5)));
                    vst1q_u16(outptr0, vcombine_u16(_bf0, _bf2));
                    vst1_u16(outptr0 + 8, _bf4);
                    vst1q_u16(outptr0 + out_hstep, vcombine_u16(_bf1, _bf3));
                    vst1_u16(outptr0 + out_hstep + 8, _bf5);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
                vst1q_f32(outptr + 12, _sum3);
                vst1q_f32(outptr + 16, _sum4);
                vst1q_f32(outptr + 20, _sum5);
            }

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
                if (pC)
                {
                    float32x2_t _c = vld1_f32(pC);
                    float32x2x2_t _cc = vzip_f32(_c, _c);
                    float32x4_t _bias = vcombine_f32(_cc.val[0], _cc.val[1]);
                    _sum0 = _bias;
                    _sum1 = _bias;
                    _sum2 = _bias;
                    _sum3 = _bias;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                }
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    uint16x4_t _bf0 = float2bfloat(vcombine_f32(vget_low_f32(_sum0), vget_low_f32(_sum1)));
                    uint16x4_t _bf1 = float2bfloat(vcombine_f32(vget_high_f32(_sum0), vget_high_f32(_sum1)));
                    uint16x4_t _bf2 = float2bfloat(vcombine_f32(vget_low_f32(_sum2), vget_low_f32(_sum3)));
                    uint16x4_t _bf3 = float2bfloat(vcombine_f32(vget_high_f32(_sum2), vget_high_f32(_sum3)));
                    vst1q_u16(outptr0, vcombine_u16(_bf0, _bf2));
                    vst1q_u16(outptr0 + out_hstep, vcombine_u16(_bf1, _bf3));
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
                vst1q_f32(outptr + 12, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    float32x2_t _c = vld1_f32(pC);
                    float32x2x2_t _cc = vzip_f32(_c, _c);
                    float32x4_t _bias = vcombine_f32(_cc.val[0], _cc.val[1]);
                    _sum0 = _bias;
                    _sum1 = _bias;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    uint16x4_t _bf0 = float2bfloat(vcombine_f32(vget_low_f32(_sum0), vget_low_f32(_sum1)));
                    uint16x4_t _bf1 = float2bfloat(vcombine_f32(vget_high_f32(_sum0), vget_high_f32(_sum1)));
                    vst1_u16(outptr0, _bf0);
                    vst1_u16(outptr0 + out_hstep, _bf1);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float32x4_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    float32x2_t _c = vld1_f32(pC);
                    float32x2x2_t _cc = vzip_f32(_c, _c);
                    _sum = vcombine_f32(_cc.val[0], _cc.val[1]);
                }
                else
                {
                    _sum = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum = vld1q_f32(outptr);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    uint16x4_t _bf0 = float2bfloat(_sum);
                    uint32x2_t _bf0_32 = vreinterpret_u32_u16(_bf0);
                    vst1_lane_u32((uint32_t*)outptr0, _bf0_32, 0);
                    vst1_lane_u32((uint32_t*)(outptr0 + out_hstep), _bf0_32, 1);
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum);
            }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum10 = pC[0];
                    sum11 = pC[1];
                }
                else
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum01 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                sum10 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[1]);
                sum11 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[1]);

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum00);
                    outptr0[1] = float32_to_bfloat16(sum10);
                    outptr0[out_hstep] = float32_to_bfloat16(sum01);
                    outptr0[out_hstep + 1] = float32_to_bfloat16(sum11);
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
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[out_hstep] = float32_to_bfloat16(sum1);
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
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

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
                if (pC)
                {
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[0]);
                    _sum2 = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                }
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 8, float2bfloat(_sum2));
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
            }

            outptr += 12;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    _sum = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum = vdupq_n_f32(0.f);
                }
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum));
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum);
            }

            outptr += 4;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[1] = float32_to_bfloat16(sum1);
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
                if (pC)
                {
                    sum = pC[0];
                }
                else
                {
                    sum = 0.f;
                }
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
            for (; kk < max_kk; kk += 1)
            {
                float pA0 = bfloat16_to_float32(pA[0]);
                float pB0 = bfloat16_to_float32(pB[0]);

                sum += pA0 * pB0;
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum);
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

static void convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_bf16 = (int)(get_cpu_level2_cache_size() / sizeof(unsigned short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __aarch64__
        int tile_size = (l2_cache_size_bf16 - 32) / 12;
#elif __ARM_NEON
        int tile_size = (l2_cache_size_bf16 - 16) / 8;
#else
        int tile_size = (l2_cache_size_bf16 - 2) / 3;
#endif

#if __aarch64__
        TILE_K = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __aarch64__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __aarch64__
        int nn_M = (M + 31) / 32;
#elif __ARM_NEON
        int nn_M = (M + 15) / 16;
#else
        int nn_M = (M + 7) / 8;
#endif

#if __aarch64__
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_M = std::max(4, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __aarch64__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __aarch64__
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __ARM_NEON
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_bf16 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_bf16 - TILE_M * TILE_K) / (TILE_M * 2 + TILE_K);
        }

#if __aarch64__
        TILE_N = std::max(4, tile_size / 4 * 4);
#elif __ARM_NEON
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __aarch64__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __ARM_NEON
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static void convolution_im2col_gemm_transform_kernel_bf16s(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        convolution_im2col_gemm_transform_kernel_bf16s_bf16(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
        return;
    }
#endif

    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        elempack = inch % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        cast_float32_to_bfloat16(kernel, A_data);
        A_data = A_data.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)2u);

        for (int q = 0; q < outch; q += 1)
        {
            unsigned short* g00 = A_data.row<unsigned short>(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_bf16(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        return convolution_im2col_gemm_bf16s_bf16(bottom_blob, top_blob, AT, bias, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
    }
#endif

    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);
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

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile_bf16(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
    {
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT_tileX.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end, opt.use_a53_a55_optimized_kernel);
            }
        }
    }

    return 0;
}
