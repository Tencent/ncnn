// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_im2col_pack_A_tile_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
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

static void convolution_im2col_input_tile_conv1x1s1d1_fp16(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    unsigned short* pp = B;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                // transpose8x12
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v16.8h, v17.8h, v18.8h, v19.8h}, [%0] \n"
                    "uzp1   v20.8h, v0.8h, v4.8h    \n"
                    "uzp2   v26.8h, v0.8h, v4.8h    \n"
                    "uzp1   v21.8h, v16.8h, v1.8h   \n"
                    "uzp2   v27.8h, v16.8h, v1.8h   \n"
                    "sub    %0, %0, #128            \n"
                    "uzp1   v22.8h, v5.8h, v17.8h   \n"
                    "uzp2   v28.8h, v5.8h, v17.8h   \n"
                    "uzp1   v23.8h, v2.8h, v6.8h    \n"
                    "uzp2   v29.8h, v2.8h, v6.8h    \n"
                    "uzp1   v24.8h, v18.8h, v3.8h   \n"
                    "uzp2   v30.8h, v18.8h, v3.8h   \n"
                    "uzp1   v25.8h, v7.8h, v19.8h   \n"
                    "uzp2   v31.8h, v7.8h, v19.8h   \n"
                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%1], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p0 + 8);
                uint16x8_t _r2 = vld1q_u16(p0 + 8 * 2);
                uint16x8_t _r3 = vld1q_u16(p0 + 8 * 3);
                uint16x8_t _r4 = vld1q_u16(p0 + 8 * 4);
                uint16x8_t _r5 = vld1q_u16(p0 + 8 * 5);
                uint16x8_t _r6 = vld1q_u16(p0 + 8 * 6);
                uint16x8_t _r7 = vld1q_u16(p0 + 8 * 7);
                uint16x8_t _r8 = vld1q_u16(p0 + 8 * 8);
                uint16x8_t _r9 = vld1q_u16(p0 + 8 * 9);
                uint16x8_t _ra = vld1q_u16(p0 + 8 * 10);
                uint16x8_t _rb = vld1q_u16(p0 + 8 * 11);
                transpose8x12_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                vst1q_u16(pp + 8 * 8, _r8);
                vst1q_u16(pp + 8 * 9, _r9);
                vst1q_u16(pp + 8 * 10, _ra);
                vst1q_u16(pp + 8 * 11, _rb);
                pp += 96;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
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
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                uint16x8_t _r01 = vld1q_u16(p0);
                uint16x4_t _r2 = vld1_u16(p0 + 8);
                vst1q_u16(pp, _r01);
                vst1_u16(pp + 8, _r2);
                pp += 12;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                // transpose8x8
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0] \n"
                    "uzp1   v16.8h, v0.8h, v4.8h    \n"
                    "uzp2   v20.8h, v0.8h, v4.8h    \n"
                    "uzp1   v17.8h, v1.8h, v5.8h    \n"
                    "uzp2   v21.8h, v1.8h, v5.8h    \n"
                    "sub    %0, %0, #64             \n"
                    "uzp1   v18.8h, v2.8h, v6.8h    \n"
                    "uzp2   v22.8h, v2.8h, v6.8h    \n"
                    "uzp1   v19.8h, v3.8h, v7.8h    \n"
                    "uzp2   v23.8h, v3.8h, v7.8h    \n"
                    "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"
                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else  // NCNN_GNU_INLINE_ASM
                uint16x8_t _r0 = vld1q_u16(p0);
                uint16x8_t _r1 = vld1q_u16(p0 + 8);
                uint16x8_t _r2 = vld1q_u16(p0 + 8 * 2);
                uint16x8_t _r3 = vld1q_u16(p0 + 8 * 3);
                uint16x8_t _r4 = vld1q_u16(p0 + 8 * 4);
                uint16x8_t _r5 = vld1q_u16(p0 + 8 * 5);
                uint16x8_t _r6 = vld1q_u16(p0 + 8 * 6);
                uint16x8_t _r7 = vld1q_u16(p0 + 8 * 7);
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
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
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
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                uint16x8_t _r0 = vld1q_u16(p0);
                vst1q_u16(pp, _r0);
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                // transpose8x4
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "st4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // NCNN_GNU_INLINE_ASM
                uint16x8x4_t _r0;
                _r0.val[0] = vld1q_u16(p0);
                _r0.val[1] = vld1q_u16(p0 + 8);
                _r0.val[2] = vld1q_u16(p0 + 8 * 2);
                _r0.val[3] = vld1q_u16(p0 + 8 * 3);
                vst4q_u16(pp, _r0);
                pp += 32;
#endif // NCNN_GNU_INLINE_ASM
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
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
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                vst1_u16(pp, _r0);
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __ARM_NEON
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                // transpose8x2
                uint16x8x2_t _r0;
                _r0.val[0] = vld1q_u16(p0);
                _r0.val[1] = vld1q_u16(p0 + 8);
                vst2q_u16(pp, _r0);
                pp += 16;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
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
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __ARM_NEON
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

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

static void convolution_im2col_input_tile_fp16(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_fp16(bottom_blob, B, j, max_jj, k, max_kk);
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (elempack == 8)
            {
                uint16x8_t _r0 = vld1q_u16(sptr0);
                uint16x8_t _r1 = vld1q_u16(sptr1);
                uint16x8_t _r2 = vld1q_u16(sptr2);
                uint16x8_t _r3 = vld1q_u16(sptr3);
                uint16x8_t _r4 = vld1q_u16(sptr4);
                uint16x8_t _r5 = vld1q_u16(sptr5);
                uint16x8_t _r6 = vld1q_u16(sptr6);
                uint16x8_t _r7 = vld1q_u16(sptr7);
                uint16x8_t _r8 = vld1q_u16(sptr8);
                uint16x8_t _r9 = vld1q_u16(sptr9);
                uint16x8_t _ra = vld1q_u16(sptra);
                uint16x8_t _rb = vld1q_u16(sptrb);
                transpose8x12_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                vst1q_u16(pp, _r0);
                vst1q_u16(pp + 8, _r1);
                vst1q_u16(pp + 8 * 2, _r2);
                vst1q_u16(pp + 8 * 3, _r3);
                vst1q_u16(pp + 8 * 4, _r4);
                vst1q_u16(pp + 8 * 5, _r5);
                vst1q_u16(pp + 8 * 6, _r6);
                vst1q_u16(pp + 8 * 7, _r7);
                vst1q_u16(pp + 8 * 8, _r8);
                vst1q_u16(pp + 8 * 9, _r9);
                vst1q_u16(pp + 8 * 10, _ra);
                vst1q_u16(pp + 8 * 11, _rb);
                pp += 96;
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (elempack == 8)
            {
                uint16x8_t _r0 = vld1q_u16(sptr0);
                uint16x8_t _r1 = vld1q_u16(sptr1);
                uint16x8_t _r2 = vld1q_u16(sptr2);
                uint16x8_t _r3 = vld1q_u16(sptr3);
                uint16x8_t _r4 = vld1q_u16(sptr4);
                uint16x8_t _r5 = vld1q_u16(sptr5);
                uint16x8_t _r6 = vld1q_u16(sptr6);
                uint16x8_t _r7 = vld1q_u16(sptr7);
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
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (elempack == 8)
            {
                uint16x8x4_t _r0;
                _r0.val[0] = vld1q_u16(sptr0);
                _r0.val[1] = vld1q_u16(sptr1);
                _r0.val[2] = vld1q_u16(sptr2);
                _r0.val[3] = vld1q_u16(sptr3);
                vst4q_u16(pp, _r0);
                pp += 32;
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        int kk = 0;
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
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (elempack == 8)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp[2] = sptr0[1];
                pp[3] = sptr1[1];
                pp[4] = sptr0[2];
                pp[5] = sptr1[2];
                pp[6] = sptr0[3];
                pp[7] = sptr1[3];
                pp[8 + 0] = sptr0[4];
                pp[8 + 1] = sptr1[4];
                pp[8 + 2] = sptr0[5];
                pp[8 + 3] = sptr1[5];
                pp[8 + 4] = sptr0[6];
                pp[8 + 5] = sptr1[6];
                pp[8 + 6] = sptr0[7];
                pp[8 + 7] = sptr1[7];
                pp += 16;
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
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
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (elempack == 8)
            {
                pp[0] = sptr[0];
                pp[1] = sptr[1];
                pp[2] = sptr[2];
                pp[3] = sptr[3];
                pp[4] = sptr[4];
                pp[5] = sptr[5];
                pp[6] = sptr[6];
                pp[7] = sptr[7];
                pp += 8;
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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
    }
}

static void convolution_gemm_transB_packed_tile_fp16sa(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end, int use_a53_a55_optimized_kernel)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_fp16sa %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;

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
            pC = (const __fp16*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const __fp16* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            if (use_a53_a55_optimized_kernel)
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%0], #64 \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                    "subs   %0, %0, #128                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "ld1    {v20.8h}, [%8]              \n"
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

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.8h}, [%1], #16          \n"
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.8h}, [%2], #16          \n"

                    "ldr    d1, [%2], #8                \n"
                    "ldr    x21, [%2], #8               \n"

                    ".align 4                           \n"
                    "4:                                 \n"
                    "ldr    d5, [%1], #8                \n"
                    "fmla   v20.8h, v4.8h, v0.h[0]      \n"
                    "ldr    x25, [%1], #8               \n"
                    "fmla   v21.8h, v4.8h, v0.h[1]      \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v22.8h, v4.8h, v0.h[2]      \n"
                    "ldr    x22, [%2], #8               \n"
                    "fmla   v23.8h, v4.8h, v0.h[3]      \n"
                    "ldr    d6, [%1], #8                \n"
                    "fmla   v24.8h, v4.8h, v0.h[4]      \n"
                    "ldr    x26, [%1], #8               \n"
                    "fmla   v25.8h, v4.8h, v0.h[5]      \n"
                    "ins    v1.d[1], x21                \n"
                    "fmla   v26.8h, v4.8h, v0.h[6]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v27.8h, v4.8h, v0.h[7]      \n"
                    "ldr    x23, [%2], #8               \n"
                    "fmla   v28.8h, v4.8h, v1.h[0]      \n"
                    "prfm   pldl1keep, [%2, #256]       \n" // NOTE PRELOAD
                    "fmla   v29.8h, v4.8h, v1.h[1]      \n"
                    "ins    v5.d[1], x25                \n"
                    "fmla   v30.8h, v4.8h, v1.h[2]      \n"
                    "ldr    d8, [%2], #8                \n"
                    "fmla   v31.8h, v4.8h, v1.h[3]      \n"
                    "ldr    x20, [%2], #8               \n"
                    "fmla   v20.8h, v5.8h, v1.h[4]      \n"
                    "ldr    d7, [%1], #8                \n"
                    "fmla   v21.8h, v5.8h, v1.h[5]      \n"
                    "ins    v2.d[1], x22                \n"
                    "fmla   v22.8h, v5.8h, v1.h[6]      \n"
                    "ldr    x27, [%1], #8               \n"
                    "fmla   v23.8h, v5.8h, v1.h[7]      \n"
                    "ldr    d9, [%2], #8                \n"
                    "fmla   v24.8h, v5.8h, v2.h[0]      \n"
                    "ldr    x21, [%2], #8               \n"
                    "fmla   v25.8h, v5.8h, v2.h[1]      \n"
                    "ins    v6.d[1], x26                \n"
                    "fmla   v26.8h, v5.8h, v2.h[2]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                    "fmla   v27.8h, v5.8h, v2.h[3]      \n"
                    "ldr    d4, [%1], #8                \n"
                    "fmla   v28.8h, v5.8h, v2.h[4]      \n"
                    "ldr    x24, [%1], #8               \n"
                    "fmla   v29.8h, v5.8h, v2.h[5]      \n"
                    "ins    v3.d[1], x23                \n"
                    "fmla   v30.8h, v5.8h, v2.h[6]      \n"
                    "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                    "fmla   v31.8h, v5.8h, v2.h[7]      \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v20.8h, v6.8h, v3.h[0]      \n"
                    "fmla   v21.8h, v6.8h, v3.h[1]      \n"
                    "fmla   v22.8h, v6.8h, v3.h[2]      \n"
                    "fmla   v23.8h, v6.8h, v3.h[3]      \n"
                    "fmla   v24.8h, v6.8h, v3.h[4]      \n"
                    "fmla   v25.8h, v6.8h, v3.h[5]      \n"
                    "ins    v8.d[1], x20                \n"
                    "fmla   v26.8h, v6.8h, v3.h[6]      \n"
                    "ldr    x20, [%2], #8               \n"
                    "fmla   v27.8h, v6.8h, v3.h[7]      \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v28.8h, v6.8h, v8.h[0]      \n"
                    "fmla   v29.8h, v6.8h, v8.h[1]      \n"
                    "ins    v7.d[1], x27                \n"
                    "fmla   v30.8h, v6.8h, v8.h[2]      \n"
                    "fmla   v31.8h, v6.8h, v8.h[3]      \n"
                    "fmla   v20.8h, v7.8h, v8.h[4]      \n"
                    "fmla   v21.8h, v7.8h, v8.h[5]      \n"
                    "ins    v9.d[1], x21                \n"
                    "fmla   v22.8h, v7.8h, v8.h[6]      \n"
                    "fmla   v23.8h, v7.8h, v8.h[7]      \n"
                    "ldr    x21, [%2], #8               \n"
                    "fmla   v24.8h, v7.8h, v9.h[0]      \n"
                    "fmla   v25.8h, v7.8h, v9.h[1]      \n"
                    "ins    v4.d[1], x24                \n"
                    "fmla   v26.8h, v7.8h, v9.h[2]      \n"
                    "fmla   v27.8h, v7.8h, v9.h[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.8h, v7.8h, v9.h[4]      \n"
                    "fmla   v29.8h, v7.8h, v9.h[5]      \n"
                    "fmla   v30.8h, v7.8h, v9.h[6]      \n"
                    "ins    v0.d[1], x20                \n"
                    "fmla   v31.8h, v7.8h, v9.h[7]      \n"
                    "bne    4b                          \n"

                    "sub    %1, %1, #16                 \n"
                    "sub    %2, %2, #32                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"
                    "ld1    {v4.8h}, [%1], #16          \n"
                    "fmla   v20.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v21.8h, v4.8h, v0.h[1]      \n"
                    "fmla   v22.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v23.8h, v4.8h, v0.h[3]      \n"
                    "fmla   v24.8h, v4.8h, v1.h[0]      \n"
                    "fmla   v25.8h, v4.8h, v1.h[1]      \n"
                    "fmla   v26.8h, v4.8h, v1.h[2]      \n"
                    "fmla   v27.8h, v4.8h, v1.h[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.8h, v4.8h, v2.h[0]      \n"
                    "fmla   v29.8h, v4.8h, v2.h[1]      \n"
                    "fmla   v30.8h, v4.8h, v2.h[2]      \n"
                    "fmla   v31.8h, v4.8h, v2.h[3]      \n"
                    "bne    6b                          \n"

                    "7:                                 \n"
                    "tst    %w11, #255                  \n"
                    "beq    11f                         \n"

                    // if out_elempack == 8
                    "cmp    %w12, #8                    \n"
                    "bne    8f                          \n"

                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%3], #64 \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%3], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%3], #64 \n"
                    "b      10f                         \n"

                    // if out_elempack == 4
                    "8:                                 \n"
                    "cmp    %w12, #4                    \n"
                    "bne    9f                          \n"

                    "zip1   v12.2d, v20.2d, v21.2d      \n"
                    "zip2   v18.2d, v20.2d, v21.2d      \n"
                    "zip1   v13.2d, v22.2d, v23.2d      \n"
                    "zip2   v19.2d, v22.2d, v23.2d      \n"
                    "zip1   v14.2d, v24.2d, v25.2d      \n"
                    "zip2   v20.2d, v24.2d, v25.2d      \n"
                    "zip1   v15.2d, v26.2d, v27.2d      \n"
                    "zip2   v21.2d, v26.2d, v27.2d      \n"
                    "zip1   v16.2d, v28.2d, v29.2d      \n"
                    "zip2   v22.2d, v28.2d, v29.2d      \n"
                    "zip1   v17.2d, v30.2d, v31.2d      \n"
                    "zip2   v23.2d, v30.2d, v31.2d      \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n"
                    "st1    {v16.8h, v17.8h}, [%3], #32 \n"
                    "st1    {v18.8h, v19.8h, v20.8h, v21.8h}, [x4], #64 \n"
                    "st1    {v22.8h, v23.8h}, [x4]      \n"
                    "b      10f                         \n"

                    // if out_elempack == 1
                    "9:                                 \n"
                    // transpose8x12
                    "zip1   v18.8h, v20.8h, v21.8h      \n"
                    "zip2   v19.8h, v20.8h, v21.8h      \n"
                    "zip1   v20.8h, v22.8h, v23.8h      \n"
                    "zip2   v21.8h, v22.8h, v23.8h      \n"
                    "zip1   v22.8h, v24.8h, v25.8h      \n"
                    "zip2   v23.8h, v24.8h, v25.8h      \n"
                    "zip1   v24.8h, v26.8h, v27.8h      \n"
                    "zip2   v25.8h, v26.8h, v27.8h      \n"
                    "zip1   v26.8h, v28.8h, v29.8h      \n"
                    "zip2   v27.8h, v28.8h, v29.8h      \n"
                    "zip1   v28.8h, v30.8h, v31.8h      \n"
                    "zip2   v29.8h, v30.8h, v31.8h      \n"

                    "zip1   v0.4s, v18.4s, v20.4s       \n"
                    "zip2   v3.4s, v18.4s, v20.4s       \n"
                    "zip1   v6.4s, v19.4s, v21.4s       \n"
                    "zip2   v9.4s, v19.4s, v21.4s       \n"
                    "zip1   v1.4s, v22.4s, v24.4s       \n"
                    "zip2   v4.4s, v22.4s, v24.4s       \n"
                    "zip1   v7.4s, v23.4s, v25.4s       \n"
                    "zip2   v10.4s, v23.4s, v25.4s      \n"
                    "zip1   v2.4s, v26.4s, v28.4s       \n"
                    "zip2   v5.4s, v26.4s, v28.4s       \n"
                    "zip1   v8.4s, v27.4s, v29.4s       \n"
                    "zip2   v11.4s, v27.4s, v29.4s      \n"

                    "mov    v12.d[0], v0.d[1]           \n"
                    "mov    v13.d[0], v1.d[1]           \n"
                    "mov    v14.d[0], v2.d[1]           \n"
                    "mov    v15.d[0], v3.d[1]           \n"
                    "mov    v16.d[0], v4.d[1]           \n"
                    "mov    v17.d[0], v5.d[1]           \n"
                    "mov    v18.d[0], v6.d[1]           \n"
                    "mov    v19.d[0], v7.d[1]           \n"
                    "mov    v20.d[0], v8.d[1]           \n"
                    "mov    v21.d[0], v9.d[1]           \n"
                    "mov    v22.d[0], v10.d[1]          \n"
                    "mov    v23.d[0], v11.d[1]          \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                    "st1    {v12.4h, v13.4h, v14.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v15.4h, v16.4h, v17.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v18.4h, v19.4h, v20.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v21.4h, v22.4h, v23.4h}, [x4] \n"

                    "10:                                \n"
                    "add    %0, %0, #192                \n"
                    "b      12f                         \n"

                    "11:                                \n"
                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%0], #64 \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                    "12:                                \n"

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

                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%0], #64 \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                    "subs   %0, %0, #128                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "ld1    {v20.8h}, [%8]              \n"
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
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n"

                    "fmla   v20.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v21.8h, v4.8h, v0.h[1]      \n"
                    "fmla   v22.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v23.8h, v4.8h, v0.h[3]      \n"
                    "fmla   v24.8h, v4.8h, v0.h[4]      \n"
                    "fmla   v25.8h, v4.8h, v0.h[5]      \n"
                    "fmla   v26.8h, v4.8h, v0.h[6]      \n"
                    "fmla   v27.8h, v4.8h, v0.h[7]      \n"
                    "fmla   v28.8h, v4.8h, v1.h[0]      \n"
                    "fmla   v29.8h, v4.8h, v1.h[1]      \n"
                    "fmla   v30.8h, v4.8h, v1.h[2]      \n"
                    "fmla   v31.8h, v4.8h, v1.h[3]      \n"

                    "fmla   v20.8h, v5.8h, v1.h[4]      \n"
                    "fmla   v21.8h, v5.8h, v1.h[5]      \n"
                    "fmla   v22.8h, v5.8h, v1.h[6]      \n"
                    "fmla   v23.8h, v5.8h, v1.h[7]      \n"
                    "fmla   v24.8h, v5.8h, v2.h[0]      \n"
                    "fmla   v25.8h, v5.8h, v2.h[1]      \n"
                    "fmla   v26.8h, v5.8h, v2.h[2]      \n"
                    "fmla   v27.8h, v5.8h, v2.h[3]      \n"
                    "fmla   v28.8h, v5.8h, v2.h[4]      \n"
                    "fmla   v29.8h, v5.8h, v2.h[5]      \n"
                    "fmla   v30.8h, v5.8h, v2.h[6]      \n"
                    "fmla   v31.8h, v5.8h, v2.h[7]      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v8.8h, v9.8h}, [%2], #32   \n"

                    "fmla   v20.8h, v6.8h, v3.h[0]      \n"
                    "fmla   v21.8h, v6.8h, v3.h[1]      \n"
                    "fmla   v22.8h, v6.8h, v3.h[2]      \n"
                    "fmla   v23.8h, v6.8h, v3.h[3]      \n"
                    "fmla   v24.8h, v6.8h, v3.h[4]      \n"
                    "fmla   v25.8h, v6.8h, v3.h[5]      \n"
                    "fmla   v26.8h, v6.8h, v3.h[6]      \n"
                    "fmla   v27.8h, v6.8h, v3.h[7]      \n"
                    "fmla   v28.8h, v6.8h, v8.h[0]      \n"
                    "fmla   v29.8h, v6.8h, v8.h[1]      \n"
                    "fmla   v30.8h, v6.8h, v8.h[2]      \n"
                    "fmla   v31.8h, v6.8h, v8.h[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v20.8h, v7.8h, v8.h[4]      \n"
                    "fmla   v21.8h, v7.8h, v8.h[5]      \n"
                    "fmla   v22.8h, v7.8h, v8.h[6]      \n"
                    "fmla   v23.8h, v7.8h, v8.h[7]      \n"
                    "fmla   v24.8h, v7.8h, v9.h[0]      \n"
                    "fmla   v25.8h, v7.8h, v9.h[1]      \n"
                    "fmla   v26.8h, v7.8h, v9.h[2]      \n"
                    "fmla   v27.8h, v7.8h, v9.h[3]      \n"
                    "fmla   v28.8h, v7.8h, v9.h[4]      \n"
                    "fmla   v29.8h, v7.8h, v9.h[5]      \n"
                    "fmla   v30.8h, v7.8h, v9.h[6]      \n"
                    "fmla   v31.8h, v7.8h, v9.h[7]      \n"

                    "bne    4b                          \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"
                    "ld1    {v4.8h}, [%1], #16          \n"
                    "fmla   v20.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v21.8h, v4.8h, v0.h[1]      \n"
                    "fmla   v22.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v23.8h, v4.8h, v0.h[3]      \n"
                    "fmla   v24.8h, v4.8h, v1.h[0]      \n"
                    "fmla   v25.8h, v4.8h, v1.h[1]      \n"
                    "fmla   v26.8h, v4.8h, v1.h[2]      \n"
                    "fmla   v27.8h, v4.8h, v1.h[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.8h, v4.8h, v2.h[0]      \n"
                    "fmla   v29.8h, v4.8h, v2.h[1]      \n"
                    "fmla   v30.8h, v4.8h, v2.h[2]      \n"
                    "fmla   v31.8h, v4.8h, v2.h[3]      \n"
                    "bne    6b                          \n"

                    "7:                                 \n"
                    "tst    %w11, #255                  \n"
                    "beq    11f                         \n"

                    // if out_elempack == 8
                    "cmp    %w12, #8                    \n"
                    "bne    8f                          \n"

                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%3], #64 \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%3], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%3], #64 \n"
                    "b      10f                         \n"

                    // if out_elempack == 4
                    "8:                                 \n"
                    "cmp    %w12, #4                    \n"
                    "bne    9f                          \n"

                    "zip1   v12.2d, v20.2d, v21.2d      \n"
                    "zip2   v18.2d, v20.2d, v21.2d      \n"
                    "zip1   v13.2d, v22.2d, v23.2d      \n"
                    "zip2   v19.2d, v22.2d, v23.2d      \n"
                    "zip1   v14.2d, v24.2d, v25.2d      \n"
                    "zip2   v20.2d, v24.2d, v25.2d      \n"
                    "zip1   v15.2d, v26.2d, v27.2d      \n"
                    "zip2   v21.2d, v26.2d, v27.2d      \n"
                    "zip1   v16.2d, v28.2d, v29.2d      \n"
                    "zip2   v22.2d, v28.2d, v29.2d      \n"
                    "zip1   v17.2d, v30.2d, v31.2d      \n"
                    "zip2   v23.2d, v30.2d, v31.2d      \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n"
                    "st1    {v16.8h, v17.8h}, [%3], #32 \n"
                    "st1    {v18.8h, v19.8h, v20.8h, v21.8h}, [x4], #64 \n"
                    "st1    {v22.8h, v23.8h}, [x4]      \n"
                    "b      10f                         \n"

                    // if out_elempack == 1
                    "9:                                 \n"
                    // transpose8x12
                    "zip1   v18.8h, v20.8h, v21.8h      \n"
                    "zip2   v19.8h, v20.8h, v21.8h      \n"
                    "zip1   v20.8h, v22.8h, v23.8h      \n"
                    "zip2   v21.8h, v22.8h, v23.8h      \n"
                    "zip1   v22.8h, v24.8h, v25.8h      \n"
                    "zip2   v23.8h, v24.8h, v25.8h      \n"
                    "zip1   v24.8h, v26.8h, v27.8h      \n"
                    "zip2   v25.8h, v26.8h, v27.8h      \n"
                    "zip1   v26.8h, v28.8h, v29.8h      \n"
                    "zip2   v27.8h, v28.8h, v29.8h      \n"
                    "zip1   v28.8h, v30.8h, v31.8h      \n"
                    "zip2   v29.8h, v30.8h, v31.8h      \n"

                    "zip1   v0.4s, v18.4s, v20.4s       \n"
                    "zip2   v3.4s, v18.4s, v20.4s       \n"
                    "zip1   v6.4s, v19.4s, v21.4s       \n"
                    "zip2   v9.4s, v19.4s, v21.4s       \n"
                    "zip1   v1.4s, v22.4s, v24.4s       \n"
                    "zip2   v4.4s, v22.4s, v24.4s       \n"
                    "zip1   v7.4s, v23.4s, v25.4s       \n"
                    "zip2   v10.4s, v23.4s, v25.4s      \n"
                    "zip1   v2.4s, v26.4s, v28.4s       \n"
                    "zip2   v5.4s, v26.4s, v28.4s       \n"
                    "zip1   v8.4s, v27.4s, v29.4s       \n"
                    "zip2   v11.4s, v27.4s, v29.4s      \n"

                    "mov    v12.d[0], v0.d[1]           \n"
                    "mov    v13.d[0], v1.d[1]           \n"
                    "mov    v14.d[0], v2.d[1]           \n"
                    "mov    v15.d[0], v3.d[1]           \n"
                    "mov    v16.d[0], v4.d[1]           \n"
                    "mov    v17.d[0], v5.d[1]           \n"
                    "mov    v18.d[0], v6.d[1]           \n"
                    "mov    v19.d[0], v7.d[1]           \n"
                    "mov    v20.d[0], v8.d[1]           \n"
                    "mov    v21.d[0], v9.d[1]           \n"
                    "mov    v22.d[0], v10.d[1]          \n"
                    "mov    v23.d[0], v11.d[1]          \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                    "st1    {v12.4h, v13.4h, v14.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v15.4h, v16.4h, v17.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v18.4h, v19.4h, v20.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v21.4h, v22.4h, v23.4h}, [x4] \n"

                    "10:                                \n"
                    "add    %0, %0, #192                \n"
                    "b      12f                         \n"

                    "11:                                \n"
                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%0], #64 \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                    "12:                                \n"

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
#else  // NCNN_GNU_INLINE_ASM
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
                if (pC)
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
                else
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
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1q_f16(outptr + 8);
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

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
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
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8, _sum1);
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
                    transpose8x12_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    vst1_f16(outptr0, vget_low_f16(_sum0));
                    vst1_f16(outptr0 + 4, vget_high_f16(_sum0));
                    vst1_f16(outptr0 + 8, vget_low_f16(_sum1));
                    vst1_f16(outptr0 + out_hstep, vget_high_f16(_sum1));
                    vst1_f16(outptr0 + out_hstep + 4, vget_low_f16(_sum2));
                    vst1_f16(outptr0 + out_hstep + 8, vget_high_f16(_sum2));
                    vst1_f16(outptr0 + out_hstep * 2, vget_low_f16(_sum3));
                    vst1_f16(outptr0 + out_hstep * 2 + 4, vget_high_f16(_sum3));
                    vst1_f16(outptr0 + out_hstep * 2 + 8, vget_low_f16(_sum4));
                    vst1_f16(outptr0 + out_hstep * 3, vget_high_f16(_sum4));
                    vst1_f16(outptr0 + out_hstep * 3 + 4, vget_low_f16(_sum5));
                    vst1_f16(outptr0 + out_hstep * 3 + 8, vget_high_f16(_sum5));
                    vst1_f16(outptr0 + out_hstep * 4, vget_low_f16(_sum6));
                    vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum6));
                    vst1_f16(outptr0 + out_hstep * 4 + 8, vget_low_f16(_sum7));
                    vst1_f16(outptr0 + out_hstep * 5, vget_high_f16(_sum7));
                    vst1_f16(outptr0 + out_hstep * 5 + 4, vget_low_f16(_sum8));
                    vst1_f16(outptr0 + out_hstep * 5 + 8, vget_high_f16(_sum8));
                    vst1_f16(outptr0 + out_hstep * 6, vget_low_f16(_sum9));
                    vst1_f16(outptr0 + out_hstep * 6 + 4, vget_high_f16(_sum9));
                    vst1_f16(outptr0 + out_hstep * 6 + 8, vget_low_f16(_suma));
                    vst1_f16(outptr0 + out_hstep * 7, vget_high_f16(_suma));
                    vst1_f16(outptr0 + out_hstep * 7 + 4, vget_low_f16(_sumb));
                    vst1_f16(outptr0 + out_hstep * 7 + 8, vget_high_f16(_sumb));

                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f16(outptr, _sum0);
                vst1q_f16(outptr + 8, _sum1);
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
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const __fp16* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            if (use_a53_a55_optimized_kernel)
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                    "subs   %0, %0, #64                 \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "ld1    {v24.8h}, [%8]              \n"
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

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.8h}, [%1], #16          \n"
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.8h}, [%2], #16          \n"

                    "ldr    d5, [%1], #8                \n"
                    "ldr    x25, [%1], #8               \n"

                    ".align 4                           \n"
                    "4:                                 \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v24.8h, v4.8h, v0.h[0]      \n"
                    "ldr    x21, [%2], #8               \n"
                    "fmla   v25.8h, v4.8h, v0.h[1]      \n"
                    "ins    v5.d[1], x25                \n"
                    "fmla   v26.8h, v4.8h, v0.h[2]      \n"
                    "ldr    d6, [%1], #8                \n"
                    "fmla   v27.8h, v4.8h, v0.h[3]      \n"
                    "ldr    x26, [%1], #8               \n"
                    "fmla   v28.8h, v4.8h, v0.h[4]      \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v29.8h, v4.8h, v0.h[5]      \n"
                    "ins    v1.d[1], x21                \n"
                    "fmla   v30.8h, v4.8h, v0.h[6]      \n"
                    "ldr    x22, [%2], #8               \n"
                    "fmla   v31.8h, v4.8h, v0.h[7]      \n"
                    "ldr    d7, [%1], #8                \n"
                    "fmla   v24.8h, v5.8h, v1.h[0]      \n"
                    "ldr    x27, [%1], #8               \n"
                    "fmla   v25.8h, v5.8h, v1.h[1]      \n"
                    "ins    v6.d[1], x26                \n"
                    "fmla   v26.8h, v5.8h, v1.h[2]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v27.8h, v5.8h, v1.h[3]      \n"
                    "ldr    x23, [%2], #8               \n"
                    "fmla   v28.8h, v5.8h, v1.h[4]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                    "fmla   v29.8h, v5.8h, v1.h[5]      \n"
                    "ins    v2.d[1], x22                \n"
                    "fmla   v30.8h, v5.8h, v1.h[6]      \n"
                    "ldr    d4, [%1], #8                \n"
                    "fmla   v31.8h, v5.8h, v1.h[7]      \n"
                    "ldr    x24, [%1], #8               \n"
                    "fmla   v24.8h, v6.8h, v2.h[0]      \n"
                    "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                    "fmla   v25.8h, v6.8h, v2.h[1]      \n"
                    "ins    v7.d[1], x27                \n"
                    "fmla   v26.8h, v6.8h, v2.h[2]      \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v27.8h, v6.8h, v2.h[3]      \n"
                    "ldr    x20, [%2], #8               \n"
                    "fmla   v28.8h, v6.8h, v2.h[4]      \n"
                    "ldr    d5, [%1], #8                \n"
                    "fmla   v29.8h, v6.8h, v2.h[5]      \n"
                    "ins    v3.d[1], x23                \n"
                    "fmla   v30.8h, v6.8h, v2.h[6]      \n"
                    "ldr    x25, [%1], #8               \n"
                    "fmla   v31.8h, v6.8h, v2.h[7]      \n"
                    "fmla   v24.8h, v7.8h, v3.h[0]      \n"
                    "fmla   v25.8h, v7.8h, v3.h[1]      \n"
                    "fmla   v26.8h, v7.8h, v3.h[2]      \n"
                    "ins    v4.d[1], x24                \n"
                    "fmla   v27.8h, v7.8h, v3.h[3]      \n"
                    "fmla   v28.8h, v7.8h, v3.h[4]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v29.8h, v7.8h, v3.h[5]      \n"
                    "fmla   v30.8h, v7.8h, v3.h[6]      \n"
                    "ins    v0.d[1], x20                \n"
                    "fmla   v31.8h, v7.8h, v3.h[7]      \n"
                    "bne    4b                          \n"

                    "sub    %1, %1, #32                 \n"
                    "sub    %2, %2, #16                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.8h}, [%2], #16          \n"
                    "ld1    {v4.8h}, [%1], #16          \n"
                    "fmla   v24.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v25.8h, v4.8h, v0.h[1]      \n"
                    "fmla   v26.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v27.8h, v4.8h, v0.h[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.8h, v4.8h, v0.h[4]      \n"
                    "fmla   v29.8h, v4.8h, v0.h[5]      \n"
                    "fmla   v30.8h, v4.8h, v0.h[6]      \n"
                    "fmla   v31.8h, v4.8h, v0.h[7]      \n"
                    "bne    6b                          \n"

                    "7:                                 \n"
                    "tst    %w11, #255                  \n"
                    "beq    11f                         \n"

                    // if out_elempack == 8
                    "cmp    %w12, #8                    \n"
                    "bne    8f                          \n"

                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%3], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%3], #64 \n"
                    "b      10f                         \n"

                    // if out_elempack == 4
                    "8:                                 \n"
                    "cmp    %w12, #4                    \n"
                    "bne    9f                          \n"

                    "zip1   v16.2d, v24.2d, v25.2d      \n"
                    "zip2   v20.2d, v24.2d, v25.2d      \n"
                    "zip1   v17.2d, v26.2d, v27.2d      \n"
                    "zip2   v21.2d, v26.2d, v27.2d      \n"
                    "zip1   v18.2d, v28.2d, v29.2d      \n"
                    "zip2   v22.2d, v28.2d, v29.2d      \n"
                    "zip1   v19.2d, v30.2d, v31.2d      \n"
                    "zip2   v23.2d, v30.2d, v31.2d      \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%3], #64 \n"
                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [x4] \n"
                    "b      10f                         \n"

                    // if out_elempack == 1
                    "9:                                 \n"
                    // transpose8x8
                    "zip1   v22.8h, v24.8h, v25.8h      \n"
                    "zip2   v23.8h, v24.8h, v25.8h      \n"
                    "zip1   v24.8h, v26.8h, v27.8h      \n"
                    "zip2   v25.8h, v26.8h, v27.8h      \n"
                    "zip1   v26.8h, v28.8h, v29.8h      \n"
                    "zip2   v27.8h, v28.8h, v29.8h      \n"
                    "zip1   v28.8h, v30.8h, v31.8h      \n"
                    "zip2   v29.8h, v30.8h, v31.8h      \n"

                    "zip1   v16.4s, v22.4s, v24.4s      \n"
                    "zip2   v17.4s, v22.4s, v24.4s      \n"
                    "zip1   v18.4s, v23.4s, v25.4s      \n"
                    "zip2   v19.4s, v23.4s, v25.4s      \n"
                    "zip1   v20.4s, v26.4s, v28.4s      \n"
                    "zip2   v21.4s, v26.4s, v28.4s      \n"
                    "zip1   v22.4s, v27.4s, v29.4s      \n"
                    "zip2   v23.4s, v27.4s, v29.4s      \n"

                    "zip1   v24.2d, v16.2d, v20.2d      \n"
                    "zip2   v25.2d, v16.2d, v20.2d      \n"
                    "zip1   v26.2d, v17.2d, v21.2d      \n"
                    "zip2   v27.2d, v17.2d, v21.2d      \n"
                    "zip1   v28.2d, v18.2d, v22.2d      \n"
                    "zip2   v29.2d, v18.2d, v22.2d      \n"
                    "zip1   v30.2d, v19.2d, v23.2d      \n"
                    "zip2   v31.2d, v19.2d, v23.2d      \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v24.8h}, [%3], #16         \n"
                    "st1    {v25.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v26.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v27.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v28.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v29.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v30.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v31.8h}, [x4]              \n"

                    "10:                                \n"
                    "add    %0, %0, #128                \n"
                    "b      12f                         \n"

                    "11:                                \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                    "12:                                \n"

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
                    : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                    "subs   %0, %0, #64                 \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "ld1    {v24.8h}, [%8]              \n"
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
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n"
                    "fmla   v24.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v25.8h, v4.8h, v0.h[1]      \n"
                    "fmla   v26.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v27.8h, v4.8h, v0.h[3]      \n"
                    "fmla   v28.8h, v4.8h, v0.h[4]      \n"
                    "fmla   v29.8h, v4.8h, v0.h[5]      \n"
                    "fmla   v30.8h, v4.8h, v0.h[6]      \n"
                    "fmla   v31.8h, v4.8h, v0.h[7]      \n"
                    "fmla   v24.8h, v5.8h, v1.h[0]      \n"
                    "fmla   v25.8h, v5.8h, v1.h[1]      \n"
                    "fmla   v26.8h, v5.8h, v1.h[2]      \n"
                    "fmla   v27.8h, v5.8h, v1.h[3]      \n"
                    "fmla   v28.8h, v5.8h, v1.h[4]      \n"
                    "fmla   v29.8h, v5.8h, v1.h[5]      \n"
                    "fmla   v30.8h, v5.8h, v1.h[6]      \n"
                    "fmla   v31.8h, v5.8h, v1.h[7]      \n"
                    "fmla   v24.8h, v6.8h, v2.h[0]      \n"
                    "fmla   v25.8h, v6.8h, v2.h[1]      \n"
                    "fmla   v26.8h, v6.8h, v2.h[2]      \n"
                    "fmla   v27.8h, v6.8h, v2.h[3]      \n"
                    "fmla   v28.8h, v6.8h, v2.h[4]      \n"
                    "fmla   v29.8h, v6.8h, v2.h[5]      \n"
                    "fmla   v30.8h, v6.8h, v2.h[6]      \n"
                    "fmla   v31.8h, v6.8h, v2.h[7]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v24.8h, v7.8h, v3.h[0]      \n"
                    "fmla   v25.8h, v7.8h, v3.h[1]      \n"
                    "fmla   v26.8h, v7.8h, v3.h[2]      \n"
                    "fmla   v27.8h, v7.8h, v3.h[3]      \n"
                    "fmla   v28.8h, v7.8h, v3.h[4]      \n"
                    "fmla   v29.8h, v7.8h, v3.h[5]      \n"
                    "fmla   v30.8h, v7.8h, v3.h[6]      \n"
                    "fmla   v31.8h, v7.8h, v3.h[7]      \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.8h}, [%2], #16          \n"
                    "ld1    {v4.8h}, [%1], #16          \n"
                    "fmla   v24.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v25.8h, v4.8h, v0.h[1]      \n"
                    "fmla   v26.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v27.8h, v4.8h, v0.h[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.8h, v4.8h, v0.h[4]      \n"
                    "fmla   v29.8h, v4.8h, v0.h[5]      \n"
                    "fmla   v30.8h, v4.8h, v0.h[6]      \n"
                    "fmla   v31.8h, v4.8h, v0.h[7]      \n"
                    "bne    6b                          \n"

                    "7:                                 \n"
                    "tst    %w11, #255                  \n"
                    "beq    11f                         \n"

                    // if out_elempack == 8
                    "cmp    %w12, #8                    \n"
                    "bne    8f                          \n"

                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%3], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%3], #64 \n"
                    "b      10f                         \n"

                    // if out_elempack == 4
                    "8:                                 \n"
                    "cmp    %w12, #4                    \n"
                    "bne    9f                          \n"

                    "zip1   v16.2d, v24.2d, v25.2d      \n"
                    "zip2   v20.2d, v24.2d, v25.2d      \n"
                    "zip1   v17.2d, v26.2d, v27.2d      \n"
                    "zip2   v21.2d, v26.2d, v27.2d      \n"
                    "zip1   v18.2d, v28.2d, v29.2d      \n"
                    "zip2   v22.2d, v28.2d, v29.2d      \n"
                    "zip1   v19.2d, v30.2d, v31.2d      \n"
                    "zip2   v23.2d, v30.2d, v31.2d      \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%3], #64 \n"
                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [x4] \n"
                    "b      10f                         \n"

                    // if out_elempack == 1
                    "9:                                 \n"
                    // transpose8x8
                    "zip1   v22.8h, v24.8h, v25.8h      \n"
                    "zip2   v23.8h, v24.8h, v25.8h      \n"
                    "zip1   v24.8h, v26.8h, v27.8h      \n"
                    "zip2   v25.8h, v26.8h, v27.8h      \n"
                    "zip1   v26.8h, v28.8h, v29.8h      \n"
                    "zip2   v27.8h, v28.8h, v29.8h      \n"
                    "zip1   v28.8h, v30.8h, v31.8h      \n"
                    "zip2   v29.8h, v30.8h, v31.8h      \n"

                    "zip1   v16.4s, v22.4s, v24.4s      \n"
                    "zip2   v17.4s, v22.4s, v24.4s      \n"
                    "zip1   v18.4s, v23.4s, v25.4s      \n"
                    "zip2   v19.4s, v23.4s, v25.4s      \n"
                    "zip1   v20.4s, v26.4s, v28.4s      \n"
                    "zip2   v21.4s, v26.4s, v28.4s      \n"
                    "zip1   v22.4s, v27.4s, v29.4s      \n"
                    "zip2   v23.4s, v27.4s, v29.4s      \n"

                    "zip1   v24.2d, v16.2d, v20.2d      \n"
                    "zip2   v25.2d, v16.2d, v20.2d      \n"
                    "zip1   v26.2d, v17.2d, v21.2d      \n"
                    "zip2   v27.2d, v17.2d, v21.2d      \n"
                    "zip1   v28.2d, v18.2d, v22.2d      \n"
                    "zip2   v29.2d, v18.2d, v22.2d      \n"
                    "zip1   v30.2d, v19.2d, v23.2d      \n"
                    "zip2   v31.2d, v19.2d, v23.2d      \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v24.8h}, [%3], #16         \n"
                    "st1    {v25.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v26.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v27.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v28.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v29.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v30.8h}, [x4]              \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v31.8h}, [x4]              \n"

                    "10:                                \n"
                    "add    %0, %0, #128                \n"
                    "b      12f                         \n"

                    "11:                                \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                    "12:                                \n"

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
            }
#else  // NCNN_GNU_INLINE_ASM
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
                if (pC)
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
                else
                {
                    _sum0 = vdupq_n_f16(0.f);
                    _sum1 = vdupq_n_f16(0.f);
                    _sum2 = vdupq_n_f16(0.f);
                    _sum3 = vdupq_n_f16(0.f);
                    _sum4 = vdupq_n_f16(0.f);
                    _sum5 = vdupq_n_f16(0.f);
                    _sum6 = vdupq_n_f16(0.f);
                    _sum7 = vdupq_n_f16(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1q_f16(outptr + 8);
                _sum2 = vld1q_f16(outptr + 8 * 2);
                _sum3 = vld1q_f16(outptr + 8 * 3);
                _sum4 = vld1q_f16(outptr + 8 * 4);
                _sum5 = vld1q_f16(outptr + 8 * 5);
                _sum6 = vld1q_f16(outptr + 8 * 6);
                _sum7 = vld1q_f16(outptr + 8 * 7);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pA = vld1q_f16(pA);

                float16x8_t _pB = vld1q_f16(pB);

                _sum0 = vfmaq_laneq_f16(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _pA, _pB, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _pA, _pB, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _pA, _pB, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _pA, _pB, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _pA, _pB, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _pA, _pB, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _pA, _pB, 7);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8, _sum1);
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
                    vst1q_f16(outptr0 + out_hstep, _sum1);
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
                vst1q_f16(outptr + 8, _sum1);
                vst1q_f16(outptr + 8 * 2, _sum2);
                vst1q_f16(outptr + 8 * 3, _sum3);
                vst1q_f16(outptr + 8 * 4, _sum4);
                vst1q_f16(outptr + 8 * 5, _sum5);
                vst1q_f16(outptr + 8 * 6, _sum6);
                vst1q_f16(outptr + 8 * 7, _sum7);
            }

            outptr += 64;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const __fp16* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v28.8h}, [%8]              \n"
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
                "ld1    {v0.8h, v1.8h}, [%2], #32   \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n"
                "fmla   v28.8h, v4.8h, v0.h[0]      \n"
                "fmla   v29.8h, v4.8h, v0.h[1]      \n"
                "fmla   v30.8h, v4.8h, v0.h[2]      \n"
                "fmla   v31.8h, v4.8h, v0.h[3]      \n"
                "fmla   v28.8h, v5.8h, v0.h[4]      \n"
                "fmla   v29.8h, v5.8h, v0.h[5]      \n"
                "fmla   v30.8h, v5.8h, v0.h[6]      \n"
                "fmla   v31.8h, v5.8h, v0.h[7]      \n"
                "fmla   v28.8h, v6.8h, v1.h[0]      \n"
                "fmla   v29.8h, v6.8h, v1.h[1]      \n"
                "fmla   v30.8h, v6.8h, v1.h[2]      \n"
                "fmla   v31.8h, v6.8h, v1.h[3]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.8h, v7.8h, v1.h[4]      \n"
                "fmla   v29.8h, v7.8h, v1.h[5]      \n"
                "fmla   v30.8h, v7.8h, v1.h[6]      \n"
                "fmla   v31.8h, v7.8h, v1.h[7]      \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4h}, [%2], #8           \n"
                "ld1    {v4.8h}, [%1], #16          \n"
                "fmla   v28.8h, v4.8h, v0.h[0]      \n"
                "fmla   v29.8h, v4.8h, v0.h[1]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.8h, v4.8h, v0.h[2]      \n"
                "fmla   v31.8h, v4.8h, v0.h[3]      \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "tst    %w11, #255                  \n"
                "beq    11f                         \n"

                // if out_elempack == 8
                "cmp    %w12, #8                    \n"
                "bne    8f                          \n"

                "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%3], #64 \n"
                "b      10f                         \n"

                // if out_elempack == 4
                "8:                                 \n"
                "cmp    %w12, #4                    \n"
                "bne    9f                          \n"

                "zip1   v24.2d, v28.2d, v29.2d      \n"
                "zip2   v26.2d, v28.2d, v29.2d      \n"
                "zip1   v25.2d, v30.2d, v31.2d      \n"
                "zip2   v27.2d, v30.2d, v31.2d      \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 1          \n"
                "st1    {v24.8h, v25.8h}, [%3], #32 \n"
                "st1    {v26.8h, v27.8h}, [x4]      \n"
                "b      10f                         \n"

                // if out_elempack == 1
                "9:                                 \n"
                // transpose8x4
                "zip1   v24.8h, v28.8h, v29.8h      \n"
                "zip2   v25.8h, v28.8h, v29.8h      \n"
                "zip1   v26.8h, v30.8h, v31.8h      \n"
                "zip2   v27.8h, v30.8h, v31.8h      \n"

                "zip1   v20.4s, v24.4s, v26.4s      \n"
                "zip2   v22.4s, v24.4s, v26.4s      \n"
                "zip1   v24.4s, v25.4s, v27.4s      \n"
                "zip2   v26.4s, v25.4s, v27.4s      \n"

                "mov    v21.d[0], v20.d[1]          \n"
                "mov    v23.d[0], v22.d[1]          \n"
                "mov    v25.d[0], v24.d[1]          \n"
                "mov    v27.d[0], v26.d[1]          \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v20.4h}, [%3], #8          \n"
                "st1    {v21.4h}, [x4]              \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v22.4h}, [x4]              \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v23.4h}, [x4]              \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v24.4h}, [x4]              \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v25.4h}, [x4]              \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v26.4h}, [x4]              \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v27.4h}, [x4]              \n"

                "10:                                \n"
                "add    %0, %0, #64                 \n"
                "b      12f                         \n"

                "11:                                \n"
                "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                "12:                                \n"

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
                : "cc", "memory", "x4", "v0", "v1", "v4", "v5", "v6", "v7", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float16x8_t _sum0;
            float16x8_t _sum1;
            float16x8_t _sum2;
            float16x8_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f16(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f16(0.f);
                    _sum1 = vdupq_n_f16(0.f);
                    _sum2 = vdupq_n_f16(0.f);
                    _sum3 = vdupq_n_f16(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1q_f16(outptr + 8);
                _sum2 = vld1q_f16(outptr + 8 * 2);
                _sum3 = vld1q_f16(outptr + 8 * 3);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pA = vld1q_f16(pA);

                float16x4_t _pB = vld1_f16(pB);

                _sum0 = vfmaq_lane_f16(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _pA, _pB, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _pA, _pB, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _pA, _pB, 3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _sum0);
                    vst1q_f16(outptr0 + 8, _sum1);
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
                vst1q_f16(outptr + 8, _sum1);
                vst1q_f16(outptr + 8 * 2, _sum2);
                vst1q_f16(outptr + 8 * 3, _sum3);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const __fp16* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v30.8h, v31.8h}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v30.8h}, [%8]              \n"
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
                "ld1    {v0.8h}, [%2], #16          \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n"
                "fmla   v28.8h, v4.8h, v0.h[0]      \n"
                "fmla   v29.8h, v4.8h, v0.h[1]      \n"
                "fmla   v30.8h, v5.8h, v0.h[2]      \n"
                "fmla   v31.8h, v5.8h, v0.h[3]      \n"
                "fmla   v28.8h, v6.8h, v0.h[4]      \n"
                "fmla   v29.8h, v6.8h, v0.h[5]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.8h, v7.8h, v0.h[6]      \n"
                "fmla   v31.8h, v7.8h, v0.h[7]      \n"
                "bne    4b                          \n"
                "fadd   v30.8h, v30.8h, v28.8h      \n"
                "fadd   v31.8h, v31.8h, v29.8h      \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.s}[0], [%2], #4         \n"
                "ld1    {v4.8h}, [%1], #16          \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.8h, v4.8h, v0.h[0]      \n"
                "fmla   v31.8h, v4.8h, v0.h[1]      \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "tst    %w11, #255                  \n"
                "beq    11f                         \n"

                // if out_elempack == 8
                "cmp    %w12, #8                    \n"
                "bne    8f                          \n"

                "st1    {v30.8h, v31.8h}, [%3], #32 \n"
                "b      10f                         \n"

                // if out_elempack == 4
                "8:                                 \n"
                "cmp    %w12, #4                    \n"
                "bne    9f                          \n"

                "zip1   v28.2d, v30.2d, v31.2d      \n"
                "zip2   v29.2d, v30.2d, v31.2d      \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 1          \n"
                "st1    {v28.8h}, [%3], #16         \n"
                "st1    {v29.8h}, [x4]              \n"
                "b      10f                         \n"

                // if out_elempack == 1
                "9:                                 \n"
                // transpose8x2
                "zip1   v28.8h, v30.8h, v31.8h      \n"
                "zip2   v29.8h, v30.8h, v31.8h      \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v28.s}[0], [%3], #4        \n"
                "st1    {v28.s}[1], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v28.s}[2], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v28.s}[3], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v29.s}[0], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v29.s}[1], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v29.s}[2], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v29.s}[3], [x4]            \n"

                "10:                                \n"
                "add    %0, %0, #64                 \n"
                "b      12f                         \n"

                "11:                                \n"
                "st1    {v30.8h, v31.8h}, [%0], #32 \n"

                "12:                                \n"

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
                : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float16x8_t _sum0;
            float16x8_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f16(pC);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f16(0.f);
                    _sum1 = vdupq_n_f16(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
                _sum1 = vld1q_f16(outptr + 8);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pA = vld1q_f16(pA);

                float16x4_t _pB = vld1_f16(pB);

                _sum0 = vfmaq_lane_f16(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _pA, _pB, 1);

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
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj < max_jj; jj += 1)
        {
            const __fp16* pA = pAT;

            float16x8_t _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f16(pC);
                }
                else
                {
                    _sum0 = vdupq_n_f16(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f16(outptr);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float16x8_t _pA = vld1q_f16(pA);

                float16x8_t _pB = vld1q_dup_f16(pB);

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
            pC = (const __fp16*)CT_tile + i + ii;
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
                if (pC)
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
                else
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
                if (pC)
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
                else
                {
                    _sum0 = vdup_n_f16(0.f);
                    _sum1 = vdup_n_f16(0.f);
                    _sum2 = vdup_n_f16(0.f);
                    _sum3 = vdup_n_f16(0.f);
                    _sum4 = vdup_n_f16(0.f);
                    _sum5 = vdup_n_f16(0.f);
                    _sum6 = vdup_n_f16(0.f);
                    _sum7 = vdup_n_f16(0.f);
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
                if (pC)
                {
                    _sum0 = vld1_f16(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = vdup_n_f16(0.f);
                    _sum1 = vdup_n_f16(0.f);
                    _sum2 = vdup_n_f16(0.f);
                    _sum3 = vdup_n_f16(0.f);
                }
            }
            else
            {
                _sum0 = vld1_f16(outptr);
                _sum1 = vld1_f16(outptr + 4);
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
                    vst1_f16(outptr0 + out_hstep, _sum1);
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
                if (pC)
                {
                    _sum0 = vld1_f16(pC);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = vdup_n_f16(0.f);
                    _sum1 = vdup_n_f16(0.f);
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

                _sum0 = vfma_n_f16(_sum0, _pA, pB[0]);
                _sum1 = vfma_n_f16(_sum1, _pA, pB[1]);

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
                if (pC)
                {
                    _sum0 = vld1_f16(pC);
                }
                else
                {
                    _sum0 = vdup_n_f16(0.f);
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
            pC = (const __fp16*)CT_tile + i + ii;
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
                if (pC)
                {
                    _sum00 = vdup_n_f16(pC[0]);
                    _sum01 = vdup_n_f16(pC[0]);
                    _sum02 = vdup_n_f16(pC[0]);
                    _sum10 = vdup_n_f16(pC[1]);
                    _sum11 = vdup_n_f16(pC[1]);
                    _sum12 = vdup_n_f16(pC[1]);
                }
                else
                {
                    _sum00 = vdup_n_f16(0.f);
                    _sum01 = vdup_n_f16(0.f);
                    _sum02 = vdup_n_f16(0.f);
                    _sum10 = vdup_n_f16(0.f);
                    _sum11 = vdup_n_f16(0.f);
                    _sum12 = vdup_n_f16(0.f);
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

                float16x4_t _pA0 = vld1_dup_f16(pA);
                float16x4_t _pA1 = vld1_dup_f16(pA + 1);

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
                if (pC)
                {
                    _sum00 = vdup_n_f16(pC[0]);
                    _sum01 = vdup_n_f16(pC[0]);
                    _sum10 = vdup_n_f16(pC[1]);
                    _sum11 = vdup_n_f16(pC[1]);
                }
                else
                {
                    _sum00 = vdup_n_f16(0.f);
                    _sum01 = vdup_n_f16(0.f);
                    _sum10 = vdup_n_f16(0.f);
                    _sum11 = vdup_n_f16(0.f);
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

                float16x4_t _pA0 = vld1_dup_f16(pA);
                float16x4_t _pA1 = vld1_dup_f16(pA + 1);

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
                if (pC)
                {
                    _sum0 = vdup_n_f16(pC[0]);
                    _sum1 = vdup_n_f16(pC[1]);
                }
                else
                {
                    _sum0 = vdup_n_f16(0.f);
                    _sum1 = vdup_n_f16(0.f);
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

                _sum0 = vfma_n_f16(_sum0, _pB, pA[0]);
                _sum1 = vfma_n_f16(_sum1, _pB, pA[1]);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_f16(outptr0, (_sum0));
                    vst1_f16(outptr0 + out_hstep, (_sum1));
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
            pC = (const __fp16*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            float16x4_t _sum0;
            float16x4_t _sum1;
            float16x4_t _sum2;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vdup_n_f16(pC[0]);
                    _sum1 = vdup_n_f16(pC[0]);
                    _sum2 = vdup_n_f16(pC[0]);
                }
                else
                {
                    _sum0 = vdup_n_f16(0.f);
                    _sum1 = vdup_n_f16(0.f);
                    _sum2 = vdup_n_f16(0.f);
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
                if (pC)
                {
                    _sum0 = vdup_n_f16(pC[0]);
                    _sum1 = vdup_n_f16(pC[0]);
                }
                else
                {
                    _sum0 = vdup_n_f16(0.f);
                    _sum1 = vdup_n_f16(0.f);
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
                if (pC)
                {
                    _sum = vdup_n_f16(pC[0]);
                }
                else
                {
                    _sum = vdup_n_f16(0.f);
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

static void convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp16 = (int)(get_cpu_level2_cache_size() / sizeof(unsigned short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
        int tile_size = (l2_cache_size_fp16 - 32) / 12;

        TILE_K = std::max(8, tile_size / 8 * 8);

        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
    }

    // solve M
    {
        int nn_M = (M + 63) / 64;

        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);

        if (nT > 1)
        {
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_fp16 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp16 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

        TILE_N = std::max(4, tile_size / 4 * 4);

        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
    }
}

static void convolution_im2col_gemm_transform_kernel_fp16sa(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel_fp16sa %p", kernel.data);
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
    if (opt.use_packing_layout)
    {
        elempack = inch % 8 == 0 ? 8 : inch % 4 == 0 ? 4 : 1;
    }

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        cast_float32_to_float16(kernel, A_data);
        A_data = A_data.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)2u);

        for (int q = 0; q < outch; q += 1)
        {
            __fp16* g00 = A_data.row<__fp16>(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = (__fp16)k00[k];
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

            convolution_im2col_pack_A_tile_fp16(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_fp16sa %p %p %p %p", bottom_blob.data, top_blob.data, AT.data, bias.data);
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa(M, N, K, TILE_M, TILE_N, TILE_K, nT);

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
        convolution_im2col_input_tile_fp16(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
    {
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
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

                convolution_gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end, opt.use_a53_a55_optimized_kernel);
            }
        }
    }

    return 0;
}
