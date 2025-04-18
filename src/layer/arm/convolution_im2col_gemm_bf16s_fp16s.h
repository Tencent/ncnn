// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void convolution_im2col_pack_A_tile_bf16_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
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

static void convolution_im2col_input_tile_conv1x1s1d1_bf16_fp16(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
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

static void convolution_im2col_input_tile_bf16_fp16(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_bf16_fp16(bottom_blob, B, j, max_jj, k, max_kk);
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
