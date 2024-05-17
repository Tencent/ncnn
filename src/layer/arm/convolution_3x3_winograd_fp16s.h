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

static void conv3x3s1_winograd_pack_A_tile_fp16(const Mat& A, Mat& AT, int batch, int max_ii, int max_kk)
{
    const int N = max_kk * batch;

    for (int b = 0; b < batch; b++)
    {
        unsigned short* pp = AT.row<unsigned short>(b);

        int ii = 0;
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[2 * N];
                pp[3] = p0[3 * N];
                pp[4] = p0[4 * N];
                pp[5] = p0[5 * N];
                pp[6] = p0[6 * N];
                pp[7] = p0[7 * N];
                p0 += batch;
                pp += 8;
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[2 * N];
                pp[3] = p0[3 * N];
                p0 += batch;
                pp += 4;
            }
        }
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                p0 += batch;
                pp += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += batch;
                pp += 1;
            }
        }
    }
}

static void conv3x3s1_winograd_transpose_pack_B_tile_fp16(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk, int nT)
{
    #pragma omp parallel for num_threads(nT)
    for (int b = 0; b < batch; b++)
    {
        unsigned short* pp = BT.row<unsigned short>(b);

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                // transpose 8x12
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v8.8h, v9.8h, v10.8h, v11.8h}, [%0] \n"

                    "uzp1   v12.8h, v0.8h, v4.8h        \n"
                    "uzp2   v16.8h, v0.8h, v4.8h        \n"
                    "uzp1   v13.8h, v1.8h, v5.8h        \n"
                    "uzp2   v17.8h, v1.8h, v5.8h        \n"
                    "uzp1   v14.8h, v2.8h, v6.8h        \n"
                    "uzp2   v18.8h, v2.8h, v6.8h        \n"
                    "uzp1   v15.8h, v3.8h, v7.8h        \n"
                    "uzp2   v19.8h, v3.8h, v7.8h        \n"
                    "uzp1   v20.8h, v8.8h, v9.8h        \n"
                    "uzp2   v22.8h, v8.8h, v9.8h        \n"
                    "uzp1   v21.8h, v10.8h, v11.8h      \n"
                    "uzp2   v23.8h, v10.8h, v11.8h      \n"

                    "sub    %0, %0, #128                \n"

                    "ext    v24.16b, v20.16b, v20.16b, #8 \n"
                    "ext    v26.16b, v22.16b, v22.16b, #8 \n"
                    "ext    v25.16b, v21.16b, v21.16b, #8 \n"
                    "ext    v27.16b, v23.16b, v23.16b, #8 \n"

                    "st1    {v12.8h}, [%1], #16         \n"
                    "st1    {v20.4h}, [%1], #8          \n"
                    "st1    {v13.8h}, [%1], #16         \n"
                    "st1    {v24.4h}, [%1], #8          \n"
                    "st1    {v14.8h}, [%1], #16         \n"
                    "st1    {v21.4h}, [%1], #8          \n"
                    "st1    {v15.8h}, [%1], #16         \n"
                    "st1    {v25.4h}, [%1], #8          \n"
                    "st1    {v16.8h}, [%1], #16         \n"
                    "st1    {v22.4h}, [%1], #8          \n"
                    "st1    {v17.8h}, [%1], #16         \n"
                    "st1    {v26.4h}, [%1], #8          \n"
                    "st1    {v18.8h}, [%1], #16         \n"
                    "st1    {v23.4h}, [%1], #8          \n"
                    "st1    {v19.8h}, [%1], #16         \n"
                    "st1    {v27.4h}, [%1], #8          \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
                p0 += max_jj * batch * 8;
#else  // NCNN_GNU_INLINE_ASM
                uint16x8x4_t _r0 = vld4q_u16(p0);
                uint16x8x4_t _r1 = vld4q_u16(p0 + 32);
                uint16x8x4_t _r2 = vld4q_u16(p0 + 64);
                uint16x8x2_t _r04lm = vuzpq_u16(_r0.val[0], _r1.val[0]);
                uint16x8x2_t _r15lm = vuzpq_u16(_r0.val[1], _r1.val[1]);
                uint16x8x2_t _r26lm = vuzpq_u16(_r0.val[2], _r1.val[2]);
                uint16x8x2_t _r37lm = vuzpq_u16(_r0.val[3], _r1.val[3]);
                uint16x8x2_t _r0145h = vuzpq_u16(_r2.val[0], _r2.val[1]);
                uint16x8x2_t _r2367h = vuzpq_u16(_r2.val[2], _r2.val[3]);
                vst1q_u16(pp, _r04lm.val[0]);
                vst1_u16(pp + 8, vget_low_u16(_r0145h.val[0]));
                vst1q_u16(pp + 12, _r15lm.val[0]);
                vst1_u16(pp + 20, vget_high_u16(_r0145h.val[0]));
                vst1q_u16(pp + 24, _r26lm.val[0]);
                vst1_u16(pp + 32, vget_low_u16(_r2367h.val[0]));
                vst1q_u16(pp + 36, _r37lm.val[0]);
                vst1_u16(pp + 44, vget_high_u16(_r2367h.val[0]));
                vst1q_u16(pp + 48, _r04lm.val[1]);
                vst1_u16(pp + 56, vget_low_u16(_r0145h.val[1]));
                vst1q_u16(pp + 60, _r15lm.val[1]);
                vst1_u16(pp + 68, vget_high_u16(_r0145h.val[1]));
                vst1q_u16(pp + 72, _r26lm.val[1]);
                vst1_u16(pp + 80, vget_low_u16(_r2367h.val[1]));
                vst1q_u16(pp + 84, _r37lm.val[1]);
                vst1_u16(pp + 92, vget_high_u16(_r2367h.val[1]));
                p0 += max_jj * batch * 8;
                pp += 96;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // transpose 4x12
                uint16x8x4_t _r01 = vld4q_u16(p0);
                uint16x4x4_t _r2 = vld4_u16(p0 + 32);
                vst1q_u16(pp, _r01.val[0]);
                vst1_u16(pp + 8, _r2.val[0]);
                vst1q_u16(pp + 12, _r01.val[1]);
                vst1_u16(pp + 20, _r2.val[1]);
                vst1q_u16(pp + 24, _r01.val[2]);
                vst1_u16(pp + 32, _r2.val[2]);
                vst1q_u16(pp + 36, _r01.val[3]);
                vst1_u16(pp + 44, _r2.val[3]);
                p0 += max_jj * batch * 4;
                pp += 48;
            }
            p0 -= (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                // transpose 2x12
                uint16x8x2_t _r01 = vld2q_u16(p0);
                uint16x4x2_t _r2 = vld2_u16(p0 + 16);
                vst1q_u16(pp, _r01.val[0]);
                vst1_u16(pp + 8, _r2.val[0]);
                vst1q_u16(pp + 12, _r01.val[1]);
                vst1_u16(pp + 20, _r2.val[1]);
                p0 += max_jj * batch * 2;
                pp += 24;
            }
            p0 -= (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                uint16x8_t _r01 = vld1q_u16(p0);
                uint16x4_t _r2 = vld1_u16(p0 + 8);
                vst1q_u16(pp, _r01);
                vst1_u16(pp + 8, _r2);
                p0 += max_jj * batch;
                pp += 12;
            }
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                // transpose 8x8
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0] \n"

                    "uzp1   v8.8h, v0.8h, v4.8h         \n"
                    "uzp2   v12.8h, v0.8h, v4.8h        \n"
                    "uzp1   v9.8h, v1.8h, v5.8h         \n"
                    "uzp2   v13.8h, v1.8h, v5.8h        \n"

                    "sub    %0, %0, #64                 \n"

                    "uzp1   v10.8h, v2.8h, v6.8h        \n"
                    "uzp2   v14.8h, v2.8h, v6.8h        \n"
                    "uzp1   v11.8h, v3.8h, v7.8h        \n"
                    "uzp2   v15.8h, v3.8h, v7.8h        \n"

                    "st1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%1], #64 \n"
                    "st1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                p0 += max_jj * batch * 8;
#else  // NCNN_GNU_INLINE_ASM
                uint16x8x4_t _r0 = vld4q_u16(p0);
                uint16x8x4_t _r1 = vld4q_u16(p0 + 32);
                uint16x8x2_t _r04 = vuzpq_u16(_r0.val[0], _r1.val[0]);
                uint16x8x2_t _r15 = vuzpq_u16(_r0.val[1], _r1.val[1]);
                uint16x8x2_t _r26 = vuzpq_u16(_r0.val[2], _r1.val[2]);
                uint16x8x2_t _r37 = vuzpq_u16(_r0.val[3], _r1.val[3]);
                vst1q_u16(pp, _r04.val[0]);
                vst1q_u16(pp + 8, _r15.val[0]);
                vst1q_u16(pp + 8 * 2, _r26.val[0]);
                vst1q_u16(pp + 8 * 3, _r37.val[0]);
                vst1q_u16(pp + 8 * 4, _r04.val[1]);
                vst1q_u16(pp + 8 * 5, _r15.val[1]);
                vst1q_u16(pp + 8 * 6, _r26.val[1]);
                vst1q_u16(pp + 8 * 7, _r37.val[1]);
                p0 += max_jj * batch * 8;
                pp += 64;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // transpose 4x8
                uint16x8x4_t _r0 = vld4q_u16(p0);
                vst1q_u16(pp, _r0.val[0]);
                vst1q_u16(pp + 8, _r0.val[1]);
                vst1q_u16(pp + 16, _r0.val[2]);
                vst1q_u16(pp + 24, _r0.val[3]);
                p0 += max_jj * batch * 4;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                // transpose 2x8
                uint16x8x2_t _r0 = vld2q_u16(p0);
                vst1q_u16(pp, _r0.val[0]);
                vst1q_u16(pp + 8, _r0.val[1]);
                p0 += max_jj * batch * 2;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                uint16x8_t _r0 = vld1q_u16(p0);
                vst1q_u16(pp, _r0);
                p0 += max_jj * batch;
                pp += 8;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                // transpose 8x4
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
                p0 += max_jj * batch * 8;
#else  // NCNN_GNU_INLINE_ASM
                uint16x8x4_t _r0;
                _r0.val[0] = vld1q_u16(p0);
                _r0.val[1] = vld1q_u16(p0 + 8);
                _r0.val[2] = vld1q_u16(p0 + 16);
                _r0.val[3] = vld1q_u16(p0 + 24);
                vst4q_u16(pp, _r0);
                p0 += max_jj * batch * 8;
                pp += 32;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // transpose 4x4
                uint16x4x4_t _r0;
                _r0.val[0] = vld1_u16(p0);
                _r0.val[1] = vld1_u16(p0 + 4);
                _r0.val[2] = vld1_u16(p0 + 8);
                _r0.val[3] = vld1_u16(p0 + 12);
                vst4_u16(pp, _r0);
                p0 += max_jj * batch * 4;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                // transpose 2x4
                uint16x4x2_t _r0 = vld2_u16(p0);
                vst1_u16(pp, _r0.val[0]);
                vst1_u16(pp + 4, _r0.val[1]);
                p0 += max_jj * batch * 2;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                vst1_u16(pp, _r0);
                p0 += max_jj * batch;
                pp += 4;
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                // transpose 8x2
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]       \n"
                    "ld1    {v0.8h, v1.8h}, [%0]        \n"
                    "st2    {v0.8h, v1.8h}, [%1], #32   \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp)
                    : "memory", "v0", "v1");
                p0 += max_jj * batch * 8;
#else  // NCNN_GNU_INLINE_ASM
                uint16x8x2_t _r0;
                _r0.val[0] = vld1q_u16(p0);
                _r0.val[1] = vld1q_u16(p0 + 8);
                vst2q_u16(pp, _r0);
                p0 += max_jj * batch * 8;
                pp += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // transpose 4x2
                uint16x4x2_t _r0;
                _r0.val[0] = vld1_u16(p0);
                _r0.val[1] = vld1_u16(p0 + 4);
                vst2_u16(pp, _r0);
                p0 += max_jj * batch * 4;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[2];
                pp[2] = p0[1];
                pp[3] = p0[3];
                p0 += max_jj * batch * 2;
                pp += 4;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                p0 += max_jj * batch;
                pp += 2;
            }
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _r0 = vld1q_u16(p0);
                vst1q_u16(pp, _r0);
                p0 += max_jj * batch * 8;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x4_t _r0 = vld1_u16(p0);
                vst1_u16(pp, _r0);
                p0 += max_jj * batch * 4;
                pp += 4;
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                p0 += max_jj * batch * 2;
                pp += 2;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += max_jj * batch;
                pp += 1;
            }
        }
    }
}

static void conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk, int use_a53_a55_optimized_kernel)
{
    // NCNN_LOGE("conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa %d %d %d", max_ii, max_jj, max_kk);
    __fp16* outptr = top_blob;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
        for (int b = 0; b < batch; b++)
        {
            const __fp16* pAT = AT_tile.row<const __fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<const __fp16>(b);

            int jj = 0;
            for (; jj + 11 < max_jj; jj += 12)
            {
                const __fp16* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                if (use_a53_a55_optimized_kernel)
                {
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%0], #64 \n"
                        "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                        "subs   %0, %0, #128                \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v20.16b, v20.16b, v20.16b   \n"
                        "eor    v21.16b, v21.16b, v21.16b   \n"
                        "eor    v22.16b, v22.16b, v22.16b   \n"
                        "eor    v23.16b, v23.16b, v23.16b   \n"
                        "eor    v24.16b, v24.16b, v24.16b   \n"
                        "eor    v25.16b, v25.16b, v25.16b   \n"
                        "eor    v26.16b, v26.16b, v26.16b   \n"
                        "eor    v27.16b, v27.16b, v27.16b   \n"
                        "eor    v28.16b, v28.16b, v28.16b   \n"
                        "eor    v29.16b, v29.16b, v29.16b   \n"
                        "eor    v30.16b, v30.16b, v30.16b   \n"
                        "eor    v31.16b, v31.16b, v31.16b   \n"

                        "1:                                 \n"
                        "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                        "cmp    w4, #0                      \n"
                        "beq    3f                          \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.8h}, [%1], #16          \n"
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.8h}, [%2], #16          \n"

                        "ldr    d1, [%2], #8                \n"
                        "ldr    x21, [%2], #8               \n"

                        ".align 4                           \n"
                        "2:                                 \n"
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
                        "bne    2b                          \n"

                        "sub    %1, %1, #16                 \n"
                        "sub    %2, %2, #32                 \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
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
                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%0], #64 \n"
                        "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                else
                {
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%0], #64 \n"
                        "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                        "subs   %0, %0, #128                \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v20.16b, v20.16b, v20.16b   \n"
                        "eor    v21.16b, v21.16b, v21.16b   \n"
                        "eor    v22.16b, v22.16b, v22.16b   \n"
                        "eor    v23.16b, v23.16b, v23.16b   \n"
                        "eor    v24.16b, v24.16b, v24.16b   \n"
                        "eor    v25.16b, v25.16b, v25.16b   \n"
                        "eor    v26.16b, v26.16b, v26.16b   \n"
                        "eor    v27.16b, v27.16b, v27.16b   \n"
                        "eor    v28.16b, v28.16b, v28.16b   \n"
                        "eor    v29.16b, v29.16b, v29.16b   \n"
                        "eor    v30.16b, v30.16b, v30.16b   \n"
                        "eor    v31.16b, v31.16b, v31.16b   \n"

                        "1:                                 \n"
                        "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                        "cmp    w4, #0                      \n"
                        "beq    3f                          \n"

                        "2:                                 \n"
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

                        "bne    2b                          \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
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
                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%0], #64 \n"
                        "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
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
                else
                {
                    _sum0 = vld1q_f16(outptr);
                    _sum1 = vld1q_f16(outptr + 8);
                    _sum2 = vld1q_f16(outptr + 16);
                    _sum3 = vld1q_f16(outptr + 24);
                    _sum4 = vld1q_f16(outptr + 32);
                    _sum5 = vld1q_f16(outptr + 40);
                    _sum6 = vld1q_f16(outptr + 48);
                    _sum7 = vld1q_f16(outptr + 56);
                    _sum8 = vld1q_f16(outptr + 64);
                    _sum9 = vld1q_f16(outptr + 72);
                    _suma = vld1q_f16(outptr + 80);
                    _sumb = vld1q_f16(outptr + 88);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x8_t _pA = vld1q_f16(pA);
                    float16x8_t _pB0 = vld1q_f16(pB);
                    float16x4_t _pB2 = vld1_f16(pB + 8);
                    _sum0 = vfmaq_laneq_f16(_sum0, _pA, _pB0, 0);
                    _sum1 = vfmaq_laneq_f16(_sum1, _pA, _pB0, 1);
                    _sum2 = vfmaq_laneq_f16(_sum2, _pA, _pB0, 2);
                    _sum3 = vfmaq_laneq_f16(_sum3, _pA, _pB0, 3);
                    _sum4 = vfmaq_laneq_f16(_sum4, _pA, _pB0, 4);
                    _sum5 = vfmaq_laneq_f16(_sum5, _pA, _pB0, 5);
                    _sum6 = vfmaq_laneq_f16(_sum6, _pA, _pB0, 6);
                    _sum7 = vfmaq_laneq_f16(_sum7, _pA, _pB0, 7);
                    _sum8 = vfmaq_lane_f16(_sum8, _pA, _pB2, 0);
                    _sum9 = vfmaq_lane_f16(_sum9, _pA, _pB2, 1);
                    _suma = vfmaq_lane_f16(_suma, _pA, _pB2, 2);
                    _sumb = vfmaq_lane_f16(_sumb, _pA, _pB2, 3);

                    pA += 8;
                    pB += 12;
                }

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
                outptr += 8 * 12;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                if (use_a53_a55_optimized_kernel)
                {
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                        "subs   %0, %0, #64                 \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v24.16b, v24.16b, v24.16b   \n"
                        "eor    v25.16b, v25.16b, v25.16b   \n"
                        "eor    v26.16b, v26.16b, v26.16b   \n"
                        "eor    v27.16b, v27.16b, v27.16b   \n"
                        "eor    v28.16b, v28.16b, v28.16b   \n"
                        "eor    v29.16b, v29.16b, v29.16b   \n"
                        "eor    v30.16b, v30.16b, v30.16b   \n"
                        "eor    v31.16b, v31.16b, v31.16b   \n"

                        "1:                                 \n"
                        "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                        "cmp    w4, #0                      \n"
                        "beq    3f                          \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.8h}, [%1], #16          \n"
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.8h}, [%2], #16          \n"

                        "ldr    d5, [%1], #8                \n"
                        "ldr    x25, [%1], #8               \n"

                        ".align 4                           \n"
                        "2:                                 \n"
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
                        "bne    2b                          \n"

                        "sub    %1, %1, #32                 \n"
                        "sub    %2, %2, #16                 \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
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
                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                else
                {
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0]      \n"
                        "subs   %0, %0, #64                 \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v24.16b, v24.16b, v24.16b   \n"
                        "eor    v25.16b, v25.16b, v25.16b   \n"
                        "eor    v26.16b, v26.16b, v26.16b   \n"
                        "eor    v27.16b, v27.16b, v27.16b   \n"
                        "eor    v28.16b, v28.16b, v28.16b   \n"
                        "eor    v29.16b, v29.16b, v29.16b   \n"
                        "eor    v30.16b, v30.16b, v30.16b   \n"
                        "eor    v31.16b, v31.16b, v31.16b   \n"

                        "1:                                 \n"
                        "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                        "cmp    w4, #0                      \n"
                        "beq    3f                          \n"

                        "2:                                 \n"
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n"

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

                        "bne    2b                          \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
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
                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
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
                    _sum0 = vdupq_n_f16(0.f);
                    _sum1 = vdupq_n_f16(0.f);
                    _sum2 = vdupq_n_f16(0.f);
                    _sum3 = vdupq_n_f16(0.f);
                    _sum4 = vdupq_n_f16(0.f);
                    _sum5 = vdupq_n_f16(0.f);
                    _sum6 = vdupq_n_f16(0.f);
                    _sum7 = vdupq_n_f16(0.f);
                }
                else
                {
                    _sum0 = vld1q_f16(outptr);
                    _sum1 = vld1q_f16(outptr + 8);
                    _sum2 = vld1q_f16(outptr + 16);
                    _sum3 = vld1q_f16(outptr + 24);
                    _sum4 = vld1q_f16(outptr + 32);
                    _sum5 = vld1q_f16(outptr + 40);
                    _sum6 = vld1q_f16(outptr + 48);
                    _sum7 = vld1q_f16(outptr + 56);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f16(outptr, _sum0);
                vst1q_f16(outptr + 8, _sum1);
                vst1q_f16(outptr + 8 * 2, _sum2);
                vst1q_f16(outptr + 8 * 3, _sum3);
                vst1q_f16(outptr + 8 * 4, _sum4);
                vst1q_f16(outptr + 8 * 5, _sum5);
                vst1q_f16(outptr + 8 * 6, _sum6);
                vst1q_f16(outptr + 8 * 7, _sum7);
                outptr += 8 * 8;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0] \n"
                    "b      1f                          \n"

                    "0:                                 \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"
                    "eor    v29.16b, v29.16b, v29.16b   \n"
                    "eor    v30.16b, v30.16b, v30.16b   \n"
                    "eor    v31.16b, v31.16b, v31.16b   \n"

                    "1:                                 \n"
                    "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    3f                          \n"

                    "2:                                 \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.8h, v1.8h}, [%2], #32   \n"

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

                    "bne    2b                          \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1    {v0.4h}, [%2], #8           \n"
                    "ld1    {v4.8h}, [%1], #16          \n"
                    "fmla   v28.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v29.8h, v4.8h, v0.h[1]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v31.8h, v4.8h, v0.h[3]      \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v1", "v4", "v5", "v6", "v7", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
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
                }
                else
                {
                    _sum0 = vld1q_f16(outptr);
                    _sum1 = vld1q_f16(outptr + 8);
                    _sum2 = vld1q_f16(outptr + 16);
                    _sum3 = vld1q_f16(outptr + 24);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f16(outptr, _sum0);
                vst1q_f16(outptr + 8, _sum1);
                vst1q_f16(outptr + 8 * 2, _sum2);
                vst1q_f16(outptr + 8 * 3, _sum3);
                outptr += 8 * 4;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                float16x8_t _sum0;
                float16x8_t _sum1;

                if (k == 0)
                {
                    _sum0 = vdupq_n_f16(0.f);
                    _sum1 = vdupq_n_f16(0.f);
                }
                else
                {
                    _sum0 = vld1q_f16(outptr);
                    _sum1 = vld1q_f16(outptr + 8);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x8_t _pA = vld1q_f16(pA);
                    _sum0 = vfmaq_n_f16(_sum0, _pA, pB[0]);
                    _sum1 = vfmaq_n_f16(_sum1, _pA, pB[1]);

                    pA += 8;
                    pB += 2;
                }

                vst1q_f16(outptr, _sum0);
                vst1q_f16(outptr + 8, _sum1);
                outptr += 8 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const __fp16* pA = pAT;

                float16x8_t _sum;

                if (k == 0)
                {
                    _sum = vdupq_n_f16(0.f);
                }
                else
                {
                    _sum = vld1q_f16(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x8_t _pA = vld1q_f16(pA);
                    _sum = vfmaq_n_f16(_sum, _pA, pB[0]);

                    pA += 8;
                    pB += 1;
                }

                vst1q_f16(outptr, _sum);
                outptr += 8;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        for (int b = 0; b < batch; b++)
        {
            const __fp16* pAT = AT_tile.row<const __fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<const __fp16>(b);

            int jj = 0;
            for (; jj + 11 < max_jj; jj += 12)
            {
                const __fp16* pA = pAT;

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
                }
                else
                {
                    _sum0 = vld1_f16(outptr);
                    _sum1 = vld1_f16(outptr + 4);
                    _sum2 = vld1_f16(outptr + 8);
                    _sum3 = vld1_f16(outptr + 12);
                    _sum4 = vld1_f16(outptr + 16);
                    _sum5 = vld1_f16(outptr + 20);
                    _sum6 = vld1_f16(outptr + 24);
                    _sum7 = vld1_f16(outptr + 28);
                    _sum8 = vld1_f16(outptr + 32);
                    _sum9 = vld1_f16(outptr + 36);
                    _suma = vld1_f16(outptr + 40);
                    _sumb = vld1_f16(outptr + 44);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x4_t _pA = vld1_f16(pA);
                    float16x8_t _pB0 = vld1q_f16(pB);
                    float16x4_t _pB2 = vld1_f16(pB + 8);
                    _sum0 = vfma_laneq_f16(_sum0, _pA, _pB0, 0);
                    _sum1 = vfma_laneq_f16(_sum1, _pA, _pB0, 1);
                    _sum2 = vfma_laneq_f16(_sum2, _pA, _pB0, 2);
                    _sum3 = vfma_laneq_f16(_sum3, _pA, _pB0, 3);
                    _sum4 = vfma_laneq_f16(_sum4, _pA, _pB0, 4);
                    _sum5 = vfma_laneq_f16(_sum5, _pA, _pB0, 5);
                    _sum6 = vfma_laneq_f16(_sum6, _pA, _pB0, 6);
                    _sum7 = vfma_laneq_f16(_sum7, _pA, _pB0, 7);
                    _sum8 = vfma_lane_f16(_sum8, _pA, _pB2, 0);
                    _sum9 = vfma_lane_f16(_sum9, _pA, _pB2, 1);
                    _suma = vfma_lane_f16(_suma, _pA, _pB2, 2);
                    _sumb = vfma_lane_f16(_sumb, _pA, _pB2, 3);

                    pA += 4;
                    pB += 12;
                }

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
                outptr += 4 * 12;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

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
                }
                else
                {
                    _sum0 = vld1_f16(outptr);
                    _sum1 = vld1_f16(outptr + 4);
                    _sum2 = vld1_f16(outptr + 8);
                    _sum3 = vld1_f16(outptr + 12);
                    _sum4 = vld1_f16(outptr + 16);
                    _sum5 = vld1_f16(outptr + 20);
                    _sum6 = vld1_f16(outptr + 24);
                    _sum7 = vld1_f16(outptr + 28);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x4_t _pA = vld1_f16(pA);
                    float16x8_t _pB = vld1q_f16(pB);
                    _sum0 = vfma_laneq_f16(_sum0, _pA, _pB, 0);
                    _sum1 = vfma_laneq_f16(_sum1, _pA, _pB, 1);
                    _sum2 = vfma_laneq_f16(_sum2, _pA, _pB, 2);
                    _sum3 = vfma_laneq_f16(_sum3, _pA, _pB, 3);
                    _sum4 = vfma_laneq_f16(_sum4, _pA, _pB, 4);
                    _sum5 = vfma_laneq_f16(_sum5, _pA, _pB, 5);
                    _sum6 = vfma_laneq_f16(_sum6, _pA, _pB, 6);
                    _sum7 = vfma_laneq_f16(_sum7, _pA, _pB, 7);

                    pA += 4;
                    pB += 8;
                }

                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
                vst1_f16(outptr + 4 * 2, _sum2);
                vst1_f16(outptr + 4 * 3, _sum3);
                vst1_f16(outptr + 4 * 4, _sum4);
                vst1_f16(outptr + 4 * 5, _sum5);
                vst1_f16(outptr + 4 * 6, _sum6);
                vst1_f16(outptr + 4 * 7, _sum7);
                outptr += 4 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

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
                }
                else
                {
                    _sum0 = vld1_f16(outptr);
                    _sum1 = vld1_f16(outptr + 4);
                    _sum2 = vld1_f16(outptr + 8);
                    _sum3 = vld1_f16(outptr + 12);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
                vst1_f16(outptr + 4 * 2, _sum2);
                vst1_f16(outptr + 4 * 3, _sum3);
                outptr += 4 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                float16x4_t _sum0;
                float16x4_t _sum1;

                if (k == 0)
                {
                    _sum0 = vdup_n_f16(0.f);
                    _sum1 = vdup_n_f16(0.f);
                }
                else
                {
                    _sum0 = vld1_f16(outptr);
                    _sum1 = vld1_f16(outptr + 4);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x4_t _pA = vld1_f16(pA);
                    _sum0 = vfma_n_f16(_sum0, _pA, pB[0]);
                    _sum1 = vfma_n_f16(_sum1, _pA, pB[1]);

                    pA += 4;
                    pB += 2;
                }

                vst1_f16(outptr, _sum0);
                vst1_f16(outptr + 4, _sum1);
                outptr += 4 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const __fp16* pA = pAT;

                float16x4_t _sum;

                if (k == 0)
                {
                    _sum = vdup_n_f16(0.f);
                }
                else
                {
                    _sum = vld1_f16(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x4_t _pA = vld1_f16(pA);
                    _sum = vfma_n_f16(_sum, _pA, pB[0]);

                    pA += 4;
                    pB += 1;
                }

                vst1_f16(outptr, _sum);
                outptr += 4;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        for (int b = 0; b < batch; b++)
        {
            const __fp16* pAT = AT_tile.row<const __fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<const __fp16>(b);

            int jj = 0;
            for (; jj + 11 < max_jj; jj += 12)
            {
                const __fp16* pA = pAT;

                float16x8_t _sum01;
                float16x4_t _sum2;
                float16x8_t _sum34;
                float16x4_t _sum5;

                if (k == 0)
                {
                    _sum01 = vdupq_n_f16(0.f);
                    _sum2 = vdup_n_f16(0.f);
                    _sum34 = vdupq_n_f16(0.f);
                    _sum5 = vdup_n_f16(0.f);
                }
                else
                {
                    float16x8x2_t _tmp0123 = vld2q_f16(outptr);
                    float16x4x2_t _tmp45 = vld2_f16(outptr + 16);
                    _sum01 = _tmp0123.val[0];
                    _sum2 = _tmp45.val[0];
                    _sum34 = _tmp0123.val[1];
                    _sum5 = _tmp45.val[1];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x8_t _pB0 = vld1q_f16(pB);
                    float16x4_t _pB2 = vld1_f16(pB + 8);
                    _sum01 = vfmaq_n_f16(_sum01, _pB0, pA[0]);
                    _sum2 = vfma_n_f16(_sum2, _pB2, pA[0]);
                    _sum34 = vfmaq_n_f16(_sum34, _pB0, pA[1]);
                    _sum5 = vfma_n_f16(_sum5, _pB2, pA[1]);
                    pA += 2;
                    pB += 12;
                }

                float16x8x2_t _tmp0123;
                _tmp0123.val[0] = _sum01;
                _tmp0123.val[1] = _sum34;
                float16x4x2_t _tmp45;
                _tmp45.val[0] = _sum2;
                _tmp45.val[1] = _sum5;
                vst2q_f16(outptr, _tmp0123);
                vst2_f16(outptr + 16, _tmp45);
                outptr += 2 * 12;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

                float16x8_t _sum01;
                float16x8_t _sum23;

                if (k == 0)
                {
                    _sum01 = vdupq_n_f16(0.f);
                    _sum23 = vdupq_n_f16(0.f);
                }
                else
                {
                    float16x8x2_t _tmp0123 = vld2q_f16(outptr);
                    _sum01 = _tmp0123.val[0];
                    _sum23 = _tmp0123.val[1];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x8_t _pB = vld1q_f16(pB);
                    _sum01 = vfmaq_n_f16(_sum01, _pB, pA[0]);
                    _sum23 = vfmaq_n_f16(_sum23, _pB, pA[1]);
                    pA += 2;
                    pB += 8;
                }

                float16x8x2_t _tmp0123;
                _tmp0123.val[0] = _sum01;
                _tmp0123.val[1] = _sum23;
                vst2q_f16(outptr, _tmp0123);
                outptr += 2 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

                float16x4_t _sum0;
                float16x4_t _sum1;

                if (k == 0)
                {
                    _sum0 = vdup_n_f16(0.f);
                    _sum1 = vdup_n_f16(0.f);
                }
                else
                {
                    float16x4x2_t _tmp01 = vld2_f16(outptr);
                    _sum0 = _tmp01.val[0];
                    _sum1 = _tmp01.val[1];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x4_t _pB = vld1_f16(pB);
                    _sum0 = vfma_n_f16(_sum0, _pB, pA[0]);
                    _sum1 = vfma_n_f16(_sum1, _pB, pA[1]);
                    pA += 2;
                    pB += 4;
                }

                float16x4x2_t _tmp01;
                _tmp01.val[0] = _sum0;
                _tmp01.val[1] = _sum1;
                vst2_f16(outptr, _tmp01);
                outptr += 2 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                __fp16 sum00 = 0.f;
                __fp16 sum01 = 0.f;
                __fp16 sum10 = 0.f;
                __fp16 sum11 = 0.f;

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
                    sum01 = outptr[1];
                    sum10 = outptr[2];
                    sum11 = outptr[3];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum01 += pA[1] * pB[0];
                    sum10 += pA[0] * pB[1];
                    sum11 += pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                }

                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
                outptr += 2 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const __fp16* pA = pAT;

                __fp16 sum0 = 0.f;
                __fp16 sum1 = 0.f;

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

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[1] * pB[0];
                    pA += 2;
                    pB += 1;
                }

                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr += 2;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        for (int b = 0; b < batch; b++)
        {
            const __fp16* pAT = AT_tile.row<const __fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<const __fp16>(b);

            int jj = 0;
            for (; jj + 11 < max_jj; jj += 12)
            {
                const __fp16* pA = pAT;

                float16x8_t _sum01;
                float16x4_t _sum2;

                if (k == 0)
                {
                    _sum01 = vdupq_n_f16(0.f);
                    _sum2 = vdup_n_f16(0.f);
                }
                else
                {
                    _sum01 = vld1q_f16(outptr);
                    _sum2 = vld1_f16(outptr + 8);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x8_t _pB0 = vld1q_f16(pB);
                    float16x4_t _pB2 = vld1_f16(pB + 8);
                    _sum01 = vfmaq_n_f16(_sum01, _pB0, pA[0]);
                    _sum2 = vfma_n_f16(_sum2, _pB2, pA[0]);
                    pA += 1;
                    pB += 12;
                }

                vst1q_f16(outptr, _sum01);
                vst1_f16(outptr + 8, _sum2);
                outptr += 12;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

                float16x8_t _sum01;

                if (k == 0)
                {
                    _sum01 = vdupq_n_f16(0.f);
                }
                else
                {
                    _sum01 = vld1q_f16(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x8_t _pB = vld1q_f16(pB);
                    _sum01 = vfmaq_n_f16(_sum01, _pB, pA[0]);
                    pA += 1;
                    pB += 8;
                }

                vst1q_f16(outptr, _sum01);
                outptr += 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

                float16x4_t _sum;

                if (k == 0)
                {
                    _sum = vdup_n_f16(0.f);
                }
                else
                {
                    _sum = vld1_f16(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float16x4_t _pB = vld1_f16(pB);
                    _sum = vfma_n_f16(_sum, _pB, pA[0]);
                    pA += 1;
                    pB += 4;
                }

                vst1_f16(outptr, _sum);
                outptr += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                __fp16 sum0 = 0.f;
                __fp16 sum1 = 0.f;

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

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[0] * pB[1];
                    pA += 1;
                    pB += 2;
                }

                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr += 2;
            }
            for (; jj < max_jj; jj++)
            {
                const __fp16* pA = pAT;

                __fp16 sum = 0.f;

                if (k == 0)
                {
                    sum = 0.f;
                }
                else
                {
                    sum = outptr[0];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum += pA[0] * pB[0];
                    pA += 1;
                    pB += 1;
                }

                outptr[0] = sum;
                outptr += 1;
            }
        }
    }
}

static void conv3x3s1_winograd_get_optimal_tile_mnk_fp16(int M, int N, int K, int B, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp16 = (int)(get_cpu_level2_cache_size() / sizeof(unsigned short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // we shall take B into account for batched gemm, but that will be slower on arm in practice, why ?
    (void)B;

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
        TILE_M = 8;
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

static inline void conv3x3s1_winograd23_transform_kernel_tile_fp16sa(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const float ktm[4][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {1.0f / 2, 1.0f / 2, 1.0f / 2},
    //     {1.0f / 2, -1.0f / 2, 1.0f / 2},
    //     {0.0f, 0.0f, 1.0f}
    // };

    __fp16* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            float tmp[4][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = r0 * 0.5f + r1 * 0.5f + r2 * 0.5f;
                tmp[2][m] = r0 * 0.5f - r1 * 0.5f + r2 * 0.5f;
                tmp[3][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 4; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = r0 * 0.5f + r1 * 0.5f + r2 * 0.5f;
                float z2 = r0 * 0.5f - r1 * 0.5f + r2 * 0.5f;
                float z3 = r2;

                ptmp[0] = (__fp16)z0;
                ptmp[1] = (__fp16)z1;
                ptmp[2] = (__fp16)z2;
                ptmp[3] = (__fp16)z3;
                ptmp += 4;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_fp16sa(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, (size_t)2u);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd23_transform_kernel_tile_fp16sa(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile_fp16(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd23_transform_input_tile_fp16sa(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w - 1) / 2;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[4][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel((k + kk) / elempack).row<const __fp16>(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                float16x8_t _r0 = vdupq_n_f16(0.f);
                float16x8_t _r1 = vdupq_n_f16(0.f);
                float16x8_t _r2 = vdupq_n_f16(0.f);
                float16x8_t _r3 = vdupq_n_f16(0.f);

                if (ti * 2 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = vld1q_f16(r0);
                        if (tj * 2 + 1 < w) _r1 = vld1q_f16(r0 + 8);
                        if (tj * 2 + 2 < w) _r2 = vld1q_f16(r0 + 16);
                        if (tj * 2 + 3 < w) _r3 = vld1q_f16(r0 + 24);
                    }
                    if (elempack == 4)
                    {
                        const __fp16* r1 = r0 + N;

                        _r0 = vcombine_f16(vld1_f16(r0), vld1_f16(r1));
                        if (tj * 2 + 1 < w)
                        {
                            _r1 = vcombine_f16(vld1_f16(r0 + 4), vld1_f16(r1 + 4));
                        }
                        if (tj * 2 + 2 < w)
                        {
                            _r2 = vcombine_f16(vld1_f16(r0 + 8), vld1_f16(r1 + 8));
                        }
                        if (tj * 2 + 3 < w)
                        {
                            _r3 = vcombine_f16(vld1_f16(r0 + 12), vld1_f16(r1 + 12));
                        }
                    }
                    if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;
                        const __fp16* r2 = r0 + N * 2;
                        const __fp16* r3 = r0 + N * 3;
                        const __fp16* r4 = r0 + N * 4;
                        const __fp16* r5 = r0 + N * 5;
                        const __fp16* r6 = r0 + N * 6;
                        const __fp16* r7 = r0 + N * 7;

                        float16x4_t _t0 = vld1_f16(r0);
                        float16x4_t _t1 = vld1_f16(r1);
                        float16x4_t _t2 = vld1_f16(r2);
                        float16x4_t _t3 = vld1_f16(r3);
                        float16x4_t _t4 = vld1_f16(r4);
                        float16x4_t _t5 = vld1_f16(r5);
                        float16x4_t _t6 = vld1_f16(r6);
                        float16x4_t _t7 = vld1_f16(r7);

                        transpose4x4_ph(_t0, _t1, _t2, _t3);
                        transpose4x4_ph(_t4, _t5, _t6, _t7);

                        _r0 = vcombine_f16(_t0, _t4);
                        if (tj * 2 + 1 < w)
                        {
                            _r1 = vcombine_f16(_t1, _t5);
                        }
                        if (tj * 2 + 2 < w)
                        {
                            _r2 = vcombine_f16(_t2, _t6);
                        }
                        if (tj * 2 + 3 < w)
                        {
                            _r3 = vcombine_f16(_t3, _t7);
                        }
                    }
                }

                float16x8_t _tmp0 = vsubq_f16(_r0, _r2);
                float16x8_t _tmp1 = vaddq_f16(_r1, _r2);
                float16x8_t _tmp2 = vsubq_f16(_r2, _r1);
                float16x8_t _tmp3 = vsubq_f16(_r3, _r1);

                vst1q_f16(tmp[0][m], _tmp0);
                vst1q_f16(tmp[1][m], _tmp1);
                vst1q_f16(tmp[2][m], _tmp2);
                vst1q_f16(tmp[3][m], _tmp3);

                r0 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 16 + jj * 8;
            __fp16* p1 = p0 + max_jj * 8;
            __fp16* p2 = p0 + max_jj * 8 * 2;
            __fp16* p3 = p0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                float16x8_t _r0 = vld1q_f16(tmp[m][0]);
                float16x8_t _r1 = vld1q_f16(tmp[m][1]);
                float16x8_t _r2 = vld1q_f16(tmp[m][2]);
                float16x8_t _r3 = vld1q_f16(tmp[m][3]);

                float16x8_t _tmp0 = vsubq_f16(_r0, _r2);
                float16x8_t _tmp1 = vaddq_f16(_r1, _r2);
                float16x8_t _tmp2 = vsubq_f16(_r2, _r1);
                float16x8_t _tmp3 = vsubq_f16(_r3, _r1);

                vst1q_f16(p0, _tmp0);
                vst1q_f16(p1, _tmp1);
                vst1q_f16(p2, _tmp2);
                vst1q_f16(p3, _tmp3);

                p0 += max_jj * 4 * 8;
                p1 += max_jj * 4 * 8;
                p2 += max_jj * 4 * 8;
                p3 += max_jj * 4 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[4][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel((k + kk) / elempack).row<const __fp16>(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                float16x4_t _r0 = vdup_n_f16(0.f);
                float16x4_t _r1 = vdup_n_f16(0.f);
                float16x4_t _r2 = vdup_n_f16(0.f);
                float16x4_t _r3 = vdup_n_f16(0.f);

                if (ti * 2 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = vld1_f16(r0);
                        if (tj * 2 + 1 < w) _r1 = vld1_f16(r0 + 4);
                        if (tj * 2 + 2 < w) _r2 = vld1_f16(r0 + 8);
                        if (tj * 2 + 3 < w) _r3 = vld1_f16(r0 + 12);
                    }
                    if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;
                        const __fp16* r2 = r0 + N * 2;
                        const __fp16* r3 = r0 + N * 3;

                        float16x4_t _t0 = vld1_f16(r0);
                        float16x4_t _t1 = vld1_f16(r1);
                        float16x4_t _t2 = vld1_f16(r2);
                        float16x4_t _t3 = vld1_f16(r3);

                        transpose4x4_ph(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 2 + 1 < w) _r1 = _t1;
                        if (tj * 2 + 2 < w) _r2 = _t2;
                        if (tj * 2 + 3 < w) _r3 = _t3;
                    }
                }

                float16x4_t _tmp0 = vsub_f16(_r0, _r2);
                float16x4_t _tmp1 = vadd_f16(_r1, _r2);
                float16x4_t _tmp2 = vsub_f16(_r2, _r1);
                float16x4_t _tmp3 = vsub_f16(_r3, _r1);

                vst1_f16(tmp[0][m], _tmp0);
                vst1_f16(tmp[1][m], _tmp1);
                vst1_f16(tmp[2][m], _tmp2);
                vst1_f16(tmp[3][m], _tmp3);

                r0 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 16 + jj * 4;
            __fp16* p1 = p0 + max_jj * 4;
            __fp16* p2 = p0 + max_jj * 4 * 2;
            __fp16* p3 = p0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                float16x4_t _r0 = vld1_f16(tmp[m][0]);
                float16x4_t _r1 = vld1_f16(tmp[m][1]);
                float16x4_t _r2 = vld1_f16(tmp[m][2]);
                float16x4_t _r3 = vld1_f16(tmp[m][3]);

                float16x4_t _tmp0 = vsub_f16(_r0, _r2);
                float16x4_t _tmp1 = vadd_f16(_r1, _r2);
                float16x4_t _tmp2 = vsub_f16(_r2, _r1);
                float16x4_t _tmp3 = vsub_f16(_r3, _r1);

                vst1_f16(p0, _tmp0);
                vst1_f16(p1, _tmp1);
                vst1_f16(p2, _tmp2);
                vst1_f16(p3, _tmp3);

                p0 += max_jj * 4 * 4;
                p1 += max_jj * 4 * 4;
                p2 += max_jj * 4 * 4;
                p3 += max_jj * 4 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        __fp16 tmp[4][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel(k + kk).row<const __fp16>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                __fp16 r00 = 0.f;
                __fp16 r01 = 0.f;
                __fp16 r10 = 0.f;
                __fp16 r11 = 0.f;
                __fp16 r20 = 0.f;
                __fp16 r21 = 0.f;
                __fp16 r30 = 0.f;
                __fp16 r31 = 0.f;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 2 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 2 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 2 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                    }
                }

                tmp[0][m][0] = r00 - r20;
                tmp[0][m][1] = r01 - r21;
                tmp[1][m][0] = r10 + r20;
                tmp[1][m][1] = r11 + r21;
                tmp[2][m][0] = r20 - r10;
                tmp[2][m][1] = r21 - r11;
                tmp[3][m][0] = r30 - r10;
                tmp[3][m][1] = r31 - r11;

                r0 += w;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 16 + jj * 2;
            __fp16* p1 = p0 + max_jj * 2;
            __fp16* p2 = p0 + max_jj * 2 * 2;
            __fp16* p3 = p0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];

                p0[0] = r00 - r20;
                p0[1] = r01 - r21;
                p1[0] = r10 + r20;
                p1[1] = r11 + r21;
                p2[0] = r20 - r10;
                p2[1] = r21 - r11;
                p3[0] = r30 - r10;
                p3[1] = r31 - r11;

                p0 += max_jj * 4 * 2;
                p1 += max_jj * 4 * 2;
                p2 += max_jj * 4 * 2;
                p3 += max_jj * 4 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        __fp16 tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0123 = bottom_blob.channel(k + kk).row<const __fp16>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                __fp16 r0 = 0.f;
                __fp16 r1 = 0.f;
                __fp16 r2 = 0.f;
                __fp16 r3 = 0.f;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 2 + 1 < w) r1 = r0123[1];
                        if (tj * 2 + 2 < w) r2 = r0123[2];
                        if (tj * 2 + 3 < w) r3 = r0123[3];
                    }
                }

                tmp[0][m] = r0 - r2;
                tmp[1][m] = r1 + r2;
                tmp[2][m] = r2 - r1;
                tmp[3][m] = r3 - r1;

                r0123 += w;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 16 + jj;
            __fp16* p1 = p0 + max_jj;
            __fp16* p2 = p0 + max_jj * 2;
            __fp16* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];

                p0[0] = r0 - r2;
                p1[0] = r1 + r2;
                p2[0] = r2 - r1;
                p3[0] = r3 - r1;

                p0 += max_jj * 4;
                p1 += max_jj * 4;
                p2 += max_jj * 4;
                p3 += max_jj * 4;
            }
        }
    }
}

static inline void conv3x3s1_winograd23_transform_output_tile_fp16sa(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 1) / 2;

    const __fp16* biasptr = bias;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
        float16x8_t _bias0 = biasptr ? vld1q_f16(biasptr + i + ii) : vdupq_n_f16(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[2][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 16 + jj * 8;
            const __fp16* r1 = r0 + max_jj * 8;
            const __fp16* r2 = r0 + max_jj * 8 * 2;
            const __fp16* r3 = r0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                float16x8_t _r0 = vld1q_f16(r0);
                float16x8_t _r1 = vld1q_f16(r1);
                float16x8_t _r2 = vld1q_f16(r2);
                float16x8_t _r3 = vld1q_f16(r3);

                float16x8_t _tmp0 = vaddq_f16(vaddq_f16(_r0, _r1), _r2);
                float16x8_t _tmp1 = vaddq_f16(vsubq_f16(_r1, _r2), _r3);

                vst1q_f16(tmp[0][m], _tmp0);
                vst1q_f16(tmp[1][m], _tmp1);

                r0 += max_jj * 4 * 8;
                r1 += max_jj * 4 * 8;
                r2 += max_jj * 4 * 8;
                r3 += max_jj * 4 * 8;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float16x8_t _r0 = vld1q_f16(tmp[m][0]);
                float16x8_t _r1 = vld1q_f16(tmp[m][1]);
                float16x8_t _r2 = vld1q_f16(tmp[m][2]);
                float16x8_t _r3 = vld1q_f16(tmp[m][3]);

                float16x8_t _tmp0 = vaddq_f16(_bias0, vaddq_f16(vaddq_f16(_r0, _r1), _r2));
                float16x8_t _tmp1 = vaddq_f16(_bias0, vaddq_f16(vsubq_f16(_r1, _r2), _r3));

                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _tmp0);
                    if (tj * 2 + 1 < outw)
                    {
                        vst1q_f16(outptr0 + 8, _tmp1);
                    }
                }
                if (out_elempack == 4)
                {
                    __fp16* outptr1 = outptr0 + N;

                    vst1_f16(outptr0, vget_low_f16(_tmp0));
                    vst1_f16(outptr1, vget_high_f16(_tmp0));
                    if (tj * 2 + 1 < outw)
                    {
                        vst1_f16(outptr0 + 4, vget_low_f16(_tmp1));
                        vst1_f16(outptr1 + 4, vget_high_f16(_tmp1));
                    }
                }
                if (out_elempack == 1)
                {
                    __fp16 tmp0[8];
                    __fp16 tmp1[8];
                    vst1q_f16(tmp0, _tmp0);
                    vst1q_f16(tmp1, _tmp1);

                    __fp16* outptr1 = outptr0 + N;
                    __fp16* outptr2 = outptr0 + N * 2;
                    __fp16* outptr3 = outptr0 + N * 3;
                    __fp16* outptr4 = outptr0 + N * 4;
                    __fp16* outptr5 = outptr0 + N * 5;
                    __fp16* outptr6 = outptr0 + N * 6;
                    __fp16* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];

                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float16x4_t _bias0 = biasptr ? vld1_f16(biasptr + i + ii) : vdup_n_f16(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[2][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 16 + jj * 4;
            const __fp16* r1 = r0 + max_jj * 4;
            const __fp16* r2 = r0 + max_jj * 4 * 2;
            const __fp16* r3 = r0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                float16x4_t _r0 = vld1_f16(r0);
                float16x4_t _r1 = vld1_f16(r1);
                float16x4_t _r2 = vld1_f16(r2);
                float16x4_t _r3 = vld1_f16(r3);

                float16x4_t _tmp0 = vadd_f16(vadd_f16(_r0, _r1), _r2);
                float16x4_t _tmp1 = vadd_f16(vsub_f16(_r1, _r2), _r3);

                vst1_f16(tmp[0][m], _tmp0);
                vst1_f16(tmp[1][m], _tmp1);

                r0 += max_jj * 4 * 4;
                r1 += max_jj * 4 * 4;
                r2 += max_jj * 4 * 4;
                r3 += max_jj * 4 * 4;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float16x4_t _r0 = vld1_f16(tmp[m][0]);
                float16x4_t _r1 = vld1_f16(tmp[m][1]);
                float16x4_t _r2 = vld1_f16(tmp[m][2]);
                float16x4_t _r3 = vld1_f16(tmp[m][3]);

                float16x4_t _tmp0 = vadd_f16(_bias0, vadd_f16(vadd_f16(_r0, _r1), _r2));
                float16x4_t _tmp1 = vadd_f16(_bias0, vadd_f16(vsub_f16(_r1, _r2), _r3));

                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, _tmp0);
                    if (tj * 2 + 1 < outw) vst1_f16(outptr0 + 4, _tmp1);
                }
                if (out_elempack == 1)
                {
                    __fp16 tmp0[4];
                    __fp16 tmp1[4];
                    vst1_f16(tmp0, _tmp0);
                    vst1_f16(tmp1, _tmp1);

                    __fp16* outptr1 = outptr0 + N;
                    __fp16* outptr2 = outptr0 + N * 2;
                    __fp16* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];

                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : 0.f;
        __fp16 bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        __fp16 tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 16 + jj * 2;
            const __fp16* r1 = r0 + max_jj * 2;
            const __fp16* r2 = r0 + max_jj * 2 * 2;
            const __fp16* r3 = r0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m][0] = r0[0] + r1[0] + r2[0];
                tmp[0][m][1] = r0[1] + r1[1] + r2[1];
                tmp[1][m][0] = r1[0] - r2[0] + r3[0];
                tmp[1][m][1] = r1[1] - r2[1] + r3[1];

                r0 += max_jj * 4 * 2;
                r1 += max_jj * 4 * 2;
                r2 += max_jj * 4 * 2;
                r3 += max_jj * 4 * 2;
            }

            __fp16* outptr0 = top_blob.channel(i + ii).row<__fp16>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];

                __fp16 tmp00 = bias0 + r00 + r10 + r20;
                __fp16 tmp01 = bias1 + r01 + r11 + r21;
                __fp16 tmp10 = bias0 + r10 - r20 + r30;
                __fp16 tmp11 = bias1 + r11 - r21 + r31;

                // if (out_elempack == 1)
                {
                    __fp16* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : 0.f;

        __fp16 tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 16 + jj;
            const __fp16* r1 = r0 + max_jj;
            const __fp16* r2 = r0 + max_jj * 2;
            const __fp16* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            __fp16* outptr0 = top_blob.channel(i + ii).row<__fp16>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];

                __fp16 tmp0 = bias0 + r0 + r1 + r2;
                __fp16 tmp1 = bias0 + r1 - r2 + r3;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 2 + 1 < outw) outptr0[1] = tmp1;
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd23_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 2n+2, winograd F(2,3)
    int w_tiles = (outw + 1) / 2;
    int h_tiles = (outh + 1) / 2;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 16;

    // NCNN_LOGE("conv3x3s1_winograd23_fp16sa %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 2u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd23_transform_input_tile_fp16sa(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 2u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd23_transform_input_tile_fp16sa(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 2u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile_fp16sa(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_kernel_tile_fp16sa(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    __fp16* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            const float sq2 = 1.41421356237f;
            // const float ktm[6][3] = {
            //     {1.0f, 0.0f, 0.0f},
            //     {-2.0f / 3, -sq2 / 3, -1.0f / 3},
            //     {-2.0f / 3, sq2 / 3, -1.0f / 3},
            //     {1.0f / 6, sq2 / 6, 1.0f / 3},
            //     {1.0f / 6, -sq2 / 6, 1.0f / 3},
            //     {0.0f, 0.0f, 1.0f}
            // };
            const float ktm0 = 2.0f / 3;
            const float ktm1 = sq2 / 3;
            const float ktm2 = 1.0f / 3;
            const float ktm3 = 1.0f / 6;
            const float ktm4 = sq2 / 6;

            float tmp[6][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = -r0 * ktm0 - r1 * ktm1 - r2 * ktm2;
                tmp[2][m] = -r0 * ktm0 + r1 * ktm1 - r2 * ktm2;
                tmp[3][m] = r0 * ktm3 + r1 * ktm4 + r2 * ktm2;
                tmp[4][m] = r0 * ktm3 - r1 * ktm4 + r2 * ktm2;
                tmp[5][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = -r0 * ktm0 - r1 * ktm1 - r2 * ktm2;
                float z2 = -r0 * ktm0 + r1 * ktm1 - r2 * ktm2;
                float z3 = r0 * ktm3 + r1 * ktm4 + r2 * ktm2;
                float z4 = r0 * ktm3 - r1 * ktm4 + r2 * ktm2;
                float z5 = r2;

                ptmp[0] = (__fp16)z0;
                ptmp[1] = (__fp16)z1;
                ptmp[2] = (__fp16)z2;
                ptmp[3] = (__fp16)z3;
                ptmp[4] = (__fp16)z4;
                ptmp[5] = (__fp16)z5;
                ptmp += 6;
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_kernel_fp16sa(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, (size_t)2u);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd43_transform_kernel_tile_fp16sa(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile_fp16(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile_fp16sa(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    const __fp16 sq2 = 1.41421356237;
    const __fp16 msq2_d2 = -1.41421356237 / 2;

    // const float itm[6][6] = {
    //     {1.0f,  0.0f,  -2.5f,  0.0f,  1.0f, 0.0f},
    //     {0.0f, -sq2,   -2.0f,  sq2/2, 1.0f, 0.0f},
    //     {0.0f,  sq2,   -2.0f, -sq2/2, 1.0f, 0.0f},
    //     {0.0f, -sq2/2, -0.5f,  sq2,   1.0f, 0.0f},
    //     {0.0f,  sq2/2, -0.5f, -sq2,   1.0f, 0.0f},
    //     {0.0f,  1.0f,   0.0f,  -2.5f, 0.0f, 1.0f}
    // };

    // 0 =  r00 + r04 - 2.5f * r02
    // 1 = -(sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 2 =  (sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 3 =  (sq2 * r03 - sq2_d2 * r01) + (r04 - 0.5f * r02)
    // 4 = -(sq2 * r03 - sq2_d2 * r01) + (r04 - 0.5f * r02)
    // 5 =  r01 + r05 - 2.5f * r03

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 1) / 4;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[6][6][8];

        const __fp16 coeffs[8] = {sq2, msq2_d2, -2.f, -0.5f, -2.5f, 0.f, 0.f, 0.f};
        float16x8_t _coeffs = vld1q_f16(coeffs);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel((k + kk) / elempack).row<const __fp16>(ti * 4) + (tj * 4) * elempack;

            for (int m = 0; m < 6; m++)
            {
                float16x8_t _r0 = vdupq_n_f16(0.f);
                float16x8_t _r1 = vdupq_n_f16(0.f);
                float16x8_t _r2 = vdupq_n_f16(0.f);
                float16x8_t _r3 = vdupq_n_f16(0.f);
                float16x8_t _r4 = vdupq_n_f16(0.f);
                float16x8_t _r5 = vdupq_n_f16(0.f);

                if (ti * 4 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = vld1q_f16(r0);
                        if (tj * 4 + 1 < w) _r1 = vld1q_f16(r0 + 8);
                        if (tj * 4 + 2 < w) _r2 = vld1q_f16(r0 + 16);
                        if (tj * 4 + 3 < w) _r3 = vld1q_f16(r0 + 24);
                        if (tj * 4 + 4 < w) _r4 = vld1q_f16(r0 + 32);
                        if (tj * 4 + 5 < w) _r5 = vld1q_f16(r0 + 40);
                    }
                    if (elempack == 4)
                    {
                        const __fp16* r1 = r0 + N;

                        _r0 = vcombine_f16(vld1_f16(r0), vld1_f16(r1));
                        if (tj * 4 + 1 < w)
                        {
                            _r1 = vcombine_f16(vld1_f16(r0 + 4), vld1_f16(r1 + 4));
                        }
                        if (tj * 4 + 2 < w)
                        {
                            _r2 = vcombine_f16(vld1_f16(r0 + 8), vld1_f16(r1 + 8));
                        }
                        if (tj * 4 + 3 < w)
                        {
                            _r3 = vcombine_f16(vld1_f16(r0 + 12), vld1_f16(r1 + 12));
                        }
                        if (tj * 4 + 4 < w)
                        {
                            _r4 = vcombine_f16(vld1_f16(r0 + 16), vld1_f16(r1 + 16));
                        }
                        if (tj * 4 + 5 < w)
                        {
                            _r5 = vcombine_f16(vld1_f16(r0 + 20), vld1_f16(r1 + 20));
                        }
                    }
                    if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;
                        const __fp16* r2 = r0 + N * 2;
                        const __fp16* r3 = r0 + N * 3;
                        const __fp16* r4 = r0 + N * 4;
                        const __fp16* r5 = r0 + N * 5;
                        const __fp16* r6 = r0 + N * 6;
                        const __fp16* r7 = r0 + N * 7;

                        float16x4_t _t0 = vld1_f16(r0);
                        float16x4_t _t1 = vld1_f16(r1);
                        float16x4_t _t2 = vld1_f16(r2);
                        float16x4_t _t3 = vld1_f16(r3);
                        float16x4_t _t4 = vld1_f16(r4);
                        float16x4_t _t5 = vld1_f16(r5);
                        float16x4_t _t6 = vld1_f16(r6);
                        float16x4_t _t7 = vld1_f16(r7);

                        transpose4x4_ph(_t0, _t1, _t2, _t3);
                        transpose4x4_ph(_t4, _t5, _t6, _t7);

                        _r0 = vcombine_f16(_t0, _t4);
                        if (tj * 4 + 1 < w)
                        {
                            _r1 = vcombine_f16(_t1, _t5);
                        }
                        if (tj * 4 + 2 < w)
                        {
                            _r2 = vcombine_f16(_t2, _t6);
                        }
                        if (tj * 4 + 3 < w)
                        {
                            _r3 = vcombine_f16(_t3, _t7);
                        }
                        if (tj * 4 + 4 < w)
                        {
                            __fp16 tmp[8] = {r0[4], r1[4], r2[4], r3[4], r4[4], r5[4], r6[4], r7[4]};
                            _r4 = vld1q_f16(tmp);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            __fp16 tmp[8] = {r0[5], r1[5], r2[5], r3[5], r4[5], r5[5], r6[5], r7[5]};
                            _r5 = vld1q_f16(tmp);
                        }
                    }
                }

                float16x8_t _tmp12a = vfmaq_laneq_f16(vmulq_laneq_f16(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float16x8_t _tmp12b = vfmaq_laneq_f16(_r4, _r2, _coeffs, 2);
                float16x8_t _tmp34a = vfmaq_laneq_f16(vmulq_laneq_f16(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float16x8_t _tmp34b = vfmaq_laneq_f16(_r4, _r2, _coeffs, 3);

                float16x8_t _tmp0 = vfmaq_laneq_f16(vaddq_f16(_r0, _r4), _r2, _coeffs, 4);
                float16x8_t _tmp1 = vsubq_f16(_tmp12b, _tmp12a);
                float16x8_t _tmp2 = vaddq_f16(_tmp12b, _tmp12a);
                float16x8_t _tmp3 = vaddq_f16(_tmp34b, _tmp34a);
                float16x8_t _tmp4 = vsubq_f16(_tmp34b, _tmp34a);
                float16x8_t _tmp5 = vfmaq_laneq_f16(vaddq_f16(_r1, _r5), _r3, _coeffs, 4);

                vst1q_f16(tmp[0][m], _tmp0);
                vst1q_f16(tmp[1][m], _tmp1);
                vst1q_f16(tmp[2][m], _tmp2);
                vst1q_f16(tmp[3][m], _tmp3);
                vst1q_f16(tmp[4][m], _tmp4);
                vst1q_f16(tmp[5][m], _tmp5);

                r0 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 36 + jj * 8;
            __fp16* p1 = p0 + max_jj * 8;
            __fp16* p2 = p0 + max_jj * 8 * 2;
            __fp16* p3 = p0 + max_jj * 8 * 3;
            __fp16* p4 = p0 + max_jj * 8 * 4;
            __fp16* p5 = p0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
                float16x8_t _r0 = vld1q_f16(tmp[m][0]);
                float16x8_t _r1 = vld1q_f16(tmp[m][1]);
                float16x8_t _r2 = vld1q_f16(tmp[m][2]);
                float16x8_t _r3 = vld1q_f16(tmp[m][3]);
                float16x8_t _r4 = vld1q_f16(tmp[m][4]);
                float16x8_t _r5 = vld1q_f16(tmp[m][5]);

                float16x8_t _tmp12a = vfmaq_laneq_f16(vmulq_laneq_f16(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float16x8_t _tmp12b = vfmaq_laneq_f16(_r4, _r2, _coeffs, 2);
                float16x8_t _tmp34a = vfmaq_laneq_f16(vmulq_laneq_f16(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float16x8_t _tmp34b = vfmaq_laneq_f16(_r4, _r2, _coeffs, 3);

                float16x8_t _tmp0 = vfmaq_laneq_f16(vaddq_f16(_r0, _r4), _r2, _coeffs, 4);
                float16x8_t _tmp1 = vsubq_f16(_tmp12b, _tmp12a);
                float16x8_t _tmp2 = vaddq_f16(_tmp12b, _tmp12a);
                float16x8_t _tmp3 = vaddq_f16(_tmp34b, _tmp34a);
                float16x8_t _tmp4 = vsubq_f16(_tmp34b, _tmp34a);
                float16x8_t _tmp5 = vfmaq_laneq_f16(vaddq_f16(_r1, _r5), _r3, _coeffs, 4);

                vst1q_f16(p0, _tmp0);
                vst1q_f16(p1, _tmp1);
                vst1q_f16(p2, _tmp2);
                vst1q_f16(p3, _tmp3);
                vst1q_f16(p4, _tmp4);
                vst1q_f16(p5, _tmp5);

                p0 += max_jj * 6 * 8;
                p1 += max_jj * 6 * 8;
                p2 += max_jj * 6 * 8;
                p3 += max_jj * 6 * 8;
                p4 += max_jj * 6 * 8;
                p5 += max_jj * 6 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[6][6][4];

        const __fp16 coeffs[8] = {sq2, msq2_d2, -2.f, -0.5f, -2.5f, 0.f, 0.f, 0.f};
        float16x8_t _coeffs = vld1q_f16(coeffs);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel((k + kk) / elempack).row<const __fp16>(ti * 4) + (tj * 4) * elempack;

            for (int m = 0; m < 6; m++)
            {
                float16x4_t _r0 = vdup_n_f16(0.f);
                float16x4_t _r1 = vdup_n_f16(0.f);
                float16x4_t _r2 = vdup_n_f16(0.f);
                float16x4_t _r3 = vdup_n_f16(0.f);
                float16x4_t _r4 = vdup_n_f16(0.f);
                float16x4_t _r5 = vdup_n_f16(0.f);

                if (ti * 4 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = vld1_f16(r0);
                        if (tj * 4 + 1 < w) _r1 = vld1_f16(r0 + 4);
                        if (tj * 4 + 2 < w) _r2 = vld1_f16(r0 + 8);
                        if (tj * 4 + 3 < w) _r3 = vld1_f16(r0 + 12);
                        if (tj * 4 + 4 < w) _r4 = vld1_f16(r0 + 16);
                        if (tj * 4 + 5 < w) _r5 = vld1_f16(r0 + 20);
                    }
                    if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;
                        const __fp16* r2 = r0 + N * 2;
                        const __fp16* r3 = r0 + N * 3;

                        float16x4_t _t0 = vld1_f16(r0);
                        float16x4_t _t1 = vld1_f16(r1);
                        float16x4_t _t2 = vld1_f16(r2);
                        float16x4_t _t3 = vld1_f16(r3);

                        transpose4x4_ph(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 4 + 1 < w) _r1 = _t1;
                        if (tj * 4 + 2 < w) _r2 = _t2;
                        if (tj * 4 + 3 < w) _r3 = _t3;
                        if (tj * 4 + 4 < w)
                        {
                            __fp16 tmp[4] = {r0[4], r1[4], r2[4], r3[4]};
                            _r4 = vld1_f16(tmp);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            __fp16 tmp[4] = {r0[5], r1[5], r2[5], r3[5]};
                            _r5 = vld1_f16(tmp);
                        }
                    }
                }

                float16x4_t _tmp12a = vfma_laneq_f16(vmul_laneq_f16(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float16x4_t _tmp12b = vfma_laneq_f16(_r4, _r2, _coeffs, 2);
                float16x4_t _tmp34a = vfma_laneq_f16(vmul_laneq_f16(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float16x4_t _tmp34b = vfma_laneq_f16(_r4, _r2, _coeffs, 3);

                float16x4_t _tmp0 = vfma_laneq_f16(vadd_f16(_r0, _r4), _r2, _coeffs, 4);
                float16x4_t _tmp1 = vsub_f16(_tmp12b, _tmp12a);
                float16x4_t _tmp2 = vadd_f16(_tmp12b, _tmp12a);
                float16x4_t _tmp3 = vadd_f16(_tmp34b, _tmp34a);
                float16x4_t _tmp4 = vsub_f16(_tmp34b, _tmp34a);
                float16x4_t _tmp5 = vfma_laneq_f16(vadd_f16(_r1, _r5), _r3, _coeffs, 4);

                vst1_f16(tmp[0][m], _tmp0);
                vst1_f16(tmp[1][m], _tmp1);
                vst1_f16(tmp[2][m], _tmp2);
                vst1_f16(tmp[3][m], _tmp3);
                vst1_f16(tmp[4][m], _tmp4);
                vst1_f16(tmp[5][m], _tmp5);

                r0 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 36 + jj * 4;
            __fp16* p1 = p0 + max_jj * 4;
            __fp16* p2 = p0 + max_jj * 4 * 2;
            __fp16* p3 = p0 + max_jj * 4 * 3;
            __fp16* p4 = p0 + max_jj * 4 * 4;
            __fp16* p5 = p0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
                float16x4_t _r0 = vld1_f16(tmp[m][0]);
                float16x4_t _r1 = vld1_f16(tmp[m][1]);
                float16x4_t _r2 = vld1_f16(tmp[m][2]);
                float16x4_t _r3 = vld1_f16(tmp[m][3]);
                float16x4_t _r4 = vld1_f16(tmp[m][4]);
                float16x4_t _r5 = vld1_f16(tmp[m][5]);

                float16x4_t _tmp12a = vfma_laneq_f16(vmul_laneq_f16(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float16x4_t _tmp12b = vfma_laneq_f16(_r4, _r2, _coeffs, 2);
                float16x4_t _tmp34a = vfma_laneq_f16(vmul_laneq_f16(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float16x4_t _tmp34b = vfma_laneq_f16(_r4, _r2, _coeffs, 3);

                float16x4_t _tmp0 = vfma_laneq_f16(vadd_f16(_r0, _r4), _r2, _coeffs, 4);
                float16x4_t _tmp1 = vsub_f16(_tmp12b, _tmp12a);
                float16x4_t _tmp2 = vadd_f16(_tmp12b, _tmp12a);
                float16x4_t _tmp3 = vadd_f16(_tmp34b, _tmp34a);
                float16x4_t _tmp4 = vsub_f16(_tmp34b, _tmp34a);
                float16x4_t _tmp5 = vfma_laneq_f16(vadd_f16(_r1, _r5), _r3, _coeffs, 4);

                vst1_f16(p0, _tmp0);
                vst1_f16(p1, _tmp1);
                vst1_f16(p2, _tmp2);
                vst1_f16(p3, _tmp3);
                vst1_f16(p4, _tmp4);
                vst1_f16(p5, _tmp5);

                p0 += max_jj * 6 * 4;
                p1 += max_jj * 6 * 4;
                p2 += max_jj * 6 * 4;
                p3 += max_jj * 6 * 4;
                p4 += max_jj * 6 * 4;
                p5 += max_jj * 6 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        __fp16 tmp[6][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel(k + kk).row<const __fp16>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                __fp16 r00 = 0.f;
                __fp16 r01 = 0.f;
                __fp16 r10 = 0.f;
                __fp16 r11 = 0.f;
                __fp16 r20 = 0.f;
                __fp16 r21 = 0.f;
                __fp16 r30 = 0.f;
                __fp16 r31 = 0.f;
                __fp16 r40 = 0.f;
                __fp16 r41 = 0.f;
                __fp16 r50 = 0.f;
                __fp16 r51 = 0.f;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 4 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 4 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 4 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                        if (tj * 4 + 4 < w)
                        {
                            r40 = r0[4];
                            r41 = r1[4];
                        }
                        if (tj * 4 + 5 < w)
                        {
                            r50 = r0[5];
                            r51 = r1[5];
                        }
                    }
                }

                __fp16 tmp12a0 = sq2 * r10 + msq2_d2 * r30;
                __fp16 tmp12a1 = sq2 * r11 + msq2_d2 * r31;
                __fp16 tmp12b0 = r40 - (__fp16)2.f * r20;
                __fp16 tmp12b1 = r41 - (__fp16)2.f * r21;
                __fp16 tmp34a0 = sq2 * r30 + msq2_d2 * r10;
                __fp16 tmp34a1 = sq2 * r31 + msq2_d2 * r11;
                __fp16 tmp34b0 = r40 - (__fp16)0.5f * r20;
                __fp16 tmp34b1 = r41 - (__fp16)0.5f * r21;

                tmp[0][m][0] = r00 + r40 - (__fp16)2.5f * r20;
                tmp[0][m][1] = r01 + r41 - (__fp16)2.5f * r21;
                tmp[1][m][0] = tmp12b0 - tmp12a0;
                tmp[1][m][1] = tmp12b1 - tmp12a1;
                tmp[2][m][0] = tmp12b0 + tmp12a0;
                tmp[2][m][1] = tmp12b1 + tmp12a1;
                tmp[3][m][0] = tmp34b0 + tmp34a0;
                tmp[3][m][1] = tmp34b1 + tmp34a1;
                tmp[4][m][0] = tmp34b0 - tmp34a0;
                tmp[4][m][1] = tmp34b1 - tmp34a1;
                tmp[5][m][0] = r10 + r50 - (__fp16)2.5f * r30;
                tmp[5][m][1] = r11 + r51 - (__fp16)2.5f * r31;

                r0 += w;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 36 + jj * 2;
            __fp16* p1 = p0 + max_jj * 2;
            __fp16* p2 = p0 + max_jj * 2 * 2;
            __fp16* p3 = p0 + max_jj * 2 * 3;
            __fp16* p4 = p0 + max_jj * 2 * 4;
            __fp16* p5 = p0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];
                __fp16 r40 = tmp[m][4][0];
                __fp16 r41 = tmp[m][4][1];
                __fp16 r50 = tmp[m][5][0];
                __fp16 r51 = tmp[m][5][1];

                __fp16 tmp12a0 = sq2 * r10 + msq2_d2 * r30;
                __fp16 tmp12a1 = sq2 * r11 + msq2_d2 * r31;
                __fp16 tmp12b0 = r40 - (__fp16)2.f * r20;
                __fp16 tmp12b1 = r41 - (__fp16)2.f * r21;
                __fp16 tmp34a0 = sq2 * r30 + msq2_d2 * r10;
                __fp16 tmp34a1 = sq2 * r31 + msq2_d2 * r11;
                __fp16 tmp34b0 = r40 - (__fp16)0.5f * r20;
                __fp16 tmp34b1 = r41 - (__fp16)0.5f * r21;

                p0[0] = r00 + r40 - (__fp16)2.5f * r20;
                p0[1] = r01 + r41 - (__fp16)2.5f * r21;
                p1[0] = tmp12b0 - tmp12a0;
                p1[1] = tmp12b1 - tmp12a1;
                p2[0] = tmp12b0 + tmp12a0;
                p2[1] = tmp12b1 + tmp12a1;
                p3[0] = tmp34b0 + tmp34a0;
                p3[1] = tmp34b1 + tmp34a1;
                p4[0] = tmp34b0 - tmp34a0;
                p4[1] = tmp34b1 - tmp34a1;
                p5[0] = r10 + r50 - (__fp16)2.5f * r30;
                p5[1] = r11 + r51 - (__fp16)2.5f * r31;

                p0 += max_jj * 6 * 2;
                p1 += max_jj * 6 * 2;
                p2 += max_jj * 6 * 2;
                p3 += max_jj * 6 * 2;
                p4 += max_jj * 6 * 2;
                p5 += max_jj * 6 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        __fp16 tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0123 = bottom_blob.channel(k + kk).row<const __fp16>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                __fp16 r0 = 0.f;
                __fp16 r1 = 0.f;
                __fp16 r2 = 0.f;
                __fp16 r3 = 0.f;
                __fp16 r4 = 0.f;
                __fp16 r5 = 0.f;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 4 + 1 < w) r1 = r0123[1];
                        if (tj * 4 + 2 < w) r2 = r0123[2];
                        if (tj * 4 + 3 < w) r3 = r0123[3];
                        if (tj * 4 + 4 < w) r4 = r0123[4];
                        if (tj * 4 + 5 < w) r5 = r0123[5];
                    }
                }

                __fp16 tmp12a = sq2 * r1 + msq2_d2 * r3;
                __fp16 tmp12b = r4 - (__fp16)2.f * r2;
                __fp16 tmp34a = sq2 * r3 + msq2_d2 * r1;
                __fp16 tmp34b = r4 - (__fp16)0.5f * r2;

                tmp[0][m] = r0 + r4 - (__fp16)2.5f * r2;
                tmp[1][m] = tmp12b - tmp12a;
                tmp[2][m] = tmp12b + tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r1 + r5 - (__fp16)2.5f * r3;

                r0123 += w;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 36 + jj;
            __fp16* p1 = p0 + max_jj;
            __fp16* p2 = p0 + max_jj * 2;
            __fp16* p3 = p0 + max_jj * 3;
            __fp16* p4 = p0 + max_jj * 4;
            __fp16* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];
                __fp16 r4 = tmp[m][4];
                __fp16 r5 = tmp[m][5];

                __fp16 tmp12a = sq2 * r1 + msq2_d2 * r3;
                __fp16 tmp12b = r4 - (__fp16)2.f * r2;
                __fp16 tmp34a = sq2 * r3 + msq2_d2 * r1;
                __fp16 tmp34b = r4 - (__fp16)0.5f * r2;

                p0[0] = r0 + r4 - (__fp16)2.5f * r2;
                p1[0] = tmp12b - tmp12a;
                p2[0] = tmp12b + tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r1 + r5 - (__fp16)2.5f * r3;

                p0 += max_jj * 6;
                p1 += max_jj * 6;
                p2 += max_jj * 6;
                p3 += max_jj * 6;
                p4 += max_jj * 6;
                p5 += max_jj * 6;
            }
        }
    }
}

static inline void conv3x3s1_winograd43_transform_output_tile_fp16sa(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    const __fp16 sq2 = 1.41421356237;
    const __fp16 sq2_m2 = 1.41421356237 * 2;
    const __fp16 sq2_d2 = 1.41421356237 / 2;
    const __fp16 sq2_d4 = 1.41421356237 / 4;

    // const float otm[4][6] = {
    //     {1.0f, 1.0f,   1.0f,  1.0f,  1.0f,   0.0f},
    //     {0.0f, sq2/2, -sq2/2, sq2,   -sq2,   0.0f},
    //     {0.0f, 0.5f,   0.5f,  2.0f,  2.0f,   0.0f},
    //     {0.0f, sq2/4, -sq2/4, sq2*2, -sq2*2, 1.0f}
    // };

    // 0 = r00 + (r01 + r02) + (r03 + r04)
    // 1 =       (r01 - r02) * sq2_d2 + (r03 - r04) * sq2
    // 2 =       (r01 + r02) * 0.5f + (r03 + r04) * 2
    // 3 = r05 + (r01 - r02) * sq2_d4 + (r03 - r04) * sq2_m2

    const __fp16 coeffs[8] = {sq2, sq2_d2, sq2_d4, sq2_m2, 0.5f, 2.f, 0.f, 0.f};
    float16x8_t _coeffs = vld1q_f16(coeffs);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 3) / 4;

    const __fp16* biasptr = bias;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
        float16x8_t _bias0 = biasptr ? vld1q_f16(biasptr + i + ii) : vdupq_n_f16(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[4][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 36 + jj * 8;
            const __fp16* r1 = r0 + max_jj * 8;
            const __fp16* r2 = r0 + max_jj * 8 * 2;
            const __fp16* r3 = r0 + max_jj * 8 * 3;
            const __fp16* r4 = r0 + max_jj * 8 * 4;
            const __fp16* r5 = r0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
                float16x8_t _r0 = vld1q_f16(r0);
                float16x8_t _r1 = vld1q_f16(r1);
                float16x8_t _r2 = vld1q_f16(r2);
                float16x8_t _r3 = vld1q_f16(r3);
                float16x8_t _r4 = vld1q_f16(r4);
                float16x8_t _r5 = vld1q_f16(r5);

                float16x8_t _tmp02a = vaddq_f16(_r1, _r2);
                float16x8_t _tmp02b = vaddq_f16(_r3, _r4);
                float16x8_t _tmp13a = vsubq_f16(_r1, _r2);
                float16x8_t _tmp13b = vsubq_f16(_r3, _r4);

                float16x8_t _tmp0 = vaddq_f16(vaddq_f16(_r0, _tmp02a), _tmp02b);
                float16x8_t _tmp1 = vfmaq_laneq_f16(vmulq_laneq_f16(_tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float16x8_t _tmp2 = vfmaq_laneq_f16(vmulq_laneq_f16(_tmp02a, _coeffs, 4), _tmp02b, _coeffs, 5);
                float16x8_t _tmp3 = vfmaq_laneq_f16(vfmaq_laneq_f16(_r5, _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);

                vst1q_f16(tmp[0][m], _tmp0);
                vst1q_f16(tmp[1][m], _tmp1);
                vst1q_f16(tmp[2][m], _tmp2);
                vst1q_f16(tmp[3][m], _tmp3);

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float16x8_t _r0 = vld1q_f16(tmp[m][0]);
                float16x8_t _r1 = vld1q_f16(tmp[m][1]);
                float16x8_t _r2 = vld1q_f16(tmp[m][2]);
                float16x8_t _r3 = vld1q_f16(tmp[m][3]);
                float16x8_t _r4 = vld1q_f16(tmp[m][4]);
                float16x8_t _r5 = vld1q_f16(tmp[m][5]);

                float16x8_t _tmp02a = vaddq_f16(_r1, _r2);
                float16x8_t _tmp02b = vaddq_f16(_r3, _r4);
                float16x8_t _tmp13a = vsubq_f16(_r1, _r2);
                float16x8_t _tmp13b = vsubq_f16(_r3, _r4);

                float16x8_t _tmp0 = vaddq_f16(vaddq_f16(_r0, _tmp02a), vaddq_f16(_tmp02b, _bias0));
                float16x8_t _tmp1 = vfmaq_laneq_f16(vfmaq_laneq_f16(_bias0, _tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float16x8_t _tmp2 = vfmaq_laneq_f16(vfmaq_laneq_f16(_bias0, _tmp02a, _coeffs, 4), _tmp02b, _coeffs, 5);
                float16x8_t _tmp3 = vfmaq_laneq_f16(vfmaq_laneq_f16(vaddq_f16(_r5, _bias0), _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);

                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) vst1q_f16(outptr0 + 8, _tmp1);
                    if (tj * 4 + 2 < outw) vst1q_f16(outptr0 + 16, _tmp2);
                    if (tj * 4 + 3 < outw) vst1q_f16(outptr0 + 24, _tmp3);
                }
                if (out_elempack == 4)
                {
                    __fp16* outptr1 = outptr0 + N;

                    vst1_f16(outptr0, vget_low_f16(_tmp0));
                    vst1_f16(outptr1, vget_high_f16(_tmp0));
                    if (tj * 4 + 1 < outw)
                    {
                        vst1_f16(outptr0 + 4, vget_low_f16(_tmp1));
                        vst1_f16(outptr1 + 4, vget_high_f16(_tmp1));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        vst1_f16(outptr0 + 8, vget_low_f16(_tmp2));
                        vst1_f16(outptr1 + 8, vget_high_f16(_tmp2));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        vst1_f16(outptr0 + 12, vget_low_f16(_tmp3));
                        vst1_f16(outptr1 + 12, vget_high_f16(_tmp3));
                    }
                }
                if (out_elempack == 1)
                {
                    __fp16 tmp0[8];
                    __fp16 tmp1[8];
                    __fp16 tmp2[8];
                    __fp16 tmp3[8];
                    vst1q_f16(tmp0, _tmp0);
                    vst1q_f16(tmp1, _tmp1);
                    vst1q_f16(tmp2, _tmp2);
                    vst1q_f16(tmp3, _tmp3);

                    __fp16* outptr1 = outptr0 + N;
                    __fp16* outptr2 = outptr0 + N * 2;
                    __fp16* outptr3 = outptr0 + N * 3;
                    __fp16* outptr4 = outptr0 + N * 4;
                    __fp16* outptr5 = outptr0 + N * 5;
                    __fp16* outptr6 = outptr0 + N * 6;
                    __fp16* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                        outptr4[2] = tmp2[4];
                        outptr5[2] = tmp2[5];
                        outptr6[2] = tmp2[6];
                        outptr7[2] = tmp2[7];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                        outptr4[3] = tmp3[4];
                        outptr5[3] = tmp3[5];
                        outptr6[3] = tmp3[6];
                        outptr7[3] = tmp3[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float16x4_t _bias0 = biasptr ? vld1_f16(biasptr + i + ii) : vdup_n_f16(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[4][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 36 + jj * 4;
            const __fp16* r1 = r0 + max_jj * 4;
            const __fp16* r2 = r0 + max_jj * 4 * 2;
            const __fp16* r3 = r0 + max_jj * 4 * 3;
            const __fp16* r4 = r0 + max_jj * 4 * 4;
            const __fp16* r5 = r0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
                float16x4_t _r0 = vld1_f16(r0);
                float16x4_t _r1 = vld1_f16(r1);
                float16x4_t _r2 = vld1_f16(r2);
                float16x4_t _r3 = vld1_f16(r3);
                float16x4_t _r4 = vld1_f16(r4);
                float16x4_t _r5 = vld1_f16(r5);

                float16x4_t _tmp02a = vadd_f16(_r1, _r2);
                float16x4_t _tmp02b = vadd_f16(_r3, _r4);
                float16x4_t _tmp13a = vsub_f16(_r1, _r2);
                float16x4_t _tmp13b = vsub_f16(_r3, _r4);

                float16x4_t _tmp0 = vadd_f16(vadd_f16(_r0, _tmp02a), _tmp02b);
                float16x4_t _tmp1 = vfma_laneq_f16(vmul_laneq_f16(_tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float16x4_t _tmp2 = vfma_laneq_f16(vmul_laneq_f16(_tmp02a, _coeffs, 4), _tmp02b, _coeffs, 5);
                float16x4_t _tmp3 = vfma_laneq_f16(vfma_laneq_f16(_r5, _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);

                vst1_f16(tmp[0][m], _tmp0);
                vst1_f16(tmp[1][m], _tmp1);
                vst1_f16(tmp[2][m], _tmp2);
                vst1_f16(tmp[3][m], _tmp3);

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float16x4_t _r0 = vld1_f16(tmp[m][0]);
                float16x4_t _r1 = vld1_f16(tmp[m][1]);
                float16x4_t _r2 = vld1_f16(tmp[m][2]);
                float16x4_t _r3 = vld1_f16(tmp[m][3]);
                float16x4_t _r4 = vld1_f16(tmp[m][4]);
                float16x4_t _r5 = vld1_f16(tmp[m][5]);

                float16x4_t _tmp02a = vadd_f16(_r1, _r2);
                float16x4_t _tmp02b = vadd_f16(_r3, _r4);
                float16x4_t _tmp13a = vsub_f16(_r1, _r2);
                float16x4_t _tmp13b = vsub_f16(_r3, _r4);

                float16x4_t _tmp0 = vadd_f16(vadd_f16(_r0, _tmp02a), vadd_f16(_tmp02b, _bias0));
                float16x4_t _tmp1 = vfma_laneq_f16(vfma_laneq_f16(_bias0, _tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float16x4_t _tmp2 = vfma_laneq_f16(vfma_laneq_f16(_bias0, _tmp02a, _coeffs, 4), _tmp02b, _coeffs, 5);
                float16x4_t _tmp3 = vfma_laneq_f16(vfma_laneq_f16(vadd_f16(_r5, _bias0), _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);

                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) vst1_f16(outptr0 + 4, _tmp1);
                    if (tj * 4 + 2 < outw) vst1_f16(outptr0 + 8, _tmp2);
                    if (tj * 4 + 3 < outw) vst1_f16(outptr0 + 12, _tmp3);
                }
                if (out_elempack == 1)
                {
                    __fp16 tmp0[4];
                    __fp16 tmp1[4];
                    __fp16 tmp2[4];
                    __fp16 tmp3[4];
                    vst1_f16(tmp0, _tmp0);
                    vst1_f16(tmp1, _tmp1);
                    vst1_f16(tmp2, _tmp2);
                    vst1_f16(tmp3, _tmp3);

                    __fp16* outptr1 = outptr0 + N;
                    __fp16* outptr2 = outptr0 + N * 2;
                    __fp16* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : 0.f;
        __fp16 bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        __fp16 tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 36 + jj * 2;
            const __fp16* r1 = r0 + max_jj * 2;
            const __fp16* r2 = r0 + max_jj * 2 * 2;
            const __fp16* r3 = r0 + max_jj * 2 * 3;
            const __fp16* r4 = r0 + max_jj * 2 * 4;
            const __fp16* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                __fp16 tmp02a0 = r1[0] + r2[0];
                __fp16 tmp02a1 = r1[1] + r2[1];
                __fp16 tmp02b0 = r3[0] + r4[0];
                __fp16 tmp02b1 = r3[1] + r4[1];
                __fp16 tmp13a0 = r1[0] - r2[0];
                __fp16 tmp13a1 = r1[1] - r2[1];
                __fp16 tmp13b0 = r3[0] - r4[0];
                __fp16 tmp13b1 = r3[1] - r4[1];

                tmp[0][m][0] = r0[0] + tmp02a0 + tmp02b0;
                tmp[0][m][1] = r0[1] + tmp02a1 + tmp02b1;
                tmp[1][m][0] = tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                tmp[1][m][1] = tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                tmp[2][m][0] = tmp02a0 * (__fp16)0.5f + tmp02b0 * (__fp16)2;
                tmp[2][m][1] = tmp02a1 * (__fp16)0.5f + tmp02b1 * (__fp16)2;
                tmp[3][m][0] = r5[0] + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                tmp[3][m][1] = r5[1] + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            __fp16* outptr0 = top_blob.channel(i + ii).row<__fp16>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];
                __fp16 r40 = tmp[m][4][0];
                __fp16 r41 = tmp[m][4][1];
                __fp16 r50 = tmp[m][5][0];
                __fp16 r51 = tmp[m][5][1];

                __fp16 tmp02a0 = r10 + r20;
                __fp16 tmp02a1 = r11 + r21;
                __fp16 tmp02b0 = r30 + r40;
                __fp16 tmp02b1 = r31 + r41;
                __fp16 tmp13a0 = r10 - r20;
                __fp16 tmp13a1 = r11 - r21;
                __fp16 tmp13b0 = r30 - r40;
                __fp16 tmp13b1 = r31 - r41;

                __fp16 tmp00 = bias0 + r00 + tmp02a0 + tmp02b0;
                __fp16 tmp01 = bias1 + r01 + tmp02a1 + tmp02b1;
                __fp16 tmp10 = bias0 + tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                __fp16 tmp11 = bias1 + tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                __fp16 tmp20 = bias0 + tmp02a0 * (__fp16)0.5f + tmp02b0 * (__fp16)2;
                __fp16 tmp21 = bias1 + tmp02a1 * (__fp16)0.5f + tmp02b1 * (__fp16)2;
                __fp16 tmp30 = bias0 + r50 + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                __fp16 tmp31 = bias1 + r51 + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;

                // if (out_elempack == 1)
                {
                    __fp16* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp20;
                        outptr1[2] = tmp21;
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp30;
                        outptr1[3] = tmp31;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : 0.f;

        __fp16 tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 36 + jj;
            const __fp16* r1 = r0 + max_jj;
            const __fp16* r2 = r0 + max_jj * 2;
            const __fp16* r3 = r0 + max_jj * 3;
            const __fp16* r4 = r0 + max_jj * 4;
            const __fp16* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                __fp16 tmp02a = r1[0] + r2[0];
                __fp16 tmp02b = r3[0] + r4[0];
                __fp16 tmp13a = r1[0] - r2[0];
                __fp16 tmp13b = r3[0] - r4[0];

                tmp[0][m] = r0[0] + tmp02a + tmp02b;
                tmp[1][m] = tmp13a * sq2_d2 + tmp13b * sq2;
                tmp[2][m] = tmp02a * (__fp16)0.5f + tmp02b * (__fp16)2;
                tmp[3][m] = r5[0] + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            __fp16* outptr0 = top_blob.channel(i + ii).row<__fp16>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];
                __fp16 r4 = tmp[m][4];
                __fp16 r5 = tmp[m][5];

                __fp16 tmp02a = r1 + r2;
                __fp16 tmp02b = r3 + r4;
                __fp16 tmp13a = r1 - r2;
                __fp16 tmp13b = r3 - r4;

                __fp16 tmp0 = bias0 + r0 + tmp02a + tmp02b;
                __fp16 tmp1 = bias0 + tmp13a * sq2_d2 + tmp13b * sq2;
                __fp16 tmp2 = bias0 + tmp02a * (__fp16)0.5f + tmp02b * (__fp16)2;
                __fp16 tmp3 = bias0 + r5 + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 4 + 1 < outw) outptr0[1] = tmp1;
                    if (tj * 4 + 2 < outw) outptr0[2] = tmp2;
                    if (tj * 4 + 3 < outw) outptr0[3] = tmp3;
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd43_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 4n+2, winograd F(4,3)
    int w_tiles = (outw + 3) / 4;
    int h_tiles = (outh + 3) / 4;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 36;

    // NCNN_LOGE("conv3x3s1_winograd43_fp16sa %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 2u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd43_transform_input_tile_fp16sa(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 2u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd43_transform_input_tile_fp16sa(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 2u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile_fp16sa(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_kernel_tile_fp16sa(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    __fp16* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            // const float ktm[8][3] = {
            //     {1.0f, 0.0f, 0.0f},
            //     {-2.0f / 9, -2.0f / 9, -2.0f / 9},
            //     {-2.0f / 9, 2.0f / 9, -2.0f / 9},
            //     {1.0f / 90, 1.0f / 45, 2.0f / 45},
            //     {1.0f / 90, -1.0f / 45, 2.0f / 45},
            //     {1.0f / 45, 1.0f / 90, 1.0f / 180},
            //     {1.0f / 45, -1.0f / 90, 1.0f / 180},
            //     {0.0f, 0.0f, 1.0f}
            // };
            const float ktm0 = 2.0f / 9;
            const float ktm1 = 1.0f / 45;
            const float ktm2 = 2.0f / 45;
            const float ktm3 = 1.0f / 90;
            const float ktm4 = 1.0f / 180;

            float tmp[8][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = -r0 * ktm0 - r1 * ktm0 - r2 * ktm0;
                tmp[2][m] = -r0 * ktm0 + r1 * ktm0 - r2 * ktm0;
                tmp[3][m] = r0 * ktm3 + r1 * ktm1 + r2 * ktm2;
                tmp[4][m] = r0 * ktm3 - r1 * ktm1 + r2 * ktm2;
                tmp[5][m] = r0 * ktm1 + r1 * ktm3 + r2 * ktm4;
                tmp[6][m] = r0 * ktm1 - r1 * ktm3 + r2 * ktm4;
                tmp[7][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 8; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = -r0 * ktm0 - r1 * ktm0 - r2 * ktm0;
                float z2 = -r0 * ktm0 + r1 * ktm0 - r2 * ktm0;
                float z3 = r0 * ktm3 + r1 * ktm1 + r2 * ktm2;
                float z4 = r0 * ktm3 - r1 * ktm1 + r2 * ktm2;
                float z5 = r0 * ktm1 + r1 * ktm3 + r2 * ktm4;
                float z6 = r0 * ktm1 - r1 * ktm3 + r2 * ktm4;
                float z7 = r2;

                ptmp[0] = (__fp16)z0;
                ptmp[1] = (__fp16)z1;
                ptmp[2] = (__fp16)z2;
                ptmp[3] = (__fp16)z3;
                ptmp[4] = (__fp16)z4;
                ptmp[5] = (__fp16)z5;
                ptmp[6] = (__fp16)z6;
                ptmp[7] = (__fp16)z7;
                ptmp += 8;
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_kernel_fp16sa(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 64;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, (size_t)2u);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd63_transform_kernel_tile_fp16sa(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile_fp16(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_input_tile_fp16sa(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[8][8] = {
    //     {1.0f, 0.0f,-5.25f, 0.00f, 5.25f, 0.00f,-1.0f, 0.0f},
    //     {0.0f, 1.0f, 1.00f,-4.25f,-4.25f, 1.00f, 1.0f, 0.0f},
    //     {0.0f,-1.0f, 1.00f, 4.25f,-4.25f,-1.00f, 1.0f, 0.0f},
    //     {0.0f, 0.5f, 0.25f,-2.50f,-1.25f, 2.00f, 1.0f, 0.0f},
    //     {0.0f,-0.5f, 0.25f, 2.50f,-1.25f,-2.00f, 1.0f, 0.0f},
    //     {0.0f, 2.0f, 4.00f,-2.50f,-5.00f, 0.50f, 1.0f, 0.0f},
    //     {0.0f,-2.0f, 4.00f, 2.50f,-5.00f,-0.50f, 1.0f, 0.0f},
    //     {0.0f,-1.0f, 0.00f, 5.25f, 0.00f,-5.25f, 0.0f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 3) / 6;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[8][8][8];

        const __fp16 coeffs[8] = {5.25f, -4.25f, -1.25f, 0.25f, -2.5f, 0.5f, 2.f, 4.f};
        float16x8_t _coeffs = vld1q_f16(coeffs);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel((k + kk) / elempack).row<const __fp16>(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                float16x8_t _r0 = vdupq_n_f16(0.f);
                float16x8_t _r1 = vdupq_n_f16(0.f);
                float16x8_t _r2 = vdupq_n_f16(0.f);
                float16x8_t _r3 = vdupq_n_f16(0.f);
                float16x8_t _r4 = vdupq_n_f16(0.f);
                float16x8_t _r5 = vdupq_n_f16(0.f);
                float16x8_t _r6 = vdupq_n_f16(0.f);
                float16x8_t _r7 = vdupq_n_f16(0.f);

                if (ti * 6 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = vld1q_f16(r0);
                        if (tj * 6 + 1 < w) _r1 = vld1q_f16(r0 + 8);
                        if (tj * 6 + 2 < w) _r2 = vld1q_f16(r0 + 16);
                        if (tj * 6 + 3 < w) _r3 = vld1q_f16(r0 + 24);
                        if (tj * 6 + 4 < w) _r4 = vld1q_f16(r0 + 32);
                        if (tj * 6 + 5 < w) _r5 = vld1q_f16(r0 + 40);
                        if (tj * 6 + 6 < w) _r6 = vld1q_f16(r0 + 48);
                        if (tj * 6 + 7 < w) _r7 = vld1q_f16(r0 + 56);
                    }
                    if (elempack == 4)
                    {
                        const __fp16* r1 = r0 + N;

                        _r0 = vcombine_f16(vld1_f16(r0), vld1_f16(r1));
                        if (tj * 6 + 1 < w)
                        {
                            _r1 = vcombine_f16(vld1_f16(r0 + 4), vld1_f16(r1 + 4));
                        }
                        if (tj * 6 + 2 < w)
                        {
                            _r2 = vcombine_f16(vld1_f16(r0 + 8), vld1_f16(r1 + 8));
                        }
                        if (tj * 6 + 3 < w)
                        {
                            _r3 = vcombine_f16(vld1_f16(r0 + 12), vld1_f16(r1 + 12));
                        }
                        if (tj * 6 + 4 < w)
                        {
                            _r4 = vcombine_f16(vld1_f16(r0 + 16), vld1_f16(r1 + 16));
                        }
                        if (tj * 6 + 5 < w)
                        {
                            _r5 = vcombine_f16(vld1_f16(r0 + 20), vld1_f16(r1 + 20));
                        }
                        if (tj * 6 + 6 < w)
                        {
                            _r6 = vcombine_f16(vld1_f16(r0 + 24), vld1_f16(r1 + 24));
                        }
                        if (tj * 6 + 7 < w)
                        {
                            _r7 = vcombine_f16(vld1_f16(r0 + 28), vld1_f16(r1 + 28));
                        }
                    }
                    if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;
                        const __fp16* r2 = r0 + N * 2;
                        const __fp16* r3 = r0 + N * 3;
                        const __fp16* r4 = r0 + N * 4;
                        const __fp16* r5 = r0 + N * 5;
                        const __fp16* r6 = r0 + N * 6;
                        const __fp16* r7 = r0 + N * 7;

                        float16x4_t _t0 = vld1_f16(r0);
                        float16x4_t _t1 = vld1_f16(r1);
                        float16x4_t _t2 = vld1_f16(r2);
                        float16x4_t _t3 = vld1_f16(r3);
                        float16x4_t _t4 = vld1_f16(r4);
                        float16x4_t _t5 = vld1_f16(r5);
                        float16x4_t _t6 = vld1_f16(r6);
                        float16x4_t _t7 = vld1_f16(r7);

                        transpose4x4_ph(_t0, _t1, _t2, _t3);
                        transpose4x4_ph(_t4, _t5, _t6, _t7);

                        _r0 = vcombine_f16(_t0, _t4);
                        if (tj * 6 + 1 < w)
                        {
                            _r1 = vcombine_f16(_t1, _t5);
                        }
                        if (tj * 6 + 2 < w)
                        {
                            _r2 = vcombine_f16(_t2, _t6);
                        }
                        if (tj * 6 + 3 < w)
                        {
                            _r3 = vcombine_f16(_t3, _t7);
                        }
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = vld1_f16(r0 + 4);
                            _t1 = vld1_f16(r1 + 4);
                            _t2 = vld1_f16(r2 + 4);
                            _t3 = vld1_f16(r3 + 4);
                            _t4 = vld1_f16(r4 + 4);
                            _t5 = vld1_f16(r5 + 4);
                            _t6 = vld1_f16(r6 + 4);
                            _t7 = vld1_f16(r7 + 4);

                            transpose4x4_ph(_t0, _t1, _t2, _t3);
                            transpose4x4_ph(_t4, _t5, _t6, _t7);

                            _r4 = vcombine_f16(_t0, _t4);
                            if (tj * 6 + 5 < w)
                            {
                                _r5 = vcombine_f16(_t1, _t5);
                            }
                            if (tj * 6 + 6 < w)
                            {
                                _r6 = vcombine_f16(_t2, _t6);
                            }
                            if (tj * 6 + 7 < w)
                            {
                                _r7 = vcombine_f16(_t3, _t7);
                            }
                        }
                    }
                }

                float16x8_t _tmp12a = vfmaq_laneq_f16(vaddq_f16(_r2, _r6), _r4, _coeffs, 1);
                float16x8_t _tmp12b = vfmaq_laneq_f16(vaddq_f16(_r1, _r5), _r3, _coeffs, 1);
                float16x8_t _tmp34a = vfmaq_laneq_f16(vfmaq_laneq_f16(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float16x8_t _tmp34b = vfmaq_laneq_f16(vfmaq_laneq_f16(vmulq_laneq_f16(_r1, _coeffs, 5), _r3, _coeffs, 4), _r5, _coeffs, 6);
                float16x8_t _tmp56a = vfmaq_laneq_f16(_r6, vfmaq_laneq_f16(_r2, _r4, _coeffs, 2), _coeffs, 7);
                float16x8_t _tmp56b = vfmaq_laneq_f16(vfmaq_laneq_f16(vmulq_laneq_f16(_r1, _coeffs, 6), _r3, _coeffs, 4), _r5, _coeffs, 5);

                float16x8_t _tmp0 = vfmaq_laneq_f16(vsubq_f16(_r0, _r6), vsubq_f16(_r4, _r2), _coeffs, 0);
                float16x8_t _tmp1 = vaddq_f16(_tmp12a, _tmp12b);
                float16x8_t _tmp2 = vsubq_f16(_tmp12a, _tmp12b);
                float16x8_t _tmp3 = vaddq_f16(_tmp34a, _tmp34b);
                float16x8_t _tmp4 = vsubq_f16(_tmp34a, _tmp34b);
                float16x8_t _tmp5 = vaddq_f16(_tmp56a, _tmp56b);
                float16x8_t _tmp6 = vsubq_f16(_tmp56a, _tmp56b);
                float16x8_t _tmp7 = vfmaq_laneq_f16(vsubq_f16(_r7, _r1), vsubq_f16(_r3, _r5), _coeffs, 0);

                vst1q_f16(tmp[0][m], _tmp0);
                vst1q_f16(tmp[1][m], _tmp1);
                vst1q_f16(tmp[2][m], _tmp2);
                vst1q_f16(tmp[3][m], _tmp3);
                vst1q_f16(tmp[4][m], _tmp4);
                vst1q_f16(tmp[5][m], _tmp5);
                vst1q_f16(tmp[6][m], _tmp6);
                vst1q_f16(tmp[7][m], _tmp7);

                r0 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 64 + jj * 8;
            __fp16* p1 = p0 + max_jj * 8;
            __fp16* p2 = p0 + max_jj * 8 * 2;
            __fp16* p3 = p0 + max_jj * 8 * 3;
            __fp16* p4 = p0 + max_jj * 8 * 4;
            __fp16* p5 = p0 + max_jj * 8 * 5;
            __fp16* p6 = p0 + max_jj * 8 * 6;
            __fp16* p7 = p0 + max_jj * 8 * 7;

            for (int m = 0; m < 8; m++)
            {
                float16x8_t _r0 = vld1q_f16(tmp[m][0]);
                float16x8_t _r1 = vld1q_f16(tmp[m][1]);
                float16x8_t _r2 = vld1q_f16(tmp[m][2]);
                float16x8_t _r3 = vld1q_f16(tmp[m][3]);
                float16x8_t _r4 = vld1q_f16(tmp[m][4]);
                float16x8_t _r5 = vld1q_f16(tmp[m][5]);
                float16x8_t _r6 = vld1q_f16(tmp[m][6]);
                float16x8_t _r7 = vld1q_f16(tmp[m][7]);

                float16x8_t _tmp12a = vfmaq_laneq_f16(vaddq_f16(_r2, _r6), _r4, _coeffs, 1);
                float16x8_t _tmp12b = vfmaq_laneq_f16(vaddq_f16(_r1, _r5), _r3, _coeffs, 1);
                float16x8_t _tmp34a = vfmaq_laneq_f16(vfmaq_laneq_f16(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float16x8_t _tmp34b = vfmaq_laneq_f16(vfmaq_laneq_f16(vmulq_laneq_f16(_r1, _coeffs, 5), _r3, _coeffs, 4), _r5, _coeffs, 6);
                float16x8_t _tmp56a = vfmaq_laneq_f16(_r6, vfmaq_laneq_f16(_r2, _r4, _coeffs, 2), _coeffs, 7);
                float16x8_t _tmp56b = vfmaq_laneq_f16(vfmaq_laneq_f16(vmulq_laneq_f16(_r1, _coeffs, 6), _r3, _coeffs, 4), _r5, _coeffs, 5);

                float16x8_t _tmp0 = vfmaq_laneq_f16(vsubq_f16(_r0, _r6), vsubq_f16(_r4, _r2), _coeffs, 0);
                float16x8_t _tmp1 = vaddq_f16(_tmp12a, _tmp12b);
                float16x8_t _tmp2 = vsubq_f16(_tmp12a, _tmp12b);
                float16x8_t _tmp3 = vaddq_f16(_tmp34a, _tmp34b);
                float16x8_t _tmp4 = vsubq_f16(_tmp34a, _tmp34b);
                float16x8_t _tmp5 = vaddq_f16(_tmp56a, _tmp56b);
                float16x8_t _tmp6 = vsubq_f16(_tmp56a, _tmp56b);
                float16x8_t _tmp7 = vfmaq_laneq_f16(vsubq_f16(_r7, _r1), vsubq_f16(_r3, _r5), _coeffs, 0);

                vst1q_f16(p0, _tmp0);
                vst1q_f16(p1, _tmp1);
                vst1q_f16(p2, _tmp2);
                vst1q_f16(p3, _tmp3);
                vst1q_f16(p4, _tmp4);
                vst1q_f16(p5, _tmp5);
                vst1q_f16(p6, _tmp6);
                vst1q_f16(p7, _tmp7);

                p0 += max_jj * 8 * 8;
                p1 += max_jj * 8 * 8;
                p2 += max_jj * 8 * 8;
                p3 += max_jj * 8 * 8;
                p4 += max_jj * 8 * 8;
                p5 += max_jj * 8 * 8;
                p6 += max_jj * 8 * 8;
                p7 += max_jj * 8 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[8][8][4];

        const __fp16 coeffs[8] = {5.25f, -4.25f, -1.25f, 0.25f, -2.5f, 0.5f, 2.f, 4.f};
        float16x8_t _coeffs = vld1q_f16(coeffs);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel((k + kk) / elempack).row<const __fp16>(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                float16x4_t _r0 = vdup_n_f16(0.f);
                float16x4_t _r1 = vdup_n_f16(0.f);
                float16x4_t _r2 = vdup_n_f16(0.f);
                float16x4_t _r3 = vdup_n_f16(0.f);
                float16x4_t _r4 = vdup_n_f16(0.f);
                float16x4_t _r5 = vdup_n_f16(0.f);
                float16x4_t _r6 = vdup_n_f16(0.f);
                float16x4_t _r7 = vdup_n_f16(0.f);

                if (ti * 6 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = vld1_f16(r0);
                        if (tj * 6 + 1 < w) _r1 = vld1_f16(r0 + 4);
                        if (tj * 6 + 2 < w) _r2 = vld1_f16(r0 + 8);
                        if (tj * 6 + 3 < w) _r3 = vld1_f16(r0 + 12);
                        if (tj * 6 + 4 < w) _r4 = vld1_f16(r0 + 16);
                        if (tj * 6 + 5 < w) _r5 = vld1_f16(r0 + 20);
                        if (tj * 6 + 6 < w) _r6 = vld1_f16(r0 + 24);
                        if (tj * 6 + 7 < w) _r7 = vld1_f16(r0 + 28);
                    }
                    if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;
                        const __fp16* r2 = r0 + N * 2;
                        const __fp16* r3 = r0 + N * 3;

                        float16x4_t _t0 = vld1_f16(r0);
                        float16x4_t _t1 = vld1_f16(r1);
                        float16x4_t _t2 = vld1_f16(r2);
                        float16x4_t _t3 = vld1_f16(r3);

                        transpose4x4_ph(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 6 + 1 < w) _r1 = _t1;
                        if (tj * 6 + 2 < w) _r2 = _t2;
                        if (tj * 6 + 3 < w) _r3 = _t3;
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = vld1_f16(r0 + 4);
                            _t1 = vld1_f16(r1 + 4);
                            _t2 = vld1_f16(r2 + 4);
                            _t3 = vld1_f16(r3 + 4);

                            transpose4x4_ph(_t0, _t1, _t2, _t3);

                            _r4 = _t0;
                            if (tj * 6 + 5 < w) _r5 = _t1;
                            if (tj * 6 + 6 < w) _r6 = _t2;
                            if (tj * 6 + 7 < w) _r7 = _t3;
                        }
                    }
                }

                float16x4_t _tmp12a = vfma_laneq_f16(vadd_f16(_r2, _r6), _r4, _coeffs, 1);
                float16x4_t _tmp12b = vfma_laneq_f16(vadd_f16(_r1, _r5), _r3, _coeffs, 1);
                float16x4_t _tmp34a = vfma_laneq_f16(vfma_laneq_f16(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float16x4_t _tmp34b = vfma_laneq_f16(vfma_laneq_f16(vmul_laneq_f16(_r1, _coeffs, 5), _r3, _coeffs, 4), _r5, _coeffs, 6);
                float16x4_t _tmp56a = vfma_laneq_f16(_r6, vfma_laneq_f16(_r2, _r4, _coeffs, 2), _coeffs, 7);
                float16x4_t _tmp56b = vfma_laneq_f16(vfma_laneq_f16(vmul_laneq_f16(_r1, _coeffs, 6), _r3, _coeffs, 4), _r5, _coeffs, 5);

                float16x4_t _tmp0 = vfma_laneq_f16(vsub_f16(_r0, _r6), vsub_f16(_r4, _r2), _coeffs, 0);
                float16x4_t _tmp1 = vadd_f16(_tmp12a, _tmp12b);
                float16x4_t _tmp2 = vsub_f16(_tmp12a, _tmp12b);
                float16x4_t _tmp3 = vadd_f16(_tmp34a, _tmp34b);
                float16x4_t _tmp4 = vsub_f16(_tmp34a, _tmp34b);
                float16x4_t _tmp5 = vadd_f16(_tmp56a, _tmp56b);
                float16x4_t _tmp6 = vsub_f16(_tmp56a, _tmp56b);
                float16x4_t _tmp7 = vfma_laneq_f16(vsub_f16(_r7, _r1), vsub_f16(_r3, _r5), _coeffs, 0);

                vst1_f16(tmp[0][m], _tmp0);
                vst1_f16(tmp[1][m], _tmp1);
                vst1_f16(tmp[2][m], _tmp2);
                vst1_f16(tmp[3][m], _tmp3);
                vst1_f16(tmp[4][m], _tmp4);
                vst1_f16(tmp[5][m], _tmp5);
                vst1_f16(tmp[6][m], _tmp6);
                vst1_f16(tmp[7][m], _tmp7);

                r0 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 64 + jj * 4;
            __fp16* p1 = p0 + max_jj * 4;
            __fp16* p2 = p0 + max_jj * 4 * 2;
            __fp16* p3 = p0 + max_jj * 4 * 3;
            __fp16* p4 = p0 + max_jj * 4 * 4;
            __fp16* p5 = p0 + max_jj * 4 * 5;
            __fp16* p6 = p0 + max_jj * 4 * 6;
            __fp16* p7 = p0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
                float16x4_t _r0 = vld1_f16(tmp[m][0]);
                float16x4_t _r1 = vld1_f16(tmp[m][1]);
                float16x4_t _r2 = vld1_f16(tmp[m][2]);
                float16x4_t _r3 = vld1_f16(tmp[m][3]);
                float16x4_t _r4 = vld1_f16(tmp[m][4]);
                float16x4_t _r5 = vld1_f16(tmp[m][5]);
                float16x4_t _r6 = vld1_f16(tmp[m][6]);
                float16x4_t _r7 = vld1_f16(tmp[m][7]);

                float16x4_t _tmp12a = vfma_laneq_f16(vadd_f16(_r2, _r6), _r4, _coeffs, 1);
                float16x4_t _tmp12b = vfma_laneq_f16(vadd_f16(_r1, _r5), _r3, _coeffs, 1);
                float16x4_t _tmp34a = vfma_laneq_f16(vfma_laneq_f16(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float16x4_t _tmp34b = vfma_laneq_f16(vfma_laneq_f16(vmul_laneq_f16(_r1, _coeffs, 5), _r3, _coeffs, 4), _r5, _coeffs, 6);
                float16x4_t _tmp56a = vfma_laneq_f16(_r6, vfma_laneq_f16(_r2, _r4, _coeffs, 2), _coeffs, 7);
                float16x4_t _tmp56b = vfma_laneq_f16(vfma_laneq_f16(vmul_laneq_f16(_r1, _coeffs, 6), _r3, _coeffs, 4), _r5, _coeffs, 5);

                float16x4_t _tmp0 = vfma_laneq_f16(vsub_f16(_r0, _r6), vsub_f16(_r4, _r2), _coeffs, 0);
                float16x4_t _tmp1 = vadd_f16(_tmp12a, _tmp12b);
                float16x4_t _tmp2 = vsub_f16(_tmp12a, _tmp12b);
                float16x4_t _tmp3 = vadd_f16(_tmp34a, _tmp34b);
                float16x4_t _tmp4 = vsub_f16(_tmp34a, _tmp34b);
                float16x4_t _tmp5 = vadd_f16(_tmp56a, _tmp56b);
                float16x4_t _tmp6 = vsub_f16(_tmp56a, _tmp56b);
                float16x4_t _tmp7 = vfma_laneq_f16(vsub_f16(_r7, _r1), vsub_f16(_r3, _r5), _coeffs, 0);

                vst1_f16(p0, _tmp0);
                vst1_f16(p1, _tmp1);
                vst1_f16(p2, _tmp2);
                vst1_f16(p3, _tmp3);
                vst1_f16(p4, _tmp4);
                vst1_f16(p5, _tmp5);
                vst1_f16(p6, _tmp6);
                vst1_f16(p7, _tmp7);

                p0 += max_jj * 8 * 4;
                p1 += max_jj * 8 * 4;
                p2 += max_jj * 8 * 4;
                p3 += max_jj * 8 * 4;
                p4 += max_jj * 8 * 4;
                p5 += max_jj * 8 * 4;
                p6 += max_jj * 8 * 4;
                p7 += max_jj * 8 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        __fp16 tmp[8][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = bottom_blob.channel(k + kk).row<const __fp16>(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
                __fp16 r00 = 0.f;
                __fp16 r01 = 0.f;
                __fp16 r10 = 0.f;
                __fp16 r11 = 0.f;
                __fp16 r20 = 0.f;
                __fp16 r21 = 0.f;
                __fp16 r30 = 0.f;
                __fp16 r31 = 0.f;
                __fp16 r40 = 0.f;
                __fp16 r41 = 0.f;
                __fp16 r50 = 0.f;
                __fp16 r51 = 0.f;
                __fp16 r60 = 0.f;
                __fp16 r61 = 0.f;
                __fp16 r70 = 0.f;
                __fp16 r71 = 0.f;

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const __fp16* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 6 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 6 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 6 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                        if (tj * 6 + 4 < w)
                        {
                            r40 = r0[4];
                            r41 = r1[4];
                        }
                        if (tj * 6 + 5 < w)
                        {
                            r50 = r0[5];
                            r51 = r1[5];
                        }
                        if (tj * 6 + 6 < w)
                        {
                            r60 = r0[6];
                            r61 = r1[6];
                        }
                        if (tj * 6 + 7 < w)
                        {
                            r70 = r0[7];
                            r71 = r1[7];
                        }
                    }
                }

                __fp16 tmp12a0 = r20 + r60 - r40 * (__fp16)4.25f;
                __fp16 tmp12a1 = r21 + r61 - r41 * (__fp16)4.25f;
                __fp16 tmp12b0 = r10 + r50 - r30 * (__fp16)4.25f;
                __fp16 tmp12b1 = r11 + r51 - r31 * (__fp16)4.25f;
                __fp16 tmp34a0 = r60 + r20 * (__fp16)0.25f - r40 * (__fp16)1.25f;
                __fp16 tmp34a1 = r61 + r21 * (__fp16)0.25f - r41 * (__fp16)1.25f;
                __fp16 tmp34b0 = r10 * (__fp16)0.5f - r30 * (__fp16)2.5f + r50 * (__fp16)2.f;
                __fp16 tmp34b1 = r11 * (__fp16)0.5f - r31 * (__fp16)2.5f + r51 * (__fp16)2.f;
                __fp16 tmp56a0 = r20 * (__fp16)4.f - r40 * (__fp16)5.f + r60;
                __fp16 tmp56a1 = r21 * (__fp16)4.f - r41 * (__fp16)5.f + r61;
                __fp16 tmp56b0 = r10 * (__fp16)2.f - r30 * (__fp16)2.5f + r50 * (__fp16)0.5f;
                __fp16 tmp56b1 = r11 * (__fp16)2.f - r31 * (__fp16)2.5f + r51 * (__fp16)0.5f;

                tmp[0][m][0] = r00 - r60 + (r40 - r20) * (__fp16)5.25f;
                tmp[0][m][1] = r01 - r61 + (r41 - r21) * (__fp16)5.25f;
                tmp[1][m][0] = tmp12a0 + tmp12b0;
                tmp[1][m][1] = tmp12a1 + tmp12b1;
                tmp[2][m][0] = tmp12a0 - tmp12b0;
                tmp[2][m][1] = tmp12a1 - tmp12b1;
                tmp[3][m][0] = tmp34a0 + tmp34b0;
                tmp[3][m][1] = tmp34a1 + tmp34b1;
                tmp[4][m][0] = tmp34a0 - tmp34b0;
                tmp[4][m][1] = tmp34a1 - tmp34b1;
                tmp[5][m][0] = tmp56a0 + tmp56b0;
                tmp[5][m][1] = tmp56a1 + tmp56b1;
                tmp[6][m][0] = tmp56a0 - tmp56b0;
                tmp[6][m][1] = tmp56a1 - tmp56b1;
                tmp[7][m][0] = r70 - r10 + (r30 - r50) * (__fp16)5.25f;
                tmp[7][m][1] = r71 - r11 + (r31 - r51) * (__fp16)5.25f;

                r0 += w;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 64 + jj * 2;
            __fp16* p1 = p0 + max_jj * 2;
            __fp16* p2 = p0 + max_jj * 2 * 2;
            __fp16* p3 = p0 + max_jj * 2 * 3;
            __fp16* p4 = p0 + max_jj * 2 * 4;
            __fp16* p5 = p0 + max_jj * 2 * 5;
            __fp16* p6 = p0 + max_jj * 2 * 6;
            __fp16* p7 = p0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];
                __fp16 r40 = tmp[m][4][0];
                __fp16 r41 = tmp[m][4][1];
                __fp16 r50 = tmp[m][5][0];
                __fp16 r51 = tmp[m][5][1];
                __fp16 r60 = tmp[m][6][0];
                __fp16 r61 = tmp[m][6][1];
                __fp16 r70 = tmp[m][7][0];
                __fp16 r71 = tmp[m][7][1];

                __fp16 tmp12a0 = r20 + r60 - r40 * (__fp16)4.25f;
                __fp16 tmp12a1 = r21 + r61 - r41 * (__fp16)4.25f;
                __fp16 tmp12b0 = r10 + r50 - r30 * (__fp16)4.25f;
                __fp16 tmp12b1 = r11 + r51 - r31 * (__fp16)4.25f;
                __fp16 tmp34a0 = r60 + r20 * (__fp16)0.25f - r40 * (__fp16)1.25f;
                __fp16 tmp34a1 = r61 + r21 * (__fp16)0.25f - r41 * (__fp16)1.25f;
                __fp16 tmp34b0 = r10 * (__fp16)0.5f - r30 * (__fp16)2.5f + r50 * (__fp16)2.f;
                __fp16 tmp34b1 = r11 * (__fp16)0.5f - r31 * (__fp16)2.5f + r51 * (__fp16)2.f;
                __fp16 tmp56a0 = r20 * (__fp16)4.f - r40 * (__fp16)5.f + r60;
                __fp16 tmp56a1 = r21 * (__fp16)4.f - r41 * (__fp16)5.f + r61;
                __fp16 tmp56b0 = r10 * (__fp16)2.f - r30 * (__fp16)2.5f + r50 * (__fp16)0.5f;
                __fp16 tmp56b1 = r11 * (__fp16)2.f - r31 * (__fp16)2.5f + r51 * (__fp16)0.5f;

                p0[0] = r00 - r60 + (r40 - r20) * (__fp16)5.25f;
                p0[1] = r01 - r61 + (r41 - r21) * (__fp16)5.25f;
                p1[0] = tmp12a0 + tmp12b0;
                p1[1] = tmp12a1 + tmp12b1;
                p2[0] = tmp12a0 - tmp12b0;
                p2[1] = tmp12a1 - tmp12b1;
                p3[0] = tmp34a0 + tmp34b0;
                p3[1] = tmp34a1 + tmp34b1;
                p4[0] = tmp34a0 - tmp34b0;
                p4[1] = tmp34a1 - tmp34b1;
                p5[0] = tmp56a0 + tmp56b0;
                p5[1] = tmp56a1 + tmp56b1;
                p6[0] = tmp56a0 - tmp56b0;
                p6[1] = tmp56a1 - tmp56b1;
                p7[0] = r70 - r10 + (r30 - r50) * (__fp16)5.25f;
                p7[1] = r71 - r11 + (r31 - r51) * (__fp16)5.25f;

                p0 += max_jj * 8 * 2;
                p1 += max_jj * 8 * 2;
                p2 += max_jj * 8 * 2;
                p3 += max_jj * 8 * 2;
                p4 += max_jj * 8 * 2;
                p5 += max_jj * 8 * 2;
                p6 += max_jj * 8 * 2;
                p7 += max_jj * 8 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        __fp16 tmp[8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0123 = bottom_blob.channel(k + kk).row<const __fp16>(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
                __fp16 r0 = 0.f;
                __fp16 r1 = 0.f;
                __fp16 r2 = 0.f;
                __fp16 r3 = 0.f;
                __fp16 r4 = 0.f;
                __fp16 r5 = 0.f;
                __fp16 r6 = 0.f;
                __fp16 r7 = 0.f;

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 6 + 1 < w) r1 = r0123[1];
                        if (tj * 6 + 2 < w) r2 = r0123[2];
                        if (tj * 6 + 3 < w) r3 = r0123[3];
                        if (tj * 6 + 4 < w) r4 = r0123[4];
                        if (tj * 6 + 5 < w) r5 = r0123[5];
                        if (tj * 6 + 6 < w) r6 = r0123[6];
                        if (tj * 6 + 7 < w) r7 = r0123[7];
                    }
                }

                __fp16 tmp12a = r2 + r6 - r4 * (__fp16)4.25f;
                __fp16 tmp12b = r1 + r5 - r3 * (__fp16)4.25f;
                __fp16 tmp34a = r6 + r2 * (__fp16)0.25f - r4 * (__fp16)1.25f;
                __fp16 tmp34b = r1 * (__fp16)0.5f - r3 * (__fp16)2.5f + r5 * (__fp16)2.f;
                __fp16 tmp56a = r2 * (__fp16)4.f - r4 * (__fp16)5.f + r6;
                __fp16 tmp56b = r1 * (__fp16)2.f - r3 * (__fp16)2.5f + r5 * (__fp16)0.5f;

                tmp[0][m] = r0 - r6 + (r4 - r2) * (__fp16)5.25f;
                tmp[1][m] = tmp12a + tmp12b;
                tmp[2][m] = tmp12a - tmp12b;
                tmp[3][m] = tmp34a + tmp34b;
                tmp[4][m] = tmp34a - tmp34b;
                tmp[5][m] = tmp56a + tmp56b;
                tmp[6][m] = tmp56a - tmp56b;
                tmp[7][m] = r7 - r1 + (r3 - r5) * (__fp16)5.25f;

                r0123 += w;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 64 + jj;
            __fp16* p1 = p0 + max_jj;
            __fp16* p2 = p0 + max_jj * 2;
            __fp16* p3 = p0 + max_jj * 3;
            __fp16* p4 = p0 + max_jj * 4;
            __fp16* p5 = p0 + max_jj * 5;
            __fp16* p6 = p0 + max_jj * 6;
            __fp16* p7 = p0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];
                __fp16 r4 = tmp[m][4];
                __fp16 r5 = tmp[m][5];
                __fp16 r6 = tmp[m][6];
                __fp16 r7 = tmp[m][7];

                __fp16 tmp12a = r2 + r6 - r4 * (__fp16)4.25f;
                __fp16 tmp12b = r1 + r5 - r3 * (__fp16)4.25f;
                __fp16 tmp34a = r6 + r2 * (__fp16)0.25f - r4 * (__fp16)1.25f;
                __fp16 tmp34b = r1 * (__fp16)0.5f - r3 * (__fp16)2.5f + r5 * (__fp16)2.f;
                __fp16 tmp56a = r2 * (__fp16)4.f - r4 * (__fp16)5.f + r6;
                __fp16 tmp56b = r1 * (__fp16)2.f - r3 * (__fp16)2.5f + r5 * (__fp16)0.5f;

                p0[0] = r0 - r6 + (r4 - r2) * (__fp16)5.25f;
                p1[0] = tmp12a + tmp12b;
                p2[0] = tmp12a - tmp12b;
                p3[0] = tmp34a + tmp34b;
                p4[0] = tmp34a - tmp34b;
                p5[0] = tmp56a + tmp56b;
                p6[0] = tmp56a - tmp56b;
                p7[0] = r7 - r1 + (r3 - r5) * (__fp16)5.25f;

                p0 += max_jj * 8;
                p1 += max_jj * 8;
                p2 += max_jj * 8;
                p3 += max_jj * 8;
                p4 += max_jj * 8;
                p5 += max_jj * 8;
                p6 += max_jj * 8;
                p7 += max_jj * 8;
            }
        }
    }
}

static inline void conv3x3s1_winograd63_transform_output_tile_fp16sa(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[6][8] = {
    //     {1.0f, 1.0f,  1.0f,  1.0f,  1.0f, 32.0f, 32.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  2.0f, -2.0f, 16.0f,-16.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f,  4.0f,  4.0f,  8.0f,  8.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  8.0f, -8.0f,  4.0f, -4.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f, 16.0f, 16.0f,  2.0f,  2.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 32.0f,-32.0f,  1.0f, -1.0f, 1.0f}
    // };

    const __fp16 coeffs[8] = {32.f, 16.f, 8.f, 4.f, 2.f, 0.f, 0.f, 0.f};
    float16x8_t _coeffs = vld1q_f16(coeffs);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 5) / 6;

    const __fp16* biasptr = bias;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
        float16x8_t _bias0 = biasptr ? vld1q_f16(biasptr + i + ii) : vdupq_n_f16(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[6][8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 64 + jj * 8;
            const __fp16* r1 = r0 + max_jj * 8;
            const __fp16* r2 = r0 + max_jj * 8 * 2;
            const __fp16* r3 = r0 + max_jj * 8 * 3;
            const __fp16* r4 = r0 + max_jj * 8 * 4;
            const __fp16* r5 = r0 + max_jj * 8 * 5;
            const __fp16* r6 = r0 + max_jj * 8 * 6;
            const __fp16* r7 = r0 + max_jj * 8 * 7;

            for (int m = 0; m < 8; m++)
            {
                float16x8_t _r0 = vld1q_f16(r0);
                float16x8_t _r1 = vld1q_f16(r1);
                float16x8_t _r2 = vld1q_f16(r2);
                float16x8_t _r3 = vld1q_f16(r3);
                float16x8_t _r4 = vld1q_f16(r4);
                float16x8_t _r5 = vld1q_f16(r5);
                float16x8_t _r6 = vld1q_f16(r6);
                float16x8_t _r7 = vld1q_f16(r7);

                float16x8_t _tmp024a = vaddq_f16(_r1, _r2);
                float16x8_t _tmp135a = vsubq_f16(_r1, _r2);
                float16x8_t _tmp024b = vaddq_f16(_r3, _r4);
                float16x8_t _tmp135b = vsubq_f16(_r3, _r4);
                float16x8_t _tmp024c = vaddq_f16(_r5, _r6);
                float16x8_t _tmp135c = vsubq_f16(_r5, _r6);

                float16x8_t _tmp0 = vaddq_f16(vaddq_f16(_r0, _tmp024a), vfmaq_laneq_f16(_tmp024b, _tmp024c, _coeffs, 0));
                float16x8_t _tmp1 = vfmaq_laneq_f16(vfmaq_laneq_f16(_tmp135a, _tmp135b, _coeffs, 4), _tmp135c, _coeffs, 1);
                float16x8_t _tmp2 = vfmaq_laneq_f16(vfmaq_laneq_f16(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2);
                float16x8_t _tmp3 = vfmaq_laneq_f16(vfmaq_laneq_f16(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3);
                float16x8_t _tmp4 = vfmaq_laneq_f16(vfmaq_laneq_f16(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _coeffs, 4);
                float16x8_t _tmp5 = vaddq_f16(vaddq_f16(_r7, _tmp135a), vfmaq_laneq_f16(_tmp135c, _tmp135b, _coeffs, 0));

                vst1q_f16(tmp[0][m], _tmp0);
                vst1q_f16(tmp[1][m], _tmp1);
                vst1q_f16(tmp[2][m], _tmp2);
                vst1q_f16(tmp[3][m], _tmp3);
                vst1q_f16(tmp[4][m], _tmp4);
                vst1q_f16(tmp[5][m], _tmp5);

                r0 += max_jj * 8 * 8;
                r1 += max_jj * 8 * 8;
                r2 += max_jj * 8 * 8;
                r3 += max_jj * 8 * 8;
                r4 += max_jj * 8 * 8;
                r5 += max_jj * 8 * 8;
                r6 += max_jj * 8 * 8;
                r7 += max_jj * 8 * 8;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float16x8_t _r0 = vld1q_f16(tmp[m][0]);
                float16x8_t _r1 = vld1q_f16(tmp[m][1]);
                float16x8_t _r2 = vld1q_f16(tmp[m][2]);
                float16x8_t _r3 = vld1q_f16(tmp[m][3]);
                float16x8_t _r4 = vld1q_f16(tmp[m][4]);
                float16x8_t _r5 = vld1q_f16(tmp[m][5]);
                float16x8_t _r6 = vld1q_f16(tmp[m][6]);
                float16x8_t _r7 = vld1q_f16(tmp[m][7]);

                float16x8_t _tmp024a = vaddq_f16(_r1, _r2);
                float16x8_t _tmp135a = vsubq_f16(_r1, _r2);
                float16x8_t _tmp024b = vaddq_f16(_r3, _r4);
                float16x8_t _tmp135b = vsubq_f16(_r3, _r4);
                float16x8_t _tmp024c = vaddq_f16(_r5, _r6);
                float16x8_t _tmp135c = vsubq_f16(_r5, _r6);

                float16x8_t _tmp0 = vaddq_f16(_bias0, vaddq_f16(vaddq_f16(_r0, _tmp024a), vfmaq_laneq_f16(_tmp024b, _tmp024c, _coeffs, 0)));
                float16x8_t _tmp1 = vaddq_f16(_bias0, vfmaq_laneq_f16(vfmaq_laneq_f16(_tmp135a, _tmp135b, _coeffs, 4), _tmp135c, _coeffs, 1));
                float16x8_t _tmp2 = vaddq_f16(_bias0, vfmaq_laneq_f16(vfmaq_laneq_f16(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2));
                float16x8_t _tmp3 = vaddq_f16(_bias0, vfmaq_laneq_f16(vfmaq_laneq_f16(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3));
                float16x8_t _tmp4 = vaddq_f16(_bias0, vfmaq_laneq_f16(vfmaq_laneq_f16(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _coeffs, 4));
                float16x8_t _tmp5 = vaddq_f16(_bias0, vaddq_f16(vaddq_f16(_r7, _tmp135a), vfmaq_laneq_f16(_tmp135c, _tmp135b, _coeffs, 0)));

                if (out_elempack == 8)
                {
                    vst1q_f16(outptr0, _tmp0);
                    if (tj * 6 + 1 < outw) vst1q_f16(outptr0 + 8, _tmp1);
                    if (tj * 6 + 2 < outw) vst1q_f16(outptr0 + 16, _tmp2);
                    if (tj * 6 + 3 < outw) vst1q_f16(outptr0 + 24, _tmp3);
                    if (tj * 6 + 4 < outw) vst1q_f16(outptr0 + 32, _tmp4);
                    if (tj * 6 + 5 < outw) vst1q_f16(outptr0 + 40, _tmp5);
                }
                if (out_elempack == 4)
                {
                    __fp16* outptr1 = outptr0 + N;

                    vst1_f16(outptr0, vget_low_f16(_tmp0));
                    vst1_f16(outptr1, vget_high_f16(_tmp0));
                    if (tj * 6 + 1 < outw)
                    {
                        vst1_f16(outptr0 + 4, vget_low_f16(_tmp1));
                        vst1_f16(outptr1 + 4, vget_high_f16(_tmp1));
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        vst1_f16(outptr0 + 8, vget_low_f16(_tmp2));
                        vst1_f16(outptr1 + 8, vget_high_f16(_tmp2));
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        vst1_f16(outptr0 + 12, vget_low_f16(_tmp3));
                        vst1_f16(outptr1 + 12, vget_high_f16(_tmp3));
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        vst1_f16(outptr0 + 16, vget_low_f16(_tmp4));
                        vst1_f16(outptr1 + 16, vget_high_f16(_tmp4));
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        vst1_f16(outptr0 + 20, vget_low_f16(_tmp5));
                        vst1_f16(outptr1 + 20, vget_high_f16(_tmp5));
                    }
                }
                if (out_elempack == 1)
                {
                    __fp16 tmp0[8];
                    __fp16 tmp1[8];
                    __fp16 tmp2[8];
                    __fp16 tmp3[8];
                    __fp16 tmp4[8];
                    __fp16 tmp5[8];
                    vst1q_f16(tmp0, _tmp0);
                    vst1q_f16(tmp1, _tmp1);
                    vst1q_f16(tmp2, _tmp2);
                    vst1q_f16(tmp3, _tmp3);
                    vst1q_f16(tmp4, _tmp4);
                    vst1q_f16(tmp5, _tmp5);

                    __fp16* outptr1 = outptr0 + N;
                    __fp16* outptr2 = outptr0 + N * 2;
                    __fp16* outptr3 = outptr0 + N * 3;
                    __fp16* outptr4 = outptr0 + N * 4;
                    __fp16* outptr5 = outptr0 + N * 5;
                    __fp16* outptr6 = outptr0 + N * 6;
                    __fp16* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                        outptr4[2] = tmp2[4];
                        outptr5[2] = tmp2[5];
                        outptr6[2] = tmp2[6];
                        outptr7[2] = tmp2[7];
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                        outptr4[3] = tmp3[4];
                        outptr5[3] = tmp3[5];
                        outptr6[3] = tmp3[6];
                        outptr7[3] = tmp3[7];
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                        outptr4[4] = tmp4[4];
                        outptr5[4] = tmp4[5];
                        outptr6[4] = tmp4[6];
                        outptr7[4] = tmp4[7];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                        outptr4[5] = tmp5[4];
                        outptr5[5] = tmp5[5];
                        outptr6[5] = tmp5[6];
                        outptr7[5] = tmp5[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float16x4_t _bias0 = biasptr ? vld1_f16(biasptr + i + ii) : vdup_n_f16(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        __fp16 tmp[6][8][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 64 + jj * 4;
            const __fp16* r1 = r0 + max_jj * 4;
            const __fp16* r2 = r0 + max_jj * 4 * 2;
            const __fp16* r3 = r0 + max_jj * 4 * 3;
            const __fp16* r4 = r0 + max_jj * 4 * 4;
            const __fp16* r5 = r0 + max_jj * 4 * 5;
            const __fp16* r6 = r0 + max_jj * 4 * 6;
            const __fp16* r7 = r0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
                float16x4_t _r0 = vld1_f16(r0);
                float16x4_t _r1 = vld1_f16(r1);
                float16x4_t _r2 = vld1_f16(r2);
                float16x4_t _r3 = vld1_f16(r3);
                float16x4_t _r4 = vld1_f16(r4);
                float16x4_t _r5 = vld1_f16(r5);
                float16x4_t _r6 = vld1_f16(r6);
                float16x4_t _r7 = vld1_f16(r7);

                float16x4_t _tmp024a = vadd_f16(_r1, _r2);
                float16x4_t _tmp135a = vsub_f16(_r1, _r2);
                float16x4_t _tmp024b = vadd_f16(_r3, _r4);
                float16x4_t _tmp135b = vsub_f16(_r3, _r4);
                float16x4_t _tmp024c = vadd_f16(_r5, _r6);
                float16x4_t _tmp135c = vsub_f16(_r5, _r6);

                float16x4_t _tmp0 = vadd_f16(vadd_f16(_r0, _tmp024a), vfma_laneq_f16(_tmp024b, _tmp024c, _coeffs, 0));
                float16x4_t _tmp1 = vfma_laneq_f16(vfma_laneq_f16(_tmp135a, _tmp135b, _coeffs, 4), _tmp135c, _coeffs, 1);
                float16x4_t _tmp2 = vfma_laneq_f16(vfma_laneq_f16(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2);
                float16x4_t _tmp3 = vfma_laneq_f16(vfma_laneq_f16(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3);
                float16x4_t _tmp4 = vfma_laneq_f16(vfma_laneq_f16(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _coeffs, 4);
                float16x4_t _tmp5 = vadd_f16(vadd_f16(_r7, _tmp135a), vfma_laneq_f16(_tmp135c, _tmp135b, _coeffs, 0));

                vst1_f16(tmp[0][m], _tmp0);
                vst1_f16(tmp[1][m], _tmp1);
                vst1_f16(tmp[2][m], _tmp2);
                vst1_f16(tmp[3][m], _tmp3);
                vst1_f16(tmp[4][m], _tmp4);
                vst1_f16(tmp[5][m], _tmp5);

                r0 += max_jj * 8 * 4;
                r1 += max_jj * 8 * 4;
                r2 += max_jj * 8 * 4;
                r3 += max_jj * 8 * 4;
                r4 += max_jj * 8 * 4;
                r5 += max_jj * 8 * 4;
                r6 += max_jj * 8 * 4;
                r7 += max_jj * 8 * 4;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float16x4_t _r0 = vld1_f16(tmp[m][0]);
                float16x4_t _r1 = vld1_f16(tmp[m][1]);
                float16x4_t _r2 = vld1_f16(tmp[m][2]);
                float16x4_t _r3 = vld1_f16(tmp[m][3]);
                float16x4_t _r4 = vld1_f16(tmp[m][4]);
                float16x4_t _r5 = vld1_f16(tmp[m][5]);
                float16x4_t _r6 = vld1_f16(tmp[m][6]);
                float16x4_t _r7 = vld1_f16(tmp[m][7]);

                float16x4_t _tmp024a = vadd_f16(_r1, _r2);
                float16x4_t _tmp135a = vsub_f16(_r1, _r2);
                float16x4_t _tmp024b = vadd_f16(_r3, _r4);
                float16x4_t _tmp135b = vsub_f16(_r3, _r4);
                float16x4_t _tmp024c = vadd_f16(_r5, _r6);
                float16x4_t _tmp135c = vsub_f16(_r5, _r6);

                float16x4_t _tmp0 = vadd_f16(_bias0, vadd_f16(vadd_f16(_r0, _tmp024a), vfma_laneq_f16(_tmp024b, _tmp024c, _coeffs, 0)));
                float16x4_t _tmp1 = vadd_f16(_bias0, vfma_laneq_f16(vfma_laneq_f16(_tmp135a, _tmp135b, _coeffs, 4), _tmp135c, _coeffs, 1));
                float16x4_t _tmp2 = vadd_f16(_bias0, vfma_laneq_f16(vfma_laneq_f16(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2));
                float16x4_t _tmp3 = vadd_f16(_bias0, vfma_laneq_f16(vfma_laneq_f16(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3));
                float16x4_t _tmp4 = vadd_f16(_bias0, vfma_laneq_f16(vfma_laneq_f16(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _coeffs, 4));
                float16x4_t _tmp5 = vadd_f16(_bias0, vadd_f16(vadd_f16(_r7, _tmp135a), vfma_laneq_f16(_tmp135c, _tmp135b, _coeffs, 0)));

                if (out_elempack == 4)
                {
                    vst1_f16(outptr0, _tmp0);
                    if (tj * 6 + 1 < outw) vst1_f16(outptr0 + 4, _tmp1);
                    if (tj * 6 + 2 < outw) vst1_f16(outptr0 + 8, _tmp2);
                    if (tj * 6 + 3 < outw) vst1_f16(outptr0 + 12, _tmp3);
                    if (tj * 6 + 4 < outw) vst1_f16(outptr0 + 16, _tmp4);
                    if (tj * 6 + 5 < outw) vst1_f16(outptr0 + 20, _tmp5);
                }
                if (out_elempack == 1)
                {
                    __fp16 tmp0[4];
                    __fp16 tmp1[4];
                    __fp16 tmp2[4];
                    __fp16 tmp3[4];
                    __fp16 tmp4[4];
                    __fp16 tmp5[4];
                    vst1_f16(tmp0, _tmp0);
                    vst1_f16(tmp1, _tmp1);
                    vst1_f16(tmp2, _tmp2);
                    vst1_f16(tmp3, _tmp3);
                    vst1_f16(tmp4, _tmp4);
                    vst1_f16(tmp5, _tmp5);

                    __fp16* outptr1 = outptr0 + N;
                    __fp16* outptr2 = outptr0 + N * 2;
                    __fp16* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : 0.f;
        __fp16 bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        __fp16 tmp[6][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 64 + jj * 2;
            const __fp16* r1 = r0 + max_jj * 2;
            const __fp16* r2 = r0 + max_jj * 2 * 2;
            const __fp16* r3 = r0 + max_jj * 2 * 3;
            const __fp16* r4 = r0 + max_jj * 2 * 4;
            const __fp16* r5 = r0 + max_jj * 2 * 5;
            const __fp16* r6 = r0 + max_jj * 2 * 6;
            const __fp16* r7 = r0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
                __fp16 tmp024a0 = r1[0] + r2[0];
                __fp16 tmp024a1 = r1[1] + r2[1];
                __fp16 tmp135a0 = r1[0] - r2[0];
                __fp16 tmp135a1 = r1[1] - r2[1];
                __fp16 tmp024b0 = r3[0] + r4[0];
                __fp16 tmp024b1 = r3[1] + r4[1];
                __fp16 tmp135b0 = r3[0] - r4[0];
                __fp16 tmp135b1 = r3[1] - r4[1];
                __fp16 tmp024c0 = r5[0] + r6[0];
                __fp16 tmp024c1 = r5[1] + r6[1];
                __fp16 tmp135c0 = r5[0] - r6[0];
                __fp16 tmp135c1 = r5[1] - r6[1];

                tmp[0][m][0] = r0[0] + tmp024a0 + tmp024b0 + tmp024c0 * (__fp16)32;
                tmp[0][m][1] = r0[1] + tmp024a1 + tmp024b1 + tmp024c1 * (__fp16)32;
                tmp[1][m][0] = tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * (__fp16)16;
                tmp[1][m][1] = tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * (__fp16)16;
                tmp[2][m][0] = tmp024a0 + tmp024b0 * (__fp16)4 + tmp024c0 * (__fp16)8;
                tmp[2][m][1] = tmp024a1 + tmp024b1 * (__fp16)4 + tmp024c1 * (__fp16)8;
                tmp[3][m][0] = tmp135a0 + tmp135b0 * (__fp16)8 + tmp135c0 * (__fp16)4;
                tmp[3][m][1] = tmp135a1 + tmp135b1 * (__fp16)8 + tmp135c1 * (__fp16)4;
                tmp[4][m][0] = tmp024a0 + tmp024b0 * (__fp16)16 + tmp024c0 + tmp024c0;
                tmp[4][m][1] = tmp024a1 + tmp024b1 * (__fp16)16 + tmp024c1 + tmp024c1;
                tmp[5][m][0] = r7[0] + tmp135a0 + tmp135b0 * (__fp16)32 + tmp135c0;
                tmp[5][m][1] = r7[1] + tmp135a1 + tmp135b1 * (__fp16)32 + tmp135c1;

                r0 += max_jj * 8 * 2;
                r1 += max_jj * 8 * 2;
                r2 += max_jj * 8 * 2;
                r3 += max_jj * 8 * 2;
                r4 += max_jj * 8 * 2;
                r5 += max_jj * 8 * 2;
                r6 += max_jj * 8 * 2;
                r7 += max_jj * 8 * 2;
            }

            __fp16* outptr0 = top_blob.channel(i + ii).row<__fp16>(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];
                __fp16 r40 = tmp[m][4][0];
                __fp16 r41 = tmp[m][4][1];
                __fp16 r50 = tmp[m][5][0];
                __fp16 r51 = tmp[m][5][1];
                __fp16 r60 = tmp[m][6][0];
                __fp16 r61 = tmp[m][6][1];
                __fp16 r70 = tmp[m][7][0];
                __fp16 r71 = tmp[m][7][1];

                __fp16 tmp024a0 = r10 + r20;
                __fp16 tmp024a1 = r11 + r21;
                __fp16 tmp135a0 = r10 - r20;
                __fp16 tmp135a1 = r11 - r21;
                __fp16 tmp024b0 = r30 + r40;
                __fp16 tmp024b1 = r31 + r41;
                __fp16 tmp135b0 = r30 - r40;
                __fp16 tmp135b1 = r31 - r41;
                __fp16 tmp024c0 = r50 + r60;
                __fp16 tmp024c1 = r51 + r61;
                __fp16 tmp135c0 = r50 - r60;
                __fp16 tmp135c1 = r51 - r61;

                __fp16 tmp00 = bias0 + r00 + tmp024a0 + tmp024b0 + tmp024c0 * (__fp16)32;
                __fp16 tmp01 = bias1 + r01 + tmp024a1 + tmp024b1 + tmp024c1 * (__fp16)32;
                __fp16 tmp10 = bias0 + tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * (__fp16)16;
                __fp16 tmp11 = bias1 + tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * (__fp16)16;
                __fp16 tmp20 = bias0 + tmp024a0 + tmp024b0 * (__fp16)4 + tmp024c0 * (__fp16)8;
                __fp16 tmp21 = bias1 + tmp024a1 + tmp024b1 * (__fp16)4 + tmp024c1 * (__fp16)8;
                __fp16 tmp30 = bias0 + tmp135a0 + tmp135b0 * (__fp16)8 + tmp135c0 * (__fp16)4;
                __fp16 tmp31 = bias1 + tmp135a1 + tmp135b1 * (__fp16)8 + tmp135c1 * (__fp16)4;
                __fp16 tmp40 = bias0 + tmp024a0 + tmp024b0 * (__fp16)16 + tmp024c0 + tmp024c0;
                __fp16 tmp41 = bias1 + tmp024a1 + tmp024b1 * (__fp16)16 + tmp024c1 + tmp024c1;
                __fp16 tmp50 = bias0 + r70 + tmp135a0 + tmp135b0 * (__fp16)32 + tmp135c0;
                __fp16 tmp51 = bias1 + r71 + tmp135a1 + tmp135b1 * (__fp16)32 + tmp135c1;

                // if (out_elempack == 1)
                {
                    __fp16* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp20;
                        outptr1[2] = tmp21;
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp30;
                        outptr1[3] = tmp31;
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp40;
                        outptr1[4] = tmp41;
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp50;
                        outptr1[5] = tmp51;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : 0.f;

        __fp16 tmp[6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 64 + jj;
            const __fp16* r1 = r0 + max_jj;
            const __fp16* r2 = r0 + max_jj * 2;
            const __fp16* r3 = r0 + max_jj * 3;
            const __fp16* r4 = r0 + max_jj * 4;
            const __fp16* r5 = r0 + max_jj * 5;
            const __fp16* r6 = r0 + max_jj * 6;
            const __fp16* r7 = r0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                __fp16 tmp024a = r1[0] + r2[0];
                __fp16 tmp135a = r1[0] - r2[0];
                __fp16 tmp024b = r3[0] + r4[0];
                __fp16 tmp135b = r3[0] - r4[0];
                __fp16 tmp024c = r5[0] + r6[0];
                __fp16 tmp135c = r5[0] - r6[0];

                tmp[0][m] = r0[0] + tmp024a + tmp024b + tmp024c * (__fp16)32;
                tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * (__fp16)16;
                tmp[2][m] = tmp024a + tmp024b * (__fp16)4 + tmp024c * (__fp16)8;
                tmp[3][m] = tmp135a + tmp135b * (__fp16)8 + tmp135c * (__fp16)4;
                tmp[4][m] = tmp024a + tmp024b * (__fp16)16 + tmp024c + tmp024c;
                tmp[5][m] = r7[0] + tmp135a + tmp135b * (__fp16)32 + tmp135c;

                r0 += max_jj * 8;
                r1 += max_jj * 8;
                r2 += max_jj * 8;
                r3 += max_jj * 8;
                r4 += max_jj * 8;
                r5 += max_jj * 8;
                r6 += max_jj * 8;
                r7 += max_jj * 8;
            }

            __fp16* outptr0 = top_blob.channel(i + ii).row<__fp16>(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];
                __fp16 r4 = tmp[m][4];
                __fp16 r5 = tmp[m][5];
                __fp16 r6 = tmp[m][6];
                __fp16 r7 = tmp[m][7];

                __fp16 tmp024a = r1 + r2;
                __fp16 tmp135a = r1 - r2;
                __fp16 tmp024b = r3 + r4;
                __fp16 tmp135b = r3 - r4;
                __fp16 tmp024c = r5 + r6;
                __fp16 tmp135c = r5 - r6;

                __fp16 tmp0 = bias0 + r0 + tmp024a + tmp024b + tmp024c * (__fp16)32;
                __fp16 tmp1 = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * (__fp16)16;
                __fp16 tmp2 = bias0 + tmp024a + tmp024b * (__fp16)4 + tmp024c * (__fp16)8;
                __fp16 tmp3 = bias0 + tmp135a + tmp135b * (__fp16)8 + tmp135c * (__fp16)4;
                __fp16 tmp4 = bias0 + tmp024a + tmp024b * (__fp16)16 + tmp024c + tmp024c;
                __fp16 tmp5 = bias0 + r7 + tmp135a + tmp135b * (__fp16)32 + tmp135c;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 6 + 1 < outw) outptr0[1] = tmp1;
                    if (tj * 6 + 2 < outw) outptr0[2] = tmp2;
                    if (tj * 6 + 3 < outw) outptr0[3] = tmp3;
                    if (tj * 6 + 4 < outw) outptr0[4] = tmp4;
                    if (tj * 6 + 5 < outw) outptr0[5] = tmp5;
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd63_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 6n+2, winograd F(6,3)
    int w_tiles = (outw + 5) / 6;
    int h_tiles = (outh + 5) / 6;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 64;

    // NCNN_LOGE("conv3x3s1_winograd63_fp16sa %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 2u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd63_transform_input_tile_fp16sa(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 2u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd63_transform_input_tile_fp16sa(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 2u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd63_transform_output_tile_fp16sa(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}
