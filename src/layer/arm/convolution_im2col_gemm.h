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

void pretty_print(const ncnn::Mat& m)
{
    return;
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
    printf("------------------------------------------------\n");
}
static void convolution_im2col_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, int maxk, int inch, int outch)
{
    // A = (maxk, inch), outch
    // const int elempack = A.elempack;
    const int A_hstep = maxk * inch;

    float* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        // if (elempack == 1)
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
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

            int kk = 0;
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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
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
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void convolution_im2col_pack_B_tile(const Mat& B, Mat& BT, int max_jj, int max_kk)
{
    NCNN_LOGE("convolution_im2col_pack_B_tile %d %d", max_jj, max_kk);
    const int elempack = B.elempack;

    float* pp = BT;

    int jj = 0;
#if __ARM_NEON
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (jj) * max_kk;
            const float* p1 = (const float*)B + (jj + 4) * max_kk;
            const float* p2 = (const float*)B + (jj + 8) * max_kk;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p1));
                vst1q_f32(pp + 8, vld1q_f32(p2));
                pp += 12;
                p0 += 4;
                p1 += 4;
                p2 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (jj) * max_kk;
            const float* p1 = (const float*)B + (jj + 1) * max_kk;
            const float* p2 = (const float*)B + (jj + 2) * max_kk;
            const float* p3 = (const float*)B + (jj + 3) * max_kk;
            const float* p4 = (const float*)B + (jj + 4) * max_kk;
            const float* p5 = (const float*)B + (jj + 5) * max_kk;
            const float* p6 = (const float*)B + (jj + 6) * max_kk;
            const float* p7 = (const float*)B + (jj + 7) * max_kk;
            const float* p8 = (const float*)B + (jj + 8) * max_kk;
            const float* p9 = (const float*)B + (jj + 9) * max_kk;
            const float* pa = (const float*)B + (jj + 10) * max_kk;
            const float* pb = (const float*)B + (jj + 11) * max_kk;

            int kk = 0;
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
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (jj) * max_kk;
            const float* p1 = (const float*)B + (jj + 4) * max_kk;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p1));
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (jj) * max_kk;
            const float* p1 = (const float*)B + (jj + 1) * max_kk;
            const float* p2 = (const float*)B + (jj + 2) * max_kk;
            const float* p3 = (const float*)B + (jj + 3) * max_kk;
            const float* p4 = (const float*)B + (jj + 4) * max_kk;
            const float* p5 = (const float*)B + (jj + 5) * max_kk;
            const float* p6 = (const float*)B + (jj + 6) * max_kk;
            const float* p7 = (const float*)B + (jj + 7) * max_kk;

            int kk = 0;
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
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (jj) * max_kk;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (jj) * max_kk;
            const float* p1 = (const float*)B + (jj + 1) * max_kk;
            const float* p2 = (const float*)B + (jj + 2) * max_kk;
            const float* p3 = (const float*)B + (jj + 3) * max_kk;

            int kk = 0;
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
            const float* p0 = (const float*)B + (jj) * max_kk;
            const float* p1 = (const float*)B + (jj + 1) * max_kk;

            int kk = 0;
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
            const float* p0 = (const float*)B + (jj) * max_kk;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void convolution_im2col_transpose_pack_B_tile(const Mat& B, Mat& BT, int max_jj, int max_kk)
{
    NCNN_LOGE("convolution_im2col_transpose_pack_B_tile %d %d", max_jj, max_kk);
    float* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        const float* p0 = B;

        int kk = 0;
        p0 += jj * 8;
        for (; kk + 7 < max_kk; kk += 8)
        {
            // transpose 8x12
#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0]  \n"

                "uzp1   v24.4s, v0.4s, v4.4s        \n"
                "uzp2   v25.4s, v0.4s, v4.4s        \n"
                "uzp1   v26.4s, v1.4s, v5.4s        \n"
                "uzp2   v27.4s, v1.4s, v5.4s        \n"
                "uzp1   v28.4s, v2.4s, v6.4s        \n"
                "uzp2   v29.4s, v2.4s, v6.4s        \n"
                "uzp1   v30.4s, v3.4s, v7.4s        \n"
                "uzp2   v31.4s, v3.4s, v7.4s        \n"

                "uzp1   v0.4s, v8.4s, v12.4s        \n"
                "uzp2   v1.4s, v8.4s, v12.4s        \n"
                "uzp1   v2.4s, v9.4s, v13.4s        \n"
                "uzp2   v3.4s, v9.4s, v13.4s        \n"
                "uzp1   v4.4s, v10.4s, v14.4s       \n"
                "uzp2   v5.4s, v10.4s, v14.4s       \n"
                "uzp1   v6.4s, v11.4s, v15.4s       \n"
                "uzp2   v7.4s, v11.4s, v15.4s       \n"

                "sub    %0, %0, #320                \n"

                "uzp1   v8.4s, v16.4s, v20.4s       \n"
                "uzp2   v9.4s, v16.4s, v20.4s       \n"
                "uzp1   v10.4s, v17.4s, v21.4s      \n"
                "uzp2   v11.4s, v17.4s, v21.4s      \n"
                "uzp1   v12.4s, v18.4s, v22.4s      \n"
                "uzp2   v13.4s, v18.4s, v22.4s      \n"
                "uzp1   v14.4s, v19.4s, v23.4s      \n"
                "uzp2   v15.4s, v19.4s, v23.4s      \n"

                "st1    {v24.4s}, [%1], #16         \n"
                "st1    {v0.4s}, [%1], #16          \n"
                "st1    {v8.4s}, [%1], #16          \n"
                "st1    {v26.4s}, [%1], #16         \n"
                "st1    {v2.4s}, [%1], #16          \n"
                "st1    {v10.4s}, [%1], #16         \n"
                "st1    {v28.4s}, [%1], #16         \n"
                "st1    {v4.4s}, [%1], #16          \n"
                "st1    {v12.4s}, [%1], #16         \n"
                "st1    {v30.4s}, [%1], #16         \n"
                "st1    {v6.4s}, [%1], #16          \n"
                "st1    {v14.4s}, [%1], #16         \n"

                "st1    {v25.4s}, [%1], #16         \n"
                "st1    {v1.4s}, [%1], #16          \n"
                "st1    {v9.4s}, [%1], #16          \n"
                "st1    {v27.4s}, [%1], #16         \n"
                "st1    {v3.4s}, [%1], #16          \n"
                "st1    {v11.4s}, [%1], #16         \n"
                "st1    {v29.4s}, [%1], #16         \n"
                "st1    {v5.4s}, [%1], #16          \n"
                "st1    {v13.4s}, [%1], #16         \n"
                "st1    {v31.4s}, [%1], #16         \n"
                "st1    {v7.4s}, [%1], #16          \n"
                "st1    {v15.4s}, [%1], #16         \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            p0 += max_jj * 8;
#else  // NCNN_GNU_INLINE_ASM
            float32x4x4_t _r0 = vld4q_f32(p0);
            float32x4x4_t _r1 = vld4q_f32(p0 + 16);
            float32x4x4_t _r2 = vld4q_f32(p0 + 32);
            float32x4x4_t _r3 = vld4q_f32(p0 + 48);
            float32x4x4_t _r4 = vld4q_f32(p0 + 64);
            float32x4x4_t _r5 = vld4q_f32(p0 + 80);
            float32x4x2_t _r04l = vuzpq_f32(_r0.val[0], _r1.val[0]);
            float32x4x2_t _r15l = vuzpq_f32(_r0.val[1], _r1.val[1]);
            float32x4x2_t _r26l = vuzpq_f32(_r0.val[2], _r1.val[2]);
            float32x4x2_t _r37l = vuzpq_f32(_r0.val[3], _r1.val[3]);
            float32x4x2_t _r04m = vuzpq_f32(_r2.val[0], _r3.val[0]);
            float32x4x2_t _r15m = vuzpq_f32(_r2.val[1], _r3.val[1]);
            float32x4x2_t _r26m = vuzpq_f32(_r2.val[2], _r3.val[2]);
            float32x4x2_t _r37m = vuzpq_f32(_r2.val[3], _r3.val[3]);
            float32x4x2_t _r04h = vuzpq_f32(_r4.val[0], _r5.val[0]);
            float32x4x2_t _r15h = vuzpq_f32(_r4.val[1], _r5.val[1]);
            float32x4x2_t _r26h = vuzpq_f32(_r4.val[2], _r5.val[2]);
            float32x4x2_t _r37h = vuzpq_f32(_r4.val[3], _r5.val[3]);
            vst1q_f32(pp, _r04l.val[0]);
            vst1q_f32(pp + 4, _r04m.val[0]);
            vst1q_f32(pp + 4 * 2, _r04h.val[0]);
            vst1q_f32(pp + 4 * 3, _r15l.val[0]);
            vst1q_f32(pp + 4 * 4, _r15m.val[0]);
            vst1q_f32(pp + 4 * 5, _r15h.val[0]);
            vst1q_f32(pp + 4 * 6, _r26l.val[0]);
            vst1q_f32(pp + 4 * 7, _r26m.val[0]);
            vst1q_f32(pp + 4 * 8, _r26h.val[0]);
            vst1q_f32(pp + 4 * 9, _r37l.val[0]);
            vst1q_f32(pp + 4 * 10, _r37m.val[0]);
            vst1q_f32(pp + 4 * 11, _r37h.val[0]);
            vst1q_f32(pp + 4 * 12, _r04l.val[1]);
            vst1q_f32(pp + 4 * 13, _r04m.val[1]);
            vst1q_f32(pp + 4 * 14, _r04h.val[1]);
            vst1q_f32(pp + 4 * 15, _r15l.val[1]);
            vst1q_f32(pp + 4 * 16, _r15m.val[1]);
            vst1q_f32(pp + 4 * 17, _r15h.val[1]);
            vst1q_f32(pp + 4 * 18, _r26l.val[1]);
            vst1q_f32(pp + 4 * 19, _r26m.val[1]);
            vst1q_f32(pp + 4 * 20, _r26h.val[1]);
            vst1q_f32(pp + 4 * 21, _r37l.val[1]);
            vst1q_f32(pp + 4 * 22, _r37m.val[1]);
            vst1q_f32(pp + 4 * 23, _r37h.val[1]);
            p0 += max_jj * 8;
            pp += 96;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 8;
        p0 += jj * 4;
        for (; kk + 3 < max_kk; kk += 4)
        {
            // transpose 4x12
            float32x4x4_t _r0 = vld4q_f32(p0);
            float32x4x4_t _r1 = vld4q_f32(p0 + 16);
            float32x4x4_t _r2 = vld4q_f32(p0 + 32);
            vst1q_f32(pp, _r0.val[0]);
            vst1q_f32(pp + 4, _r1.val[0]);
            vst1q_f32(pp + 4 * 2, _r2.val[0]);
            vst1q_f32(pp + 4 * 3, _r0.val[1]);
            vst1q_f32(pp + 4 * 4, _r1.val[1]);
            vst1q_f32(pp + 4 * 5, _r2.val[1]);
            vst1q_f32(pp + 4 * 6, _r0.val[2]);
            vst1q_f32(pp + 4 * 7, _r1.val[2]);
            vst1q_f32(pp + 4 * 8, _r2.val[2]);
            vst1q_f32(pp + 4 * 9, _r0.val[3]);
            vst1q_f32(pp + 4 * 10, _r1.val[3]);
            vst1q_f32(pp + 4 * 11, _r2.val[3]);
            p0 += max_jj * 4;
            pp += 48;
        }
        p0 -= jj * 2;
        for (; kk + 1 < max_kk; kk += 2)
        {
            // transpose 2x12
            float32x4x2_t _r0 = vld2q_f32(p0);
            float32x4x2_t _r1 = vld2q_f32(p0 + 8);
            float32x4x2_t _r2 = vld2q_f32(p0 + 16);
            vst1q_f32(pp, _r0.val[0]);
            vst1q_f32(pp + 4, _r1.val[0]);
            vst1q_f32(pp + 4 * 2, _r2.val[0]);
            vst1q_f32(pp + 4 * 3, _r0.val[1]);
            vst1q_f32(pp + 4 * 4, _r1.val[1]);
            vst1q_f32(pp + 4 * 5, _r2.val[1]);
            p0 += max_jj * 2;
            pp += 24;
        }
        p0 -= jj;
        for (; kk < max_kk; kk++)
        {
            float32x4_t _r0 = vld1q_f32(p0);
            float32x4_t _r1 = vld1q_f32(p0 + 4);
            float32x4_t _r2 = vld1q_f32(p0 + 8);
            vst1q_f32(pp, _r0);
            vst1q_f32(pp + 4, _r1);
            vst1q_f32(pp + 8, _r2);
            p0 += max_jj;
            pp += 12;
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = B;

        int kk = 0;
#if __aarch64__
        p0 += jj * 8;
        for (; kk + 7 < max_kk; kk += 8)
        {
            // transpose 8x8
#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0] \n"

                "uzp1   v16.4s, v0.4s, v4.4s        \n"
                "uzp2   v24.4s, v0.4s, v4.4s        \n"
                "uzp1   v18.4s, v1.4s, v5.4s        \n"
                "uzp2   v26.4s, v1.4s, v5.4s        \n"
                "uzp1   v20.4s, v2.4s, v6.4s        \n"
                "uzp2   v28.4s, v2.4s, v6.4s        \n"
                "uzp1   v22.4s, v3.4s, v7.4s        \n"
                "uzp2   v30.4s, v3.4s, v7.4s        \n"

                "sub    %0, %0, #192                \n"

                "uzp1   v17.4s, v8.4s, v12.4s       \n"
                "uzp2   v25.4s, v8.4s, v12.4s       \n"
                "uzp1   v19.4s, v9.4s, v13.4s       \n"
                "uzp2   v27.4s, v9.4s, v13.4s       \n"
                "uzp1   v21.4s, v10.4s, v14.4s      \n"
                "uzp2   v29.4s, v10.4s, v14.4s      \n"
                "uzp1   v23.4s, v11.4s, v15.4s      \n"
                "uzp2   v31.4s, v11.4s, v15.4s      \n"

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%1], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%1], #64 \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            p0 += max_jj * 8;
#else  // NCNN_GNU_INLINE_ASM
            float32x4x4_t _r0 = vld4q_f32(p0);
            float32x4x4_t _r1 = vld4q_f32(p0 + 16);
            float32x4x4_t _r2 = vld4q_f32(p0 + 32);
            float32x4x4_t _r3 = vld4q_f32(p0 + 48);
            float32x4x2_t _r04l = vuzpq_f32(_r0.val[0], _r1.val[0]);
            float32x4x2_t _r15l = vuzpq_f32(_r0.val[1], _r1.val[1]);
            float32x4x2_t _r26l = vuzpq_f32(_r0.val[2], _r1.val[2]);
            float32x4x2_t _r37l = vuzpq_f32(_r0.val[3], _r1.val[3]);
            float32x4x2_t _r04h = vuzpq_f32(_r2.val[0], _r3.val[0]);
            float32x4x2_t _r15h = vuzpq_f32(_r2.val[1], _r3.val[1]);
            float32x4x2_t _r26h = vuzpq_f32(_r2.val[2], _r3.val[2]);
            float32x4x2_t _r37h = vuzpq_f32(_r2.val[3], _r3.val[3]);
            vst1q_f32(pp, _r04l.val[0]);
            vst1q_f32(pp + 4, _r04h.val[0]);
            vst1q_f32(pp + 4 * 2, _r15l.val[0]);
            vst1q_f32(pp + 4 * 3, _r15h.val[0]);
            vst1q_f32(pp + 4 * 4, _r26l.val[0]);
            vst1q_f32(pp + 4 * 5, _r26h.val[0]);
            vst1q_f32(pp + 4 * 6, _r37l.val[0]);
            vst1q_f32(pp + 4 * 7, _r37h.val[0]);
            vst1q_f32(pp + 4 * 8, _r04l.val[1]);
            vst1q_f32(pp + 4 * 9, _r04h.val[1]);
            vst1q_f32(pp + 4 * 10, _r15l.val[1]);
            vst1q_f32(pp + 4 * 11, _r15h.val[1]);
            vst1q_f32(pp + 4 * 12, _r26l.val[1]);
            vst1q_f32(pp + 4 * 13, _r26h.val[1]);
            vst1q_f32(pp + 4 * 14, _r37l.val[1]);
            vst1q_f32(pp + 4 * 15, _r37h.val[1]);
            p0 += max_jj * 8;
            pp += 64;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 8;
#endif // __aarch64__
        p0 += jj * 4;
        for (; kk + 3 < max_kk; kk += 4)
        {
            // transpose 4x8
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                "sub    %0, %0, #64                 \n"
                "st1    {v0.4s}, [%1], #16          \n"
                "st1    {v4.4s}, [%1], #16          \n"
                "st1    {v1.4s}, [%1], #16          \n"
                "st1    {v5.4s}, [%1], #16          \n"
                "st1    {v2.4s}, [%1], #16          \n"
                "st1    {v6.4s}, [%1], #16          \n"
                "st1    {v3.4s}, [%1], #16          \n"
                "st1    {v7.4s}, [%1], #16          \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]          \n"
                "vldm       %0!, {d0-d7}        \n"
                "pld        [%0, #512]          \n"
                "vldm       %0, {d16-d23}       \n"

                "vtrn.32    q0, q1              \n"
                "vtrn.32    q2, q3              \n"
                "vtrn.32    q8, q9              \n"
                "vtrn.32    q10, q11            \n"
                "vswp       d1, d4              \n"
                "vswp       d3, d6              \n"
                "vswp       d17, d20            \n"
                "vswp       d19, d22            \n"
                "vswp       q1, q8              \n"
                "vswp       q3, q10             \n"

                "vst1.f32   {d0-d3}, [%1 :128]! \n"
                "vst1.f32   {d16-d19}, [%1 :128]! \n"
                "sub        %0, %0, #64         \n"
                "vst1.f32   {d4-d7}, [%1 :128]! \n"
                "vst1.f32   {d20-d23}, [%1 :128]! \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
#endif // __aarch64__
            p0 += max_jj * 4;
#else  // NCNN_GNU_INLINE_ASM
            float32x4x4_t _r0 = vld4q_f32(p0);
            float32x4x4_t _r1 = vld4q_f32(p0 + 16);
            vst1q_f32(pp, _r0.val[0]);
            vst1q_f32(pp + 4, _r1.val[0]);
            vst1q_f32(pp + 4 * 2, _r0.val[1]);
            vst1q_f32(pp + 4 * 3, _r1.val[1]);
            vst1q_f32(pp + 4 * 4, _r0.val[2]);
            vst1q_f32(pp + 4 * 5, _r1.val[2]);
            vst1q_f32(pp + 4 * 6, _r0.val[3]);
            vst1q_f32(pp + 4 * 7, _r1.val[3]);
            p0 += max_jj * 4;
            pp += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 4;
        p0 += jj * 2;
        for (; kk + 1 < max_kk; kk += 2)
        {
            // transpose 2x8
            float32x4x2_t _r0 = vld2q_f32(p0);
            float32x4x2_t _r1 = vld2q_f32(p0 + 8);
            vst1q_f32(pp, _r0.val[0]);
            vst1q_f32(pp + 4, _r1.val[0]);
            vst1q_f32(pp + 4 * 2, _r0.val[1]);
            vst1q_f32(pp + 4 * 3, _r1.val[1]);
            p0 += max_jj * 2;
            pp += 16;
        }
        p0 -= jj * 2;
        p0 += jj;
        for (; kk < max_kk; kk++)
        {
            float32x4_t _r0 = vld1q_f32(p0);
            float32x4_t _r1 = vld1q_f32(p0 + 4);
            vst1q_f32(pp, _r0);
            vst1q_f32(pp + 4, _r1);
            p0 += max_jj;
            pp += 8;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = B;

        int kk = 0;
#if __aarch64__
        p0 += jj * 8;
        for (; kk + 7 < max_kk; kk += 8)
        {
            // transpose 8x4
#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"

                "uzp1   v8.4s, v0.4s, v4.4s         \n"
                "uzp2   v12.4s, v0.4s, v4.4s        \n"
                "uzp1   v9.4s, v1.4s, v5.4s         \n"
                "uzp2   v13.4s, v1.4s, v5.4s        \n"

                "sub    %0, %0, #64                 \n"

                "uzp1   v10.4s, v2.4s, v6.4s        \n"
                "uzp2   v14.4s, v2.4s, v6.4s        \n"
                "uzp1   v11.4s, v3.4s, v7.4s        \n"
                "uzp2   v15.4s, v3.4s, v7.4s        \n"

                "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
            p0 += max_jj * 8;
#else  // NCNN_GNU_INLINE_ASM
            float32x4x4_t _r0;
            float32x4x4_t _r1;
            _r0.val[0] = vld1q_f32(p0);
            _r1.val[0] = vld1q_f32(p0 + 4);
            _r0.val[1] = vld1q_f32(p0 + 8);
            _r1.val[1] = vld1q_f32(p0 + 12);
            _r0.val[2] = vld1q_f32(p0 + 16);
            _r1.val[2] = vld1q_f32(p0 + 20);
            _r0.val[3] = vld1q_f32(p0 + 24);
            _r1.val[3] = vld1q_f32(p0 + 28);
            vst4q_f32(pp, _r0);
            vst4q_f32(pp + 16, _r1);
            p0 += max_jj * 8;
            pp += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 8;
#endif // __aarch64__
        p0 += jj * 4;
        for (; kk + 3 < max_kk; kk += 4)
        {
            // transpose 4x4
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                "st4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]          \n"
                "vldm       %0, {d0-d7}         \n"
                "vtrn.32    q0, q1              \n"
                "vtrn.32    q2, q3              \n"
                "vswp       d1, d4              \n"
                "vswp       d3, d6              \n"
                "vstm       %1!, {d0-d7}        \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
            p0 += max_jj * 4;
#else  // NCNN_GNU_INLINE_ASM
            float32x4x4_t _r0;
            _r0.val[0] = vld1q_f32(p0);
            _r0.val[1] = vld1q_f32(p0 + 4);
            _r0.val[2] = vld1q_f32(p0 + 8);
            _r0.val[3] = vld1q_f32(p0 + 12);
            vst4q_f32(pp, _r0);
            p0 += max_jj * 4;
            pp += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 4;
        p0 += jj * 2;
        for (; kk + 1 < max_kk; kk += 2)
        {
            // transpose 2x4
            float32x4x2_t _r0 = vld2q_f32(p0);
            vst1q_f32(pp, _r0.val[0]);
            vst1q_f32(pp + 4, _r0.val[1]);
            p0 += max_jj * 2;
            pp += 8;
        }
        p0 -= jj * 2;
        p0 += jj;
        for (; kk < max_kk; kk++)
        {
            float32x4_t _r0 = vld1q_f32(p0);
            vst1q_f32(pp, _r0);
            p0 += max_jj;
            pp += 4;
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = B;

        int kk = 0;
#if __ARM_NEON
#if __aarch64__
        p0 += jj * 8;
        for (; kk + 7 < max_kk; kk += 8)
        {
            // transpose 8x2
#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"

                "zip1   v4.4s, v0.4s, v2.4s         \n"
                "zip2   v5.4s, v0.4s, v2.4s         \n"
                "zip1   v6.4s, v1.4s, v3.4s         \n"
                "zip2   v7.4s, v1.4s, v3.4s         \n"

                "st1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
            p0 += max_jj * 8;
#else  // NCNN_GNU_INLINE_ASM
            float32x4x2_t _r0;
            float32x4x2_t _r1;
            _r0.val[0] = vld1q_f32(p0);
            _r1.val[0] = vld1q_f32(p0 + 4);
            _r0.val[1] = vld1q_f32(p0 + 8);
            _r1.val[1] = vld1q_f32(p0 + 12);
            vst2q_f32(pp, _r0);
            vst2q_f32(pp + 8, _r1);
            p0 += max_jj * 8;
            pp += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 8;
#endif // __aarch64__
        p0 += jj * 4;
        for (; kk + 3 < max_kk; kk += 4)
        {
            // transpose 4x2
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #256]       \n"
                "ld1    {v0.4s, v1.4s}, [%0]        \n"
                "st2    {v0.4s, v1.4s}, [%1], #32   \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0", "v1");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #256]          \n"
                "vld1.f32   {d0-d3}, [%0 :128]  \n"
                "vst2.f32   {d0-d3}, [%1 :128]! \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "q0", "q1");
#endif // __aarch64__
            p0 += max_jj * 4;
#else  // NCNN_GNU_INLINE_ASM
            float32x4x2_t _r0;
            _r0.val[0] = vld1q_f32(p0);
            _r0.val[1] = vld1q_f32(p0 + 4);
            vst2q_f32(pp, _r0);
            p0 += max_jj * 4;
            pp += 8;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 4;
#endif // __ARM_NEON
        p0 += jj * 2;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[2];
            pp[2] = p0[1];
            pp[3] = p0[3];
            p0 += max_jj * 2;
            pp += 4;
        }
        p0 -= jj * 2;
        p0 += jj;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            p0 += max_jj;
            pp += 2;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const float* p0 = B;

        int kk = 0;
#if __ARM_NEON
#if __aarch64__
        p0 += jj * 8;
        for (; kk + 7 < max_kk; kk += 8)
        {
#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "prfm   pldl1keep, [%0, #256]       \n"
                "ld1    {v0.4s, v1.4s}, [%0]        \n"
                "st1    {v0.4s, v1.4s}, [%1], #32   \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0", "v1");
            p0 += max_jj * 8;
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _r0 = vld1q_f32(p0);
            float32x4_t _r1 = vld1q_f32(p0 + 4);
            vst1q_f32(pp, _r0);
            vst1q_f32(pp + 4, _r1);
            p0 += max_jj * 8;
            pp += 8;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 8;
#endif // __aarch64__
        p0 += jj * 4;
        for (; kk + 3 < max_kk; kk += 4)
        {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #128]       \n"
                "ld1    {v0.4s}, [%0]               \n"
                "st1    {v0.4s}, [%1], #16          \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "v0");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #128]          \n"
                "vld1.f32   {d0-d1}, [%0]       \n"
                "vst1.f32   {d0-d1}, [%1]!      \n"
                : "=r"(p0), // %0
                "=r"(pp)  // %1
                : "0"(p0),
                "1"(pp)
                : "memory", "q0");
#endif // __aarch64__
            p0 += max_jj * 4;
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _r0 = vld1q_f32(p0);
            vst1q_f32(pp, _r0);
            p0 += max_jj * 4;
            pp += 4;
#endif // NCNN_GNU_INLINE_ASM
        }
        p0 -= jj * 4;
#endif // __ARM_NEON
        p0 += jj * 2;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            p0 += max_jj * 2;
            pp += 2;
        }
        p0 -= jj * 2;
        p0 += jj;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            p0 += max_jj;
            pp += 1;
        }
    }
}

static void convolution_gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    NCNN_LOGE("convolution_gemm_transB_packed_tile %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

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
                if (pC)
                {
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

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

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum10);
                    vst1q_f32(outptr0 + 4 * 2, _sum20);
                    vst1q_f32(outptr0 + 4 * 3, _sum30);
                    vst1q_f32(outptr0 + 4 * 4, _sum40);
                    vst1q_f32(outptr0 + 4 * 5, _sum50);
                    vst1q_f32(outptr0 + 4 * 6, _sum60);
                    vst1q_f32(outptr0 + 4 * 7, _sum70);
                    vst1q_f32(outptr0 + 4 * 8, _sum80);
                    vst1q_f32(outptr0 + 4 * 9, _sum90);
                    vst1q_f32(outptr0 + 4 * 10, _suma0);
                    vst1q_f32(outptr0 + 4 * 11, _sumb0);

                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 2, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 3, _sum31);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 4, _sum41);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 5, _sum51);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 6, _sum61);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 7, _sum71);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 8, _sum81);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 9, _sum91);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 10, _suma1);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 11, _sumb1);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x12_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71, _sum80, _sum81, _sum90, _sum91, _suma0, _suma1, _sumb0, _sumb1);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + 8, _sum10);
                    vst1q_f32(outptr0 + out_hstep, _sum11);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum20);
                    vst1q_f32(outptr0 + out_hstep + 8, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum30);
                    vst1q_f32(outptr0 + out_hstep * 2 + 4, _sum31);
                    vst1q_f32(outptr0 + out_hstep * 2 + 8, _sum40);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum41);
                    vst1q_f32(outptr0 + out_hstep * 3 + 4, _sum50);
                    vst1q_f32(outptr0 + out_hstep * 3 + 8, _sum51);
                    vst1q_f32(outptr0 + out_hstep * 4, _sum60);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum61);
                    vst1q_f32(outptr0 + out_hstep * 4 + 8, _sum70);
                    vst1q_f32(outptr0 + out_hstep * 5, _sum71);
                    vst1q_f32(outptr0 + out_hstep * 5 + 4, _sum80);
                    vst1q_f32(outptr0 + out_hstep * 5 + 8, _sum81);
                    vst1q_f32(outptr0 + out_hstep * 6, _sum90);
                    vst1q_f32(outptr0 + out_hstep * 6 + 4, _sum91);
                    vst1q_f32(outptr0 + out_hstep * 6 + 8, _suma0);
                    vst1q_f32(outptr0 + out_hstep * 7, _suma1);
                    vst1q_f32(outptr0 + out_hstep * 7 + 4, _sumb0);
                    vst1q_f32(outptr0 + out_hstep * 7 + 8, _sumb1);

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
                if (pC)
                {
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);

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

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum10);
                    vst1q_f32(outptr0 + 4 * 2, _sum20);
                    vst1q_f32(outptr0 + 4 * 3, _sum30);
                    vst1q_f32(outptr0 + 4 * 4, _sum40);
                    vst1q_f32(outptr0 + 4 * 5, _sum50);
                    vst1q_f32(outptr0 + 4 * 6, _sum60);
                    vst1q_f32(outptr0 + 4 * 7, _sum70);

                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 2, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 3, _sum31);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 4, _sum41);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 5, _sum51);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 6, _sum61);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 7, _sum71);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep, _sum10);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum20);
                    vst1q_f32(outptr0 + out_hstep * 2 + 4, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum30);
                    vst1q_f32(outptr0 + out_hstep * 3 + 4, _sum31);
                    vst1q_f32(outptr0 + out_hstep * 4, _sum40);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum41);
                    vst1q_f32(outptr0 + out_hstep * 5, _sum50);
                    vst1q_f32(outptr0 + out_hstep * 5 + 4, _sum51);
                    vst1q_f32(outptr0 + out_hstep * 6, _sum60);
                    vst1q_f32(outptr0 + out_hstep * 6 + 4, _sum61);
                    vst1q_f32(outptr0 + out_hstep * 7, _sum70);
                    vst1q_f32(outptr0 + out_hstep * 7 + 4, _sum71);

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
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB0 = vld1q_f32(pB);

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

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum10);
                    vst1q_f32(outptr0 + 4 * 2, _sum20);
                    vst1q_f32(outptr0 + 4 * 3, _sum30);

                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 2, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 3, _sum31);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + out_hstep * 1, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum10);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 4, _sum20);
                    vst1q_f32(outptr0 + out_hstep * 5, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 6, _sum30);
                    vst1q_f32(outptr0 + out_hstep * 7, _sum31);

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
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x2_t _pB0 = vld1_f32(pB);

                _sum00 = vfmaq_lane_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pA1, _pB0, 1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum10);

                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum11);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    float sum1[8];
                    vst1q_f32(sum0, _sum00);
                    vst1q_f32(sum0 + 4, _sum01);
                    vst1q_f32(sum1, _sum10);
                    vst1q_f32(sum1 + 4, _sum11);

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
        }
        for (; jj < max_jj; jj += 1)
        {
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB = vld1q_dup_f32(pB);

                _sum00 = vfmaq_f32(_sum00, _pA0, _pB);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    vst1q_f32(sum0, _sum00);
                    vst1q_f32(sum0 + 4, _sum01);

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
        }

        pAT += max_kk * 8;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

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
                if (pC)
                {
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

#if __aarch64__
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
#else // __aarch64__
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _pA, vget_low_f32(_pB1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _pA, vget_low_f32(_pB1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _pA, vget_high_f32(_pB1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _pA, vget_high_f32(_pB1), 1);
                _sum8 = vmlaq_lane_f32(_sum8, _pA, vget_low_f32(_pB2), 0);
                _sum9 = vmlaq_lane_f32(_sum9, _pA, vget_low_f32(_pB2), 1);
                _suma = vmlaq_lane_f32(_suma, _pA, vget_high_f32(_pB2), 0);
                _sumb = vmlaq_lane_f32(_sumb, _pA, vget_high_f32(_pB2), 1);
#endif // __aarch64__

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 4 * 2, _sum2);
                    vst1q_f32(outptr0 + 4 * 3, _sum3);
                    vst1q_f32(outptr0 + 4 * 4, _sum4);
                    vst1q_f32(outptr0 + 4 * 5, _sum5);
                    vst1q_f32(outptr0 + 4 * 6, _sum6);
                    vst1q_f32(outptr0 + 4 * 7, _sum7);
                    vst1q_f32(outptr0 + 4 * 8, _sum8);
                    vst1q_f32(outptr0 + 4 * 9, _sum9);
                    vst1q_f32(outptr0 + 4 * 10, _suma);
                    vst1q_f32(outptr0 + 4 * 11, _sumb);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose4x12_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 8, _sum2);
                    vst1q_f32(outptr0 + out_hstep, _sum3);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum4);
                    vst1q_f32(outptr0 + out_hstep + 8, _sum5);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum6);
                    vst1q_f32(outptr0 + out_hstep * 2 + 4, _sum7);
                    vst1q_f32(outptr0 + out_hstep * 2 + 8, _sum8);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum9);
                    vst1q_f32(outptr0 + out_hstep * 3 + 4, _suma);
                    vst1q_f32(outptr0 + out_hstep * 3 + 8, _sumb);
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
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);

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

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 4 * 2, _sum2);
                    vst1q_f32(outptr0 + 4 * 3, _sum3);
                    vst1q_f32(outptr0 + 4 * 4, _sum4);
                    vst1q_f32(outptr0 + 4 * 5, _sum5);
                    vst1q_f32(outptr0 + 4 * 6, _sum6);
                    vst1q_f32(outptr0 + 4 * 7, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + out_hstep, _sum2);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum3);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum4);
                    vst1q_f32(outptr0 + out_hstep * 2 + 4, _sum5);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum6);
                    vst1q_f32(outptr0 + out_hstep * 3 + 4, _sum7);
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
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x4_t _pB = vld1q_f32(pB);

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

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 4 * 2, _sum2);
                    vst1q_f32(outptr0 + 4 * 3, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + out_hstep * 1, _sum1);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum2);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum3);
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
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x2_t _pB = vld1_f32(pB);

#if __aarch64__
                _sum0 = vfmaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pA, _pB, 1);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, _pB, 1);
#endif

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    vst1q_f32(sum0, _sum0);
                    vst1q_f32(sum1, _sum1);

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
        }
        for (; jj < max_jj; jj += 1)
        {
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x4_t _pB = vdupq_n_f32(pB[0]);

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, _pB);
#else
                _sum0 = vmlaq_f32(_sum0, _pA, _pB);
#endif

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    vst1q_f32(sum0, _sum0);

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
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum02;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum12;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vdupq_n_f32(pC[0]);
                    _sum01 = vdupq_n_f32(pC[0]);
                    _sum02 = vdupq_n_f32(pC[0]);
                    _sum10 = vdupq_n_f32(pC[1]);
                    _sum11 = vdupq_n_f32(pC[1]);
                    _sum12 = vdupq_n_f32(pC[1]);
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum02 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum12 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                float32x4x2_t _tmp45 = vld2q_f32(outptr + 16);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum02 = _tmp45.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
                _sum12 = _tmp45.val[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

                float32x2_t _pA = vld1_f32(pA);
#if __aarch64__
                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum02 = vfmaq_lane_f32(_sum02, _pB2, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
                _sum12 = vfmaq_lane_f32(_sum12, _pB2, _pA, 1);
#else
                _sum00 = vmlaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vmlaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum02 = vmlaq_lane_f32(_sum02, _pB2, _pA, 0);
                _sum10 = vmlaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vmlaq_lane_f32(_sum11, _pB1, _pA, 1);
                _sum12 = vmlaq_lane_f32(_sum12, _pB2, _pA, 1);
#endif

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + 8, _sum02);
                    vst1q_f32(outptr0 + out_hstep, _sum10);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep + 8, _sum12);
                    outptr0 += 12;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                float32x4x2_t _tmp45;
                _tmp45.val[0] = _sum02;
                _tmp45.val[1] = _sum12;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
                vst2q_f32(outptr + 16, _tmp45);
            }

            outptr += 24;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vdupq_n_f32(pC[0]);
                    _sum01 = vdupq_n_f32(pC[0]);
                    _sum10 = vdupq_n_f32(pC[1]);
                    _sum11 = vdupq_n_f32(pC[1]);
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
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);

                float32x2_t _pA = vld1_f32(pA);
#if __aarch64__
                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
#else
                _sum00 = vmlaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vmlaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vmlaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vmlaq_lane_f32(_sum11, _pB1, _pA, 1);
#endif

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep, _sum10);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum11);
                    outptr0 += 8;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
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
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[1]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                _sum0 = _tmp01.val[0];
                _sum1 = _tmp01.val[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = vld1q_f32(pB);

                float32x2_t _pA = vld1_f32(pA);
#if __aarch64__
                _sum0 = vfmaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pB, _pA, 1);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pB, _pA, 1);
#endif

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + out_hstep, _sum1);
                    outptr0 += 4;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum0;
                _tmp01.val[1] = _sum1;
                vst2q_f32(outptr, _tmp01);
            }

            outptr += 8;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

                float32x4_t _pA0 = vdupq_n_f32(pA[0]);
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA0, _pB2);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vmlaq_f32(_sum2, _pA0, _pB2);
#endif

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 8, _sum2);
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);

                float32x4_t _pA0 = vdupq_n_f32(pA[0]);
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = vld1q_f32(pB);
                float32x4_t _pA = vdupq_n_f32(pA[0]);

#if __aarch64__
                _sum = vfmaq_f32(_sum, _pA, _pB);
#else
                _sum = vmlaq_f32(_sum, _pA, _pB);
#endif

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum);
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

static void convolution_im2col_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
    // const int h = bottom_blob.h;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    // const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    // const int outh = (h - kernel_extent_h) / stride_h + 1;

    // const int gap = (w * stride_h - outw * stride_w) * elempack;
    // bottom_im2col(size, maxk * channels, elemsize, elempack);

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    float* p0 = (float*)B;

    for (int jj = 0; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        if (elempack == 4)
        {
            for (int kk = 0; kk < max_kk / 4; kk++)
            {
                int p = (k + kk) / maxk;
                int uv = (k + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x = stride_w * dx + dilation_w * v;
                int y = stride_h * dy + dilation_h * u;

                const float* sptr = img.row(y) + x * 4;

                p0[0] = sptr[0];
                p0[1] = sptr[1];
                p0[2] = sptr[2];
                p0[3] = sptr[3];

                p0 += 4;
            }
        }

        if (elempack == 1)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                int p = (k + kk) / maxk;
                int uv = (k + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x = stride_w * dx + dilation_w * v;
                int y = stride_h * dy + dilation_h * u;

                const float* sptr = img.row(y) + x;

                p0[0] = sptr[0];

                p0 += 1;
            }
        }
    }
}

static void convolution_im2col_gemm_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    TILE_M = (M + 7) / 8 * 8;
    TILE_K = (K + 3) / 4 * 4;
    // get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    // Mat A_data = kernel.reshape(maxk * inch, outch);
    // wrap inch elempack

    int elempack = 1;
    if (opt.use_packing_layout)
    {
        elempack = inch % 4 == 0 ? 4 : 1;
    }


    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch);

        for (int q = 0; q < outch; q += 1)
        {
            float* g00 = A_data.row(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk, maxk, inch, outch);
        }
    }

    pretty_print(AT);
}

static void convolution_im2col_gemm(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    NCNN_LOGE("convolution_im2col_gemm  bottom_blob %d %d %d", bottom_blob.w, bottom_blob.h, bottom_blob.c);
    NCNN_LOGE("convolution_im2col_gemm           AT %d %d %d", AT.w, AT.h, AT.c);
    NCNN_LOGE("convolution_im2col_gemm         bias %d", bias.w);

    pretty_print(bottom_blob);

    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    TILE_M = (M + 7) / 8 * 8;
    TILE_N = (N + 3) / 4 * 4;
    TILE_K = (K + 3) / 4 * 4;
    // get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.blob_allocator);

    const int nn_NK = nn_N * nn_K;

    {
        // Mat B_tileX(TILE_N * TILE_K, 1, nT, bottom_blob.elemsize, bottom_blob.elempack, opt.workspace_allocator);
        Mat B_tileX(TILE_N * TILE_K, 1, nT, 4u, opt.workspace_allocator);

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

            // im2col
            convolution_im2col_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);

            pretty_print(B_tile);

            Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

            // convolution_im2col_transpose_pack_B_tile(B_tile, BT_tile, max_jj, max_kk);
            convolution_im2col_pack_B_tile(B_tile, BT_tile, max_jj, max_kk);

            // pretty_print(BT_tile);
        }
    }

    pretty_print(BT);

    Mat topT_tileX(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

                convolution_gemm_transB_packed_tile(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }

    pretty_print(top_blob);
}
