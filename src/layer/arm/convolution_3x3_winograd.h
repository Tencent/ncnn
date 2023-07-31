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

static void conv3x3s1_winograd_pack_A_tile(const Mat& A, Mat& AT, int batch, int max_ii, int max_kk)
{
    const int N = max_kk * batch;

    for (int b = 0; b < batch; b++)
    {
        float* pp = AT.row(b);

        int ii = 0;
#if __ARM_NEON
#if __aarch64__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + ii * N + b;

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
#endif // __aarch64__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + ii * N + b;

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
#endif // __ARM_NEON
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + ii * N + b;

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
            const float* p0 = (const float*)A + ii * N + b;

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

static void conv3x3s1_winograd_transpose_pack_B_tile(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk, int nT)
{
    #pragma omp parallel for num_threads(nT)
    for (int b = 0; b < batch; b++)
    {
        float* pp = BT.row(b);

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            const float* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 8;
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
                p0 += max_jj * batch * 8;
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
                p0 += max_jj * batch * 8;
                pp += 96;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 4;
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
                p0 += max_jj * batch * 4;
                pp += 48;
            }
            p0 -= (b * max_jj + jj) * 2;
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
                p0 += max_jj * batch * 2;
                pp += 24;
            }
            p0 -= (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                float32x4_t _r0 = vld1q_f32(p0);
                float32x4_t _r1 = vld1q_f32(p0 + 4);
                float32x4_t _r2 = vld1q_f32(p0 + 8);
                vst1q_f32(pp, _r0);
                vst1q_f32(pp + 4, _r1);
                vst1q_f32(pp + 8, _r2);
                p0 += max_jj * batch;
                pp += 12;
            }
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* p0 = B;

            int kk = 0;
#if __aarch64__
            p0 += (b * max_jj + jj) * 8;
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
                p0 += max_jj * batch * 8;
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
                p0 += max_jj * batch * 8;
                pp += 64;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __aarch64__
            p0 += (b * max_jj + jj) * 4;
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
                p0 += max_jj * batch * 4;
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
                p0 += max_jj * batch * 4;
                pp += 32;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                // transpose 2x8
                float32x4x2_t _r0 = vld2q_f32(p0);
                float32x4x2_t _r1 = vld2q_f32(p0 + 8);
                vst1q_f32(pp, _r0.val[0]);
                vst1q_f32(pp + 4, _r1.val[0]);
                vst1q_f32(pp + 4 * 2, _r0.val[1]);
                vst1q_f32(pp + 4 * 3, _r1.val[1]);
                p0 += max_jj * batch * 2;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                float32x4_t _r0 = vld1q_f32(p0);
                float32x4_t _r1 = vld1q_f32(p0 + 4);
                vst1q_f32(pp, _r0);
                vst1q_f32(pp + 4, _r1);
                p0 += max_jj * batch;
                pp += 8;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* p0 = B;

            int kk = 0;
#if __aarch64__
            p0 += (b * max_jj + jj) * 8;
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
                p0 += max_jj * batch * 8;
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
                p0 += max_jj * batch * 8;
                pp += 32;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __aarch64__
            p0 += (b * max_jj + jj) * 4;
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
                p0 += max_jj * batch * 4;
#else  // NCNN_GNU_INLINE_ASM
                float32x4x4_t _r0;
                _r0.val[0] = vld1q_f32(p0);
                _r0.val[1] = vld1q_f32(p0 + 4);
                _r0.val[2] = vld1q_f32(p0 + 8);
                _r0.val[3] = vld1q_f32(p0 + 12);
                vst4q_f32(pp, _r0);
                p0 += max_jj * batch * 4;
                pp += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                // transpose 2x4
                float32x4x2_t _r0 = vld2q_f32(p0);
                vst1q_f32(pp, _r0.val[0]);
                vst1q_f32(pp + 4, _r0.val[1]);
                p0 += max_jj * batch * 2;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                float32x4_t _r0 = vld1q_f32(p0);
                vst1q_f32(pp, _r0);
                p0 += max_jj * batch;
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
            p0 += (b * max_jj + jj) * 8;
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
                p0 += max_jj * batch * 8;
#else  // NCNN_GNU_INLINE_ASM
                float32x4x2_t _r0;
                float32x4x2_t _r1;
                _r0.val[0] = vld1q_f32(p0);
                _r1.val[0] = vld1q_f32(p0 + 4);
                _r0.val[1] = vld1q_f32(p0 + 8);
                _r1.val[1] = vld1q_f32(p0 + 12);
                vst2q_f32(pp, _r0);
                vst2q_f32(pp + 8, _r1);
                p0 += max_jj * batch * 8;
                pp += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __aarch64__
            p0 += (b * max_jj + jj) * 4;
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
                p0 += max_jj * batch * 4;
#else  // NCNN_GNU_INLINE_ASM
                float32x4x2_t _r0;
                _r0.val[0] = vld1q_f32(p0);
                _r0.val[1] = vld1q_f32(p0 + 4);
                vst2q_f32(pp, _r0);
                p0 += max_jj * batch * 4;
                pp += 8;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 4;
#endif // __ARM_NEON
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
            const float* p0 = B;

            int kk = 0;
#if __ARM_NEON
#if __aarch64__
            p0 += (b * max_jj + jj) * 8;
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
                p0 += max_jj * batch * 8;
#else  // NCNN_GNU_INLINE_ASM
                float32x4_t _r0 = vld1q_f32(p0);
                float32x4_t _r1 = vld1q_f32(p0 + 4);
                vst1q_f32(pp, _r0);
                vst1q_f32(pp + 4, _r1);
                p0 += max_jj * batch * 8;
                pp += 8;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __aarch64__
            p0 += (b * max_jj + jj) * 4;
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
                p0 += max_jj * batch * 4;
#else  // NCNN_GNU_INLINE_ASM
                float32x4_t _r0 = vld1q_f32(p0);
                vst1q_f32(pp, _r0);
                p0 += max_jj * batch * 4;
                pp += 4;
#endif // NCNN_GNU_INLINE_ASM
            }
            p0 -= (b * max_jj + jj) * 4;
#endif // __ARM_NEON
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

static void conv3x3s1_winograd_gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk, int use_a53_a55_optimized_kernel)
{
    // NCNN_LOGE("conv3x3s1_winograd_gemm_transB_packed_tile %d %d %d", max_ii, max_jj, max_kk);
    float* outptr = top_blob;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                if (use_a53_a55_optimized_kernel && cpu_support_arm_asimdhp())
                {
                    // a55
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                        "subs   %0, %0, #320                \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v8.16b, v8.16b, v8.16b      \n"
                        "eor    v9.16b, v9.16b, v9.16b      \n"
                        "eor    v10.16b, v10.16b, v10.16b   \n"
                        "eor    v11.16b, v11.16b, v11.16b   \n"
                        "eor    v12.16b, v12.16b, v12.16b   \n"
                        "eor    v13.16b, v13.16b, v13.16b   \n"
                        "eor    v14.16b, v14.16b, v14.16b   \n"
                        "eor    v15.16b, v15.16b, v15.16b   \n"
                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "eor    v17.16b, v17.16b, v17.16b   \n"
                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"
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
                        "ld1    {v4.4s}, [%1], #16          \n"
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s}, [%2], #16          \n"

                        "ldr    d5, [%1], #8                \n"
                        "ldr    x25, [%1], #8               \n"

                        ".align 4                           \n"
                        "2:                                 \n"
                        "ldr    d1, [%2], #8                \n"
                        "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                        "ldr    x21, [%2], #8               \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]      \n"
                        "ins    v5.d[1], x25                \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]      \n"
                        "ldr    d2, [%2], #8                \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]      \n"
                        "ldr    x22, [%2], #8               \n"
                        "fmla   v9.4s, v5.4s, v0.s[0]       \n"
                        "ldr    d6, [%1], #8                \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]      \n"
                        "ins    v1.d[1], x21                \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]      \n"
                        "ldr    x26, [%1], #8               \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]      \n"
                        "ldr    d3, [%2], #8                \n"
                        "fmla   v16.4s, v4.4s, v1.s[0]      \n"
                        "ldr    x23, [%2], #8               \n"
                        "fmla   v18.4s, v4.4s, v1.s[1]      \n"
                        "ldr    d7, [%1], #8                \n"
                        "fmla   v20.4s, v4.4s, v1.s[2]      \n"
                        "ldr    x27, [%1], #8               \n"
                        "fmla   v22.4s, v4.4s, v1.s[3]      \n"
                        "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                        "fmla   v17.4s, v5.4s, v1.s[0]      \n"
                        "ldr    d0, [%2], #8                \n"
                        "fmla   v19.4s, v5.4s, v1.s[1]      \n"
                        "ins    v2.d[1], x22                \n"
                        "fmla   v21.4s, v5.4s, v1.s[2]      \n"
                        "ldr    x20, [%2], #8               \n"
                        "fmla   v23.4s, v5.4s, v1.s[3]      \n"
                        "fmla   v24.4s, v4.4s, v2.s[0]      \n"
                        "ldr    d1, [%2], #8                \n"
                        "fmla   v26.4s, v4.4s, v2.s[1]      \n"
                        "ins    v6.d[1], x26                \n"
                        "fmla   v28.4s, v4.4s, v2.s[2]      \n"
                        "ldr    x21, [%2], #8               \n"
                        "fmla   v30.4s, v4.4s, v2.s[3]      \n"
                        "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                        "fmla   v25.4s, v5.4s, v2.s[0]      \n"
                        "ldr    d4, [%1], #8                \n"
                        "fmla   v27.4s, v5.4s, v2.s[1]      \n"
                        "ins    v3.d[1], x23                \n"
                        "fmla   v29.4s, v5.4s, v2.s[2]      \n"
                        "ldr    x24, [%1], #8               \n"
                        "fmla   v31.4s, v5.4s, v2.s[3]      \n"
                        "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                        "ldr    d2, [%2], #8                \n"
                        "fmla   v10.4s, v6.4s, v3.s[1]      \n"
                        "ins    v7.d[1], x27                \n"
                        "fmla   v12.4s, v6.4s, v3.s[2]      \n"
                        "ldr    x22, [%2], #8               \n"
                        "fmla   v14.4s, v6.4s, v3.s[3]      \n"
                        "fmla   v9.4s, v7.4s, v3.s[0]       \n"
                        "ldr    d5, [%1], #8                \n"
                        "fmla   v11.4s, v7.4s, v3.s[1]      \n"
                        "ins    v0.d[1], x20                \n"
                        "fmla   v13.4s, v7.4s, v3.s[2]      \n"
                        "ldr    x25, [%1], #8               \n"
                        "fmla   v15.4s, v7.4s, v3.s[3]      \n"
                        "fmla   v16.4s, v6.4s, v0.s[0]      \n"
                        "ldr    d3, [%2], #8                \n"
                        "fmla   v18.4s, v6.4s, v0.s[1]      \n"
                        "ldr    x23, [%2], #8               \n"
                        "fmla   v20.4s, v6.4s, v0.s[2]      \n"
                        "fmla   v22.4s, v6.4s, v0.s[3]      \n"
                        "fmla   v17.4s, v7.4s, v0.s[0]      \n"
                        "fmla   v19.4s, v7.4s, v0.s[1]      \n"
                        "ins    v1.d[1], x21                \n"
                        "fmla   v21.4s, v7.4s, v0.s[2]      \n"
                        "fmla   v23.4s, v7.4s, v0.s[3]      \n"
                        "prfm   pldl1keep, [%2, #256]       \n" // NOTE PRELOAD
                        "fmla   v24.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v26.4s, v6.4s, v1.s[1]      \n"
                        "ins    v4.d[1], x24                \n"
                        "fmla   v28.4s, v6.4s, v1.s[2]      \n"
                        "ldr    d0, [%2], #8                \n"
                        "fmla   v30.4s, v6.4s, v1.s[3]      \n"
                        "ldr    x20, [%2], #8               \n"
                        "fmla   v25.4s, v7.4s, v1.s[0]      \n"
                        "ldr    d6, [%1], #8                \n"
                        "fmla   v27.4s, v7.4s, v1.s[1]      \n"
                        "ins    v2.d[1], x22                \n"
                        "fmla   v29.4s, v7.4s, v1.s[2]      \n"
                        "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                        "ldr    x26, [%1], #8               \n"
                        "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                        "ldr    d1, [%2], #8                \n"
                        "fmla   v10.4s, v4.4s, v2.s[1]      \n"
                        "ins    v5.d[1], x25                \n"
                        "fmla   v12.4s, v4.4s, v2.s[2]      \n"
                        "ldr    x21, [%2], #8               \n"
                        "fmla   v14.4s, v4.4s, v2.s[3]      \n"
                        "ldr    d7, [%1], #8                \n"
                        "fmla   v9.4s, v5.4s, v2.s[0]       \n"
                        "ldr    x27, [%1], #8               \n"
                        "fmla   v11.4s, v5.4s, v2.s[1]      \n"
                        "ins    v3.d[1], x23                \n"
                        "fmla   v13.4s, v5.4s, v2.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v2.s[3]      \n"
                        "fmla   v16.4s, v4.4s, v3.s[0]      \n"
                        "ldr    d2, [%2], #8                \n"
                        "fmla   v18.4s, v4.4s, v3.s[1]      \n"
                        "ldr    x22, [%2], #8               \n"
                        "fmla   v20.4s, v4.4s, v3.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v3.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v3.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v3.s[1]      \n"
                        "ins    v0.d[1], x20                \n"
                        "fmla   v21.4s, v5.4s, v3.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v3.s[3]      \n"
                        "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                        "ldr    d3, [%2], #8                \n"
                        "fmla   v26.4s, v4.4s, v0.s[1]      \n"
                        "ldr    x23, [%2], #8               \n"
                        "fmla   v28.4s, v4.4s, v0.s[2]      \n"
                        "ins    v6.d[1], x26                \n"
                        "fmla   v30.4s, v4.4s, v0.s[3]      \n"
                        "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                        "fmla   v25.4s, v5.4s, v0.s[0]      \n"
                        "ldr    d4, [%1], #8                \n"
                        "fmla   v27.4s, v5.4s, v0.s[1]      \n"
                        "ins    v1.d[1], x21                \n"
                        "fmla   v29.4s, v5.4s, v0.s[2]      \n"
                        "ldr    x24, [%1], #8               \n"
                        "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                        "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                        "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                        "ldr    d0, [%2], #8                \n"
                        "fmla   v10.4s, v6.4s, v1.s[1]      \n"
                        "ins    v7.d[1], x27                \n"
                        "fmla   v12.4s, v6.4s, v1.s[2]      \n"
                        "ldr    x20, [%2], #8               \n"
                        "fmla   v14.4s, v6.4s, v1.s[3]      \n"
                        "ldr    d5, [%1], #8                \n"
                        "fmla   v9.4s, v7.4s, v1.s[0]       \n"
                        "ldr    x25, [%1], #8               \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]      \n"
                        "ins    v2.d[1], x22                \n"
                        "fmla   v13.4s, v7.4s, v1.s[2]      \n"
                        "fmla   v15.4s, v7.4s, v1.s[3]      \n"
                        "fmla   v16.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v18.4s, v6.4s, v2.s[1]      \n"
                        "fmla   v20.4s, v6.4s, v2.s[2]      \n"
                        "fmla   v22.4s, v6.4s, v2.s[3]      \n"
                        "fmla   v17.4s, v7.4s, v2.s[0]      \n"
                        "fmla   v19.4s, v7.4s, v2.s[1]      \n"
                        "ins    v3.d[1], x23                \n"
                        "fmla   v21.4s, v7.4s, v2.s[2]      \n"
                        "fmla   v23.4s, v7.4s, v2.s[3]      \n"
                        "fmla   v24.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v26.4s, v6.4s, v3.s[1]      \n"
                        "fmla   v28.4s, v6.4s, v3.s[2]      \n"
                        "ins    v4.d[1], x24                \n"
                        "fmla   v30.4s, v6.4s, v3.s[3]      \n"
                        "fmla   v25.4s, v7.4s, v3.s[0]      \n"
                        "subs   w4, w4, #1                  \n"
                        "fmla   v27.4s, v7.4s, v3.s[1]      \n"
                        "fmla   v29.4s, v7.4s, v3.s[2]      \n"
                        "ins    v0.d[1], x20                \n"
                        "fmla   v31.4s, v7.4s, v3.s[3]      \n"
                        "bne    2b                          \n"

                        "sub    %1, %1, #32                 \n"
                        "sub    %2, %2, #16                 \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%2], #48 \n"
                        "ld1    {v4.4s, v5.4s}, [%1], #32   \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]      \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]      \n"
                        "fmla   v16.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v18.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v20.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v1.s[3]      \n"
                        "fmla   v24.4s, v4.4s, v2.s[0]      \n"
                        "fmla   v26.4s, v4.4s, v2.s[1]      \n"
                        "fmla   v28.4s, v4.4s, v2.s[2]      \n"
                        "fmla   v30.4s, v4.4s, v2.s[3]      \n"

                        "subs   w4, w4, #1                  \n"

                        "fmla   v9.4s, v5.4s, v0.s[0]       \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v1.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v1.s[1]      \n"
                        "fmla   v21.4s, v5.4s, v1.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v1.s[3]      \n"
                        "fmla   v25.4s, v5.4s, v2.s[0]      \n"
                        "fmla   v27.4s, v5.4s, v2.s[1]      \n"
                        "fmla   v29.4s, v5.4s, v2.s[2]      \n"
                        "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                        "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                else if (use_a53_a55_optimized_kernel && !cpu_support_arm_asimdhp())
                {
                    // a53
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                        "subs   %0, %0, #320                \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v8.16b, v8.16b, v8.16b      \n"
                        "eor    v9.16b, v9.16b, v9.16b      \n"
                        "eor    v10.16b, v10.16b, v10.16b   \n"
                        "eor    v11.16b, v11.16b, v11.16b   \n"
                        "eor    v12.16b, v12.16b, v12.16b   \n"
                        "eor    v13.16b, v13.16b, v13.16b   \n"
                        "eor    v14.16b, v14.16b, v14.16b   \n"
                        "eor    v15.16b, v15.16b, v15.16b   \n"
                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "eor    v17.16b, v17.16b, v17.16b   \n"
                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"
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

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v4.4s}, [%1], #16          \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v0.4s}, [%2], #16          \n"

                        "ldr    d1, [%2]                    \n"
                        "ldr    x21, [%2, #8]               \n"
                        "ldr    d2, [%2, #16]               \n"
                        "ldr    x22, [%2, #24]              \n"
                        "add    %2, %2, #32                 \n"

                        ".align 4                           \n"
                        "2:                                 \n"

                        "ldr    d5, [%1]                    \n"
                        "ins    v1.d[1], x21                \n"
                        "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                        "ldr    x25, [%1, #8]               \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]      \n"

                        "ldr    d6, [%1]                    \n"
                        "ins    v2.d[1], x22                \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]      \n"
                        "ldr    x26, [%1, #8]               \n"
                        "fmla   v16.4s, v4.4s, v1.s[0]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v18.4s, v4.4s, v1.s[1]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%1, #256]       \n" // NOTE PRELOAD
                        "fmla   v20.4s, v4.4s, v1.s[2]      \n"
                        "nop                                \n"
                        "fmla   v22.4s, v4.4s, v1.s[3]      \n"
                        "nop                                \n"
                        "fmla   v24.4s, v4.4s, v2.s[0]      \n"

                        "ldr    d3, [%2]                    \n"
                        "ins    v5.d[1], x25                \n"
                        "fmla   v26.4s, v4.4s, v2.s[1]      \n"
                        "ldr    x23, [%2, #8]               \n"
                        "fmla   v28.4s, v4.4s, v2.s[2]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v30.4s, v4.4s, v2.s[3]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%2, #384]       \n" // NOTE PRELOAD
                        "fmla   v9.4s, v5.4s, v0.s[0]       \n"
                        "nop                                \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]      \n"
                        "nop                                \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]      \n"

                        "nop                                \n"
                        "nop                                \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]      \n"
                        "nop                                \n"
                        "fmla   v17.4s, v5.4s, v1.s[0]      \n"
                        "nop                                \n"
                        "fmla   v19.4s, v5.4s, v1.s[1]      \n"

                        "ldr    d0, [%2]                    \n"
                        "ins    v6.d[1], x26                \n"
                        "fmla   v21.4s, v5.4s, v1.s[2]      \n"
                        "ldr    x20, [%2, #8]               \n"
                        "fmla   v23.4s, v5.4s, v1.s[3]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v25.4s, v5.4s, v2.s[0]      \n"

                        "ldr    d1, [%2]                    \n"
                        "ins    v3.d[1], x23                \n"
                        "fmla   v27.4s, v5.4s, v2.s[1]      \n"
                        "ldr    x21, [%2, #8]               \n"
                        "fmla   v29.4s, v5.4s, v2.s[2]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                        "ldr    d7, [%1]                    \n"
                        "ins    v0.d[1], x20                \n"
                        "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                        "ldr    x27, [%1, #8]               \n"
                        "fmla   v10.4s, v6.4s, v3.s[1]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v12.4s, v6.4s, v3.s[2]      \n"

                        "ldr    d4, [%1]                    \n"
                        "ins    v1.d[1], x21                \n"
                        "fmla   v14.4s, v6.4s, v3.s[3]      \n"
                        "ldr    x24, [%1, #8]               \n"
                        "fmla   v16.4s, v6.4s, v0.s[0]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v18.4s, v6.4s, v0.s[1]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%1, #256]       \n" // NOTE PRELOAD
                        "fmla   v20.4s, v6.4s, v0.s[2]      \n"
                        "nop                                \n"
                        "fmla   v22.4s, v6.4s, v0.s[3]      \n"
                        "nop                                \n"
                        "fmla   v24.4s, v6.4s, v1.s[0]      \n"

                        "ldr    d2, [%2]                    \n"
                        "ins    v7.d[1], x27                \n"
                        "fmla   v26.4s, v6.4s, v1.s[1]      \n"
                        "ldr    x22, [%2, #8]               \n"
                        "fmla   v28.4s, v6.4s, v1.s[2]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v30.4s, v6.4s, v1.s[3]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%2, #384]       \n" // NOTE PRELOAD
                        "fmla   v9.4s, v7.4s, v3.s[0]       \n"
                        "nop                                \n"
                        "fmla   v11.4s, v7.4s, v3.s[1]      \n"
                        "nop                                \n"
                        "fmla   v13.4s, v7.4s, v3.s[2]      \n"

                        "nop                                \n"
                        "nop                                \n"
                        "fmla   v15.4s, v7.4s, v3.s[3]      \n"
                        "nop                                \n"
                        "fmla   v17.4s, v7.4s, v0.s[0]      \n"
                        "nop                                \n"
                        "fmla   v19.4s, v7.4s, v0.s[1]      \n"

                        "ldr    d3, [%2]                    \n"
                        "ins    v4.d[1], x24                \n"
                        "fmla   v21.4s, v7.4s, v0.s[2]      \n"
                        "ldr    x23, [%2, #8]               \n"
                        "fmla   v23.4s, v7.4s, v0.s[3]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v25.4s, v7.4s, v1.s[0]      \n"

                        "ldr    d0, [%2]                    \n"
                        "ins    v2.d[1], x22                \n"
                        "fmla   v27.4s, v7.4s, v1.s[1]      \n"
                        "ldr    x20, [%2, #8]               \n"
                        "fmla   v29.4s, v7.4s, v1.s[2]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                        "ldr    d5, [%1]                    \n"
                        "ins    v3.d[1], x23                \n"
                        "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                        "ldr    x25, [%1, #8]               \n"
                        "fmla   v10.4s, v4.4s, v2.s[1]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v12.4s, v4.4s, v2.s[2]      \n"

                        "ldr    d6, [%1]                    \n"
                        "ins    v0.d[1], x20                \n"
                        "fmla   v14.4s, v4.4s, v2.s[3]      \n"
                        "ldr    x26, [%1, #8]               \n"
                        "fmla   v16.4s, v4.4s, v3.s[0]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v18.4s, v4.4s, v3.s[1]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%1, #256]       \n" // NOTE PRELOAD
                        "fmla   v20.4s, v4.4s, v3.s[2]      \n"
                        "nop                                \n"
                        "fmla   v22.4s, v4.4s, v3.s[3]      \n"
                        "nop                                \n"
                        "fmla   v24.4s, v4.4s, v0.s[0]      \n"

                        "ldr    d1, [%2]                    \n"
                        "ins    v5.d[1], x25                \n"
                        "fmla   v26.4s, v4.4s, v0.s[1]      \n"
                        "ldr    x21, [%2, #8]               \n"
                        "fmla   v28.4s, v4.4s, v0.s[2]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v30.4s, v4.4s, v0.s[3]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%2, #384]       \n" // NOTE PRELOAD
                        "fmla   v9.4s, v5.4s, v2.s[0]       \n"
                        "nop                                \n"
                        "fmla   v11.4s, v5.4s, v2.s[1]      \n"
                        "nop                                \n"
                        "fmla   v13.4s, v5.4s, v2.s[2]      \n"

                        "nop                                \n"
                        "nop                                \n"
                        "fmla   v15.4s, v5.4s, v2.s[3]      \n"
                        "nop                                \n"
                        "fmla   v17.4s, v5.4s, v3.s[0]      \n"
                        "nop                                \n"
                        "fmla   v19.4s, v5.4s, v3.s[1]      \n"

                        "ldr    d2, [%2]                    \n"
                        "ins    v6.d[1], x26                \n"
                        "fmla   v21.4s, v5.4s, v3.s[2]      \n"
                        "ldr    x22, [%2, #8]               \n"
                        "fmla   v23.4s, v5.4s, v3.s[3]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v25.4s, v5.4s, v0.s[0]      \n"

                        "ldr    d3, [%2]                    \n"
                        "ins    v1.d[1], x21                \n"
                        "fmla   v27.4s, v5.4s, v0.s[1]      \n"
                        "ldr    x23, [%2, #8]               \n"
                        "fmla   v29.4s, v5.4s, v0.s[2]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                        "ldr    d7, [%1]                    \n"
                        "ins    v2.d[1], x22                \n"
                        "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                        "ldr    x27, [%1, #8]               \n"
                        "fmla   v10.4s, v6.4s, v1.s[1]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v12.4s, v6.4s, v1.s[2]      \n"

                        "ldr    d4, [%1]                    \n"
                        "ins    v3.d[1], x23                \n"
                        "fmla   v14.4s, v6.4s, v1.s[3]      \n"
                        "ldr    x24, [%1, #8]               \n"
                        "fmla   v16.4s, v6.4s, v2.s[0]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v18.4s, v6.4s, v2.s[1]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%1, #256]       \n" // NOTE PRELOAD
                        "fmla   v20.4s, v6.4s, v2.s[2]      \n"
                        "nop                                \n"
                        "fmla   v22.4s, v6.4s, v2.s[3]      \n"
                        "nop                                \n"
                        "fmla   v24.4s, v6.4s, v3.s[0]      \n"

                        "ldr    d0, [%2]                    \n"
                        "ins    v7.d[1], x27                \n"
                        "fmla   v26.4s, v6.4s, v3.s[1]      \n"
                        "ldr    x20, [%2, #8]               \n"
                        "fmla   v28.4s, v6.4s, v3.s[2]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v30.4s, v6.4s, v3.s[3]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%2, #384]       \n" // NOTE PRELOAD
                        "fmla   v9.4s, v7.4s, v1.s[0]       \n"
                        "nop                                \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]      \n"
                        "nop                                \n"
                        "fmla   v13.4s, v7.4s, v1.s[2]      \n"

                        "nop                                \n"
                        "nop                                \n"
                        "fmla   v15.4s, v7.4s, v1.s[3]      \n"
                        "subs   w4, w4, #1                  \n"
                        "fmla   v17.4s, v7.4s, v2.s[0]      \n"
                        "nop                                \n"
                        "fmla   v19.4s, v7.4s, v2.s[1]      \n"

                        "ldr    d1, [%2]                    \n"
                        "ins    v4.d[1], x24                \n"
                        "fmla   v21.4s, v7.4s, v2.s[2]      \n"
                        "ldr    x21, [%2, #8]               \n"
                        "fmla   v23.4s, v7.4s, v2.s[3]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v25.4s, v7.4s, v3.s[0]      \n"

                        "ldr    d2, [%2]                    \n"
                        "ins    v0.d[1], x20                \n"
                        "fmla   v27.4s, v7.4s, v3.s[1]      \n"
                        "ldr    x22, [%2, #8]               \n"
                        "fmla   v29.4s, v7.4s, v3.s[2]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                        "bne    2b                          \n"

                        "sub    %1, %1, #16                 \n"
                        "sub    %2, %2, #48                 \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%2], #48 \n"
                        "ld1    {v4.4s, v5.4s}, [%1], #32   \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]      \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]      \n"
                        "fmla   v16.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v18.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v20.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v1.s[3]      \n"
                        "fmla   v24.4s, v4.4s, v2.s[0]      \n"
                        "fmla   v26.4s, v4.4s, v2.s[1]      \n"
                        "fmla   v28.4s, v4.4s, v2.s[2]      \n"
                        "fmla   v30.4s, v4.4s, v2.s[3]      \n"

                        "subs   w4, w4, #1                  \n"

                        "fmla   v9.4s, v5.4s, v0.s[0]       \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v1.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v1.s[1]      \n"
                        "fmla   v21.4s, v5.4s, v1.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v1.s[3]      \n"
                        "fmla   v25.4s, v5.4s, v2.s[0]      \n"
                        "fmla   v27.4s, v5.4s, v2.s[1]      \n"
                        "fmla   v29.4s, v5.4s, v2.s[2]      \n"
                        "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                        "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                else
                {
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                        "subs   %0, %0, #320                \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v8.16b, v8.16b, v8.16b      \n"
                        "eor    v9.16b, v9.16b, v9.16b      \n"
                        "eor    v10.16b, v10.16b, v10.16b   \n"
                        "eor    v11.16b, v11.16b, v11.16b   \n"
                        "eor    v12.16b, v12.16b, v12.16b   \n"
                        "eor    v13.16b, v13.16b, v13.16b   \n"
                        "eor    v14.16b, v14.16b, v14.16b   \n"
                        "eor    v15.16b, v15.16b, v15.16b   \n"
                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "eor    v17.16b, v17.16b, v17.16b   \n"
                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"
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
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]      \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]      \n"
                        "fmla   v9.4s, v5.4s, v0.s[0]       \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]      \n"

                        "fmla   v16.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v18.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v20.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v1.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v1.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v1.s[1]      \n"
                        "fmla   v21.4s, v5.4s, v1.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v1.s[3]      \n"

                        "fmla   v24.4s, v4.4s, v2.s[0]      \n"
                        "fmla   v26.4s, v4.4s, v2.s[1]      \n"
                        "fmla   v28.4s, v4.4s, v2.s[2]      \n"
                        "fmla   v30.4s, v4.4s, v2.s[3]      \n"
                        "fmla   v25.4s, v5.4s, v2.s[0]      \n"
                        "fmla   v27.4s, v5.4s, v2.s[1]      \n"
                        "fmla   v29.4s, v5.4s, v2.s[2]      \n"
                        "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                        "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                        "fmla   v10.4s, v6.4s, v3.s[1]      \n"
                        "fmla   v12.4s, v6.4s, v3.s[2]      \n"
                        "fmla   v14.4s, v6.4s, v3.s[3]      \n"
                        "fmla   v9.4s, v7.4s, v3.s[0]       \n"
                        "fmla   v11.4s, v7.4s, v3.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v3.s[2]      \n"
                        "fmla   v15.4s, v7.4s, v3.s[3]      \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                        "fmla   v16.4s, v6.4s, v0.s[0]      \n"
                        "fmla   v18.4s, v6.4s, v0.s[1]      \n"
                        "fmla   v20.4s, v6.4s, v0.s[2]      \n"
                        "fmla   v22.4s, v6.4s, v0.s[3]      \n"
                        "fmla   v17.4s, v7.4s, v0.s[0]      \n"
                        "fmla   v19.4s, v7.4s, v0.s[1]      \n"
                        "fmla   v21.4s, v7.4s, v0.s[2]      \n"
                        "fmla   v23.4s, v7.4s, v0.s[3]      \n"

                        "fmla   v24.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v26.4s, v6.4s, v1.s[1]      \n"
                        "fmla   v28.4s, v6.4s, v1.s[2]      \n"
                        "fmla   v30.4s, v6.4s, v1.s[3]      \n"
                        "fmla   v25.4s, v7.4s, v1.s[0]      \n"
                        "fmla   v27.4s, v7.4s, v1.s[1]      \n"
                        "fmla   v29.4s, v7.4s, v1.s[2]      \n"
                        "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"

                        "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                        "fmla   v10.4s, v4.4s, v2.s[1]      \n"
                        "fmla   v12.4s, v4.4s, v2.s[2]      \n"
                        "fmla   v14.4s, v4.4s, v2.s[3]      \n"
                        "fmla   v9.4s, v5.4s, v2.s[0]       \n"
                        "fmla   v11.4s, v5.4s, v2.s[1]      \n"
                        "fmla   v13.4s, v5.4s, v2.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v2.s[3]      \n"

                        "fmla   v16.4s, v4.4s, v3.s[0]      \n"
                        "fmla   v18.4s, v4.4s, v3.s[1]      \n"
                        "fmla   v20.4s, v4.4s, v3.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v3.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v3.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v3.s[1]      \n"
                        "fmla   v21.4s, v5.4s, v3.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v3.s[3]      \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                        "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                        "fmla   v26.4s, v4.4s, v0.s[1]      \n"
                        "fmla   v28.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v30.4s, v4.4s, v0.s[3]      \n"
                        "fmla   v25.4s, v5.4s, v0.s[0]      \n"
                        "fmla   v27.4s, v5.4s, v0.s[1]      \n"
                        "fmla   v29.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                        "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                        "fmla   v10.4s, v6.4s, v1.s[1]      \n"
                        "fmla   v12.4s, v6.4s, v1.s[2]      \n"
                        "fmla   v14.4s, v6.4s, v1.s[3]      \n"
                        "fmla   v9.4s, v7.4s, v1.s[0]       \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v1.s[2]      \n"
                        "fmla   v15.4s, v7.4s, v1.s[3]      \n"

                        "fmla   v16.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v18.4s, v6.4s, v2.s[1]      \n"
                        "fmla   v20.4s, v6.4s, v2.s[2]      \n"
                        "fmla   v22.4s, v6.4s, v2.s[3]      \n"
                        "fmla   v17.4s, v7.4s, v2.s[0]      \n"
                        "fmla   v19.4s, v7.4s, v2.s[1]      \n"
                        "fmla   v21.4s, v7.4s, v2.s[2]      \n"
                        "fmla   v23.4s, v7.4s, v2.s[3]      \n"

                        "subs   w4, w4, #1                  \n"

                        "fmla   v24.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v26.4s, v6.4s, v3.s[1]      \n"
                        "fmla   v28.4s, v6.4s, v3.s[2]      \n"
                        "fmla   v30.4s, v6.4s, v3.s[3]      \n"
                        "fmla   v25.4s, v7.4s, v3.s[0]      \n"
                        "fmla   v27.4s, v7.4s, v3.s[1]      \n"
                        "fmla   v29.4s, v7.4s, v3.s[2]      \n"
                        "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                        "bne    2b                          \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%2], #48 \n"
                        "ld1    {v4.4s, v5.4s}, [%1], #32   \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]      \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]      \n"
                        "fmla   v16.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v18.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v20.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v1.s[3]      \n"
                        "fmla   v24.4s, v4.4s, v2.s[0]      \n"
                        "fmla   v26.4s, v4.4s, v2.s[1]      \n"
                        "fmla   v28.4s, v4.4s, v2.s[2]      \n"
                        "fmla   v30.4s, v4.4s, v2.s[3]      \n"

                        "subs   w4, w4, #1                  \n"

                        "fmla   v9.4s, v5.4s, v0.s[0]       \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v1.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v1.s[1]      \n"
                        "fmla   v21.4s, v5.4s, v1.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v1.s[3]      \n"
                        "fmla   v25.4s, v5.4s, v2.s[0]      \n"
                        "fmla   v27.4s, v5.4s, v2.s[1]      \n"
                        "fmla   v29.4s, v5.4s, v2.s[2]      \n"
                        "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                        "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
#else  // NCNN_GNU_INLINE_ASM
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
                for (; kk < max_kk; kk++)
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
                outptr += 8 * 12;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                if (use_a53_a55_optimized_kernel && cpu_support_arm_asimdhp())
                {
                    // a55
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                        "subs   %0, %0, #192                \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "eor    v17.16b, v17.16b, v17.16b   \n"
                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"
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
                        "ld1    {v8.4s}, [%1], #16          \n"
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s}, [%2], #16          \n"

                        "ldr    d1, [%2], #8                \n"
                        "ldr    x21, [%2], #8               \n"

                        ".align 4                           \n"
                        "2:                                 \n"
                        "ldr    d9, [%1], #8                \n"
                        "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                        "ldr    x25, [%1], #8               \n"
                        "fmla   v18.4s, v8.4s, v0.s[1]      \n"
                        "ins    v1.d[1], x21                \n"
                        "fmla   v20.4s, v8.4s, v0.s[2]      \n"
                        "ldr    d10, [%1], #8               \n"
                        "fmla   v22.4s, v8.4s, v0.s[3]      \n"
                        "ldr    x26, [%1], #8               \n"
                        "fmla   v24.4s, v8.4s, v1.s[0]      \n"
                        "ldr    d2, [%2], #8                \n"
                        "fmla   v26.4s, v8.4s, v1.s[1]      \n"
                        "ins    v9.d[1], x25                \n"
                        "fmla   v28.4s, v8.4s, v1.s[2]      \n"
                        "ldr    x22, [%2], #8               \n"
                        "fmla   v30.4s, v8.4s, v1.s[3]      \n"
                        "ldr    d3, [%2], #8                \n"
                        "fmla   v17.4s, v9.4s, v0.s[0]      \n"
                        "ldr    x23, [%2], #8               \n"
                        "fmla   v19.4s, v9.4s, v0.s[1]      \n"
                        "ins    v10.d[1], x26               \n"
                        "fmla   v21.4s, v9.4s, v0.s[2]      \n"
                        "ldr    d11, [%1], #8               \n"
                        "fmla   v23.4s, v9.4s, v0.s[3]      \n"
                        "ldr    x27, [%1], #8               \n"
                        "fmla   v25.4s, v9.4s, v1.s[0]      \n"
                        "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                        "fmla   v27.4s, v9.4s, v1.s[1]      \n"
                        "ins    v2.d[1], x22                \n"
                        "fmla   v29.4s, v9.4s, v1.s[2]      \n"
                        "ldr    d12, [%1], #8               \n"
                        "fmla   v31.4s, v9.4s, v1.s[3]      \n"
                        "ldr    x24, [%1], #8               \n"
                        "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                        "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                        "fmla   v18.4s, v10.4s, v2.s[1]     \n"
                        "ins    v3.d[1], x23                \n"
                        "fmla   v20.4s, v10.4s, v2.s[2]     \n"
                        "ldr    d4, [%2], #8                \n"
                        "fmla   v22.4s, v10.4s, v2.s[3]     \n"
                        "ldr    x20, [%2], #8               \n"
                        "fmla   v24.4s, v10.4s, v3.s[0]     \n"
                        "ldr    d5, [%2], #8                \n"
                        "fmla   v26.4s, v10.4s, v3.s[1]     \n"
                        "ins    v11.d[1], x27               \n"
                        "fmla   v28.4s, v10.4s, v3.s[2]     \n"
                        "ldr    x21, [%2], #8               \n"
                        "fmla   v30.4s, v10.4s, v3.s[3]     \n"
                        "ldr    d13, [%1], #8               \n"
                        "fmla   v17.4s, v11.4s, v2.s[0]     \n"
                        "ldr    x25, [%1], #8               \n"
                        "fmla   v19.4s, v11.4s, v2.s[1]     \n"
                        "ins    v12.d[1], x24               \n"
                        "fmla   v21.4s, v11.4s, v2.s[2]     \n"
                        "ldr    d14, [%1], #8               \n"
                        "fmla   v23.4s, v11.4s, v2.s[3]     \n"
                        "ldr    x26, [%1], #8               \n"
                        "fmla   v25.4s, v11.4s, v3.s[0]     \n"
                        "ldr    d6, [%2], #8                \n"
                        "fmla   v27.4s, v11.4s, v3.s[1]     \n"
                        "ins    v4.d[1], x20                \n"
                        "fmla   v29.4s, v11.4s, v3.s[2]     \n"
                        "ldr    x22, [%2], #8               \n"
                        "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                        "ldr    d7, [%2], #8                \n"
                        "fmla   v16.4s, v12.4s, v4.s[0]     \n"
                        "ldr    x23, [%2], #8               \n"
                        "fmla   v18.4s, v12.4s, v4.s[1]     \n"
                        "ins    v5.d[1], x21                \n"
                        "fmla   v20.4s, v12.4s, v4.s[2]     \n"
                        "ldr    d15, [%1], #8               \n"
                        "fmla   v22.4s, v12.4s, v4.s[3]     \n"
                        "ldr    x27, [%1], #8               \n"
                        "fmla   v24.4s, v12.4s, v5.s[0]     \n"
                        "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                        "fmla   v26.4s, v12.4s, v5.s[1]     \n"
                        "ins    v13.d[1], x25               \n"
                        "fmla   v28.4s, v12.4s, v5.s[2]     \n"
                        "ldr    d8, [%1], #8                \n"
                        "fmla   v30.4s, v12.4s, v5.s[3]     \n"
                        "ldr    x24, [%1], #8               \n"
                        "fmla   v17.4s, v13.4s, v4.s[0]     \n"
                        "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                        "fmla   v19.4s, v13.4s, v4.s[1]     \n"
                        "ins    v14.d[1], x26               \n"
                        "fmla   v21.4s, v13.4s, v4.s[2]     \n"
                        "ldr    d0, [%2], #8                \n"
                        "fmla   v23.4s, v13.4s, v4.s[3]     \n"
                        "ldr    x20, [%2], #8               \n"
                        "fmla   v25.4s, v13.4s, v5.s[0]     \n"
                        "ldr    d1, [%2], #8                \n"
                        "fmla   v27.4s, v13.4s, v5.s[1]     \n"
                        "ins    v6.d[1], x22                \n"
                        "fmla   v29.4s, v13.4s, v5.s[2]     \n"
                        "ldr    x21, [%2], #8               \n"
                        "fmla   v31.4s, v13.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v14.4s, v6.s[1]     \n"
                        "ins    v7.d[1], x23                \n"
                        "fmla   v20.4s, v14.4s, v6.s[2]     \n"
                        "fmla   v22.4s, v14.4s, v6.s[3]     \n"
                        "fmla   v24.4s, v14.4s, v7.s[0]     \n"
                        "fmla   v26.4s, v14.4s, v7.s[1]     \n"
                        "ins    v15.d[1], x27               \n"
                        "fmla   v28.4s, v14.4s, v7.s[2]     \n"
                        "fmla   v30.4s, v14.4s, v7.s[3]     \n"
                        "fmla   v17.4s, v15.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v15.4s, v6.s[1]     \n"
                        "ins    v8.d[1], x24                \n"
                        "fmla   v21.4s, v15.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v15.4s, v6.s[3]     \n"
                        "fmla   v25.4s, v15.4s, v7.s[0]     \n"
                        "subs   w4, w4, #1                  \n"
                        "fmla   v27.4s, v15.4s, v7.s[1]     \n"
                        "fmla   v29.4s, v15.4s, v7.s[2]     \n"
                        "ins    v0.d[1], x20                \n"
                        "fmla   v31.4s, v15.4s, v7.s[3]     \n"
                        "bne    2b                          \n"

                        "sub    %1, %1, #16                 \n"
                        "sub    %2, %2, #32                 \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
                        "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                        "ld1    {v4.4s, v5.4s}, [%1], #32   \n"
                        "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                        "fmla   v18.4s, v4.4s, v0.s[1]      \n"
                        "fmla   v20.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v0.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v0.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v0.s[1]      \n"
                        "fmla   v21.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                        "subs   w4, w4, #1                  \n"
                        "fmla   v24.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v26.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v28.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v30.4s, v4.4s, v1.s[3]      \n"
                        "fmla   v25.4s, v5.4s, v1.s[0]      \n"
                        "fmla   v27.4s, v5.4s, v1.s[1]      \n"
                        "fmla   v29.4s, v5.4s, v1.s[2]      \n"
                        "fmla   v31.4s, v5.4s, v1.s[3]      \n"
                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                else if (use_a53_a55_optimized_kernel && !cpu_support_arm_asimdhp())
                {
                    // a53
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                        "subs   %0, %0, #192                \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "eor    v17.16b, v17.16b, v17.16b   \n"
                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"
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
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ldr    d0, [%2]                    \n"
                        "ldr    x20, [%2, #8]               \n"
                        "ins    v0.d[1], x20                \n"
                        "add    %2, %2, #16                 \n"

                        "ldr    d8, [%1]                    \n"
                        "ldr    x24, [%1, #8]               \n"
                        "ins    v8.d[1], x24                \n"
                        "add    %1, %1, #16                 \n"

                        "ldr    d1, [%2]                    \n"
                        "ldr    x21, [%2, #8]               \n"
                        "add    %2, %2, #16                 \n"

                        "ldr    d9, [%1]                    \n"
                        "ldr    x25, [%1, #8]               \n"
                        "add    %1, %1, #16                 \n"

                        ".align 4                           \n"
                        "2:                                 \n"

                        "ldr    d2, [%2]                    \n"
                        "ins    v1.d[1], x21                \n"
                        "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                        "ldr    x22, [%2, #8]               \n"
                        "fmla   v18.4s, v8.4s, v0.s[1]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v20.4s, v8.4s, v0.s[2]      \n"

                        "ldr    d10, [%1]                   \n"
                        "ins    v9.d[1], x25                \n"
                        "fmla   v22.4s, v8.4s, v0.s[3]      \n"
                        "ldr    x26, [%1, #8]               \n"
                        "fmla   v24.4s, v8.4s, v1.s[0]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v26.4s, v8.4s, v1.s[1]      \n"

                        "ldr    d3, [%2]                    \n"
                        "ins    v2.d[1], x22                \n"
                        "fmla   v28.4s, v8.4s, v1.s[2]      \n"
                        "ldr    x23, [%2, #8]               \n"
                        "fmla   v30.4s, v8.4s, v1.s[3]      \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v17.4s, v9.4s, v0.s[0]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                        "fmla   v19.4s, v9.4s, v0.s[1]      \n"
                        "nop                                \n"
                        "fmla   v21.4s, v9.4s, v0.s[2]      \n"
                        "nop                                \n"
                        "fmla   v23.4s, v9.4s, v0.s[3]      \n"

                        "ldr    d11, [%1]                   \n"
                        "ins    v10.d[1], x26               \n"
                        "fmla   v25.4s, v9.4s, v1.s[0]      \n"
                        "ldr    x27, [%1, #8]               \n"
                        "fmla   v27.4s, v9.4s, v1.s[1]      \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v29.4s, v9.4s, v1.s[2]      \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                        "fmla   v31.4s, v9.4s, v1.s[3]      \n"
                        "nop                                \n"
                        "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                        "nop                                \n"
                        "fmla   v18.4s, v10.4s, v2.s[1]     \n"

                        "ldr    d4, [%2]                    \n"
                        "ins    v3.d[1], x23                \n"
                        "fmla   v20.4s, v10.4s, v2.s[2]     \n"
                        "ldr    x20, [%2, #8]               \n"
                        "fmla   v22.4s, v10.4s, v2.s[3]     \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v24.4s, v10.4s, v3.s[0]     \n"

                        "ldr    d12, [%1]                   \n"
                        "ins    v11.d[1], x27               \n"
                        "fmla   v26.4s, v10.4s, v3.s[1]     \n"
                        "ldr    x24, [%1, #8]               \n"
                        "fmla   v28.4s, v10.4s, v3.s[2]     \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v30.4s, v10.4s, v3.s[3]     \n"

                        "ldr    d5, [%2]                    \n"
                        "ins    v4.d[1], x20                \n"
                        "fmla   v17.4s, v11.4s, v2.s[0]     \n"
                        "ldr    x21, [%2, #8]               \n"
                        "fmla   v19.4s, v11.4s, v2.s[1]     \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v21.4s, v11.4s, v2.s[2]     \n"

                        "ldr    d13, [%1]                   \n"
                        "ins    v12.d[1], x24               \n"
                        "fmla   v23.4s, v11.4s, v2.s[3]     \n"
                        "ldr    x25, [%1, #8]               \n"
                        "fmla   v25.4s, v11.4s, v3.s[0]     \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v27.4s, v11.4s, v3.s[1]     \n"

                        "ldr    d6, [%2]                    \n"
                        "ins    v5.d[1], x21                \n"
                        "fmla   v29.4s, v11.4s, v3.s[2]     \n"
                        "ldr    x22, [%2, #8]               \n"
                        "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v16.4s, v12.4s, v4.s[0]     \n"

                        "ldr    d14, [%1]                   \n"
                        "ins    v13.d[1], x25               \n"
                        "fmla   v18.4s, v12.4s, v4.s[1]     \n"
                        "ldr    x26, [%1, #8]               \n"
                        "fmla   v20.4s, v12.4s, v4.s[2]     \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v22.4s, v12.4s, v4.s[3]     \n"

                        "ldr    d7, [%2]                    \n"
                        "ins    v6.d[1], x22                \n"
                        "fmla   v24.4s, v12.4s, v5.s[0]     \n"
                        "ldr    x23, [%2, #8]               \n"
                        "fmla   v26.4s, v12.4s, v5.s[1]     \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v28.4s, v12.4s, v5.s[2]     \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                        "fmla   v30.4s, v12.4s, v5.s[3]     \n"
                        "nop                                \n"
                        "fmla   v17.4s, v13.4s, v4.s[0]     \n"
                        "nop                                \n"
                        "fmla   v19.4s, v13.4s, v4.s[1]     \n"

                        "ldr    d15, [%1]                   \n"
                        "ins    v14.d[1], x26               \n"
                        "fmla   v21.4s, v13.4s, v4.s[2]     \n"
                        "ldr    x27, [%1, #8]               \n"
                        "fmla   v23.4s, v13.4s, v4.s[3]     \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v25.4s, v13.4s, v5.s[0]     \n"

                        "nop                                \n"
                        "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                        "fmla   v27.4s, v13.4s, v5.s[1]     \n"
                        "nop                                \n"
                        "fmla   v29.4s, v13.4s, v5.s[2]     \n"
                        "nop                                \n"
                        "fmla   v31.4s, v13.4s, v5.s[3]     \n"

                        "ldr    d0, [%2]                    \n"
                        "ins    v7.d[1], x23                \n"
                        "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                        "ldr    x20, [%2, #8]               \n"
                        "fmla   v18.4s, v14.4s, v6.s[1]     \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v20.4s, v14.4s, v6.s[2]     \n"

                        "ldr    d8, [%1]                    \n"
                        "ins    v15.d[1], x27               \n"
                        "fmla   v22.4s, v14.4s, v6.s[3]     \n"
                        "ldr    x24, [%1, #8]               \n"
                        "fmla   v24.4s, v14.4s, v7.s[0]     \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v26.4s, v14.4s, v7.s[1]     \n"

                        "ldr    d1, [%2]                    \n"
                        "ins    v0.d[1], x20                \n"
                        "fmla   v28.4s, v14.4s, v7.s[2]     \n"
                        "ldr    x21, [%2, #8]               \n"
                        "fmla   v30.4s, v14.4s, v7.s[3]     \n"
                        "add    %2, %2, #16                 \n"
                        "fmla   v17.4s, v15.4s, v6.s[0]     \n"

                        "ldr    d9, [%1]                    \n"
                        "ins    v8.d[1], x24                \n"
                        "fmla   v19.4s, v15.4s, v6.s[1]     \n"
                        "ldr    x25, [%1, #8]               \n"
                        "fmla   v21.4s, v15.4s, v6.s[2]     \n"
                        "add    %1, %1, #16                 \n"
                        "fmla   v23.4s, v15.4s, v6.s[3]     \n"

                        "nop                                \n"
                        "nop                                \n"
                        "fmla   v25.4s, v15.4s, v7.s[0]     \n"
                        "subs   w4, w4, #1                  \n"
                        "fmla   v27.4s, v15.4s, v7.s[1]     \n"
                        "nop                                \n"
                        "fmla   v29.4s, v15.4s, v7.s[2]     \n"

                        "nop                                \n"
                        "nop                                \n"
                        "fmla   v31.4s, v15.4s, v7.s[3]     \n"
                        "nop                                \n"
                        "nop                                \n"
                        "nop                                \n"
                        "nop                                \n"

                        "bne    2b                          \n"

                        "sub    %1, %1, #32                 \n"
                        "sub    %2, %2, #32                 \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
                        "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                        "ld1    {v4.4s, v5.4s}, [%1], #32   \n"
                        "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                        "fmla   v18.4s, v4.4s, v0.s[1]      \n"
                        "fmla   v20.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v0.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v0.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v0.s[1]      \n"
                        "fmla   v21.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                        "subs   w4, w4, #1                  \n"
                        "fmla   v24.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v26.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v28.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v30.4s, v4.4s, v1.s[3]      \n"
                        "fmla   v25.4s, v5.4s, v1.s[0]      \n"
                        "fmla   v27.4s, v5.4s, v1.s[1]      \n"
                        "fmla   v29.4s, v5.4s, v1.s[2]      \n"
                        "fmla   v31.4s, v5.4s, v1.s[3]      \n"
                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                else
                {
                    asm volatile(
                        "cbz    %w7, 0f                     \n"

                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                        "subs   %0, %0, #192                \n"
                        "b      1f                          \n"

                        "0:                                 \n"
                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "eor    v17.16b, v17.16b, v17.16b   \n"
                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"
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
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"
                        "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v18.4s, v8.4s, v0.s[1]      \n"
                        "fmla   v20.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v22.4s, v8.4s, v0.s[3]      \n"
                        "fmla   v24.4s, v8.4s, v1.s[0]      \n"
                        "fmla   v26.4s, v8.4s, v1.s[1]      \n"
                        "fmla   v28.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v30.4s, v8.4s, v1.s[3]      \n"
                        "fmla   v17.4s, v9.4s, v0.s[0]      \n"
                        "fmla   v19.4s, v9.4s, v0.s[1]      \n"
                        "fmla   v21.4s, v9.4s, v0.s[2]      \n"
                        "fmla   v23.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v25.4s, v9.4s, v1.s[0]      \n"
                        "fmla   v27.4s, v9.4s, v1.s[1]      \n"
                        "fmla   v29.4s, v9.4s, v1.s[2]      \n"
                        "fmla   v31.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v10.4s, v2.s[1]     \n"
                        "fmla   v20.4s, v10.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v10.4s, v2.s[3]     \n"
                        "fmla   v24.4s, v10.4s, v3.s[0]     \n"
                        "fmla   v26.4s, v10.4s, v3.s[1]     \n"
                        "fmla   v28.4s, v10.4s, v3.s[2]     \n"
                        "fmla   v30.4s, v10.4s, v3.s[3]     \n"
                        "fmla   v17.4s, v11.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v11.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v11.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v11.4s, v2.s[3]     \n"
                        "fmla   v25.4s, v11.4s, v3.s[0]     \n"
                        "fmla   v27.4s, v11.4s, v3.s[1]     \n"
                        "fmla   v29.4s, v11.4s, v3.s[2]     \n"
                        "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n"
                        "fmla   v16.4s, v12.4s, v4.s[0]     \n"
                        "fmla   v18.4s, v12.4s, v4.s[1]     \n"
                        "fmla   v20.4s, v12.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v12.4s, v4.s[3]     \n"
                        "fmla   v24.4s, v12.4s, v5.s[0]     \n"
                        "fmla   v26.4s, v12.4s, v5.s[1]     \n"
                        "fmla   v28.4s, v12.4s, v5.s[2]     \n"
                        "fmla   v30.4s, v12.4s, v5.s[3]     \n"
                        "fmla   v17.4s, v13.4s, v4.s[0]     \n"
                        "fmla   v19.4s, v13.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v13.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v13.4s, v4.s[3]     \n"
                        "fmla   v25.4s, v13.4s, v5.s[0]     \n"
                        "fmla   v27.4s, v13.4s, v5.s[1]     \n"
                        "fmla   v29.4s, v13.4s, v5.s[2]     \n"
                        "fmla   v31.4s, v13.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v14.4s, v6.s[1]     \n"
                        "fmla   v20.4s, v14.4s, v6.s[2]     \n"
                        "fmla   v22.4s, v14.4s, v6.s[3]     \n"
                        "fmla   v24.4s, v14.4s, v7.s[0]     \n"
                        "fmla   v26.4s, v14.4s, v7.s[1]     \n"
                        "fmla   v28.4s, v14.4s, v7.s[2]     \n"
                        "fmla   v30.4s, v14.4s, v7.s[3]     \n"
                        "subs   w4, w4, #1                  \n"
                        "fmla   v17.4s, v15.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v15.4s, v6.s[1]     \n"
                        "fmla   v21.4s, v15.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v15.4s, v6.s[3]     \n"
                        "fmla   v25.4s, v15.4s, v7.s[0]     \n"
                        "fmla   v27.4s, v15.4s, v7.s[1]     \n"
                        "fmla   v29.4s, v15.4s, v7.s[2]     \n"
                        "fmla   v31.4s, v15.4s, v7.s[3]     \n"
                        "bne    2b                          \n"

                        "3:                                 \n"
                        "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                        "cmp    w4, #0                      \n"
                        "beq    5f                          \n"

                        "4:                                 \n"
                        "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                        "ld1    {v4.4s, v5.4s}, [%1], #32   \n"
                        "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                        "fmla   v18.4s, v4.4s, v0.s[1]      \n"
                        "fmla   v20.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v22.4s, v4.4s, v0.s[3]      \n"
                        "fmla   v17.4s, v5.4s, v0.s[0]      \n"
                        "fmla   v19.4s, v5.4s, v0.s[1]      \n"
                        "fmla   v21.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                        "subs   w4, w4, #1                  \n"
                        "fmla   v24.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v26.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v28.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v30.4s, v4.4s, v1.s[3]      \n"
                        "fmla   v25.4s, v5.4s, v1.s[0]      \n"
                        "fmla   v27.4s, v5.4s, v1.s[1]      \n"
                        "fmla   v29.4s, v5.4s, v1.s[2]      \n"
                        "fmla   v31.4s, v5.4s, v1.s[3]      \n"
                        "bne    4b                          \n"

                        "5:                                 \n"
                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(pA),     // %1
                        "=r"(pB)      // %2
                        : "0"(outptr),
                        "1"(pA),
                        "2"(pB),
                        "r"(max_kk), // %6
                        "r"(k)       // %7
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
#else  // NCNN_GNU_INLINE_ASM
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
                for (; kk < max_kk; kk++)
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
                outptr += 8 * 8;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
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
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v26.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v28.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v30.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v25.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v27.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v26.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v28.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v30.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v25.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v27.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v29.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                    "fmla   v24.4s, v8.4s, v2.s[0]      \n"
                    "fmla   v26.4s, v8.4s, v2.s[1]      \n"
                    "fmla   v28.4s, v8.4s, v2.s[2]      \n"
                    "fmla   v30.4s, v8.4s, v2.s[3]      \n"
                    "fmla   v25.4s, v9.4s, v2.s[0]      \n"
                    "fmla   v27.4s, v9.4s, v2.s[1]      \n"
                    "fmla   v29.4s, v9.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v9.4s, v2.s[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v24.4s, v10.4s, v3.s[0]     \n"
                    "fmla   v26.4s, v10.4s, v3.s[1]     \n"
                    "fmla   v28.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v30.4s, v10.4s, v3.s[3]     \n"
                    "fmla   v25.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v27.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v29.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "bne    2b                          \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1    {v0.4s}, [%2], #16          \n"
                    "ld1    {v4.4s, v5.4s}, [%1], #32   \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v26.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v28.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v30.4s, v4.4s, v0.s[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v25.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v27.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
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
                    _sum01 = vld1q_f32(outptr + 4 * 1);
                    _sum10 = vld1q_f32(outptr + 4 * 2);
                    _sum11 = vld1q_f32(outptr + 4 * 3);
                    _sum20 = vld1q_f32(outptr + 4 * 4);
                    _sum21 = vld1q_f32(outptr + 4 * 5);
                    _sum30 = vld1q_f32(outptr + 4 * 6);
                    _sum31 = vld1q_f32(outptr + 4 * 7);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                outptr += 8 * 4;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
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
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                    "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v30.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v28.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v30.4s, v6.4s, v0.s[3]      \n"
                    "fmla   v29.4s, v7.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v0.s[3]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                    "fmla   v28.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v30.4s, v8.4s, v1.s[1]      \n"
                    "fmla   v29.4s, v9.4s, v1.s[0]      \n"
                    "fmla   v31.4s, v9.4s, v1.s[1]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.4s, v10.4s, v1.s[2]     \n"
                    "fmla   v30.4s, v10.4s, v1.s[3]     \n"
                    "fmla   v29.4s, v11.4s, v1.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v1.s[3]     \n"
                    "bne    2b                          \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1    {v0.2s}, [%2], #8           \n"
                    "ld1    {v4.4s, v5.4s}, [%1], #32   \n"
                    "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v30.4s, v4.4s, v0.s[1]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v29.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[1]      \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
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
                    _sum01 = vld1q_f32(outptr + 4 * 1);
                    _sum10 = vld1q_f32(outptr + 4 * 2);
                    _sum11 = vld1q_f32(outptr + 4 * 3);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                outptr += 8 * 2;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj < max_jj; jj++)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v30.4s, v31.4s}, [%0]      \n"
                    "b      1f                          \n"

                    "0:                                 \n"
                    "eor    v30.16b, v30.16b, v30.16b   \n"
                    "eor    v31.16b, v31.16b, v31.16b   \n"

                    "1:                                 \n"
                    "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    3f                          \n"

                    "eor    v28.16b, v28.16b, v28.16b   \n"
                    "eor    v29.16b, v29.16b, v29.16b   \n"
                    "2:                                 \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v0.4s}, [%2], #16          \n"
                    "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v30.4s, v6.4s, v0.s[1]      \n"
                    "fmla   v31.4s, v7.4s, v0.s[1]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                    "fmla   v28.4s, v8.4s, v0.s[2]      \n"
                    "fmla   v29.4s, v9.4s, v0.s[2]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v10.4s, v0.s[3]     \n"
                    "fmla   v31.4s, v11.4s, v0.s[3]     \n"
                    "bne    2b                          \n"
                    "fadd   v30.4s, v30.4s, v28.4s      \n"
                    "fadd   v31.4s, v31.4s, v29.4s      \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1r   {v0.4s}, [%2], #4           \n"
                    "ld1    {v4.4s, v5.4s}, [%1], #32   \n"
                    "fmla   v30.4s, v4.4s, v0.4s        \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v31.4s, v5.4s, v0.4s        \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v30.4s, v31.4s}, [%0], #32 \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
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

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float32x4_t _pA0 = vld1q_f32(pA);
                    float32x4_t _pA1 = vld1q_f32(pA + 4);

                    float32x4_t _pB = vld1q_dup_f32(pB);

                    _sum00 = vfmaq_f32(_sum00, _pA0, _pB);
                    _sum01 = vfmaq_f32(_sum01, _pA1, _pB);

                    pA += 8;
                    pB += 1;
                }

                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                outptr += 8;
#endif // NCNN_GNU_INLINE_ASM
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
#if __aarch64__
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
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
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"
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
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n"
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
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"
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
                    "bne    2b                          \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1    {v0.4s, v1.4s, v2.4s}, [%2], #48 \n"
                    "ld1    {v16.4s}, [%1], #16         \n"
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
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
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

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float32x4_t _pA = vld1q_f32(pA);
                    float32x4_t _pB0 = vld1q_f32(pB);
                    float32x4_t _pB1 = vld1q_f32(pB + 4);
                    float32x4_t _pB2 = vld1q_f32(pB + 8);

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
                outptr += 4 * 12;
#endif // NCNN_GNU_INLINE_ASM
            }
#endif // __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
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
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"
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
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n"
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
                    "bne    2b                          \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                    "ld1    {v16.4s}, [%1], #16         \n"
                    "fmla   v24.4s, v16.4s, v0.s[0]     \n"
                    "fmla   v25.4s, v16.4s, v0.s[1]     \n"
                    "fmla   v26.4s, v16.4s, v0.s[2]     \n"
                    "fmla   v27.4s, v16.4s, v0.s[3]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.4s, v16.4s, v1.s[0]     \n"
                    "fmla   v29.4s, v16.4s, v1.s[1]     \n"
                    "fmla   v30.4s, v16.4s, v1.s[2]     \n"
                    "fmla   v31.4s, v16.4s, v1.s[3]     \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
                asm volatile(
                    "cmp        %7, #0              \n"
                    "beq        0f                  \n"

                    "vldm       %0!, {d16-d23}      \n"
                    "vldm       %0, {d24-d31}       \n"
                    "sub        %0, %0, #64         \n"
                    "b          1f                  \n"

                    "0:                             \n"
                    "veor       q8, q8              \n"
                    "veor       q9, q9              \n"
                    "veor       q10, q10            \n"
                    "veor       q11, q11            \n"
                    "veor       q12, q12            \n"
                    "veor       q13, q13            \n"
                    "veor       q14, q14            \n"
                    "veor       q15, q15            \n"

                    "1:                             \n"
                    "lsr        r4, %6, #2          \n" // r4 = max_kk >> 2
                    "cmp        r4, #0              \n"
                    "beq        3f                  \n"

                    "2:                             \n"
                    "pld        [%1, #512]          \n"
                    "vldm       %1!, {d8-d15}       \n"
                    "pld        [%2, #512]          \n"
                    "vldm       %2!, {d0-d7}        \n"
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
                    "pld        [%2, #512]          \n"
                    "vldm       %2!, {d0-d7}        \n"
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
                    "bne        2b                  \n"

                    "3:                             \n"
                    "and        r4, %6, #3          \n" // r4 = remain = max_kk & 3
                    "cmp        r4, #0              \n"
                    "beq        5f                  \n"

                    "4:                             \n"
                    "vldm       %2!, {d0-d3}        \n"
                    "vld1.f32   {d8-d9}, [%1 :128]! \n"
                    "vmla.f32   q8, q4, d0[0]       \n"
                    "vmla.f32   q9, q4, d0[1]       \n"
                    "vmla.f32   q10, q4, d1[0]      \n"
                    "vmla.f32   q11, q4, d1[1]      \n"
                    "subs       r4, r4, #1          \n"
                    "vmla.f32   q12, q4, d2[0]      \n"
                    "vmla.f32   q13, q4, d2[1]      \n"
                    "vmla.f32   q14, q4, d3[0]      \n"
                    "vmla.f32   q15, q4, d3[1]      \n"
                    "bne        4b                  \n"

                    "5:                             \n"
                    "vstm       %0!, {d16-d23}      \n"
                    "vstm       %0!, {d24-d31}      \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
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

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
                outptr += 4 * 8;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0] \n"
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
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"
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
                    "bne    2b                          \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1    {v0.4s}, [%2], #16          \n"
                    "ld1    {v16.4s}, [%1], #16         \n"
                    "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                    "fmla   v29.4s, v16.4s, v0.s[1]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v16.4s, v0.s[2]     \n"
                    "fmla   v31.4s, v16.4s, v0.s[3]     \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
#else  // __aarch64__
                asm volatile(
                    "cmp        %7, #0              \n"
                    "beq        0f                  \n"

                    "vldm       %0, {d24-d31}       \n"
                    "b          1f                  \n"

                    "0:                             \n"
                    "veor       q12, q12            \n"
                    "veor       q13, q13            \n"
                    "veor       q14, q14            \n"
                    "veor       q15, q15            \n"

                    "1:                             \n"
                    "lsr        r4, %6, #2          \n" // r4 = max_kk >> 2
                    "cmp        r4, #0              \n"
                    "beq        3f                  \n"

                    "2:                             \n"
                    "pld        [%1, #512]          \n"
                    "vldm       %1!, {d8-d15}       \n"
                    "pld        [%2, #512]          \n"
                    "vldm       %2!, {d0-d7}        \n"
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
                    "bne        2b                  \n"

                    "3:                             \n"
                    "and        r4, %6, #3          \n" // r4 = remain = max_kk & 3
                    "cmp        r4, #0              \n"
                    "beq        5f                  \n"

                    "4:                             \n"
                    "vld1.f32   {d0-d1}, [%2 :128]! \n"
                    "vld1.f32   {d8-d9}, [%1 :128]! \n"
                    "vmla.f32   q12, q4, d0[0]      \n"
                    "vmla.f32   q13, q4, d0[1]      \n"
                    "subs       r4, r4, #1          \n"
                    "vmla.f32   q14, q4, d1[0]      \n"
                    "vmla.f32   q15, q4, d1[1]      \n"
                    "bne        4b                  \n"

                    "5:                             \n"
                    "vstm       %0!, {d24-d31}      \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
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

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                outptr += 4 * 4;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v30.4s, v31.4s}, [%0]      \n"
                    "b      1f                          \n"

                    "0:                                 \n"
                    "eor    v30.16b, v30.16b, v30.16b   \n"
                    "eor    v31.16b, v31.16b, v31.16b   \n"

                    "1:                                 \n"
                    "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    3f                          \n"

                    "eor    v28.16b, v28.16b, v28.16b   \n"
                    "eor    v29.16b, v29.16b, v29.16b   \n"
                    "2:                                 \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                    "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                    "fmla   v29.4s, v16.4s, v0.s[1]     \n"
                    "fmla   v30.4s, v17.4s, v0.s[2]     \n"
                    "fmla   v31.4s, v17.4s, v0.s[3]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.4s, v18.4s, v1.s[0]     \n"
                    "fmla   v29.4s, v18.4s, v1.s[1]     \n"
                    "fmla   v30.4s, v19.4s, v1.s[2]     \n"
                    "fmla   v31.4s, v19.4s, v1.s[3]     \n"
                    "bne    2b                          \n"
                    "fadd   v30.4s, v30.4s, v28.4s      \n"
                    "fadd   v31.4s, v31.4s, v29.4s      \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1    {v0.2s}, [%2], #8           \n"
                    "ld1    {v16.4s}, [%1], #16         \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v16.4s, v0.s[0]     \n"
                    "fmla   v31.4s, v16.4s, v0.s[1]     \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v30.4s, v31.4s}, [%0], #32 \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
#else  // __aarch64__
                asm volatile(
                    "cmp        %7, #0              \n"
                    "beq        0f                  \n"

                    "vld1.f32   {d28-d31}, [%0 :128] \n"
                    "b          1f                  \n"

                    "0:                             \n"
                    "veor       q14, q14            \n"
                    "veor       q15, q15            \n"

                    "1:                             \n"
                    "lsr        r4, %6, #2          \n" // r4 = max_kk >> 2
                    "cmp        r4, #0              \n"
                    "beq        3f                  \n"

                    "veor       q12, q12            \n"
                    "veor       q13, q13            \n"
                    "2:                             \n"
                    "pld        [%1, #512]          \n"
                    "vldm       %1!, {d8-d15}       \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d0-d3}, [%2 :128]! \n"
                    "vmla.f32   q12, q4, d0[0]      \n"
                    "vmla.f32   q13, q4, d0[1]      \n"
                    "vmla.f32   q14, q5, d1[0]      \n"
                    "vmla.f32   q15, q5, d1[1]      \n"
                    "subs       r4, r4, #1          \n"
                    "vmla.f32   q12, q6, d2[0]      \n"
                    "vmla.f32   q13, q6, d2[1]      \n"
                    "vmla.f32   q14, q7, d3[0]      \n"
                    "vmla.f32   q15, q7, d3[1]      \n"
                    "bne        2b                  \n"
                    "vadd.f32   q14, q14, q12       \n"
                    "vadd.f32   q15, q15, q13       \n"

                    "3:                             \n"
                    "and        r4, %6, #3          \n" // r4 = remain = max_kk & 3
                    "cmp        r4, #0              \n"
                    "beq        5f                  \n"

                    "4:                             \n"
                    "vld1.f32   {d0}, [%2 :64]!     \n"
                    "vld1.f32   {d8-d9}, [%1 :128]! \n"
                    "subs       r4, r4, #1          \n"
                    "vmla.f32   q14, q4, d0[0]      \n"
                    "vmla.f32   q15, q4, d0[1]      \n"
                    "bne        4b                  \n"

                    "5:                             \n"
                    "vst1.f32   {d28-d31}, [%0 :128]! \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
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

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                outptr += 4 * 2;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; jj < max_jj; jj++)
            {
                const float* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "cbz    %w7, 0f                     \n"

                    "ld1    {v31.4s}, [%0]              \n"
                    "b      1f                          \n"

                    "0:                                 \n"
                    "eor    v31.16b, v31.16b, v31.16b   \n"

                    "1:                                 \n"
                    "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    3f                          \n"

                    "eor    v28.16b, v28.16b, v28.16b   \n"
                    "eor    v29.16b, v29.16b, v29.16b   \n"
                    "eor    v30.16b, v30.16b, v30.16b   \n"
                    "2:                                 \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v0.4s}, [%2], #16          \n"
                    "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                    "fmla   v29.4s, v17.4s, v0.s[1]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v18.4s, v0.s[2]     \n"
                    "fmla   v31.4s, v19.4s, v0.s[3]     \n"
                    "bne    2b                          \n"
                    "fadd   v30.4s, v30.4s, v28.4s      \n"
                    "fadd   v31.4s, v31.4s, v29.4s      \n"
                    "fadd   v31.4s, v31.4s, v30.4s      \n"

                    "3:                                 \n"
                    "and    w4, %w6, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "ld1r   {v0.4s}, [%2], #4           \n"
                    "ld1    {v16.4s}, [%1], #16         \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v31.4s, v16.4s, v0.4s       \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "st1    {v31.4s}, [%0], #16         \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
#else  // __aarch64__
                asm volatile(
                    "cmp        %7, #0              \n"
                    "beq        0f                  \n"

                    "vld1.f32   {d30-d31}, [%0 :128] \n"
                    "b          1f                  \n"

                    "0:                             \n"
                    "veor       q15, q15            \n"

                    "1:                             \n"
                    "lsr        r4, %6, #2          \n" // r4 = max_kk >> 2
                    "cmp        r4, #0              \n"
                    "beq        3f                  \n"

                    "veor       q12, q12            \n"
                    "veor       q13, q13            \n"
                    "veor       q14, q14            \n"
                    "2:                             \n"
                    "pld        [%1, #512]          \n"
                    "vldm       %1!, {d8-d15}       \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%2 :64]!  \n"
                    "vmla.f32   q12, q4, d0[0]      \n"
                    "vmla.f32   q13, q5, d0[1]      \n"
                    "subs       r4, r4, #1          \n"
                    "vmla.f32   q14, q6, d1[0]      \n"
                    "vmla.f32   q15, q7, d1[1]      \n"
                    "bne        2b                  \n"
                    "vadd.f32   q14, q14, q12       \n"
                    "vadd.f32   q15, q15, q13       \n"
                    "vadd.f32   q15, q15, q14       \n"

                    "3:                             \n"
                    "and        r4, %6, #3          \n" // r4 = remain = max_kk & 3
                    "cmp        r4, #0              \n"
                    "beq        5f                  \n"

                    "4:                             \n"
                    "vld1.f32   {d0[0]}, [%2]!      \n"
                    "vld1.f32   {d8-d9}, [%1 :128]! \n"
                    "subs       r4, r4, #1          \n"
                    "vmla.f32   q15, q4, d0[0]      \n"
                    "bne        4b                  \n"

                    "5:                             \n"
                    "vst1.f32   {d30-d31}, [%0 :128]! \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB)      // %2
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "r"(max_kk), // %6
                    "r"(k)       // %7
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                float32x4_t _sum;

                if (k == 0)
                {
                    _sum = vdupq_n_f32(0.f);
                }
                else
                {
                    _sum = vld1q_f32(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    float32x4_t _pA = vld1q_f32(pA);
                    float32x4_t _pB = vdupq_n_f32(pB[0]);

#if __aarch64__
                    _sum = vfmaq_f32(_sum, _pA, _pB);
#else
                    _sum = vmlaq_f32(_sum, _pA, _pB);
#endif

                    pA += 4;
                    pB += 1;
                }

                vst1q_f32(outptr, _sum);
                outptr += 4;
#endif // NCNN_GNU_INLINE_ASM
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
#if __ARM_NEON
#if __aarch64__
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

                float32x4_t _sum00;
                float32x4_t _sum01;
                float32x4_t _sum02;
                float32x4_t _sum10;
                float32x4_t _sum11;
                float32x4_t _sum12;

                if (k == 0)
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum02 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum12 = vdupq_n_f32(0.f);
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

                int kk = 0;
                for (; kk < max_kk; kk++)
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
                outptr += 2 * 12;
            }
#endif // __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

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
                    float32x4x2_t _tmp01 = vld2q_f32(outptr);
                    float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                    _sum00 = _tmp01.val[0];
                    _sum01 = _tmp23.val[0];
                    _sum10 = _tmp01.val[1];
                    _sum11 = _tmp23.val[1];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
                outptr += 2 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                float32x4_t _sum0;
                float32x4_t _sum1;

                if (k == 0)
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
                else
                {
                    float32x4x2_t _tmp01 = vld2q_f32(outptr);
                    _sum0 = _tmp01.val[0];
                    _sum1 = _tmp01.val[1];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum0;
                _tmp01.val[1] = _sum1;
                vst2q_f32(outptr, _tmp01);
                outptr += 2 * 4;
            }
#endif // __ARM_NEON
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                float sum00 = 0.f;
                float sum01 = 0.f;
                float sum10 = 0.f;
                float sum11 = 0.f;

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
                const float* pA = pAT;

                float sum0 = 0.f;
                float sum1 = 0.f;

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
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
#if __ARM_NEON
#if __aarch64__
            for (; jj + 11 < max_jj; jj += 12)
            {
                const float* pA = pAT;

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

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
                outptr += 12;
            }
#endif // __aarch64__
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

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

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                outptr += 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                float32x4_t _sum;

                if (k == 0)
                {
                    _sum = vdupq_n_f32(0.f);
                }
                else
                {
                    _sum = vld1q_f32(outptr);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
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

                vst1q_f32(outptr, _sum);
                outptr += 4;
            }
#endif // __ARM_NEON
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                float sum0 = 0.f;
                float sum1 = 0.f;

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
                const float* pA = pAT;

                float sum = 0.f;

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

static void conv3x3s1_winograd_get_optimal_tile_mnk(int M, int N, int K, int B, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp32 = (int)(get_cpu_level2_cache_size() / sizeof(float));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // we shall take B into account for batched gemm, but that will be slower on arm in practice, why ?
    (void)B;

    // solve K
    {
        // try not to split K
#if __aarch64__
        int tile_size = (l2_cache_size_fp32 - 32) / 12;
#elif __ARM_NEON
        int tile_size = (l2_cache_size_fp32 - 16) / 8;
#else
        int tile_size = (l2_cache_size_fp32 - 2) / 3;
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
        TILE_M = 8;
#elif __ARM_NEON
        TILE_M = 4;
#else
        TILE_M = 2;
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

#if __aarch64__
        TILE_M = std::max(8, TILE_M);
#elif __ARM_NEON
        TILE_M = std::max(4, TILE_M);
#else
        TILE_M = std::max(2, TILE_M);
#endif
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / (TILE_M + TILE_K);
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

#if __aarch64__
        TILE_N = std::max(4, TILE_N);
#elif __ARM_NEON
        TILE_N = std::max(4, TILE_N);
#else
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static inline void conv3x3s1_winograd23_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const float ktm[4][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {1.0f / 2, 1.0f / 2, 1.0f / 2},
    //     {1.0f / 2, -1.0f / 2, 1.0f / 2},
    //     {0.0f, 0.0f, 1.0f}
    // };

    float* ptmp = A;

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

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp += 4;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd23_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd23_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
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
#if __ARM_NEON
#if __aarch64__
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
        float tmp[4][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r00 = vdupq_n_f32(0.f);
                float32x4_t _r01 = vdupq_n_f32(0.f);
                float32x4_t _r10 = vdupq_n_f32(0.f);
                float32x4_t _r11 = vdupq_n_f32(0.f);
                float32x4_t _r20 = vdupq_n_f32(0.f);
                float32x4_t _r21 = vdupq_n_f32(0.f);
                float32x4_t _r30 = vdupq_n_f32(0.f);
                float32x4_t _r31 = vdupq_n_f32(0.f);

                if (ti * 2 + m < h)
                {
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r00 = vld1q_f32(r0);
                        _r01 = vld1q_f32(r1);
                        if (tj * 2 + 1 < w)
                        {
                            _r10 = vld1q_f32(r0 + 4);
                            _r11 = vld1q_f32(r1 + 4);
                        }
                        if (tj * 2 + 2 < w)
                        {
                            _r20 = vld1q_f32(r0 + 8);
                            _r21 = vld1q_f32(r1 + 8);
                        }
                        if (tj * 2 + 3 < w)
                        {
                            _r30 = vld1q_f32(r0 + 12);
                            _r31 = vld1q_f32(r1 + 12);
                        }
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4_t _t2 = vld1q_f32(r2);
                        float32x4_t _t3 = vld1q_f32(r3);
                        float32x4_t _t4 = vld1q_f32(r4);
                        float32x4_t _t5 = vld1q_f32(r5);
                        float32x4_t _t6 = vld1q_f32(r6);
                        float32x4_t _t7 = vld1q_f32(r7);

                        transpose4x4_ps(_t0, _t1, _t2, _t3);
                        transpose4x4_ps(_t4, _t5, _t6, _t7);

                        _r00 = _t0;
                        _r01 = _t4;
                        if (tj * 2 + 1 < w)
                        {
                            _r10 = _t1;
                            _r11 = _t5;
                        }
                        if (tj * 2 + 2 < w)
                        {
                            _r20 = _t2;
                            _r21 = _t6;
                        }
                        if (tj * 2 + 3 < w)
                        {
                            _r30 = _t3;
                            _r31 = _t7;
                        }
                    }
                }

                float32x4_t _tmp00 = vsubq_f32(_r00, _r20);
                float32x4_t _tmp01 = vsubq_f32(_r01, _r21);
                float32x4_t _tmp10 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp11 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp20 = vsubq_f32(_r20, _r10);
                float32x4_t _tmp21 = vsubq_f32(_r21, _r11);
                float32x4_t _tmp30 = vsubq_f32(_r30, _r10);
                float32x4_t _tmp31 = vsubq_f32(_r31, _r11);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);

                float32x4_t _tmp00 = vsubq_f32(_r00, _r20);
                float32x4_t _tmp01 = vsubq_f32(_r01, _r21);
                float32x4_t _tmp10 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp11 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp20 = vsubq_f32(_r20, _r10);
                float32x4_t _tmp21 = vsubq_f32(_r21, _r11);
                float32x4_t _tmp30 = vsubq_f32(_r30, _r10);
                float32x4_t _tmp31 = vsubq_f32(_r31, _r11);

                vst1q_f32(p0, _tmp00);
                vst1q_f32(p0 + 4, _tmp01);
                vst1q_f32(p1, _tmp10);
                vst1q_f32(p1 + 4, _tmp11);
                vst1q_f32(p2, _tmp20);
                vst1q_f32(p2 + 4, _tmp21);
                vst1q_f32(p3, _tmp30);
                vst1q_f32(p3 + 4, _tmp31);

                p0 += max_jj * 4 * 8;
                p1 += max_jj * 4 * 8;
                p2 += max_jj * 4 * 8;
                p3 += max_jj * 4 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
#else // __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __aarch64__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r0 = vdupq_n_f32(0.f);
                float32x4_t _r1 = vdupq_n_f32(0.f);
                float32x4_t _r2 = vdupq_n_f32(0.f);
                float32x4_t _r3 = vdupq_n_f32(0.f);

                if (ti * 2 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        if (tj * 2 + 1 < w) _r1 = vld1q_f32(r0 + 4);
                        if (tj * 2 + 2 < w) _r2 = vld1q_f32(r0 + 8);
                        if (tj * 2 + 3 < w) _r3 = vld1q_f32(r0 + 12);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4_t _t2 = vld1q_f32(r2);
                        float32x4_t _t3 = vld1q_f32(r3);

                        transpose4x4_ps(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 2 + 1 < w) _r1 = _t1;
                        if (tj * 2 + 2 < w) _r2 = _t2;
                        if (tj * 2 + 3 < w) _r3 = _t3;
                    }
                }

                float32x4_t _tmp0 = vsubq_f32(_r0, _r2);
                float32x4_t _tmp1 = vaddq_f32(_r1, _r2);
                float32x4_t _tmp2 = vsubq_f32(_r2, _r1);
                float32x4_t _tmp3 = vsubq_f32(_r3, _r1);

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);

                float32x4_t _tmp0 = vsubq_f32(_r0, _r2);
                float32x4_t _tmp1 = vaddq_f32(_r1, _r2);
                float32x4_t _tmp2 = vsubq_f32(_r2, _r1);
                float32x4_t _tmp3 = vsubq_f32(_r3, _r1);

                vst1q_f32(p0, _tmp0);
                vst1q_f32(p1, _tmp1);
                vst1q_f32(p2, _tmp2);
                vst1q_f32(p3, _tmp3);

                p0 += max_jj * 4 * 4;
                p1 += max_jj * 4 * 4;
                p2 += max_jj * 4 * 4;
                p3 += max_jj * 4 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __ARM_NEON
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __ARM_NEON
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[4][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vdup_n_f32(0.f);
                float32x2_t _r1 = vdup_n_f32(0.f);
                float32x2_t _r2 = vdup_n_f32(0.f);
                float32x2_t _r3 = vdup_n_f32(0.f);
#else
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
#endif

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

#if __ARM_NEON
                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4x2_t _t01 = vzipq_f32(_t0, _t1);

                        _r0 = vget_low_f32(_t01.val[0]);
                        if (tj * 2 + 1 < w) _r1 = vget_high_f32(_t01.val[0]);
                        if (tj * 2 + 2 < w) _r2 = vget_low_f32(_t01.val[1]);
                        if (tj * 2 + 3 < w) _r3 = vget_high_f32(_t01.val[1]);
#else
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
#endif
                    }
                }

#if __ARM_NEON
                float32x2_t _tmp0 = vsub_f32(_r0, _r2);
                float32x2_t _tmp1 = vadd_f32(_r1, _r2);
                float32x2_t _tmp2 = vsub_f32(_r2, _r1);
                float32x2_t _tmp3 = vsub_f32(_r3, _r1);

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
#else
                tmp[0][m][0] = r00 - r20;
                tmp[0][m][1] = r01 - r21;
                tmp[1][m][0] = r10 + r20;
                tmp[1][m][1] = r11 + r21;
                tmp[2][m][0] = r20 - r10;
                tmp[2][m][1] = r21 - r11;
                tmp[3][m][0] = r30 - r10;
                tmp[3][m][1] = r31 - r11;
#endif

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);

                float32x2_t _tmp0 = vsub_f32(_r0, _r2);
                float32x2_t _tmp1 = vadd_f32(_r1, _r2);
                float32x2_t _tmp2 = vsub_f32(_r2, _r1);
                float32x2_t _tmp3 = vsub_f32(_r3, _r1);

                vst1_f32(p0, _tmp0);
                vst1_f32(p1, _tmp1);
                vst1_f32(p2, _tmp2);
                vst1_f32(p3, _tmp3);
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];

                p0[0] = r00 - r20;
                p0[1] = r01 - r21;
                p1[0] = r10 + r20;
                p1[1] = r11 + r21;
                p2[0] = r20 - r10;
                p2[1] = r21 - r11;
                p3[0] = r30 - r10;
                p3[1] = r31 - r11;
#endif

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
        float tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;

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

            float* p0 = (float*)B + kk * max_jj * 16 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

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

static inline void conv3x3s1_winograd23_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
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

    const float* biasptr = bias;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = biasptr ? vld1q_f32(biasptr + i + ii + 4) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[2][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);

                float32x4_t _tmp00 = vaddq_f32(vaddq_f32(_r00, _r10), _r20);
                float32x4_t _tmp01 = vaddq_f32(vaddq_f32(_r01, _r11), _r21);
                float32x4_t _tmp10 = vaddq_f32(vsubq_f32(_r10, _r20), _r30);
                float32x4_t _tmp11 = vaddq_f32(vsubq_f32(_r11, _r21), _r31);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);

                r0 += max_jj * 4 * 8;
                r1 += max_jj * 4 * 8;
                r2 += max_jj * 4 * 8;
                r3 += max_jj * 4 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);

                float32x4_t _tmp00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r00, _r10), _r20));
                float32x4_t _tmp01 = vaddq_f32(_bias1, vaddq_f32(vaddq_f32(_r01, _r11), _r21));
                float32x4_t _tmp10 = vaddq_f32(_bias0, vaddq_f32(vsubq_f32(_r10, _r20), _r30));
                float32x4_t _tmp11 = vaddq_f32(_bias1, vaddq_f32(vsubq_f32(_r11, _r21), _r31));

                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    vst1q_f32(outptr0, _tmp00);
                    vst1q_f32(outptr1, _tmp01);
                    if (tj * 2 + 1 < outw)
                    {
                        vst1q_f32(outptr0 + 4, _tmp10);
                        vst1q_f32(outptr1 + 4, _tmp11);
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    vst1q_f32(tmp0, _tmp00);
                    vst1q_f32(tmp0 + 4, _tmp01);
                    vst1q_f32(tmp1, _tmp10);
                    vst1q_f32(tmp1 + 4, _tmp11);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[2][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _r1 = vld1q_f32(r1);
                float32x4_t _r2 = vld1q_f32(r2);
                float32x4_t _r3 = vld1q_f32(r3);

                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _r1), _r2);
                float32x4_t _tmp1 = vaddq_f32(vsubq_f32(_r1, _r2), _r3);

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);

                r0 += max_jj * 4 * 4;
                r1 += max_jj * 4 * 4;
                r2 += max_jj * 4 * 4;
                r3 += max_jj * 4 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);

                float32x4_t _tmp0 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r0, _r1), _r2));
                float32x4_t _tmp1 = vaddq_f32(_bias0, vaddq_f32(vsubq_f32(_r1, _r2), _r3));

                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _tmp0);
                    if (tj * 2 + 1 < outw) vst1q_f32(outptr0 + 4, _tmp1);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    vst1q_f32(tmp0, _tmp0);
                    vst1q_f32(tmp1, _tmp1);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        float32x2_t _bias0 = biasptr ? vld1_f32(biasptr + i + ii) : vdup_n_f32(0.f);
#else
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;
#endif

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(r0);
                float32x2_t _r1 = vld1_f32(r1);
                float32x2_t _r2 = vld1_f32(r2);
                float32x2_t _r3 = vld1_f32(r3);

                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _r1), _r2);
                float32x2_t _tmp1 = vadd_f32(vsub_f32(_r1, _r2), _r3);

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
#else
                tmp[0][m][0] = r0[0] + r1[0] + r2[0];
                tmp[0][m][1] = r0[1] + r1[1] + r2[1];
                tmp[1][m][0] = r1[0] - r2[0] + r3[0];
                tmp[1][m][1] = r1[1] - r2[1] + r3[1];
#endif

                r0 += max_jj * 4 * 2;
                r1 += max_jj * 4 * 2;
                r2 += max_jj * 4 * 2;
                r3 += max_jj * 4 * 2;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);

                float32x2_t _tmp0 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r0, _r1), _r2));
                float32x2_t _tmp1 = vadd_f32(_bias0, vadd_f32(vsub_f32(_r1, _r2), _r3));
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];

                float tmp00 = bias0 + r00 + r10 + r20;
                float tmp01 = bias1 + r01 + r11 + r21;
                float tmp10 = bias0 + r10 - r20 + r30;
                float tmp11 = bias1 + r11 - r21 + r31;
#endif

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

#if __ARM_NEON
                    outptr0[0] = vget_lane_f32(_tmp0, 0);
                    outptr1[0] = vget_lane_f32(_tmp0, 1);
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = vget_lane_f32(_tmp1, 0);
                        outptr1[1] = vget_lane_f32(_tmp1, 1);
                    }
#else
                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
#endif
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

                float tmp0 = bias0 + r0 + r1 + r2;
                float tmp1 = bias0 + r1 - r2 + r3;

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

static void conv3x3s1_winograd23(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
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

    // NCNN_LOGE("conv3x3s1_winograd23 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd23_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);

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
            conv3x3s1_winograd23_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

                conv3x3s1_winograd_gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    float* ptmp = A;

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

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp[4] = z4;
                ptmp[5] = z5;
                ptmp += 6;
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd43_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    const float sq2 = 1.41421356237;
    const float sq2_d2 = 1.41421356237 / 2;

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
#if __ARM_NEON
#if __aarch64__
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
        float tmp[6][6][8];

        const float coeffs[4] = {sq2, -sq2_d2, -2.f, -0.5f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _vm2_5 = vdupq_n_f32(-2.5f);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 4) + (tj * 4) * elempack;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r00 = vdupq_n_f32(0.f);
                float32x4_t _r01 = vdupq_n_f32(0.f);
                float32x4_t _r10 = vdupq_n_f32(0.f);
                float32x4_t _r11 = vdupq_n_f32(0.f);
                float32x4_t _r20 = vdupq_n_f32(0.f);
                float32x4_t _r21 = vdupq_n_f32(0.f);
                float32x4_t _r30 = vdupq_n_f32(0.f);
                float32x4_t _r31 = vdupq_n_f32(0.f);
                float32x4_t _r40 = vdupq_n_f32(0.f);
                float32x4_t _r41 = vdupq_n_f32(0.f);
                float32x4_t _r50 = vdupq_n_f32(0.f);
                float32x4_t _r51 = vdupq_n_f32(0.f);

                if (ti * 4 + m < h)
                {
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r00 = vld1q_f32(r0);
                        _r01 = vld1q_f32(r1);
                        if (tj * 4 + 1 < w)
                        {
                            _r10 = vld1q_f32(r0 + 4);
                            _r11 = vld1q_f32(r1 + 4);
                        }
                        if (tj * 4 + 2 < w)
                        {
                            _r20 = vld1q_f32(r0 + 8);
                            _r21 = vld1q_f32(r1 + 8);
                        }
                        if (tj * 4 + 3 < w)
                        {
                            _r30 = vld1q_f32(r0 + 12);
                            _r31 = vld1q_f32(r1 + 12);
                        }
                        if (tj * 4 + 4 < w)
                        {
                            _r40 = vld1q_f32(r0 + 16);
                            _r41 = vld1q_f32(r1 + 16);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            _r50 = vld1q_f32(r0 + 20);
                            _r51 = vld1q_f32(r1 + 20);
                        }
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4_t _t2 = vld1q_f32(r2);
                        float32x4_t _t3 = vld1q_f32(r3);
                        float32x4_t _t4 = vld1q_f32(r4);
                        float32x4_t _t5 = vld1q_f32(r5);
                        float32x4_t _t6 = vld1q_f32(r6);
                        float32x4_t _t7 = vld1q_f32(r7);

                        transpose4x4_ps(_t0, _t1, _t2, _t3);
                        transpose4x4_ps(_t4, _t5, _t6, _t7);

                        _r00 = _t0;
                        _r01 = _t4;
                        if (tj * 4 + 1 < w)
                        {
                            _r10 = _t1;
                            _r11 = _t5;
                        }
                        if (tj * 4 + 2 < w)
                        {
                            _r20 = _t2;
                            _r21 = _t6;
                        }
                        if (tj * 4 + 3 < w)
                        {
                            _r30 = _t3;
                            _r31 = _t7;
                        }
                        if (tj * 4 + 4 < w)
                        {
                            float tmp[8] = {r0[4], r1[4], r2[4], r3[4], r4[4], r5[4], r6[4], r7[4]};
                            _r40 = vld1q_f32(tmp);
                            _r41 = vld1q_f32(tmp + 4);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            float tmp[8] = {r0[5], r1[5], r2[5], r3[5], r4[5], r5[5], r6[5], r7[5]};
                            _r50 = vld1q_f32(tmp);
                            _r51 = vld1q_f32(tmp + 4);
                        }
                    }
                }

                float32x4_t _tmp12a0 = vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs, 0), _r30, _coeffs, 1);
                float32x4_t _tmp12a1 = vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs, 0), _r31, _coeffs, 1);
                float32x4_t _tmp12b0 = vfmaq_laneq_f32(_r40, _r20, _coeffs, 2);
                float32x4_t _tmp12b1 = vfmaq_laneq_f32(_r41, _r21, _coeffs, 2);
                float32x4_t _tmp34a0 = vfmaq_laneq_f32(vmulq_laneq_f32(_r30, _coeffs, 0), _r10, _coeffs, 1);
                float32x4_t _tmp34a1 = vfmaq_laneq_f32(vmulq_laneq_f32(_r31, _coeffs, 0), _r11, _coeffs, 1);
                float32x4_t _tmp34b0 = vfmaq_laneq_f32(_r40, _r20, _coeffs, 3);
                float32x4_t _tmp34b1 = vfmaq_laneq_f32(_r41, _r21, _coeffs, 3);

                float32x4_t _tmp00 = vfmaq_f32(vaddq_f32(_r00, _r40), _r20, _vm2_5);
                float32x4_t _tmp01 = vfmaq_f32(vaddq_f32(_r01, _r41), _r21, _vm2_5);
                float32x4_t _tmp10 = vsubq_f32(_tmp12b0, _tmp12a0);
                float32x4_t _tmp11 = vsubq_f32(_tmp12b1, _tmp12a1);
                float32x4_t _tmp20 = vaddq_f32(_tmp12b0, _tmp12a0);
                float32x4_t _tmp21 = vaddq_f32(_tmp12b1, _tmp12a1);
                float32x4_t _tmp30 = vaddq_f32(_tmp34b0, _tmp34a0);
                float32x4_t _tmp31 = vaddq_f32(_tmp34b1, _tmp34a1);
                float32x4_t _tmp40 = vsubq_f32(_tmp34b0, _tmp34a0);
                float32x4_t _tmp41 = vsubq_f32(_tmp34b1, _tmp34a1);
                float32x4_t _tmp50 = vfmaq_f32(vaddq_f32(_r10, _r50), _r30, _vm2_5);
                float32x4_t _tmp51 = vfmaq_f32(vaddq_f32(_r11, _r51), _r31, _vm2_5);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);
                vst1q_f32(tmp[4][m], _tmp40);
                vst1q_f32(tmp[4][m] + 4, _tmp41);
                vst1q_f32(tmp[5][m], _tmp50);
                vst1q_f32(tmp[5][m] + 4, _tmp51);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;
            float* p4 = p0 + max_jj * 8 * 4;
            float* p5 = p0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);
                float32x4_t _r40 = vld1q_f32(tmp[m][4]);
                float32x4_t _r41 = vld1q_f32(tmp[m][4] + 4);
                float32x4_t _r50 = vld1q_f32(tmp[m][5]);
                float32x4_t _r51 = vld1q_f32(tmp[m][5] + 4);

                float32x4_t _tmp12a0 = vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs, 0), _r30, _coeffs, 1);
                float32x4_t _tmp12a1 = vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs, 0), _r31, _coeffs, 1);
                float32x4_t _tmp12b0 = vfmaq_laneq_f32(_r40, _r20, _coeffs, 2);
                float32x4_t _tmp12b1 = vfmaq_laneq_f32(_r41, _r21, _coeffs, 2);
                float32x4_t _tmp34a0 = vfmaq_laneq_f32(vmulq_laneq_f32(_r30, _coeffs, 0), _r10, _coeffs, 1);
                float32x4_t _tmp34a1 = vfmaq_laneq_f32(vmulq_laneq_f32(_r31, _coeffs, 0), _r11, _coeffs, 1);
                float32x4_t _tmp34b0 = vfmaq_laneq_f32(_r40, _r20, _coeffs, 3);
                float32x4_t _tmp34b1 = vfmaq_laneq_f32(_r41, _r21, _coeffs, 3);

                float32x4_t _tmp00 = vfmaq_f32(vaddq_f32(_r00, _r40), _r20, _vm2_5);
                float32x4_t _tmp01 = vfmaq_f32(vaddq_f32(_r01, _r41), _r21, _vm2_5);
                float32x4_t _tmp10 = vsubq_f32(_tmp12b0, _tmp12a0);
                float32x4_t _tmp11 = vsubq_f32(_tmp12b1, _tmp12a1);
                float32x4_t _tmp20 = vaddq_f32(_tmp12b0, _tmp12a0);
                float32x4_t _tmp21 = vaddq_f32(_tmp12b1, _tmp12a1);
                float32x4_t _tmp30 = vaddq_f32(_tmp34b0, _tmp34a0);
                float32x4_t _tmp31 = vaddq_f32(_tmp34b1, _tmp34a1);
                float32x4_t _tmp40 = vsubq_f32(_tmp34b0, _tmp34a0);
                float32x4_t _tmp41 = vsubq_f32(_tmp34b1, _tmp34a1);
                float32x4_t _tmp50 = vfmaq_f32(vaddq_f32(_r10, _r50), _r30, _vm2_5);
                float32x4_t _tmp51 = vfmaq_f32(vaddq_f32(_r11, _r51), _r31, _vm2_5);

                vst1q_f32(p0, _tmp00);
                vst1q_f32(p0 + 4, _tmp01);
                vst1q_f32(p1, _tmp10);
                vst1q_f32(p1 + 4, _tmp11);
                vst1q_f32(p2, _tmp20);
                vst1q_f32(p2 + 4, _tmp21);
                vst1q_f32(p3, _tmp30);
                vst1q_f32(p3 + 4, _tmp31);
                vst1q_f32(p4, _tmp40);
                vst1q_f32(p4 + 4, _tmp41);
                vst1q_f32(p5, _tmp50);
                vst1q_f32(p5 + 4, _tmp51);

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
#else // __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __aarch64__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][6][4];

        const float coeffs[4] = {sq2, -sq2_d2, -2.f, -0.5f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _vm2_5 = vdupq_n_f32(-2.5f);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 4) + (tj * 4) * elempack;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r0 = vdupq_n_f32(0.f);
                float32x4_t _r1 = vdupq_n_f32(0.f);
                float32x4_t _r2 = vdupq_n_f32(0.f);
                float32x4_t _r3 = vdupq_n_f32(0.f);
                float32x4_t _r4 = vdupq_n_f32(0.f);
                float32x4_t _r5 = vdupq_n_f32(0.f);

                if (ti * 4 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        if (tj * 4 + 1 < w) _r1 = vld1q_f32(r0 + 4);
                        if (tj * 4 + 2 < w) _r2 = vld1q_f32(r0 + 8);
                        if (tj * 4 + 3 < w) _r3 = vld1q_f32(r0 + 12);
                        if (tj * 4 + 4 < w) _r4 = vld1q_f32(r0 + 16);
                        if (tj * 4 + 5 < w) _r5 = vld1q_f32(r0 + 20);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4_t _t2 = vld1q_f32(r2);
                        float32x4_t _t3 = vld1q_f32(r3);

                        transpose4x4_ps(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 4 + 1 < w) _r1 = _t1;
                        if (tj * 4 + 2 < w) _r2 = _t2;
                        if (tj * 4 + 3 < w) _r3 = _t3;
                        if (tj * 4 + 4 < w)
                        {
                            float tmp[4] = {r0[4], r1[4], r2[4], r3[4]};
                            _r4 = vld1q_f32(tmp);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            float tmp[4] = {r0[5], r1[5], r2[5], r3[5]};
                            _r5 = vld1q_f32(tmp);
                        }
                    }
                }

#if __aarch64__
                float32x4_t _tmp12a = vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float32x4_t _tmp12b = vfmaq_laneq_f32(_r4, _r2, _coeffs, 2);
                float32x4_t _tmp34a = vfmaq_laneq_f32(vmulq_laneq_f32(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float32x4_t _tmp34b = vfmaq_laneq_f32(_r4, _r2, _coeffs, 3);
#else
                float32x4_t _tmp12a = vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_low_f32(_coeffs), 0), _r3, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp12b = vmlaq_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp34a = vmlaq_lane_f32(vmulq_lane_f32(_r3, vget_low_f32(_coeffs), 0), _r1, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp34b = vmlaq_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 1);
#endif

#if __aarch64__
                float32x4_t _tmp0 = vfmaq_f32(vaddq_f32(_r0, _r4), _r2, _vm2_5);
#else
                float32x4_t _tmp0 = vmlaq_f32(vaddq_f32(_r0, _r4), _r2, _vm2_5);
#endif
                float32x4_t _tmp1 = vsubq_f32(_tmp12b, _tmp12a);
                float32x4_t _tmp2 = vaddq_f32(_tmp12b, _tmp12a);
                float32x4_t _tmp3 = vaddq_f32(_tmp34b, _tmp34a);
                float32x4_t _tmp4 = vsubq_f32(_tmp34b, _tmp34a);
#if __aarch64__
                float32x4_t _tmp5 = vfmaq_f32(vaddq_f32(_r1, _r5), _r3, _vm2_5);
#else
                float32x4_t _tmp5 = vmlaq_f32(vaddq_f32(_r1, _r5), _r3, _vm2_5);
#endif

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);
                vst1q_f32(tmp[4][m], _tmp4);
                vst1q_f32(tmp[5][m], _tmp5);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;
            float* p4 = p0 + max_jj * 4 * 4;
            float* p5 = p0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);
                float32x4_t _r4 = vld1q_f32(tmp[m][4]);
                float32x4_t _r5 = vld1q_f32(tmp[m][5]);

#if __aarch64__
                float32x4_t _tmp12a = vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float32x4_t _tmp12b = vfmaq_laneq_f32(_r4, _r2, _coeffs, 2);
                float32x4_t _tmp34a = vfmaq_laneq_f32(vmulq_laneq_f32(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float32x4_t _tmp34b = vfmaq_laneq_f32(_r4, _r2, _coeffs, 3);
#else
                float32x4_t _tmp12a = vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_low_f32(_coeffs), 0), _r3, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp12b = vmlaq_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp34a = vmlaq_lane_f32(vmulq_lane_f32(_r3, vget_low_f32(_coeffs), 0), _r1, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp34b = vmlaq_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 1);
#endif

#if __aarch64__
                float32x4_t _tmp0 = vfmaq_f32(vaddq_f32(_r0, _r4), _r2, _vm2_5);
#else
                float32x4_t _tmp0 = vmlaq_f32(vaddq_f32(_r0, _r4), _r2, _vm2_5);
#endif
                float32x4_t _tmp1 = vsubq_f32(_tmp12b, _tmp12a);
                float32x4_t _tmp2 = vaddq_f32(_tmp12b, _tmp12a);
                float32x4_t _tmp3 = vaddq_f32(_tmp34b, _tmp34a);
                float32x4_t _tmp4 = vsubq_f32(_tmp34b, _tmp34a);
#if __aarch64__
                float32x4_t _tmp5 = vfmaq_f32(vaddq_f32(_r1, _r5), _r3, _vm2_5);
#else
                float32x4_t _tmp5 = vmlaq_f32(vaddq_f32(_r1, _r5), _r3, _vm2_5);
#endif

                vst1q_f32(p0, _tmp0);
                vst1q_f32(p1, _tmp1);
                vst1q_f32(p2, _tmp2);
                vst1q_f32(p3, _tmp3);
                vst1q_f32(p4, _tmp4);
                vst1q_f32(p5, _tmp5);

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
#else // __ARM_NEON
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __ARM_NEON
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[6][6][2];

#if __ARM_NEON
        const float coeffs[4] = {sq2, -sq2_d2, -2.f, -0.5f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x2_t _vm2_5 = vdup_n_f32(-2.5f);
#endif

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vdup_n_f32(0.f);
                float32x2_t _r1 = vdup_n_f32(0.f);
                float32x2_t _r2 = vdup_n_f32(0.f);
                float32x2_t _r3 = vdup_n_f32(0.f);
                float32x2_t _r4 = vdup_n_f32(0.f);
                float32x2_t _r5 = vdup_n_f32(0.f);
#else
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
                float r40 = 0.f;
                float r41 = 0.f;
                float r50 = 0.f;
                float r51 = 0.f;
#endif

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

#if __ARM_NEON
                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4x2_t _t01 = vzipq_f32(_t0, _t1);

                        _r0 = vget_low_f32(_t01.val[0]);
                        if (tj * 4 + 1 < w) _r1 = vget_high_f32(_t01.val[0]);
                        if (tj * 4 + 2 < w) _r2 = vget_low_f32(_t01.val[1]);
                        if (tj * 4 + 3 < w) _r3 = vget_high_f32(_t01.val[1]);
                        if (tj * 4 + 4 < w)
                        {
                            float tmp[2] = {r0[4], r1[4]};
                            _r4 = vld1_f32(tmp);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            float tmp[2] = {r0[5], r1[5]};
                            _r5 = vld1_f32(tmp);
                        }
#else
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
#endif
                    }
                }

#if __ARM_NEON
#if __aarch64__
                float32x2_t _tmp12a = vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float32x2_t _tmp12b = vfma_laneq_f32(_r4, _r2, _coeffs, 2);
                float32x2_t _tmp34a = vfma_laneq_f32(vmul_laneq_f32(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float32x2_t _tmp34b = vfma_laneq_f32(_r4, _r2, _coeffs, 3);
#else
                float32x2_t _tmp12a = vmla_lane_f32(vmul_lane_f32(_r1, vget_low_f32(_coeffs), 0), _r3, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp12b = vmla_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp34a = vmla_lane_f32(vmul_lane_f32(_r3, vget_low_f32(_coeffs), 0), _r1, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp34b = vmla_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 1);
#endif

#if __aarch64__
                float32x2_t _tmp0 = vfma_f32(vadd_f32(_r0, _r4), _r2, _vm2_5);
#else
                float32x2_t _tmp0 = vmla_f32(vadd_f32(_r0, _r4), _r2, _vm2_5);
#endif
                float32x2_t _tmp1 = vsub_f32(_tmp12b, _tmp12a);
                float32x2_t _tmp2 = vadd_f32(_tmp12b, _tmp12a);
                float32x2_t _tmp3 = vadd_f32(_tmp34b, _tmp34a);
                float32x2_t _tmp4 = vsub_f32(_tmp34b, _tmp34a);
#if __aarch64__
                float32x2_t _tmp5 = vfma_f32(vadd_f32(_r1, _r5), _r3, _vm2_5);
#else
                float32x2_t _tmp5 = vmla_f32(vadd_f32(_r1, _r5), _r3, _vm2_5);
#endif

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
                vst1_f32(tmp[4][m], _tmp4);
                vst1_f32(tmp[5][m], _tmp5);
#else
                float tmp12a0 = sq2 * r10 - sq2_d2 * r30;
                float tmp12a1 = sq2 * r11 - sq2_d2 * r31;
                float tmp12b0 = r40 - 2 * r20;
                float tmp12b1 = r41 - 2 * r21;
                float tmp34a0 = sq2 * r30 - sq2_d2 * r10;
                float tmp34a1 = sq2 * r31 - sq2_d2 * r11;
                float tmp34b0 = r40 - 0.5f * r20;
                float tmp34b1 = r41 - 0.5f * r21;

                tmp[0][m][0] = r00 + r40 - 2.5f * r20;
                tmp[0][m][1] = r01 + r41 - 2.5f * r21;
                tmp[1][m][0] = tmp12b0 - tmp12a0;
                tmp[1][m][1] = tmp12b1 - tmp12a1;
                tmp[2][m][0] = tmp12b0 + tmp12a0;
                tmp[2][m][1] = tmp12b1 + tmp12a1;
                tmp[3][m][0] = tmp34b0 + tmp34a0;
                tmp[3][m][1] = tmp34b1 + tmp34a1;
                tmp[4][m][0] = tmp34b0 - tmp34a0;
                tmp[4][m][1] = tmp34b1 - tmp34a1;
                tmp[5][m][0] = r10 + r50 - 2.5f * r30;
                tmp[5][m][1] = r11 + r51 - 2.5f * r31;
#endif

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;
            float* p4 = p0 + max_jj * 2 * 4;
            float* p5 = p0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);
                float32x2_t _r4 = vld1_f32(tmp[m][4]);
                float32x2_t _r5 = vld1_f32(tmp[m][5]);

#if __aarch64__
                float32x2_t _tmp12a = vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float32x2_t _tmp12b = vfma_laneq_f32(_r4, _r2, _coeffs, 2);
                float32x2_t _tmp34a = vfma_laneq_f32(vmul_laneq_f32(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float32x2_t _tmp34b = vfma_laneq_f32(_r4, _r2, _coeffs, 3);
#else
                float32x2_t _tmp12a = vmla_lane_f32(vmul_lane_f32(_r1, vget_low_f32(_coeffs), 0), _r3, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp12b = vmla_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp34a = vmla_lane_f32(vmul_lane_f32(_r3, vget_low_f32(_coeffs), 0), _r1, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp34b = vmla_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 1);
#endif

#if __aarch64__
                float32x2_t _tmp0 = vfma_f32(vadd_f32(_r0, _r4), _r2, _vm2_5);
#else
                float32x2_t _tmp0 = vmla_f32(vadd_f32(_r0, _r4), _r2, _vm2_5);
#endif
                float32x2_t _tmp1 = vsub_f32(_tmp12b, _tmp12a);
                float32x2_t _tmp2 = vadd_f32(_tmp12b, _tmp12a);
                float32x2_t _tmp3 = vadd_f32(_tmp34b, _tmp34a);
                float32x2_t _tmp4 = vsub_f32(_tmp34b, _tmp34a);
#if __aarch64__
                float32x2_t _tmp5 = vfma_f32(vadd_f32(_r1, _r5), _r3, _vm2_5);
#else
                float32x2_t _tmp5 = vmla_f32(vadd_f32(_r1, _r5), _r3, _vm2_5);
#endif

                vst1_f32(p0, _tmp0);
                vst1_f32(p1, _tmp1);
                vst1_f32(p2, _tmp2);
                vst1_f32(p3, _tmp3);
                vst1_f32(p4, _tmp4);
                vst1_f32(p5, _tmp5);
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];

                float tmp12a0 = sq2 * r10 - sq2_d2 * r30;
                float tmp12a1 = sq2 * r11 - sq2_d2 * r31;
                float tmp12b0 = r40 - 2 * r20;
                float tmp12b1 = r41 - 2 * r21;
                float tmp34a0 = sq2 * r30 - sq2_d2 * r10;
                float tmp34a1 = sq2 * r31 - sq2_d2 * r11;
                float tmp34b0 = r40 - 0.5f * r20;
                float tmp34b1 = r41 - 0.5f * r21;

                p0[0] = r00 + r40 - 2.5f * r20;
                p0[1] = r01 + r41 - 2.5f * r21;
                p1[0] = tmp12b0 - tmp12a0;
                p1[1] = tmp12b1 - tmp12a1;
                p2[0] = tmp12b0 + tmp12a0;
                p2[1] = tmp12b1 + tmp12a1;
                p3[0] = tmp34b0 + tmp34a0;
                p3[1] = tmp34b1 + tmp34a1;
                p4[0] = tmp34b0 - tmp34a0;
                p4[1] = tmp34b1 - tmp34a1;
                p5[0] = r10 + r50 - 2.5f * r30;
                p5[1] = r11 + r51 - 2.5f * r31;
#endif

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
        float tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;
                float r4 = 0.f;
                float r5 = 0.f;

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

                float tmp12a = sq2 * r1 - sq2_d2 * r3;
                float tmp12b = r4 - 2 * r2;
                float tmp34a = sq2 * r3 - sq2_d2 * r1;
                float tmp34b = r4 - 0.5f * r2;

                tmp[0][m] = r0 + r4 - 2.5f * r2;
                tmp[1][m] = tmp12b - tmp12a;
                tmp[2][m] = tmp12b + tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r1 + r5 - 2.5f * r3;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;
            float* p4 = p0 + max_jj * 4;
            float* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float tmp12a = sq2 * r1 - sq2_d2 * r3;
                float tmp12b = r4 - 2 * r2;
                float tmp34a = sq2 * r3 - sq2_d2 * r1;
                float tmp34b = r4 - 0.5f * r2;

                p0[0] = r0 + r4 - 2.5f * r2;
                p1[0] = tmp12b - tmp12a;
                p2[0] = tmp12b + tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r1 + r5 - 2.5f * r3;

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

static inline void conv3x3s1_winograd43_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    const float sq2 = 1.41421356237;
    const float sq2_m2 = 1.41421356237 * 2;
    const float sq2_d2 = 1.41421356237 / 2;
    const float sq2_d4 = 1.41421356237 / 4;

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

#if __ARM_NEON
    const float coeffs[6] = {sq2, sq2_d2, sq2_d4, sq2_m2, 0.5f, 2.f};
    float32x4_t _coeffs = vld1q_f32(coeffs);
    float32x2_t _coeffs2 = vld1_f32(coeffs + 4);
#endif // __ARM_NEON

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 3) / 4;

    const float* biasptr = bias;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = biasptr ? vld1q_f32(biasptr + i + ii + 4) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;
            const float* r4 = r0 + max_jj * 8 * 4;
            const float* r5 = r0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r50 = vld1q_f32(r5);
                float32x4_t _r51 = vld1q_f32(r5 + 4);

                float32x4_t _tmp02a0 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp02a1 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp02b0 = vaddq_f32(_r30, _r40);
                float32x4_t _tmp02b1 = vaddq_f32(_r31, _r41);
                float32x4_t _tmp13a0 = vsubq_f32(_r10, _r20);
                float32x4_t _tmp13a1 = vsubq_f32(_r11, _r21);
                float32x4_t _tmp13b0 = vsubq_f32(_r30, _r40);
                float32x4_t _tmp13b1 = vsubq_f32(_r31, _r41);

                float32x4_t _tmp00 = vaddq_f32(vaddq_f32(_r00, _tmp02a0), _tmp02b0);
                float32x4_t _tmp01 = vaddq_f32(vaddq_f32(_r01, _tmp02a1), _tmp02b1);
                float32x4_t _tmp10 = vfmaq_laneq_f32(vmulq_laneq_f32(_tmp13a0, _coeffs, 1), _tmp13b0, _coeffs, 0);
                float32x4_t _tmp11 = vfmaq_laneq_f32(vmulq_laneq_f32(_tmp13a1, _coeffs, 1), _tmp13b1, _coeffs, 0);
                float32x4_t _tmp20 = vfmaq_lane_f32(vmulq_lane_f32(_tmp02a0, _coeffs2, 0), _tmp02b0, _coeffs2, 1);
                float32x4_t _tmp21 = vfmaq_lane_f32(vmulq_lane_f32(_tmp02a1, _coeffs2, 0), _tmp02b1, _coeffs2, 1);
                float32x4_t _tmp30 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r50, _tmp13a0, _coeffs, 2), _tmp13b0, _coeffs, 3);
                float32x4_t _tmp31 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r51, _tmp13a1, _coeffs, 2), _tmp13b1, _coeffs, 3);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);
                float32x4_t _r40 = vld1q_f32(tmp[m][4]);
                float32x4_t _r41 = vld1q_f32(tmp[m][4] + 4);
                float32x4_t _r50 = vld1q_f32(tmp[m][5]);
                float32x4_t _r51 = vld1q_f32(tmp[m][5] + 4);

                float32x4_t _tmp02a0 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp02a1 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp02b0 = vaddq_f32(_r30, _r40);
                float32x4_t _tmp02b1 = vaddq_f32(_r31, _r41);
                float32x4_t _tmp13a0 = vsubq_f32(_r10, _r20);
                float32x4_t _tmp13a1 = vsubq_f32(_r11, _r21);
                float32x4_t _tmp13b0 = vsubq_f32(_r30, _r40);
                float32x4_t _tmp13b1 = vsubq_f32(_r31, _r41);

                float32x4_t _tmp00 = vaddq_f32(vaddq_f32(_r00, _tmp02a0), vaddq_f32(_tmp02b0, _bias0));
                float32x4_t _tmp01 = vaddq_f32(vaddq_f32(_r01, _tmp02a1), vaddq_f32(_tmp02b1, _bias1));
                float32x4_t _tmp10 = vfmaq_laneq_f32(vfmaq_laneq_f32(_bias0, _tmp13a0, _coeffs, 1), _tmp13b0, _coeffs, 0);
                float32x4_t _tmp11 = vfmaq_laneq_f32(vfmaq_laneq_f32(_bias1, _tmp13a1, _coeffs, 1), _tmp13b1, _coeffs, 0);
                float32x4_t _tmp20 = vfmaq_lane_f32(vfmaq_lane_f32(_bias0, _tmp02a0, _coeffs2, 0), _tmp02b0, _coeffs2, 1);
                float32x4_t _tmp21 = vfmaq_lane_f32(vfmaq_lane_f32(_bias1, _tmp02a1, _coeffs2, 0), _tmp02b1, _coeffs2, 1);
                float32x4_t _tmp30 = vfmaq_laneq_f32(vfmaq_laneq_f32(vaddq_f32(_r50, _bias0), _tmp13a0, _coeffs, 2), _tmp13b0, _coeffs, 3);
                float32x4_t _tmp31 = vfmaq_laneq_f32(vfmaq_laneq_f32(vaddq_f32(_r51, _bias1), _tmp13a1, _coeffs, 2), _tmp13b1, _coeffs, 3);

                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    vst1q_f32(outptr0, _tmp00);
                    vst1q_f32(outptr1, _tmp01);
                    if (tj * 4 + 1 < outw)
                    {
                        vst1q_f32(outptr0 + 4, _tmp10);
                        vst1q_f32(outptr1 + 4, _tmp11);
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        vst1q_f32(outptr0 + 8, _tmp20);
                        vst1q_f32(outptr1 + 8, _tmp21);
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        vst1q_f32(outptr0 + 12, _tmp30);
                        vst1q_f32(outptr1 + 12, _tmp31);
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    float tmp2[8];
                    float tmp3[8];
                    vst1q_f32(tmp0, _tmp00);
                    vst1q_f32(tmp0 + 4, _tmp01);
                    vst1q_f32(tmp1, _tmp10);
                    vst1q_f32(tmp1 + 4, _tmp11);
                    vst1q_f32(tmp2, _tmp20);
                    vst1q_f32(tmp2 + 4, _tmp21);
                    vst1q_f32(tmp3, _tmp30);
                    vst1q_f32(tmp3 + 4, _tmp31);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;
            const float* r4 = r0 + max_jj * 4 * 4;
            const float* r5 = r0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _r1 = vld1q_f32(r1);
                float32x4_t _r2 = vld1q_f32(r2);
                float32x4_t _r3 = vld1q_f32(r3);
                float32x4_t _r4 = vld1q_f32(r4);
                float32x4_t _r5 = vld1q_f32(r5);

                float32x4_t _tmp02a = vaddq_f32(_r1, _r2);
                float32x4_t _tmp02b = vaddq_f32(_r3, _r4);
                float32x4_t _tmp13a = vsubq_f32(_r1, _r2);
                float32x4_t _tmp13b = vsubq_f32(_r3, _r4);

                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _tmp02a), _tmp02b);
#if __aarch64__
                float32x4_t _tmp1 = vfmaq_laneq_f32(vmulq_laneq_f32(_tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float32x4_t _tmp2 = vfmaq_lane_f32(vmulq_lane_f32(_tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x4_t _tmp3 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r5, _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);
#else
                float32x4_t _tmp1 = vmlaq_lane_f32(vmulq_lane_f32(_tmp13a, vget_low_f32(_coeffs), 1), _tmp13b, vget_low_f32(_coeffs), 0);
                float32x4_t _tmp2 = vmlaq_lane_f32(vmulq_lane_f32(_tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x4_t _tmp3 = vmlaq_lane_f32(vmlaq_lane_f32(_r5, _tmp13a, vget_high_f32(_coeffs), 0), _tmp13b, vget_high_f32(_coeffs), 1);
#endif

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);
                float32x4_t _r4 = vld1q_f32(tmp[m][4]);
                float32x4_t _r5 = vld1q_f32(tmp[m][5]);

                float32x4_t _tmp02a = vaddq_f32(_r1, _r2);
                float32x4_t _tmp02b = vaddq_f32(_r3, _r4);
                float32x4_t _tmp13a = vsubq_f32(_r1, _r2);
                float32x4_t _tmp13b = vsubq_f32(_r3, _r4);

                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _tmp02a), vaddq_f32(_tmp02b, _bias0));
#if __aarch64__
                float32x4_t _tmp1 = vfmaq_laneq_f32(vfmaq_laneq_f32(_bias0, _tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float32x4_t _tmp2 = vfmaq_lane_f32(vfmaq_lane_f32(_bias0, _tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x4_t _tmp3 = vfmaq_laneq_f32(vfmaq_laneq_f32(vaddq_f32(_r5, _bias0), _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);
#else
                float32x4_t _tmp1 = vmlaq_lane_f32(vmlaq_lane_f32(_bias0, _tmp13a, vget_low_f32(_coeffs), 1), _tmp13b, vget_low_f32(_coeffs), 0);
                float32x4_t _tmp2 = vmlaq_lane_f32(vmlaq_lane_f32(_bias0, _tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x4_t _tmp3 = vmlaq_lane_f32(vmlaq_lane_f32(vaddq_f32(_r5, _bias0), _tmp13a, vget_high_f32(_coeffs), 0), _tmp13b, vget_high_f32(_coeffs), 1);
#endif

                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _tmp0);
                    if (tj * 4 + 1 < outw) vst1q_f32(outptr0 + 4, _tmp1);
                    if (tj * 4 + 2 < outw) vst1q_f32(outptr0 + 8, _tmp2);
                    if (tj * 4 + 3 < outw) vst1q_f32(outptr0 + 12, _tmp3);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    float tmp2[4];
                    float tmp3[4];
                    vst1q_f32(tmp0, _tmp0);
                    vst1q_f32(tmp1, _tmp1);
                    vst1q_f32(tmp2, _tmp2);
                    vst1q_f32(tmp3, _tmp3);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        float32x2_t _bias0 = biasptr ? vld1_f32(biasptr + i + ii) : vdup_n_f32(0.f);
#else
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;
#endif

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;
            const float* r4 = r0 + max_jj * 2 * 4;
            const float* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(r0);
                float32x2_t _r1 = vld1_f32(r1);
                float32x2_t _r2 = vld1_f32(r2);
                float32x2_t _r3 = vld1_f32(r3);
                float32x2_t _r4 = vld1_f32(r4);
                float32x2_t _r5 = vld1_f32(r5);

                float32x2_t _tmp02a = vadd_f32(_r1, _r2);
                float32x2_t _tmp02b = vadd_f32(_r3, _r4);
                float32x2_t _tmp13a = vsub_f32(_r1, _r2);
                float32x2_t _tmp13b = vsub_f32(_r3, _r4);

                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _tmp02a), _tmp02b);
#if __aarch64__
                float32x2_t _tmp1 = vfma_laneq_f32(vmul_laneq_f32(_tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float32x2_t _tmp2 = vfma_lane_f32(vmul_lane_f32(_tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x2_t _tmp3 = vfma_laneq_f32(vfma_laneq_f32(_r5, _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);
#else
                float32x2_t _tmp1 = vmla_lane_f32(vmul_lane_f32(_tmp13a, vget_low_f32(_coeffs), 1), _tmp13b, vget_low_f32(_coeffs), 0);
                float32x2_t _tmp2 = vmla_lane_f32(vmul_lane_f32(_tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x2_t _tmp3 = vmla_lane_f32(vmla_lane_f32(_r5, _tmp13a, vget_high_f32(_coeffs), 0), _tmp13b, vget_high_f32(_coeffs), 1);
#endif

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
#else
                float tmp02a0 = r1[0] + r2[0];
                float tmp02a1 = r1[1] + r2[1];
                float tmp02b0 = r3[0] + r4[0];
                float tmp02b1 = r3[1] + r4[1];
                float tmp13a0 = r1[0] - r2[0];
                float tmp13a1 = r1[1] - r2[1];
                float tmp13b0 = r3[0] - r4[0];
                float tmp13b1 = r3[1] - r4[1];

                tmp[0][m][0] = r0[0] + tmp02a0 + tmp02b0;
                tmp[0][m][1] = r0[1] + tmp02a1 + tmp02b1;
                tmp[1][m][0] = tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                tmp[1][m][1] = tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                tmp[2][m][0] = tmp02a0 * 0.5f + tmp02b0 * 2;
                tmp[2][m][1] = tmp02a1 * 0.5f + tmp02b1 * 2;
                tmp[3][m][0] = r5[0] + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                tmp[3][m][1] = r5[1] + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;
#endif

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);
                float32x2_t _r4 = vld1_f32(tmp[m][4]);
                float32x2_t _r5 = vld1_f32(tmp[m][5]);

                float32x2_t _tmp02a = vadd_f32(_r1, _r2);
                float32x2_t _tmp02b = vadd_f32(_r3, _r4);
                float32x2_t _tmp13a = vsub_f32(_r1, _r2);
                float32x2_t _tmp13b = vsub_f32(_r3, _r4);

                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _tmp02a), vadd_f32(_tmp02b, _bias0));
#if __aarch64__
                float32x2_t _tmp1 = vfma_laneq_f32(vfma_laneq_f32(_bias0, _tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float32x2_t _tmp2 = vfma_lane_f32(vfma_lane_f32(_bias0, _tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x2_t _tmp3 = vfma_laneq_f32(vfma_laneq_f32(vadd_f32(_r5, _bias0), _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);
#else
                float32x2_t _tmp1 = vmla_lane_f32(vmla_lane_f32(_bias0, _tmp13a, vget_low_f32(_coeffs), 1), _tmp13b, vget_low_f32(_coeffs), 0);
                float32x2_t _tmp2 = vmla_lane_f32(vmla_lane_f32(_bias0, _tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x2_t _tmp3 = vmla_lane_f32(vmla_lane_f32(vadd_f32(_r5, _bias0), _tmp13a, vget_high_f32(_coeffs), 0), _tmp13b, vget_high_f32(_coeffs), 1);
#endif
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];

                float tmp02a0 = r10 + r20;
                float tmp02a1 = r11 + r21;
                float tmp02b0 = r30 + r40;
                float tmp02b1 = r31 + r41;
                float tmp13a0 = r10 - r20;
                float tmp13a1 = r11 - r21;
                float tmp13b0 = r30 - r40;
                float tmp13b1 = r31 - r41;

                float tmp00 = bias0 + r00 + tmp02a0 + tmp02b0;
                float tmp01 = bias1 + r01 + tmp02a1 + tmp02b1;
                float tmp10 = bias0 + tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                float tmp11 = bias1 + tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                float tmp20 = bias0 + tmp02a0 * 0.5f + tmp02b0 * 2;
                float tmp21 = bias1 + tmp02a1 * 0.5f + tmp02b1 * 2;
                float tmp30 = bias0 + r50 + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                float tmp31 = bias1 + r51 + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;
#endif

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

#if __ARM_NEON
                    outptr0[0] = vget_lane_f32(_tmp0, 0);
                    outptr1[0] = vget_lane_f32(_tmp0, 1);
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = vget_lane_f32(_tmp1, 0);
                        outptr1[1] = vget_lane_f32(_tmp1, 1);
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = vget_lane_f32(_tmp2, 0);
                        outptr1[2] = vget_lane_f32(_tmp2, 1);
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = vget_lane_f32(_tmp3, 0);
                        outptr1[3] = vget_lane_f32(_tmp3, 1);
                    }
#else
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
#endif
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;
            const float* r4 = r0 + max_jj * 4;
            const float* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                float tmp02a = r1[0] + r2[0];
                float tmp02b = r3[0] + r4[0];
                float tmp13a = r1[0] - r2[0];
                float tmp13b = r3[0] - r4[0];

                tmp[0][m] = r0[0] + tmp02a + tmp02b;
                tmp[1][m] = tmp13a * sq2_d2 + tmp13b * sq2;
                tmp[2][m] = tmp02a * 0.5f + tmp02b * 2;
                tmp[3][m] = r5[0] + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float tmp02a = r1 + r2;
                float tmp02b = r3 + r4;
                float tmp13a = r1 - r2;
                float tmp13b = r3 - r4;

                float tmp0 = bias0 + r0 + tmp02a + tmp02b;
                float tmp1 = bias0 + tmp13a * sq2_d2 + tmp13b * sq2;
                float tmp2 = bias0 + tmp02a * 0.5f + tmp02b * 2;
                float tmp3 = bias0 + r5 + tmp13a * sq2_d4 + tmp13b * sq2_m2;

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

static void conv3x3s1_winograd43(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
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

    // NCNN_LOGE("conv3x3s1_winograd43 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd43_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);

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
            conv3x3s1_winograd43_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

                conv3x3s1_winograd_gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    float* ptmp = A;

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

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp[4] = z4;
                ptmp[5] = z5;
                ptmp[6] = z6;
                ptmp[7] = z7;
                ptmp += 8;
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 64;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd63_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
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
#if __ARM_NEON
#if __aarch64__
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
        float tmp[8][8][8];

        const float coeffs[8] = {5.25f, -4.25f, -1.25f, 0.25f, -2.5f, 0.5f, 2.f, 4.f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _coeffs2 = vld1q_f32(coeffs + 4);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r00 = vdupq_n_f32(0.f);
                float32x4_t _r01 = vdupq_n_f32(0.f);
                float32x4_t _r10 = vdupq_n_f32(0.f);
                float32x4_t _r11 = vdupq_n_f32(0.f);
                float32x4_t _r20 = vdupq_n_f32(0.f);
                float32x4_t _r21 = vdupq_n_f32(0.f);
                float32x4_t _r30 = vdupq_n_f32(0.f);
                float32x4_t _r31 = vdupq_n_f32(0.f);
                float32x4_t _r40 = vdupq_n_f32(0.f);
                float32x4_t _r41 = vdupq_n_f32(0.f);
                float32x4_t _r50 = vdupq_n_f32(0.f);
                float32x4_t _r51 = vdupq_n_f32(0.f);
                float32x4_t _r60 = vdupq_n_f32(0.f);
                float32x4_t _r61 = vdupq_n_f32(0.f);
                float32x4_t _r70 = vdupq_n_f32(0.f);
                float32x4_t _r71 = vdupq_n_f32(0.f);

                if (ti * 6 + m < h)
                {
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r00 = vld1q_f32(r0);
                        _r01 = vld1q_f32(r1);
                        if (tj * 6 + 1 < w)
                        {
                            _r10 = vld1q_f32(r0 + 4);
                            _r11 = vld1q_f32(r1 + 4);
                        }
                        if (tj * 6 + 2 < w)
                        {
                            _r20 = vld1q_f32(r0 + 8);
                            _r21 = vld1q_f32(r1 + 8);
                        }
                        if (tj * 6 + 3 < w)
                        {
                            _r30 = vld1q_f32(r0 + 12);
                            _r31 = vld1q_f32(r1 + 12);
                        }
                        if (tj * 6 + 4 < w)
                        {
                            _r40 = vld1q_f32(r0 + 16);
                            _r41 = vld1q_f32(r1 + 16);
                        }
                        if (tj * 6 + 5 < w)
                        {
                            _r50 = vld1q_f32(r0 + 20);
                            _r51 = vld1q_f32(r1 + 20);
                        }
                        if (tj * 6 + 6 < w)
                        {
                            _r60 = vld1q_f32(r0 + 24);
                            _r61 = vld1q_f32(r1 + 24);
                        }
                        if (tj * 6 + 7 < w)
                        {
                            _r70 = vld1q_f32(r0 + 28);
                            _r71 = vld1q_f32(r1 + 28);
                        }
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4_t _t2 = vld1q_f32(r2);
                        float32x4_t _t3 = vld1q_f32(r3);
                        float32x4_t _t4 = vld1q_f32(r4);
                        float32x4_t _t5 = vld1q_f32(r5);
                        float32x4_t _t6 = vld1q_f32(r6);
                        float32x4_t _t7 = vld1q_f32(r7);

                        transpose4x4_ps(_t0, _t1, _t2, _t3);
                        transpose4x4_ps(_t4, _t5, _t6, _t7);

                        _r00 = _t0;
                        _r01 = _t4;
                        if (tj * 6 + 1 < w)
                        {
                            _r10 = _t1;
                            _r11 = _t5;
                        }
                        if (tj * 6 + 2 < w)
                        {
                            _r20 = _t2;
                            _r21 = _t6;
                        }
                        if (tj * 6 + 3 < w)
                        {
                            _r30 = _t3;
                            _r31 = _t7;
                        }
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = vld1q_f32(r0 + 4);
                            _t1 = vld1q_f32(r1 + 4);
                            _t2 = vld1q_f32(r2 + 4);
                            _t3 = vld1q_f32(r3 + 4);
                            _t4 = vld1q_f32(r4 + 4);
                            _t5 = vld1q_f32(r5 + 4);
                            _t6 = vld1q_f32(r6 + 4);
                            _t7 = vld1q_f32(r7 + 4);

                            transpose4x4_ps(_t0, _t1, _t2, _t3);
                            transpose4x4_ps(_t4, _t5, _t6, _t7);

                            _r40 = _t0;
                            _r41 = _t4;
                            if (tj * 6 + 5 < w)
                            {
                                _r50 = _t1;
                                _r51 = _t5;
                            }
                            if (tj * 6 + 6 < w)
                            {
                                _r60 = _t2;
                                _r61 = _t6;
                            }
                            if (tj * 6 + 7 < w)
                            {
                                _r70 = _t3;
                                _r71 = _t7;
                            }
                        }
                    }
                }

                float32x4_t _tmp12a0 = vfmaq_laneq_f32(vaddq_f32(_r20, _r60), _r40, _coeffs, 1);
                float32x4_t _tmp12a1 = vfmaq_laneq_f32(vaddq_f32(_r21, _r61), _r41, _coeffs, 1);
                float32x4_t _tmp12b0 = vfmaq_laneq_f32(vaddq_f32(_r10, _r50), _r30, _coeffs, 1);
                float32x4_t _tmp12b1 = vfmaq_laneq_f32(vaddq_f32(_r11, _r51), _r31, _coeffs, 1);
                float32x4_t _tmp34a0 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r60, _r20, _coeffs, 3), _r40, _coeffs, 2);
                float32x4_t _tmp34a1 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r61, _r21, _coeffs, 3), _r41, _coeffs, 2);
                float32x4_t _tmp34b0 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs2, 1), _r30, _coeffs2, 0), _r50, _coeffs2, 2);
                float32x4_t _tmp34b1 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs2, 1), _r31, _coeffs2, 0), _r51, _coeffs2, 2);
                float32x4_t _tmp56a0 = vfmaq_laneq_f32(_r60, vfmaq_laneq_f32(_r20, _r40, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56a1 = vfmaq_laneq_f32(_r61, vfmaq_laneq_f32(_r21, _r41, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56b0 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs2, 2), _r30, _coeffs2, 0), _r50, _coeffs2, 1);
                float32x4_t _tmp56b1 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs2, 2), _r31, _coeffs2, 0), _r51, _coeffs2, 1);

                float32x4_t _tmp00 = vfmaq_laneq_f32(vsubq_f32(_r00, _r60), vsubq_f32(_r40, _r20), _coeffs, 0);
                float32x4_t _tmp01 = vfmaq_laneq_f32(vsubq_f32(_r01, _r61), vsubq_f32(_r41, _r21), _coeffs, 0);
                float32x4_t _tmp10 = vaddq_f32(_tmp12a0, _tmp12b0);
                float32x4_t _tmp11 = vaddq_f32(_tmp12a1, _tmp12b1);
                float32x4_t _tmp20 = vsubq_f32(_tmp12a0, _tmp12b0);
                float32x4_t _tmp21 = vsubq_f32(_tmp12a1, _tmp12b1);
                float32x4_t _tmp30 = vaddq_f32(_tmp34a0, _tmp34b0);
                float32x4_t _tmp31 = vaddq_f32(_tmp34a1, _tmp34b1);
                float32x4_t _tmp40 = vsubq_f32(_tmp34a0, _tmp34b0);
                float32x4_t _tmp41 = vsubq_f32(_tmp34a1, _tmp34b1);
                float32x4_t _tmp50 = vaddq_f32(_tmp56a0, _tmp56b0);
                float32x4_t _tmp51 = vaddq_f32(_tmp56a1, _tmp56b1);
                float32x4_t _tmp60 = vsubq_f32(_tmp56a0, _tmp56b0);
                float32x4_t _tmp61 = vsubq_f32(_tmp56a1, _tmp56b1);
                float32x4_t _tmp70 = vfmaq_laneq_f32(vsubq_f32(_r70, _r10), vsubq_f32(_r30, _r50), _coeffs, 0);
                float32x4_t _tmp71 = vfmaq_laneq_f32(vsubq_f32(_r71, _r11), vsubq_f32(_r31, _r51), _coeffs, 0);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);
                vst1q_f32(tmp[4][m], _tmp40);
                vst1q_f32(tmp[4][m] + 4, _tmp41);
                vst1q_f32(tmp[5][m], _tmp50);
                vst1q_f32(tmp[5][m] + 4, _tmp51);
                vst1q_f32(tmp[6][m], _tmp60);
                vst1q_f32(tmp[6][m] + 4, _tmp61);
                vst1q_f32(tmp[7][m], _tmp70);
                vst1q_f32(tmp[7][m] + 4, _tmp71);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;
            float* p4 = p0 + max_jj * 8 * 4;
            float* p5 = p0 + max_jj * 8 * 5;
            float* p6 = p0 + max_jj * 8 * 6;
            float* p7 = p0 + max_jj * 8 * 7;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);
                float32x4_t _r40 = vld1q_f32(tmp[m][4]);
                float32x4_t _r41 = vld1q_f32(tmp[m][4] + 4);
                float32x4_t _r50 = vld1q_f32(tmp[m][5]);
                float32x4_t _r51 = vld1q_f32(tmp[m][5] + 4);
                float32x4_t _r60 = vld1q_f32(tmp[m][6]);
                float32x4_t _r61 = vld1q_f32(tmp[m][6] + 4);
                float32x4_t _r70 = vld1q_f32(tmp[m][7]);
                float32x4_t _r71 = vld1q_f32(tmp[m][7] + 4);

                float32x4_t _tmp12a0 = vfmaq_laneq_f32(vaddq_f32(_r20, _r60), _r40, _coeffs, 1);
                float32x4_t _tmp12a1 = vfmaq_laneq_f32(vaddq_f32(_r21, _r61), _r41, _coeffs, 1);
                float32x4_t _tmp12b0 = vfmaq_laneq_f32(vaddq_f32(_r10, _r50), _r30, _coeffs, 1);
                float32x4_t _tmp12b1 = vfmaq_laneq_f32(vaddq_f32(_r11, _r51), _r31, _coeffs, 1);
                float32x4_t _tmp34a0 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r60, _r20, _coeffs, 3), _r40, _coeffs, 2);
                float32x4_t _tmp34a1 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r61, _r21, _coeffs, 3), _r41, _coeffs, 2);
                float32x4_t _tmp34b0 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs2, 1), _r30, _coeffs2, 0), _r50, _coeffs2, 2);
                float32x4_t _tmp34b1 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs2, 1), _r31, _coeffs2, 0), _r51, _coeffs2, 2);
                float32x4_t _tmp56a0 = vfmaq_laneq_f32(_r60, vfmaq_laneq_f32(_r20, _r40, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56a1 = vfmaq_laneq_f32(_r61, vfmaq_laneq_f32(_r21, _r41, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56b0 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs2, 2), _r30, _coeffs2, 0), _r50, _coeffs2, 1);
                float32x4_t _tmp56b1 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs2, 2), _r31, _coeffs2, 0), _r51, _coeffs2, 1);

                float32x4_t _tmp00 = vfmaq_laneq_f32(vsubq_f32(_r00, _r60), vsubq_f32(_r40, _r20), _coeffs, 0);
                float32x4_t _tmp01 = vfmaq_laneq_f32(vsubq_f32(_r01, _r61), vsubq_f32(_r41, _r21), _coeffs, 0);
                float32x4_t _tmp10 = vaddq_f32(_tmp12a0, _tmp12b0);
                float32x4_t _tmp11 = vaddq_f32(_tmp12a1, _tmp12b1);
                float32x4_t _tmp20 = vsubq_f32(_tmp12a0, _tmp12b0);
                float32x4_t _tmp21 = vsubq_f32(_tmp12a1, _tmp12b1);
                float32x4_t _tmp30 = vaddq_f32(_tmp34a0, _tmp34b0);
                float32x4_t _tmp31 = vaddq_f32(_tmp34a1, _tmp34b1);
                float32x4_t _tmp40 = vsubq_f32(_tmp34a0, _tmp34b0);
                float32x4_t _tmp41 = vsubq_f32(_tmp34a1, _tmp34b1);
                float32x4_t _tmp50 = vaddq_f32(_tmp56a0, _tmp56b0);
                float32x4_t _tmp51 = vaddq_f32(_tmp56a1, _tmp56b1);
                float32x4_t _tmp60 = vsubq_f32(_tmp56a0, _tmp56b0);
                float32x4_t _tmp61 = vsubq_f32(_tmp56a1, _tmp56b1);
                float32x4_t _tmp70 = vfmaq_laneq_f32(vsubq_f32(_r70, _r10), vsubq_f32(_r30, _r50), _coeffs, 0);
                float32x4_t _tmp71 = vfmaq_laneq_f32(vsubq_f32(_r71, _r11), vsubq_f32(_r31, _r51), _coeffs, 0);

                vst1q_f32(p0, _tmp00);
                vst1q_f32(p0 + 4, _tmp01);
                vst1q_f32(p1, _tmp10);
                vst1q_f32(p1 + 4, _tmp11);
                vst1q_f32(p2, _tmp20);
                vst1q_f32(p2 + 4, _tmp21);
                vst1q_f32(p3, _tmp30);
                vst1q_f32(p3 + 4, _tmp31);
                vst1q_f32(p4, _tmp40);
                vst1q_f32(p4 + 4, _tmp41);
                vst1q_f32(p5, _tmp50);
                vst1q_f32(p5 + 4, _tmp51);
                vst1q_f32(p6, _tmp60);
                vst1q_f32(p6 + 4, _tmp61);
                vst1q_f32(p7, _tmp70);
                vst1q_f32(p7 + 4, _tmp71);

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
#else // __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __aarch64__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[8][8][4];

        const float coeffs[8] = {5.25f, -4.25f, -1.25f, 0.25f, -2.5f, 0.5f, 2.f, 4.f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _coeffs2 = vld1q_f32(coeffs + 4);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r0 = vdupq_n_f32(0.f);
                float32x4_t _r1 = vdupq_n_f32(0.f);
                float32x4_t _r2 = vdupq_n_f32(0.f);
                float32x4_t _r3 = vdupq_n_f32(0.f);
                float32x4_t _r4 = vdupq_n_f32(0.f);
                float32x4_t _r5 = vdupq_n_f32(0.f);
                float32x4_t _r6 = vdupq_n_f32(0.f);
                float32x4_t _r7 = vdupq_n_f32(0.f);

                if (ti * 6 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        if (tj * 6 + 1 < w) _r1 = vld1q_f32(r0 + 4);
                        if (tj * 6 + 2 < w) _r2 = vld1q_f32(r0 + 8);
                        if (tj * 6 + 3 < w) _r3 = vld1q_f32(r0 + 12);
                        if (tj * 6 + 4 < w) _r4 = vld1q_f32(r0 + 16);
                        if (tj * 6 + 5 < w) _r5 = vld1q_f32(r0 + 20);
                        if (tj * 6 + 6 < w) _r6 = vld1q_f32(r0 + 24);
                        if (tj * 6 + 7 < w) _r7 = vld1q_f32(r0 + 28);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4_t _t2 = vld1q_f32(r2);
                        float32x4_t _t3 = vld1q_f32(r3);

                        transpose4x4_ps(_t0, _t1, _t2, _t3);

                        _r0 = _t0;
                        if (tj * 6 + 1 < w) _r1 = _t1;
                        if (tj * 6 + 2 < w) _r2 = _t2;
                        if (tj * 6 + 3 < w) _r3 = _t3;
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = vld1q_f32(r0 + 4);
                            _t1 = vld1q_f32(r1 + 4);
                            _t2 = vld1q_f32(r2 + 4);
                            _t3 = vld1q_f32(r3 + 4);

                            transpose4x4_ps(_t0, _t1, _t2, _t3);

                            _r4 = _t0;
                            if (tj * 6 + 5 < w) _r5 = _t1;
                            if (tj * 6 + 6 < w) _r6 = _t2;
                            if (tj * 6 + 7 < w) _r7 = _t3;
                        }
                    }
                }

#if __aarch64__
                float32x4_t _tmp12a = vfmaq_laneq_f32(vaddq_f32(_r2, _r6), _r4, _coeffs, 1);
                float32x4_t _tmp12b = vfmaq_laneq_f32(vaddq_f32(_r1, _r5), _r3, _coeffs, 1);
                float32x4_t _tmp34a = vfmaq_laneq_f32(vfmaq_laneq_f32(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float32x4_t _tmp34b = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs2, 1), _r3, _coeffs2, 0), _r5, _coeffs2, 2);
                float32x4_t _tmp56a = vfmaq_laneq_f32(_r6, vfmaq_laneq_f32(_r2, _r4, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56b = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs2, 2), _r3, _coeffs2, 0), _r5, _coeffs2, 1);
#else
                float32x4_t _tmp12a = vmlaq_lane_f32(vaddq_f32(_r2, _r6), _r4, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp12b = vmlaq_lane_f32(vaddq_f32(_r1, _r5), _r3, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp34a = vmlaq_lane_f32(vmlaq_lane_f32(_r6, _r2, vget_high_f32(_coeffs), 1), _r4, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp34b = vmlaq_lane_f32(vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_low_f32(_coeffs2), 1), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_high_f32(_coeffs2), 0);
                float32x4_t _tmp56a = vmlaq_lane_f32(_r6, vmlaq_lane_f32(_r2, _r4, vget_high_f32(_coeffs), 0), vget_high_f32(_coeffs2), 1);
                float32x4_t _tmp56b = vmlaq_lane_f32(vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_high_f32(_coeffs2), 0), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_low_f32(_coeffs2), 1);
#endif

#if __aarch64__
                float32x4_t _tmp0 = vfmaq_laneq_f32(vsubq_f32(_r0, _r6), vsubq_f32(_r4, _r2), _coeffs, 0);
#else
                float32x4_t _tmp0 = vmlaq_lane_f32(vsubq_f32(_r0, _r6), vsubq_f32(_r4, _r2), vget_low_f32(_coeffs), 0);
#endif
                float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
                float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);
                float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
                float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);
                float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
                float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);
#if __aarch64__
                float32x4_t _tmp7 = vfmaq_laneq_f32(vsubq_f32(_r7, _r1), vsubq_f32(_r3, _r5), _coeffs, 0);
#else
                float32x4_t _tmp7 = vmlaq_lane_f32(vsubq_f32(_r7, _r1), vsubq_f32(_r3, _r5), vget_low_f32(_coeffs), 0);
#endif

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);
                vst1q_f32(tmp[4][m], _tmp4);
                vst1q_f32(tmp[5][m], _tmp5);
                vst1q_f32(tmp[6][m], _tmp6);
                vst1q_f32(tmp[7][m], _tmp7);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;
            float* p4 = p0 + max_jj * 4 * 4;
            float* p5 = p0 + max_jj * 4 * 5;
            float* p6 = p0 + max_jj * 4 * 6;
            float* p7 = p0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);
                float32x4_t _r4 = vld1q_f32(tmp[m][4]);
                float32x4_t _r5 = vld1q_f32(tmp[m][5]);
                float32x4_t _r6 = vld1q_f32(tmp[m][6]);
                float32x4_t _r7 = vld1q_f32(tmp[m][7]);

#if __aarch64__
                float32x4_t _tmp12a = vfmaq_laneq_f32(vaddq_f32(_r2, _r6), _r4, _coeffs, 1);
                float32x4_t _tmp12b = vfmaq_laneq_f32(vaddq_f32(_r1, _r5), _r3, _coeffs, 1);
                float32x4_t _tmp34a = vfmaq_laneq_f32(vfmaq_laneq_f32(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float32x4_t _tmp34b = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs2, 1), _r3, _coeffs2, 0), _r5, _coeffs2, 2);
                float32x4_t _tmp56a = vfmaq_laneq_f32(_r6, vfmaq_laneq_f32(_r2, _r4, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56b = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs2, 2), _r3, _coeffs2, 0), _r5, _coeffs2, 1);
#else
                float32x4_t _tmp12a = vmlaq_lane_f32(vaddq_f32(_r2, _r6), _r4, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp12b = vmlaq_lane_f32(vaddq_f32(_r1, _r5), _r3, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp34a = vmlaq_lane_f32(vmlaq_lane_f32(_r6, _r2, vget_high_f32(_coeffs), 1), _r4, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp34b = vmlaq_lane_f32(vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_low_f32(_coeffs2), 1), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_high_f32(_coeffs2), 0);
                float32x4_t _tmp56a = vmlaq_lane_f32(_r6, vmlaq_lane_f32(_r2, _r4, vget_high_f32(_coeffs), 0), vget_high_f32(_coeffs2), 1);
                float32x4_t _tmp56b = vmlaq_lane_f32(vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_high_f32(_coeffs2), 0), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_low_f32(_coeffs2), 1);
#endif

#if __aarch64__
                float32x4_t _tmp0 = vfmaq_laneq_f32(vsubq_f32(_r0, _r6), vsubq_f32(_r4, _r2), _coeffs, 0);
#else
                float32x4_t _tmp0 = vmlaq_lane_f32(vsubq_f32(_r0, _r6), vsubq_f32(_r4, _r2), vget_low_f32(_coeffs), 0);
#endif
                float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
                float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);
                float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
                float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);
                float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
                float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);
#if __aarch64__
                float32x4_t _tmp7 = vfmaq_laneq_f32(vsubq_f32(_r7, _r1), vsubq_f32(_r3, _r5), _coeffs, 0);
#else
                float32x4_t _tmp7 = vmlaq_lane_f32(vsubq_f32(_r7, _r1), vsubq_f32(_r3, _r5), vget_low_f32(_coeffs), 0);
#endif

                vst1q_f32(p0, _tmp0);
                vst1q_f32(p1, _tmp1);
                vst1q_f32(p2, _tmp2);
                vst1q_f32(p3, _tmp3);
                vst1q_f32(p4, _tmp4);
                vst1q_f32(p5, _tmp5);
                vst1q_f32(p6, _tmp6);
                vst1q_f32(p7, _tmp7);

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
#else // __ARM_NEON
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __ARM_NEON
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[8][8][2];

#if __ARM_NEON
        const float coeffs[8] = {5.25f, -4.25f, -1.25f, 0.25f, -2.5f, 0.5f, 2.f, 4.f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _coeffs2 = vld1q_f32(coeffs + 4);
#endif

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vdup_n_f32(0.f);
                float32x2_t _r1 = vdup_n_f32(0.f);
                float32x2_t _r2 = vdup_n_f32(0.f);
                float32x2_t _r3 = vdup_n_f32(0.f);
                float32x2_t _r4 = vdup_n_f32(0.f);
                float32x2_t _r5 = vdup_n_f32(0.f);
                float32x2_t _r6 = vdup_n_f32(0.f);
                float32x2_t _r7 = vdup_n_f32(0.f);
#else
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
                float r40 = 0.f;
                float r41 = 0.f;
                float r50 = 0.f;
                float r51 = 0.f;
                float r60 = 0.f;
                float r61 = 0.f;
                float r70 = 0.f;
                float r71 = 0.f;
#endif

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

#if __ARM_NEON
                        float32x4_t _t0 = vld1q_f32(r0);
                        float32x4_t _t1 = vld1q_f32(r1);
                        float32x4x2_t _t01 = vzipq_f32(_t0, _t1);

                        _r0 = vget_low_f32(_t01.val[0]);
                        if (tj * 6 + 1 < w) _r1 = vget_high_f32(_t01.val[0]);
                        if (tj * 6 + 2 < w) _r2 = vget_low_f32(_t01.val[1]);
                        if (tj * 6 + 3 < w) _r3 = vget_high_f32(_t01.val[1]);
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = vld1q_f32(r0 + 4);
                            _t1 = vld1q_f32(r1 + 4);
                            _t01 = vzipq_f32(_t0, _t1);

                            _r4 = vget_low_f32(_t01.val[0]);
                            if (tj * 6 + 5 < w) _r5 = vget_high_f32(_t01.val[0]);
                            if (tj * 6 + 6 < w) _r6 = vget_low_f32(_t01.val[1]);
                            if (tj * 6 + 7 < w) _r7 = vget_high_f32(_t01.val[1]);
                        }
#else
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
#endif
                    }
                }

#if __ARM_NEON
#if __aarch64__
                float32x2_t _tmp12a = vfma_laneq_f32(vadd_f32(_r2, _r6), _r4, _coeffs, 1);
                float32x2_t _tmp12b = vfma_laneq_f32(vadd_f32(_r1, _r5), _r3, _coeffs, 1);
                float32x2_t _tmp34a = vfma_laneq_f32(vfma_laneq_f32(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float32x2_t _tmp34b = vfma_laneq_f32(vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs2, 1), _r3, _coeffs2, 0), _r5, _coeffs2, 2);
                float32x2_t _tmp56a = vfma_laneq_f32(_r6, vfma_laneq_f32(_r2, _r4, _coeffs, 2), _coeffs2, 3);
                float32x2_t _tmp56b = vfma_laneq_f32(vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs2, 2), _r3, _coeffs2, 0), _r5, _coeffs2, 1);
#else
                float32x2_t _tmp12a = vmla_lane_f32(vadd_f32(_r2, _r6), _r4, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp12b = vmla_lane_f32(vadd_f32(_r1, _r5), _r3, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp34a = vmla_lane_f32(vmla_lane_f32(_r6, _r2, vget_high_f32(_coeffs), 1), _r4, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp34b = vmla_lane_f32(vmla_lane_f32(vmul_lane_f32(_r1, vget_low_f32(_coeffs2), 1), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_high_f32(_coeffs2), 0);
                float32x2_t _tmp56a = vmla_lane_f32(_r6, vmla_lane_f32(_r2, _r4, vget_high_f32(_coeffs), 0), vget_high_f32(_coeffs2), 1);
                float32x2_t _tmp56b = vmla_lane_f32(vmla_lane_f32(vmul_lane_f32(_r1, vget_high_f32(_coeffs2), 0), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_low_f32(_coeffs2), 1);
#endif

#if __aarch64__
                float32x2_t _tmp0 = vfma_laneq_f32(vsub_f32(_r0, _r6), vsub_f32(_r4, _r2), _coeffs, 0);
#else
                float32x2_t _tmp0 = vmla_lane_f32(vsub_f32(_r0, _r6), vsub_f32(_r4, _r2), vget_low_f32(_coeffs), 0);
#endif
                float32x2_t _tmp1 = vadd_f32(_tmp12a, _tmp12b);
                float32x2_t _tmp2 = vsub_f32(_tmp12a, _tmp12b);
                float32x2_t _tmp3 = vadd_f32(_tmp34a, _tmp34b);
                float32x2_t _tmp4 = vsub_f32(_tmp34a, _tmp34b);
                float32x2_t _tmp5 = vadd_f32(_tmp56a, _tmp56b);
                float32x2_t _tmp6 = vsub_f32(_tmp56a, _tmp56b);
#if __aarch64__
                float32x2_t _tmp7 = vfma_laneq_f32(vsub_f32(_r7, _r1), vsub_f32(_r3, _r5), _coeffs, 0);
#else
                float32x2_t _tmp7 = vmla_lane_f32(vsub_f32(_r7, _r1), vsub_f32(_r3, _r5), vget_low_f32(_coeffs), 0);
#endif

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
                vst1_f32(tmp[4][m], _tmp4);
                vst1_f32(tmp[5][m], _tmp5);
                vst1_f32(tmp[6][m], _tmp6);
                vst1_f32(tmp[7][m], _tmp7);
#else
                float tmp12a0 = r20 + r60 - r40 * 4.25f;
                float tmp12a1 = r21 + r61 - r41 * 4.25f;
                float tmp12b0 = r10 + r50 - r30 * 4.25f;
                float tmp12b1 = r11 + r51 - r31 * 4.25f;
                float tmp34a0 = r60 + r20 * 0.25f - r40 * 1.25f;
                float tmp34a1 = r61 + r21 * 0.25f - r41 * 1.25f;
                float tmp34b0 = r10 * 0.5f - r30 * 2.5f + r50 * 2.f;
                float tmp34b1 = r11 * 0.5f - r31 * 2.5f + r51 * 2.f;
                float tmp56a0 = r20 * 4.f - r40 * 5.f + r60;
                float tmp56a1 = r21 * 4.f - r41 * 5.f + r61;
                float tmp56b0 = r10 * 2.f - r30 * 2.5f + r50 * 0.5f;
                float tmp56b1 = r11 * 2.f - r31 * 2.5f + r51 * 0.5f;

                tmp[0][m][0] = r00 - r60 + (r40 - r20) * 5.25f;
                tmp[0][m][1] = r01 - r61 + (r41 - r21) * 5.25f;
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
                tmp[7][m][0] = r70 - r10 + (r30 - r50) * 5.25f;
                tmp[7][m][1] = r71 - r11 + (r31 - r51) * 5.25f;
#endif

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;
            float* p4 = p0 + max_jj * 2 * 4;
            float* p5 = p0 + max_jj * 2 * 5;
            float* p6 = p0 + max_jj * 2 * 6;
            float* p7 = p0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);
                float32x2_t _r4 = vld1_f32(tmp[m][4]);
                float32x2_t _r5 = vld1_f32(tmp[m][5]);
                float32x2_t _r6 = vld1_f32(tmp[m][6]);
                float32x2_t _r7 = vld1_f32(tmp[m][7]);

#if __aarch64__
                float32x2_t _tmp12a = vfma_laneq_f32(vadd_f32(_r2, _r6), _r4, _coeffs, 1);
                float32x2_t _tmp12b = vfma_laneq_f32(vadd_f32(_r1, _r5), _r3, _coeffs, 1);
                float32x2_t _tmp34a = vfma_laneq_f32(vfma_laneq_f32(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float32x2_t _tmp34b = vfma_laneq_f32(vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs2, 1), _r3, _coeffs2, 0), _r5, _coeffs2, 2);
                float32x2_t _tmp56a = vfma_laneq_f32(_r6, vfma_laneq_f32(_r2, _r4, _coeffs, 2), _coeffs2, 3);
                float32x2_t _tmp56b = vfma_laneq_f32(vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs2, 2), _r3, _coeffs2, 0), _r5, _coeffs2, 1);
#else
                float32x2_t _tmp12a = vmla_lane_f32(vadd_f32(_r2, _r6), _r4, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp12b = vmla_lane_f32(vadd_f32(_r1, _r5), _r3, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp34a = vmla_lane_f32(vmla_lane_f32(_r6, _r2, vget_high_f32(_coeffs), 1), _r4, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp34b = vmla_lane_f32(vmla_lane_f32(vmul_lane_f32(_r1, vget_low_f32(_coeffs2), 1), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_high_f32(_coeffs2), 0);
                float32x2_t _tmp56a = vmla_lane_f32(_r6, vmla_lane_f32(_r2, _r4, vget_high_f32(_coeffs), 0), vget_high_f32(_coeffs2), 1);
                float32x2_t _tmp56b = vmla_lane_f32(vmla_lane_f32(vmul_lane_f32(_r1, vget_high_f32(_coeffs2), 0), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_low_f32(_coeffs2), 1);
#endif

#if __aarch64__
                float32x2_t _tmp0 = vfma_laneq_f32(vsub_f32(_r0, _r6), vsub_f32(_r4, _r2), _coeffs, 0);
#else
                float32x2_t _tmp0 = vmla_lane_f32(vsub_f32(_r0, _r6), vsub_f32(_r4, _r2), vget_low_f32(_coeffs), 0);
#endif
                float32x2_t _tmp1 = vadd_f32(_tmp12a, _tmp12b);
                float32x2_t _tmp2 = vsub_f32(_tmp12a, _tmp12b);
                float32x2_t _tmp3 = vadd_f32(_tmp34a, _tmp34b);
                float32x2_t _tmp4 = vsub_f32(_tmp34a, _tmp34b);
                float32x2_t _tmp5 = vadd_f32(_tmp56a, _tmp56b);
                float32x2_t _tmp6 = vsub_f32(_tmp56a, _tmp56b);
#if __aarch64__
                float32x2_t _tmp7 = vfma_laneq_f32(vsub_f32(_r7, _r1), vsub_f32(_r3, _r5), _coeffs, 0);
#else
                float32x2_t _tmp7 = vmla_lane_f32(vsub_f32(_r7, _r1), vsub_f32(_r3, _r5), vget_low_f32(_coeffs), 0);
#endif

                vst1_f32(p0, _tmp0);
                vst1_f32(p1, _tmp1);
                vst1_f32(p2, _tmp2);
                vst1_f32(p3, _tmp3);
                vst1_f32(p4, _tmp4);
                vst1_f32(p5, _tmp5);
                vst1_f32(p6, _tmp6);
                vst1_f32(p7, _tmp7);
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];
                float r60 = tmp[m][6][0];
                float r61 = tmp[m][6][1];
                float r70 = tmp[m][7][0];
                float r71 = tmp[m][7][1];

                float tmp12a0 = r20 + r60 - r40 * 4.25f;
                float tmp12a1 = r21 + r61 - r41 * 4.25f;
                float tmp12b0 = r10 + r50 - r30 * 4.25f;
                float tmp12b1 = r11 + r51 - r31 * 4.25f;
                float tmp34a0 = r60 + r20 * 0.25f - r40 * 1.25f;
                float tmp34a1 = r61 + r21 * 0.25f - r41 * 1.25f;
                float tmp34b0 = r10 * 0.5f - r30 * 2.5f + r50 * 2.f;
                float tmp34b1 = r11 * 0.5f - r31 * 2.5f + r51 * 2.f;
                float tmp56a0 = r20 * 4.f - r40 * 5.f + r60;
                float tmp56a1 = r21 * 4.f - r41 * 5.f + r61;
                float tmp56b0 = r10 * 2.f - r30 * 2.5f + r50 * 0.5f;
                float tmp56b1 = r11 * 2.f - r31 * 2.5f + r51 * 0.5f;

                p0[0] = r00 - r60 + (r40 - r20) * 5.25f;
                p0[1] = r01 - r61 + (r41 - r21) * 5.25f;
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
                p7[0] = r70 - r10 + (r30 - r50) * 5.25f;
                p7[1] = r71 - r11 + (r31 - r51) * 5.25f;
#endif

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
        float tmp[8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;
                float r4 = 0.f;
                float r5 = 0.f;
                float r6 = 0.f;
                float r7 = 0.f;

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

                float tmp12a = r2 + r6 - r4 * 4.25f;
                float tmp12b = r1 + r5 - r3 * 4.25f;
                float tmp34a = r6 + r2 * 0.25f - r4 * 1.25f;
                float tmp34b = r1 * 0.5f - r3 * 2.5f + r5 * 2.f;
                float tmp56a = r2 * 4.f - r4 * 5.f + r6;
                float tmp56b = r1 * 2.f - r3 * 2.5f + r5 * 0.5f;

                tmp[0][m] = r0 - r6 + (r4 - r2) * 5.25f;
                tmp[1][m] = tmp12a + tmp12b;
                tmp[2][m] = tmp12a - tmp12b;
                tmp[3][m] = tmp34a + tmp34b;
                tmp[4][m] = tmp34a - tmp34b;
                tmp[5][m] = tmp56a + tmp56b;
                tmp[6][m] = tmp56a - tmp56b;
                tmp[7][m] = r7 - r1 + (r3 - r5) * 5.25f;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;
            float* p4 = p0 + max_jj * 4;
            float* p5 = p0 + max_jj * 5;
            float* p6 = p0 + max_jj * 6;
            float* p7 = p0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];
                float r6 = tmp[m][6];
                float r7 = tmp[m][7];

                float tmp12a = r2 + r6 - r4 * 4.25f;
                float tmp12b = r1 + r5 - r3 * 4.25f;
                float tmp34a = r6 + r2 * 0.25f - r4 * 1.25f;
                float tmp34b = r1 * 0.5f - r3 * 2.5f + r5 * 2.f;
                float tmp56a = r2 * 4.f - r4 * 5.f + r6;
                float tmp56b = r1 * 2.f - r3 * 2.5f + r5 * 0.5f;

                p0[0] = r0 - r6 + (r4 - r2) * 5.25f;
                p1[0] = tmp12a + tmp12b;
                p2[0] = tmp12a - tmp12b;
                p3[0] = tmp34a + tmp34b;
                p4[0] = tmp34a - tmp34b;
                p5[0] = tmp56a + tmp56b;
                p6[0] = tmp56a - tmp56b;
                p7[0] = r7 - r1 + (r3 - r5) * 5.25f;

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

static inline void conv3x3s1_winograd63_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[6][8] = {
    //     {1.0f, 1.0f,  1.0f,  1.0f,  1.0f, 32.0f, 32.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  2.0f, -2.0f, 16.0f,-16.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f,  4.0f,  4.0f,  8.0f,  8.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  8.0f, -8.0f,  4.0f, -4.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f, 16.0f, 16.0f,  2.0f,  2.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 32.0f,-32.0f,  1.0f, -1.0f, 1.0f}
    // };

#if __ARM_NEON
    const float coeffs[4] = {32.f, 16.f, 8.f, 4.f};
    float32x4_t _coeffs = vld1q_f32(coeffs);
    float32x2_t _v2 = vdup_n_f32(2.f);
#endif

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 5) / 6;

    const float* biasptr = bias;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = biasptr ? vld1q_f32(biasptr + i + ii + 4) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;
            const float* r4 = r0 + max_jj * 8 * 4;
            const float* r5 = r0 + max_jj * 8 * 5;
            const float* r6 = r0 + max_jj * 8 * 6;
            const float* r7 = r0 + max_jj * 8 * 7;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r50 = vld1q_f32(r5);
                float32x4_t _r51 = vld1q_f32(r5 + 4);
                float32x4_t _r60 = vld1q_f32(r6);
                float32x4_t _r61 = vld1q_f32(r6 + 4);
                float32x4_t _r70 = vld1q_f32(r7);
                float32x4_t _r71 = vld1q_f32(r7 + 4);

                float32x4_t _tmp024a0 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp024a1 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp135a0 = vsubq_f32(_r10, _r20);
                float32x4_t _tmp135a1 = vsubq_f32(_r11, _r21);
                float32x4_t _tmp024b0 = vaddq_f32(_r30, _r40);
                float32x4_t _tmp024b1 = vaddq_f32(_r31, _r41);
                float32x4_t _tmp135b0 = vsubq_f32(_r30, _r40);
                float32x4_t _tmp135b1 = vsubq_f32(_r31, _r41);
                float32x4_t _tmp024c0 = vaddq_f32(_r50, _r60);
                float32x4_t _tmp024c1 = vaddq_f32(_r51, _r61);
                float32x4_t _tmp135c0 = vsubq_f32(_r50, _r60);
                float32x4_t _tmp135c1 = vsubq_f32(_r51, _r61);

                float32x4_t _tmp00 = vaddq_f32(vaddq_f32(_r00, _tmp024a0), vfmaq_laneq_f32(_tmp024b0, _tmp024c0, _coeffs, 0));
                float32x4_t _tmp01 = vaddq_f32(vaddq_f32(_r01, _tmp024a1), vfmaq_laneq_f32(_tmp024b1, _tmp024c1, _coeffs, 0));
                float32x4_t _tmp10 = vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a0, _tmp135b0, _v2, 0), _tmp135c0, _coeffs, 1);
                float32x4_t _tmp11 = vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a1, _tmp135b1, _v2, 0), _tmp135c1, _coeffs, 1);
                float32x4_t _tmp20 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a0, _tmp024b0, _coeffs, 3), _tmp024c0, _coeffs, 2);
                float32x4_t _tmp21 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a1, _tmp024b1, _coeffs, 3), _tmp024c1, _coeffs, 2);
                float32x4_t _tmp30 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a0, _tmp135b0, _coeffs, 2), _tmp135c0, _coeffs, 3);
                float32x4_t _tmp31 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a1, _tmp135b1, _coeffs, 2), _tmp135c1, _coeffs, 3);
                float32x4_t _tmp40 = vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a0, _tmp024b0, _coeffs, 1), _tmp024c0, _v2, 0);
                float32x4_t _tmp41 = vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a1, _tmp024b1, _coeffs, 1), _tmp024c1, _v2, 0);
                float32x4_t _tmp50 = vaddq_f32(vaddq_f32(_r70, _tmp135a0), vfmaq_laneq_f32(_tmp135c0, _tmp135b0, _coeffs, 0));
                float32x4_t _tmp51 = vaddq_f32(vaddq_f32(_r71, _tmp135a1), vfmaq_laneq_f32(_tmp135c1, _tmp135b1, _coeffs, 0));

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);
                vst1q_f32(tmp[4][m], _tmp40);
                vst1q_f32(tmp[4][m] + 4, _tmp41);
                vst1q_f32(tmp[5][m], _tmp50);
                vst1q_f32(tmp[5][m] + 4, _tmp51);

                r0 += max_jj * 8 * 8;
                r1 += max_jj * 8 * 8;
                r2 += max_jj * 8 * 8;
                r3 += max_jj * 8 * 8;
                r4 += max_jj * 8 * 8;
                r5 += max_jj * 8 * 8;
                r6 += max_jj * 8 * 8;
                r7 += max_jj * 8 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);
                float32x4_t _r40 = vld1q_f32(tmp[m][4]);
                float32x4_t _r41 = vld1q_f32(tmp[m][4] + 4);
                float32x4_t _r50 = vld1q_f32(tmp[m][5]);
                float32x4_t _r51 = vld1q_f32(tmp[m][5] + 4);
                float32x4_t _r60 = vld1q_f32(tmp[m][6]);
                float32x4_t _r61 = vld1q_f32(tmp[m][6] + 4);
                float32x4_t _r70 = vld1q_f32(tmp[m][7]);
                float32x4_t _r71 = vld1q_f32(tmp[m][7] + 4);

                float32x4_t _tmp024a0 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp024a1 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp135a0 = vsubq_f32(_r10, _r20);
                float32x4_t _tmp135a1 = vsubq_f32(_r11, _r21);
                float32x4_t _tmp024b0 = vaddq_f32(_r30, _r40);
                float32x4_t _tmp024b1 = vaddq_f32(_r31, _r41);
                float32x4_t _tmp135b0 = vsubq_f32(_r30, _r40);
                float32x4_t _tmp135b1 = vsubq_f32(_r31, _r41);
                float32x4_t _tmp024c0 = vaddq_f32(_r50, _r60);
                float32x4_t _tmp024c1 = vaddq_f32(_r51, _r61);
                float32x4_t _tmp135c0 = vsubq_f32(_r50, _r60);
                float32x4_t _tmp135c1 = vsubq_f32(_r51, _r61);

                float32x4_t _tmp00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r00, _tmp024a0), vfmaq_laneq_f32(_tmp024b0, _tmp024c0, _coeffs, 0)));
                float32x4_t _tmp01 = vaddq_f32(_bias1, vaddq_f32(vaddq_f32(_r01, _tmp024a1), vfmaq_laneq_f32(_tmp024b1, _tmp024c1, _coeffs, 0)));
                float32x4_t _tmp10 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a0, _tmp135b0, _v2, 0), _tmp135c0, _coeffs, 1));
                float32x4_t _tmp11 = vaddq_f32(_bias1, vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a1, _tmp135b1, _v2, 0), _tmp135c1, _coeffs, 1));
                float32x4_t _tmp20 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a0, _tmp024b0, _coeffs, 3), _tmp024c0, _coeffs, 2));
                float32x4_t _tmp21 = vaddq_f32(_bias1, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a1, _tmp024b1, _coeffs, 3), _tmp024c1, _coeffs, 2));
                float32x4_t _tmp30 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a0, _tmp135b0, _coeffs, 2), _tmp135c0, _coeffs, 3));
                float32x4_t _tmp31 = vaddq_f32(_bias1, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a1, _tmp135b1, _coeffs, 2), _tmp135c1, _coeffs, 3));
                float32x4_t _tmp40 = vaddq_f32(_bias0, vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a0, _tmp024b0, _coeffs, 1), _tmp024c0, _v2, 0));
                float32x4_t _tmp41 = vaddq_f32(_bias1, vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a1, _tmp024b1, _coeffs, 1), _tmp024c1, _v2, 0));
                float32x4_t _tmp50 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r70, _tmp135a0), vfmaq_laneq_f32(_tmp135c0, _tmp135b0, _coeffs, 0)));
                float32x4_t _tmp51 = vaddq_f32(_bias1, vaddq_f32(vaddq_f32(_r71, _tmp135a1), vfmaq_laneq_f32(_tmp135c1, _tmp135b1, _coeffs, 0)));

                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    vst1q_f32(outptr0, _tmp00);
                    vst1q_f32(outptr1, _tmp01);
                    if (tj * 6 + 1 < outw)
                    {
                        vst1q_f32(outptr0 + 4, _tmp10);
                        vst1q_f32(outptr1 + 4, _tmp11);
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        vst1q_f32(outptr0 + 8, _tmp20);
                        vst1q_f32(outptr1 + 8, _tmp21);
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        vst1q_f32(outptr0 + 12, _tmp30);
                        vst1q_f32(outptr1 + 12, _tmp31);
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        vst1q_f32(outptr0 + 16, _tmp40);
                        vst1q_f32(outptr1 + 16, _tmp41);
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        vst1q_f32(outptr0 + 20, _tmp50);
                        vst1q_f32(outptr1 + 20, _tmp51);
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    float tmp2[8];
                    float tmp3[8];
                    float tmp4[8];
                    float tmp5[8];
                    vst1q_f32(tmp0, _tmp00);
                    vst1q_f32(tmp0 + 4, _tmp01);
                    vst1q_f32(tmp1, _tmp10);
                    vst1q_f32(tmp1 + 4, _tmp11);
                    vst1q_f32(tmp2, _tmp20);
                    vst1q_f32(tmp2 + 4, _tmp21);
                    vst1q_f32(tmp3, _tmp30);
                    vst1q_f32(tmp3 + 4, _tmp31);
                    vst1q_f32(tmp4, _tmp40);
                    vst1q_f32(tmp4 + 4, _tmp41);
                    vst1q_f32(tmp5, _tmp50);
                    vst1q_f32(tmp5 + 4, _tmp51);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][8][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;
            const float* r4 = r0 + max_jj * 4 * 4;
            const float* r5 = r0 + max_jj * 4 * 5;
            const float* r6 = r0 + max_jj * 4 * 6;
            const float* r7 = r0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _r1 = vld1q_f32(r1);
                float32x4_t _r2 = vld1q_f32(r2);
                float32x4_t _r3 = vld1q_f32(r3);
                float32x4_t _r4 = vld1q_f32(r4);
                float32x4_t _r5 = vld1q_f32(r5);
                float32x4_t _r6 = vld1q_f32(r6);
                float32x4_t _r7 = vld1q_f32(r7);

                float32x4_t _tmp024a = vaddq_f32(_r1, _r2);
                float32x4_t _tmp135a = vsubq_f32(_r1, _r2);
                float32x4_t _tmp024b = vaddq_f32(_r3, _r4);
                float32x4_t _tmp135b = vsubq_f32(_r3, _r4);
                float32x4_t _tmp024c = vaddq_f32(_r5, _r6);
                float32x4_t _tmp135c = vsubq_f32(_r5, _r6);

#if __aarch64__
                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _tmp024a), vfmaq_laneq_f32(_tmp024b, _tmp024c, _coeffs, 0));
                float32x4_t _tmp1 = vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a, _tmp135b, _v2, 0), _tmp135c, _coeffs, 1);
                float32x4_t _tmp2 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2);
                float32x4_t _tmp3 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3);
                float32x4_t _tmp4 = vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _v2, 0);
                float32x4_t _tmp5 = vaddq_f32(vaddq_f32(_r7, _tmp135a), vfmaq_laneq_f32(_tmp135c, _tmp135b, _coeffs, 0));
#else
                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _tmp024a), vmlaq_lane_f32(_tmp024b, _tmp024c, vget_low_f32(_coeffs), 0));
                float32x4_t _tmp1 = vmlaq_lane_f32(vmlaq_lane_f32(_tmp135a, _tmp135b, _v2, 0), _tmp135c, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp2 = vmlaq_lane_f32(vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeffs), 1), _tmp024c, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp3 = vmlaq_lane_f32(vmlaq_lane_f32(_tmp135a, _tmp135b, vget_high_f32(_coeffs), 0), _tmp135c, vget_high_f32(_coeffs), 1);
                float32x4_t _tmp4 = vmlaq_lane_f32(vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeffs), 1), _tmp024c, _v2, 0);
                float32x4_t _tmp5 = vaddq_f32(vaddq_f32(_r7, _tmp135a), vmlaq_lane_f32(_tmp135c, _tmp135b, vget_low_f32(_coeffs), 0));
#endif

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);
                vst1q_f32(tmp[4][m], _tmp4);
                vst1q_f32(tmp[5][m], _tmp5);

                r0 += max_jj * 8 * 4;
                r1 += max_jj * 8 * 4;
                r2 += max_jj * 8 * 4;
                r3 += max_jj * 8 * 4;
                r4 += max_jj * 8 * 4;
                r5 += max_jj * 8 * 4;
                r6 += max_jj * 8 * 4;
                r7 += max_jj * 8 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);
                float32x4_t _r4 = vld1q_f32(tmp[m][4]);
                float32x4_t _r5 = vld1q_f32(tmp[m][5]);
                float32x4_t _r6 = vld1q_f32(tmp[m][6]);
                float32x4_t _r7 = vld1q_f32(tmp[m][7]);

                float32x4_t _tmp024a = vaddq_f32(_r1, _r2);
                float32x4_t _tmp135a = vsubq_f32(_r1, _r2);
                float32x4_t _tmp024b = vaddq_f32(_r3, _r4);
                float32x4_t _tmp135b = vsubq_f32(_r3, _r4);
                float32x4_t _tmp024c = vaddq_f32(_r5, _r6);
                float32x4_t _tmp135c = vsubq_f32(_r5, _r6);

#if __aarch64__
                float32x4_t _tmp0 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r0, _tmp024a), vfmaq_laneq_f32(_tmp024b, _tmp024c, _coeffs, 0)));
                float32x4_t _tmp1 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a, _tmp135b, _v2, 0), _tmp135c, _coeffs, 1));
                float32x4_t _tmp2 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2));
                float32x4_t _tmp3 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3));
                float32x4_t _tmp4 = vaddq_f32(_bias0, vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _v2, 0));
                float32x4_t _tmp5 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r7, _tmp135a), vfmaq_laneq_f32(_tmp135c, _tmp135b, _coeffs, 0)));
#else
                float32x4_t _tmp0 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r0, _tmp024a), vmlaq_lane_f32(_tmp024b, _tmp024c, vget_low_f32(_coeffs), 0)));
                float32x4_t _tmp1 = vaddq_f32(_bias0, vmlaq_lane_f32(vmlaq_lane_f32(_tmp135a, _tmp135b, _v2, 0), _tmp135c, vget_low_f32(_coeffs), 1));
                float32x4_t _tmp2 = vaddq_f32(_bias0, vmlaq_lane_f32(vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeffs), 1), _tmp024c, vget_high_f32(_coeffs), 0));
                float32x4_t _tmp3 = vaddq_f32(_bias0, vmlaq_lane_f32(vmlaq_lane_f32(_tmp135a, _tmp135b, vget_high_f32(_coeffs), 0), _tmp135c, vget_high_f32(_coeffs), 1));
                float32x4_t _tmp4 = vaddq_f32(_bias0, vmlaq_lane_f32(vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeffs), 1), _tmp024c, _v2, 0));
                float32x4_t _tmp5 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r7, _tmp135a), vmlaq_lane_f32(_tmp135c, _tmp135b, vget_low_f32(_coeffs), 0)));
#endif

                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _tmp0);
                    if (tj * 6 + 1 < outw) vst1q_f32(outptr0 + 4, _tmp1);
                    if (tj * 6 + 2 < outw) vst1q_f32(outptr0 + 8, _tmp2);
                    if (tj * 6 + 3 < outw) vst1q_f32(outptr0 + 12, _tmp3);
                    if (tj * 6 + 4 < outw) vst1q_f32(outptr0 + 16, _tmp4);
                    if (tj * 6 + 5 < outw) vst1q_f32(outptr0 + 20, _tmp5);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    float tmp2[4];
                    float tmp3[4];
                    float tmp4[4];
                    float tmp5[4];
                    vst1q_f32(tmp0, _tmp0);
                    vst1q_f32(tmp1, _tmp1);
                    vst1q_f32(tmp2, _tmp2);
                    vst1q_f32(tmp3, _tmp3);
                    vst1q_f32(tmp4, _tmp4);
                    vst1q_f32(tmp5, _tmp5);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        float32x2_t _bias0 = biasptr ? vld1_f32(biasptr + i + ii) : vdup_n_f32(0.f);
#else
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;
#endif

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[6][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;
            const float* r4 = r0 + max_jj * 2 * 4;
            const float* r5 = r0 + max_jj * 2 * 5;
            const float* r6 = r0 + max_jj * 2 * 6;
            const float* r7 = r0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(r0);
                float32x2_t _r1 = vld1_f32(r1);
                float32x2_t _r2 = vld1_f32(r2);
                float32x2_t _r3 = vld1_f32(r3);
                float32x2_t _r4 = vld1_f32(r4);
                float32x2_t _r5 = vld1_f32(r5);
                float32x2_t _r6 = vld1_f32(r6);
                float32x2_t _r7 = vld1_f32(r7);

                float32x2_t _tmp024a = vadd_f32(_r1, _r2);
                float32x2_t _tmp135a = vsub_f32(_r1, _r2);
                float32x2_t _tmp024b = vadd_f32(_r3, _r4);
                float32x2_t _tmp135b = vsub_f32(_r3, _r4);
                float32x2_t _tmp024c = vadd_f32(_r5, _r6);
                float32x2_t _tmp135c = vsub_f32(_r5, _r6);

#if __aarch64__
                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _tmp024a), vfma_laneq_f32(_tmp024b, _tmp024c, _coeffs, 0));
                float32x2_t _tmp1 = vfma_laneq_f32(vfma_f32(_tmp135a, _tmp135b, _v2), _tmp135c, _coeffs, 1);
                float32x2_t _tmp2 = vfma_laneq_f32(vfma_laneq_f32(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2);
                float32x2_t _tmp3 = vfma_laneq_f32(vfma_laneq_f32(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3);
                float32x2_t _tmp4 = vfma_f32(vfma_laneq_f32(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _v2);
                float32x2_t _tmp5 = vadd_f32(vadd_f32(_r7, _tmp135a), vfma_laneq_f32(_tmp135c, _tmp135b, _coeffs, 0));
#else
                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _tmp024a), vmla_lane_f32(_tmp024b, _tmp024c, vget_low_f32(_coeffs), 0));
                float32x2_t _tmp1 = vmla_lane_f32(vmla_f32(_tmp135a, _tmp135b, _v2), _tmp135c, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp2 = vmla_lane_f32(vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeffs), 1), _tmp024c, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp3 = vmla_lane_f32(vmla_lane_f32(_tmp135a, _tmp135b, vget_high_f32(_coeffs), 0), _tmp135c, vget_high_f32(_coeffs), 1);
                float32x2_t _tmp4 = vmla_f32(vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeffs), 1), _tmp024c, _v2);
                float32x2_t _tmp5 = vadd_f32(vadd_f32(_r7, _tmp135a), vmla_lane_f32(_tmp135c, _tmp135b, vget_low_f32(_coeffs), 0));
#endif

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
                vst1_f32(tmp[4][m], _tmp4);
                vst1_f32(tmp[5][m], _tmp5);
#else
                float tmp024a0 = r1[0] + r2[0];
                float tmp024a1 = r1[1] + r2[1];
                float tmp135a0 = r1[0] - r2[0];
                float tmp135a1 = r1[1] - r2[1];
                float tmp024b0 = r3[0] + r4[0];
                float tmp024b1 = r3[1] + r4[1];
                float tmp135b0 = r3[0] - r4[0];
                float tmp135b1 = r3[1] - r4[1];
                float tmp024c0 = r5[0] + r6[0];
                float tmp024c1 = r5[1] + r6[1];
                float tmp135c0 = r5[0] - r6[0];
                float tmp135c1 = r5[1] - r6[1];

                tmp[0][m][0] = r0[0] + tmp024a0 + tmp024b0 + tmp024c0 * 32;
                tmp[0][m][1] = r0[1] + tmp024a1 + tmp024b1 + tmp024c1 * 32;
                tmp[1][m][0] = tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * 16;
                tmp[1][m][1] = tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * 16;
                tmp[2][m][0] = tmp024a0 + tmp024b0 * 4 + tmp024c0 * 8;
                tmp[2][m][1] = tmp024a1 + tmp024b1 * 4 + tmp024c1 * 8;
                tmp[3][m][0] = tmp135a0 + tmp135b0 * 8 + tmp135c0 * 4;
                tmp[3][m][1] = tmp135a1 + tmp135b1 * 8 + tmp135c1 * 4;
                tmp[4][m][0] = tmp024a0 + tmp024b0 * 16 + tmp024c0 + tmp024c0;
                tmp[4][m][1] = tmp024a1 + tmp024b1 * 16 + tmp024c1 + tmp024c1;
                tmp[5][m][0] = r7[0] + tmp135a0 + tmp135b0 * 32 + tmp135c0;
                tmp[5][m][1] = r7[1] + tmp135a1 + tmp135b1 * 32 + tmp135c1;
#endif

                r0 += max_jj * 8 * 2;
                r1 += max_jj * 8 * 2;
                r2 += max_jj * 8 * 2;
                r3 += max_jj * 8 * 2;
                r4 += max_jj * 8 * 2;
                r5 += max_jj * 8 * 2;
                r6 += max_jj * 8 * 2;
                r7 += max_jj * 8 * 2;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);
                float32x2_t _r4 = vld1_f32(tmp[m][4]);
                float32x2_t _r5 = vld1_f32(tmp[m][5]);
                float32x2_t _r6 = vld1_f32(tmp[m][6]);
                float32x2_t _r7 = vld1_f32(tmp[m][7]);

                float32x2_t _tmp024a = vadd_f32(_r1, _r2);
                float32x2_t _tmp135a = vsub_f32(_r1, _r2);
                float32x2_t _tmp024b = vadd_f32(_r3, _r4);
                float32x2_t _tmp135b = vsub_f32(_r3, _r4);
                float32x2_t _tmp024c = vadd_f32(_r5, _r6);
                float32x2_t _tmp135c = vsub_f32(_r5, _r6);

#if __aarch64__
                float32x2_t _tmp0 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r0, _tmp024a), vfma_laneq_f32(_tmp024b, _tmp024c, _coeffs, 0)));
                float32x2_t _tmp1 = vadd_f32(_bias0, vfma_laneq_f32(vfma_f32(_tmp135a, _tmp135b, _v2), _tmp135c, _coeffs, 1));
                float32x2_t _tmp2 = vadd_f32(_bias0, vfma_laneq_f32(vfma_laneq_f32(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2));
                float32x2_t _tmp3 = vadd_f32(_bias0, vfma_laneq_f32(vfma_laneq_f32(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3));
                float32x2_t _tmp4 = vadd_f32(_bias0, vfma_f32(vfma_laneq_f32(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _v2));
                float32x2_t _tmp5 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r7, _tmp135a), vfma_laneq_f32(_tmp135c, _tmp135b, _coeffs, 0)));
#else
                float32x2_t _tmp0 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r0, _tmp024a), vmla_lane_f32(_tmp024b, _tmp024c, vget_low_f32(_coeffs), 0)));
                float32x2_t _tmp1 = vadd_f32(_bias0, vmla_lane_f32(vmla_f32(_tmp135a, _tmp135b, _v2), _tmp135c, vget_low_f32(_coeffs), 1));
                float32x2_t _tmp2 = vadd_f32(_bias0, vmla_lane_f32(vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeffs), 1), _tmp024c, vget_high_f32(_coeffs), 0));
                float32x2_t _tmp3 = vadd_f32(_bias0, vmla_lane_f32(vmla_lane_f32(_tmp135a, _tmp135b, vget_high_f32(_coeffs), 0), _tmp135c, vget_high_f32(_coeffs), 1));
                float32x2_t _tmp4 = vadd_f32(_bias0, vmla_f32(vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeffs), 1), _tmp024c, _v2));
                float32x2_t _tmp5 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r7, _tmp135a), vmla_lane_f32(_tmp135c, _tmp135b, vget_low_f32(_coeffs), 0)));
#endif
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];
                float r60 = tmp[m][6][0];
                float r61 = tmp[m][6][1];
                float r70 = tmp[m][7][0];
                float r71 = tmp[m][7][1];

                float tmp024a0 = r10 + r20;
                float tmp024a1 = r11 + r21;
                float tmp135a0 = r10 - r20;
                float tmp135a1 = r11 - r21;
                float tmp024b0 = r30 + r40;
                float tmp024b1 = r31 + r41;
                float tmp135b0 = r30 - r40;
                float tmp135b1 = r31 - r41;
                float tmp024c0 = r50 + r60;
                float tmp024c1 = r51 + r61;
                float tmp135c0 = r50 - r60;
                float tmp135c1 = r51 - r61;

                float tmp00 = bias0 + r00 + tmp024a0 + tmp024b0 + tmp024c0 * 32;
                float tmp01 = bias1 + r01 + tmp024a1 + tmp024b1 + tmp024c1 * 32;
                float tmp10 = bias0 + tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * 16;
                float tmp11 = bias1 + tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * 16;
                float tmp20 = bias0 + tmp024a0 + tmp024b0 * 4 + tmp024c0 * 8;
                float tmp21 = bias1 + tmp024a1 + tmp024b1 * 4 + tmp024c1 * 8;
                float tmp30 = bias0 + tmp135a0 + tmp135b0 * 8 + tmp135c0 * 4;
                float tmp31 = bias1 + tmp135a1 + tmp135b1 * 8 + tmp135c1 * 4;
                float tmp40 = bias0 + tmp024a0 + tmp024b0 * 16 + tmp024c0 + tmp024c0;
                float tmp41 = bias1 + tmp024a1 + tmp024b1 * 16 + tmp024c1 + tmp024c1;
                float tmp50 = bias0 + r70 + tmp135a0 + tmp135b0 * 32 + tmp135c0;
                float tmp51 = bias1 + r71 + tmp135a1 + tmp135b1 * 32 + tmp135c1;
#endif

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

#if __ARM_NEON
                    outptr0[0] = vget_lane_f32(_tmp0, 0);
                    outptr1[0] = vget_lane_f32(_tmp0, 1);
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = vget_lane_f32(_tmp1, 0);
                        outptr1[1] = vget_lane_f32(_tmp1, 1);
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = vget_lane_f32(_tmp2, 0);
                        outptr1[2] = vget_lane_f32(_tmp2, 1);
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = vget_lane_f32(_tmp3, 0);
                        outptr1[3] = vget_lane_f32(_tmp3, 1);
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = vget_lane_f32(_tmp4, 0);
                        outptr1[4] = vget_lane_f32(_tmp4, 1);
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = vget_lane_f32(_tmp5, 0);
                        outptr1[5] = vget_lane_f32(_tmp5, 1);
                    }
#else
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
#endif
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;
            const float* r4 = r0 + max_jj * 4;
            const float* r5 = r0 + max_jj * 5;
            const float* r6 = r0 + max_jj * 6;
            const float* r7 = r0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                float tmp024a = r1[0] + r2[0];
                float tmp135a = r1[0] - r2[0];
                float tmp024b = r3[0] + r4[0];
                float tmp135b = r3[0] - r4[0];
                float tmp024c = r5[0] + r6[0];
                float tmp135c = r5[0] - r6[0];

                tmp[0][m] = r0[0] + tmp024a + tmp024b + tmp024c * 32;
                tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;
                tmp[5][m] = r7[0] + tmp135a + tmp135b * 32 + tmp135c;

                r0 += max_jj * 8;
                r1 += max_jj * 8;
                r2 += max_jj * 8;
                r3 += max_jj * 8;
                r4 += max_jj * 8;
                r5 += max_jj * 8;
                r6 += max_jj * 8;
                r7 += max_jj * 8;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];
                float r6 = tmp[m][6];
                float r7 = tmp[m][7];

                float tmp024a = r1 + r2;
                float tmp135a = r1 - r2;
                float tmp024b = r3 + r4;
                float tmp135b = r3 - r4;
                float tmp024c = r5 + r6;
                float tmp135c = r5 - r6;

                float tmp0 = bias0 + r0 + tmp024a + tmp024b + tmp024c * 32;
                float tmp1 = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                float tmp2 = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                float tmp3 = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                float tmp4 = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;
                float tmp5 = bias0 + r7 + tmp135a + tmp135b * 32 + tmp135c;

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

static void conv3x3s1_winograd63(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
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

    // NCNN_LOGE("conv3x3s1_winograd63 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd63_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);

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
            conv3x3s1_winograd63_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

                conv3x3s1_winograd_gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd63_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}
