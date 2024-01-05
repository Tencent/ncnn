// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "gemm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

#if NCNN_BF16
#include "gemm_bf16s_fp16s.h"
#include "gemm_bf16s.h"
#endif

Gemm_arm::Gemm_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_VFPV4
    support_fp16_storage = cpu_support_arm_vfpv4();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif

    nT = 0;
}

void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
            const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
            const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
            const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
            const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _r0l = vld1q_f32(p0);
                float32x4_t _r0h = vld1q_f32(p0 + 4);
                float32x4_t _r1l = vld1q_f32(p1);
                float32x4_t _r1h = vld1q_f32(p1 + 4);
                float32x4_t _r2l = vld1q_f32(p2);
                float32x4_t _r2h = vld1q_f32(p2 + 4);
                float32x4_t _r3l = vld1q_f32(p3);
                float32x4_t _r3h = vld1q_f32(p3 + 4);
                float32x4_t _r4l = vld1q_f32(p4);
                float32x4_t _r4h = vld1q_f32(p4 + 4);
                float32x4_t _r5l = vld1q_f32(p5);
                float32x4_t _r5h = vld1q_f32(p5 + 4);
                float32x4_t _r6l = vld1q_f32(p6);
                float32x4_t _r6h = vld1q_f32(p6 + 4);
                float32x4_t _r7l = vld1q_f32(p7);
                float32x4_t _r7h = vld1q_f32(p7 + 4);
                transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h);
                vst1q_f32(pp, _r0l);
                vst1q_f32(pp + 4, _r0h);
                vst1q_f32(pp + 8, _r1l);
                vst1q_f32(pp + 12, _r1h);
                vst1q_f32(pp + 8 * 2, _r2l);
                vst1q_f32(pp + 8 * 2 + 4, _r2h);
                vst1q_f32(pp + 8 * 3, _r3l);
                vst1q_f32(pp + 8 * 3 + 4, _r3h);
                vst1q_f32(pp + 8 * 4, _r4l);
                vst1q_f32(pp + 8 * 4 + 4, _r4h);
                vst1q_f32(pp + 8 * 5, _r5l);
                vst1q_f32(pp + 8 * 5 + 4, _r5h);
                vst1q_f32(pp + 8 * 6, _r6l);
                vst1q_f32(pp + 8 * 6 + 4, _r6h);
                vst1q_f32(pp + 8 * 7, _r7l);
                vst1q_f32(pp + 8 * 7 + 4, _r7h);
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
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123;
                _r0123.val[0] = vld1q_f32(p0);
                _r0123.val[1] = vld1q_f32(p1);
                _r0123.val[2] = vld1q_f32(p2);
                _r0123.val[3] = vld1q_f32(p3);
                vst4q_f32(pp, _r0123);
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
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x2_t _r01;
                _r01.val[0] = vld1q_f32(p0);
                _r01.val[1] = vld1q_f32(p1);
                vst2q_f32(pp, _r01);
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
    }
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_f32(pp, vld1q_f32(p0));
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

static void transpose_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123 = vld4q_f32(p0);
                float32x4x4_t _r4567 = vld4q_f32(p0 + 16);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r4567.val[0]);
                vst1q_f32(pp + 4 * 2, _r0123.val[1]);
                vst1q_f32(pp + 4 * 3, _r4567.val[1]);
                vst1q_f32(pp + 4 * 4, _r0123.val[2]);
                vst1q_f32(pp + 4 * 5, _r4567.val[2]);
                vst1q_f32(pp + 4 * 6, _r0123.val[3]);
                vst1q_f32(pp + 4 * 7, _r4567.val[3]);
                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p0 + 4));
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123 = vld4q_f32(p0);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r0123.val[1]);
                vst1q_f32(pp + 4 * 2, _r0123.val[2]);
                vst1q_f32(pp + 4 * 3, _r0123.val[3]);
                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x2_t _r01;
                _r01.val[0] = vld1q_f32(p0);
                _r01.val[1] = vld1q_f32(p0 + 4);
                vst2q_f32(pp, _r01);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;
            const float* p2 = (const float*)B + (j + jj + 8) * B_hstep + k * 4;

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
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _r0 = vld1q_f32(p0);
                float32x4_t _r1 = vld1q_f32(p1);
                float32x4_t _r2 = vld1q_f32(p2);
                float32x4_t _r3 = vld1q_f32(p3);
                float32x4_t _r4 = vld1q_f32(p4);
                float32x4_t _r5 = vld1q_f32(p5);
                float32x4_t _r6 = vld1q_f32(p6);
                float32x4_t _r7 = vld1q_f32(p7);
                float32x4_t _r8 = vld1q_f32(p8);
                float32x4_t _r9 = vld1q_f32(p9);
                float32x4_t _ra = vld1q_f32(pa);
                float32x4_t _rb = vld1q_f32(pb);

                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                transpose4x4_ps(_r8, _r9, _ra, _rb);

                vst1q_f32(pp, _r0);
                vst1q_f32(pp + 4, _r4);
                vst1q_f32(pp + 4 * 2, _r8);
                vst1q_f32(pp + 4 * 3, _r1);
                vst1q_f32(pp + 4 * 4, _r5);
                vst1q_f32(pp + 4 * 5, _r9);
                vst1q_f32(pp + 4 * 6, _r2);
                vst1q_f32(pp + 4 * 7, _r6);
                vst1q_f32(pp + 4 * 8, _ra);
                vst1q_f32(pp + 4 * 9, _r3);
                vst1q_f32(pp + 4 * 10, _r7);
                vst1q_f32(pp + 4 * 11, _rb);
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
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;

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
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
            const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
            const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
            const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
            const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _r0 = vld1q_f32(p0);
                float32x4_t _r1 = vld1q_f32(p1);
                float32x4_t _r2 = vld1q_f32(p2);
                float32x4_t _r3 = vld1q_f32(p3);
                float32x4_t _r4 = vld1q_f32(p4);
                float32x4_t _r5 = vld1q_f32(p5);
                float32x4_t _r6 = vld1q_f32(p6);
                float32x4_t _r7 = vld1q_f32(p7);

                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);

                vst1q_f32(pp, _r0);
                vst1q_f32(pp + 4, _r4);
                vst1q_f32(pp + 4 * 2, _r1);
                vst1q_f32(pp + 4 * 3, _r5);
                vst1q_f32(pp + 4 * 4, _r2);
                vst1q_f32(pp + 4 * 5, _r6);
                vst1q_f32(pp + 4 * 6, _r3);
                vst1q_f32(pp + 4 * 7, _r7);
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
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123;
                _r0123.val[0] = vld1q_f32(p0);
                _r0123.val[1] = vld1q_f32(p1);
                _r0123.val[2] = vld1q_f32(p2);
                _r0123.val[3] = vld1q_f32(p3);
                vst4q_f32(pp, _r0123);
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
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x2_t _r01;
                _r01.val[0] = vld1q_f32(p0);
                _r01.val[1] = vld1q_f32(p1);
                vst2q_f32(pp, _r01);
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
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_f32(pp, vld1q_f32(p0));
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

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123 = vld4q_f32(p0);
                float32x4x4_t _r4567 = vld4q_f32(p0 + 16);
                float32x4x4_t _r89ab = vld4q_f32(p0 + 32);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r4567.val[0]);
                vst1q_f32(pp + 4 * 2, _r89ab.val[0]);
                vst1q_f32(pp + 4 * 3, _r0123.val[1]);
                vst1q_f32(pp + 4 * 4, _r4567.val[1]);
                vst1q_f32(pp + 4 * 5, _r89ab.val[1]);
                vst1q_f32(pp + 4 * 6, _r0123.val[2]);
                vst1q_f32(pp + 4 * 7, _r4567.val[2]);
                vst1q_f32(pp + 4 * 8, _r89ab.val[2]);
                vst1q_f32(pp + 4 * 9, _r0123.val[3]);
                vst1q_f32(pp + 4 * 10, _r4567.val[3]);
                vst1q_f32(pp + 4 * 11, _r89ab.val[3]);
                pp += 48;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p0 + 4));
                vst1q_f32(pp + 8, vld1q_f32(p0 + 8));
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123 = vld4q_f32(p0);
                float32x4x4_t _r4567 = vld4q_f32(p0 + 16);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r4567.val[0]);
                vst1q_f32(pp + 4 * 2, _r0123.val[1]);
                vst1q_f32(pp + 4 * 3, _r4567.val[1]);
                vst1q_f32(pp + 4 * 4, _r0123.val[2]);
                vst1q_f32(pp + 4 * 5, _r4567.val[2]);
                vst1q_f32(pp + 4 * 6, _r0123.val[3]);
                vst1q_f32(pp + 4 * 7, _r4567.val[3]);
                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p0 + 4));
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123 = vld4q_f32(p0);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r0123.val[1]);
                vst1q_f32(pp + 4 * 2, _r0123.val[2]);
                vst1q_f32(pp + 4 * 3, _r0123.val[3]);
                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x2_t _r01;
                _r01.val[0] = vld1q_f32(p0);
                _r01.val[1] = vld1q_f32(p0 + 4);
                vst2q_f32(pp, _r01);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                float32x4x4_t _r0;
                float32x4x4_t _r1;
                _r0.val[0] = vld1q_f32(pp);
                _r1.val[0] = vld1q_f32(pp + 4);
                _r0.val[1] = vld1q_f32(pp + 8);
                _r1.val[1] = vld1q_f32(pp + 12);
                _r0.val[2] = vld1q_f32(pp + 16);
                _r1.val[2] = vld1q_f32(pp + 20);
                _r0.val[3] = vld1q_f32(pp + 24);
                _r1.val[3] = vld1q_f32(pp + 28);
                vst4q_f32(p0, _r0);
                vst4q_f32(p0 + 16, _r1);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                float32x4_t _r0 = vld1q_f32(pp);
                float32x4_t _r1 = vld1q_f32(pp + 4);
                vst1q_f32(p0, _r0);
                vst1q_f32(p0 + 4, _r1);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                float32x4x4_t _r0123;
                _r0123.val[0] = vld1q_f32(pp);
                _r0123.val[1] = vld1q_f32(pp + 4);
                _r0123.val[2] = vld1q_f32(pp + 8);
                _r0123.val[3] = vld1q_f32(pp + 12);
                vst4q_f32(p0, _r0123);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                float32x4_t _r0 = vld1q_f32(pp);
                vst1q_f32(p0, _r0);
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[1];
                p0[5] = pp[3];
                p0[6] = pp[5];
                p0[7] = pp[7];
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
#if __ARM_NEON
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                float32x4_t _r0 = vld1q_f32(pp);
                vst1q_f32(p0, _r0);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

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

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                        _sum20 = vdupq_n_f32(pC[0]);
                        _sum21 = vdupq_n_f32(pC[0]);
                        _sum30 = vdupq_n_f32(pC[0]);
                        _sum31 = vdupq_n_f32(pC[0]);
                        _sum40 = vdupq_n_f32(pC[0]);
                        _sum41 = vdupq_n_f32(pC[0]);
                        _sum50 = vdupq_n_f32(pC[0]);
                        _sum51 = vdupq_n_f32(pC[0]);
                        _sum60 = vdupq_n_f32(pC[0]);
                        _sum61 = vdupq_n_f32(pC[0]);
                        _sum70 = vdupq_n_f32(pC[0]);
                        _sum71 = vdupq_n_f32(pC[0]);
                        _sum80 = vdupq_n_f32(pC[0]);
                        _sum81 = vdupq_n_f32(pC[0]);
                        _sum90 = vdupq_n_f32(pC[0]);
                        _sum91 = vdupq_n_f32(pC[0]);
                        _suma0 = vdupq_n_f32(pC[0]);
                        _suma1 = vdupq_n_f32(pC[0]);
                        _sumb0 = vdupq_n_f32(pC[0]);
                        _sumb1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        _sum40 = vld1q_f32(pC + 4 * 8);
                        _sum41 = vld1q_f32(pC + 4 * 9);
                        _sum50 = vld1q_f32(pC + 4 * 10);
                        _sum51 = vld1q_f32(pC + 4 * 11);
                        _sum60 = vld1q_f32(pC + 4 * 12);
                        _sum61 = vld1q_f32(pC + 4 * 13);
                        _sum70 = vld1q_f32(pC + 4 * 14);
                        _sum71 = vld1q_f32(pC + 4 * 15);
                        _sum80 = vld1q_f32(pC + 4 * 16);
                        _sum81 = vld1q_f32(pC + 4 * 17);
                        _sum90 = vld1q_f32(pC + 4 * 18);
                        _sum91 = vld1q_f32(pC + 4 * 19);
                        _suma0 = vld1q_f32(pC + 4 * 20);
                        _suma1 = vld1q_f32(pC + 4 * 21);
                        _sumb0 = vld1q_f32(pC + 4 * 22);
                        _sumb1 = vld1q_f32(pC + 4 * 23);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum40 = vdupq_n_f32(pC[4]);
                        _sum50 = vdupq_n_f32(pC[5]);
                        _sum60 = vdupq_n_f32(pC[6]);
                        _sum70 = vdupq_n_f32(pC[7]);
                        _sum80 = vdupq_n_f32(pC[8]);
                        _sum90 = vdupq_n_f32(pC[9]);
                        _suma0 = vdupq_n_f32(pC[10]);
                        _sumb0 = vdupq_n_f32(pC[11]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        _sum41 = _sum40;
                        _sum51 = _sum50;
                        _sum61 = _sum60;
                        _sum71 = _sum70;
                        _sum81 = _sum80;
                        _sum91 = _sum90;
                        _suma1 = _suma0;
                        _sumb1 = _sumb0;
                        pC += 12;
                    }
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
            for (; kk + 3 < max_kk; kk += 4)
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

                _pA0 = vld1q_f32(pA);
                _pA1 = vld1q_f32(pA + 4);

                _pB0 = vld1q_f32(pB);
                _pB1 = vld1q_f32(pB + 4);
                _pB2 = vld1q_f32(pB + 8);

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

                _pA0 = vld1q_f32(pA);
                _pA1 = vld1q_f32(pA + 4);

                _pB0 = vld1q_f32(pB);
                _pB1 = vld1q_f32(pB + 4);
                _pB2 = vld1q_f32(pB + 8);

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

                _pA0 = vld1q_f32(pA);
                _pA1 = vld1q_f32(pA + 4);

                _pB0 = vld1q_f32(pB);
                _pB1 = vld1q_f32(pB + 4);
                _pB2 = vld1q_f32(pB + 8);

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

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                        _sum20 = vdupq_n_f32(pC[0]);
                        _sum21 = vdupq_n_f32(pC[0]);
                        _sum30 = vdupq_n_f32(pC[0]);
                        _sum31 = vdupq_n_f32(pC[0]);
                        _sum40 = vdupq_n_f32(pC[0]);
                        _sum41 = vdupq_n_f32(pC[0]);
                        _sum50 = vdupq_n_f32(pC[0]);
                        _sum51 = vdupq_n_f32(pC[0]);
                        _sum60 = vdupq_n_f32(pC[0]);
                        _sum61 = vdupq_n_f32(pC[0]);
                        _sum70 = vdupq_n_f32(pC[0]);
                        _sum71 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        _sum40 = vld1q_f32(pC + 4 * 8);
                        _sum41 = vld1q_f32(pC + 4 * 9);
                        _sum50 = vld1q_f32(pC + 4 * 10);
                        _sum51 = vld1q_f32(pC + 4 * 11);
                        _sum60 = vld1q_f32(pC + 4 * 12);
                        _sum61 = vld1q_f32(pC + 4 * 13);
                        _sum70 = vld1q_f32(pC + 4 * 14);
                        _sum71 = vld1q_f32(pC + 4 * 15);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum40 = vdupq_n_f32(pC[4]);
                        _sum50 = vdupq_n_f32(pC[5]);
                        _sum60 = vdupq_n_f32(pC[6]);
                        _sum70 = vdupq_n_f32(pC[7]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        _sum41 = _sum40;
                        _sum51 = _sum50;
                        _sum61 = _sum60;
                        _sum71 = _sum70;
                        pC += 8;
                    }
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
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                        _sum20 = vdupq_n_f32(pC[0]);
                        _sum21 = vdupq_n_f32(pC[0]);
                        _sum30 = vdupq_n_f32(pC[0]);
                        _sum31 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        pC += 4;
                    }
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
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        pC += 2;
                    }
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
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        pC += 1;
                    }
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

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                        _sum3 = vdupq_n_f32(pC[0]);
                        _sum4 = vdupq_n_f32(pC[0]);
                        _sum5 = vdupq_n_f32(pC[0]);
                        _sum6 = vdupq_n_f32(pC[0]);
                        _sum7 = vdupq_n_f32(pC[0]);
                        _sum8 = vdupq_n_f32(pC[0]);
                        _sum9 = vdupq_n_f32(pC[0]);
                        _suma = vdupq_n_f32(pC[0]);
                        _sumb = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        _sum4 = vld1q_f32(pC + 16);
                        _sum5 = vld1q_f32(pC + 20);
                        _sum6 = vld1q_f32(pC + 24);
                        _sum7 = vld1q_f32(pC + 28);
                        _sum8 = vld1q_f32(pC + 32);
                        _sum9 = vld1q_f32(pC + 36);
                        _suma = vld1q_f32(pC + 40);
                        _sumb = vld1q_f32(pC + 44);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        _sum4 = vdupq_n_f32(pC[4]);
                        _sum5 = vdupq_n_f32(pC[5]);
                        _sum6 = vdupq_n_f32(pC[6]);
                        _sum7 = vdupq_n_f32(pC[7]);
                        _sum8 = vdupq_n_f32(pC[8]);
                        _sum9 = vdupq_n_f32(pC[9]);
                        _suma = vdupq_n_f32(pC[10]);
                        _sumb = vdupq_n_f32(pC[11]);
                        pC += 12;
                    }
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
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
                _sum4 = vdupq_n_f32(0.f);
                _sum5 = vdupq_n_f32(0.f);
                _sum6 = vdupq_n_f32(0.f);
                _sum7 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                        _sum3 = vdupq_n_f32(pC[0]);
                        _sum4 = vdupq_n_f32(pC[0]);
                        _sum5 = vdupq_n_f32(pC[0]);
                        _sum6 = vdupq_n_f32(pC[0]);
                        _sum7 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        _sum4 = vld1q_f32(pC + 16);
                        _sum5 = vld1q_f32(pC + 20);
                        _sum6 = vld1q_f32(pC + 24);
                        _sum7 = vld1q_f32(pC + 28);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        _sum4 = vdupq_n_f32(pC[4]);
                        _sum5 = vdupq_n_f32(pC[5]);
                        _sum6 = vdupq_n_f32(pC[6]);
                        _sum7 = vdupq_n_f32(pC[7]);
                        pC += 8;
                    }
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
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                        _sum3 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        pC += 4;
                    }
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
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        pC += 2;
                    }
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
                _sum0 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        pC += 1;
                    }
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
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum02 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum12 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum02 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                        _sum12 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum02 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum11 = vdupq_n_f32(pC[1]);
                        _sum12 = vdupq_n_f32(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                        float32x4x2_t _tmp45 = vld2q_f32(pC + 16);
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
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum02 = vld1q_f32(pC + 8);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum12 = _sum02;
                        pC += 12;
                    }
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

                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum02 = vfmaq_lane_f32(_sum02, _pB2, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
                _sum12 = vfmaq_lane_f32(_sum12, _pB2, _pA, 1);

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
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum11 = vdupq_n_f32(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                        _sum00 = _tmp01.val[0];
                        _sum01 = _tmp23.val[0];
                        _sum10 = _tmp01.val[1];
                        _sum11 = _tmp23.val[1];
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        pC += 8;
                    }
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
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        _sum0 = _tmp01.val[0];
                        _sum1 = _tmp01.val[1];
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                        pC += 4;
                    }
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

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        pC += 12;
                    }
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

                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA0, _pB2);

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
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
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
                _sum = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = vld1q_f32(pC);
                        pC += 4;
                    }
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

#if __aarch64__
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(1, tile_size);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __aarch64__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(float) / TILE_K);

#if __aarch64__
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(4, tile_size / 4 * 4);
#elif __ARM_NEON
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
            TILE_N = std::max(1, tile_size);
#endif
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __aarch64__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __aarch64__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __ARM_NEON
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }

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

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __aarch64__
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#elif __ARM_NEON
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_N > 0)
    {
#if __aarch64__
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#elif __ARM_NEON
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = constant_TILE_N;
#endif
    }

    if (constant_TILE_K > 0)
    {
#if __aarch64__
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#elif __ARM_NEON
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

static int gemm_arm(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
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
    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

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
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

static int gemm_AT_arm(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
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
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

static int gemm_BT_arm(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 4u, opt.workspace_allocator);

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

static int gemm_AT_BT_arm(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

int Gemm_arm::create_pipeline(const Option& opt)
{
#if NCNN_ARM82
    if (cpu_support_arm_asimdhp() && opt.use_fp16_storage)
    {
        if (opt.use_fp16_arithmetic)
            return create_pipeline_fp16sa(opt);
        else
            return create_pipeline_fp16s(opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        return create_pipeline_bf16s(opt);
    }
#endif

#if NCNN_VFPV4
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
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

        B_data.release();
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        CT_data = C_data;

#if __ARM_NEON
        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
            int C_elempack = constantM % 4 == 0 ? 4 : 1;
            convert_packing(C_data, CT_data, C_elempack, opt);
        }
#endif // __ARM_NEON

        // pre-multiply C with beta
        if (beta != 1.f)
        {
            Mat C2;
            C2.create_like(CT_data);

            const int size = CT_data.total() * CT_data.elempack;
            for (int i = 0; i < size; i++)
            {
                C2[i] = CT_data[i] * beta;
            }

            CT_data = C2;
        }

        C_data.release();
    }

    if (constantA || constantB || constantC)
    {
        nT = opt.num_threads;
    }

    return 0;
}

int Gemm_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = constantA ? AT_data : bottom_blobs[0];
    int elembits = bottom_blob.elembits();

#if NCNN_ARM82
    if (cpu_support_arm_asimdhp() && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

#if NCNN_VFPV4
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        return forward_fp16s(bottom_blobs, top_blobs, opt);
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

                const int size = C.total() * C.elempack;
                for (int i = 0; i < size; i++)
                {
                    CT_data[i] = C[i] * beta;
                }

                C = CT_data;
            }
        }
    }

    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON
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
        ret = gemm_AT_BT_arm(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_arm(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_arm(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_arm(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
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

#if NCNN_BF16
static int gemm_arm_bf16s(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16s_fp16s(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 2u, opt.workspace_allocator);
    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

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
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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
                        transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                bool k_end = !output_transpose && k + TILE_K >= K;
                float _alpha = k + TILE_K >= K ? alpha : 1.f;

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_fp32_to_bf16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_AT_arm_bf16s(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16s_fp16s(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

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
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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
                float _alpha = k + TILE_K >= K ? alpha : 1.f;

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_fp32_to_bf16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_BT_arm_bf16s(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16s_fp16s(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 2u, opt.workspace_allocator);

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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
                        transpose_pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        pack_A_tile_bf16_fp16(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                bool k_end = !output_transpose && k + TILE_K >= K;
                float _alpha = k + TILE_K >= K ? alpha : 1.f;

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_fp32_to_bf16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static int gemm_AT_BT_arm_bf16s(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, float alpha, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16s_fp16s(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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
                float _alpha = k + TILE_K >= K ? alpha : 1.f;

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, _alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile_fp32_to_bf16(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

int Gemm_arm::create_pipeline_bf16s(const Option& opt)
{
    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_bf16s_fp16s(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

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
                    transpose_pack_A_tile_fp32_to_bf16(A_data, AT_tile, i, max_ii, k, max_kk);
                }
                else
                {
                    pack_A_tile_fp32_to_bf16(A_data, AT_tile, i, max_ii, k, max_kk);
                }
            }
        }

        A_data.release();
    }

    if (constantB)
    {
        const int N = constantN;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_bf16s_fp16s(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

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
                    pack_B_tile_fp32_to_bf16(B_data, BT_tile, j, max_jj, k, max_kk);
                }
                else
                {
                    transpose_pack_B_tile_fp32_to_bf16(B_data, BT_tile, j, max_jj, k, max_kk);
                }
            }
        }

        B_data.release();
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        CT_data = C_data;

#if __ARM_NEON
        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
            int C_elempack = constantM % 4 == 0 ? 4 : 1;
            convert_packing(C_data, CT_data, C_elempack, opt);
        }
#endif // __ARM_NEON

        // pre-multiply C with beta
        if (beta != 1.f)
        {
            Mat C2;
            C2.create_like(CT_data);

            const int size = CT_data.total() * CT_data.elempack;
            for (int i = 0; i < size; i++)
            {
                C2[i] = CT_data[i] * beta;
            }

            CT_data = C2;
        }

        C_data.release();
    }

    if (constantA || constantB || constantC)
    {
        nT = opt.num_threads;
    }

    return 0;
}

int Gemm_arm::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

            // cast to fp32
            {
                Mat CT_data;
                cast_bfloat16_to_float32(C, CT_data);
                C = CT_data;
            }

            // pre-multiply C with beta
            if (beta != 1.f)
            {
                Mat CT_data;
                CT_data.create_like(C, opt.workspace_allocator);

                const int size = C.total() * C.elempack;
                for (int i = 0; i < size; i++)
                {
                    CT_data[i] = C[i] * beta;
                }

                C = CT_data;
            }
        }
    }

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % 4 == 0 ? 4 : 1;
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
        ret = gemm_AT_BT_arm_bf16s(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_arm_bf16s(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_arm_bf16s(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_arm_bf16s(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, alpha, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }

    return ret;
}
#endif // NCNN_BF16

} // namespace ncnn
