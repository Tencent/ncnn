// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "gemm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

Gemm_riscv::Gemm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
    one_blob_only = false;
    support_inplace = false;

    nT = 0;
#if __riscv_vector
    // When processing float data,
    // even if the current hardware provides vector registers of more than 128 bits,
    // vl=4 is still used, even though this will waste the width of the vector register.
    vl = vsetvlmax_e32m1();
    vl = vl >= 4 ? 4 : vl;
#else
    vl = 0;
#endif // __riscv_vector
}

static void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, size_t vl)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, vl), vl);
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
                vfloat32m1_t _r0l = vle32_v_f32m1(p0, vl);
                vfloat32m1_t _r0h = vle32_v_f32m1(p0 + 4, vl);
                vfloat32m1_t _r1l = vle32_v_f32m1(p1, vl);
                vfloat32m1_t _r1h = vle32_v_f32m1(p1 + 4, vl);
                vfloat32m1_t _r2l = vle32_v_f32m1(p2, vl);
                vfloat32m1_t _r2h = vle32_v_f32m1(p2 + 4, vl);
                vfloat32m1_t _r3l = vle32_v_f32m1(p3, vl);
                vfloat32m1_t _r3h = vle32_v_f32m1(p3 + 4, vl);
                vfloat32m1_t _r4l = vle32_v_f32m1(p4, vl);
                vfloat32m1_t _r4h = vle32_v_f32m1(p4 + 4, vl);
                vfloat32m1_t _r5l = vle32_v_f32m1(p5, vl);
                vfloat32m1_t _r5h = vle32_v_f32m1(p5 + 4, vl);
                vfloat32m1_t _r6l = vle32_v_f32m1(p6, vl);
                vfloat32m1_t _r6h = vle32_v_f32m1(p6 + 4, vl);
                vfloat32m1_t _r7l = vle32_v_f32m1(p7, vl);
                vfloat32m1_t _r7h = vle32_v_f32m1(p7 + 4, vl);

                vsseg8e32_v_f32m1(pp, _r0l, _r1l, _r2l, _r3l, _r4l, _r5l, _r6l, _r7l, vl);
                vsseg8e32_v_f32m1(pp + 32, _r0h, _r1h, _r2h, _r3h, _r4h, _r5h, _r6h, _r7h, vl);

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
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
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
                vfloat32m1_t v0 = vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = vle32_v_f32m1(p1, vl);
                vfloat32m1_t v2 = vle32_v_f32m1(p2, vl);
                vfloat32m1_t v3 = vle32_v_f32m1(p3, vl);
                vsseg4e32_v_f32m1(pp, v0, v1, v2, v3, vl);
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
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t v0 = vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = vle32_v_f32m1(p1, vl);
                vsseg2e32_v_f32m1(pp, v0, v1, vl);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __riscv_vector
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
#if __riscv_vector
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                pp += 4;
                p0 += 4;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, size_t vl)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t _r0;
                vfloat32m1_t _r1;
                vfloat32m1_t _r2;
                vfloat32m1_t _r3;
                vfloat32m1_t _r4;
                vfloat32m1_t _r5;
                vfloat32m1_t _r6;
                vfloat32m1_t _r7;
                vlseg4e32_v_f32m1(&_r0, &_r1, &_r2, &_r3, p0, vl);
                vlseg4e32_v_f32m1(&_r4, &_r5, &_r6, &_r7, p0 + 16, vl);
                vse32_v_f32m1(pp, _r0, vl);
                vse32_v_f32m1(pp + 4, _r4, vl);
                vse32_v_f32m1(pp + 4 * 2, _r1, vl);
                vse32_v_f32m1(pp + 4 * 3, _r5, vl);
                vse32_v_f32m1(pp + 4 * 4, _r2, vl);
                vse32_v_f32m1(pp + 4 * 5, _r6, vl);
                vse32_v_f32m1(pp + 4 * 6, _r3, vl);
                vse32_v_f32m1(pp + 4 * 7, _r7, vl);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, vl), vl);
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t _r0;
                vfloat32m1_t _r1;
                vfloat32m1_t _r2;
                vfloat32m1_t _r3;
                vlseg4e32_v_f32m1(&_r0, &_r1, &_r2, &_r3, p0, vl);
                vse32_v_f32m1(pp, _r0, vl);
                vse32_v_f32m1(pp + 4, _r1, vl);
                vse32_v_f32m1(pp + 4 * 2, _r2, vl);
                vse32_v_f32m1(pp + 4 * 3, _r3, vl);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t v0 = vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = vle32_v_f32m1(p0 + 4, vl);
                vsseg2e32_v_f32m1(pp, v0, v1, vl);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __riscv_vector
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
#if __riscv_vector
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __riscv_vector
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

static void pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, size_t vl)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;
            const float* p2 = (const float*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, vl), vl);
                vse32_v_f32m1(pp + 8, vle32_v_f32m1(p2, vl), vl);
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
                vfloat32m1_t _r0 = vle32_v_f32m1(p0, vl);
                vfloat32m1_t _r1 = vle32_v_f32m1(p1, vl);
                vfloat32m1_t _r2 = vle32_v_f32m1(p2, vl);
                vfloat32m1_t _r3 = vle32_v_f32m1(p3, vl);
                vfloat32m1_t _r4 = vle32_v_f32m1(p4, vl);
                vfloat32m1_t _r5 = vle32_v_f32m1(p5, vl);
                vfloat32m1_t _r6 = vle32_v_f32m1(p6, vl);
                vfloat32m1_t _r7 = vle32_v_f32m1(p7, vl);
                vfloat32m1_t _r8 = vle32_v_f32m1(p8, vl);
                vfloat32m1_t _r9 = vle32_v_f32m1(p9, vl);
                vfloat32m1_t _ra = vle32_v_f32m1(pa, vl);
                vfloat32m1_t _rb = vle32_v_f32m1(pb, vl);

                transpose4x4_ps(_r0, _r1, _r2, _r3, vl);
                transpose4x4_ps(_r4, _r5, _r6, _r7, vl);
                transpose4x4_ps(_r8, _r9, _ra, _rb, vl);

                vse32_v_f32m1(pp, _r0, vl);
                vse32_v_f32m1(pp + 4, _r4, vl);
                vse32_v_f32m1(pp + 4 * 2, _r8, vl);
                vse32_v_f32m1(pp + 4 * 3, _r1, vl);
                vse32_v_f32m1(pp + 4 * 4, _r5, vl);
                vse32_v_f32m1(pp + 4 * 5, _r9, vl);
                vse32_v_f32m1(pp + 4 * 6, _r2, vl);
                vse32_v_f32m1(pp + 4 * 7, _r6, vl);
                vse32_v_f32m1(pp + 4 * 8, _ra, vl);
                vse32_v_f32m1(pp + 4 * 9, _r3, vl);
                vse32_v_f32m1(pp + 4 * 10, _r7, vl);
                vse32_v_f32m1(pp + 4 * 11, _rb, vl);
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
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, vl), vl);
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
                vfloat32m1_t _r0 = vle32_v_f32m1(p0, vl);
                vfloat32m1_t _r1 = vle32_v_f32m1(p1, vl);
                vfloat32m1_t _r2 = vle32_v_f32m1(p2, vl);
                vfloat32m1_t _r3 = vle32_v_f32m1(p3, vl);
                vfloat32m1_t _r4 = vle32_v_f32m1(p4, vl);
                vfloat32m1_t _r5 = vle32_v_f32m1(p5, vl);
                vfloat32m1_t _r6 = vle32_v_f32m1(p6, vl);
                vfloat32m1_t _r7 = vle32_v_f32m1(p7, vl);

                vsseg8e32_v_f32m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
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
                vfloat32m1_t v0 = vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = vle32_v_f32m1(p1, vl);
                vfloat32m1_t v2 = vle32_v_f32m1(p2, vl);
                vfloat32m1_t v3 = vle32_v_f32m1(p3, vl);
                vsseg4e32_v_f32m1(pp, v0, v1, v2, v3, vl);
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
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t v0 = vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = vle32_v_f32m1(p1, vl);
                vsseg2e32_v_f32m1(pp, v0, v1, vl);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __riscv_vector
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
#if __riscv_vector
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                pp += 4;
                p0 += 4;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, size_t vl)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t _r0;
                vfloat32m1_t _r1;
                vfloat32m1_t _r2;
                vfloat32m1_t _r3;
                vfloat32m1_t _r4;
                vfloat32m1_t _r5;
                vfloat32m1_t _r6;
                vfloat32m1_t _r7;
                vfloat32m1_t _r8;
                vfloat32m1_t _r9;
                vfloat32m1_t _ra;
                vfloat32m1_t _rb;
                vlseg4e32_v_f32m1(&_r0, &_r1, &_r2, &_r3, p0, vl);
                vlseg4e32_v_f32m1(&_r4, &_r5, &_r6, &_r7, p0 + 16, vl);
                vlseg4e32_v_f32m1(&_r8, &_r9, &_ra, &_rb, p0 + 32, vl);
                vse32_v_f32m1(pp, _r0, vl);
                vse32_v_f32m1(pp + 4, _r4, vl);
                vse32_v_f32m1(pp + 4 * 2, _r8, vl);
                vse32_v_f32m1(pp + 4 * 3, _r1, vl);
                vse32_v_f32m1(pp + 4 * 4, _r5, vl);
                vse32_v_f32m1(pp + 4 * 5, _r9, vl);
                vse32_v_f32m1(pp + 4 * 6, _r2, vl);
                vse32_v_f32m1(pp + 4 * 7, _r6, vl);
                vse32_v_f32m1(pp + 4 * 8, _ra, vl);
                vse32_v_f32m1(pp + 4 * 9, _r3, vl);
                vse32_v_f32m1(pp + 4 * 10, _r7, vl);
                vse32_v_f32m1(pp + 4 * 11, _rb, vl);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, vl), vl);
                vse32_v_f32m1(pp + 8, vle32_v_f32m1(p0 + 8, vl), vl);
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t _r0;
                vfloat32m1_t _r1;
                vfloat32m1_t _r2;
                vfloat32m1_t _r3;
                vfloat32m1_t _r4;
                vfloat32m1_t _r5;
                vfloat32m1_t _r6;
                vfloat32m1_t _r7;
                vlseg4e32_v_f32m1(&_r0, &_r1, &_r2, &_r3, p0, vl);
                vlseg4e32_v_f32m1(&_r4, &_r5, &_r6, &_r7, p0 + 16, vl);
                vse32_v_f32m1(pp, _r0, vl);
                vse32_v_f32m1(pp + 4, _r4, vl);
                vse32_v_f32m1(pp + 4 * 2, _r1, vl);
                vse32_v_f32m1(pp + 4 * 3, _r5, vl);
                vse32_v_f32m1(pp + 4 * 4, _r2, vl);
                vse32_v_f32m1(pp + 4 * 5, _r6, vl);
                vse32_v_f32m1(pp + 4 * 6, _r3, vl);
                vse32_v_f32m1(pp + 4 * 7, _r7, vl);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, vl), vl);
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
                vfloat32m1_t _r0;
                vfloat32m1_t _r1;
                vfloat32m1_t _r2;
                vfloat32m1_t _r3;
                vlseg4e32_v_f32m1(&_r0, &_r1, &_r2, &_r3, p0, vl);
                vse32_v_f32m1(pp, _r0, vl);
                vse32_v_f32m1(pp + 4, _r1, vl);
                vse32_v_f32m1(pp + 4 * 2, _r2, vl);
                vse32_v_f32m1(pp + 4 * 3, _r3, vl);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __riscv_vector
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t v0 = vle32_v_f32m1(p0, vl);
                vfloat32m1_t v1 = vle32_v_f32m1(p0 + 4, vl);
                vsseg2e32_v_f32m1(pp, v0, v1, vl);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __riscv_vector
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
#if __riscv_vector
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, vl), vl);
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __riscv_vector
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

static void transpose_unpack_output_tile(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj, size_t vl)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vfloat32m1_t v0 = vle32_v_f32m1(pp, vl);
                vfloat32m1_t v1 = vle32_v_f32m1(pp + 8, vl);
                vfloat32m1_t v2 = vle32_v_f32m1(pp + 16, vl);
                vfloat32m1_t v3 = vle32_v_f32m1(pp + 24, vl);
                vsseg4e32_v_f32m1(p0, v0, v1, v2, v3, vl);
                v0 = vle32_v_f32m1(pp + 4, vl);
                v1 = vle32_v_f32m1(pp + 12, vl);
                v2 = vle32_v_f32m1(pp + 20, vl);
                v3 = vle32_v_f32m1(pp + 28, vl);
                vsseg4e32_v_f32m1(p0 + 16, v0, v1, v2, v3, vl);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vfloat32m1_t _r0 = vle32_v_f32m1(pp, vl);
                vfloat32m1_t _r1 = vle32_v_f32m1(pp + 4, vl);
                vse32_v_f32m1(p0, _r0, vl);
                vse32_v_f32m1(p0 + 4, _r1, vl);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vfloat32m1_t v0 = vle32_v_f32m1(pp, vl);
                vfloat32m1_t v1 = vle32_v_f32m1(pp + 4, vl);
                vfloat32m1_t v2 = vle32_v_f32m1(pp + 8, vl);
                vfloat32m1_t v3 = vle32_v_f32m1(pp + 12, vl);
                vsseg4e32_v_f32m1(p0, v0, v1, v2, v3, vl);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vfloat32m1_t _r0 = vle32_v_f32m1(pp, vl);
                vse32_v_f32m1(p0, _r0, vl);
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
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
#endif // __riscv_vector
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
#if __riscv_vector
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vfloat32m1_t _r0 = vle32_v_f32m1(pp, vl);
                vse32_v_f32m1(p0, _r0, vl);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __riscv_vector
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

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end, size_t vl)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __riscv_vector
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
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;
            vfloat32m1_t _sum20;
            vfloat32m1_t _sum21;
            vfloat32m1_t _sum30;
            vfloat32m1_t _sum31;
            vfloat32m1_t _sum40;
            vfloat32m1_t _sum41;
            vfloat32m1_t _sum50;
            vfloat32m1_t _sum51;
            vfloat32m1_t _sum60;
            vfloat32m1_t _sum61;
            vfloat32m1_t _sum70;
            vfloat32m1_t _sum71;
            vfloat32m1_t _sum80;
            vfloat32m1_t _sum81;
            vfloat32m1_t _sum90;
            vfloat32m1_t _sum91;
            vfloat32m1_t _suma0;
            vfloat32m1_t _suma1;
            vfloat32m1_t _sumb0;
            vfloat32m1_t _sumb1;

            if (k == 0)
            {
                _sum00 = vfmv_v_f_f32m1(0.f, vl);
                _sum01 = vfmv_v_f_f32m1(0.f, vl);
                _sum10 = vfmv_v_f_f32m1(0.f, vl);
                _sum11 = vfmv_v_f_f32m1(0.f, vl);
                _sum20 = vfmv_v_f_f32m1(0.f, vl);
                _sum21 = vfmv_v_f_f32m1(0.f, vl);
                _sum30 = vfmv_v_f_f32m1(0.f, vl);
                _sum31 = vfmv_v_f_f32m1(0.f, vl);
                _sum40 = vfmv_v_f_f32m1(0.f, vl);
                _sum41 = vfmv_v_f_f32m1(0.f, vl);
                _sum50 = vfmv_v_f_f32m1(0.f, vl);
                _sum51 = vfmv_v_f_f32m1(0.f, vl);
                _sum60 = vfmv_v_f_f32m1(0.f, vl);
                _sum61 = vfmv_v_f_f32m1(0.f, vl);
                _sum70 = vfmv_v_f_f32m1(0.f, vl);
                _sum71 = vfmv_v_f_f32m1(0.f, vl);
                _sum80 = vfmv_v_f_f32m1(0.f, vl);
                _sum81 = vfmv_v_f_f32m1(0.f, vl);
                _sum90 = vfmv_v_f_f32m1(0.f, vl);
                _sum91 = vfmv_v_f_f32m1(0.f, vl);
                _suma0 = vfmv_v_f_f32m1(0.f, vl);
                _suma1 = vfmv_v_f_f32m1(0.f, vl);
                _sumb0 = vfmv_v_f_f32m1(0.f, vl);
                _sumb1 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum11 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum20 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum21 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum30 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum31 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum40 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum41 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum50 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum51 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum60 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum61 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum70 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum71 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum80 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum81 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum90 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum91 = vfmv_v_f_f32m1(pC[0], vl);
                        _suma0 = vfmv_v_f_f32m1(pC[0], vl);
                        _suma1 = vfmv_v_f_f32m1(pC[0], vl);
                        _sumb0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sumb1 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4, vl);
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
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4 * 1, vl);
                        _sum10 = vle32_v_f32m1(pC + 4 * 2, vl);
                        _sum11 = vle32_v_f32m1(pC + 4 * 3, vl);
                        _sum20 = vle32_v_f32m1(pC + 4 * 4, vl);
                        _sum21 = vle32_v_f32m1(pC + 4 * 5, vl);
                        _sum30 = vle32_v_f32m1(pC + 4 * 6, vl);
                        _sum31 = vle32_v_f32m1(pC + 4 * 7, vl);
                        _sum40 = vle32_v_f32m1(pC + 4 * 8, vl);
                        _sum41 = vle32_v_f32m1(pC + 4 * 9, vl);
                        _sum50 = vle32_v_f32m1(pC + 4 * 10, vl);
                        _sum51 = vle32_v_f32m1(pC + 4 * 11, vl);
                        _sum60 = vle32_v_f32m1(pC + 4 * 12, vl);
                        _sum61 = vle32_v_f32m1(pC + 4 * 13, vl);
                        _sum70 = vle32_v_f32m1(pC + 4 * 14, vl);
                        _sum71 = vle32_v_f32m1(pC + 4 * 15, vl);
                        _sum80 = vle32_v_f32m1(pC + 4 * 16, vl);
                        _sum81 = vle32_v_f32m1(pC + 4 * 17, vl);
                        _sum90 = vle32_v_f32m1(pC + 4 * 18, vl);
                        _sum91 = vle32_v_f32m1(pC + 4 * 19, vl);
                        _suma0 = vle32_v_f32m1(pC + 4 * 20, vl);
                        _suma1 = vle32_v_f32m1(pC + 4 * 21, vl);
                        _sumb0 = vle32_v_f32m1(pC + 4 * 22, vl);
                        _sumb1 = vle32_v_f32m1(pC + 4 * 23, vl);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum20 = vfmv_v_f_f32m1(pC[2], vl);
                        _sum30 = vfmv_v_f_f32m1(pC[3], vl);
                        _sum40 = vfmv_v_f_f32m1(pC[4], vl);
                        _sum50 = vfmv_v_f_f32m1(pC[5], vl);
                        _sum60 = vfmv_v_f_f32m1(pC[6], vl);
                        _sum70 = vfmv_v_f_f32m1(pC[7], vl);
                        _sum80 = vfmv_v_f_f32m1(pC[8], vl);
                        _sum90 = vfmv_v_f_f32m1(pC[9], vl);
                        _suma0 = vfmv_v_f_f32m1(pC[10], vl);
                        _sumb0 = vfmv_v_f_f32m1(pC[11], vl);
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
                _sum00 = vle32_v_f32m1(outptr, vl);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, vl);
                _sum10 = vle32_v_f32m1(outptr + 4 * 2, vl);
                _sum11 = vle32_v_f32m1(outptr + 4 * 3, vl);
                _sum20 = vle32_v_f32m1(outptr + 4 * 4, vl);
                _sum21 = vle32_v_f32m1(outptr + 4 * 5, vl);
                _sum30 = vle32_v_f32m1(outptr + 4 * 6, vl);
                _sum31 = vle32_v_f32m1(outptr + 4 * 7, vl);
                _sum40 = vle32_v_f32m1(outptr + 4 * 8, vl);
                _sum41 = vle32_v_f32m1(outptr + 4 * 9, vl);
                _sum50 = vle32_v_f32m1(outptr + 4 * 10, vl);
                _sum51 = vle32_v_f32m1(outptr + 4 * 11, vl);
                _sum60 = vle32_v_f32m1(outptr + 4 * 12, vl);
                _sum61 = vle32_v_f32m1(outptr + 4 * 13, vl);
                _sum70 = vle32_v_f32m1(outptr + 4 * 14, vl);
                _sum71 = vle32_v_f32m1(outptr + 4 * 15, vl);
                _sum80 = vle32_v_f32m1(outptr + 4 * 16, vl);
                _sum81 = vle32_v_f32m1(outptr + 4 * 17, vl);
                _sum90 = vle32_v_f32m1(outptr + 4 * 18, vl);
                _sum91 = vle32_v_f32m1(outptr + 4 * 19, vl);
                _suma0 = vle32_v_f32m1(outptr + 4 * 20, vl);
                _suma1 = vle32_v_f32m1(outptr + 4 * 21, vl);
                _sumb0 = vle32_v_f32m1(outptr + 4 * 22, vl);
                _sumb1 = vle32_v_f32m1(outptr + 4 * 23, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, vl);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pA0, pB[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pA1, pB[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pA0, pB[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pA1, pB[1], _sum11, vl);
                _sum20 = vfmadd_vf_f32m1(_pA0, pB[2], _sum20, vl);
                _sum21 = vfmadd_vf_f32m1(_pA1, pB[2], _sum21, vl);
                _sum30 = vfmadd_vf_f32m1(_pA0, pB[3], _sum30, vl);
                _sum31 = vfmadd_vf_f32m1(_pA1, pB[3], _sum31, vl);
                _sum40 = vfmadd_vf_f32m1(_pA0, pB[4], _sum40, vl);
                _sum41 = vfmadd_vf_f32m1(_pA1, pB[4], _sum41, vl);
                _sum50 = vfmadd_vf_f32m1(_pA0, pB[5], _sum50, vl);
                _sum51 = vfmadd_vf_f32m1(_pA1, pB[5], _sum51, vl);
                _sum60 = vfmadd_vf_f32m1(_pA0, pB[6], _sum60, vl);
                _sum61 = vfmadd_vf_f32m1(_pA1, pB[6], _sum61, vl);
                _sum70 = vfmadd_vf_f32m1(_pA0, pB[7], _sum70, vl);
                _sum71 = vfmadd_vf_f32m1(_pA1, pB[7], _sum71, vl);
                _sum80 = vfmadd_vf_f32m1(_pA0, pB[8], _sum80, vl);
                _sum81 = vfmadd_vf_f32m1(_pA1, pB[8], _sum81, vl);
                _sum90 = vfmadd_vf_f32m1(_pA0, pB[9], _sum90, vl);
                _sum91 = vfmadd_vf_f32m1(_pA1, pB[9], _sum91, vl);
                _suma0 = vfmadd_vf_f32m1(_pA0, pB[10], _suma0, vl);
                _suma1 = vfmadd_vf_f32m1(_pA1, pB[10], _suma1, vl);
                _sumb0 = vfmadd_vf_f32m1(_pA0, pB[11], _sumb0, vl);
                _sumb1 = vfmadd_vf_f32m1(_pA1, pB[11], _sumb1, vl);

                pA += 8;
                pB += 12;

                _pA0 = vle32_v_f32m1(pA, vl);
                _pA1 = vle32_v_f32m1(pA + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pA0, pB[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pA1, pB[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pA0, pB[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pA1, pB[1], _sum11, vl);
                _sum20 = vfmadd_vf_f32m1(_pA0, pB[2], _sum20, vl);
                _sum21 = vfmadd_vf_f32m1(_pA1, pB[2], _sum21, vl);
                _sum30 = vfmadd_vf_f32m1(_pA0, pB[3], _sum30, vl);
                _sum31 = vfmadd_vf_f32m1(_pA1, pB[3], _sum31, vl);
                _sum40 = vfmadd_vf_f32m1(_pA0, pB[4], _sum40, vl);
                _sum41 = vfmadd_vf_f32m1(_pA1, pB[4], _sum41, vl);
                _sum50 = vfmadd_vf_f32m1(_pA0, pB[5], _sum50, vl);
                _sum51 = vfmadd_vf_f32m1(_pA1, pB[5], _sum51, vl);
                _sum60 = vfmadd_vf_f32m1(_pA0, pB[6], _sum60, vl);
                _sum61 = vfmadd_vf_f32m1(_pA1, pB[6], _sum61, vl);
                _sum70 = vfmadd_vf_f32m1(_pA0, pB[7], _sum70, vl);
                _sum71 = vfmadd_vf_f32m1(_pA1, pB[7], _sum71, vl);
                _sum80 = vfmadd_vf_f32m1(_pA0, pB[8], _sum80, vl);
                _sum81 = vfmadd_vf_f32m1(_pA1, pB[8], _sum81, vl);
                _sum90 = vfmadd_vf_f32m1(_pA0, pB[9], _sum90, vl);
                _sum91 = vfmadd_vf_f32m1(_pA1, pB[9], _sum91, vl);
                _suma0 = vfmadd_vf_f32m1(_pA0, pB[10], _suma0, vl);
                _suma1 = vfmadd_vf_f32m1(_pA1, pB[10], _suma1, vl);
                _sumb0 = vfmadd_vf_f32m1(_pA0, pB[11], _sumb0, vl);
                _sumb1 = vfmadd_vf_f32m1(_pA1, pB[11], _sumb1, vl);

                pA += 8;
                pB += 12;

                _pA0 = vle32_v_f32m1(pA, vl);
                _pA1 = vle32_v_f32m1(pA + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pA0, pB[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pA1, pB[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pA0, pB[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pA1, pB[1], _sum11, vl);
                _sum20 = vfmadd_vf_f32m1(_pA0, pB[2], _sum20, vl);
                _sum21 = vfmadd_vf_f32m1(_pA1, pB[2], _sum21, vl);
                _sum30 = vfmadd_vf_f32m1(_pA0, pB[3], _sum30, vl);
                _sum31 = vfmadd_vf_f32m1(_pA1, pB[3], _sum31, vl);
                _sum40 = vfmadd_vf_f32m1(_pA0, pB[4], _sum40, vl);
                _sum41 = vfmadd_vf_f32m1(_pA1, pB[4], _sum41, vl);
                _sum50 = vfmadd_vf_f32m1(_pA0, pB[5], _sum50, vl);
                _sum51 = vfmadd_vf_f32m1(_pA1, pB[5], _sum51, vl);
                _sum60 = vfmadd_vf_f32m1(_pA0, pB[6], _sum60, vl);
                _sum61 = vfmadd_vf_f32m1(_pA1, pB[6], _sum61, vl);
                _sum70 = vfmadd_vf_f32m1(_pA0, pB[7], _sum70, vl);
                _sum71 = vfmadd_vf_f32m1(_pA1, pB[7], _sum71, vl);
                _sum80 = vfmadd_vf_f32m1(_pA0, pB[8], _sum80, vl);
                _sum81 = vfmadd_vf_f32m1(_pA1, pB[8], _sum81, vl);
                _sum90 = vfmadd_vf_f32m1(_pA0, pB[9], _sum90, vl);
                _sum91 = vfmadd_vf_f32m1(_pA1, pB[9], _sum91, vl);
                _suma0 = vfmadd_vf_f32m1(_pA0, pB[10], _suma0, vl);
                _suma1 = vfmadd_vf_f32m1(_pA1, pB[10], _suma1, vl);
                _sumb0 = vfmadd_vf_f32m1(_pA0, pB[11], _sumb0, vl);
                _sumb1 = vfmadd_vf_f32m1(_pA1, pB[11], _sumb1, vl);

                pA += 8;
                pB += 12;

                _pA0 = vle32_v_f32m1(pA, vl);
                _pA1 = vle32_v_f32m1(pA + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pA0, pB[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pA1, pB[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pA0, pB[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pA1, pB[1], _sum11, vl);
                _sum20 = vfmadd_vf_f32m1(_pA0, pB[2], _sum20, vl);
                _sum21 = vfmadd_vf_f32m1(_pA1, pB[2], _sum21, vl);
                _sum30 = vfmadd_vf_f32m1(_pA0, pB[3], _sum30, vl);
                _sum31 = vfmadd_vf_f32m1(_pA1, pB[3], _sum31, vl);
                _sum40 = vfmadd_vf_f32m1(_pA0, pB[4], _sum40, vl);
                _sum41 = vfmadd_vf_f32m1(_pA1, pB[4], _sum41, vl);
                _sum50 = vfmadd_vf_f32m1(_pA0, pB[5], _sum50, vl);
                _sum51 = vfmadd_vf_f32m1(_pA1, pB[5], _sum51, vl);
                _sum60 = vfmadd_vf_f32m1(_pA0, pB[6], _sum60, vl);
                _sum61 = vfmadd_vf_f32m1(_pA1, pB[6], _sum61, vl);
                _sum70 = vfmadd_vf_f32m1(_pA0, pB[7], _sum70, vl);
                _sum71 = vfmadd_vf_f32m1(_pA1, pB[7], _sum71, vl);
                _sum80 = vfmadd_vf_f32m1(_pA0, pB[8], _sum80, vl);
                _sum81 = vfmadd_vf_f32m1(_pA1, pB[8], _sum81, vl);
                _sum90 = vfmadd_vf_f32m1(_pA0, pB[9], _sum90, vl);
                _sum91 = vfmadd_vf_f32m1(_pA1, pB[9], _sum91, vl);
                _suma0 = vfmadd_vf_f32m1(_pA0, pB[10], _suma0, vl);
                _suma1 = vfmadd_vf_f32m1(_pA1, pB[10], _suma1, vl);
                _sumb0 = vfmadd_vf_f32m1(_pA0, pB[11], _sumb0, vl);
                _sumb1 = vfmadd_vf_f32m1(_pA1, pB[11], _sumb1, vl);

                pA += 8;
                pB += 12;
            }
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, vl);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pA0, pB[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pA1, pB[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pA0, pB[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pA1, pB[1], _sum11, vl);
                _sum20 = vfmadd_vf_f32m1(_pA0, pB[2], _sum20, vl);
                _sum21 = vfmadd_vf_f32m1(_pA1, pB[2], _sum21, vl);
                _sum30 = vfmadd_vf_f32m1(_pA0, pB[3], _sum30, vl);
                _sum31 = vfmadd_vf_f32m1(_pA1, pB[3], _sum31, vl);
                _sum40 = vfmadd_vf_f32m1(_pA0, pB[4], _sum40, vl);
                _sum41 = vfmadd_vf_f32m1(_pA1, pB[4], _sum41, vl);
                _sum50 = vfmadd_vf_f32m1(_pA0, pB[5], _sum50, vl);
                _sum51 = vfmadd_vf_f32m1(_pA1, pB[5], _sum51, vl);
                _sum60 = vfmadd_vf_f32m1(_pA0, pB[6], _sum60, vl);
                _sum61 = vfmadd_vf_f32m1(_pA1, pB[6], _sum61, vl);
                _sum70 = vfmadd_vf_f32m1(_pA0, pB[7], _sum70, vl);
                _sum71 = vfmadd_vf_f32m1(_pA1, pB[7], _sum71, vl);
                _sum80 = vfmadd_vf_f32m1(_pA0, pB[8], _sum80, vl);
                _sum81 = vfmadd_vf_f32m1(_pA1, pB[8], _sum81, vl);
                _sum90 = vfmadd_vf_f32m1(_pA0, pB[9], _sum90, vl);
                _sum91 = vfmadd_vf_f32m1(_pA1, pB[9], _sum91, vl);
                _suma0 = vfmadd_vf_f32m1(_pA0, pB[10], _suma0, vl);
                _suma1 = vfmadd_vf_f32m1(_pA1, pB[10], _suma1, vl);
                _sumb0 = vfmadd_vf_f32m1(_pA0, pB[11], _sumb0, vl);
                _sumb1 = vfmadd_vf_f32m1(_pA1, pB[11], _sumb1, vl);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum10, vl);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum20, vl);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum30, vl);
                    vse32_v_f32m1(outptr0 + 4 * 4, _sum40, vl);
                    vse32_v_f32m1(outptr0 + 4 * 5, _sum50, vl);
                    vse32_v_f32m1(outptr0 + 4 * 6, _sum60, vl);
                    vse32_v_f32m1(outptr0 + 4 * 7, _sum70, vl);
                    vse32_v_f32m1(outptr0 + 4 * 8, _sum80, vl);
                    vse32_v_f32m1(outptr0 + 4 * 9, _sum90, vl);
                    vse32_v_f32m1(outptr0 + 4 * 10, _suma0, vl);
                    vse32_v_f32m1(outptr0 + 4 * 11, _sumb0, vl);

                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum11, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 2, _sum21, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 3, _sum31, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 4, _sum41, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 5, _sum51, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 6, _sum61, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 7, _sum71, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 8, _sum81, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 9, _sum91, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 10, _suma1, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 11, _sumb1, vl);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x12_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71, _sum80, _sum81, _sum90, _sum91, _suma0, _suma1, _sumb0, _sumb1, vl);

                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum01, vl);
                    vse32_v_f32m1(outptr0 + 8, _sum10, vl);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum11, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum20, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 8, _sum21, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum30, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 4, _sum31, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 8, _sum40, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum41, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 4, _sum50, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 8, _sum51, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum60, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum61, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 8, _sum70, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 5, _sum71, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 5 + 4, _sum80, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 5 + 8, _sum81, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 6, _sum90, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 6 + 4, _sum91, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 6 + 8, _suma0, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 7, _suma1, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 7 + 4, _sumb0, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 7 + 8, _sumb1, vl);

                    outptr0 += 12;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum00, vl);
                vse32_v_f32m1(outptr + 4, _sum01, vl);
                vse32_v_f32m1(outptr + 4 * 2, _sum10, vl);
                vse32_v_f32m1(outptr + 4 * 3, _sum11, vl);
                vse32_v_f32m1(outptr + 4 * 4, _sum20, vl);
                vse32_v_f32m1(outptr + 4 * 5, _sum21, vl);
                vse32_v_f32m1(outptr + 4 * 6, _sum30, vl);
                vse32_v_f32m1(outptr + 4 * 7, _sum31, vl);
                vse32_v_f32m1(outptr + 4 * 8, _sum40, vl);
                vse32_v_f32m1(outptr + 4 * 9, _sum41, vl);
                vse32_v_f32m1(outptr + 4 * 10, _sum50, vl);
                vse32_v_f32m1(outptr + 4 * 11, _sum51, vl);
                vse32_v_f32m1(outptr + 4 * 12, _sum60, vl);
                vse32_v_f32m1(outptr + 4 * 13, _sum61, vl);
                vse32_v_f32m1(outptr + 4 * 14, _sum70, vl);
                vse32_v_f32m1(outptr + 4 * 15, _sum71, vl);
                vse32_v_f32m1(outptr + 4 * 16, _sum80, vl);
                vse32_v_f32m1(outptr + 4 * 17, _sum81, vl);
                vse32_v_f32m1(outptr + 4 * 18, _sum90, vl);
                vse32_v_f32m1(outptr + 4 * 19, _sum91, vl);
                vse32_v_f32m1(outptr + 4 * 20, _suma0, vl);
                vse32_v_f32m1(outptr + 4 * 21, _suma1, vl);
                vse32_v_f32m1(outptr + 4 * 22, _sumb0, vl);
                vse32_v_f32m1(outptr + 4 * 23, _sumb1, vl);
            }

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;
            vfloat32m1_t _sum20;
            vfloat32m1_t _sum21;
            vfloat32m1_t _sum30;
            vfloat32m1_t _sum31;
            vfloat32m1_t _sum40;
            vfloat32m1_t _sum41;
            vfloat32m1_t _sum50;
            vfloat32m1_t _sum51;
            vfloat32m1_t _sum60;
            vfloat32m1_t _sum61;
            vfloat32m1_t _sum70;
            vfloat32m1_t _sum71;

            if (k == 0)
            {
                _sum00 = vfmv_v_f_f32m1(0.f, vl);
                _sum01 = vfmv_v_f_f32m1(0.f, vl);
                _sum10 = vfmv_v_f_f32m1(0.f, vl);
                _sum11 = vfmv_v_f_f32m1(0.f, vl);
                _sum20 = vfmv_v_f_f32m1(0.f, vl);
                _sum21 = vfmv_v_f_f32m1(0.f, vl);
                _sum30 = vfmv_v_f_f32m1(0.f, vl);
                _sum31 = vfmv_v_f_f32m1(0.f, vl);
                _sum40 = vfmv_v_f_f32m1(0.f, vl);
                _sum41 = vfmv_v_f_f32m1(0.f, vl);
                _sum50 = vfmv_v_f_f32m1(0.f, vl);
                _sum51 = vfmv_v_f_f32m1(0.f, vl);
                _sum60 = vfmv_v_f_f32m1(0.f, vl);
                _sum61 = vfmv_v_f_f32m1(0.f, vl);
                _sum70 = vfmv_v_f_f32m1(0.f, vl);
                _sum71 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum11 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum20 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum21 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum30 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum31 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum40 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum41 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum50 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum51 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum60 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum61 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum70 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum71 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4, vl);
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
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4 * 1, vl);
                        _sum10 = vle32_v_f32m1(pC + 4 * 2, vl);
                        _sum11 = vle32_v_f32m1(pC + 4 * 3, vl);
                        _sum20 = vle32_v_f32m1(pC + 4 * 4, vl);
                        _sum21 = vle32_v_f32m1(pC + 4 * 5, vl);
                        _sum30 = vle32_v_f32m1(pC + 4 * 6, vl);
                        _sum31 = vle32_v_f32m1(pC + 4 * 7, vl);
                        _sum40 = vle32_v_f32m1(pC + 4 * 8, vl);
                        _sum41 = vle32_v_f32m1(pC + 4 * 9, vl);
                        _sum50 = vle32_v_f32m1(pC + 4 * 10, vl);
                        _sum51 = vle32_v_f32m1(pC + 4 * 11, vl);
                        _sum60 = vle32_v_f32m1(pC + 4 * 12, vl);
                        _sum61 = vle32_v_f32m1(pC + 4 * 13, vl);
                        _sum70 = vle32_v_f32m1(pC + 4 * 14, vl);
                        _sum71 = vle32_v_f32m1(pC + 4 * 15, vl);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum20 = vfmv_v_f_f32m1(pC[2], vl);
                        _sum30 = vfmv_v_f_f32m1(pC[3], vl);
                        _sum40 = vfmv_v_f_f32m1(pC[4], vl);
                        _sum50 = vfmv_v_f_f32m1(pC[5], vl);
                        _sum60 = vfmv_v_f_f32m1(pC[6], vl);
                        _sum70 = vfmv_v_f_f32m1(pC[7], vl);
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
                _sum00 = vle32_v_f32m1(outptr, vl);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, vl);
                _sum10 = vle32_v_f32m1(outptr + 4 * 2, vl);
                _sum11 = vle32_v_f32m1(outptr + 4 * 3, vl);
                _sum20 = vle32_v_f32m1(outptr + 4 * 4, vl);
                _sum21 = vle32_v_f32m1(outptr + 4 * 5, vl);
                _sum30 = vle32_v_f32m1(outptr + 4 * 6, vl);
                _sum31 = vle32_v_f32m1(outptr + 4 * 7, vl);
                _sum40 = vle32_v_f32m1(outptr + 4 * 8, vl);
                _sum41 = vle32_v_f32m1(outptr + 4 * 9, vl);
                _sum50 = vle32_v_f32m1(outptr + 4 * 10, vl);
                _sum51 = vle32_v_f32m1(outptr + 4 * 11, vl);
                _sum60 = vle32_v_f32m1(outptr + 4 * 12, vl);
                _sum61 = vle32_v_f32m1(outptr + 4 * 13, vl);
                _sum70 = vle32_v_f32m1(outptr + 4 * 14, vl);
                _sum71 = vle32_v_f32m1(outptr + 4 * 15, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, vl);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pA0, pB[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pA1, pB[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pA0, pB[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pA1, pB[1], _sum11, vl);
                _sum20 = vfmadd_vf_f32m1(_pA0, pB[2], _sum20, vl);
                _sum21 = vfmadd_vf_f32m1(_pA1, pB[2], _sum21, vl);
                _sum30 = vfmadd_vf_f32m1(_pA0, pB[3], _sum30, vl);
                _sum31 = vfmadd_vf_f32m1(_pA1, pB[3], _sum31, vl);
                _sum40 = vfmadd_vf_f32m1(_pA0, pB[4], _sum40, vl);
                _sum41 = vfmadd_vf_f32m1(_pA1, pB[4], _sum41, vl);
                _sum50 = vfmadd_vf_f32m1(_pA0, pB[5], _sum50, vl);
                _sum51 = vfmadd_vf_f32m1(_pA1, pB[5], _sum51, vl);
                _sum60 = vfmadd_vf_f32m1(_pA0, pB[6], _sum60, vl);
                _sum61 = vfmadd_vf_f32m1(_pA1, pB[6], _sum61, vl);
                _sum70 = vfmadd_vf_f32m1(_pA0, pB[7], _sum70, vl);
                _sum71 = vfmadd_vf_f32m1(_pA1, pB[7], _sum71, vl);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum10, vl);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum20, vl);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum30, vl);
                    vse32_v_f32m1(outptr0 + 4 * 4, _sum40, vl);
                    vse32_v_f32m1(outptr0 + 4 * 5, _sum50, vl);
                    vse32_v_f32m1(outptr0 + 4 * 6, _sum60, vl);
                    vse32_v_f32m1(outptr0 + 4 * 7, _sum70, vl);

                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum11, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 2, _sum21, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 3, _sum31, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 4, _sum41, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 5, _sum51, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 6, _sum61, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 7, _sum71, vl);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71, vl);

                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum01, vl);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum10, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum11, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum20, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 4, _sum21, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum30, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 4, _sum31, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum40, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum41, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 5, _sum50, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 5 + 4, _sum51, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 6, _sum60, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 6 + 4, _sum61, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 7, _sum70, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 7 + 4, _sum71, vl);

                    outptr0 += 8;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum00, vl);
                vse32_v_f32m1(outptr + 4, _sum01, vl);
                vse32_v_f32m1(outptr + 4 * 2, _sum10, vl);
                vse32_v_f32m1(outptr + 4 * 3, _sum11, vl);
                vse32_v_f32m1(outptr + 4 * 4, _sum20, vl);
                vse32_v_f32m1(outptr + 4 * 5, _sum21, vl);
                vse32_v_f32m1(outptr + 4 * 6, _sum30, vl);
                vse32_v_f32m1(outptr + 4 * 7, _sum31, vl);
                vse32_v_f32m1(outptr + 4 * 8, _sum40, vl);
                vse32_v_f32m1(outptr + 4 * 9, _sum41, vl);
                vse32_v_f32m1(outptr + 4 * 10, _sum50, vl);
                vse32_v_f32m1(outptr + 4 * 11, _sum51, vl);
                vse32_v_f32m1(outptr + 4 * 12, _sum60, vl);
                vse32_v_f32m1(outptr + 4 * 13, _sum61, vl);
                vse32_v_f32m1(outptr + 4 * 14, _sum70, vl);
                vse32_v_f32m1(outptr + 4 * 15, _sum71, vl);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;
            vfloat32m1_t _sum20;
            vfloat32m1_t _sum21;
            vfloat32m1_t _sum30;
            vfloat32m1_t _sum31;

            if (k == 0)
            {
                _sum00 = vfmv_v_f_f32m1(0.f, vl);
                _sum01 = vfmv_v_f_f32m1(0.f, vl);
                _sum10 = vfmv_v_f_f32m1(0.f, vl);
                _sum11 = vfmv_v_f_f32m1(0.f, vl);
                _sum20 = vfmv_v_f_f32m1(0.f, vl);
                _sum21 = vfmv_v_f_f32m1(0.f, vl);
                _sum30 = vfmv_v_f_f32m1(0.f, vl);
                _sum31 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum11 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum20 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum21 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum30 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum31 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4, vl);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum20 = _sum00;
                        _sum21 = _sum01;
                        _sum30 = _sum00;
                        _sum31 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4 * 1, vl);
                        _sum10 = vle32_v_f32m1(pC + 4 * 2, vl);
                        _sum11 = vle32_v_f32m1(pC + 4 * 3, vl);
                        _sum20 = vle32_v_f32m1(pC + 4 * 4, vl);
                        _sum21 = vle32_v_f32m1(pC + 4 * 5, vl);
                        _sum30 = vle32_v_f32m1(pC + 4 * 6, vl);
                        _sum31 = vle32_v_f32m1(pC + 4 * 7, vl);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum20 = vfmv_v_f_f32m1(pC[2], vl);
                        _sum30 = vfmv_v_f_f32m1(pC[3], vl);
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
                _sum00 = vle32_v_f32m1(outptr, vl);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, vl);
                _sum10 = vle32_v_f32m1(outptr + 4 * 2, vl);
                _sum11 = vle32_v_f32m1(outptr + 4 * 3, vl);
                _sum20 = vle32_v_f32m1(outptr + 4 * 4, vl);
                _sum21 = vle32_v_f32m1(outptr + 4 * 5, vl);
                _sum30 = vle32_v_f32m1(outptr + 4 * 6, vl);
                _sum31 = vle32_v_f32m1(outptr + 4 * 7, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, vl);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pA0, pB[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pA1, pB[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pA0, pB[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pA1, pB[1], _sum11, vl);
                _sum20 = vfmadd_vf_f32m1(_pA0, pB[2], _sum20, vl);
                _sum21 = vfmadd_vf_f32m1(_pA1, pB[2], _sum21, vl);
                _sum30 = vfmadd_vf_f32m1(_pA0, pB[3], _sum30, vl);
                _sum31 = vfmadd_vf_f32m1(_pA1, pB[3], _sum31, vl);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum10, vl);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum20, vl);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum30, vl);

                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum11, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 2, _sum21, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 3, _sum31, vl);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, vl);

                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 1, _sum01, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum10, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum11, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum20, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 5, _sum21, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 6, _sum30, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 7, _sum31, vl);

                    outptr0 += 4;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum00, vl);
                vse32_v_f32m1(outptr + 4, _sum01, vl);
                vse32_v_f32m1(outptr + 4 * 2, _sum10, vl);
                vse32_v_f32m1(outptr + 4 * 3, _sum11, vl);
                vse32_v_f32m1(outptr + 4 * 4, _sum20, vl);
                vse32_v_f32m1(outptr + 4 * 5, _sum21, vl);
                vse32_v_f32m1(outptr + 4 * 6, _sum30, vl);
                vse32_v_f32m1(outptr + 4 * 7, _sum31, vl);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;

            if (k == 0)
            {
                _sum00 = vfmv_v_f_f32m1(0.f, vl);
                _sum01 = vfmv_v_f_f32m1(0.f, vl);
                _sum10 = vfmv_v_f_f32m1(0.f, vl);
                _sum11 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum11 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4, vl);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4 * 1, vl);
                        _sum10 = vle32_v_f32m1(pC + 4 * 2, vl);
                        _sum11 = vle32_v_f32m1(pC + 4 * 3, vl);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum00 = vle32_v_f32m1(outptr, vl);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, vl);
                _sum10 = vle32_v_f32m1(outptr + 4 * 2, vl);
                _sum11 = vle32_v_f32m1(outptr + 4 * 3, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, vl);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pA0, pB[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pA1, pB[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pA0, pB[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pA1, pB[1], _sum11, vl);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum10, vl);

                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum11, vl);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    float sum1[8];
                    vse32_v_f32m1(sum0, _sum00, vl);
                    vse32_v_f32m1(sum0 + 4, _sum01, vl);
                    vse32_v_f32m1(sum1, _sum10, vl);
                    vse32_v_f32m1(sum1 + 4, _sum11, vl);

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
                vse32_v_f32m1(outptr, _sum00, vl);
                vse32_v_f32m1(outptr + 4, _sum01, vl);
                vse32_v_f32m1(outptr + 4 * 2, _sum10, vl);
                vse32_v_f32m1(outptr + 4 * 3, _sum11, vl);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;

            if (k == 0)
            {
                _sum00 = vfmv_v_f_f32m1(0.f, vl);
                _sum01 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4, vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4, vl);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = _sum00;
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum00 = vle32_v_f32m1(outptr, vl);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, vl);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, vl);

                vfloat32m1_t _pB = vfmv_v_f_f32m1(pB[0], vl);

                _sum00 = vfmadd_vv_f32m1(_pA0, _pB, _sum00, vl);
                _sum01 = vfmadd_vv_f32m1(_pA1, _pB, _sum01, vl);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, vl);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    vse32_v_f32m1(sum0, _sum00, vl);
                    vse32_v_f32m1(sum0 + 4, _sum01, vl);

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
                vse32_v_f32m1(outptr, _sum00, vl);
                vse32_v_f32m1(outptr + 4, _sum01, vl);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
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
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;
            vfloat32m1_t _sum4;
            vfloat32m1_t _sum5;
            vfloat32m1_t _sum6;
            vfloat32m1_t _sum7;
            vfloat32m1_t _sum8;
            vfloat32m1_t _sum9;
            vfloat32m1_t _suma;
            vfloat32m1_t _sumb;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m1(0.f, vl);
                _sum1 = vfmv_v_f_f32m1(0.f, vl);
                _sum2 = vfmv_v_f_f32m1(0.f, vl);
                _sum3 = vfmv_v_f_f32m1(0.f, vl);
                _sum4 = vfmv_v_f_f32m1(0.f, vl);
                _sum5 = vfmv_v_f_f32m1(0.f, vl);
                _sum6 = vfmv_v_f_f32m1(0.f, vl);
                _sum7 = vfmv_v_f_f32m1(0.f, vl);
                _sum8 = vfmv_v_f_f32m1(0.f, vl);
                _sum9 = vfmv_v_f_f32m1(0.f, vl);
                _suma = vfmv_v_f_f32m1(0.f, vl);
                _sumb = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum2 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum3 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum4 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum5 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum6 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum7 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum8 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum9 = vfmv_v_f_f32m1(pC[0], vl);
                        _suma = vfmv_v_f_f32m1(pC[0], vl);
                        _sumb = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
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
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = vle32_v_f32m1(pC + 4, vl);
                        _sum2 = vle32_v_f32m1(pC + 8, vl);
                        _sum3 = vle32_v_f32m1(pC + 12, vl);
                        _sum4 = vle32_v_f32m1(pC + 16, vl);
                        _sum5 = vle32_v_f32m1(pC + 20, vl);
                        _sum6 = vle32_v_f32m1(pC + 24, vl);
                        _sum7 = vle32_v_f32m1(pC + 28, vl);
                        _sum8 = vle32_v_f32m1(pC + 32, vl);
                        _sum9 = vle32_v_f32m1(pC + 36, vl);
                        _suma = vle32_v_f32m1(pC + 40, vl);
                        _sumb = vle32_v_f32m1(pC + 44, vl);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m1(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m1(pC[3], vl);
                        _sum4 = vfmv_v_f_f32m1(pC[4], vl);
                        _sum5 = vfmv_v_f_f32m1(pC[5], vl);
                        _sum6 = vfmv_v_f_f32m1(pC[6], vl);
                        _sum7 = vfmv_v_f_f32m1(pC[7], vl);
                        _sum8 = vfmv_v_f_f32m1(pC[8], vl);
                        _sum9 = vfmv_v_f_f32m1(pC[9], vl);
                        _suma = vfmv_v_f_f32m1(pC[10], vl);
                        _sumb = vfmv_v_f_f32m1(pC[11], vl);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, vl);
                _sum1 = vle32_v_f32m1(outptr + 4 * 1, vl);
                _sum2 = vle32_v_f32m1(outptr + 4 * 2, vl);
                _sum3 = vle32_v_f32m1(outptr + 4 * 3, vl);
                _sum4 = vle32_v_f32m1(outptr + 4 * 4, vl);
                _sum5 = vle32_v_f32m1(outptr + 4 * 5, vl);
                _sum6 = vle32_v_f32m1(outptr + 4 * 6, vl);
                _sum7 = vle32_v_f32m1(outptr + 4 * 7, vl);
                _sum8 = vle32_v_f32m1(outptr + 4 * 8, vl);
                _sum9 = vle32_v_f32m1(outptr + 4 * 9, vl);
                _suma = vle32_v_f32m1(outptr + 4 * 10, vl);
                _sumb = vle32_v_f32m1(outptr + 4 * 11, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, vl);

                _sum0 = vfmadd_vf_f32m1(_pA, pB[0], _sum0, vl);
                _sum1 = vfmadd_vf_f32m1(_pA, pB[1], _sum1, vl);
                _sum2 = vfmadd_vf_f32m1(_pA, pB[2], _sum2, vl);
                _sum3 = vfmadd_vf_f32m1(_pA, pB[3], _sum3, vl);
                _sum4 = vfmadd_vf_f32m1(_pA, pB[4], _sum4, vl);
                _sum5 = vfmadd_vf_f32m1(_pA, pB[5], _sum5, vl);
                _sum6 = vfmadd_vf_f32m1(_pA, pB[6], _sum6, vl);
                _sum7 = vfmadd_vf_f32m1(_pA, pB[7], _sum7, vl);
                _sum8 = vfmadd_vf_f32m1(_pA, pB[8], _sum8, vl);
                _sum9 = vfmadd_vf_f32m1(_pA, pB[9], _sum9, vl);
                _suma = vfmadd_vf_f32m1(_pA, pB[10], _suma, vl);
                _sumb = vfmadd_vf_f32m1(_pA, pB[11], _sumb, vl);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum1, vl);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum2, vl);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum3, vl);
                    vse32_v_f32m1(outptr0 + 4 * 4, _sum4, vl);
                    vse32_v_f32m1(outptr0 + 4 * 5, _sum5, vl);
                    vse32_v_f32m1(outptr0 + 4 * 6, _sum6, vl);
                    vse32_v_f32m1(outptr0 + 4 * 7, _sum7, vl);
                    vse32_v_f32m1(outptr0 + 4 * 8, _sum8, vl);
                    vse32_v_f32m1(outptr0 + 4 * 9, _sum9, vl);
                    vse32_v_f32m1(outptr0 + 4 * 10, _suma, vl);
                    vse32_v_f32m1(outptr0 + 4 * 11, _sumb, vl);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose4x12_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb, vl);

                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum1, vl);
                    vse32_v_f32m1(outptr0 + 8, _sum2, vl);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum3, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum4, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 8, _sum5, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum6, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 4, _sum7, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 8, _sum8, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum9, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 4, _suma, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 8, _sumb, vl);
                    outptr0 += 12;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, vl);
                vse32_v_f32m1(outptr + 4, _sum1, vl);
                vse32_v_f32m1(outptr + 4 * 2, _sum2, vl);
                vse32_v_f32m1(outptr + 4 * 3, _sum3, vl);
                vse32_v_f32m1(outptr + 4 * 4, _sum4, vl);
                vse32_v_f32m1(outptr + 4 * 5, _sum5, vl);
                vse32_v_f32m1(outptr + 4 * 6, _sum6, vl);
                vse32_v_f32m1(outptr + 4 * 7, _sum7, vl);
                vse32_v_f32m1(outptr + 4 * 8, _sum8, vl);
                vse32_v_f32m1(outptr + 4 * 9, _sum9, vl);
                vse32_v_f32m1(outptr + 4 * 10, _suma, vl);
                vse32_v_f32m1(outptr + 4 * 11, _sumb, vl);
            }

            outptr += 48;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;
            vfloat32m1_t _sum4;
            vfloat32m1_t _sum5;
            vfloat32m1_t _sum6;
            vfloat32m1_t _sum7;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m1(0.f, vl);
                _sum1 = vfmv_v_f_f32m1(0.f, vl);
                _sum2 = vfmv_v_f_f32m1(0.f, vl);
                _sum3 = vfmv_v_f_f32m1(0.f, vl);
                _sum4 = vfmv_v_f_f32m1(0.f, vl);
                _sum5 = vfmv_v_f_f32m1(0.f, vl);
                _sum6 = vfmv_v_f_f32m1(0.f, vl);
                _sum7 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum2 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum3 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum4 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum5 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum6 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum7 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
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
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = vle32_v_f32m1(pC + 4, vl);
                        _sum2 = vle32_v_f32m1(pC + 8, vl);
                        _sum3 = vle32_v_f32m1(pC + 12, vl);
                        _sum4 = vle32_v_f32m1(pC + 16, vl);
                        _sum5 = vle32_v_f32m1(pC + 20, vl);
                        _sum6 = vle32_v_f32m1(pC + 24, vl);
                        _sum7 = vle32_v_f32m1(pC + 28, vl);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m1(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m1(pC[3], vl);
                        _sum4 = vfmv_v_f_f32m1(pC[4], vl);
                        _sum5 = vfmv_v_f_f32m1(pC[5], vl);
                        _sum6 = vfmv_v_f_f32m1(pC[6], vl);
                        _sum7 = vfmv_v_f_f32m1(pC[7], vl);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, vl);
                _sum1 = vle32_v_f32m1(outptr + 4 * 1, vl);
                _sum2 = vle32_v_f32m1(outptr + 4 * 2, vl);
                _sum3 = vle32_v_f32m1(outptr + 4 * 3, vl);
                _sum4 = vle32_v_f32m1(outptr + 4 * 4, vl);
                _sum5 = vle32_v_f32m1(outptr + 4 * 5, vl);
                _sum6 = vle32_v_f32m1(outptr + 4 * 6, vl);
                _sum7 = vle32_v_f32m1(outptr + 4 * 7, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, vl);

                _sum0 = vfmadd_vf_f32m1(_pA, pB[0], _sum0, vl);
                _sum1 = vfmadd_vf_f32m1(_pA, pB[1], _sum1, vl);
                _sum2 = vfmadd_vf_f32m1(_pA, pB[2], _sum2, vl);
                _sum3 = vfmadd_vf_f32m1(_pA, pB[3], _sum3, vl);
                _sum4 = vfmadd_vf_f32m1(_pA, pB[4], _sum4, vl);
                _sum5 = vfmadd_vf_f32m1(_pA, pB[5], _sum5, vl);
                _sum6 = vfmadd_vf_f32m1(_pA, pB[6], _sum6, vl);
                _sum7 = vfmadd_vf_f32m1(_pA, pB[7], _sum7, vl);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum1, vl);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum2, vl);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum3, vl);
                    vse32_v_f32m1(outptr0 + 4 * 4, _sum4, vl);
                    vse32_v_f32m1(outptr0 + 4 * 5, _sum5, vl);
                    vse32_v_f32m1(outptr0 + 4 * 6, _sum6, vl);
                    vse32_v_f32m1(outptr0 + 4 * 7, _sum7, vl);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, vl);

                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum1, vl);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum2, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum3, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum4, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 4, _sum5, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum6, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 4, _sum7, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, vl);
                vse32_v_f32m1(outptr + 4, _sum1, vl);
                vse32_v_f32m1(outptr + 4 * 2, _sum2, vl);
                vse32_v_f32m1(outptr + 4 * 3, _sum3, vl);
                vse32_v_f32m1(outptr + 4 * 4, _sum4, vl);
                vse32_v_f32m1(outptr + 4 * 5, _sum5, vl);
                vse32_v_f32m1(outptr + 4 * 6, _sum6, vl);
                vse32_v_f32m1(outptr + 4 * 7, _sum7, vl);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m1(0.f, vl);
                _sum1 = vfmv_v_f_f32m1(0.f, vl);
                _sum2 = vfmv_v_f_f32m1(0.f, vl);
                _sum3 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum2 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum3 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = vle32_v_f32m1(pC + 4, vl);
                        _sum2 = vle32_v_f32m1(pC + 8, vl);
                        _sum3 = vle32_v_f32m1(pC + 12, vl);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m1(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m1(pC[3], vl);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, vl);
                _sum1 = vle32_v_f32m1(outptr + 4 * 1, vl);
                _sum2 = vle32_v_f32m1(outptr + 4 * 2, vl);
                _sum3 = vle32_v_f32m1(outptr + 4 * 3, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, vl);

                _sum0 = vfmadd_vf_f32m1(_pA, pB[0], _sum0, vl);
                _sum1 = vfmadd_vf_f32m1(_pA, pB[1], _sum1, vl);
                _sum2 = vfmadd_vf_f32m1(_pA, pB[2], _sum2, vl);
                _sum3 = vfmadd_vf_f32m1(_pA, pB[3], _sum3, vl);
                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum1, vl);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum2, vl);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum3, vl);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3, vl);

                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 1, _sum1, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum2, vl);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum3, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, vl);
                vse32_v_f32m1(outptr + 4, _sum1, vl);
                vse32_v_f32m1(outptr + 4 * 2, _sum2, vl);
                vse32_v_f32m1(outptr + 4 * 3, _sum3, vl);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m1(0.f, vl);
                _sum1 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = vle32_v_f32m1(pC + 4, vl);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[1], vl);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, vl);
                _sum1 = vle32_v_f32m1(outptr + 4, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, vl);

                _sum0 = vfmadd_vf_f32m1(_pA, pB[0], _sum0, vl);
                _sum1 = vfmadd_vf_f32m1(_pA, pB[1], _sum1, vl);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum1, vl);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    vse32_v_f32m1(sum0, _sum0, vl);
                    vse32_v_f32m1(sum1, _sum1, vl);

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
                vse32_v_f32m1(outptr, _sum0, vl);
                vse32_v_f32m1(outptr + 4, _sum1, vl);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat32m1_t _sum0;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, vl);
                vfloat32m1_t _pB = vfmv_v_f_f32m1(pB[0], vl);

                _sum0 = vfmadd_vv_f32m1(_pA, _pB, _sum0, vl);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, vl);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    vse32_v_f32m1(sum0, _sum0, vl);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, vl);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __riscv_vector
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
#if __riscv_vector
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum02;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;
            vfloat32m1_t _sum12;

            if (k == 0)
            {
                _sum00 = vfmv_v_f_f32m1(0.f, vl);
                _sum01 = vfmv_v_f_f32m1(0.f, vl);
                _sum02 = vfmv_v_f_f32m1(0.f, vl);
                _sum10 = vfmv_v_f_f32m1(0.f, vl);
                _sum11 = vfmv_v_f_f32m1(0.f, vl);
                _sum12 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum02 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum11 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum12 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum02 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum11 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum12 = vfmv_v_f_f32m1(pC[1], vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vlseg2e32_v_f32m1(&_sum00, &_sum10, pC, vl);
                        vlseg2e32_v_f32m1(&_sum01, &_sum11, pC + 8, vl);
                        vlseg2e32_v_f32m1(&_sum02, &_sum12, pC + 16, vl);
                        pC += 24;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4, vl);
                        _sum02 = vle32_v_f32m1(pC + 8, vl);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum12 = _sum02;
                        pC += 12;
                    }
                }
            }
            else
            {
                vlseg2e32_v_f32m1(&_sum00, &_sum10, outptr, vl);
                vlseg2e32_v_f32m1(&_sum01, &_sum11, outptr + 8, vl);
                vlseg2e32_v_f32m1(&_sum02, &_sum12, outptr + 16, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, vl);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, vl);
                vfloat32m1_t _pB2 = vle32_v_f32m1(pB + 8, vl);

                _sum00 = vfmadd_vf_f32m1(_pB0, pA[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pB1, pA[0], _sum01, vl);
                _sum02 = vfmadd_vf_f32m1(_pB2, pA[0], _sum02, vl);
                _sum10 = vfmadd_vf_f32m1(_pB0, pA[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pB1, pA[1], _sum11, vl);
                _sum12 = vfmadd_vf_f32m1(_pB2, pA[1], _sum12, vl);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum01, vl);
                    vse32_v_f32m1(outptr0 + 8, _sum02, vl);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum10, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum11, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 8, _sum12, vl);
                    outptr0 += 12;
                }
            }
            else
            {
                vsseg2e32_v_f32m1(outptr, _sum00, _sum10, vl);
                vsseg2e32_v_f32m1(outptr + 8, _sum01, _sum11, vl);
                vsseg2e32_v_f32m1(outptr + 16, _sum02, _sum12, vl);
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;

            if (k == 0)
            {
                _sum00 = vfmv_v_f_f32m1(0.f, vl);
                _sum01 = vfmv_v_f_f32m1(0.f, vl);
                _sum10 = vfmv_v_f_f32m1(0.f, vl);
                _sum11 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum11 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum01 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum10 = vfmv_v_f_f32m1(pC[1], vl);
                        _sum11 = vfmv_v_f_f32m1(pC[1], vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vlseg2e32_v_f32m1(&_sum00, &_sum10, pC, vl);
                        vlseg2e32_v_f32m1(&_sum01, &_sum11, pC + 8, vl);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vle32_v_f32m1(pC, vl);
                        _sum01 = vle32_v_f32m1(pC + 4, vl);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        pC += 8;
                    }
                }
            }
            else
            {
                vlseg2e32_v_f32m1(&_sum00, &_sum10, outptr, vl);
                vlseg2e32_v_f32m1(&_sum01, &_sum11, outptr + 8, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, vl);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, vl);

                _sum00 = vfmadd_vf_f32m1(_pB0, pA[0], _sum00, vl);
                _sum01 = vfmadd_vf_f32m1(_pB1, pA[0], _sum01, vl);
                _sum10 = vfmadd_vf_f32m1(_pB0, pA[1], _sum10, vl);
                _sum11 = vfmadd_vf_f32m1(_pB1, pA[1], _sum11, vl);
                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum00, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum01, vl);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum10, vl);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum11, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                vsseg2e32_v_f32m1(outptr, _sum00, _sum10, vl);
                vsseg2e32_v_f32m1(outptr + 8, _sum01, _sum11, vl);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m1(0.f, vl);
                _sum1 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[1], vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vlseg2e32_v_f32m1(&_sum0, &_sum1, pC, vl);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
            }
            else
            {
                vfloat32m1_t _tmp0;
                vfloat32m1_t _tmp1;
                vlseg2e32_v_f32m1(&_tmp0, &_tmp1, outptr, vl);
                _sum0 = _tmp0;
                _sum1 = _tmp1;
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB = vle32_v_f32m1(pB, vl);

                _sum0 = vfmadd_vf_f32m1(_pB, pA[0], _sum0, vl);
                _sum1 = vfmadd_vf_f32m1(_pB, pA[1], _sum1, vl);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum1, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                vsseg2e32_v_f32m1(outptr, _sum0, _sum1, vl);
            }

            outptr += 8;
        }
#endif // __riscv_vector
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
#if __riscv_vector
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m1(0.f, vl);
                _sum1 = vfmv_v_f_f32m1(0.f, vl);
                _sum2 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum2 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = vle32_v_f32m1(pC + 4, vl);
                        _sum2 = vle32_v_f32m1(pC + 8, vl);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, vl);
                _sum1 = vle32_v_f32m1(outptr + 4, vl);
                _sum2 = vle32_v_f32m1(outptr + 8, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, vl);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, vl);
                vfloat32m1_t _pB2 = vle32_v_f32m1(pB + 8, vl);

                vfloat32m1_t _pA0 = vfmv_v_f_f32m1(pA[0], vl);

                _sum0 = vfmadd_vv_f32m1(_pA0, _pB0, _sum0, vl);
                _sum1 = vfmadd_vv_f32m1(_pA0, _pB1, _sum1, vl);
                _sum2 = vfmadd_vv_f32m1(_pA0, _pB2, _sum2, vl);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum1, vl);
                    vse32_v_f32m1(outptr0 + 8, _sum2, vl);
                    outptr0 += 12;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, vl);
                vse32_v_f32m1(outptr + 4, _sum1, vl);
                vse32_v_f32m1(outptr + 8, _sum2, vl);
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m1(0.f, vl);
                _sum1 = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vfmv_v_f_f32m1(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m1(pC, vl);
                        _sum1 = vle32_v_f32m1(pC + 4, vl);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, vl);
                _sum1 = vle32_v_f32m1(outptr + 4, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, vl);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, vl);

                vfloat32m1_t _pA0 = vfmv_v_f_f32m1(pA[0], vl);
                _sum0 = vfmadd_vv_f32m1(_pA0, _pB0, _sum0, vl);
                _sum1 = vfmadd_vv_f32m1(_pA0, _pB1, _sum1, vl);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum0, vl);
                    vse32_v_f32m1(outptr0 + 4, _sum1, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, vl);
                vse32_v_f32m1(outptr + 4, _sum1, vl);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m1_t _sum;

            if (k == 0)
            {
                _sum = vfmv_v_f_f32m1(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = vfmv_v_f_f32m1(pC[0], vl);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = vle32_v_f32m1(pC, vl);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum = vle32_v_f32m1(outptr, vl);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB = vle32_v_f32m1(pB, vl);
                vfloat32m1_t _pA = vfmv_v_f_f32m1(pA[0], vl);

                _sum = vfmadd_vv_f32m1(_pA, _pB, _sum, vl);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum, vl);
            }

            outptr += 4;
        }
#endif // __riscv_vector
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

    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(8, tile_size / 8 * 8);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(float) / TILE_K);
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

static int gemm_riscv(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, size_t vl, const Option& opt)
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
#if TIME_TEST
    gettimeofday(&start_time, NULL);
#endif
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
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk, vl);
        }
        else
        {
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk, vl);
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
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj, vl);
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
                        transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk, vl);
                    }
                    else
                    {
                        pack_A_tile(A, AT_tile, i, max_ii, k, max_kk, vl);
                    }
                }

                bool k_end = !output_transpose && k + TILE_K >= K;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end, vl);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj, vl);
            }
        }
    }

    return 0;
}

static int gemm_AT_riscv(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, size_t vl, const Option& opt)
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
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk, vl);
        }
        else
        {
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk, vl);
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
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj, vl);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);
                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end, vl);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj, vl);
            }
        }
    }

    return 0;
}

static int gemm_BT_riscv(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, size_t vl, const Option& opt)
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
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj, vl);
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
                        transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk, vl);
                    }
                    else
                    {
                        pack_A_tile(A, AT_tile, i, max_ii, k, max_kk, vl);
                    }
                }

                bool k_end = !output_transpose && k + TILE_K >= K;

                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end, vl);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj, vl);
            }
        }
    }

    return 0;
}

static int gemm_AT_BT_riscv(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, size_t vl, const Option& opt)
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
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj, vl);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;

                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end, vl);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj, vl);
            }
        }
    }

    return 0;
}

int Gemm_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1, bottom_blob);
    std::vector<Mat> top_blobs(1, top_blob);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Gemm_riscv::create_pipeline(const Option& opt)
{
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
                    transpose_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk, vl);
                }
                else
                {
                    pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk, vl);
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
                    pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk, vl);
                }
                else
                {
                    transpose_pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk, vl);
                }
            }
        }

        B_data.release();
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        CT_data = C_data;

#if __riscv_vector
        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
            int C_elempack = constantM % 4 == 0 ? 4 : 1;
            convert_packing(C_data, CT_data, C_elempack, opt);
        }
#endif // __riscv_vector

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

int Gemm_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % 4 == 0 ? 4 : 1;
    }
#endif // __riscv_vector
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
        ret = gemm_AT_BT_riscv(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, vl, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_riscv(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, vl, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_riscv(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, vl, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_riscv(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, vl, opt);
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

} // namespace ncnn
