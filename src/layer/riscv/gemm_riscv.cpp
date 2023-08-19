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

namespace ncnn {

Gemm_riscv::Gemm_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
    one_blob_only = false;
    support_inplace = false;
}

void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL);
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
                vfloat32m1_t _r0l = vle32_v_f32m1(p0, VL);
                vfloat32m1_t _r0h = vle32_v_f32m1(p0 + 4, VL);
                vfloat32m1_t _r1l = vle32_v_f32m1(p1, VL);
                vfloat32m1_t _r1h = vle32_v_f32m1(p1 + 4, VL);
                vfloat32m1_t _r2l = vle32_v_f32m1(p2, VL);
                vfloat32m1_t _r2h = vle32_v_f32m1(p2 + 4, VL);
                vfloat32m1_t _r3l = vle32_v_f32m1(p3, VL);
                vfloat32m1_t _r3h = vle32_v_f32m1(p3 + 4, VL);
                vfloat32m1_t _r4l = vle32_v_f32m1(p4, VL);
                vfloat32m1_t _r4h = vle32_v_f32m1(p4 + 4, VL);
                vfloat32m1_t _r5l = vle32_v_f32m1(p5, VL);
                vfloat32m1_t _r5h = vle32_v_f32m1(p5 + 4, VL);
                vfloat32m1_t _r6l = vle32_v_f32m1(p6, VL);
                vfloat32m1_t _r6h = vle32_v_f32m1(p6 + 4, VL);
                vfloat32m1_t _r7l = vle32_v_f32m1(p7, VL);
                vfloat32m1_t _r7h = vle32_v_f32m1(p7 + 4, VL);
                transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h);
                vse32_v_f32m1(pp, _r0l, VL);
                vse32_v_f32m1(pp + 4, _r0h, VL);
                vse32_v_f32m1(pp + 8, _r1l, VL);
                vse32_v_f32m1(pp + 12, _r1h, VL);
                vse32_v_f32m1(pp + 8 * 2, _r2l, VL);
                vse32_v_f32m1(pp + 8 * 2 + 4, _r2h, VL);
                vse32_v_f32m1(pp + 8 * 3, _r3l, VL);
                vse32_v_f32m1(pp + 8 * 3 + 4, _r3h, VL);
                vse32_v_f32m1(pp + 8 * 4, _r4l, VL);
                vse32_v_f32m1(pp + 8 * 4 + 4, _r4h, VL);
                vse32_v_f32m1(pp + 8 * 5, _r5l, VL);
                vse32_v_f32m1(pp + 8 * 5 + 4, _r5h, VL);
                vse32_v_f32m1(pp + 8 * 6, _r6l, VL);
                vse32_v_f32m1(pp + 8 * 6 + 4, _r6h, VL);
                vse32_v_f32m1(pp + 8 * 7, _r7l, VL);
                vse32_v_f32m1(pp + 8 * 7 + 4, _r7h, VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
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
                // vfloat32m4_t _r0123;
                // vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                // vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL);
                // vse32_v_f32m1(pp + 8, vle32_v_f32m1(p2, VL), VL);
                // vse32_v_f32m1(pp + 12, vle32_v_f32m1(p3, VL), VL);
                store_float_v4(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p1, VL), vle32_v_f32m1(p2, VL), vle32_v_f32m1(p3, VL), pp);
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
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // vfloat32m1x2_t _r01;
                // _r01.val[0] = vle32_v_f32m1(p0, VL);
                // _r01.val[1] = vle32_v_f32m1(p1, VL);
                store_float32_v2(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p1, VL), pp);
                // vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL); 
                // vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL); 
                // vsseg2e32_v_f32m1x2(pp, _r01, VL);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
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
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += 4;
            }
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
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r4567, 0), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r4567, 1), VL);
                vse32_v_f32m1(pp + 4 * 4, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 5, vget_f32m1x4_f32m1(_r4567, 2), VL);
                vse32_v_f32m1(pp + 4 * 6, vget_f32m1x4_f32m1(_r0123, 3), VL);
                vse32_v_f32m1(pp + 4 * 7, vget_f32m1x4_f32m1(_r4567, 3), VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, VL), VL);
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
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r0123, 3), VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // vfloat32m1x2_t _r01;
                // vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL); 
                // vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, VL), VL);
                store_float32_v2(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p0 + 4, VL), pp);
                // vsseg2e32_v_f32m1x2(pp, _r01, VL);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
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
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
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
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;
            const float* p2 = (const float*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL);
                vse32_v_f32m1(pp + 8, vle32_v_f32m1(p2, VL), VL);
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
                vfloat32m1_t _r0 = vle32_v_f32m1(p0, VL);
                vfloat32m1_t _r1 = vle32_v_f32m1(p1, VL);
                vfloat32m1_t _r2 = vle32_v_f32m1(p2, VL);
                vfloat32m1_t _r3 = vle32_v_f32m1(p3, VL);
                vfloat32m1_t _r4 = vle32_v_f32m1(p4, VL);
                vfloat32m1_t _r5 = vle32_v_f32m1(p5, VL);
                vfloat32m1_t _r6 = vle32_v_f32m1(p6, VL);
                vfloat32m1_t _r7 = vle32_v_f32m1(p7, VL);
                vfloat32m1_t _r8 = vle32_v_f32m1(p8, VL);
                vfloat32m1_t _r9 = vle32_v_f32m1(p9, VL);
                vfloat32m1_t _ra = vle32_v_f32m1(pa, VL);
                vfloat32m1_t _rb = vle32_v_f32m1(pb, VL);

                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                transpose4x4_ps(_r8, _r9, _ra, _rb);

                vse32_v_f32m1(pp, _r0, VL);
                vse32_v_f32m1(pp + 4, _r4, VL);
                vse32_v_f32m1(pp + 4 * 2, _r8, VL);
                vse32_v_f32m1(pp + 4 * 3, _r1, VL);
                vse32_v_f32m1(pp + 4 * 4, _r5, VL);
                vse32_v_f32m1(pp + 4 * 5, _r9, VL);
                vse32_v_f32m1(pp + 4 * 6, _r2, VL);
                vse32_v_f32m1(pp + 4 * 7, _r6, VL);
                vse32_v_f32m1(pp + 4 * 8, _ra, VL);
                vse32_v_f32m1(pp + 4 * 9, _r3, VL);
                vse32_v_f32m1(pp + 4 * 10, _r7, VL);
                vse32_v_f32m1(pp + 4 * 11, _rb, VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL);
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
                vfloat32m1_t _r0 = vle32_v_f32m1(p0, VL);
                vfloat32m1_t _r1 = vle32_v_f32m1(p1, VL);
                vfloat32m1_t _r2 = vle32_v_f32m1(p2, VL);
                vfloat32m1_t _r3 = vle32_v_f32m1(p3, VL);
                vfloat32m1_t _r4 = vle32_v_f32m1(p4, VL);
                vfloat32m1_t _r5 = vle32_v_f32m1(p5, VL);
                vfloat32m1_t _r6 = vle32_v_f32m1(p6, VL);
                vfloat32m1_t _r7 = vle32_v_f32m1(p7, VL);

                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);

                vse32_v_f32m1(pp, _r0, VL);
                vse32_v_f32m1(pp + 4, _r4, VL);
                vse32_v_f32m1(pp + 4 * 2, _r1, VL);
                vse32_v_f32m1(pp + 4 * 3, _r5, VL);
                vse32_v_f32m1(pp + 4 * 4, _r2, VL);
                vse32_v_f32m1(pp + 4 * 5, _r6, VL);
                vse32_v_f32m1(pp + 4 * 6, _r3, VL);
                vse32_v_f32m1(pp + 4 * 7, _r7, VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
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
                // vfloat32m1x4_t _r0123;
                // vget_f32m1x4_f32m1(_r0123, 0) = vle32_v_f32m1(p0, VL);
                // vget_f32m1x4_f32m1(_r0123, 1) = vle32_v_f32m1(p1, VL);
                // vget_f32m1x4_f32m1(_r0123, 2) = vle32_v_f32m1(p2, VL);
                // vget_f32m1x4_f32m1(_r0123, 3) = vle32_v_f32m1(p3, VL);
                // vst4q_f32(pp, _r0123);
                store_float_v4(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p1, VL), vle32_v_f32m1(p2, VL), vle32_v_f32m1(p3, VL), pp);
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
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // float32x4x2_t _r01;
                // vget_f32m1x4_f32m1(_r01, 0) = vle32_v_f32m1(p0, VL);
                // vget_f32m1x4_f32m1(_r01, 1) = vle32_v_f32m1(p1, VL);
                // vst2q_f32(pp, _r01);
                store_float32_v2(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p1, VL), pp);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
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
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += 4;
            }
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
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                vfloat32m1x4_t _r89ab = vlseg4e32_v_f32m1x4(p0 + 32, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r4567, 0), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r89ab, 0), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 4, vget_f32m1x4_f32m1(_r4567, 1), VL);
                vse32_v_f32m1(pp + 4 * 5, vget_f32m1x4_f32m1(_r89ab, 1), VL);
                vse32_v_f32m1(pp + 4 * 6, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 7, vget_f32m1x4_f32m1(_r4567, 2), VL);
                vse32_v_f32m1(pp + 4 * 8, vget_f32m1x4_f32m1(_r89ab, 2), VL);
                vse32_v_f32m1(pp + 4 * 9, vget_f32m1x4_f32m1(_r0123, 3), VL);
                vse32_v_f32m1(pp + 4 * 10, vget_f32m1x4_f32m1(_r4567, 3), VL);
                vse32_v_f32m1(pp + 4 * 11, vget_f32m1x4_f32m1(_r89ab, 3), VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, VL), VL);
                vse32_v_f32m1(pp + 8, vle32_v_f32m1(p0 + 8, VL), VL);
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
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r4567, 0), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r4567, 1), VL);
                vse32_v_f32m1(pp + 4 * 4, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 5, vget_f32m1x4_f32m1(_r4567, 2), VL);
                vse32_v_f32m1(pp + 4 * 6, vget_f32m1x4_f32m1(_r0123, 3), VL);
                vse32_v_f32m1(pp + 4 * 7, vget_f32m1x4_f32m1(_r4567, 3), VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, VL), VL);
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
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r0123, 3), VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // float32x4x2_t _r01;
                // vget_f32m1x4_f32m1(_r01, 0) = vle32_v_f32m1(p0, VL);
                // vget_f32m1x4_f32m1(_r01, 1) = vle32_v_f32m1(p0 + 4, VL);
                // vst2q_f32(pp, _r01);
                store_float32_v2(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p0 + 4, VL), pp);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
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
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
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



int Gemm_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1, bottom_blob);
    std::vector<Mat> top_blobs(1, top_blob);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Gemm_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A0 = constantA ? A_data : bottom_blobs[0];
    const Mat& B0 = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

    size_t elemsize = A0.elemsize;

    Mat A;
    if (transA == 0)
    {
        A = A0;
    }
    else
    {
        // transpose A to row-major
        A.create((A0.dims == 3 ? A0.c : A0.h), A0.w, elemsize, opt.workspace_allocator);

        const int A0_hstep = A0.dims == 3 ? (int)A0.cstep : A0.w;

        for (int i = 0; i < A.h; i++)
        {
            float* ptr = A.row(i);
            for (int j = 0; j < A.w; j++)
            {
                ptr[j] = A0[j * A0_hstep + i];
            }
        }
    }

    Mat B;
    if (transB == 0)
    {
        // transpose B to col-major
        B.create((B0.dims == 3 ? B0.c : B0.h), B0.w, elemsize, opt.workspace_allocator);

        const int B0_hstep = B0.dims == 3 ? (int)B0.cstep : B0.w;

        for (int i = 0; i < B.h; i++)
        {
            float* ptr = B.row(i);
            for (int j = 0; j < B.w; j++)
            {
                ptr[j] = B0[j * B0_hstep + i];
            }
        }
    }
    else
    {
        B = B0;
    }

    const int M = A.dims == 3 ? A.c : A.h;
    const int K = A.w; // assert A.w == B.w
    const int N = B.dims == 3 ? B.c : B.h;

    const float* ptrC = 0;
    int broadcast_type_C = 0;
    if (constantC)
    {
        ptrC = C_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB)
        {
            ptrC = bottom_blobs.size() == 1 ? bottom_blobs[0] : 0;
        }
        else if (constantA)
        {
            ptrC = bottom_blobs.size() == 2 ? bottom_blobs[1] : 0;
        }
        else if (constantB)
        {
            ptrC = bottom_blobs.size() == 2 ? bottom_blobs[1] : 0;
        }
        else
        {
            ptrC = bottom_blobs.size() == 3 ? bottom_blobs[2] : 0;
        }

        if (ptrC)
        {
            const Mat& C = bottom_blobs[bottom_blobs.size() - 1];

            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N, elemsize, opt.blob_allocator);
        else
            top_blob.create(M, N, elemsize, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M, elemsize, opt.blob_allocator);
        else
            top_blob.create(N, M, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < M; i++)
    {
        const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

        const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
        const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

        const float* ptrA = (const float*)A + i * A_hstep;

        for (int j = 0; j < N; j++)
        {
            const float* ptrB = (const float*)B + j * B_hstep;

            float sum = 0.f;
            if (ptrC)
            {
                if (broadcast_type_C == 0)
                {
                    sum = ptrC[0];
                }
                if (broadcast_type_C == 1)
                {
                    sum = ptrC[i];
                }
                if (broadcast_type_C == 2)
                {
                    sum = ptrC[i];
                }
                if (broadcast_type_C == 3)
                {
                    sum = ptrC[i * N + j];
                }
                if (broadcast_type_C == 4)
                {
                    sum = ptrC[j];
                }

                sum *= beta;
            }

            for (int k = 0; k < K; k++)
            {
                sum += ptrA[k] * ptrB[k];
            }

            sum *= alpha;

            if (output_transpose)
            {
                top_blob[j * out_hstep + i] = sum;
            }
            else
            {
                top_blob[i * out_hstep + j] = sum;
            }
        }
    }

    return 0;
}

} // namespace ncnn
