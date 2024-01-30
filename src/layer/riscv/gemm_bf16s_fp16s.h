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

static void pack_A_tile_bf16_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    int vl;
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1q_u16(pp, vle16_v_u16m1(p0), vl);
                pp += 8;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vl = 4;
                vuint16m1_t _r00 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r01 = vle16_v_u16m1(p1, vl);
                vsseg2e16_v_u16m1(pp, _r00, _r01, vl);
                // vuint16m1_t _r0 = vcombine_u16(vld1_u16(p0), vld1_u16(p1));
                // vst1q_u16(pp, _r0);
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

            int n = max_kk;
            while (n > 0)
            {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p4, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p5, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p6, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p7, vl);

                vsseg8e16_v_u16m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 8 * vl;
                p0 += vl;
                p1 += vl;
                p2 += vl;
                p3 += vl;
                p4 += vl;
                p5 += vl;
                p6 += vl;
                p7 += vl;
                n -= vl;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
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

            int n = max_kk;
            while (n > 0)
            {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
                vsseg4e16_v_u16m1(pp, _r0, _r1, _r2, _r3, vl);
                pp += 4 * vl;
                p0 += vl;
                p1 += vl;
                p2 += vl;
                p3 += vl;
                n -= vl;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            int n = max_kk;
            while (n > 0)
            {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vsseg2e16_v_u16m1(pp, _r0, _r1, vl);
                pp += 2 * vl;
                p0 += vl;
                p1 += vl;
                n -= vl;
            }
#else
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0 += 1;
                p1 += 1;
            }
#endif // __riscv_vector
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            int n = max_kk;
            while (n > 0)
            {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vse16_v_u16m1(pp, _r0, vl);
                pp += 1 * vl;
                p0 += vl;
                n -= vl;
            }
#else
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += 1;
            }
#endif
        }
    }
}

static void transpose_pack_A_tile_bf16_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    int vl;
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;
                vuint16m1_t _r4;
                vuint16m1_t _r5;
                vuint16m1_t _r6;
                vuint16m1_t _r7;
                vlseg8e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, &_r4, &_r5, &_r6, &_r7, p0, vl);
                vse16_v_u16m1(pp, _r0, vl);
                vse16_v_u16m1(pp + 8, _r1, vl);
                vse16_v_u16m1(pp + 16, _r2, vl);
                vse16_v_u16m1(pp + 24, _r3, vl);
                vse16_v_u16m1(pp + 32, _r4, vl);
                vse16_v_u16m1(pp + 40, _r5, vl);
                vse16_v_u16m1(pp + 48, _r6, vl);
                vse16_v_u16m1(pp + 56, _r7, vl);
                pp += 64;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 8;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;
                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, p0, vl);
                vse16_v_u16m1(pp, _r0, vl);
                vse16_v_u16m1(pp + 4, _r1, vl);
                vse16_v_u16m1(pp + 8, _r2, vl);
                vse16_v_u16m1(pp + 12, _r3, vl);
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);
                vsseg4e16_v_u16m1(pp, _r0, _r1, _r2, _r3, vl);
                pp += 32;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;
                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, p0, vl);
                vse16_v_u16m1(pp, _r0, vl);
                vse16_v_u16m1(pp + 4, _r1, vl);
                vse16_v_u16m1(pp + 8, _r2, vl);
                vse16_v_u16m1(pp + 12, _r3, vl);
                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vsseg2e16_v_u16m1(pp, _r0, _r1, vl);
                pp += 16;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r010 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r011 = vle16_v_u16m1(p0 + 4, vl);
                vsseg2e16_v_u16m1(pp, _r010, _r011, vl);

                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

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
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
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
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __riscv_vector
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

static void pack_B_tile_bf16_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    int vl;
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
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
                    vl = 8;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                    vl = 4;
                    vse16_v_u16m1(pp + 8, vle16_v_u16m1(p1, vl), vl);
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 4;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                    vl = 8;
                    vse16_v_u16m1(pp + 4, vle16_v_u16m1(p1, vl), vl);
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

            for (int kk = 0; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                vse16_v_u16m1(pp + 4, vle16_v_u16m1(p1, vl), vl);
                vse16_v_u16m1(pp + 8, vle16_v_u16m1(p2, vl), vl);
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
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p4, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p5, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p6, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p7, vl);
                vuint16m1_t _r8 = vle16_v_u16m1(p8, vl);
                vuint16m1_t _r9 = vle16_v_u16m1(p9, vl);
                vuint16m1_t _ra = vle16_v_u16m1(pa, vl);
                vuint16m1_t _rb = vle16_v_u16m1(pb, vl);

                vsse16_v_u16m1(pp, 12 * sizeof(unsigned short), _r0, vl);
                vsse16_v_u16m1(pp + 1, 12 * sizeof(unsigned short), _r1, vl);
                vsse16_v_u16m1(pp + 2, 12 * sizeof(unsigned short), _r2, vl);
                vsse16_v_u16m1(pp + 3, 12 * sizeof(unsigned short), _r3, vl);
                vsse16_v_u16m1(pp + 4, 12 * sizeof(unsigned short), _r4, vl);
                vsse16_v_u16m1(pp + 5, 12 * sizeof(unsigned short), _r5, vl);
                vsse16_v_u16m1(pp + 6, 12 * sizeof(unsigned short), _r6, vl);
                vsse16_v_u16m1(pp + 7, 12 * sizeof(unsigned short), _r7, vl);
                vsse16_v_u16m1(pp + 8, 12 * sizeof(unsigned short), _r8, vl);
                vsse16_v_u16m1(pp + 9, 12 * sizeof(unsigned short), _r9, vl);
                vsse16_v_u16m1(pp + 10, 12 * sizeof(unsigned short), _ra, vl);
                vsse16_v_u16m1(pp + 11, 12 * sizeof(unsigned short), _rb, vl);

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
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 8;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                    pp += 8;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 4;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0 + 4, vl), vl);
                    vse16_v_u16m1(pp + 4, vle16_v_u16m1(p1, vl), vl);
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

            for (int kk = 0; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                vse16_v_u16m1(pp + 4, vle16_v_u16m1(p1, vl), vl);
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

            int n = max_kk;
            while (n > 0)
            {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p4, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p5, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p6, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p7, vl);
                vsseg8e16_v_u16m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 8 * vl;
                p0 += vl;
                p1 += vl;
                p2 += vl;
                p3 += vl;
                p4 += vl;
                p5 += vl;
                p6 += vl;
                p7 += vl;

                n -= vl;
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
                    vl = 8;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                    pp += 4;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 4;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0 + 4, vl), vl);
                    pp += 4;
                    p0 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
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

            int n = max_kk;
            while (n > 0)
            {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
                vsseg4e16_v_u16m1(pp, _r0, _r1, _r2, _r3, vl);
                pp += 4 * vl;
                p0 += vl;
                p1 += vl;
                p2 += vl;
                p3 += vl;
                n -= vl;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __riscv_vector

            int n = max_kk;
            while (n > 0)
            {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vsseg2e16_v_u16m1(pp, _r0, _r1, vl);
                pp += 2 * vl;
                p0 += vl;
                p1 += vl;
                n -= vl;
            }
#else
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
#endif // __riscv_vector
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            int kk = 0;
#if __riscv_vector

            int n = max_kk;
            while (n > 0)
            {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vse16_v_u16m1(pp, _r0, vl);
                pp += vl;
                p0 += vl;
                n -= vl;
            }
#else
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
#endif // __riscv_vector
        }
    }
}

static void transpose_pack_B_tile_bf16_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    int vl;
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;

            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p0 + 32, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p0 + 40, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p0 + 48, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p0 + 56, vl);
                vuint16m1_t _r8 = vle16_v_u16m1(p0 + 64, vl);
                vuint16m1_t _r9 = vle16_v_u16m1(p0 + 72, vl);
                vuint16m1_t _ra = vle16_v_u16m1(p0 + 80, vl);
                vuint16m1_t _rb = vle16_v_u16m1(p0 + 88, vl);

                vsse16_v_u16m1(pp, 12 * sizeof(unsigned short), _r0, vl);
                vsse16_v_u16m1(pp + 1, 12 * sizeof(unsigned short), _r1, vl);
                vsse16_v_u16m1(pp + 2, 12 * sizeof(unsigned short), _r2, vl);
                vsse16_v_u16m1(pp + 3, 12 * sizeof(unsigned short), _r3, vl);
                vsse16_v_u16m1(pp + 4, 12 * sizeof(unsigned short), _r4, vl);
                vsse16_v_u16m1(pp + 5, 12 * sizeof(unsigned short), _r5, vl);
                vsse16_v_u16m1(pp + 6, 12 * sizeof(unsigned short), _r6, vl);
                vsse16_v_u16m1(pp + 7, 12 * sizeof(unsigned short), _r7, vl);
                vsse16_v_u16m1(pp + 8, 12 * sizeof(unsigned short), _r8, vl);
                vsse16_v_u16m1(pp + 9, 12 * sizeof(unsigned short), _r9, vl);
                vsse16_v_u16m1(pp + 10, 12 * sizeof(unsigned short), _ra, vl);
                vsse16_v_u16m1(pp + 11, 12 * sizeof(unsigned short), _rb, vl);

                pp += 96;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p0 + 32, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p0 + 40, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p0 + 48, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p0 + 56, vl);
                vuint16m1_t _r8 = vle16_v_u16m1(p0 + 64, vl);
                vuint16m1_t _r9 = vle16_v_u16m1(p0 + 72, vl);
                vuint16m1_t _ra = vle16_v_u16m1(p0 + 80, vl);
                vuint16m1_t _rb = vle16_v_u16m1(p0 + 88, vl);

                vsse16_v_u16m1(pp, 12 * sizeof(unsigned short), _r0, vl);
                vsse16_v_u16m1(pp + 1, 12 * sizeof(unsigned short), _r1, vl);
                vsse16_v_u16m1(pp + 2, 12 * sizeof(unsigned short), _r2, vl);
                vsse16_v_u16m1(pp + 3, 12 * sizeof(unsigned short), _r3, vl);
                vsse16_v_u16m1(pp + 4, 12 * sizeof(unsigned short), _r4, vl);
                vsse16_v_u16m1(pp + 5, 12 * sizeof(unsigned short), _r5, vl);
                vsse16_v_u16m1(pp + 6, 12 * sizeof(unsigned short), _r6, vl);
                vsse16_v_u16m1(pp + 7, 12 * sizeof(unsigned short), _r7, vl);
                vsse16_v_u16m1(pp + 8, 12 * sizeof(unsigned short), _r8, vl);
                vsse16_v_u16m1(pp + 9, 12 * sizeof(unsigned short), _r9, vl);
                vsse16_v_u16m1(pp + 10, 12 * sizeof(unsigned short), _ra, vl);
                vsse16_v_u16m1(pp + 11, 12 * sizeof(unsigned short), _rb, vl);

                pp += 48;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vl = 12;
                vse16_v_u16m2(pp, vle16_v_u16m2(p0, vl), vl);
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p0 + 32, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p0 + 40, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p0 + 48, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p0 + 56, vl);
                vsseg8e16_v_u16m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);
                pp += 64;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 8;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, p0, vl);

                vse16_v_u16m1(pp, _r0, vl);
                vse16_v_u16m1(pp + 8, _r1, vl);
                vse16_v_u16m1(pp + 16, _r2, vl);
                vse16_v_u16m1(pp + 24, _r3, vl);

                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
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
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);

                vsseg4e16_v_u16m1(pp, _r0, _r1, _r2, _r3, vl);
                pp += 32;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, p0, vl);

                vse16_v_u16m1(pp, _r0, vl);
                vse16_v_u16m1(pp + 4, _r1, vl);
                vse16_v_u16m1(pp + 8, _r2, vl);
                vse16_v_u16m1(pp + 12, _r3, vl);
                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __riscv_vector
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vsseg2e16_v_u16m1(pp, _r0, _r1, vl);
                pp += 16;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vsseg2e16_v_u16m1(pp, _r0, _r1, vl);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

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
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
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
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __riscv_vector
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

static void transpose_unpack_output_tile_bf16_fp16(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    int vl;
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const unsigned short* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 4, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 12, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(pp + 16, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(pp + 20, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(pp + 24, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(pp + 28, vl);

                vsseg8e16_v_u16m1(p0 + 4, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 32;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8 * 2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 8 * 3, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(pp + 8 * 4, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(pp + 8 * 5, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(pp + 8 * 6, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(pp + 8 * 7, vl);

                vsseg8e16_v_u16m1(p0, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 64;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 4, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 12, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(pp + 16, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(pp + 20, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(pp + 24, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(pp + 28, vl);

                vsseg8e16_v_u16m1(p0, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 32;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8 * 2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 8 * 3, vl);

                vsseg4e16_v_u16m1(p0, _r0, _r1, _r2, _r3, vl);

                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                vl = 4;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, pp, vl);
                vse16_v_u16m1(p0 + 4, _r0, vl);
                vse16_v_u16m1(p0 + 8 + 4, _r1, vl);
                vse16_v_u16m1(p0 + 16 + 4, _r2, vl);
                vse16_v_u16m1(p0 + 24 + 4, _r3, vl);

                pp += 16;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                vl = 8;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, pp, vl);
                vse16_v_u16m1(p0, _r0, vl);
                vse16_v_u16m1(p0 + 8, _r1, vl);
                vse16_v_u16m1(p0 + 8 * 2, _r2, vl);
                vse16_v_u16m1(p0 + 8 * 3, _r3, vl);

                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, pp, vl);
                vse16_v_u16m1(p0, _r0, vl);
                vse16_v_u16m1(p0 + 8, _r1, vl);
                vse16_v_u16m1(p0 + 16, _r2, vl);
                vse16_v_u16m1(p0 + 24, _r3, vl);

                pp += 16;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 4, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 12, vl);

                vsseg4e16_v_u16m1(p0, _r0, _r1, _r2, _r3, vl);

                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);

                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                p0[0 + 4] = pp[0];
                p0[1 + 4] = pp[2];
                p0[2 + 4] = pp[4];
                p0[3 + 4] = pp[6];
                p0[8 + 4] = pp[1];
                p0[9 + 4] = pp[3];
                p0[10 + 4] = pp[5];
                p0[11 + 4] = pp[7];
                pp += 8;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[8];
                p0[5] = pp[10];
                p0[6] = pp[12];
                p0[7] = pp[14];
                p0[8] = pp[1];
                p0[9] = pp[3];
                p0[10] = pp[5];
                p0[11] = pp[7];
                p0[12] = pp[9];
                p0[13] = pp[11];
                p0[14] = pp[13];
                p0[15] = pp[15];
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[8] = pp[1];
                p0[9] = pp[3];
                p0[10] = pp[5];
                p0[11] = pp[7];
                pp += 8;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

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
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

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
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0 + 4, _r0, vl);

                pp += 4;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);

                pp += 4;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);

                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void get_optimal_tile_mnk_bf16s_fp16s(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
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
