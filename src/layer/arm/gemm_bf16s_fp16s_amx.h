// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_bf16_fp16_amx(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
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
                uint16x8_t _r0 = vcombine_u16(vld1_u16(p0), vld1_u16(p1));
                vst1q_u16(pp, _r0);
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
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
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
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
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
    }
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
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
}

static void transpose_pack_A_tile_bf16_fp16_amx(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);
                vst1q_u16(pp, _r04.val[0]);
                vst1q_u16(pp + 8, _r15.val[0]);
                vst1q_u16(pp + 16, _r26.val[0]);
                vst1q_u16(pp + 24, _r37.val[0]);
                vst1q_u16(pp + 32, _r04.val[1]);
                vst1q_u16(pp + 40, _r15.val[1]);
                vst1q_u16(pp + 48, _r26.val[1]);
                vst1q_u16(pp + 56, _r37.val[1]);
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
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                vst1q_u16(pp, _r0123.val[0]);
                vst1q_u16(pp + 8, _r0123.val[1]);
                vst1q_u16(pp + 16, _r0123.val[2]);
                vst1q_u16(pp + 24, _r0123.val[3]);
                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123;
                _r0123.val[0] = vld1q_u16(p0);
                _r0123.val[1] = vld1q_u16(p0 + 8);
                _r0123.val[2] = vld1q_u16(p0 + 16);
                _r0123.val[3] = vld1q_u16(p0 + 24);
                vst4q_u16(pp, _r0123);
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
                uint16x4x4_t _r0123 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0123.val[0], _r0123.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0123.val[2], _r0123.val[3]));
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
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x2_t _r01;
                _r01.val[0] = vld1q_u16(p0);
                _r01.val[1] = vld1q_u16(p0 + 8);
                vst2q_u16(pp, _r01);
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
                uint16x4x2_t _r01;
                _r01.val[0] = vld1_u16(p0);
                _r01.val[1] = vld1_u16(p0 + 4);
                vst2_u16(pp, _r01);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
#if __ARM_NEON
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vst1q_u16(pp, vld1q_u16(p0));
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
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
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

static void pack_B_tile_bf16_fp16_amx(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
    for (; jj + 31 < max_jj; jj += 32)
    // if (false)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 8;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 16) * B_hstep + k * 8;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 24) * B_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                vst1q_u16(pp + 8, vld1q_u16(p1));
                vst1q_u16(pp + 16, vld1q_u16(p2));
                vst1q_u16(pp + 24, vld1q_u16(p3));
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 4;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 12) * B_hstep + k * 4;
            const unsigned short* p4 = (const unsigned short*)B + (j + jj + 16) * B_hstep + k * 4;
            const unsigned short* p5 = (const unsigned short*)B + (j + jj + 20) * B_hstep + k * 4;
            const unsigned short* p6 = (const unsigned short*)B + (j + jj + 24) * B_hstep + k * 4;
            const unsigned short* p7 = (const unsigned short*)B + (j + jj + 28) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
                vst1_u16(pp + 4, vld1_u16(p1));
                vst1_u16(pp + 8, vld1_u16(p2));
                vst1_u16(pp + 12, vld1_u16(p3));
                vst1_u16(pp + 16, vld1_u16(p4));
                vst1_u16(pp + 20, vld1_u16(p5));
                vst1_u16(pp + 24, vld1_u16(p6));
                vst1_u16(pp + 28, vld1_u16(p7));
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
            const unsigned short* pc = (const unsigned short*)B + (j + jj + 12) * B_hstep + k;
            const unsigned short* pd = (const unsigned short*)B + (j + jj + 13) * B_hstep + k;
            const unsigned short* pe = (const unsigned short*)B + (j + jj + 14) * B_hstep + k;
            const unsigned short* pf = (const unsigned short*)B + (j + jj + 15) * B_hstep + k;
            const unsigned short* pg = (const unsigned short*)B + (j + jj + 16) * B_hstep + k;
            const unsigned short* ph = (const unsigned short*)B + (j + jj + 17) * B_hstep + k;
            const unsigned short* pi = (const unsigned short*)B + (j + jj + 18) * B_hstep + k;
            const unsigned short* pj = (const unsigned short*)B + (j + jj + 19) * B_hstep + k;
            const unsigned short* pk = (const unsigned short*)B + (j + jj + 20) * B_hstep + k;
            const unsigned short* pl = (const unsigned short*)B + (j + jj + 21) * B_hstep + k;
            const unsigned short* pm = (const unsigned short*)B + (j + jj + 22) * B_hstep + k;
            const unsigned short* pn = (const unsigned short*)B + (j + jj + 23) * B_hstep + k;
            const unsigned short* po = (const unsigned short*)B + (j + jj + 24) * B_hstep + k;
            const unsigned short* _pp = (const unsigned short*)B + (j + jj + 25) * B_hstep + k;
            const unsigned short* pq = (const unsigned short*)B + (j + jj + 26) * B_hstep + k;
            const unsigned short* pr = (const unsigned short*)B + (j + jj + 27) * B_hstep + k;
            const unsigned short* ps = (const unsigned short*)B + (j + jj + 28) * B_hstep + k;
            const unsigned short* pt = (const unsigned short*)B + (j + jj + 29) * B_hstep + k;
            const unsigned short* pu = (const unsigned short*)B + (j + jj + 30) * B_hstep + k;
            const unsigned short* pv = (const unsigned short*)B + (j + jj + 31) * B_hstep + k;

            int kk = 0;
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
                uint16x4_t _r8 = vld1_u16(p8);
                uint16x4_t _r9 = vld1_u16(p9);
                uint16x4_t _ra = vld1_u16(pa);
                uint16x4_t _rb = vld1_u16(pb);
                uint16x4_t _rc = vld1_u16(pc);
                uint16x4_t _rd = vld1_u16(pd);
                uint16x4_t _re = vld1_u16(pe);
                uint16x4_t _rf = vld1_u16(pf);
                uint16x4_t _rg = vld1_u16(pg);
                uint16x4_t _rh = vld1_u16(ph);
                uint16x4_t _ri = vld1_u16(pi);
                uint16x4_t _rj = vld1_u16(pj);
                uint16x4_t _rk = vld1_u16(pk);
                uint16x4_t _rl = vld1_u16(pl);
                uint16x4_t _rm = vld1_u16(pm);
                uint16x4_t _rn = vld1_u16(pn);
                uint16x4_t _ro = vld1_u16(po);
                uint16x4_t _rp = vld1_u16(_pp);
                uint16x4_t _rq = vld1_u16(pq);
                uint16x4_t _rr = vld1_u16(pr);
                uint16x4_t _rs = vld1_u16(ps);
                uint16x4_t _rt = vld1_u16(pt);
                uint16x4_t _ru = vld1_u16(pu);
                uint16x4_t _rv = vld1_u16(pv);

                transpose4x4_u16(_r0, _r1, _r2, _r3);
                transpose4x4_u16(_r4, _r5, _r6, _r7);
                transpose4x4_u16(_r8, _r9, _ra, _rb);
                transpose4x4_u16(_rc, _rd, _re, _rf);
                transpose4x4_u16(_rg, _rh, _ri, _rj);
                transpose4x4_u16(_rk, _rl, _rm, _rn);
                transpose4x4_u16(_ro, _rp, _rq, _rr);
                transpose4x4_u16(_rs, _rt, _ru, _rv);

                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r4);
                vst1_u16(pp + 4 * 2, _r8);
                vst1_u16(pp + 4 * 3, _rc);
                vst1_u16(pp + 4 * 4, _rg);
                vst1_u16(pp + 4 * 5, _rk);
                vst1_u16(pp + 4 * 6, _ro);
                vst1_u16(pp + 4 * 7, _rs);
                vst1_u16(pp + 4 * 8, _r1);
                vst1_u16(pp + 4 * 9, _r5);
                vst1_u16(pp + 4 * 10, _r9);
                vst1_u16(pp + 4 * 11, _rd);
                vst1_u16(pp + 4 * 12, _rh);
                vst1_u16(pp + 4 * 13, _rl);
                vst1_u16(pp + 4 * 14, _rp);
                vst1_u16(pp + 4 * 15, _rt);
                vst1_u16(pp + 4 * 16, _r2);
                vst1_u16(pp + 4 * 17, _r6);
                vst1_u16(pp + 4 * 18, _ra);
                vst1_u16(pp + 4 * 19, _re);
                vst1_u16(pp + 4 * 20, _ri);
                vst1_u16(pp + 4 * 21, _rm);
                vst1_u16(pp + 4 * 22, _rq);
                vst1_u16(pp + 4 * 23, _ru);
                vst1_u16(pp + 4 * 24, _r3);
                vst1_u16(pp + 4 * 25, _r7);
                vst1_u16(pp + 4 * 26, _rb);
                vst1_u16(pp + 4 * 27, _rf);
                vst1_u16(pp + 4 * 28, _rj);
                vst1_u16(pp + 4 * 29, _rn);
                vst1_u16(pp + 4 * 30, _rr);
                vst1_u16(pp + 4 * 31, _rv);
                pp += 128;
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
                pc += 4;
                pd += 4;
                pe += 4;
                pf += 4;
                pg += 4;
                ph += 4;
                pi += 4;
                pj += 4;
                pk += 4;
                pl += 4;
                pm += 4;
                pn += 4;
                po += 4;
                _pp += 4;
                pq += 4;
                pr += 4;
                ps += 4;
                pt += 4;
                pu += 4;
                pv += 4;
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
                pp[12] = pc[0];
                pp[13] = pd[0];
                pp[14] = pe[0];
                pp[15] = pf[0];
                pp[16] = pg[0];
                pp[17] = ph[0];
                pp[18] = pi[0];
                pp[19] = pj[0];
                pp[20] = pk[0];
                pp[21] = pl[0];
                pp[22] = pm[0];
                pp[23] = pn[0];
                pp[24] = po[0];
                pp[25] = _pp[0];
                pp[26] = pq[0];
                pp[27] = pr[0];
                pp[28] = ps[0];
                pp[29] = pt[0];
                pp[30] = pu[0];
                pp[31] = pv[0];
                pp += 32;
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
                pc++;
                pd++;
                pe++;
                pf++;
                pg++;
                ph++;
                pi++;
                pj++;
                pk++;
                pl++;
                pm++;
                pn++;
                po++;
                _pp++;
                pq++;
                pr++;
                ps++;
                pt++;
                pu++;
                pv++;
            }
        }
    }
#if __ARM_NEON
#if __aarch64__
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
                    vst1q_u16(pp, vld1q_u16(p0));
                    vst1_u16(pp + 8, vld1_u16(p1));
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1_u16(pp, vld1_u16(p0 + 4));
                    vst1q_u16(pp + 4, vld1q_u16(p1));
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
                vst1_u16(pp, vld1_u16(p0));
                vst1_u16(pp + 4, vld1_u16(p1));
                vst1_u16(pp + 8, vld1_u16(p2));
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
                uint16x4_t _r0 = vld1_u16(p0);
                uint16x4_t _r1 = vld1_u16(p1);
                uint16x4_t _r2 = vld1_u16(p2);
                uint16x4_t _r3 = vld1_u16(p3);
                uint16x4_t _r4 = vld1_u16(p4);
                uint16x4_t _r5 = vld1_u16(p5);
                uint16x4_t _r6 = vld1_u16(p6);
                uint16x4_t _r7 = vld1_u16(p7);
                uint16x4_t _r8 = vld1_u16(p8);
                uint16x4_t _r9 = vld1_u16(p9);
                uint16x4_t _ra = vld1_u16(pa);
                uint16x4_t _rb = vld1_u16(pb);

                transpose4x4_u16(_r0, _r1, _r2, _r3);
                transpose4x4_u16(_r4, _r5, _r6, _r7);
                transpose4x4_u16(_r8, _r9, _ra, _rb);

                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r4);
                vst1_u16(pp + 4 * 2, _r8);
                vst1_u16(pp + 4 * 3, _r1);
                vst1_u16(pp + 4 * 4, _r5);
                vst1_u16(pp + 4 * 5, _r9);
                vst1_u16(pp + 4 * 6, _r2);
                vst1_u16(pp + 4 * 7, _r6);
                vst1_u16(pp + 4 * 8, _ra);
                vst1_u16(pp + 4 * 9, _r3);
                vst1_u16(pp + 4 * 10, _r7);
                vst1_u16(pp + 4 * 11, _rb);
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
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1q_u16(pp, vld1q_u16(p0));
                    pp += 8;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1q_u16(pp, vcombine_u16(vld1_u16(p0 + 4), vld1_u16(p1)));
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
                uint16x8_t _r0 = vcombine_u16(vld1_u16(p0), vld1_u16(p1));
                vst1q_u16(pp, _r0);
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

                transpose4x4_u16(_r0, _r1, _r2, _r3);
                transpose4x4_u16(_r4, _r5, _r6, _r7);

                vst1_u16(pp, _r0);
                vst1_u16(pp + 4, _r4);
                vst1_u16(pp + 4 * 2, _r1);
                vst1_u16(pp + 4 * 3, _r5);
                vst1_u16(pp + 4 * 4, _r2);
                vst1_u16(pp + 4 * 5, _r6);
                vst1_u16(pp + 4 * 6, _r3);
                vst1_u16(pp + 4 * 7, _r7);
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
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1_u16(pp, vld1_u16(p0));
                    pp += 4;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vst1_u16(pp, vld1_u16(p0 + 4));
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
                vst1q_u16(pp, vld1q_u16(p0));
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                vst1_u16(pp, vld1_u16(p0));
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
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

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
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

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
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_bf16_fp16_amx(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
    // if (false)
    for (; jj + 31 < max_jj; jj += 32)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);

                uint16x8x4_t _r89ab = vld4q_u16(p0 + 64);
                uint16x8x4_t _rcdef = vld4q_u16(p0 + 96);

                uint16x8x4_t _rghij = vld4q_u16(p0 + 128);
                uint16x8x4_t _rklmn = vld4q_u16(p0 + 160);

                uint16x8x4_t _ropqr = vld4q_u16(p0 + 192);
                uint16x8x4_t _rstuv = vld4q_u16(p0 + 224);

                uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);

                uint16x8x2_t _r8c = vuzpq_u16(_r89ab.val[0], _rcdef.val[0]);
                uint16x8x2_t _r9d = vuzpq_u16(_r89ab.val[1], _rcdef.val[1]);
                uint16x8x2_t _rae = vuzpq_u16(_r89ab.val[2], _rcdef.val[2]);
                uint16x8x2_t _rbf = vuzpq_u16(_r89ab.val[3], _rcdef.val[3]);

                uint16x8x2_t _rgk = vuzpq_u16(_rghij.val[0], _rklmn.val[0]);
                uint16x8x2_t _rhl = vuzpq_u16(_rghij.val[1], _rklmn.val[1]);
                uint16x8x2_t _rim = vuzpq_u16(_rghij.val[2], _rklmn.val[2]);
                uint16x8x2_t _rjn = vuzpq_u16(_rghij.val[3], _rklmn.val[3]);

                uint16x8x2_t _ros = vuzpq_u16(_ropqr.val[0], _rstuv.val[0]);
                uint16x8x2_t _rpt = vuzpq_u16(_ropqr.val[1], _rstuv.val[1]);
                uint16x8x2_t _rqu = vuzpq_u16(_ropqr.val[2], _rstuv.val[2]);
                uint16x8x2_t _rrv = vuzpq_u16(_ropqr.val[3], _rstuv.val[3]);

                vst1q_u16(pp, _r04.val[0]);
                vst1q_u16(pp + 8, _r8c.val[0]);
                vst1q_u16(pp + 16, _rgk.val[0]);
                vst1q_u16(pp + 24, _ros.val[0]);
                vst1q_u16(pp + 32, _r15.val[0]);
                vst1q_u16(pp + 40, _r9d.val[0]);
                vst1q_u16(pp + 48, _rhl.val[0]);
                vst1q_u16(pp + 56, _rpt.val[0]);
                vst1q_u16(pp + 64, _r26.val[0]);
                vst1q_u16(pp + 72, _rae.val[0]);
                vst1q_u16(pp + 80, _rim.val[0]);
                vst1q_u16(pp + 88, _rqu.val[0]);
                vst1q_u16(pp + 96, _r37.val[0]);
                vst1q_u16(pp + 104, _rbf.val[0]);
                vst1q_u16(pp + 112, _rjn.val[0]);
                vst1q_u16(pp + 120, _rrv.val[0]);
                vst1q_u16(pp + 128, _r04.val[1]);
                vst1q_u16(pp + 136, _r8c.val[1]);
                vst1q_u16(pp + 144, _rgk.val[1]);
                vst1q_u16(pp + 152, _ros.val[1]);
                vst1q_u16(pp + 160, _r15.val[1]);
                vst1q_u16(pp + 168, _r9d.val[1]);
                vst1q_u16(pp + 176, _rhl.val[1]);
                vst1q_u16(pp + 184, _rpt.val[1]);
                vst1q_u16(pp + 192, _r26.val[1]);
                vst1q_u16(pp + 200, _rae.val[1]);
                vst1q_u16(pp + 208, _rim.val[1]);
                vst1q_u16(pp + 216, _rqu.val[1]);
                vst1q_u16(pp + 224, _r37.val[1]);
                vst1q_u16(pp + 232, _rbf.val[1]);
                vst1q_u16(pp + 240, _rjn.val[1]);
                vst1q_u16(pp + 248, _rrv.val[1]);

                pp += 256;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                uint16x8x4_t _r89ab = vld4q_u16(p0 + 64);
                uint16x8x4_t _rcdef = vld4q_u16(p0 + 96);

                vst1q_u16(pp, _r0123.val[0]);
                vst1q_u16(pp + 8, _r4567.val[0]);
                vst1q_u16(pp + 16, _r89ab.val[0]);
                vst1q_u16(pp + 24, _rcdef.val[0]);

                vst1q_u16(pp + 32, _r0123.val[1]);
                vst1q_u16(pp + 40, _r4567.val[1]);
                vst1q_u16(pp + 48, _r89ab.val[1]);
                vst1q_u16(pp + 56, _rcdef.val[1]);

                vst1q_u16(pp + 64, _r0123.val[2]);
                vst1q_u16(pp + 72, _r4567.val[2]);
                vst1q_u16(pp + 80, _r89ab.val[2]);
                vst1q_u16(pp + 88, _rcdef.val[2]);

                vst1q_u16(pp + 96, _r0123.val[3]);
                vst1q_u16(pp + 104, _r4567.val[3]);
                vst1q_u16(pp + 112, _r89ab.val[3]);
                vst1q_u16(pp + 120, _rcdef.val[3]);
                pp += 128;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_u16(pp, vld1q_u16(p0));
                vst1q_u16(pp + 8, vld1q_u16(p0 + 8));
                vst1q_u16(pp + 16, vld1q_u16(p0 + 16));
                vst1q_u16(pp + 24, vld1q_u16(p0 + 24));
                pp += 32;
                p0 += B_hstep;
            }
        }
    }
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                uint16x8x4_t _r89ab = vld4q_u16(p0 + 64);
                uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);
                uint16x4x2_t _r04_1 = vuzp_u16(vget_low_u16(_r89ab.val[0]), vget_high_u16(_r89ab.val[0]));
                uint16x4x2_t _r15_1 = vuzp_u16(vget_low_u16(_r89ab.val[1]), vget_high_u16(_r89ab.val[1]));
                uint16x4x2_t _r26_1 = vuzp_u16(vget_low_u16(_r89ab.val[2]), vget_high_u16(_r89ab.val[2]));
                uint16x4x2_t _r37_1 = vuzp_u16(vget_low_u16(_r89ab.val[3]), vget_high_u16(_r89ab.val[3]));
                vst1q_u16(pp, _r04.val[0]);
                vst1_u16(pp + 8, _r04_1.val[0]);
                vst1q_u16(pp + 12, _r15.val[0]);
                vst1_u16(pp + 20, _r15_1.val[0]);
                vst1q_u16(pp + 24, _r26.val[0]);
                vst1_u16(pp + 32, _r26_1.val[0]);
                vst1q_u16(pp + 36, _r37.val[0]);
                vst1_u16(pp + 44, _r37_1.val[0]);
                vst1q_u16(pp + 48, _r04.val[1]);
                vst1_u16(pp + 56, _r04_1.val[1]);
                vst1q_u16(pp + 60, _r15.val[1]);
                vst1_u16(pp + 68, _r15_1.val[1]);
                vst1q_u16(pp + 72, _r26.val[1]);
                vst1_u16(pp + 80, _r26_1.val[1]);
                vst1q_u16(pp + 84, _r37.val[1]);
                vst1_u16(pp + 92, _r37_1.val[1]);
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
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x4x4_t _r89ab = vld4_u16(p0 + 32);
                vst1q_u16(pp, _r0123.val[0]);
                vst1_u16(pp + 8, _r89ab.val[0]);
                vst1q_u16(pp + 12, _r0123.val[1]);
                vst1_u16(pp + 20, _r89ab.val[1]);
                vst1q_u16(pp + 24, _r0123.val[2]);
                vst1_u16(pp + 32, _r89ab.val[2]);
                vst1q_u16(pp + 36, _r0123.val[3]);
                vst1_u16(pp + 44, _r89ab.val[3]);
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
                vst1q_u16(pp, vld1q_u16(p0));
                vst1_u16(pp + 8, vld1_u16(p0 + 8));
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);
                vst1q_u16(pp, _r04.val[0]);
                vst1q_u16(pp + 8, _r15.val[0]);
                vst1q_u16(pp + 16, _r26.val[0]);
                vst1q_u16(pp + 24, _r37.val[0]);
                vst1q_u16(pp + 32, _r04.val[1]);
                vst1q_u16(pp + 40, _r15.val[1]);
                vst1q_u16(pp + 48, _r26.val[1]);
                vst1q_u16(pp + 56, _r37.val[1]);
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
                uint16x8x4_t _r0123 = vld4q_u16(p0);
                vst1q_u16(pp, _r0123.val[0]);
                vst1q_u16(pp + 8, _r0123.val[1]);
                vst1q_u16(pp + 16, _r0123.val[2]);
                vst1q_u16(pp + 24, _r0123.val[3]);
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
                vst1q_u16(pp, vld1q_u16(p0));
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
                uint16x8x4_t _r0123;
                _r0123.val[0] = vld1q_u16(p0);
                _r0123.val[1] = vld1q_u16(p0 + 8);
                _r0123.val[2] = vld1q_u16(p0 + 16);
                _r0123.val[3] = vld1q_u16(p0 + 24);
                vst4q_u16(pp, _r0123);
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
                uint16x4x4_t _r0123 = vld4_u16(p0);
                vst1q_u16(pp, vcombine_u16(_r0123.val[0], _r0123.val[1]));
                vst1q_u16(pp + 8, vcombine_u16(_r0123.val[2], _r0123.val[3]));
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
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __ARM_NEON
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8x2_t _r01;
                _r01.val[0] = vld1q_u16(p0);
                _r01.val[1] = vld1q_u16(p0 + 8);
                vst2q_u16(pp, _r01);
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
                uint16x4x2_t _r01;
                _r01.val[0] = vld1_u16(p0);
                _r01.val[1] = vld1_u16(p0 + 4);
                vst2_u16(pp, _r01);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
#if __ARM_NEON
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vst1q_u16(pp, vld1q_u16(p0));
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
                vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
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

static void transpose_unpack_output_tile_bf16_fp16_amx(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const unsigned short* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                uint16x8_t _r0 = vld1q_u16(pp);
                uint16x8_t _r1 = vld1q_u16(pp + 8);
                uint16x8_t _r2 = vld1q_u16(pp + 8 * 2);
                uint16x8_t _r3 = vld1q_u16(pp + 8 * 3);
                transpose8x4_u16(_r0, _r1, _r2, _r3);
                vst1_u16(p0 + 4, vget_low_u16(_r0));
                vst1_u16(p0 + 8 + 4, vget_high_u16(_r0));
                vst1_u16(p0 + 8 * 2 + 4, vget_low_u16(_r1));
                vst1_u16(p0 + 8 * 3 + 4, vget_high_u16(_r1));
                vst1_u16(p0 + 8 * 4 + 4, vget_low_u16(_r2));
                vst1_u16(p0 + 8 * 5 + 4, vget_high_u16(_r2));
                vst1_u16(p0 + 8 * 6 + 4, vget_low_u16(_r3));
                vst1_u16(p0 + 8 * 7 + 4, vget_high_u16(_r3));
                pp += 32;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                uint16x8_t _r0 = vld1q_u16(pp);
                uint16x8_t _r1 = vld1q_u16(pp + 8);
                uint16x8_t _r2 = vld1q_u16(pp + 8 * 2);
                uint16x8_t _r3 = vld1q_u16(pp + 8 * 3);
                uint16x8_t _r4 = vld1q_u16(pp + 8 * 4);
                uint16x8_t _r5 = vld1q_u16(pp + 8 * 5);
                uint16x8_t _r6 = vld1q_u16(pp + 8 * 6);
                uint16x8_t _r7 = vld1q_u16(pp + 8 * 7);
                transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                vst1q_u16(p0, _r0);
                vst1q_u16(p0 + 8, _r1);
                vst1q_u16(p0 + 8 * 2, _r2);
                vst1q_u16(p0 + 8 * 3, _r3);
                vst1q_u16(p0 + 8 * 4, _r4);
                vst1q_u16(p0 + 8 * 5, _r5);
                vst1q_u16(p0 + 8 * 6, _r6);
                vst1q_u16(p0 + 8 * 7, _r7);
                pp += 64;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                uint16x8_t _r0 = vld1q_u16(pp);
                uint16x8_t _r1 = vld1q_u16(pp + 8);
                uint16x8_t _r2 = vld1q_u16(pp + 8 * 2);
                uint16x8_t _r3 = vld1q_u16(pp + 8 * 3);
                transpose8x4_u16(_r0, _r1, _r2, _r3);
                vst1_u16(p0, vget_low_u16(_r0));
                vst1_u16(p0 + 8, vget_high_u16(_r0));
                vst1_u16(p0 + 8 * 2, vget_low_u16(_r1));
                vst1_u16(p0 + 8 * 3, vget_high_u16(_r1));
                vst1_u16(p0 + 8 * 4, vget_low_u16(_r2));
                vst1_u16(p0 + 8 * 5, vget_high_u16(_r2));
                vst1_u16(p0 + 8 * 6, vget_low_u16(_r3));
                vst1_u16(p0 + 8 * 7, vget_high_u16(_r3));
                pp += 32;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x8x4_t _r0123;
                _r0123.val[0] = vld1q_u16(pp);
                _r0123.val[1] = vld1q_u16(pp + 8);
                _r0123.val[2] = vld1q_u16(pp + 8 * 2);
                _r0123.val[3] = vld1q_u16(pp + 8 * 3);
                vst4q_u16(p0, _r0123);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                uint16x8_t _r0 = vld1q_u16(pp);
                vst1q_u16(p0, _r0);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __aarch64__
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                uint16x4x4_t _r0123 = vld4_u16(pp);
                vst1_u16(p0 + 4, _r0123.val[0]);
                vst1_u16(p0 + 8 + 4, _r0123.val[1]);
                vst1_u16(p0 + 16 + 4, _r0123.val[2]);
                vst1_u16(p0 + 24 + 4, _r0123.val[3]);
                pp += 16;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                uint16x8x4_t _r0123 = vld4q_u16(pp);
                vst1q_u16(p0, _r0123.val[0]);
                vst1q_u16(p0 + 8, _r0123.val[1]);
                vst1q_u16(p0 + 16, _r0123.val[2]);
                vst1q_u16(p0 + 24, _r0123.val[3]);
                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                uint16x4x4_t _r0123 = vld4_u16(pp);
                vst1_u16(p0, _r0123.val[0]);
                vst1_u16(p0 + 8, _r0123.val[1]);
                vst1_u16(p0 + 16, _r0123.val[2]);
                vst1_u16(p0 + 24, _r0123.val[3]);
                pp += 16;
                p0 += out_hstep * 8;
            }
        }
#endif // __aarch64__
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x4x4_t _r0123;
                _r0123.val[0] = vld1_u16(pp);
                _r0123.val[1] = vld1_u16(pp + 4);
                _r0123.val[2] = vld1_u16(pp + 8);
                _r0123.val[3] = vld1_u16(pp + 12);
                vst4_u16(p0, _r0123);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                uint16x4_t _r0 = vld1_u16(pp);
                vst1_u16(p0, _r0);
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
#if __aarch64__
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
#endif // __aarch64__
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
#endif // __ARM_NEON
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
#if __ARM_NEON
#if __aarch64__
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                uint16x4_t _r0 = vld1_u16(pp);
                vst1_u16(p0 + 4, _r0);
                pp += 4;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                uint16x8_t _r0 = vld1q_u16(pp);
                vst1q_u16(p0, _r0);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                uint16x4_t _r0 = vld1_u16(pp);
                vst1_u16(p0, _r0);
                pp += 4;
                p0 += out_hstep * 8;
            }
        }
#endif // __aarch64__
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x4_t _r0 = vld1_u16(pp);
                vst1_u16(p0, _r0);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
