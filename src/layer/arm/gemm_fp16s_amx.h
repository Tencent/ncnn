// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_fp32_to_fp16_amx(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
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
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            uint16x8_t _r1 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            uint16x8_t _r2 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            uint16x8_t _r3 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
            uint16x8_t _r4 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p4)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4 + 4)));
            uint16x8_t _r5 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p5)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5 + 4)));
            uint16x8_t _r6 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p6)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6 + 4)));
            uint16x8_t _r7 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p7)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7 + 4)));
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
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp[4] = float32_to_float16(p4[0]);
            pp[5] = float32_to_float16(p5[0]);
            pp[6] = float32_to_float16(p6[0]);
            pp[7] = float32_to_float16(p7[0]);
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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x4_t _r0123;
            _r0123.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            _r0123.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            _r0123.val[2] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            _r0123.val[3] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
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
            _r0123.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            _r0123.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            _r0123.val[2] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            _r0123.val[3] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            vst4_u16(pp, _r0123);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x2_t _r01;
            _r01.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            _r01.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            vst2q_u16(pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x2_t _r01;
            _r01.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            _r01.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            vst2_u16(pp, _r01);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_fp16_amx(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += A_hstep;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += A_hstep;
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p0[1]);
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_fp32_to_fp16_amx(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;
    // NCNN_LOGE("pack_B_tile_fp32_to_fp16_amx");

    unsigned short* pp = BT;

    int jj = 0;
    for (; jj + 31 < max_jj; jj += 32)
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
        const float* pc = (const float*)B + (j + jj + 12) * B_hstep + k;
        const float* pd = (const float*)B + (j + jj + 13) * B_hstep + k;
        const float* pe = (const float*)B + (j + jj + 14) * B_hstep + k;
        const float* pf = (const float*)B + (j + jj + 15) * B_hstep + k;
        const float* pg = (const float*)B + (j + jj + 16) * B_hstep + k;
        const float* ph = (const float*)B + (j + jj + 17) * B_hstep + k;
        const float* pi = (const float*)B + (j + jj + 18) * B_hstep + k;
        const float* pj = (const float*)B + (j + jj + 19) * B_hstep + k;
        const float* pk = (const float*)B + (j + jj + 20) * B_hstep + k;
        const float* pl = (const float*)B + (j + jj + 21) * B_hstep + k;
        const float* pm = (const float*)B + (j + jj + 22) * B_hstep + k;
        const float* pn = (const float*)B + (j + jj + 23) * B_hstep + k;
        const float* po = (const float*)B + (j + jj + 24) * B_hstep + k;
        const float* _pp = (const float*)B + (j + jj + 25) * B_hstep + k;
        const float* pq = (const float*)B + (j + jj + 26) * B_hstep + k;
        const float* pr = (const float*)B + (j + jj + 27) * B_hstep + k;
        const float* ps = (const float*)B + (j + jj + 28) * B_hstep + k;
        const float* pt = (const float*)B + (j + jj + 29) * B_hstep + k;
        const float* pu = (const float*)B + (j + jj + 30) * B_hstep + k;
        const float* pv = (const float*)B + (j + jj + 31) * B_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            uint16x8_t _r1 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            uint16x8_t _r2 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            uint16x8_t _r3 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
            uint16x8_t _r4 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p4)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4 + 4)));
            uint16x8_t _r5 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p5)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5 + 4)));
            uint16x8_t _r6 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p6)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6 + 4)));
            uint16x8_t _r7 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p7)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7 + 4)));

            uint16x8_t _r8 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p8)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p8 + 4)));
            uint16x8_t _r9 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p9)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p9 + 4)));
            uint16x8_t _ra = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pa)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pa + 4)));
            uint16x8_t _rb = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pb)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pb + 4)));
            uint16x8_t _rc = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pc)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pc + 4)));
            uint16x8_t _rd = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pd)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pd + 4)));
            uint16x8_t _re = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pe)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pe + 4)));
            uint16x8_t _rf = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pf)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pf + 4)));

            uint16x8_t _rg = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pg)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pg + 4)));
            uint16x8_t _rh = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(ph)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(ph + 4)));
            uint16x8_t _ri = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pi)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pi + 4)));
            uint16x8_t _rj = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pj)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pj + 4)));
            uint16x8_t _rk = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pk)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pk + 4)));
            uint16x8_t _rl = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pl)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pl + 4)));
            uint16x8_t _rm = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pm)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pm + 4)));
            uint16x8_t _rn = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pn)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pn + 4)));

            uint16x8_t _ro = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(po)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(po + 4)));
            uint16x8_t _rp = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(_pp)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(_pp + 4)));
            uint16x8_t _rq = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pq)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pq + 4)));
            uint16x8_t _rr = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pr)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pr + 4)));
            uint16x8_t _rs = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(ps)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(ps + 4)));
            uint16x8_t _rt = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pt)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pt + 4)));
            uint16x8_t _ru = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pu)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pu + 4)));
            uint16x8_t _rv = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pv)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pv + 4)));

            transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            transpose8x8_u16(_r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
            transpose8x8_u16(_rg, _rh, _ri, _rj, _rk, _rl, _rm, _rn);
            transpose8x8_u16(_ro, _rp, _rq, _rr, _rs, _rt, _ru, _rv);
            vst1q_u16(pp, _r0);
            vst1q_u16(pp + 8, _r8);
            vst1q_u16(pp + 8 * 2, _rg);
            vst1q_u16(pp + 8 * 3, _ro);
            vst1q_u16(pp + 8 * 4, _r1);
            vst1q_u16(pp + 8 * 5, _r9);
            vst1q_u16(pp + 8 * 6, _rh);
            vst1q_u16(pp + 8 * 7, _rp);
            vst1q_u16(pp + 8 * 8, _r2);
            vst1q_u16(pp + 8 * 9, _ra);
            vst1q_u16(pp + 8 * 10, _ri);
            vst1q_u16(pp + 8 * 11, _rq);
            vst1q_u16(pp + 8 * 12, _r3);
            vst1q_u16(pp + 8 * 13, _rb);
            vst1q_u16(pp + 8 * 14, _rj);
            vst1q_u16(pp + 8 * 15, _rr);
            vst1q_u16(pp + 8 * 16, _r4);
            vst1q_u16(pp + 8 * 17, _rc);
            vst1q_u16(pp + 8 * 18, _rk);
            vst1q_u16(pp + 8 * 19, _rs);
            vst1q_u16(pp + 8 * 20, _r5);
            vst1q_u16(pp + 8 * 21, _rd);
            vst1q_u16(pp + 8 * 22, _rl);
            vst1q_u16(pp + 8 * 23, _rt);
            vst1q_u16(pp + 8 * 24, _r6);
            vst1q_u16(pp + 8 * 25, _re);
            vst1q_u16(pp + 8 * 26, _rm);
            vst1q_u16(pp + 8 * 27, _ru);
            vst1q_u16(pp + 8 * 28, _r7);
            vst1q_u16(pp + 8 * 29, _rf);
            vst1q_u16(pp + 8 * 30, _rn);
            vst1q_u16(pp + 8 * 31, _rv);
            pp += 256;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
            p8 += 8;
            p9 += 8;
            pa += 8;
            pb += 8;
            pc += 8;
            pd += 8;
            pe += 8;
            pf += 8;
            pg += 8;
            ph += 8;
            pi += 8;
            pj += 8;
            pk += 8;
            pl += 8;
            pm += 8;
            pn += 8;
            po += 8;
            _pp += 8;
            pq += 8;
            pr += 8;
            ps += 8;
            pt += 8;
            pu += 8;
            pv += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            uint16x4_t _r1 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            uint16x4_t _r2 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            uint16x4_t _r3 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            uint16x4_t _r4 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4));
            uint16x4_t _r5 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5));
            uint16x4_t _r6 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6));
            uint16x4_t _r7 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7));

            uint16x4_t _r8 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p8));
            uint16x4_t _r9 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p9));
            uint16x4_t _ra = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pa));
            uint16x4_t _rb = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pb));
            uint16x4_t _rc = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pc));
            uint16x4_t _rd = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pd));
            uint16x4_t _re = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pe));
            uint16x4_t _rf = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pf));

            uint16x4_t _rg = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pg));
            uint16x4_t _rh = (uint16x4_t)vcvt_f16_f32(vld1q_f32(ph));
            uint16x4_t _ri = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pi));
            uint16x4_t _rj = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pj));
            uint16x4_t _rk = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pk));
            uint16x4_t _rl = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pl));
            uint16x4_t _rm = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pm));
            uint16x4_t _rn = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pn));

            uint16x4_t _ro = (uint16x4_t)vcvt_f16_f32(vld1q_f32(po));
            uint16x4_t _rp = (uint16x4_t)vcvt_f16_f32(vld1q_f32(_pp));
            uint16x4_t _rq = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pq));
            uint16x4_t _rr = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pr));
            uint16x4_t _rs = (uint16x4_t)vcvt_f16_f32(vld1q_f32(ps));
            uint16x4_t _rt = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pt));
            uint16x4_t _ru = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pu));
            uint16x4_t _rv = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pv));

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
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp[4] = float32_to_float16(p4[0]);
            pp[5] = float32_to_float16(p5[0]);
            pp[6] = float32_to_float16(p6[0]);
            pp[7] = float32_to_float16(p7[0]);
            pp[8] = float32_to_float16(p8[0]);
            pp[9] = float32_to_float16(p9[0]);
            pp[10] = float32_to_float16(pa[0]);
            pp[11] = float32_to_float16(pb[0]);
            pp[12] = float32_to_float16(pc[0]);
            pp[13] = float32_to_float16(pd[0]);
            pp[14] = float32_to_float16(pe[0]);
            pp[15] = float32_to_float16(pf[0]);
            pp[16] = float32_to_float16(pg[0]);
            pp[17] = float32_to_float16(ph[0]);
            pp[18] = float32_to_float16(pi[0]);
            pp[19] = float32_to_float16(pj[0]);
            pp[20] = float32_to_float16(pk[0]);
            pp[21] = float32_to_float16(pl[0]);
            pp[22] = float32_to_float16(pm[0]);
            pp[23] = float32_to_float16(pn[0]);
            pp[24] = float32_to_float16(po[0]);
            pp[25] = float32_to_float16(_pp[0]);
            pp[26] = float32_to_float16(pq[0]);
            pp[27] = float32_to_float16(pr[0]);
            pp[28] = float32_to_float16(ps[0]);
            pp[29] = float32_to_float16(pt[0]);
            pp[30] = float32_to_float16(pu[0]);
            pp[31] = float32_to_float16(pv[0]);
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
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
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
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            uint16x4_t _r1 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            uint16x4_t _r2 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            uint16x4_t _r3 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            uint16x4_t _r4 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4));
            uint16x4_t _r5 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5));
            uint16x4_t _r6 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6));
            uint16x4_t _r7 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7));
            uint16x4_t _r8 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p8));
            uint16x4_t _r9 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p9));
            uint16x4_t _ra = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pa));
            uint16x4_t _rb = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pb));

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
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp[4] = float32_to_float16(p4[0]);
            pp[5] = float32_to_float16(p5[0]);
            pp[6] = float32_to_float16(p6[0]);
            pp[7] = float32_to_float16(p7[0]);
            pp[8] = float32_to_float16(p8[0]);
            pp[9] = float32_to_float16(p9[0]);
            pp[10] = float32_to_float16(pa[0]);
            pp[11] = float32_to_float16(pb[0]);
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
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
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
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            uint16x8_t _r1 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            uint16x8_t _r2 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            uint16x8_t _r3 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
            uint16x8_t _r4 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p4)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4 + 4)));
            uint16x8_t _r5 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p5)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5 + 4)));
            uint16x8_t _r6 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p6)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6 + 4)));
            uint16x8_t _r7 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p7)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7 + 4)));
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
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            uint16x4_t _r1 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            uint16x4_t _r2 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            uint16x4_t _r3 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            uint16x4_t _r4 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4));
            uint16x4_t _r5 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5));
            uint16x4_t _r6 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6));
            uint16x4_t _r7 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7));

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
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp[4] = float32_to_float16(p4[0]);
            pp[5] = float32_to_float16(p5[0]);
            pp[6] = float32_to_float16(p6[0]);
            pp[7] = float32_to_float16(p7[0]);
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x4_t _r0123;
            _r0123.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            _r0123.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            _r0123.val[2] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            _r0123.val[3] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
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
            _r0123.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            _r0123.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            _r0123.val[2] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            _r0123.val[3] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            vst4_u16(pp, _r0123);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x2_t _r01;
            _r01.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            _r01.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            vst2q_u16(pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x2_t _r01;
            _r01.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            _r01.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            vst2_u16(pp, _r01);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_fp16_amx(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;
    // NCNN_LOGE("transpose_pack_B_tile_fp32_to_fp16_amx");

    unsigned short* pp = BT;

    int jj = 0;
    for (; jj + 31 < max_jj; jj += 32)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vst1_u16(pp, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)));
            vst1_u16(pp + 4, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1_u16(pp + 8, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 8)));
            vst1_u16(pp + 12, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 12)));
            vst1_u16(pp + 16, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 16)));
            vst1_u16(pp + 20, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 20)));
            vst1_u16(pp + 24, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 24)));
            vst1_u16(pp + 28, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 28)));
            pp += 32;
            p0 += B_hstep;
        }
    }
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vst1_u16(pp, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)));
            vst1_u16(pp + 4, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1_u16(pp + 8, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 8)));
            pp += 12;
            p0 += B_hstep;
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += B_hstep;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += B_hstep;
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p0[1]);
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void transpose_unpack_output_tile_fp32_to_fp16_amx(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x8x4_t _r0;
                _r0.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 4)));
                _r0.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 8)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 12)));
                _r0.val[2] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 16)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 20)));
                _r0.val[3] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 24)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 28)));
                vst4q_u16(p0, _r0);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 4)));
                vst1q_u16(p0, _r0);
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
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x4x4_t _r0123;
                _r0123.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp));
                _r0123.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 4));
                _r0123.val[2] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 8));
                _r0123.val[3] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 12));
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
                uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp));
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
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = float32_to_float16(pp[0]);
                p0[1] = float32_to_float16(pp[2]);
                p0[2] = float32_to_float16(pp[4]);
                p0[3] = float32_to_float16(pp[6]);
                p0[4] = float32_to_float16(pp[1]);
                p0[5] = float32_to_float16(pp[3]);
                p0[6] = float32_to_float16(pp[5]);
                p0[7] = float32_to_float16(pp[7]);
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
                p0[0] = float32_to_float16(pp[0]);
                p0[1] = float32_to_float16(pp[1]);
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
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp));
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
                p0[0] = float32_to_float16(pp[0]);
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}
