// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static inline unsigned char get_weight_wq_int4(const unsigned char* p, int k)
{
    return (p[k / 2] >> ((k & 1) * 4)) & 15;
}

static inline signed char get_packed_weight_wq_int4(const unsigned char* p, size_t index)
{
    return (signed char)(p[index / 2] << ((index & 1) ? 0 : 4) & 0xf0);
}

static inline void loongarch_wq_int4_pack_pair(unsigned char* ptr, int n)
{
    unsigned char tmp[16];
    for (int i = 0; i < n; i++)
        tmp[i] = (ptr[i / 2] >> ((i & 1) * 4) & 15) | (ptr[(i + n) / 2] >> (((i + n) & 1) * 4) & 15) << 4;
    for (int i = 0; i < n; i++)
        ptr[i] = tmp[i];
}

static inline unsigned short loongarch_wq_int4_load_u16(const unsigned char* ptr)
{
    return (unsigned short)(unsigned char)get_packed_weight_wq_int4(ptr, 0)
           | (unsigned short)(unsigned char)get_packed_weight_wq_int4(ptr, 1) << 8;
}

static inline unsigned int loongarch_wq_int4_load_u32(const unsigned char* ptr)
{
    return (unsigned int)loongarch_wq_int4_load_u16(ptr)
           | (unsigned int)loongarch_wq_int4_load_u16(ptr + 1) << 16;
}

#if __loongarch_sx
static inline __m128i loongarch_wq_int4_load16(const unsigned char* ptr)
{
    __m128i _p = __lsx_vldrepl_d(ptr, 0);
    __m128i _lo = __lsx_vslli_b(_p, 4);
    __m128i _hi = __lsx_vandi_b(_p, 0xf0);
    return __lsx_vilvl_b(_hi, _lo);
}

static inline __m128i loongarch_wq_int4_load16_pair(const unsigned char* ptr)
{
    __m128i _p = __lsx_vldrepl_d(ptr, 0);
    __m128i _lo = __lsx_vslli_b(_p, 4);
    __m128i _hi = __lsx_vandi_b(_p, 0xf0);
    return __lsx_vilvl_d(_hi, _lo);
}

static inline __m128i loongarch_wq_int4_load8(const unsigned char* ptr)
{
    __m128i _p = __lsx_vldrepl_w(ptr, 0);
    __m128i _lo = __lsx_vslli_b(_p, 4);
    __m128i _hi = __lsx_vandi_b(_p, 0xf0);
    return __lsx_vilvl_b(_hi, _lo);
}

#if __loongarch_asx
static inline __m256i loongarch_wq_int4_load32(const unsigned char* ptr)
{
    __m128i _p = __lsx_vld(ptr, 0);
    __m128i _lo = __lsx_vslli_b(_p, 4);
    __m128i _hi = __lsx_vandi_b(_p, 0xf0);
    return __lasx_concat_128(_lo, _hi);
}
#endif // __loongarch_asx
#endif // __loongarch_sx

// group-major, output-major within each K4/K1 fragment
static void pack_B_tile_wq_int4(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
    const int block_count = (K + block_size - 1) / block_size;
    unsigned char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
        const unsigned char* p4 = B.row<const unsigned char>(j + jj + 4);
        const unsigned char* p5 = B.row<const unsigned char>(j + jj + 5);
        const unsigned char* p6 = B.row<const unsigned char>(j + jj + 6);
        const unsigned char* p7 = B.row<const unsigned char>(j + jj + 7);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);
        const float* ps4 = B_scales.row(j + jj + 4);
        const float* ps5 = B_scales.row(j + jj + 5);
        const float* ps6 = B_scales.row(j + jj + 6);
        const float* ps7 = B_scales.row(j + jj + 7);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _p0 = __lsx_vldrepl_w(p0, 0);
                __m128i _p1 = __lsx_vldrepl_w(p1, 0);
                __m128i _p2 = __lsx_vldrepl_w(p2, 0);
                __m128i _p3 = __lsx_vldrepl_w(p3, 0);
                __m128i _p4 = __lsx_vldrepl_w(p4, 0);
                __m128i _p5 = __lsx_vldrepl_w(p5, 0);
                __m128i _p6 = __lsx_vldrepl_w(p6, 0);
                __m128i _p7 = __lsx_vldrepl_w(p7, 0);
                __m128i _p01 = __lsx_vilvl_w(_p1, _p0);
                __m128i _p23 = __lsx_vilvl_w(_p3, _p2);
                __m128i _p45 = __lsx_vilvl_w(_p5, _p4);
                __m128i _p67 = __lsx_vilvl_w(_p7, _p6);
                __m128i _p0123 = __lsx_vilvl_d(_p23, _p01);
                __m128i _p4567 = __lsx_vilvl_d(_p67, _p45);
                __m128i _r0 = __lsx_vpickev_h(_p0123, _p0123);
                __m128i _r1 = __lsx_vpickod_h(_p0123, _p0123);
                __m128i _r2 = __lsx_vpickev_h(_p4567, _p4567);
                __m128i _r3 = __lsx_vpickod_h(_p4567, _p4567);
                _r0 = __lsx_vilvl_d(_r2, _r0);
                _r1 = __lsx_vilvl_d(_r3, _r1);
                __lasx_xvst(__lasx_concat_128(_r0, _r1), pp, 0);
                loongarch_wq_int4_pack_pair(pp, 16);
                loongarch_wq_int4_pack_pair(pp + 16, 16);
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
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp[4] = p2[0];
                pp[5] = p2[1];
                pp[6] = p3[0];
                pp[7] = p3[1];
                pp[8] = p4[0];
                pp[9] = p4[1];
                pp[10] = p5[0];
                pp[11] = p5[1];
                pp[12] = p6[0];
                pp[13] = p6[1];
                pp[14] = p7[0];
                pp[15] = p7[1];
                loongarch_wq_int4_pack_pair(pp, 16);
                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = (p0[0] & 15) | (p1[0] & 15) << 4;
                pp[1] = (p2[0] & 15) | (p3[0] & 15) << 4;
                pp[2] = (p4[0] & 15) | (p5[0] & 15) << 4;
                pp[3] = (p6[0] & 15) | (p7[0] & 15) << 4;
                pp[4] = p0[0] >> 4 | (p1[0] & 240);
                pp[5] = p2[0] >> 4 | (p3[0] & 240);
                pp[6] = p4[0] >> 4 | (p5[0] & 240);
                pp[7] = p6[0] >> 4 | (p7[0] & 240);
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
            for (; kk < max_kk; kk++)
            {
                pp[0] = (p0[0] & 15) | (p1[0] & 15) << 4;
                pp[1] = (p2[0] & 15) | (p3[0] & 15) << 4;
                pp[2] = (p4[0] & 15) | (p5[0] & 15) << 4;
                pp[3] = (p6[0] & 15) | (p7[0] & 15) << 4;
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
            }
            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
            *pd++ = (1.f / *ps2++) * 0.0625f;
            *pd++ = (1.f / *ps3++) * 0.0625f;
            *pd++ = (1.f / *ps4++) * 0.0625f;
            *pd++ = (1.f / *ps5++) * 0.0625f;
            *pd++ = (1.f / *ps6++) * 0.0625f;
            *pd++ = (1.f / *ps7++) * 0.0625f;
        }
    }
#endif // __loongarch_asx
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _p0 = __lsx_vldrepl_w(p0, 0);
                __m128i _p1 = __lsx_vldrepl_w(p1, 0);
                __m128i _p2 = __lsx_vldrepl_w(p2, 0);
                __m128i _p3 = __lsx_vldrepl_w(p3, 0);
                __m128i _p01 = __lsx_vilvl_w(_p1, _p0);
                __m128i _p23 = __lsx_vilvl_w(_p3, _p2);
                __m128i _p0123 = __lsx_vilvl_d(_p23, _p01);
                __m128i _r0 = __lsx_vpickev_h(_p0123, _p0123);
                __m128i _r1 = __lsx_vpickod_h(_p0123, _p0123);
                __lsx_vst(__lsx_vilvl_d(_r1, _r0), pp, 0);
                loongarch_wq_int4_pack_pair(pp, 8);
                loongarch_wq_int4_pack_pair(pp + 8, 8);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp[4] = p2[0];
                pp[5] = p2[1];
                pp[6] = p3[0];
                pp[7] = p3[1];
                loongarch_wq_int4_pack_pair(pp, 8);
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = (p0[0] & 15) | (p1[0] & 15) << 4;
                pp[1] = (p2[0] & 15) | (p3[0] & 15) << 4;
                pp[2] = p0[0] >> 4 | (p1[0] & 240);
                pp[3] = p2[0] >> 4 | (p3[0] & 240);
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = (p0[0] & 15) | (p1[0] & 15) << 4;
                pp[1] = (p2[0] & 15) | (p3[0] & 15) << 4;
                pp += 2;
                p0++;
                p1++;
                p2++;
                p3++;
            }
            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
            *pd++ = (1.f / *ps2++) * 0.0625f;
            *pd++ = (1.f / *ps3++) * 0.0625f;
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[kk / 2];
                pp[1] = p0[kk / 2 + 1];
                pp[2] = p1[kk / 2];
                pp[3] = p1[kk / 2 + 1];
                pp += 4;
            }
            for (; kk < max_kk; kk++)
                *pp++ = get_weight_wq_int4(p0, kk) | get_weight_wq_int4(p1, kk) << 4;
            const size_t consumed = ((size_t)max_kk + 1) / 2;
            p0 += consumed;
            p1 += consumed;
            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);
        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            const int bytes = (max_kk + 1) / 2;
            for (int kk = 0; kk < bytes; kk++)
                *pp++ = *p0++;
            *pd++ = (1.f / *ps0++) * 0.0625f;
        }
    }
}

static void gemm_transB_packed_tile_wq_int4(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
    const signed char* pAT = AT_tile;
    const float* pAT_descales = AT_descales_tile;
    const unsigned char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int A_hstep = AT_tile.w;
    const int A_descales_hstep = AT_descales_tile.w;
    const int block_count = (K + block_size - 1) / block_size;
    const int block_start = k / block_size;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)8 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            __m256 _fsum0;
            __m256 _fsum1;
            __m256 _fsum2;
            __m256 _fsum3;
            __m256 _fsum4;
            __m256 _fsum5;
            __m256 _fsum6;
            __m256 _fsum7;
            if (k == 0)
            {
                _fsum0 = (__m256)__lasx_xvldi(0);
                _fsum1 = (__m256)__lasx_xvldi(0);
                _fsum2 = (__m256)__lasx_xvldi(0);
                _fsum3 = (__m256)__lasx_xvldi(0);
                _fsum4 = (__m256)__lasx_xvldi(0);
                _fsum5 = (__m256)__lasx_xvldi(0);
                _fsum6 = (__m256)__lasx_xvldi(0);
                _fsum7 = (__m256)__lasx_xvldi(0);
            }
            else
            {
                _fsum0 = (__m256)__lasx_xvld(outptr, 0);
                _fsum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _fsum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _fsum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                _fsum4 = (__m256)__lasx_xvld(outptr + 32, 0);
                _fsum5 = (__m256)__lasx_xvld(outptr + 40, 0);
                _fsum6 = (__m256)__lasx_xvld(outptr + 48, 0);
                _fsum7 = (__m256)__lasx_xvld(outptr + 56, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum2 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum3 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum4 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum5 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum6 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum7 = __lasx_xvreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA = __lasx_xvld(pA, 0);
                    __m256i _pA1 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB0 = loongarch_wq_int4_load32(pB);

                    __m256i _s0 = __lasx_xvmulwev_h_b(_pA, _pB0);
                    __m256i _s1 = __lasx_xvmulwev_h_b(_pA1, _pB0);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB0);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB0);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB0, _LSX_SHUFFLE(1, 0, 3, 2));
                    _s0 = __lasx_xvmulwev_h_b(_pA, _pB1);
                    _s1 = __lasx_xvmulwev_h_b(_pA1, _pB1);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB1);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB1);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s1, _s1));
                    __m256i _pA2 = __lasx_xvpermi_q(_pA, _pA, _LSX_SHUFFLE(0, 0, 0, 1));
                    __m256i _pA3 = __lasx_xvshuf4i_w(_pA2, _LSX_SHUFFLE(0, 3, 2, 1));
                    _s0 = __lasx_xvmulwev_h_b(_pA2, _pB0);
                    _s1 = __lasx_xvmulwev_h_b(_pA3, _pB0);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA2, _pB0);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA3, _pB0);
                    _sum4 = __lasx_xvadd_w(_sum4, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum5 = __lasx_xvadd_w(_sum5, __lasx_xvhaddw_w_h(_s1, _s1));
                    _s0 = __lasx_xvmulwev_h_b(_pA2, _pB1);
                    _s1 = __lasx_xvmulwev_h_b(_pA3, _pB1);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA2, _pB1);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA3, _pB1);
                    _sum6 = __lasx_xvadd_w(_sum6, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum7 = __lasx_xvadd_w(_sum7, __lasx_xvhaddw_w_h(_s1, _s1));
                    pB += 16;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                    _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);
                    __m256i _pA1 = __lasx_xvshuf4i_h(_pA, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _pB08 = loongarch_wq_int4_load8(pB);
                    __m256i _pB0 = __lasx_concat_128(_pB08, _pB08);
                    _pB0 = __lasx_xvilvl_b(__lasx_xvslti_b(_pB0, 0), _pB0);
                    __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);
                    __m256i _s1 = __lasx_xvmul_h(_pA1, _pB0);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));
                    __m256i _pB1 = __lasx_xvshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 3, 2));
                    _s0 = __lasx_xvmul_h(_pA, _pB1);
                    _s1 = __lasx_xvmul_h(_pA1, _pB1);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(_s0));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(_s1));
                    __m256i _pA2 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _pA3 = __lasx_xvshuf4i_h(_pA2, _LSX_SHUFFLE(0, 3, 2, 1));
                    _s0 = __lasx_xvmul_h(_pA2, _pB0);
                    _s1 = __lasx_xvmul_h(_pA3, _pB0);
                    _sum4 = __lasx_xvadd_w(_sum4, __lasx_vext2xv_w_h(_s0));
                    _sum5 = __lasx_xvadd_w(_sum5, __lasx_vext2xv_w_h(_s1));
                    _s0 = __lasx_xvmul_h(_pA2, _pB1);
                    _s1 = __lasx_xvmul_h(_pA3, _pB1);
                    _sum6 = __lasx_xvadd_w(_sum6, __lasx_vext2xv_w_h(_s0));
                    _sum7 = __lasx_xvadd_w(_sum7, __lasx_vext2xv_w_h(_s1));
                    pB += 4;
                    pA += 8;
                }

                __m256 _descaleB = (__m256)__lasx_xvld(pB_descales, 0);
                __m256 _descaleB1 = (__m256)__lasx_xvshuf4i_w((__m256i)_descaleB, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _descaleA = (__m256)__lasx_xvld(pA_descales, 0);
                __m256 _descaleA1 = (__m256)__lasx_xvshuf4i_w((__m256i)_descaleA, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256 _descaleA2 = (__m256)__lasx_xvpermi_q((__m256i)_descaleA, (__m256i)_descaleA, _LSX_SHUFFLE(0, 0, 0, 1));
                __m256 _descaleA3 = (__m256)__lasx_xvshuf4i_w((__m256i)_descaleA2, _LSX_SHUFFLE(0, 3, 2, 1));
                _fsum0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_descaleA, _descaleB), _fsum0);
                _fsum1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_descaleA1, _descaleB), _fsum1);
                _fsum2 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum2), __lasx_xvfmul_s(_descaleA, _descaleB1), _fsum2);
                _fsum3 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum3), __lasx_xvfmul_s(_descaleA1, _descaleB1), _fsum3);
                _fsum4 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum4), __lasx_xvfmul_s(_descaleA2, _descaleB), _fsum4);
                _fsum5 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum5), __lasx_xvfmul_s(_descaleA3, _descaleB), _fsum5);
                _fsum6 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum6), __lasx_xvfmul_s(_descaleA2, _descaleB1), _fsum6);
                _fsum7 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum7), __lasx_xvfmul_s(_descaleA3, _descaleB1), _fsum7);
                pA_descales += 8;
                pB_descales += 8;
            }
            __lasx_xvst(_fsum0, outptr, 0);
            __lasx_xvst(_fsum1, outptr + 8, 0);
            __lasx_xvst(_fsum2, outptr + 16, 0);
            __lasx_xvst(_fsum3, outptr + 24, 0);
            __lasx_xvst(_fsum4, outptr + 32, 0);
            __lasx_xvst(_fsum5, outptr + 40, 0);
            __lasx_xvst(_fsum6, outptr + 48, 0);
            __lasx_xvst(_fsum7, outptr + 56, 0);
            outptr += 64;
            pB_panel += ((size_t)8 * K + 1) / 2;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            __m128 _fsum00;
            __m128 _fsum01;
            __m128 _fsum10;
            __m128 _fsum11;
            __m128 _fsum20;
            __m128 _fsum21;
            __m128 _fsum30;
            __m128 _fsum31;
            if (k == 0)
            {
                _fsum00 = (__m128)__lsx_vldi(0);
                _fsum01 = (__m128)__lsx_vldi(0);
                _fsum10 = (__m128)__lsx_vldi(0);
                _fsum11 = (__m128)__lsx_vldi(0);
                _fsum20 = (__m128)__lsx_vldi(0);
                _fsum21 = (__m128)__lsx_vldi(0);
                _fsum30 = (__m128)__lsx_vldi(0);
                _fsum31 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum00 = (__m128)__lsx_vld(outptr, 0);
                _fsum01 = (__m128)__lsx_vld(outptr + 4, 0);
                _fsum10 = (__m128)__lsx_vld(outptr + 8, 0);
                _fsum11 = (__m128)__lsx_vld(outptr + 12, 0);
                _fsum20 = (__m128)__lsx_vld(outptr + 16, 0);
                _fsum21 = (__m128)__lsx_vld(outptr + 20, 0);
                _fsum30 = (__m128)__lsx_vld(outptr + 24, 0);
                _fsum31 = (__m128)__lsx_vld(outptr + 28, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                __m128i _sum20 = __lsx_vreplgr2vr_w(0);
                __m128i _sum21 = __lsx_vreplgr2vr_w(0);
                __m128i _sum30 = __lsx_vreplgr2vr_w(0);
                __m128i _sum31 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vld(pA + 16, 0);
                    __m128i _pB0 = loongarch_wq_int4_load16_pair(pB);

                    __m128i _s = __lsx_vmulwev_h_b(_pA0, _pB0);
                    _s = __lsx_vmaddwod_h_b(_s, _pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmulwev_h_b(_pA1, _pB0);
                    _s = __lsx_vmaddwod_h_b(_s, _pA1, _pB0);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s, _s));

                    __m128i _pB0r = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    _s = __lsx_vmulwev_h_b(_pA0, _pB0r);
                    _s = __lsx_vmaddwod_h_b(_s, _pA0, _pB0r);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmulwev_h_b(_pA1, _pB0r);
                    _s = __lsx_vmaddwod_h_b(_s, _pA1, _pB0r);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s, _s));

                    __m128i _pA0r = __lsx_vshuf4i_w(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pA1r = __lsx_vshuf4i_w(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                    _s = __lsx_vmulwev_h_b(_pA0r, _pB0);
                    _s = __lsx_vmaddwod_h_b(_s, _pA0r, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmulwev_h_b(_pA1r, _pB0);
                    _s = __lsx_vmaddwod_h_b(_s, _pA1r, _pB0);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vhaddw_w_h(_s, _s));

                    _s = __lsx_vmulwev_h_b(_pA0r, _pB0r);
                    _s = __lsx_vmaddwod_h_b(_s, _pA0r, _pB0r);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmulwev_h_b(_pA1r, _pB0r);
                    _s = __lsx_vmaddwod_h_b(_s, _pA1r, _pB0r);
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vhaddw_w_h(_s, _s));
                    pB += 8;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    _pA0 = __lsx_vilvl_b(__lsx_vslti_b(_pA0, 0), _pA0);
                    __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                    _pA1 = __lsx_vilvl_b(__lsx_vslti_b(_pA1, 0), _pA1);
                    __m128i _pB0 = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB));
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    __m128i _s = __lsx_vmul_h(_pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1, _pB0);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    __m128i _pB0r = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    _s = __lsx_vmul_h(_pA0, _pB0r);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1, _pB0r);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    __m128i _pA0r = __lsx_vshuf4i_h(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pA1r = __lsx_vshuf4i_h(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                    _s = __lsx_vmul_h(_pA0r, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1r, _pB0);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA0r, _pB0r);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1r, _pB0r);
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    pB += 2;
                    pA += 8;
                }

                __m128 _descaleB = (__m128)__lsx_vld(pB_descales, 0);
                __m128 _descaleBr = (__m128)__lsx_vshuf4i_w((__m128i)_descaleB, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128 _descaleA0 = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _descaleA1 = (__m128)__lsx_vld(pA_descales + 4, 0);
                __m128 _descaleA0r = (__m128)__lsx_vshuf4i_w((__m128i)_descaleA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _descaleA1r = (__m128)__lsx_vshuf4i_w((__m128i)_descaleA1, _LSX_SHUFFLE(1, 0, 3, 2));
                _fsum00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_descaleA0, _descaleB), _fsum00);
                _fsum01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_descaleA1, _descaleB), _fsum01);
                _fsum10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_descaleA0, _descaleBr), _fsum10);
                _fsum11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_descaleA1, _descaleBr), _fsum11);
                _fsum20 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum20), __lsx_vfmul_s(_descaleA0r, _descaleB), _fsum20);
                _fsum21 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum21), __lsx_vfmul_s(_descaleA1r, _descaleB), _fsum21);
                _fsum30 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum30), __lsx_vfmul_s(_descaleA0r, _descaleBr), _fsum30);
                _fsum31 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum31), __lsx_vfmul_s(_descaleA1r, _descaleBr), _fsum31);
                pA_descales += 8;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_fsum00, outptr, 0);
            __lsx_vst((__m128i)_fsum01, outptr + 4, 0);
            __lsx_vst((__m128i)_fsum10, outptr + 8, 0);
            __lsx_vst((__m128i)_fsum11, outptr + 12, 0);
            __lsx_vst((__m128i)_fsum20, outptr + 16, 0);
            __lsx_vst((__m128i)_fsum21, outptr + 20, 0);
            __lsx_vst((__m128i)_fsum30, outptr + 24, 0);
            __lsx_vst((__m128i)_fsum31, outptr + 28, 0);
            outptr += 32;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            __m128 _fsum00;
            __m128 _fsum01;
            __m128 _fsum10;
            __m128 _fsum11;
            if (k == 0)
            {
                _fsum00 = (__m128)__lsx_vldi(0);
                _fsum01 = (__m128)__lsx_vldi(0);
                _fsum10 = (__m128)__lsx_vldi(0);
                _fsum11 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum00 = (__m128)__lsx_vld(outptr, 0);
                _fsum01 = (__m128)__lsx_vld(outptr + 4, 0);
                _fsum10 = (__m128)__lsx_vld(outptr + 8, 0);
                _fsum11 = (__m128)__lsx_vld(outptr + 12, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vld(pA + 16, 0);
                    __m128i _pB = loongarch_wq_int4_load8(pB);
                    __m128i _pB0 = __lsx_vreplvei_w(_pB, 0);
                    __m128i _pB1 = __lsx_vreplvei_w(_pB, 1);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s1, _s1));
                    _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s1, _s1));
                    pA += 32;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA0 = __lsx_vreplvei_d(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_d(_pA, 1);
                    __m128i _pB = __lsx_vreplgr2vr_h((short)loongarch_wq_int4_load_u16(pB));
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pB0 = __lsx_vreplvei_h(_pB, 0);
                    __m128i _pB1 = __lsx_vreplvei_h(_pB, 1);
                    __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                    __m128i _s1 = __lsx_vmul_h(_pA1, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmul_h(_pA0, _pB1);
                    _s1 = __lsx_vmul_h(_pA1, _pB1);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 8;
                    pB += 1;
                }
                __m128 _descaleA0 = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _descaleA1 = (__m128)__lsx_vld(pA_descales + 4, 0);
                __m128 _descaleB = __lsx_vreplfr2vr_s(pB_descales[0]);
                _fsum00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_descaleA0, _descaleB), _fsum00);
                _fsum01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_descaleA1, _descaleB), _fsum01);
                _descaleB = __lsx_vreplfr2vr_s(pB_descales[1]);
                _fsum10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_descaleA0, _descaleB), _fsum10);
                _fsum11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_descaleA1, _descaleB), _fsum11);
                pA_descales += 8;
                pB_descales += 2;
            }
            __lsx_vst((__m128i)_fsum00, outptr, 0);
            __lsx_vst((__m128i)_fsum01, outptr + 4, 0);
            __lsx_vst((__m128i)_fsum10, outptr + 8, 0);
            __lsx_vst((__m128i)_fsum11, outptr + 12, 0);
            outptr += 16;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            __m128 _fsum0;
            __m128 _fsum1;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
                _fsum1 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
                _fsum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vld(pA + 16, 0);
                    __m128i _pB = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB));
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                    pA += 32;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA0 = __lsx_vreplvei_d(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_d(_pA, 1);
                    __m128i _pB = __lsx_vreplgr2vr_h((signed char)get_packed_weight_wq_int4(pB, kk & 1));
                    __m128i _s0 = __lsx_vmul_h(_pA0, _pB);
                    __m128i _s1 = __lsx_vmul_h(_pA1, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 8;
                    if ((kk & 1) || kk + 1 == max_kk0)
                        pB++;
                }
                __m128 _descaleB = __lsx_vreplfr2vr_s(*pB_descales++);
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s((__m128)__lsx_vld(pA_descales, 0), _descaleB), _fsum0);
                _fsum1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s((__m128)__lsx_vld(pA_descales + 4, 0), _descaleB), _fsum1);
                pA_descales += 8;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            __lsx_vst((__m128i)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 8;
        pAT_descales += A_descales_hstep * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)8 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            __m256 _fsum0;
            __m256 _fsum1;
            __m256 _fsum2;
            __m256 _fsum3;
            if (k == 0)
            {
                _fsum0 = (__m256)__lasx_xvldi(0);
                _fsum1 = (__m256)__lasx_xvldi(0);
                _fsum2 = (__m256)__lasx_xvldi(0);
                _fsum3 = (__m256)__lasx_xvldi(0);
            }
            else
            {
                _fsum0 = (__m256)__lasx_xvld(outptr, 0);
                _fsum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _fsum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _fsum3 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum2 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum3 = __lasx_xvreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA4 = __lsx_vld(pA, 0);
                    __m256i _pA = __lasx_concat_128(_pA4, _pA4);
                    __m256i _pA1 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB = loongarch_wq_int4_load32(pB);
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s, _s));
                    pB += 16;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA4 = __lsx_vldrepl_w(pA, 0);
                    _pA4 = __lsx_vilvl_b(__lsx_vslti_b(_pA4, 0), _pA4);
                    __m128i _pB04 = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB));
                    _pB04 = __lsx_vilvl_b(__lsx_vslti_b(_pB04, 0), _pB04);
                    __m128i _pB48 = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB + 2));
                    _pB48 = __lsx_vilvl_b(__lsx_vslti_b(_pB48, 0), _pB48);
                    __m128i _pA4r = __lsx_vshuf4i_h(_pA4, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB04r = __lsx_vshuf4i_h(_pB04, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _pB48r = __lsx_vshuf4i_h(_pB48, _LSX_SHUFFLE(0, 3, 2, 1));

                    __m128i _s0 = __lsx_vmul_h(_pA4, _pB04);
                    __m128i _s1 = __lsx_vmul_h(_pA4, _pB04r);
                    __m128i _s2 = __lsx_vmul_h(_pA4r, _pB04);
                    __m128i _s3 = __lsx_vmul_h(_pA4r, _pB04r);
                    __m128i _s4 = __lsx_vmul_h(_pA4, _pB48);
                    __m128i _s5 = __lsx_vmul_h(_pA4, _pB48r);
                    __m128i _s6 = __lsx_vmul_h(_pA4r, _pB48);
                    __m128i _s7 = __lsx_vmul_h(_pA4r, _pB48r);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_concat_128(__lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0), __lsx_vilvl_h(__lsx_vslti_h(_s4, 0), _s4)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_concat_128(__lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1), __lsx_vilvl_h(__lsx_vslti_h(_s5, 0), _s5)));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_concat_128(__lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2), __lsx_vilvl_h(__lsx_vslti_h(_s6, 0), _s6)));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_concat_128(__lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3), __lsx_vilvl_h(__lsx_vslti_h(_s7, 0), _s7)));
                    pB += 4;
                    pA += 4;
                }

                __m256 _descaleB = (__m256)__lasx_xvld(pB_descales, 0);
                __m128i _descaleA4 = __lsx_vld(pA_descales, 0);
                __m256 _descaleA = (__m256)__lasx_concat_128(_descaleA4, _descaleA4);
                __m256 _descaleA1 = (__m256)__lasx_xvshuf4i_w((__m256i)_descaleA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _descaleBr = (__m256)__lasx_xvshuf4i_w((__m256i)_descaleB, _LSX_SHUFFLE(0, 3, 2, 1));
                _fsum0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_descaleB, _descaleA), _fsum0);
                _fsum1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_descaleBr, _descaleA), _fsum1);
                _fsum2 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum2), __lasx_xvfmul_s(_descaleB, _descaleA1), _fsum2);
                _fsum3 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum3), __lasx_xvfmul_s(_descaleBr, _descaleA1), _fsum3);
                pA_descales += 4;
                pB_descales += 8;
            }

            __lasx_xvst(_fsum0, outptr, 0);
            __lasx_xvst(_fsum1, outptr + 8, 0);
            __lasx_xvst(_fsum2, outptr + 16, 0);
            __lasx_xvst(_fsum3, outptr + 24, 0);
            outptr += 32;
            pB_panel += ((size_t)8 * K + 1) / 2;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            __m128 _fsum0;
            __m128 _fsum1;
            __m128 _fsum2;
            __m128 _fsum3;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
                _fsum1 = (__m128)__lsx_vldi(0);
                _fsum2 = (__m128)__lsx_vldi(0);
                _fsum3 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
                _fsum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _fsum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _fsum3 = (__m128)__lsx_vld(outptr + 12, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                __m128i _sum2 = __lsx_vreplgr2vr_w(0);
                __m128i _sum3 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB = loongarch_wq_int4_load16_pair(pB);
                    __m128i _pB1 = __lsx_vshuf4i_w(_pB, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s, _s));
                    pB += 8;
                    pA += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA1 = __lsx_vshuf4i_h(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB));
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pB1 = __lsx_vshuf4i_h(_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                    __m128i _s0 = __lsx_vmul_h(_pA, _pB);
                    __m128i _s1 = __lsx_vmul_h(_pA, _pB1);
                    __m128i _s2 = __lsx_vmul_h(_pA1, _pB);
                    __m128i _s3 = __lsx_vmul_h(_pA1, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2));
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3));
                    pB += 2;
                    pA += 4;
                }
                __m128 _descaleB = (__m128)__lsx_vld(pB_descales, 0);
                __m128 _descaleA = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _descaleA1 = (__m128)__lsx_vshuf4i_w((__m128i)_descaleA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _descaleB1 = (__m128)__lsx_vshuf4i_w((__m128i)_descaleB, _LSX_SHUFFLE(0, 3, 2, 1));
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_descaleB, _descaleA), _fsum0);
                _fsum1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s(_descaleB1, _descaleA), _fsum1);
                _fsum2 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum2), __lsx_vfmul_s(_descaleB, _descaleA1), _fsum2);
                _fsum3 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum3), __lsx_vfmul_s(_descaleB1, _descaleA1), _fsum3);
                pA_descales += 4;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            __lsx_vst((__m128i)_fsum1, outptr + 4, 0);
            __lsx_vst((__m128i)_fsum2, outptr + 8, 0);
            __lsx_vst((__m128i)_fsum3, outptr + 12, 0);
            outptr += 16;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            __m128 _fsum0;
            __m128 _fsum1;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
                _fsum1 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
                _fsum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pB0 = loongarch_wq_int4_load8(pB);
                    __m128i _pB1 = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(2, 3, 0, 1));
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB0), _pA, _pB0);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                    pA += 16;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pB0 = __lsx_vreplgr2vr_h((short)loongarch_wq_int4_load_u16(pB));
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    __m128i _pB1 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s0 = __lsx_vmul_h(_pA, _pB0);
                    __m128i _s1 = __lsx_vmul_h(_pA, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 4;
                    pB += 1;
                }
                __m128 _descaleA = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _descaleB0 = (__m128)__lsx_vldrepl_d(pB_descales, 0);
                __m128 _descaleB1 = (__m128)__lsx_vshuf4i_w((__m128i)_descaleB0, _LSX_SHUFFLE(2, 3, 0, 1));
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_descaleA, _descaleB0), _fsum0);
                _fsum1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s(_descaleA, _descaleB1), _fsum1);
                pA_descales += 4;
                pB_descales += 2;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            __lsx_vst((__m128i)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            __m128 _fsum0;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pB = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB));
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    pA += 16;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(_pA, __lsx_vreplgr2vr_h(get_packed_weight_wq_int4(pB, kk & 1)));
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    pA += 4;
                    if ((kk & 1) || kk + 1 == max_kk0)
                        pB++;
                }
                __m128 _scale = __lsx_vfmul_s((__m128)__lsx_vld(pA_descales, 0), __lsx_vreplfr2vr_s(*pB_descales++));
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), _scale, _fsum0);
                pA_descales += 4;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            outptr += 4;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep * 4;
        pAT_descales += A_descales_hstep * 4;
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)8 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            __m256 _fsum0;
            __m256 _fsum1;
            if (k == 0)
            {
                _fsum0 = (__m256)__lasx_xvldi(0);
                _fsum1 = (__m256)__lasx_xvldi(0);
            }
            else
            {
                _fsum0 = (__m256)__lasx_xvld(outptr, 0);
                _fsum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pB = loongarch_wq_int4_load32(pB);
                    __m256i _pA0 = __lasx_xvldrepl_w(pA, 0);
                    __m256i _pA1 = __lasx_xvldrepl_w(pA + 4, 0);
                    __m256i _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s, _s));
                    pB += 16;
                    pA += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pB = loongarch_wq_int4_load8(pB);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pA = __lsx_vldrepl_h(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 0), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 1), _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(__lasx_cast_128(_s1)));
                    pB += 4;
                    pA += 2;
                }
                __m256 _descaleB = (__m256)__lasx_xvld(pB_descales, 0);
                _fsum0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_descaleB, (__m256)__lasx_xvreplfr2vr_s(pA_descales[0])), _fsum0);
                _fsum1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_descaleB, (__m256)__lasx_xvreplfr2vr_s(pA_descales[1])), _fsum1);
                pA_descales += 2;
                pB_descales += 8;
            }
            __lasx_xvst(_fsum0, outptr, 0);
            __lasx_xvst(_fsum1, outptr + 8, 0);
            outptr += 16;
            pB_panel += ((size_t)8 * K + 1) / 2;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            __m128 _fsum0;
            __m128 _fsum1;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
                _fsum1 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
                _fsum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pB = loongarch_wq_int4_load16_pair(pB);
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s, _s));
                    pB += 8;
                    pA += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pB = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB));
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pA = __lsx_vldrepl_h(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 0), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 1), _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pB += 2;
                    pA += 2;
                }
                __m128 _descaleB = (__m128)__lsx_vld(pB_descales, 0);
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_descaleB, __lsx_vreplfr2vr_s(pA_descales[0])), _fsum0);
                _fsum1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s(_descaleB, __lsx_vreplfr2vr_s(pA_descales[1])), _fsum1);
                pA_descales += 2;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            __lsx_vst((__m128i)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            float out00;
            float out01;
            float out10;
            float out11;
            if (k == 0)
            {
                out00 = 0.f;
                out01 = 0.f;
                out10 = 0.f;
                out11 = 0.f;
            }
            else
            {
                out00 = outptr[0];
                out01 = outptr[1];
                out10 = outptr[2];
                out11 = outptr[3];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum00 += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    sum01 += pA[0] * get_packed_weight_wq_int4(pB, 4) + pA[1] * get_packed_weight_wq_int4(pB, 5) + pA[2] * get_packed_weight_wq_int4(pB, 6) + pA[3] * get_packed_weight_wq_int4(pB, 7);
                    sum10 += pA[4] * get_packed_weight_wq_int4(pB, 0) + pA[5] * get_packed_weight_wq_int4(pB, 1) + pA[6] * get_packed_weight_wq_int4(pB, 2) + pA[7] * get_packed_weight_wq_int4(pB, 3);
                    sum11 += pA[4] * get_packed_weight_wq_int4(pB, 4) + pA[5] * get_packed_weight_wq_int4(pB, 5) + pA[6] * get_packed_weight_wq_int4(pB, 6) + pA[7] * get_packed_weight_wq_int4(pB, 7);
                    pB += 4;
                    pA += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum00 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    sum01 += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    sum10 += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    sum11 += pA[1] * get_packed_weight_wq_int4(pB, 1);
                    pB += 1;
                    pA += 2;
                }
                const float bscale0 = pB_descales[0];
                const float bscale1 = pB_descales[1];
                const float ascale0 = pA_descales[0];
                const float ascale1 = pA_descales[1];
                out00 += sum00 * ascale0 * bscale0;
                out01 += sum01 * ascale0 * bscale1;
                out10 += sum10 * ascale1 * bscale0;
                out11 += sum11 * ascale1 * bscale1;
                pA_descales += 2;
                pB_descales += 2;
            }
            outptr[0] = out00;
            outptr[1] = out01;
            outptr[2] = out10;
            outptr[3] = out11;
            outptr += 4;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            float out0;
            float out1;
            if (k == 0)
            {
                out0 = 0.f;
                out1 = 0.f;
            }
            else
            {
                out0 = outptr[0];
                out1 = outptr[1];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum0 = 0;
                int sum1 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __loongarch_sx
                __m128i _sum = __lsx_vreplgr2vr_w(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pB0 = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB));
                    __m128i _pB1 = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB + 2));
                    __m128i _pB = __lsx_vilvl_d(_pB1, _pB0);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum = __lsx_vadd_w(_sum, __lsx_vhaddw_w_h(_s, _s));
                    pA += 16;
                    pB += 4;
                }
                sum0 = __lsx_vpickve2gr_w(_sum, 0) + __lsx_vpickve2gr_w(_sum, 2);
                sum1 = __lsx_vpickve2gr_w(_sum, 1) + __lsx_vpickve2gr_w(_sum, 3);
#endif // __loongarch_sx
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0 += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    sum1 += pA[4] * get_packed_weight_wq_int4(pB, 0) + pA[5] * get_packed_weight_wq_int4(pB, 1) + pA[6] * get_packed_weight_wq_int4(pB, 2) + pA[7] * get_packed_weight_wq_int4(pB, 3);
                    pB += 2;
                    pA += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * get_packed_weight_wq_int4(pB, kk & 1);
                    sum1 += pA[1] * get_packed_weight_wq_int4(pB, kk & 1);
                    if ((kk & 1) || kk + 1 == max_kk0)
                        pB++;
                    pA += 2;
                }
                const float bscale = *pB_descales++;
                out0 += sum0 * pA_descales[0] * bscale;
                out1 += sum1 * pA_descales[1] * bscale;
                pA_descales += 2;
            }
            outptr[0] = out0;
            outptr[1] = out1;
            outptr += 2;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)8 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            __m256 _fsum0;
            if (k == 0)
            {
                _fsum0 = (__m256)__lasx_xvldi(0);
            }
            else
            {
                _fsum0 = (__m256)__lasx_xvld(outptr, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pB = loongarch_wq_int4_load32(pB);
                    __m256i _pA0 = __lasx_xvldrepl_w(pA, 0);
                    __m256i _s0 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    pB += 16;
                    pA += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pB = loongarch_wq_int4_load8(pB);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h(pA[0]), _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    pB += 4;
                    pA++;
                }
                __m256 _descaleB = (__m256)__lasx_xvld(pB_descales, 0);
                _fsum0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_descaleB, (__m256)__lasx_xvreplfr2vr_s(*pA_descales++)), _fsum0);
                pB_descales += 8;
            }
            __lasx_xvst(_fsum0, outptr, 0);
            outptr += 8;
            pB_panel += ((size_t)8 * K + 1) / 2;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            __m128 _fsum0;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pB = loongarch_wq_int4_load16_pair(pB);
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    pB += 8;
                    pA += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pB = __lsx_vreplgr2vr_w((int)loongarch_wq_int4_load_u32(pB));
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h(pA[0]), _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    pB += 2;
                    pA++;
                }
                __m128 _descaleB = (__m128)__lsx_vld(pB_descales, 0);
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_descaleB, __lsx_vreplfr2vr_s(*pA_descales++)), _fsum0);
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            outptr += 4;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            float out0;
            float out1;
            if (k == 0)
            {
                out0 = 0.f;
                out1 = 0.f;
            }
            else
            {
                out0 = outptr[0];
                out1 = outptr[1];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum0 = 0;
                int sum1 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __loongarch_sx
                __m128i _sum = __lsx_vreplgr2vr_w(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                    __m128i _pA = __lsx_vilvl_d(_pA1, _pA0);
                    __m128i _pB = loongarch_wq_int4_load16(pB);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum = __lsx_vadd_w(_sum, __lsx_vhaddw_w_h(_s, _s));
                    pA += 8;
                    pB += 8;
                }
                sum0 = __lsx_vpickve2gr_w(_sum, 0) + __lsx_vpickve2gr_w(_sum, 2);
                sum1 = __lsx_vpickve2gr_w(_sum, 1) + __lsx_vpickve2gr_w(_sum, 3);
#endif // __loongarch_sx
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0 += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    sum1 += pA[0] * get_packed_weight_wq_int4(pB, 4) + pA[1] * get_packed_weight_wq_int4(pB, 5) + pA[2] * get_packed_weight_wq_int4(pB, 6) + pA[3] * get_packed_weight_wq_int4(pB, 7);
                    pB += 4;
                    pA += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    sum1 += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    pB += 1;
                    pA++;
                }
                const float ascale = *pA_descales++;
                out0 += sum0 * ascale * pB_descales[0];
                out1 += sum1 * ascale * pB_descales[1];
                pB_descales += 2;
            }
            outptr[0] = out0;
            outptr[1] = out1;
            outptr += 2;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            float out0;
            if (k == 0)
            {
                out0 = 0.f;
            }
            else
            {
                out0 = outptr[0];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum0 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0 += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    pB += 2;
                    pA += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * get_packed_weight_wq_int4(pB, kk & 1);
                    if ((kk & 1) || kk + 1 == max_kk0)
                        pB++;
                    pA++;
                }
                out0 += sum0 * *pA_descales++ * *pB_descales++;
            }
            *outptr++ = out0;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void get_optimal_tile_mnk_wq_int4(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (sizeof(signed char) + 0.5f + sizeof(float) + 8.f / block_size));

#if __loongarch_sx
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(2, tile_size / 2 * 2);
#endif

    TILE_K = std::max(block_size, tile_size / block_size * block_size);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + block_size - 1) / block_size * block_size);
        TILE_K = std::min(TILE_K, K);

        if (nn_K == 1)
        {
            const float linear_footprint = (1.5f + 8.f / block_size) * TILE_K;
            tile_size = std::max(1, (int)((sqrtf(linear_footprint * linear_footprint + 16.f * l2_cache_size) - linear_footprint) / 8.f));

#if __loongarch_sx
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(8, tile_size / 8 * 8);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
            TILE_N = std::max(2, tile_size / 2 * 2);
#endif
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __loongarch_sx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2);
#endif
    }

    if (nT > 1)
    {
#if __loongarch_sx
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __loongarch_sx
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_N > 0)
    {
#if __loongarch_sx
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#else
        TILE_N = (constant_TILE_N + 1) / 2 * 2;
#endif // __loongarch_sx
    }
    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, (constant_TILE_K + block_size - 1) / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(TILE_K, K);
    }
}
