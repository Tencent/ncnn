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

static inline void mips_wq_int4_pack_pair(unsigned char* ptr)
{
    unsigned char tmp[8];
    for (int i = 0; i < 8; i++)
        tmp[i] = (ptr[i / 2] >> ((i & 1) * 4) & 15) | (ptr[(i + 8) / 2] >> (((i + 8) & 1) * 4) & 15) << 4;
    for (int i = 0; i < 8; i++)
        ptr[i] = tmp[i];
}

#if __mips_msa
static inline v16i8 mips_wq_int4_load2(const unsigned char* ptr)
{
    v16i8 _p = __msa_fill_b((signed char)ptr[0]);
    v16i8 _lo = __msa_slli_b(_p, 4);
    v16i8 _hi = (v16i8)__msa_andi_b((v16u8)_p, 0xf0);
    return __msa_ilvr_b(_hi, _lo);
}

static inline v16i8 mips_wq_int4_load4(const unsigned char* ptr)
{
    v16i8 _p = (v16i8)__msa_fill_h((unsigned short)ptr[0] | (unsigned short)ptr[1] << 8);
    v16i8 _lo = __msa_slli_b(_p, 4);
    v16i8 _hi = (v16i8)__msa_andi_b((v16u8)_p, 0xf0);
    return __msa_ilvr_b(_hi, _lo);
}

static inline v16i8 mips_wq_int4_load16(const unsigned char* ptr)
{
    v16i8 _p = (v16i8)__msa_loadl_d(ptr);
    v16i8 _lo = __msa_slli_b(_p, 4);
    v16i8 _hi = (v16i8)__msa_andi_b((v16u8)_p, 0xf0);
    return __msa_ilvr_b(_hi, _lo);
}

static inline v16i8 mips_wq_int4_load16_pair(const unsigned char* ptr)
{
    v16i8 _p = (v16i8)__msa_loadl_d(ptr);
    v16i8 _lo = __msa_slli_b(_p, 4);
    v16i8 _hi = (v16i8)__msa_andi_b((v16u8)_p, 0xf0);
    return (v16i8)__msa_ilvr_d((v2i64)_hi, (v2i64)_lo);
}

static inline v16i8 mips_wq_int4_load8(const unsigned char* ptr)
{
    v16i8 _p = (v16i8)__msa_fill_w_ptr(ptr);
    v16i8 _lo = __msa_slli_b(_p, 4);
    v16i8 _hi = (v16i8)__msa_andi_b((v16u8)_p, 0xf0);
    return __msa_ilvr_b(_hi, _lo);
}
#endif // __mips_msa

// group-major, output-major within each K4/K1 fragment
static void pack_B_tile_wq_int4(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
    const int block_count = (K + block_size - 1) / block_size;
    unsigned char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __mips_msa
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
                v16i8 _p = (v16i8)__msa_set_w(__msa_load_w(p0), __msa_load_w(p1), __msa_load_w(p2), __msa_load_w(p3));
                v8i16 _p0 = __msa_pckev_h((v8i16)_p, (v8i16)_p);
                v8i16 _p1 = __msa_pckod_h((v8i16)_p, (v8i16)_p);
                __msa_st_b((v16i8)__msa_ilvr_d((v2i64)_p1, (v2i64)_p0), pp, 0);
                mips_wq_int4_pack_pair(pp);
                mips_wq_int4_pack_pair(pp + 8);
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
                mips_wq_int4_pack_pair(pp);
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
#endif // __mips_msa
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
            p0 += ((size_t)max_kk + 1) / 2;
            p1 += ((size_t)max_kk + 1) / 2;
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
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        const v8i16 _one = __msa_fill_h(1);

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            v4f32 _fsum2;
            v4f32 _fsum3;
            v4f32 _fsum4;
            v4f32 _fsum5;
            v4f32 _fsum6;
            v4f32 _fsum7;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
                _fsum2 = (v4f32)__msa_fill_w(0);
                _fsum3 = (v4f32)__msa_fill_w(0);
                _fsum4 = (v4f32)__msa_fill_w(0);
                _fsum5 = (v4f32)__msa_fill_w(0);
                _fsum6 = (v4f32)__msa_fill_w(0);
                _fsum7 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum4 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _fsum5 = (v4f32)__msa_ld_w(outptr + 12, 0);
                _fsum2 = (v4f32)__msa_ld_w(outptr + 16, 0);
                _fsum6 = (v4f32)__msa_ld_w(outptr + 20, 0);
                _fsum3 = (v4f32)__msa_ld_w(outptr + 24, 0);
                _fsum7 = (v4f32)__msa_ld_w(outptr + 28, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                v4i32 _sum4 = __msa_fill_w(0);
                v4i32 _sum5 = __msa_fill_w(0);
                v4i32 _sum6 = __msa_fill_w(0);
                v4i32 _sum7 = __msa_fill_w(0);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 64);
                    v16i8 _pA0 = __msa_ld_b(pA, 0);
                    v16i8 _pA0r = (v16i8)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                    v16i8 _pB = mips_wq_int4_load16_pair(pB);
                    v16i8 _pBr = (v16i8)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA0, _pBr), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA0r, _pB), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA0r, _pBr), _one);

                    v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    v16i8 _pA1r = (v16i8)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                    _sum4 = __msa_dpadd_s_w(_sum4, __msa_dotp_s_h(_pA1, _pB), _one);
                    _sum5 = __msa_dpadd_s_w(_sum5, __msa_dotp_s_h(_pA1, _pBr), _one);
                    _sum6 = __msa_dpadd_s_w(_sum6, __msa_dotp_s_h(_pA1r, _pB), _one);
                    _sum7 = __msa_dpadd_s_w(_sum7, __msa_dotp_s_h(_pA1r, _pBr), _one);
                    pA += 32;
                    pB += 8;
                }

                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA0 = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                    v8i16 _pA1 = (v8i16)__msa_fill_w(*(const int*)(pA + 4));
                    _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);
                    v8i16 _pA0r = __msa_shf_h(_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                    v8i16 _pA1r = __msa_shf_h(_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                    v8i16 _pB0 = (v8i16)mips_wq_int4_load4(pB);
                    _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                    v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                    v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA0, _pB0r);
                    v8i16 _s2 = __msa_mulv_h(_pA0r, _pB0);
                    v8i16 _s3 = __msa_mulv_h(_pA0r, _pB0r);
                    v8i16 _s4 = __msa_mulv_h(_pA1, _pB0);
                    v8i16 _s5 = __msa_mulv_h(_pA1, _pB0r);
                    v8i16 _s6 = __msa_mulv_h(_pA1r, _pB0);
                    v8i16 _s7 = __msa_mulv_h(_pA1r, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                    _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s4, 0), _s4));
                    _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s5, 0), _s5));
                    _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s6, 0), _s6));
                    _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s7, 0), _s7));
                    pA += 8;
                    pB += 2;
                }

                v4f32 _descaleB = (v4f32)__msa_ld_w(pB_descales, 0);
                v4f32 _descaleBr = (v4f32)__msa_shf_w((v4i32)_descaleB, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _descaleA0 = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleA1 = (v4f32)__msa_ld_w(pA_descales + 4, 0);
                v4f32 _descaleA0r = (v4f32)__msa_shf_w((v4i32)_descaleA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _descaleA1r = (v4f32)__msa_shf_w((v4i32)_descaleA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _scale = __msa_fmul_w(_descaleA0, _descaleB);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA0, _descaleBr);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleA0r, _descaleB);
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleA0r, _descaleBr);
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                _scale = __msa_fmul_w(_descaleA1, _descaleB);
                _fsum4 = __ncnn_msa_fmadd_w(_fsum4, (v4f32)__msa_ffint_s_w(_sum4), _scale);
                _scale = __msa_fmul_w(_descaleA1, _descaleBr);
                _fsum5 = __ncnn_msa_fmadd_w(_fsum5, (v4f32)__msa_ffint_s_w(_sum5), _scale);
                _scale = __msa_fmul_w(_descaleA1r, _descaleB);
                _fsum6 = __ncnn_msa_fmadd_w(_fsum6, (v4f32)__msa_ffint_s_w(_sum6), _scale);
                _scale = __msa_fmul_w(_descaleA1r, _descaleBr);
                _fsum7 = __ncnn_msa_fmadd_w(_fsum7, (v4f32)__msa_ffint_s_w(_sum7), _scale);
                pA_descales += 8;
                pB_descales += 4;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum4, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum5, outptr + 12, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 16, 0);
            __msa_st_w((v4i32)_fsum6, outptr + 20, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 24, 0);
            __msa_st_w((v4i32)_fsum7, outptr + 28, 0);
            outptr += 32;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            v4f32 _fsum2;
            v4f32 _fsum3;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
                _fsum2 = (v4f32)__msa_fill_w(0);
                _fsum3 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum2 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _fsum3 = (v4f32)__msa_ld_w(outptr + 12, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA0 = __msa_ld_b(pA, 0);
                    v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    v16i8 _pB = mips_wq_int4_load8(pB);
                    v16i8 _pBr = (v16i8)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA0, _pBr), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA1, _pB), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA1, _pBr), _one);
                    pA += 32;
                    pB += 4;
                }

                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA0 = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                    v8i16 _pA1 = (v8i16)__msa_fill_w(*(const int*)(pA + 4));
                    _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);
                    int b01 = (unsigned char)get_packed_weight_wq_int4(pB, 0) | ((unsigned char)get_packed_weight_wq_int4(pB, 1) << 8);
                    v8i16 _pB0 = (v8i16)__msa_fill_w(b01);
                    _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                    _pB0 = __msa_shf_h(_pB0, _MSA_SHUFFLE(1, 0, 1, 0));
                    v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                    v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA0, _pB0r);
                    v8i16 _s2 = __msa_mulv_h(_pA1, _pB0);
                    v8i16 _s3 = __msa_mulv_h(_pA1, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                    pA += 8;
                    pB += 1;
                }

                v4f32 _descaleA0 = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleA1 = (v4f32)__msa_ld_w(pA_descales + 4, 0);
                v4f32 _descaleB = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1));
                v4f32 _descaleBr = (v4f32)__msa_shf_w((v4i32)_descaleB, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _scale = __msa_fmul_w(_descaleA0, _descaleB);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA0, _descaleBr);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleA1, _descaleB);
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleA1, _descaleBr);
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                pA_descales += 8;
                pB_descales += 2;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            outptr += 16;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA0 = __msa_ld_b(pA, 0);
                    v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    v16i8 _pB = mips_wq_int4_load4(pB);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA1, _pB), _one);
                    pA += 32;
                    pB += 2;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_d_ptr(pA);
                    v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    v8i16 _s = __msa_mulv_h(_pA, __msa_fill_h(get_packed_weight_wq_int4(pB, 0)));
                    v8i16 _sign = __msa_clti_s_h(_s, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s));
                    _pA8 = (v16i8)__msa_fill_d_ptr(pA + 8);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    _s = __msa_mulv_h(_pA, __msa_fill_h(get_packed_weight_wq_int4(pB, 1)));
                    _sign = __msa_clti_s_h(_s, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s));
                    pA += 16;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_d_ptr(pA);
                    v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    v8i16 _s = __msa_mulv_h(_pA, __msa_fill_h(get_packed_weight_wq_int4(pB, 0)));
                    v8i16 _sign = __msa_clti_s_h(_s, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s));
                    pA += 8;
                    pB++;
                }

                v4f32 _descaleA0 = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleA1 = (v4f32)__msa_ld_w(pA_descales + 4, 0);
                v4f32 _scale = __msa_fmul_w(_descaleA0, __msa_fill_w_f32(pB_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA1, __msa_fill_w_f32(pB_descales[0]));
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                pA_descales += 8;
                pB_descales++;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }

        pAT += (size_t)8 * A_hstep;
        pAT_descales += (size_t)8 * A_descales_hstep;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        const v8i16 _one = __msa_fill_h(1);

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            v4f32 _fsum2;
            v4f32 _fsum3;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
                _fsum2 = (v4f32)__msa_fill_w(0);
                _fsum3 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _fsum2 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _fsum3 = (v4f32)__msa_ld_w(outptr + 12, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pAr = (v16i8)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    v16i8 _pB0 = mips_wq_int4_load16_pair(pB);
                    v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pAr, _pB0), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pAr, _pB0r), _one);
                    pA += 16;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    v8i16 _pAr = __msa_shf_h(_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    v8i16 _pB0 = (v8i16)mips_wq_int4_load4(pB);
                    _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                    v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                    v8i16 _s2 = __msa_mulv_h(_pAr, _pB0);
                    v8i16 _s3 = __msa_mulv_h(_pAr, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                    pA += 4;
                    pB += 2;
                }
                v4f32 _descaleB = (v4f32)__msa_ld_w(pB_descales, 0);
                v4f32 _descaleBr = (v4f32)__msa_shf_w((v4i32)_descaleB, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _descaleA = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleAr = (v4f32)__msa_shf_w((v4i32)_descaleA, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleBr);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleAr, _descaleB);
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleAr, _descaleBr);
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                pA_descales += 4;
                pB_descales += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            outptr += 16;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pB0 = mips_wq_int4_load8(pB);
                    v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                    pA += 16;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    v16i8 _pB8 = mips_wq_int4_load2(pB);
                    v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    pA += 4;
                    pB += 1;
                }
                v4f32 _descaleA = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleB = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1));
                v4f32 _descaleBr = (v4f32)__msa_shf_w((v4i32)_descaleB, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleBr);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                pA_descales += 4;
                pB_descales += 2;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pB0 = mips_wq_int4_load4(pB);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    pA += 16;
                    pB += 2;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    v8i16 _pB0 = __msa_fill_h(get_packed_weight_wq_int4(pB, 0));
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _pA = (v8i16)__msa_fill_w(*(const int*)(pA + 4));
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    _pB0 = __msa_fill_h(get_packed_weight_wq_int4(pB, 1));
                    _s0 = __msa_mulv_h(_pA, _pB0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    pA += 8;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    v8i16 _pB0 = __msa_fill_h(get_packed_weight_wq_int4(pB, 0));
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    pA += 4;
                    pB++;
                }
                v4f32 _descaleA = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _scale = __msa_fmul_w(_descaleA, __msa_fill_w_f32(pB_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                pA_descales += 4;
                pB_descales++;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            outptr += 4;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }

        pAT += (size_t)4 * A_hstep;
        pAT_descales += (size_t)4 * A_descales_hstep;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __mips_msa
        const v8i16 _one = __msa_fill_h(1);
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    v16i8 _pB0 = mips_wq_int4_load16_pair(pB);
                    v16i8 _pB01 = (v16i8)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                    v16i8 _pB23 = (v16i8)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB01), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB23), _one);
                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    _sum0 = __msa_addv_w(_sum0, __msa_set_w(pA[0] * get_packed_weight_wq_int4(pB, 0), pA[1] * get_packed_weight_wq_int4(pB, 0), pA[0] * get_packed_weight_wq_int4(pB, 1), pA[1] * get_packed_weight_wq_int4(pB, 1)));
                    _sum1 = __msa_addv_w(_sum1, __msa_set_w(pA[0] * get_packed_weight_wq_int4(pB, 2), pA[1] * get_packed_weight_wq_int4(pB, 2), pA[0] * get_packed_weight_wq_int4(pB, 3), pA[1] * get_packed_weight_wq_int4(pB, 3)));
                    pA += 2;
                    pB += 2;
                }
                v4f32 _descaleA = (v4f32)__msa_fill_d_ptr(pA_descales);
                v4f32 _descaleB0 = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales + 1));
                v4f32 _descaleB1 = (v4f32)__msa_set_w(__msa_load_w(pB_descales + 2), __msa_load_w(pB_descales + 2), __msa_load_w(pB_descales + 3), __msa_load_w(pB_descales + 3));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB0);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleB1);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                pA_descales += 2;
                pB_descales += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
#if __mips_msa
            v4f32 _fsum;
            if (k == 0)
            {
                _fsum = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum = (v4f32)__msa_ld_w(outptr, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    v16i8 _pB = mips_wq_int4_load8(pB);
                    v16i8 _pB01 = (v16i8)__msa_ilvr_w((v4i32)_pB, (v4i32)_pB);
                    _sum = __msa_dpadd_s_w(_sum, __msa_dotp_s_h(_pA, _pB01), _one);
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    _sum = __msa_addv_w(_sum, __msa_set_w(pA[0] * get_packed_weight_wq_int4(pB, 0), pA[1] * get_packed_weight_wq_int4(pB, 0), pA[0] * get_packed_weight_wq_int4(pB, 1), pA[1] * get_packed_weight_wq_int4(pB, 1)));
                    pA += 2;
                    pB += 1;
                }
                v4f32 _descaleA = (v4f32)__msa_fill_d_ptr(pA_descales);
                v4f32 _descaleB = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales + 1));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB);
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales += 2;
                pB_descales += 2;
            }
            __msa_st_w((v4i32)_fsum, outptr, 0);
#else
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
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum00_i = 0;
                int sum01_i = 0;
                int sum10_i = 0;
                int sum11_i = 0;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum00_i += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    sum01_i += pA[4] * get_packed_weight_wq_int4(pB, 0) + pA[5] * get_packed_weight_wq_int4(pB, 1) + pA[6] * get_packed_weight_wq_int4(pB, 2) + pA[7] * get_packed_weight_wq_int4(pB, 3);
                    sum10_i += pA[0] * get_packed_weight_wq_int4(pB, 4) + pA[1] * get_packed_weight_wq_int4(pB, 5) + pA[2] * get_packed_weight_wq_int4(pB, 6) + pA[3] * get_packed_weight_wq_int4(pB, 7);
                    sum11_i += pA[4] * get_packed_weight_wq_int4(pB, 4) + pA[5] * get_packed_weight_wq_int4(pB, 5) + pA[6] * get_packed_weight_wq_int4(pB, 6) + pA[7] * get_packed_weight_wq_int4(pB, 7);
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum00_i += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    sum01_i += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    sum10_i += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    sum11_i += pA[1] * get_packed_weight_wq_int4(pB, 1);
                    pA += 2;
                    pB += 1;
                }
                sum00 += sum00_i * pA_descales[0] * pB_descales[0];
                sum01 += sum01_i * pA_descales[1] * pB_descales[0];
                sum10 += sum10_i * pA_descales[0] * pB_descales[1];
                sum11 += sum11_i * pA_descales[1] * pB_descales[1];
                pA_descales += 2;
                pB_descales += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
#endif // __mips_msa
            outptr += 4;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float sum0;
            float sum1;
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

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum0_i = 0;
                int sum1_i = 0;
                int kk = 0;
#if __mips_msa
                v4i32 _sum = __msa_fill_w(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pB = mips_wq_int4_load8(pB);
                    _pB = (v16i8)__msa_ilvr_w((v4i32)_pB, (v4i32)_pB);
                    _sum = __msa_dpadd_s_w(_sum, __msa_dotp_s_h(_pA, _pB), _one);
                    pA += 16;
                    pB += 4;
                }
                sum0_i = __msa_copy_s_w(_sum, 0) + __msa_copy_s_w(_sum, 2);
                sum1_i = __msa_copy_s_w(_sum, 1) + __msa_copy_s_w(_sum, 3);
#endif // __mips_msa
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0_i += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    sum1_i += pA[4] * get_packed_weight_wq_int4(pB, 0) + pA[5] * get_packed_weight_wq_int4(pB, 1) + pA[6] * get_packed_weight_wq_int4(pB, 2) + pA[7] * get_packed_weight_wq_int4(pB, 3);
                    pA += 8;
                    pB += 2;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    sum0_i += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    sum1_i += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    sum0_i += pA[2] * get_packed_weight_wq_int4(pB, 1);
                    sum1_i += pA[3] * get_packed_weight_wq_int4(pB, 1);
                    pA += 4;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0_i += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    sum1_i += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    pA += 2;
                    pB++;
                }
                const float bscale = *pB_descales++;
                sum0 += sum0_i * pA_descales[0] * bscale;
                sum1 += sum1_i * pA_descales[1] * bscale;
                pA_descales += 2;
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }

        pAT += (size_t)2 * A_hstep;
        pAT_descales += (size_t)2 * A_descales_hstep;
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __mips_msa
        const v8i16 _one = __msa_fill_h(1);
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                int kk = 0;
                {
                    v4i32 _sum1 = __msa_fill_w(0);
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __builtin_prefetch(pA + 32);
                        __builtin_prefetch(pB + 64);
                        v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                        v16i8 _pB0 = mips_wq_int4_load16_pair(pB);
                        _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);

                        _pA = (v16i8)__msa_fill_w(*(const int*)(pA + 4));
                        _pB0 = mips_wq_int4_load16_pair(pB + 8);
                        _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0), _one);
                        pA += 8;
                        pB += 16;
                    }
                    _sum0 = __msa_addv_w(_sum0, _sum1);
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                    v16i8 _pB0 = mips_wq_int4_load16_pair(pB);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    _sum0 = __msa_addv_w(_sum0, __msa_set_w(pA[0] * get_packed_weight_wq_int4(pB, 0), pA[0] * get_packed_weight_wq_int4(pB, 1), pA[0] * get_packed_weight_wq_int4(pB, 2), pA[0] * get_packed_weight_wq_int4(pB, 3)));
                    pA++;
                    pB += 2;
                }
                v4f32 _descaleB = (v4f32)__msa_ld_w(pB_descales, 0);
                v4f32 _scale = __msa_fmul_w(_descaleB, __msa_fill_w_f32(pA_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                pA_descales++;
                pB_descales += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            outptr += 4;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float sum0;
            float sum1;
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

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum0_i = 0;
                int sum1_i = 0;
                int kk = 0;
#if __mips_msa
                v4i32 _sum = __msa_fill_w(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __builtin_prefetch(pA + 16);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    _pA = (v16i8)__msa_ilvr_w((v4i32)_pA, (v4i32)_pA);
                    v16i8 _pB = mips_wq_int4_load16(pB);
                    _sum = __msa_dpadd_s_w(_sum, __msa_dotp_s_h(_pA, _pB), _one);
                    pA += 8;
                    pB += 8;
                }
                sum0_i = __msa_copy_s_w(_sum, 0) + __msa_copy_s_w(_sum, 2);
                sum1_i = __msa_copy_s_w(_sum, 1) + __msa_copy_s_w(_sum, 3);
#endif // __mips_msa
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0_i += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    sum1_i += pA[0] * get_packed_weight_wq_int4(pB, 4) + pA[1] * get_packed_weight_wq_int4(pB, 5) + pA[2] * get_packed_weight_wq_int4(pB, 6) + pA[3] * get_packed_weight_wq_int4(pB, 7);
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0_i += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    sum1_i += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    pA++;
                    pB += 1;
                }
                const float ascale = *pA_descales++;
                sum0 += sum0_i * ascale * pB_descales[0];
                sum1 += sum1_i * ascale * pB_descales[1];
                pB_descales += 2;
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float sum0;
            if (k == 0)
            {
                sum0 = 0.f;
            }
            else
            {
                sum0 = outptr[0];
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum0_i = 0;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0_i += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    pA += 4;
                    pB += 2;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    sum0_i += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1);
                    pA += 2;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0_i += *pA++ * get_packed_weight_wq_int4(pB, 0);
                    pB++;
                }
                sum0 += sum0_i * pA_descales[0] * pB_descales[0];
                pA_descales++;
                pB_descales++;
            }

            *outptr++ = sum0;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void get_optimal_tile_mnk_wq_int4(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const int l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    {
#if __mips_msa
        int tile_size = (int)((l2_cache_size_int8 - 8 * 4 * sizeof(float)) / (8.f + 4 * 0.5f + (8.f + 4.f) * sizeof(float) / block_size));
#else
        int tile_size = (int)((l2_cache_size_int8 - 2 * 2 * sizeof(float)) / (2.f + 2 * 0.5f + (2.f + 2.f) * sizeof(float) / block_size));
#endif
        TILE_K = std::max(block_size, tile_size / block_size * block_size);

        if (K > 0)
        {
            int nn_K = (K + TILE_K - 1) / TILE_K;
            TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + block_size - 1) / block_size * block_size);
            if (TILE_K >= K)
                TILE_K = K;
        }
    }

#if __mips_msa
    TILE_M = 8;
#else
    TILE_M = 2;
#endif
    if (M > 0)
    {
        TILE_M *= std::min(nT, get_physical_cpu_count());
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __mips_msa
        TILE_M = std::max(8, std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8));
#else
        TILE_M = std::max(2, std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2));
#endif
        if (nT > 1)
        {
#if __mips_msa
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        const int tile_block_count = (TILE_K + block_size - 1) / block_size;
        const float fixed_footprint = TILE_M * TILE_K + TILE_M * tile_block_count * sizeof(float);
        const float n_footprint = TILE_K * 0.5f + tile_block_count * sizeof(float) + TILE_M * sizeof(float);
        int tile_size = (int)((l2_cache_size_int8 - fixed_footprint) / std::max(1.f, n_footprint));
#if __mips_msa
        TILE_N = std::max(4, tile_size / 4 * 4);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(4, std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4));
#else
        TILE_N = std::max(2, tile_size / 2 * 2);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(2, std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2));
#endif
    }
    else
    {
#if __mips_msa
        TILE_N = 4;
#else
        TILE_N = 2;
#endif
    }

    if (constant_TILE_M > 0)
    {
#if __mips_msa
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_N > 0)
    {
#if __mips_msa
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = (constant_TILE_N + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, (constant_TILE_K + block_size - 1) / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(TILE_K, K);
    }
}
