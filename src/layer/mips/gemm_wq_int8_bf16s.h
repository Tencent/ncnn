// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if __mips_msa
    const int elempack = A.elempack;
#endif // __mips_msa
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int local_block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __mips_msa
        for (; ii + 7 < max_ii; ii += 8)
        {
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0a);
                        v4f32 _p1 = bfloat2float_msa(p0a + 4);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                        p0a += 8;
                    }

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const v4f32 _zero = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax03 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                    v4f32 _absmax47 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                    v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax03);
                    v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax47);

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), _scale03);
                        v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _scale03);
                        v4f32 _p2 = __msa_fmul_w(bfloat2float_msa(p0 + 16), _scale03);
                        v4f32 _p3 = __msa_fmul_w(bfloat2float_msa(p0 + 24), _scale03);
                        v4f32 _p4 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _scale47);
                        v4f32 _p5 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _scale47);
                        v4f32 _p6 = __msa_fmul_w(bfloat2float_msa(p0 + 20), _scale47);
                        v4f32 _p7 = __msa_fmul_w(bfloat2float_msa(p0 + 28), _scale47);

                        v16i8 _q0 = float2int8(_p0);
                        v16i8 _q1 = float2int8(_p1);
                        v16i8 _q2 = float2int8(_p2);
                        v16i8 _q3 = float2int8(_p3);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __msa_st_b(_q0, pp, 0);
                        _q0 = float2int8(_p4);
                        _q1 = float2int8(_p5);
                        _q2 = float2int8(_p6);
                        _q3 = float2int8(_p7);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __msa_st_b(_q0, pp + 16, 0);
                        pp += 32;
                        p0 += 32;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0);
                        v4f32 _p1 = bfloat2float_msa(p0 + 4);
                        _p0 = __msa_fmul_w(_p0, _scale03);
                        _p1 = __msa_fmul_w(_p1, _scale47);
                        v16i8 _q0 = float2int8(_p0);
                        v16i8 _q1 = float2int8(_p1);
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                        pp += 8;
                        p0 += 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 4;
                const unsigned short* p1 = p0 + A_hstep * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    const unsigned short* p1a = p1;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0a);
                        v4f32 _p1 = bfloat2float_msa(p1a);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                    }

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const v4f32 _zero = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax03 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                    v4f32 _absmax47 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                    v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax03);
                    v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax47);

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), _scale03);
                        v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _scale03);
                        v4f32 _p2 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _scale03);
                        v4f32 _p3 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _scale03);
                        v4f32 _p4 = __msa_fmul_w(bfloat2float_msa(p1), _scale47);
                        v4f32 _p5 = __msa_fmul_w(bfloat2float_msa(p1 + 4), _scale47);
                        v4f32 _p6 = __msa_fmul_w(bfloat2float_msa(p1 + 8), _scale47);
                        v4f32 _p7 = __msa_fmul_w(bfloat2float_msa(p1 + 12), _scale47);

                        v16i8 _q0 = float2int8(_p0);
                        v16i8 _q1 = float2int8(_p1);
                        v16i8 _q2 = float2int8(_p2);
                        v16i8 _q3 = float2int8(_p3);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __msa_st_b(_q0, pp, 0);
                        _q0 = float2int8(_p4);
                        _q1 = float2int8(_p5);
                        _q2 = float2int8(_p6);
                        _q3 = float2int8(_p7);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __msa_st_b(_q0, pp + 16, 0);
                        pp += 32;
                        p0 += 16;
                        p1 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0);
                        v4f32 _p1 = bfloat2float_msa(p1);
                        _p0 = __msa_fmul_w(_p0, _scale03);
                        _p1 = __msa_fmul_w(_p1, _scale47);
                        v16i8 _q0 = float2int8(_p0);
                        v16i8 _q1 = float2int8(_p1);
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                        pp += 8;
                        p0 += 4;
                        p1 += 4;
                    }
                }
            }
            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
                const unsigned short* p1 = p0 + A_hstep;
                const unsigned short* p2 = p1 + A_hstep;
                const unsigned short* p3 = p2 + A_hstep;
                const unsigned short* p4 = p3 + A_hstep;
                const unsigned short* p5 = p4 + A_hstep;
                const unsigned short* p6 = p5 + A_hstep;
                const unsigned short* p7 = p6 + A_hstep;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    const unsigned short* p1a = p1;
                    const unsigned short* p2a = p2;
                    const unsigned short* p3a = p3;
                    const unsigned short* p4a = p4;
                    const unsigned short* p5a = p5;
                    const unsigned short* p6a = p6;
                    const unsigned short* p7a = p7;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0a);
                        v4f32 _p1 = bfloat2float_msa(p1a);
                        v4f32 _p2 = bfloat2float_msa(p2a);
                        v4f32 _p3 = bfloat2float_msa(p3a);
                        v4f32 _p4 = bfloat2float_msa(p4a);
                        v4f32 _p5 = bfloat2float_msa(p5a);
                        v4f32 _p6 = bfloat2float_msa(p6a);
                        v4f32 _p7 = bfloat2float_msa(p7a);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                        _absmax2 = __msa_fmax_w(_absmax2, (v4f32)__msa_and_v((v16u8)_p2, _abs_mask));
                        _absmax3 = __msa_fmax_w(_absmax3, (v4f32)__msa_and_v((v16u8)_p3, _abs_mask));
                        _absmax4 = __msa_fmax_w(_absmax4, (v4f32)__msa_and_v((v16u8)_p4, _abs_mask));
                        _absmax5 = __msa_fmax_w(_absmax5, (v4f32)__msa_and_v((v16u8)_p5, _abs_mask));
                        _absmax6 = __msa_fmax_w(_absmax6, (v4f32)__msa_and_v((v16u8)_p6, _abs_mask));
                        _absmax7 = __msa_fmax_w(_absmax7, (v4f32)__msa_and_v((v16u8)_p7, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                        p2a += 4;
                        p3a += 4;
                        p4a += 4;
                        p5a += 4;
                        p6a += 4;
                        p7a += 4;
                    }

                    transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                    transpose4x4_ps(_absmax4, _absmax5, _absmax6, _absmax7);
                    _absmax0 = __msa_fmax_w(__msa_fmax_w(_absmax0, _absmax1), __msa_fmax_w(_absmax2, _absmax3));
                    _absmax1 = __msa_fmax_w(__msa_fmax_w(_absmax4, _absmax5), __msa_fmax_w(_absmax6, _absmax7));

                    for (; kk < max_kk0; kk++)
                    {
                        v8i16 _p0 = (v8i16)__msa_fill_w(0);
                        _p0 = __msa_insert_h(_p0, 0, p0a[0]);
                        _p0 = __msa_insert_h(_p0, 1, p1a[0]);
                        _p0 = __msa_insert_h(_p0, 2, p2a[0]);
                        _p0 = __msa_insert_h(_p0, 3, p3a[0]);
                        v8i16 _p1 = (v8i16)__msa_fill_w(0);
                        _p1 = __msa_insert_h(_p1, 0, p4a[0]);
                        _p1 = __msa_insert_h(_p1, 1, p5a[0]);
                        _p1 = __msa_insert_h(_p1, 2, p6a[0]);
                        _p1 = __msa_insert_h(_p1, 3, p7a[0]);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)bfloat2float_msa((v4i32)_p0), _abs_mask));
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)bfloat2float_msa((v4i32)_p1), _abs_mask));
                        p0a++;
                        p1a++;
                        p2a++;
                        p3a++;
                        p4a++;
                        p5a++;
                        p6a++;
                        p7a++;
                    }

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const v4f32 _zero = (v4f32)__msa_fill_w(0);
                    _absmax0 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                    _absmax1 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                    v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax0);
                    v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax1);

                    v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale03, 0);
                    v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale03, 1);
                    v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale03, 2);
                    v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale03, 3);
                    v4f32 _scale4 = (v4f32)__msa_splati_w((v4i32)_scale47, 0);
                    v4f32 _scale5 = (v4f32)__msa_splati_w((v4i32)_scale47, 1);
                    v4f32 _scale6 = (v4f32)__msa_splati_w((v4i32)_scale47, 2);
                    v4f32 _scale7 = (v4f32)__msa_splati_w((v4i32)_scale47, 3);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0);
                        v4f32 _p1 = bfloat2float_msa(p1);
                        v4f32 _p2 = bfloat2float_msa(p2);
                        v4f32 _p3 = bfloat2float_msa(p3);
                        v4f32 _p4 = bfloat2float_msa(p4);
                        v4f32 _p5 = bfloat2float_msa(p5);
                        v4f32 _p6 = bfloat2float_msa(p6);
                        v4f32 _p7 = bfloat2float_msa(p7);
                        _p0 = __msa_fmul_w(_p0, _scale0);
                        _p1 = __msa_fmul_w(_p1, _scale1);
                        _p2 = __msa_fmul_w(_p2, _scale2);
                        _p3 = __msa_fmul_w(_p3, _scale3);
                        _p4 = __msa_fmul_w(_p4, _scale4);
                        _p5 = __msa_fmul_w(_p5, _scale5);
                        _p6 = __msa_fmul_w(_p6, _scale6);
                        _p7 = __msa_fmul_w(_p7, _scale7);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                        ((int64_t*)pp)[3] = float2int8(_p6, _p7);
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
                    for (; kk < max_kk0; kk++)
                    {
                        v8i16 _p0 = (v8i16)__msa_fill_w(0);
                        _p0 = __msa_insert_h(_p0, 0, p0[0]);
                        _p0 = __msa_insert_h(_p0, 1, p1[0]);
                        _p0 = __msa_insert_h(_p0, 2, p2[0]);
                        _p0 = __msa_insert_h(_p0, 3, p3[0]);
                        v8i16 _p1 = (v8i16)__msa_fill_w(0);
                        _p1 = __msa_insert_h(_p1, 0, p4[0]);
                        _p1 = __msa_insert_h(_p1, 1, p5[0]);
                        _p1 = __msa_insert_h(_p1, 2, p6[0]);
                        _p1 = __msa_insert_h(_p1, 3, p7[0]);
                        v4f32 _f0 = __msa_fmul_w(bfloat2float_msa((v4i32)_p0), _scale03);
                        v4f32 _f1 = __msa_fmul_w(bfloat2float_msa((v4i32)_p1), _scale47);
                        ((int64_t*)pp)[0] = float2int8(_f0, _f1);
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
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * elempack;
            const unsigned short* p1 = 0;
            const unsigned short* p2 = 0;
            const unsigned short* p3 = 0;
            if (elempack == 1)
            {
                p1 = p0 + A_hstep;
                p2 = p1 + A_hstep;
                p3 = p2 + A_hstep;
            }

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax3 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const unsigned short* p2a = p2;
                const unsigned short* p3a = p3;
                int kk = 0;

                if (elempack == 4)
                {
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p = bfloat2float_msa(p0a);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p, _abs_mask));
                        p0a += 4;
                    }
                }

                if (elempack == 1)
                {
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0a);
                        v4f32 _p1 = bfloat2float_msa(p1a);
                        v4f32 _p2 = bfloat2float_msa(p2a);
                        v4f32 _p3 = bfloat2float_msa(p3a);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                        _absmax2 = __msa_fmax_w(_absmax2, (v4f32)__msa_and_v((v16u8)_p2, _abs_mask));
                        _absmax3 = __msa_fmax_w(_absmax3, (v4f32)__msa_and_v((v16u8)_p3, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                        p2a += 4;
                        p3a += 4;
                    }
                    transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                    _absmax0 = __msa_fmax_w(__msa_fmax_w(_absmax0, _absmax1), __msa_fmax_w(_absmax2, _absmax3));

                    for (; kk < max_kk0; kk++)
                    {
                        v8i16 _p = (v8i16)__msa_fill_w(0);
                        _p = __msa_insert_h(_p, 0, p0a[0]);
                        _p = __msa_insert_h(_p, 1, p1a[0]);
                        _p = __msa_insert_h(_p, 2, p2a[0]);
                        _p = __msa_insert_h(_p, 3, p3a[0]);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)bfloat2float_msa((v4i32)_p), _abs_mask));
                        p0a++;
                        p1a++;
                        p2a++;
                        p3a++;
                    }
                }

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                pd += 4;

                _absmax0 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, (v4f32)__msa_fill_w(0)), (v16u8)_absmax0, (v16u8)_v127);
                v4f32 _scale = __msa_fdiv_w(_v127, _absmax0);

                if (elempack == 4)
                {
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), _scale);
                        v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _scale);
                        v4f32 _p2 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _scale);
                        v4f32 _p3 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _scale);

                        v16i8 _q0 = float2int8(_p0);
                        v16i8 _q1 = float2int8(_p1);
                        v16i8 _q2 = float2int8(_p2);
                        v16i8 _q3 = float2int8(_p3);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __msa_st_b(_q0, pp, 0);
                        pp += 16;
                        p0 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p = __msa_fmul_w(bfloat2float_msa(p0), _scale);
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                        pp += 4;
                        p0 += 4;
                    }
                }

                if (elempack == 1)
                {
                    v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale, 0);
                    v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale, 1);
                    v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale, 2);
                    v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale, 3);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0);
                        v4f32 _p1 = bfloat2float_msa(p1);
                        v4f32 _p2 = bfloat2float_msa(p2);
                        v4f32 _p3 = bfloat2float_msa(p3);
                        _p0 = __msa_fmul_w(_p0, _scale0);
                        _p1 = __msa_fmul_w(_p1, _scale1);
                        _p2 = __msa_fmul_w(_p2, _scale2);
                        _p3 = __msa_fmul_w(_p3, _scale3);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        pp += 16;
                        p0 += 4;
                        p1 += 4;
                        p2 += 4;
                        p3 += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        v8i16 _p = (v8i16)__msa_fill_w(0);
                        _p = __msa_insert_h(_p, 0, p0[0]);
                        _p = __msa_insert_h(_p, 1, p1[0]);
                        _p = __msa_insert_h(_p, 2, p2[0]);
                        _p = __msa_insert_h(_p, 3, p3[0]);
                        v4f32 _f = __msa_fmul_w(bfloat2float_msa((v4i32)_p), _scale);
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_f), 0);
                        pp += 4;
                        p0++;
                        p1++;
                        p2++;
                        p3++;
                    }
                }
            }
        }
#endif // __mips_msa
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0a++);
                    float v1 = bfloat16_to_float32(*p1a++);
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v00 = bfloat16_to_float32(p0[0]);
                    float v01 = bfloat16_to_float32(p0[1]);
                    float v02 = bfloat16_to_float32(p0[2]);
                    float v03 = bfloat16_to_float32(p0[3]);
                    float v10 = bfloat16_to_float32(p1[0]);
                    float v11 = bfloat16_to_float32(p1[1]);
                    float v12 = bfloat16_to_float32(p1[2]);
                    float v13 = bfloat16_to_float32(p1[3]);
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v02 * scale0);
                    pp[3] = float2int8(v03 * scale0);
                    pp[4] = float2int8(v10 * scale1);
                    pp[5] = float2int8(v11 * scale1);
                    pp[6] = float2int8(v12 * scale1);
                    pp[7] = float2int8(v13 * scale1);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0++);
                    float v1 = bfloat16_to_float32(*p1++);
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0a++);
                    absmax0 = std::max(absmax0, fabsf(v0));
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                *pd++ = absmax0 / 127.f;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[1]);
                    float v2 = bfloat16_to_float32(p0[2]);
                    float v3 = bfloat16_to_float32(p0[3]);
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    pp[2] = float2int8(v2 * scale0);
                    pp[3] = float2int8(v3 * scale0);
                    pp += 4;
                    p0 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0++);
                    *pp++ = float2int8(v0 * scale0);
                }
            }
        }
        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 8;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    v4f32 _s = __msa_fill_w_f32(*psa++);
                    v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0a), _s);
                    v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p0a + 4), _s);
                    _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                    _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                    p0a += 8;
                }

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                pd += 8;

                const v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _absmax03 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                v4f32 _absmax47 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax03);
                v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax47);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(ps[0])), _scale03);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 8), __msa_fill_w_f32(ps[1])), _scale03);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 16), __msa_fill_w_f32(ps[2])), _scale03);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 24), __msa_fill_w_f32(ps[3])), _scale03);
                    v4f32 _p4 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 4), __msa_fill_w_f32(ps[0])), _scale47);
                    v4f32 _p5 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 12), __msa_fill_w_f32(ps[1])), _scale47);
                    v4f32 _p6 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 20), __msa_fill_w_f32(ps[2])), _scale47);
                    v4f32 _p7 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 28), __msa_fill_w_f32(ps[3])), _scale47);

                    v16i8 _q0 = float2int8(_p0);
                    v16i8 _q1 = float2int8(_p1);
                    v16i8 _q2 = float2int8(_p2);
                    v16i8 _q3 = float2int8(_p3);
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __msa_st_b(_q0, pp, 0);
                    _q0 = float2int8(_p4);
                    _q1 = float2int8(_p5);
                    _q2 = float2int8(_p6);
                    _q3 = float2int8(_p7);
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __msa_st_b(_q0, pp + 16, 0);
                    pp += 32;
                    p0 += 32;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _s = __msa_fill_w_f32(*ps++);
                    v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), _s);
                    v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _s);
                    _p0 = __msa_fmul_w(_p0, _scale03);
                    _p1 = __msa_fmul_w(_p1, _scale47);
                    v16i8 _q0 = float2int8(_p0);
                    v16i8 _q1 = float2int8(_p1);
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                    pp += 8;
                    p0 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = p0 + A_hstep * 4;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    v4f32 _s = __msa_fill_w_f32(*psa++);
                    v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0a), _s);
                    v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p1a), _s);
                    _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                    _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                    p0a += 4;
                    p1a += 4;
                }

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                pd += 8;

                const v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _absmax03 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                v4f32 _absmax47 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax03);
                v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax47);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(ps[0])), _scale03);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 4), __msa_fill_w_f32(ps[1])), _scale03);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 8), __msa_fill_w_f32(ps[2])), _scale03);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 12), __msa_fill_w_f32(ps[3])), _scale03);
                    v4f32 _p4 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p1), __msa_fill_w_f32(ps[0])), _scale47);
                    v4f32 _p5 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p1 + 4), __msa_fill_w_f32(ps[1])), _scale47);
                    v4f32 _p6 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p1 + 8), __msa_fill_w_f32(ps[2])), _scale47);
                    v4f32 _p7 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p1 + 12), __msa_fill_w_f32(ps[3])), _scale47);

                    v16i8 _q0 = float2int8(_p0);
                    v16i8 _q1 = float2int8(_p1);
                    v16i8 _q2 = float2int8(_p2);
                    v16i8 _q3 = float2int8(_p3);
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __msa_st_b(_q0, pp, 0);
                    _q0 = float2int8(_p4);
                    _q1 = float2int8(_p5);
                    _q2 = float2int8(_p6);
                    _q3 = float2int8(_p7);
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __msa_st_b(_q0, pp + 16, 0);
                    pp += 32;
                    p0 += 16;
                    p1 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _s = __msa_fill_w_f32(*ps++);
                    v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), _s);
                    v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p1), _s);
                    _p0 = __msa_fmul_w(_p0, _scale03);
                    _p1 = __msa_fmul_w(_p1, _scale47);
                    v16i8 _q0 = float2int8(_p0);
                    v16i8 _q1 = float2int8(_p1);
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;
            const unsigned short* p2 = p1 + A_hstep;
            const unsigned short* p3 = p2 + A_hstep;
            const unsigned short* p4 = p3 + A_hstep;
            const unsigned short* p5 = p4 + A_hstep;
            const unsigned short* p6 = p5 + A_hstep;
            const unsigned short* p7 = p6 + A_hstep;

            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const unsigned short* p2a = p2;
                const unsigned short* p3a = p3;
                const unsigned short* p4a = p4;
                const unsigned short* p5a = p5;
                const unsigned short* p6a = p6;
                const unsigned short* p7a = p7;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = bfloat2float_msa(p0a);
                    v4f32 _p1 = bfloat2float_msa(p1a);
                    v4f32 _p2 = bfloat2float_msa(p2a);
                    v4f32 _p3 = bfloat2float_msa(p3a);
                    v4f32 _p4 = bfloat2float_msa(p4a);
                    v4f32 _p5 = bfloat2float_msa(p5a);
                    v4f32 _p6 = bfloat2float_msa(p6a);
                    v4f32 _p7 = bfloat2float_msa(p7a);
                    v4f32 _s = (v4f32)__msa_ld_w(psa, 0);
                    _p0 = (v4f32)__msa_and_v((v16u8)_p0, _abs_mask);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = (v4f32)__msa_and_v((v16u8)_p1, _abs_mask);
                    _p1 = __msa_fmul_w(_p1, _s);
                    _p2 = (v4f32)__msa_and_v((v16u8)_p2, _abs_mask);
                    _p2 = __msa_fmul_w(_p2, _s);
                    _p3 = (v4f32)__msa_and_v((v16u8)_p3, _abs_mask);
                    _p3 = __msa_fmul_w(_p3, _s);
                    _p4 = (v4f32)__msa_and_v((v16u8)_p4, _abs_mask);
                    _p4 = __msa_fmul_w(_p4, _s);
                    _p5 = (v4f32)__msa_and_v((v16u8)_p5, _abs_mask);
                    _p5 = __msa_fmul_w(_p5, _s);
                    _p6 = (v4f32)__msa_and_v((v16u8)_p6, _abs_mask);
                    _p6 = __msa_fmul_w(_p6, _s);
                    _p7 = (v4f32)__msa_and_v((v16u8)_p7, _abs_mask);
                    _p7 = __msa_fmul_w(_p7, _s);
                    _absmax0 = __msa_fmax_w(_absmax0, _p0);
                    _absmax1 = __msa_fmax_w(_absmax1, _p1);
                    _absmax2 = __msa_fmax_w(_absmax2, _p2);
                    _absmax3 = __msa_fmax_w(_absmax3, _p3);
                    _absmax4 = __msa_fmax_w(_absmax4, _p4);
                    _absmax5 = __msa_fmax_w(_absmax5, _p5);
                    _absmax6 = __msa_fmax_w(_absmax6, _p6);
                    _absmax7 = __msa_fmax_w(_absmax7, _p7);
                    p0a += 4;
                    p1a += 4;
                    p2a += 4;
                    p3a += 4;
                    p4a += 4;
                    p5a += 4;
                    p6a += 4;
                    p7a += 4;
                    psa += 4;
                }

                transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                transpose4x4_ps(_absmax4, _absmax5, _absmax6, _absmax7);
                _absmax0 = __msa_fmax_w(__msa_fmax_w(_absmax0, _absmax1), __msa_fmax_w(_absmax2, _absmax3));
                _absmax1 = __msa_fmax_w(__msa_fmax_w(_absmax4, _absmax5), __msa_fmax_w(_absmax6, _absmax7));

                for (; kk < max_kk0; kk++)
                {
                    v8i16 _p0 = (v8i16)__msa_fill_w(0);
                    _p0 = __msa_insert_h(_p0, 0, p0a[0]);
                    _p0 = __msa_insert_h(_p0, 1, p1a[0]);
                    _p0 = __msa_insert_h(_p0, 2, p2a[0]);
                    _p0 = __msa_insert_h(_p0, 3, p3a[0]);
                    v8i16 _p1 = (v8i16)__msa_fill_w(0);
                    _p1 = __msa_insert_h(_p1, 0, p4a[0]);
                    _p1 = __msa_insert_h(_p1, 1, p5a[0]);
                    _p1 = __msa_insert_h(_p1, 2, p6a[0]);
                    _p1 = __msa_insert_h(_p1, 3, p7a[0]);
                    v4f32 _s = __msa_fill_w_f32(*psa++);
                    v4f32 _f0 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)bfloat2float_msa((v4i32)_p0), _abs_mask), _s);
                    v4f32 _f1 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)bfloat2float_msa((v4i32)_p1), _abs_mask), _s);
                    _absmax0 = __msa_fmax_w(_absmax0, _f0);
                    _absmax1 = __msa_fmax_w(_absmax1, _f1);
                    p0a++;
                    p1a++;
                    p2a++;
                    p3a++;
                    p4a++;
                    p5a++;
                    p6a++;
                    p7a++;
                }

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                pd += 8;

                const v4f32 _zero = (v4f32)__msa_fill_w(0);
                _absmax0 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                _absmax1 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax0);
                v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax1);

                v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale03, 0);
                v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale03, 1);
                v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale03, 2);
                v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale03, 3);
                v4f32 _scale4 = (v4f32)__msa_splati_w((v4i32)_scale47, 0);
                v4f32 _scale5 = (v4f32)__msa_splati_w((v4i32)_scale47, 1);
                v4f32 _scale6 = (v4f32)__msa_splati_w((v4i32)_scale47, 2);
                v4f32 _scale7 = (v4f32)__msa_splati_w((v4i32)_scale47, 3);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = bfloat2float_msa(p0);
                    v4f32 _p1 = bfloat2float_msa(p1);
                    v4f32 _p2 = bfloat2float_msa(p2);
                    v4f32 _p3 = bfloat2float_msa(p3);
                    v4f32 _p4 = bfloat2float_msa(p4);
                    v4f32 _p5 = bfloat2float_msa(p5);
                    v4f32 _p6 = bfloat2float_msa(p6);
                    v4f32 _p7 = bfloat2float_msa(p7);
                    v4f32 _s = (v4f32)__msa_ld_w(ps, 0);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = __msa_fmul_w(_p1, _s);
                    _p2 = __msa_fmul_w(_p2, _s);
                    _p3 = __msa_fmul_w(_p3, _s);
                    _p4 = __msa_fmul_w(_p4, _s);
                    _p5 = __msa_fmul_w(_p5, _s);
                    _p6 = __msa_fmul_w(_p6, _s);
                    _p7 = __msa_fmul_w(_p7, _s);
                    _p0 = __msa_fmul_w(_p0, _scale0);
                    _p1 = __msa_fmul_w(_p1, _scale1);
                    _p2 = __msa_fmul_w(_p2, _scale2);
                    _p3 = __msa_fmul_w(_p3, _scale3);
                    _p4 = __msa_fmul_w(_p4, _scale4);
                    _p5 = __msa_fmul_w(_p5, _scale5);
                    _p6 = __msa_fmul_w(_p6, _scale6);
                    _p7 = __msa_fmul_w(_p7, _scale7);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                    ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                    pp += 32;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                    p4 += 4;
                    p5 += 4;
                    p6 += 4;
                    p7 += 4;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v8i16 _p0 = (v8i16)__msa_fill_w(0);
                    _p0 = __msa_insert_h(_p0, 0, p0[0]);
                    _p0 = __msa_insert_h(_p0, 1, p1[0]);
                    _p0 = __msa_insert_h(_p0, 2, p2[0]);
                    _p0 = __msa_insert_h(_p0, 3, p3[0]);
                    v8i16 _p1 = (v8i16)__msa_fill_w(0);
                    _p1 = __msa_insert_h(_p1, 0, p4[0]);
                    _p1 = __msa_insert_h(_p1, 1, p5[0]);
                    _p1 = __msa_insert_h(_p1, 2, p6[0]);
                    _p1 = __msa_insert_h(_p1, 3, p7[0]);
                    v4f32 _s = __msa_fill_w_f32(*ps++);
                    v4f32 _f0 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa((v4i32)_p0), _s), _scale03);
                    v4f32 _f1 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa((v4i32)_p1), _s), _scale47);
                    ((int64_t*)pp)[0] = float2int8(_f0, _f1);
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
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 4;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    v4f32 _p = __msa_fmul_w(bfloat2float_msa(p0a), __msa_fill_w_f32(*psa++));
                    _absmax = __msa_fmax_w(_absmax, (v4f32)__msa_and_v((v16u8)_p, _abs_mask));
                    p0a += 4;
                }

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax, _v127), pd, 0);
                pd += 4;

                const v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _absmax_safe = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax, _zero), (v16u8)_absmax, (v16u8)_v127);
                v4f32 _scale = __msa_fdiv_w(_v127, _absmax_safe);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(ps[0])), _scale);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 4), __msa_fill_w_f32(ps[1])), _scale);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 8), __msa_fill_w_f32(ps[2])), _scale);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 12), __msa_fill_w_f32(ps[3])), _scale);
                    v16i8 _q0 = float2int8(_p0);
                    v16i8 _q1 = float2int8(_p1);
                    v16i8 _q2 = float2int8(_p2);
                    v16i8 _q3 = float2int8(_p3);
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __msa_st_b(_q0, pp, 0);
                    pp += 16;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _p = __msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(*ps++));
                    _p = __msa_fmul_w(_p, _scale);
                    v16i8 _q = float2int8(_p);
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q, 0);
                    pp += 4;
                    p0 += 4;
                }
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;
            const unsigned short* p2 = p1 + A_hstep;
            const unsigned short* p3 = p2 + A_hstep;

            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax3 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const unsigned short* p2a = p2;
                const unsigned short* p3a = p3;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = bfloat2float_msa(p0a);
                    v4f32 _p1 = bfloat2float_msa(p1a);
                    v4f32 _p2 = bfloat2float_msa(p2a);
                    v4f32 _p3 = bfloat2float_msa(p3a);
                    v4f32 _s = (v4f32)__msa_ld_w(psa, 0);
                    _p0 = (v4f32)__msa_and_v((v16u8)_p0, _abs_mask);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = (v4f32)__msa_and_v((v16u8)_p1, _abs_mask);
                    _p1 = __msa_fmul_w(_p1, _s);
                    _p2 = (v4f32)__msa_and_v((v16u8)_p2, _abs_mask);
                    _p2 = __msa_fmul_w(_p2, _s);
                    _p3 = (v4f32)__msa_and_v((v16u8)_p3, _abs_mask);
                    _p3 = __msa_fmul_w(_p3, _s);
                    _absmax0 = __msa_fmax_w(_absmax0, _p0);
                    _absmax1 = __msa_fmax_w(_absmax1, _p1);
                    _absmax2 = __msa_fmax_w(_absmax2, _p2);
                    _absmax3 = __msa_fmax_w(_absmax3, _p3);
                    p0a += 4;
                    p1a += 4;
                    p2a += 4;
                    p3a += 4;
                    psa += 4;
                }
                transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                _absmax0 = __msa_fmax_w(__msa_fmax_w(_absmax0, _absmax1), __msa_fmax_w(_absmax2, _absmax3));

                for (; kk < max_kk0; kk++)
                {
                    v8i16 _p = (v8i16)__msa_fill_w(0);
                    _p = __msa_insert_h(_p, 0, p0a[0]);
                    _p = __msa_insert_h(_p, 1, p1a[0]);
                    _p = __msa_insert_h(_p, 2, p2a[0]);
                    _p = __msa_insert_h(_p, 3, p3a[0]);
                    v4f32 _f = (v4f32)__msa_and_v((v16u8)bfloat2float_msa((v4i32)_p), _abs_mask);
                    _f = __msa_fmul_w(_f, __msa_fill_w_f32(*psa++));
                    _absmax0 = __msa_fmax_w(_absmax0, _f);
                    p0a++;
                    p1a++;
                    p2a++;
                    p3a++;
                }

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                pd += 4;

                _absmax0 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, (v4f32)__msa_fill_w(0)), (v16u8)_absmax0, (v16u8)_v127);
                v4f32 _scale = __msa_fdiv_w(_v127, _absmax0);

                v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale, 0);
                v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale, 1);
                v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale, 2);
                v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale, 3);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = bfloat2float_msa(p0);
                    v4f32 _p1 = bfloat2float_msa(p1);
                    v4f32 _p2 = bfloat2float_msa(p2);
                    v4f32 _p3 = bfloat2float_msa(p3);
                    v4f32 _s = (v4f32)__msa_ld_w(ps, 0);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = __msa_fmul_w(_p1, _s);
                    _p2 = __msa_fmul_w(_p2, _s);
                    _p3 = __msa_fmul_w(_p3, _s);
                    _p0 = __msa_fmul_w(_p0, _scale0);
                    _p1 = __msa_fmul_w(_p1, _scale1);
                    _p2 = __msa_fmul_w(_p2, _scale2);
                    _p3 = __msa_fmul_w(_p3, _scale3);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v8i16 _p = (v8i16)__msa_fill_w(0);
                    _p = __msa_insert_h(_p, 0, p0[0]);
                    _p = __msa_insert_h(_p, 1, p1[0]);
                    _p = __msa_insert_h(_p, 2, p2[0]);
                    _p = __msa_insert_h(_p, 3, p3[0]);
                    v4f32 _f = __msa_fmul_w(bfloat2float_msa((v4i32)_p), __msa_fill_w_f32(*ps++));
                    _f = __msa_fmul_w(_f, _scale);
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_f), 0);
                    pp += 4;
                    p0++;
                    p1++;
                    p2++;
                    p3++;
                }
            }
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
        const unsigned short* p1 = p0 + A_hstep;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < local_block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const unsigned short* p0a = p0;
            const unsigned short* p1a = p1;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(*p0a++);
                float v1 = bfloat16_to_float32(*p1a++);
                const float s = *psa++;

                absmax0 = std::max(absmax0, fabsf(v0) * s);
                absmax1 = std::max(absmax1, fabsf(v1) * s);
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v00 = bfloat16_to_float32(p0[0]);
                float v01 = bfloat16_to_float32(p0[1]);
                float v02 = bfloat16_to_float32(p0[2]);
                float v03 = bfloat16_to_float32(p0[3]);
                float v10 = bfloat16_to_float32(p1[0]);
                float v11 = bfloat16_to_float32(p1[1]);
                float v12 = bfloat16_to_float32(p1[2]);
                float v13 = bfloat16_to_float32(p1[3]);
                v00 *= ps[0];
                v01 *= ps[1];
                v02 *= ps[2];
                v03 *= ps[3];
                v10 *= ps[0];
                v11 *= ps[1];
                v12 *= ps[2];
                v13 *= ps[3];
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v02 * scale0);
                pp[3] = float2int8(v03 * scale0);
                pp[4] = float2int8(v10 * scale1);
                pp[5] = float2int8(v11 * scale1);
                pp[6] = float2int8(v12 * scale1);
                pp[7] = float2int8(v13 * scale1);
                pp += 8;
                p0 += 4;
                p1 += 4;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(*p0++);
                float v1 = bfloat16_to_float32(*p1++);
                const float s = *ps++;
                v0 *= s;
                v1 *= s;
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < local_block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            const unsigned short* p0a = p0;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(*p0a++);
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(v0) * s);
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            *pd++ = absmax0 / 127.f;

            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v0 = bfloat16_to_float32(p0[0]);
                float v1 = bfloat16_to_float32(p0[1]);
                float v2 = bfloat16_to_float32(p0[2]);
                float v3 = bfloat16_to_float32(p0[3]);
                v0 *= ps[0];
                v1 *= ps[1];
                v2 *= ps[2];
                v3 *= ps[3];
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale0);
                pp[2] = float2int8(v2 * scale0);
                pp[3] = float2int8(v3 * scale0);
                pp += 4;
                p0 += 4;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(*p0++);
                v0 *= *ps++;
                *pp++ = float2int8(v0 * scale0);
            }
        }
    }
}

static void transpose_quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int local_block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __mips_msa
        for (; ii + 7 < max_ii; ii += 8)
        {
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0a);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                        v4f32 _p01 = bfloat2float_msa(p0a + 4);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p01, _abs_mask));
                        v4f32 _p10 = bfloat2float_msa(p0a + 8);
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                        v4f32 _p11 = bfloat2float_msa(p0a + 12);
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p11, _abs_mask));
                        v4f32 _p20 = bfloat2float_msa(p0a + 16);
                        _absmax2 = __msa_fmax_w(_absmax2, (v4f32)__msa_and_v((v16u8)_p20, _abs_mask));
                        v4f32 _p21 = bfloat2float_msa(p0a + 20);
                        _absmax2 = __msa_fmax_w(_absmax2, (v4f32)__msa_and_v((v16u8)_p21, _abs_mask));
                        v4f32 _p30 = bfloat2float_msa(p0a + 24);
                        _absmax3 = __msa_fmax_w(_absmax3, (v4f32)__msa_and_v((v16u8)_p30, _abs_mask));
                        v4f32 _p31 = bfloat2float_msa(p0a + 28);
                        _absmax3 = __msa_fmax_w(_absmax3, (v4f32)__msa_and_v((v16u8)_p31, _abs_mask));
                        v4f32 _p40 = bfloat2float_msa(p0a + 32);
                        _absmax4 = __msa_fmax_w(_absmax4, (v4f32)__msa_and_v((v16u8)_p40, _abs_mask));
                        v4f32 _p41 = bfloat2float_msa(p0a + 36);
                        _absmax4 = __msa_fmax_w(_absmax4, (v4f32)__msa_and_v((v16u8)_p41, _abs_mask));
                        v4f32 _p50 = bfloat2float_msa(p0a + 40);
                        _absmax5 = __msa_fmax_w(_absmax5, (v4f32)__msa_and_v((v16u8)_p50, _abs_mask));
                        v4f32 _p51 = bfloat2float_msa(p0a + 44);
                        _absmax5 = __msa_fmax_w(_absmax5, (v4f32)__msa_and_v((v16u8)_p51, _abs_mask));
                        v4f32 _p60 = bfloat2float_msa(p0a + 48);
                        _absmax6 = __msa_fmax_w(_absmax6, (v4f32)__msa_and_v((v16u8)_p60, _abs_mask));
                        v4f32 _p61 = bfloat2float_msa(p0a + 52);
                        _absmax6 = __msa_fmax_w(_absmax6, (v4f32)__msa_and_v((v16u8)_p61, _abs_mask));
                        v4f32 _p70 = bfloat2float_msa(p0a + 56);
                        _absmax7 = __msa_fmax_w(_absmax7, (v4f32)__msa_and_v((v16u8)_p70, _abs_mask));
                        v4f32 _p71 = bfloat2float_msa(p0a + 60);
                        _absmax7 = __msa_fmax_w(_absmax7, (v4f32)__msa_and_v((v16u8)_p71, _abs_mask));
                        p0a += A_hstep * 8;
                    }

                    transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                    transpose4x4_ps(_absmax4, _absmax5, _absmax6, _absmax7);
                    _absmax0 = __msa_fmax_w(__msa_fmax_w(_absmax0, _absmax1), __msa_fmax_w(_absmax2, _absmax3));
                    _absmax1 = __msa_fmax_w(__msa_fmax_w(_absmax4, _absmax5), __msa_fmax_w(_absmax6, _absmax7));

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const v4f32 _zero = (v4f32)__msa_fill_w(0);
                    _absmax0 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                    _absmax1 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                    v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax0);
                    v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax1);

                    v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale03, 0);
                    v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale03, 1);
                    v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale03, 2);
                    v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale03, 3);
                    v4f32 _scale4 = (v4f32)__msa_splati_w((v4i32)_scale47, 0);
                    v4f32 _scale5 = (v4f32)__msa_splati_w((v4i32)_scale47, 1);
                    v4f32 _scale6 = (v4f32)__msa_splati_w((v4i32)_scale47, 2);
                    v4f32 _scale7 = (v4f32)__msa_splati_w((v4i32)_scale47, 3);
                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0), _scale0);
                        v4f32 _p01 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _scale0);
                        v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _scale1);
                        v4f32 _p11 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _scale1);
                        v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0 + 16), _scale2);
                        v4f32 _p21 = __msa_fmul_w(bfloat2float_msa(p0 + 20), _scale2);
                        v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0 + 24), _scale3);
                        v4f32 _p31 = __msa_fmul_w(bfloat2float_msa(p0 + 28), _scale3);
                        v4f32 _p40 = __msa_fmul_w(bfloat2float_msa(p0 + 32), _scale4);
                        v4f32 _p41 = __msa_fmul_w(bfloat2float_msa(p0 + 36), _scale4);
                        v4f32 _p50 = __msa_fmul_w(bfloat2float_msa(p0 + 40), _scale5);
                        v4f32 _p51 = __msa_fmul_w(bfloat2float_msa(p0 + 44), _scale5);
                        v4f32 _p60 = __msa_fmul_w(bfloat2float_msa(p0 + 48), _scale6);
                        v4f32 _p61 = __msa_fmul_w(bfloat2float_msa(p0 + 52), _scale6);
                        v4f32 _p70 = __msa_fmul_w(bfloat2float_msa(p0 + 56), _scale7);
                        v4f32 _p71 = __msa_fmul_w(bfloat2float_msa(p0 + 60), _scale7);
                        ((int64_t*)pp)[0] = float2int8(_p00, _p10);
                        ((int64_t*)pp)[1] = float2int8(_p20, _p30);
                        ((int64_t*)pp)[2] = float2int8(_p40, _p50);
                        ((int64_t*)pp)[3] = float2int8(_p60, _p70);
                        ((int64_t*)pp)[4] = float2int8(_p01, _p11);
                        ((int64_t*)pp)[5] = float2int8(_p21, _p31);
                        ((int64_t*)pp)[6] = float2int8(_p41, _p51);
                        ((int64_t*)pp)[7] = float2int8(_p61, _p71);
                        pp += 64;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax10 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax20 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax30 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax40 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax50 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax60 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax70 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0a);
                        _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                        v4f32 _p10 = bfloat2float_msa(p0a + 4);
                        _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                        v4f32 _p20 = bfloat2float_msa(p0a + 8);
                        _absmax20 = __msa_fmax_w(_absmax20, (v4f32)__msa_and_v((v16u8)_p20, _abs_mask));
                        v4f32 _p30 = bfloat2float_msa(p0a + 12);
                        _absmax30 = __msa_fmax_w(_absmax30, (v4f32)__msa_and_v((v16u8)_p30, _abs_mask));
                        v4f32 _p40 = bfloat2float_msa(p0a + 16);
                        _absmax40 = __msa_fmax_w(_absmax40, (v4f32)__msa_and_v((v16u8)_p40, _abs_mask));
                        v4f32 _p50 = bfloat2float_msa(p0a + 20);
                        _absmax50 = __msa_fmax_w(_absmax50, (v4f32)__msa_and_v((v16u8)_p50, _abs_mask));
                        v4f32 _p60 = bfloat2float_msa(p0a + 24);
                        _absmax60 = __msa_fmax_w(_absmax60, (v4f32)__msa_and_v((v16u8)_p60, _abs_mask));
                        v4f32 _p70 = bfloat2float_msa(p0a + 28);
                        _absmax70 = __msa_fmax_w(_absmax70, (v4f32)__msa_and_v((v16u8)_p70, _abs_mask));
                        p0a += A_hstep * 4;
                    }

                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    transpose4x4_ps(_absmax40, _absmax50, _absmax60, _absmax70);
                    _absmax00 = __msa_fmax_w(__msa_fmax_w(_absmax00, _absmax10), __msa_fmax_w(_absmax20, _absmax30));
                    _absmax10 = __msa_fmax_w(__msa_fmax_w(_absmax40, _absmax50), __msa_fmax_w(_absmax60, _absmax70));

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax00, _v127), pd, 0);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax10, _v127), pd + 4, 0);
                    pd += 8;

                    const v4f32 _zero = (v4f32)__msa_fill_w(0);
                    _absmax00 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax00, _zero), (v16u8)_absmax00, (v16u8)_v127);
                    _absmax10 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax10, _zero), (v16u8)_absmax10, (v16u8)_v127);
                    v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax00);
                    v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax10);

                    v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale03, 0);
                    v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale03, 1);
                    v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale03, 2);
                    v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale03, 3);
                    v4f32 _scale4 = (v4f32)__msa_splati_w((v4i32)_scale47, 0);
                    v4f32 _scale5 = (v4f32)__msa_splati_w((v4i32)_scale47, 1);
                    v4f32 _scale6 = (v4f32)__msa_splati_w((v4i32)_scale47, 2);
                    v4f32 _scale7 = (v4f32)__msa_splati_w((v4i32)_scale47, 3);
                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0), _scale0);
                        v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _scale1);
                        v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _scale2);
                        v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _scale3);
                        v4f32 _p40 = __msa_fmul_w(bfloat2float_msa(p0 + 16), _scale4);
                        v4f32 _p50 = __msa_fmul_w(bfloat2float_msa(p0 + 20), _scale5);
                        v4f32 _p60 = __msa_fmul_w(bfloat2float_msa(p0 + 24), _scale6);
                        v4f32 _p70 = __msa_fmul_w(bfloat2float_msa(p0 + 28), _scale7);
                        ((int64_t*)pp)[0] = float2int8(_p00, _p10);
                        ((int64_t*)pp)[1] = float2int8(_p20, _p30);
                        ((int64_t*)pp)[2] = float2int8(_p40, _p50);
                        ((int64_t*)pp)[3] = float2int8(_p60, _p70);
                        pp += 32;
                        p0 += A_hstep * 4;
                    }
                }
            }

            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0a);
                        v4f32 _p1 = bfloat2float_msa(p0a + 4);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                        p0a += A_hstep;
                    }

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const v4f32 _zero = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax03 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                    v4f32 _absmax47 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                    v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax03);
                    v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax47);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const unsigned short* p1 = p0 + A_hstep;
                        const unsigned short* p2 = p1 + A_hstep;
                        const unsigned short* p3 = p2 + A_hstep;
                        v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), _scale03);
                        v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p1), _scale03);
                        v4f32 _p2 = __msa_fmul_w(bfloat2float_msa(p2), _scale03);
                        v4f32 _p3 = __msa_fmul_w(bfloat2float_msa(p3), _scale03);
                        v4f32 _p4 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _scale47);
                        v4f32 _p5 = __msa_fmul_w(bfloat2float_msa(p1 + 4), _scale47);
                        v4f32 _p6 = __msa_fmul_w(bfloat2float_msa(p2 + 4), _scale47);
                        v4f32 _p7 = __msa_fmul_w(bfloat2float_msa(p3 + 4), _scale47);

                        v16i8 _q0 = float2int8(_p0);
                        v16i8 _q1 = float2int8(_p1);
                        v16i8 _q2 = float2int8(_p2);
                        v16i8 _q3 = float2int8(_p3);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __msa_st_b(_q0, pp, 0);
                        _q0 = float2int8(_p4);
                        _q1 = float2int8(_p5);
                        _q2 = float2int8(_p6);
                        _q3 = float2int8(_p7);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __msa_st_b(_q0, pp + 16, 0);
                        pp += 32;
                        p0 = p3 + A_hstep;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = bfloat2float_msa(p0);
                        v4f32 _p1 = bfloat2float_msa(p0 + 4);
                        v16i8 _q0 = float2int8(__msa_fmul_w(_p0, _scale03));
                        v16i8 _q1 = float2int8(__msa_fmul_w(_p1, _scale47));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                        pp += 8;
                        p0 += A_hstep;
                    }
                }
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax01 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax10 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax11 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax20 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax21 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax30 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax31 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0a);
                        _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                        v4f32 _p01 = bfloat2float_msa(p0a + 4);
                        _absmax01 = __msa_fmax_w(_absmax01, (v4f32)__msa_and_v((v16u8)_p01, _abs_mask));
                        v4f32 _p10 = bfloat2float_msa(p0a + 8);
                        _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                        v4f32 _p11 = bfloat2float_msa(p0a + 12);
                        _absmax11 = __msa_fmax_w(_absmax11, (v4f32)__msa_and_v((v16u8)_p11, _abs_mask));
                        v4f32 _p20 = bfloat2float_msa(p0a + 16);
                        _absmax20 = __msa_fmax_w(_absmax20, (v4f32)__msa_and_v((v16u8)_p20, _abs_mask));
                        v4f32 _p21 = bfloat2float_msa(p0a + 20);
                        _absmax21 = __msa_fmax_w(_absmax21, (v4f32)__msa_and_v((v16u8)_p21, _abs_mask));
                        v4f32 _p30 = bfloat2float_msa(p0a + 24);
                        _absmax30 = __msa_fmax_w(_absmax30, (v4f32)__msa_and_v((v16u8)_p30, _abs_mask));
                        v4f32 _p31 = bfloat2float_msa(p0a + 28);
                        _absmax31 = __msa_fmax_w(_absmax31, (v4f32)__msa_and_v((v16u8)_p31, _abs_mask));
                        p0a += A_hstep * 8;
                    }

                    _absmax00 = __msa_fmax_w(_absmax00, _absmax01);
                    _absmax10 = __msa_fmax_w(_absmax10, _absmax11);
                    _absmax20 = __msa_fmax_w(_absmax20, _absmax21);
                    _absmax30 = __msa_fmax_w(_absmax30, _absmax31);
                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    _absmax00 = __msa_fmax_w(__msa_fmax_w(_absmax00, _absmax10), __msa_fmax_w(_absmax20, _absmax30));

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax00, _v127), pd, 0);
                    pd += 4;

                    _absmax00 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax00, (v4f32)__msa_fill_w(0)), (v16u8)_absmax00, (v16u8)_v127);
                    v4f32 _scale = __msa_fdiv_w(_v127, _absmax00);
                    v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale, 0);
                    v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale, 1);
                    v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale, 2);
                    v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale, 3);
                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0), _scale0);
                        v4f32 _p01 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _scale0);
                        v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _scale1);
                        v4f32 _p11 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _scale1);
                        v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0 + 16), _scale2);
                        v4f32 _p21 = __msa_fmul_w(bfloat2float_msa(p0 + 20), _scale2);
                        v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0 + 24), _scale3);
                        v4f32 _p31 = __msa_fmul_w(bfloat2float_msa(p0 + 28), _scale3);
                        ((int64_t*)pp)[0] = float2int8(_p00, _p10);
                        ((int64_t*)pp)[1] = float2int8(_p20, _p30);
                        ((int64_t*)pp)[2] = float2int8(_p01, _p11);
                        ((int64_t*)pp)[3] = float2int8(_p21, _p31);
                        pp += 32;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax10 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax20 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax30 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0a);
                        _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                        v4f32 _p10 = bfloat2float_msa(p0a + 4);
                        _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                        v4f32 _p20 = bfloat2float_msa(p0a + 8);
                        _absmax20 = __msa_fmax_w(_absmax20, (v4f32)__msa_and_v((v16u8)_p20, _abs_mask));
                        v4f32 _p30 = bfloat2float_msa(p0a + 12);
                        _absmax30 = __msa_fmax_w(_absmax30, (v4f32)__msa_and_v((v16u8)_p30, _abs_mask));
                        p0a += A_hstep * 4;
                    }

                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    _absmax00 = __msa_fmax_w(__msa_fmax_w(_absmax00, _absmax10), __msa_fmax_w(_absmax20, _absmax30));

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax00, _v127), pd, 0);
                    pd += 4;

                    _absmax00 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax00, (v4f32)__msa_fill_w(0)), (v16u8)_absmax00, (v16u8)_v127);
                    v4f32 _scale = __msa_fdiv_w(_v127, _absmax00);
                    v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale, 0);
                    v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale, 1);
                    v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale, 2);
                    v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale, 3);
                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0), _scale0);
                        v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _scale1);
                        v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _scale2);
                        v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _scale3);
                        ((int64_t*)pp)[0] = float2int8(_p00, _p10);
                        ((int64_t*)pp)[1] = float2int8(_p20, _p30);
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
                }
            }

            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p = bfloat2float_msa(p0a);
                        _absmax = __msa_fmax_w(_absmax, (v4f32)__msa_and_v((v16u8)_p, _abs_mask));
                        p0a += A_hstep;
                    }

                    const v4f32 _v127 = __msa_fill_w_f32(127.f);
                    __msa_st_w((v4i32)__msa_fdiv_w(_absmax, _v127), pd, 0);
                    pd += 4;

                    const v4f32 _zero = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax_safe = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax, _zero), (v16u8)_absmax, (v16u8)_v127);
                    v4f32 _scale = __msa_fdiv_w(_v127, _absmax_safe);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const unsigned short* p1 = p0 + A_hstep;
                        const unsigned short* p2 = p1 + A_hstep;
                        const unsigned short* p3 = p2 + A_hstep;
                        v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), _scale);
                        v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p1), _scale);
                        v4f32 _p2 = __msa_fmul_w(bfloat2float_msa(p2), _scale);
                        v4f32 _p3 = __msa_fmul_w(bfloat2float_msa(p3), _scale);

                        v16i8 _q0 = float2int8(_p0);
                        v16i8 _q1 = float2int8(_p1);
                        v16i8 _q2 = float2int8(_p2);
                        v16i8 _q3 = float2int8(_p3);
                        transpose16x4_epi8(_q0, _q1, _q2, _q3);
                        __msa_st_b(_q0, pp, 0);
                        pp += 16;
                        p0 = p3 + A_hstep;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p = __msa_fmul_w(bfloat2float_msa(p0), _scale);
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                        pp += 4;
                        p0 += A_hstep;
                    }
                }
            }
        }
#endif // __mips_msa
        for (; ii + 1 < max_ii; ii += 2)
        {
#if __mips_msa
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax01 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax10 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax11 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0a);
                        _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                        v4f32 _p01 = bfloat2float_msa(p0a + 4);
                        _absmax01 = __msa_fmax_w(_absmax01, (v4f32)__msa_and_v((v16u8)_p01, _abs_mask));
                        v4f32 _p10 = bfloat2float_msa(p0a + 8);
                        _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                        v4f32 _p11 = bfloat2float_msa(p0a + 12);
                        _absmax11 = __msa_fmax_w(_absmax11, (v4f32)__msa_and_v((v16u8)_p11, _abs_mask));
                        p0a += A_hstep * 8;
                    }

                    const float absmax0 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax00, _absmax01));
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    const float absmax1 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax10, _absmax11));
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0);
                        _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                        v4f32 _p01 = bfloat2float_msa(p0 + 4);
                        _p01 = __msa_fmul_w(_p01, __msa_fill_w_f32(scale0));
                        ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p01), 0);
                        v4f32 _p10 = bfloat2float_msa(p0 + 8);
                        _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                        v4f32 _p11 = bfloat2float_msa(p0 + 12);
                        _p11 = __msa_fmul_w(_p11, __msa_fill_w_f32(scale1));
                        ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p11), 0);
                        pp += 16;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax10 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0a);
                        _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                        v4f32 _p10 = bfloat2float_msa(p0a + 4);
                        _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                        p0a += A_hstep * 4;
                    }

                    const float absmax0 = __msa_reduce_fmax_w(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    const float absmax1 = __msa_reduce_fmax_w(_absmax10);
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0);
                        _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                        v4f32 _p10 = bfloat2float_msa(p0 + 4);
                        _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                        pp += 8;
                        p0 += A_hstep * 4;
                    }
                }
            }
#endif // __mips_msa
            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0a[0]);
                        float v1 = bfloat16_to_float32(p0a[1]);
                        absmax0 = std::max(absmax0, fabsf(v0));
                        absmax1 = std::max(absmax1, fabsf(v1));
                        p0a += A_hstep;
                    }

                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float v00 = bfloat16_to_float32(p0[0]);
                        float v10 = bfloat16_to_float32(p0[1]);
                        float v01 = bfloat16_to_float32(p0[A_hstep]);
                        float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                        float v02 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v12 = bfloat16_to_float32(p0[A_hstep * 2 + 1]);
                        float v03 = bfloat16_to_float32(p0[A_hstep * 3]);
                        float v13 = bfloat16_to_float32(p0[A_hstep * 3 + 1]);
                        pp[0] = float2int8(v00 * scale0);
                        pp[1] = float2int8(v01 * scale0);
                        pp[2] = float2int8(v02 * scale0);
                        pp[3] = float2int8(v03 * scale0);
                        pp[4] = float2int8(v10 * scale1);
                        pp[5] = float2int8(v11 * scale1);
                        pp[6] = float2int8(v12 * scale1);
                        pp[7] = float2int8(v13 * scale1);
                        p0 += A_hstep * 4;
                        pp += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[1]);
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale1);
                        pp += 2;
                        p0 += A_hstep;
                    }
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
#if __mips_msa
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax01 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0a);
                        _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                        v4f32 _p01 = bfloat2float_msa(p0a + 4);
                        _absmax01 = __msa_fmax_w(_absmax01, (v4f32)__msa_and_v((v16u8)_p01, _abs_mask));
                        p0a += A_hstep * 8;
                    }

                    const float absmax0 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax00, _absmax01));
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    pd += 1;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0);
                        _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                        v4f32 _p01 = bfloat2float_msa(p0 + 4);
                        _p01 = __msa_fmul_w(_p01, __msa_fill_w_f32(scale0));
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p01), 0);
                        pp += 8;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    v4f32 _absmax00 = (v4f32)__msa_fill_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0a);
                        _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                        p0a += A_hstep * 4;
                    }

                    const float absmax0 = __msa_reduce_fmax_w(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    pd += 1;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0);
                        _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                        pp += 4;
                        p0 += A_hstep * 4;
                    }
                }
            }
#endif // __mips_msa
            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax0 = 0.f;
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(*p0a);
                        absmax0 = std::max(absmax0, fabsf(v0));
                        p0a += A_hstep;
                    }

                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    *pd++ = absmax0 / 127.f;

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[A_hstep]);
                        float v2 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v3 = bfloat16_to_float32(p0[A_hstep * 3]);
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale0);
                        pp[2] = float2int8(v2 * scale0);
                        pp[3] = float2int8(v3 * scale0);
                        p0 += A_hstep * 4;
                        pp += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(*p0);
                        *pp++ = float2int8(v0 * scale0);
                        p0 += A_hstep;
                    }
                }
            }
        }
        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _s1 = (v4f32)__msa_ld_w(psa + 4, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0a), _s0);
                    _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                    v4f32 _p01 = __msa_fmul_w(bfloat2float_msa(p0a + 4), _s1);
                    _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p01, _abs_mask));
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0a + 8), _s0);
                    _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                    v4f32 _p11 = __msa_fmul_w(bfloat2float_msa(p0a + 12), _s1);
                    _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p11, _abs_mask));
                    v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0a + 16), _s0);
                    _absmax2 = __msa_fmax_w(_absmax2, (v4f32)__msa_and_v((v16u8)_p20, _abs_mask));
                    v4f32 _p21 = __msa_fmul_w(bfloat2float_msa(p0a + 20), _s1);
                    _absmax2 = __msa_fmax_w(_absmax2, (v4f32)__msa_and_v((v16u8)_p21, _abs_mask));
                    v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0a + 24), _s0);
                    _absmax3 = __msa_fmax_w(_absmax3, (v4f32)__msa_and_v((v16u8)_p30, _abs_mask));
                    v4f32 _p31 = __msa_fmul_w(bfloat2float_msa(p0a + 28), _s1);
                    _absmax3 = __msa_fmax_w(_absmax3, (v4f32)__msa_and_v((v16u8)_p31, _abs_mask));
                    v4f32 _p40 = __msa_fmul_w(bfloat2float_msa(p0a + 32), _s0);
                    _absmax4 = __msa_fmax_w(_absmax4, (v4f32)__msa_and_v((v16u8)_p40, _abs_mask));
                    v4f32 _p41 = __msa_fmul_w(bfloat2float_msa(p0a + 36), _s1);
                    _absmax4 = __msa_fmax_w(_absmax4, (v4f32)__msa_and_v((v16u8)_p41, _abs_mask));
                    v4f32 _p50 = __msa_fmul_w(bfloat2float_msa(p0a + 40), _s0);
                    _absmax5 = __msa_fmax_w(_absmax5, (v4f32)__msa_and_v((v16u8)_p50, _abs_mask));
                    v4f32 _p51 = __msa_fmul_w(bfloat2float_msa(p0a + 44), _s1);
                    _absmax5 = __msa_fmax_w(_absmax5, (v4f32)__msa_and_v((v16u8)_p51, _abs_mask));
                    v4f32 _p60 = __msa_fmul_w(bfloat2float_msa(p0a + 48), _s0);
                    _absmax6 = __msa_fmax_w(_absmax6, (v4f32)__msa_and_v((v16u8)_p60, _abs_mask));
                    v4f32 _p61 = __msa_fmul_w(bfloat2float_msa(p0a + 52), _s1);
                    _absmax6 = __msa_fmax_w(_absmax6, (v4f32)__msa_and_v((v16u8)_p61, _abs_mask));
                    v4f32 _p70 = __msa_fmul_w(bfloat2float_msa(p0a + 56), _s0);
                    _absmax7 = __msa_fmax_w(_absmax7, (v4f32)__msa_and_v((v16u8)_p70, _abs_mask));
                    v4f32 _p71 = __msa_fmul_w(bfloat2float_msa(p0a + 60), _s1);
                    _absmax7 = __msa_fmax_w(_absmax7, (v4f32)__msa_and_v((v16u8)_p71, _abs_mask));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                transpose4x4_ps(_absmax4, _absmax5, _absmax6, _absmax7);
                _absmax0 = __msa_fmax_w(__msa_fmax_w(_absmax0, _absmax1), __msa_fmax_w(_absmax2, _absmax3));
                _absmax1 = __msa_fmax_w(__msa_fmax_w(_absmax4, _absmax5), __msa_fmax_w(_absmax6, _absmax7));

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                pd += 8;

                const v4f32 _zero = (v4f32)__msa_fill_w(0);
                _absmax0 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                _absmax1 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax0);
                v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax1);

                v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale03, 0);
                v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale03, 1);
                v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale03, 2);
                v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale03, 3);
                v4f32 _scale4 = (v4f32)__msa_splati_w((v4i32)_scale47, 0);
                v4f32 _scale5 = (v4f32)__msa_splati_w((v4i32)_scale47, 1);
                v4f32 _scale6 = (v4f32)__msa_splati_w((v4i32)_scale47, 2);
                v4f32 _scale7 = (v4f32)__msa_splati_w((v4i32)_scale47, 3);
                int kk = 0;
                for (; kk < max_kk0; kk += 8)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _s1 = (v4f32)__msa_ld_w(ps + 4, 0);
                    v4f32 _p00 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), _s0), _scale0);
                    v4f32 _p01 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 4), _s1), _scale0);
                    v4f32 _p10 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 8), _s0), _scale1);
                    v4f32 _p11 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 12), _s1), _scale1);
                    v4f32 _p20 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 16), _s0), _scale2);
                    v4f32 _p21 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 20), _s1), _scale2);
                    v4f32 _p30 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 24), _s0), _scale3);
                    v4f32 _p31 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 28), _s1), _scale3);
                    v4f32 _p40 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 32), _s0), _scale4);
                    v4f32 _p41 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 36), _s1), _scale4);
                    v4f32 _p50 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 40), _s0), _scale5);
                    v4f32 _p51 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 44), _s1), _scale5);
                    v4f32 _p60 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 48), _s0), _scale6);
                    v4f32 _p61 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 52), _s1), _scale6);
                    v4f32 _p70 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 56), _s0), _scale7);
                    v4f32 _p71 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 60), _s1), _scale7);
                    ((int64_t*)pp)[0] = float2int8(_p00, _p10);
                    ((int64_t*)pp)[1] = float2int8(_p20, _p30);
                    ((int64_t*)pp)[2] = float2int8(_p40, _p50);
                    ((int64_t*)pp)[3] = float2int8(_p60, _p70);
                    ((int64_t*)pp)[4] = float2int8(_p01, _p11);
                    ((int64_t*)pp)[5] = float2int8(_p21, _p31);
                    ((int64_t*)pp)[6] = float2int8(_p41, _p51);
                    ((int64_t*)pp)[7] = float2int8(_p61, _p71);
                    pp += 64;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax10 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax20 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax30 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax40 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax50 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax60 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax70 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0a), _s0);
                    _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0a + 4), _s0);
                    _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                    v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0a + 8), _s0);
                    _absmax20 = __msa_fmax_w(_absmax20, (v4f32)__msa_and_v((v16u8)_p20, _abs_mask));
                    v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0a + 12), _s0);
                    _absmax30 = __msa_fmax_w(_absmax30, (v4f32)__msa_and_v((v16u8)_p30, _abs_mask));
                    v4f32 _p40 = __msa_fmul_w(bfloat2float_msa(p0a + 16), _s0);
                    _absmax40 = __msa_fmax_w(_absmax40, (v4f32)__msa_and_v((v16u8)_p40, _abs_mask));
                    v4f32 _p50 = __msa_fmul_w(bfloat2float_msa(p0a + 20), _s0);
                    _absmax50 = __msa_fmax_w(_absmax50, (v4f32)__msa_and_v((v16u8)_p50, _abs_mask));
                    v4f32 _p60 = __msa_fmul_w(bfloat2float_msa(p0a + 24), _s0);
                    _absmax60 = __msa_fmax_w(_absmax60, (v4f32)__msa_and_v((v16u8)_p60, _abs_mask));
                    v4f32 _p70 = __msa_fmul_w(bfloat2float_msa(p0a + 28), _s0);
                    _absmax70 = __msa_fmax_w(_absmax70, (v4f32)__msa_and_v((v16u8)_p70, _abs_mask));
                    p0a += A_hstep * 4;
                    psa += 4;
                }

                transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                transpose4x4_ps(_absmax40, _absmax50, _absmax60, _absmax70);
                _absmax00 = __msa_fmax_w(__msa_fmax_w(_absmax00, _absmax10), __msa_fmax_w(_absmax20, _absmax30));
                _absmax10 = __msa_fmax_w(__msa_fmax_w(_absmax40, _absmax50), __msa_fmax_w(_absmax60, _absmax70));

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax00, _v127), pd, 0);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax10, _v127), pd + 4, 0);
                pd += 8;

                const v4f32 _zero = (v4f32)__msa_fill_w(0);
                _absmax00 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax00, _zero), (v16u8)_absmax00, (v16u8)_v127);
                _absmax10 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax10, _zero), (v16u8)_absmax10, (v16u8)_v127);
                v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax00);
                v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax10);

                v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale03, 0);
                v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale03, 1);
                v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale03, 2);
                v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale03, 3);
                v4f32 _scale4 = (v4f32)__msa_splati_w((v4i32)_scale47, 0);
                v4f32 _scale5 = (v4f32)__msa_splati_w((v4i32)_scale47, 1);
                v4f32 _scale6 = (v4f32)__msa_splati_w((v4i32)_scale47, 2);
                v4f32 _scale7 = (v4f32)__msa_splati_w((v4i32)_scale47, 3);
                int kk = 0;
                for (; kk < max_kk0; kk += 4)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _p00 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), _s0), _scale0);
                    v4f32 _p10 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 4), _s0), _scale1);
                    v4f32 _p20 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 8), _s0), _scale2);
                    v4f32 _p30 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 12), _s0), _scale3);
                    v4f32 _p40 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 16), _s0), _scale4);
                    v4f32 _p50 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 20), _s0), _scale5);
                    v4f32 _p60 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 24), _s0), _scale6);
                    v4f32 _p70 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 28), _s0), _scale7);
                    ((int64_t*)pp)[0] = float2int8(_p00, _p10);
                    ((int64_t*)pp)[1] = float2int8(_p20, _p30);
                    ((int64_t*)pp)[2] = float2int8(_p40, _p50);
                    ((int64_t*)pp)[3] = float2int8(_p60, _p70);
                    pp += 32;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                int kk = 0;
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _p0 = bfloat2float_msa(p0a);
                    v4f32 _p1 = bfloat2float_msa(p0a + 4);
                    v4f32 _s = __msa_fill_w_f32(*psa++);
                    _p0 = (v4f32)__msa_and_v((v16u8)_p0, _abs_mask);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = (v4f32)__msa_and_v((v16u8)_p1, _abs_mask);
                    _p1 = __msa_fmul_w(_p1, _s);
                    _absmax0 = __msa_fmax_w(_absmax0, _p0);
                    _absmax1 = __msa_fmax_w(_absmax1, _p1);
                    p0a += A_hstep;
                }

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax0, _v127), pd, 0);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax1, _v127), pd + 4, 0);
                pd += 8;

                const v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _absmax03 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax0, _zero), (v16u8)_absmax0, (v16u8)_v127);
                v4f32 _absmax47 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax1, _zero), (v16u8)_absmax1, (v16u8)_v127);
                v4f32 _scale03 = __msa_fdiv_w(_v127, _absmax03);
                v4f32 _scale47 = __msa_fdiv_w(_v127, _absmax47);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned short* p1 = p0 + A_hstep;
                    const unsigned short* p2 = p1 + A_hstep;
                    const unsigned short* p3 = p2 + A_hstep;
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(ps[0])), _scale03);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p1), __msa_fill_w_f32(ps[1])), _scale03);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p2), __msa_fill_w_f32(ps[2])), _scale03);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p3), __msa_fill_w_f32(ps[3])), _scale03);
                    v4f32 _p4 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 4), __msa_fill_w_f32(ps[0])), _scale47);
                    v4f32 _p5 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p1 + 4), __msa_fill_w_f32(ps[1])), _scale47);
                    v4f32 _p6 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p2 + 4), __msa_fill_w_f32(ps[2])), _scale47);
                    v4f32 _p7 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p3 + 4), __msa_fill_w_f32(ps[3])), _scale47);

                    v16i8 _q0 = float2int8(_p0);
                    v16i8 _q1 = float2int8(_p1);
                    v16i8 _q2 = float2int8(_p2);
                    v16i8 _q3 = float2int8(_p3);
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __msa_st_b(_q0, pp, 0);
                    _q0 = float2int8(_p4);
                    _q1 = float2int8(_p5);
                    _q2 = float2int8(_p6);
                    _q3 = float2int8(_p7);
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __msa_st_b(_q0, pp + 16, 0);
                    pp += 32;
                    p0 = p3 + A_hstep;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = *ps++;
                    v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(s));
                    v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p0 + 4), __msa_fill_w_f32(s));
                    v16i8 _q0 = float2int8(__msa_fmul_w(_p0, _scale03));
                    v16i8 _q1 = float2int8(__msa_fmul_w(_p1, _scale47));
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                    pp += 8;
                    p0 += A_hstep;
                }
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax01 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax10 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax11 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax20 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax21 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax30 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax31 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _s1 = (v4f32)__msa_ld_w(psa + 4, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0a), _s0);
                    _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                    v4f32 _p01 = __msa_fmul_w(bfloat2float_msa(p0a + 4), _s1);
                    _absmax01 = __msa_fmax_w(_absmax01, (v4f32)__msa_and_v((v16u8)_p01, _abs_mask));
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0a + 8), _s0);
                    _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                    v4f32 _p11 = __msa_fmul_w(bfloat2float_msa(p0a + 12), _s1);
                    _absmax11 = __msa_fmax_w(_absmax11, (v4f32)__msa_and_v((v16u8)_p11, _abs_mask));
                    v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0a + 16), _s0);
                    _absmax20 = __msa_fmax_w(_absmax20, (v4f32)__msa_and_v((v16u8)_p20, _abs_mask));
                    v4f32 _p21 = __msa_fmul_w(bfloat2float_msa(p0a + 20), _s1);
                    _absmax21 = __msa_fmax_w(_absmax21, (v4f32)__msa_and_v((v16u8)_p21, _abs_mask));
                    v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0a + 24), _s0);
                    _absmax30 = __msa_fmax_w(_absmax30, (v4f32)__msa_and_v((v16u8)_p30, _abs_mask));
                    v4f32 _p31 = __msa_fmul_w(bfloat2float_msa(p0a + 28), _s1);
                    _absmax31 = __msa_fmax_w(_absmax31, (v4f32)__msa_and_v((v16u8)_p31, _abs_mask));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                _absmax00 = __msa_fmax_w(_absmax00, _absmax01);
                _absmax10 = __msa_fmax_w(_absmax10, _absmax11);
                _absmax20 = __msa_fmax_w(_absmax20, _absmax21);
                _absmax30 = __msa_fmax_w(_absmax30, _absmax31);
                transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                _absmax00 = __msa_fmax_w(__msa_fmax_w(_absmax00, _absmax10), __msa_fmax_w(_absmax20, _absmax30));

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax00, _v127), pd, 0);
                pd += 4;

                _absmax00 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax00, (v4f32)__msa_fill_w(0)), (v16u8)_absmax00, (v16u8)_v127);
                v4f32 _scale = __msa_fdiv_w(_v127, _absmax00);
                v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale, 0);
                v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale, 1);
                v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale, 2);
                v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale, 3);
                int kk = 0;
                for (; kk < max_kk0; kk += 8)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _s1 = (v4f32)__msa_ld_w(ps + 4, 0);
                    v4f32 _p00 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), _s0), _scale0);
                    v4f32 _p01 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 4), _s1), _scale0);
                    v4f32 _p10 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 8), _s0), _scale1);
                    v4f32 _p11 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 12), _s1), _scale1);
                    v4f32 _p20 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 16), _s0), _scale2);
                    v4f32 _p21 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 20), _s1), _scale2);
                    v4f32 _p30 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 24), _s0), _scale3);
                    v4f32 _p31 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 28), _s1), _scale3);
                    ((int64_t*)pp)[0] = float2int8(_p00, _p10);
                    ((int64_t*)pp)[1] = float2int8(_p20, _p30);
                    ((int64_t*)pp)[2] = float2int8(_p01, _p11);
                    ((int64_t*)pp)[3] = float2int8(_p21, _p31);
                    pp += 32;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax10 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax20 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax30 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0a), _s0);
                    _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0a + 4), _s0);
                    _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                    v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0a + 8), _s0);
                    _absmax20 = __msa_fmax_w(_absmax20, (v4f32)__msa_and_v((v16u8)_p20, _abs_mask));
                    v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0a + 12), _s0);
                    _absmax30 = __msa_fmax_w(_absmax30, (v4f32)__msa_and_v((v16u8)_p30, _abs_mask));
                    p0a += A_hstep * 4;
                    psa += 4;
                }

                transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                _absmax00 = __msa_fmax_w(__msa_fmax_w(_absmax00, _absmax10), __msa_fmax_w(_absmax20, _absmax30));

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax00, _v127), pd, 0);
                pd += 4;

                _absmax00 = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax00, (v4f32)__msa_fill_w(0)), (v16u8)_absmax00, (v16u8)_v127);
                v4f32 _scale = __msa_fdiv_w(_v127, _absmax00);
                v4f32 _scale0 = (v4f32)__msa_splati_w((v4i32)_scale, 0);
                v4f32 _scale1 = (v4f32)__msa_splati_w((v4i32)_scale, 1);
                v4f32 _scale2 = (v4f32)__msa_splati_w((v4i32)_scale, 2);
                v4f32 _scale3 = (v4f32)__msa_splati_w((v4i32)_scale, 3);
                int kk = 0;
                for (; kk < max_kk0; kk += 4)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _p00 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), _s0), _scale0);
                    v4f32 _p10 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 4), _s0), _scale1);
                    v4f32 _p20 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 8), _s0), _scale2);
                    v4f32 _p30 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0 + 12), _s0), _scale3);
                    ((int64_t*)pp)[0] = float2int8(_p00, _p10);
                    ((int64_t*)pp)[1] = float2int8(_p20, _p30);
                    pp += 16;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                int kk = 0;
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _p = bfloat2float_msa(p0a);
                    _p = (v4f32)__msa_and_v((v16u8)_p, _abs_mask);
                    _p = __msa_fmul_w(_p, __msa_fill_w_f32(*psa++));
                    _absmax = __msa_fmax_w(_absmax, _p);
                    p0a += A_hstep;
                }

                const v4f32 _v127 = __msa_fill_w_f32(127.f);
                __msa_st_w((v4i32)__msa_fdiv_w(_absmax, _v127), pd, 0);
                pd += 4;

                const v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _absmax_safe = (v4f32)__msa_bsel_v((v16u8)__msa_fceq_w(_absmax, _zero), (v16u8)_absmax, (v16u8)_v127);
                v4f32 _scale = __msa_fdiv_w(_v127, _absmax_safe);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned short* p1 = p0 + A_hstep;
                    const unsigned short* p2 = p1 + A_hstep;
                    const unsigned short* p3 = p2 + A_hstep;
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(ps[0])), _scale);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p1), __msa_fill_w_f32(ps[1])), _scale);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p2), __msa_fill_w_f32(ps[2])), _scale);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w(bfloat2float_msa(p3), __msa_fill_w_f32(ps[3])), _scale);

                    v16i8 _q0 = float2int8(_p0);
                    v16i8 _q1 = float2int8(_p1);
                    v16i8 _q2 = float2int8(_p2);
                    v16i8 _q3 = float2int8(_p3);
                    transpose16x4_epi8(_q0, _q1, _q2, _q3);
                    __msa_st_b(_q0, pp, 0);
                    pp += 16;
                    p0 = p3 + A_hstep;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = *ps++;
                    v4f32 _p = __msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(s));
                    _p = __msa_fmul_w(_p, _scale);
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                    pp += 4;
                    p0 += A_hstep;
                }
            }
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __mips_msa
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax01 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax10 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax11 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _s1 = (v4f32)__msa_ld_w(psa + 4, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0a), _s0);
                    _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                    v4f32 _p01 = __msa_fmul_w(bfloat2float_msa(p0a + 4), _s1);
                    _absmax01 = __msa_fmax_w(_absmax01, (v4f32)__msa_and_v((v16u8)_p01, _abs_mask));
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0a + 8), _s0);
                    _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                    v4f32 _p11 = __msa_fmul_w(bfloat2float_msa(p0a + 12), _s1);
                    _absmax11 = __msa_fmax_w(_absmax11, (v4f32)__msa_and_v((v16u8)_p11, _abs_mask));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                const float absmax0 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax00, _absmax01));
                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                pd[0] = absmax0 / 127.f;
                const float absmax1 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax10, _absmax11));
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                int kk = 0;
                for (; kk < max_kk0; kk += 8)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _s1 = (v4f32)__msa_ld_w(ps + 4, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0), _s0);
                    _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                    v4f32 _p01 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _s1);
                    _p01 = __msa_fmul_w(_p01, __msa_fill_w_f32(scale0));
                    ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p01), 0);
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _s0);
                    _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                    v4f32 _p11 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _s1);
                    _p11 = __msa_fmul_w(_p11, __msa_fill_w_f32(scale1));
                    ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p11), 0);
                    pp += 16;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax10 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0a), _s0);
                    _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0a + 4), _s0);
                    _absmax10 = __msa_fmax_w(_absmax10, (v4f32)__msa_and_v((v16u8)_p10, _abs_mask));
                    p0a += A_hstep * 4;
                    psa += 4;
                }

                const float absmax0 = __msa_reduce_fmax_w(_absmax00);
                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                pd[0] = absmax0 / 127.f;
                const float absmax1 = __msa_reduce_fmax_w(_absmax10);
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                int kk = 0;
                for (; kk < max_kk0; kk += 4)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0), _s0);
                    _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _s0);
                    _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                    pp += 8;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
        }
#endif // __mips_msa
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0a[0]);
                    float v1 = bfloat16_to_float32(p0a[1]);
                    const float s = *psa++;

                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    absmax1 = std::max(absmax1, fabsf(v1) * s);
                    p0a += A_hstep;
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v00 = bfloat16_to_float32(p0[0]);
                    float v10 = bfloat16_to_float32(p0[1]);
                    float v01 = bfloat16_to_float32(p0[A_hstep]);
                    float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                    float v02 = bfloat16_to_float32(p0[A_hstep * 2]);
                    float v12 = bfloat16_to_float32(p0[A_hstep * 2 + 1]);
                    float v03 = bfloat16_to_float32(p0[A_hstep * 3]);
                    float v13 = bfloat16_to_float32(p0[A_hstep * 3 + 1]);
                    v00 *= ps[0];
                    v10 *= ps[0];
                    v01 *= ps[1];
                    v11 *= ps[1];
                    v02 *= ps[2];
                    v12 *= ps[2];
                    v03 *= ps[3];
                    v13 *= ps[3];
                    ps += 4;
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v02 * scale0);
                    pp[3] = float2int8(v03 * scale0);
                    pp[4] = float2int8(v10 * scale1);
                    pp[5] = float2int8(v11 * scale1);
                    pp[6] = float2int8(v12 * scale1);
                    pp[7] = float2int8(v13 * scale1);
                    p0 += A_hstep * 4;
                    pp += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[1]);
                    const float s = *ps++;
                    v0 *= s;
                    v1 *= s;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                    p0 += A_hstep;
                }
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
#if __mips_msa
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax00 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax01 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _s1 = (v4f32)__msa_ld_w(psa + 4, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0a), _s0);
                    _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                    v4f32 _p01 = __msa_fmul_w(bfloat2float_msa(p0a + 4), _s1);
                    _absmax01 = __msa_fmax_w(_absmax01, (v4f32)__msa_and_v((v16u8)_p01, _abs_mask));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                const float absmax0 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax00, _absmax01));
                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                pd[0] = absmax0 / 127.f;
                pd += 1;

                int kk = 0;
                for (; kk < max_kk0; kk += 8)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _s1 = (v4f32)__msa_ld_w(ps + 4, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0), _s0);
                    _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                    v4f32 _p01 = __msa_fmul_w(bfloat2float_msa(p0 + 4), _s1);
                    _p01 = __msa_fmul_w(_p01, __msa_fill_w_f32(scale0));
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p01), 0);
                    pp += 8;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax00 = (v4f32)__msa_fill_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0a), _s0);
                    _absmax00 = __msa_fmax_w(_absmax00, (v4f32)__msa_and_v((v16u8)_p00, _abs_mask));
                    p0a += A_hstep * 4;
                    psa += 4;
                }

                const float absmax0 = __msa_reduce_fmax_w(_absmax00);
                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                pd[0] = absmax0 / 127.f;
                pd += 1;

                int kk = 0;
                for (; kk < max_kk0; kk += 4)
                {
                    v4f32 _s0 = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _p00 = __msa_fmul_w(bfloat2float_msa(p0), _s0);
                    _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                    pp += 4;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
        }
#endif // __mips_msa
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0a);
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    p0a += A_hstep;
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                *pd++ = absmax0 / 127.f;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[A_hstep]);
                    float v2 = bfloat16_to_float32(p0[A_hstep * 2]);
                    float v3 = bfloat16_to_float32(p0[A_hstep * 3]);
                    v0 *= ps[0];
                    v1 *= ps[1];
                    v2 *= ps[2];
                    v3 *= ps[3];
                    ps += 4;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    pp[2] = float2int8(v2 * scale0);
                    pp[3] = float2int8(v3 * scale0);
                    p0 += A_hstep * 4;
                    pp += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0);
                    v0 *= *ps++;
                    *pp++ = float2int8(v0 * scale0);
                    p0 += A_hstep;
                }
            }
        }
    }
}
