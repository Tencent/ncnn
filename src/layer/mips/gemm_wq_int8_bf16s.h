// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
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

                for (int g = 0; g < block_count; g++)
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

                    float absmax[8];
                    __msa_st_w((v4i32)_absmax0, absmax, 0);
                    __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                    const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                    const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                    const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                    const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                    const float scale4 = absmax[4] == 0.f ? 1.f : 127.f / absmax[4];
                    const float scale5 = absmax[5] == 0.f ? 1.f : 127.f / absmax[5];
                    const float scale6 = absmax[6] == 0.f ? 1.f : 127.f / absmax[6];
                    const float scale7 = absmax[7] == 0.f ? 1.f : 127.f / absmax[7];
                    pd[0] = absmax[0] / 127.f;
                    pd[1] = absmax[1] / 127.f;
                    pd[2] = absmax[2] / 127.f;
                    pd[3] = absmax[3] / 127.f;
                    pd[4] = absmax[4] / 127.f;
                    pd[5] = absmax[5] / 127.f;
                    pd[6] = absmax[6] / 127.f;
                    pd[7] = absmax[7] / 127.f;
                    pd += 8;

                    v4f32 _scale03 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                    v4f32 _scale47 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));

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

                for (int g = 0; g < block_count; g++)
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

                    float absmax[8];
                    __msa_st_w((v4i32)_absmax0, absmax, 0);
                    __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                    const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                    const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                    const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                    const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                    const float scale4 = absmax[4] == 0.f ? 1.f : 127.f / absmax[4];
                    const float scale5 = absmax[5] == 0.f ? 1.f : 127.f / absmax[5];
                    const float scale6 = absmax[6] == 0.f ? 1.f : 127.f / absmax[6];
                    const float scale7 = absmax[7] == 0.f ? 1.f : 127.f / absmax[7];
                    pd[0] = absmax[0] / 127.f;
                    pd[1] = absmax[1] / 127.f;
                    pd[2] = absmax[2] / 127.f;
                    pd[3] = absmax[3] / 127.f;
                    pd[4] = absmax[4] / 127.f;
                    pd[5] = absmax[5] / 127.f;
                    pd[6] = absmax[6] / 127.f;
                    pd[7] = absmax[7] / 127.f;
                    pd += 8;

                    v4f32 _scale03 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                    v4f32 _scale47 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));

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

                for (int g = 0; g < block_count; g++)
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

                    float absmax0 = __msa_reduce_fmax_w(_absmax0);
                    float absmax1 = __msa_reduce_fmax_w(_absmax1);
                    float absmax2 = __msa_reduce_fmax_w(_absmax2);
                    float absmax3 = __msa_reduce_fmax_w(_absmax3);
                    float absmax4 = __msa_reduce_fmax_w(_absmax4);
                    float absmax5 = __msa_reduce_fmax_w(_absmax5);
                    float absmax6 = __msa_reduce_fmax_w(_absmax6);
                    float absmax7 = __msa_reduce_fmax_w(_absmax7);

                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(*p0a++)));
                        absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(*p1a++)));
                        absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(*p2a++)));
                        absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(*p3a++)));
                        absmax4 = std::max(absmax4, fabsf(bfloat16_to_float32(*p4a++)));
                        absmax5 = std::max(absmax5, fabsf(bfloat16_to_float32(*p5a++)));
                        absmax6 = std::max(absmax6, fabsf(bfloat16_to_float32(*p6a++)));
                        absmax7 = std::max(absmax7, fabsf(bfloat16_to_float32(*p7a++)));
                    }

                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                    const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                    const float scale4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
                    const float scale5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
                    const float scale6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
                    const float scale7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    pd[4] = absmax4 / 127.f;
                    pd[5] = absmax5 / 127.f;
                    pd[6] = absmax6 / 127.f;
                    pd[7] = absmax7 / 127.f;
                    pd += 8;

                    v4f32 _scale0 = __msa_fill_w_f32(scale0);
                    v4f32 _scale1 = __msa_fill_w_f32(scale1);
                    v4f32 _scale2 = __msa_fill_w_f32(scale2);
                    v4f32 _scale3 = __msa_fill_w_f32(scale3);
                    v4f32 _scale4 = __msa_fill_w_f32(scale4);
                    v4f32 _scale5 = __msa_fill_w_f32(scale5);
                    v4f32 _scale6 = __msa_fill_w_f32(scale6);
                    v4f32 _scale7 = __msa_fill_w_f32(scale7);
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
                        pp[0] = float2int8(bfloat16_to_float32(*p0) * scale0);
                        pp[1] = float2int8(bfloat16_to_float32(*p1) * scale1);
                        pp[2] = float2int8(bfloat16_to_float32(*p2) * scale2);
                        pp[3] = float2int8(bfloat16_to_float32(*p3) * scale3);
                        pp[4] = float2int8(bfloat16_to_float32(*p4) * scale4);
                        pp[5] = float2int8(bfloat16_to_float32(*p5) * scale5);
                        pp[6] = float2int8(bfloat16_to_float32(*p6) * scale6);
                        pp[7] = float2int8(bfloat16_to_float32(*p7) * scale7);
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

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                float absmax2 = 0.f;
                float absmax3 = 0.f;

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
                    float absmax[4];
                    __msa_st_w((v4i32)_absmax0, absmax, 0);
                    absmax0 = absmax[0];
                    absmax1 = absmax[1];
                    absmax2 = absmax[2];
                    absmax3 = absmax[3];
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
                    absmax0 = __msa_reduce_fmax_w(_absmax0);
                    absmax1 = __msa_reduce_fmax_w(_absmax1);
                    absmax2 = __msa_reduce_fmax_w(_absmax2);
                    absmax3 = __msa_reduce_fmax_w(_absmax3);
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(*p0a++);
                        float v1 = bfloat16_to_float32(*p1a++);
                        float v2 = bfloat16_to_float32(*p2a++);
                        float v3 = bfloat16_to_float32(*p3a++);
                        absmax0 = std::max(absmax0, fabsf(v0));
                        absmax1 = std::max(absmax1, fabsf(v1));
                        absmax2 = std::max(absmax2, fabsf(v2));
                        absmax3 = std::max(absmax3, fabsf(v3));
                    }
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd += 4;

                if (elempack == 4)
                {
                    v4f32 _scale = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
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
                    v4f32 _scale0 = __msa_fill_w_f32(scale0);
                    v4f32 _scale1 = __msa_fill_w_f32(scale1);
                    v4f32 _scale2 = __msa_fill_w_f32(scale2);
                    v4f32 _scale3 = __msa_fill_w_f32(scale3);
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
                        float v0 = bfloat16_to_float32(*p0++);
                        float v1 = bfloat16_to_float32(*p1++);
                        float v2 = bfloat16_to_float32(*p2++);
                        float v3 = bfloat16_to_float32(*p3++);
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale1);
                        pp[2] = float2int8(v2 * scale2);
                        pp[3] = float2int8(v3 * scale3);
                        pp += 4;
                    }
                }
            }
        }
#endif // __mips_msa
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;

            for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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

                float absmax[8];
                __msa_st_w((v4i32)_absmax0, absmax, 0);
                __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                const float scale4 = absmax[4] == 0.f ? 1.f : 127.f / absmax[4];
                const float scale5 = absmax[5] == 0.f ? 1.f : 127.f / absmax[5];
                const float scale6 = absmax[6] == 0.f ? 1.f : 127.f / absmax[6];
                const float scale7 = absmax[7] == 0.f ? 1.f : 127.f / absmax[7];
                pd[0] = absmax[0] / 127.f;
                pd[1] = absmax[1] / 127.f;
                pd[2] = absmax[2] / 127.f;
                pd[3] = absmax[3] / 127.f;
                pd[4] = absmax[4] / 127.f;
                pd[5] = absmax[5] / 127.f;
                pd[6] = absmax[6] / 127.f;
                pd[7] = absmax[7] / 127.f;
                pd += 8;

                v4f32 _scale03 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                v4f32 _scale47 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));

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

            for (int g = 0; g < block_count; g++)
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

                float absmax[8];
                __msa_st_w((v4i32)_absmax0, absmax, 0);
                __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                const float scale4 = absmax[4] == 0.f ? 1.f : 127.f / absmax[4];
                const float scale5 = absmax[5] == 0.f ? 1.f : 127.f / absmax[5];
                const float scale6 = absmax[6] == 0.f ? 1.f : 127.f / absmax[6];
                const float scale7 = absmax[7] == 0.f ? 1.f : 127.f / absmax[7];
                pd[0] = absmax[0] / 127.f;
                pd[1] = absmax[1] / 127.f;
                pd[2] = absmax[2] / 127.f;
                pd[3] = absmax[3] / 127.f;
                pd[4] = absmax[4] / 127.f;
                pd[5] = absmax[5] / 127.f;
                pd[6] = absmax[6] / 127.f;
                pd[7] = absmax[7] / 127.f;
                pd += 8;

                v4f32 _scale03 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                v4f32 _scale47 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));

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

            for (int g = 0; g < block_count; g++)
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

                float absmax0 = __msa_reduce_fmax_w(_absmax0);
                float absmax1 = __msa_reduce_fmax_w(_absmax1);
                float absmax2 = __msa_reduce_fmax_w(_absmax2);
                float absmax3 = __msa_reduce_fmax_w(_absmax3);
                float absmax4 = __msa_reduce_fmax_w(_absmax4);
                float absmax5 = __msa_reduce_fmax_w(_absmax5);
                float absmax6 = __msa_reduce_fmax_w(_absmax6);
                float absmax7 = __msa_reduce_fmax_w(_absmax7);

                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(*p0a++)) * s);
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(*p1a++)) * s);
                    absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(*p2a++)) * s);
                    absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(*p3a++)) * s);
                    absmax4 = std::max(absmax4, fabsf(bfloat16_to_float32(*p4a++)) * s);
                    absmax5 = std::max(absmax5, fabsf(bfloat16_to_float32(*p5a++)) * s);
                    absmax6 = std::max(absmax6, fabsf(bfloat16_to_float32(*p6a++)) * s);
                    absmax7 = std::max(absmax7, fabsf(bfloat16_to_float32(*p7a++)) * s);
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                const float scale4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
                const float scale5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
                const float scale6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
                const float scale7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                pd += 8;

                v4f32 _scale0 = __msa_fill_w_f32(scale0);
                v4f32 _scale1 = __msa_fill_w_f32(scale1);
                v4f32 _scale2 = __msa_fill_w_f32(scale2);
                v4f32 _scale3 = __msa_fill_w_f32(scale3);
                v4f32 _scale4 = __msa_fill_w_f32(scale4);
                v4f32 _scale5 = __msa_fill_w_f32(scale5);
                v4f32 _scale6 = __msa_fill_w_f32(scale6);
                v4f32 _scale7 = __msa_fill_w_f32(scale7);
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
                    const float s = *ps++;
                    pp[0] = float2int8(bfloat16_to_float32(*p0) * s * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(*p1) * s * scale1);
                    pp[2] = float2int8(bfloat16_to_float32(*p2) * s * scale2);
                    pp[3] = float2int8(bfloat16_to_float32(*p3) * s * scale3);
                    pp[4] = float2int8(bfloat16_to_float32(*p4) * s * scale4);
                    pp[5] = float2int8(bfloat16_to_float32(*p5) * s * scale5);
                    pp[6] = float2int8(bfloat16_to_float32(*p6) * s * scale6);
                    pp[7] = float2int8(bfloat16_to_float32(*p7) * s * scale7);
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

            for (int g = 0; g < block_count; g++)
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

                float absmax[4];
                __msa_st_w((v4i32)_absmax, absmax, 0);
                const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                pd[0] = absmax[0] / 127.f;
                pd[1] = absmax[1] / 127.f;
                pd[2] = absmax[2] / 127.f;
                pd[3] = absmax[3] / 127.f;
                pd += 4;

                v4f32 _scale = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));

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

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                float absmax2 = 0.f;
                float absmax3 = 0.f;

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
                absmax0 = __msa_reduce_fmax_w(_absmax0);
                absmax1 = __msa_reduce_fmax_w(_absmax1);
                absmax2 = __msa_reduce_fmax_w(_absmax2);
                absmax3 = __msa_reduce_fmax_w(_absmax3);

                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0a++);
                    float v1 = bfloat16_to_float32(*p1a++);
                    float v2 = bfloat16_to_float32(*p2a++);
                    float v3 = bfloat16_to_float32(*p3a++);
                    const float s = *psa++;

                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    absmax1 = std::max(absmax1, fabsf(v1) * s);
                    absmax2 = std::max(absmax2, fabsf(v2) * s);
                    absmax3 = std::max(absmax3, fabsf(v3) * s);
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd += 4;

                v4f32 _scale0 = __msa_fill_w_f32(scale0);
                v4f32 _scale1 = __msa_fill_w_f32(scale1);
                v4f32 _scale2 = __msa_fill_w_f32(scale2);
                v4f32 _scale3 = __msa_fill_w_f32(scale3);
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
                    float v0 = bfloat16_to_float32(*p0++);
                    float v1 = bfloat16_to_float32(*p1++);
                    float v2 = bfloat16_to_float32(*p2++);
                    float v3 = bfloat16_to_float32(*p3++);
                    const float s = *ps++;
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp[2] = float2int8(v2 * scale2);
                    pp[3] = float2int8(v3 * scale3);
                    pp += 4;
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

        for (int g = 0; g < block_count; g++)
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

        for (int g = 0; g < block_count; g++)
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
    const int block_count = (max_kk + block_size - 1) / block_size;
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

                for (int g = 0; g < block_count; g++)
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
                    v4f32 _absmax40 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax41 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax50 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax51 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax60 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax61 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax70 = (v4f32)__msa_fill_w(0);
                    v4f32 _absmax71 = (v4f32)__msa_fill_w(0);

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
                        v4f32 _p40 = bfloat2float_msa(p0a + 32);
                        _absmax40 = __msa_fmax_w(_absmax40, (v4f32)__msa_and_v((v16u8)_p40, _abs_mask));
                        v4f32 _p41 = bfloat2float_msa(p0a + 36);
                        _absmax41 = __msa_fmax_w(_absmax41, (v4f32)__msa_and_v((v16u8)_p41, _abs_mask));
                        v4f32 _p50 = bfloat2float_msa(p0a + 40);
                        _absmax50 = __msa_fmax_w(_absmax50, (v4f32)__msa_and_v((v16u8)_p50, _abs_mask));
                        v4f32 _p51 = bfloat2float_msa(p0a + 44);
                        _absmax51 = __msa_fmax_w(_absmax51, (v4f32)__msa_and_v((v16u8)_p51, _abs_mask));
                        v4f32 _p60 = bfloat2float_msa(p0a + 48);
                        _absmax60 = __msa_fmax_w(_absmax60, (v4f32)__msa_and_v((v16u8)_p60, _abs_mask));
                        v4f32 _p61 = bfloat2float_msa(p0a + 52);
                        _absmax61 = __msa_fmax_w(_absmax61, (v4f32)__msa_and_v((v16u8)_p61, _abs_mask));
                        v4f32 _p70 = bfloat2float_msa(p0a + 56);
                        _absmax70 = __msa_fmax_w(_absmax70, (v4f32)__msa_and_v((v16u8)_p70, _abs_mask));
                        v4f32 _p71 = bfloat2float_msa(p0a + 60);
                        _absmax71 = __msa_fmax_w(_absmax71, (v4f32)__msa_and_v((v16u8)_p71, _abs_mask));
                        p0a += A_hstep * 8;
                    }

                    const float absmax0 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax00, _absmax01));
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    const float absmax1 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax10, _absmax11));
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    const float absmax2 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax20, _absmax21));
                    const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                    pd[2] = absmax2 / 127.f;
                    const float absmax3 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax30, _absmax31));
                    const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                    pd[3] = absmax3 / 127.f;
                    const float absmax4 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax40, _absmax41));
                    const float scale4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
                    pd[4] = absmax4 / 127.f;
                    const float absmax5 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax50, _absmax51));
                    const float scale5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
                    pd[5] = absmax5 / 127.f;
                    const float absmax6 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax60, _absmax61));
                    const float scale6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
                    pd[6] = absmax6 / 127.f;
                    const float absmax7 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax70, _absmax71));
                    const float scale7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
                    pd[7] = absmax7 / 127.f;
                    pd += 8;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0);
                        _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                        v4f32 _p01 = bfloat2float_msa(p0 + 4);
                        _p01 = __msa_fmul_w(_p01, __msa_fill_w_f32(scale0));
                        ((int*)pp)[8] = __msa_copy_s_w((v4i32)float2int8(_p01), 0);
                        v4f32 _p10 = bfloat2float_msa(p0 + 8);
                        _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                        v4f32 _p11 = bfloat2float_msa(p0 + 12);
                        _p11 = __msa_fmul_w(_p11, __msa_fill_w_f32(scale1));
                        ((int*)pp)[9] = __msa_copy_s_w((v4i32)float2int8(_p11), 0);
                        v4f32 _p20 = bfloat2float_msa(p0 + 16);
                        _p20 = __msa_fmul_w(_p20, __msa_fill_w_f32(scale2));
                        ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p20), 0);
                        v4f32 _p21 = bfloat2float_msa(p0 + 20);
                        _p21 = __msa_fmul_w(_p21, __msa_fill_w_f32(scale2));
                        ((int*)pp)[10] = __msa_copy_s_w((v4i32)float2int8(_p21), 0);
                        v4f32 _p30 = bfloat2float_msa(p0 + 24);
                        _p30 = __msa_fmul_w(_p30, __msa_fill_w_f32(scale3));
                        ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p30), 0);
                        v4f32 _p31 = bfloat2float_msa(p0 + 28);
                        _p31 = __msa_fmul_w(_p31, __msa_fill_w_f32(scale3));
                        ((int*)pp)[11] = __msa_copy_s_w((v4i32)float2int8(_p31), 0);
                        v4f32 _p40 = bfloat2float_msa(p0 + 32);
                        _p40 = __msa_fmul_w(_p40, __msa_fill_w_f32(scale4));
                        ((int*)pp)[4] = __msa_copy_s_w((v4i32)float2int8(_p40), 0);
                        v4f32 _p41 = bfloat2float_msa(p0 + 36);
                        _p41 = __msa_fmul_w(_p41, __msa_fill_w_f32(scale4));
                        ((int*)pp)[12] = __msa_copy_s_w((v4i32)float2int8(_p41), 0);
                        v4f32 _p50 = bfloat2float_msa(p0 + 40);
                        _p50 = __msa_fmul_w(_p50, __msa_fill_w_f32(scale5));
                        ((int*)pp)[5] = __msa_copy_s_w((v4i32)float2int8(_p50), 0);
                        v4f32 _p51 = bfloat2float_msa(p0 + 44);
                        _p51 = __msa_fmul_w(_p51, __msa_fill_w_f32(scale5));
                        ((int*)pp)[13] = __msa_copy_s_w((v4i32)float2int8(_p51), 0);
                        v4f32 _p60 = bfloat2float_msa(p0 + 48);
                        _p60 = __msa_fmul_w(_p60, __msa_fill_w_f32(scale6));
                        ((int*)pp)[6] = __msa_copy_s_w((v4i32)float2int8(_p60), 0);
                        v4f32 _p61 = bfloat2float_msa(p0 + 52);
                        _p61 = __msa_fmul_w(_p61, __msa_fill_w_f32(scale6));
                        ((int*)pp)[14] = __msa_copy_s_w((v4i32)float2int8(_p61), 0);
                        v4f32 _p70 = bfloat2float_msa(p0 + 56);
                        _p70 = __msa_fmul_w(_p70, __msa_fill_w_f32(scale7));
                        ((int*)pp)[7] = __msa_copy_s_w((v4i32)float2int8(_p70), 0);
                        v4f32 _p71 = bfloat2float_msa(p0 + 60);
                        _p71 = __msa_fmul_w(_p71, __msa_fill_w_f32(scale7));
                        ((int*)pp)[15] = __msa_copy_s_w((v4i32)float2int8(_p71), 0);
                        pp += 64;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < block_count; g++)
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

                    const float absmax0 = __msa_reduce_fmax_w(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    const float absmax1 = __msa_reduce_fmax_w(_absmax10);
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    const float absmax2 = __msa_reduce_fmax_w(_absmax20);
                    const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                    pd[2] = absmax2 / 127.f;
                    const float absmax3 = __msa_reduce_fmax_w(_absmax30);
                    const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                    pd[3] = absmax3 / 127.f;
                    const float absmax4 = __msa_reduce_fmax_w(_absmax40);
                    const float scale4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
                    pd[4] = absmax4 / 127.f;
                    const float absmax5 = __msa_reduce_fmax_w(_absmax50);
                    const float scale5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
                    pd[5] = absmax5 / 127.f;
                    const float absmax6 = __msa_reduce_fmax_w(_absmax60);
                    const float scale6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
                    pd[6] = absmax6 / 127.f;
                    const float absmax7 = __msa_reduce_fmax_w(_absmax70);
                    const float scale7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
                    pd[7] = absmax7 / 127.f;
                    pd += 8;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0);
                        _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                        v4f32 _p10 = bfloat2float_msa(p0 + 4);
                        _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                        v4f32 _p20 = bfloat2float_msa(p0 + 8);
                        _p20 = __msa_fmul_w(_p20, __msa_fill_w_f32(scale2));
                        ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p20), 0);
                        v4f32 _p30 = bfloat2float_msa(p0 + 12);
                        _p30 = __msa_fmul_w(_p30, __msa_fill_w_f32(scale3));
                        ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p30), 0);
                        v4f32 _p40 = bfloat2float_msa(p0 + 16);
                        _p40 = __msa_fmul_w(_p40, __msa_fill_w_f32(scale4));
                        ((int*)pp)[4] = __msa_copy_s_w((v4i32)float2int8(_p40), 0);
                        v4f32 _p50 = bfloat2float_msa(p0 + 20);
                        _p50 = __msa_fmul_w(_p50, __msa_fill_w_f32(scale5));
                        ((int*)pp)[5] = __msa_copy_s_w((v4i32)float2int8(_p50), 0);
                        v4f32 _p60 = bfloat2float_msa(p0 + 24);
                        _p60 = __msa_fmul_w(_p60, __msa_fill_w_f32(scale6));
                        ((int*)pp)[6] = __msa_copy_s_w((v4i32)float2int8(_p60), 0);
                        v4f32 _p70 = bfloat2float_msa(p0 + 28);
                        _p70 = __msa_fmul_w(_p70, __msa_fill_w_f32(scale7));
                        ((int*)pp)[7] = __msa_copy_s_w((v4i32)float2int8(_p70), 0);
                        pp += 32;
                        p0 += A_hstep * 4;
                    }
                }
            }

            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < block_count; g++)
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

                    float absmax[8];
                    __msa_st_w((v4i32)_absmax0, absmax, 0);
                    __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                    const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                    const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                    const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                    const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                    const float scale4 = absmax[4] == 0.f ? 1.f : 127.f / absmax[4];
                    const float scale5 = absmax[5] == 0.f ? 1.f : 127.f / absmax[5];
                    const float scale6 = absmax[6] == 0.f ? 1.f : 127.f / absmax[6];
                    const float scale7 = absmax[7] == 0.f ? 1.f : 127.f / absmax[7];
                    pd[0] = absmax[0] / 127.f;
                    pd[1] = absmax[1] / 127.f;
                    pd[2] = absmax[2] / 127.f;
                    pd[3] = absmax[3] / 127.f;
                    pd[4] = absmax[4] / 127.f;
                    pd[5] = absmax[5] / 127.f;
                    pd[6] = absmax[6] / 127.f;
                    pd[7] = absmax[7] / 127.f;
                    pd += 8;

                    v4f32 _scale03 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                    v4f32 _scale47 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));
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

                for (int g = 0; g < block_count; g++)
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

                    const float absmax0 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax00, _absmax01));
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    const float absmax1 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax10, _absmax11));
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    const float absmax2 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax20, _absmax21));
                    const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                    pd[2] = absmax2 / 127.f;
                    const float absmax3 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax30, _absmax31));
                    const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                    pd[3] = absmax3 / 127.f;
                    pd += 4;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0);
                        _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                        v4f32 _p01 = bfloat2float_msa(p0 + 4);
                        _p01 = __msa_fmul_w(_p01, __msa_fill_w_f32(scale0));
                        ((int*)pp)[4] = __msa_copy_s_w((v4i32)float2int8(_p01), 0);
                        v4f32 _p10 = bfloat2float_msa(p0 + 8);
                        _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                        v4f32 _p11 = bfloat2float_msa(p0 + 12);
                        _p11 = __msa_fmul_w(_p11, __msa_fill_w_f32(scale1));
                        ((int*)pp)[5] = __msa_copy_s_w((v4i32)float2int8(_p11), 0);
                        v4f32 _p20 = bfloat2float_msa(p0 + 16);
                        _p20 = __msa_fmul_w(_p20, __msa_fill_w_f32(scale2));
                        ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p20), 0);
                        v4f32 _p21 = bfloat2float_msa(p0 + 20);
                        _p21 = __msa_fmul_w(_p21, __msa_fill_w_f32(scale2));
                        ((int*)pp)[6] = __msa_copy_s_w((v4i32)float2int8(_p21), 0);
                        v4f32 _p30 = bfloat2float_msa(p0 + 24);
                        _p30 = __msa_fmul_w(_p30, __msa_fill_w_f32(scale3));
                        ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p30), 0);
                        v4f32 _p31 = bfloat2float_msa(p0 + 28);
                        _p31 = __msa_fmul_w(_p31, __msa_fill_w_f32(scale3));
                        ((int*)pp)[7] = __msa_copy_s_w((v4i32)float2int8(_p31), 0);
                        pp += 32;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < block_count; g++)
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

                    const float absmax0 = __msa_reduce_fmax_w(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    const float absmax1 = __msa_reduce_fmax_w(_absmax10);
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    const float absmax2 = __msa_reduce_fmax_w(_absmax20);
                    const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                    pd[2] = absmax2 / 127.f;
                    const float absmax3 = __msa_reduce_fmax_w(_absmax30);
                    const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                    pd[3] = absmax3 / 127.f;
                    pd += 4;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        v4f32 _p00 = bfloat2float_msa(p0);
                        _p00 = __msa_fmul_w(_p00, __msa_fill_w_f32(scale0));
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p00), 0);
                        v4f32 _p10 = bfloat2float_msa(p0 + 4);
                        _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                        ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                        v4f32 _p20 = bfloat2float_msa(p0 + 8);
                        _p20 = __msa_fmul_w(_p20, __msa_fill_w_f32(scale2));
                        ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p20), 0);
                        v4f32 _p30 = bfloat2float_msa(p0 + 12);
                        _p30 = __msa_fmul_w(_p30, __msa_fill_w_f32(scale3));
                        ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p30), 0);
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
                }
            }

            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < block_count; g++)
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

                    float absmax[4];
                    __msa_st_w((v4i32)_absmax, absmax, 0);
                    const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                    const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                    const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                    const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                    pd[0] = absmax[0] / 127.f;
                    pd[1] = absmax[1] / 127.f;
                    pd[2] = absmax[2] / 127.f;
                    pd[3] = absmax[3] / 127.f;
                    pd += 4;

                    v4f32 _scale = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
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
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[1]);
                        float v2 = bfloat16_to_float32(p0[2]);
                        float v3 = bfloat16_to_float32(p0[3]);
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale1);
                        pp[2] = float2int8(v2 * scale2);
                        pp[3] = float2int8(v3 * scale3);
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

                for (int g = 0; g < block_count; g++)
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

                for (int g = 0; g < block_count; g++)
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
                for (int g = 0; g < block_count; g++)
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

                for (int g = 0; g < block_count; g++)
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

                for (int g = 0; g < block_count; g++)
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

                for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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
                v4f32 _absmax40 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax41 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax50 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax51 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax60 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax61 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax70 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax71 = (v4f32)__msa_fill_w(0);

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
                    v4f32 _p40 = __msa_fmul_w(bfloat2float_msa(p0a + 32), _s0);
                    _absmax40 = __msa_fmax_w(_absmax40, (v4f32)__msa_and_v((v16u8)_p40, _abs_mask));
                    v4f32 _p41 = __msa_fmul_w(bfloat2float_msa(p0a + 36), _s1);
                    _absmax41 = __msa_fmax_w(_absmax41, (v4f32)__msa_and_v((v16u8)_p41, _abs_mask));
                    v4f32 _p50 = __msa_fmul_w(bfloat2float_msa(p0a + 40), _s0);
                    _absmax50 = __msa_fmax_w(_absmax50, (v4f32)__msa_and_v((v16u8)_p50, _abs_mask));
                    v4f32 _p51 = __msa_fmul_w(bfloat2float_msa(p0a + 44), _s1);
                    _absmax51 = __msa_fmax_w(_absmax51, (v4f32)__msa_and_v((v16u8)_p51, _abs_mask));
                    v4f32 _p60 = __msa_fmul_w(bfloat2float_msa(p0a + 48), _s0);
                    _absmax60 = __msa_fmax_w(_absmax60, (v4f32)__msa_and_v((v16u8)_p60, _abs_mask));
                    v4f32 _p61 = __msa_fmul_w(bfloat2float_msa(p0a + 52), _s1);
                    _absmax61 = __msa_fmax_w(_absmax61, (v4f32)__msa_and_v((v16u8)_p61, _abs_mask));
                    v4f32 _p70 = __msa_fmul_w(bfloat2float_msa(p0a + 56), _s0);
                    _absmax70 = __msa_fmax_w(_absmax70, (v4f32)__msa_and_v((v16u8)_p70, _abs_mask));
                    v4f32 _p71 = __msa_fmul_w(bfloat2float_msa(p0a + 60), _s1);
                    _absmax71 = __msa_fmax_w(_absmax71, (v4f32)__msa_and_v((v16u8)_p71, _abs_mask));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                const float absmax0 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax00, _absmax01));
                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                pd[0] = absmax0 / 127.f;
                const float absmax1 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax10, _absmax11));
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[1] = absmax1 / 127.f;
                const float absmax2 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax20, _absmax21));
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                pd[2] = absmax2 / 127.f;
                const float absmax3 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax30, _absmax31));
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                pd[3] = absmax3 / 127.f;
                const float absmax4 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax40, _absmax41));
                const float scale4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
                pd[4] = absmax4 / 127.f;
                const float absmax5 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax50, _absmax51));
                const float scale5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
                pd[5] = absmax5 / 127.f;
                const float absmax6 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax60, _absmax61));
                const float scale6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
                pd[6] = absmax6 / 127.f;
                const float absmax7 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax70, _absmax71));
                const float scale7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
                pd[7] = absmax7 / 127.f;
                pd += 8;

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
                    ((int*)pp)[8] = __msa_copy_s_w((v4i32)float2int8(_p01), 0);
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _s0);
                    _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                    v4f32 _p11 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _s1);
                    _p11 = __msa_fmul_w(_p11, __msa_fill_w_f32(scale1));
                    ((int*)pp)[9] = __msa_copy_s_w((v4i32)float2int8(_p11), 0);
                    v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0 + 16), _s0);
                    _p20 = __msa_fmul_w(_p20, __msa_fill_w_f32(scale2));
                    ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p20), 0);
                    v4f32 _p21 = __msa_fmul_w(bfloat2float_msa(p0 + 20), _s1);
                    _p21 = __msa_fmul_w(_p21, __msa_fill_w_f32(scale2));
                    ((int*)pp)[10] = __msa_copy_s_w((v4i32)float2int8(_p21), 0);
                    v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0 + 24), _s0);
                    _p30 = __msa_fmul_w(_p30, __msa_fill_w_f32(scale3));
                    ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p30), 0);
                    v4f32 _p31 = __msa_fmul_w(bfloat2float_msa(p0 + 28), _s1);
                    _p31 = __msa_fmul_w(_p31, __msa_fill_w_f32(scale3));
                    ((int*)pp)[11] = __msa_copy_s_w((v4i32)float2int8(_p31), 0);
                    v4f32 _p40 = __msa_fmul_w(bfloat2float_msa(p0 + 32), _s0);
                    _p40 = __msa_fmul_w(_p40, __msa_fill_w_f32(scale4));
                    ((int*)pp)[4] = __msa_copy_s_w((v4i32)float2int8(_p40), 0);
                    v4f32 _p41 = __msa_fmul_w(bfloat2float_msa(p0 + 36), _s1);
                    _p41 = __msa_fmul_w(_p41, __msa_fill_w_f32(scale4));
                    ((int*)pp)[12] = __msa_copy_s_w((v4i32)float2int8(_p41), 0);
                    v4f32 _p50 = __msa_fmul_w(bfloat2float_msa(p0 + 40), _s0);
                    _p50 = __msa_fmul_w(_p50, __msa_fill_w_f32(scale5));
                    ((int*)pp)[5] = __msa_copy_s_w((v4i32)float2int8(_p50), 0);
                    v4f32 _p51 = __msa_fmul_w(bfloat2float_msa(p0 + 44), _s1);
                    _p51 = __msa_fmul_w(_p51, __msa_fill_w_f32(scale5));
                    ((int*)pp)[13] = __msa_copy_s_w((v4i32)float2int8(_p51), 0);
                    v4f32 _p60 = __msa_fmul_w(bfloat2float_msa(p0 + 48), _s0);
                    _p60 = __msa_fmul_w(_p60, __msa_fill_w_f32(scale6));
                    ((int*)pp)[6] = __msa_copy_s_w((v4i32)float2int8(_p60), 0);
                    v4f32 _p61 = __msa_fmul_w(bfloat2float_msa(p0 + 52), _s1);
                    _p61 = __msa_fmul_w(_p61, __msa_fill_w_f32(scale6));
                    ((int*)pp)[14] = __msa_copy_s_w((v4i32)float2int8(_p61), 0);
                    v4f32 _p70 = __msa_fmul_w(bfloat2float_msa(p0 + 56), _s0);
                    _p70 = __msa_fmul_w(_p70, __msa_fill_w_f32(scale7));
                    ((int*)pp)[7] = __msa_copy_s_w((v4i32)float2int8(_p70), 0);
                    v4f32 _p71 = __msa_fmul_w(bfloat2float_msa(p0 + 60), _s1);
                    _p71 = __msa_fmul_w(_p71, __msa_fill_w_f32(scale7));
                    ((int*)pp)[15] = __msa_copy_s_w((v4i32)float2int8(_p71), 0);
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

            for (int g = 0; g < block_count; g++)
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

                const float absmax0 = __msa_reduce_fmax_w(_absmax00);
                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                pd[0] = absmax0 / 127.f;
                const float absmax1 = __msa_reduce_fmax_w(_absmax10);
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[1] = absmax1 / 127.f;
                const float absmax2 = __msa_reduce_fmax_w(_absmax20);
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                pd[2] = absmax2 / 127.f;
                const float absmax3 = __msa_reduce_fmax_w(_absmax30);
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                pd[3] = absmax3 / 127.f;
                const float absmax4 = __msa_reduce_fmax_w(_absmax40);
                const float scale4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
                pd[4] = absmax4 / 127.f;
                const float absmax5 = __msa_reduce_fmax_w(_absmax50);
                const float scale5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
                pd[5] = absmax5 / 127.f;
                const float absmax6 = __msa_reduce_fmax_w(_absmax60);
                const float scale6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
                pd[6] = absmax6 / 127.f;
                const float absmax7 = __msa_reduce_fmax_w(_absmax70);
                const float scale7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
                pd[7] = absmax7 / 127.f;
                pd += 8;

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
                    v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _s0);
                    _p20 = __msa_fmul_w(_p20, __msa_fill_w_f32(scale2));
                    ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p20), 0);
                    v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _s0);
                    _p30 = __msa_fmul_w(_p30, __msa_fill_w_f32(scale3));
                    ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p30), 0);
                    v4f32 _p40 = __msa_fmul_w(bfloat2float_msa(p0 + 16), _s0);
                    _p40 = __msa_fmul_w(_p40, __msa_fill_w_f32(scale4));
                    ((int*)pp)[4] = __msa_copy_s_w((v4i32)float2int8(_p40), 0);
                    v4f32 _p50 = __msa_fmul_w(bfloat2float_msa(p0 + 20), _s0);
                    _p50 = __msa_fmul_w(_p50, __msa_fill_w_f32(scale5));
                    ((int*)pp)[5] = __msa_copy_s_w((v4i32)float2int8(_p50), 0);
                    v4f32 _p60 = __msa_fmul_w(bfloat2float_msa(p0 + 24), _s0);
                    _p60 = __msa_fmul_w(_p60, __msa_fill_w_f32(scale6));
                    ((int*)pp)[6] = __msa_copy_s_w((v4i32)float2int8(_p60), 0);
                    v4f32 _p70 = __msa_fmul_w(bfloat2float_msa(p0 + 28), _s0);
                    _p70 = __msa_fmul_w(_p70, __msa_fill_w_f32(scale7));
                    ((int*)pp)[7] = __msa_copy_s_w((v4i32)float2int8(_p70), 0);
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

            for (int g = 0; g < block_count; g++)
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

                float absmax[8];
                __msa_st_w((v4i32)_absmax0, absmax, 0);
                __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                const float scale4 = absmax[4] == 0.f ? 1.f : 127.f / absmax[4];
                const float scale5 = absmax[5] == 0.f ? 1.f : 127.f / absmax[5];
                const float scale6 = absmax[6] == 0.f ? 1.f : 127.f / absmax[6];
                const float scale7 = absmax[7] == 0.f ? 1.f : 127.f / absmax[7];
                pd[0] = absmax[0] / 127.f;
                pd[1] = absmax[1] / 127.f;
                pd[2] = absmax[2] / 127.f;
                pd[3] = absmax[3] / 127.f;
                pd[4] = absmax[4] / 127.f;
                pd[5] = absmax[5] / 127.f;
                pd[6] = absmax[6] / 127.f;
                pd[7] = absmax[7] / 127.f;
                pd += 8;

                v4f32 _scale03 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                v4f32 _scale47 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));
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

            for (int g = 0; g < block_count; g++)
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

                const float absmax0 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax00, _absmax01));
                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                pd[0] = absmax0 / 127.f;
                const float absmax1 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax10, _absmax11));
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[1] = absmax1 / 127.f;
                const float absmax2 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax20, _absmax21));
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                pd[2] = absmax2 / 127.f;
                const float absmax3 = __msa_reduce_fmax_w(__msa_fmax_w(_absmax30, _absmax31));
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                pd[3] = absmax3 / 127.f;
                pd += 4;

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
                    ((int*)pp)[4] = __msa_copy_s_w((v4i32)float2int8(_p01), 0);
                    v4f32 _p10 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _s0);
                    _p10 = __msa_fmul_w(_p10, __msa_fill_w_f32(scale1));
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p10), 0);
                    v4f32 _p11 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _s1);
                    _p11 = __msa_fmul_w(_p11, __msa_fill_w_f32(scale1));
                    ((int*)pp)[5] = __msa_copy_s_w((v4i32)float2int8(_p11), 0);
                    v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0 + 16), _s0);
                    _p20 = __msa_fmul_w(_p20, __msa_fill_w_f32(scale2));
                    ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p20), 0);
                    v4f32 _p21 = __msa_fmul_w(bfloat2float_msa(p0 + 20), _s1);
                    _p21 = __msa_fmul_w(_p21, __msa_fill_w_f32(scale2));
                    ((int*)pp)[6] = __msa_copy_s_w((v4i32)float2int8(_p21), 0);
                    v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0 + 24), _s0);
                    _p30 = __msa_fmul_w(_p30, __msa_fill_w_f32(scale3));
                    ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p30), 0);
                    v4f32 _p31 = __msa_fmul_w(bfloat2float_msa(p0 + 28), _s1);
                    _p31 = __msa_fmul_w(_p31, __msa_fill_w_f32(scale3));
                    ((int*)pp)[7] = __msa_copy_s_w((v4i32)float2int8(_p31), 0);
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

            for (int g = 0; g < block_count; g++)
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

                const float absmax0 = __msa_reduce_fmax_w(_absmax00);
                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                pd[0] = absmax0 / 127.f;
                const float absmax1 = __msa_reduce_fmax_w(_absmax10);
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[1] = absmax1 / 127.f;
                const float absmax2 = __msa_reduce_fmax_w(_absmax20);
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                pd[2] = absmax2 / 127.f;
                const float absmax3 = __msa_reduce_fmax_w(_absmax30);
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                pd[3] = absmax3 / 127.f;
                pd += 4;

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
                    v4f32 _p20 = __msa_fmul_w(bfloat2float_msa(p0 + 8), _s0);
                    _p20 = __msa_fmul_w(_p20, __msa_fill_w_f32(scale2));
                    ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p20), 0);
                    v4f32 _p30 = __msa_fmul_w(bfloat2float_msa(p0 + 12), _s0);
                    _p30 = __msa_fmul_w(_p30, __msa_fill_w_f32(scale3));
                    ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p30), 0);
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

            for (int g = 0; g < block_count; g++)
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

                float absmax[4];
                __msa_st_w((v4i32)_absmax, absmax, 0);
                const float scale0 = absmax[0] == 0.f ? 1.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 1.f : 127.f / absmax[1];
                const float scale2 = absmax[2] == 0.f ? 1.f : 127.f / absmax[2];
                const float scale3 = absmax[3] == 0.f ? 1.f : 127.f / absmax[3];
                pd[0] = absmax[0] / 127.f;
                pd[1] = absmax[1] / 127.f;
                pd[2] = absmax[2] / 127.f;
                pd[3] = absmax[3] / 127.f;
                pd += 4;

                v4f32 _scale = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
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
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[1]);
                    float v2 = bfloat16_to_float32(p0[2]);
                    float v3 = bfloat16_to_float32(p0[3]);
                    const float s = *ps++;
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp[2] = float2int8(v2 * scale2);
                    pp[3] = float2int8(v3 * scale3);
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

            for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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

static void unpack_output_tile_wq_int8_bf16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
#if __mips_msa
    const int c_elempack = C.elempack;
#endif // __mips_msa
    const float* pp = topT;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        v4f32 _c0123 = (v4f32)__msa_fill_w(0);
        v4f32 _c4567 = (v4f32)__msa_fill_w(0);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                _c0123 = __msa_fill_w_f32(c);
                _c4567 = _c0123;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                _c0123 = (v4f32)__msa_ld_w(pC, 0);
                _c4567 = (v4f32)__msa_ld_w(pC + 4, 0);
                if (beta != 1.f)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _c0123 = __msa_fmul_w(_c0123, _beta);
                    _c4567 = __msa_fmul_w(_c4567, _beta);
                }
            }
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 64);
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f4 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f5 = (v4f32)__msa_ld_w(pp + 12, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 16, 0);
            v4f32 _f6 = (v4f32)__msa_ld_w(pp + 20, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 24, 0);
            v4f32 _f7 = (v4f32)__msa_ld_w(pp + 28, 0);

            _f2 = (v4f32)__msa_shf_w((v4i32)_f2, _MSA_SHUFFLE(1, 0, 3, 2));
            _f3 = (v4f32)__msa_shf_w((v4i32)_f3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            _f1 = (v4f32)__msa_shf_w((v4i32)_f1, _MSA_SHUFFLE(2, 1, 0, 3));
            _f2 = (v4f32)__msa_shf_w((v4i32)_f2, _MSA_SHUFFLE(1, 0, 3, 2));
            _f3 = (v4f32)__msa_shf_w((v4i32)_f3, _MSA_SHUFFLE(0, 3, 2, 1));

            _f6 = (v4f32)__msa_shf_w((v4i32)_f6, _MSA_SHUFFLE(1, 0, 3, 2));
            _f7 = (v4f32)__msa_shf_w((v4i32)_f7, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_f4, _f5, _f6, _f7);
            _f5 = (v4f32)__msa_shf_w((v4i32)_f5, _MSA_SHUFFLE(2, 1, 0, 3));
            _f6 = (v4f32)__msa_shf_w((v4i32)_f6, _MSA_SHUFFLE(1, 0, 3, 2));
            _f7 = (v4f32)__msa_shf_w((v4i32)_f7, _MSA_SHUFFLE(0, 3, 2, 1));

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0123, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0123, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0123, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0123, 3));
                    _f4 = __msa_fadd_w(_f4, (v4f32)__msa_splati_w((v4i32)_c4567, 0));
                    _f5 = __msa_fadd_w(_f5, (v4f32)__msa_splati_w((v4i32)_c4567, 1));
                    _f6 = __msa_fadd_w(_f6, (v4f32)__msa_splati_w((v4i32)_c4567, 2));
                    _f7 = __msa_fadd_w(_f7, (v4f32)__msa_splati_w((v4i32)_c4567, 3));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    if (c_elempack == 8)
                    {
                        v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pC + 8, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pC + 16, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pC + 24, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w(_c0, _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w(_c1, _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w(_c2, _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w(_c3, _beta));

                        _c0 = (v4f32)__msa_ld_w(pC + 4, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 12, 0);
                        _c2 = (v4f32)__msa_ld_w(pC + 20, 0);
                        _c3 = (v4f32)__msa_ld_w(pC + 28, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w(_c0, _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w(_c1, _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w(_c2, _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w(_c3, _beta));
                    }
                    else if (c_elempack == 4)
                    {
                        v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pC + 8, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pC + 12, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w(_c0, _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w(_c1, _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w(_c2, _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w(_c3, _beta));

                        const float* pC1 = pC + c_hstep * 4;
                        _c0 = (v4f32)__msa_ld_w(pC1, 0);
                        _c1 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                        _c2 = (v4f32)__msa_ld_w(pC1 + 8, 0);
                        _c3 = (v4f32)__msa_ld_w(pC1 + 12, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w(_c0, _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w(_c1, _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w(_c2, _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w(_c3, _beta));
                    }
                    else // if (c_elempack == 1)
                    {
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 4, 0), _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 5, 0), _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 6, 0), _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 7, 0), _beta));
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4f32 _c = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta);
                    _f0 = __msa_fadd_w(_f0, _c);
                    _f1 = __msa_fadd_w(_f1, _c);
                    _f2 = __msa_fadd_w(_f2, _c);
                    _f3 = __msa_fadd_w(_f3, _c);
                    _f4 = __msa_fadd_w(_f4, _c);
                    _f5 = __msa_fadd_w(_f5, _c);
                    _f6 = __msa_fadd_w(_f6, _c);
                    _f7 = __msa_fadd_w(_f7, _c);
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
                _f6 = __msa_fmul_w(_f6, _alpha);
                _f7 = __msa_fmul_w(_f7, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);
            v8i16 _bf2 = (v8i16)float2bfloat_msa(_f2);
            v8i16 _bf3 = (v8i16)float2bfloat_msa(_f3);
            v8i16 _bf4 = (v8i16)float2bfloat_msa(_f4);
            v8i16 _bf5 = (v8i16)float2bfloat_msa(_f5);
            v8i16 _bf6 = (v8i16)float2bfloat_msa(_f6);
            v8i16 _bf7 = (v8i16)float2bfloat_msa(_f7);

            v4f32 _g0 = (v4f32)__msa_ld_w(pp + 32, 0);
            v4f32 _g4 = (v4f32)__msa_ld_w(pp + 36, 0);
            v4f32 _g1 = (v4f32)__msa_ld_w(pp + 40, 0);
            v4f32 _g5 = (v4f32)__msa_ld_w(pp + 44, 0);
            v4f32 _g2 = (v4f32)__msa_ld_w(pp + 48, 0);
            v4f32 _g6 = (v4f32)__msa_ld_w(pp + 52, 0);
            v4f32 _g3 = (v4f32)__msa_ld_w(pp + 56, 0);
            v4f32 _g7 = (v4f32)__msa_ld_w(pp + 60, 0);
            pp += 64;

            _g2 = (v4f32)__msa_shf_w((v4i32)_g2, _MSA_SHUFFLE(1, 0, 3, 2));
            _g3 = (v4f32)__msa_shf_w((v4i32)_g3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_g0, _g1, _g2, _g3);
            _g1 = (v4f32)__msa_shf_w((v4i32)_g1, _MSA_SHUFFLE(2, 1, 0, 3));
            _g2 = (v4f32)__msa_shf_w((v4i32)_g2, _MSA_SHUFFLE(1, 0, 3, 2));
            _g3 = (v4f32)__msa_shf_w((v4i32)_g3, _MSA_SHUFFLE(0, 3, 2, 1));

            _g6 = (v4f32)__msa_shf_w((v4i32)_g6, _MSA_SHUFFLE(1, 0, 3, 2));
            _g7 = (v4f32)__msa_shf_w((v4i32)_g7, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_g4, _g5, _g6, _g7);
            _g5 = (v4f32)__msa_shf_w((v4i32)_g5, _MSA_SHUFFLE(2, 1, 0, 3));
            _g6 = (v4f32)__msa_shf_w((v4i32)_g6, _MSA_SHUFFLE(1, 0, 3, 2));
            _g7 = (v4f32)__msa_shf_w((v4i32)_g7, _MSA_SHUFFLE(0, 3, 2, 1));

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _g0 = __msa_fadd_w(_g0, (v4f32)__msa_splati_w((v4i32)_c0123, 0));
                    _g1 = __msa_fadd_w(_g1, (v4f32)__msa_splati_w((v4i32)_c0123, 1));
                    _g2 = __msa_fadd_w(_g2, (v4f32)__msa_splati_w((v4i32)_c0123, 2));
                    _g3 = __msa_fadd_w(_g3, (v4f32)__msa_splati_w((v4i32)_c0123, 3));
                    _g4 = __msa_fadd_w(_g4, (v4f32)__msa_splati_w((v4i32)_c4567, 0));
                    _g5 = __msa_fadd_w(_g5, (v4f32)__msa_splati_w((v4i32)_c4567, 1));
                    _g6 = __msa_fadd_w(_g6, (v4f32)__msa_splati_w((v4i32)_c4567, 2));
                    _g7 = __msa_fadd_w(_g7, (v4f32)__msa_splati_w((v4i32)_c4567, 3));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    if (c_elempack == 8)
                    {
                        v4f32 _c0 = (v4f32)__msa_ld_w(pC + 32, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pC + 40, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pC + 48, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pC + 56, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _g0 = __msa_fadd_w(_g0, __msa_fmul_w(_c0, _beta));
                        _g1 = __msa_fadd_w(_g1, __msa_fmul_w(_c1, _beta));
                        _g2 = __msa_fadd_w(_g2, __msa_fmul_w(_c2, _beta));
                        _g3 = __msa_fadd_w(_g3, __msa_fmul_w(_c3, _beta));

                        _c0 = (v4f32)__msa_ld_w(pC + 36, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 44, 0);
                        _c2 = (v4f32)__msa_ld_w(pC + 52, 0);
                        _c3 = (v4f32)__msa_ld_w(pC + 60, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _g4 = __msa_fadd_w(_g4, __msa_fmul_w(_c0, _beta));
                        _g5 = __msa_fadd_w(_g5, __msa_fmul_w(_c1, _beta));
                        _g6 = __msa_fadd_w(_g6, __msa_fmul_w(_c2, _beta));
                        _g7 = __msa_fadd_w(_g7, __msa_fmul_w(_c3, _beta));
                        pC += 64;
                    }
                    else if (c_elempack == 4)
                    {
                        v4f32 _c0 = (v4f32)__msa_ld_w(pC + 16, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pC + 20, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pC + 24, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pC + 28, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _g0 = __msa_fadd_w(_g0, __msa_fmul_w(_c0, _beta));
                        _g1 = __msa_fadd_w(_g1, __msa_fmul_w(_c1, _beta));
                        _g2 = __msa_fadd_w(_g2, __msa_fmul_w(_c2, _beta));
                        _g3 = __msa_fadd_w(_g3, __msa_fmul_w(_c3, _beta));

                        const float* pC1 = pC + c_hstep * 4;
                        _c0 = (v4f32)__msa_ld_w(pC1 + 16, 0);
                        _c1 = (v4f32)__msa_ld_w(pC1 + 20, 0);
                        _c2 = (v4f32)__msa_ld_w(pC1 + 24, 0);
                        _c3 = (v4f32)__msa_ld_w(pC1 + 28, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _g4 = __msa_fadd_w(_g4, __msa_fmul_w(_c0, _beta));
                        _g5 = __msa_fadd_w(_g5, __msa_fmul_w(_c1, _beta));
                        _g6 = __msa_fadd_w(_g6, __msa_fmul_w(_c2, _beta));
                        _g7 = __msa_fadd_w(_g7, __msa_fmul_w(_c3, _beta));
                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        _g0 = __msa_fadd_w(_g0, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                        _g1 = __msa_fadd_w(_g1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep + 4, 0), _beta));
                        _g2 = __msa_fadd_w(_g2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2 + 4, 0), _beta));
                        _g3 = __msa_fadd_w(_g3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3 + 4, 0), _beta));
                        _g4 = __msa_fadd_w(_g4, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 4 + 4, 0), _beta));
                        _g5 = __msa_fadd_w(_g5, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 5 + 4, 0), _beta));
                        _g6 = __msa_fadd_w(_g6, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 6 + 4, 0), _beta));
                        _g7 = __msa_fadd_w(_g7, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 7 + 4, 0), _beta));
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4f32 _c = __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta);
                    pC += 8;
                    _g0 = __msa_fadd_w(_g0, _c);
                    _g1 = __msa_fadd_w(_g1, _c);
                    _g2 = __msa_fadd_w(_g2, _c);
                    _g3 = __msa_fadd_w(_g3, _c);
                    _g4 = __msa_fadd_w(_g4, _c);
                    _g5 = __msa_fadd_w(_g5, _c);
                    _g6 = __msa_fadd_w(_g6, _c);
                    _g7 = __msa_fadd_w(_g7, _c);
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _g0 = __msa_fmul_w(_g0, _alpha);
                _g1 = __msa_fmul_w(_g1, _alpha);
                _g2 = __msa_fmul_w(_g2, _alpha);
                _g3 = __msa_fmul_w(_g3, _alpha);
                _g4 = __msa_fmul_w(_g4, _alpha);
                _g5 = __msa_fmul_w(_g5, _alpha);
                _g6 = __msa_fmul_w(_g6, _alpha);
                _g7 = __msa_fmul_w(_g7, _alpha);
            }

            v8i16 _bg0 = (v8i16)float2bfloat_msa(_g0);
            v8i16 _bg1 = (v8i16)float2bfloat_msa(_g1);
            v8i16 _bg2 = (v8i16)float2bfloat_msa(_g2);
            v8i16 _bg3 = (v8i16)float2bfloat_msa(_g3);
            v8i16 _bg4 = (v8i16)float2bfloat_msa(_g4);
            v8i16 _bg5 = (v8i16)float2bfloat_msa(_g5);
            v8i16 _bg6 = (v8i16)float2bfloat_msa(_g6);
            v8i16 _bg7 = (v8i16)float2bfloat_msa(_g7);

            _bf0 = (v8i16)__msa_ilvr_d((v2i64)_bg0, (v2i64)_bf0);
            _bf1 = (v8i16)__msa_ilvr_d((v2i64)_bg1, (v2i64)_bf1);
            _bf2 = (v8i16)__msa_ilvr_d((v2i64)_bg2, (v2i64)_bf2);
            _bf3 = (v8i16)__msa_ilvr_d((v2i64)_bg3, (v2i64)_bf3);
            _bf4 = (v8i16)__msa_ilvr_d((v2i64)_bg4, (v2i64)_bf4);
            _bf5 = (v8i16)__msa_ilvr_d((v2i64)_bg5, (v2i64)_bf5);
            _bf6 = (v8i16)__msa_ilvr_d((v2i64)_bg6, (v2i64)_bf6);
            _bf7 = (v8i16)__msa_ilvr_d((v2i64)_bg7, (v2i64)_bf7);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __msa_st_h(_bf0, p0, 0);
                    __msa_st_h(_bf1, p0 + 8, 0);
                    __msa_st_h(_bf2, p0 + 16, 0);
                    __msa_st_h(_bf3, p0 + 24, 0);
                    __msa_st_h(_bf4, p0 + 32, 0);
                    __msa_st_h(_bf5, p0 + 40, 0);
                    __msa_st_h(_bf6, p0 + 48, 0);
                    __msa_st_h(_bf7, p0 + 56, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                    __msa_storel_d((v4i32)_bf4, p0 + 16);
                    __msa_storel_d((v4i32)_bf5, p0 + 20);
                    __msa_storel_d((v4i32)_bf6, p0 + 24);
                    __msa_storel_d((v4i32)_bf7, p0 + 28);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf0, (v16i8)_bf0, 8), p1);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf1, (v16i8)_bf1, 8), p1 + 4);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf2, (v16i8)_bf2, 8), p1 + 8);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf3, (v16i8)_bf3, 8), p1 + 12);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf4, (v16i8)_bf4, 8), p1 + 16);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf5, (v16i8)_bf5, 8), p1 + 20);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf6, (v16i8)_bf6, 8), p1 + 24);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf7, (v16i8)_bf7, 8), p1 + 28);
                }
                if (out_elempack == 1)
                {
                    transpose8x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);
                    __msa_st_h(_bf0, p0, 0);
                    __msa_st_h(_bf1, p0 + out_hstep, 0);
                    __msa_st_h(_bf2, p0 + out_hstep * 2, 0);
                    __msa_st_h(_bf3, p0 + out_hstep * 3, 0);
                    __msa_st_h(_bf4, p0 + out_hstep * 4, 0);
                    __msa_st_h(_bf5, p0 + out_hstep * 5, 0);
                    __msa_st_h(_bf6, p0 + out_hstep * 6, 0);
                    __msa_st_h(_bf7, p0 + out_hstep * 7, 0);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose8x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);
                    __msa_st_h(_bf0, p0, 0);
                    __msa_st_h(_bf1, p0 + 8, 0);
                    __msa_st_h(_bf2, p0 + 16, 0);
                    __msa_st_h(_bf3, p0 + 24, 0);
                    __msa_st_h(_bf4, p0 + 32, 0);
                    __msa_st_h(_bf5, p0 + 40, 0);
                    __msa_st_h(_bf6, p0 + 48, 0);
                    __msa_st_h(_bf7, p0 + 56, 0);
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose8x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                    __msa_storel_d((v4i32)_bf4, p0 + 16);
                    __msa_storel_d((v4i32)_bf5, p0 + 20);
                    __msa_storel_d((v4i32)_bf6, p0 + 24);
                    __msa_storel_d((v4i32)_bf7, p0 + 28);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf0, (v16i8)_bf0, 8), p1);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf1, (v16i8)_bf1, 8), p1 + 4);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf2, (v16i8)_bf2, 8), p1 + 8);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf3, (v16i8)_bf3, 8), p1 + 12);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf4, (v16i8)_bf4, 8), p1 + 16);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf5, (v16i8)_bf5, 8), p1 + 20);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf6, (v16i8)_bf6, 8), p1 + 24);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_bf7, (v16i8)_bf7, 8), p1 + 28);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    __msa_st_h(_bf0, p0, 0);
                    __msa_st_h(_bf1, p0 + out_hstep, 0);
                    __msa_st_h(_bf2, p0 + out_hstep * 2, 0);
                    __msa_st_h(_bf3, p0 + out_hstep * 3, 0);
                    __msa_st_h(_bf4, p0 + out_hstep * 4, 0);
                    __msa_st_h(_bf5, p0 + out_hstep * 5, 0);
                    __msa_st_h(_bf6, p0 + out_hstep * 6, 0);
                    __msa_st_h(_bf7, p0 + out_hstep * 7, 0);
                    p0 += 8;
                }
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f4 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f5 = (v4f32)__msa_ld_w(pp + 12, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 16, 0);
            v4f32 _f6 = (v4f32)__msa_ld_w(pp + 20, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 24, 0);
            v4f32 _f7 = (v4f32)__msa_ld_w(pp + 28, 0);
            pp += 32;

            _f2 = (v4f32)__msa_shf_w((v4i32)_f2, _MSA_SHUFFLE(1, 0, 3, 2));
            _f3 = (v4f32)__msa_shf_w((v4i32)_f3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            _f1 = (v4f32)__msa_shf_w((v4i32)_f1, _MSA_SHUFFLE(2, 1, 0, 3));
            _f2 = (v4f32)__msa_shf_w((v4i32)_f2, _MSA_SHUFFLE(1, 0, 3, 2));
            _f3 = (v4f32)__msa_shf_w((v4i32)_f3, _MSA_SHUFFLE(0, 3, 2, 1));

            _f6 = (v4f32)__msa_shf_w((v4i32)_f6, _MSA_SHUFFLE(1, 0, 3, 2));
            _f7 = (v4f32)__msa_shf_w((v4i32)_f7, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_f4, _f5, _f6, _f7);
            _f5 = (v4f32)__msa_shf_w((v4i32)_f5, _MSA_SHUFFLE(2, 1, 0, 3));
            _f6 = (v4f32)__msa_shf_w((v4i32)_f6, _MSA_SHUFFLE(1, 0, 3, 2));
            _f7 = (v4f32)__msa_shf_w((v4i32)_f7, _MSA_SHUFFLE(0, 3, 2, 1));

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0123, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0123, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0123, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0123, 3));
                    _f4 = __msa_fadd_w(_f4, (v4f32)__msa_splati_w((v4i32)_c4567, 0));
                    _f5 = __msa_fadd_w(_f5, (v4f32)__msa_splati_w((v4i32)_c4567, 1));
                    _f6 = __msa_fadd_w(_f6, (v4f32)__msa_splati_w((v4i32)_c4567, 2));
                    _f7 = __msa_fadd_w(_f7, (v4f32)__msa_splati_w((v4i32)_c4567, 3));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    if (c_elempack == 8)
                    {
                        v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pC + 8, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pC + 16, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pC + 24, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w(_c0, _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w(_c1, _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w(_c2, _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w(_c3, _beta));

                        _c0 = (v4f32)__msa_ld_w(pC + 4, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 12, 0);
                        _c2 = (v4f32)__msa_ld_w(pC + 20, 0);
                        _c3 = (v4f32)__msa_ld_w(pC + 28, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w(_c0, _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w(_c1, _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w(_c2, _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w(_c3, _beta));
                        pC += 32;
                    }
                    else if (c_elempack == 4)
                    {
                        v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pC + 8, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pC + 12, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w(_c0, _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w(_c1, _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w(_c2, _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w(_c3, _beta));

                        const float* pC1 = pC + c_hstep * 4;
                        _c0 = (v4f32)__msa_ld_w(pC1, 0);
                        _c1 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                        _c2 = (v4f32)__msa_ld_w(pC1 + 8, 0);
                        _c3 = (v4f32)__msa_ld_w(pC1 + 12, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w(_c0, _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w(_c1, _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w(_c2, _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w(_c3, _beta));
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 4, 0), _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 5, 0), _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 6, 0), _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 7, 0), _beta));
                        pC += 4;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c = (v4f32)__msa_ld_w(pC, 0);
                    pC += 4;
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c);
                    _f1 = __msa_fadd_w(_f1, _c);
                    _f2 = __msa_fadd_w(_f2, _c);
                    _f3 = __msa_fadd_w(_f3, _c);
                    _f4 = __msa_fadd_w(_f4, _c);
                    _f5 = __msa_fadd_w(_f5, _c);
                    _f6 = __msa_fadd_w(_f6, _c);
                    _f7 = __msa_fadd_w(_f7, _c);
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
                _f6 = __msa_fmul_w(_f6, _alpha);
                _f7 = __msa_fmul_w(_f7, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);
            v8i16 _bf2 = (v8i16)float2bfloat_msa(_f2);
            v8i16 _bf3 = (v8i16)float2bfloat_msa(_f3);
            v8i16 _bf4 = (v8i16)float2bfloat_msa(_f4);
            v8i16 _bf5 = (v8i16)float2bfloat_msa(_f5);
            v8i16 _bf6 = (v8i16)float2bfloat_msa(_f6);
            v8i16 _bf7 = (v8i16)float2bfloat_msa(_f7);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf4, p0 + 4);
                    __msa_storel_d((v4i32)_bf1, p0 + 8);
                    __msa_storel_d((v4i32)_bf5, p0 + 12);
                    __msa_storel_d((v4i32)_bf2, p0 + 16);
                    __msa_storel_d((v4i32)_bf6, p0 + 20);
                    __msa_storel_d((v4i32)_bf3, p0 + 24);
                    __msa_storel_d((v4i32)_bf7, p0 + 28);
                }
                if (out_elempack == 4)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                    __msa_storel_d((v4i32)_bf4, p0 + 16);
                    __msa_storel_d((v4i32)_bf5, p0 + 20);
                    __msa_storel_d((v4i32)_bf6, p0 + 24);
                    __msa_storel_d((v4i32)_bf7, p0 + 28);
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    v8i16 _tmpf0 = _bf0;
                    v8i16 _tmpf1 = _bf1;
                    _bf1 = (v8i16)__msa_sldi_b((v16i8)_tmpf0, (v16i8)_tmpf0, 8);
                    _bf2 = _tmpf1;
                    _bf3 = (v8i16)__msa_sldi_b((v16i8)_tmpf1, (v16i8)_tmpf1, 8);
                    transpose4x4_epi16(_bf4, _bf5, _bf6, _bf7);
                    v8i16 _tmpf4 = _bf4;
                    v8i16 _tmpf5 = _bf5;
                    _bf5 = (v8i16)__msa_sldi_b((v16i8)_tmpf4, (v16i8)_tmpf4, 8);
                    _bf6 = _tmpf5;
                    _bf7 = (v8i16)__msa_sldi_b((v16i8)_tmpf5, (v16i8)_tmpf5, 8);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf4, p0 + 4);
                    __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                    __msa_storel_d((v4i32)_bf5, p0 + out_hstep + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + out_hstep * 2);
                    __msa_storel_d((v4i32)_bf6, p0 + out_hstep * 2 + 4);
                    __msa_storel_d((v4i32)_bf3, p0 + out_hstep * 3);
                    __msa_storel_d((v4i32)_bf7, p0 + out_hstep * 3 + 4);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    v8i16 _tmpf0 = _bf0;
                    v8i16 _tmpf1 = _bf1;
                    _bf1 = (v8i16)__msa_sldi_b((v16i8)_tmpf0, (v16i8)_tmpf0, 8);
                    _bf2 = _tmpf1;
                    _bf3 = (v8i16)__msa_sldi_b((v16i8)_tmpf1, (v16i8)_tmpf1, 8);
                    transpose4x4_epi16(_bf4, _bf5, _bf6, _bf7);
                    v8i16 _tmpf4 = _bf4;
                    v8i16 _tmpf5 = _bf5;
                    _bf5 = (v8i16)__msa_sldi_b((v16i8)_tmpf4, (v16i8)_tmpf4, 8);
                    _bf6 = _tmpf5;
                    _bf7 = (v8i16)__msa_sldi_b((v16i8)_tmpf5, (v16i8)_tmpf5, 8);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf4, p0 + 4);
                    __msa_storel_d((v4i32)_bf1, p0 + 8);
                    __msa_storel_d((v4i32)_bf5, p0 + 12);
                    __msa_storel_d((v4i32)_bf2, p0 + 16);
                    __msa_storel_d((v4i32)_bf6, p0 + 20);
                    __msa_storel_d((v4i32)_bf3, p0 + 24);
                    __msa_storel_d((v4i32)_bf7, p0 + 28);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose4x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    v8i16 _tmpf0 = _bf0;
                    v8i16 _tmpf1 = _bf1;
                    _bf1 = (v8i16)__msa_sldi_b((v16i8)_tmpf0, (v16i8)_tmpf0, 8);
                    _bf2 = _tmpf1;
                    _bf3 = (v8i16)__msa_sldi_b((v16i8)_tmpf1, (v16i8)_tmpf1, 8);
                    transpose4x4_epi16(_bf4, _bf5, _bf6, _bf7);
                    v8i16 _tmpf4 = _bf4;
                    v8i16 _tmpf5 = _bf5;
                    _bf5 = (v8i16)__msa_sldi_b((v16i8)_tmpf4, (v16i8)_tmpf4, 8);
                    _bf6 = _tmpf5;
                    _bf7 = (v8i16)__msa_sldi_b((v16i8)_tmpf5, (v16i8)_tmpf5, 8);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                    __msa_storel_d((v4i32)_bf4, p1);
                    __msa_storel_d((v4i32)_bf5, p1 + 4);
                    __msa_storel_d((v4i32)_bf6, p1 + 8);
                    __msa_storel_d((v4i32)_bf7, p1 + 12);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                    __msa_storel_d((v4i32)_bf2, p0 + out_hstep * 2);
                    __msa_storel_d((v4i32)_bf3, p0 + out_hstep * 3);
                    __msa_storel_d((v4i32)_bf4, p0 + out_hstep * 4);
                    __msa_storel_d((v4i32)_bf5, p0 + out_hstep * 5);
                    __msa_storel_d((v4i32)_bf6, p0 + out_hstep * 6);
                    __msa_storel_d((v4i32)_bf7, p0 + out_hstep * 7);
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);
            v4i32 _sum2 = __msa_ld_w(pp + 8, 0);
            v4i32 _sum3 = __msa_ld_w(pp + 12, 0);
            pp += 16;

            v4i32 _sum0e = __msa_shf_w(_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum0o = __msa_shf_w(_sum0, _MSA_SHUFFLE(2, 0, 3, 1));
            v4i32 _sum2e = __msa_shf_w(_sum2, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum2o = __msa_shf_w(_sum2, _MSA_SHUFFLE(2, 0, 3, 1));
            v4i32 _sum4e = __msa_shf_w(_sum1, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum4o = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 0, 3, 1));
            v4i32 _sum6e = __msa_shf_w(_sum3, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum6o = __msa_shf_w(_sum3, _MSA_SHUFFLE(2, 0, 3, 1));

            v4f32 _f0 = (v4f32)__msa_ilvr_w(_sum2o, _sum0e);
            v4f32 _f1 = (v4f32)__msa_ilvr_w(_sum0o, _sum2e);
            v4f32 _f4 = (v4f32)__msa_ilvr_w(_sum6o, _sum4e);
            v4f32 _f5 = (v4f32)__msa_ilvr_w(_sum4o, _sum6e);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f4 = __msa_fadd_w(_f4, _c4567);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                    _f5 = __msa_fadd_w(_f5, _c4567);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0;
                    v4f32 _c1;
                    v4f32 _c4;
                    v4f32 _c5;
                    if (c_elempack == 8)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC, 0);
                        _c4 = (v4f32)__msa_ld_w(pC + 4, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 8, 0);
                        _c5 = (v4f32)__msa_ld_w(pC + 12, 0);
                        pC += 16;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        const float* pC1 = pC + c_hstep * 4;
                        _c4 = (v4f32)__msa_ld_w(pC1, 0);
                        _c5 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                        _c4 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                        _c1 = (v4f32)__msa_set_w(__msa_load_w(pC + 1), __msa_load_w(pC + c_hstep + 1), __msa_load_w(pC + c_hstep * 2 + 1), __msa_load_w(pC + c_hstep * 3 + 1));
                        _c5 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 1), __msa_load_w(pC + c_hstep * 5 + 1), __msa_load_w(pC + c_hstep * 6 + 1), __msa_load_w(pC + c_hstep * 7 + 1));
                        pC += 2;
                    }
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                        _c1 = __msa_fmul_w(_c1, _beta);
                        _c5 = __msa_fmul_w(_c5, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c4);
                    _f1 = __msa_fadd_w(_f1, _c1);
                    _f5 = __msa_fadd_w(_f5, _c5);
                }
                if (broadcast_type_C == 4)
                {
                    float c0 = pC[0];
                    float c1 = pC[1];
                    pC += 2;
                    if (beta != 1.f)
                    {
                        c0 *= beta;
                        c1 *= beta;
                    }
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c0));
                    _f4 = __msa_fadd_w(_f4, __msa_fill_w_f32(c0));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(c1));
                    _f5 = __msa_fadd_w(_f5, __msa_fill_w_f32(c1));
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);
            v8i16 _bf4 = (v8i16)float2bfloat_msa(_f4);
            v8i16 _bf5 = (v8i16)float2bfloat_msa(_f5);

            if (output_transpose)
            {
                __msa_storel_d((v4i32)_bf0, p0);
                __msa_storel_d((v4i32)_bf4, p0 + 4);
                __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                __msa_storel_d((v4i32)_bf5, p0 + out_hstep + 4);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 8)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf4, p0 + 4);
                    __msa_storel_d((v4i32)_bf1, p0 + 8);
                    __msa_storel_d((v4i32)_bf5, p0 + 12);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf4, p1);
                    __msa_storel_d((v4i32)_bf5, p1 + 4);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned int v0 = (unsigned short)__msa_copy_s_h(_bf0, 0) | ((unsigned int)(unsigned short)__msa_copy_s_h(_bf1, 0) << 16);
                    unsigned int v1 = (unsigned short)__msa_copy_s_h(_bf0, 1) | ((unsigned int)(unsigned short)__msa_copy_s_h(_bf1, 1) << 16);
                    unsigned int v2 = (unsigned short)__msa_copy_s_h(_bf0, 2) | ((unsigned int)(unsigned short)__msa_copy_s_h(_bf1, 2) << 16);
                    unsigned int v3 = (unsigned short)__msa_copy_s_h(_bf0, 3) | ((unsigned int)(unsigned short)__msa_copy_s_h(_bf1, 3) << 16);
                    unsigned int v4 = (unsigned short)__msa_copy_s_h(_bf4, 0) | ((unsigned int)(unsigned short)__msa_copy_s_h(_bf5, 0) << 16);
                    unsigned int v5 = (unsigned short)__msa_copy_s_h(_bf4, 1) | ((unsigned int)(unsigned short)__msa_copy_s_h(_bf5, 1) << 16);
                    unsigned int v6 = (unsigned short)__msa_copy_s_h(_bf4, 2) | ((unsigned int)(unsigned short)__msa_copy_s_h(_bf5, 2) << 16);
                    unsigned int v7 = (unsigned short)__msa_copy_s_h(_bf4, 3) | ((unsigned int)(unsigned short)__msa_copy_s_h(_bf5, 3) << 16);
                    memcpy(p0, &v0, 4);
                    memcpy(p0 + out_hstep, &v1, 4);
                    memcpy(p0 + out_hstep * 2, &v2, 4);
                    memcpy(p0 + out_hstep * 3, &v3, 4);
                    memcpy(p0 + out_hstep * 4, &v4, 4);
                    memcpy(p0 + out_hstep * 5, &v5, 4);
                    memcpy(p0 + out_hstep * 6, &v6, 4);
                    memcpy(p0 + out_hstep * 7, &v7, 4);
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f4 = (v4f32)__msa_ld_w(pp + 4, 0);
            pp += 8;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f4 = __msa_fadd_w(_f4, _c4567);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0;
                    v4f32 _c4;
                    if (c_elempack == 8)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC, 0);
                        _c4 = (v4f32)__msa_ld_w(pC + 4, 0);
                        pC += 8;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC, 0);
                        _c4 = (v4f32)__msa_ld_w(pC + c_hstep * 4, 0);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                        _c4 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                        pC++;
                    }
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c4);
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0];
                    pC++;
                    if (beta != 1.f)
                        c *= beta;
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c));
                    _f4 = __msa_fadd_w(_f4, __msa_fill_w_f32(c));
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
            }

            if (output_transpose)
            {
                __msa_storel_d(float2bfloat_msa(_f0), p0);
                __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 8)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p1);
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
                    v8i16 _bf4 = (v8i16)float2bfloat_msa(_f4);
                    p0[0] = (unsigned short)__msa_copy_s_h(_bf0, 0);
                    p0[out_hstep] = (unsigned short)__msa_copy_s_h(_bf0, 1);
                    p0[out_hstep * 2] = (unsigned short)__msa_copy_s_h(_bf0, 2);
                    p0[out_hstep * 3] = (unsigned short)__msa_copy_s_h(_bf0, 3);
                    p0[out_hstep * 4] = (unsigned short)__msa_copy_s_h(_bf4, 0);
                    p0[out_hstep * 5] = (unsigned short)__msa_copy_s_h(_bf4, 1);
                    p0[out_hstep * 6] = (unsigned short)__msa_copy_s_h(_bf4, 2);
                    p0[out_hstep * 7] = (unsigned short)__msa_copy_s_h(_bf4, 3);
                    p0++;
                }
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        v4f32 _c0123 = (v4f32)__msa_fill_w(0);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                _c0123 = __msa_fill_w_f32(c);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                _c0123 = (v4f32)__msa_ld_w(pC, 0);
                if (beta != 1.f)
                    _c0123 = __msa_fmul_w(_c0123, __msa_fill_w_f32(beta));
            }
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 32);
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 12, 0);
            v4f32 _f4 = (v4f32)__msa_ld_w(pp + 16, 0);
            v4f32 _f5 = (v4f32)__msa_ld_w(pp + 20, 0);
            v4f32 _f6 = (v4f32)__msa_ld_w(pp + 24, 0);
            v4f32 _f7 = (v4f32)__msa_ld_w(pp + 28, 0);
            pp += 32;

            _f2 = (v4f32)__msa_shf_w((v4i32)_f2, _MSA_SHUFFLE(1, 0, 3, 2));
            _f3 = (v4f32)__msa_shf_w((v4i32)_f3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            _f1 = (v4f32)__msa_shf_w((v4i32)_f1, _MSA_SHUFFLE(2, 1, 0, 3));
            _f2 = (v4f32)__msa_shf_w((v4i32)_f2, _MSA_SHUFFLE(1, 0, 3, 2));
            _f3 = (v4f32)__msa_shf_w((v4i32)_f3, _MSA_SHUFFLE(0, 3, 2, 1));

            _f6 = (v4f32)__msa_shf_w((v4i32)_f6, _MSA_SHUFFLE(1, 0, 3, 2));
            _f7 = (v4f32)__msa_shf_w((v4i32)_f7, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_f4, _f5, _f6, _f7);
            _f5 = (v4f32)__msa_shf_w((v4i32)_f5, _MSA_SHUFFLE(2, 1, 0, 3));
            _f6 = (v4f32)__msa_shf_w((v4i32)_f6, _MSA_SHUFFLE(1, 0, 3, 2));
            _f7 = (v4f32)__msa_shf_w((v4i32)_f7, _MSA_SHUFFLE(0, 3, 2, 1));

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0123, 0));
                    _f4 = __msa_fadd_w(_f4, (v4f32)__msa_splati_w((v4i32)_c0123, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0123, 1));
                    _f5 = __msa_fadd_w(_f5, (v4f32)__msa_splati_w((v4i32)_c0123, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0123, 2));
                    _f6 = __msa_fadd_w(_f6, (v4f32)__msa_splati_w((v4i32)_c0123, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0123, 3));
                    _f7 = __msa_fadd_w(_f7, (v4f32)__msa_splati_w((v4i32)_c0123, 3));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    if (c_elempack == 4)
                    {
                        v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pC + 8, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pC + 12, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w(_c0, _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w(_c1, _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w(_c2, _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w(_c3, _beta));

                        _c0 = (v4f32)__msa_ld_w(pC + 16, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 20, 0);
                        _c2 = (v4f32)__msa_ld_w(pC + 24, 0);
                        _c3 = (v4f32)__msa_ld_w(pC + 28, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w(_c0, _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w(_c1, _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w(_c2, _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w(_c3, _beta));
                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep + 4, 0), _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2 + 4, 0), _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3 + 4, 0), _beta));
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    v4f32 _c4 = (v4f32)__msa_ld_w(pC + 4, 0);
                    pC += 8;
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f4 = __msa_fadd_w(_f4, _c4);
                    _f5 = __msa_fadd_w(_f5, _c4);
                    _f6 = __msa_fadd_w(_f6, _c4);
                    _f7 = __msa_fadd_w(_f7, _c4);
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
                _f6 = __msa_fmul_w(_f6, _alpha);
                _f7 = __msa_fmul_w(_f7, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);
            v8i16 _bf2 = (v8i16)float2bfloat_msa(_f2);
            v8i16 _bf3 = (v8i16)float2bfloat_msa(_f3);
            v8i16 _bf4 = (v8i16)float2bfloat_msa(_f4);
            v8i16 _bf5 = (v8i16)float2bfloat_msa(_f5);
            v8i16 _bf6 = (v8i16)float2bfloat_msa(_f6);
            v8i16 _bf7 = (v8i16)float2bfloat_msa(_f7);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf4, p0 + 4);
                    __msa_storel_d((v4i32)_bf1, p0 + 8);
                    __msa_storel_d((v4i32)_bf5, p0 + 12);
                    __msa_storel_d((v4i32)_bf2, p0 + 16);
                    __msa_storel_d((v4i32)_bf6, p0 + 20);
                    __msa_storel_d((v4i32)_bf3, p0 + 24);
                    __msa_storel_d((v4i32)_bf7, p0 + 28);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                    __msa_storel_d((v4i32)_bf4, p1);
                    __msa_storel_d((v4i32)_bf5, p1 + 4);
                    __msa_storel_d((v4i32)_bf6, p1 + 8);
                    __msa_storel_d((v4i32)_bf7, p1 + 12);
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    v8i16 _tmpf0 = _bf0;
                    v8i16 _tmpf1 = _bf1;
                    _bf1 = (v8i16)__msa_sldi_b((v16i8)_tmpf0, (v16i8)_tmpf0, 8);
                    _bf2 = _tmpf1;
                    _bf3 = (v8i16)__msa_sldi_b((v16i8)_tmpf1, (v16i8)_tmpf1, 8);
                    transpose4x4_epi16(_bf4, _bf5, _bf6, _bf7);
                    v8i16 _tmpf4 = _bf4;
                    v8i16 _tmpf5 = _bf5;
                    _bf5 = (v8i16)__msa_sldi_b((v16i8)_tmpf4, (v16i8)_tmpf4, 8);
                    _bf6 = _tmpf5;
                    _bf7 = (v8i16)__msa_sldi_b((v16i8)_tmpf5, (v16i8)_tmpf5, 8);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                    __msa_storel_d((v4i32)_bf2, p0 + out_hstep * 2);
                    __msa_storel_d((v4i32)_bf3, p0 + out_hstep * 3);
                    __msa_storel_d((v4i32)_bf4, p0 + out_hstep * 4);
                    __msa_storel_d((v4i32)_bf5, p0 + out_hstep * 5);
                    __msa_storel_d((v4i32)_bf6, p0 + out_hstep * 6);
                    __msa_storel_d((v4i32)_bf7, p0 + out_hstep * 7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    v8i16 _tmpf0 = _bf0;
                    v8i16 _tmpf1 = _bf1;
                    _bf1 = (v8i16)__msa_sldi_b((v16i8)_tmpf0, (v16i8)_tmpf0, 8);
                    _bf2 = _tmpf1;
                    _bf3 = (v8i16)__msa_sldi_b((v16i8)_tmpf1, (v16i8)_tmpf1, 8);
                    transpose4x4_epi16(_bf4, _bf5, _bf6, _bf7);
                    v8i16 _tmpf4 = _bf4;
                    v8i16 _tmpf5 = _bf5;
                    _bf5 = (v8i16)__msa_sldi_b((v16i8)_tmpf4, (v16i8)_tmpf4, 8);
                    _bf6 = _tmpf5;
                    _bf7 = (v8i16)__msa_sldi_b((v16i8)_tmpf5, (v16i8)_tmpf5, 8);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                    __msa_storel_d((v4i32)_bf4, p0 + 16);
                    __msa_storel_d((v4i32)_bf5, p0 + 20);
                    __msa_storel_d((v4i32)_bf6, p0 + 24);
                    __msa_storel_d((v4i32)_bf7, p0 + 28);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf4, p0 + 4);
                    __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                    __msa_storel_d((v4i32)_bf5, p0 + out_hstep + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + out_hstep * 2);
                    __msa_storel_d((v4i32)_bf6, p0 + out_hstep * 2 + 4);
                    __msa_storel_d((v4i32)_bf3, p0 + out_hstep * 3);
                    __msa_storel_d((v4i32)_bf7, p0 + out_hstep * 3 + 4);
                    p0 += 8;
                }
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

            _f2 = (v4f32)__msa_shf_w((v4i32)_f2, _MSA_SHUFFLE(1, 0, 3, 2));
            _f3 = (v4f32)__msa_shf_w((v4i32)_f3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            _f1 = (v4f32)__msa_shf_w((v4i32)_f1, _MSA_SHUFFLE(2, 1, 0, 3));
            _f2 = (v4f32)__msa_shf_w((v4i32)_f2, _MSA_SHUFFLE(1, 0, 3, 2));
            _f3 = (v4f32)__msa_shf_w((v4i32)_f3, _MSA_SHUFFLE(0, 3, 2, 1));

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0123, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0123, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0123, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0123, 3));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    if (c_elempack == 4)
                    {
                        v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pC + 8, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pC + 12, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w(_c0, _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w(_c1, _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w(_c2, _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w(_c3, _beta));
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                        pC += 4;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    pC += 4;
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);
            v8i16 _bf2 = (v8i16)float2bfloat_msa(_f2);
            v8i16 _bf3 = (v8i16)float2bfloat_msa(_f3);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    transpose4x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    v8i16 _tmpf0 = _bf0;
                    v8i16 _tmpf1 = _bf1;
                    _bf1 = (v8i16)__msa_sldi_b((v16i8)_tmpf0, (v16i8)_tmpf0, 8);
                    _bf2 = _tmpf1;
                    _bf3 = (v8i16)__msa_sldi_b((v16i8)_tmpf1, (v16i8)_tmpf1, 8);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 8);
                    __msa_storel_d((v4i32)_bf2, p0 + 16);
                    __msa_storel_d((v4i32)_bf3, p0 + 24);
                }
                if (out_elempack == 4)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    v8i16 _tmpf0 = _bf0;
                    v8i16 _tmpf1 = _bf1;
                    _bf1 = (v8i16)__msa_sldi_b((v16i8)_tmpf0, (v16i8)_tmpf0, 8);
                    _bf2 = _tmpf1;
                    _bf3 = (v8i16)__msa_sldi_b((v16i8)_tmpf1, (v16i8)_tmpf1, 8);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                    __msa_storel_d((v4i32)_bf2, p0 + out_hstep * 2);
                    __msa_storel_d((v4i32)_bf3, p0 + out_hstep * 3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    v8i16 _tmpf0 = _bf0;
                    v8i16 _tmpf1 = _bf1;
                    _bf1 = (v8i16)__msa_sldi_b((v16i8)_tmpf0, (v16i8)_tmpf0, 8);
                    _bf2 = _tmpf1;
                    _bf3 = (v8i16)__msa_sldi_b((v16i8)_tmpf1, (v16i8)_tmpf1, 8);
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                    __msa_storel_d((v4i32)_bf2, p0 + out_hstep * 2);
                    __msa_storel_d((v4i32)_bf3, p0 + out_hstep * 3);
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);
            pp += 8;

            v4i32 _sum0e = __msa_shf_w(_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum0o = __msa_shf_w(_sum0, _MSA_SHUFFLE(2, 0, 3, 1));
            v4i32 _sum1e = __msa_shf_w(_sum1, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum1o = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 0, 3, 1));

            v4f32 _f0 = (v4f32)__msa_ilvr_w(_sum1o, _sum0e);
            v4f32 _f1 = (v4f32)__msa_ilvr_w(_sum0o, _sum1e);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    v4i32 _c0;
                    v4i32 _c1;
                    if (c_elempack == 4)
                    {
                        _c0 = __msa_ld_w(pC, 0);
                        _c1 = __msa_ld_w(pC + 4, 0);
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = __msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                        _c1 = __msa_set_w(__msa_load_w(pC + 1), __msa_load_w(pC + c_hstep + 1), __msa_load_w(pC + c_hstep * 2 + 1), __msa_load_w(pC + c_hstep * 3 + 1));
                        pC += 2;
                    }
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = (v4i32)__msa_fmul_w((v4f32)_c0, _beta);
                        _c1 = (v4i32)__msa_fmul_w((v4f32)_c1, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, (v4f32)_c0);
                    _f1 = __msa_fadd_w(_f1, (v4f32)_c1);
                }
                if (broadcast_type_C == 4)
                {
                    float c0 = pC[0];
                    float c1 = pC[1];
                    pC += 2;
                    if (beta != 1.f)
                    {
                        c0 *= beta;
                        c1 *= beta;
                    }
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c0));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(c1));
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);

            if (output_transpose)
            {
                __msa_storel_d((v4i32)_bf0, p0);
                __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 4)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    v8i16 _tmp = __msa_ilvr_h(_bf1, _bf0);
                    ((int*)p0)[0] = __msa_copy_s_w((v4i32)_tmp, 0);
                    ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w((v4i32)_tmp, 1);
                    ((int*)(p0 + out_hstep * 2))[0] = __msa_copy_s_w((v4i32)_tmp, 2);
                    ((int*)(p0 + out_hstep * 3))[0] = __msa_copy_s_w((v4i32)_tmp, 3);
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    v4i32 _c0;
                    if (c_elempack == 4)
                    {
                        _c0 = __msa_ld_w(pC, 0);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = __msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                        pC++;
                    }
                    if (beta != 1.f)
                        _c0 = (v4i32)__msa_fmul_w((v4f32)_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, (v4f32)_c0);
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0];
                    pC++;
                    if (beta != 1.f)
                        c *= beta;
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c));
                }
            }

            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            if (output_transpose)
            {
                __msa_storel_d(float2bfloat_msa(_f0), p0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 4)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    p0[0] = float32_to_bfloat16(_f0[0]);
                    p0[out_hstep] = float32_to_bfloat16(_f0[1]);
                    p0[out_hstep * 2] = float32_to_bfloat16(_f0[2]);
                    p0[out_hstep * 3] = float32_to_bfloat16(_f0[3]);
                    p0++;
                }
            }
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }
        float c0 = 0.f;
        float c1 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
                c1 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
            }
        }

        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 16);
            v4i32 _s0 = __msa_ld_w(pp, 0);
            v4i32 _s1 = __msa_ld_w(pp + 4, 0);
            v4i32 _s2 = __msa_ld_w(pp + 8, 0);
            v4i32 _s3 = __msa_ld_w(pp + 12, 0);
            pp += 16;

            v4f32 _f0 = (v4f32)__msa_pckev_w(_s1, _s0);
            v4f32 _f1 = (v4f32)__msa_pckev_w(_s3, _s2);
            v4f32 _f2 = (v4f32)__msa_pckod_w(_s1, _s0);
            v4f32 _f3 = (v4f32)__msa_pckod_w(_s3, _s2);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c0));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(c0));
                    _f2 = __msa_fadd_w(_f2, __msa_fill_w_f32(c1));
                    _f3 = __msa_fadd_w(_f3, __msa_fill_w_f32(c1));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC + c_hstep, 0);
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC + c_hstep + 4, 0);
                    pC += 8;
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c1 = __msa_fmul_w(_c1, _beta);
                        _c2 = __msa_fmul_w(_c2, _beta);
                        _c3 = __msa_fmul_w(_c3, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c1);
                    _f2 = __msa_fadd_w(_f2, _c2);
                    _f3 = __msa_fadd_w(_f3, _c3);
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                    pC += 8;
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c1 = __msa_fmul_w(_c1, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c1);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c1);
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);
            v8i16 _bf2 = (v8i16)float2bfloat_msa(_f2);
            v8i16 _bf3 = (v8i16)float2bfloat_msa(_f3);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                    __msa_storel_d((v4i32)_bf2, p0 + 8);
                    __msa_storel_d((v4i32)_bf3, p0 + 12);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf2, p0 + 4);
                    __msa_storel_d((v4i32)_bf1, p1);
                    __msa_storel_d((v4i32)_bf3, p1 + 4);
                }
                if (out_elempack == 1)
                {
                    v8i16 _tmp0 = __msa_ilvr_h(_bf2, _bf0);
                    v8i16 _tmp1 = __msa_ilvr_h(_bf3, _bf1);
                    ((int*)p0)[0] = __msa_copy_s_w((v4i32)_tmp0, 0);
                    ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w((v4i32)_tmp0, 1);
                    ((int*)(p0 + out_hstep * 2))[0] = __msa_copy_s_w((v4i32)_tmp0, 2);
                    ((int*)(p0 + out_hstep * 3))[0] = __msa_copy_s_w((v4i32)_tmp0, 3);
                    ((int*)(p0 + out_hstep * 4))[0] = __msa_copy_s_w((v4i32)_tmp1, 0);
                    ((int*)(p0 + out_hstep * 5))[0] = __msa_copy_s_w((v4i32)_tmp1, 1);
                    ((int*)(p0 + out_hstep * 6))[0] = __msa_copy_s_w((v4i32)_tmp1, 2);
                    ((int*)(p0 + out_hstep * 7))[0] = __msa_copy_s_w((v4i32)_tmp1, 3);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                __msa_storel_d((v4i32)_bf0, p0);
                __msa_storel_d((v4i32)_bf1, p0 + 4);
                __msa_storel_d((v4i32)_bf2, p0 + out_hstep);
                __msa_storel_d((v4i32)_bf3, p0 + out_hstep + 4);
                p0 += 8;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4i32 _s0 = __msa_ld_w(pp, 0);
            v4i32 _s1 = __msa_ld_w(pp + 4, 0);
            pp += 8;

            v4f32 _f0 = (v4f32)__msa_pckev_w(_s1, _s0);
            v4f32 _f1 = (v4f32)__msa_pckod_w(_s1, _s0);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c0));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(c1));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC + c_hstep, 0);
                    pC += 4;
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c1 = __msa_fmul_w(_c1, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c1);
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    pC += 4;
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                }
                if (out_elempack == 1)
                {
                    v8i16 _tmp = __msa_ilvr_h(_bf1, _bf0);
                    ((int*)p0)[0] = __msa_copy_s_w((v4i32)_tmp, 0);
                    ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w((v4i32)_tmp, 1);
                    ((int*)(p0 + out_hstep * 2))[0] = __msa_copy_s_w((v4i32)_tmp, 2);
                    ((int*)(p0 + out_hstep * 3))[0] = __msa_copy_s_w((v4i32)_tmp, 3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                __msa_storel_d((v4i32)_bf0, p0);
                __msa_storel_d((v4i32)_bf1, p0 + out_hstep);
                p0 += 4;
            }
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __mips_msa
            v4f32 _f = (v4f32)__msa_ld_w(pp, 0);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __msa_fadd_w(_f, (v4f32)__msa_set_w(__msa_load_w(&c0), __msa_load_w(&c1), __msa_load_w(&c0), __msa_load_w(&c1)));
                if (broadcast_type_C == 3)
                {
                    v4f32 _c = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + 1), __msa_load_w(pC + c_hstep + 1));
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float cc0 = pC[0];
                    float cc1 = pC[1];
                    if (beta != 1.f)
                    {
                        cc0 *= beta;
                        cc1 *= beta;
                    }
                    _f = __msa_fadd_w(_f, (v4f32)__msa_set_w(__msa_load_w(&cc0), __msa_load_w(&cc0), __msa_load_w(&cc1), __msa_load_w(&cc1)));
                    pC += 2;
                }
            }

            if (alpha != 1.f)
                _f = __msa_fmul_w(_f, __msa_fill_w_f32(alpha));

            v4i32 _f0 = __msa_pckev_w((v4i32)_f, (v4i32)_f);
            v4i32 _f1 = __msa_pckod_w((v4i32)_f, (v4i32)_f);
            v8i16 _bf0 = (v8i16)float2bfloat_msa((v4f32)_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa((v4f32)_f1);

            if (output_transpose)
            {
                v8i16 _tmp = __msa_ilvr_h(_bf1, _bf0);
                ((int*)p0)[0] = __msa_copy_s_w((v4i32)_tmp, 0);
                ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w((v4i32)_tmp, 1);
                p0 += out_hstep * 2;
            }
            else
            {
                ((int*)p0)[0] = __msa_copy_s_w((v4i32)_bf0, 0);
                ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w((v4i32)_bf1, 0);
                p0 += 2;
            }
#else
            float sum00 = pp[0];
            float sum01 = pp[1];
            float sum10 = pp[2];
            float sum11 = pp[3];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    sum00 += c0;
                    sum01 += c1;
                    sum10 += c0;
                    sum11 += c1;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum00 += c0;
                    sum10 += c0;
                    sum01 += c1;
                    sum11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    float c00 = pC[0];
                    float c01 = pC[c_hstep];
                    float c10 = pC[1];
                    float c11 = pC[c_hstep + 1];
                    if (beta != 1.f)
                    {
                        c00 *= beta;
                        c01 *= beta;
                        c10 *= beta;
                        c11 *= beta;
                    }
                    sum00 += c00;
                    sum01 += c01;
                    sum10 += c10;
                    sum11 += c11;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float c0 = pC[0];
                    float c1 = pC[1];
                    if (beta != 1.f)
                    {
                        c0 *= beta;
                        c1 *= beta;
                    }
                    sum00 += c0;
                    sum01 += c0;
                    sum10 += c1;
                    sum11 += c1;
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum01 *= alpha;
                sum10 *= alpha;
                sum11 *= alpha;
            }

            if (output_transpose)
            {
                p0[0] = float32_to_bfloat16(sum00);
                p0[1] = float32_to_bfloat16(sum01);
                p0[out_hstep] = float32_to_bfloat16(sum10);
                p0[out_hstep + 1] = float32_to_bfloat16(sum11);
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum00);
                p0[out_hstep] = float32_to_bfloat16(sum01);
                p0[1] = float32_to_bfloat16(sum10);
                p0[out_hstep + 1] = float32_to_bfloat16(sum11);
                p0 += 2;
            }
#endif // __mips_msa
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    sum0 += c0;
                    sum1 += c1;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += c0;
                    sum1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    float c0 = pC[0];
                    float c1 = pC[c_hstep];
                    pC++;
                    if (beta != 1.f)
                    {
                        c0 *= beta;
                        c1 *= beta;
                    }
                    sum0 += c0;
                    sum1 += c1;
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0];
                    pC++;
                    if (beta != 1.f)
                        c *= beta;
                    sum0 += c;
                    sum1 += c;
                }
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }

            if (output_transpose)
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0[1] = float32_to_bfloat16(sum1);
                p0 += out_hstep;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0[out_hstep] = float32_to_bfloat16(sum1);
                p0++;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[0];
                if (beta != 1.f)
                    c0 *= beta;
            }
        }

        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 8);
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            pp += 8;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c0 = __msa_fill_w_f32(c0);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                    pC += 8;
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c1 = __msa_fmul_w(_c1, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c1);
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
            }

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
            v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + 4);
                }
                if (out_elempack == 4)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                    __msa_storel_d((v4i32)_bf1, p0 + out_hstep * 4);
                }
                if (out_elempack == 1)
                {
                    p0[0] = (unsigned short)__msa_copy_s_h(_bf0, 0);
                    p0[out_hstep] = (unsigned short)__msa_copy_s_h(_bf0, 1);
                    p0[out_hstep * 2] = (unsigned short)__msa_copy_s_h(_bf0, 2);
                    p0[out_hstep * 3] = (unsigned short)__msa_copy_s_h(_bf0, 3);
                    p0[out_hstep * 4] = (unsigned short)__msa_copy_s_h(_bf1, 0);
                    p0[out_hstep * 5] = (unsigned short)__msa_copy_s_h(_bf1, 1);
                    p0[out_hstep * 6] = (unsigned short)__msa_copy_s_h(_bf1, 2);
                    p0[out_hstep * 7] = (unsigned short)__msa_copy_s_h(_bf1, 3);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                __msa_storel_d((v4i32)_bf0, p0);
                __msa_storel_d((v4i32)_bf1, p0 + 4);
                p0 += 8;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            pp += 4;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c0));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    pC += 4;
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                }
            }

            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));

            v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    __msa_storel_d((v4i32)_bf0, p0);
                }
                if (out_elempack == 1)
                {
                    p0[0] = (unsigned short)__msa_copy_s_h(_bf0, 0);
                    p0[out_hstep] = (unsigned short)__msa_copy_s_h(_bf0, 1);
                    p0[out_hstep * 2] = (unsigned short)__msa_copy_s_h(_bf0, 2);
                    p0[out_hstep * 3] = (unsigned short)__msa_copy_s_h(_bf0, 3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                __msa_storel_d((v4i32)_bf0, p0);
                p0 += 4;
            }
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    sum0 += c0;
                    sum1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += c0;
                    sum1 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    float c0 = pC[0];
                    float c1 = pC[1];
                    if (beta != 1.f)
                    {
                        c0 *= beta;
                        c1 *= beta;
                    }
                    sum0 += c0;
                    sum1 += c1;
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }

            if (output_transpose)
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0[out_hstep] = float32_to_bfloat16(sum1);
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0[1] = float32_to_bfloat16(sum1);
                p0 += 2;
            }
            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = *pp++;
            if (pC)
            {
                float c = 0.f;
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    c = c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    c = pC[0];
                    pC++;
                }
                if ((broadcast_type_C == 3 || broadcast_type_C == 4) && beta != 1.f)
                    c *= beta;
                sum0 += c;
            }

            if (alpha != 1.f)
                sum0 *= alpha;
            if (output_transpose)
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0 += out_hstep;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0++;
            }
        }
    }
}
