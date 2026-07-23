// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
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
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;
            const unsigned short* p2 = p1 + A_hstep;
            const unsigned short* p3 = p2 + A_hstep;

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
    for (; ii + 3 < max_ii; ii += 4)
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
                    const unsigned short* p1 = p0 + A_hstep;
                    const unsigned short* p2 = p1 + A_hstep;
                    const unsigned short* p3 = p2 + A_hstep;
                    v4f32 _p0 = bfloat2float_msa(p0);
                    v4f32 _p1 = bfloat2float_msa(p1);
                    v4f32 _p2 = bfloat2float_msa(p2);
                    v4f32 _p3 = bfloat2float_msa(p3);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);

                    v4f32 _p4 = bfloat2float_msa(p0 + 4);
                    v4f32 _p5 = bfloat2float_msa(p1 + 4);
                    v4f32 _p6 = bfloat2float_msa(p2 + 4);
                    v4f32 _p7 = bfloat2float_msa(p3 + 4);
                    transpose4x4_ps(_p4, _p5, _p6, _p7);
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
                    p0 = p3 + A_hstep;
                }
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _p0 = bfloat2float_msa(p0);
                    v4f32 _p1 = bfloat2float_msa(p0 + 4);
                    v4f32 _scale0123 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                    v4f32 _scale4567 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));
                    v16i8 _q0 = float2int8(__msa_fmul_w(_p0, _scale0123));
                    v16i8 _q1 = float2int8(__msa_fmul_w(_p1, _scale4567));
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                    ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                    pp += 8;
                    p0 += A_hstep;
                }
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
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

                v4f32 _scale0 = __msa_fill_w_f32(scale0);
                v4f32 _scale1 = __msa_fill_w_f32(scale1);
                v4f32 _scale2 = __msa_fill_w_f32(scale2);
                v4f32 _scale3 = __msa_fill_w_f32(scale3);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned short* p1 = p0 + A_hstep;
                    const unsigned short* p2 = p1 + A_hstep;
                    const unsigned short* p3 = p2 + A_hstep;
                    v4f32 _p0 = bfloat2float_msa(p0);
                    v4f32 _p1 = bfloat2float_msa(p1);
                    v4f32 _p2 = bfloat2float_msa(p2);
                    v4f32 _p3 = bfloat2float_msa(p3);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);
                    _p0 = __msa_fmul_w(_p0, _scale0);
                    _p1 = __msa_fmul_w(_p1, _scale1);
                    _p2 = __msa_fmul_w(_p2, _scale2);
                    _p3 = __msa_fmul_w(_p3, _scale3);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
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
#endif // __mips_msa
        for (; ii + 1 < max_ii; ii += 2)
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
        for (; ii < max_ii; ii++)
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
        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
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
                const unsigned short* p1 = p0 + A_hstep;
                const unsigned short* p2 = p1 + A_hstep;
                const unsigned short* p3 = p2 + A_hstep;
                v4f32 _p0 = bfloat2float_msa(p0);
                v4f32 _p1 = bfloat2float_msa(p1);
                v4f32 _p2 = bfloat2float_msa(p2);
                v4f32 _p3 = bfloat2float_msa(p3);
                transpose4x4_ps(_p0, _p1, _p2, _p3);

                v4f32 _p4 = bfloat2float_msa(p0 + 4);
                v4f32 _p5 = bfloat2float_msa(p1 + 4);
                v4f32 _p6 = bfloat2float_msa(p2 + 4);
                v4f32 _p7 = bfloat2float_msa(p3 + 4);
                transpose4x4_ps(_p4, _p5, _p6, _p7);
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
                p0 = p3 + A_hstep;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                const float s = *ps++;
                v4f32 _p0 = __msa_fmul_w(bfloat2float_msa(p0), __msa_fill_w_f32(s));
                v4f32 _p1 = __msa_fmul_w(bfloat2float_msa(p0 + 4), __msa_fill_w_f32(s));
                v4f32 _scale0123 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                v4f32 _scale4567 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));
                v16i8 _q0 = float2int8(__msa_fmul_w(_p0, _scale0123));
                v16i8 _q1 = float2int8(__msa_fmul_w(_p1, _scale4567));
                ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
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

            v4f32 _scale0 = __msa_fill_w_f32(scale0);
            v4f32 _scale1 = __msa_fill_w_f32(scale1);
            v4f32 _scale2 = __msa_fill_w_f32(scale2);
            v4f32 _scale3 = __msa_fill_w_f32(scale3);
            kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                const unsigned short* p1 = p0 + A_hstep;
                const unsigned short* p2 = p1 + A_hstep;
                const unsigned short* p3 = p2 + A_hstep;
                v4f32 _p0 = bfloat2float_msa(p0);
                v4f32 _p1 = bfloat2float_msa(p1);
                v4f32 _p2 = bfloat2float_msa(p2);
                v4f32 _p3 = bfloat2float_msa(p3);
                transpose4x4_ps(_p0, _p1, _p2, _p3);
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
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
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
    for (; ii < max_ii; ii++)
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
