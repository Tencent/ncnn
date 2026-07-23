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
#if __loongarch_sx
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
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
                const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vldi(0);
                __m128 _absmax1 = (__m128)__lsx_vldi(0);
                __m128 _absmax2 = (__m128)__lsx_vldi(0);
                __m128 _absmax3 = (__m128)__lsx_vldi(0);
                __m128 _absmax4 = (__m128)__lsx_vldi(0);
                __m128 _absmax5 = (__m128)__lsx_vldi(0);
                __m128 _absmax6 = (__m128)__lsx_vldi(0);
                __m128 _absmax7 = (__m128)__lsx_vldi(0);
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
                    __m128 _p0 = bfloat2float_lsx(p0a);
                    __m128 _p1 = bfloat2float_lsx(p1a);
                    __m128 _p2 = bfloat2float_lsx(p2a);
                    __m128 _p3 = bfloat2float_lsx(p3a);
                    __m128 _p4 = bfloat2float_lsx(p4a);
                    __m128 _p5 = bfloat2float_lsx(p5a);
                    __m128 _p6 = bfloat2float_lsx(p6a);
                    __m128 _p7 = bfloat2float_lsx(p7a);
                    _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                    _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                    _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)_p2, _abs_mask));
                    _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)_p3, _abs_mask));
                    _absmax4 = __lsx_vfmax_s(_absmax4, (__m128)__lsx_vand_v((__m128i)_p4, _abs_mask));
                    _absmax5 = __lsx_vfmax_s(_absmax5, (__m128)__lsx_vand_v((__m128i)_p5, _abs_mask));
                    _absmax6 = __lsx_vfmax_s(_absmax6, (__m128)__lsx_vand_v((__m128i)_p6, _abs_mask));
                    _absmax7 = __lsx_vfmax_s(_absmax7, (__m128)__lsx_vand_v((__m128i)_p7, _abs_mask));
                    p0a += 4;
                    p1a += 4;
                    p2a += 4;
                    p3a += 4;
                    p4a += 4;
                    p5a += 4;
                    p6a += 4;
                    p7a += 4;
                }
                float absmax0 = __lsx_reduce_fmax_s(_absmax0);
                float absmax1 = __lsx_reduce_fmax_s(_absmax1);
                float absmax2 = __lsx_reduce_fmax_s(_absmax2);
                float absmax3 = __lsx_reduce_fmax_s(_absmax3);
                float absmax4 = __lsx_reduce_fmax_s(_absmax4);
                float absmax5 = __lsx_reduce_fmax_s(_absmax5);
                float absmax6 = __lsx_reduce_fmax_s(_absmax6);
                float absmax7 = __lsx_reduce_fmax_s(_absmax7);
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

                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                pd += 8;

                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
                const float scale4 = absmax4 == 0.f ? 0.f : 127.f / absmax4;
                const float scale5 = absmax5 == 0.f ? 0.f : 127.f / absmax5;
                const float scale6 = absmax6 == 0.f ? 0.f : 127.f / absmax6;
                const float scale7 = absmax7 == 0.f ? 0.f : 127.f / absmax7;
                __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
                __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
                __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
                __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
                __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
                __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0);
                    __m128 _p1 = bfloat2float_lsx(p1);
                    __m128 _p2 = bfloat2float_lsx(p2);
                    __m128 _p3 = bfloat2float_lsx(p3);
                    __m128 _p4 = bfloat2float_lsx(p4);
                    __m128 _p5 = bfloat2float_lsx(p5);
                    __m128 _p6 = bfloat2float_lsx(p6);
                    __m128 _p7 = bfloat2float_lsx(p7);
                    *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                    *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                    *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_p4, _scale4), __lsx_vfmul_s(_p5, _scale5));
                    *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_p6, _scale6), __lsx_vfmul_s(_p7, _scale7));
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
                    pp[0] = float2int8(bfloat16_to_float32(*p0++) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(*p1++) * scale1);
                    pp[2] = float2int8(bfloat16_to_float32(*p2++) * scale2);
                    pp[3] = float2int8(bfloat16_to_float32(*p3++) * scale3);
                    pp[4] = float2int8(bfloat16_to_float32(*p4++) * scale4);
                    pp[5] = float2int8(bfloat16_to_float32(*p5++) * scale5);
                    pp[6] = float2int8(bfloat16_to_float32(*p6++) * scale6);
                    pp[7] = float2int8(bfloat16_to_float32(*p7++) * scale7);
                    pp += 8;
                }
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;
            const unsigned short* p2 = p1 + A_hstep;
            const unsigned short* p3 = p2 + A_hstep;
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                __m128 _absmax0 = (__m128)__lsx_vldi(0);
                __m128 _absmax1 = (__m128)__lsx_vldi(0);
                __m128 _absmax2 = (__m128)__lsx_vldi(0);
                __m128 _absmax3 = (__m128)__lsx_vldi(0);
                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const unsigned short* p2a = p2;
                const unsigned short* p3a = p3;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0a);
                    __m128 _p1 = bfloat2float_lsx(p1a);
                    __m128 _p2 = bfloat2float_lsx(p2a);
                    __m128 _p3 = bfloat2float_lsx(p3a);
                    _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                    _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                    _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)_p2, _abs_mask));
                    _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)_p3, _abs_mask));
                    p0a += 4;
                    p1a += 4;
                    p2a += 4;
                    p3a += 4;
                }
                float absmax0 = __lsx_reduce_fmax_s(_absmax0);
                float absmax1 = __lsx_reduce_fmax_s(_absmax1);
                float absmax2 = __lsx_reduce_fmax_s(_absmax2);
                float absmax3 = __lsx_reduce_fmax_s(_absmax3);
                for (; kk < max_kk0; kk++)
                {
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(*p0a++)));
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(*p1a++)));
                    absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(*p2a++)));
                    absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(*p3a++)));
                }
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd += 4;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
                __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
                __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0);
                    __m128 _p1 = bfloat2float_lsx(p1);
                    __m128 _p2 = bfloat2float_lsx(p2);
                    __m128 _p3 = bfloat2float_lsx(p3);
                    *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                    *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    pp[0] = float2int8(bfloat16_to_float32(*p0++) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(*p1++) * scale1);
                    pp[2] = float2int8(bfloat16_to_float32(*p2++) * scale2);
                    pp[3] = float2int8(bfloat16_to_float32(*p3++) * scale3);
                    pp += 4;
                }
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;
#if __loongarch_sx
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                __m128 _absmax0 = (__m128)__lsx_vldi(0);
                __m128 _absmax1 = (__m128)__lsx_vldi(0);
                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0a);
                    __m128 _p1 = bfloat2float_lsx(p1a);
                    _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                    _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                    p0a += 4;
                    p1a += 4;
                }
                float absmax0 = __lsx_reduce_fmax_s(_absmax0);
                float absmax1 = __lsx_reduce_fmax_s(_absmax1);
                for (; kk < max_kk0; kk++)
                {
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(*p0a++)));
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(*p1a++)));
                }
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0);
                    __m128 _p1 = bfloat2float_lsx(p1);
                    *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    pp[0] = float2int8(bfloat16_to_float32(*p0++) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(*p1++) * scale1);
                    pp += 2;
                }
            }
#else
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(*p0a++)));
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(*p1a++)));
                }
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale0);
                    pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale0);
                    pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale0);
                    pp[4] = float2int8(bfloat16_to_float32(p1[0]) * scale1);
                    pp[5] = float2int8(bfloat16_to_float32(p1[1]) * scale1);
                    pp[6] = float2int8(bfloat16_to_float32(p1[2]) * scale1);
                    pp[7] = float2int8(bfloat16_to_float32(p1[3]) * scale1);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    pp[0] = float2int8(bfloat16_to_float32(*p0++) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(*p1++) * scale1);
                    pp += 2;
                }
            }
#endif // __loongarch_sx
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax = 0.f;
                const unsigned short* p0a = p0;
                int kk = 0;
#if __loongarch_sx
                const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
#if __loongarch_asx
                const __m256i _abs_mask256 = __lasx_xvreplgr2vr_w(0x7fffffff);
                __m256 _absmax256 = (__m256)__lasx_xvldi(0);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _p = bfloat2float_lasx((const __m128i*)p0a);
                    _p = (__m256)__lasx_xvand_v((__m256i)_p, _abs_mask256);
                    _absmax256 = __lasx_xvfmax_s(_absmax256, _p);
                    p0a += 8;
                }
                absmax = __lasx_reduce_fmax_s(_absmax256);
#endif // __loongarch_asx
                __m128 _absmax128 = __lsx_vreplfr2vr_s(absmax);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p = bfloat2float_lsx(p0a);
                    _p = (__m128)__lsx_vand_v((__m128i)_p, _abs_mask);
                    _absmax128 = __lsx_vfmax_s(_absmax128, _p);
                    p0a += 4;
                }
                absmax = __lsx_reduce_fmax_s(_absmax128);
#endif // __loongarch_sx
                for (; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0a++);
                    absmax = std::max(absmax, fabsf(v));
                }

                if (absmax == 0.f)
                {
                    *pd++ = 0.f;
                    for (int kk = 0; kk < max_kk0; kk++)
                        *pp++ = 0;
                    p0 += max_kk0;
                    continue;
                }

                const float scale = 127.f / absmax;
                *pd++ = absmax / 127.f;
                kk = 0;
#if __loongarch_sx
#if __loongarch_asx
                __m256 _scale256 = (__m256)__lasx_xvreplfr2vr_s(scale);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _p = bfloat2float_lasx((const __m128i*)p0);
                    _p = __lasx_xvfmul_s(_p, _scale256);
                    __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p)), pp, 0, 0);
                    pp += 8;
                    p0 += 8;
                }
#endif // __loongarch_asx
                __m128 _scale128 = __lsx_vreplfr2vr_s(scale);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p = bfloat2float_lsx(p0);
                    _p = __lsx_vfmul_s(_p, _scale128);
                    __lsx_vstelm_w(float2int8(_p), pp, 0, 0);
                    pp += 4;
                    p0 += 4;
                }
#endif // __loongarch_sx
                for (; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0++);
                    *pp++ = float2int8(v * scale);
                }
            }
        }

        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
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
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            __m128 _absmax2 = (__m128)__lsx_vldi(0);
            __m128 _absmax3 = (__m128)__lsx_vldi(0);
            __m128 _absmax4 = (__m128)__lsx_vldi(0);
            __m128 _absmax5 = (__m128)__lsx_vldi(0);
            __m128 _absmax6 = (__m128)__lsx_vldi(0);
            __m128 _absmax7 = (__m128)__lsx_vldi(0);
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
                __m128 _p0 = bfloat2float_lsx(p0a);
                __m128 _p1 = bfloat2float_lsx(p1a);
                __m128 _p2 = bfloat2float_lsx(p2a);
                __m128 _p3 = bfloat2float_lsx(p3a);
                __m128 _p4 = bfloat2float_lsx(p4a);
                __m128 _p5 = bfloat2float_lsx(p5a);
                __m128 _p6 = bfloat2float_lsx(p6a);
                __m128 _p7 = bfloat2float_lsx(p7a);
                __m128 _s = (__m128)__lsx_vld(psa, 0);
                _p0 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p0, _abs_mask), _s);
                _p1 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p1, _abs_mask), _s);
                _p2 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p2, _abs_mask), _s);
                _p3 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p3, _abs_mask), _s);
                _p4 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p4, _abs_mask), _s);
                _p5 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p5, _abs_mask), _s);
                _p6 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p6, _abs_mask), _s);
                _p7 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p7, _abs_mask), _s);
                _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
                _absmax2 = __lsx_vfmax_s(_absmax2, _p2);
                _absmax3 = __lsx_vfmax_s(_absmax3, _p3);
                _absmax4 = __lsx_vfmax_s(_absmax4, _p4);
                _absmax5 = __lsx_vfmax_s(_absmax5, _p5);
                _absmax6 = __lsx_vfmax_s(_absmax6, _p6);
                _absmax7 = __lsx_vfmax_s(_absmax7, _p7);
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
            float absmax0 = __lsx_reduce_fmax_s(_absmax0);
            float absmax1 = __lsx_reduce_fmax_s(_absmax1);
            float absmax2 = __lsx_reduce_fmax_s(_absmax2);
            float absmax3 = __lsx_reduce_fmax_s(_absmax3);
            float absmax4 = __lsx_reduce_fmax_s(_absmax4);
            float absmax5 = __lsx_reduce_fmax_s(_absmax5);
            float absmax6 = __lsx_reduce_fmax_s(_absmax6);
            float absmax7 = __lsx_reduce_fmax_s(_absmax7);
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

            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd[4] = absmax4 / 127.f;
            pd[5] = absmax5 / 127.f;
            pd[6] = absmax6 / 127.f;
            pd[7] = absmax7 / 127.f;
            pd += 8;

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
            const float scale4 = absmax4 == 0.f ? 0.f : 127.f / absmax4;
            const float scale5 = absmax5 == 0.f ? 0.f : 127.f / absmax5;
            const float scale6 = absmax6 == 0.f ? 0.f : 127.f / absmax6;
            const float scale7 = absmax7 == 0.f ? 0.f : 127.f / absmax7;
            __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
            __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
            __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
            __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
            kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p0 = bfloat2float_lsx(p0);
                __m128 _p1 = bfloat2float_lsx(p1);
                __m128 _p2 = bfloat2float_lsx(p2);
                __m128 _p3 = bfloat2float_lsx(p3);
                __m128 _p4 = bfloat2float_lsx(p4);
                __m128 _p5 = bfloat2float_lsx(p5);
                __m128 _p6 = bfloat2float_lsx(p6);
                __m128 _p7 = bfloat2float_lsx(p7);
                __m128 _s = (__m128)__lsx_vld(ps, 0);
                _p0 = __lsx_vfmul_s(_p0, _s);
                _p1 = __lsx_vfmul_s(_p1, _s);
                _p2 = __lsx_vfmul_s(_p2, _s);
                _p3 = __lsx_vfmul_s(_p3, _s);
                _p4 = __lsx_vfmul_s(_p4, _s);
                _p5 = __lsx_vfmul_s(_p5, _s);
                _p6 = __lsx_vfmul_s(_p6, _s);
                _p7 = __lsx_vfmul_s(_p7, _s);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_p4, _scale4), __lsx_vfmul_s(_p5, _scale5));
                *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_p6, _scale6), __lsx_vfmul_s(_p7, _scale7));
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
                pp[0] = float2int8(bfloat16_to_float32(*p0++) * s * scale0);
                pp[1] = float2int8(bfloat16_to_float32(*p1++) * s * scale1);
                pp[2] = float2int8(bfloat16_to_float32(*p2++) * s * scale2);
                pp[3] = float2int8(bfloat16_to_float32(*p3++) * s * scale3);
                pp[4] = float2int8(bfloat16_to_float32(*p4++) * s * scale4);
                pp[5] = float2int8(bfloat16_to_float32(*p5++) * s * scale5);
                pp[6] = float2int8(bfloat16_to_float32(*p6++) * s * scale6);
                pp[7] = float2int8(bfloat16_to_float32(*p7++) * s * scale7);
                pp += 8;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const unsigned short* p1 = p0 + A_hstep;
        const unsigned short* p2 = p1 + A_hstep;
        const unsigned short* p3 = p2 + A_hstep;
        const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            __m128 _absmax2 = (__m128)__lsx_vldi(0);
            __m128 _absmax3 = (__m128)__lsx_vldi(0);
            const unsigned short* p0a = p0;
            const unsigned short* p1a = p1;
            const unsigned short* p2a = p2;
            const unsigned short* p3a = p3;
            const float* psa = ps;
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p0 = bfloat2float_lsx(p0a);
                __m128 _p1 = bfloat2float_lsx(p1a);
                __m128 _p2 = bfloat2float_lsx(p2a);
                __m128 _p3 = bfloat2float_lsx(p3a);
                __m128 _s = (__m128)__lsx_vld(psa, 0);
                _p0 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p0, _abs_mask), _s);
                _p1 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p1, _abs_mask), _s);
                _p2 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p2, _abs_mask), _s);
                _p3 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p3, _abs_mask), _s);
                _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
                _absmax2 = __lsx_vfmax_s(_absmax2, _p2);
                _absmax3 = __lsx_vfmax_s(_absmax3, _p3);
                p0a += 4;
                p1a += 4;
                p2a += 4;
                p3a += 4;
                psa += 4;
            }
            float absmax0 = __lsx_reduce_fmax_s(_absmax0);
            float absmax1 = __lsx_reduce_fmax_s(_absmax1);
            float absmax2 = __lsx_reduce_fmax_s(_absmax2);
            float absmax3 = __lsx_reduce_fmax_s(_absmax3);
            for (; kk < max_kk0; kk++)
            {
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(*p0a++)) * s);
                absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(*p1a++)) * s);
                absmax2 = std::max(absmax2, fabsf(bfloat16_to_float32(*p2a++)) * s);
                absmax3 = std::max(absmax3, fabsf(bfloat16_to_float32(*p3a++)) * s);
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd += 4;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
            __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p0 = bfloat2float_lsx(p0);
                __m128 _p1 = bfloat2float_lsx(p1);
                __m128 _p2 = bfloat2float_lsx(p2);
                __m128 _p3 = bfloat2float_lsx(p3);
                __m128 _s = (__m128)__lsx_vld(ps, 0);
                _p0 = __lsx_vfmul_s(_p0, _s);
                _p1 = __lsx_vfmul_s(_p1, _s);
                _p2 = __lsx_vfmul_s(_p2, _s);
                _p3 = __lsx_vfmul_s(_p3, _s);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                const float s = *ps++;
                pp[0] = float2int8(bfloat16_to_float32(*p0++) * s * scale0);
                pp[1] = float2int8(bfloat16_to_float32(*p1++) * s * scale1);
                pp[2] = float2int8(bfloat16_to_float32(*p2++) * s * scale2);
                pp[3] = float2int8(bfloat16_to_float32(*p3++) * s * scale3);
                pp += 4;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const unsigned short* p1 = p0 + A_hstep;
#if __loongarch_sx
        const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            const unsigned short* p0a = p0;
            const unsigned short* p1a = p1;
            const float* psa = ps;
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p0 = bfloat2float_lsx(p0a);
                __m128 _p1 = bfloat2float_lsx(p1a);
                __m128 _s = (__m128)__lsx_vld(psa, 0);
                _p0 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p0, _abs_mask), _s);
                _p1 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p1, _abs_mask), _s);
                _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
                p0a += 4;
                p1a += 4;
                psa += 4;
            }
            float absmax0 = __lsx_reduce_fmax_s(_absmax0);
            float absmax1 = __lsx_reduce_fmax_s(_absmax1);
            for (; kk < max_kk0; kk++)
            {
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(*p0a++)) * s);
                absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(*p1a++)) * s);
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p0 = bfloat2float_lsx(p0);
                __m128 _p1 = bfloat2float_lsx(p1);
                __m128 _s = (__m128)__lsx_vld(ps, 0);
                _p0 = __lsx_vfmul_s(_p0, _s);
                _p1 = __lsx_vfmul_s(_p1, _s);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                pp += 8;
                p0 += 4;
                p1 += 4;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                const float s = *ps++;
                pp[0] = float2int8(bfloat16_to_float32(*p0++) * s * scale0);
                pp[1] = float2int8(bfloat16_to_float32(*p1++) * s * scale1);
                pp += 2;
            }
        }
#else
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
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(*p0a++)) * s);
                absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(*p1a++)) * s);
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                const float s0 = ps[0];
                const float s1 = ps[1];
                const float s2 = ps[2];
                const float s3 = ps[3];
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * s0 * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * s1 * scale0);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * s2 * scale0);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * s3 * scale0);
                pp[4] = float2int8(bfloat16_to_float32(p1[0]) * s0 * scale1);
                pp[5] = float2int8(bfloat16_to_float32(p1[1]) * s1 * scale1);
                pp[6] = float2int8(bfloat16_to_float32(p1[2]) * s2 * scale1);
                pp[7] = float2int8(bfloat16_to_float32(p1[3]) * s3 * scale1);
                pp += 8;
                p0 += 4;
                p1 += 4;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                const float s = *ps++;
                pp[0] = float2int8(bfloat16_to_float32(*p0++) * s * scale0);
                pp[1] = float2int8(bfloat16_to_float32(*p1++) * s * scale1);
                pp += 2;
            }
        }
#endif // __loongarch_sx
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax = 0.f;
            const unsigned short* p0a = p0;
            const float* psa = ps;
            int kk = 0;
#if __loongarch_sx
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
#if __loongarch_asx
            const __m256i _abs_mask256 = __lasx_xvreplgr2vr_w(0x7fffffff);
            __m256 _absmax256 = (__m256)__lasx_xvldi(0);
            for (; kk + 7 < max_kk0; kk += 8)
            {
                __m256 _p = bfloat2float_lasx((const __m128i*)p0a);
                _p = __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)_p, _abs_mask256), (__m256)__lasx_xvld(psa, 0));
                _absmax256 = __lasx_xvfmax_s(_absmax256, _p);
                p0a += 8;
                psa += 8;
            }
            absmax = __lasx_reduce_fmax_s(_absmax256);
#endif // __loongarch_asx
            __m128 _absmax128 = __lsx_vreplfr2vr_s(absmax);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p = bfloat2float_lsx(p0a);
                _p = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p, _abs_mask), (__m128)__lsx_vld(psa, 0));
                _absmax128 = __lsx_vfmax_s(_absmax128, _p);
                p0a += 4;
                psa += 4;
            }
            absmax = __lsx_reduce_fmax_s(_absmax128);
#endif // __loongarch_sx
            for (; kk < max_kk0; kk++)
            {
                float v = bfloat16_to_float32(*p0a++);
                v = fabsf(v) * *psa++;
                absmax = std::max(absmax, v);
            }

            if (absmax == 0.f)
            {
                *pd++ = 0.f;
                for (int kk = 0; kk < max_kk0; kk++)
                    *pp++ = 0;
                p0 += max_kk0;
                ps += max_kk0;
                continue;
            }

            const float scale = 127.f / absmax;
            *pd++ = absmax / 127.f;
            kk = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _scale256 = (__m256)__lasx_xvreplfr2vr_s(scale);
            for (; kk + 7 < max_kk0; kk += 8)
            {
                __m256 _p = bfloat2float_lasx((const __m128i*)p0);
                _p = __lasx_xvfmul_s(_p, (__m256)__lasx_xvld(ps, 0));
                _p = __lasx_xvfmul_s(_p, _scale256);
                __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p)), pp, 0, 0);
                pp += 8;
                p0 += 8;
                ps += 8;
            }
#endif // __loongarch_asx
            __m128 _scale128 = __lsx_vreplfr2vr_s(scale);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                __m128 _p = bfloat2float_lsx(p0);
                _p = __lsx_vfmul_s(_p, (__m128)__lsx_vld(ps, 0));
                _p = __lsx_vfmul_s(_p, _scale128);
                __lsx_vstelm_w(float2int8(_p), pp, 0, 0);
                pp += 4;
                p0 += 4;
                ps += 4;
            }
#endif // __loongarch_sx
            for (; kk < max_kk0; kk++)
            {
                float v = bfloat16_to_float32(*p0++);
                v *= *ps++;
                *pp++ = float2int8(v * scale);
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
#if __loongarch_sx
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vldi(0);
                __m128 _absmax1 = (__m128)__lsx_vldi(0);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p0 = bfloat2float_lsx(p0a);
                    __m128 _p1 = bfloat2float_lsx(p0a + 4);
                    _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                    _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                    p0a += A_hstep;
                }

                float absmax[8];
                __lsx_vst(_absmax0, absmax, 0);
                __lsx_vst(_absmax1, absmax + 4, 0);
                const float absmax0 = absmax[0];
                const float absmax1 = absmax[1];
                const float absmax2 = absmax[2];
                const float absmax3 = absmax[3];
                const float absmax4 = absmax[4];
                const float absmax5 = absmax[5];
                const float absmax6 = absmax[6];
                const float absmax7 = absmax[7];

                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                pd += 8;

                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
                const float scale4 = absmax4 == 0.f ? 0.f : 127.f / absmax4;
                const float scale5 = absmax5 == 0.f ? 0.f : 127.f / absmax5;
                const float scale6 = absmax6 == 0.f ? 0.f : 127.f / absmax6;
                const float scale7 = absmax7 == 0.f ? 0.f : 127.f / absmax7;
                __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
                __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
                __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
                __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
                __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
                __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
                const float scales0[4] = {scale0, scale1, scale2, scale3};
                const float scales1[4] = {scale4, scale5, scale6, scale7};
                __m128 _scales0 = (__m128)__lsx_vld(scales0, 0);
                __m128 _scales1 = (__m128)__lsx_vld(scales1, 0);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned short* p1 = p0 + A_hstep;
                    const unsigned short* p2 = p1 + A_hstep;
                    const unsigned short* p3 = p2 + A_hstep;
                    __m128 _p0 = bfloat2float_lsx(p0);
                    __m128 _p1 = bfloat2float_lsx(p1);
                    __m128 _p2 = bfloat2float_lsx(p2);
                    __m128 _p3 = bfloat2float_lsx(p3);
                    __m128 _p4 = bfloat2float_lsx(p0 + 4);
                    __m128 _p5 = bfloat2float_lsx(p1 + 4);
                    __m128 _p6 = bfloat2float_lsx(p2 + 4);
                    __m128 _p7 = bfloat2float_lsx(p3 + 4);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);
                    transpose4x4_ps(_p4, _p5, _p6, _p7);
                    *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                    *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                    *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_p4, _scale4), __lsx_vfmul_s(_p5, _scale5));
                    *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_p6, _scale6), __lsx_vfmul_s(_p7, _scale7));
                    pp += 32;
                    p0 += (size_t)4 * A_hstep;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p0 = bfloat2float_lsx(p0);
                    __m128 _p1 = bfloat2float_lsx(p0 + 4);
                    _p0 = __lsx_vfmul_s(_p0, _scales0);
                    _p1 = __lsx_vfmul_s(_p1, _scales1);
                    *((int64_t*)pp) = float2int8(_p0, _p1);
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

                const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax = (__m128)__lsx_vldi(0);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p = bfloat2float_lsx(p0a);
                    _p = (__m128)__lsx_vand_v((__m128i)_p, _abs_mask);
                    _absmax = __lsx_vfmax_s(_absmax, _p);
                    p0a += A_hstep;
                }

                float absmax[4];
                __lsx_vst(_absmax, absmax, 0);
                pd[0] = absmax[0] / 127.f;
                pd[1] = absmax[1] / 127.f;
                pd[2] = absmax[2] / 127.f;
                pd[3] = absmax[3] / 127.f;
                pd += 4;

                const float scales[4] = {
                    absmax[0] == 0.f ? 0.f : 127.f / absmax[0],
                    absmax[1] == 0.f ? 0.f : 127.f / absmax[1],
                    absmax[2] == 0.f ? 0.f : 127.f / absmax[2],
                    absmax[3] == 0.f ? 0.f : 127.f / absmax[3]
                };
                __m128 _scale = (__m128)__lsx_vld(scales, 0);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned short* p1 = p0 + A_hstep;
                    const unsigned short* p2 = p1 + A_hstep;
                    const unsigned short* p3 = p2 + A_hstep;
                    __m128 _p0 = bfloat2float_lsx(p0);
                    __m128 _p1 = bfloat2float_lsx(p1);
                    __m128 _p2 = bfloat2float_lsx(p2);
                    __m128 _p3 = bfloat2float_lsx(p3);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);
                    *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, __lsx_vreplfr2vr_s(scales[0])), __lsx_vfmul_s(_p1, __lsx_vreplfr2vr_s(scales[1])));
                    *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, __lsx_vreplfr2vr_s(scales[2])), __lsx_vfmul_s(_p3, __lsx_vreplfr2vr_s(scales[3])));
                    pp += 16;
                    p0 += (size_t)4 * A_hstep;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = bfloat2float_lsx(p0);
                    const int q = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(_p, _scale)), 0);
                    pp[0] = (signed char)q;
                    pp[1] = (signed char)(q >> 8);
                    pp[2] = (signed char)(q >> 16);
                    pp[3] = (signed char)(q >> 24);
                    pp += 4;
                    p0 += A_hstep;
                }
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
#if __loongarch_sx
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                __m128 _absmax = (__m128)__lsx_vldi(0);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p = bfloat2float_lsx(p0a);
                    _absmax = __lsx_vfmax_s(_absmax, (__m128)__lsx_vand_v((__m128i)_p, _abs_mask));
                    p0a += A_hstep;
                }
                float absmax[2];
                __lsx_vstelm_d((__m128i)_absmax, absmax, 0, 0);
                pd[0] = absmax[0] / 127.f;
                pd[1] = absmax[1] / 127.f;
                pd += 2;
                const float scale0 = absmax[0] == 0.f ? 0.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 0.f : 127.f / absmax[1];
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned short* p1 = p0 + A_hstep;
                    const unsigned short* p2 = p1 + A_hstep;
                    const unsigned short* p3 = p2 + A_hstep;
                    pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(p1[0]) * scale0);
                    pp[2] = float2int8(bfloat16_to_float32(p2[0]) * scale0);
                    pp[3] = float2int8(bfloat16_to_float32(p3[0]) * scale0);
                    pp[4] = float2int8(bfloat16_to_float32(p0[1]) * scale1);
                    pp[5] = float2int8(bfloat16_to_float32(p1[1]) * scale1);
                    pp[6] = float2int8(bfloat16_to_float32(p2[1]) * scale1);
                    pp[7] = float2int8(bfloat16_to_float32(p3[1]) * scale1);
                    pp += 8;
                    p0 += (size_t)4 * A_hstep;
                }
                for (; kk < max_kk0; kk++)
                {
                    pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale1);
                    pp += 2;
                    p0 += A_hstep;
                }
            }
#else
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])));
                    absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[1])));
                    p0a += A_hstep;
                }
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned short* p1 = p0 + A_hstep;
                    const unsigned short* p2 = p1 + A_hstep;
                    const unsigned short* p3 = p2 + A_hstep;
                    pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(p1[0]) * scale0);
                    pp[2] = float2int8(bfloat16_to_float32(p2[0]) * scale0);
                    pp[3] = float2int8(bfloat16_to_float32(p3[0]) * scale0);
                    pp[4] = float2int8(bfloat16_to_float32(p0[1]) * scale1);
                    pp[5] = float2int8(bfloat16_to_float32(p1[1]) * scale1);
                    pp[6] = float2int8(bfloat16_to_float32(p2[1]) * scale1);
                    pp[7] = float2int8(bfloat16_to_float32(p3[1]) * scale1);
                    pp += 8;
                    p0 += (size_t)4 * A_hstep;
                }
                for (; kk < max_kk0; kk++)
                {
                    pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale1);
                    pp += 2;
                    p0 += A_hstep;
                }
            }
#endif // __loongarch_sx
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                float absmax = 0.f;
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0a);
                    absmax = std::max(absmax, fabsf(v));
                    p0a += A_hstep;
                }

                if (absmax == 0.f)
                {
                    *pd++ = 0.f;
                    for (int kk = 0; kk < max_kk0; kk++)
                        *pp++ = 0;
                    p0 += (size_t)max_kk0 * A_hstep;
                    continue;
                }

                const float scale = 127.f / absmax;
                *pd++ = absmax / 127.f;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0);
                    *pp++ = float2int8(v * scale);
                    p0 += A_hstep;
                }
            }
        }

        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            const unsigned short* p0a = p0;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                __m128 _p0 = bfloat2float_lsx(p0a);
                __m128 _p1 = bfloat2float_lsx(p0a + 4);
                __m128 _s = __lsx_vreplfr2vr_s(*psa++);
                _p0 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p0, _abs_mask), _s);
                _p1 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p1, _abs_mask), _s);
                _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
                p0a += A_hstep;
            }

            float absmax[8];
            __lsx_vst(_absmax0, absmax, 0);
            __lsx_vst(_absmax1, absmax + 4, 0);
            const float absmax0 = absmax[0];
            const float absmax1 = absmax[1];
            const float absmax2 = absmax[2];
            const float absmax3 = absmax[3];
            const float absmax4 = absmax[4];
            const float absmax5 = absmax[5];
            const float absmax6 = absmax[6];
            const float absmax7 = absmax[7];

            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd[4] = absmax4 / 127.f;
            pd[5] = absmax5 / 127.f;
            pd[6] = absmax6 / 127.f;
            pd[7] = absmax7 / 127.f;
            pd += 8;

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
            const float scale4 = absmax4 == 0.f ? 0.f : 127.f / absmax4;
            const float scale5 = absmax5 == 0.f ? 0.f : 127.f / absmax5;
            const float scale6 = absmax6 == 0.f ? 0.f : 127.f / absmax6;
            const float scale7 = absmax7 == 0.f ? 0.f : 127.f / absmax7;
            __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
            __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
            __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
            __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
            const float scales0[4] = {scale0, scale1, scale2, scale3};
            const float scales1[4] = {scale4, scale5, scale6, scale7};
            __m128 _scales0 = (__m128)__lsx_vld(scales0, 0);
            __m128 _scales1 = (__m128)__lsx_vld(scales1, 0);

            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                const unsigned short* p1 = p0 + A_hstep;
                const unsigned short* p2 = p1 + A_hstep;
                const unsigned short* p3 = p2 + A_hstep;
                __m128 _p0 = bfloat2float_lsx(p0);
                __m128 _p1 = bfloat2float_lsx(p1);
                __m128 _p2 = bfloat2float_lsx(p2);
                __m128 _p3 = bfloat2float_lsx(p3);
                __m128 _p4 = bfloat2float_lsx(p0 + 4);
                __m128 _p5 = bfloat2float_lsx(p1 + 4);
                __m128 _p6 = bfloat2float_lsx(p2 + 4);
                __m128 _p7 = bfloat2float_lsx(p3 + 4);
                __m128 _s0 = __lsx_vreplfr2vr_s(ps[0]);
                __m128 _s1 = __lsx_vreplfr2vr_s(ps[1]);
                __m128 _s2 = __lsx_vreplfr2vr_s(ps[2]);
                __m128 _s3 = __lsx_vreplfr2vr_s(ps[3]);
                _p0 = __lsx_vfmul_s(_p0, _s0);
                _p1 = __lsx_vfmul_s(_p1, _s1);
                _p2 = __lsx_vfmul_s(_p2, _s2);
                _p3 = __lsx_vfmul_s(_p3, _s3);
                _p4 = __lsx_vfmul_s(_p4, _s0);
                _p5 = __lsx_vfmul_s(_p5, _s1);
                _p6 = __lsx_vfmul_s(_p6, _s2);
                _p7 = __lsx_vfmul_s(_p7, _s3);
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                transpose4x4_ps(_p4, _p5, _p6, _p7);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_p4, _scale4), __lsx_vfmul_s(_p5, _scale5));
                *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_p6, _scale6), __lsx_vfmul_s(_p7, _scale7));
                pp += 32;
                p0 += (size_t)4 * A_hstep;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                __m128 _p0 = bfloat2float_lsx(p0);
                __m128 _p1 = bfloat2float_lsx(p0 + 4);
                __m128 _s = __lsx_vreplfr2vr_s(*ps++);
                _p0 = __lsx_vfmul_s(_p0, _s);
                _p1 = __lsx_vfmul_s(_p1, _s);
                _p0 = __lsx_vfmul_s(_p0, _scales0);
                _p1 = __lsx_vfmul_s(_p1, _scales1);
                *((int64_t*)pp) = float2int8(_p0, _p1);
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

            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax = (__m128)__lsx_vldi(0);
            const unsigned short* p0a = p0;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                __m128 _p = bfloat2float_lsx(p0a);
                _p = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p, _abs_mask), __lsx_vreplfr2vr_s(*psa++));
                _absmax = __lsx_vfmax_s(_absmax, _p);
                p0a += A_hstep;
            }

            float absmax[4];
            __lsx_vst(_absmax, absmax, 0);
            pd[0] = absmax[0] / 127.f;
            pd[1] = absmax[1] / 127.f;
            pd[2] = absmax[2] / 127.f;
            pd[3] = absmax[3] / 127.f;
            pd += 4;

            const float scales[4] = {
                absmax[0] == 0.f ? 0.f : 127.f / absmax[0],
                absmax[1] == 0.f ? 0.f : 127.f / absmax[1],
                absmax[2] == 0.f ? 0.f : 127.f / absmax[2],
                absmax[3] == 0.f ? 0.f : 127.f / absmax[3]
            };
            __m128 _scale = (__m128)__lsx_vld(scales, 0);

            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                const unsigned short* p1 = p0 + A_hstep;
                const unsigned short* p2 = p1 + A_hstep;
                const unsigned short* p3 = p2 + A_hstep;
                __m128 _p0 = bfloat2float_lsx(p0);
                __m128 _p1 = bfloat2float_lsx(p1);
                __m128 _p2 = bfloat2float_lsx(p2);
                __m128 _p3 = bfloat2float_lsx(p3);
                _p0 = __lsx_vfmul_s(_p0, __lsx_vreplfr2vr_s(ps[0]));
                _p1 = __lsx_vfmul_s(_p1, __lsx_vreplfr2vr_s(ps[1]));
                _p2 = __lsx_vfmul_s(_p2, __lsx_vreplfr2vr_s(ps[2]));
                _p3 = __lsx_vfmul_s(_p3, __lsx_vreplfr2vr_s(ps[3]));
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, __lsx_vreplfr2vr_s(scales[0])), __lsx_vfmul_s(_p1, __lsx_vreplfr2vr_s(scales[1])));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, __lsx_vreplfr2vr_s(scales[2])), __lsx_vfmul_s(_p3, __lsx_vreplfr2vr_s(scales[3])));
                pp += 16;
                p0 += (size_t)4 * A_hstep;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                __m128 _p = bfloat2float_lsx(p0);
                _p = __lsx_vfmul_s(_p, __lsx_vreplfr2vr_s(*ps++));
                const int q = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(_p, _scale)), 0);
                pp[0] = (signed char)q;
                pp[1] = (signed char)(q >> 8);
                pp[2] = (signed char)(q >> 16);
                pp[3] = (signed char)(q >> 24);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
#if __loongarch_sx
        const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            __m128 _absmax = (__m128)__lsx_vldi(0);
            const unsigned short* p0a = p0;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                __m128 _p = bfloat2float_lsx(p0a);
                _p = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p, _abs_mask), __lsx_vreplfr2vr_s(*psa++));
                _absmax = __lsx_vfmax_s(_absmax, _p);
                p0a += A_hstep;
            }
            float absmax[2];
            __lsx_vstelm_d((__m128i)_absmax, absmax, 0, 0);
            pd[0] = absmax[0] / 127.f;
            pd[1] = absmax[1] / 127.f;
            pd += 2;
            const float scale0 = absmax[0] == 0.f ? 0.f : 127.f / absmax[0];
            const float scale1 = absmax[1] == 0.f ? 0.f : 127.f / absmax[1];
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                const unsigned short* p1 = p0 + A_hstep;
                const unsigned short* p2 = p1 + A_hstep;
                const unsigned short* p3 = p2 + A_hstep;
                const float s0 = ps[0];
                const float s1 = ps[1];
                const float s2 = ps[2];
                const float s3 = ps[3];
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * s0 * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p1[0]) * s1 * scale0);
                pp[2] = float2int8(bfloat16_to_float32(p2[0]) * s2 * scale0);
                pp[3] = float2int8(bfloat16_to_float32(p3[0]) * s3 * scale0);
                pp[4] = float2int8(bfloat16_to_float32(p0[1]) * s0 * scale1);
                pp[5] = float2int8(bfloat16_to_float32(p1[1]) * s1 * scale1);
                pp[6] = float2int8(bfloat16_to_float32(p2[1]) * s2 * scale1);
                pp[7] = float2int8(bfloat16_to_float32(p3[1]) * s3 * scale1);
                pp += 8;
                p0 += (size_t)4 * A_hstep;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                const float s = *ps++;
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * s * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * s * scale1);
                pp += 2;
                p0 += A_hstep;
            }
        }
#else
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
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(bfloat16_to_float32(p0a[0])) * s);
                absmax1 = std::max(absmax1, fabsf(bfloat16_to_float32(p0a[1])) * s);
                p0a += A_hstep;
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                const unsigned short* p1 = p0 + A_hstep;
                const unsigned short* p2 = p1 + A_hstep;
                const unsigned short* p3 = p2 + A_hstep;
                const float s0 = ps[0];
                const float s1 = ps[1];
                const float s2 = ps[2];
                const float s3 = ps[3];
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * s0 * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p1[0]) * s1 * scale0);
                pp[2] = float2int8(bfloat16_to_float32(p2[0]) * s2 * scale0);
                pp[3] = float2int8(bfloat16_to_float32(p3[0]) * s3 * scale0);
                pp[4] = float2int8(bfloat16_to_float32(p0[1]) * s0 * scale1);
                pp[5] = float2int8(bfloat16_to_float32(p1[1]) * s1 * scale1);
                pp[6] = float2int8(bfloat16_to_float32(p2[1]) * s2 * scale1);
                pp[7] = float2int8(bfloat16_to_float32(p3[1]) * s3 * scale1);
                pp += 8;
                p0 += (size_t)4 * A_hstep;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                const float s = *ps++;
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * s * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * s * scale1);
                pp += 2;
                p0 += A_hstep;
            }
        }
#endif // __loongarch_sx
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

            float absmax = 0.f;
            const unsigned short* p0a = p0;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v = bfloat16_to_float32(*p0a);
                v = fabsf(v) * *psa++;
                absmax = std::max(absmax, v);
                p0a += A_hstep;
            }

            if (absmax == 0.f)
            {
                *pd++ = 0.f;
                for (int kk = 0; kk < max_kk0; kk++)
                    *pp++ = 0;
                p0 += (size_t)max_kk0 * A_hstep;
                ps += max_kk0;
                continue;
            }

            const float scale = 127.f / absmax;
            *pd++ = absmax / 127.f;

            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v = bfloat16_to_float32(*p0);
                v *= *ps++;
                *pp++ = float2int8(v * scale);
                p0 += A_hstep;
            }
        }
    }
}
