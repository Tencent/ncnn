// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// K-major, row-interleaved MR-packn/MR2/MR1
static void quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int local_block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __riscv_vector && __riscv_zvfh
        const int packn = csrr_vlenb() / 4;
        const int packn_fp16 = packn * 2;
        const int packn_a = elempack == packn_fp16 ? packn_fp16 : packn;
        const size_t vl = __riscv_vsetvl_e32m4(packn_a);
        const size_t vl_packn = __riscv_vsetvl_e32m4(packn);
        const ptrdiff_t A_stride = (ptrdiff_t)A_hstep * sizeof(__fp16);
        for (; ii + (packn_a - 1) < max_ii; ii += packn_a)
        {
            const __fp16* p0 = (const __fp16*)A + (size_t)(i + ii) * A_hstep + k;
            if (elempack == packn_fp16)
                p0 = (const __fp16*)A + (size_t)(i + ii) / packn_fp16 * A_hstep * packn_fp16 + k * packn_fp16;

            signed char* pp0 = pp;
            signed char* pp1 = pp + (size_t)max_kk * packn;
            float* pd0 = pd;
            float* pd1 = pd + (size_t)local_block_count * packn;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(0.f, vl);
                const __fp16* p0a = p0;

                if (elempack == packn_fp16)
                {
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0a, vl);
                        vfloat32m4_t _v = __riscv_vfabs_v_f32m4(__riscv_vfwcvt_f_f_v_f32m4(_p, vl), vl);
                        _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl);
                        p0a += packn_fp16;
                    }
                }
                if (elempack == 1)
                {
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0a, A_stride, vl);
                        vfloat32m4_t _v = __riscv_vfabs_v_f32m4(__riscv_vfwcvt_f_f_v_f32m4(_p, vl), vl);
                        _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl);
                        p0a++;
                    }
                }

                vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl);
                _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl), vl);
                vfloat32m4_t _descale = __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl);

                if (elempack == packn_fp16)
                {
                    __riscv_vse32_v_f32m4(pd0, _descale, vl_packn);
                    __riscv_vse32_v_f32m4(pd1, __riscv_vslidedown_vx_f32m4(_descale, packn, vl), vl_packn);
                    pd0 += packn;
                    pd1 += packn;
                }
                if (elempack == 1)
                {
                    __riscv_vse32_v_f32m4(pd0, _descale, vl);
                    pd0 += packn;
                }

                if (elempack == packn_fp16)
                {
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0, vl);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl);
                        vint8m1_t _q = float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl), vl);
                        __riscv_vse8_v_i8m1(pp0, _q, vl_packn);
                        __riscv_vse8_v_i8m1(pp1, __riscv_vslidedown_vx_i8m1(_q, packn, vl), vl_packn);
                        pp0 += packn;
                        pp1 += packn;
                        p0 += packn_fp16;
                    }
                }
                if (elempack == 1)
                {
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0, A_stride, vl);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl);
                        __riscv_vse8_v_i8m1(pp0, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl), vl), vl);
                        pp0 += packn;
                        p0++;
                    }
                }
            }

            pp += (size_t)max_kk * packn_a;
            pd += (size_t)local_block_count * packn_a;
        }
#endif // __riscv_vector && __riscv_zvfh
        for (; ii + 1 < max_ii; ii += 2)
        {
            const __fp16* p0 = (const __fp16*)A + (size_t)(i + ii) * A_hstep + k;
            const __fp16* p1 = p0 + A_hstep;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const __fp16* p0a = p0;
                const __fp16* p1a = p1;

                int kk = 0;
#if __riscv_vector && __riscv_zvfh
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                    vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p0a, vl), vl);
                    vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p1a, vl), vl);
                    _v0 = __riscv_vfabs_v_f32m4(_v0, vl);
                    _v1 = __riscv_vfabs_v_f32m4(_v1, vl);
                    absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                    absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                    p0a += vl;
                    p1a += vl;
                    kk += vl;
                }
#else
                for (; kk < max_kk0; kk++)
                {
                    absmax0 = std::max(absmax0, fabsf((float)*p0a++));
                    absmax1 = std::max(absmax1, fabsf((float)*p1a++));
                }
#endif // __riscv_vector && __riscv_zvfh

                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                kk = 0;
#if __riscv_vector && __riscv_zvfh
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                    vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p0, vl), vl);
                    vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p1, vl), vl);
                    vint8m1_t _q0 = float2int8(__riscv_vfmul_vf_f32m4(_v0, scale0, vl), vl);
                    vint8m1_t _q1 = float2int8(__riscv_vfmul_vf_f32m4(_v1, scale1, vl), vl);
                    vint8m1x2_t _q = __riscv_vcreate_v_i8m1x2(_q0, _q1);
                    __riscv_vsseg2e8_v_i8m1x2(pp, _q, vl);
                    pp += vl * 2;
                    p0 += vl;
                    p1 += vl;
                    kk += vl;
                }
#else
                for (; kk < max_kk0; kk++)
                {
                    pp[0] = float2int8((float)*p0++ * scale0);
                    pp[1] = float2int8((float)*p1++ * scale1);
                    pp += 2;
                }
#endif // __riscv_vector && __riscv_zvfh
            }
        }
        for (; ii < max_ii; ii++)
        {
            const __fp16* p0 = (const __fp16*)A + (size_t)(i + ii) * A_hstep + k;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax = 0.f;
                const __fp16* p0a = p0;

                int kk = 0;
#if __riscv_vector && __riscv_zvfh
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p0a, vl), vl);
                    _v = __riscv_vfabs_v_f32m4(_v, vl);
                    absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                    p0a += vl;
                    kk += vl;
                }
#else
                for (; kk < max_kk0; kk++)
                    absmax = std::max(absmax, fabsf((float)*p0a++));
#endif // __riscv_vector && __riscv_zvfh

                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;

                kk = 0;
#if __riscv_vector && __riscv_zvfh
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p0, vl), vl);
                    __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vf_f32m4(_v, scale, vl), vl), vl);
                    pp += vl;
                    p0 += vl;
                    kk += vl;
                }
#else
                for (; kk < max_kk0; kk++)
                    *pp++ = float2int8((float)*p0++ * scale);
#endif // __riscv_vector && __riscv_zvfh
            }
        }
        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __riscv_vector && __riscv_zvfh
    const int packn = csrr_vlenb() / 4;
    const int packn_fp16 = packn * 2;
    const int packn_a = elempack == packn_fp16 ? packn_fp16 : packn;
    const size_t vl = __riscv_vsetvl_e32m4(packn_a);
    const size_t vl_packn = __riscv_vsetvl_e32m4(packn);
    const ptrdiff_t A_stride = (ptrdiff_t)A_hstep * sizeof(__fp16);
    for (; ii + (packn_a - 1) < max_ii; ii += packn_a)
    {
        const __fp16* p0 = (const __fp16*)A + (size_t)(i + ii) * A_hstep + k;
        if (elempack == packn_fp16)
            p0 = (const __fp16*)A + (size_t)(i + ii) / packn_fp16 * A_hstep * packn_fp16 + k * packn_fp16;
        const float* ps = input_scale_ptr;

        signed char* pp0 = pp;
        signed char* pp1 = pp + (size_t)max_kk * packn;
        float* pd0 = pd;
        float* pd1 = pd + (size_t)local_block_count * packn;

        for (int g = 0; g < local_block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(0.f, vl);
            const __fp16* p0a = p0;
            const float* psa = ps;

            if (elempack == packn_fp16)
            {
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0a, vl);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl);
                    _v = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(_v, *psa++, vl), vl);
                    _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl);
                    p0a += packn_fp16;
                }
            }
            if (elempack == 1)
            {
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0a, A_stride, vl);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl);
                    _v = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(_v, *psa++, vl), vl);
                    _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl);
                    p0a++;
                }
            }

            vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl);
            _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl), vl);
            vfloat32m4_t _descale = __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl);

            if (elempack == packn_fp16)
            {
                __riscv_vse32_v_f32m4(pd0, _descale, vl_packn);
                __riscv_vse32_v_f32m4(pd1, __riscv_vslidedown_vx_f32m4(_descale, packn, vl), vl_packn);
                pd0 += packn;
                pd1 += packn;
            }
            if (elempack == 1)
            {
                __riscv_vse32_v_f32m4(pd0, _descale, vl);
                pd0 += packn;
            }

            if (elempack == packn_fp16)
            {
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0, vl);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl);
                    _v = __riscv_vfmul_vf_f32m4(_v, *ps++, vl);
                    vint8m1_t _q = float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl), vl);
                    __riscv_vse8_v_i8m1(pp0, _q, vl_packn);
                    __riscv_vse8_v_i8m1(pp1, __riscv_vslidedown_vx_i8m1(_q, packn, vl), vl_packn);
                    pp0 += packn;
                    pp1 += packn;
                    p0 += packn_fp16;
                }
            }
            if (elempack == 1)
            {
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0, A_stride, vl);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl);
                    _v = __riscv_vfmul_vf_f32m4(_v, *ps++, vl);
                    __riscv_vse8_v_i8m1(pp0, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl), vl), vl);
                    pp0 += packn;
                    p0++;
                }
            }
        }

        pp += (size_t)max_kk * packn_a;
        pd += (size_t)local_block_count * packn_a;
    }
#endif // __riscv_vector && __riscv_zvfh
    for (; ii + 1 < max_ii; ii += 2)
    {
        const __fp16* p0 = (const __fp16*)A + (size_t)(i + ii) * A_hstep + k;
        const __fp16* p1 = p0 + A_hstep;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < local_block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const __fp16* p0a = p0;
            const __fp16* p1a = p1;
            const float* psa = ps;

            int kk = 0;
#if __riscv_vector && __riscv_zvfh
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p0a, vl), vl);
                vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p1a, vl), vl);
                vfloat32m4_t _s = __riscv_vle32_v_f32m4(psa, vl);
                _v0 = __riscv_vfabs_v_f32m4(__riscv_vfmul_vv_f32m4(_v0, _s, vl), vl);
                _v1 = __riscv_vfabs_v_f32m4(__riscv_vfmul_vv_f32m4(_v1, _s, vl), vl);
                absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                p0a += vl;
                p1a += vl;
                psa += vl;
                kk += vl;
            }
#else
            for (; kk < max_kk0; kk++)
            {
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf((float)*p0a++) * s);
                absmax1 = std::max(absmax1, fabsf((float)*p1a++) * s);
            }
#endif // __riscv_vector && __riscv_zvfh

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

            kk = 0;
#if __riscv_vector && __riscv_zvfh
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p0, vl), vl);
                vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p1, vl), vl);
                vfloat32m4_t _s = __riscv_vle32_v_f32m4(ps, vl);
                _v0 = __riscv_vfmul_vv_f32m4(_v0, _s, vl);
                _v1 = __riscv_vfmul_vv_f32m4(_v1, _s, vl);
                vint8m1_t _q0 = float2int8(__riscv_vfmul_vf_f32m4(_v0, scale0, vl), vl);
                vint8m1_t _q1 = float2int8(__riscv_vfmul_vf_f32m4(_v1, scale1, vl), vl);
                vint8m1x2_t _q = __riscv_vcreate_v_i8m1x2(_q0, _q1);
                __riscv_vsseg2e8_v_i8m1x2(pp, _q, vl);
                pp += vl * 2;
                p0 += vl;
                p1 += vl;
                ps += vl;
                kk += vl;
            }
#else
            for (; kk < max_kk0; kk++)
            {
                const float s = *ps++;
                pp[0] = float2int8((float)*p0++ * s * scale0);
                pp[1] = float2int8((float)*p1++ * s * scale1);
                pp += 2;
            }
#endif // __riscv_vector && __riscv_zvfh
        }
    }
    for (; ii < max_ii; ii++)
    {
        const __fp16* p0 = (const __fp16*)A + (size_t)(i + ii) * A_hstep + k;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < local_block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax = 0.f;
            const __fp16* p0a = p0;
            const float* psa = ps;

            int kk = 0;
#if __riscv_vector && __riscv_zvfh
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p0a, vl), vl);
                _v = __riscv_vfabs_v_f32m4(__riscv_vfmul_vv_f32m4(_v, __riscv_vle32_v_f32m4(psa, vl), vl), vl);
                absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                p0a += vl;
                psa += vl;
                kk += vl;
            }
#else
            for (; kk < max_kk0; kk++)
                absmax = std::max(absmax, fabsf((float)*p0a++) * *psa++);
#endif // __riscv_vector && __riscv_zvfh

            const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
            *pd++ = absmax / 127.f;

            kk = 0;
#if __riscv_vector && __riscv_zvfh
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p0, vl), vl);
                _v = __riscv_vfmul_vv_f32m4(_v, __riscv_vle32_v_f32m4(ps, vl), vl);
                __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vf_f32m4(_v, scale, vl), vl), vl);
                pp += vl;
                p0 += vl;
                ps += vl;
                kk += vl;
            }
#else
            for (; kk < max_kk0; kk++)
                *pp++ = float2int8((float)*p0++ * *ps++ * scale);
#endif // __riscv_vector && __riscv_zvfh
        }
    }
}

// K-major, row-interleaved MR-packn/MR2/MR1
static void transpose_quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int local_block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __riscv_vector && __riscv_zvfh
        const int packn = csrr_vlenb() / 4;
        const int packn_fp16 = packn * 2;
        const size_t vl_packn = __riscv_vsetvl_e32m4(packn);
        for (; ii + (packn - 1) < max_ii; ii += packn)
        {
            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const int k0 = k + g * block_size;
                const int k0_pack = k0 / packn_fp16;
                const int k0_lane = k0 % packn_fp16;
                const __fp16* p0_base = (const __fp16*)A + (size_t)k0_pack * A_hstep * packn_fp16 + (i + ii) * packn_fp16 + k0_lane;
                const int head_kk = std::min(max_kk0, packn_fp16 - k0_lane);
                vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(0.f, vl_packn);

                if (elempack == packn_fp16)
                {
                    const __fp16* p0a = p0_base;

                    int kk = 0;
                    for (; kk < head_kk; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0a, packn_fp16 * sizeof(__fp16), vl_packn);
                        vfloat32m4_t _v = __riscv_vfabs_v_f32m4(__riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn), vl_packn);
                        _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl_packn);
                        p0a++;
                    }
                    if (kk == packn_fp16 - k0_lane)
                        p0a += A_hstep * packn_fp16 - packn_fp16;
                    for (; kk + (packn_fp16 - 1) < max_kk0; kk += packn_fp16)
                    {
                        for (int l = 0; l < packn_fp16; l++)
                        {
                            vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0a + l, packn_fp16 * sizeof(__fp16), vl_packn);
                            vfloat32m4_t _v = __riscv_vfabs_v_f32m4(__riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn), vl_packn);
                            _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl_packn);
                        }
                        p0a += A_hstep * packn_fp16;
                    }
                }
                if (elempack == 1)
                {
                    const __fp16* p0 = (const __fp16*)A + (size_t)(k + g * block_size) * A_hstep + i + ii;

                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0, vl_packn);
                        vfloat32m4_t _v = __riscv_vfabs_v_f32m4(__riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn), vl_packn);
                        _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl_packn);
                        p0 += A_hstep;
                    }
                }

                vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl_packn);
                _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl_packn), vl_packn);
                __riscv_vse32_v_f32m4(pd, __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl_packn), vl_packn);
                pd += packn;

                if (elempack == packn_fp16)
                {
                    const __fp16* p0 = p0_base;

                    int kk = 0;
                    for (; kk < head_kk; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0, packn_fp16 * sizeof(__fp16), vl_packn);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                        __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl_packn), vl_packn), vl_packn);
                        pp += packn;
                        p0++;
                    }
                    if (kk == packn_fp16 - k0_lane)
                        p0 += A_hstep * packn_fp16 - packn_fp16;
                    for (; kk + (packn_fp16 - 1) < max_kk0; kk += packn_fp16)
                    {
                        for (int l = 0; l < packn_fp16; l++)
                        {
                            vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0 + l, packn_fp16 * sizeof(__fp16), vl_packn);
                            vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                            __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl_packn), vl_packn), vl_packn);
                            pp += packn;
                        }
                        p0 += A_hstep * packn_fp16;
                    }
                }
                if (elempack == 1)
                {
                    const __fp16* p0 = (const __fp16*)A + (size_t)(k + g * block_size) * A_hstep + i + ii;

                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0, vl_packn);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                        __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl_packn), vl_packn), vl_packn);
                        pp += packn;
                        p0 += A_hstep;
                    }
                }
            }
        }
#endif // __riscv_vector && __riscv_zvfh
        for (; ii + 1 < max_ii; ii += 2)
        {
#if __riscv_vector && __riscv_zvfh
            if (elempack == packn_fp16)
            {
                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const int k0 = k + g * block_size;
                    const int k0_pack = k0 / packn_fp16;
                    const int k0_lane = k0 % packn_fp16;
                    const __fp16* p0 = (const __fp16*)A + (size_t)k0_pack * A_hstep * packn_fp16 + (i + ii) * packn_fp16 + k0_lane;
                    const __fp16* p0a = p0;
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;

                    int kk = 0;
                    int kk_lane = k0_lane;
                    const size_t vl = packn_fp16;
                    vuint16m2_t _idx = __riscv_vid_v_u16m2(vl);
                    while (kk < max_kk0)
                    {
                        const int n = std::min(max_kk0 - kk, packn_fp16 - kk_lane);
                        vbool8_t _mask = __riscv_vmsltu_vx_u16m2_b8(_idx, n, vl);
                        vfloat16m2_t _p0 = __riscv_vle16_v_f16m2_m(_mask, p0a, vl);
                        vfloat16m2_t _p1 = __riscv_vle16_v_f16m2_m(_mask, p0a + packn_fp16, vl);
                        vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(_p0, vl);
                        vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(_p1, vl);
                        _v0 = __riscv_vfmerge_vfm_f32m4(__riscv_vfabs_v_f32m4(_v0, vl), 0.f, __riscv_vmnot_m_b8(_mask, vl), vl);
                        _v1 = __riscv_vfmerge_vfm_f32m4(__riscv_vfabs_v_f32m4(_v1, vl), 0.f, __riscv_vmnot_m_b8(_mask, vl), vl);
                        absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                        absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                        p0a += n;
                        kk += n;
                        kk_lane += n;
                        if (kk_lane == packn_fp16)
                        {
                            p0a += A_hstep * packn_fp16 - packn_fp16;
                            kk_lane = 0;
                        }
                    }

                    const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    kk = 0;
                    kk_lane = k0_lane;
                    while (kk < max_kk0)
                    {
                        const int n = std::min(max_kk0 - kk, packn_fp16 - kk_lane);
                        vbool8_t _mask = __riscv_vmsltu_vx_u16m2_b8(_idx, n, vl);
                        vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0, vl), vl);
                        vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0 + packn_fp16, vl), vl);
                        vint8m1x2_t _q = __riscv_vcreate_v_i8m1x2(
                                             float2int8(__riscv_vfmul_vf_f32m4(_v0, scale0, vl), vl),
                                             float2int8(__riscv_vfmul_vf_f32m4(_v1, scale1, vl), vl));
                        __riscv_vsseg2e8_v_i8m1x2_m(_mask, pp, _q, vl);
                        pp += n * 2;
                        p0 += n;
                        kk += n;
                        kk_lane += n;
                        if (kk_lane == packn_fp16)
                        {
                            p0 += A_hstep * packn_fp16 - packn_fp16;
                            kk_lane = 0;
                        }
                    }
                }
            }
#endif // __riscv_vector && __riscv_zvfh
            if (elempack == 1)
            {
                const __fp16* p0 = (const __fp16*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    const __fp16* p0a = p0;

                    int kk = 0;
#if __riscv_vector && __riscv_zvfh
                    while (kk < max_kk0)
                    {
                        const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                        vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0a, A_hstep * sizeof(__fp16), vl), vl);
                        vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0a + 1, A_hstep * sizeof(__fp16), vl), vl);
                        _v0 = __riscv_vfabs_v_f32m4(_v0, vl);
                        _v1 = __riscv_vfabs_v_f32m4(_v1, vl);
                        absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                        absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                        p0a += vl * A_hstep;
                        kk += vl;
                    }
#else
                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf((float)p0a[0]));
                        absmax1 = std::max(absmax1, fabsf((float)p0a[1]));
                        p0a += A_hstep;
                    }
#endif // __riscv_vector && __riscv_zvfh

                    const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    kk = 0;
#if __riscv_vector && __riscv_zvfh
                    while (kk < max_kk0)
                    {
                        const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                        vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0, A_hstep * sizeof(__fp16), vl), vl);
                        vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0 + 1, A_hstep * sizeof(__fp16), vl), vl);
                        vint8m1_t _q0 = float2int8(__riscv_vfmul_vf_f32m4(_v0, scale0, vl), vl);
                        vint8m1_t _q1 = float2int8(__riscv_vfmul_vf_f32m4(_v1, scale1, vl), vl);
                        vint8m1x2_t _q = __riscv_vcreate_v_i8m1x2(_q0, _q1);
                        __riscv_vsseg2e8_v_i8m1x2(pp, _q, vl);
                        pp += vl * 2;
                        p0 += vl * A_hstep;
                        kk += vl;
                    }
#else
                    for (; kk < max_kk0; kk++)
                    {
                        pp[0] = float2int8((float)p0[0] * scale0);
                        pp[1] = float2int8((float)p0[1] * scale1);
                        pp += 2;
                        p0 += A_hstep;
                    }
#endif // __riscv_vector && __riscv_zvfh
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
#if __riscv_vector && __riscv_zvfh
            if (elempack == packn_fp16)
            {
                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const int k0 = k + g * block_size;
                    const int k0_pack = k0 / packn_fp16;
                    const int k0_lane = k0 % packn_fp16;
                    const __fp16* p0 = (const __fp16*)A + (size_t)k0_pack * A_hstep * packn_fp16 + (i + ii) * packn_fp16 + k0_lane;
                    const __fp16* p0a = p0;
                    float absmax = 0.f;

                    int kk = 0;
                    int kk_lane = k0_lane;
                    const size_t vl = packn_fp16;
                    vuint16m2_t _idx = __riscv_vid_v_u16m2(vl);
                    while (kk < max_kk0)
                    {
                        const int n = std::min(max_kk0 - kk, packn_fp16 - kk_lane);
                        vbool8_t _mask = __riscv_vmsltu_vx_u16m2_b8(_idx, n, vl);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0a, vl), vl);
                        _v = __riscv_vfmerge_vfm_f32m4(__riscv_vfabs_v_f32m4(_v, vl), 0.f, __riscv_vmnot_m_b8(_mask, vl), vl);
                        absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                        p0a += n;
                        kk += n;
                        kk_lane += n;
                        if (kk_lane == packn_fp16)
                        {
                            p0a += A_hstep * packn_fp16 - packn_fp16;
                            kk_lane = 0;
                        }
                    }

                    const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                    *pd++ = absmax / 127.f;

                    kk = 0;
                    kk_lane = k0_lane;
                    while (kk < max_kk0)
                    {
                        const int n = std::min(max_kk0 - kk, packn_fp16 - kk_lane);
                        vbool8_t _mask = __riscv_vmsltu_vx_u16m2_b8(_idx, n, vl);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0, vl), vl);
                        vint8m1_t _q = float2int8(__riscv_vfmul_vf_f32m4(_v, scale, vl), vl);
                        __riscv_vse8_v_i8m1_m(_mask, pp, _q, vl);
                        pp += n;
                        p0 += n;
                        kk += n;
                        kk_lane += n;
                        if (kk_lane == packn_fp16)
                        {
                            p0 += A_hstep * packn_fp16 - packn_fp16;
                            kk_lane = 0;
                        }
                    }
                }
            }
#endif // __riscv_vector && __riscv_zvfh
            if (elempack == 1)
            {
                const __fp16* p0 = (const __fp16*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax = 0.f;
                    const __fp16* p0a = p0;

                    int kk = 0;
#if __riscv_vector && __riscv_zvfh
                    while (kk < max_kk0)
                    {
                        const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0a, A_hstep * sizeof(__fp16), vl), vl);
                        _v = __riscv_vfabs_v_f32m4(_v, vl);
                        absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                        p0a += vl * A_hstep;
                        kk += vl;
                    }
#else
                    for (; kk < max_kk0; kk++)
                    {
                        absmax = std::max(absmax, fabsf((float)*p0a));
                        p0a += A_hstep;
                    }
#endif // __riscv_vector && __riscv_zvfh

                    const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                    *pd++ = absmax / 127.f;

                    kk = 0;
#if __riscv_vector && __riscv_zvfh
                    while (kk < max_kk0)
                    {
                        const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0, A_hstep * sizeof(__fp16), vl), vl);
                        __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vf_f32m4(_v, scale, vl), vl), vl);
                        pp += vl;
                        p0 += vl * A_hstep;
                        kk += vl;
                    }
#else
                    for (; kk < max_kk0; kk++)
                    {
                        *pp++ = float2int8((float)*p0 * scale);
                        p0 += A_hstep;
                    }
#endif // __riscv_vector && __riscv_zvfh
                }
            }
        }
        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __riscv_vector && __riscv_zvfh
    const int packn = csrr_vlenb() / 4;
    const int packn_fp16 = packn * 2;
    const size_t vl_packn = __riscv_vsetvl_e32m4(packn);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        for (int g = 0; g < local_block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            const int k0 = k + g * block_size;
            const int k0_pack = k0 / packn_fp16;
            const int k0_lane = k0 % packn_fp16;
            const __fp16* p0_base = (const __fp16*)A + (size_t)k0_pack * A_hstep * packn_fp16 + (i + ii) * packn_fp16 + k0_lane;
            const int head_kk = std::min(max_kk0, packn_fp16 - k0_lane);
            vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(0.f, vl_packn);

            if (elempack == packn_fp16)
            {
                const __fp16* p0 = p0_base;
                const float* ps = input_scale_ptr + g * block_size;

                int kk = 0;
                for (; kk < head_kk; kk++)
                {
                    vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0, packn_fp16 * sizeof(__fp16), vl_packn);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                    _v = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(_v, *ps++, vl_packn), vl_packn);
                    _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl_packn);
                    p0++;
                }
                if (kk == packn_fp16 - k0_lane)
                    p0 += A_hstep * packn_fp16 - packn_fp16;
                for (; kk + (packn_fp16 - 1) < max_kk0; kk += packn_fp16)
                {
                    for (int l = 0; l < packn_fp16; l++)
                    {
                        vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0 + l, packn_fp16 * sizeof(__fp16), vl_packn);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                        _v = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(_v, *ps++, vl_packn), vl_packn);
                        _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl_packn);
                    }
                    p0 += A_hstep * packn_fp16;
                }
            }
            if (elempack == 1)
            {
                const __fp16* p0 = (const __fp16*)A + (size_t)(k + g * block_size) * A_hstep + i + ii;
                const float* ps = input_scale_ptr + g * block_size;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0, vl_packn);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                    _v = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(_v, *ps++, vl_packn), vl_packn);
                    _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl_packn);
                    p0 += A_hstep;
                }
            }

            vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl_packn);
            _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl_packn), vl_packn);
            __riscv_vse32_v_f32m4(pd, __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl_packn), vl_packn);
            pd += packn;

            if (elempack == packn_fp16)
            {
                const __fp16* p0 = p0_base;
                const float* ps = input_scale_ptr + g * block_size;

                int kk = 0;
                for (; kk < head_kk; kk++)
                {
                    vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0, packn_fp16 * sizeof(__fp16), vl_packn);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                    _v = __riscv_vfmul_vf_f32m4(_v, *ps++, vl_packn);
                    __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl_packn), vl_packn), vl_packn);
                    pp += packn;
                    p0++;
                }
                if (kk == packn_fp16 - k0_lane)
                    p0 += A_hstep * packn_fp16 - packn_fp16;
                for (; kk + (packn_fp16 - 1) < max_kk0; kk += packn_fp16)
                {
                    for (int l = 0; l < packn_fp16; l++)
                    {
                        vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0 + l, packn_fp16 * sizeof(__fp16), vl_packn);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                        _v = __riscv_vfmul_vf_f32m4(_v, *ps++, vl_packn);
                        __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl_packn), vl_packn), vl_packn);
                        pp += packn;
                    }
                    p0 += A_hstep * packn_fp16;
                }
            }
            if (elempack == 1)
            {
                const __fp16* p0 = (const __fp16*)A + (size_t)(k + g * block_size) * A_hstep + i + ii;
                const float* ps = input_scale_ptr + g * block_size;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0, vl_packn);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                    _v = __riscv_vfmul_vf_f32m4(_v, *ps++, vl_packn);
                    __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl_packn), vl_packn), vl_packn);
                    pp += packn;
                    p0 += A_hstep;
                }
            }
        }
    }
#endif // __riscv_vector && __riscv_zvfh
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector && __riscv_zvfh
        if (elempack == packn_fp16)
        {
            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const int k0 = k + g * block_size;
                const int k0_pack = k0 / packn_fp16;
                const int k0_lane = k0 % packn_fp16;
                const __fp16* p0 = (const __fp16*)A + (size_t)k0_pack * A_hstep * packn_fp16 + (i + ii) * packn_fp16 + k0_lane;
                const __fp16* p0a = p0;
                const float* ps = input_scale_ptr + g * block_size;
                const float* psa = ps;
                float absmax0 = 0.f;
                float absmax1 = 0.f;

                int kk = 0;
                int kk_lane = k0_lane;
                const size_t vl = packn_fp16;
                vuint16m2_t _idx = __riscv_vid_v_u16m2(vl);
                while (kk < max_kk0)
                {
                    const int n = std::min(max_kk0 - kk, packn_fp16 - kk_lane);
                    vbool8_t _mask = __riscv_vmsltu_vx_u16m2_b8(_idx, n, vl);
                    vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0a, vl), vl);
                    vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0a + packn_fp16, vl), vl);
                    vfloat32m4_t _s = __riscv_vle32_v_f32m4_m(_mask, psa, vl);
                    _v0 = __riscv_vfmul_vv_f32m4(__riscv_vfabs_v_f32m4(_v0, vl), _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m4(__riscv_vfabs_v_f32m4(_v1, vl), _s, vl);
                    _v0 = __riscv_vfmerge_vfm_f32m4(_v0, 0.f, __riscv_vmnot_m_b8(_mask, vl), vl);
                    _v1 = __riscv_vfmerge_vfm_f32m4(_v1, 0.f, __riscv_vmnot_m_b8(_mask, vl), vl);
                    absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                    absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                    p0a += n;
                    psa += n;
                    kk += n;
                    kk_lane += n;
                    if (kk_lane == packn_fp16)
                    {
                        p0a += A_hstep * packn_fp16 - packn_fp16;
                        kk_lane = 0;
                    }
                }

                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                kk = 0;
                kk_lane = k0_lane;
                while (kk < max_kk0)
                {
                    const int n = std::min(max_kk0 - kk, packn_fp16 - kk_lane);
                    vbool8_t _mask = __riscv_vmsltu_vx_u16m2_b8(_idx, n, vl);
                    vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0, vl), vl);
                    vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0 + packn_fp16, vl), vl);
                    vfloat32m4_t _s = __riscv_vle32_v_f32m4_m(_mask, ps, vl);
                    _v0 = __riscv_vfmul_vv_f32m4(_v0, _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m4(_v1, _s, vl);
                    vint8m1x2_t _q = __riscv_vcreate_v_i8m1x2(
                                         float2int8(__riscv_vfmul_vf_f32m4(_v0, scale0, vl), vl),
                                         float2int8(__riscv_vfmul_vf_f32m4(_v1, scale1, vl), vl));
                    __riscv_vsseg2e8_v_i8m1x2_m(_mask, pp, _q, vl);
                    pp += n * 2;
                    p0 += n;
                    ps += n;
                    kk += n;
                    kk_lane += n;
                    if (kk_lane == packn_fp16)
                    {
                        p0 += A_hstep * packn_fp16 - packn_fp16;
                        kk_lane = 0;
                    }
                }
            }
        }
#endif // __riscv_vector && __riscv_zvfh
        if (elempack == 1)
        {
            const __fp16* p0 = (const __fp16*)A + (size_t)k * A_hstep + i + ii;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const __fp16* p0a = p0;
                const float* psa = ps;

                int kk = 0;
#if __riscv_vector && __riscv_zvfh
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                    vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0a, A_hstep * sizeof(__fp16), vl), vl);
                    vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0a + 1, A_hstep * sizeof(__fp16), vl), vl);
                    vfloat32m4_t _s = __riscv_vle32_v_f32m4(psa, vl);
                    _v0 = __riscv_vfabs_v_f32m4(__riscv_vfmul_vv_f32m4(_v0, _s, vl), vl);
                    _v1 = __riscv_vfabs_v_f32m4(__riscv_vfmul_vv_f32m4(_v1, _s, vl), vl);
                    absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                    absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                    p0a += vl * A_hstep;
                    psa += vl;
                    kk += vl;
                }
#else
                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf((float)p0a[0]) * s);
                    absmax1 = std::max(absmax1, fabsf((float)p0a[1]) * s);
                    p0a += A_hstep;
                }
#endif // __riscv_vector && __riscv_zvfh

                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                kk = 0;
#if __riscv_vector && __riscv_zvfh
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                    vfloat32m4_t _v0 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0, A_hstep * sizeof(__fp16), vl), vl);
                    vfloat32m4_t _v1 = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0 + 1, A_hstep * sizeof(__fp16), vl), vl);
                    vfloat32m4_t _s = __riscv_vle32_v_f32m4(ps, vl);
                    _v0 = __riscv_vfmul_vv_f32m4(_v0, _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m4(_v1, _s, vl);
                    vint8m1_t _q0 = float2int8(__riscv_vfmul_vf_f32m4(_v0, scale0, vl), vl);
                    vint8m1_t _q1 = float2int8(__riscv_vfmul_vf_f32m4(_v1, scale1, vl), vl);
                    vint8m1x2_t _q = __riscv_vcreate_v_i8m1x2(_q0, _q1);
                    __riscv_vsseg2e8_v_i8m1x2(pp, _q, vl);
                    pp += vl * 2;
                    p0 += vl * A_hstep;
                    ps += vl;
                    kk += vl;
                }
#else
                for (; kk < max_kk0; kk++)
                {
                    const float s = *ps++;
                    pp[0] = float2int8((float)p0[0] * s * scale0);
                    pp[1] = float2int8((float)p0[1] * s * scale1);
                    pp += 2;
                    p0 += A_hstep;
                }
#endif // __riscv_vector && __riscv_zvfh
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
#if __riscv_vector && __riscv_zvfh
        if (elempack == packn_fp16)
        {
            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const int k0 = k + g * block_size;
                const int k0_pack = k0 / packn_fp16;
                const int k0_lane = k0 % packn_fp16;
                const __fp16* p0 = (const __fp16*)A + (size_t)k0_pack * A_hstep * packn_fp16 + (i + ii) * packn_fp16 + k0_lane;
                const __fp16* p0a = p0;
                const float* ps = input_scale_ptr + g * block_size;
                const float* psa = ps;
                float absmax = 0.f;

                int kk = 0;
                int kk_lane = k0_lane;
                const size_t vl = packn_fp16;
                vuint16m2_t _idx = __riscv_vid_v_u16m2(vl);
                while (kk < max_kk0)
                {
                    const int n = std::min(max_kk0 - kk, packn_fp16 - kk_lane);
                    vbool8_t _mask = __riscv_vmsltu_vx_u16m2_b8(_idx, n, vl);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0a, vl), vl);
                    vfloat32m4_t _s = __riscv_vle32_v_f32m4_m(_mask, psa, vl);
                    _v = __riscv_vfmul_vv_f32m4(__riscv_vfabs_v_f32m4(_v, vl), _s, vl);
                    _v = __riscv_vfmerge_vfm_f32m4(_v, 0.f, __riscv_vmnot_m_b8(_mask, vl), vl);
                    absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                    p0a += n;
                    psa += n;
                    kk += n;
                    kk_lane += n;
                    if (kk_lane == packn_fp16)
                    {
                        p0a += A_hstep * packn_fp16 - packn_fp16;
                        kk_lane = 0;
                    }
                }

                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;

                kk = 0;
                kk_lane = k0_lane;
                while (kk < max_kk0)
                {
                    const int n = std::min(max_kk0 - kk, packn_fp16 - kk_lane);
                    vbool8_t _mask = __riscv_vmsltu_vx_u16m2_b8(_idx, n, vl);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2_m(_mask, p0, vl), vl);
                    _v = __riscv_vfmul_vv_f32m4(_v, __riscv_vle32_v_f32m4_m(_mask, ps, vl), vl);
                    vint8m1_t _q = float2int8(__riscv_vfmul_vf_f32m4(_v, scale, vl), vl);
                    __riscv_vse8_v_i8m1_m(_mask, pp, _q, vl);
                    pp += n;
                    p0 += n;
                    ps += n;
                    kk += n;
                    kk_lane += n;
                    if (kk_lane == packn_fp16)
                    {
                        p0 += A_hstep * packn_fp16 - packn_fp16;
                        kk_lane = 0;
                    }
                }
            }
        }
#endif // __riscv_vector && __riscv_zvfh
        if (elempack == 1)
        {
            const __fp16* p0 = (const __fp16*)A + (size_t)k * A_hstep + i + ii;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax = 0.f;
                const __fp16* p0a = p0;
                const float* psa = ps;

                int kk = 0;
#if __riscv_vector && __riscv_zvfh
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0a, A_hstep * sizeof(__fp16), vl), vl);
                    _v = __riscv_vfabs_v_f32m4(__riscv_vfmul_vv_f32m4(_v, __riscv_vle32_v_f32m4(psa, vl), vl), vl);
                    absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                    p0a += vl * A_hstep;
                    psa += vl;
                    kk += vl;
                }
#else
                for (; kk < max_kk0; kk++)
                {
                    absmax = std::max(absmax, fabsf((float)*p0a) * *psa++);
                    p0a += A_hstep;
                }
#endif // __riscv_vector && __riscv_zvfh

                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;

                kk = 0;
#if __riscv_vector && __riscv_zvfh
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e16m2(max_kk0 - kk);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vlse16_v_f16m2(p0, A_hstep * sizeof(__fp16), vl), vl);
                    _v = __riscv_vfmul_vv_f32m4(_v, __riscv_vle32_v_f32m4(ps, vl), vl);
                    __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vf_f32m4(_v, scale, vl), vl), vl);
                    pp += vl;
                    p0 += vl * A_hstep;
                    ps += vl;
                    kk += vl;
                }
#else
                for (; kk < max_kk0; kk++)
                {
                    *pp++ = float2int8((float)*p0 * *ps++ * scale);
                    p0 += A_hstep;
                }
#endif // __riscv_vector && __riscv_zvfh
            }
        }
    }
}
