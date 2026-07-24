// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// K-major, row-interleaved MR-packn/MR2/MR1
static void quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
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
            float* pd1 = pd + (size_t)block_count * packn;

            for (int g = 0; g < block_count; g++)
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
            pd += (size_t)block_count * packn_a;
        }
#endif // __riscv_vector && __riscv_zvfh
        for (; ii + 1 < max_ii; ii += 2)
        {
            const __fp16* p0 = (const __fp16*)A + (size_t)(i + ii) * A_hstep + k;
            const __fp16* p1 = p0 + A_hstep;

            for (int g = 0; g < block_count; g++)
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

#if __riscv_vector && __riscv_zvfh
                vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(absmax0, vl_packn);
                _absmax = __riscv_vslideup_vx_f32m4(_absmax, __riscv_vfmv_v_f_f32m4(absmax1, vl_packn), 1, vl_packn);
                vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl_packn);
                _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl_packn), vl_packn);
                vbool8_t _mask2 = __riscv_vmsltu_vx_u32m4_b8(__riscv_vid_v_u32m4(vl_packn), 2, vl_packn);
                __riscv_vse32_v_f32m4_m(_mask2, pd, __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl_packn), vl_packn);
                const float scale0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(_scale, 0));
                const float scale1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(__riscv_vslidedown_vx_f32m4(_scale, 1, vl_packn), 0));
#else
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
#endif // __riscv_vector && __riscv_zvfh
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

            for (int g = 0; g < block_count; g++)
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
        float* pd1 = pd + (size_t)block_count * packn;

        for (int g = 0; g < block_count; g++)
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
        pd += (size_t)block_count * packn_a;
    }
#endif // __riscv_vector && __riscv_zvfh
    for (; ii + 1 < max_ii; ii += 2)
    {
        const __fp16* p0 = (const __fp16*)A + (size_t)(i + ii) * A_hstep + k;
        const __fp16* p1 = p0 + A_hstep;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
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

#if __riscv_vector && __riscv_zvfh
            vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(absmax0, vl_packn);
            _absmax = __riscv_vslideup_vx_f32m4(_absmax, __riscv_vfmv_v_f_f32m4(absmax1, vl_packn), 1, vl_packn);
            vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl_packn);
            _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl_packn), vl_packn);
            vbool8_t _mask2 = __riscv_vmsltu_vx_u32m4_b8(__riscv_vid_v_u32m4(vl_packn), 2, vl_packn);
            __riscv_vse32_v_f32m4_m(_mask2, pd, __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl_packn), vl_packn);
            const float scale0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(_scale, 0));
            const float scale1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(__riscv_vslidedown_vx_f32m4(_scale, 1, vl_packn), 0));
#else
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
#endif // __riscv_vector && __riscv_zvfh
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

        for (int g = 0; g < block_count; g++)
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
    const int block_count = (max_kk + block_size - 1) / block_size;
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
            for (int g = 0; g < block_count; g++)
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
                    for (; kk < max_kk0; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0a, packn_fp16 * sizeof(__fp16), vl_packn);
                        vfloat32m4_t _v = __riscv_vfabs_v_f32m4(__riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn), vl_packn);
                        _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl_packn);
                        p0a++;
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
                    for (; kk < max_kk0; kk++)
                    {
                        vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0, packn_fp16 * sizeof(__fp16), vl_packn);
                        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                        __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl_packn), vl_packn), vl_packn);
                        pp += packn;
                        p0++;
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
                for (int g = 0; g < block_count; g++)
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

                    vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(absmax0, vl_packn);
                    _absmax = __riscv_vslideup_vx_f32m4(_absmax, __riscv_vfmv_v_f_f32m4(absmax1, vl_packn), 1, vl_packn);
                    vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl_packn);
                    _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl_packn), vl_packn);
                    vbool8_t _mask2 = __riscv_vmsltu_vx_u32m4_b8(__riscv_vid_v_u32m4(vl_packn), 2, vl_packn);
                    __riscv_vse32_v_f32m4_m(_mask2, pd, __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl_packn), vl_packn);
                    const float scale0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(_scale, 0));
                    const float scale1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(__riscv_vslidedown_vx_f32m4(_scale, 1, vl_packn), 0));
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

                for (int g = 0; g < block_count; g++)
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

#if __riscv_vector && __riscv_zvfh
                    vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(absmax0, vl_packn);
                    _absmax = __riscv_vslideup_vx_f32m4(_absmax, __riscv_vfmv_v_f_f32m4(absmax1, vl_packn), 1, vl_packn);
                    vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl_packn);
                    _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl_packn), vl_packn);
                    vbool8_t _mask2 = __riscv_vmsltu_vx_u32m4_b8(__riscv_vid_v_u32m4(vl_packn), 2, vl_packn);
                    __riscv_vse32_v_f32m4_m(_mask2, pd, __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl_packn), vl_packn);
                    const float scale0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(_scale, 0));
                    const float scale1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(__riscv_vslidedown_vx_f32m4(_scale, 1, vl_packn), 0));
#else
                    const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
#endif // __riscv_vector && __riscv_zvfh
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
                for (int g = 0; g < block_count; g++)
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

                for (int g = 0; g < block_count; g++)
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
        for (int g = 0; g < block_count; g++)
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
                for (; kk < max_kk0; kk++)
                {
                    vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0, packn_fp16 * sizeof(__fp16), vl_packn);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                    _v = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(_v, *ps++, vl_packn), vl_packn);
                    _absmax = __riscv_vfmax_vv_f32m4(_absmax, _v, vl_packn);
                    p0++;
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
                for (; kk < max_kk0; kk++)
                {
                    vfloat16m2_t _p = __riscv_vlse16_v_f16m2(p0, packn_fp16 * sizeof(__fp16), vl_packn);
                    vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(_p, vl_packn);
                    _v = __riscv_vfmul_vf_f32m4(_v, *ps++, vl_packn);
                    __riscv_vse8_v_i8m1(pp, float2int8(__riscv_vfmul_vv_f32m4(_v, _scale, vl_packn), vl_packn), vl_packn);
                    pp += packn;
                    p0++;
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
            for (int g = 0; g < block_count; g++)
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

                vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(absmax0, vl_packn);
                _absmax = __riscv_vslideup_vx_f32m4(_absmax, __riscv_vfmv_v_f_f32m4(absmax1, vl_packn), 1, vl_packn);
                vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl_packn);
                _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl_packn), vl_packn);
                vbool8_t _mask2 = __riscv_vmsltu_vx_u32m4_b8(__riscv_vid_v_u32m4(vl_packn), 2, vl_packn);
                __riscv_vse32_v_f32m4_m(_mask2, pd, __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl_packn), vl_packn);
                const float scale0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(_scale, 0));
                const float scale1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(__riscv_vslidedown_vx_f32m4(_scale, 1, vl_packn), 0));
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

            for (int g = 0; g < block_count; g++)
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

#if __riscv_vector && __riscv_zvfh
                vfloat32m4_t _absmax = __riscv_vfmv_v_f_f32m4(absmax0, vl_packn);
                _absmax = __riscv_vslideup_vx_f32m4(_absmax, __riscv_vfmv_v_f_f32m4(absmax1, vl_packn), 1, vl_packn);
                vfloat32m4_t _scale = __riscv_vfrdiv_vf_f32m4(_absmax, 127.f, vl_packn);
                _scale = __riscv_vfmerge_vfm_f32m4(_scale, 0.f, __riscv_vmfeq_vf_f32m4_b8(_absmax, 0.f, vl_packn), vl_packn);
                vbool8_t _mask2 = __riscv_vmsltu_vx_u32m4_b8(__riscv_vid_v_u32m4(vl_packn), 2, vl_packn);
                __riscv_vse32_v_f32m4_m(_mask2, pd, __riscv_vfmul_vf_f32m4(_absmax, 1.f / 127.f, vl_packn), vl_packn);
                const float scale0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(_scale, 0));
                const float scale1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(__riscv_vslidedown_vx_f32m4(_scale, 1, vl_packn), 0));
#else
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
#endif // __riscv_vector && __riscv_zvfh
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
            for (int g = 0; g < block_count; g++)
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

            for (int g = 0; g < block_count; g++)
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

static void unpack_output_tile_wq_int8_fp16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const float* pp = topT;
    const int out_elempack = top_blob.elempack;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    __fp16* outptr = (__fp16*)top_blob + (size_t)i / out_elempack * out_hstep * out_elempack + j * out_elempack + i % out_elempack;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const int packn_fp16 = packn * 2;
    const size_t vl_packn = __riscv_vsetvl_e32m2(packn);
    const ptrdiff_t c_stride = (ptrdiff_t)c_hstep * sizeof(float);
    const ptrdiff_t out_stride = (ptrdiff_t)out_hstep * sizeof(__fp16);
    const float* pC3 = C;
    if (pC3 && broadcast_type_C == 3)
        pC3 += (size_t)i / c_elempack * c_hstep * c_elempack + j * c_elempack + i % c_elempack;
    bool pC3_high = false;
    bool outptr_high = false;
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC = pC3;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        float c0 = 0.f;
        vfloat32m2_t _c = __riscv_vfmv_v_f_f32m2(0.f, vl_packn);
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = pC[0] * beta;
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                _c = __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(pC, vl_packn), beta, vl_packn);
        }

        __fp16* out0 = outptr;
        for (int jj = 0; jj < max_jj; jj++)
        {
            vfloat32m2_t _sum = __riscv_vle32_v_f32m2(pp, vl_packn);

            if (pC)
            {
                if (broadcast_type_C == 0)
                    _sum = __riscv_vfadd_vf_f32m2(_sum, c0, vl_packn);
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _sum = __riscv_vfadd_vv_f32m2(_sum, _c, vl_packn);
                if (broadcast_type_C == 3)
                {
                    vfloat32m2_t _c0;
                    if (c_elempack != 1)
                        _c0 = __riscv_vle32_v_f32m2(pC, vl_packn);
                    else
                        _c0 = __riscv_vlse32_v_f32m2(pC, c_stride, vl_packn);
                    if (beta == 1.f)
                        _sum = __riscv_vfadd_vv_f32m2(_sum, _c0, vl_packn);
                    else
                        _sum = __riscv_vfmacc_vf_f32m2(_sum, beta, _c0, vl_packn);
                }
                if (broadcast_type_C == 4)
                {
                    _sum = __riscv_vfadd_vf_f32m2(_sum, *pC * beta, vl_packn);
                }
                if (broadcast_type_C == 3)
                    pC += c_elempack;
                if (broadcast_type_C == 4)
                    pC++;
            }

            if (alpha != 1.f)
                _sum = __riscv_vfmul_vf_f32m2(_sum, alpha, vl_packn);

            if (out_elempack == packn_fp16)
                __riscv_vse16_v_f16m1(out0, __riscv_vfncvt_f_f_w_f16m1(_sum, vl_packn), vl_packn);
            if (out_elempack == 1)
                __riscv_vsse16_v_f16m1(out0, out_stride, __riscv_vfncvt_f_f_w_f16m1(_sum, vl_packn), vl_packn);
            pp += packn;
            out0 += out_elempack;
        }
        if (pC3 && broadcast_type_C == 3)
        {
            if (c_elempack == packn_fp16)
            {
                if (pC3_high)
                    pC3 += c_hstep * packn_fp16 - packn;
                else
                    pC3 += packn;
                pC3_high = !pC3_high;
            }
            else
            {
                pC3 += c_hstep * packn;
            }
        }
        if (out_elempack == packn_fp16)
        {
            if (outptr_high)
                outptr += out_hstep * packn_fp16 - packn;
            else
                outptr += packn;
            outptr_high = !outptr_high;
        }
        if (out_elempack == 1)
            outptr += out_hstep * packn;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
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

        __fp16* out0 = outptr;
        __fp16* out1 = out0 + out_hstep;
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
#if __riscv_vector
        while (jj < max_jj)
        {
            const size_t vl = __riscv_vsetvl_e32m4(max_jj - jj);
            vfloat32m4x2_t _s = __riscv_vlseg2e32_v_f32m4x2(pp, vl);
            vfloat32m4_t _sum0 = __riscv_vget_v_f32m4x2_f32m4(_s, 0);
            vfloat32m4_t _sum1 = __riscv_vget_v_f32m4x2_f32m4(_s, 1);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _sum0 = __riscv_vfadd_vf_f32m4(_sum0, c0, vl);
                    _sum1 = __riscv_vfadd_vf_f32m4(_sum1, c1, vl);
                }
                if (broadcast_type_C == 3)
                {
                    vfloat32m4_t _c0 = __riscv_vle32_v_f32m4(pC, vl);
                    vfloat32m4_t _c1 = __riscv_vle32_v_f32m4(pC + c_hstep, vl);
                    if (beta == 1.f)
                    {
                        _sum0 = __riscv_vfadd_vv_f32m4(_sum0, _c0, vl);
                        _sum1 = __riscv_vfadd_vv_f32m4(_sum1, _c1, vl);
                    }
                    else
                    {
                        _sum0 = __riscv_vfmacc_vf_f32m4(_sum0, beta, _c0, vl);
                        _sum1 = __riscv_vfmacc_vf_f32m4(_sum1, beta, _c1, vl);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                    if (beta == 1.f)
                    {
                        _sum0 = __riscv_vfadd_vv_f32m4(_sum0, _c, vl);
                        _sum1 = __riscv_vfadd_vv_f32m4(_sum1, _c, vl);
                    }
                    else
                    {
                        _sum0 = __riscv_vfmacc_vf_f32m4(_sum0, beta, _c, vl);
                        _sum1 = __riscv_vfmacc_vf_f32m4(_sum1, beta, _c, vl);
                    }
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += vl;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f32m4(_sum0, alpha, vl);
                _sum1 = __riscv_vfmul_vf_f32m4(_sum1, alpha, vl);
            }

            __riscv_vse16_v_f16m2(out0, __riscv_vfncvt_f_f_w_f16m2(_sum0, vl), vl);
            __riscv_vse16_v_f16m2(out1, __riscv_vfncvt_f_f_w_f16m2(_sum1, vl), vl);
            pp += vl * 2;
            jj += (int)vl;
            out0 += vl;
            out1 += vl;
        }
#else
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum00 = pp[0];
            float sum10 = pp[1];
            float sum01 = pp[2];
            float sum11 = pp[3];
            float sum02 = pp[4];
            float sum12 = pp[5];
            float sum03 = pp[6];
            float sum13 = pp[7];
            pp += 8;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum00 += c0;
                    sum01 += c0;
                    sum02 += c0;
                    sum03 += c0;
                    sum10 += c1;
                    sum11 += c1;
                    sum12 += c1;
                    sum13 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum02 += pC[2] * beta;
                    sum03 += pC[3] * beta;
                    sum10 += pC[c_hstep] * beta;
                    sum11 += pC[c_hstep + 1] * beta;
                    sum12 += pC[c_hstep + 2] * beta;
                    sum13 += pC[c_hstep + 3] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum02 += pC[2] * beta;
                    sum03 += pC[3] * beta;
                    sum10 += pC[0] * beta;
                    sum11 += pC[1] * beta;
                    sum12 += pC[2] * beta;
                    sum13 += pC[3] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += 4;
            }

            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum10 *= alpha;
                sum01 *= alpha;
                sum11 *= alpha;
                sum02 *= alpha;
                sum12 *= alpha;
                sum03 *= alpha;
                sum13 *= alpha;
            }

            out0[0] = (__fp16)sum00;
            out0[1] = (__fp16)sum01;
            out0[2] = (__fp16)sum02;
            out0[3] = (__fp16)sum03;
            out1[0] = (__fp16)sum10;
            out1[1] = (__fp16)sum11;
            out1[2] = (__fp16)sum12;
            out1[3] = (__fp16)sum13;
            out0 += 4;
            out1 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = pp[0];
            float sum10 = pp[1];
            float sum01 = pp[2];
            float sum11 = pp[3];
            pp += 4;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum00 += c0;
                    sum01 += c0;
                    sum10 += c1;
                    sum11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum10 += pC[c_hstep] * beta;
                    sum11 += pC[c_hstep + 1] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum10 += pC[0] * beta;
                    sum11 += pC[1] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += 2;
            }

            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum10 *= alpha;
                sum01 *= alpha;
                sum11 *= alpha;
            }

            out0[0] = (__fp16)sum00;
            out0[1] = (__fp16)sum01;
            out1[0] = (__fp16)sum10;
            out1[1] = (__fp16)sum11;
            out0 += 2;
            out1 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += c0;
                    sum1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[c_hstep] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[0] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC++;
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }
            out0[0] = (__fp16)sum0;
            out1[0] = (__fp16)sum1;
            out0++;
            out1++;
        }
#endif // __riscv_vector
        outptr += out_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
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

        __fp16* out0 = outptr;
        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[0] * beta;
        }
        int jj = 0;
#if __riscv_vector
        while (jj < max_jj)
        {
            const size_t vl = __riscv_vsetvl_e32m4(max_jj - jj);
            vfloat32m4_t _sum = __riscv_vle32_v_f32m4(pp, vl);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _sum = __riscv_vfadd_vf_f32m4(_sum, c0, vl);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                    if (beta == 1.f)
                        _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                    else
                        _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
                    pC += vl;
                }
            }

            if (alpha != 1.f)
                _sum = __riscv_vfmul_vf_f32m4(_sum, alpha, vl);

            __riscv_vse16_v_f16m2(out0, __riscv_vfncvt_f_f_w_f16m2(_sum, vl), vl);
            pp += vl;
            jj += (int)vl;
            out0 += vl;
        }
#else
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            float sum2 = pp[2];
            float sum3 = pp[3];
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += c0;
                    sum1 += c0;
                    sum2 += c0;
                    sum3 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                    sum2 += pC[2] * beta;
                    sum3 += pC[3] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                    sum2 += pC[2] * beta;
                    sum3 += pC[3] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += 4;
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
                sum2 *= alpha;
                sum3 *= alpha;
            }
            out0[0] = (__fp16)sum0;
            out0[1] = (__fp16)sum1;
            out0[2] = (__fp16)sum2;
            out0[3] = (__fp16)sum3;
            out0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += c0;
                    sum1 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += 2;
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }
            out0[0] = (__fp16)sum0;
            out0[1] = (__fp16)sum1;
            out0 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum = *pp++;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    sum += c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    sum += pC[0] * beta;
                    pC++;
                }
            }

            if (alpha != 1.f)
                sum *= alpha;
            out0[0] = (__fp16)sum;
            out0++;
        }
#endif // __riscv_vector
        outptr += out_hstep;
    }
}

static void transpose_unpack_output_tile_wq_int8_fp16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const float* pp = topT;
    const int out_elempack = top_blob.elempack;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    __fp16* outptr = (__fp16*)top_blob + (size_t)j / out_elempack * out_hstep * out_elempack + i * out_elempack + j % out_elempack;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const int packn_fp16 = packn * 2;
    const size_t vl_packn = __riscv_vsetvl_e32m2(packn);
    const ptrdiff_t c_stride = (ptrdiff_t)c_hstep * sizeof(float);
    const ptrdiff_t out_stride = (ptrdiff_t)out_hstep * sizeof(__fp16);
    const float* pC3 = C;
    if (pC3 && broadcast_type_C == 3)
        pC3 += (size_t)i / c_elempack * c_hstep * c_elempack + j * c_elempack + i % c_elempack;
    bool pC3_high = false;
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC = pC3;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        float c0 = 0.f;
        vfloat32m2_t _c = __riscv_vfmv_v_f_f32m2(0.f, vl_packn);
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = pC[0] * beta;
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                _c = __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(pC, vl_packn), beta, vl_packn);
        }

        __fp16* out0 = outptr;
        for (int jj = 0; jj < max_jj; jj += out_elempack)
        {
            const int max_jj0 = std::min(max_jj - jj, out_elempack);
            for (int p = 0; p < max_jj0; p++)
            {
                vfloat32m2_t _sum = __riscv_vle32_v_f32m2(pp, vl_packn);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                        _sum = __riscv_vfadd_vf_f32m2(_sum, c0, vl_packn);
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        _sum = __riscv_vfadd_vv_f32m2(_sum, _c, vl_packn);
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m2_t _c0;
                        if (c_elempack != 1)
                            _c0 = __riscv_vle32_v_f32m2(pC, vl_packn);
                        else
                            _c0 = __riscv_vlse32_v_f32m2(pC, c_stride, vl_packn);
                        if (beta == 1.f)
                            _sum = __riscv_vfadd_vv_f32m2(_sum, _c0, vl_packn);
                        else
                            _sum = __riscv_vfmacc_vf_f32m2(_sum, beta, _c0, vl_packn);
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum = __riscv_vfadd_vf_f32m2(_sum, *pC * beta, vl_packn);
                    }
                    if (broadcast_type_C == 3)
                        pC += c_elempack;
                    if (broadcast_type_C == 4)
                        pC++;
                }

                if (alpha != 1.f)
                    _sum = __riscv_vfmul_vf_f32m2(_sum, alpha, vl_packn);

                if (out_elempack == packn_fp16)
                {
                    __riscv_vsse16_v_f16m1(out0 + p, packn_fp16 * sizeof(__fp16), __riscv_vfncvt_f_f_w_f16m1(_sum, vl_packn), vl_packn);
                }
                if (out_elempack == 1)
                {
                    __riscv_vse16_v_f16m1(out0, __riscv_vfncvt_f_f_w_f16m1(_sum, vl_packn), vl_packn);
                }
                pp += packn;
            }
            out0 += out_hstep * out_elempack;
        }
        if (pC3 && broadcast_type_C == 3)
        {
            if (c_elempack == packn_fp16)
            {
                if (pC3_high)
                    pC3 += c_hstep * packn_fp16 - packn;
                else
                    pC3 += packn;
                pC3_high = !pC3_high;
            }
            else
            {
                pC3 += c_hstep * packn;
            }
        }
        outptr += packn * out_elempack;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
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

        __fp16* out0 = outptr;
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
#if __riscv_vector
        if (out_elempack == packn_fp16)
        {
            for (; jj < max_jj; jj += packn_fp16)
            {
                const size_t vl = __riscv_vsetvl_e32m2(packn_fp16);
                vfloat32m2x2_t _s = __riscv_vlseg2e32_v_f32m2x2(pp, vl);
                vfloat32m2_t _sum0 = __riscv_vget_v_f32m2x2_f32m2(_s, 0);
                vfloat32m2_t _sum1 = __riscv_vget_v_f32m2x2_f32m2(_s, 1);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfadd_vf_f32m2(_sum0, c0, vl);
                        _sum1 = __riscv_vfadd_vf_f32m2(_sum1, c1, vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m2_t _c0 = __riscv_vle32_v_f32m2(pC, vl);
                        vfloat32m2_t _c1 = __riscv_vle32_v_f32m2(pC + c_hstep, vl);
                        if (beta == 1.f)
                        {
                            _sum0 = __riscv_vfadd_vv_f32m2(_sum0, _c0, vl);
                            _sum1 = __riscv_vfadd_vv_f32m2(_sum1, _c1, vl);
                        }
                        else
                        {
                            _sum0 = __riscv_vfmacc_vf_f32m2(_sum0, beta, _c0, vl);
                            _sum1 = __riscv_vfmacc_vf_f32m2(_sum1, beta, _c1, vl);
                        }
                    }
                    if (broadcast_type_C == 4)
                    {
                        vfloat32m2_t _c = __riscv_vle32_v_f32m2(pC, vl);
                        if (beta == 1.f)
                        {
                            _sum0 = __riscv_vfadd_vv_f32m2(_sum0, _c, vl);
                            _sum1 = __riscv_vfadd_vv_f32m2(_sum1, _c, vl);
                        }
                        else
                        {
                            _sum0 = __riscv_vfmacc_vf_f32m2(_sum0, beta, _c, vl);
                            _sum1 = __riscv_vfmacc_vf_f32m2(_sum1, beta, _c, vl);
                        }
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                        pC += packn_fp16;
                }

                if (alpha != 1.f)
                {
                    _sum0 = __riscv_vfmul_vf_f32m2(_sum0, alpha, vl);
                    _sum1 = __riscv_vfmul_vf_f32m2(_sum1, alpha, vl);
                }

                __riscv_vse16_v_f16m1(out0, __riscv_vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                __riscv_vse16_v_f16m1(out0 + packn_fp16, __riscv_vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                pp += packn_fp16 * 2;
                out0 += out_hstep * packn_fp16;
            }
        }
        if (out_elempack == 1)
        {
            while (jj < max_jj)
            {
                const size_t vl = __riscv_vsetvl_e32m4(max_jj - jj);
                vfloat32m4x2_t _s = __riscv_vlseg2e32_v_f32m4x2(pp, vl);
                vfloat32m4_t _sum0 = __riscv_vget_v_f32m4x2_f32m4(_s, 0);
                vfloat32m4_t _sum1 = __riscv_vget_v_f32m4x2_f32m4(_s, 1);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfadd_vf_f32m4(_sum0, c0, vl);
                        _sum1 = __riscv_vfadd_vf_f32m4(_sum1, c1, vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m4_t _c0 = __riscv_vle32_v_f32m4(pC, vl);
                        vfloat32m4_t _c1 = __riscv_vle32_v_f32m4(pC + c_hstep, vl);
                        if (beta == 1.f)
                        {
                            _sum0 = __riscv_vfadd_vv_f32m4(_sum0, _c0, vl);
                            _sum1 = __riscv_vfadd_vv_f32m4(_sum1, _c1, vl);
                        }
                        else
                        {
                            _sum0 = __riscv_vfmacc_vf_f32m4(_sum0, beta, _c0, vl);
                            _sum1 = __riscv_vfmacc_vf_f32m4(_sum1, beta, _c1, vl);
                        }
                    }
                    if (broadcast_type_C == 4)
                    {
                        vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                        if (beta == 1.f)
                        {
                            _sum0 = __riscv_vfadd_vv_f32m4(_sum0, _c, vl);
                            _sum1 = __riscv_vfadd_vv_f32m4(_sum1, _c, vl);
                        }
                        else
                        {
                            _sum0 = __riscv_vfmacc_vf_f32m4(_sum0, beta, _c, vl);
                            _sum1 = __riscv_vfmacc_vf_f32m4(_sum1, beta, _c, vl);
                        }
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                        pC += vl;
                }
                if (alpha != 1.f)
                {
                    _sum0 = __riscv_vfmul_vf_f32m4(_sum0, alpha, vl);
                    _sum1 = __riscv_vfmul_vf_f32m4(_sum1, alpha, vl);
                }

                vfloat16m2x2_t _sum = __riscv_vcreate_v_f16m2x2(
                                          __riscv_vfncvt_f_f_w_f16m2(_sum0, vl),
                                          __riscv_vfncvt_f_f_w_f16m2(_sum1, vl));
                if (out_hstep == 2)
                    __riscv_vsseg2e16_v_f16m2x2(out0, _sum, vl);
                else
                    __riscv_vssseg2e16_v_f16m2x2(out0, out_stride, _sum, vl);
                pp += vl * 2;
                jj += (int)vl;
                out0 += out_hstep * vl;
            }
        }
#else
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum00 = pp[0];
            float sum10 = pp[1];
            float sum01 = pp[2];
            float sum11 = pp[3];
            float sum02 = pp[4];
            float sum12 = pp[5];
            float sum03 = pp[6];
            float sum13 = pp[7];
            pp += 8;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum00 += c0;
                    sum01 += c0;
                    sum02 += c0;
                    sum03 += c0;
                    sum10 += c1;
                    sum11 += c1;
                    sum12 += c1;
                    sum13 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum02 += pC[2] * beta;
                    sum03 += pC[3] * beta;
                    sum10 += pC[c_hstep] * beta;
                    sum11 += pC[c_hstep + 1] * beta;
                    sum12 += pC[c_hstep + 2] * beta;
                    sum13 += pC[c_hstep + 3] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum02 += pC[2] * beta;
                    sum03 += pC[3] * beta;
                    sum10 += pC[0] * beta;
                    sum11 += pC[1] * beta;
                    sum12 += pC[2] * beta;
                    sum13 += pC[3] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += 4;
            }

            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum10 *= alpha;
                sum01 *= alpha;
                sum11 *= alpha;
                sum02 *= alpha;
                sum12 *= alpha;
                sum03 *= alpha;
                sum13 *= alpha;
            }

            out0[0] = (__fp16)sum00;
            out0[1] = (__fp16)sum10;
            out0[out_hstep] = (__fp16)sum01;
            out0[out_hstep + 1] = (__fp16)sum11;
            out0[out_hstep * 2] = (__fp16)sum02;
            out0[out_hstep * 2 + 1] = (__fp16)sum12;
            out0[out_hstep * 3] = (__fp16)sum03;
            out0[out_hstep * 3 + 1] = (__fp16)sum13;
            out0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = pp[0];
            float sum10 = pp[1];
            float sum01 = pp[2];
            float sum11 = pp[3];
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum00 += c0;
                    sum01 += c0;
                    sum10 += c1;
                    sum11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum10 += pC[c_hstep] * beta;
                    sum11 += pC[c_hstep + 1] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum10 += pC[0] * beta;
                    sum11 += pC[1] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += 2;
            }

            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum10 *= alpha;
                sum01 *= alpha;
                sum11 *= alpha;
            }

            out0[0] = (__fp16)sum00;
            out0[1] = (__fp16)sum10;
            out0[out_hstep] = (__fp16)sum01;
            out0[out_hstep + 1] = (__fp16)sum11;
            out0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += c0;
                    sum1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[c_hstep] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[0] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC++;
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }
            out0[0] = (__fp16)sum0;
            out0[1] = (__fp16)sum1;
            out0 += out_hstep;
        }
#endif // __riscv_vector
        outptr += 2 * out_elempack;
    }
    for (; ii < max_ii; ii++)
    {
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

        __fp16* out0 = outptr;
        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[0] * beta;
        }
        int jj = 0;
#if __riscv_vector
        if (out_elempack == packn_fp16)
        {
            for (; jj < max_jj; jj += packn_fp16)
            {
                const size_t vl = __riscv_vsetvl_e32m2(packn_fp16);
                vfloat32m2_t _sum = __riscv_vle32_v_f32m2(pp, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                        _sum = __riscv_vfadd_vf_f32m2(_sum, c0, vl);
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        vfloat32m2_t _c = __riscv_vle32_v_f32m2(pC, vl);
                        if (beta == 1.f)
                            _sum = __riscv_vfadd_vv_f32m2(_sum, _c, vl);
                        else
                            _sum = __riscv_vfmacc_vf_f32m2(_sum, beta, _c, vl);
                        pC += packn_fp16;
                    }
                }

                if (alpha != 1.f)
                    _sum = __riscv_vfmul_vf_f32m2(_sum, alpha, vl);

                __riscv_vse16_v_f16m1(out0, __riscv_vfncvt_f_f_w_f16m1(_sum, vl), vl);
                pp += packn_fp16;
                out0 += out_hstep * packn_fp16;
            }
        }
        if (out_elempack == 1)
        {
            while (jj < max_jj)
            {
                const size_t vl = __riscv_vsetvl_e32m4(max_jj - jj);
                vfloat32m4_t _sum = __riscv_vle32_v_f32m4(pp, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                        _sum = __riscv_vfadd_vf_f32m4(_sum, c0, vl);
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                        if (beta == 1.f)
                            _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                        else
                            _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
                        pC += vl;
                    }
                }

                if (alpha != 1.f)
                    _sum = __riscv_vfmul_vf_f32m4(_sum, alpha, vl);

                if (out_hstep == 1)
                    __riscv_vse16_v_f16m2(out0, __riscv_vfncvt_f_f_w_f16m2(_sum, vl), vl);
                else
                    __riscv_vsse16_v_f16m2(out0, out_stride, __riscv_vfncvt_f_f_w_f16m2(_sum, vl), vl);
                pp += vl;
                jj += (int)vl;
                out0 += out_hstep * vl;
            }
        }
#else
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            float sum2 = pp[2];
            float sum3 = pp[3];
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += c0;
                    sum1 += c0;
                    sum2 += c0;
                    sum3 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                    sum2 += pC[2] * beta;
                    sum3 += pC[3] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                    sum2 += pC[2] * beta;
                    sum3 += pC[3] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += 4;
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
                sum2 *= alpha;
                sum3 *= alpha;
            }
            out0[0] = (__fp16)sum0;
            out0[out_hstep] = (__fp16)sum1;
            out0[out_hstep * 2] = (__fp16)sum2;
            out0[out_hstep * 3] = (__fp16)sum3;
            out0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += c0;
                    sum1 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC += 2;
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }
            out0[0] = (__fp16)sum0;
            out0[out_hstep] = (__fp16)sum1;
            out0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum = *pp++;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    sum += c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    sum += pC[0] * beta;
                    pC++;
                }
            }

            if (alpha != 1.f)
                sum *= alpha;
            out0[0] = (__fp16)sum;
            out0 += out_hstep;
        }
#endif // __riscv_vector
        outptr += out_elempack;
    }
}
