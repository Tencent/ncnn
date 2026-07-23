// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// output-major tile, block-major within each output tile
static void pack_B_tile_wq_int8(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
    const size_t B_hstep = B.w;
#if __riscv_vector
    const ptrdiff_t B_stride = (ptrdiff_t)B_hstep;
    const size_t vl4 = __riscv_vsetvl_e8m1(4);
    const size_t vl2 = __riscv_vsetvl_e8m1(2);
#endif // __riscv_vector
    const int block_count = (K + block_size - 1) / block_size;
    signed char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __riscv_vector
                __riscv_vse8_v_i8m1(pp, __riscv_vlse8_v_i8m1(p0, B_stride, vl4), vl4);
                __riscv_vse8_v_i8m1(pp + 4, __riscv_vlse8_v_i8m1(p0 + 1, B_stride, vl4), vl4);
                __riscv_vse8_v_i8m1(pp + 8, __riscv_vlse8_v_i8m1(p0 + 2, B_stride, vl4), vl4);
                __riscv_vse8_v_i8m1(pp + 12, __riscv_vlse8_v_i8m1(p0 + 3, B_stride, vl4), vl4);
#else
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[B_hstep];
                pp[5] = p0[B_hstep + 1];
                pp[6] = p0[B_hstep + 2];
                pp[7] = p0[B_hstep + 3];
                pp[8] = p0[B_hstep * 2];
                pp[9] = p0[B_hstep * 2 + 1];
                pp[10] = p0[B_hstep * 2 + 2];
                pp[11] = p0[B_hstep * 2 + 3];
                pp[12] = p0[B_hstep * 3];
                pp[13] = p0[B_hstep * 3 + 1];
                pp[14] = p0[B_hstep * 3 + 2];
                pp[15] = p0[B_hstep * 3 + 3];
#endif // __riscv_vector
                p0 += 4;
                pp += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __riscv_vector
                __riscv_vse8_v_i8m1(pp, __riscv_vlse8_v_i8m1(p0, B_stride, vl4), vl4);
                __riscv_vse8_v_i8m1(pp + 4, __riscv_vlse8_v_i8m1(p0 + 1, B_stride, vl4), vl4);
#else
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[B_hstep];
                pp[3] = p0[B_hstep + 1];
                pp[4] = p0[B_hstep * 2];
                pp[5] = p0[B_hstep * 2 + 1];
                pp[6] = p0[B_hstep * 3];
                pp[7] = p0[B_hstep * 3 + 1];
#endif // __riscv_vector
                p0 += 2;
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
#if __riscv_vector
                __riscv_vse8_v_i8m1(pp, __riscv_vlse8_v_i8m1(p0, B_stride, vl4), vl4);
#else
                pp[0] = p0[0];
                pp[1] = p0[B_hstep];
                pp[2] = p0[B_hstep * 2];
                pp[3] = p0[B_hstep * 3];
#endif // __riscv_vector
                p0++;
                pp += 4;
            }

            pd[0] = 1.f / *ps0++;
            pd[1] = 1.f / *ps1++;
            pd[2] = 1.f / *ps2++;
            pd[3] = 1.f / *ps3++;
            pd += 4;
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __riscv_vector
                __riscv_vse8_v_i8m1(pp, __riscv_vlse8_v_i8m1(p0, B_stride, vl2), vl2);
                __riscv_vse8_v_i8m1(pp + 2, __riscv_vlse8_v_i8m1(p0 + 1, B_stride, vl2), vl2);
                __riscv_vse8_v_i8m1(pp + 4, __riscv_vlse8_v_i8m1(p0 + 2, B_stride, vl2), vl2);
                __riscv_vse8_v_i8m1(pp + 6, __riscv_vlse8_v_i8m1(p0 + 3, B_stride, vl2), vl2);
#else
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[B_hstep];
                pp[5] = p0[B_hstep + 1];
                pp[6] = p0[B_hstep + 2];
                pp[7] = p0[B_hstep + 3];
#endif // __riscv_vector
                p0 += 4;
                pp += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __riscv_vector
                __riscv_vse8_v_i8m1(pp, __riscv_vlse8_v_i8m1(p0, B_stride, vl2), vl2);
                __riscv_vse8_v_i8m1(pp + 2, __riscv_vlse8_v_i8m1(p0 + 1, B_stride, vl2), vl2);
#else
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[B_hstep];
                pp[3] = p0[B_hstep + 1];
#endif // __riscv_vector
                p0 += 2;
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
#if __riscv_vector
                __riscv_vse8_v_i8m1(pp, __riscv_vlse8_v_i8m1(p0, B_stride, vl2), vl2);
#else
                pp[0] = p0[0];
                pp[1] = p0[B_hstep];
#endif // __riscv_vector
                p0++;
                pp += 2;
            }

            pd[0] = 1.f / *ps0++;
            pd[1] = 1.f / *ps1++;
            pd += 2;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            for (int kk = 0; kk < max_kk; kk++)
                *pp++ = *p0++;
            *pd++ = 1.f / *ps0++;
        }
    }
}

// K-major, row-interleaved MR-packn/MR2/MR1
static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __riscv_vector
        const int packn = csrr_vlenb() / 4;
        const size_t vl = __riscv_vsetvl_e32m1(packn);
        const ptrdiff_t A_stride = (ptrdiff_t)A_hstep * sizeof(float);
        for (; ii + (packn - 1) < max_ii; ii += packn)
        {
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                vfloat32m1_t _absmax = __riscv_vfmv_v_f_f32m1(0.f, vl);
                const float* p0a = p0;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat32m1_t _v = __riscv_vlse32_v_f32m1(p0a, A_stride, vl);
                    _absmax = __riscv_vfmax_vv_f32m1(_absmax, __riscv_vfabs_v_f32m1(_v, vl), vl);
                    p0a++;
                }

                vfloat32m1_t _scale = __riscv_vfrdiv_vf_f32m1(_absmax, 127.f, vl);
                _scale = __riscv_vfmerge_vfm_f32m1(_scale, 0.f, __riscv_vmfeq_vf_f32m1_b32(_absmax, 0.f, vl), vl);
                __riscv_vse32_v_f32m1(pd, __riscv_vfmul_vf_f32m1(_absmax, 1.f / 127.f, vl), vl);
                pd += packn;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat32m1_t _v = __riscv_vlse32_v_f32m1(p0, A_stride, vl);
                    vint32m1_t _v32 = __riscv_vfcvt_x_f_v_i32m1_rm(__riscv_vfmul_vv_f32m1(_v, _scale, vl), __RISCV_FRM_RMM, vl);
                    _v32 = __riscv_vmax_vx_i32m1(_v32, -127, vl);
                    _v32 = __riscv_vmin_vx_i32m1(_v32, 127, vl);
                    vint32m4_t _v32x4 = __riscv_vundefined_i32m4();
                    _v32x4 = __riscv_vset_v_i32m1_i32m4(_v32x4, 0, _v32);
                    vint16m2_t _v16 = __riscv_vnclip_wx_i16m2(_v32x4, 0, __RISCV_VXRM_RNU, vl);
                    __riscv_vse8_v_i8m1(pp, __riscv_vnclip_wx_i8m1(_v16, 0, __RISCV_VXRM_RNU, vl), vl);
                    pp += packn;
                    p0++;
                }
            }
        }
#endif // __riscv_vector
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;
            const float* p1 = p0 + A_hstep;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0;
                const float* p1a = p1;

                int kk = 0;
#if __riscv_vector
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                    vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(p0a, vl);
                    vfloat32m8_t _v1 = __riscv_vle32_v_f32m8(p1a, vl);
                    _v0 = __riscv_vfabs_v_f32m8(_v0, vl);
                    _v1 = __riscv_vfabs_v_f32m8(_v1, vl);
                    absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                    absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                    p0a += vl;
                    p1a += vl;
                    kk += vl;
                }
#endif // __riscv_vector
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0a++;
                    float v1 = *p1a++;
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                }

                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                kk = 0;
#if __riscv_vector
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                    vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(p0, vl);
                    vfloat32m8_t _v1 = __riscv_vle32_v_f32m8(p1, vl);
                    vint8m2_t _q0 = float2int8(__riscv_vfmul_vf_f32m8(_v0, scale0, vl), vl);
                    vint8m2_t _q1 = float2int8(__riscv_vfmul_vf_f32m8(_v1, scale1, vl), vl);
                    vint8m2x2_t _q = __riscv_vcreate_v_i8m2x2(_q0, _q1);
                    __riscv_vsseg2e8_v_i8m2x2(pp, _q, vl);
                    pp += vl * 2;
                    p0 += vl;
                    p1 += vl;
                    kk += vl;
                }
#endif // __riscv_vector
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0++;
                    float v1 = *p1++;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax = 0.f;
                const float* p0a = p0;

                int kk = 0;
#if __riscv_vector
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                    vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0a, vl);
                    _v = __riscv_vfabs_v_f32m8(_v, vl);
                    absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                    p0a += vl;
                    kk += vl;
                }
#endif // __riscv_vector
                for (; kk < max_kk0; kk++)
                {
                    float v = *p0a++;
                    absmax = std::max(absmax, fabsf(v));
                }

                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;

                kk = 0;
#if __riscv_vector
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                    vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0, vl);
                    __riscv_vse8_v_i8m2(pp, float2int8(__riscv_vfmul_vf_f32m8(_v, scale, vl), vl), vl);
                    pp += vl;
                    p0 += vl;
                    kk += vl;
                }
#endif // __riscv_vector
                for (; kk < max_kk0; kk++)
                {
                    float v = *p0++;
                    *pp++ = float2int8(v * scale);
                }
            }
        }
        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    const ptrdiff_t A_stride = (ptrdiff_t)A_hstep * sizeof(float);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            vfloat32m1_t _absmax = __riscv_vfmv_v_f_f32m1(0.f, vl);
            const float* p0a = p0;
            const float* psa = ps;

            for (int kk = 0; kk < max_kk0; kk++)
            {
                vfloat32m1_t _v = __riscv_vlse32_v_f32m1(p0a, A_stride, vl);
                _v = __riscv_vfabs_v_f32m1(_v, vl);
                _v = __riscv_vfmul_vf_f32m1(_v, *psa++, vl);
                _absmax = __riscv_vfmax_vv_f32m1(_absmax, _v, vl);
                p0a++;
            }

            vfloat32m1_t _scale = __riscv_vfrdiv_vf_f32m1(_absmax, 127.f, vl);
            _scale = __riscv_vfmerge_vfm_f32m1(_scale, 0.f, __riscv_vmfeq_vf_f32m1_b32(_absmax, 0.f, vl), vl);
            __riscv_vse32_v_f32m1(pd, __riscv_vfmul_vf_f32m1(_absmax, 1.f / 127.f, vl), vl);
            pd += packn;

            for (int kk = 0; kk < max_kk0; kk++)
            {
                vfloat32m1_t _v = __riscv_vlse32_v_f32m1(p0, A_stride, vl);
                _v = __riscv_vfmul_vf_f32m1(_v, *ps++, vl);
                vint32m1_t _v32 = __riscv_vfcvt_x_f_v_i32m1_rm(__riscv_vfmul_vv_f32m1(_v, _scale, vl), __RISCV_FRM_RMM, vl);
                _v32 = __riscv_vmax_vx_i32m1(_v32, -127, vl);
                _v32 = __riscv_vmin_vx_i32m1(_v32, 127, vl);
                vint32m4_t _v32x4 = __riscv_vundefined_i32m4();
                _v32x4 = __riscv_vset_v_i32m1_i32m4(_v32x4, 0, _v32);
                vint16m2_t _v16 = __riscv_vnclip_wx_i16m2(_v32x4, 0, __RISCV_VXRM_RNU, vl);
                __riscv_vse8_v_i8m1(pp, __riscv_vnclip_wx_i8m1(_v16, 0, __RISCV_VXRM_RNU, vl), vl);
                pp += packn;
                p0++;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;
        const float* p1 = p0 + A_hstep;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0;
            const float* p1a = p1;
            const float* psa = ps;

            int kk = 0;
#if __riscv_vector
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(p0a, vl);
                vfloat32m8_t _v1 = __riscv_vle32_v_f32m8(p1a, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(psa, vl);
                _v0 = __riscv_vfabs_v_f32m8(_v0, vl);
                _v1 = __riscv_vfabs_v_f32m8(_v1, vl);
                _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                p0a += vl;
                p1a += vl;
                psa += vl;
                kk += vl;
            }
#endif // __riscv_vector
            for (; kk < max_kk0; kk++)
            {
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(*p0a++) * s);
                absmax1 = std::max(absmax1, fabsf(*p1a++) * s);
            }

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

            kk = 0;
#if __riscv_vector
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(p0, vl);
                vfloat32m8_t _v1 = __riscv_vle32_v_f32m8(p1, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(ps, vl);
                _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                vint8m2_t _q0 = float2int8(__riscv_vfmul_vf_f32m8(_v0, scale0, vl), vl);
                vint8m2_t _q1 = float2int8(__riscv_vfmul_vf_f32m8(_v1, scale1, vl), vl);
                vint8m2x2_t _q = __riscv_vcreate_v_i8m2x2(_q0, _q1);
                __riscv_vsseg2e8_v_i8m2x2(pp, _q, vl);
                pp += vl * 2;
                p0 += vl;
                p1 += vl;
                ps += vl;
                kk += vl;
            }
#endif // __riscv_vector
            for (; kk < max_kk0; kk++)
            {
                float v0 = *p0++;
                float v1 = *p1++;
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
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax = 0.f;
            const float* p0a = p0;
            const float* psa = ps;

            int kk = 0;
#if __riscv_vector
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0a, vl);
                _v = __riscv_vfabs_v_f32m8(_v, vl);
                _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(psa, vl), vl);
                absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                p0a += vl;
                psa += vl;
                kk += vl;
            }
#endif // __riscv_vector
            for (; kk < max_kk0; kk++)
            {
                absmax = std::max(absmax, fabsf(*p0a++) * *psa++);
            }

            const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
            *pd++ = absmax / 127.f;

            kk = 0;
#if __riscv_vector
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0, vl);
                _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(ps, vl), vl);
                __riscv_vse8_v_i8m2(pp, float2int8(__riscv_vfmul_vf_f32m8(_v, scale, vl), vl), vl);
                pp += vl;
                p0 += vl;
                ps += vl;
                kk += vl;
            }
#endif // __riscv_vector
            for (; kk < max_kk0; kk++)
            {
                float v = *p0++;
                v *= *ps++;
                *pp++ = float2int8(v * scale);
            }
        }
    }
}

// K-major, row-interleaved MR-packn/MR2/MR1
static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __riscv_vector
        const int packn = csrr_vlenb() / 4;
        const size_t vl = __riscv_vsetvl_e32m1(packn);
        for (; ii + (packn - 1) < max_ii; ii += packn)
        {
            const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                vfloat32m1_t _absmax = __riscv_vfmv_v_f_f32m1(0.f, vl);
                const float* p0a = p0;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat32m1_t _v = __riscv_vle32_v_f32m1(p0a, vl);
                    _absmax = __riscv_vfmax_vv_f32m1(_absmax, __riscv_vfabs_v_f32m1(_v, vl), vl);
                    p0a += A_hstep;
                }

                vfloat32m1_t _scale = __riscv_vfrdiv_vf_f32m1(_absmax, 127.f, vl);
                _scale = __riscv_vfmerge_vfm_f32m1(_scale, 0.f, __riscv_vmfeq_vf_f32m1_b32(_absmax, 0.f, vl), vl);
                __riscv_vse32_v_f32m1(pd, __riscv_vfmul_vf_f32m1(_absmax, 1.f / 127.f, vl), vl);
                pd += packn;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    vfloat32m1_t _v = __riscv_vle32_v_f32m1(p0, vl);
                    vint32m1_t _v32 = __riscv_vfcvt_x_f_v_i32m1_rm(__riscv_vfmul_vv_f32m1(_v, _scale, vl), __RISCV_FRM_RMM, vl);
                    _v32 = __riscv_vmax_vx_i32m1(_v32, -127, vl);
                    _v32 = __riscv_vmin_vx_i32m1(_v32, 127, vl);
                    vint32m4_t _v32x4 = __riscv_vundefined_i32m4();
                    _v32x4 = __riscv_vset_v_i32m1_i32m4(_v32x4, 0, _v32);
                    vint16m2_t _v16 = __riscv_vnclip_wx_i16m2(_v32x4, 0, __RISCV_VXRM_RNU, vl);
                    __riscv_vse8_v_i8m1(pp, __riscv_vnclip_wx_i8m1(_v16, 0, __RISCV_VXRM_RNU, vl), vl);
                    pp += packn;
                    p0 += A_hstep;
                }
            }
        }
#endif // __riscv_vector
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0;

                int kk = 0;
#if __riscv_vector
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                    vfloat32m8_t _v0 = __riscv_vlse32_v_f32m8(p0a, (ptrdiff_t)A_hstep * sizeof(float), vl);
                    vfloat32m8_t _v1 = __riscv_vlse32_v_f32m8(p0a + 1, (ptrdiff_t)A_hstep * sizeof(float), vl);
                    _v0 = __riscv_vfabs_v_f32m8(_v0, vl);
                    _v1 = __riscv_vfabs_v_f32m8(_v1, vl);
                    absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                    absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                    p0a += vl * A_hstep;
                    kk += vl;
                }
#endif // __riscv_vector
                for (; kk < max_kk0; kk++)
                {
                    float v0 = p0a[0];
                    float v1 = p0a[1];
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                    p0a += A_hstep;
                }

                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                kk = 0;
#if __riscv_vector
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                    vfloat32m8_t _v0 = __riscv_vlse32_v_f32m8(p0, (ptrdiff_t)A_hstep * sizeof(float), vl);
                    vfloat32m8_t _v1 = __riscv_vlse32_v_f32m8(p0 + 1, (ptrdiff_t)A_hstep * sizeof(float), vl);
                    vint8m2_t _q0 = float2int8(__riscv_vfmul_vf_f32m8(_v0, scale0, vl), vl);
                    vint8m2_t _q1 = float2int8(__riscv_vfmul_vf_f32m8(_v1, scale1, vl), vl);
                    vint8m2x2_t _q = __riscv_vcreate_v_i8m2x2(_q0, _q1);
                    __riscv_vsseg2e8_v_i8m2x2(pp, _q, vl);
                    pp += vl * 2;
                    p0 += vl * A_hstep;
                    kk += vl;
                }
#endif // __riscv_vector
                for (; kk < max_kk0; kk++)
                {
                    float v0 = p0[0];
                    float v1 = p0[1];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                    p0 += A_hstep;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax = 0.f;
                const float* p0a = p0;

                int kk = 0;
#if __riscv_vector
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                    vfloat32m8_t _v = __riscv_vlse32_v_f32m8(p0a, (ptrdiff_t)A_hstep * sizeof(float), vl);
                    _v = __riscv_vfabs_v_f32m8(_v, vl);
                    absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                    p0a += vl * A_hstep;
                    kk += vl;
                }
#endif // __riscv_vector
                for (; kk < max_kk0; kk++)
                {
                    float v = *p0a;
                    absmax = std::max(absmax, fabsf(v));
                    p0a += A_hstep;
                }

                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;

                kk = 0;
#if __riscv_vector
                while (kk < max_kk0)
                {
                    const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                    vfloat32m8_t _v = __riscv_vlse32_v_f32m8(p0, (ptrdiff_t)A_hstep * sizeof(float), vl);
                    __riscv_vse8_v_i8m2(pp, float2int8(__riscv_vfmul_vf_f32m8(_v, scale, vl), vl), vl);
                    pp += vl;
                    p0 += vl * A_hstep;
                    kk += vl;
                }
#endif // __riscv_vector
                for (; kk < max_kk0; kk++)
                {
                    float v = *p0;
                    *pp++ = float2int8(v * scale);
                    p0 += A_hstep;
                }
            }
        }
        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            vfloat32m1_t _absmax = __riscv_vfmv_v_f_f32m1(0.f, vl);
            const float* p0a = p0;
            const float* psa = ps;

            for (int kk = 0; kk < max_kk0; kk++)
            {
                vfloat32m1_t _v = __riscv_vle32_v_f32m1(p0a, vl);
                _v = __riscv_vfabs_v_f32m1(_v, vl);
                _v = __riscv_vfmul_vf_f32m1(_v, *psa++, vl);
                _absmax = __riscv_vfmax_vv_f32m1(_absmax, _v, vl);
                p0a += A_hstep;
            }

            vfloat32m1_t _scale = __riscv_vfrdiv_vf_f32m1(_absmax, 127.f, vl);
            _scale = __riscv_vfmerge_vfm_f32m1(_scale, 0.f, __riscv_vmfeq_vf_f32m1_b32(_absmax, 0.f, vl), vl);
            __riscv_vse32_v_f32m1(pd, __riscv_vfmul_vf_f32m1(_absmax, 1.f / 127.f, vl), vl);
            pd += packn;

            for (int kk = 0; kk < max_kk0; kk++)
            {
                vfloat32m1_t _v = __riscv_vle32_v_f32m1(p0, vl);
                _v = __riscv_vfmul_vf_f32m1(_v, *ps++, vl);
                vint32m1_t _v32 = __riscv_vfcvt_x_f_v_i32m1_rm(__riscv_vfmul_vv_f32m1(_v, _scale, vl), __RISCV_FRM_RMM, vl);
                _v32 = __riscv_vmax_vx_i32m1(_v32, -127, vl);
                _v32 = __riscv_vmin_vx_i32m1(_v32, 127, vl);
                vint32m4_t _v32x4 = __riscv_vundefined_i32m4();
                _v32x4 = __riscv_vset_v_i32m1_i32m4(_v32x4, 0, _v32);
                vint16m2_t _v16 = __riscv_vnclip_wx_i16m2(_v32x4, 0, __RISCV_VXRM_RNU, vl);
                __riscv_vse8_v_i8m1(pp, __riscv_vnclip_wx_i8m1(_v16, 0, __RISCV_VXRM_RNU, vl), vl);
                pp += packn;
                p0 += A_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0;
            const float* psa = ps;

            int kk = 0;
#if __riscv_vector
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                vfloat32m8_t _v0 = __riscv_vlse32_v_f32m8(p0a, (ptrdiff_t)A_hstep * sizeof(float), vl);
                vfloat32m8_t _v1 = __riscv_vlse32_v_f32m8(p0a + 1, (ptrdiff_t)A_hstep * sizeof(float), vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(psa, vl);
                _v0 = __riscv_vfabs_v_f32m8(_v0, vl);
                _v1 = __riscv_vfabs_v_f32m8(_v1, vl);
                _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                p0a += vl * A_hstep;
                psa += vl;
                kk += vl;
            }
#endif // __riscv_vector
            for (; kk < max_kk0; kk++)
            {
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(p0a[0]) * s);
                absmax1 = std::max(absmax1, fabsf(p0a[1]) * s);
                p0a += A_hstep;
            }

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

            kk = 0;
#if __riscv_vector
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                vfloat32m8_t _v0 = __riscv_vlse32_v_f32m8(p0, (ptrdiff_t)A_hstep * sizeof(float), vl);
                vfloat32m8_t _v1 = __riscv_vlse32_v_f32m8(p0 + 1, (ptrdiff_t)A_hstep * sizeof(float), vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(ps, vl);
                _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                vint8m2_t _q0 = float2int8(__riscv_vfmul_vf_f32m8(_v0, scale0, vl), vl);
                vint8m2_t _q1 = float2int8(__riscv_vfmul_vf_f32m8(_v1, scale1, vl), vl);
                vint8m2x2_t _q = __riscv_vcreate_v_i8m2x2(_q0, _q1);
                __riscv_vsseg2e8_v_i8m2x2(pp, _q, vl);
                pp += vl * 2;
                p0 += vl * A_hstep;
                ps += vl;
                kk += vl;
            }
#endif // __riscv_vector
            for (; kk < max_kk0; kk++)
            {
                float v0 = p0[0];
                float v1 = p0[1];
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
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax = 0.f;
            const float* p0a = p0;
            const float* psa = ps;

            int kk = 0;
#if __riscv_vector
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                vfloat32m8_t _v = __riscv_vlse32_v_f32m8(p0a, (ptrdiff_t)A_hstep * sizeof(float), vl);
                _v = __riscv_vfabs_v_f32m8(_v, vl);
                _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(psa, vl), vl);
                absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                p0a += vl * A_hstep;
                psa += vl;
                kk += vl;
            }
#endif // __riscv_vector
            for (; kk < max_kk0; kk++)
            {
                absmax = std::max(absmax, fabsf(*p0a) * *psa++);
                p0a += A_hstep;
            }

            const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
            *pd++ = absmax / 127.f;

            kk = 0;
#if __riscv_vector
            while (kk < max_kk0)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk0 - kk);
                vfloat32m8_t _v = __riscv_vlse32_v_f32m8(p0, (ptrdiff_t)A_hstep * sizeof(float), vl);
                _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(ps, vl), vl);
                __riscv_vse8_v_i8m2(pp, float2int8(__riscv_vfmul_vf_f32m8(_v, scale, vl), vl), vl);
                pp += vl;
                p0 += vl * A_hstep;
                ps += vl;
                kk += vl;
            }
#endif // __riscv_vector
            for (; kk < max_kk0; kk++)
            {
                float v = *p0;
                v *= *ps++;
                *pp++ = float2int8(v * scale);
                p0 += A_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
    const signed char* pAT = AT_tile;
    const float* pAT_descales = AT_descales_tile;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int A_hstep = AT_tile.w;
    const int A_descales_hstep = AT_descales_tile.w;
    const int block_count = (K + block_size - 1) / block_size;
    const int block_start = k / block_size;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl4 = __riscv_vsetvl_e8m1(4);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        const size_t vl = __riscv_vsetvl_e32m1(packn);

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            vfloat32m1_t _fsum2;
            vfloat32m1_t _fsum3;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                _fsum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _fsum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
                _fsum2 = __riscv_vle32_v_f32m1(outptr + packn * 2, vl);
                _fsum3 = __riscv_vle32_v_f32m1(outptr + packn * 3, vl);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum2 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum3 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    uint32_t b = *(const uint32_t*)pB;
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = *(const uint32_t*)(pB + 4);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 2, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = *(const uint32_t*)(pB + 8);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 3, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = *(const uint32_t*)(pB + 12);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += packn * 4;
                    pB += 16;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    uint32_t b = *(const uint32_t*)pB;
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = *(const uint32_t*)(pB + 4);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += packn * 2;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    const uint32_t b = *(const uint32_t*)pB;
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += packn;
                    pB += 4;
                }

                vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), _ad, vl);
                _fsum0 = __riscv_vfmacc_vf_f32m1(_fsum0, pB_descales[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), _ad, vl);
                _fsum1 = __riscv_vfmacc_vf_f32m1(_fsum1, pB_descales[1], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum2, vl), _ad, vl);
                _fsum2 = __riscv_vfmacc_vf_f32m1(_fsum2, pB_descales[2], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum3, vl), _ad, vl);
                _fsum3 = __riscv_vfmacc_vf_f32m1(_fsum3, pB_descales[3], _v, vl);
                pA_descales += packn;
                pB_descales += 4;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + packn, _fsum1, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 2, _fsum2, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 3, _fsum3, vl);
            outptr += packn * 4;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                _fsum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _fsum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    uint16_t b = *(const uint16_t*)pB;
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = *(const uint16_t*)(pB + 2);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 2, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = *(const uint16_t*)(pB + 4);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 3, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = *(const uint16_t*)(pB + 6);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += packn * 4;
                    pB += 8;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    uint16_t b = *(const uint16_t*)pB;
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = *(const uint16_t*)(pB + 2);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += packn * 2;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    const uint16_t b = *(const uint16_t*)pB;
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += packn;
                    pB += 2;
                }

                vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), _ad, vl);
                _fsum0 = __riscv_vfmacc_vf_f32m1(_fsum0, pB_descales[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), _ad, vl);
                _fsum1 = __riscv_vfmacc_vf_f32m1(_fsum1, pB_descales[1], _v, vl);
                pA_descales += packn;
                pB_descales += 2;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + packn, _fsum1, vl);
            outptr += packn * 2;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            vfloat32m1_t _fsum;
            if (k == 0)
                _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            else
                _fsum = __riscv_vle32_v_f32m1(outptr, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint8m1_t _b = __riscv_vle8_v_i8m1(pB, vl4);
                    const signed char b0 = __riscv_vmv_x_s_i8m1_i8(_b);
                    _b = __riscv_vslidedown_vx_i8m1(_b, 1, vl4);
                    const signed char b1 = __riscv_vmv_x_s_i8m1_i8(_b);
                    _b = __riscv_vslidedown_vx_i8m1(_b, 1, vl4);
                    const signed char b2 = __riscv_vmv_x_s_i8m1_i8(_b);
                    _b = __riscv_vslidedown_vx_i8m1(_b, 1, vl4);
                    const signed char b3 = __riscv_vmv_x_s_i8m1_i8(_b);
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, b0, _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, b1, _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 2, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, b2, _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 3, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, b3, _a, vl);
                    pA += packn * 4;
                    pB += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const uint16_t b = *(const uint16_t*)pB;
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, (signed char)b, _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, (signed char)(b >> 8), _a, vl);
                    pA += packn * 2;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pB[0], _a, vl);
                    pA += packn;
                    pB++;
                }

                vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), _ad, vl);
                _fsum = __riscv_vfmacc_vf_f32m1(_fsum, pB_descales[0], _v, vl);
                pA_descales += packn;
                pB_descales++;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += packn;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * packn;
        pAT_descales += A_descales_hstep * packn;
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vget_v_f32m1x2_f32m1(_s, 0);
                _fsum1 = __riscv_vget_v_f32m1x2_f32m1(_s, 1);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 8, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[4], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[5], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 12, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[6], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[7], _b0, vl);
                    pA += 8;
                    pB += 16;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b, vl);
                    pA += 2;
                    pB += 4;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), pA_descales[0], vl);
                _fsum0 = __riscv_vfmacc_vv_f32m1(_fsum0, _bd, _v, vl);
                _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), pA_descales[1], vl);
                _fsum1 = __riscv_vfmacc_vv_f32m1(_fsum1, _bd, _v, vl);
                pA_descales += 2;
                pB_descales += 4;
            }

            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
            outptr += 8;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(2);
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vget_v_f32m1x2_f32m1(_s, 0);
                _fsum1 = __riscv_vget_v_f32m1x2_f32m1(_s, 1);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[4], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[5], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 6, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[6], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[7], _b0, vl);
                    pA += 8;
                    pB += 8;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b, vl);
                    pA += 2;
                    pB += 2;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), pA_descales[0], vl);
                _fsum0 = __riscv_vfmacc_vv_f32m1(_fsum0, _bd, _v, vl);
                _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), pA_descales[1], vl);
                _fsum1 = __riscv_vfmacc_vv_f32m1(_fsum1, _bd, _v, vl);
                pA_descales += 2;
                pB_descales += 2;
            }

            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
            outptr += 4;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            const size_t vl = __riscv_vsetvl_e32m1(1);
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vget_v_f32m1x2_f32m1(_s, 0);
                _fsum1 = __riscv_vget_v_f32m1x2_f32m1(_s, 1);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 1, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[4], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[5], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 3, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[6], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[7], _b0, vl);
                    pA += 8;
                    pB += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 1, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b, vl);
                    pA += 2;
                    pB++;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), pA_descales[0], vl);
                _fsum0 = __riscv_vfmacc_vv_f32m1(_fsum0, _bd, _v, vl);
                _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), pA_descales[1], vl);
                _fsum1 = __riscv_vfmacc_vv_f32m1(_fsum1, _bd, _v, vl);
                pA_descales += 2;
                pB_descales++;
            }

            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
            outptr += 2;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _fsum;
            if (k == 0)
                _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            else
                _fsum = __riscv_vle32_v_f32m1(outptr, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 8, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[2], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 12, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[3], _b0, vl);
                    pA += 4;
                    pB += 16;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    pA += 2;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    pA++;
                    pB += 4;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), pA_descales[0], vl);
                _fsum = __riscv_vfmacc_vv_f32m1(_fsum, _bd, _v, vl);
                pA_descales++;
                pB_descales += 4;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += 4;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(2);
            vfloat32m1_t _fsum;
            if (k == 0)
                _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            else
                _fsum = __riscv_vle32_v_f32m1(outptr, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[2], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 6, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[3], _b0, vl);
                    pA += 4;
                    pB += 8;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    pA++;
                    pB += 2;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), pA_descales[0], vl);
                _fsum = __riscv_vfmacc_vv_f32m1(_fsum, _bd, _v, vl);
                pA_descales++;
                pB_descales += 2;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += 2;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            const size_t vl = __riscv_vsetvl_e32m1(1);
            vfloat32m1_t _fsum;
            if (k == 0)
                _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            else
                _fsum = __riscv_vle32_v_f32m1(outptr, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 1, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[2], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 3, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[3], _b0, vl);
                    pA += 4;
                    pB += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB + 1, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    pA += 2;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    pA++;
                    pB++;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), pA_descales[0], vl);
                _fsum = __riscv_vfmacc_vv_f32m1(_fsum, _bd, _v, vl);
                pA_descales++;
                pB_descales++;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr++;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            float sum00;
            float sum01;
            float sum02;
            float sum03;
            float sum10;
            float sum11;
            float sum12;
            float sum13;
            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum02 = 0.f;
                sum03 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;
                sum12 = 0.f;
                sum13 = 0.f;
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
                sum02 = outptr[4];
                sum12 = outptr[5];
                sum03 = outptr[6];
                sum13 = outptr[7];
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s00 = 0, s01 = 0, s02 = 0, s03 = 0;
                int s10 = 0, s11 = 0, s12 = 0, s13 = 0;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 4;
                    const signed char* b2 = b1 + 4;
                    const signed char* b3 = b2 + 4;
                    s00 += pA[0] * b0[0] + pA[2] * b0[1] + pA[4] * b0[2] + pA[6] * b0[3];
                    s01 += pA[0] * b1[0] + pA[2] * b1[1] + pA[4] * b1[2] + pA[6] * b1[3];
                    s02 += pA[0] * b2[0] + pA[2] * b2[1] + pA[4] * b2[2] + pA[6] * b2[3];
                    s03 += pA[0] * b3[0] + pA[2] * b3[1] + pA[4] * b3[2] + pA[6] * b3[3];
                    s10 += pA[1] * b0[0] + pA[3] * b0[1] + pA[5] * b0[2] + pA[7] * b0[3];
                    s11 += pA[1] * b1[0] + pA[3] * b1[1] + pA[5] * b1[2] + pA[7] * b1[3];
                    s12 += pA[1] * b2[0] + pA[3] * b2[1] + pA[5] * b2[2] + pA[7] * b2[3];
                    s13 += pA[1] * b3[0] + pA[3] * b3[1] + pA[5] * b3[2] + pA[7] * b3[3];
                    pA += 8;
                    pB += 16;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 2;
                    const signed char* b2 = b1 + 2;
                    const signed char* b3 = b2 + 2;
                    s00 += pA[0] * b0[0] + pA[2] * b0[1];
                    s01 += pA[0] * b1[0] + pA[2] * b1[1];
                    s02 += pA[0] * b2[0] + pA[2] * b2[1];
                    s03 += pA[0] * b3[0] + pA[2] * b3[1];
                    s10 += pA[1] * b0[0] + pA[3] * b0[1];
                    s11 += pA[1] * b1[0] + pA[3] * b1[1];
                    s12 += pA[1] * b2[0] + pA[3] * b2[1];
                    s13 += pA[1] * b3[0] + pA[3] * b3[1];
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    s00 += pA[0] * pB[0];
                    s01 += pA[0] * pB[1];
                    s02 += pA[0] * pB[2];
                    s03 += pA[0] * pB[3];
                    s10 += pA[1] * pB[0];
                    s11 += pA[1] * pB[1];
                    s12 += pA[1] * pB[2];
                    s13 += pA[1] * pB[3];
                    pA += 2;
                    pB += 4;
                }

                const float ad0 = pA_descales[0];
                const float ad1 = pA_descales[1];
                const float* bd = pB_descales;
                sum00 += s00 * ad0 * bd[0];
                sum01 += s01 * ad0 * bd[1];
                sum02 += s02 * ad0 * bd[2];
                sum03 += s03 * ad0 * bd[3];
                sum10 += s10 * ad1 * bd[0];
                sum11 += s11 * ad1 * bd[1];
                sum12 += s12 * ad1 * bd[2];
                sum13 += s13 * ad1 * bd[3];
                pA_descales += 2;
                pB_descales += 4;
            }

            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;
            outptr[4] = sum02;
            outptr[5] = sum12;
            outptr[6] = sum03;
            outptr[7] = sum13;
            outptr += 8;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
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
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s00 = 0, s01 = 0, s10 = 0, s11 = 0;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 4;
                    s00 += pA[0] * b0[0] + pA[2] * b0[1] + pA[4] * b0[2] + pA[6] * b0[3];
                    s01 += pA[0] * b1[0] + pA[2] * b1[1] + pA[4] * b1[2] + pA[6] * b1[3];
                    s10 += pA[1] * b0[0] + pA[3] * b0[1] + pA[5] * b0[2] + pA[7] * b0[3];
                    s11 += pA[1] * b1[0] + pA[3] * b1[1] + pA[5] * b1[2] + pA[7] * b1[3];
                    pA += 8;
                    pB += 8;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 2;
                    s00 += pA[0] * b0[0] + pA[2] * b0[1];
                    s01 += pA[0] * b1[0] + pA[2] * b1[1];
                    s10 += pA[1] * b0[0] + pA[3] * b0[1];
                    s11 += pA[1] * b1[0] + pA[3] * b1[1];
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    s00 += pA[0] * pB[0];
                    s01 += pA[0] * pB[1];
                    s10 += pA[1] * pB[0];
                    s11 += pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                }
                const float ad0 = pA_descales[0];
                const float ad1 = pA_descales[1];
                const float* bd = pB_descales;
                sum00 += s00 * ad0 * bd[0];
                sum01 += s01 * ad0 * bd[1];
                sum10 += s10 * ad1 * bd[0];
                sum11 += s11 * ad1 * bd[1];
                pA_descales += 2;
                pB_descales += 2;
            }
            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;
            outptr += 4;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
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
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s0 = 0, s1 = 0;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const signed char* b = pB;
                    s0 += pA[0] * b[0] + pA[2] * b[1] + pA[4] * b[2] + pA[6] * b[3];
                    s1 += pA[1] * b[0] + pA[3] * b[1] + pA[5] * b[2] + pA[7] * b[3];
                    pA += 8;
                    pB += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const signed char* b = pB;
                    s0 += pA[0] * b[0] + pA[2] * b[1];
                    s1 += pA[1] * b[0] + pA[3] * b[1];
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    s0 += pA[0] * pB[0];
                    s1 += pA[1] * pB[0];
                    pA += 2;
                    pB++;
                }
                const float bd = pB_descales[0];
                sum0 += s0 * pA_descales[0] * bd;
                sum1 += s1 * pA_descales[1] * bd;
                pA_descales += 2;
                pB_descales++;
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            float sum0;
            float sum1;
            float sum2;
            float sum3;
            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;
                sum2 = 0.f;
                sum3 = 0.f;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 4;
                    const signed char* b2 = b1 + 4;
                    const signed char* b3 = b2 + 4;
                    s0 += pA[0] * b0[0] + pA[1] * b0[1] + pA[2] * b0[2] + pA[3] * b0[3];
                    s1 += pA[0] * b1[0] + pA[1] * b1[1] + pA[2] * b1[2] + pA[3] * b1[3];
                    s2 += pA[0] * b2[0] + pA[1] * b2[1] + pA[2] * b2[2] + pA[3] * b2[3];
                    s3 += pA[0] * b3[0] + pA[1] * b3[1] + pA[2] * b3[2] + pA[3] * b3[3];
                    pA += 4;
                    pB += 16;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 2;
                    const signed char* b2 = b1 + 2;
                    const signed char* b3 = b2 + 2;
                    s0 += pA[0] * b0[0] + pA[1] * b0[1];
                    s1 += pA[0] * b1[0] + pA[1] * b1[1];
                    s2 += pA[0] * b2[0] + pA[1] * b2[1];
                    s3 += pA[0] * b3[0] + pA[1] * b3[1];
                    pA += 2;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    s0 += pA[0] * pB[0];
                    s1 += pA[0] * pB[1];
                    s2 += pA[0] * pB[2];
                    s3 += pA[0] * pB[3];
                    pA++;
                    pB += 4;
                }
                const float ad = pA_descales[0];
                const float* bd = pB_descales;
                sum0 += s0 * ad * bd[0];
                sum1 += s1 * ad * bd[1];
                sum2 += s2 * ad * bd[2];
                sum3 += s3 * ad * bd[3];
                pA_descales++;
                pB_descales += 4;
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;
            outptr += 4;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
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
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s0 = 0, s1 = 0;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 4;
                    s0 += pA[0] * b0[0] + pA[1] * b0[1] + pA[2] * b0[2] + pA[3] * b0[3];
                    s1 += pA[0] * b1[0] + pA[1] * b1[1] + pA[2] * b1[2] + pA[3] * b1[3];
                    pA += 4;
                    pB += 8;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 2;
                    s0 += pA[0] * b0[0] + pA[1] * b0[1];
                    s1 += pA[0] * b1[0] + pA[1] * b1[1];
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    s0 += pA[0] * pB[0];
                    s1 += pA[0] * pB[1];
                    pA++;
                    pB += 2;
                }
                const float ad = pA_descales[0];
                const float* bd = pB_descales;
                sum0 += s0 * ad * bd[0];
                sum1 += s1 * ad * bd[1];
                pA_descales++;
                pB_descales += 2;
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            float sum;
            if (k == 0)
                sum = 0.f;
            else
                sum = outptr[0];
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s = 0;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const signed char* b = pB;
                    s += pA[0] * b[0] + pA[1] * b[1] + pA[2] * b[2] + pA[3] * b[3];
                    pA += 4;
                    pB += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const signed char* b = pB;
                    s += pA[0] * b[0] + pA[1] * b[1];
                    pA += 2;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    s += pA[0] * pB[0];
                    pA++;
                    pB++;
                }
                sum += s * pA_descales[0] * pB_descales[0];
                pA_descales++;
                pB_descales++;
            }
            outptr[0] = sum;
            outptr++;
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
#endif // __riscv_vector
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const float* pp = topT;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    float* outptr = (float*)top_blob + (size_t)i * out_hstep + j;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl_packn = __riscv_vsetvl_e32m1(packn);
    const ptrdiff_t c_stride = (ptrdiff_t)c_hstep * sizeof(float);
    const ptrdiff_t out_stride = (ptrdiff_t)out_hstep * sizeof(float);
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
                pC += (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        float c0 = 0.f;
        vfloat32m1_t _c = __riscv_vfmv_v_f_f32m1(0.f, vl_packn);
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = pC[0] * beta;
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                _c = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(pC, vl_packn), beta, vl_packn);
        }

        float* out0 = outptr;
        for (int jj = 0; jj < max_jj; jj++)
        {
            vfloat32m1_t _sum = __riscv_vle32_v_f32m1(pp, vl_packn);

            if (pC)
            {
                if (broadcast_type_C == 0)
                    _sum = __riscv_vfadd_vf_f32m1(_sum, c0, vl_packn);
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _sum = __riscv_vfadd_vv_f32m1(_sum, _c, vl_packn);
                if (broadcast_type_C == 3)
                {
                    vfloat32m1_t _c0 = __riscv_vlse32_v_f32m1(pC, c_stride, vl_packn);
                    if (beta == 1.f)
                        _sum = __riscv_vfadd_vv_f32m1(_sum, _c0, vl_packn);
                    else
                        _sum = __riscv_vfmacc_vf_f32m1(_sum, beta, _c0, vl_packn);
                }
                if (broadcast_type_C == 4)
                {
                    _sum = __riscv_vfadd_vf_f32m1(_sum, *pC * beta, vl_packn);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC++;
            }

            if (alpha != 1.f)
                _sum = __riscv_vfmul_vf_f32m1(_sum, alpha, vl_packn);

            __riscv_vsse32_v_f32m1(out0, out_stride, _sum, vl_packn);
            pp += packn;
            out0++;
        }
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

        float* out0 = outptr;
        float* out1 = out0 + out_hstep;
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

            __riscv_vse32_v_f32m4(out0, _sum0, vl);
            __riscv_vse32_v_f32m4(out1, _sum1, vl);
            pp += vl * 2;
            jj += (int)vl;
            out0 += vl;
            out1 += vl;
        }
#endif // __riscv_vector
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

            out0[0] = sum00;
            out0[1] = sum01;
            out0[2] = sum02;
            out0[3] = sum03;
            out1[0] = sum10;
            out1[1] = sum11;
            out1[2] = sum12;
            out1[3] = sum13;
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

            out0[0] = sum00;
            out0[1] = sum01;
            out1[0] = sum10;
            out1[1] = sum11;
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
            out0[0] = sum0;
            out1[0] = sum1;
            out0++;
            out1++;
        }
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

        float* out0 = outptr;
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

            __riscv_vse32_v_f32m4(out0, _sum, vl);
            pp += vl;
            jj += (int)vl;
            out0 += vl;
        }
#endif // __riscv_vector
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
            out0[0] = sum0;
            out0[1] = sum1;
            out0[2] = sum2;
            out0[3] = sum3;
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
            out0[0] = sum0;
            out0[1] = sum1;
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
            out0[0] = sum;
            out0++;
        }
        outptr += out_hstep;
    }
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const float* pp = topT;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    float* outptr = (float*)top_blob + (size_t)j * out_hstep + i;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl_packn = __riscv_vsetvl_e32m1(packn);
    const ptrdiff_t c_stride = (ptrdiff_t)c_hstep * sizeof(float);
    const ptrdiff_t out_stride = (ptrdiff_t)out_hstep * sizeof(float);
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
                pC += (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        float c0 = 0.f;
        vfloat32m1_t _c = __riscv_vfmv_v_f_f32m1(0.f, vl_packn);
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = pC[0] * beta;
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                _c = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(pC, vl_packn), beta, vl_packn);
        }

        float* out0 = outptr;
        for (int jj = 0; jj < max_jj; jj++)
        {
            vfloat32m1_t _sum = __riscv_vle32_v_f32m1(pp, vl_packn);

            if (pC)
            {
                if (broadcast_type_C == 0)
                    _sum = __riscv_vfadd_vf_f32m1(_sum, c0, vl_packn);
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _sum = __riscv_vfadd_vv_f32m1(_sum, _c, vl_packn);
                if (broadcast_type_C == 3)
                {
                    vfloat32m1_t _c0 = __riscv_vlse32_v_f32m1(pC, c_stride, vl_packn);
                    if (beta == 1.f)
                        _sum = __riscv_vfadd_vv_f32m1(_sum, _c0, vl_packn);
                    else
                        _sum = __riscv_vfmacc_vf_f32m1(_sum, beta, _c0, vl_packn);
                }
                if (broadcast_type_C == 4)
                {
                    _sum = __riscv_vfadd_vf_f32m1(_sum, *pC * beta, vl_packn);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC++;
            }

            if (alpha != 1.f)
                _sum = __riscv_vfmul_vf_f32m1(_sum, alpha, vl_packn);

            __riscv_vse32_v_f32m1(out0, _sum, vl_packn);
            pp += packn;
            out0 += out_hstep;
        }
        outptr += packn;
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

        float* out0 = outptr;
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

            vfloat32m4x2_t _sum = __riscv_vcreate_v_f32m4x2(_sum0, _sum1);
            if (out_hstep == 2)
                __riscv_vsseg2e32_v_f32m4x2(out0, _sum, vl);
            else
                __riscv_vssseg2e32_v_f32m4x2(out0, out_stride, _sum, vl);
            pp += vl * 2;
            jj += (int)vl;
            out0 += out_hstep * vl;
        }
#endif // __riscv_vector
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

            out0[0] = sum00;
            out0[1] = sum10;
            out0[out_hstep] = sum01;
            out0[out_hstep + 1] = sum11;
            out0[out_hstep * 2] = sum02;
            out0[out_hstep * 2 + 1] = sum12;
            out0[out_hstep * 3] = sum03;
            out0[out_hstep * 3 + 1] = sum13;
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

            out0[0] = sum00;
            out0[1] = sum10;
            out0[out_hstep] = sum01;
            out0[out_hstep + 1] = sum11;
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
            out0[0] = sum0;
            out0[1] = sum1;
            out0 += out_hstep;
        }
        outptr += 2;
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

        float* out0 = outptr;
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

            if (out_hstep == 1)
                __riscv_vse32_v_f32m4(out0, _sum, vl);
            else
                __riscv_vsse32_v_f32m4(out0, out_stride, _sum, vl);
            pp += vl;
            jj += (int)vl;
            out0 += out_hstep * vl;
        }
#endif // __riscv_vector
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
            out0[0] = sum0;
            out0[out_hstep] = sum1;
            out0[out_hstep * 2] = sum2;
            out0[out_hstep * 3] = sum3;
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
            out0[0] = sum0;
            out0[out_hstep] = sum1;
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
            out0[0] = sum;
            out0 += out_hstep;
        }
        outptr++;
    }
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif // __riscv_vector

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(float)));
#if __riscv_vector
    TILE_M = std::max(packn, tile_size / packn * packn);
    TILE_N = std::max(packn * 4, tile_size / (packn * 4) * (packn * 4));
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(4, tile_size / 4 * 4);
#endif // __riscv_vector
    TILE_K = std::max(block_size, tile_size / block_size * block_size);

    if (K > 0)
    {
        const int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + block_size - 1) / block_size * block_size);
        TILE_K = std::min(TILE_K, K);

        if (nn_K == 1)
        {
            tile_size = std::max(1, (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K));
#if __riscv_vector
            TILE_M = std::max(packn, tile_size / packn * packn);
            TILE_N = std::max(packn * 4, tile_size / (packn * 4) * (packn * 4));
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
            TILE_N = std::max(4, tile_size / 4 * 4);
#endif // __riscv_vector
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        const int nn_M = (M + TILE_M - 1) / TILE_M;
#if __riscv_vector
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif // __riscv_vector
    }

    if (N > 0)
    {
        const int nn_N = (N + TILE_N - 1) / TILE_N;
#if __riscv_vector
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + packn * 4 - 1) / (packn * 4) * (packn * 4));
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#endif // __riscv_vector
    }

    if (nT > 1)
    {
#if __riscv_vector
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + packn - 1) / packn * packn);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif // __riscv_vector
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __riscv_vector
        TILE_M = (constant_TILE_M + packn - 1) / packn * packn;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif // __riscv_vector
    }

    if (constant_TILE_N > 0)
    {
#if __riscv_vector
        TILE_N = (constant_TILE_N + packn * 4 - 1) / (packn * 4) * (packn * 4);
#else
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#endif // __riscv_vector
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, constant_TILE_K / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(TILE_K, K);
    }
}
