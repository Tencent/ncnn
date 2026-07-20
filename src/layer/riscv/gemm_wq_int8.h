// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// output-major tile, block-major within each output tile
static int pack_B_wq_int8(const Mat& B, const Mat& B_scales, Mat& packed_B, Mat& packed_B_descales, int N, int K, int block_size, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb();
#else
    const int packn = 4;
#endif
    const int block_count = (K + block_size - 1) / block_size;

    packed_B.create(N * K, (size_t)1u, opt.blob_allocator);
    if (packed_B.empty())
        return -100;
    packed_B.cstep = (size_t)N * K;

    packed_B_descales.create(N * block_count, (size_t)4u, opt.blob_allocator);
    if (packed_B_descales.empty())
        return -100;
    packed_B_descales.cstep = (size_t)N * block_count;

    const int nn_N = (N + packn - 1) / packn;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_N; ppj++)
    {
        const int j = ppj * packn;
        const int max_jj = std::min(N - j, packn);
        signed char* pp = (signed char*)packed_B + (size_t)j * K;
        float* pd = (float*)packed_B_descales + (size_t)j * block_count;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* p0 = B.row<const signed char>(j + jj);
            const signed char* p1 = B.row<const signed char>(j + jj + 1);
            const signed char* p2 = B.row<const signed char>(j + jj + 2);
            const signed char* p3 = B.row<const signed char>(j + jj + 3);
            const float* ps0 = B_scales.row(j + jj);
            const float* ps1 = B_scales.row(j + jj + 1);
            const float* ps2 = B_scales.row(j + jj + 2);
            const float* ps3 = B_scales.row(j + jj + 3);
#if __riscv_vector
            const size_t vl = __riscv_vsetvl_e8mf4(4);
            const ptrdiff_t B_stride = (ptrdiff_t)B.w;
#endif

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
#if __riscv_vector
                    __riscv_vse8_v_i8mf4(pp, __riscv_vlse8_v_i8mf4(p0, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 4, __riscv_vlse8_v_i8mf4(p0 + 1, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 8, __riscv_vlse8_v_i8mf4(p0 + 2, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 12, __riscv_vlse8_v_i8mf4(p0 + 3, B_stride, vl), vl);
#else
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp[4] = p1[0];
                    pp[5] = p1[1];
                    pp[6] = p1[2];
                    pp[7] = p1[3];
                    pp[8] = p2[0];
                    pp[9] = p2[1];
                    pp[10] = p2[2];
                    pp[11] = p2[3];
                    pp[12] = p3[0];
                    pp[13] = p3[1];
                    pp[14] = p3[2];
                    pp[15] = p3[3];
#endif
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                    pp += 16;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
#if __riscv_vector
                    __riscv_vse8_v_i8mf4(pp, __riscv_vlse8_v_i8mf4(p0, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 4, __riscv_vlse8_v_i8mf4(p0 + 1, B_stride, vl), vl);
#else
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
                    pp[4] = p2[0];
                    pp[5] = p2[1];
                    pp[6] = p3[0];
                    pp[7] = p3[1];
#endif
                    p0 += 2;
                    p1 += 2;
                    p2 += 2;
                    p3 += 2;
                    pp += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = *p0++;
                    pp[1] = *p1++;
                    pp[2] = *p2++;
                    pp[3] = *p3++;
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
            const signed char* p1 = B.row<const signed char>(j + jj + 1);
            const float* ps0 = B_scales.row(j + jj);
            const float* ps1 = B_scales.row(j + jj + 1);
#if __riscv_vector
            const size_t vl = __riscv_vsetvl_e8mf4(2);
            const ptrdiff_t B_stride = (ptrdiff_t)B.w;
#endif

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
#if __riscv_vector
                    __riscv_vse8_v_i8mf4(pp, __riscv_vlse8_v_i8mf4(p0, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 2, __riscv_vlse8_v_i8mf4(p0 + 1, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 4, __riscv_vlse8_v_i8mf4(p0 + 2, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 6, __riscv_vlse8_v_i8mf4(p0 + 3, B_stride, vl), vl);
#else
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp[4] = p1[0];
                    pp[5] = p1[1];
                    pp[6] = p1[2];
                    pp[7] = p1[3];
#endif
                    p0 += 4;
                    p1 += 4;
                    pp += 8;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
#if __riscv_vector
                    __riscv_vse8_v_i8mf4(pp, __riscv_vlse8_v_i8mf4(p0, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 2, __riscv_vlse8_v_i8mf4(p0 + 1, B_stride, vl), vl);
#else
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
#endif
                    p0 += 2;
                    p1 += 2;
                    pp += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = *p0++;
                    pp[1] = *p1++;
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

    return 0;
}

// K-major, row-interleaved MR-packn/MR2/MR1
static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const float* input_scale_ptr)
{
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int K = max_kk;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const float* A_data = (const float*)A + k;
    input_scale_ptr = input_scale_ptr ? input_scale_ptr + k : 0;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    const ptrdiff_t A_stride = (ptrdiff_t)A_hstep * sizeof(float);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = A_data + (size_t)(i + ii) * A_hstep;
        const float* p0g = p0;
        const float* psg = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            vfloat32m1_t _absmax = __riscv_vfmv_v_f_f32m1(0.f, vl);

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _v = __riscv_vlse32_v_f32m1(p0g + kk, A_stride, vl);
                if (psg)
                    _v = __riscv_vfmul_vf_f32m1(_v, psg[kk], vl);
                _absmax = __riscv_vfmax_vv_f32m1(_absmax, __riscv_vfabs_v_f32m1(_v, vl), vl);
            }

            vfloat32m1_t _scale = __riscv_vfrdiv_vf_f32m1(_absmax, 127.f, vl);
            _scale = __riscv_vfmerge_vfm_f32m1(_scale, 0.f, __riscv_vmfeq_vf_f32m1_b32(_absmax, 0.f, vl), vl);
            __riscv_vse32_v_f32m1(pd, __riscv_vfmul_vf_f32m1(_absmax, 1.f / 127.f, vl), vl);
            pd += packn;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _v = __riscv_vlse32_v_f32m1(p0g + kk, A_stride, vl);
                if (psg)
                    _v = __riscv_vfmul_vf_f32m1(_v, psg[kk], vl);
                vint32m1_t _v32 = __riscv_vfcvt_x_f_v_i32m1_rm(__riscv_vfmul_vv_f32m1(_v, _scale, vl), __RISCV_FRM_RMM, vl);
                _v32 = __riscv_vmax_vx_i32m1(_v32, -127, vl);
                _v32 = __riscv_vmin_vx_i32m1(_v32, 127, vl);
                vint16mf2_t _v16 = __riscv_vnclip_wx_i16mf2(_v32, 0, __RISCV_VXRM_RNU, vl);
                __riscv_vse8_v_i8mf4(pp, __riscv_vnclip_wx_i8mf4(_v16, 0, __RISCV_VXRM_RNU, vl), vl);
                pp += packn;
            }
            p0g += max_kk;
            if (psg)
                psg += max_kk;
        }
    }
#endif
    for (; ii + 1 < max_ii; ii += 2)
    {
        const int i0 = i + ii;
        const int i1 = i + ii + 1;
        const float* p0 = A_data + (size_t)i0 * A_hstep;
        const float* p1 = A_data + (size_t)i1 * A_hstep;
        const float* p0g = p0;
        const float* p1g = p1;
        const float* psg = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;

#if __riscv_vector
            int kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(p0g + kk, vl);
                vfloat32m8_t _v1 = __riscv_vle32_v_f32m8(p1g + kk, vl);
                if (psg)
                {
                    vfloat32m8_t _s = __riscv_vle32_v_f32m8(psg + kk, vl);
                    _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                }
                _v0 = __riscv_vfabs_v_f32m8(_v0, vl);
                _v1 = __riscv_vfabs_v_f32m8(_v1, vl);
                absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = p0g[kk];
                float v1 = p1g[kk];
                if (psg)
                {
                    v0 *= psg[kk];
                    v1 *= psg[kk];
                }
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
            }
#endif

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

#if __riscv_vector
            kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(p0g + kk, vl);
                vfloat32m8_t _v1 = __riscv_vle32_v_f32m8(p1g + kk, vl);
                if (psg)
                {
                    vfloat32m8_t _s = __riscv_vle32_v_f32m8(psg + kk, vl);
                    _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                }
                vint8m2_t _q0 = float2int8(__riscv_vfmul_vf_f32m8(_v0, scale0, vl), vl);
                vint8m2_t _q1 = float2int8(__riscv_vfmul_vf_f32m8(_v1, scale1, vl), vl);
                __riscv_vsse8_v_i8m2(pp, 2, _q0, vl);
                __riscv_vsse8_v_i8m2(pp + 1, 2, _q1, vl);
                pp += vl * 2;
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = p0g[kk];
                float v1 = p1g[kk];
                if (psg)
                {
                    v0 *= psg[kk];
                    v1 *= psg[kk];
                }
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
            }
#endif
            p0g += max_kk;
            p1g += max_kk;
            if (psg)
                psg += max_kk;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const int i0 = i + ii;
        const float* p0 = A_data + (size_t)i0 * A_hstep;
        const float* p0g = p0;
        const float* psg = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            float absmax = 0.f;

#if __riscv_vector
            int kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0g + kk, vl);
                if (psg)
                    _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(psg + kk, vl), vl);
                _v = __riscv_vfabs_v_f32m8(_v, vl);
                absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v = p0g[kk];
                if (psg)
                    v *= psg[kk];
                absmax = std::max(absmax, fabsf(v));
            }
#endif

            const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
            *pd++ = absmax / 127.f;

#if __riscv_vector
            kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0g + kk, vl);
                if (psg)
                    _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(psg + kk, vl), vl);
                __riscv_vse8_v_i8m2(pp, float2int8(__riscv_vfmul_vf_f32m8(_v, scale, vl), vl), vl);
                pp += vl;
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v = p0g[kk];
                if (psg)
                {
                    v *= psg[kk];
                }
                *pp++ = float2int8(v * scale);
            }
#endif
            p0g += max_kk;
            if (psg)
                psg += max_kk;
        }
    }
}

// K-major, row-interleaved MR-packn/MR2/MR1
static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const float* input_scale_ptr)
{
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int K = max_kk;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const float* A_data = (const float*)A + (size_t)k * A_hstep;
    input_scale_ptr = input_scale_ptr ? input_scale_ptr + k : 0;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const int i0 = i + ii;
        const float* p0g = A_data + i0;
        const float* psg = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            vfloat32m1_t _absmax = __riscv_vfmv_v_f_f32m1(0.f, vl);
            const float* pAk = p0g;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _v = __riscv_vle32_v_f32m1(pAk, vl);
                if (psg)
                    _v = __riscv_vfmul_vf_f32m1(_v, psg[kk], vl);
                _absmax = __riscv_vfmax_vv_f32m1(_absmax, __riscv_vfabs_v_f32m1(_v, vl), vl);
                pAk += A_hstep;
            }

            vfloat32m1_t _scale = __riscv_vfrdiv_vf_f32m1(_absmax, 127.f, vl);
            _scale = __riscv_vfmerge_vfm_f32m1(_scale, 0.f, __riscv_vmfeq_vf_f32m1_b32(_absmax, 0.f, vl), vl);
            __riscv_vse32_v_f32m1(pd, __riscv_vfmul_vf_f32m1(_absmax, 1.f / 127.f, vl), vl);
            pd += packn;
            pAk = p0g;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _v = __riscv_vle32_v_f32m1(pAk, vl);
                if (psg)
                    _v = __riscv_vfmul_vf_f32m1(_v, psg[kk], vl);
                vint32m1_t _v32 = __riscv_vfcvt_x_f_v_i32m1_rm(__riscv_vfmul_vv_f32m1(_v, _scale, vl), __RISCV_FRM_RMM, vl);
                _v32 = __riscv_vmax_vx_i32m1(_v32, -127, vl);
                _v32 = __riscv_vmin_vx_i32m1(_v32, 127, vl);
                vint16mf2_t _v16 = __riscv_vnclip_wx_i16mf2(_v32, 0, __RISCV_VXRM_RNU, vl);
                __riscv_vse8_v_i8mf4(pp, __riscv_vnclip_wx_i8mf4(_v16, 0, __RISCV_VXRM_RNU, vl), vl);
                pp += packn;
                pAk += A_hstep;
            }
            p0g += (size_t)max_kk * A_hstep;
            if (psg)
                psg += max_kk;
        }
    }
#endif
    for (; ii + 1 < max_ii; ii += 2)
    {
        const int i0 = i + ii;
        const float* p0g = A_data + i0;
        const float* psg = input_scale_ptr;
        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;

#if __riscv_vector
            int kk = 0;
            const float* pAk = p0g;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v0 = __riscv_vlse32_v_f32m8(pAk, (ptrdiff_t)A_hstep * sizeof(float), vl);
                vfloat32m8_t _v1 = __riscv_vlse32_v_f32m8(pAk + 1, (ptrdiff_t)A_hstep * sizeof(float), vl);
                if (psg)
                {
                    vfloat32m8_t _s = __riscv_vle32_v_f32m8(psg + kk, vl);
                    _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                }
                _v0 = __riscv_vfabs_v_f32m8(_v0, vl);
                _v1 = __riscv_vfabs_v_f32m8(_v1, vl);
                absmax0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v0, __riscv_vfmv_s_f_f32m1(absmax0, 1), vl));
                absmax1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v1, __riscv_vfmv_s_f_f32m1(absmax1, 1), vl));
                pAk += vl * A_hstep;
                kk += vl;
            }
#else
            const float* pAk = p0g;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = pAk[0];
                float v1 = pAk[1];
                if (psg)
                {
                    const float s = psg[kk];
                    v0 *= s;
                    v1 *= s;
                }
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
                pAk += A_hstep;
            }
#endif

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

#if __riscv_vector
            kk = 0;
            pAk = p0g;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v0 = __riscv_vlse32_v_f32m8(pAk, (ptrdiff_t)A_hstep * sizeof(float), vl);
                vfloat32m8_t _v1 = __riscv_vlse32_v_f32m8(pAk + 1, (ptrdiff_t)A_hstep * sizeof(float), vl);
                if (psg)
                {
                    vfloat32m8_t _s = __riscv_vle32_v_f32m8(psg + kk, vl);
                    _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                }
                vint8m2_t _q0 = float2int8(__riscv_vfmul_vf_f32m8(_v0, scale0, vl), vl);
                vint8m2_t _q1 = float2int8(__riscv_vfmul_vf_f32m8(_v1, scale1, vl), vl);
                __riscv_vsse8_v_i8m2(pp, 2, _q0, vl);
                __riscv_vsse8_v_i8m2(pp + 1, 2, _q1, vl);
                pp += vl * 2;
                pAk += vl * A_hstep;
                kk += vl;
            }
#else
            pAk = p0g;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = pAk[0];
                float v1 = pAk[1];
                if (psg)
                {
                    const float s = psg[kk];
                    v0 *= s;
                    v1 *= s;
                }
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
                pAk += A_hstep;
            }
#endif
            p0g += (size_t)max_kk * A_hstep;
            if (psg)
                psg += max_kk;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const int i0 = i + ii;
        const float* p0g = A_data + i0;
        const float* psg = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            float absmax = 0.f;

#if __riscv_vector
            int kk = 0;
            const float* pAk = p0g;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v = __riscv_vlse32_v_f32m8(pAk, (ptrdiff_t)A_hstep * sizeof(float), vl);
                if (psg)
                    _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(psg + kk, vl), vl);
                _v = __riscv_vfabs_v_f32m8(_v, vl);
                absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                pAk += vl * A_hstep;
                kk += vl;
            }
#else
            const float* pAk = p0g;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v = *pAk;
                if (psg)
                    v *= psg[kk];
                absmax = std::max(absmax, fabsf(v));
                pAk += A_hstep;
            }
#endif

            const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
            *pd++ = absmax / 127.f;

#if __riscv_vector
            kk = 0;
            pAk = p0g;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v = __riscv_vlse32_v_f32m8(pAk, (ptrdiff_t)A_hstep * sizeof(float), vl);
                if (psg)
                    _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(psg + kk, vl), vl);
                __riscv_vse8_v_i8m2(pp, float2int8(__riscv_vfmul_vf_f32m8(_v, scale, vl), vl), vl);
                pp += vl;
                pAk += vl * A_hstep;
                kk += vl;
            }
#else
            pAk = p0g;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v = *pAk;
                if (psg)
                {
                    v *= psg[kk];
                }
                *pp++ = float2int8(v * scale);
                pAk += A_hstep;
            }
#endif
            p0g += (size_t)max_kk * A_hstep;
            if (psg)
                psg += max_kk;
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k0, int max_kk0, int B_hstep, int block_size)
{
    const signed char* pAT = AT_tile;
    const float* pAT_descales = AT_descales_tile;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int K = max_kk0;
    const int block_count = (max_kk0 + block_size - 1) / block_size;
    const int num_blocks = (B_hstep + block_size - 1) / block_size;
    const int block_start = k0 / block_size;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl4 = __riscv_vsetvl_e8mf4(4);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;
        const int mr = packn;
        const size_t vl = __riscv_vsetvl_e32m1(mr);

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum5 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum6 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum7 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            const signed char* pB0 = pB + (size_t)4 * k0;
            const signed char* pB1 = pB + (size_t)4 * B_hstep + (size_t)4 * k0;
            const float* pB_descales0 = pB_descales + (size_t)4 * block_start;
            const float* pB_descales1 = pB_descales + (size_t)4 * num_blocks + (size_t)4 * block_start;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum2 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum3 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum4 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum5 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum6 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum7 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    uint32_t b0 = *(const uint32_t*)pB0;
                    uint32_t b1 = *(const uint32_t*)pB1;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vwmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vwmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vwmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vwmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr, vl), vl);
                    b0 = *(const uint32_t*)(pB0 + 4);
                    b1 = *(const uint32_t*)(pB1 + 4);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vwmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vwmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vwmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vwmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr * 2, vl), vl);
                    b0 = *(const uint32_t*)(pB0 + 8);
                    b1 = *(const uint32_t*)(pB1 + 8);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vwmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vwmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vwmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vwmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr * 3, vl), vl);
                    b0 = *(const uint32_t*)(pB0 + 12);
                    b1 = *(const uint32_t*)(pB1 + 12);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vwmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vwmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vwmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vwmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    pA += mr * 4;
                    pB0 += 16;
                    pB1 += 16;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    uint32_t b0 = *(const uint32_t*)pB0;
                    uint32_t b1 = *(const uint32_t*)pB1;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vwmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vwmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vwmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vwmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr, vl), vl);
                    b0 = *(const uint32_t*)(pB0 + 4);
                    b1 = *(const uint32_t*)(pB1 + 4);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vwmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vwmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vwmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vwmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    pA += mr * 2;
                    pB0 += 8;
                    pB1 += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    const uint32_t b0 = *(const uint32_t*)pB0;
                    const uint32_t b1 = *(const uint32_t*)pB1;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vwmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vwmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vwmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vwmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    pA += mr;
                    pB0 += 4;
                    pB1 += 4;
                }

                vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), _ad, vl);
                _fsum0 = __riscv_vfmacc_vf_f32m1(_fsum0, pB_descales0[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), _ad, vl);
                _fsum1 = __riscv_vfmacc_vf_f32m1(_fsum1, pB_descales0[1], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum2, vl), _ad, vl);
                _fsum2 = __riscv_vfmacc_vf_f32m1(_fsum2, pB_descales0[2], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum3, vl), _ad, vl);
                _fsum3 = __riscv_vfmacc_vf_f32m1(_fsum3, pB_descales0[3], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum4, vl), _ad, vl);
                _fsum4 = __riscv_vfmacc_vf_f32m1(_fsum4, pB_descales1[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum5, vl), _ad, vl);
                _fsum5 = __riscv_vfmacc_vf_f32m1(_fsum5, pB_descales1[1], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum6, vl), _ad, vl);
                _fsum6 = __riscv_vfmacc_vf_f32m1(_fsum6, pB_descales1[2], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum7, vl), _ad, vl);
                _fsum7 = __riscv_vfmacc_vf_f32m1(_fsum7, pB_descales1[3], _v, vl);
                pA_descales += mr;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }

            if (k0 != 0)
            {
                _fsum0 = __riscv_vfadd_vv_f32m1(_fsum0, __riscv_vle32_v_f32m1(outptr, vl), vl);
                _fsum1 = __riscv_vfadd_vv_f32m1(_fsum1, __riscv_vle32_v_f32m1(outptr + mr, vl), vl);
                _fsum2 = __riscv_vfadd_vv_f32m1(_fsum2, __riscv_vle32_v_f32m1(outptr + mr * 2, vl), vl);
                _fsum3 = __riscv_vfadd_vv_f32m1(_fsum3, __riscv_vle32_v_f32m1(outptr + mr * 3, vl), vl);
                _fsum4 = __riscv_vfadd_vv_f32m1(_fsum4, __riscv_vle32_v_f32m1(outptr + mr * 4, vl), vl);
                _fsum5 = __riscv_vfadd_vv_f32m1(_fsum5, __riscv_vle32_v_f32m1(outptr + mr * 5, vl), vl);
                _fsum6 = __riscv_vfadd_vv_f32m1(_fsum6, __riscv_vle32_v_f32m1(outptr + mr * 6, vl), vl);
                _fsum7 = __riscv_vfadd_vv_f32m1(_fsum7, __riscv_vle32_v_f32m1(outptr + mr * 7, vl), vl);
            }
            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + mr, _fsum1, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 2, _fsum2, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 3, _fsum3, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 4, _fsum4, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 5, _fsum5, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 6, _fsum6, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 7, _fsum7, vl);
            outptr += mr * 8;
            pB = pB1 + (size_t)4 * (B_hstep - k0 - max_kk0);
            pB_descales = pB_descales1 + (size_t)4 * (num_blocks - block_start - block_count);
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            vfloat32m1_t _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum2 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum3 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    uint32_t b = *(const uint32_t*)pB;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr, vl), vl);
                    b = *(const uint32_t*)(pB + 4);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr * 2, vl), vl);
                    b = *(const uint32_t*)(pB + 8);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr * 3, vl), vl);
                    b = *(const uint32_t*)(pB + 12);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += mr * 4;
                    pB += 16;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    uint32_t b = *(const uint32_t*)pB;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr, vl), vl);
                    b = *(const uint32_t*)(pB + 4);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += mr * 2;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    const uint32_t b = *(const uint32_t*)pB;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += mr;
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
                pA_descales += mr;
                pB_descales += 4;
            }

            if (k0 != 0)
            {
                _fsum0 = __riscv_vfadd_vv_f32m1(_fsum0, __riscv_vle32_v_f32m1(outptr, vl), vl);
                _fsum1 = __riscv_vfadd_vv_f32m1(_fsum1, __riscv_vle32_v_f32m1(outptr + mr, vl), vl);
                _fsum2 = __riscv_vfadd_vv_f32m1(_fsum2, __riscv_vle32_v_f32m1(outptr + mr * 2, vl), vl);
                _fsum3 = __riscv_vfadd_vv_f32m1(_fsum3, __riscv_vle32_v_f32m1(outptr + mr * 3, vl), vl);
            }
            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + mr, _fsum1, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 2, _fsum2, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 3, _fsum3, vl);
            outptr += mr * 4;
            pB += (size_t)4 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)4 * (num_blocks - block_start - block_count);
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            vfloat32m1_t _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    uint16_t b = *(const uint16_t*)pB;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr, vl), vl);
                    b = *(const uint16_t*)(pB + 2);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr * 2, vl), vl);
                    b = *(const uint16_t*)(pB + 4);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr * 3, vl), vl);
                    b = *(const uint16_t*)(pB + 6);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += mr * 4;
                    pB += 8;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    uint16_t b = *(const uint16_t*)pB;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr, vl), vl);
                    b = *(const uint16_t*)(pB + 2);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += mr * 2;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    const uint16_t b = *(const uint16_t*)pB;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += mr;
                    pB += 2;
                }

                vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), _ad, vl);
                _fsum0 = __riscv_vfmacc_vf_f32m1(_fsum0, pB_descales[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), _ad, vl);
                _fsum1 = __riscv_vfmacc_vf_f32m1(_fsum1, pB_descales[1], _v, vl);
                pA_descales += mr;
                pB_descales += 2;
            }

            if (k0 != 0)
            {
                _fsum0 = __riscv_vfadd_vv_f32m1(_fsum0, __riscv_vle32_v_f32m1(outptr, vl), vl);
                _fsum1 = __riscv_vfadd_vv_f32m1(_fsum1, __riscv_vle32_v_f32m1(outptr + mr, vl), vl);
            }
            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + mr, _fsum1, vl);
            outptr += mr * 2;
            pB += (size_t)2 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)2 * (num_blocks - block_start - block_count);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            vfloat32m1_t _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint8mf4_t _b = __riscv_vle8_v_i8mf4(pB, vl4);
                    const signed char b0 = __riscv_vmv_x_s_i8mf4_i8(_b);
                    _b = __riscv_vslidedown_vx_i8mf4(_b, 1, vl4);
                    const signed char b1 = __riscv_vmv_x_s_i8mf4_i8(_b);
                    _b = __riscv_vslidedown_vx_i8mf4(_b, 1, vl4);
                    const signed char b2 = __riscv_vmv_x_s_i8mf4_i8(_b);
                    _b = __riscv_vslidedown_vx_i8mf4(_b, 1, vl4);
                    const signed char b3 = __riscv_vmv_x_s_i8mf4_i8(_b);
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, b0, _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, b1, _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr * 2, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, b2, _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr * 3, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, b3, _a, vl);
                    pA += mr * 4;
                    pB += 4;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const uint16_t b = *(const uint16_t*)pB;
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, (signed char)b, _a, vl);
                    _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA + mr, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, (signed char)(b >> 8), _a, vl);
                    pA += mr * 2;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pB[0], _a, vl);
                    pA += mr;
                    pB++;
                }

                vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), _ad, vl);
                _fsum = __riscv_vfmacc_vf_f32m1(_fsum, pB_descales[0], _v, vl);
                pA_descales += mr;
                pB_descales++;
            }

            if (k0 != 0)
                _fsum = __riscv_vfadd_vv_f32m1(_fsum, __riscv_vle32_v_f32m1(outptr, vl), vl);
            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += mr;
            pB += B_hstep - k0 - max_kk0;
            pB_descales += num_blocks - block_start - block_count;
        }

        pAT += K * mr;
        pAT_descales += block_count * mr;
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 4, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 8, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[4], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[5], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 12, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[6], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[7], _b0, vl);
                    pA += 8;
                    pB += 16;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 4, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _b = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b, vl);
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

            if (k0 != 0)
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vfadd_vv_f32m1(_fsum0, __riscv_vget_v_f32m1x2_f32m1(_s, 0), vl);
                _fsum1 = __riscv_vfadd_vv_f32m1(_fsum1, __riscv_vget_v_f32m1x2_f32m1(_s, 1), vl);
            }
            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
            outptr += 8;
            pB += (size_t)4 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)4 * (num_blocks - block_start - block_count);
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(2);
            vfloat32m1_t _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 2, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 4, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[4], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[5], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 6, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[6], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[7], _b0, vl);
                    pA += 8;
                    pB += 8;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 2, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _b = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b, vl);
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

            if (k0 != 0)
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vfadd_vv_f32m1(_fsum0, __riscv_vget_v_f32m1x2_f32m1(_s, 0), vl);
                _fsum1 = __riscv_vfadd_vv_f32m1(_fsum1, __riscv_vget_v_f32m1x2_f32m1(_s, 1), vl);
            }
            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
            outptr += 4;
            pB += (size_t)2 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)2 * (num_blocks - block_start - block_count);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            const size_t vl = __riscv_vsetvl_e32m1(1);
            vfloat32m1_t _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 1, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 2, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[4], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[5], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 3, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[6], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[7], _b0, vl);
                    pA += 8;
                    pB += 4;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 1, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _b = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, pA[1], _b, vl);
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

            if (k0 != 0)
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vfadd_vv_f32m1(_fsum0, __riscv_vget_v_f32m1x2_f32m1(_s, 0), vl);
                _fsum1 = __riscv_vfadd_vv_f32m1(_fsum1, __riscv_vget_v_f32m1x2_f32m1(_s, 1), vl);
            }
            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
            outptr += 2;
            pB += B_hstep - k0 - max_kk0;
            pB_descales += num_blocks - block_start - block_count;
        }

        pAT += K * 2;
        pAT_descales += block_count * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 4, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 8, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[2], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 12, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[3], _b0, vl);
                    pA += 4;
                    pB += 16;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 4, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    pA += 2;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _b = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    pA++;
                    pB += 4;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), pA_descales[0], vl);
                _fsum = __riscv_vfmacc_vv_f32m1(_fsum, _bd, _v, vl);
                pA_descales++;
                pB_descales += 4;
            }

            if (k0 != 0)
                _fsum = __riscv_vfadd_vv_f32m1(_fsum, __riscv_vle32_v_f32m1(outptr, vl), vl);
            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += 4;
            pB += (size_t)4 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)4 * (num_blocks - block_start - block_count);
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(2);
            vfloat32m1_t _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 2, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 4, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[2], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 6, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[3], _b0, vl);
                    pA += 4;
                    pB += 8;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 2, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _b = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    pA++;
                    pB += 2;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), pA_descales[0], vl);
                _fsum = __riscv_vfmacc_vv_f32m1(_fsum, _bd, _v, vl);
                pA_descales++;
                pB_descales += 2;
            }

            if (k0 != 0)
                _fsum = __riscv_vfadd_vv_f32m1(_fsum, __riscv_vle32_v_f32m1(outptr, vl), vl);
            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += 2;
            pB += (size_t)2 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)2 * (num_blocks - block_start - block_count);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            const size_t vl = __riscv_vsetvl_e32m1(1);
            vfloat32m1_t _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 1, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 2, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[2], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 3, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[3], _b0, vl);
                    pA += 4;
                    pB += 4;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    vint16mf2_t _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b0 = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB + 1, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    pA += 2;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    vint16mf2_t _b = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pB, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    pA++;
                    pB++;
                }

                vfloat32m1_t _bd = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), pA_descales[0], vl);
                _fsum = __riscv_vfmacc_vv_f32m1(_fsum, _bd, _v, vl);
                pA_descales++;
                pB_descales++;
            }

            if (k0 != 0)
                _fsum = __riscv_vfadd_vv_f32m1(_fsum, __riscv_vle32_v_f32m1(outptr, vl), vl);
            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr++;
            pB += B_hstep - k0 - max_kk0;
            pB_descales += num_blocks - block_start - block_count;
        }

        pAT += K;
        pAT_descales += block_count;
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;
        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            float sum00 = 0.f;
            float sum01 = 0.f;
            float sum02 = 0.f;
            float sum03 = 0.f;
            float sum10 = 0.f;
            float sum11 = 0.f;
            float sum12 = 0.f;
            float sum13 = 0.f;

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int s00 = 0, s01 = 0, s02 = 0, s03 = 0;
                int s10 = 0, s11 = 0, s12 = 0, s13 = 0;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
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
                for (; kk + 1 < max_kk; kk += 2)
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
                for (; kk < max_kk; kk++)
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

            if (k0 != 0)
            {
                sum00 += outptr[0];
                sum10 += outptr[1];
                sum01 += outptr[2];
                sum11 += outptr[3];
                sum02 += outptr[4];
                sum12 += outptr[5];
                sum03 += outptr[6];
                sum13 += outptr[7];
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
            pB += (size_t)4 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)4 * (num_blocks - block_start - block_count);
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            float sum00 = 0.f, sum01 = 0.f, sum10 = 0.f, sum11 = 0.f;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int s00 = 0, s01 = 0, s10 = 0, s11 = 0;
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
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
                for (; kk + 1 < max_kk; kk += 2)
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
                for (; kk < max_kk; kk++)
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
            if (k0 != 0)
            {
                sum00 += outptr[0];
                sum10 += outptr[1];
                sum01 += outptr[2];
                sum11 += outptr[3];
            }
            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;
            outptr += 4;
            pB += (size_t)2 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)2 * (num_blocks - block_start - block_count);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            float sum0 = 0.f, sum1 = 0.f;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int s0 = 0, s1 = 0;
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const signed char* b = pB;
                    s0 += pA[0] * b[0] + pA[2] * b[1] + pA[4] * b[2] + pA[6] * b[3];
                    s1 += pA[1] * b[0] + pA[3] * b[1] + pA[5] * b[2] + pA[7] * b[3];
                    pA += 8;
                    pB += 4;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const signed char* b = pB;
                    s0 += pA[0] * b[0] + pA[2] * b[1];
                    s1 += pA[1] * b[0] + pA[3] * b[1];
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
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
            if (k0 != 0)
            {
                sum0 += outptr[0];
                sum1 += outptr[1];
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
            pB += B_hstep - k0 - max_kk0;
            pB_descales += num_blocks - block_start - block_count;
        }
        pAT += K * 2;
        pAT_descales += block_count * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;
        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
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
                for (; kk + 1 < max_kk; kk += 2)
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
                for (; kk < max_kk; kk++)
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
            if (k0 != 0)
            {
                sum0 += outptr[0];
                sum1 += outptr[1];
                sum2 += outptr[2];
                sum3 += outptr[3];
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;
            outptr += 4;
            pB += (size_t)4 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)4 * (num_blocks - block_start - block_count);
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            float sum0 = 0.f, sum1 = 0.f;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int s0 = 0, s1 = 0;
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 4;
                    s0 += pA[0] * b0[0] + pA[1] * b0[1] + pA[2] * b0[2] + pA[3] * b0[3];
                    s1 += pA[0] * b1[0] + pA[1] * b1[1] + pA[2] * b1[2] + pA[3] * b1[3];
                    pA += 4;
                    pB += 8;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const signed char* b0 = pB;
                    const signed char* b1 = b0 + 2;
                    s0 += pA[0] * b0[0] + pA[1] * b0[1];
                    s1 += pA[0] * b1[0] + pA[1] * b1[1];
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
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
            if (k0 != 0)
            {
                sum0 += outptr[0];
                sum1 += outptr[1];
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
            pB += (size_t)2 * (B_hstep - k0 - max_kk0);
            pB_descales += (size_t)2 * (num_blocks - block_start - block_count);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            float sum = 0.f;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int s = 0;
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const signed char* b = pB;
                    s += pA[0] * b[0] + pA[1] * b[1] + pA[2] * b[2] + pA[3] * b[3];
                    pA += 4;
                    pB += 4;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const signed char* b = pB;
                    s += pA[0] * b[0] + pA[1] * b[1];
                    pA += 2;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    s += pA[0] * pB[0];
                    pA++;
                    pB++;
                }
                sum += s * pA_descales[0] * pB_descales[0];
                pA_descales++;
                pB_descales++;
            }
            if (k0 != 0)
                sum += outptr[0];
            outptr[0] = sum;
            outptr++;
            pB += B_hstep - k0 - max_kk0;
            pB_descales += num_blocks - block_start - block_count;
        }
        pAT += K;
        pAT_descales += block_count;
    }
#endif
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int N, float alpha, float beta)
{
    const float* pp = topT;
    beta *= alpha;
    (void)N;
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
                pC += i + ii;
            if (broadcast_type_C == 3)
                pC += (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC += j;
        }

        float c0 = 0.f;
        vfloat32m1_t _c = __riscv_vfmv_v_f_f32m1(0.f, vl_packn);
        if (pC && broadcast_type_C == 0)
            c0 = pC[0] * beta;
        if (pC && (broadcast_type_C == 1 || broadcast_type_C == 2))
            _c = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(pC, vl_packn), beta, vl_packn);

        float* out0 = outptr;
        for (int jj = 0; jj < max_jj; jj++)
        {
            vfloat32m1_t _sum = __riscv_vle32_v_f32m1(pp, vl_packn);
            if (alpha != 1.f)
                _sum = __riscv_vfmul_vf_f32m1(_sum, alpha, vl_packn);
            if (pC)
            {
                if (broadcast_type_C == 0)
                    _sum = __riscv_vfadd_vf_f32m1(_sum, c0, vl_packn);
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _sum = __riscv_vfadd_vv_f32m1(_sum, _c, vl_packn);
                if (broadcast_type_C == 3)
                {
                    vfloat32m1_t _c0 = __riscv_vlse32_v_f32m1(pC, c_stride, vl_packn);
                    pC++;
                    if (beta == 1.f)
                        _sum = __riscv_vfadd_vv_f32m1(_sum, _c0, vl_packn);
                    else
                        _sum = __riscv_vfmacc_vf_f32m1(_sum, beta, _c0, vl_packn);
                }
                if (broadcast_type_C == 4)
                {
                    _sum = __riscv_vfadd_vf_f32m1(_sum, *pC * beta, vl_packn);
                    pC++;
                }
            }
            __riscv_vsse32_v_f32m1(out0, out_stride, _sum, vl_packn);
            out0++;
            pp += packn;
        }
        outptr += out_hstep * packn;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* out0 = outptr;
        float* out1 = out0 + out_hstep;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC += i + ii;
            if (broadcast_type_C == 3)
                pC += (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC += j;
        }

        float c0 = 0.f;
        float c1 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            c0 = pC[0] * beta;
            c1 = pC[broadcast_type_C == 0 ? 0 : 1] * beta;
        }
        int jj = 0;
#if __riscv_vector
        const size_t vl = __riscv_vsetvl_e32m4(max_jj);
        vfloat32m4x2_t _s = __riscv_vlseg2e32_v_f32m4x2(pp, vl);
        vfloat32m4_t _sum0 = __riscv_vget_v_f32m4x2_f32m4(_s, 0);
        vfloat32m4_t _sum1 = __riscv_vget_v_f32m4x2_f32m4(_s, 1);
        if (alpha != 1.f)
        {
            _sum0 = __riscv_vfmul_vf_f32m4(_sum0, alpha, vl);
            _sum1 = __riscv_vfmul_vf_f32m4(_sum1, alpha, vl);
        }

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
        }

        __riscv_vse32_v_f32m4(out0, _sum0, vl);
        __riscv_vse32_v_f32m4(out1, _sum1, vl);
        jj += (int)vl;
        pp += vl * 2;
        out0 += vl;
        out1 += vl;
        if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
            pC += vl;
#endif // __riscv_vector
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum00 = pp[0] * alpha;
            float sum10 = pp[1] * alpha;
            float sum01 = pp[2] * alpha;
            float sum11 = pp[3] * alpha;
            float sum02 = pp[4] * alpha;
            float sum12 = pp[5] * alpha;
            float sum03 = pp[6] * alpha;
            float sum13 = pp[7] * alpha;

            if (pC)
            {
                if (broadcast_type_C == 0)
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
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                    pC += 4;
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
                    pC += 4;
                }
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
            pp += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = pp[0] * alpha;
            float sum10 = pp[1] * alpha;
            float sum01 = pp[2] * alpha;
            float sum11 = pp[3] * alpha;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    sum00 += c0;
                    sum01 += c0;
                    sum10 += c1;
                    sum11 += c1;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    sum00 += pC[0] * beta;
                    sum01 += pC[1] * beta;
                    sum10 += pC[0] * beta;
                    sum11 += pC[1] * beta;
                    pC += 2;
                }
            }

            out0[0] = sum00;
            out0[1] = sum01;
            out1[0] = sum10;
            out1[1] = sum11;
            out0 += 2;
            out1 += 2;
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
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
                    sum0 += pC[0] * beta;
                    sum1 += pC[c_hstep] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[0] * beta;
                    pC++;
                }
            }
            out0[0] = sum0;
            out1[0] = sum1;
            out0++;
            out1++;
            pp += 2;
        }
        outptr += out_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        float* out0 = outptr;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC += i + ii;
            if (broadcast_type_C == 3)
                pC += (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC += j;
        }

        float c0 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
            c0 = pC[0] * beta;
        int jj = 0;
#if __riscv_vector
        const size_t vl = __riscv_vsetvl_e32m4(max_jj);
        vfloat32m4_t _sum = __riscv_vle32_v_f32m4(pp, vl);
        if (alpha != 1.f)
            _sum = __riscv_vfmul_vf_f32m4(_sum, alpha, vl);

        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                _sum = __riscv_vfadd_vf_f32m4(_sum, c0, vl);
            if (broadcast_type_C == 3)
            {
                vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                if (beta == 1.f)
                    _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                else
                    _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
            }
            if (broadcast_type_C == 4)
            {
                vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                if (beta == 1.f)
                    _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                else
                    _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
            }
        }

        __riscv_vse32_v_f32m4(out0, _sum, vl);
        jj += (int)vl;
        pp += vl;
        out0 += vl;
        if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
            pC += vl;
#endif // __riscv_vector
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
            float sum2 = pp[2] * alpha;
            float sum3 = pp[3] * alpha;
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
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                    sum2 += pC[2] * beta;
                    sum3 += pC[3] * beta;
                    pC += 4;
                }
            }
            out0[0] = sum0;
            out0[1] = sum1;
            out0[2] = sum2;
            out0[3] = sum3;
            out0 += 4;
            pp += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
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
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[1] * beta;
                    pC += 2;
                }
            }
            out0[0] = sum0;
            out0[1] = sum1;
            out0 += 2;
            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum = *pp * alpha;
            if (pC)
            {
                if (broadcast_type_C == 0)
                    sum += c0;
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    sum += c0;
                if (broadcast_type_C == 3)
                {
                    sum += pC[0] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    sum += pC[0] * beta;
                    pC++;
                }
            }
            out0[0] = sum;
            out0++;
            pp++;
        }
        outptr += out_hstep;
    }
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int N, float alpha, float beta)
{
    const float* pp = topT;
    beta *= alpha;
    (void)N;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    float* outptr = (float*)top_blob + (size_t)j * out_hstep + i;

    int ii = 0;
#if __riscv_vector
    const ptrdiff_t out_stride = (ptrdiff_t)out_hstep * sizeof(float);
    const int packn = csrr_vlenb() / 4;
    const size_t vl_packn = __riscv_vsetvl_e32m1(packn);
    const ptrdiff_t c_stride = (ptrdiff_t)c_hstep * sizeof(float);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC += i + ii;
            if (broadcast_type_C == 3)
                pC += (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC += j;
        }

        float c0 = 0.f;
        vfloat32m1_t _c = __riscv_vfmv_v_f_f32m1(0.f, vl_packn);
        if (pC && broadcast_type_C == 0)
            c0 = pC[0] * beta;
        if (pC && (broadcast_type_C == 1 || broadcast_type_C == 2))
            _c = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(pC, vl_packn), beta, vl_packn);

        float* out0 = outptr;
        for (int jj = 0; jj < max_jj; jj++)
        {
            vfloat32m1_t _sum = __riscv_vle32_v_f32m1(pp, vl_packn);
            if (alpha != 1.f)
                _sum = __riscv_vfmul_vf_f32m1(_sum, alpha, vl_packn);
            if (pC)
            {
                if (broadcast_type_C == 0)
                    _sum = __riscv_vfadd_vf_f32m1(_sum, c0, vl_packn);
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _sum = __riscv_vfadd_vv_f32m1(_sum, _c, vl_packn);
                if (broadcast_type_C == 3)
                {
                    vfloat32m1_t _c0 = __riscv_vlse32_v_f32m1(pC, c_stride, vl_packn);
                    pC++;
                    if (beta == 1.f)
                        _sum = __riscv_vfadd_vv_f32m1(_sum, _c0, vl_packn);
                    else
                        _sum = __riscv_vfmacc_vf_f32m1(_sum, beta, _c0, vl_packn);
                }
                if (broadcast_type_C == 4)
                {
                    _sum = __riscv_vfadd_vf_f32m1(_sum, *pC * beta, vl_packn);
                    pC++;
                }
            }
            __riscv_vse32_v_f32m1(out0, _sum, vl_packn);
            out0 += out_hstep;
            pp += packn;
        }
        outptr += packn;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* out0 = outptr;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC += i + ii;
            if (broadcast_type_C == 3)
                pC += (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC += j;
        }

        float c0 = 0.f;
        float c1 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            c0 = pC[0] * beta;
            c1 = pC[broadcast_type_C == 0 ? 0 : 1] * beta;
        }
        int jj = 0;
#if __riscv_vector
        const size_t vl = __riscv_vsetvl_e32m4(max_jj);
        vfloat32m4x2_t _s = __riscv_vlseg2e32_v_f32m4x2(pp, vl);
        vfloat32m4_t _sum0 = __riscv_vget_v_f32m4x2_f32m4(_s, 0);
        vfloat32m4_t _sum1 = __riscv_vget_v_f32m4x2_f32m4(_s, 1);
        if (alpha != 1.f)
        {
            _sum0 = __riscv_vfmul_vf_f32m4(_sum0, alpha, vl);
            _sum1 = __riscv_vfmul_vf_f32m4(_sum1, alpha, vl);
        }
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
        }
        vfloat32m4x2_t _sum = __riscv_vcreate_v_f32m4x2(_sum0, _sum1);
        if (out_hstep == 2)
            __riscv_vsseg2e32_v_f32m4x2(out0, _sum, vl);
        else
            __riscv_vssseg2e32_v_f32m4x2(out0, out_stride, _sum, vl);
        jj += (int)vl;
        pp += vl * 2;
        out0 += out_hstep * vl;
        if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
            pC += vl;
#endif // __riscv_vector
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum00 = pp[0] * alpha;
            float sum10 = pp[1] * alpha;
            float sum01 = pp[2] * alpha;
            float sum11 = pp[3] * alpha;
            float sum02 = pp[4] * alpha;
            float sum12 = pp[5] * alpha;
            float sum03 = pp[6] * alpha;
            float sum13 = pp[7] * alpha;
            if (pC)
            {
                if (broadcast_type_C == 0)
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
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
            out0[0] = sum00;
            out0[1] = sum10;
            out0[out_hstep] = sum01;
            out0[out_hstep + 1] = sum11;
            out0[out_hstep * 2] = sum02;
            out0[out_hstep * 2 + 1] = sum12;
            out0[out_hstep * 3] = sum03;
            out0[out_hstep * 3 + 1] = sum13;
            out0 += out_hstep * 4;
            pp += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = pp[0] * alpha;
            float sum10 = pp[1] * alpha;
            float sum01 = pp[2] * alpha;
            float sum11 = pp[3] * alpha;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    sum00 += c0;
                    sum01 += c0;
                    sum10 += c1;
                    sum11 += c1;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
            out0[0] = sum00;
            out0[1] = sum10;
            out0[out_hstep] = sum01;
            out0[out_hstep + 1] = sum11;
            out0 += out_hstep * 2;
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
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
            out0[0] = sum0;
            out0[1] = sum1;
            out0 += out_hstep;
            pp += 2;
        }
        outptr += 2;
    }
    for (; ii < max_ii; ii++)
    {
        float* out0 = outptr;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC += i + ii;
            if (broadcast_type_C == 3)
                pC += (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC += j;
        }

        float c0 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
            c0 = pC[0] * beta;
        int jj = 0;
#if __riscv_vector
        const size_t vl = __riscv_vsetvl_e32m4(max_jj);
        vfloat32m4_t _sum = __riscv_vle32_v_f32m4(pp, vl);
        if (alpha != 1.f)
            _sum = __riscv_vfmul_vf_f32m4(_sum, alpha, vl);
        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                _sum = __riscv_vfadd_vf_f32m4(_sum, c0, vl);
            if (broadcast_type_C == 3)
            {
                vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                if (beta == 1.f)
                    _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                else
                    _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
            }
            if (broadcast_type_C == 4)
            {
                vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                if (beta == 1.f)
                    _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                else
                    _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
            }
        }
        if (out_hstep == 1)
            __riscv_vse32_v_f32m4(out0, _sum, vl);
        else
            __riscv_vsse32_v_f32m4(out0, out_stride, _sum, vl);
        jj += (int)vl;
        pp += vl;
        out0 += out_hstep * vl;
        if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
            pC += vl;
#endif // __riscv_vector
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
            float sum2 = pp[2] * alpha;
            float sum3 = pp[3] * alpha;
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
            out0[0] = sum0;
            out0[out_hstep] = sum1;
            out0[out_hstep * 2] = sum2;
            out0[out_hstep * 3] = sum3;
            out0 += out_hstep * 4;
            pp += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
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
            out0[0] = sum0;
            out0[out_hstep] = sum1;
            out0 += out_hstep * 2;
            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum = *pp * alpha;
            if (pC)
            {
                if (broadcast_type_C == 0)
                    sum += c0;
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    sum += c0;
                if (broadcast_type_C == 3)
                    sum += pC[0] * beta;
                if (broadcast_type_C == 4)
                    sum += pC[0] * beta;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    pC++;
            }
            out0[0] = sum;
            out0 += out_hstep;
            pp++;
        }
        outptr++;
    }
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();
    int tile_k = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(float)));

#if __riscv_vector
    const int packm = std::max(8, csrr_vlenb() / 4);
    const int packn = csrr_vlenb();
#else
    const int packm = 8;
    const int packn = 4;
#endif

    TILE_M = packm;
    TILE_N = packn;
    TILE_K = std::max(block_size, tile_k / block_size * block_size);

    if (K > 0)
    {
        if (TILE_K >= K)
        {
            TILE_K = K;
        }
        else
        {
            const int nn_K = (K + TILE_K - 1) / TILE_K;
            tile_k = (K + nn_K - 1) / nn_K;
            TILE_K = std::max(block_size, tile_k / block_size * block_size);
        }
    }

    // take constant TILE_M/N value when provided
    if (constant_TILE_M > 0)
    {
        TILE_M = (constant_TILE_M + (packm - 1)) / packm * packm;
    }

    if (constant_TILE_N > 0)
    {
        TILE_N = (constant_TILE_N + (packn - 1)) / packn * packn;
    }

    // one driver tile follows the natural producer slab
    TILE_M = std::min(TILE_M, packm);
    TILE_N = std::min(TILE_N, packn);

    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, constant_TILE_K / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(TILE_K, K);
    }

    (void)M;
    (void)N;
    (void)nT;
}
