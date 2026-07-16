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
            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const signed char* p0 = B.row<const signed char>(j + jj) + k0 + kk;
#if __riscv_vector
                    const size_t vl = __riscv_vsetvl_e8mf4(4);
                    const ptrdiff_t B_stride = (ptrdiff_t)B.w;
                    __riscv_vse8_v_i8mf4(pp, __riscv_vlse8_v_i8mf4(p0, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 4, __riscv_vlse8_v_i8mf4(p0 + 1, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 8, __riscv_vlse8_v_i8mf4(p0 + 2, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 12, __riscv_vlse8_v_i8mf4(p0 + 3, B_stride, vl), vl);
#else
                    const signed char* p1 = B.row<const signed char>(j + jj + 1) + k0 + kk;
                    const signed char* p2 = B.row<const signed char>(j + jj + 2) + k0 + kk;
                    const signed char* p3 = B.row<const signed char>(j + jj + 3) + k0 + kk;
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
                    pp += 16;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const signed char* p0 = B.row<const signed char>(j + jj) + k0 + kk;
#if __riscv_vector
                    const size_t vl = __riscv_vsetvl_e8mf4(4);
                    const ptrdiff_t B_stride = (ptrdiff_t)B.w;
                    __riscv_vse8_v_i8mf4(pp, __riscv_vlse8_v_i8mf4(p0, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 4, __riscv_vlse8_v_i8mf4(p0 + 1, B_stride, vl), vl);
#else
                    const signed char* p1 = B.row<const signed char>(j + jj + 1) + k0 + kk;
                    const signed char* p2 = B.row<const signed char>(j + jj + 2) + k0 + kk;
                    const signed char* p3 = B.row<const signed char>(j + jj + 3) + k0 + kk;
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
                    pp[4] = p2[0];
                    pp[5] = p2[1];
                    pp[6] = p3[0];
                    pp[7] = p3[1];
#endif
                    pp += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = B.row<const signed char>(j + jj)[k0 + kk];
                    pp[1] = B.row<const signed char>(j + jj + 1)[k0 + kk];
                    pp[2] = B.row<const signed char>(j + jj + 2)[k0 + kk];
                    pp[3] = B.row<const signed char>(j + jj + 3)[k0 + kk];
                    pp += 4;
                }

                pd[0] = 1.f / B_scales.row(j + jj)[g];
                pd[1] = 1.f / B_scales.row(j + jj + 1)[g];
                pd[2] = 1.f / B_scales.row(j + jj + 2)[g];
                pd[3] = 1.f / B_scales.row(j + jj + 3)[g];
                pd += 4;
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const signed char* p0 = B.row<const signed char>(j + jj) + k0 + kk;
#if __riscv_vector
                    const size_t vl = __riscv_vsetvl_e8mf4(2);
                    const ptrdiff_t B_stride = (ptrdiff_t)B.w;
                    __riscv_vse8_v_i8mf4(pp, __riscv_vlse8_v_i8mf4(p0, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 2, __riscv_vlse8_v_i8mf4(p0 + 1, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 4, __riscv_vlse8_v_i8mf4(p0 + 2, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 6, __riscv_vlse8_v_i8mf4(p0 + 3, B_stride, vl), vl);
#else
                    const signed char* p1 = B.row<const signed char>(j + jj + 1) + k0 + kk;
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp[4] = p1[0];
                    pp[5] = p1[1];
                    pp[6] = p1[2];
                    pp[7] = p1[3];
#endif
                    pp += 8;
                }
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const signed char* p0 = B.row<const signed char>(j + jj) + k0 + kk;
#if __riscv_vector
                    const size_t vl = __riscv_vsetvl_e8mf4(2);
                    const ptrdiff_t B_stride = (ptrdiff_t)B.w;
                    __riscv_vse8_v_i8mf4(pp, __riscv_vlse8_v_i8mf4(p0, B_stride, vl), vl);
                    __riscv_vse8_v_i8mf4(pp + 2, __riscv_vlse8_v_i8mf4(p0 + 1, B_stride, vl), vl);
#else
                    const signed char* p1 = B.row<const signed char>(j + jj + 1) + k0 + kk;
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
#endif
                    pp += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = B.row<const signed char>(j + jj)[k0 + kk];
                    pp[1] = B.row<const signed char>(j + jj + 1)[k0 + kk];
                    pp += 2;
                }

                pd[0] = 1.f / B_scales.row(j + jj)[g];
                pd[1] = 1.f / B_scales.row(j + jj + 1)[g];
                pd += 2;
            }
        }
        for (; jj < max_jj; jj++)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);
                const signed char* p0 = B.row<const signed char>(j + jj) + k0;
                for (int kk = 0; kk < max_kk; kk++)
                    *pp++ = p0[kk];
                *pd++ = 1.f / B_scales.row(j + jj)[g];
            }
        }
    }

    return 0;
}

// K-major, row-interleaved MR-packn/MR2/MR1
static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int block_size, const float* input_scale_ptr)
{
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int K = AT_tile.w;
    const int block_count = AT_descales_tile.w;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    const ptrdiff_t A_stride = (ptrdiff_t)A_hstep * sizeof(float);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            vfloat32m1_t _absmax = __riscv_vfmv_v_f_f32m1(0.f, vl);

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _v = __riscv_vlse32_v_f32m1(p0 + k0 + kk, A_stride, vl);
                if (input_scale_ptr)
                    _v = __riscv_vfmul_vf_f32m1(_v, input_scale_ptr[k0 + kk], vl);
                _absmax = __riscv_vfmax_vv_f32m1(_absmax, __riscv_vfabs_v_f32m1(_v, vl), vl);
            }

            vfloat32m1_t _scale = __riscv_vfrdiv_vf_f32m1(_absmax, 127.f, vl);
            _scale = __riscv_vfmerge_vfm_f32m1(_scale, 0.f, __riscv_vmfeq_vf_f32m1_b32(_absmax, 0.f, vl), vl);
            __riscv_vse32_v_f32m1(pd, __riscv_vfmul_vf_f32m1(_absmax, 1.f / 127.f, vl), vl);
            pd += packn;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _v = __riscv_vlse32_v_f32m1(p0 + k0 + kk, A_stride, vl);
                if (input_scale_ptr)
                    _v = __riscv_vfmul_vf_f32m1(_v, input_scale_ptr[k0 + kk], vl);
                vint32m1_t _v32 = __riscv_vfcvt_x_f_v_i32m1_rm(__riscv_vfmul_vv_f32m1(_v, _scale, vl), __RISCV_FRM_RMM, vl);
                _v32 = __riscv_vmax_vx_i32m1(_v32, -127, vl);
                _v32 = __riscv_vmin_vx_i32m1(_v32, 127, vl);
                const vint16mf2_t _v16 = __riscv_vnclip_wx_i16mf2(_v32, 0, __RISCV_VXRM_RNU, vl);
                __riscv_vse8_v_i8mf4(pp, __riscv_vnclip_wx_i8mf4(_v16, 0, __RISCV_VXRM_RNU, vl), vl);
                pp += packn;
            }
        }
    }
#endif
    for (; ii + 1 < max_ii; ii += 2)
    {
        const int i0 = i + ii;
        const int i1 = i + ii + 1;
        const float* p0 = (const float*)A + (size_t)i0 * A_hstep;
        const float* p1 = (const float*)A + (size_t)i1 * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;

#if __riscv_vector
            int kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(p0 + k0 + kk, vl);
                vfloat32m8_t _v1 = __riscv_vle32_v_f32m8(p1 + k0 + kk, vl);
                if (input_scale_ptr)
                {
                    const vfloat32m8_t _s = __riscv_vle32_v_f32m8(input_scale_ptr + k0 + kk, vl);
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
                const int k = k0 + kk;
                float v0 = p0[k];
                float v1 = p1[k];
                if (input_scale_ptr)
                {
                    v0 *= input_scale_ptr[k];
                    v1 *= input_scale_ptr[k];
                }
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
            }
#endif

            volatile double scale0_fp64 = absmax0 == 0.f ? 0.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 0.0 : 127.0 / (double)absmax1;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

#if __riscv_vector
            kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(p0 + k0 + kk, vl);
                vfloat32m8_t _v1 = __riscv_vle32_v_f32m8(p1 + k0 + kk, vl);
                if (input_scale_ptr)
                {
                    const vfloat32m8_t _s = __riscv_vle32_v_f32m8(input_scale_ptr + k0 + kk, vl);
                    _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                }
                const vint8m2_t _q0 = float2int8(__riscv_vfmul_vf_f32m8(_v0, scale0, vl), vl);
                const vint8m2_t _q1 = float2int8(__riscv_vfmul_vf_f32m8(_v1, scale1, vl), vl);
                __riscv_vsse8_v_i8m2(pp, 2, _q0, vl);
                __riscv_vsse8_v_i8m2(pp + 1, 2, _q1, vl);
                pp += vl * 2;
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v0 = p0[k];
                float v1 = p1[k];
                if (input_scale_ptr)
                {
                    v0 *= input_scale_ptr[k];
                    v1 *= input_scale_ptr[k];
                    asm volatile(""
                                 : "+f"(v0), "+f"(v1));
                }
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
            }
#endif
        }
    }
    for (; ii < max_ii; ii++)
    {
        const int i0 = i + ii;
        const float* p0 = (const float*)A + (size_t)i0 * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax = 0.f;

#if __riscv_vector
            int kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0 + k0 + kk, vl);
                if (input_scale_ptr)
                    _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(input_scale_ptr + k0 + kk, vl), vl);
                _v = __riscv_vfabs_v_f32m8(_v, vl);
                absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = p0[k];
                if (input_scale_ptr)
                    v *= input_scale_ptr[k];
                absmax = std::max(absmax, fabsf(v));
            }
#endif

            volatile double scale_fp64 = absmax == 0.f ? 0.0 : 127.0 / (double)absmax;
            const float scale = (float)scale_fp64;
            *pd++ = absmax / 127.f;

#if __riscv_vector
            kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0 + k0 + kk, vl);
                if (input_scale_ptr)
                    _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(input_scale_ptr + k0 + kk, vl), vl);
                __riscv_vse8_v_i8m2(pp, float2int8(__riscv_vfmul_vf_f32m8(_v, scale, vl), vl), vl);
                pp += vl;
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = p0[k];
                if (input_scale_ptr)
                {
                    v *= input_scale_ptr[k];
                    asm volatile(""
                                 : "+f"(v));
                }
                *pp++ = float2int8(v * scale);
            }
#endif
        }
    }
}

// K-major, row-interleaved MR-packn/MR2/MR1
static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int block_size, const float* input_scale_ptr)
{
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int K = AT_tile.w;
    const int block_count = AT_descales_tile.w;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const int i0 = i + ii;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            vfloat32m1_t _absmax = __riscv_vfmv_v_f_f32m1(0.f, vl);

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _v = __riscv_vle32_v_f32m1((const float*)A + (size_t)(k0 + kk) * A_hstep + i0, vl);
                if (input_scale_ptr)
                    _v = __riscv_vfmul_vf_f32m1(_v, input_scale_ptr[k0 + kk], vl);
                _absmax = __riscv_vfmax_vv_f32m1(_absmax, __riscv_vfabs_v_f32m1(_v, vl), vl);
            }

            vfloat32m1_t _scale = __riscv_vfrdiv_vf_f32m1(_absmax, 127.f, vl);
            _scale = __riscv_vfmerge_vfm_f32m1(_scale, 0.f, __riscv_vmfeq_vf_f32m1_b32(_absmax, 0.f, vl), vl);
            __riscv_vse32_v_f32m1(pd, __riscv_vfmul_vf_f32m1(_absmax, 1.f / 127.f, vl), vl);
            pd += packn;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _v = __riscv_vle32_v_f32m1((const float*)A + (size_t)(k0 + kk) * A_hstep + i0, vl);
                if (input_scale_ptr)
                    _v = __riscv_vfmul_vf_f32m1(_v, input_scale_ptr[k0 + kk], vl);
                vint32m1_t _v32 = __riscv_vfcvt_x_f_v_i32m1_rm(__riscv_vfmul_vv_f32m1(_v, _scale, vl), __RISCV_FRM_RMM, vl);
                _v32 = __riscv_vmax_vx_i32m1(_v32, -127, vl);
                _v32 = __riscv_vmin_vx_i32m1(_v32, 127, vl);
                const vint16mf2_t _v16 = __riscv_vnclip_wx_i16mf2(_v32, 0, __RISCV_VXRM_RNU, vl);
                __riscv_vse8_v_i8mf4(pp, __riscv_vnclip_wx_i8mf4(_v16, 0, __RISCV_VXRM_RNU, vl), vl);
                pp += packn;
            }
        }
    }
#endif
    for (; ii + 1 < max_ii; ii += 2)
    {
        const int i0 = i + ii;
        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;

#if __riscv_vector
            int kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v0 = __riscv_vlse32_v_f32m8((const float*)A + (size_t)(k0 + kk) * A_hstep + i0, (ptrdiff_t)A_hstep * sizeof(float), vl);
                vfloat32m8_t _v1 = __riscv_vlse32_v_f32m8((const float*)A + (size_t)(k0 + kk) * A_hstep + i0 + 1, (ptrdiff_t)A_hstep * sizeof(float), vl);
                if (input_scale_ptr)
                {
                    const vfloat32m8_t _s = __riscv_vle32_v_f32m8(input_scale_ptr + k0 + kk, vl);
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
                const int k = k0 + kk;
                const float* p0 = (const float*)A + (size_t)k * A_hstep + i0;
                float v0 = p0[0];
                float v1 = p0[1];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k];
                    v0 *= s;
                    v1 *= s;
                }
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
            }
#endif

            volatile double scale0_fp64 = absmax0 == 0.f ? 0.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 0.0 : 127.0 / (double)absmax1;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

#if __riscv_vector
            kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v0 = __riscv_vlse32_v_f32m8((const float*)A + (size_t)(k0 + kk) * A_hstep + i0, (ptrdiff_t)A_hstep * sizeof(float), vl);
                vfloat32m8_t _v1 = __riscv_vlse32_v_f32m8((const float*)A + (size_t)(k0 + kk) * A_hstep + i0 + 1, (ptrdiff_t)A_hstep * sizeof(float), vl);
                if (input_scale_ptr)
                {
                    const vfloat32m8_t _s = __riscv_vle32_v_f32m8(input_scale_ptr + k0 + kk, vl);
                    _v0 = __riscv_vfmul_vv_f32m8(_v0, _s, vl);
                    _v1 = __riscv_vfmul_vv_f32m8(_v1, _s, vl);
                }
                const vint8m2_t _q0 = float2int8(__riscv_vfmul_vf_f32m8(_v0, scale0, vl), vl);
                const vint8m2_t _q1 = float2int8(__riscv_vfmul_vf_f32m8(_v1, scale1, vl), vl);
                __riscv_vsse8_v_i8m2(pp, 2, _q0, vl);
                __riscv_vsse8_v_i8m2(pp + 1, 2, _q1, vl);
                pp += vl * 2;
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                const float* p0 = (const float*)A + (size_t)k * A_hstep + i0;
                float v0 = p0[0];
                float v1 = p0[1];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k];
                    v0 *= s;
                    v1 *= s;
                    asm volatile(""
                                 : "+f"(v0), "+f"(v1));
                }
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
            }
#endif
        }
    }
    for (; ii < max_ii; ii++)
    {
        const int i0 = i + ii;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax = 0.f;

#if __riscv_vector
            int kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v = __riscv_vlse32_v_f32m8((const float*)A + (size_t)(k0 + kk) * A_hstep + i0, (ptrdiff_t)A_hstep * sizeof(float), vl);
                if (input_scale_ptr)
                    _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(input_scale_ptr + k0 + kk, vl), vl);
                _v = __riscv_vfabs_v_f32m8(_v, vl);
                absmax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(_v, __riscv_vfmv_s_f_f32m1(absmax, 1), vl));
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ((const float*)A)[(size_t)k * A_hstep + i0];
                if (input_scale_ptr)
                    v *= input_scale_ptr[k];
                absmax = std::max(absmax, fabsf(v));
            }
#endif

            volatile double scale_fp64 = absmax == 0.f ? 0.0 : 127.0 / (double)absmax;
            const float scale = (float)scale_fp64;
            *pd++ = absmax / 127.f;

#if __riscv_vector
            kk = 0;
            while (kk < max_kk)
            {
                const size_t vl = __riscv_vsetvl_e32m8(max_kk - kk);
                vfloat32m8_t _v = __riscv_vlse32_v_f32m8((const float*)A + (size_t)(k0 + kk) * A_hstep + i0, (ptrdiff_t)A_hstep * sizeof(float), vl);
                if (input_scale_ptr)
                    _v = __riscv_vfmul_vv_f32m8(_v, __riscv_vle32_v_f32m8(input_scale_ptr + k0 + kk, vl), vl);
                __riscv_vse8_v_i8m2(pp, float2int8(__riscv_vfmul_vf_f32m8(_v, scale, vl), vl), vl);
                pp += vl;
                kk += vl;
            }
#else
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ((const float*)A)[(size_t)k * A_hstep + i0];
                if (input_scale_ptr)
                {
                    v *= input_scale_ptr[k];
                    asm volatile(""
                                 : "+f"(v));
                }
                *pp++ = float2int8(v * scale);
            }
#endif
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int block_size)
{
    const signed char* pAT = AT_tile;
    const float* pAT_descales = AT_descales_tile;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int K = AT_tile.w;
    const int block_count = AT_descales_tile.w;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl4 = __riscv_vsetvl_e8mf4(4);
    for (; ii < max_ii;)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;
        const int mr = ii + (packn - 1) < max_ii ? packn : ii + 1 < max_ii ? 2 : 1;
        const size_t vl = __riscv_vsetvl_e32m1(mr);

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
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
                    const vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    const uint32_t b = *(const uint32_t*)pB;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += mr;
                    pB += 4;
                }

                const vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
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

            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + mr, _fsum1, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 2, _fsum2, vl);
            __riscv_vse32_v_f32m1(outptr + mr * 3, _fsum3, vl);
            outptr += mr * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
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
                    const vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    const uint16_t b = *(const uint16_t*)pB;
                    _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += mr;
                    pB += 2;
                }

                const vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), _ad, vl);
                _fsum0 = __riscv_vfmacc_vf_f32m1(_fsum0, pB_descales[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), _ad, vl);
                _fsum1 = __riscv_vfmacc_vf_f32m1(_fsum1, pB_descales[1], _v, vl);
                pA_descales += mr;
                pB_descales += 2;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + mr, _fsum1, vl);
            outptr += mr * 2;
        }
        for (; jj < max_jj; jj++)
        {
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
                    const vint16mf2_t _a = __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(pA, vl), vl);
                    _sum = __riscv_vwmacc_vx_i32m1(_sum, pB[0], _a, vl);
                    pA += mr;
                    pB++;
                }

                const vfloat32m1_t _ad = __riscv_vle32_v_f32m1(pA_descales, vl);
                const vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), _ad, vl);
                _fsum = __riscv_vfmacc_vf_f32m1(_fsum, pB_descales[0], _v, vl);
                pA_descales += mr;
                pB_descales++;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += mr;
        }

        pAT += K * mr;
        pAT_descales += block_count * mr;
        ii += mr;
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;
        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
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

            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;
            outptr[4] = sum02;
            outptr[5] = sum12;
            outptr[6] = sum03;
            outptr[7] = sum13;
            outptr += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
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
            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;
            outptr += 4;
        }
        for (; jj < max_jj; jj++)
        {
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
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
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
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;
            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
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
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }
        for (; jj < max_jj; jj++)
        {
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
            outptr[0] = sum;
            outptr++;
        }
        pAT += K;
        pAT_descales += block_count;
    }
#endif
}

static void unpack_output_tile_wq_int8(const float* pp, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int N, float alpha, float beta)
{
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
                    const vfloat32m1_t _c0 = __riscv_vlse32_v_f32m1(pC + jj, c_stride, vl_packn);
                    if (beta == 1.f)
                        _sum = __riscv_vfadd_vv_f32m1(_sum, _c0, vl_packn);
                    else
                        _sum = __riscv_vfmacc_vf_f32m1(_sum, beta, _c0, vl_packn);
                }
                if (broadcast_type_C == 4)
                    _sum = __riscv_vfadd_vf_f32m1(_sum, pC[jj] * beta, vl_packn);
            }
            __riscv_vsse32_v_f32m1(outptr + jj, out_stride, _sum, vl_packn);
            pp += packn;
        }
        outptr += out_hstep * packn;
    }
    for (; ii + 1 < max_ii; ii += 2)
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
        float c1 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            c0 = pC[0] * beta;
            c1 = pC[broadcast_type_C == 0 ? 0 : 1] * beta;
        }

        float* out0 = outptr;
        float* out1 = out0 + out_hstep;
        const size_t vl = __riscv_vsetvl_e32m4(max_jj);
        const vfloat32m4x2_t _s = __riscv_vlseg2e32_v_f32m4x2(pp, vl);
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
                const vfloat32m4_t _c0 = __riscv_vle32_v_f32m4(pC, vl);
                const vfloat32m4_t _c1 = __riscv_vle32_v_f32m4(pC + c_hstep, vl);
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
                const vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
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
        pp += vl * 2;
        outptr += out_hstep * 2;
    }
    for (; ii < max_ii; ii++)
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
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
            c0 = pC[0] * beta;

        float* out0 = outptr;
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
                const vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                if (beta == 1.f)
                    _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                else
                    _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
            }
            if (broadcast_type_C == 4)
            {
                const vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                if (beta == 1.f)
                    _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                else
                    _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
            }
        }

        __riscv_vse32_v_f32m4(out0, _sum, vl);
        pp += vl;
        outptr += out_hstep;
    }
#else
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
            pp += 8;

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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = pp[0] * alpha;
            float sum10 = pp[1] * alpha;
            float sum01 = pp[2] * alpha;
            float sum11 = pp[3] * alpha;
            pp += 4;

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
            }

            out0[0] = sum00;
            out0[1] = sum01;
            out1[0] = sum10;
            out1[1] = sum11;
            out0 += 2;
            out1 += 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
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
                    sum0 += pC[0] * beta;
                    sum1 += pC[c_hstep] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[0] * beta;
                }
            }
            out0[0] = sum0;
            out1[0] = sum1;
            out0++;
            out1++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
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
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
            float sum2 = pp[2] * alpha;
            float sum3 = pp[3] * alpha;
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
            }
            out0[0] = sum0;
            out0[1] = sum1;
            out0[2] = sum2;
            out0[3] = sum3;
            out0 += 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
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
            }
            out0[0] = sum0;
            out0[1] = sum1;
            out0 += 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum = *pp++ * alpha;
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
            }
            out0[0] = sum;
            out0++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
        outptr += out_hstep;
    }
#endif
}

static void transpose_unpack_output_tile_wq_int8(const float* pp, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int N, float alpha, float beta)
{
    beta *= alpha;
    (void)N;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    float* outptr = (float*)top_blob + (size_t)j * out_hstep + i;
#if __riscv_vector
    const ptrdiff_t out_stride = (ptrdiff_t)out_hstep * sizeof(float);
#endif

    int ii = 0;
#if __riscv_vector
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
                    const vfloat32m1_t _c0 = __riscv_vlse32_v_f32m1(pC + jj, c_stride, vl_packn);
                    if (beta == 1.f)
                        _sum = __riscv_vfadd_vv_f32m1(_sum, _c0, vl_packn);
                    else
                        _sum = __riscv_vfmacc_vf_f32m1(_sum, beta, _c0, vl_packn);
                }
                if (broadcast_type_C == 4)
                    _sum = __riscv_vfadd_vf_f32m1(_sum, pC[jj] * beta, vl_packn);
            }
            __riscv_vse32_v_f32m1(outptr + (size_t)jj * out_hstep, _sum, vl_packn);
            pp += packn;
        }
        outptr += packn;
    }
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

        const size_t vl = __riscv_vsetvl_e32m4(max_jj);
        const vfloat32m4x2_t _s = __riscv_vlseg2e32_v_f32m4x2(pp, vl);
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
                const vfloat32m4_t _c0 = __riscv_vle32_v_f32m4(pC, vl);
                const vfloat32m4_t _c1 = __riscv_vle32_v_f32m4(pC + c_hstep, vl);
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
                const vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
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
        const vfloat32m4x2_t _sum = __riscv_vcreate_v_f32m4x2(_sum0, _sum1);
        if (out_hstep == 2)
            __riscv_vsseg2e32_v_f32m4x2(out0, _sum, vl);
        else
            __riscv_vssseg2e32_v_f32m4x2(out0, out_stride, _sum, vl);
        pp += vl * 2;
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
                const vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
                if (beta == 1.f)
                    _sum = __riscv_vfadd_vv_f32m4(_sum, _c, vl);
                else
                    _sum = __riscv_vfmacc_vf_f32m4(_sum, beta, _c, vl);
            }
            if (broadcast_type_C == 4)
            {
                const vfloat32m4_t _c = __riscv_vle32_v_f32m4(pC, vl);
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
        pp += vl;
        outptr++;
    }
#else
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
            pp += 8;
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = pp[0] * alpha;
            float sum10 = pp[1] * alpha;
            float sum01 = pp[2] * alpha;
            float sum11 = pp[3] * alpha;
            pp += 4;
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
            }
            out0[0] = sum00;
            out0[1] = sum10;
            out0[out_hstep] = sum01;
            out0[out_hstep + 1] = sum11;
            out0 += out_hstep * 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
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
                    sum0 += pC[0] * beta;
                    sum1 += pC[c_hstep] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pC[0] * beta;
                    sum1 += pC[0] * beta;
                }
            }
            out0[0] = sum0;
            out0[1] = sum1;
            out0 += out_hstep;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
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
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
            float sum2 = pp[2] * alpha;
            float sum3 = pp[3] * alpha;
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
            }
            out0[0] = sum0;
            out0[out_hstep] = sum1;
            out0[out_hstep * 2] = sum2;
            out0[out_hstep * 3] = sum3;
            out0 += out_hstep * 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = pp[0] * alpha;
            float sum1 = pp[1] * alpha;
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
            }
            out0[0] = sum0;
            out0[out_hstep] = sum1;
            out0 += out_hstep * 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum = *pp++ * alpha;
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
            }
            out0[0] = sum;
            out0 += out_hstep;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
        outptr++;
    }
#endif
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
#if __riscv_vector
    const int packm = std::max(8, csrr_vlenb() / 4);
    const int packn = csrr_vlenb();
#else
    const int packm = 8;
    const int packn = 4;
#endif

    TILE_M = packm;
    TILE_N = packn;
    TILE_K = K;

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

    (void)M;
    (void)N;
    (void)constant_TILE_K;
    (void)nT;
}
