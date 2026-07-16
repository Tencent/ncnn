// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
int pack_B_wq_int8_loongson_mmi(const Mat& B, const Mat& B_scales, Mat& packed_B, Mat& packed_B_descales, int N, int K, int block_size, const Option& opt);
void quantize_A_tile_wq_int8_loongson_mmi(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int block_size, const float* input_scale_ptr);
void transpose_quantize_A_tile_wq_int8_loongson_mmi(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int block_size, const float* input_scale_ptr);
void gemm_transB_packed_tile_wq_int8_loongson_mmi(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int block_size);
#endif

// group-major, output-major within each K4/K2/K1 fragment
static int pack_B_wq_int8(const Mat& B, const Mat& B_scales, Mat& packed_B, Mat& packed_B_descales, int N, int K, int block_size, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
        return pack_B_wq_int8_loongson_mmi(B, B_scales, packed_B, packed_B_descales, N, K, block_size, opt);
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

    int j = 0;
#if __mips_msa
    const int nn8 = N / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn8; ppj++)
    {
        const int j = ppj * 8;
        signed char* pp = (signed char*)packed_B + (size_t)j * K;
        float* pd = (float*)packed_B_descales + (size_t)j * block_count;

        for (int jj = 0; jj < 8; jj += 4)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const signed char* p0 = B.row<const signed char>(j + jj) + k0 + kk;
                    const signed char* p1 = B.row<const signed char>(j + jj + 1) + k0 + kk;
                    const signed char* p2 = B.row<const signed char>(j + jj + 2) + k0 + kk;
                    const signed char* p3 = B.row<const signed char>(j + jj + 3) + k0 + kk;
                    const v16i8 _p = (v16i8)__msa_set_w(__msa_load_w(p0), __msa_load_w(p1), __msa_load_w(p2), __msa_load_w(p3));
                    __msa_st_b(_p, pp, 0);
                    pp += 16;
                }
                if (kk + 1 < max_kk)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        const signed char* p0 = B.row<const signed char>(j + jj + n) + k0 + kk;
                        pp[0] = p0[0];
                        pp[1] = p0[1];
                        pp += 2;
                    }
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    for (int n = 0; n < 4; n++)
                        *pp++ = B.row<const signed char>(j + jj + n)[k0 + kk];
                }

                for (int n = 0; n < 4; n++)
                    *pd++ = 1.f / B_scales.row(j + jj + n)[g];
            }
        }
    }
    j += nn8 * 8;

    const int nn4 = (N - j) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn4; ppj++)
    {
        const int j = nn8 * 8 + ppj * 4;
        signed char* pp = (signed char*)packed_B + (size_t)j * K;
        float* pd = (float*)packed_B_descales + (size_t)j * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const signed char* p0 = B.row<const signed char>(j) + k0 + kk;
                const signed char* p1 = B.row<const signed char>(j + 1) + k0 + kk;
                const signed char* p2 = B.row<const signed char>(j + 2) + k0 + kk;
                const signed char* p3 = B.row<const signed char>(j + 3) + k0 + kk;
                const v16i8 _p = (v16i8)__msa_set_w(__msa_load_w(p0), __msa_load_w(p1), __msa_load_w(p2), __msa_load_w(p3));
                __msa_st_b(_p, pp, 0);
                pp += 16;
            }
            if (kk + 1 < max_kk)
            {
                for (int n = 0; n < 4; n++)
                {
                    const signed char* p0 = B.row<const signed char>(j + n) + k0 + kk;
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp += 2;
                }
                kk += 2;
            }
            if (kk < max_kk)
            {
                for (int n = 0; n < 4; n++)
                    *pp++ = B.row<const signed char>(j + n)[k0 + kk];
            }

            for (int n = 0; n < 4; n++)
                *pd++ = 1.f / B_scales.row(j + n)[g];
        }
    }
    j += nn4 * 4;
#endif // __mips_msa

    const int nn2 = (N - j) / 2;
    const int j2 = j;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn2; ppj++)
    {
        const int j = j2 + ppj * 2;
        signed char* pp = (signed char*)packed_B + (size_t)j * K;
        float* pd = (float*)packed_B_descales + (size_t)j * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const signed char* p0 = B.row<const signed char>(j) + k0 + kk;
                const signed char* p1 = B.row<const signed char>(j + 1) + k0 + kk;
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p1[0];
                pp[5] = p1[1];
                pp[6] = p1[2];
                pp[7] = p1[3];
                pp += 8;
            }
            if (kk + 1 < max_kk)
            {
                const signed char* p0 = B.row<const signed char>(j) + k0 + kk;
                const signed char* p1 = B.row<const signed char>(j + 1) + k0 + kk;
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp += 4;
                kk += 2;
            }
            if (kk < max_kk)
            {
                pp[0] = B.row<const signed char>(j)[k0 + kk];
                pp[1] = B.row<const signed char>(j + 1)[k0 + kk];
                pp += 2;
            }

            *pd++ = 1.f / B_scales.row(j)[g];
            *pd++ = 1.f / B_scales.row(j + 1)[g];
        }
    }
    j += nn2 * 2;

    if (j < N)
    {
        signed char* pp = (signed char*)packed_B + (size_t)j * K;
        float* pd = (float*)packed_B_descales + (size_t)j * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const signed char* p0 = B.row<const signed char>(j) + k0;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[kk];
                pp[1] = p0[kk + 1];
                pp[2] = p0[kk + 2];
                pp[3] = p0[kk + 3];
                pp += 4;
            }
            if (kk + 1 < max_kk)
            {
                pp[0] = p0[kk];
                pp[1] = p0[kk + 1];
                pp += 2;
                kk += 2;
            }
            if (kk < max_kk)
                *pp++ = p0[kk];

            *pd++ = 1.f / B_scales.row(j)[g];
        }
    }

    return 0;
}

// group-major, row-major within each K4/K2/K1 fragment
static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int block_size, const float* input_scale_ptr)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        quantize_A_tile_wq_int8_loongson_mmi(A, AT_tile, AT_descales_tile, i, max_ii, block_size, input_scale_ptr);
        return;
    }
#endif

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int K = AT_tile.w;
    const int block_count = AT_descales_tile.w;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep;
        const float* p1 = (const float*)A + (size_t)(i + ii + 1) * A_hstep;
        const float* p2 = (const float*)A + (size_t)(i + ii + 2) * A_hstep;
        const float* p3 = (const float*)A + (size_t)(i + ii + 3) * A_hstep;
        const float* p4 = (const float*)A + (size_t)(i + ii + 4) * A_hstep;
        const float* p5 = (const float*)A + (size_t)(i + ii + 5) * A_hstep;
        const float* p6 = (const float*)A + (size_t)(i + ii + 6) * A_hstep;
        const float* p7 = (const float*)A + (size_t)(i + ii + 7) * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                v4f32 _p0 = (v4f32)__msa_ld_w(p0 + k0 + kk, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(p1 + k0 + kk, 0);
                v4f32 _p2 = (v4f32)__msa_ld_w(p2 + k0 + kk, 0);
                v4f32 _p3 = (v4f32)__msa_ld_w(p3 + k0 + kk, 0);
                v4f32 _p4 = (v4f32)__msa_ld_w(p4 + k0 + kk, 0);
                v4f32 _p5 = (v4f32)__msa_ld_w(p5 + k0 + kk, 0);
                v4f32 _p6 = (v4f32)__msa_ld_w(p6 + k0 + kk, 0);
                v4f32 _p7 = (v4f32)__msa_ld_w(p7 + k0 + kk, 0);
                if (input_scale_ptr)
                {
                    const v4f32 _s = (v4f32)__msa_ld_w(input_scale_ptr + k0 + kk, 0);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = __msa_fmul_w(_p1, _s);
                    _p2 = __msa_fmul_w(_p2, _s);
                    _p3 = __msa_fmul_w(_p3, _s);
                    _p4 = __msa_fmul_w(_p4, _s);
                    _p5 = __msa_fmul_w(_p5, _s);
                    _p6 = __msa_fmul_w(_p6, _s);
                    _p7 = __msa_fmul_w(_p7, _s);
                }
                _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                _absmax2 = __msa_fmax_w(_absmax2, (v4f32)__msa_and_v((v16u8)_p2, _abs_mask));
                _absmax3 = __msa_fmax_w(_absmax3, (v4f32)__msa_and_v((v16u8)_p3, _abs_mask));
                _absmax4 = __msa_fmax_w(_absmax4, (v4f32)__msa_and_v((v16u8)_p4, _abs_mask));
                _absmax5 = __msa_fmax_w(_absmax5, (v4f32)__msa_and_v((v16u8)_p5, _abs_mask));
                _absmax6 = __msa_fmax_w(_absmax6, (v4f32)__msa_and_v((v16u8)_p6, _abs_mask));
                _absmax7 = __msa_fmax_w(_absmax7, (v4f32)__msa_and_v((v16u8)_p7, _abs_mask));
            }

            float absmax0 = __msa_reduce_fmax_w(_absmax0);
            float absmax1 = __msa_reduce_fmax_w(_absmax1);
            float absmax2 = __msa_reduce_fmax_w(_absmax2);
            float absmax3 = __msa_reduce_fmax_w(_absmax3);
            float absmax4 = __msa_reduce_fmax_w(_absmax4);
            float absmax5 = __msa_reduce_fmax_w(_absmax5);
            float absmax6 = __msa_reduce_fmax_w(_absmax6);
            float absmax7 = __msa_reduce_fmax_w(_absmax7);

            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0[k] * s));
                absmax1 = std::max(absmax1, fabsf(p1[k] * s));
                absmax2 = std::max(absmax2, fabsf(p2[k] * s));
                absmax3 = std::max(absmax3, fabsf(p3[k] * s));
                absmax4 = std::max(absmax4, fabsf(p4[k] * s));
                absmax5 = std::max(absmax5, fabsf(p5[k] * s));
                absmax6 = std::max(absmax6, fabsf(p6[k] * s));
                absmax7 = std::max(absmax7, fabsf(p7[k] * s));
            }

            volatile double scale0_fp64 = absmax0 == 0.f ? 1.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 1.0 : 127.0 / (double)absmax1;
            volatile double scale2_fp64 = absmax2 == 0.f ? 1.0 : 127.0 / (double)absmax2;
            volatile double scale3_fp64 = absmax3 == 0.f ? 1.0 : 127.0 / (double)absmax3;
            volatile double scale4_fp64 = absmax4 == 0.f ? 1.0 : 127.0 / (double)absmax4;
            volatile double scale5_fp64 = absmax5 == 0.f ? 1.0 : 127.0 / (double)absmax5;
            volatile double scale6_fp64 = absmax6 == 0.f ? 1.0 : 127.0 / (double)absmax6;
            volatile double scale7_fp64 = absmax7 == 0.f ? 1.0 : 127.0 / (double)absmax7;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            const float scale2 = (float)scale2_fp64;
            const float scale3 = (float)scale3_fp64;
            const float scale4 = (float)scale4_fp64;
            const float scale5 = (float)scale5_fp64;
            const float scale6 = (float)scale6_fp64;
            const float scale7 = (float)scale7_fp64;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd[4] = absmax4 / 127.f;
            pd[5] = absmax5 / 127.f;
            pd[6] = absmax6 / 127.f;
            pd[7] = absmax7 / 127.f;
            pd += 8;

            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const v4f32 _s = input_scale_ptr ? (v4f32)__msa_ld_w(input_scale_ptr + k0 + kk, 0) : __msa_fill_w_f32(1.f);
                v4f32 _p = (v4f32)__msa_ld_w(p0 + k0 + kk, 0);
                _p = __msa_fmul_w(__msa_fmul_w(_p, _s), __msa_fill_w_f32(scale0));
                ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                _p = (v4f32)__msa_ld_w(p1 + k0 + kk, 0);
                _p = __msa_fmul_w(__msa_fmul_w(_p, _s), __msa_fill_w_f32(scale1));
                ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                _p = (v4f32)__msa_ld_w(p2 + k0 + kk, 0);
                _p = __msa_fmul_w(__msa_fmul_w(_p, _s), __msa_fill_w_f32(scale2));
                ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                _p = (v4f32)__msa_ld_w(p3 + k0 + kk, 0);
                _p = __msa_fmul_w(__msa_fmul_w(_p, _s), __msa_fill_w_f32(scale3));
                ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                _p = (v4f32)__msa_ld_w(p4 + k0 + kk, 0);
                _p = __msa_fmul_w(__msa_fmul_w(_p, _s), __msa_fill_w_f32(scale4));
                ((int*)pp)[4] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                _p = (v4f32)__msa_ld_w(p5 + k0 + kk, 0);
                _p = __msa_fmul_w(__msa_fmul_w(_p, _s), __msa_fill_w_f32(scale5));
                ((int*)pp)[5] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                _p = (v4f32)__msa_ld_w(p6 + k0 + kk, 0);
                _p = __msa_fmul_w(__msa_fmul_w(_p, _s), __msa_fill_w_f32(scale6));
                ((int*)pp)[6] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                _p = (v4f32)__msa_ld_w(p7 + k0 + kk, 0);
                _p = __msa_fmul_w(__msa_fmul_w(_p, _s), __msa_fill_w_f32(scale7));
                ((int*)pp)[7] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                pp += 32;
            }
            if (kk + 1 < max_kk)
            {
                const int k = k0 + kk;
                const float s0 = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                const float s1 = input_scale_ptr ? input_scale_ptr[k + 1] : 1.f;
                pp[0] = float2int8(p0[k] * s0 * scale0);
                pp[1] = float2int8(p0[k + 1] * s1 * scale0);
                pp[2] = float2int8(p1[k] * s0 * scale1);
                pp[3] = float2int8(p1[k + 1] * s1 * scale1);
                pp[4] = float2int8(p2[k] * s0 * scale2);
                pp[5] = float2int8(p2[k + 1] * s1 * scale2);
                pp[6] = float2int8(p3[k] * s0 * scale3);
                pp[7] = float2int8(p3[k + 1] * s1 * scale3);
                pp[8] = float2int8(p4[k] * s0 * scale4);
                pp[9] = float2int8(p4[k + 1] * s1 * scale4);
                pp[10] = float2int8(p5[k] * s0 * scale5);
                pp[11] = float2int8(p5[k + 1] * s1 * scale5);
                pp[12] = float2int8(p6[k] * s0 * scale6);
                pp[13] = float2int8(p6[k + 1] * s1 * scale6);
                pp[14] = float2int8(p7[k] * s0 * scale7);
                pp[15] = float2int8(p7[k + 1] * s1 * scale7);
                pp += 16;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                pp[0] = float2int8(p0[k] * s * scale0);
                pp[1] = float2int8(p1[k] * s * scale1);
                pp[2] = float2int8(p2[k] * s * scale2);
                pp[3] = float2int8(p3[k] * s * scale3);
                pp[4] = float2int8(p4[k] * s * scale4);
                pp[5] = float2int8(p5[k] * s * scale5);
                pp[6] = float2int8(p6[k] * s * scale6);
                pp[7] = float2int8(p7[k] * s * scale7);
                pp += 8;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int i0 = i + ii;
        const int i1 = i + ii + 1;
        const int i2 = i + ii + 2;
        const int i3 = i + ii + 3;
        const float* p0 = (const float*)A + (size_t)i0 * A_hstep;
        const float* p1 = (const float*)A + (size_t)i1 * A_hstep;
        const float* p2 = (const float*)A + (size_t)i2 * A_hstep;
        const float* p3 = (const float*)A + (size_t)i3 * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            float absmax2 = 0.f;
            float absmax3 = 0.f;

            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax3 = (v4f32)__msa_fill_w(0);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                v4f32 _p0 = (v4f32)__msa_ld_w(p0 + k0 + kk, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(p1 + k0 + kk, 0);
                v4f32 _p2 = (v4f32)__msa_ld_w(p2 + k0 + kk, 0);
                v4f32 _p3 = (v4f32)__msa_ld_w(p3 + k0 + kk, 0);
                if (input_scale_ptr)
                {
                    const v4f32 _s = (v4f32)__msa_ld_w(input_scale_ptr + k0 + kk, 0);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = __msa_fmul_w(_p1, _s);
                    _p2 = __msa_fmul_w(_p2, _s);
                    _p3 = __msa_fmul_w(_p3, _s);
                }
                _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                _absmax2 = __msa_fmax_w(_absmax2, (v4f32)__msa_and_v((v16u8)_p2, _abs_mask));
                _absmax3 = __msa_fmax_w(_absmax3, (v4f32)__msa_and_v((v16u8)_p3, _abs_mask));
            }
            absmax0 = __msa_reduce_fmax_w(_absmax0);
            absmax1 = __msa_reduce_fmax_w(_absmax1);
            absmax2 = __msa_reduce_fmax_w(_absmax2);
            absmax3 = __msa_reduce_fmax_w(_absmax3);

            for (; kk < max_kk; kk++)
            {
                float v0 = p0[k0 + kk];
                float v1 = p1[k0 + kk];
                float v2 = p2[k0 + kk];
                float v3 = p3[k0 + kk];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k0 + kk];
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                }
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
                absmax2 = std::max(absmax2, fabsf(v2));
                absmax3 = std::max(absmax3, fabsf(v3));
            }

            volatile double scale0_fp64 = absmax0 == 0.f ? 1.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 1.0 : 127.0 / (double)absmax1;
            volatile double scale2_fp64 = absmax2 == 0.f ? 1.0 : 127.0 / (double)absmax2;
            volatile double scale3_fp64 = absmax3 == 0.f ? 1.0 : 127.0 / (double)absmax3;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            const float scale2 = (float)scale2_fp64;
            const float scale3 = (float)scale3_fp64;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd += 4;

            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float v00 = p0[k0 + kk];
                float v01 = p0[k0 + kk + 1];
                float v02 = p0[k0 + kk + 2];
                float v03 = p0[k0 + kk + 3];
                float v10 = p1[k0 + kk];
                float v11 = p1[k0 + kk + 1];
                float v12 = p1[k0 + kk + 2];
                float v13 = p1[k0 + kk + 3];
                float v20 = p2[k0 + kk];
                float v21 = p2[k0 + kk + 1];
                float v22 = p2[k0 + kk + 2];
                float v23 = p2[k0 + kk + 3];
                float v30 = p3[k0 + kk];
                float v31 = p3[k0 + kk + 1];
                float v32 = p3[k0 + kk + 2];
                float v33 = p3[k0 + kk + 3];
                if (input_scale_ptr)
                {
                    const float s0 = input_scale_ptr[k0 + kk];
                    const float s1 = input_scale_ptr[k0 + kk + 1];
                    const float s2 = input_scale_ptr[k0 + kk + 2];
                    const float s3 = input_scale_ptr[k0 + kk + 3];
                    v00 *= s0;
                    v01 *= s1;
                    v02 *= s2;
                    v03 *= s3;
                    v10 *= s0;
                    v11 *= s1;
                    v12 *= s2;
                    v13 *= s3;
                    v20 *= s0;
                    v21 *= s1;
                    v22 *= s2;
                    v23 *= s3;
                    v30 *= s0;
                    v31 *= s1;
                    v32 *= s2;
                    v33 *= s3;
                    asm volatile(""
                                 : "+f"(v00), "+f"(v01), "+f"(v02), "+f"(v03));
                    asm volatile(""
                                 : "+f"(v10), "+f"(v11), "+f"(v12), "+f"(v13));
                    asm volatile(""
                                 : "+f"(v20), "+f"(v21), "+f"(v22), "+f"(v23));
                    asm volatile(""
                                 : "+f"(v30), "+f"(v31), "+f"(v32), "+f"(v33));
                }
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v02 * scale0);
                pp[3] = float2int8(v03 * scale0);
                pp[4] = float2int8(v10 * scale1);
                pp[5] = float2int8(v11 * scale1);
                pp[6] = float2int8(v12 * scale1);
                pp[7] = float2int8(v13 * scale1);
                pp[8] = float2int8(v20 * scale2);
                pp[9] = float2int8(v21 * scale2);
                pp[10] = float2int8(v22 * scale2);
                pp[11] = float2int8(v23 * scale2);
                pp[12] = float2int8(v30 * scale3);
                pp[13] = float2int8(v31 * scale3);
                pp[14] = float2int8(v32 * scale3);
                pp[15] = float2int8(v33 * scale3);
                pp += 16;
            }
            if (kk + 1 < max_kk)
            {
                float v00 = p0[k0 + kk];
                float v01 = p0[k0 + kk + 1];
                float v10 = p1[k0 + kk];
                float v11 = p1[k0 + kk + 1];
                float v20 = p2[k0 + kk];
                float v21 = p2[k0 + kk + 1];
                float v30 = p3[k0 + kk];
                float v31 = p3[k0 + kk + 1];
                if (input_scale_ptr)
                {
                    const float s0 = input_scale_ptr[k0 + kk];
                    const float s1 = input_scale_ptr[k0 + kk + 1];
                    v00 *= s0;
                    v01 *= s1;
                    v10 *= s0;
                    v11 *= s1;
                    v20 *= s0;
                    v21 *= s1;
                    v30 *= s0;
                    v31 *= s1;
                    asm volatile(""
                                 : "+f"(v00), "+f"(v01), "+f"(v10), "+f"(v11));
                    asm volatile(""
                                 : "+f"(v20), "+f"(v21), "+f"(v30), "+f"(v31));
                }
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v10 * scale1);
                pp[3] = float2int8(v11 * scale1);
                pp[4] = float2int8(v20 * scale2);
                pp[5] = float2int8(v21 * scale2);
                pp[6] = float2int8(v30 * scale3);
                pp[7] = float2int8(v31 * scale3);
                pp += 8;
                kk += 2;
            }
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v0 = p0[k];
                float v1 = p1[k];
                float v2 = p2[k];
                float v3 = p3[k];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k];
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    asm volatile(""
                                 : "+f"(v0), "+f"(v1), "+f"(v2), "+f"(v3));
                }
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
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v0 = p0[k];
                float v1 = p1[k];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k];
                    v0 *= s;
                    v1 *= s;
                }
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
            }

            volatile double scale0_fp64 = absmax0 == 0.f ? 1.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 1.0 : 127.0 / (double)absmax1;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int r = 0; r < 4; r++)
                {
                    const int k = k0 + kk + r;
                    float v0 = p0[k];
                    float v1 = p1[k];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k];
                        v1 *= input_scale_ptr[k];
                        asm volatile(""
                                     : "+f"(v0), "+f"(v1));
                    }
                    pp[r] = float2int8(v0 * scale0);
                    pp[4 + r] = float2int8(v1 * scale1);
                }
                pp += 8;
            }
            if (kk + 1 < max_kk)
            {
                for (int r = 0; r < 2; r++)
                {
                    const int k = k0 + kk + r;
                    float v0 = p0[k];
                    float v1 = p1[k];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k];
                        v1 *= input_scale_ptr[k];
                        asm volatile(""
                                     : "+f"(v0), "+f"(v1));
                    }
                    pp[r] = float2int8(v0 * scale0);
                    pp[2 + r] = float2int8(v1 * scale1);
                }
                pp += 4;
                kk += 2;
            }
            for (; kk < max_kk; kk++)
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
            float absmax0 = 0.f;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v0 = p0[k];
                if (input_scale_ptr)
                    v0 *= input_scale_ptr[k];
                absmax0 = std::max(absmax0, fabsf(v0));
            }

            volatile double scale0_fp64 = absmax0 == 0.f ? 1.0 : 127.0 / (double)absmax0;
            const float scale0 = (float)scale0_fp64;
            *pd++ = absmax0 / 127.f;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int r = 0; r < 4; r++)
                {
                    const int k = k0 + kk + r;
                    float v0 = p0[k];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k];
                        asm volatile(""
                                     : "+f"(v0));
                    }
                    pp[r] = float2int8(v0 * scale0);
                }
                pp += 4;
            }
            if (kk + 1 < max_kk)
            {
                for (int r = 0; r < 2; r++)
                {
                    const int k = k0 + kk + r;
                    float v0 = p0[k];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k];
                        asm volatile(""
                                     : "+f"(v0));
                    }
                    pp[r] = float2int8(v0 * scale0);
                }
                pp += 2;
                kk += 2;
            }
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v0 = p0[k];
                if (input_scale_ptr)
                {
                    v0 *= input_scale_ptr[k];
                    asm volatile(""
                                 : "+f"(v0));
                }
                *pp++ = float2int8(v0 * scale0);
            }
        }
    }
}

// group-major, row-major within each K4/K2/K1 fragment
static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int block_size, const float* input_scale_ptr)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        transpose_quantize_A_tile_wq_int8_loongson_mmi(A, AT_tile, AT_descales_tile, i, max_ii, block_size, input_scale_ptr);
        return;
    }
#endif

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int K = AT_tile.w;
    const int block_count = AT_descales_tile.w;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const int i0 = i + ii;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                v4f32 _p0 = (v4f32)__msa_ld_w((const float*)A + (size_t)k * A_hstep + i0, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w((const float*)A + (size_t)k * A_hstep + i0 + 4, 0);
                if (input_scale_ptr)
                {
                    const v4f32 _s = __msa_fill_w_f32(input_scale_ptr[k]);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = __msa_fmul_w(_p1, _s);
                }
                _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
            }

            float absmax[8];
            __msa_st_w((v4i32)_absmax0, absmax, 0);
            __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
            volatile double scale0_fp64 = absmax[0] == 0.f ? 1.0 : 127.0 / (double)absmax[0];
            volatile double scale1_fp64 = absmax[1] == 0.f ? 1.0 : 127.0 / (double)absmax[1];
            volatile double scale2_fp64 = absmax[2] == 0.f ? 1.0 : 127.0 / (double)absmax[2];
            volatile double scale3_fp64 = absmax[3] == 0.f ? 1.0 : 127.0 / (double)absmax[3];
            volatile double scale4_fp64 = absmax[4] == 0.f ? 1.0 : 127.0 / (double)absmax[4];
            volatile double scale5_fp64 = absmax[5] == 0.f ? 1.0 : 127.0 / (double)absmax[5];
            volatile double scale6_fp64 = absmax[6] == 0.f ? 1.0 : 127.0 / (double)absmax[6];
            volatile double scale7_fp64 = absmax[7] == 0.f ? 1.0 : 127.0 / (double)absmax[7];
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            const float scale2 = (float)scale2_fp64;
            const float scale3 = (float)scale3_fp64;
            const float scale4 = (float)scale4_fp64;
            const float scale5 = (float)scale5_fp64;
            const float scale6 = (float)scale6_fp64;
            const float scale7 = (float)scale7_fp64;
            pd[0] = absmax[0] / 127.f;
            pd[1] = absmax[1] / 127.f;
            pd[2] = absmax[2] / 127.f;
            pd[3] = absmax[3] / 127.f;
            pd[4] = absmax[4] / 127.f;
            pd[5] = absmax[5] / 127.f;
            pd[6] = absmax[6] / 127.f;
            pd[7] = absmax[7] / 127.f;
            pd += 8;

            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float* p0 = (const float*)A + (size_t)(k0 + kk) * A_hstep + i0;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
                if (input_scale_ptr)
                {
                    _p0 = __msa_fmul_w(_p0, __msa_fill_w_f32(input_scale_ptr[k0 + kk]));
                    _p1 = __msa_fmul_w(_p1, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 1]));
                    _p2 = __msa_fmul_w(_p2, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 2]));
                    _p3 = __msa_fmul_w(_p3, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 3]));
                }
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(__msa_fmul_w(_p0, __msa_fill_w_f32(scale0))), 0);
                ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(__msa_fmul_w(_p1, __msa_fill_w_f32(scale1))), 0);
                ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(__msa_fmul_w(_p2, __msa_fill_w_f32(scale2))), 0);
                ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(__msa_fmul_w(_p3, __msa_fill_w_f32(scale3))), 0);

                _p0 = (v4f32)__msa_ld_w(p0 + 4, 0);
                _p1 = (v4f32)__msa_ld_w(p1 + 4, 0);
                _p2 = (v4f32)__msa_ld_w(p2 + 4, 0);
                _p3 = (v4f32)__msa_ld_w(p3 + 4, 0);
                if (input_scale_ptr)
                {
                    _p0 = __msa_fmul_w(_p0, __msa_fill_w_f32(input_scale_ptr[k0 + kk]));
                    _p1 = __msa_fmul_w(_p1, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 1]));
                    _p2 = __msa_fmul_w(_p2, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 2]));
                    _p3 = __msa_fmul_w(_p3, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 3]));
                }
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                ((int*)pp)[4] = __msa_copy_s_w((v4i32)float2int8(__msa_fmul_w(_p0, __msa_fill_w_f32(scale4))), 0);
                ((int*)pp)[5] = __msa_copy_s_w((v4i32)float2int8(__msa_fmul_w(_p1, __msa_fill_w_f32(scale5))), 0);
                ((int*)pp)[6] = __msa_copy_s_w((v4i32)float2int8(__msa_fmul_w(_p2, __msa_fill_w_f32(scale6))), 0);
                ((int*)pp)[7] = __msa_copy_s_w((v4i32)float2int8(__msa_fmul_w(_p3, __msa_fill_w_f32(scale7))), 0);
                pp += 32;
            }
            if (kk + 1 < max_kk)
            {
                const float* p0 = (const float*)A + (size_t)(k0 + kk) * A_hstep + i0;
                const float* p1 = p0 + A_hstep;
                const float s0 = input_scale_ptr ? input_scale_ptr[k0 + kk] : 1.f;
                const float s1 = input_scale_ptr ? input_scale_ptr[k0 + kk + 1] : 1.f;
                v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), __msa_fill_w_f32(s0));
                v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p1, 0), __msa_fill_w_f32(s1));
                v16i8 _q0 = float2int8(__msa_fmul_w(_p0, (v4f32) {
                    scale0, scale1, scale2, scale3
                }));
                v16i8 _q1 = float2int8(__msa_fmul_w(_p1, (v4f32) {
                    scale0, scale1, scale2, scale3
                }));
                pp[0] = __msa_copy_s_b(_q0, 0);
                pp[1] = __msa_copy_s_b(_q1, 0);
                pp[2] = __msa_copy_s_b(_q0, 1);
                pp[3] = __msa_copy_s_b(_q1, 1);
                pp[4] = __msa_copy_s_b(_q0, 2);
                pp[5] = __msa_copy_s_b(_q1, 2);
                pp[6] = __msa_copy_s_b(_q0, 3);
                pp[7] = __msa_copy_s_b(_q1, 3);
                _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), __msa_fill_w_f32(s0));
                _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p1 + 4, 0), __msa_fill_w_f32(s1));
                _q0 = float2int8(__msa_fmul_w(_p0, (v4f32) {
                    scale4, scale5, scale6, scale7
                }));
                _q1 = float2int8(__msa_fmul_w(_p1, (v4f32) {
                    scale4, scale5, scale6, scale7
                }));
                pp[8] = __msa_copy_s_b(_q0, 0);
                pp[9] = __msa_copy_s_b(_q1, 0);
                pp[10] = __msa_copy_s_b(_q0, 1);
                pp[11] = __msa_copy_s_b(_q1, 1);
                pp[12] = __msa_copy_s_b(_q0, 2);
                pp[13] = __msa_copy_s_b(_q1, 2);
                pp[14] = __msa_copy_s_b(_q0, 3);
                pp[15] = __msa_copy_s_b(_q1, 3);
                pp += 16;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                const float* p0 = (const float*)A + (size_t)k * A_hstep + i0;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), __msa_fill_w_f32(s));
                v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), __msa_fill_w_f32(s));
                const v16i8 _q0 = float2int8(__msa_fmul_w(_p0, (v4f32) {
                    scale0, scale1, scale2, scale3
                }));
                const v16i8 _q1 = float2int8(__msa_fmul_w(_p1, (v4f32) {
                    scale4, scale5, scale6, scale7
                }));
                ((int*)pp)[0] = __msa_copy_s_w((v4i32)_q0, 0);
                ((int*)pp)[1] = __msa_copy_s_w((v4i32)_q1, 0);
                pp += 8;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int i0 = i + ii;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax = (v4f32)__msa_fill_w(0);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                v4f32 _p = (v4f32)__msa_ld_w((const float*)A + (size_t)k * A_hstep + i0, 0);
                if (input_scale_ptr)
                    _p = __msa_fmul_w(_p, __msa_fill_w_f32(input_scale_ptr[k]));
                _absmax = __msa_fmax_w(_absmax, (v4f32)__msa_and_v((v16u8)_p, _abs_mask));
            }

            float absmax[4];
            __msa_st_w((v4i32)_absmax, absmax, 0);
            volatile double scale0_fp64 = absmax[0] == 0.f ? 1.0 : 127.0 / (double)absmax[0];
            volatile double scale1_fp64 = absmax[1] == 0.f ? 1.0 : 127.0 / (double)absmax[1];
            volatile double scale2_fp64 = absmax[2] == 0.f ? 1.0 : 127.0 / (double)absmax[2];
            volatile double scale3_fp64 = absmax[3] == 0.f ? 1.0 : 127.0 / (double)absmax[3];
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            const float scale2 = (float)scale2_fp64;
            const float scale3 = (float)scale3_fp64;
            pd[0] = absmax[0] / 127.f;
            pd[1] = absmax[1] / 127.f;
            pd[2] = absmax[2] / 127.f;
            pd[3] = absmax[3] / 127.f;
            pd += 4;

            const v4f32 _scale0 = __msa_fill_w_f32(scale0);
            const v4f32 _scale1 = __msa_fill_w_f32(scale1);
            const v4f32 _scale2 = __msa_fill_w_f32(scale2);
            const v4f32 _scale3 = __msa_fill_w_f32(scale3);
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float* p0 = (const float*)A + (size_t)(k0 + kk) * A_hstep + i0;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
                if (input_scale_ptr)
                {
                    _p0 = __msa_fmul_w(_p0, __msa_fill_w_f32(input_scale_ptr[k0 + kk]));
                    _p1 = __msa_fmul_w(_p1, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 1]));
                    _p2 = __msa_fmul_w(_p2, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 2]));
                    _p3 = __msa_fmul_w(_p3, __msa_fill_w_f32(input_scale_ptr[k0 + kk + 3]));
                }
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                _p0 = __msa_fmul_w(_p0, _scale0);
                _p1 = __msa_fmul_w(_p1, _scale1);
                _p2 = __msa_fmul_w(_p2, _scale2);
                _p3 = __msa_fmul_w(_p3, _scale3);
                ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p0), 0);
                ((int*)pp)[1] = __msa_copy_s_w((v4i32)float2int8(_p1), 0);
                ((int*)pp)[2] = __msa_copy_s_w((v4i32)float2int8(_p2), 0);
                ((int*)pp)[3] = __msa_copy_s_w((v4i32)float2int8(_p3), 0);
                pp += 16;
            }
            if (kk + 1 < max_kk)
            {
                const float* p0 = (const float*)A + (size_t)(k0 + kk) * A_hstep + i0;
                const float* p1 = p0 + A_hstep;
                float v00 = p0[0];
                float v10 = p0[1];
                float v20 = p0[2];
                float v30 = p0[3];
                float v01 = p1[0];
                float v11 = p1[1];
                float v21 = p1[2];
                float v31 = p1[3];
                if (input_scale_ptr)
                {
                    const float s0 = input_scale_ptr[k0 + kk];
                    const float s1 = input_scale_ptr[k0 + kk + 1];
                    v00 *= s0;
                    v10 *= s0;
                    v20 *= s0;
                    v30 *= s0;
                    v01 *= s1;
                    v11 *= s1;
                    v21 *= s1;
                    v31 *= s1;
                    asm volatile(""
                                 : "+f"(v00), "+f"(v01), "+f"(v10), "+f"(v11));
                    asm volatile(""
                                 : "+f"(v20), "+f"(v21), "+f"(v30), "+f"(v31));
                }
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v10 * scale1);
                pp[3] = float2int8(v11 * scale1);
                pp[4] = float2int8(v20 * scale2);
                pp[5] = float2int8(v21 * scale2);
                pp[6] = float2int8(v30 * scale3);
                pp[7] = float2int8(v31 * scale3);
                pp += 8;
                kk += 2;
            }
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                const float* p0 = (const float*)A + (size_t)k * A_hstep + i0;
                float v0 = p0[0];
                float v1 = p0[1];
                float v2 = p0[2];
                float v3 = p0[3];
                if (input_scale_ptr)
                {
                    const float s = input_scale_ptr[k];
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    asm volatile(""
                                 : "+f"(v0), "+f"(v1), "+f"(v2), "+f"(v3));
                }
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
        const int i0 = i + ii;
        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
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

            volatile double scale0_fp64 = absmax0 == 0.f ? 1.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 1.0 : 127.0 / (double)absmax1;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int r = 0; r < 4; r++)
                {
                    const int k = k0 + kk + r;
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
                    pp[r] = float2int8(v0 * scale0);
                    pp[4 + r] = float2int8(v1 * scale1);
                }
                pp += 8;
            }
            if (kk + 1 < max_kk)
            {
                for (int r = 0; r < 2; r++)
                {
                    const int k = k0 + kk + r;
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
                    pp[r] = float2int8(v0 * scale0);
                    pp[2 + r] = float2int8(v1 * scale1);
                }
                pp += 4;
                kk += 2;
            }
            for (; kk < max_kk; kk++)
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
        }
    }
    for (; ii < max_ii; ii++)
    {
        const int i0 = i + ii;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax0 = 0.f;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v0 = ((const float*)A)[(size_t)k * A_hstep + i0];
                if (input_scale_ptr)
                    v0 *= input_scale_ptr[k];
                absmax0 = std::max(absmax0, fabsf(v0));
            }

            volatile double scale0_fp64 = absmax0 == 0.f ? 1.0 : 127.0 / (double)absmax0;
            const float scale0 = (float)scale0_fp64;
            *pd++ = absmax0 / 127.f;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int r = 0; r < 4; r++)
                {
                    const int k = k0 + kk + r;
                    float v0 = ((const float*)A)[(size_t)k * A_hstep + i0];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k];
                        asm volatile(""
                                     : "+f"(v0));
                    }
                    pp[r] = float2int8(v0 * scale0);
                }
                pp += 4;
            }
            if (kk + 1 < max_kk)
            {
                for (int r = 0; r < 2; r++)
                {
                    const int k = k0 + kk + r;
                    float v0 = ((const float*)A)[(size_t)k * A_hstep + i0];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k];
                        asm volatile(""
                                     : "+f"(v0));
                    }
                    pp[r] = float2int8(v0 * scale0);
                }
                pp += 2;
                kk += 2;
            }
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v0 = ((const float*)A)[(size_t)k * A_hstep + i0];
                if (input_scale_ptr)
                {
                    v0 *= input_scale_ptr[k];
                    asm volatile(""
                                 : "+f"(v0));
                }
                *pp++ = float2int8(v0 * scale0);
            }
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        gemm_transB_packed_tile_wq_int8_loongson_mmi(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, block_size);
        return;
    }
#endif

    const signed char* pAT = AT_tile;
    const int A_hstep = AT_tile.w;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = AT_descales_tile.w;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int K = AT_tile.w;
    const int num_blocks = (K + block_size - 1) / block_size;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;
        const float* pBD = pBT_descales;
        const v8i16 _one = __msa_fill_h(1);

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum2 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum3 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum4 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum5 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum6 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum7 = (v4f32)__msa_fill_w(0);

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                v4i32 _sum4 = __msa_fill_w(0);
                v4i32 _sum5 = __msa_fill_w(0);
                v4i32 _sum6 = __msa_fill_w(0);
                v4i32 _sum7 = __msa_fill_w(0);

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 64);
                    const v16i8 _pA0 = __msa_ld_b(pA, 0);
                    const v16i8 _pA0r = (v16i8)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                    const v16i8 _pB = __msa_ld_b(pB, 0);
                    const v16i8 _pBr = (v16i8)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA0, _pBr), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA0r, _pB), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA0r, _pBr), _one);

                    const v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    const v16i8 _pA1r = (v16i8)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                    _sum4 = __msa_dpadd_s_w(_sum4, __msa_dotp_s_h(_pA1, _pB), _one);
                    _sum5 = __msa_dpadd_s_w(_sum5, __msa_dotp_s_h(_pA1, _pBr), _one);
                    _sum6 = __msa_dpadd_s_w(_sum6, __msa_dotp_s_h(_pA1r, _pB), _one);
                    _sum7 = __msa_dpadd_s_w(_sum7, __msa_dotp_s_h(_pA1r, _pBr), _one);
                    pA += 32;
                    pB += 16;
                }

                _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(1, 0, 3, 2));
                transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                _sum1 = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(0, 3, 2, 1));
                _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(1, 0, 3, 2));
                transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
                _sum5 = __msa_shf_w(_sum5, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(0, 3, 2, 1));

                if (kk + 1 < max_kk)
                {
                    const v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    const v8i16 _pA = (v8i16)__msa_ld_b(pA, 0);
                    v8i16 _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 0), _pB);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 1), _pB);
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 2), _pB);
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 3), _pB);
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 4), _pB);
                    _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 5), _pB);
                    _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 6), _pB);
                    _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 7), _pB);
                    _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 16;
                    pB += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    const v16i8 _pA8 = (v16i8)__msa_fill_d_ptr(pA);
                    const v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    v8i16 _pB = (v8i16)__msa_fill_w(*(const int*)pB);
                    _pB = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB, 0), (v16i8)_pB);
                    v8i16 _s = __msa_mulv_h(__msa_splati_h(_pA, 0), _pB);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 1), _pB);
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 2), _pB);
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 3), _pB);
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 4), _pB);
                    _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 5), _pB);
                    _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 6), _pB);
                    _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 7), _pB);
                    _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 8;
                    pB += 4;
                }

                const v4f32 _descaleB = (v4f32)__msa_ld_w(pBD, 0);
                const v4f32 _descaleA0 = (v4f32)__msa_ld_w(pAD, 0);
                const v4f32 _descaleA1 = (v4f32)__msa_ld_w(pAD + 4, 0);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA0, 0))));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1), __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA0, 1))));
                _fsum2 = __msa_fadd_w(_fsum2, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum2), __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA0, 2))));
                _fsum3 = __msa_fadd_w(_fsum3, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum3), __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA0, 3))));
                _fsum4 = __msa_fadd_w(_fsum4, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum4), __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA1, 0))));
                _fsum5 = __msa_fadd_w(_fsum5, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum5), __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA1, 1))));
                _fsum6 = __msa_fadd_w(_fsum6, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum6), __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA1, 2))));
                _fsum7 = __msa_fadd_w(_fsum7, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum7), __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA1, 3))));
                pAD += 8;
                pBD += 4;
            }

            transpose4x4_ps(_fsum0, _fsum1, _fsum2, _fsum3);
            transpose4x4_ps(_fsum4, _fsum5, _fsum6, _fsum7);
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum4, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum5, outptr + 12, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 16, 0);
            __msa_st_w((v4i32)_fsum6, outptr + 20, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 24, 0);
            __msa_st_w((v4i32)_fsum7, outptr + 28, 0);
            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum2 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum3 = (v4f32)__msa_fill_w(0);

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA0 = __msa_ld_b(pA, 0);
                    const v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    const v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    const v16i8 _pBr = (v16i8)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA0, _pBr), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA1, _pB), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA1, _pBr), _one);
                    pA += 32;
                    pB += 8;
                }

                const v4i32 _sum0e = __msa_shf_w(_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
                const v4i32 _sum0o = __msa_shf_w(_sum0, _MSA_SHUFFLE(2, 0, 3, 1));
                const v4i32 _sum1e = __msa_shf_w(_sum1, _MSA_SHUFFLE(3, 1, 2, 0));
                const v4i32 _sum1o = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum0x = (v4i32)__msa_ilvr_w(_sum1o, _sum0e);
                v4i32 _sum1x = (v4i32)__msa_ilvr_w(_sum0o, _sum1e);
                const v4i32 _sum2e = __msa_shf_w(_sum2, _MSA_SHUFFLE(3, 1, 2, 0));
                const v4i32 _sum2o = __msa_shf_w(_sum2, _MSA_SHUFFLE(2, 0, 3, 1));
                const v4i32 _sum3e = __msa_shf_w(_sum3, _MSA_SHUFFLE(3, 1, 2, 0));
                const v4i32 _sum3o = __msa_shf_w(_sum3, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum2x = (v4i32)__msa_ilvr_w(_sum3o, _sum2e);
                v4i32 _sum3x = (v4i32)__msa_ilvr_w(_sum2o, _sum3e);

                if (kk + 1 < max_kk)
                {
                    const v16i8 _pA = __msa_ld_b(pA, 0);
                    const v8i16 _pB = (v8i16)__msa_fill_w(*(const int*)pB);
                    const v8i16 _s0 = __msa_dotp_s_h(_pA, (v16i8)__msa_splati_h(_pB, 0));
                    const v8i16 _s1 = __msa_dotp_s_h(_pA, (v16i8)__msa_splati_h(_pB, 1));
                    const v8i16 _sign0 = __msa_clti_s_h(_s0, 0);
                    const v8i16 _sign1 = __msa_clti_s_h(_s1, 0);
                    _sum0x = __msa_addv_w(_sum0x, (v4i32)__msa_ilvr_h(_sign0, _s0));
                    _sum1x = __msa_addv_w(_sum1x, (v4i32)__msa_ilvr_h(_sign1, _s1));
                    _sum2x = __msa_addv_w(_sum2x, (v4i32)__msa_ilvl_h(_sign0, _s0));
                    _sum3x = __msa_addv_w(_sum3x, (v4i32)__msa_ilvl_h(_sign1, _s1));
                    pA += 16;
                    pB += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    const v16i8 _pA8 = (v16i8)__msa_fill_d_ptr(pA);
                    const v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    const v16i8 _pB8 = (v16i8)__msa_fill_h(*(const short*)pB);
                    const v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(__msa_splati_b(_pB8, 0), 0), __msa_splati_b(_pB8, 0));
                    const v8i16 _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(__msa_splati_b(_pB8, 1), 0), __msa_splati_b(_pB8, 1));
                    const v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    const v8i16 _s1 = __msa_mulv_h(_pA, _pB1);
                    const v8i16 _sign0 = __msa_clti_s_h(_s0, 0);
                    const v8i16 _sign1 = __msa_clti_s_h(_s1, 0);
                    _sum0x = __msa_addv_w(_sum0x, (v4i32)__msa_ilvr_h(_sign0, _s0));
                    _sum1x = __msa_addv_w(_sum1x, (v4i32)__msa_ilvr_h(_sign1, _s1));
                    _sum2x = __msa_addv_w(_sum2x, (v4i32)__msa_ilvl_h(_sign0, _s0));
                    _sum3x = __msa_addv_w(_sum3x, (v4i32)__msa_ilvl_h(_sign1, _s1));
                    pA += 8;
                    pB += 2;
                }

                const v4f32 _descaleA0 = (v4f32)__msa_ld_w(pAD, 0);
                const v4f32 _descaleA1 = (v4f32)__msa_ld_w(pAD + 4, 0);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0x), __msa_fmul_w(_descaleA0, __msa_fill_w_f32(pBD[0]))));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum2x), __msa_fmul_w(_descaleA1, __msa_fill_w_f32(pBD[0]))));
                _fsum2 = __msa_fadd_w(_fsum2, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1x), __msa_fmul_w(_descaleA0, __msa_fill_w_f32(pBD[1]))));
                _fsum3 = __msa_fadd_w(_fsum3, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum3x), __msa_fmul_w(_descaleA1, __msa_fill_w_f32(pBD[1]))));
                pAD += 8;
                pBD += 2;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            outptr += 16;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA0 = __msa_ld_b(pA, 0);
                    const v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    const v16i8 _pB = (v16i8)__msa_fill_w(*(const int*)pB);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA1, _pB), _one);
                    pA += 32;
                    pB += 4;
                }
                if (kk + 1 < max_kk)
                {
                    const v16i8 _pA = __msa_ld_b(pA, 0);
                    const v16i8 _pB = (v16i8)__msa_fill_h(*(const short*)pB);
                    const v8i16 _s = __msa_dotp_s_h(_pA, _pB);
                    const v8i16 _sign = __msa_clti_s_h(_s, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s));
                    pA += 16;
                    pB += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    const v16i8 _pA8 = (v16i8)__msa_fill_d_ptr(pA);
                    const v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    const v8i16 _s = __msa_mulv_h(_pA, __msa_fill_h(pB[0]));
                    const v8i16 _sign = __msa_clti_s_h(_s, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s));
                    pA += 8;
                    pB++;
                }

                const v4f32 _descaleA0 = (v4f32)__msa_ld_w(pAD, 0);
                const v4f32 _descaleA1 = (v4f32)__msa_ld_w(pAD + 4, 0);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleA0, __msa_fill_w_f32(pBD[0]))));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1), __msa_fmul_w(_descaleA1, __msa_fill_w_f32(pBD[0]))));
                pAD += 8;
                pBD++;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
        }

        pAT += (size_t)8 * A_hstep;
        pAT_descales += (size_t)8 * A_descales_hstep;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;
        const float* pBD = pBT_descales;
        const v8i16 _one = __msa_fill_h(1);

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + (size_t)4 * K;
            const float* pBD0 = pBD;
            const float* pBD1 = pBD + (size_t)4 * num_blocks;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum2 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum3 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum4 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum5 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum6 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum7 = (v4f32)__msa_fill_w(0);

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                v4i32 _sum4 = __msa_fill_w(0);
                v4i32 _sum5 = __msa_fill_w(0);
                v4i32 _sum6 = __msa_fill_w(0);
                v4i32 _sum7 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA = __msa_ld_b(pA, 0);
                    const v16i8 _pAr = (v16i8)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    const v16i8 _pB0 = __msa_ld_b(pB0, 0);
                    const v16i8 _pB1 = __msa_ld_b(pB1, 0);
                    const v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    const v16i8 _pB1r = (v16i8)__msa_shf_w((v4i32)_pB1, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pAr, _pB0), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pAr, _pB0r), _one);
                    _sum4 = __msa_dpadd_s_w(_sum4, __msa_dotp_s_h(_pA, _pB1), _one);
                    _sum5 = __msa_dpadd_s_w(_sum5, __msa_dotp_s_h(_pA, _pB1r), _one);
                    _sum6 = __msa_dpadd_s_w(_sum6, __msa_dotp_s_h(_pAr, _pB1), _one);
                    _sum7 = __msa_dpadd_s_w(_sum7, __msa_dotp_s_h(_pAr, _pB1r), _one);
                    pA += 16;
                    pB0 += 16;
                    pB1 += 16;
                }

                const signed char* pA2 = pA;
                const signed char* pB02 = pB0;
                const signed char* pB12 = pB1;
                const bool has_k2 = kk + 1 < max_kk;
                if (kk + 1 < max_kk)
                {
                    pA += 8;
                    pB0 += 8;
                    pB1 += 8;
                    kk += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    const v8i16 _pAr = __msa_shf_h(_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    v8i16 _pB0 = (v8i16)__msa_fill_w(*(const int*)pB0);
                    v8i16 _pB1 = (v8i16)__msa_fill_w(*(const int*)pB1);
                    _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                    _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB1, 0), (v16i8)_pB1);
                    const v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    const v8i16 _pB1r = __msa_shf_h(_pB1, _MSA_SHUFFLE(0, 3, 2, 1));
                    const v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    const v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                    const v8i16 _s2 = __msa_mulv_h(_pAr, _pB0);
                    const v8i16 _s3 = __msa_mulv_h(_pAr, _pB0r);
                    const v8i16 _s4 = __msa_mulv_h(_pA, _pB1);
                    const v8i16 _s5 = __msa_mulv_h(_pA, _pB1r);
                    const v8i16 _s6 = __msa_mulv_h(_pAr, _pB1);
                    const v8i16 _s7 = __msa_mulv_h(_pAr, _pB1r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                    _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s4, 0), _s4));
                    _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s5, 0), _s5));
                    _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s6, 0), _s6));
                    _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s7, 0), _s7));
                    pA += 4;
                    pB0 += 4;
                    pB1 += 4;
                }

                _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(1, 0, 3, 2));
                transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                _sum1 = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(0, 3, 2, 1));
                _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(1, 0, 3, 2));
                transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
                _sum5 = __msa_shf_w(_sum5, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(0, 3, 2, 1));
                if (has_k2)
                {
                    const v8i16 _pA = (v8i16)__msa_fill_d_ptr(pA2);
                    const v16i8 _pB0 = (v16i8)__msa_fill_d_ptr(pB02);
                    const v16i8 _pB1 = (v16i8)__msa_fill_d_ptr(pB12);
                    v8i16 _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 0), _pB0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 1), _pB0);
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 2), _pB0);
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 3), _pB0);
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 0), _pB1);
                    _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 1), _pB1);
                    _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 2), _pB1);
                    _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_dotp_s_h((v16i8)__msa_splati_h(_pA, 3), _pB1);
                    _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                }

                const v4f32 _descaleB0 = (v4f32)__msa_ld_w(pBD0, 0);
                const v4f32 _descaleB1 = (v4f32)__msa_ld_w(pBD1, 0);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleB0, __msa_fill_w_f32(pAD[0]))));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1), __msa_fmul_w(_descaleB0, __msa_fill_w_f32(pAD[1]))));
                _fsum2 = __msa_fadd_w(_fsum2, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum2), __msa_fmul_w(_descaleB0, __msa_fill_w_f32(pAD[2]))));
                _fsum3 = __msa_fadd_w(_fsum3, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum3), __msa_fmul_w(_descaleB0, __msa_fill_w_f32(pAD[3]))));
                _fsum4 = __msa_fadd_w(_fsum4, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum4), __msa_fmul_w(_descaleB1, __msa_fill_w_f32(pAD[0]))));
                _fsum5 = __msa_fadd_w(_fsum5, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum5), __msa_fmul_w(_descaleB1, __msa_fill_w_f32(pAD[1]))));
                _fsum6 = __msa_fadd_w(_fsum6, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum6), __msa_fmul_w(_descaleB1, __msa_fill_w_f32(pAD[2]))));
                _fsum7 = __msa_fadd_w(_fsum7, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum7), __msa_fmul_w(_descaleB1, __msa_fill_w_f32(pAD[3]))));
                pAD += 4;
                pBD0 += 4;
                pBD1 += 4;
            }

            transpose4x4_ps(_fsum0, _fsum1, _fsum2, _fsum3);
            transpose4x4_ps(_fsum4, _fsum5, _fsum6, _fsum7);
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            __msa_st_w((v4i32)_fsum4, outptr + 16, 0);
            __msa_st_w((v4i32)_fsum5, outptr + 20, 0);
            __msa_st_w((v4i32)_fsum6, outptr + 24, 0);
            __msa_st_w((v4i32)_fsum7, outptr + 28, 0);
            outptr += 32;
            pB = pB1;
            pBD = pBD1;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum2 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum3 = (v4f32)__msa_fill_w(0);

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA = __msa_ld_b(pA, 0);
                    const v16i8 _pAr = (v16i8)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    const v16i8 _pB0 = __msa_ld_b(pB, 0);
                    const v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pAr, _pB0), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pAr, _pB0r), _one);
                    pA += 16;
                    pB += 16;
                }
                v8i16 _sum2_0 = __msa_fill_h(0);
                v8i16 _sum2_1 = __msa_fill_h(0);
                v8i16 _sum2_2 = __msa_fill_h(0);
                v8i16 _sum2_3 = __msa_fill_h(0);
                if (kk + 1 < max_kk)
                {
                    const v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    const v8i16 _pA = (v8i16)__msa_fill_d_ptr(pA);
                    const v16i8 _pA0 = (v16i8)__msa_splati_h(_pA, 0);
                    const v16i8 _pA1 = (v16i8)__msa_splati_h(_pA, 1);
                    const v16i8 _pA2 = (v16i8)__msa_splati_h(_pA, 2);
                    const v16i8 _pA3 = (v16i8)__msa_splati_h(_pA, 3);
                    _sum2_0 = __msa_dotp_s_h(_pA0, _pB);
                    _sum2_1 = __msa_dotp_s_h(_pA1, _pB);
                    _sum2_2 = __msa_dotp_s_h(_pA2, _pB);
                    _sum2_3 = __msa_dotp_s_h(_pA3, _pB);
                    pA += 8;
                    pB += 8;
                    kk += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    const v8i16 _pAr = __msa_shf_h(_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    v8i16 _pB0 = (v8i16)__msa_fill_w(*(const int*)pB);
                    _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                    const v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    const v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    const v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                    const v8i16 _s2 = __msa_mulv_h(_pAr, _pB0);
                    const v8i16 _s3 = __msa_mulv_h(_pAr, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                    pA += 4;
                    pB += 4;
                }
                _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(1, 0, 3, 2));
                transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                _sum1 = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
                _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(0, 3, 2, 1));
                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_0, 0), _sum2_0));
                _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_1, 0), _sum2_1));
                _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_2, 0), _sum2_2));
                _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_3, 0), _sum2_3));
                const v4f32 _descaleB = (v4f32)__msa_ld_w(pBD, 0);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleB, __msa_fill_w_f32(pAD[0]))));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1), __msa_fmul_w(_descaleB, __msa_fill_w_f32(pAD[1]))));
                _fsum2 = __msa_fadd_w(_fsum2, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum2), __msa_fmul_w(_descaleB, __msa_fill_w_f32(pAD[2]))));
                _fsum3 = __msa_fadd_w(_fsum3, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum3), __msa_fmul_w(_descaleB, __msa_fill_w_f32(pAD[3]))));
                pAD += 4;
                pBD += 4;
            }
            transpose4x4_ps(_fsum0, _fsum1, _fsum2, _fsum3);
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA = __msa_ld_b(pA, 0);
                    const v16i8 _pB0 = (v16i8)__msa_fill_d_ptr(pB);
                    const v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                    pA += 16;
                    pB += 8;
                }
                v8i16 _sum2_0 = __msa_fill_h(0);
                v8i16 _sum2_1 = __msa_fill_h(0);
                if (kk + 1 < max_kk)
                {
                    const v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    const v8i16 _pB = (v8i16)__msa_fill_w(*(const int*)pB);
                    const v16i8 _pB0 = (v16i8)__msa_splati_h(_pB, 0);
                    const v16i8 _pB1 = (v16i8)__msa_splati_h(_pB, 1);
                    _sum2_0 = __msa_dotp_s_h(_pA, _pB0);
                    _sum2_1 = __msa_dotp_s_h(_pA, _pB1);
                    pA += 8;
                    pB += 4;
                    kk += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    const v16i8 _pB8 = (v16i8)__msa_fill_h(*(const short*)pB);
                    const v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    const v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    const v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    const v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    pA += 4;
                    pB += 2;
                }
                const v4i32 _sum0e = __msa_shf_w(_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
                const v4i32 _sum0o = __msa_shf_w(_sum0, _MSA_SHUFFLE(2, 0, 3, 1));
                const v4i32 _sum1e = __msa_shf_w(_sum1, _MSA_SHUFFLE(3, 1, 2, 0));
                const v4i32 _sum1o = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum0x = (v4i32)__msa_ilvr_w(_sum1o, _sum0e);
                v4i32 _sum1x = (v4i32)__msa_ilvr_w(_sum0o, _sum1e);
                _sum0x = __msa_addv_w(_sum0x, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_0, 0), _sum2_0));
                _sum1x = __msa_addv_w(_sum1x, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_1, 0), _sum2_1));
                const v4f32 _descaleA = (v4f32)__msa_ld_w(pAD, 0);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0x), __msa_fmul_w(_descaleA, __msa_fill_w_f32(pBD[0]))));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1x), __msa_fmul_w(_descaleA, __msa_fill_w_f32(pBD[1]))));
                pAD += 4;
                pBD += 2;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA = __msa_ld_b(pA, 0);
                    const v16i8 _pB0 = (v16i8)__msa_fill_w(*(const int*)pB);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    pA += 16;
                    pB += 4;
                }
                v8i16 _sum2_0 = __msa_fill_h(0);
                if (kk + 1 < max_kk)
                {
                    const v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    const v16i8 _pB = (v16i8)__msa_fill_h(*(const short*)pB);
                    _sum2_0 = __msa_dotp_s_h(_pA, _pB);
                    pA += 8;
                    pB += 2;
                    kk += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    const v8i16 _pB0 = __msa_fill_h(pB[0]);
                    const v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    pA += 4;
                    pB++;
                }
                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_0, 0), _sum2_0));
                const v4f32 _descaleA = (v4f32)__msa_ld_w(pAD, 0);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleA, __msa_fill_w_f32(pBD[0]))));
                pAD += 4;
                pBD++;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            outptr += 4;
        }

        pAT += (size_t)4 * A_hstep;
        pAT_descales += (size_t)4 * A_descales_hstep;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;
        const float* pBD = pBT_descales;

        int jj = 0;
#if __mips_msa
        const v8i16 _one = __msa_fill_h(1);
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + (size_t)4 * K;
            const float* pBD0 = pBD;
            const float* pBD1 = pBD + (size_t)4 * num_blocks;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum2 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum3 = (v4f32)__msa_fill_w(0);
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    const v16i8 _pB0 = __msa_ld_b(pB0, 0);
                    const v16i8 _pB00 = (v16i8)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                    const v16i8 _pB01 = (v16i8)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);
                    const v16i8 _pB1 = __msa_ld_b(pB1, 0);
                    const v16i8 _pB10 = (v16i8)__msa_ilvr_w((v4i32)_pB1, (v4i32)_pB1);
                    const v16i8 _pB11 = (v16i8)__msa_ilvl_w((v4i32)_pB1, (v4i32)_pB1);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB00), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB01), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA, _pB10), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA, _pB11), _one);
                    pA += 8;
                    pB0 += 16;
                    pB1 += 16;
                }
                if (kk + 1 < max_kk)
                {
                    const v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    const v16i8 _pA0 = (v16i8)__msa_splati_h(_pA, 0);
                    const v16i8 _pA1 = (v16i8)__msa_splati_h(_pA, 1);
                    const v16i8 _pB0 = (v16i8)__msa_fill_d_ptr(pB0);
                    const v16i8 _pB1 = (v16i8)__msa_fill_d_ptr(pB1);
                    const v8i16 _s00 = __msa_dotp_s_h(_pA0, _pB0);
                    const v8i16 _s01 = __msa_dotp_s_h(_pA1, _pB0);
                    const v8i16 _s10 = __msa_dotp_s_h(_pA0, _pB1);
                    const v8i16 _s11 = __msa_dotp_s_h(_pA1, _pB1);
                    const v8i16 _s0 = (v8i16)__msa_ilvr_h(_s01, _s00);
                    const v8i16 _s2 = (v8i16)__msa_ilvr_h(_s11, _s10);
                    const v8i16 _sign0 = __msa_clti_s_h(_s0, 0);
                    const v8i16 _sign2 = __msa_clti_s_h(_s2, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign0, _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign0, _s0));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(_sign2, _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvl_h(_sign2, _s2));
                    pA += 4;
                    pB0 += 8;
                    pB1 += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    const v16i8 _pA8 = (v16i8)__msa_fill_h(*(const short*)pA);
                    const v16i8 _pA0b = __msa_splati_b(_pA8, 0);
                    const v16i8 _pA1b = __msa_splati_b(_pA8, 1);
                    const v8i16 _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA0b, 0), _pA0b);
                    const v8i16 _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA1b, 0), _pA1b);
                    const v16i8 _pB08 = (v16i8)__msa_fill_w(*(const int*)pB0);
                    const v16i8 _pB18 = (v16i8)__msa_fill_w(*(const int*)pB1);
                    const v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB08, 0), _pB08);
                    const v8i16 _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB18, 0), _pB18);
                    const v8i16 _s00 = __msa_mulv_h(_pA0, _pB0);
                    const v8i16 _s01 = __msa_mulv_h(_pA1, _pB0);
                    const v8i16 _s10 = __msa_mulv_h(_pA0, _pB1);
                    const v8i16 _s11 = __msa_mulv_h(_pA1, _pB1);
                    const v8i16 _s0 = (v8i16)__msa_ilvr_h(_s01, _s00);
                    const v8i16 _s2 = (v8i16)__msa_ilvr_h(_s11, _s10);
                    const v8i16 _sign0 = __msa_clti_s_h(_s0, 0);
                    const v8i16 _sign2 = __msa_clti_s_h(_s2, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign0, _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign0, _s0));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(_sign2, _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvl_h(_sign2, _s2));
                    pA += 2;
                    pB0 += 4;
                    pB1 += 4;
                }
                const v4f32 _descaleA = (v4f32) {
                    pAD[0], pAD[1], pAD[0], pAD[1]
                };
                const v4f32 _descaleB0 = (v4f32) {
                    pBD0[0], pBD0[0], pBD0[1], pBD0[1]
                };
                const v4f32 _descaleB1 = (v4f32) {
                    pBD0[2], pBD0[2], pBD0[3], pBD0[3]
                };
                const v4f32 _descaleB2 = (v4f32) {
                    pBD1[0], pBD1[0], pBD1[1], pBD1[1]
                };
                const v4f32 _descaleB3 = (v4f32) {
                    pBD1[2], pBD1[2], pBD1[3], pBD1[3]
                };
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleA, _descaleB0)));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1), __msa_fmul_w(_descaleA, _descaleB1)));
                _fsum2 = __msa_fadd_w(_fsum2, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum2), __msa_fmul_w(_descaleA, _descaleB2)));
                _fsum3 = __msa_fadd_w(_fsum3, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum3), __msa_fmul_w(_descaleA, _descaleB3)));
                pAD += 2;
                pBD0 += 4;
                pBD1 += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            outptr += 16;
            pB = pB1;
            pBD = pBD1;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    const v16i8 _pB0 = __msa_ld_b(pB, 0);
                    const v16i8 _pB01 = (v16i8)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                    const v16i8 _pB23 = (v16i8)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB01), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB23), _one);
                    pA += 8;
                    pB += 16;
                }
                if (kk + 1 < max_kk)
                {
                    const v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    const v16i8 _pA0 = (v16i8)__msa_splati_h(_pA, 0);
                    const v16i8 _pA1 = (v16i8)__msa_splati_h(_pA, 1);
                    const v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    const v8i16 _s00 = __msa_dotp_s_h(_pA0, _pB);
                    const v8i16 _s01 = __msa_dotp_s_h(_pA1, _pB);
                    const v8i16 _s0 = (v8i16)__msa_ilvr_h(_s01, _s00);
                    const v8i16 _sign = __msa_clti_s_h(_s0, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s0));
                    pA += 4;
                    pB += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    const v16i8 _pA8 = (v16i8)__msa_fill_h(*(const short*)pA);
                    const v16i8 _pA0b = __msa_splati_b(_pA8, 0);
                    const v16i8 _pA1b = __msa_splati_b(_pA8, 1);
                    const v8i16 _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA0b, 0), _pA0b);
                    const v8i16 _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA1b, 0), _pA1b);
                    const v16i8 _pB8 = (v16i8)__msa_fill_w(*(const int*)pB);
                    const v8i16 _pB = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    const v8i16 _s00 = __msa_mulv_h(_pA0, _pB);
                    const v8i16 _s01 = __msa_mulv_h(_pA1, _pB);
                    const v8i16 _s0 = (v8i16)__msa_ilvr_h(_s01, _s00);
                    const v8i16 _sign = __msa_clti_s_h(_s0, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s0));
                    pA += 2;
                    pB += 4;
                }
                const v4f32 _descaleA = (v4f32) {
                    pAD[0], pAD[1], pAD[0], pAD[1]
                };
                const v4f32 _descaleB0 = (v4f32) {
                    pBD[0], pBD[0], pBD[1], pBD[1]
                };
                const v4f32 _descaleB1 = (v4f32) {
                    pBD[2], pBD[2], pBD[3], pBD[3]
                };
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleA, _descaleB0)));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1), __msa_fmul_w(_descaleA, _descaleB1)));
                pAD += 2;
                pBD += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            float sum00 = 0.f;
            float sum01 = 0.f;
            float sum10 = 0.f;
            float sum11 = 0.f;

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int sum00_i = 0;
                int sum01_i = 0;
                int sum10_i = 0;
                int sum11_i = 0;
                int kk = 0;
#if __mips_loongson_mmi
                int32x2_t _sum00 = __mmi_pzerow_s();
                int32x2_t _sum01 = __mmi_pzerow_s();
                int32x2_t _sum10 = __mmi_pzerow_s();
                int32x2_t _sum11 = __mmi_pzerow_s();
#if NCNN_GNU_INLINE_ASM
                double _tmp0;
                double _tmp1;
                double _tmp2;
                double _tmp3;
                double _tmp4;
                double _tmp5;
                double _tmp6;
                double _tmp7;
                double _shift;
                const int shift_8 = 8;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(
                        "ld         $0, 32(%1)      \n"
                        "ldc1       %6, 0(%0)       \n"
                        "ldc1       %8, 0(%1)       \n"
#if __mips64
                        "dmtc1      $0, %7          \n"
#else
                        "mtc1       $0, %7          \n"
#endif
                        "mtc1       %21, %14        \n"
#if __mips64
                        "daddiu     %0, %0, 8       \n"
                        "daddiu     %1, %1, 8       \n"
#else
                        "addiu      %0, %0, 8       \n"
                        "addiu      %1, %1, 8       \n"
#endif
                        "punpcklbh  %10, %6, %7     \n"
                        "punpckhbh  %11, %6, %7     \n"
                        "punpcklbh  %12, %8, %7     \n"
                        "punpckhbh  %13, %8, %7     \n"
                        "psllh      %10, %10, %14   \n"
                        "psllh      %11, %11, %14   \n"
                        "psllh      %12, %12, %14   \n"
                        "psllh      %13, %13, %14   \n"
                        "psrah      %10, %10, %14   \n"
                        "psrah      %11, %11, %14   \n"
                        "psrah      %12, %12, %14   \n"
                        "psrah      %13, %13, %14   \n"
                        "pmaddhw    %6, %10, %12    \n"
                        "pmaddhw    %7, %11, %12    \n"
                        "pmaddhw    %8, %10, %13    \n"
                        "pmaddhw    %9, %11, %13    \n"
                        "paddw      %2, %2, %6      \n"
                        "paddw      %3, %3, %7      \n"
                        "paddw      %4, %4, %8      \n"
                        "paddw      %5, %5, %9      \n"
                        : "=r"(pA),
                        "=r"(pB),
                        "=f"(_sum00),
                        "=f"(_sum01),
                        "=f"(_sum10),
                        "=f"(_sum11),
                        "=&f"(_tmp0),
                        "=&f"(_tmp1),
                        "=&f"(_tmp2),
                        "=&f"(_tmp3),
                        "=&f"(_tmp4),
                        "=&f"(_tmp5),
                        "=&f"(_tmp6),
                        "=&f"(_tmp7),
                        "=&f"(_shift)
                        : "0"(pA),
                        "1"(pB),
                        "2"(_sum00),
                        "3"(_sum01),
                        "4"(_sum10),
                        "5"(_sum11),
                        "r"(shift_8)
                        : "memory");
                }
#else  // NCNN_GNU_INLINE_ASM
                const int8x8_t _zero = __mmi_pzerob_s();
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __builtin_prefetch(pB + 32);
                    const int8x8_t _pA = __mmi_pldb_s(pA);
                    const int8x8_t _pB = __mmi_pldb_s(pB);
                    int16x4_t _pA0 = (int16x4_t)__mmi_punpcklbh_s(_pA, _zero);
                    int16x4_t _pA1 = (int16x4_t)__mmi_punpckhbh_s(_pA, _zero);
                    int16x4_t _pB0 = (int16x4_t)__mmi_punpcklbh_s(_pB, _zero);
                    int16x4_t _pB1 = (int16x4_t)__mmi_punpckhbh_s(_pB, _zero);
                    _pA0 = __mmi_psrah_s(__mmi_psllh_s(_pA0, 8), 8);
                    _pA1 = __mmi_psrah_s(__mmi_psllh_s(_pA1, 8), 8);
                    _pB0 = __mmi_psrah_s(__mmi_psllh_s(_pB0, 8), 8);
                    _pB1 = __mmi_psrah_s(__mmi_psllh_s(_pB1, 8), 8);
                    _sum00 = __mmi_paddw_s(_sum00, __mmi_pmaddhw(_pA0, _pB0));
                    _sum01 = __mmi_paddw_s(_sum01, __mmi_pmaddhw(_pA1, _pB0));
                    _sum10 = __mmi_paddw_s(_sum10, __mmi_pmaddhw(_pA0, _pB1));
                    _sum11 = __mmi_paddw_s(_sum11, __mmi_pmaddhw(_pA1, _pB1));
                    pA += 8;
                    pB += 8;
                }
#endif // NCNN_GNU_INLINE_ASM
                _sum00 = __mmi_paddw_s(_sum00, __mmi_punpckhwd_s(_sum00, _sum00));
                _sum01 = __mmi_paddw_s(_sum01, __mmi_punpckhwd_s(_sum01, _sum01));
                _sum10 = __mmi_paddw_s(_sum10, __mmi_punpckhwd_s(_sum10, _sum10));
                _sum11 = __mmi_paddw_s(_sum11, __mmi_punpckhwd_s(_sum11, _sum11));
                sum00_i += _sum00[0];
                sum01_i += _sum01[0];
                sum10_i += _sum10[0];
                sum11_i += _sum11[0];
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk; kk += 4)
                {
                    sum00_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum01_i += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    sum10_i += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    sum11_i += pA[4] * pB[4] + pA[5] * pB[5] + pA[6] * pB[6] + pA[7] * pB[7];
                    pA += 8;
                    pB += 8;
                }
                if (kk + 1 < max_kk)
                {
                    sum00_i += pA[0] * pB[0] + pA[1] * pB[1];
                    sum01_i += pA[2] * pB[0] + pA[3] * pB[1];
                    sum10_i += pA[0] * pB[2] + pA[1] * pB[3];
                    sum11_i += pA[2] * pB[2] + pA[3] * pB[3];
                    pA += 4;
                    pB += 4;
                    kk += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    sum00_i += pA[0] * pB[0];
                    sum01_i += pA[1] * pB[0];
                    sum10_i += pA[0] * pB[1];
                    sum11_i += pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                }
                sum00 += sum00_i * pAD[0] * pBD[0];
                sum01 += sum01_i * pAD[1] * pBD[0];
                sum10 += sum10_i * pAD[0] * pBD[1];
                sum11 += sum11_i * pAD[1] * pBD[1];
                pAD += 2;
                pBD += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
            outptr += 4;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            float sum0 = 0.f;
            float sum1 = 0.f;

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int sum0_i = 0;
                int sum1_i = 0;
                int kk = 0;
#if __mips_loongson_mmi
                int32x2_t _sum0 = __mmi_pzerow_s();
                int32x2_t _sum1 = __mmi_pzerow_s();
#if NCNN_GNU_INLINE_ASM
                double _tmp0;
                double _tmp1;
                double _tmp2;
                double _tmp3;
                double _tmp4;
                double _tmp5;
                double _tmp6;
                double _shift;
                const int shift_8 = 8;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(
                        "ld         $0, 16(%1)      \n"
                        "ldc1       %4, 0(%0)       \n"
                        "lwc1       %6, 0(%1)       \n"
#if __mips64
                        "dmtc1      $0, %5          \n"
#else
                        "mtc1       $0, %5          \n"
#endif
                        "mtc1       %16, %11        \n"
                        "punpcklwd  %6, %6, %6      \n"
#if __mips64
                        "daddiu     %0, %0, 8       \n"
                        "daddiu     %1, %1, 4       \n"
#else
                        "addiu      %0, %0, 8       \n"
                        "addiu      %1, %1, 4       \n"
#endif
                        "punpcklbh  %8, %4, %5      \n"
                        "punpckhbh  %9, %4, %5      \n"
                        "punpcklbh  %10, %6, %5     \n"
                        "psllh      %8, %8, %11     \n"
                        "psllh      %9, %9, %11     \n"
                        "psllh      %10, %10, %11   \n"
                        "psrah      %8, %8, %11     \n"
                        "psrah      %9, %9, %11     \n"
                        "psrah      %10, %10, %11   \n"
                        "pmaddhw    %4, %8, %10     \n"
                        "pmaddhw    %5, %9, %10     \n"
                        "paddw      %2, %2, %4      \n"
                        "paddw      %3, %3, %5      \n"
                        : "=r"(pA),
                        "=r"(pB),
                        "=f"(_sum0),
                        "=f"(_sum1),
                        "=&f"(_tmp0),
                        "=&f"(_tmp1),
                        "=&f"(_tmp2),
                        "=&f"(_tmp3),
                        "=&f"(_tmp4),
                        "=&f"(_tmp5),
                        "=&f"(_tmp6),
                        "=&f"(_shift)
                        : "0"(pA),
                        "1"(pB),
                        "2"(_sum0),
                        "3"(_sum1),
                        "r"(shift_8)
                        : "memory");
                }
#else  // NCNN_GNU_INLINE_ASM
                const int8x8_t _zero = __mmi_pzerob_s();
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __builtin_prefetch(pB + 16);
                    const int8x8_t _pA = __mmi_pldb_s(pA);
                    const int8x8_t _pB = (int8x8_t)__mmi_pfillw_s(*(const int*)pB);
                    int16x4_t _pA0 = (int16x4_t)__mmi_punpcklbh_s(_pA, _zero);
                    int16x4_t _pA1 = (int16x4_t)__mmi_punpckhbh_s(_pA, _zero);
                    int16x4_t _pB0 = (int16x4_t)__mmi_punpcklbh_s(_pB, _zero);
                    _pA0 = __mmi_psrah_s(__mmi_psllh_s(_pA0, 8), 8);
                    _pA1 = __mmi_psrah_s(__mmi_psllh_s(_pA1, 8), 8);
                    _pB0 = __mmi_psrah_s(__mmi_psllh_s(_pB0, 8), 8);
                    _sum0 = __mmi_paddw_s(_sum0, __mmi_pmaddhw(_pA0, _pB0));
                    _sum1 = __mmi_paddw_s(_sum1, __mmi_pmaddhw(_pA1, _pB0));
                    pA += 8;
                    pB += 4;
                }
#endif // NCNN_GNU_INLINE_ASM
                _sum0 = __mmi_paddw_s(_sum0, __mmi_punpckhwd_s(_sum0, _sum0));
                _sum1 = __mmi_paddw_s(_sum1, __mmi_punpckhwd_s(_sum1, _sum1));
                sum0_i += _sum0[0];
                sum1_i += _sum1[0];
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk; kk += 4)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum1_i += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    pA += 8;
                    pB += 4;
                }
                if (kk + 1 < max_kk)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1];
                    sum1_i += pA[2] * pB[0] + pA[3] * pB[1];
                    pA += 4;
                    pB += 2;
                    kk += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    sum0_i += pA[0] * pB[0];
                    sum1_i += pA[1] * pB[0];
                    pA += 2;
                    pB++;
                }
                sum0 += sum0_i * pAD[0] * pBD[0];
                sum1 += sum1_i * pAD[1] * pBD[0];
                pAD += 2;
                pBD++;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }

        pAT += (size_t)2 * A_hstep;
        pAT_descales += (size_t)2 * A_descales_hstep;
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pB = pBT;
        const float* pBD = pBT_descales;

        int jj = 0;
#if __mips_msa
        const v8i16 _one = __msa_fill_h(1);
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + (size_t)4 * K;
            const float* pBD0 = pBD;
            const float* pBD1 = pBD + (size_t)4 * num_blocks;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            v4f32 _fsum1 = (v4f32)__msa_fill_w(0);
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                    const v16i8 _pB0 = __msa_ld_b(pB0, 0);
                    const v16i8 _pB1 = __msa_ld_b(pB1, 0);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB1), _one);
                    pA += 4;
                    pB0 += 16;
                    pB1 += 16;
                }
                if (kk + 1 < max_kk)
                {
                    const v16i8 _pA = (v16i8)__msa_fill_h(*(const short*)pA);
                    const v8i16 _s0 = __msa_dotp_s_h(_pA, (v16i8)__msa_fill_d_ptr(pB0));
                    const v8i16 _s1 = __msa_dotp_s_h(_pA, (v16i8)__msa_fill_d_ptr(pB1));
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    pA += 2;
                    pB0 += 8;
                    pB1 += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    const v8i16 _pA = __msa_fill_h(pA[0]);
                    const v16i8 _pB08 = (v16i8)__msa_fill_w(*(const int*)pB0);
                    const v16i8 _pB18 = (v16i8)__msa_fill_w(*(const int*)pB1);
                    const v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB08, 0), _pB08);
                    const v8i16 _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB18, 0), _pB18);
                    const v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    const v8i16 _s1 = __msa_mulv_h(_pA, _pB1);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    pA++;
                    pB0 += 4;
                    pB1 += 4;
                }
                const v4f32 _descaleB0 = (v4f32)__msa_ld_w(pBD0, 0);
                const v4f32 _descaleB1 = (v4f32)__msa_ld_w(pBD1, 0);
                const v4f32 _descaleA = __msa_fill_w_f32(pAD[0]);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleA, _descaleB0)));
                _fsum1 = __msa_fadd_w(_fsum1, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum1), __msa_fmul_w(_descaleA, _descaleB1)));
                pAD++;
                pBD0 += 4;
                pBD1 += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB = pB1;
            pBD = pBD1;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            v4f32 _fsum0 = (v4f32)__msa_fill_w(0);
            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                    const v16i8 _pB0 = __msa_ld_b(pB, 0);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    pA += 4;
                    pB += 16;
                }
                if (kk + 1 < max_kk)
                {
                    const v16i8 _pA = (v16i8)__msa_fill_h(*(const short*)pA);
                    const v8i16 _s = __msa_dotp_s_h(_pA, (v16i8)__msa_fill_d_ptr(pB));
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 2;
                    pB += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    const v16i8 _pB8 = (v16i8)__msa_fill_w(*(const int*)pB);
                    const v8i16 _pB = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    const v8i16 _s = __msa_mulv_h(__msa_fill_h(pA[0]), _pB);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA++;
                    pB += 4;
                }
                const v4f32 _descaleB = (v4f32)__msa_ld_w(pBD, 0);
                _fsum0 = __msa_fadd_w(_fsum0, __msa_fmul_w((v4f32)__msa_ffint_s_w(_sum0), __msa_fmul_w(_descaleB, __msa_fill_w_f32(pAD[0]))));
                pAD++;
                pBD += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            outptr += 4;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            float sum0 = 0.f;
            float sum1 = 0.f;

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int sum0_i = 0;
                int sum1_i = 0;
                int kk = 0;
#if __mips_loongson_mmi
                int32x2_t _sum0 = __mmi_pzerow_s();
                int32x2_t _sum1 = __mmi_pzerow_s();
#if NCNN_GNU_INLINE_ASM
                double _tmp0;
                double _tmp1;
                double _tmp2;
                double _tmp3;
                double _tmp4;
                double _tmp5;
                double _tmp6;
                double _shift;
                const int shift_8 = 8;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(
                        "ld         $0, 32(%1)      \n"
                        "lwc1       %4, 0(%0)       \n"
                        "ldc1       %6, 0(%1)       \n"
#if __mips64
                        "dmtc1      $0, %5          \n"
#else
                        "mtc1       $0, %5          \n"
#endif
                        "mtc1       %16, %11        \n"
                        "punpcklwd  %4, %4, %4      \n"
#if __mips64
                        "daddiu     %0, %0, 4       \n"
                        "daddiu     %1, %1, 8       \n"
#else
                        "addiu      %0, %0, 4       \n"
                        "addiu      %1, %1, 8       \n"
#endif
                        "punpcklbh  %8, %4, %5      \n"
                        "punpcklbh  %9, %6, %5      \n"
                        "punpckhbh  %10, %6, %5     \n"
                        "psllh      %8, %8, %11     \n"
                        "psllh      %9, %9, %11     \n"
                        "psllh      %10, %10, %11   \n"
                        "psrah      %8, %8, %11     \n"
                        "psrah      %9, %9, %11     \n"
                        "psrah      %10, %10, %11   \n"
                        "pmaddhw    %4, %8, %9      \n"
                        "pmaddhw    %5, %8, %10     \n"
                        "paddw      %2, %2, %4      \n"
                        "paddw      %3, %3, %5      \n"
                        : "=r"(pA),
                        "=r"(pB),
                        "=f"(_sum0),
                        "=f"(_sum1),
                        "=&f"(_tmp0),
                        "=&f"(_tmp1),
                        "=&f"(_tmp2),
                        "=&f"(_tmp3),
                        "=&f"(_tmp4),
                        "=&f"(_tmp5),
                        "=&f"(_tmp6),
                        "=&f"(_shift)
                        : "0"(pA),
                        "1"(pB),
                        "2"(_sum0),
                        "3"(_sum1),
                        "r"(shift_8)
                        : "memory");
                }
#else  // NCNN_GNU_INLINE_ASM
                const int8x8_t _zero = __mmi_pzerob_s();
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __builtin_prefetch(pB + 32);
                    const int8x8_t _pA = (int8x8_t)__mmi_pfillw_s(*(const int*)pA);
                    const int8x8_t _pB = __mmi_pldb_s(pB);
                    int16x4_t _pA0 = (int16x4_t)__mmi_punpcklbh_s(_pA, _zero);
                    int16x4_t _pB0 = (int16x4_t)__mmi_punpcklbh_s(_pB, _zero);
                    int16x4_t _pB1 = (int16x4_t)__mmi_punpckhbh_s(_pB, _zero);
                    _pA0 = __mmi_psrah_s(__mmi_psllh_s(_pA0, 8), 8);
                    _pB0 = __mmi_psrah_s(__mmi_psllh_s(_pB0, 8), 8);
                    _pB1 = __mmi_psrah_s(__mmi_psllh_s(_pB1, 8), 8);
                    _sum0 = __mmi_paddw_s(_sum0, __mmi_pmaddhw(_pA0, _pB0));
                    _sum1 = __mmi_paddw_s(_sum1, __mmi_pmaddhw(_pA0, _pB1));
                    pA += 4;
                    pB += 8;
                }
#endif // NCNN_GNU_INLINE_ASM
                _sum0 = __mmi_paddw_s(_sum0, __mmi_punpckhwd_s(_sum0, _sum0));
                _sum1 = __mmi_paddw_s(_sum1, __mmi_punpckhwd_s(_sum1, _sum1));
                sum0_i += _sum0[0];
                sum1_i += _sum1[0];
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk; kk += 4)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum1_i += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    pA += 4;
                    pB += 8;
                }
                if (kk + 1 < max_kk)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1];
                    sum1_i += pA[0] * pB[2] + pA[1] * pB[3];
                    pA += 2;
                    pB += 4;
                    kk += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    sum0_i += pA[0] * pB[0];
                    sum1_i += pA[0] * pB[1];
                    pA++;
                    pB += 2;
                }
                sum0 += sum0_i * pAD[0] * pBD[0];
                sum1 += sum1_i * pAD[0] * pBD[1];
                pAD++;
                pBD += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pAT;
            const float* pAD = pAT_descales;
            float sum0 = 0.f;

            for (int k = 0; k < K; k += block_size)
            {
                const int max_kk = std::min(K - k, block_size);
                int sum0_i = 0;
                int kk = 0;
#if __mips_loongson_mmi
                int32x2_t _sum0 = __mmi_pzerow_s();
#if NCNN_GNU_INLINE_ASM
                double _tmp0;
                double _tmp1;
                double _tmp2;
                double _tmp3;
                double _tmp4;
                double _tmp5;
                double _shift;
                const int shift_8 = 8;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(
                        "ld         $0, 16(%1)      \n"
                        "lwc1       %3, 0(%0)       \n"
                        "lwc1       %5, 0(%1)       \n"
#if __mips64
                        "dmtc1      $0, %4          \n"
#else
                        "mtc1       $0, %4          \n"
#endif
                        "mtc1       %13, %9         \n"
                        "punpcklwd  %3, %3, %3      \n"
                        "punpcklwd  %5, %5, %5      \n"
#if __mips64
                        "daddiu     %0, %0, 4       \n"
                        "daddiu     %1, %1, 4       \n"
#else
                        "addiu      %0, %0, 4       \n"
                        "addiu      %1, %1, 4       \n"
#endif
                        "punpcklbh  %7, %3, %4      \n"
                        "punpcklbh  %8, %5, %4      \n"
                        "psllh      %7, %7, %9      \n"
                        "psllh      %8, %8, %9      \n"
                        "psrah      %7, %7, %9      \n"
                        "psrah      %8, %8, %9      \n"
                        "pmaddhw    %3, %7, %8      \n"
                        "paddw      %2, %2, %3      \n"
                        : "=r"(pA),
                        "=r"(pB),
                        "=f"(_sum0),
                        "=&f"(_tmp0),
                        "=&f"(_tmp1),
                        "=&f"(_tmp2),
                        "=&f"(_tmp3),
                        "=&f"(_tmp4),
                        "=&f"(_tmp5),
                        "=&f"(_shift)
                        : "0"(pA),
                        "1"(pB),
                        "2"(_sum0),
                        "r"(shift_8)
                        : "memory");
                }
#else  // NCNN_GNU_INLINE_ASM
                const int8x8_t _zero = __mmi_pzerob_s();
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x8_t _pA = (int8x8_t)__mmi_pfillw_s(*(const int*)pA);
                    const int8x8_t _pB = (int8x8_t)__mmi_pfillw_s(*(const int*)pB);
                    int16x4_t _pA0 = (int16x4_t)__mmi_punpcklbh_s(_pA, _zero);
                    int16x4_t _pB0 = (int16x4_t)__mmi_punpcklbh_s(_pB, _zero);
                    _pA0 = __mmi_psrah_s(__mmi_psllh_s(_pA0, 8), 8);
                    _pB0 = __mmi_psrah_s(__mmi_psllh_s(_pB0, 8), 8);
                    _sum0 = __mmi_paddw_s(_sum0, __mmi_pmaddhw(_pA0, _pB0));
                    pA += 4;
                    pB += 4;
                }
#endif // NCNN_GNU_INLINE_ASM
                _sum0 = __mmi_paddw_s(_sum0, __mmi_punpckhwd_s(_sum0, _sum0));
                sum0_i += _sum0[0];
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk; kk += 4)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    pA += 4;
                    pB += 4;
                }
                if (kk + 1 < max_kk)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                    kk += 2;
                }
                for (; kk < max_kk; kk++)
                    sum0_i += *pA++ * *pB++;
                sum0 += sum0_i * pAD[0] * pBD[0];
                pAD++;
                pBD++;
            }

            *outptr++ = sum0;
        }

        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void unpack_output_tile_wq_int8(const float* pp, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    float* outptr = (float*)top_blob + (size_t)i * out_hstep + j;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = outptr;
        float* outptr1 = outptr0 + out_hstep;
        float* outptr2 = outptr0 + out_hstep * 2;
        float* outptr3 = outptr0 + out_hstep * 3;
        float* outptr4 = outptr0 + out_hstep * 4;
        float* outptr5 = outptr0 + out_hstep * 5;
        float* outptr6 = outptr0 + out_hstep * 6;
        float* outptr7 = outptr0 + out_hstep * 7;

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
                    const v4f32 _beta = __msa_fill_w_f32(beta);
                    _c0123 = __msa_fmul_w(_c0123, _beta);
                    _c4567 = __msa_fmul_w(_c4567, _beta);
                }
            }
        }

        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;
        const float* pC2 = pC0 ? pC0 + c_hstep * 2 : 0;
        const float* pC3 = pC0 ? pC0 + c_hstep * 3 : 0;
        const float* pC4 = pC0 ? pC0 + c_hstep * 4 : 0;
        const float* pC5 = pC0 ? pC0 + c_hstep * 5 : 0;
        const float* pC6 = pC0 ? pC0 + c_hstep * 6 : 0;
        const float* pC7 = pC0 ? pC0 + c_hstep * 7 : 0;

        int jj = 0;
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
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            transpose4x4_ps(_f4, _f5, _f6, _f7);

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
                    const v4f32 _beta = __msa_fill_w_f32(beta);
                    v4f32 _c = (v4f32)__msa_ld_w(pC0, 0);
                    _f0 = __msa_fadd_w(_f0, beta == 1.f ? _c : __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC1, 0);
                    _f1 = __msa_fadd_w(_f1, beta == 1.f ? _c : __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC2, 0);
                    _f2 = __msa_fadd_w(_f2, beta == 1.f ? _c : __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC3, 0);
                    _f3 = __msa_fadd_w(_f3, beta == 1.f ? _c : __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC4, 0);
                    _f4 = __msa_fadd_w(_f4, beta == 1.f ? _c : __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC5, 0);
                    _f5 = __msa_fadd_w(_f5, beta == 1.f ? _c : __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC6, 0);
                    _f6 = __msa_fadd_w(_f6, beta == 1.f ? _c : __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC7, 0);
                    _f7 = __msa_fadd_w(_f7, beta == 1.f ? _c : __msa_fmul_w(_c, _beta));
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c = (v4f32)__msa_ld_w(pC, 0);
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
                const v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
                _f6 = __msa_fmul_w(_f6, _alpha);
                _f7 = __msa_fmul_w(_f7, _alpha);
            }
            __msa_st_w((v4i32)_f0, outptr0, 0);
            __msa_st_w((v4i32)_f1, outptr1, 0);
            __msa_st_w((v4i32)_f2, outptr2, 0);
            __msa_st_w((v4i32)_f3, outptr3, 0);
            __msa_st_w((v4i32)_f4, outptr4, 0);
            __msa_st_w((v4i32)_f5, outptr5, 0);
            __msa_st_w((v4i32)_f6, outptr6, 0);
            __msa_st_w((v4i32)_f7, outptr7, 0);
            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
            outptr4 += 4;
            outptr5 += 4;
            outptr6 += 4;
            outptr7 += 4;
            if (pC0)
            {
                pC0 += 4;
                pC1 += 4;
                pC2 += 4;
                pC3 += 4;
                pC4 += 4;
                pC5 += 4;
                pC6 += 4;
                pC7 += 4;
            }
            if (pC && broadcast_type_C == 4)
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f4 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f5 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

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
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    v4f32 _c4 = (v4f32) {
                        pC4[0], pC5[0], pC6[0], pC7[0]
                    };
                    v4f32 _c1 = (v4f32) {
                        pC0[1], pC1[1], pC2[1], pC3[1]
                    };
                    v4f32 _c5 = (v4f32) {
                        pC4[1], pC5[1], pC6[1], pC7[1]
                    };
                    if (beta != 1.f)
                    {
                        const v4f32 _beta = __msa_fill_w_f32(beta);
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
                const v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
            }
            ((int*)outptr0)[0] = __msa_copy_s_w((v4i32)_f0, 0);
            ((int*)outptr0)[1] = __msa_copy_s_w((v4i32)_f1, 0);
            ((int*)outptr1)[0] = __msa_copy_s_w((v4i32)_f0, 1);
            ((int*)outptr1)[1] = __msa_copy_s_w((v4i32)_f1, 1);
            ((int*)outptr2)[0] = __msa_copy_s_w((v4i32)_f0, 2);
            ((int*)outptr2)[1] = __msa_copy_s_w((v4i32)_f1, 2);
            ((int*)outptr3)[0] = __msa_copy_s_w((v4i32)_f0, 3);
            ((int*)outptr3)[1] = __msa_copy_s_w((v4i32)_f1, 3);
            ((int*)outptr4)[0] = __msa_copy_s_w((v4i32)_f4, 0);
            ((int*)outptr4)[1] = __msa_copy_s_w((v4i32)_f5, 0);
            ((int*)outptr5)[0] = __msa_copy_s_w((v4i32)_f4, 1);
            ((int*)outptr5)[1] = __msa_copy_s_w((v4i32)_f5, 1);
            ((int*)outptr6)[0] = __msa_copy_s_w((v4i32)_f4, 2);
            ((int*)outptr6)[1] = __msa_copy_s_w((v4i32)_f5, 2);
            ((int*)outptr7)[0] = __msa_copy_s_w((v4i32)_f4, 3);
            ((int*)outptr7)[1] = __msa_copy_s_w((v4i32)_f5, 3);
            outptr0 += 2;
            outptr1 += 2;
            outptr2 += 2;
            outptr3 += 2;
            outptr4 += 2;
            outptr5 += 2;
            outptr6 += 2;
            outptr7 += 2;
            if (pC0)
            {
                pC0 += 2;
                pC1 += 2;
                pC2 += 2;
                pC3 += 2;
                pC4 += 2;
                pC5 += 2;
                pC6 += 2;
                pC7 += 2;
            }
            if (pC && broadcast_type_C == 4)
                pC += 2;
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
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    v4f32 _c4 = (v4f32) {
                        pC4[0], pC5[0], pC6[0], pC7[0]
                    };
                    if (beta != 1.f)
                    {
                        const v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c4);
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0];
                    if (beta != 1.f)
                        c *= beta;
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c));
                    _f4 = __msa_fadd_w(_f4, __msa_fill_w_f32(c));
                }
            }

            if (alpha != 1.f)
            {
                const v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
            }
            ((int*)outptr0)[0] = __msa_copy_s_w((v4i32)_f0, 0);
            ((int*)outptr1)[0] = __msa_copy_s_w((v4i32)_f0, 1);
            ((int*)outptr2)[0] = __msa_copy_s_w((v4i32)_f0, 2);
            ((int*)outptr3)[0] = __msa_copy_s_w((v4i32)_f0, 3);
            ((int*)outptr4)[0] = __msa_copy_s_w((v4i32)_f4, 0);
            ((int*)outptr5)[0] = __msa_copy_s_w((v4i32)_f4, 1);
            ((int*)outptr6)[0] = __msa_copy_s_w((v4i32)_f4, 2);
            ((int*)outptr7)[0] = __msa_copy_s_w((v4i32)_f4, 3);
            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            outptr4++;
            outptr5++;
            outptr6++;
            outptr7++;
        }
        outptr += out_hstep * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = outptr;
        float* outptr1 = outptr0 + out_hstep;
        float* outptr2 = outptr0 + out_hstep * 2;
        float* outptr3 = outptr0 + out_hstep * 3;

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

        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;
        const float* pC2 = pC0 ? pC0 + c_hstep * 2 : 0;
        const float* pC3 = pC0 ? pC0 + c_hstep * 3 : 0;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 12, 0);
            v4f32 _f4 = (v4f32)__msa_ld_w(pp + 16, 0);
            v4f32 _f5 = (v4f32)__msa_ld_w(pp + 20, 0);
            v4f32 _f6 = (v4f32)__msa_ld_w(pp + 24, 0);
            v4f32 _f7 = (v4f32)__msa_ld_w(pp + 28, 0);
            pp += 32;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                    _f2 = __msa_fadd_w(_f2, _c0123);
                    _f3 = __msa_fadd_w(_f3, _c0123);
                    _f4 = __msa_fadd_w(_f4, _c0123);
                    _f5 = __msa_fadd_w(_f5, _c0123);
                    _f6 = __msa_fadd_w(_f6, _c0123);
                    _f7 = __msa_fadd_w(_f7, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC1, 0);
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC2, 0);
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC3, 0);
                    transpose4x4_ps(_c0, _c1, _c2, _c3);
                    v4f32 _c4 = (v4f32)__msa_ld_w(pC0 + 4, 0);
                    v4f32 _c5 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                    v4f32 _c6 = (v4f32)__msa_ld_w(pC2 + 4, 0);
                    v4f32 _c7 = (v4f32)__msa_ld_w(pC3 + 4, 0);
                    transpose4x4_ps(_c4, _c5, _c6, _c7);
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c1 = __msa_fmul_w(_c1, _beta);
                        _c2 = __msa_fmul_w(_c2, _beta);
                        _c3 = __msa_fmul_w(_c3, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                        _c5 = __msa_fmul_w(_c5, _beta);
                        _c6 = __msa_fmul_w(_c6, _beta);
                        _c7 = __msa_fmul_w(_c7, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c1);
                    _f2 = __msa_fadd_w(_f2, _c2);
                    _f3 = __msa_fadd_w(_f3, _c3);
                    _f4 = __msa_fadd_w(_f4, _c4);
                    _f5 = __msa_fadd_w(_f5, _c5);
                    _f6 = __msa_fadd_w(_f6, _c6);
                    _f7 = __msa_fadd_w(_f7, _c7);
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    v4f32 _c4 = (v4f32)__msa_ld_w(pC + 4, 0);
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0, 3));
                    _f4 = __msa_fadd_w(_f4, (v4f32)__msa_splati_w((v4i32)_c4, 0));
                    _f5 = __msa_fadd_w(_f5, (v4f32)__msa_splati_w((v4i32)_c4, 1));
                    _f6 = __msa_fadd_w(_f6, (v4f32)__msa_splati_w((v4i32)_c4, 2));
                    _f7 = __msa_fadd_w(_f7, (v4f32)__msa_splati_w((v4i32)_c4, 3));
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

            transpose4x4_ps(_f0, _f1, _f2, _f3);
            transpose4x4_ps(_f4, _f5, _f6, _f7);
            __msa_st_w((v4i32)_f0, outptr0, 0);
            __msa_st_w((v4i32)_f4, outptr0 + 4, 0);
            __msa_st_w((v4i32)_f1, outptr1, 0);
            __msa_st_w((v4i32)_f5, outptr1 + 4, 0);
            __msa_st_w((v4i32)_f2, outptr2, 0);
            __msa_st_w((v4i32)_f6, outptr2 + 4, 0);
            __msa_st_w((v4i32)_f3, outptr3, 0);
            __msa_st_w((v4i32)_f7, outptr3 + 4, 0);
            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
            if (pC0)
            {
                pC0 += 8;
                pC1 += 8;
                pC2 += 8;
                pC3 += 8;
            }
            if (pC && broadcast_type_C == 4)
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                    _f2 = __msa_fadd_w(_f2, _c0123);
                    _f3 = __msa_fadd_w(_f3, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC1, 0);
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC2, 0);
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC3, 0);
                    transpose4x4_ps(_c0, _c1, _c2, _c3);
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
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0, 3));
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

            transpose4x4_ps(_f0, _f1, _f2, _f3);
            __msa_st_w((v4i32)_f0, outptr0, 0);
            __msa_st_w((v4i32)_f1, outptr1, 0);
            __msa_st_w((v4i32)_f2, outptr2, 0);
            __msa_st_w((v4i32)_f3, outptr3, 0);
            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
            if (pC0)
            {
                pC0 += 4;
                pC1 += 4;
                pC2 += 4;
                pC3 += 4;
            }
            if (pC && broadcast_type_C == 4)
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            pp += 8;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    v4f32 _c1 = (v4f32) {
                        pC0[1], pC1[1], pC2[1], pC3[1]
                    };
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
                    float c0 = pC[0];
                    float c1 = pC[1];
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

            v4f32 _f2 = (v4f32)__msa_fill_w(0);
            v4f32 _f3 = (v4f32)__msa_fill_w(0);
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            *(int64_t*)outptr0 = __msa_copy_s_d((v2i64)_f0, 0);
            *(int64_t*)outptr1 = __msa_copy_s_d((v2i64)_f1, 0);
            *(int64_t*)outptr2 = __msa_copy_s_d((v2i64)_f2, 0);
            *(int64_t*)outptr3 = __msa_copy_s_d((v2i64)_f3, 0);
            outptr0 += 2;
            outptr1 += 2;
            outptr2 += 2;
            outptr3 += 2;
            if (pC0)
            {
                pC0 += 2;
                pC1 += 2;
                pC2 += 2;
                pC3 += 2;
            }
            if (pC && broadcast_type_C == 4)
                pC += 2;
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
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0];
                    if (beta != 1.f)
                        c *= beta;
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c));
                }
            }
            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            outptr0[0] = _f0[0];
            outptr1[0] = _f0[1];
            outptr2[0] = _f0[2];
            outptr3[0] = _f0[3];
            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            if (pC0)
            {
                pC0++;
                pC1++;
                pC2++;
                pC3++;
            }
            if (pC && broadcast_type_C == 4)
                pC++;
        }
        outptr += out_hstep * 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = outptr;
        float* outptr1 = outptr0 + out_hstep;
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
        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;

        float c0 = 0.f;
        float c1 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            c0 = pC[0];
            c1 = pC[broadcast_type_C == 0 ? 0 : 1];
            if (beta != 1.f)
            {
                c0 *= beta;
                c1 *= beta;
            }
        }

        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
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
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC0 + 4, 0);
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC1, 0);
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC1 + 4, 0);
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

            __msa_st_w((v4i32)_f0, outptr0, 0);
            __msa_st_w((v4i32)_f1, outptr0 + 4, 0);
            __msa_st_w((v4i32)_f2, outptr1, 0);
            __msa_st_w((v4i32)_f3, outptr1 + 4, 0);
            outptr0 += 8;
            outptr1 += 8;
            if (pC0)
            {
                pC0 += 8;
                pC1 += 8;
            }
            if (pC && broadcast_type_C == 4)
                pC += 8;
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
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC1, 0);
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

            __msa_st_w((v4i32)_f0, outptr0, 0);
            __msa_st_w((v4i32)_f1, outptr1, 0);
            outptr0 += 4;
            outptr1 += 4;
            if (pC0)
            {
                pC0 += 4;
                pC1 += 4;
            }
            if (pC && broadcast_type_C == 4)
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _f = (v4f32)__msa_ld_w(pp, 0);
            pp += 4;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __msa_fadd_w(_f, (v4f32) {
                    c0, c1, c0, c1
                });
                if (broadcast_type_C == 3)
                {
                    v4f32 _c = (v4f32) {
                        pC0[0], pC1[0], pC0[1], pC1[1]
                    };
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
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
                    _f = __msa_fadd_w(_f, (v4f32) {
                        cc0, cc0, cc1, cc1
                    });
                }
            }

            if (alpha != 1.f)
                _f = __msa_fmul_w(_f, __msa_fill_w_f32(alpha));

            v4i32 _f0 = __msa_pckev_w((v4i32)_f, (v4i32)_f);
            v4i32 _f1 = __msa_pckod_w((v4i32)_f, (v4i32)_f);
            __msa_storel_d(_f0, outptr0);
            __msa_storel_d(_f1, outptr1);
            outptr0 += 2;
            outptr1 += 2;
            if (pC0)
            {
                pC0 += 2;
                pC1 += 2;
            }
            if (pC && broadcast_type_C == 4)
                pC += 2;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = pp[0];
            float sum01 = pp[1];
            float sum10 = pp[2];
            float sum11 = pp[3];
            pp += 4;

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
                    float c00 = pC0[0];
                    float c01 = pC1[0];
                    float c10 = pC0[1];
                    float c11 = pC1[1];
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
                }
            }

            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum01 *= alpha;
                sum10 *= alpha;
                sum11 *= alpha;
            }

            outptr0[0] = sum00;
            outptr1[0] = sum01;
            outptr0[1] = sum10;
            outptr1[1] = sum11;
            outptr0 += 2;
            outptr1 += 2;
            if (pC0)
            {
                pC0 += 2;
                pC1 += 2;
            }
            if (pC && broadcast_type_C == 4)
                pC += 2;
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
                    float c0 = pC0[0];
                    float c1 = pC1[0];
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
                    if (beta != 1.f) c *= beta;
                    sum0 += c;
                    sum1 += c;
                }
            }
            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }
            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr0++;
            outptr1++;
            if (pC0)
            {
                pC0++;
                pC1++;
            }
            if (pC && broadcast_type_C == 4)
                pC++;
        }
        outptr += out_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        float* outptr0 = outptr;
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
        const float* pC0 = pC && (broadcast_type_C == 3 || broadcast_type_C == 4) ? pC : 0;

        float c0 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            c0 = pC[0];
            if (beta != 1.f)
                c0 *= beta;
        }

        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
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
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC0 + 4, 0);
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

            __msa_st_w((v4i32)_f0, outptr0, 0);
            __msa_st_w((v4i32)_f1, outptr0 + 4, 0);
            outptr0 += 8;
            if (pC0)
                pC0 += 8;
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
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                }
            }

            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));

            __msa_st_w((v4i32)_f0, outptr0, 0);
            outptr0 += 4;
            if (pC0)
                pC0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4i32 _fi = __msa_fill_w(0);
            _fi = __msa_insert_w(_fi, 0, ((const int*)pp)[0]);
            _fi = __msa_insert_w(_fi, 1, ((const int*)pp)[1]);
            v4f32 _f = (v4f32)_fi;
            pp += 2;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __msa_fadd_w(_f, __msa_fill_w_f32(c0));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    v4i32 _ci = __msa_fill_w(0);
                    _ci = __msa_insert_w(_ci, 0, ((const int*)pC0)[0]);
                    _ci = __msa_insert_w(_ci, 1, ((const int*)pC0)[1]);
                    v4f32 _c = (v4f32)_ci;
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
                }
            }

            if (alpha != 1.f)
                _f = __msa_fmul_w(_f, __msa_fill_w_f32(alpha));

            __msa_storel_d((v4i32)_f, outptr0);
            outptr0 += 2;
            if (pC0)
                pC0 += 2;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = pp[0];
            float sum1 = pp[1];
            pp += 2;
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
                if (broadcast_type_C == 3)
                {
                    float c0 = pC0[0];
                    float c1 = pC0[1];
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
                    float c0 = pC0[0];
                    float c1 = pC0[1];
                    if (beta != 1.f)
                    {
                        c0 *= beta;
                        c1 *= beta;
                    }
                    sum0 += c0;
                    sum1 += c1;
                }
            }
            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }
            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0 += 2;
            if (pC0)
                pC0 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = *pp++;
            if (pC)
            {
                float c = 0.f;
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2) c = c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4) c = pC0[0];
                if ((broadcast_type_C == 3 || broadcast_type_C == 4) && beta != 1.f) c *= beta;
                sum0 += c;
            }
            if (alpha != 1.f) sum0 *= alpha;
            outptr0[0] = sum0;
            outptr0++;
            if (pC0)
                pC0++;
        }
        outptr += out_hstep;
    }
}

static void transpose_unpack_output_tile_wq_int8(const float* pp, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    float* outptr0 = (float*)top_blob + (size_t)j * out_hstep + i;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr = outptr0;

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
                    const v4f32 _beta = __msa_fill_w_f32(beta);
                    _c0123 = __msa_fmul_w(_c0123, _beta);
                    _c4567 = __msa_fmul_w(_c4567, _beta);
                }
            }
        }

        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;
        const float* pC2 = pC0 ? pC0 + c_hstep * 2 : 0;
        const float* pC3 = pC0 ? pC0 + c_hstep * 3 : 0;
        const float* pC4 = pC0 ? pC0 + c_hstep * 4 : 0;
        const float* pC5 = pC0 ? pC0 + c_hstep * 5 : 0;
        const float* pC6 = pC0 ? pC0 + c_hstep * 6 : 0;
        const float* pC7 = pC0 ? pC0 + c_hstep * 7 : 0;

        int jj = 0;
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

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                    _f2 = __msa_fadd_w(_f2, _c0123);
                    _f3 = __msa_fadd_w(_f3, _c0123);
                    _f4 = __msa_fadd_w(_f4, _c4567);
                    _f5 = __msa_fadd_w(_f5, _c4567);
                    _f6 = __msa_fadd_w(_f6, _c4567);
                    _f7 = __msa_fadd_w(_f7, _c4567);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _cl0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    v4f32 _ch0 = (v4f32) {
                        pC4[0], pC5[0], pC6[0], pC7[0]
                    };
                    v4f32 _cl1 = (v4f32) {
                        pC0[1], pC1[1], pC2[1], pC3[1]
                    };
                    v4f32 _ch1 = (v4f32) {
                        pC4[1], pC5[1], pC6[1], pC7[1]
                    };
                    v4f32 _cl2 = (v4f32) {
                        pC0[2], pC1[2], pC2[2], pC3[2]
                    };
                    v4f32 _ch2 = (v4f32) {
                        pC4[2], pC5[2], pC6[2], pC7[2]
                    };
                    v4f32 _cl3 = (v4f32) {
                        pC0[3], pC1[3], pC2[3], pC3[3]
                    };
                    v4f32 _ch3 = (v4f32) {
                        pC4[3], pC5[3], pC6[3], pC7[3]
                    };
                    if (beta != 1.f)
                    {
                        const v4f32 _beta = __msa_fill_w_f32(beta);
                        _cl0 = __msa_fmul_w(_cl0, _beta);
                        _ch0 = __msa_fmul_w(_ch0, _beta);
                        _cl1 = __msa_fmul_w(_cl1, _beta);
                        _ch1 = __msa_fmul_w(_ch1, _beta);
                        _cl2 = __msa_fmul_w(_cl2, _beta);
                        _ch2 = __msa_fmul_w(_ch2, _beta);
                        _cl3 = __msa_fmul_w(_cl3, _beta);
                        _ch3 = __msa_fmul_w(_ch3, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _cl0);
                    _f4 = __msa_fadd_w(_f4, _ch0);
                    _f1 = __msa_fadd_w(_f1, _cl1);
                    _f5 = __msa_fadd_w(_f5, _ch1);
                    _f2 = __msa_fadd_w(_f2, _cl2);
                    _f6 = __msa_fadd_w(_f6, _ch2);
                    _f3 = __msa_fadd_w(_f3, _cl3);
                    _f7 = __msa_fadd_w(_f7, _ch3);
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c = (v4f32)__msa_ld_w(pC, 0);
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c, 0));
                    _f4 = __msa_fadd_w(_f4, (v4f32)__msa_splati_w((v4i32)_c, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c, 1));
                    _f5 = __msa_fadd_w(_f5, (v4f32)__msa_splati_w((v4i32)_c, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c, 2));
                    _f6 = __msa_fadd_w(_f6, (v4f32)__msa_splati_w((v4i32)_c, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c, 3));
                    _f7 = __msa_fadd_w(_f7, (v4f32)__msa_splati_w((v4i32)_c, 3));
                }
            }

            if (alpha != 1.f)
            {
                const v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
                _f6 = __msa_fmul_w(_f6, _alpha);
                _f7 = __msa_fmul_w(_f7, _alpha);
            }
            __msa_st_w((v4i32)_f0, outptr, 0);
            __msa_st_w((v4i32)_f4, outptr + 4, 0);
            __msa_st_w((v4i32)_f1, outptr + out_hstep, 0);
            __msa_st_w((v4i32)_f5, outptr + out_hstep + 4, 0);
            __msa_st_w((v4i32)_f2, outptr + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f6, outptr + out_hstep * 2 + 4, 0);
            __msa_st_w((v4i32)_f3, outptr + out_hstep * 3, 0);
            __msa_st_w((v4i32)_f7, outptr + out_hstep * 3 + 4, 0);
            outptr += out_hstep * 4;
            if (pC0)
            {
                pC0 += 4;
                pC1 += 4;
                pC2 += 4;
                pC3 += 4;
                pC4 += 4;
                pC5 += 4;
                pC6 += 4;
                pC7 += 4;
            }
            if (pC && broadcast_type_C == 4)
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f4 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f5 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

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
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    v4f32 _c4 = (v4f32) {
                        pC4[0], pC5[0], pC6[0], pC7[0]
                    };
                    v4f32 _c1 = (v4f32) {
                        pC0[1], pC1[1], pC2[1], pC3[1]
                    };
                    v4f32 _c5 = (v4f32) {
                        pC4[1], pC5[1], pC6[1], pC7[1]
                    };
                    if (beta != 1.f)
                    {
                        const v4f32 _beta = __msa_fill_w_f32(beta);
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
                const v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
            }
            __msa_st_w((v4i32)_f0, outptr, 0);
            __msa_st_w((v4i32)_f4, outptr + 4, 0);
            __msa_st_w((v4i32)_f1, outptr + out_hstep, 0);
            __msa_st_w((v4i32)_f5, outptr + out_hstep + 4, 0);
            outptr += out_hstep * 2;
            if (pC0)
            {
                pC0 += 2;
                pC1 += 2;
                pC2 += 2;
                pC3 += 2;
                pC4 += 2;
                pC5 += 2;
                pC6 += 2;
                pC7 += 2;
            }
            if (pC && broadcast_type_C == 4)
                pC += 2;
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
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    v4f32 _c4 = (v4f32) {
                        pC4[0], pC5[0], pC6[0], pC7[0]
                    };
                    if (beta != 1.f)
                    {
                        const v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c4);
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0];
                    if (beta != 1.f)
                        c *= beta;
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c));
                    _f4 = __msa_fadd_w(_f4, __msa_fill_w_f32(c));
                }
            }

            if (alpha != 1.f)
            {
                const v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
            }
            __msa_st_w((v4i32)_f0, outptr, 0);
            __msa_st_w((v4i32)_f4, outptr + 4, 0);
            outptr += out_hstep;
        }
        outptr0 += 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr = outptr0;

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

        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;
        const float* pC2 = pC0 ? pC0 + c_hstep * 2 : 0;
        const float* pC3 = pC0 ? pC0 + c_hstep * 3 : 0;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 12, 0);
            v4f32 _f4 = (v4f32)__msa_ld_w(pp + 16, 0);
            v4f32 _f5 = (v4f32)__msa_ld_w(pp + 20, 0);
            v4f32 _f6 = (v4f32)__msa_ld_w(pp + 24, 0);
            v4f32 _f7 = (v4f32)__msa_ld_w(pp + 28, 0);
            pp += 32;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                    _f2 = __msa_fadd_w(_f2, _c0123);
                    _f3 = __msa_fadd_w(_f3, _c0123);
                    _f4 = __msa_fadd_w(_f4, _c0123);
                    _f5 = __msa_fadd_w(_f5, _c0123);
                    _f6 = __msa_fadd_w(_f6, _c0123);
                    _f7 = __msa_fadd_w(_f7, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC1, 0);
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC2, 0);
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC3, 0);
                    transpose4x4_ps(_c0, _c1, _c2, _c3);
                    v4f32 _c4 = (v4f32)__msa_ld_w(pC0 + 4, 0);
                    v4f32 _c5 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                    v4f32 _c6 = (v4f32)__msa_ld_w(pC2 + 4, 0);
                    v4f32 _c7 = (v4f32)__msa_ld_w(pC3 + 4, 0);
                    transpose4x4_ps(_c4, _c5, _c6, _c7);
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c1 = __msa_fmul_w(_c1, _beta);
                        _c2 = __msa_fmul_w(_c2, _beta);
                        _c3 = __msa_fmul_w(_c3, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                        _c5 = __msa_fmul_w(_c5, _beta);
                        _c6 = __msa_fmul_w(_c6, _beta);
                        _c7 = __msa_fmul_w(_c7, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c1);
                    _f2 = __msa_fadd_w(_f2, _c2);
                    _f3 = __msa_fadd_w(_f3, _c3);
                    _f4 = __msa_fadd_w(_f4, _c4);
                    _f5 = __msa_fadd_w(_f5, _c5);
                    _f6 = __msa_fadd_w(_f6, _c6);
                    _f7 = __msa_fadd_w(_f7, _c7);
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC, 0);
                    v4f32 _c4 = (v4f32)__msa_ld_w(pC + 4, 0);
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0, 3));
                    _f4 = __msa_fadd_w(_f4, (v4f32)__msa_splati_w((v4i32)_c4, 0));
                    _f5 = __msa_fadd_w(_f5, (v4f32)__msa_splati_w((v4i32)_c4, 1));
                    _f6 = __msa_fadd_w(_f6, (v4f32)__msa_splati_w((v4i32)_c4, 2));
                    _f7 = __msa_fadd_w(_f7, (v4f32)__msa_splati_w((v4i32)_c4, 3));
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

            __msa_st_w((v4i32)_f0, outptr, 0);
            __msa_st_w((v4i32)_f1, outptr + out_hstep, 0);
            __msa_st_w((v4i32)_f2, outptr + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f3, outptr + out_hstep * 3, 0);
            __msa_st_w((v4i32)_f4, outptr + out_hstep * 4, 0);
            __msa_st_w((v4i32)_f5, outptr + out_hstep * 5, 0);
            __msa_st_w((v4i32)_f6, outptr + out_hstep * 6, 0);
            __msa_st_w((v4i32)_f7, outptr + out_hstep * 7, 0);
            outptr += out_hstep * 8;
            if (pC0)
            {
                pC0 += 8;
                pC1 += 8;
                pC2 += 8;
                pC3 += 8;
            }
            if (pC && broadcast_type_C == 4)
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                    _f2 = __msa_fadd_w(_f2, _c0123);
                    _f3 = __msa_fadd_w(_f3, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC1, 0);
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC2, 0);
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC3, 0);
                    transpose4x4_ps(_c0, _c1, _c2, _c3);
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
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0, 3));
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

            __msa_st_w((v4i32)_f0, outptr, 0);
            __msa_st_w((v4i32)_f1, outptr + out_hstep, 0);
            __msa_st_w((v4i32)_f2, outptr + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f3, outptr + out_hstep * 3, 0);
            outptr += out_hstep * 4;
            if (pC0)
            {
                pC0 += 4;
                pC1 += 4;
                pC2 += 4;
                pC3 += 4;
            }
            if (pC && broadcast_type_C == 4)
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            pp += 8;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    v4f32 _c1 = (v4f32) {
                        pC0[1], pC1[1], pC2[1], pC3[1]
                    };
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
                    float c0 = pC[0];
                    float c1 = pC[1];
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

            __msa_st_w((v4i32)_f0, outptr, 0);
            __msa_st_w((v4i32)_f1, outptr + out_hstep, 0);
            outptr += out_hstep * 2;
            if (pC0)
            {
                pC0 += 2;
                pC1 += 2;
                pC2 += 2;
                pC3 += 2;
            }
            if (pC && broadcast_type_C == 4)
                pC += 2;
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
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC2[0], pC3[0]
                    };
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0];
                    if (beta != 1.f)
                        c *= beta;
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c));
                }
            }
            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            __msa_st_w((v4i32)_f0, outptr, 0);
            outptr += out_hstep;
            if (pC0)
            {
                pC0++;
                pC1++;
                pC2++;
                pC3++;
            }
            if (pC && broadcast_type_C == 4)
                pC++;
        }
        outptr0 += 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr = outptr0;
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
        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;

        float c0 = 0.f;
        float c1 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            c0 = pC[0];
            c1 = pC[broadcast_type_C == 0 ? 0 : 1];
            if (beta != 1.f)
            {
                c0 *= beta;
                c1 *= beta;
            }
        }

        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = (v4f32) {
                        c0, c1, c0, c1
                    };
                    _f0 = __msa_fadd_w(_f0, _c);
                    _f1 = __msa_fadd_w(_f1, _c);
                    _f2 = __msa_fadd_w(_f2, _c);
                    _f3 = __msa_fadd_w(_f3, _c);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC0[1], pC1[1]
                    };
                    v4f32 _c1 = (v4f32) {
                        pC0[2], pC1[2], pC0[3], pC1[3]
                    };
                    v4f32 _c2 = (v4f32) {
                        pC0[4], pC1[4], pC0[5], pC1[5]
                    };
                    v4f32 _c3 = (v4f32) {
                        pC0[6], pC1[6], pC0[7], pC1[7]
                    };
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
                    float c00 = pC[0];
                    float c01 = pC[1];
                    float c02 = pC[2];
                    float c03 = pC[3];
                    float c04 = pC[4];
                    float c05 = pC[5];
                    float c06 = pC[6];
                    float c07 = pC[7];
                    if (beta != 1.f)
                    {
                        c00 *= beta;
                        c01 *= beta;
                        c02 *= beta;
                        c03 *= beta;
                        c04 *= beta;
                        c05 *= beta;
                        c06 *= beta;
                        c07 *= beta;
                    }
                    _f0 = __msa_fadd_w(_f0, (v4f32) {
                        c00, c00, c01, c01
                    });
                    _f1 = __msa_fadd_w(_f1, (v4f32) {
                        c02, c02, c03, c03
                    });
                    _f2 = __msa_fadd_w(_f2, (v4f32) {
                        c04, c04, c05, c05
                    });
                    _f3 = __msa_fadd_w(_f3, (v4f32) {
                        c06, c06, c07, c07
                    });
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

            __msa_storel_d((v4i32)_f0, outptr);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f0, (v16i8)_f0, 8), outptr + out_hstep);
            __msa_storel_d((v4i32)_f1, outptr + out_hstep * 2);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f1, (v16i8)_f1, 8), outptr + out_hstep * 3);
            __msa_storel_d((v4i32)_f2, outptr + out_hstep * 4);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f2, (v16i8)_f2, 8), outptr + out_hstep * 5);
            __msa_storel_d((v4i32)_f3, outptr + out_hstep * 6);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f3, (v16i8)_f3, 8), outptr + out_hstep * 7);
            outptr += out_hstep * 8;
            if (pC0)
            {
                pC0 += 8;
                pC1 += 8;
            }
            if (pC && broadcast_type_C == 4)
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            pp += 8;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = (v4f32) {
                        c0, c1, c0, c1
                    };
                    _f0 = __msa_fadd_w(_f0, _c);
                    _f1 = __msa_fadd_w(_f1, _c);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32) {
                        pC0[0], pC1[0], pC0[1], pC1[1]
                    };
                    v4f32 _c1 = (v4f32) {
                        pC0[2], pC1[2], pC0[3], pC1[3]
                    };
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
                    float c00 = pC[0];
                    float c01 = pC[1];
                    float c02 = pC[2];
                    float c03 = pC[3];
                    if (beta != 1.f)
                    {
                        c00 *= beta;
                        c01 *= beta;
                        c02 *= beta;
                        c03 *= beta;
                    }
                    _f0 = __msa_fadd_w(_f0, (v4f32) {
                        c00, c00, c01, c01
                    });
                    _f1 = __msa_fadd_w(_f1, (v4f32) {
                        c02, c02, c03, c03
                    });
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
            }

            __msa_storel_d((v4i32)_f0, outptr);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f0, (v16i8)_f0, 8), outptr + out_hstep);
            __msa_storel_d((v4i32)_f1, outptr + out_hstep * 2);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f1, (v16i8)_f1, 8), outptr + out_hstep * 3);
            outptr += out_hstep * 4;
            if (pC0)
            {
                pC0 += 4;
                pC1 += 4;
            }
            if (pC && broadcast_type_C == 4)
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _f = (v4f32)__msa_ld_w(pp, 0);
            pp += 4;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __msa_fadd_w(_f, (v4f32) {
                    c0, c1, c0, c1
                });
                if (broadcast_type_C == 3)
                {
                    v4f32 _c = (v4f32) {
                        pC0[0], pC1[0], pC0[1], pC1[1]
                    };
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
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
                    _f = __msa_fadd_w(_f, (v4f32) {
                        cc0, cc0, cc1, cc1
                    });
                }
            }

            if (alpha != 1.f)
                _f = __msa_fmul_w(_f, __msa_fill_w_f32(alpha));

            __msa_storel_d((v4i32)_f, outptr);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f, (v16i8)_f, 8), outptr + out_hstep);
            outptr += out_hstep * 2;
            if (pC0)
            {
                pC0 += 2;
                pC1 += 2;
            }
            if (pC && broadcast_type_C == 4)
                pC += 2;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = pp[0];
            float sum01 = pp[1];
            float sum10 = pp[2];
            float sum11 = pp[3];
            pp += 4;
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
                    float c00 = pC0[0];
                    float c01 = pC1[0];
                    float c10 = pC0[1];
                    float c11 = pC1[1];
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
                }
            }
            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum01 *= alpha;
                sum10 *= alpha;
                sum11 *= alpha;
            }
            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[out_hstep] = sum10;
            outptr[out_hstep + 1] = sum11;
            outptr += out_hstep * 2;
            if (pC0)
            {
                pC0 += 2;
                pC1 += 2;
            }
            if (pC && broadcast_type_C == 4)
                pC += 2;
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
                    float c0 = pC0[0];
                    float c1 = pC1[0];
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
                    if (beta != 1.f) c *= beta;
                    sum0 += c;
                    sum1 += c;
                }
            }
            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += out_hstep;
            if (pC0)
            {
                pC0++;
                pC1++;
            }
            if (pC && broadcast_type_C == 4)
                pC++;
        }
        outptr0 += 2;
    }
    for (; ii < max_ii; ii++)
    {
        float* outptr = outptr0;
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
        const float* pC0 = pC && (broadcast_type_C == 3 || broadcast_type_C == 4) ? pC : 0;
        float c0 = 0.f;
        if (pC && (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            c0 = pC[0];
            if (beta != 1.f)
                c0 *= beta;
        }
        int jj = 0;
#if __mips_msa
        v4f32 _c0 = __msa_fill_w_f32(c0);

        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            pp += 8;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC0 + 4, 0);
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
            if (out_hstep == 1)
            {
                __msa_st_w((v4i32)_f0, outptr, 0);
                __msa_st_w((v4i32)_f1, outptr + 4, 0);
            }
            else
            {
                *(int*)outptr = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(outptr + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
                *(int*)(outptr + out_hstep * 2) = __msa_copy_s_w((v4i32)_f0, 2);
                *(int*)(outptr + out_hstep * 3) = __msa_copy_s_w((v4i32)_f0, 3);
                *(int*)(outptr + out_hstep * 4) = __msa_copy_s_w((v4i32)_f1, 0);
                *(int*)(outptr + out_hstep * 5) = __msa_copy_s_w((v4i32)_f1, 1);
                *(int*)(outptr + out_hstep * 6) = __msa_copy_s_w((v4i32)_f1, 2);
                *(int*)(outptr + out_hstep * 7) = __msa_copy_s_w((v4i32)_f1, 3);
            }
            outptr += out_hstep * 8;
            if (pC0)
                pC0 += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __msa_fadd_w(_f0, _c0);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    v4f32 _c = (v4f32)__msa_ld_w(pC0, 0);
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c);
                }
            }
            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            if (out_hstep == 1)
            {
                __msa_st_w((v4i32)_f0, outptr, 0);
            }
            else
            {
                *(int*)outptr = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(outptr + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
                *(int*)(outptr + out_hstep * 2) = __msa_copy_s_w((v4i32)_f0, 2);
                *(int*)(outptr + out_hstep * 3) = __msa_copy_s_w((v4i32)_f0, 3);
            }
            outptr += out_hstep * 4;
            if (pC0)
                pC0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4i32 _fi = __msa_fill_w(0);
            _fi = __msa_insert_w(_fi, 0, ((const int*)pp)[0]);
            _fi = __msa_insert_w(_fi, 1, ((const int*)pp)[1]);
            v4f32 _f0 = (v4f32)_fi;
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __msa_fadd_w(_f0, _c0);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    v4i32 _ci = __msa_fill_w(0);
                    _ci = __msa_insert_w(_ci, 0, ((const int*)pC0)[0]);
                    _ci = __msa_insert_w(_ci, 1, ((const int*)pC0)[1]);
                    v4f32 _c = (v4f32)_ci;
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c);
                }
            }
            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            if (out_hstep == 1)
            {
                *(int64_t*)outptr = __msa_copy_s_d((v2i64)_f0, 0);
            }
            else
            {
                *(int*)outptr = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(outptr + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
            }
            outptr += out_hstep * 2;
            if (pC0)
                pC0 += 2;
        }
#endif // __mips_msa
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
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    float c0 = pC0[0];
                    float c1 = pC0[1];
                    if (beta != 1.f)
                    {
                        c0 *= beta;
                        c1 *= beta;
                    }
                    sum0 += c0;
                    sum1 += c1;
                }
            }
            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }
            outptr[0] = sum0;
            outptr[out_hstep] = sum1;
            outptr += out_hstep * 2;
            if (pC0)
                pC0 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float sum0 = *pp++;
            if (pC)
            {
                float c = 0.f;
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2) c = c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    c = pC0[0];
                    if (beta != 1.f) c *= beta;
                }
                sum0 += c;
            }
            if (alpha != 1.f) sum0 *= alpha;
            outptr[0] = sum0;
            outptr += out_hstep;
            if (pC0)
                pC0++;
        }
        outptr0++;
    }
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    const int tile_size = std::max(1, (int)((float)l2_cache_size / 2 / sizeof(signed char) / std::max(1, K)));

#if __mips_msa
    const int tile_m_align = 8;
    const int tile_n_align = 8;
#else
    const int tile_m_align = 4;
    const int tile_n_align = 2;
#endif
    // one driver M tile follows the natural producer slab
    TILE_M = tile_m_align;
    TILE_N = std::max(tile_n_align, tile_size / tile_n_align * tile_n_align);
    TILE_K = K;

    if (N > 0)
    {
        const int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + tile_n_align - 1) / tile_n_align * tile_n_align);
    }

    // always take constant TILE_N value when provided
    if (constant_TILE_N > 0)
        TILE_N = (constant_TILE_N + tile_n_align - 1) / tile_n_align * tile_n_align;

    (void)M;
    (void)constant_TILE_M;
    (void)constant_TILE_K;
    (void)nT;
}
