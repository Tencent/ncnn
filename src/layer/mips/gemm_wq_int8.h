// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
void pack_B_tile_wq_int8_loongson_mmi(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_loongson_mmi(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_loongson_mmi(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void gemm_transB_packed_tile_wq_int8_loongson_mmi(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

// group-major, output-major within each K4/K1 fragment
static void pack_B_tile_wq_int8(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        pack_B_tile_wq_int8_loongson_mmi(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

    const int block_count = (K + block_size - 1) / block_size;
    signed char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __mips_msa
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

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                v16i8 _p = (v16i8)__msa_set_w(__msa_load_w(p0), __msa_load_w(p1), __msa_load_w(p2), __msa_load_w(p3));
                __msa_st_b(_p, pp, 0);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }

            pd[0] = 1.f / *ps0++;
            pd[1] = 1.f / *ps1++;
            pd[2] = 1.f / *ps2++;
            pd[3] = 1.f / *ps3++;
            pd += 4;
        }
    }
#endif // __mips_msa

    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const signed char* p1 = B.row<const signed char>(j + jj + 1);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p1[0];
                pp[5] = p1[1];
                pp[6] = p1[2];
                pp[7] = p1[3];
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }

            *pd++ = 1.f / *ps0++;
            *pd++ = 1.f / *ps1++;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += 4;
            }
            for (; kk < max_kk; kk++)
                *pp++ = *p0++;

            *pd++ = 1.f / *ps0++;
        }
    }
}

// group-major, row-major within each K4/K1 fragment
static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_BF16
    if (A.elembits() == 16)
    {
        quantize_A_tile_wq_int8_bf16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif // NCNN_BF16

#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (A.elempack == 1 && ncnn::cpu_support_loongson_mmi())
    {
        quantize_A_tile_wq_int8_loongson_mmi(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if __mips_msa
    const int elempack = A.elempack;
#endif // __mips_msa
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
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k * elempack;
            const float* p1 = p0 + A_hstep * elempack;
            const float* p2 = 0;
            const float* p3 = 0;
            const float* p4 = 0;
            const float* p5 = 0;
            const float* p6 = 0;
            const float* p7 = 0;
            if (elempack == 1)
            {
                p2 = p1 + A_hstep;
                p3 = p2 + A_hstep;
                p4 = p3 + A_hstep;
                p5 = p4 + A_hstep;
                p6 = p5 + A_hstep;
                p7 = p6 + A_hstep;
            }

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                float absmax2 = 0.f;
                float absmax3 = 0.f;
                float absmax4 = 0.f;
                float absmax5 = 0.f;
                float absmax6 = 0.f;
                float absmax7 = 0.f;

                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

                const float* p0a = p0;
                const float* p1a = p1;
                const float* p2a = p2;
                const float* p3a = p3;
                const float* p4a = p4;
                const float* p5a = p5;
                const float* p6a = p6;
                const float* p7a = p7;
                int kk = 0;

                if (elempack == 4)
                {
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p1a, 0);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                    }

                    float absmax[8];
                    __msa_st_w((v4i32)_absmax0, absmax, 0);
                    __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                    absmax0 = absmax[0];
                    absmax1 = absmax[1];
                    absmax2 = absmax[2];
                    absmax3 = absmax[3];
                    absmax4 = absmax[4];
                    absmax5 = absmax[5];
                    absmax6 = absmax[6];
                    absmax7 = absmax[7];
                }

                if (elempack == 1)
                {
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p1a, 0);
                        v4f32 _p2 = (v4f32)__msa_ld_w(p2a, 0);
                        v4f32 _p3 = (v4f32)__msa_ld_w(p3a, 0);
                        v4f32 _p4 = (v4f32)__msa_ld_w(p4a, 0);
                        v4f32 _p5 = (v4f32)__msa_ld_w(p5a, 0);
                        v4f32 _p6 = (v4f32)__msa_ld_w(p6a, 0);
                        v4f32 _p7 = (v4f32)__msa_ld_w(p7a, 0);
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

                    absmax0 = __msa_reduce_fmax_w(_absmax0);
                    absmax1 = __msa_reduce_fmax_w(_absmax1);
                    absmax2 = __msa_reduce_fmax_w(_absmax2);
                    absmax3 = __msa_reduce_fmax_w(_absmax3);
                    absmax4 = __msa_reduce_fmax_w(_absmax4);
                    absmax5 = __msa_reduce_fmax_w(_absmax5);
                    absmax6 = __msa_reduce_fmax_w(_absmax6);
                    absmax7 = __msa_reduce_fmax_w(_absmax7);

                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(*p0a++));
                        absmax1 = std::max(absmax1, fabsf(*p1a++));
                        absmax2 = std::max(absmax2, fabsf(*p2a++));
                        absmax3 = std::max(absmax3, fabsf(*p3a++));
                        absmax4 = std::max(absmax4, fabsf(*p4a++));
                        absmax5 = std::max(absmax5, fabsf(*p5a++));
                        absmax6 = std::max(absmax6, fabsf(*p6a++));
                        absmax7 = std::max(absmax7, fabsf(*p7a++));
                    }
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

                if (elempack == 4)
                {
                    v4f32 _scale0 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                    v4f32 _scale1 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));

                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale0);
                        v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _scale0);
                        v4f32 _p2 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 8, 0), _scale0);
                        v4f32 _p3 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 12, 0), _scale0);
                        transpose4x4_ps(_p0, _p1, _p2, _p3);

                        v4f32 _p4 = __msa_fmul_w((v4f32)__msa_ld_w(p1, 0), _scale1);
                        v4f32 _p5 = __msa_fmul_w((v4f32)__msa_ld_w(p1 + 4, 0), _scale1);
                        v4f32 _p6 = __msa_fmul_w((v4f32)__msa_ld_w(p1 + 8, 0), _scale1);
                        v4f32 _p7 = __msa_fmul_w((v4f32)__msa_ld_w(p1 + 12, 0), _scale1);
                        transpose4x4_ps(_p4, _p5, _p6, _p7);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                        ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                        pp += 32;
                        p0 += 16;
                        p1 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale0);
                        v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p1, 0), _scale1);
                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        pp += 8;
                        p0 += 4;
                        p1 += 4;
                    }
                }

                if (elempack == 1)
                {
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
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                        v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                        v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
                        v4f32 _p4 = (v4f32)__msa_ld_w(p4, 0);
                        v4f32 _p5 = (v4f32)__msa_ld_w(p5, 0);
                        v4f32 _p6 = (v4f32)__msa_ld_w(p6, 0);
                        v4f32 _p7 = (v4f32)__msa_ld_w(p7, 0);
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
                        pp[0] = float2int8(*p0 * scale0);
                        pp[1] = float2int8(*p1 * scale1);
                        pp[2] = float2int8(*p2 * scale2);
                        pp[3] = float2int8(*p3 * scale3);
                        pp[4] = float2int8(*p4 * scale4);
                        pp[5] = float2int8(*p5 * scale5);
                        pp[6] = float2int8(*p6 * scale6);
                        pp[7] = float2int8(*p7 * scale7);
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
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k * elempack;
            const float* p1 = 0;
            const float* p2 = 0;
            const float* p3 = 0;
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

                const float* p0a = p0;
                const float* p1a = p1;
                const float* p2a = p2;
                const float* p3a = p3;
                int kk = 0;

                if (elempack == 4)
                {
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(p0a, 0);
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
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p1a, 0);
                        v4f32 _p2 = (v4f32)__msa_ld_w(p2a, 0);
                        v4f32 _p3 = (v4f32)__msa_ld_w(p3a, 0);
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
                        float v0 = *p0a++;
                        float v1 = *p1a++;
                        float v2 = *p2a++;
                        float v3 = *p3a++;
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
                        v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale);
                        v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _scale);
                        v4f32 _p2 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 8, 0), _scale);
                        v4f32 _p3 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 12, 0), _scale);
                        transpose4x4_ps(_p0, _p1, _p2, _p3);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        pp += 16;
                        p0 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale);
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
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                        v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                        v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
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
                        float v0 = *p0++;
                        float v1 = *p1++;
                        float v2 = *p2++;
                        float v3 = *p3++;
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
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;
            const float* p1 = p0 + A_hstep;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0;
                const float* p1a = p1;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = *p0a++;
                    float v1 = *p1a++;
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
                    float v00 = p0[0];
                    float v01 = p0[1];
                    float v02 = p0[2];
                    float v03 = p0[3];
                    float v10 = p1[0];
                    float v11 = p1[1];
                    float v12 = p1[2];
                    float v13 = p1[3];
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
                float absmax0 = 0.f;
                const float* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = *p0a++;
                    absmax0 = std::max(absmax0, fabsf(v0));
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                *pd++ = absmax0 / 127.f;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v0 = p0[0];
                    float v1 = p0[1];
                    float v2 = p0[2];
                    float v3 = p0[3];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    pp[2] = float2int8(v2 * scale0);
                    pp[3] = float2int8(v3 * scale0);
                    pp += 4;
                    p0 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0++;
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
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k * elempack;
        const float* p1 = p0 + A_hstep * elempack;
        const float* p2 = 0;
        const float* p3 = 0;
        const float* p4 = 0;
        const float* p5 = 0;
        const float* p6 = 0;
        const float* p7 = 0;
        if (elempack == 1)
        {
            p2 = p1 + A_hstep;
            p3 = p2 + A_hstep;
            p4 = p3 + A_hstep;
            p5 = p4 + A_hstep;
            p6 = p5 + A_hstep;
            p7 = p6 + A_hstep;
        }

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            float absmax2 = 0.f;
            float absmax3 = 0.f;
            float absmax4 = 0.f;
            float absmax5 = 0.f;
            float absmax6 = 0.f;
            float absmax7 = 0.f;

            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

            const float* p0a = p0;
            const float* p1a = p1;
            const float* p2a = p2;
            const float* p3a = p3;
            const float* p4a = p4;
            const float* p5a = p5;
            const float* p6a = p6;
            const float* p7a = p7;
            const float* psa = ps;
            int kk = 0;

            if (elempack == 4)
            {
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _s = __msa_fill_w_f32(*psa++);
                    v4f32 _p0 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a, 0), _abs_mask), _s);
                    v4f32 _p1 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p1a, 0), _abs_mask), _s);
                    _absmax0 = __msa_fmax_w(_absmax0, _p0);
                    _absmax1 = __msa_fmax_w(_absmax1, _p1);
                    p0a += 4;
                    p1a += 4;
                }

                float absmax[8];
                __msa_st_w((v4i32)_absmax0, absmax, 0);
                __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                absmax0 = absmax[0];
                absmax1 = absmax[1];
                absmax2 = absmax[2];
                absmax3 = absmax[3];
                absmax4 = absmax[4];
                absmax5 = absmax[5];
                absmax6 = absmax[6];
                absmax7 = absmax[7];
            }

            if (elempack == 1)
            {
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1a, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(p2a, 0);
                    v4f32 _p3 = (v4f32)__msa_ld_w(p3a, 0);
                    v4f32 _p4 = (v4f32)__msa_ld_w(p4a, 0);
                    v4f32 _p5 = (v4f32)__msa_ld_w(p5a, 0);
                    v4f32 _p6 = (v4f32)__msa_ld_w(p6a, 0);
                    v4f32 _p7 = (v4f32)__msa_ld_w(p7a, 0);
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

                absmax0 = __msa_reduce_fmax_w(_absmax0);
                absmax1 = __msa_reduce_fmax_w(_absmax1);
                absmax2 = __msa_reduce_fmax_w(_absmax2);
                absmax3 = __msa_reduce_fmax_w(_absmax3);
                absmax4 = __msa_reduce_fmax_w(_absmax4);
                absmax5 = __msa_reduce_fmax_w(_absmax5);
                absmax6 = __msa_reduce_fmax_w(_absmax6);
                absmax7 = __msa_reduce_fmax_w(_absmax7);

                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(*p0a++) * s);
                    absmax1 = std::max(absmax1, fabsf(*p1a++) * s);
                    absmax2 = std::max(absmax2, fabsf(*p2a++) * s);
                    absmax3 = std::max(absmax3, fabsf(*p3a++) * s);
                    absmax4 = std::max(absmax4, fabsf(*p4a++) * s);
                    absmax5 = std::max(absmax5, fabsf(*p5a++) * s);
                    absmax6 = std::max(absmax6, fabsf(*p6a++) * s);
                    absmax7 = std::max(absmax7, fabsf(*p7a++) * s);
                }
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

            if (elempack == 4)
            {
                v4f32 _scale0 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                v4f32 _scale1 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));

                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s0 = __msa_fill_w_f32(ps[0]);
                    v4f32 _s1 = __msa_fill_w_f32(ps[1]);
                    v4f32 _s2 = __msa_fill_w_f32(ps[2]);
                    v4f32 _s3 = __msa_fill_w_f32(ps[3]);
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _s0), _scale0);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _s1), _scale0);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 8, 0), _s2), _scale0);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 12, 0), _s3), _scale0);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);

                    v4f32 _p4 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p1, 0), _s0), _scale1);
                    v4f32 _p5 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p1 + 4, 0), _s1), _scale1);
                    v4f32 _p6 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p1 + 8, 0), _s2), _scale1);
                    v4f32 _p7 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p1 + 12, 0), _s3), _scale1);
                    transpose4x4_ps(_p4, _p5, _p6, _p7);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                    ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                    pp += 32;
                    p0 += 16;
                    p1 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _s = __msa_fill_w_f32(*ps++);
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _s), _scale0);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p1, 0), _s), _scale1);
                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
            }

            if (elempack == 1)
            {
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
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                    v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
                    v4f32 _p4 = (v4f32)__msa_ld_w(p4, 0);
                    v4f32 _p5 = (v4f32)__msa_ld_w(p5, 0);
                    v4f32 _p6 = (v4f32)__msa_ld_w(p6, 0);
                    v4f32 _p7 = (v4f32)__msa_ld_w(p7, 0);
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
                    pp[0] = float2int8(*p0 * s * scale0);
                    pp[1] = float2int8(*p1 * s * scale1);
                    pp[2] = float2int8(*p2 * s * scale2);
                    pp[3] = float2int8(*p3 * s * scale3);
                    pp[4] = float2int8(*p4 * s * scale4);
                    pp[5] = float2int8(*p5 * s * scale5);
                    pp[6] = float2int8(*p6 * s * scale6);
                    pp[7] = float2int8(*p7 * s * scale7);
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
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k * elempack;
        const float* p1 = 0;
        const float* p2 = 0;
        const float* p3 = 0;
        if (elempack == 1)
        {
            p1 = p0 + A_hstep;
            p2 = p1 + A_hstep;
            p3 = p2 + A_hstep;
        }

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

            const float* p0a = p0;
            const float* p1a = p1;
            const float* p2a = p2;
            const float* p3a = p3;
            const float* psa = ps;
            int kk = 0;

            if (elempack == 4)
            {
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _s = __msa_fill_w_f32(*psa++);
                    v4f32 _p = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a, 0), _abs_mask), _s);
                    _absmax0 = __msa_fmax_w(_absmax0, _p);
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
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1a, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(p2a, 0);
                    v4f32 _p3 = (v4f32)__msa_ld_w(p3a, 0);
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
                    float v0 = *p0a++;
                    float v1 = *p1a++;
                    float v2 = *p2a++;
                    float v3 = *p3a++;
                    const float s = *psa++;

                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    absmax1 = std::max(absmax1, fabsf(v1) * s);
                    absmax2 = std::max(absmax2, fabsf(v2) * s);
                    absmax3 = std::max(absmax3, fabsf(v3) * s);
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
                    v4f32 _s0 = __msa_fill_w_f32(ps[0]);
                    v4f32 _s1 = __msa_fill_w_f32(ps[1]);
                    v4f32 _s2 = __msa_fill_w_f32(ps[2]);
                    v4f32 _s3 = __msa_fill_w_f32(ps[3]);
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _s0), _scale);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _s1), _scale);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 8, 0), _s2), _scale);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 12, 0), _s3), _scale);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    pp += 16;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _s = __msa_fill_w_f32(*ps++);
                    v4f32 _p = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _s), _scale);
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
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                    v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
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
                    float v0 = *p0++;
                    float v1 = *p1++;
                    float v2 = *p2++;
                    float v3 = *p3++;
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
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = *p0a++;
                float v1 = *p1a++;
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
                float v00 = p0[0];
                float v01 = p0[1];
                float v02 = p0[2];
                float v03 = p0[3];
                float v10 = p1[0];
                float v11 = p1[1];
                float v12 = p1[2];
                float v13 = p1[3];
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
            float absmax0 = 0.f;
            const float* p0a = p0;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = *p0a++;
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(v0) * s);
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            *pd++ = absmax0 / 127.f;

            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v0 = p0[0];
                float v1 = p0[1];
                float v2 = p0[2];
                float v3 = p0[3];
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
                float v0 = *p0++;
                v0 *= *ps++;
                *pp++ = float2int8(v0 * scale0);
            }
        }
    }
}

// group-major, row-major within each K4/K1 fragment
static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_BF16
    if (A.elembits() == 16)
    {
        transpose_quantize_A_tile_wq_int8_bf16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif // NCNN_BF16

#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (A.elempack == 1 && ncnn::cpu_support_loongson_mmi())
    {
        transpose_quantize_A_tile_wq_int8_loongson_mmi(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

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
            const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

                float absmax[8] = {0.f};
                const float* p0a = p0;
                int kk = 0;

                if (elempack == 4)
                {
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a, 0), _abs_mask);
                        v4f32 _p1 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 4, 0), _abs_mask);
                        v4f32 _p2 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 8, 0), _abs_mask);
                        v4f32 _p3 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 12, 0), _abs_mask);
                        v4f32 _p4 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 16, 0), _abs_mask);
                        v4f32 _p5 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 20, 0), _abs_mask);
                        v4f32 _p6 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 24, 0), _abs_mask);
                        v4f32 _p7 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 28, 0), _abs_mask);
                        absmax[0] = std::max(absmax[0], __msa_reduce_fmax_w(_p0));
                        absmax[1] = std::max(absmax[1], __msa_reduce_fmax_w(_p1));
                        absmax[2] = std::max(absmax[2], __msa_reduce_fmax_w(_p2));
                        absmax[3] = std::max(absmax[3], __msa_reduce_fmax_w(_p3));
                        absmax[4] = std::max(absmax[4], __msa_reduce_fmax_w(_p4));
                        absmax[5] = std::max(absmax[5], __msa_reduce_fmax_w(_p5));
                        absmax[6] = std::max(absmax[6], __msa_reduce_fmax_w(_p6));
                        absmax[7] = std::max(absmax[7], __msa_reduce_fmax_w(_p7));
                        p0a += A_hstep * 4;
                    }
                }

                if (elempack == 1)
                {
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p0a + 4, 0);
                        _absmax0 = __msa_fmax_w(_absmax0, (v4f32)__msa_and_v((v16u8)_p0, _abs_mask));
                        _absmax1 = __msa_fmax_w(_absmax1, (v4f32)__msa_and_v((v16u8)_p1, _abs_mask));
                        p0a += A_hstep;
                    }

                    __msa_st_w((v4i32)_absmax0, absmax, 0);
                    __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
                }
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

                if (elempack == 4)
                {
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale0);
                        v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _scale1);
                        v4f32 _p2 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 8, 0), _scale2);
                        v4f32 _p3 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 12, 0), _scale3);
                        v4f32 _p4 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 16, 0), _scale4);
                        v4f32 _p5 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 20, 0), _scale5);
                        v4f32 _p6 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 24, 0), _scale6);
                        v4f32 _p7 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 28, 0), _scale7);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                        ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                        pp += 32;
                        p0 += A_hstep * 4;
                    }
                }

                if (elempack == 1)
                {
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const float* p1 = p0 + A_hstep;
                        const float* p2 = p1 + A_hstep;
                        const float* p3 = p2 + A_hstep;
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                        v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                        v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
                        transpose4x4_ps(_p0, _p1, _p2, _p3);

                        v4f32 _p4 = (v4f32)__msa_ld_w(p0 + 4, 0);
                        v4f32 _p5 = (v4f32)__msa_ld_w(p1 + 4, 0);
                        v4f32 _p6 = (v4f32)__msa_ld_w(p2 + 4, 0);
                        v4f32 _p7 = (v4f32)__msa_ld_w(p3 + 4, 0);
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
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p0 + 4, 0);
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
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax = (v4f32)__msa_fill_w(0);

                float absmax[4] = {0.f};
                const float* p0a = p0;
                int kk = 0;

                if (elempack == 4)
                {
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a, 0), _abs_mask);
                        v4f32 _p1 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 4, 0), _abs_mask);
                        v4f32 _p2 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 8, 0), _abs_mask);
                        v4f32 _p3 = (v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 12, 0), _abs_mask);
                        absmax[0] = std::max(absmax[0], __msa_reduce_fmax_w(_p0));
                        absmax[1] = std::max(absmax[1], __msa_reduce_fmax_w(_p1));
                        absmax[2] = std::max(absmax[2], __msa_reduce_fmax_w(_p2));
                        absmax[3] = std::max(absmax[3], __msa_reduce_fmax_w(_p3));
                        p0a += A_hstep * 4;
                    }
                }

                if (elempack == 1)
                {
                    for (; kk < max_kk0; kk++)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(p0a, 0);
                        _absmax = __msa_fmax_w(_absmax, (v4f32)__msa_and_v((v16u8)_p, _abs_mask));
                        p0a += A_hstep;
                    }

                    __msa_st_w((v4i32)_absmax, absmax, 0);
                }
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

                if (elempack == 4)
                {
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale0);
                        v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _scale1);
                        v4f32 _p2 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 8, 0), _scale2);
                        v4f32 _p3 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 12, 0), _scale3);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
                }

                if (elempack == 1)
                {
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const float* p1 = p0 + A_hstep;
                        const float* p2 = p1 + A_hstep;
                        const float* p3 = p2 + A_hstep;
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                        v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                        v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
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
                        float v0 = p0[0];
                        float v1 = p0[1];
                        float v2 = p0[2];
                        float v3 = p0[3];
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
            const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0;

#if __mips_msa
                if (elempack == 4)
                {
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(p0a + 4, 0);
                        absmax0 = std::max(absmax0, __msa_reduce_fmax_w((v4f32)__msa_and_v((v16u8)_p0, _abs_mask)));
                        absmax1 = std::max(absmax1, __msa_reduce_fmax_w((v4f32)__msa_and_v((v16u8)_p1, _abs_mask)));
                        p0a += A_hstep * 4;
                    }
                }
#endif // __mips_msa

                if (elempack == 1)
                {
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = p0a[0];
                        float v1 = p0a[1];
                        absmax0 = std::max(absmax0, fabsf(v0));
                        absmax1 = std::max(absmax1, fabsf(v1));
                        p0a += A_hstep;
                    }
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

#if __mips_msa
                if (elempack == 4)
                {
                    v4f32 _scale0 = __msa_fill_w_f32(scale0);
                    v4f32 _scale1 = __msa_fill_w_f32(scale1);
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale0);
                        v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _scale1);
                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        pp += 8;
                        p0 += A_hstep * 4;
                    }
                }
#endif // __mips_msa

                if (elempack == 1)
                {
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float v00 = p0[0];
                        float v10 = p0[1];
                        float v01 = p0[A_hstep];
                        float v11 = p0[A_hstep + 1];
                        float v02 = p0[A_hstep * 2];
                        float v12 = p0[A_hstep * 2 + 1];
                        float v03 = p0[A_hstep * 3];
                        float v13 = p0[A_hstep * 3 + 1];
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
                        float v0 = p0[0];
                        float v1 = p0[1];
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
            const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                const float* p0a = p0;

#if __mips_msa
                if (elempack == 4)
                {
                    const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(p0a, 0);
                        absmax0 = std::max(absmax0, __msa_reduce_fmax_w((v4f32)__msa_and_v((v16u8)_p, _abs_mask)));
                        p0a += A_hstep * 4;
                    }
                }
#endif // __mips_msa

                if (elempack == 1)
                {
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = *p0a;
                        absmax0 = std::max(absmax0, fabsf(v0));
                        p0a += A_hstep;
                    }
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                *pd++ = absmax0 / 127.f;

#if __mips_msa
                if (elempack == 4)
                {
                    v4f32 _scale = __msa_fill_w_f32(scale0);
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        v4f32 _p = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale);
                        ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                        pp += 4;
                        p0 += A_hstep * 4;
                    }
                }
#endif // __mips_msa

                if (elempack == 1)
                {
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float v0 = p0[0];
                        float v1 = p0[A_hstep];
                        float v2 = p0[A_hstep * 2];
                        float v3 = p0[A_hstep * 3];
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale0);
                        pp[2] = float2int8(v2 * scale0);
                        pp[3] = float2int8(v3 * scale0);
                        p0 += A_hstep * 4;
                        pp += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = *p0;
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
        const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

            float absmax[8] = {0.f};
            const float* p0a = p0;
            const float* psa = ps;
            int kk = 0;

            if (elempack == 4)
            {
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _p0 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a, 0), _abs_mask), _s);
                    v4f32 _p1 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 4, 0), _abs_mask), _s);
                    v4f32 _p2 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 8, 0), _abs_mask), _s);
                    v4f32 _p3 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 12, 0), _abs_mask), _s);
                    v4f32 _p4 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 16, 0), _abs_mask), _s);
                    v4f32 _p5 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 20, 0), _abs_mask), _s);
                    v4f32 _p6 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 24, 0), _abs_mask), _s);
                    v4f32 _p7 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 28, 0), _abs_mask), _s);
                    absmax[0] = std::max(absmax[0], __msa_reduce_fmax_w(_p0));
                    absmax[1] = std::max(absmax[1], __msa_reduce_fmax_w(_p1));
                    absmax[2] = std::max(absmax[2], __msa_reduce_fmax_w(_p2));
                    absmax[3] = std::max(absmax[3], __msa_reduce_fmax_w(_p3));
                    absmax[4] = std::max(absmax[4], __msa_reduce_fmax_w(_p4));
                    absmax[5] = std::max(absmax[5], __msa_reduce_fmax_w(_p5));
                    absmax[6] = std::max(absmax[6], __msa_reduce_fmax_w(_p6));
                    absmax[7] = std::max(absmax[7], __msa_reduce_fmax_w(_p7));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
            }

            if (elempack == 1)
            {
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p0a + 4, 0);
                    v4f32 _s = __msa_fill_w_f32(*psa++);
                    _p0 = (v4f32)__msa_and_v((v16u8)_p0, _abs_mask);
                    _p0 = __msa_fmul_w(_p0, _s);
                    _p1 = (v4f32)__msa_and_v((v16u8)_p1, _abs_mask);
                    _p1 = __msa_fmul_w(_p1, _s);
                    _absmax0 = __msa_fmax_w(_absmax0, _p0);
                    _absmax1 = __msa_fmax_w(_absmax1, _p1);
                    p0a += A_hstep;
                }

                __msa_st_w((v4i32)_absmax0, absmax, 0);
                __msa_st_w((v4i32)_absmax1, absmax + 4, 0);
            }
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

            if (elempack == 4)
            {
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _s), _scale0);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _s), _scale1);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 8, 0), _s), _scale2);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 12, 0), _s), _scale3);
                    v4f32 _p4 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 16, 0), _s), _scale4);
                    v4f32 _p5 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 20, 0), _s), _scale5);
                    v4f32 _p6 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 24, 0), _s), _scale6);
                    v4f32 _p7 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 28, 0), _s), _scale7);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                    ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                    pp += 32;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }

            if (elempack == 1)
            {
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const float* p1 = p0 + A_hstep;
                    const float* p2 = p1 + A_hstep;
                    const float* p3 = p2 + A_hstep;
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                    v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);

                    v4f32 _p4 = (v4f32)__msa_ld_w(p0 + 4, 0);
                    v4f32 _p5 = (v4f32)__msa_ld_w(p1 + 4, 0);
                    v4f32 _p6 = (v4f32)__msa_ld_w(p2 + 4, 0);
                    v4f32 _p7 = (v4f32)__msa_ld_w(p3 + 4, 0);
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
                    v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), __msa_fill_w_f32(s));
                    v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), __msa_fill_w_f32(s));
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
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax = (v4f32)__msa_fill_w(0);

            float absmax[4] = {0.f};
            const float* p0a = p0;
            const float* psa = ps;
            int kk = 0;

            if (elempack == 4)
            {
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _p0 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a, 0), _abs_mask), _s);
                    v4f32 _p1 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 4, 0), _abs_mask), _s);
                    v4f32 _p2 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 8, 0), _abs_mask), _s);
                    v4f32 _p3 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 12, 0), _abs_mask), _s);
                    absmax[0] = std::max(absmax[0], __msa_reduce_fmax_w(_p0));
                    absmax[1] = std::max(absmax[1], __msa_reduce_fmax_w(_p1));
                    absmax[2] = std::max(absmax[2], __msa_reduce_fmax_w(_p2));
                    absmax[3] = std::max(absmax[3], __msa_reduce_fmax_w(_p3));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
            }

            if (elempack == 1)
            {
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _p = (v4f32)__msa_ld_w(p0a, 0);
                    _p = (v4f32)__msa_and_v((v16u8)_p, _abs_mask);
                    _p = __msa_fmul_w(_p, __msa_fill_w_f32(*psa++));
                    _absmax = __msa_fmax_w(_absmax, _p);
                    p0a += A_hstep;
                }

                __msa_st_w((v4i32)_absmax, absmax, 0);
            }
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

            if (elempack == 4)
            {
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _s), _scale0);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _s), _scale1);
                    v4f32 _p2 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 8, 0), _s), _scale2);
                    v4f32 _p3 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 12, 0), _s), _scale3);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    pp += 16;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }

            if (elempack == 1)
            {
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const float* p1 = p0 + A_hstep;
                    const float* p2 = p1 + A_hstep;
                    const float* p3 = p2 + A_hstep;
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                    v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
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
                    float v0 = p0[0];
                    float v1 = p0[1];
                    float v2 = p0[2];
                    float v3 = p0[3];
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
        const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0;
            const float* psa = ps;

#if __mips_msa
            if (elempack == 4)
            {
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _p0 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a, 0), _abs_mask), _s);
                    v4f32 _p1 = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a + 4, 0), _abs_mask), _s);
                    absmax0 = std::max(absmax0, __msa_reduce_fmax_w(_p0));
                    absmax1 = std::max(absmax1, __msa_reduce_fmax_w(_p1));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
            }
#endif // __mips_msa

            if (elempack == 1)
            {
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = p0a[0];
                    float v1 = p0a[1];
                    const float s = *psa++;

                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    absmax1 = std::max(absmax1, fabsf(v1) * s);
                    p0a += A_hstep;
                }
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

#if __mips_msa
            if (elempack == 4)
            {
                v4f32 _scale0 = __msa_fill_w_f32(scale0);
                v4f32 _scale1 = __msa_fill_w_f32(scale1);
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _p0 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _s), _scale0);
                    v4f32 _p1 = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0 + 4, 0), _s), _scale1);
                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    pp += 8;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
#endif // __mips_msa

            if (elempack == 1)
            {
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v00 = p0[0];
                    float v10 = p0[1];
                    float v01 = p0[A_hstep];
                    float v11 = p0[A_hstep + 1];
                    float v02 = p0[A_hstep * 2];
                    float v12 = p0[A_hstep * 2 + 1];
                    float v03 = p0[A_hstep * 3];
                    float v13 = p0[A_hstep * 3 + 1];
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
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            const float* p0a = p0;
            const float* psa = ps;

#if __mips_msa
            if (elempack == 4)
            {
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s = (v4f32)__msa_ld_w(psa, 0);
                    v4f32 _p = __msa_fmul_w((v4f32)__msa_and_v((v16u8)__msa_ld_w(p0a, 0), _abs_mask), _s);
                    absmax0 = std::max(absmax0, __msa_reduce_fmax_w(_p));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
            }
#endif // __mips_msa

            if (elempack == 1)
            {
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = *p0a;
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    p0a += A_hstep;
                }
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            *pd++ = absmax0 / 127.f;

#if __mips_msa
            if (elempack == 4)
            {
                v4f32 _scale = __msa_fill_w_f32(scale0);
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _s = (v4f32)__msa_ld_w(ps, 0);
                    v4f32 _p = __msa_fmul_w(__msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _s), _scale);
                    ((int*)pp)[0] = __msa_copy_s_w((v4i32)float2int8(_p), 0);
                    pp += 4;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
#endif // __mips_msa

            if (elempack == 1)
            {
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v0 = p0[0];
                    float v1 = p0[A_hstep];
                    float v2 = p0[A_hstep * 2];
                    float v3 = p0[A_hstep * 3];
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
                    float v0 = *p0;
                    v0 *= *ps++;
                    *pp++ = float2int8(v0 * scale0);
                    p0 += A_hstep;
                }
            }
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        gemm_transB_packed_tile_wq_int8_loongson_mmi(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

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
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        const v8i16 _one = __msa_fill_h(1);

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            v4f32 _fsum2;
            v4f32 _fsum3;
            v4f32 _fsum4;
            v4f32 _fsum5;
            v4f32 _fsum6;
            v4f32 _fsum7;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
                _fsum2 = (v4f32)__msa_fill_w(0);
                _fsum3 = (v4f32)__msa_fill_w(0);
                _fsum4 = (v4f32)__msa_fill_w(0);
                _fsum5 = (v4f32)__msa_fill_w(0);
                _fsum6 = (v4f32)__msa_fill_w(0);
                _fsum7 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum4 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _fsum5 = (v4f32)__msa_ld_w(outptr + 12, 0);
                _fsum2 = (v4f32)__msa_ld_w(outptr + 16, 0);
                _fsum6 = (v4f32)__msa_ld_w(outptr + 20, 0);
                _fsum3 = (v4f32)__msa_ld_w(outptr + 24, 0);
                _fsum7 = (v4f32)__msa_ld_w(outptr + 28, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                v4i32 _sum4 = __msa_fill_w(0);
                v4i32 _sum5 = __msa_fill_w(0);
                v4i32 _sum6 = __msa_fill_w(0);
                v4i32 _sum7 = __msa_fill_w(0);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 64);
                    v16i8 _pA0 = __msa_ld_b(pA, 0);
                    v16i8 _pA0r = (v16i8)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                    v16i8 _pB = __msa_ld_b(pB, 0);
                    v16i8 _pBr = (v16i8)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA0, _pBr), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA0r, _pB), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA0r, _pBr), _one);

                    v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    v16i8 _pA1r = (v16i8)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                    _sum4 = __msa_dpadd_s_w(_sum4, __msa_dotp_s_h(_pA1, _pB), _one);
                    _sum5 = __msa_dpadd_s_w(_sum5, __msa_dotp_s_h(_pA1, _pBr), _one);
                    _sum6 = __msa_dpadd_s_w(_sum6, __msa_dotp_s_h(_pA1r, _pB), _one);
                    _sum7 = __msa_dpadd_s_w(_sum7, __msa_dotp_s_h(_pA1r, _pBr), _one);
                    pA += 32;
                    pB += 16;
                }

                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA0 = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                    v8i16 _pA1 = (v8i16)__msa_fill_w(*(const int*)(pA + 4));
                    _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);
                    v8i16 _pA0r = __msa_shf_h(_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                    v8i16 _pA1r = __msa_shf_h(_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                    v8i16 _pB0 = (v8i16)__msa_fill_w(*(const int*)pB);
                    _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                    v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                    v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA0, _pB0r);
                    v8i16 _s2 = __msa_mulv_h(_pA0r, _pB0);
                    v8i16 _s3 = __msa_mulv_h(_pA0r, _pB0r);
                    v8i16 _s4 = __msa_mulv_h(_pA1, _pB0);
                    v8i16 _s5 = __msa_mulv_h(_pA1, _pB0r);
                    v8i16 _s6 = __msa_mulv_h(_pA1r, _pB0);
                    v8i16 _s7 = __msa_mulv_h(_pA1r, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                    _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s4, 0), _s4));
                    _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s5, 0), _s5));
                    _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s6, 0), _s6));
                    _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s7, 0), _s7));
                    pA += 8;
                    pB += 4;
                }

                v4f32 _descaleB = (v4f32)__msa_ld_w(pB_descales, 0);
                v4f32 _descaleBr = (v4f32)__msa_shf_w((v4i32)_descaleB, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _descaleA0 = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleA1 = (v4f32)__msa_ld_w(pA_descales + 4, 0);
                v4f32 _descaleA0r = (v4f32)__msa_shf_w((v4i32)_descaleA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _descaleA1r = (v4f32)__msa_shf_w((v4i32)_descaleA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _scale = __msa_fmul_w(_descaleA0, _descaleB);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA0, _descaleBr);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleA0r, _descaleB);
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleA0r, _descaleBr);
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                _scale = __msa_fmul_w(_descaleA1, _descaleB);
                _fsum4 = __ncnn_msa_fmadd_w(_fsum4, (v4f32)__msa_ffint_s_w(_sum4), _scale);
                _scale = __msa_fmul_w(_descaleA1, _descaleBr);
                _fsum5 = __ncnn_msa_fmadd_w(_fsum5, (v4f32)__msa_ffint_s_w(_sum5), _scale);
                _scale = __msa_fmul_w(_descaleA1r, _descaleB);
                _fsum6 = __ncnn_msa_fmadd_w(_fsum6, (v4f32)__msa_ffint_s_w(_sum6), _scale);
                _scale = __msa_fmul_w(_descaleA1r, _descaleBr);
                _fsum7 = __ncnn_msa_fmadd_w(_fsum7, (v4f32)__msa_ffint_s_w(_sum7), _scale);
                pA_descales += 8;
                pB_descales += 4;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum4, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum5, outptr + 12, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 16, 0);
            __msa_st_w((v4i32)_fsum6, outptr + 20, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 24, 0);
            __msa_st_w((v4i32)_fsum7, outptr + 28, 0);
            outptr += 32;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            v4f32 _fsum2;
            v4f32 _fsum3;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
                _fsum2 = (v4f32)__msa_fill_w(0);
                _fsum3 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum2 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _fsum3 = (v4f32)__msa_ld_w(outptr + 12, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA0 = __msa_ld_b(pA, 0);
                    v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    v16i8 _pBr = (v16i8)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA0, _pBr), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA1, _pB), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA1, _pBr), _one);
                    pA += 32;
                    pB += 8;
                }

                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA0 = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                    v8i16 _pA1 = (v8i16)__msa_fill_w(*(const int*)(pA + 4));
                    _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);
                    int b01 = (unsigned char)pB[0] | ((unsigned char)pB[1] << 8);
                    v8i16 _pB0 = (v8i16)__msa_fill_w(b01);
                    _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                    _pB0 = __msa_shf_h(_pB0, _MSA_SHUFFLE(1, 0, 1, 0));
                    v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                    v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA0, _pB0r);
                    v8i16 _s2 = __msa_mulv_h(_pA1, _pB0);
                    v8i16 _s3 = __msa_mulv_h(_pA1, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                    pA += 8;
                    pB += 2;
                }

                v4f32 _descaleA0 = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleA1 = (v4f32)__msa_ld_w(pA_descales + 4, 0);
                v4f32 _descaleB = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1));
                v4f32 _descaleBr = (v4f32)__msa_shf_w((v4i32)_descaleB, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _scale = __msa_fmul_w(_descaleA0, _descaleB);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA0, _descaleBr);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleA1, _descaleB);
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleA1, _descaleBr);
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                pA_descales += 8;
                pB_descales += 2;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            outptr += 16;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA0 = __msa_ld_b(pA, 0);
                    v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                    v16i8 _pB = (v16i8)__msa_fill_w(*(const int*)pB);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA0, _pB), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA1, _pB), _one);
                    pA += 32;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_d_ptr(pA);
                    v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    v8i16 _s = __msa_mulv_h(_pA, __msa_fill_h(pB[0]));
                    v8i16 _sign = __msa_clti_s_h(_s, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s));
                    pA += 8;
                    pB++;
                }

                v4f32 _descaleA0 = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleA1 = (v4f32)__msa_ld_w(pA_descales + 4, 0);
                v4f32 _scale = __msa_fmul_w(_descaleA0, __msa_fill_w_f32(pB_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA1, __msa_fill_w_f32(pB_descales[0]));
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                pA_descales += 8;
                pB_descales++;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += (size_t)8 * A_hstep;
        pAT_descales += (size_t)8 * A_descales_hstep;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        const v8i16 _one = __msa_fill_h(1);

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            v4f32 _fsum2;
            v4f32 _fsum3;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
                _fsum2 = (v4f32)__msa_fill_w(0);
                _fsum3 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _fsum2 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _fsum3 = (v4f32)__msa_ld_w(outptr + 12, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 64);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pAr = (v16i8)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    v16i8 _pB0 = __msa_ld_b(pB, 0);
                    v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pAr, _pB0), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pAr, _pB0r), _one);
                    pA += 16;
                    pB += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    v8i16 _pAr = __msa_shf_h(_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    v8i16 _pB0 = (v8i16)__msa_fill_w(*(const int*)pB);
                    _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                    v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                    v8i16 _s2 = __msa_mulv_h(_pAr, _pB0);
                    v8i16 _s3 = __msa_mulv_h(_pAr, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                    pA += 4;
                    pB += 4;
                }
                v4f32 _descaleB = (v4f32)__msa_ld_w(pB_descales, 0);
                v4f32 _descaleBr = (v4f32)__msa_shf_w((v4i32)_descaleB, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _descaleA = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleAr = (v4f32)__msa_shf_w((v4i32)_descaleA, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleBr);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleAr, _descaleB);
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleAr, _descaleBr);
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                pA_descales += 4;
                pB_descales += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            outptr += 16;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pB0 = (v16i8)__msa_fill_d_ptr(pB);
                    v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                    pA += 16;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    v16i8 _pB8 = (v16i8)__msa_fill_h(*(const short*)pB);
                    v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    pA += 4;
                    pB += 2;
                }
                v4f32 _descaleA = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleB = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1));
                v4f32 _descaleBr = (v4f32)__msa_shf_w((v4i32)_descaleB, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleBr);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                pA_descales += 4;
                pB_descales += 2;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pB0 = (v16i8)__msa_fill_w(*(const int*)pB);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    pA += 16;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                    v8i16 _pB0 = __msa_fill_h(pB[0]);
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    pA += 4;
                    pB++;
                }
                v4f32 _descaleA = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _scale = __msa_fmul_w(_descaleA, __msa_fill_w_f32(pB_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                pA_descales += 4;
                pB_descales++;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            outptr += 4;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += (size_t)4 * A_hstep;
        pAT_descales += (size_t)4 * A_descales_hstep;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __mips_msa
        const v8i16 _one = __msa_fill_h(1);
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            v4f32 _fsum1;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
                _fsum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    v16i8 _pB0 = __msa_ld_b(pB, 0);
                    v16i8 _pB01 = (v16i8)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                    v16i8 _pB23 = (v16i8)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB01), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB23), _one);
                    pA += 8;
                    pB += 16;
                }
                int sum00 = __msa_copy_s_w(_sum0, 0);
                int sum01 = __msa_copy_s_w(_sum0, 1);
                int sum10 = __msa_copy_s_w(_sum0, 2);
                int sum11 = __msa_copy_s_w(_sum0, 3);
                int sum20 = __msa_copy_s_w(_sum1, 0);
                int sum21 = __msa_copy_s_w(_sum1, 1);
                int sum30 = __msa_copy_s_w(_sum1, 2);
                int sum31 = __msa_copy_s_w(_sum1, 3);
                for (; kk < max_kk0; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum01 += pA[1] * pB[0];
                    sum10 += pA[0] * pB[1];
                    sum11 += pA[1] * pB[1];
                    sum20 += pA[0] * pB[2];
                    sum21 += pA[1] * pB[2];
                    sum30 += pA[0] * pB[3];
                    sum31 += pA[1] * pB[3];
                    pA += 2;
                    pB += 4;
                }
                _sum0 = __msa_set_w(sum00, sum01, sum10, sum11);
                _sum1 = __msa_set_w(sum20, sum21, sum30, sum31);
                v4f32 _descaleA = (v4f32)__msa_fill_d_ptr(pA_descales);
                v4f32 _descaleB0 = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales + 1));
                v4f32 _descaleB1 = (v4f32)__msa_set_w(__msa_load_w(pB_descales + 2), __msa_load_w(pB_descales + 2), __msa_load_w(pB_descales + 3), __msa_load_w(pB_descales + 3));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB0);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleB1);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                pA_descales += 2;
                pB_descales += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
#if __mips_msa
            v4f32 _fsum;
            if (k == 0)
            {
                _fsum = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum = (v4f32)__msa_ld_w(outptr, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    v16i8 _pB01 = (v16i8)__msa_ilvr_w((v4i32)_pB, (v4i32)_pB);
                    _sum = __msa_dpadd_s_w(_sum, __msa_dotp_s_h(_pA, _pB01), _one);
                    pA += 8;
                    pB += 8;
                }
                int sum00 = __msa_copy_s_w(_sum, 0);
                int sum01 = __msa_copy_s_w(_sum, 1);
                int sum10 = __msa_copy_s_w(_sum, 2);
                int sum11 = __msa_copy_s_w(_sum, 3);
                for (; kk < max_kk0; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum01 += pA[1] * pB[0];
                    sum10 += pA[0] * pB[1];
                    sum11 += pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                }
                _sum = __msa_set_w(sum00, sum01, sum10, sum11);
                v4f32 _descaleA = (v4f32)__msa_fill_d_ptr(pA_descales);
                v4f32 _descaleB = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales + 1));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB);
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales += 2;
                pB_descales += 2;
            }
            __msa_st_w((v4i32)_fsum, outptr, 0);
#else
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
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
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
                const int8x8_t _zero = __mmi_pzerob_s();
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk0; kk += 4)
                {
#if __mips_loongson_mmi
                    __builtin_prefetch(pB + 32);
                    int8x8_t _pA = __mmi_pldb_s(pA);
                    int8x8_t _pB = __mmi_pldb_s(pB);
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
#else
                    sum00_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum01_i += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    sum10_i += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    sum11_i += pA[4] * pB[4] + pA[5] * pB[5] + pA[6] * pB[6] + pA[7] * pB[7];
                    pA += 8;
                    pB += 8;
#endif // __mips_loongson_mmi
                }
#if __mips_loongson_mmi
                int tmp[2];
                __mmi_pstw_s(tmp, _sum00);
                sum00_i += tmp[0] + tmp[1];
                __mmi_pstw_s(tmp, _sum01);
                sum01_i += tmp[0] + tmp[1];
                __mmi_pstw_s(tmp, _sum10);
                sum10_i += tmp[0] + tmp[1];
                __mmi_pstw_s(tmp, _sum11);
                sum11_i += tmp[0] + tmp[1];
#endif // __mips_loongson_mmi
                for (; kk < max_kk0; kk++)
                {
                    sum00_i += pA[0] * pB[0];
                    sum01_i += pA[1] * pB[0];
                    sum10_i += pA[0] * pB[1];
                    sum11_i += pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                }
                sum00 += sum00_i * pA_descales[0] * pB_descales[0];
                sum01 += sum01_i * pA_descales[1] * pB_descales[0];
                sum10 += sum10_i * pA_descales[0] * pB_descales[1];
                sum11 += sum11_i * pA_descales[1] * pB_descales[1];
                pA_descales += 2;
                pB_descales += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
#endif // __mips_msa
            outptr += 4;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
#if __mips_msa
            v4f32 _fsum;
            if (k == 0)
            {
                _fsum = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum = (v4f32)__msa_loadl_d(outptr);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    v16i8 _pB = (v16i8)__msa_fill_w(*(const int*)pB);
                    _sum = __msa_dpadd_s_w(_sum, __msa_dotp_s_h(_pA, _pB), _one);
                    pA += 8;
                    pB += 4;
                }
                int sum0 = __msa_copy_s_w(_sum, 0);
                int sum1 = __msa_copy_s_w(_sum, 1);
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[1] * pB[0];
                    pA += 2;
                    pB++;
                }
                _sum = __msa_set_w(sum0, sum1, 0, 0);
                v4f32 _descaleA = (v4f32)__msa_fill_d_ptr(pA_descales);
                v4f32 _scale = __msa_fmul_w(_descaleA, __msa_fill_w_f32(pB_descales[0]));
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales += 2;
                pB_descales++;
            }
            __msa_storel_d((v4i32)_fsum, outptr);
#else
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

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum0_i = 0;
                int sum1_i = 0;
                int kk = 0;
#if __mips_loongson_mmi
                int32x2_t _sum0 = __mmi_pzerow_s();
                int32x2_t _sum1 = __mmi_pzerow_s();
                const int8x8_t _zero = __mmi_pzerob_s();
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk0; kk += 4)
                {
#if __mips_loongson_mmi
                    __builtin_prefetch(pB + 16);
                    int8x8_t _pA = __mmi_pldb_s(pA);
                    int8x8_t _pB = (int8x8_t)__mmi_pfillw_s(*(const int*)pB);
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
#else
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum1_i += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    pA += 8;
                    pB += 4;
#endif // __mips_loongson_mmi
                }
#if __mips_loongson_mmi
                int tmp[2];
                __mmi_pstw_s(tmp, _sum0);
                sum0_i += tmp[0] + tmp[1];
                __mmi_pstw_s(tmp, _sum1);
                sum1_i += tmp[0] + tmp[1];
#endif // __mips_loongson_mmi
                for (; kk < max_kk0; kk++)
                {
                    sum0_i += pA[0] * pB[0];
                    sum1_i += pA[1] * pB[0];
                    pA += 2;
                    pB++;
                }
                sum0 += sum0_i * pA_descales[0] * pB_descales[0];
                sum1 += sum1_i * pA_descales[1] * pB_descales[0];
                pA_descales += 2;
                pB_descales++;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
#endif // __mips_msa
            outptr += 2;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += (size_t)2 * A_hstep;
        pAT_descales += (size_t)2 * A_descales_hstep;
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __mips_msa
        const v8i16 _one = __msa_fill_h(1);
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            v4f32 _fsum0;
            if (k == 0)
            {
                _fsum0 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum0 = (v4f32)__msa_ld_w(outptr, 0);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum0 = __msa_fill_w(0);
                int kk = 0;
                {
                    v4i32 _sum1 = __msa_fill_w(0);
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __builtin_prefetch(pA + 32);
                        __builtin_prefetch(pB + 64);
                        v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                        v16i8 _pB0 = __msa_ld_b(pB, 0);
                        _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);

                        _pA = (v16i8)__msa_fill_w(*(const int*)(pA + 4));
                        _pB0 = __msa_ld_b(pB + 16, 0);
                        _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0), _one);
                        pA += 8;
                        pB += 32;
                    }
                    _sum0 = __msa_addv_w(_sum0, _sum1);
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                    v16i8 _pB0 = __msa_ld_b(pB, 0);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    pA += 4;
                    pB += 16;
                }
                int sum0 = __msa_copy_s_w(_sum0, 0);
                int sum1 = __msa_copy_s_w(_sum0, 1);
                int sum2 = __msa_copy_s_w(_sum0, 2);
                int sum3 = __msa_copy_s_w(_sum0, 3);
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[0] * pB[1];
                    sum2 += pA[0] * pB[2];
                    sum3 += pA[0] * pB[3];
                    pA++;
                    pB += 4;
                }
                _sum0 = __msa_set_w(sum0, sum1, sum2, sum3);
                v4f32 _descaleB = (v4f32)__msa_ld_w(pB_descales, 0);
                v4f32 _scale = __msa_fmul_w(_descaleB, __msa_fill_w_f32(pA_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                pA_descales++;
                pB_descales += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            outptr += 4;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
#if __mips_msa
            v4f32 _fsum;
            if (k == 0)
            {
                _fsum = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _fsum = (v4f32)__msa_loadl_d(outptr);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 16);
                    __builtin_prefetch(pB + 32);
                    v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                    v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    _sum = __msa_dpadd_s_w(_sum, __msa_dotp_s_h(_pA, _pB), _one);
                    pA += 4;
                    pB += 8;
                }
                int sum0 = __msa_copy_s_w(_sum, 0);
                int sum1 = __msa_copy_s_w(_sum, 1);
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[0] * pB[1];
                    pA++;
                    pB += 2;
                }
                _sum = __msa_set_w(sum0, sum1, 0, 0);
                v4f32 _descaleB = (v4f32)__msa_fill_d_ptr(pB_descales);
                v4f32 _scale = __msa_fmul_w(_descaleB, __msa_fill_w_f32(pA_descales[0]));
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales++;
                pB_descales += 2;
            }
            __msa_storel_d((v4i32)_fsum, outptr);
#else
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

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum0_i = 0;
                int sum1_i = 0;
                int kk = 0;
#if __mips_loongson_mmi
                int32x2_t _sum0 = __mmi_pzerow_s();
                int32x2_t _sum1 = __mmi_pzerow_s();
                const int8x8_t _zero = __mmi_pzerob_s();
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk0; kk += 4)
                {
#if __mips_loongson_mmi
                    __builtin_prefetch(pB + 32);
                    int8x8_t _pA = (int8x8_t)__mmi_pfillw_s(*(const int*)pA);
                    int8x8_t _pB = __mmi_pldb_s(pB);
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
#else
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum1_i += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    pA += 4;
                    pB += 8;
#endif // __mips_loongson_mmi
                }
#if __mips_loongson_mmi
                int tmp[2];
                __mmi_pstw_s(tmp, _sum0);
                sum0_i += tmp[0] + tmp[1];
                __mmi_pstw_s(tmp, _sum1);
                sum1_i += tmp[0] + tmp[1];
#endif // __mips_loongson_mmi
                for (; kk < max_kk0; kk++)
                {
                    sum0_i += pA[0] * pB[0];
                    sum1_i += pA[0] * pB[1];
                    pA++;
                    pB += 2;
                }
                sum0 += sum0_i * pA_descales[0] * pB_descales[0];
                sum1 += sum1_i * pA_descales[0] * pB_descales[1];
                pA_descales++;
                pB_descales += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
#endif // __mips_msa
            outptr += 2;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
#if __mips_msa
            v4f32 _fsum = (v4f32)__msa_fill_w(0);
            if (k != 0)
                _fsum[0] = outptr[0];
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                v4i32 _sum = __msa_fill_w(0);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 16);
                    __builtin_prefetch(pB + 16);
                    v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                    v16i8 _pB = (v16i8)__msa_fill_w(*(const int*)pB);
                    _sum = __msa_dpadd_s_w(_sum, __msa_dotp_s_h(_pA, _pB), _one);
                    pA += 4;
                    pB += 4;
                }
                int sum0 = __msa_copy_s_w(_sum, 0);
                for (; kk < max_kk0; kk++)
                    sum0 += *pA++ * *pB++;
                _sum = __msa_fill_w(sum0);
                v4f32 _scale = __msa_fill_w_f32(pA_descales[0] * pB_descales[0]);
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales++;
                pB_descales++;
            }
            *outptr++ = _fsum[0];
#else
            float sum0;
            if (k == 0)
            {
                sum0 = 0.f;
            }
            else
            {
                sum0 = outptr[0];
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int sum0_i = 0;
                int kk = 0;
#if __mips_loongson_mmi
                int32x2_t _sum0 = __mmi_pzerow_s();
                const int8x8_t _zero = __mmi_pzerob_s();
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk0; kk += 4)
                {
#if __mips_loongson_mmi
                    int8x8_t _pA = (int8x8_t)__mmi_pfillw_s(*(const int*)pA);
                    int8x8_t _pB = (int8x8_t)__mmi_pfillw_s(*(const int*)pB);
                    int16x4_t _pA0 = (int16x4_t)__mmi_punpcklbh_s(_pA, _zero);
                    int16x4_t _pB0 = (int16x4_t)__mmi_punpcklbh_s(_pB, _zero);
                    _pA0 = __mmi_psrah_s(__mmi_psllh_s(_pA0, 8), 8);
                    _pB0 = __mmi_psrah_s(__mmi_psllh_s(_pB0, 8), 8);
                    _sum0 = __mmi_paddw_s(_sum0, __mmi_pmaddhw(_pA0, _pB0));
                    pA += 4;
                    pB += 4;
#else
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    pA += 4;
                    pB += 4;
#endif // __mips_loongson_mmi
                }
#if __mips_loongson_mmi
                int tmp[2];
                __mmi_pstw_s(tmp, _sum0);
                sum0_i += tmp[0] + tmp[1];
#endif // __mips_loongson_mmi
                for (; kk < max_kk0; kk++)
                    sum0_i += *pA++ * *pB++;
                sum0 += sum0_i * pA_descales[0] * pB_descales[0];
                pA_descales++;
                pB_descales++;
            }

            *outptr++ = sum0;
#endif // __mips_msa
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_elemtype)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
#if __mips_msa
    const int c_elempack = C.elempack;
#endif // __mips_msa
    const float* pp = topT;
    float* outptr = output_elemtype == 1 ? (float*)top_blob + (size_t)i * out_hstep + j * out_elempack : 0;
    unsigned short* outptr_bf16s = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)i * out_hstep + j * out_elempack : 0;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0f = outptr;
        unsigned short* p0 = outptr_bf16s;

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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    float* p1 = p0f + out_hstep * 4;
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + 8, 0);
                    __msa_st_w((v4i32)_f3, p0f + 12, 0);
                    __msa_st_w((v4i32)_f4, p1, 0);
                    __msa_st_w((v4i32)_f5, p1 + 4, 0);
                    __msa_st_w((v4i32)_f6, p1 + 8, 0);
                    __msa_st_w((v4i32)_f7, p1 + 12, 0);
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                    __msa_st_w((v4i32)_f2, p0f + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_f3, p0f + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_f4, p0f + out_hstep * 4, 0);
                    __msa_st_w((v4i32)_f5, p0f + out_hstep * 5, 0);
                    __msa_st_w((v4i32)_f6, p0f + out_hstep * 6, 0);
                    __msa_st_w((v4i32)_f7, p0f + out_hstep * 7, 0);
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 16);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + 20);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 24);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + 28);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f4), p1);
                    __msa_storel_d(float2bfloat_msa(_f5), p1 + 4);
                    __msa_storel_d(float2bfloat_msa(_f6), p1 + 8);
                    __msa_storel_d(float2bfloat_msa(_f7), p1 + 12);
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + out_hstep * 2);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + out_hstep * 3);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + out_hstep * 4);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + out_hstep * 5);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + out_hstep * 6);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + out_hstep * 7);
                }
            }

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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    float* p1 = p0f + out_hstep * 4;
                    transpose4x4_ps(_g0, _g1, _g2, _g3);
                    transpose4x4_ps(_g4, _g5, _g6, _g7);
                    __msa_st_w((v4i32)_g0, p0f + 16, 0);
                    __msa_st_w((v4i32)_g1, p0f + 20, 0);
                    __msa_st_w((v4i32)_g2, p0f + 24, 0);
                    __msa_st_w((v4i32)_g3, p0f + 28, 0);
                    __msa_st_w((v4i32)_g4, p1 + 16, 0);
                    __msa_st_w((v4i32)_g5, p1 + 20, 0);
                    __msa_st_w((v4i32)_g6, p1 + 24, 0);
                    __msa_st_w((v4i32)_g7, p1 + 28, 0);
                    p0f += 32;
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_g0, p0f + 4, 0);
                    __msa_st_w((v4i32)_g1, p0f + out_hstep + 4, 0);
                    __msa_st_w((v4i32)_g2, p0f + out_hstep * 2 + 4, 0);
                    __msa_st_w((v4i32)_g3, p0f + out_hstep * 3 + 4, 0);
                    __msa_st_w((v4i32)_g4, p0f + out_hstep * 4 + 4, 0);
                    __msa_st_w((v4i32)_g5, p0f + out_hstep * 5 + 4, 0);
                    __msa_st_w((v4i32)_g6, p0f + out_hstep * 6 + 4, 0);
                    __msa_st_w((v4i32)_g7, p0f + out_hstep * 7 + 4, 0);
                    p0f += 8;
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_g0, _g1, _g2, _g3);
                    transpose4x4_ps(_g4, _g5, _g6, _g7);
                    __msa_storel_d(float2bfloat_msa(_g0), p0 + 32);
                    __msa_storel_d(float2bfloat_msa(_g4), p0 + 36);
                    __msa_storel_d(float2bfloat_msa(_g1), p0 + 40);
                    __msa_storel_d(float2bfloat_msa(_g5), p0 + 44);
                    __msa_storel_d(float2bfloat_msa(_g2), p0 + 48);
                    __msa_storel_d(float2bfloat_msa(_g6), p0 + 52);
                    __msa_storel_d(float2bfloat_msa(_g3), p0 + 56);
                    __msa_storel_d(float2bfloat_msa(_g7), p0 + 60);
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose4x4_ps(_g0, _g1, _g2, _g3);
                    transpose4x4_ps(_g4, _g5, _g6, _g7);
                    __msa_storel_d(float2bfloat_msa(_g0), p0 + 16);
                    __msa_storel_d(float2bfloat_msa(_g1), p0 + 20);
                    __msa_storel_d(float2bfloat_msa(_g2), p0 + 24);
                    __msa_storel_d(float2bfloat_msa(_g3), p0 + 28);
                    __msa_storel_d(float2bfloat_msa(_g4), p1 + 16);
                    __msa_storel_d(float2bfloat_msa(_g5), p1 + 20);
                    __msa_storel_d(float2bfloat_msa(_g6), p1 + 24);
                    __msa_storel_d(float2bfloat_msa(_g7), p1 + 28);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_g0), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_g1), p0 + out_hstep + 4);
                    __msa_storel_d(float2bfloat_msa(_g2), p0 + out_hstep * 2 + 4);
                    __msa_storel_d(float2bfloat_msa(_g3), p0 + out_hstep * 3 + 4);
                    __msa_storel_d(float2bfloat_msa(_g4), p0 + out_hstep * 4 + 4);
                    __msa_storel_d(float2bfloat_msa(_g5), p0 + out_hstep * 5 + 4);
                    __msa_storel_d(float2bfloat_msa(_g6), p0 + out_hstep * 6 + 4);
                    __msa_storel_d(float2bfloat_msa(_g7), p0 + out_hstep * 7 + 4);
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    float* p1 = p0f + out_hstep * 4;
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + 8, 0);
                    __msa_st_w((v4i32)_f3, p0f + 12, 0);
                    __msa_st_w((v4i32)_f4, p1, 0);
                    __msa_st_w((v4i32)_f5, p1 + 4, 0);
                    __msa_st_w((v4i32)_f6, p1 + 8, 0);
                    __msa_st_w((v4i32)_f7, p1 + 12, 0);
                    p0f += 16;
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                    __msa_st_w((v4i32)_f2, p0f + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_f3, p0f + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_f4, p0f + out_hstep * 4, 0);
                    __msa_st_w((v4i32)_f5, p0f + out_hstep * 5, 0);
                    __msa_st_w((v4i32)_f6, p0f + out_hstep * 6, 0);
                    __msa_st_w((v4i32)_f7, p0f + out_hstep * 7, 0);
                    p0f += 4;
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 16);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + 20);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 24);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + 28);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f4), p1);
                    __msa_storel_d(float2bfloat_msa(_f5), p1 + 4);
                    __msa_storel_d(float2bfloat_msa(_f6), p1 + 8);
                    __msa_storel_d(float2bfloat_msa(_f7), p1 + 12);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + out_hstep * 2);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + out_hstep * 3);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + out_hstep * 4);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + out_hstep * 5);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + out_hstep * 6);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + out_hstep * 7);
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    float* p1 = p0f + out_hstep * 4;
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f4, p1, 0);
                    __msa_st_w((v4i32)_f5, p1 + 4, 0);
                    p0f += 8;
                }
                if (out_elempack == 1)
                {
                    ((int*)p0f)[0] = __msa_copy_s_w((v4i32)_f0, 0);
                    ((int*)p0f)[1] = __msa_copy_s_w((v4i32)_f1, 0);
                    ((int*)(p0f + out_hstep))[0] = __msa_copy_s_w((v4i32)_f0, 1);
                    ((int*)(p0f + out_hstep))[1] = __msa_copy_s_w((v4i32)_f1, 1);
                    ((int*)(p0f + out_hstep * 2))[0] = __msa_copy_s_w((v4i32)_f0, 2);
                    ((int*)(p0f + out_hstep * 2))[1] = __msa_copy_s_w((v4i32)_f1, 2);
                    ((int*)(p0f + out_hstep * 3))[0] = __msa_copy_s_w((v4i32)_f0, 3);
                    ((int*)(p0f + out_hstep * 3))[1] = __msa_copy_s_w((v4i32)_f1, 3);
                    ((int*)(p0f + out_hstep * 4))[0] = __msa_copy_s_w((v4i32)_f4, 0);
                    ((int*)(p0f + out_hstep * 4))[1] = __msa_copy_s_w((v4i32)_f5, 0);
                    ((int*)(p0f + out_hstep * 5))[0] = __msa_copy_s_w((v4i32)_f4, 1);
                    ((int*)(p0f + out_hstep * 5))[1] = __msa_copy_s_w((v4i32)_f5, 1);
                    ((int*)(p0f + out_hstep * 6))[0] = __msa_copy_s_w((v4i32)_f4, 2);
                    ((int*)(p0f + out_hstep * 6))[1] = __msa_copy_s_w((v4i32)_f5, 2);
                    ((int*)(p0f + out_hstep * 7))[0] = __msa_copy_s_w((v4i32)_f4, 3);
                    ((int*)(p0f + out_hstep * 7))[1] = __msa_copy_s_w((v4i32)_f5, 3);
                    p0f += 2;
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + 12);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f4), p1);
                    __msa_storel_d(float2bfloat_msa(_f5), p1 + 4);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
                    v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);
                    v8i16 _bf4 = (v8i16)float2bfloat_msa(_f4);
                    v8i16 _bf5 = (v8i16)float2bfloat_msa(_f5);
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    float* p1 = p0f + out_hstep * 4;
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f4, p1, 0);
                    p0f += 4;
                }
                if (out_elempack == 1)
                {
                    ((int*)p0f)[0] = __msa_copy_s_w((v4i32)_f0, 0);
                    ((int*)(p0f + out_hstep))[0] = __msa_copy_s_w((v4i32)_f0, 1);
                    ((int*)(p0f + out_hstep * 2))[0] = __msa_copy_s_w((v4i32)_f0, 2);
                    ((int*)(p0f + out_hstep * 3))[0] = __msa_copy_s_w((v4i32)_f0, 3);
                    ((int*)(p0f + out_hstep * 4))[0] = __msa_copy_s_w((v4i32)_f4, 0);
                    ((int*)(p0f + out_hstep * 5))[0] = __msa_copy_s_w((v4i32)_f4, 1);
                    ((int*)(p0f + out_hstep * 6))[0] = __msa_copy_s_w((v4i32)_f4, 2);
                    ((int*)(p0f + out_hstep * 7))[0] = __msa_copy_s_w((v4i32)_f4, 3);
                    p0f++;
                }
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
        if (output_elemtype == 1)
            outptr += out_hstep * 8;
        else
            outptr_bf16s += out_hstep * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0f = outptr;
        unsigned short* p0 = outptr_bf16s;

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

        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;
        const float* pC2 = pC0 ? pC0 + c_hstep * 2 : 0;
        const float* pC3 = pC0 ? pC0 + c_hstep * 3 : 0;

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
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC1, 0), _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC2, 0), _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC3, 0), _beta));
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC0 + 4, 0), _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC1 + 4, 0), _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC2 + 4, 0), _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC3 + 4, 0), _beta));
                        pC0 += 8;
                        pC1 += 8;
                        pC2 += 8;
                        pC3 += 8;
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + 8, 0);
                    __msa_st_w((v4i32)_f3, p0f + 12, 0);
                    __msa_st_w((v4i32)_f4, p0f + 16, 0);
                    __msa_st_w((v4i32)_f5, p0f + 20, 0);
                    __msa_st_w((v4i32)_f6, p0f + 24, 0);
                    __msa_st_w((v4i32)_f7, p0f + 28, 0);
                    p0f += 32;
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f4, p0f + 4, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                    __msa_st_w((v4i32)_f5, p0f + out_hstep + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_f6, p0f + out_hstep * 2 + 4, 0);
                    __msa_st_w((v4i32)_f3, p0f + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_f7, p0f + out_hstep * 3 + 4, 0);
                    p0f += 8;
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 16);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + 20);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + 24);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + 28);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + out_hstep + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + out_hstep * 2);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + out_hstep * 2 + 4);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + out_hstep * 3);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + out_hstep * 3 + 4);
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
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC1, 0), _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC2, 0), _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC3, 0), _beta));
                        pC0 += 4;
                        pC1 += 4;
                        pC2 += 4;
                        pC3 += 4;
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + 8, 0);
                    __msa_st_w((v4i32)_f3, p0f + 12, 0);
                    p0f += 16;
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                    __msa_st_w((v4i32)_f2, p0f + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_f3, p0f + out_hstep * 3, 0);
                    p0f += 4;
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 12);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + out_hstep * 2);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + out_hstep * 3);
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
                        _c0 = __msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC2), __msa_load_w(pC3));
                        _c1 = __msa_set_w(__msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1), __msa_load_w(pC2 + 1), __msa_load_w(pC3 + 1));
                        pC0 += 2;
                        pC1 += 2;
                        pC2 += 2;
                        pC3 += 2;
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    p0f += 8;
                }
                if (out_elempack == 1)
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_f1, (v4i32)_f0);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_f1, (v4i32)_f0);
                    __msa_storel_d((v4i32)_tmp0, p0f);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_tmp0, (v16i8)_tmp0, 8), p0f + out_hstep);
                    __msa_storel_d((v4i32)_tmp1, p0f + out_hstep * 2);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_tmp1, (v16i8)_tmp1, 8), p0f + out_hstep * 3);
                    p0f += 2;
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_f1, (v4i32)_f0);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_f1, (v4i32)_f0);
                    ((int*)p0)[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)((v4i32)_tmp0)), 0);
                    ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)((v4i32)__msa_sldi_b((v16i8)_tmp0, (v16i8)_tmp0, 8))), 0);
                    ((int*)(p0 + out_hstep * 2))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)((v4i32)_tmp1)), 0);
                    ((int*)(p0 + out_hstep * 3))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)((v4i32)__msa_sldi_b((v16i8)_tmp1, (v16i8)_tmp1, 8))), 0);
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
                        _c0 = __msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC2), __msa_load_w(pC3));
                        pC0++;
                        pC1++;
                        pC2++;
                        pC3++;
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
            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    p0f += 4;
                }
                if (out_elempack == 1)
                {
                    p0f[0] = _f0[0];
                    p0f[out_hstep] = _f0[1];
                    p0f[out_hstep * 2] = _f0[2];
                    p0f[out_hstep * 3] = _f0[3];
                    p0f++;
                }
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
        if (output_elemtype == 1)
            outptr += out_hstep * 4;
        else
            outptr_bf16s += out_hstep * 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0f = outptr;
        unsigned short* p0 = outptr_bf16s;
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
        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;

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
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC0 + 4, 0);
                    pC0 += 8;
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC1, 0);
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                    pC1 += 8;
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

            if (output_elemtype == 1)
            {
                __msa_st_w((v4i32)_f0, p0f, 0);
                __msa_st_w((v4i32)_f1, p0f + 4, 0);
                __msa_st_w((v4i32)_f2, p0f + out_hstep, 0);
                __msa_st_w((v4i32)_f3, p0f + out_hstep + 4, 0);
                p0f += 8;
            }
            else
            {
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f0)), p0);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f1)), p0 + 4);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f2)), p0 + out_hstep);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f3)), p0 + out_hstep + 4);
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
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    pC0 += 4;
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC1, 0);
                    pC1 += 4;
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

            if (output_elemtype == 1)
            {
                __msa_st_w((v4i32)_f0, p0f, 0);
                __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                p0f += 4;
            }
            else
            {
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f0)), p0);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f1)), p0 + out_hstep);
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
                    v4f32 _c = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1));
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
                    pC0 += 2;
                    pC1 += 2;
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
            if (output_elemtype == 1)
            {
                __msa_storel_d(_f0, p0f);
                __msa_storel_d(_f1, p0f + out_hstep);
            }
            else
            {
                ((int*)p0)[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)(_f0)), 0);
                ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)(_f1)), 0);
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
                    pC0 += 2;
                    pC1 += 2;
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

            if (output_elemtype == 1)
            {
                p0f[0] = sum00;
                p0f[out_hstep] = sum01;
                p0f[1] = sum10;
                p0f[out_hstep + 1] = sum11;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum00);
                p0[out_hstep] = float32_to_bfloat16(sum01);
                p0[1] = float32_to_bfloat16(sum10);
                p0[out_hstep + 1] = float32_to_bfloat16(sum11);
            }
#endif // __mips_msa
            pp += 4;
            if (output_elemtype == 1)
                p0f += 2;
            else
                p0 += 2;
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
                    pC0++;
                    float c1 = pC1[0];
                    pC1++;
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

            if (output_elemtype == 1)
            {
                p0f[0] = sum0;
                p0f[out_hstep] = sum1;
                p0f++;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0[out_hstep] = float32_to_bfloat16(sum1);
                p0++;
            }
        }
        if (output_elemtype == 1)
            outptr += out_hstep * 2;
        else
            outptr_bf16s += out_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        float* p0f = outptr;
        unsigned short* p0 = outptr_bf16s;
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
        const float* pC0 = pC && (broadcast_type_C == 3 || broadcast_type_C == 4) ? pC : 0;

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
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC0 + 4, 0);
                    pC0 += 8;
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

            if (output_elemtype == 1)
            {
                __msa_st_w((v4i32)_f0, p0f, 0);
                __msa_st_w((v4i32)_f1, p0f + 4, 0);
                p0f += 8;
            }
            else
            {
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f0)), p0);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f1)), p0 + 4);
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
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    pC0 += 4;
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                }
            }

            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));

            if (output_elemtype == 1)
            {
                __msa_st_w((v4i32)_f0, p0f, 0);
                p0f += 4;
            }
            else
            {
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f0)), p0);
                p0 += 4;
            }
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __mips_msa
            v4i32 _fi = __msa_fill_w(0);
            _fi = __msa_insert_w(_fi, 0, ((const int*)pp)[0]);
            _fi = __msa_insert_w(_fi, 1, ((const int*)pp)[1]);
            v4f32 _f = (v4f32)_fi;

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
                    pC0 += 2;
                }
            }

            if (alpha != 1.f)
                _f = __msa_fmul_w(_f, __msa_fill_w_f32(alpha));

            if (output_elemtype == 1)
            {
                __msa_storel_d((v4i32)_f, p0f);
            }
            else
            {
                ((int*)p0)[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)((v4i32)_f)), 0);
            }
#else
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
                    float c0 = pC0[0];
                    float c1 = pC0[1];
                    if (beta != 1.f)
                    {
                        c0 *= beta;
                        c1 *= beta;
                    }
                    sum0 += c0;
                    sum1 += c1;
                    pC0 += 2;
                }
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }

            if (output_elemtype == 1)
            {
                p0f[0] = sum0;
                p0f[1] = sum1;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0[1] = float32_to_bfloat16(sum1);
            }
#endif // __mips_msa
            pp += 2;
            if (output_elemtype == 1)
                p0f += 2;
            else
                p0 += 2;
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
                    c = pC0[0];
                    pC0++;
                }
                if ((broadcast_type_C == 3 || broadcast_type_C == 4) && beta != 1.f)
                    c *= beta;
                sum0 += c;
            }

            if (alpha != 1.f)
                sum0 *= alpha;
            if (output_elemtype == 1)
            {
                p0f[0] = sum0;
                p0f++;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0++;
            }
        }
        if (output_elemtype == 1)
            outptr += out_hstep;
        else
            outptr_bf16s += out_hstep;
    }
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_elemtype)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
#if __mips_msa
    const int c_elempack = C.elempack;
#endif // __mips_msa
    const float* pp = topT;
    float* outptr = output_elemtype == 1 ? (float*)top_blob + (size_t)j * out_hstep + i * out_elempack : 0;
    unsigned short* outptr_bf16s = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)j * out_hstep + i * out_elempack : 0;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0f = outptr;
        unsigned short* p0 = outptr_bf16s;

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
            v4f32 _g0 = (v4f32)__msa_ld_w(pp + 32, 0);
            v4f32 _g4 = (v4f32)__msa_ld_w(pp + 36, 0);
            v4f32 _g1 = (v4f32)__msa_ld_w(pp + 40, 0);
            v4f32 _g5 = (v4f32)__msa_ld_w(pp + 44, 0);
            v4f32 _g2 = (v4f32)__msa_ld_w(pp + 48, 0);
            v4f32 _g6 = (v4f32)__msa_ld_w(pp + 52, 0);
            v4f32 _g3 = (v4f32)__msa_ld_w(pp + 56, 0);
            v4f32 _g7 = (v4f32)__msa_ld_w(pp + 60, 0);
            pp += 64;

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

            transpose4x4_ps(_f0, _f1, _f2, _f3);
            transpose4x4_ps(_f4, _f5, _f6, _f7);
            transpose4x4_ps(_g0, _g1, _g2, _g3);
            transpose4x4_ps(_g4, _g5, _g6, _g7);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, _c0123);
                    _g0 = __msa_fadd_w(_g0, _c0123);
                    _f1 = __msa_fadd_w(_f1, _c0123);
                    _g1 = __msa_fadd_w(_g1, _c0123);
                    _f2 = __msa_fadd_w(_f2, _c0123);
                    _g2 = __msa_fadd_w(_g2, _c0123);
                    _f3 = __msa_fadd_w(_f3, _c0123);
                    _g3 = __msa_fadd_w(_g3, _c0123);
                    _f4 = __msa_fadd_w(_f4, _c4567);
                    _g4 = __msa_fadd_w(_g4, _c4567);
                    _f5 = __msa_fadd_w(_f5, _c4567);
                    _g5 = __msa_fadd_w(_g5, _c4567);
                    _f6 = __msa_fadd_w(_f6, _c4567);
                    _g6 = __msa_fadd_w(_g6, _c4567);
                    _f7 = __msa_fadd_w(_f7, _c4567);
                    _g7 = __msa_fadd_w(_g7, _c4567);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    if (c_elempack == 8)
                    {
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + 8, 0), _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC + 12, 0), _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + 16, 0), _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC + 20, 0), _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + 24, 0), _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC + 28, 0), _beta));
                        _g0 = __msa_fadd_w(_g0, __msa_fmul_w((v4f32)__msa_ld_w(pC + 32, 0), _beta));
                        _g4 = __msa_fadd_w(_g4, __msa_fmul_w((v4f32)__msa_ld_w(pC + 36, 0), _beta));
                        _g1 = __msa_fadd_w(_g1, __msa_fmul_w((v4f32)__msa_ld_w(pC + 40, 0), _beta));
                        _g5 = __msa_fadd_w(_g5, __msa_fmul_w((v4f32)__msa_ld_w(pC + 44, 0), _beta));
                        _g2 = __msa_fadd_w(_g2, __msa_fmul_w((v4f32)__msa_ld_w(pC + 48, 0), _beta));
                        _g6 = __msa_fadd_w(_g6, __msa_fmul_w((v4f32)__msa_ld_w(pC + 52, 0), _beta));
                        _g3 = __msa_fadd_w(_g3, __msa_fmul_w((v4f32)__msa_ld_w(pC + 56, 0), _beta));
                        _g7 = __msa_fadd_w(_g7, __msa_fmul_w((v4f32)__msa_ld_w(pC + 60, 0), _beta));
                        pC += 64;
                    }
                    else if (c_elempack == 4)
                    {
                        const float* pC1 = pC + c_hstep * 4;
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC1, 0), _beta));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC1 + 4, 0), _beta));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + 8, 0), _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC1 + 8, 0), _beta));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + 12, 0), _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC1 + 12, 0), _beta));
                        _g0 = __msa_fadd_w(_g0, __msa_fmul_w((v4f32)__msa_ld_w(pC + 16, 0), _beta));
                        _g4 = __msa_fadd_w(_g4, __msa_fmul_w((v4f32)__msa_ld_w(pC1 + 16, 0), _beta));
                        _g1 = __msa_fadd_w(_g1, __msa_fmul_w((v4f32)__msa_ld_w(pC + 20, 0), _beta));
                        _g5 = __msa_fadd_w(_g5, __msa_fmul_w((v4f32)__msa_ld_w(pC1 + 20, 0), _beta));
                        _g2 = __msa_fadd_w(_g2, __msa_fmul_w((v4f32)__msa_ld_w(pC + 24, 0), _beta));
                        _g6 = __msa_fadd_w(_g6, __msa_fmul_w((v4f32)__msa_ld_w(pC1 + 24, 0), _beta));
                        _g3 = __msa_fadd_w(_g3, __msa_fmul_w((v4f32)__msa_ld_w(pC + 28, 0), _beta));
                        _g7 = __msa_fadd_w(_g7, __msa_fmul_w((v4f32)__msa_ld_w(pC1 + 28, 0), _beta));
                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        v4f32 _cl = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                        v4f32 _ch = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                        _f0 = __msa_fadd_w(_f0, __msa_fmul_w(_cl, _beta));
                        _f4 = __msa_fadd_w(_f4, __msa_fmul_w(_ch, _beta));
                        _cl = (v4f32)__msa_set_w(__msa_load_w(pC + 1), __msa_load_w(pC + c_hstep + 1), __msa_load_w(pC + c_hstep * 2 + 1), __msa_load_w(pC + c_hstep * 3 + 1));
                        _ch = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 1), __msa_load_w(pC + c_hstep * 5 + 1), __msa_load_w(pC + c_hstep * 6 + 1), __msa_load_w(pC + c_hstep * 7 + 1));
                        _f1 = __msa_fadd_w(_f1, __msa_fmul_w(_cl, _beta));
                        _f5 = __msa_fadd_w(_f5, __msa_fmul_w(_ch, _beta));
                        _cl = (v4f32)__msa_set_w(__msa_load_w(pC + 2), __msa_load_w(pC + c_hstep + 2), __msa_load_w(pC + c_hstep * 2 + 2), __msa_load_w(pC + c_hstep * 3 + 2));
                        _ch = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 2), __msa_load_w(pC + c_hstep * 5 + 2), __msa_load_w(pC + c_hstep * 6 + 2), __msa_load_w(pC + c_hstep * 7 + 2));
                        _f2 = __msa_fadd_w(_f2, __msa_fmul_w(_cl, _beta));
                        _f6 = __msa_fadd_w(_f6, __msa_fmul_w(_ch, _beta));
                        _cl = (v4f32)__msa_set_w(__msa_load_w(pC + 3), __msa_load_w(pC + c_hstep + 3), __msa_load_w(pC + c_hstep * 2 + 3), __msa_load_w(pC + c_hstep * 3 + 3));
                        _ch = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 3), __msa_load_w(pC + c_hstep * 5 + 3), __msa_load_w(pC + c_hstep * 6 + 3), __msa_load_w(pC + c_hstep * 7 + 3));
                        _f3 = __msa_fadd_w(_f3, __msa_fmul_w(_cl, _beta));
                        _f7 = __msa_fadd_w(_f7, __msa_fmul_w(_ch, _beta));
                        _cl = (v4f32)__msa_set_w(__msa_load_w(pC + 4), __msa_load_w(pC + c_hstep + 4), __msa_load_w(pC + c_hstep * 2 + 4), __msa_load_w(pC + c_hstep * 3 + 4));
                        _ch = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 4), __msa_load_w(pC + c_hstep * 5 + 4), __msa_load_w(pC + c_hstep * 6 + 4), __msa_load_w(pC + c_hstep * 7 + 4));
                        _g0 = __msa_fadd_w(_g0, __msa_fmul_w(_cl, _beta));
                        _g4 = __msa_fadd_w(_g4, __msa_fmul_w(_ch, _beta));
                        _cl = (v4f32)__msa_set_w(__msa_load_w(pC + 5), __msa_load_w(pC + c_hstep + 5), __msa_load_w(pC + c_hstep * 2 + 5), __msa_load_w(pC + c_hstep * 3 + 5));
                        _ch = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 5), __msa_load_w(pC + c_hstep * 5 + 5), __msa_load_w(pC + c_hstep * 6 + 5), __msa_load_w(pC + c_hstep * 7 + 5));
                        _g1 = __msa_fadd_w(_g1, __msa_fmul_w(_cl, _beta));
                        _g5 = __msa_fadd_w(_g5, __msa_fmul_w(_ch, _beta));
                        _cl = (v4f32)__msa_set_w(__msa_load_w(pC + 6), __msa_load_w(pC + c_hstep + 6), __msa_load_w(pC + c_hstep * 2 + 6), __msa_load_w(pC + c_hstep * 3 + 6));
                        _ch = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 6), __msa_load_w(pC + c_hstep * 5 + 6), __msa_load_w(pC + c_hstep * 6 + 6), __msa_load_w(pC + c_hstep * 7 + 6));
                        _g2 = __msa_fadd_w(_g2, __msa_fmul_w(_cl, _beta));
                        _g6 = __msa_fadd_w(_g6, __msa_fmul_w(_ch, _beta));
                        _cl = (v4f32)__msa_set_w(__msa_load_w(pC + 7), __msa_load_w(pC + c_hstep + 7), __msa_load_w(pC + c_hstep * 2 + 7), __msa_load_w(pC + c_hstep * 3 + 7));
                        _ch = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 7), __msa_load_w(pC + c_hstep * 5 + 7), __msa_load_w(pC + c_hstep * 6 + 7), __msa_load_w(pC + c_hstep * 7 + 7));
                        _g3 = __msa_fadd_w(_g3, __msa_fmul_w(_cl, _beta));
                        _g7 = __msa_fadd_w(_g7, __msa_fmul_w(_ch, _beta));
                        pC += 8;
                    }
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
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_splati_w((v4i32)_c0, 0));
                    _f4 = __msa_fadd_w(_f4, (v4f32)__msa_splati_w((v4i32)_c0, 0));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_splati_w((v4i32)_c0, 1));
                    _f5 = __msa_fadd_w(_f5, (v4f32)__msa_splati_w((v4i32)_c0, 1));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_splati_w((v4i32)_c0, 2));
                    _f6 = __msa_fadd_w(_f6, (v4f32)__msa_splati_w((v4i32)_c0, 2));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_splati_w((v4i32)_c0, 3));
                    _f7 = __msa_fadd_w(_f7, (v4f32)__msa_splati_w((v4i32)_c0, 3));
                    _g0 = __msa_fadd_w(_g0, (v4f32)__msa_splati_w((v4i32)_c1, 0));
                    _g4 = __msa_fadd_w(_g4, (v4f32)__msa_splati_w((v4i32)_c1, 0));
                    _g1 = __msa_fadd_w(_g1, (v4f32)__msa_splati_w((v4i32)_c1, 1));
                    _g5 = __msa_fadd_w(_g5, (v4f32)__msa_splati_w((v4i32)_c1, 1));
                    _g2 = __msa_fadd_w(_g2, (v4f32)__msa_splati_w((v4i32)_c1, 2));
                    _g6 = __msa_fadd_w(_g6, (v4f32)__msa_splati_w((v4i32)_c1, 2));
                    _g3 = __msa_fadd_w(_g3, (v4f32)__msa_splati_w((v4i32)_c1, 3));
                    _g7 = __msa_fadd_w(_g7, (v4f32)__msa_splati_w((v4i32)_c1, 3));
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
                _g0 = __msa_fmul_w(_g0, _alpha);
                _g1 = __msa_fmul_w(_g1, _alpha);
                _g2 = __msa_fmul_w(_g2, _alpha);
                _g3 = __msa_fmul_w(_g3, _alpha);
                _g4 = __msa_fmul_w(_g4, _alpha);
                _g5 = __msa_fmul_w(_g5, _alpha);
                _g6 = __msa_fmul_w(_g6, _alpha);
                _g7 = __msa_fmul_w(_g7, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    transpose4x4_ps(_g0, _g1, _g2, _g3);
                    transpose4x4_ps(_g4, _g5, _g6, _g7);
                    float* p1 = p0f + out_hstep * 4;
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + 8, 0);
                    __msa_st_w((v4i32)_f3, p0f + 12, 0);
                    __msa_st_w((v4i32)_f4, p0f + 16, 0);
                    __msa_st_w((v4i32)_f5, p0f + 20, 0);
                    __msa_st_w((v4i32)_f6, p0f + 24, 0);
                    __msa_st_w((v4i32)_f7, p0f + 28, 0);
                    __msa_st_w((v4i32)_g0, p1, 0);
                    __msa_st_w((v4i32)_g1, p1 + 4, 0);
                    __msa_st_w((v4i32)_g2, p1 + 8, 0);
                    __msa_st_w((v4i32)_g3, p1 + 12, 0);
                    __msa_st_w((v4i32)_g4, p1 + 16, 0);
                    __msa_st_w((v4i32)_g5, p1 + 20, 0);
                    __msa_st_w((v4i32)_g6, p1 + 24, 0);
                    __msa_st_w((v4i32)_g7, p1 + 28, 0);
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f4, p0f + 4, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                    __msa_st_w((v4i32)_f5, p0f + out_hstep + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_f6, p0f + out_hstep * 2 + 4, 0);
                    __msa_st_w((v4i32)_f3, p0f + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_f7, p0f + out_hstep * 3 + 4, 0);
                    __msa_st_w((v4i32)_g0, p0f + out_hstep * 4, 0);
                    __msa_st_w((v4i32)_g4, p0f + out_hstep * 4 + 4, 0);
                    __msa_st_w((v4i32)_g1, p0f + out_hstep * 5, 0);
                    __msa_st_w((v4i32)_g5, p0f + out_hstep * 5 + 4, 0);
                    __msa_st_w((v4i32)_g2, p0f + out_hstep * 6, 0);
                    __msa_st_w((v4i32)_g6, p0f + out_hstep * 6 + 4, 0);
                    __msa_st_w((v4i32)_g3, p0f + out_hstep * 7, 0);
                    __msa_st_w((v4i32)_g7, p0f + out_hstep * 7 + 4, 0);
                }
                p0f += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    transpose4x4_ps(_g0, _g1, _g2, _g3);
                    transpose4x4_ps(_g4, _g5, _g6, _g7);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_g0), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_g1), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 16);
                    __msa_storel_d(float2bfloat_msa(_g2), p0 + 20);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 24);
                    __msa_storel_d(float2bfloat_msa(_g3), p0 + 28);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 32);
                    __msa_storel_d(float2bfloat_msa(_g4), p0 + 36);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + 40);
                    __msa_storel_d(float2bfloat_msa(_g5), p0 + 44);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + 48);
                    __msa_storel_d(float2bfloat_msa(_g6), p0 + 52);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + 56);
                    __msa_storel_d(float2bfloat_msa(_g7), p0 + 60);
                }
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    transpose4x4_ps(_g0, _g1, _g2, _g3);
                    transpose4x4_ps(_g4, _g5, _g6, _g7);
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 16);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + 20);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + 24);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + 28);
                    __msa_storel_d(float2bfloat_msa(_g0), p1);
                    __msa_storel_d(float2bfloat_msa(_g1), p1 + 4);
                    __msa_storel_d(float2bfloat_msa(_g2), p1 + 8);
                    __msa_storel_d(float2bfloat_msa(_g3), p1 + 12);
                    __msa_storel_d(float2bfloat_msa(_g4), p1 + 16);
                    __msa_storel_d(float2bfloat_msa(_g5), p1 + 20);
                    __msa_storel_d(float2bfloat_msa(_g6), p1 + 24);
                    __msa_storel_d(float2bfloat_msa(_g7), p1 + 28);
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + out_hstep + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + out_hstep * 2);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + out_hstep * 2 + 4);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + out_hstep * 3);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + out_hstep * 3 + 4);
                    __msa_storel_d(float2bfloat_msa(_g0), p0 + out_hstep * 4);
                    __msa_storel_d(float2bfloat_msa(_g4), p0 + out_hstep * 4 + 4);
                    __msa_storel_d(float2bfloat_msa(_g1), p0 + out_hstep * 5);
                    __msa_storel_d(float2bfloat_msa(_g5), p0 + out_hstep * 5 + 4);
                    __msa_storel_d(float2bfloat_msa(_g2), p0 + out_hstep * 6);
                    __msa_storel_d(float2bfloat_msa(_g6), p0 + out_hstep * 6 + 4);
                    __msa_storel_d(float2bfloat_msa(_g3), p0 + out_hstep * 7);
                    __msa_storel_d(float2bfloat_msa(_g7), p0 + out_hstep * 7 + 4);
                }
                p0 += out_hstep * 8;
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

            transpose4x4_ps(_f0, _f1, _f2, _f3);
            transpose4x4_ps(_f4, _f5, _f6, _f7);

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
                    v4f32 _cl0;
                    v4f32 _ch0;
                    v4f32 _cl1;
                    v4f32 _ch1;
                    v4f32 _cl2;
                    v4f32 _ch2;
                    v4f32 _cl3;
                    v4f32 _ch3;
                    if (c_elempack == 8)
                    {
                        _cl0 = (v4f32)__msa_ld_w(pC, 0);
                        _ch0 = (v4f32)__msa_ld_w(pC + 4, 0);
                        _cl1 = (v4f32)__msa_ld_w(pC + 8, 0);
                        _ch1 = (v4f32)__msa_ld_w(pC + 12, 0);
                        _cl2 = (v4f32)__msa_ld_w(pC + 16, 0);
                        _ch2 = (v4f32)__msa_ld_w(pC + 20, 0);
                        _cl3 = (v4f32)__msa_ld_w(pC + 24, 0);
                        _ch3 = (v4f32)__msa_ld_w(pC + 28, 0);
                        pC += 32;
                    }
                    else if (c_elempack == 4)
                    {
                        _cl0 = (v4f32)__msa_ld_w(pC, 0);
                        _cl1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        _cl2 = (v4f32)__msa_ld_w(pC + 8, 0);
                        _cl3 = (v4f32)__msa_ld_w(pC + 12, 0);
                        const float* pC1 = pC + c_hstep * 4;
                        _ch0 = (v4f32)__msa_ld_w(pC1, 0);
                        _ch1 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                        _ch2 = (v4f32)__msa_ld_w(pC1 + 8, 0);
                        _ch3 = (v4f32)__msa_ld_w(pC1 + 12, 0);
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        _cl0 = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                        _ch0 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                        _cl1 = (v4f32)__msa_set_w(__msa_load_w(pC + 1), __msa_load_w(pC + c_hstep + 1), __msa_load_w(pC + c_hstep * 2 + 1), __msa_load_w(pC + c_hstep * 3 + 1));
                        _ch1 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 1), __msa_load_w(pC + c_hstep * 5 + 1), __msa_load_w(pC + c_hstep * 6 + 1), __msa_load_w(pC + c_hstep * 7 + 1));
                        _cl2 = (v4f32)__msa_set_w(__msa_load_w(pC + 2), __msa_load_w(pC + c_hstep + 2), __msa_load_w(pC + c_hstep * 2 + 2), __msa_load_w(pC + c_hstep * 3 + 2));
                        _ch2 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 2), __msa_load_w(pC + c_hstep * 5 + 2), __msa_load_w(pC + c_hstep * 6 + 2), __msa_load_w(pC + c_hstep * 7 + 2));
                        _cl3 = (v4f32)__msa_set_w(__msa_load_w(pC + 3), __msa_load_w(pC + c_hstep + 3), __msa_load_w(pC + c_hstep * 2 + 3), __msa_load_w(pC + c_hstep * 3 + 3));
                        _ch3 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 3), __msa_load_w(pC + c_hstep * 5 + 3), __msa_load_w(pC + c_hstep * 6 + 3), __msa_load_w(pC + c_hstep * 7 + 3));
                        pC += 4;
                    }
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
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
                    pC += 4;
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + 8, 0);
                    __msa_st_w((v4i32)_f3, p0f + 12, 0);
                    __msa_st_w((v4i32)_f4, p0f + 16, 0);
                    __msa_st_w((v4i32)_f5, p0f + 20, 0);
                    __msa_st_w((v4i32)_f6, p0f + 24, 0);
                    __msa_st_w((v4i32)_f7, p0f + 28, 0);
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f4, p0f + 4, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                    __msa_st_w((v4i32)_f5, p0f + out_hstep + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_f6, p0f + out_hstep * 2 + 4, 0);
                    __msa_st_w((v4i32)_f3, p0f + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_f7, p0f + out_hstep * 3 + 4, 0);
                }
                p0f += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 16);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + 20);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + 24);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + 28);
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + out_hstep + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + out_hstep * 2);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + out_hstep * 2 + 4);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + out_hstep * 3);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + out_hstep * 3 + 4);
                }
                p0 += out_hstep * 4;
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
                    v4f32 _c4;
                    v4f32 _c1;
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

            if (output_elemtype == 1)
            {
                __msa_st_w((v4i32)_f0, p0f, 0);
                __msa_st_w((v4i32)_f4, p0f + 4, 0);
                __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                __msa_st_w((v4i32)_f5, p0f + out_hstep + 4, 0);
                p0f += out_hstep * 2;
            }
            else
            {
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f0)), p0);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f4)), p0 + 4);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f1)), p0 + out_hstep);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f5)), p0 + out_hstep + 4);
                p0 += out_hstep * 2;
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

            if (output_elemtype == 1)
            {
                __msa_st_w((v4i32)_f0, p0f, 0);
                __msa_st_w((v4i32)_f4, p0f + 4, 0);
                p0f += out_hstep;
            }
            else
            {
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f0)), p0);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f4)), p0 + 4);
                p0 += out_hstep;
            }
        }
        if (output_elemtype == 1)
            outptr += 8 * out_elempack;
        else
            outptr_bf16s += 8 * out_elempack;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0f = outptr;
        unsigned short* p0 = outptr_bf16s;

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

        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;
        const float* pC2 = pC0 ? pC0 + c_hstep * 2 : 0;
        const float* pC3 = pC0 ? pC0 + c_hstep * 3 : 0;

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

            transpose4x4_ps(_f0, _f1, _f2, _f3);
            transpose4x4_ps(_f4, _f5, _f6, _f7);

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
                    v4f32 _c0;
                    v4f32 _c1;
                    v4f32 _c2;
                    v4f32 _c3;
                    v4f32 _c4;
                    v4f32 _c5;
                    v4f32 _c6;
                    v4f32 _c7;
                    if (c_elempack == 4)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        _c2 = (v4f32)__msa_ld_w(pC + 8, 0);
                        _c3 = (v4f32)__msa_ld_w(pC + 12, 0);
                        _c4 = (v4f32)__msa_ld_w(pC + 16, 0);
                        _c5 = (v4f32)__msa_ld_w(pC + 20, 0);
                        _c6 = (v4f32)__msa_ld_w(pC + 24, 0);
                        _c7 = (v4f32)__msa_ld_w(pC + 28, 0);
                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC0, 0);
                        _c1 = (v4f32)__msa_ld_w(pC1, 0);
                        _c2 = (v4f32)__msa_ld_w(pC2, 0);
                        _c3 = (v4f32)__msa_ld_w(pC3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _c4 = (v4f32)__msa_ld_w(pC0 + 4, 0);
                        _c5 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                        _c6 = (v4f32)__msa_ld_w(pC2 + 4, 0);
                        _c7 = (v4f32)__msa_ld_w(pC3 + 4, 0);
                        transpose4x4_ps(_c4, _c5, _c6, _c7);
                        pC0 += 8;
                        pC1 += 8;
                        pC2 += 8;
                        pC3 += 8;
                    }
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
                    pC += 8;
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    float* p1 = p0f + out_hstep * 4;
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + 8, 0);
                    __msa_st_w((v4i32)_f3, p0f + 12, 0);
                    __msa_st_w((v4i32)_f4, p1, 0);
                    __msa_st_w((v4i32)_f5, p1 + 4, 0);
                    __msa_st_w((v4i32)_f6, p1 + 8, 0);
                    __msa_st_w((v4i32)_f7, p1 + 12, 0);
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                    __msa_st_w((v4i32)_f2, p0f + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_f3, p0f + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_f4, p0f + out_hstep * 4, 0);
                    __msa_st_w((v4i32)_f5, p0f + out_hstep * 5, 0);
                    __msa_st_w((v4i32)_f6, p0f + out_hstep * 6, 0);
                    __msa_st_w((v4i32)_f7, p0f + out_hstep * 7, 0);
                }
                p0f += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 16);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + 20);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 24);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + 28);
                }
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 12);
                    __msa_storel_d(float2bfloat_msa(_f4), p1);
                    __msa_storel_d(float2bfloat_msa(_f5), p1 + 4);
                    __msa_storel_d(float2bfloat_msa(_f6), p1 + 8);
                    __msa_storel_d(float2bfloat_msa(_f7), p1 + 12);
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + out_hstep * 2);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + out_hstep * 3);
                    __msa_storel_d(float2bfloat_msa(_f4), p0 + out_hstep * 4);
                    __msa_storel_d(float2bfloat_msa(_f5), p0 + out_hstep * 5);
                    __msa_storel_d(float2bfloat_msa(_f6), p0 + out_hstep * 6);
                    __msa_storel_d(float2bfloat_msa(_f7), p0 + out_hstep * 7);
                }
                p0 += out_hstep * 8;
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

            transpose4x4_ps(_f0, _f1, _f2, _f3);

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
                    v4f32 _c0;
                    v4f32 _c1;
                    v4f32 _c2;
                    v4f32 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        _c2 = (v4f32)__msa_ld_w(pC + 8, 0);
                        _c3 = (v4f32)__msa_ld_w(pC + 12, 0);
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC0, 0);
                        _c1 = (v4f32)__msa_ld_w(pC1, 0);
                        _c2 = (v4f32)__msa_ld_w(pC2, 0);
                        _c3 = (v4f32)__msa_ld_w(pC3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        pC0 += 4;
                        pC1 += 4;
                        pC2 += 4;
                        pC3 += 4;
                    }
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
                    pC += 4;
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                    __msa_st_w((v4i32)_f2, p0f + 8, 0);
                    __msa_st_w((v4i32)_f3, p0f + 12, 0);
                }
                if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                    __msa_st_w((v4i32)_f2, p0f + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_f3, p0f + out_hstep * 3, 0);
                }
                p0f += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + 12);
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep);
                    __msa_storel_d(float2bfloat_msa(_f2), p0 + out_hstep * 2);
                    __msa_storel_d(float2bfloat_msa(_f3), p0 + out_hstep * 3);
                }
                p0 += out_hstep * 4;
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
                    v4f32 _c0;
                    v4f32 _c1;
                    if (c_elempack == 4)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC, 0);
                        _c1 = (v4f32)__msa_ld_w(pC + 4, 0);
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC2), __msa_load_w(pC3));
                        _c1 = (v4f32)__msa_set_w(__msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1), __msa_load_w(pC2 + 1), __msa_load_w(pC3 + 1));
                        pC0 += 2;
                        pC1 += 2;
                        pC2 += 2;
                        pC3 += 2;
                    }
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

            if (output_elemtype == 1)
            {
                __msa_st_w((v4i32)_f0, p0f, 0);
                __msa_st_w((v4i32)_f1, p0f + out_hstep, 0);
                p0f += out_hstep * 2;
            }
            else
            {
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f0)), p0);
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f1)), p0 + out_hstep);
                p0 += out_hstep * 2;
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
                    v4f32 _c0;
                    if (c_elempack == 4)
                    {
                        _c0 = (v4f32)__msa_ld_w(pC, 0);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC2), __msa_load_w(pC3));
                        pC0++;
                        pC1++;
                        pC2++;
                        pC3++;
                    }
                    if (beta != 1.f)
                        _c0 = __msa_fmul_w(_c0, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
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
            if (output_elemtype == 1)
            {
                __msa_st_w((v4i32)_f0, p0f, 0);
                p0f += out_hstep;
            }
            else
            {
                __msa_storel_d(float2bfloat_msa((v4f32)((v4i32)_f0)), p0);
                p0 += out_hstep;
            }
        }
        if (output_elemtype == 1)
            outptr += 4 * out_elempack;
        else
            outptr_bf16s += 4 * out_elempack;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0f = outptr;
        unsigned short* p0 = outptr_bf16s;
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
        const float* pC0 = pC && broadcast_type_C == 3 ? pC : 0;
        const float* pC1 = pC0 ? pC0 + c_hstep : 0;

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
            v4f32 _f0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _f1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _f2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _f3 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = (v4f32)__msa_set_w(__msa_load_w(&c0), __msa_load_w(&c1), __msa_load_w(&c0), __msa_load_w(&c1));
                    _f0 = __msa_fadd_w(_f0, _c);
                    _f1 = __msa_fadd_w(_f1, _c);
                    _f2 = __msa_fadd_w(_f2, _c);
                    _f3 = __msa_fadd_w(_f3, _c);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1));
                    v4f32 _c1 = (v4f32)__msa_set_w(__msa_load_w(pC0 + 2), __msa_load_w(pC1 + 2), __msa_load_w(pC0 + 3), __msa_load_w(pC1 + 3));
                    v4f32 _c2 = (v4f32)__msa_set_w(__msa_load_w(pC0 + 4), __msa_load_w(pC1 + 4), __msa_load_w(pC0 + 5), __msa_load_w(pC1 + 5));
                    v4f32 _c3 = (v4f32)__msa_set_w(__msa_load_w(pC0 + 6), __msa_load_w(pC1 + 6), __msa_load_w(pC0 + 7), __msa_load_w(pC1 + 7));
                    pC0 += 8;
                    pC1 += 8;
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
                    pC += 8;
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
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_set_w(__msa_load_w(&c00), __msa_load_w(&c00), __msa_load_w(&c01), __msa_load_w(&c01)));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_set_w(__msa_load_w(&c02), __msa_load_w(&c02), __msa_load_w(&c03), __msa_load_w(&c03)));
                    _f2 = __msa_fadd_w(_f2, (v4f32)__msa_set_w(__msa_load_w(&c04), __msa_load_w(&c04), __msa_load_w(&c05), __msa_load_w(&c05)));
                    _f3 = __msa_fadd_w(_f3, (v4f32)__msa_set_w(__msa_load_w(&c06), __msa_load_w(&c06), __msa_load_w(&c07), __msa_load_w(&c07)));
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

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    v4i32 _m0 = __msa_pckev_w((v4i32)_f1, (v4i32)_f0);
                    v4i32 _m1 = __msa_pckod_w((v4i32)_f1, (v4i32)_f0);
                    v4i32 _m2 = __msa_pckev_w((v4i32)_f3, (v4i32)_f2);
                    v4i32 _m3 = __msa_pckod_w((v4i32)_f3, (v4i32)_f2);
                    __msa_st_w(_m0, p0f, 0);
                    __msa_st_w(_m1, p0f + 4, 0);
                    __msa_st_w(_m2, p0f + out_hstep * 4, 0);
                    __msa_st_w(_m3, p0f + out_hstep * 4 + 4, 0);
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d((v4i32)_f0, p0f);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f0, (v16i8)_f0, 8), p0f + out_hstep);
                    __msa_storel_d((v4i32)_f1, p0f + out_hstep * 2);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f1, (v16i8)_f1, 8), p0f + out_hstep * 3);
                    __msa_storel_d((v4i32)_f2, p0f + out_hstep * 4);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f2, (v16i8)_f2, 8), p0f + out_hstep * 5);
                    __msa_storel_d((v4i32)_f3, p0f + out_hstep * 6);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f3, (v16i8)_f3, 8), p0f + out_hstep * 7);
                }
                p0f += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    v4f32 _m0 = (v4f32)__msa_pckev_w((v4i32)_f1, (v4i32)_f0);
                    v4f32 _m1 = (v4f32)__msa_pckod_w((v4i32)_f1, (v4i32)_f0);
                    v4f32 _m2 = (v4f32)__msa_pckev_w((v4i32)_f3, (v4i32)_f2);
                    v4f32 _m3 = (v4f32)__msa_pckod_w((v4i32)_f3, (v4i32)_f2);
                    __msa_storel_d(float2bfloat_msa(_m0), p0);
                    __msa_storel_d(float2bfloat_msa(_m2), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_m1), p0 + 8);
                    __msa_storel_d(float2bfloat_msa(_m3), p0 + 12);
                }
                if (out_elempack == 4)
                {
                    v4f32 _m0 = (v4f32)__msa_pckev_w((v4i32)_f1, (v4i32)_f0);
                    v4f32 _m1 = (v4f32)__msa_pckod_w((v4i32)_f1, (v4i32)_f0);
                    v4f32 _m2 = (v4f32)__msa_pckev_w((v4i32)_f3, (v4i32)_f2);
                    v4f32 _m3 = (v4f32)__msa_pckod_w((v4i32)_f3, (v4i32)_f2);
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __msa_storel_d(float2bfloat_msa(_m0), p0);
                    __msa_storel_d(float2bfloat_msa(_m1), p0 + 4);
                    __msa_storel_d(float2bfloat_msa(_m2), p1);
                    __msa_storel_d(float2bfloat_msa(_m3), p1 + 4);
                }
                if (out_elempack == 1)
                {
                    ((int*)p0)[0] = __msa_copy_s_w(float2bfloat_msa(_f0), 0);
                    ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)__msa_sldi_b((v16i8)_f0, (v16i8)_f0, 8)), 0);
                    ((int*)(p0 + out_hstep * 2))[0] = __msa_copy_s_w(float2bfloat_msa(_f1), 0);
                    ((int*)(p0 + out_hstep * 3))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)__msa_sldi_b((v16i8)_f1, (v16i8)_f1, 8)), 0);
                    ((int*)(p0 + out_hstep * 4))[0] = __msa_copy_s_w(float2bfloat_msa(_f2), 0);
                    ((int*)(p0 + out_hstep * 5))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)__msa_sldi_b((v16i8)_f2, (v16i8)_f2, 8)), 0);
                    ((int*)(p0 + out_hstep * 6))[0] = __msa_copy_s_w(float2bfloat_msa(_f3), 0);
                    ((int*)(p0 + out_hstep * 7))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)__msa_sldi_b((v16i8)_f3, (v16i8)_f3, 8)), 0);
                }
                p0 += out_hstep * 8;
            }
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
                    v4f32 _c = (v4f32)__msa_set_w(__msa_load_w(&c0), __msa_load_w(&c1), __msa_load_w(&c0), __msa_load_w(&c1));
                    _f0 = __msa_fadd_w(_f0, _c);
                    _f1 = __msa_fadd_w(_f1, _c);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1));
                    v4f32 _c1 = (v4f32)__msa_set_w(__msa_load_w(pC0 + 2), __msa_load_w(pC1 + 2), __msa_load_w(pC0 + 3), __msa_load_w(pC1 + 3));
                    pC0 += 4;
                    pC1 += 4;
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
                    pC += 4;
                    if (beta != 1.f)
                    {
                        c00 *= beta;
                        c01 *= beta;
                        c02 *= beta;
                        c03 *= beta;
                    }
                    _f0 = __msa_fadd_w(_f0, (v4f32)__msa_set_w(__msa_load_w(&c00), __msa_load_w(&c00), __msa_load_w(&c01), __msa_load_w(&c01)));
                    _f1 = __msa_fadd_w(_f1, (v4f32)__msa_set_w(__msa_load_w(&c02), __msa_load_w(&c02), __msa_load_w(&c03), __msa_load_w(&c03)));
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    v4i32 _m0 = __msa_pckev_w((v4i32)_f1, (v4i32)_f0);
                    v4i32 _m1 = __msa_pckod_w((v4i32)_f1, (v4i32)_f0);
                    __msa_st_w(_m0, p0f, 0);
                    __msa_st_w(_m1, p0f + 4, 0);
                }
                if (out_elempack == 1)
                {
                    __msa_storel_d((v4i32)_f0, p0f);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f0, (v16i8)_f0, 8), p0f + out_hstep);
                    __msa_storel_d((v4i32)_f1, p0f + out_hstep * 2);
                    __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f1, (v16i8)_f1, 8), p0f + out_hstep * 3);
                }
                p0f += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 4)
                {
                    v4f32 _m0 = (v4f32)__msa_pckev_w((v4i32)_f1, (v4i32)_f0);
                    v4f32 _m1 = (v4f32)__msa_pckod_w((v4i32)_f1, (v4i32)_f0);
                    __msa_storel_d(float2bfloat_msa(_m0), p0);
                    __msa_storel_d(float2bfloat_msa(_m1), p0 + 4);
                }
                if (out_elempack == 1)
                {
                    ((int*)p0)[0] = __msa_copy_s_w(float2bfloat_msa(_f0), 0);
                    ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)__msa_sldi_b((v16i8)_f0, (v16i8)_f0, 8)), 0);
                    ((int*)(p0 + out_hstep * 2))[0] = __msa_copy_s_w(float2bfloat_msa(_f1), 0);
                    ((int*)(p0 + out_hstep * 3))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)__msa_sldi_b((v16i8)_f1, (v16i8)_f1, 8)), 0);
                }
                p0 += out_hstep * 4;
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
                    v4f32 _c = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1));
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
                    pC0 += 2;
                    pC1 += 2;
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

            if (output_elemtype == 1)
            {
                __msa_storel_d((v4i32)_f, p0f);
                __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f, (v16i8)_f, 8), p0f + out_hstep);
            }
            else
            {
                ((int*)p0)[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)((v4i32)_f)), 0);
                ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w(float2bfloat_msa((v4f32)((v4i32)__msa_sldi_b((v16i8)_f, (v16i8)_f, 8))), 0);
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
                    pC0 += 2;
                    pC1 += 2;
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

            if (output_elemtype == 1)
            {
                p0f[0] = sum00;
                p0f[1] = sum01;
                p0f[out_hstep] = sum10;
                p0f[out_hstep + 1] = sum11;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum00);
                p0[1] = float32_to_bfloat16(sum01);
                p0[out_hstep] = float32_to_bfloat16(sum10);
                p0[out_hstep + 1] = float32_to_bfloat16(sum11);
            }
#endif // __mips_msa
            pp += 4;
            if (output_elemtype == 1)
                p0f += out_hstep * 2;
            else
                p0 += out_hstep * 2;
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
                    pC0++;
                    float c1 = pC1[0];
                    pC1++;
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

            if (output_elemtype == 1)
            {
                p0f[0] = sum0;
                p0f[1] = sum1;
                p0f += out_hstep;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0[1] = float32_to_bfloat16(sum1);
                p0 += out_hstep;
            }
        }
        if (output_elemtype == 1)
            outptr += 2 * out_elempack;
        else
            outptr_bf16s += 2 * out_elempack;
    }
    for (; ii < max_ii; ii++)
    {
        float* p0f = outptr;
        unsigned short* p0 = outptr_bf16s;
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
        const float* pC0 = pC && (broadcast_type_C == 3 || broadcast_type_C == 4) ? pC : 0;
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
        v4f32 _c0 = __msa_fill_w_f32(c0);

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
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    v4f32 _c0 = (v4f32)__msa_ld_w(pC0, 0);
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC0 + 4, 0);
                    pC0 += 8;
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
            if (out_elempack == 8)
            {
                if (output_elemtype == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + 4, 0);
                }
                else
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + 4);
                }
            }
            if (out_elempack == 4)
            {
                if (output_elemtype == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                    __msa_st_w((v4i32)_f1, p0f + out_hstep * 4, 0);
                }
                else
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                    __msa_storel_d(float2bfloat_msa(_f1), p0 + out_hstep * 4);
                }
            }
            if (out_elempack == 1)
            {
                if (output_elemtype == 1)
                {
                    *(int*)p0f = __msa_copy_s_w((v4i32)_f0, 0);
                    *(int*)(p0f + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
                    *(int*)(p0f + out_hstep * 2) = __msa_copy_s_w((v4i32)_f0, 2);
                    *(int*)(p0f + out_hstep * 3) = __msa_copy_s_w((v4i32)_f0, 3);
                    *(int*)(p0f + out_hstep * 4) = __msa_copy_s_w((v4i32)_f1, 0);
                    *(int*)(p0f + out_hstep * 5) = __msa_copy_s_w((v4i32)_f1, 1);
                    *(int*)(p0f + out_hstep * 6) = __msa_copy_s_w((v4i32)_f1, 2);
                    *(int*)(p0f + out_hstep * 7) = __msa_copy_s_w((v4i32)_f1, 3);
                }
                else
                {
                    v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
                    v8i16 _bf1 = (v8i16)float2bfloat_msa(_f1);
                    p0[0] = (unsigned short)__msa_copy_s_h(_bf0, 0);
                    p0[out_hstep] = (unsigned short)__msa_copy_s_h(_bf0, 1);
                    p0[out_hstep * 2] = (unsigned short)__msa_copy_s_h(_bf0, 2);
                    p0[out_hstep * 3] = (unsigned short)__msa_copy_s_h(_bf0, 3);
                    p0[out_hstep * 4] = (unsigned short)__msa_copy_s_h(_bf1, 0);
                    p0[out_hstep * 5] = (unsigned short)__msa_copy_s_h(_bf1, 1);
                    p0[out_hstep * 6] = (unsigned short)__msa_copy_s_h(_bf1, 2);
                    p0[out_hstep * 7] = (unsigned short)__msa_copy_s_h(_bf1, 3);
                }
            }

            if (output_elemtype == 1)
                p0f += out_hstep * 8;
            else
                p0 += out_hstep * 8;
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
                    pC0 += 4;
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c);
                }
            }

            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            if (out_elempack == 4)
            {
                if (output_elemtype == 1)
                {
                    __msa_st_w((v4i32)_f0, p0f, 0);
                }
                else
                {
                    __msa_storel_d(float2bfloat_msa(_f0), p0);
                }
            }
            if (out_elempack == 1)
            {
                if (output_elemtype == 1)
                {
                    *(int*)p0f = __msa_copy_s_w((v4i32)_f0, 0);
                    *(int*)(p0f + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
                    *(int*)(p0f + out_hstep * 2) = __msa_copy_s_w((v4i32)_f0, 2);
                    *(int*)(p0f + out_hstep * 3) = __msa_copy_s_w((v4i32)_f0, 3);
                }
                else
                {
                    v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
                    p0[0] = (unsigned short)__msa_copy_s_h(_bf0, 0);
                    p0[out_hstep] = (unsigned short)__msa_copy_s_h(_bf0, 1);
                    p0[out_hstep * 2] = (unsigned short)__msa_copy_s_h(_bf0, 2);
                    p0[out_hstep * 3] = (unsigned short)__msa_copy_s_h(_bf0, 3);
                }
            }

            if (output_elemtype == 1)
                p0f += out_hstep * 4;
            else
                p0 += out_hstep * 4;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __mips_msa
            v4i32 _fi = __msa_fill_w(0);
            _fi = __msa_insert_w(_fi, 0, ((const int*)pp)[0]);
            _fi = __msa_insert_w(_fi, 1, ((const int*)pp)[1]);
            v4f32 _f0 = (v4f32)_fi;
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
                    pC0 += 2;
                }
            }

            if (alpha != 1.f)
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            if (output_elemtype == 1)
            {
                *(int*)p0f = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(p0f + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
            }
            else
            {
                v8i16 _bf0 = (v8i16)float2bfloat_msa(_f0);
                p0[0] = (unsigned short)__msa_copy_s_h(_bf0, 0);
                p0[out_hstep] = (unsigned short)__msa_copy_s_h(_bf0, 1);
            }
#else
            float sum0 = pp[0];
            float sum1 = pp[1];
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
                    pC0 += 2;
                }
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }

            if (output_elemtype == 1)
            {
                p0f[0] = sum0;
                p0f[out_hstep] = sum1;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0[out_hstep] = float32_to_bfloat16(sum1);
            }
#endif // __mips_msa
            pp += 2;
            if (output_elemtype == 1)
                p0f += out_hstep * 2;
            else
                p0 += out_hstep * 2;
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
                    c = pC0[0];
                    pC0++;
                    if (beta != 1.f)
                        c *= beta;
                }
                sum0 += c;
            }

            if (alpha != 1.f)
                sum0 *= alpha;
            if (output_elemtype == 1)
            {
                p0f[0] = sum0;
                p0f += out_hstep;
            }
            else
            {
                p0[0] = float32_to_bfloat16(sum0);
                p0 += out_hstep;
            }
        }
        if (output_elemtype == 1)
            outptr++;
        else
            outptr_bf16s++;
    }
}
static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const int l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    {
#if __mips_msa
        int tile_size = (l2_cache_size_int8 - 16) / 8;
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
#endif
        TILE_K = std::max(block_size, tile_size / block_size * block_size);

        if (K > 0)
        {
            int nn_K = (K + TILE_K - 1) / TILE_K;
            TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + block_size - 1) / block_size * block_size);
            if (TILE_K >= K)
                TILE_K = K;
        }
    }

    {
#if __mips_msa
        int tile_size = (l2_cache_size_int8 - 8 * TILE_K) / std::max(1, TILE_K + 8);
        TILE_M = std::max(8, tile_size / 8 * 8);
#else
        int tile_size = (l2_cache_size_int8 - 2 * TILE_K) / std::max(1, TILE_K + 2);
        TILE_M = std::max(2, tile_size / 2 * 2);
#endif

        if (M > 0)
        {
            int nn_M = std::max(std::min(nT, get_physical_cpu_count()), (M + TILE_M - 1) / TILE_M);
#if __mips_msa
            TILE_M = std::max(8, std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8));
#else
            TILE_M = std::max(2, std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2));
#endif
        }
    }

    if (N > 0)
    {
        int tile_size = TILE_K >= K ? (l2_cache_size_int8 - TILE_M * TILE_K) / std::max(1, TILE_K) : (l2_cache_size_int8 - TILE_M * TILE_K) / std::max(1, TILE_M + TILE_K);
#if __mips_msa
        TILE_N = std::max(4, tile_size / 4 * 4);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(4, std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4));
#else
        TILE_N = std::max(2, tile_size / 2 * 2);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(2, std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2));
#endif
    }
    else
    {
#if __mips_msa
        TILE_N = 4;
#else
        TILE_N = 2;
#endif
    }

    if (constant_TILE_M > 0)
    {
#if __mips_msa
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_N > 0)
    {
#if __mips_msa
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = (constant_TILE_N + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, (constant_TILE_K + block_size - 1) / block_size * block_size);
        if (K > 0 && TILE_K >= K)
            TILE_K = K;
    }
}
