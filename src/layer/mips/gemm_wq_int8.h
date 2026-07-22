// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
void pack_B_tile_wq_int8_loongson_mmi(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_loongson_mmi(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_loongson_mmi(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void gemm_transB_packed_tile_wq_int8_loongson_mmi(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

// group-major, output-major within each K4/K2/K1 fragment
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
        const float* s0 = B_scales.row(j + jj);
        const float* s1 = B_scales.row(j + jj + 1);
        const float* s2 = B_scales.row(j + jj + 2);
        const float* s3 = B_scales.row(j + jj + 3);

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
            if (kk + 1 < max_kk)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp[4] = p2[0];
                pp[5] = p2[1];
                pp[6] = p3[0];
                pp[7] = p3[1];
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                kk += 2;
            }
            if (kk < max_kk)
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

            pd[0] = 1.f / *s0++;
            pd[1] = 1.f / *s1++;
            pd[2] = 1.f / *s2++;
            pd[3] = 1.f / *s3++;
            pd += 4;
        }
    }
#endif // __mips_msa

    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const signed char* p1 = B.row<const signed char>(j + jj + 1);
        const float* s0 = B_scales.row(j + jj);
        const float* s1 = B_scales.row(j + jj + 1);

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
            if (kk + 1 < max_kk)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp += 4;
                p0 += 2;
                p1 += 2;
                kk += 2;
            }
            if (kk < max_kk)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }

            *pd++ = 1.f / *s0++;
            *pd++ = 1.f / *s1++;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const float* s0 = B_scales.row(j + jj);

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
            if (kk + 1 < max_kk)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += 2;
                kk += 2;
            }
            if (kk < max_kk)
            {
                *pp++ = *p0++;
            }

            *pd++ = 1.f / *s0++;
        }
    }
}

// group-major, row-major within each K4/K2/K1 fragment
static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        quantize_A_tile_wq_int8_loongson_mmi(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const float* A_data = (const float*)A + k;

    if (input_scales.empty())
    {
        int ii = 0;
#if __mips_msa
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = A_data + (size_t)(i + ii) * A_hstep;
            const float* p1 = A_data + (size_t)(i + ii + 1) * A_hstep;
            const float* p2 = A_data + (size_t)(i + ii + 2) * A_hstep;
            const float* p3 = A_data + (size_t)(i + ii + 3) * A_hstep;
            const float* p4 = A_data + (size_t)(i + ii + 4) * A_hstep;
            const float* p5 = A_data + (size_t)(i + ii + 5) * A_hstep;
            const float* p6 = A_data + (size_t)(i + ii + 6) * A_hstep;
            const float* p7 = A_data + (size_t)(i + ii + 7) * A_hstep;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk0 = std::min(max_kk - k0, block_size);
                const float* p0g = p0 + k0;
                const float* p1g = p1 + k0;
                const float* p2g = p2 + k0;
                const float* p3g = p3 + k0;
                const float* p4g = p4 + k0;
                const float* p5g = p5 + k0;
                const float* p6g = p6 + k0;
                const float* p7g = p7 + k0;
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

                const float* p0a = p0g;
                const float* p1a = p1g;
                const float* p2a = p2g;
                const float* p3a = p3g;
                const float* p4a = p4g;
                const float* p5a = p5g;
                const float* p6a = p6g;
                const float* p7a = p7g;
                int kk = 0;
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
                    absmax0 = std::max(absmax0, fabsf(*p0a++));
                    absmax1 = std::max(absmax1, fabsf(*p1a++));
                    absmax2 = std::max(absmax2, fabsf(*p2a++));
                    absmax3 = std::max(absmax3, fabsf(*p3a++));
                    absmax4 = std::max(absmax4, fabsf(*p4a++));
                    absmax5 = std::max(absmax5, fabsf(*p5a++));
                    absmax6 = std::max(absmax6, fabsf(*p6a++));
                    absmax7 = std::max(absmax7, fabsf(*p7a++));
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

                const float* p0q = p0g;
                const float* p1q = p1g;
                const float* p2q = p2g;
                const float* p3q = p3g;
                const float* p4q = p4g;
                const float* p5q = p5g;
                const float* p6q = p6g;
                const float* p7q = p7g;
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
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0q, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1q, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(p2q, 0);
                    v4f32 _p3 = (v4f32)__msa_ld_w(p3q, 0);
                    v4f32 _p4 = (v4f32)__msa_ld_w(p4q, 0);
                    v4f32 _p5 = (v4f32)__msa_ld_w(p5q, 0);
                    v4f32 _p6 = (v4f32)__msa_ld_w(p6q, 0);
                    v4f32 _p7 = (v4f32)__msa_ld_w(p7q, 0);
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
                    p0q += 4;
                    p1q += 4;
                    p2q += 4;
                    p3q += 4;
                    p4q += 4;
                    p5q += 4;
                    p6q += 4;
                    p7q += 4;
                }
                if (kk + 1 < max_kk0)
                {
                    pp[0] = float2int8(p0q[0] * scale0);
                    pp[1] = float2int8(p0q[1] * scale0);
                    pp[2] = float2int8(p1q[0] * scale1);
                    pp[3] = float2int8(p1q[1] * scale1);
                    pp[4] = float2int8(p2q[0] * scale2);
                    pp[5] = float2int8(p2q[1] * scale2);
                    pp[6] = float2int8(p3q[0] * scale3);
                    pp[7] = float2int8(p3q[1] * scale3);
                    pp[8] = float2int8(p4q[0] * scale4);
                    pp[9] = float2int8(p4q[1] * scale4);
                    pp[10] = float2int8(p5q[0] * scale5);
                    pp[11] = float2int8(p5q[1] * scale5);
                    pp[12] = float2int8(p6q[0] * scale6);
                    pp[13] = float2int8(p6q[1] * scale6);
                    pp[14] = float2int8(p7q[0] * scale7);
                    pp[15] = float2int8(p7q[1] * scale7);
                    pp += 16;
                    p0q += 2;
                    p1q += 2;
                    p2q += 2;
                    p3q += 2;
                    p4q += 2;
                    p5q += 2;
                    p6q += 2;
                    p7q += 2;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    pp[0] = float2int8(*p0q * scale0);
                    pp[1] = float2int8(*p1q * scale1);
                    pp[2] = float2int8(*p2q * scale2);
                    pp[3] = float2int8(*p3q * scale3);
                    pp[4] = float2int8(*p4q * scale4);
                    pp[5] = float2int8(*p5q * scale5);
                    pp[6] = float2int8(*p6q * scale6);
                    pp[7] = float2int8(*p7q * scale7);
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
            const float* p0 = A_data + (size_t)i0 * A_hstep;
            const float* p1 = A_data + (size_t)i1 * A_hstep;
            const float* p2 = A_data + (size_t)i2 * A_hstep;
            const float* p3 = A_data + (size_t)i3 * A_hstep;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk0 = std::min(max_kk - k0, block_size);
                const float* p0g = p0 + k0;
                const float* p1g = p1 + k0;
                const float* p2g = p2 + k0;
                const float* p3g = p3 + k0;
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                float absmax2 = 0.f;
                float absmax3 = 0.f;

                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax3 = (v4f32)__msa_fill_w(0);

                const float* p0a = p0g;
                const float* p1a = p1g;
                const float* p2a = p2g;
                const float* p3a = p3g;
                int kk = 0;
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

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd += 4;

                const float* p0q = p0g;
                const float* p1q = p1g;
                const float* p2q = p2g;
                const float* p3q = p3g;
                v4f32 _scale0 = __msa_fill_w_f32(scale0);
                v4f32 _scale1 = __msa_fill_w_f32(scale1);
                v4f32 _scale2 = __msa_fill_w_f32(scale2);
                v4f32 _scale3 = __msa_fill_w_f32(scale3);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0q, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1q, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(p2q, 0);
                    v4f32 _p3 = (v4f32)__msa_ld_w(p3q, 0);
                    _p0 = __msa_fmul_w(_p0, _scale0);
                    _p1 = __msa_fmul_w(_p1, _scale1);
                    _p2 = __msa_fmul_w(_p2, _scale2);
                    _p3 = __msa_fmul_w(_p3, _scale3);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    pp += 16;
                    p0q += 4;
                    p1q += 4;
                    p2q += 4;
                    p3q += 4;
                }
                if (kk + 1 < max_kk0)
                {
                    float v00 = p0q[0];
                    float v01 = p0q[1];
                    float v10 = p1q[0];
                    float v11 = p1q[1];
                    float v20 = p2q[0];
                    float v21 = p2q[1];
                    float v30 = p3q[0];
                    float v31 = p3q[1];
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v10 * scale1);
                    pp[3] = float2int8(v11 * scale1);
                    pp[4] = float2int8(v20 * scale2);
                    pp[5] = float2int8(v21 * scale2);
                    pp[6] = float2int8(v30 * scale3);
                    pp[7] = float2int8(v31 * scale3);
                    pp += 8;
                    p0q += 2;
                    p1q += 2;
                    p2q += 2;
                    p3q += 2;
                    kk += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0q++;
                    float v1 = *p1q++;
                    float v2 = *p2q++;
                    float v3 = *p3q++;
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
            const float* p0 = A_data + (size_t)i0 * A_hstep;
            const float* p1 = A_data + (size_t)i1 * A_hstep;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk0 = std::min(max_kk - k0, block_size);
                const float* p0g = p0 + k0;
                const float* p1g = p1 + k0;
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0g;
                const float* p1a = p1g;
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

                const float* p0q = p0g;
                const float* p1q = p1g;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v00 = p0q[0];
                    float v01 = p0q[1];
                    float v02 = p0q[2];
                    float v03 = p0q[3];
                    float v10 = p1q[0];
                    float v11 = p1q[1];
                    float v12 = p1q[2];
                    float v13 = p1q[3];
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v02 * scale0);
                    pp[3] = float2int8(v03 * scale0);
                    pp[4] = float2int8(v10 * scale1);
                    pp[5] = float2int8(v11 * scale1);
                    pp[6] = float2int8(v12 * scale1);
                    pp[7] = float2int8(v13 * scale1);
                    pp += 8;
                    p0q += 4;
                    p1q += 4;
                }
                if (kk + 1 < max_kk0)
                {
                    float v00 = p0q[0];
                    float v01 = p0q[1];
                    float v10 = p1q[0];
                    float v11 = p1q[1];
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v10 * scale1);
                    pp[3] = float2int8(v11 * scale1);
                    pp += 4;
                    p0q += 2;
                    p1q += 2;
                    kk += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0q++;
                    float v1 = *p1q++;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const int i0 = i + ii;
            const float* p0 = A_data + (size_t)i0 * A_hstep;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk0 = std::min(max_kk - k0, block_size);
                const float* p0g = p0 + k0;
                float absmax0 = 0.f;
                const float* p0a = p0g;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = *p0a++;
                    absmax0 = std::max(absmax0, fabsf(v0));
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                *pd++ = absmax0 / 127.f;

                const float* p0q = p0g;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v0 = p0q[0];
                    float v1 = p0q[1];
                    float v2 = p0q[2];
                    float v3 = p0q[3];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    pp[2] = float2int8(v2 * scale0);
                    pp[3] = float2int8(v3 * scale0);
                    pp += 4;
                    p0q += 4;
                }
                if (kk + 1 < max_kk0)
                {
                    float v0 = p0q[0];
                    float v1 = p0q[1];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    pp += 2;
                    p0q += 2;
                    kk += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0q++;
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
        const float* p0 = A_data + (size_t)(i + ii) * A_hstep;
        const float* p1 = A_data + (size_t)(i + ii + 1) * A_hstep;
        const float* p2 = A_data + (size_t)(i + ii + 2) * A_hstep;
        const float* p3 = A_data + (size_t)(i + ii + 3) * A_hstep;
        const float* p4 = A_data + (size_t)(i + ii + 4) * A_hstep;
        const float* p5 = A_data + (size_t)(i + ii + 5) * A_hstep;
        const float* p6 = A_data + (size_t)(i + ii + 6) * A_hstep;
        const float* p7 = A_data + (size_t)(i + ii + 7) * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk0 = std::min(max_kk - k0, block_size);
            const float* p0g = p0 + k0;
            const float* p1g = p1 + k0;
            const float* p2g = p2 + k0;
            const float* p3g = p3 + k0;
            const float* p4g = p4 + k0;
            const float* p5g = p5 + k0;
            const float* p6g = p6 + k0;
            const float* p7g = p7 + k0;
            const float* sg = input_scale_ptr + k0;
            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax3 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax4 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax5 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax6 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax7 = (v4f32)__msa_fill_w(0);

            const float* p0a = p0g;
            const float* p1a = p1g;
            const float* p2a = p2g;
            const float* p3a = p3g;
            const float* p4a = p4g;
            const float* p5a = p5g;
            const float* p6a = p6g;
            const float* p7a = p7g;
            const float* psa = sg;
            int kk = 0;
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
                absmax0 = std::max(absmax0, fabsf(*p0a++) * s);
                absmax1 = std::max(absmax1, fabsf(*p1a++) * s);
                absmax2 = std::max(absmax2, fabsf(*p2a++) * s);
                absmax3 = std::max(absmax3, fabsf(*p3a++) * s);
                absmax4 = std::max(absmax4, fabsf(*p4a++) * s);
                absmax5 = std::max(absmax5, fabsf(*p5a++) * s);
                absmax6 = std::max(absmax6, fabsf(*p6a++) * s);
                absmax7 = std::max(absmax7, fabsf(*p7a++) * s);
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

            const float* p0q = p0g;
            const float* p1q = p1g;
            const float* p2q = p2g;
            const float* p3q = p3g;
            const float* p4q = p4g;
            const float* p5q = p5g;
            const float* p6q = p6g;
            const float* p7q = p7g;
            const float* psq = sg;
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
                v4f32 _p0 = (v4f32)__msa_ld_w(p0q, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(p1q, 0);
                v4f32 _p2 = (v4f32)__msa_ld_w(p2q, 0);
                v4f32 _p3 = (v4f32)__msa_ld_w(p3q, 0);
                v4f32 _p4 = (v4f32)__msa_ld_w(p4q, 0);
                v4f32 _p5 = (v4f32)__msa_ld_w(p5q, 0);
                v4f32 _p6 = (v4f32)__msa_ld_w(p6q, 0);
                v4f32 _p7 = (v4f32)__msa_ld_w(p7q, 0);
                v4f32 _s = (v4f32)__msa_ld_w(psq, 0);
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
                p0q += 4;
                p1q += 4;
                p2q += 4;
                p3q += 4;
                p4q += 4;
                p5q += 4;
                p6q += 4;
                p7q += 4;
                psq += 4;
            }
            if (kk + 1 < max_kk0)
            {
                const float s0 = psq[0];
                const float s1 = psq[1];
                pp[0] = float2int8(p0q[0] * s0 * scale0);
                pp[1] = float2int8(p0q[1] * s1 * scale0);
                pp[2] = float2int8(p1q[0] * s0 * scale1);
                pp[3] = float2int8(p1q[1] * s1 * scale1);
                pp[4] = float2int8(p2q[0] * s0 * scale2);
                pp[5] = float2int8(p2q[1] * s1 * scale2);
                pp[6] = float2int8(p3q[0] * s0 * scale3);
                pp[7] = float2int8(p3q[1] * s1 * scale3);
                pp[8] = float2int8(p4q[0] * s0 * scale4);
                pp[9] = float2int8(p4q[1] * s1 * scale4);
                pp[10] = float2int8(p5q[0] * s0 * scale5);
                pp[11] = float2int8(p5q[1] * s1 * scale5);
                pp[12] = float2int8(p6q[0] * s0 * scale6);
                pp[13] = float2int8(p6q[1] * s1 * scale6);
                pp[14] = float2int8(p7q[0] * s0 * scale7);
                pp[15] = float2int8(p7q[1] * s1 * scale7);
                pp += 16;
                p0q += 2;
                p1q += 2;
                p2q += 2;
                p3q += 2;
                p4q += 2;
                p5q += 2;
                p6q += 2;
                p7q += 2;
                psq += 2;
                kk += 2;
            }
            if (kk < max_kk0)
            {
                const float s = *psq;
                pp[0] = float2int8(*p0q * s * scale0);
                pp[1] = float2int8(*p1q * s * scale1);
                pp[2] = float2int8(*p2q * s * scale2);
                pp[3] = float2int8(*p3q * s * scale3);
                pp[4] = float2int8(*p4q * s * scale4);
                pp[5] = float2int8(*p5q * s * scale5);
                pp[6] = float2int8(*p6q * s * scale6);
                pp[7] = float2int8(*p7q * s * scale7);
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
        const float* p0 = A_data + (size_t)i0 * A_hstep;
        const float* p1 = A_data + (size_t)i1 * A_hstep;
        const float* p2 = A_data + (size_t)i2 * A_hstep;
        const float* p3 = A_data + (size_t)i3 * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk0 = std::min(max_kk - k0, block_size);
            const float* p0g = p0 + k0;
            const float* p1g = p1 + k0;
            const float* p2g = p2 + k0;
            const float* p3g = p3 + k0;
            const float* sg = input_scale_ptr + k0;
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            float absmax2 = 0.f;
            float absmax3 = 0.f;

            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax1 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax2 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax3 = (v4f32)__msa_fill_w(0);

            const float* p0a = p0g;
            const float* p1a = p1g;
            const float* p2a = p2g;
            const float* p3a = p3g;
            const float* psa = sg;
            int kk = 0;
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

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd += 4;

            const float* p0q = p0g;
            const float* p1q = p1g;
            const float* p2q = p2g;
            const float* p3q = p3g;
            const float* psq = sg;
            v4f32 _scale0 = __msa_fill_w_f32(scale0);
            v4f32 _scale1 = __msa_fill_w_f32(scale1);
            v4f32 _scale2 = __msa_fill_w_f32(scale2);
            v4f32 _scale3 = __msa_fill_w_f32(scale3);
            kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                v4f32 _p0 = (v4f32)__msa_ld_w(p0q, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(p1q, 0);
                v4f32 _p2 = (v4f32)__msa_ld_w(p2q, 0);
                v4f32 _p3 = (v4f32)__msa_ld_w(p3q, 0);
                v4f32 _s = (v4f32)__msa_ld_w(psq, 0);
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
                p0q += 4;
                p1q += 4;
                p2q += 4;
                p3q += 4;
                psq += 4;
            }
            if (kk + 1 < max_kk0)
            {
                float v00 = p0q[0];
                float v01 = p0q[1];
                float v10 = p1q[0];
                float v11 = p1q[1];
                float v20 = p2q[0];
                float v21 = p2q[1];
                float v30 = p3q[0];
                float v31 = p3q[1];
                const float s0 = psq[0];
                const float s1 = psq[1];
                v00 *= s0;
                v01 *= s1;
                v10 *= s0;
                v11 *= s1;
                v20 *= s0;
                v21 *= s1;
                v30 *= s0;
                v31 *= s1;
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v10 * scale1);
                pp[3] = float2int8(v11 * scale1);
                pp[4] = float2int8(v20 * scale2);
                pp[5] = float2int8(v21 * scale2);
                pp[6] = float2int8(v30 * scale3);
                pp[7] = float2int8(v31 * scale3);
                pp += 8;
                p0q += 2;
                p1q += 2;
                p2q += 2;
                p3q += 2;
                psq += 2;
                kk += 2;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = *p0q++;
                float v1 = *p1q++;
                float v2 = *p2q++;
                float v3 = *p3q++;
                const float s = *psq++;
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
        const int i0 = i + ii;
        const int i1 = i + ii + 1;
        const float* p0 = A_data + (size_t)i0 * A_hstep;
        const float* p1 = A_data + (size_t)i1 * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk0 = std::min(max_kk - k0, block_size);
            const float* p0g = p0 + k0;
            const float* p1g = p1 + k0;
            const float* sg = input_scale_ptr + k0;
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0g;
            const float* p1a = p1g;
            const float* psa = sg;
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

            const float* p0q = p0g;
            const float* p1q = p1g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v00 = p0q[0];
                float v01 = p0q[1];
                float v02 = p0q[2];
                float v03 = p0q[3];
                float v10 = p1q[0];
                float v11 = p1q[1];
                float v12 = p1q[2];
                float v13 = p1q[3];
                v00 *= psq[0];
                v01 *= psq[1];
                v02 *= psq[2];
                v03 *= psq[3];
                v10 *= psq[0];
                v11 *= psq[1];
                v12 *= psq[2];
                v13 *= psq[3];
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v02 * scale0);
                pp[3] = float2int8(v03 * scale0);
                pp[4] = float2int8(v10 * scale1);
                pp[5] = float2int8(v11 * scale1);
                pp[6] = float2int8(v12 * scale1);
                pp[7] = float2int8(v13 * scale1);
                pp += 8;
                p0q += 4;
                p1q += 4;
                psq += 4;
            }
            if (kk + 1 < max_kk0)
            {
                float v00 = p0q[0];
                float v01 = p0q[1];
                float v10 = p1q[0];
                float v11 = p1q[1];
                v00 *= psq[0];
                v01 *= psq[1];
                v10 *= psq[0];
                v11 *= psq[1];
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v10 * scale1);
                pp[3] = float2int8(v11 * scale1);
                pp += 4;
                p0q += 2;
                p1q += 2;
                psq += 2;
                kk += 2;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = *p0q++;
                float v1 = *p1q++;
                const float s = *psq++;
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
        const int i0 = i + ii;
        const float* p0 = A_data + (size_t)i0 * A_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk0 = std::min(max_kk - k0, block_size);
            const float* p0g = p0 + k0;
            const float* sg = input_scale_ptr + k0;
            float absmax0 = 0.f;
            const float* p0a = p0g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = *p0a++;
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(v0) * s);
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            *pd++ = absmax0 / 127.f;

            const float* p0q = p0g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v0 = p0q[0];
                float v1 = p0q[1];
                float v2 = p0q[2];
                float v3 = p0q[3];
                v0 *= psq[0];
                v1 *= psq[1];
                v2 *= psq[2];
                v3 *= psq[3];
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale0);
                pp[2] = float2int8(v2 * scale0);
                pp[3] = float2int8(v3 * scale0);
                pp += 4;
                p0q += 4;
                psq += 4;
            }
            if (kk + 1 < max_kk0)
            {
                float v0 = p0q[0];
                float v1 = p0q[1];
                v0 *= psq[0];
                v1 *= psq[1];
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale0);
                pp += 2;
                p0q += 2;
                psq += 2;
                kk += 2;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = *p0q++;
                v0 *= *psq++;
                *pp++ = float2int8(v0 * scale0);
            }
        }
    }
}

// group-major, row-major within each K4/K2/K1 fragment
static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        transpose_quantize_A_tile_wq_int8_loongson_mmi(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const float* A_data = (const float*)A + (size_t)k * A_hstep;

    if (input_scales.empty())
    {
        int ii = 0;
#if __mips_msa
        for (; ii + 7 < max_ii; ii += 8)
        {
            const int i0 = i + ii;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk0 = std::min(max_kk - k0, block_size);
                const float* p0g = A_data + (size_t)k0 * A_hstep + i0;
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
                v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

                const float* p0a = p0g;
                int kk = 0;
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0a, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p0a + 4, 0);
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

                const float* p0q = p0g;
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
                    const float* p0 = p0q;
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
                    p0q = p3 + A_hstep;
                }
                if (kk + 1 < max_kk0)
                {
                    const float* p0 = p0q;
                    const float* p1 = p0 + A_hstep;
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                    v4f32 _scale0123 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                    v4f32 _scale4567 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));
                    v16i8 _q0 = float2int8(__msa_fmul_w(_p0, _scale0123));
                    v16i8 _q1 = float2int8(__msa_fmul_w(_p1, _scale0123));
                    pp[0] = __msa_copy_s_b(_q0, 0);
                    pp[1] = __msa_copy_s_b(_q1, 0);
                    pp[2] = __msa_copy_s_b(_q0, 1);
                    pp[3] = __msa_copy_s_b(_q1, 1);
                    pp[4] = __msa_copy_s_b(_q0, 2);
                    pp[5] = __msa_copy_s_b(_q1, 2);
                    pp[6] = __msa_copy_s_b(_q0, 3);
                    pp[7] = __msa_copy_s_b(_q1, 3);
                    _p0 = (v4f32)__msa_ld_w(p0 + 4, 0);
                    _p1 = (v4f32)__msa_ld_w(p1 + 4, 0);
                    _q0 = float2int8(__msa_fmul_w(_p0, _scale4567));
                    _q1 = float2int8(__msa_fmul_w(_p1, _scale4567));
                    pp[8] = __msa_copy_s_b(_q0, 0);
                    pp[9] = __msa_copy_s_b(_q1, 0);
                    pp[10] = __msa_copy_s_b(_q0, 1);
                    pp[11] = __msa_copy_s_b(_q1, 1);
                    pp[12] = __msa_copy_s_b(_q0, 2);
                    pp[13] = __msa_copy_s_b(_q1, 2);
                    pp[14] = __msa_copy_s_b(_q0, 3);
                    pp[15] = __msa_copy_s_b(_q1, 3);
                    pp += 16;
                    p0q = p1 + A_hstep;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v4f32 _p0 = (v4f32)__msa_ld_w(p0q, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(p0q + 4, 0);
                    v4f32 _scale0123 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                    v4f32 _scale4567 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));
                    v16i8 _q0 = float2int8(__msa_fmul_w(_p0, _scale0123));
                    v16i8 _q1 = float2int8(__msa_fmul_w(_p1, _scale4567));
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
                const int max_kk0 = std::min(max_kk - k0, block_size);
                const float* p0g = A_data + (size_t)k0 * A_hstep + i0;
                const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
                v4f32 _absmax = (v4f32)__msa_fill_w(0);

                const float* p0a = p0g;
                int kk = 0;
                for (; kk < max_kk0; kk++)
                {
                    v4f32 _p = (v4f32)__msa_ld_w(p0a, 0);
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
                const float* p0q = p0g;
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const float* p0 = p0q;
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
                    p0q = p3 + A_hstep;
                }
                if (kk + 1 < max_kk0)
                {
                    const float* p0 = p0q;
                    const float* p1 = p0 + A_hstep;
                    float v00 = p0[0];
                    float v10 = p0[1];
                    float v20 = p0[2];
                    float v30 = p0[3];
                    float v01 = p1[0];
                    float v11 = p1[1];
                    float v21 = p1[2];
                    float v31 = p1[3];
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v10 * scale1);
                    pp[3] = float2int8(v11 * scale1);
                    pp[4] = float2int8(v20 * scale2);
                    pp[5] = float2int8(v21 * scale2);
                    pp[6] = float2int8(v30 * scale3);
                    pp[7] = float2int8(v31 * scale3);
                    pp += 8;
                    p0q = p1 + A_hstep;
                    kk += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = p0q[0];
                    float v1 = p0q[1];
                    float v2 = p0q[2];
                    float v3 = p0q[3];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp[2] = float2int8(v2 * scale2);
                    pp[3] = float2int8(v3 * scale3);
                    pp += 4;
                    p0q += A_hstep;
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
                const int max_kk0 = std::min(max_kk - k0, block_size);
                const float* p0g = A_data + (size_t)k0 * A_hstep + i0;
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0g;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = p0a[0];
                    float v1 = p0a[1];
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                    p0a += A_hstep;
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                const float* p0q = p0g;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v00 = p0q[0];
                    float v10 = p0q[1];
                    float v01 = p0q[A_hstep];
                    float v11 = p0q[A_hstep + 1];
                    float v02 = p0q[A_hstep * 2];
                    float v12 = p0q[A_hstep * 2 + 1];
                    float v03 = p0q[A_hstep * 3];
                    float v13 = p0q[A_hstep * 3 + 1];
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v02 * scale0);
                    pp[3] = float2int8(v03 * scale0);
                    pp[4] = float2int8(v10 * scale1);
                    pp[5] = float2int8(v11 * scale1);
                    pp[6] = float2int8(v12 * scale1);
                    pp[7] = float2int8(v13 * scale1);
                    p0q += A_hstep * 4;
                    pp += 8;
                }
                if (kk + 1 < max_kk0)
                {
                    float v00 = p0q[0];
                    float v10 = p0q[1];
                    float v01 = p0q[A_hstep];
                    float v11 = p0q[A_hstep + 1];
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v10 * scale1);
                    pp[3] = float2int8(v11 * scale1);
                    p0q += A_hstep * 2;
                    pp += 4;
                    kk += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = p0q[0];
                    float v1 = p0q[1];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                    p0q += A_hstep;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const int i0 = i + ii;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk0 = std::min(max_kk - k0, block_size);
                const float* p0g = A_data + (size_t)k0 * A_hstep + i0;
                float absmax0 = 0.f;
                const float* p0a = p0g;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = *p0a;
                    absmax0 = std::max(absmax0, fabsf(v0));
                    p0a += A_hstep;
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                *pd++ = absmax0 / 127.f;

                const float* p0q = p0g;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v0 = p0q[0];
                    float v1 = p0q[A_hstep];
                    float v2 = p0q[A_hstep * 2];
                    float v3 = p0q[A_hstep * 3];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    pp[2] = float2int8(v2 * scale0);
                    pp[3] = float2int8(v3 * scale0);
                    p0q += A_hstep * 4;
                    pp += 4;
                }
                if (kk + 1 < max_kk0)
                {
                    float v0 = p0q[0];
                    float v1 = p0q[A_hstep];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    p0q += A_hstep * 2;
                    pp += 2;
                    kk += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0q;
                    *pp++ = float2int8(v0 * scale0);
                    p0q += A_hstep;
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
        const int i0 = i + ii;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk0 = std::min(max_kk - k0, block_size);
            const float* p0g = A_data + (size_t)k0 * A_hstep + i0;
            const float* sg = input_scale_ptr + k0;
            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax0 = (v4f32)__msa_fill_w(0);
            v4f32 _absmax1 = (v4f32)__msa_fill_w(0);

            const float* p0a = p0g;
            const float* psa = sg;
            int kk = 0;
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

            const float* p0q = p0g;
            const float* psq = sg;
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
                const float* p0 = p0q;
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
                v4f32 _s = (v4f32)__msa_ld_w(psq, 0);
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
                p0q = p3 + A_hstep;
                psq += 4;
            }
            if (kk + 1 < max_kk0)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                const float s0 = psq[0];
                const float s1 = psq[1];
                v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), __msa_fill_w_f32(s0));
                v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p1, 0), __msa_fill_w_f32(s1));
                v4f32 _scale0123 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                v4f32 _scale4567 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));
                v16i8 _q0 = float2int8(__msa_fmul_w(_p0, _scale0123));
                v16i8 _q1 = float2int8(__msa_fmul_w(_p1, _scale0123));
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
                _q0 = float2int8(__msa_fmul_w(_p0, _scale4567));
                _q1 = float2int8(__msa_fmul_w(_p1, _scale4567));
                pp[8] = __msa_copy_s_b(_q0, 0);
                pp[9] = __msa_copy_s_b(_q1, 0);
                pp[10] = __msa_copy_s_b(_q0, 1);
                pp[11] = __msa_copy_s_b(_q1, 1);
                pp[12] = __msa_copy_s_b(_q0, 2);
                pp[13] = __msa_copy_s_b(_q1, 2);
                pp[14] = __msa_copy_s_b(_q0, 3);
                pp[15] = __msa_copy_s_b(_q1, 3);
                pp += 16;
                p0q = p1 + A_hstep;
                psq += 2;
                kk += 2;
            }
            if (kk < max_kk0)
            {
                const float s = *psq;
                v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0q, 0), __msa_fill_w_f32(s));
                v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p0q + 4, 0), __msa_fill_w_f32(s));
                v4f32 _scale0123 = (v4f32)__msa_set_w(__msa_load_w(&scale0), __msa_load_w(&scale1), __msa_load_w(&scale2), __msa_load_w(&scale3));
                v4f32 _scale4567 = (v4f32)__msa_set_w(__msa_load_w(&scale4), __msa_load_w(&scale5), __msa_load_w(&scale6), __msa_load_w(&scale7));
                v16i8 _q0 = float2int8(__msa_fmul_w(_p0, _scale0123));
                v16i8 _q1 = float2int8(__msa_fmul_w(_p1, _scale4567));
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
            const int max_kk0 = std::min(max_kk - k0, block_size);
            const float* p0g = A_data + (size_t)k0 * A_hstep + i0;
            const float* sg = input_scale_ptr + k0;
            const v16u8 _abs_mask = (v16u8)__msa_fill_w(0x7fffffff);
            v4f32 _absmax = (v4f32)__msa_fill_w(0);

            const float* p0a = p0g;
            const float* psa = sg;
            int kk = 0;
            for (; kk < max_kk0; kk++)
            {
                v4f32 _p = (v4f32)__msa_ld_w(p0a, 0);
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
            const float* p0q = p0g;
            const float* psq = sg;
            kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
                v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
                v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                v4f32 _s = (v4f32)__msa_ld_w(psq, 0);
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
                p0q = p3 + A_hstep;
                psq += 4;
            }
            if (kk + 1 < max_kk0)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                float v00 = p0[0];
                float v10 = p0[1];
                float v20 = p0[2];
                float v30 = p0[3];
                float v01 = p1[0];
                float v11 = p1[1];
                float v21 = p1[2];
                float v31 = p1[3];
                const float s0 = psq[0];
                const float s1 = psq[1];
                v00 *= s0;
                v10 *= s0;
                v20 *= s0;
                v30 *= s0;
                v01 *= s1;
                v11 *= s1;
                v21 *= s1;
                v31 *= s1;
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v10 * scale1);
                pp[3] = float2int8(v11 * scale1);
                pp[4] = float2int8(v20 * scale2);
                pp[5] = float2int8(v21 * scale2);
                pp[6] = float2int8(v30 * scale3);
                pp[7] = float2int8(v31 * scale3);
                pp += 8;
                p0q = p1 + A_hstep;
                psq += 2;
                kk += 2;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = p0q[0];
                float v1 = p0q[1];
                float v2 = p0q[2];
                float v3 = p0q[3];
                const float s = *psq++;
                v0 *= s;
                v1 *= s;
                v2 *= s;
                v3 *= s;
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp[2] = float2int8(v2 * scale2);
                pp[3] = float2int8(v3 * scale3);
                pp += 4;
                p0q += A_hstep;
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
            const int max_kk0 = std::min(max_kk - k0, block_size);
            const float* p0g = A_data + (size_t)k0 * A_hstep + i0;
            const float* sg = input_scale_ptr + k0;
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = p0a[0];
                float v1 = p0a[1];
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

            const float* p0q = p0g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v00 = p0q[0];
                float v10 = p0q[1];
                float v01 = p0q[A_hstep];
                float v11 = p0q[A_hstep + 1];
                float v02 = p0q[A_hstep * 2];
                float v12 = p0q[A_hstep * 2 + 1];
                float v03 = p0q[A_hstep * 3];
                float v13 = p0q[A_hstep * 3 + 1];
                v00 *= psq[0];
                v10 *= psq[0];
                v01 *= psq[1];
                v11 *= psq[1];
                v02 *= psq[2];
                v12 *= psq[2];
                v03 *= psq[3];
                v13 *= psq[3];
                psq += 4;
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v02 * scale0);
                pp[3] = float2int8(v03 * scale0);
                pp[4] = float2int8(v10 * scale1);
                pp[5] = float2int8(v11 * scale1);
                pp[6] = float2int8(v12 * scale1);
                pp[7] = float2int8(v13 * scale1);
                p0q += A_hstep * 4;
                pp += 8;
            }
            if (kk + 1 < max_kk0)
            {
                float v00 = p0q[0];
                float v10 = p0q[1];
                float v01 = p0q[A_hstep];
                float v11 = p0q[A_hstep + 1];
                v00 *= psq[0];
                v10 *= psq[0];
                v01 *= psq[1];
                v11 *= psq[1];
                psq += 2;
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v10 * scale1);
                pp[3] = float2int8(v11 * scale1);
                p0q += A_hstep * 2;
                pp += 4;
                kk += 2;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = p0q[0];
                float v1 = p0q[1];
                const float s = *psq++;
                v0 *= s;
                v1 *= s;
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
                p0q += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const int i0 = i + ii;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk0 = std::min(max_kk - k0, block_size);
            const float* p0g = A_data + (size_t)k0 * A_hstep + i0;
            const float* sg = input_scale_ptr + k0;
            float absmax0 = 0.f;
            const float* p0a = p0g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = *p0a;
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(v0) * s);
                p0a += A_hstep;
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            *pd++ = absmax0 / 127.f;

            const float* p0q = p0g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v0 = p0q[0];
                float v1 = p0q[A_hstep];
                float v2 = p0q[A_hstep * 2];
                float v3 = p0q[A_hstep * 3];
                v0 *= psq[0];
                v1 *= psq[1];
                v2 *= psq[2];
                v3 *= psq[3];
                psq += 4;
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale0);
                pp[2] = float2int8(v2 * scale0);
                pp[3] = float2int8(v3 * scale0);
                p0q += A_hstep * 4;
                pp += 4;
            }
            if (kk + 1 < max_kk0)
            {
                float v0 = p0q[0];
                float v1 = p0q[A_hstep];
                v0 *= psq[0];
                v1 *= psq[1];
                psq += 2;
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale0);
                p0q += A_hstep * 2;
                pp += 2;
                kk += 2;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = *p0q;
                v0 *= *psq++;
                *pp++ = float2int8(v0 * scale0);
                p0q += A_hstep;
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
    const int A_hstep = max_kk;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = (max_kk + block_size - 1) / block_size;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
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
                transpose4x4_ps(_fsum0, _fsum1, _fsum2, _fsum3);
                transpose4x4_ps(_fsum4, _fsum5, _fsum6, _fsum7);
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

                if (kk + 1 < max_kk0)
                {
                    v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    v8i16 _pA = (v8i16)__msa_ld_b(pA, 0);
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
                if (kk < max_kk0)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_d_ptr(pA);
                    v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
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

                v4f32 _descaleB = (v4f32)__msa_ld_w(pB_descales, 0);
                v4f32 _descaleA0 = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleA1 = (v4f32)__msa_ld_w(pA_descales + 4, 0);
                v4f32 _scale = __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA0, 0));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA0, 1));
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA0, 2));
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA0, 3));
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                _scale = __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA1, 0));
                _fsum4 = __ncnn_msa_fmadd_w(_fsum4, (v4f32)__msa_ffint_s_w(_sum4), _scale);
                _scale = __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA1, 1));
                _fsum5 = __ncnn_msa_fmadd_w(_fsum5, (v4f32)__msa_ffint_s_w(_sum5), _scale);
                _scale = __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA1, 2));
                _fsum6 = __ncnn_msa_fmadd_w(_fsum6, (v4f32)__msa_ffint_s_w(_sum6), _scale);
                _scale = __msa_fmul_w(_descaleB, (v4f32)__msa_splati_w((v4i32)_descaleA1, 3));
                _fsum7 = __ncnn_msa_fmadd_w(_fsum7, (v4f32)__msa_ffint_s_w(_sum7), _scale);
                pA_descales += 8;
                pB_descales += 4;
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

                v4i32 _sum0e = __msa_shf_w(_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
                v4i32 _sum0o = __msa_shf_w(_sum0, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum1e = __msa_shf_w(_sum1, _MSA_SHUFFLE(3, 1, 2, 0));
                v4i32 _sum1o = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum0x = (v4i32)__msa_ilvr_w(_sum1o, _sum0e);
                v4i32 _sum1x = (v4i32)__msa_ilvr_w(_sum0o, _sum1e);
                v4i32 _sum2e = __msa_shf_w(_sum2, _MSA_SHUFFLE(3, 1, 2, 0));
                v4i32 _sum2o = __msa_shf_w(_sum2, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum3e = __msa_shf_w(_sum3, _MSA_SHUFFLE(3, 1, 2, 0));
                v4i32 _sum3o = __msa_shf_w(_sum3, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum2x = (v4i32)__msa_ilvr_w(_sum3o, _sum2e);
                v4i32 _sum3x = (v4i32)__msa_ilvr_w(_sum2o, _sum3e);

                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v8i16 _pB = (v8i16)__msa_fill_w(*(const int*)pB);
                    v8i16 _s0 = __msa_dotp_s_h(_pA, (v16i8)__msa_splati_h(_pB, 0));
                    v8i16 _s1 = __msa_dotp_s_h(_pA, (v16i8)__msa_splati_h(_pB, 1));
                    v8i16 _sign0 = __msa_clti_s_h(_s0, 0);
                    v8i16 _sign1 = __msa_clti_s_h(_s1, 0);
                    _sum0x = __msa_addv_w(_sum0x, (v4i32)__msa_ilvr_h(_sign0, _s0));
                    _sum1x = __msa_addv_w(_sum1x, (v4i32)__msa_ilvr_h(_sign1, _s1));
                    _sum2x = __msa_addv_w(_sum2x, (v4i32)__msa_ilvl_h(_sign0, _s0));
                    _sum3x = __msa_addv_w(_sum3x, (v4i32)__msa_ilvl_h(_sign1, _s1));
                    pA += 16;
                    pB += 4;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_d_ptr(pA);
                    v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    v16i8 _pB8 = (v16i8)__msa_fill_h(*(const short*)pB);
                    v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(__msa_splati_b(_pB8, 0), 0), __msa_splati_b(_pB8, 0));
                    v8i16 _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(__msa_splati_b(_pB8, 1), 0), __msa_splati_b(_pB8, 1));
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA, _pB1);
                    v8i16 _sign0 = __msa_clti_s_h(_s0, 0);
                    v8i16 _sign1 = __msa_clti_s_h(_s1, 0);
                    _sum0x = __msa_addv_w(_sum0x, (v4i32)__msa_ilvr_h(_sign0, _s0));
                    _sum1x = __msa_addv_w(_sum1x, (v4i32)__msa_ilvr_h(_sign1, _s1));
                    _sum2x = __msa_addv_w(_sum2x, (v4i32)__msa_ilvl_h(_sign0, _s0));
                    _sum3x = __msa_addv_w(_sum3x, (v4i32)__msa_ilvl_h(_sign1, _s1));
                    pA += 8;
                    pB += 2;
                }

                v4f32 _descaleA0 = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _descaleA1 = (v4f32)__msa_ld_w(pA_descales + 4, 0);
                v4f32 _scale = __msa_fmul_w(_descaleA0, __msa_fill_w_f32(pB_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0x), _scale);
                _scale = __msa_fmul_w(_descaleA1, __msa_fill_w_f32(pB_descales[0]));
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum2x), _scale);
                _scale = __msa_fmul_w(_descaleA0, __msa_fill_w_f32(pB_descales[1]));
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum1x), _scale);
                _scale = __msa_fmul_w(_descaleA1, __msa_fill_w_f32(pB_descales[1]));
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3x), _scale);
                pA_descales += 8;
                pB_descales += 2;
            }

            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 8, 0);
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
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pB = (v16i8)__msa_fill_h(*(const short*)pB);
                    v8i16 _s = __msa_dotp_s_h(_pA, _pB);
                    v8i16 _sign = __msa_clti_s_h(_s, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s));
                    pA += 16;
                    pB += 2;
                    kk += 2;
                }
                if (kk < max_kk0)
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
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            const signed char* pB0 = pB_panel + (size_t)4 * k;
            const signed char* pB1 = pB_panel + (size_t)4 * K + (size_t)4 * k;
            const float* pB_descales0 = pB_descales_panel + (size_t)4 * block_start;
            const float* pB_descales1 = pB_descales_panel + (size_t)4 * block_count + (size_t)4 * block_start;
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
                _fsum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _fsum2 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _fsum3 = (v4f32)__msa_ld_w(outptr + 12, 0);
                _fsum4 = (v4f32)__msa_ld_w(outptr + 16, 0);
                _fsum5 = (v4f32)__msa_ld_w(outptr + 20, 0);
                _fsum6 = (v4f32)__msa_ld_w(outptr + 24, 0);
                _fsum7 = (v4f32)__msa_ld_w(outptr + 28, 0);
                transpose4x4_ps(_fsum0, _fsum1, _fsum2, _fsum3);
                transpose4x4_ps(_fsum4, _fsum5, _fsum6, _fsum7);
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
                    __builtin_prefetch(pB0 + 64);
                    __builtin_prefetch(pB1 + 64);
                    v16i8 _pA = __msa_ld_b(pA, 0);
                    v16i8 _pAr = (v16i8)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                    v16i8 _pB0 = __msa_ld_b(pB0, 0);
                    v16i8 _pB1 = __msa_ld_b(pB1, 0);
                    v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                    v16i8 _pB1r = (v16i8)__msa_shf_w((v4i32)_pB1, _MSA_SHUFFLE(0, 3, 2, 1));
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

                if (kk + 1 < max_kk0)
                {
                    v8i16 _pA = (v8i16)__msa_fill_d_ptr(pA);
                    v16i8 _pB0 = (v16i8)__msa_fill_d_ptr(pB0);
                    v16i8 _pB1 = (v16i8)__msa_fill_d_ptr(pB1);
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
                    pA += 8;
                    pB0 += 8;
                    pB1 += 8;
                    kk += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_w(*(const int*)pA);
                    v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    v16i8 _pB08 = (v16i8)__msa_fill_w(*(const int*)pB0);
                    v16i8 _pB18 = (v16i8)__msa_fill_w(*(const int*)pB1);
                    v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB08, 0), _pB08);
                    v8i16 _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB18, 0), _pB18);
                    v8i16 _s = __msa_mulv_h(__msa_splati_h(_pA, 0), _pB0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 1), _pB0);
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 2), _pB0);
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 3), _pB0);
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 0), _pB1);
                    _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 1), _pB1);
                    _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 2), _pB1);
                    _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    _s = __msa_mulv_h(__msa_splati_h(_pA, 3), _pB1);
                    _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 4;
                    pB0 += 4;
                    pB1 += 4;
                }

                v4f32 _descaleB0 = (v4f32)__msa_ld_w(pB_descales0, 0);
                v4f32 _descaleB1 = (v4f32)__msa_ld_w(pB_descales1, 0);
                v4f32 _scale = __msa_fmul_w(_descaleB0, __msa_fill_w_f32(pA_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleB0, __msa_fill_w_f32(pA_descales[1]));
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleB0, __msa_fill_w_f32(pA_descales[2]));
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleB0, __msa_fill_w_f32(pA_descales[3]));
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                _scale = __msa_fmul_w(_descaleB1, __msa_fill_w_f32(pA_descales[0]));
                _fsum4 = __ncnn_msa_fmadd_w(_fsum4, (v4f32)__msa_ffint_s_w(_sum4), _scale);
                _scale = __msa_fmul_w(_descaleB1, __msa_fill_w_f32(pA_descales[1]));
                _fsum5 = __ncnn_msa_fmadd_w(_fsum5, (v4f32)__msa_ffint_s_w(_sum5), _scale);
                _scale = __msa_fmul_w(_descaleB1, __msa_fill_w_f32(pA_descales[2]));
                _fsum6 = __ncnn_msa_fmadd_w(_fsum6, (v4f32)__msa_ffint_s_w(_sum6), _scale);
                _scale = __msa_fmul_w(_descaleB1, __msa_fill_w_f32(pA_descales[3]));
                _fsum7 = __ncnn_msa_fmadd_w(_fsum7, (v4f32)__msa_ffint_s_w(_sum7), _scale);
                pA_descales += 4;
                pB_descales0 += 4;
                pB_descales1 += 4;
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
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
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
                transpose4x4_ps(_fsum0, _fsum1, _fsum2, _fsum3);
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
                v8i16 _sum2_0 = __msa_fill_h(0);
                v8i16 _sum2_1 = __msa_fill_h(0);
                v8i16 _sum2_2 = __msa_fill_h(0);
                v8i16 _sum2_3 = __msa_fill_h(0);
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    v8i16 _pA = (v8i16)__msa_fill_d_ptr(pA);
                    v16i8 _pA0 = (v16i8)__msa_splati_h(_pA, 0);
                    v16i8 _pA1 = (v16i8)__msa_splati_h(_pA, 1);
                    v16i8 _pA2 = (v16i8)__msa_splati_h(_pA, 2);
                    v16i8 _pA3 = (v16i8)__msa_splati_h(_pA, 3);
                    _sum2_0 = __msa_dotp_s_h(_pA0, _pB);
                    _sum2_1 = __msa_dotp_s_h(_pA1, _pB);
                    _sum2_2 = __msa_dotp_s_h(_pA2, _pB);
                    _sum2_3 = __msa_dotp_s_h(_pA3, _pB);
                    pA += 8;
                    pB += 8;
                    kk += 2;
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
                v4f32 _descaleB = (v4f32)__msa_ld_w(pB_descales, 0);
                v4f32 _scale = __msa_fmul_w(_descaleB, __msa_fill_w_f32(pA_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleB, __msa_fill_w_f32(pA_descales[1]));
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleB, __msa_fill_w_f32(pA_descales[2]));
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleB, __msa_fill_w_f32(pA_descales[3]));
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                pA_descales += 4;
                pB_descales += 4;
            }
            transpose4x4_ps(_fsum0, _fsum1, _fsum2, _fsum3);
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
                v8i16 _sum2_0 = __msa_fill_h(0);
                v8i16 _sum2_1 = __msa_fill_h(0);
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    v8i16 _pB = (v8i16)__msa_fill_w(*(const int*)pB);
                    v16i8 _pB0 = (v16i8)__msa_splati_h(_pB, 0);
                    v16i8 _pB1 = (v16i8)__msa_splati_h(_pB, 1);
                    _sum2_0 = __msa_dotp_s_h(_pA, _pB0);
                    _sum2_1 = __msa_dotp_s_h(_pA, _pB1);
                    pA += 8;
                    pB += 4;
                    kk += 2;
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
                v4i32 _sum0e = __msa_shf_w(_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
                v4i32 _sum0o = __msa_shf_w(_sum0, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum1e = __msa_shf_w(_sum1, _MSA_SHUFFLE(3, 1, 2, 0));
                v4i32 _sum1o = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 0, 3, 1));
                v4i32 _sum0x = (v4i32)__msa_ilvr_w(_sum1o, _sum0e);
                v4i32 _sum1x = (v4i32)__msa_ilvr_w(_sum0o, _sum1e);
                _sum0x = __msa_addv_w(_sum0x, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_0, 0), _sum2_0));
                _sum1x = __msa_addv_w(_sum1x, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_1, 0), _sum2_1));
                v4f32 _descaleA = (v4f32)__msa_ld_w(pA_descales, 0);
                v4f32 _scale = __msa_fmul_w(_descaleA, __msa_fill_w_f32(pB_descales[0]));
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0x), _scale);
                _scale = __msa_fmul_w(_descaleA, __msa_fill_w_f32(pB_descales[1]));
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1x), _scale);
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
                v8i16 _sum2_0 = __msa_fill_h(0);
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    v16i8 _pB = (v16i8)__msa_fill_h(*(const short*)pB);
                    _sum2_0 = __msa_dotp_s_h(_pA, _pB);
                    pA += 8;
                    pB += 2;
                    kk += 2;
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
                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_sum2_0, 0), _sum2_0));
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
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            const signed char* pB0 = pB_panel + (size_t)4 * k;
            const signed char* pB1 = pB_panel + (size_t)4 * K + (size_t)4 * k;
            const float* pB_descales0 = pB_descales_panel + (size_t)4 * block_start;
            const float* pB_descales1 = pB_descales_panel + (size_t)4 * block_count + (size_t)4 * block_start;
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
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB0 + 64);
                    __builtin_prefetch(pB1 + 64);
                    v16i8 _pA = (v16i8)__msa_fill_d_ptr(pA);
                    v16i8 _pB0 = __msa_ld_b(pB0, 0);
                    v16i8 _pB00 = (v16i8)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                    v16i8 _pB01 = (v16i8)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);
                    v16i8 _pB1 = __msa_ld_b(pB1, 0);
                    v16i8 _pB10 = (v16i8)__msa_ilvr_w((v4i32)_pB1, (v4i32)_pB1);
                    v16i8 _pB11 = (v16i8)__msa_ilvl_w((v4i32)_pB1, (v4i32)_pB1);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB00), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB01), _one);
                    _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA, _pB10), _one);
                    _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA, _pB11), _one);
                    pA += 8;
                    pB0 += 16;
                    pB1 += 16;
                }
                if (kk + 1 < max_kk0)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    v16i8 _pA0 = (v16i8)__msa_splati_h(_pA, 0);
                    v16i8 _pA1 = (v16i8)__msa_splati_h(_pA, 1);
                    v16i8 _pB0 = (v16i8)__msa_fill_d_ptr(pB0);
                    v16i8 _pB1 = (v16i8)__msa_fill_d_ptr(pB1);
                    v8i16 _s00 = __msa_dotp_s_h(_pA0, _pB0);
                    v8i16 _s01 = __msa_dotp_s_h(_pA1, _pB0);
                    v8i16 _s10 = __msa_dotp_s_h(_pA0, _pB1);
                    v8i16 _s11 = __msa_dotp_s_h(_pA1, _pB1);
                    v8i16 _s0 = (v8i16)__msa_ilvr_h(_s01, _s00);
                    v8i16 _s2 = (v8i16)__msa_ilvr_h(_s11, _s10);
                    v8i16 _sign0 = __msa_clti_s_h(_s0, 0);
                    v8i16 _sign2 = __msa_clti_s_h(_s2, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign0, _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign0, _s0));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(_sign2, _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvl_h(_sign2, _s2));
                    pA += 4;
                    pB0 += 8;
                    pB1 += 8;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_h(*(const short*)pA);
                    v16i8 _pA0b = __msa_splati_b(_pA8, 0);
                    v16i8 _pA1b = __msa_splati_b(_pA8, 1);
                    v8i16 _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA0b, 0), _pA0b);
                    v8i16 _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA1b, 0), _pA1b);
                    v16i8 _pB08 = (v16i8)__msa_fill_w(*(const int*)pB0);
                    v16i8 _pB18 = (v16i8)__msa_fill_w(*(const int*)pB1);
                    v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB08, 0), _pB08);
                    v8i16 _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB18, 0), _pB18);
                    v8i16 _s00 = __msa_mulv_h(_pA0, _pB0);
                    v8i16 _s01 = __msa_mulv_h(_pA1, _pB0);
                    v8i16 _s10 = __msa_mulv_h(_pA0, _pB1);
                    v8i16 _s11 = __msa_mulv_h(_pA1, _pB1);
                    v8i16 _s0 = (v8i16)__msa_ilvr_h(_s01, _s00);
                    v8i16 _s2 = (v8i16)__msa_ilvr_h(_s11, _s10);
                    v8i16 _sign0 = __msa_clti_s_h(_s0, 0);
                    v8i16 _sign2 = __msa_clti_s_h(_s2, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign0, _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign0, _s0));
                    _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(_sign2, _s2));
                    _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvl_h(_sign2, _s2));
                    pA += 2;
                    pB0 += 4;
                    pB1 += 4;
                }
                v4f32 _descaleA = (v4f32)__msa_fill_d_ptr(pA_descales);
                v4f32 _descaleB0 = (v4f32)__msa_set_w(__msa_load_w(pB_descales0), __msa_load_w(pB_descales0), __msa_load_w(pB_descales0 + 1), __msa_load_w(pB_descales0 + 1));
                v4f32 _descaleB1 = (v4f32)__msa_set_w(__msa_load_w(pB_descales0 + 2), __msa_load_w(pB_descales0 + 2), __msa_load_w(pB_descales0 + 3), __msa_load_w(pB_descales0 + 3));
                v4f32 _descaleB2 = (v4f32)__msa_set_w(__msa_load_w(pB_descales1), __msa_load_w(pB_descales1), __msa_load_w(pB_descales1 + 1), __msa_load_w(pB_descales1 + 1));
                v4f32 _descaleB3 = (v4f32)__msa_set_w(__msa_load_w(pB_descales1 + 2), __msa_load_w(pB_descales1 + 2), __msa_load_w(pB_descales1 + 3), __msa_load_w(pB_descales1 + 3));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB0);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleB1);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleB2);
                _fsum2 = __ncnn_msa_fmadd_w(_fsum2, (v4f32)__msa_ffint_s_w(_sum2), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleB3);
                _fsum3 = __ncnn_msa_fmadd_w(_fsum3, (v4f32)__msa_ffint_s_w(_sum3), _scale);
                pA_descales += 2;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            __msa_st_w((v4i32)_fsum2, outptr + 8, 0);
            __msa_st_w((v4i32)_fsum3, outptr + 12, 0);
            outptr += 16;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
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
                if (kk + 1 < max_kk0)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    v16i8 _pA0 = (v16i8)__msa_splati_h(_pA, 0);
                    v16i8 _pA1 = (v16i8)__msa_splati_h(_pA, 1);
                    v16i8 _pB = (v16i8)__msa_fill_d_ptr(pB);
                    v8i16 _s00 = __msa_dotp_s_h(_pA0, _pB);
                    v8i16 _s01 = __msa_dotp_s_h(_pA1, _pB);
                    v8i16 _s0 = (v8i16)__msa_ilvr_h(_s01, _s00);
                    v8i16 _sign = __msa_clti_s_h(_s0, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s0));
                    pA += 4;
                    pB += 8;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_h(*(const short*)pA);
                    v16i8 _pA0b = __msa_splati_b(_pA8, 0);
                    v16i8 _pA1b = __msa_splati_b(_pA8, 1);
                    v8i16 _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA0b, 0), _pA0b);
                    v8i16 _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA1b, 0), _pA1b);
                    v16i8 _pB8 = (v16i8)__msa_fill_w(*(const int*)pB);
                    v8i16 _pB = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    v8i16 _s00 = __msa_mulv_h(_pA0, _pB);
                    v8i16 _s01 = __msa_mulv_h(_pA1, _pB);
                    v8i16 _s0 = (v8i16)__msa_ilvr_h(_s01, _s00);
                    v8i16 _sign = __msa_clti_s_h(_s0, 0);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(_sign, _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvl_h(_sign, _s0));
                    pA += 2;
                    pB += 4;
                }
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
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
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
                if (kk + 1 < max_kk0)
                {
                    v8i16 _pA = (v8i16)__msa_fill_w(*(const int*)pA);
                    v16i8 _pA0 = (v16i8)__msa_splati_h(_pA, 0);
                    v16i8 _pA1 = (v16i8)__msa_splati_h(_pA, 1);
                    v16i8 _pB = (v16i8)__msa_fill_w(*(const int*)pB);
                    v8i16 _s0 = __msa_dotp_s_h(_pA0, _pB);
                    v8i16 _s1 = __msa_dotp_s_h(_pA1, _pB);
                    v8i16 _s = (v8i16)__msa_ilvr_h(_s1, _s0);
                    _sum = __msa_addv_w(_sum, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 4;
                    pB += 4;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_h(*(const short*)pA);
                    v16i8 _pA0b = __msa_splati_b(_pA8, 0);
                    v16i8 _pA1b = __msa_splati_b(_pA8, 1);
                    v8i16 _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA0b, 0), _pA0b);
                    v8i16 _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA1b, 0), _pA1b);
                    v16i8 _pB8 = (v16i8)__msa_fill_h(*(const short*)pB);
                    v8i16 _pB = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    v8i16 _s0 = __msa_mulv_h(_pA0, _pB);
                    v8i16 _s1 = __msa_mulv_h(_pA1, _pB);
                    v8i16 _s = (v8i16)__msa_ilvr_h(_s1, _s0);
                    _sum = __msa_addv_w(_sum, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 2;
                    pB += 2;
                }
                v4f32 _descaleA = (v4f32)__msa_fill_d_ptr(pA_descales);
                v4f32 _descaleB = (v4f32)__msa_set_w(__msa_load_w(pB_descales), __msa_load_w(pB_descales), __msa_load_w(pB_descales + 1), __msa_load_w(pB_descales + 1));
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB);
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales += 2;
                pB_descales += 2;
            }
            __msa_st_w((v4i32)_fsum, outptr, 0);
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
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                    v16i8 _pB = (v16i8)__msa_fill_h(*(const short*)pB);
                    v8i16 _s = __msa_dotp_s_h(_pA, _pB);
                    _sum = __msa_addv_w(_sum, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 4;
                    pB += 2;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v16i8 _pA8 = (v16i8)__msa_fill_h(*(const short*)pA);
                    v8i16 _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pA8, 0), _pA8);
                    v8i16 _s = __msa_mulv_h(_pA, __msa_fill_h(pB[0]));
                    _sum = __msa_addv_w(_sum, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 2;
                    pB++;
                }
                v4f32 _descaleA = (v4f32)__msa_fill_d_ptr(pA_descales);
                v4f32 _scale = __msa_fmul_w(_descaleA, __msa_fill_w_f32(pB_descales[0]));
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales += 2;
                pB_descales++;
            }
            __msa_storel_d((v4i32)_fsum, outptr);
            outptr += 2;
            pB_panel += K;
            pB_descales_panel += block_count;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
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
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk + 3 < max_kk0; kk += 4)
                {
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
                }
#endif // NCNN_GNU_INLINE_ASM
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
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum00_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum01_i += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    sum10_i += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    sum11_i += pA[4] * pB[4] + pA[5] * pB[5] + pA[6] * pB[6] + pA[7] * pB[7];
                    pA += 8;
                    pB += 8;
                }
                if (kk + 1 < max_kk0)
                {
                    sum00_i += pA[0] * pB[0] + pA[1] * pB[1];
                    sum01_i += pA[2] * pB[0] + pA[3] * pB[1];
                    sum10_i += pA[0] * pB[2] + pA[1] * pB[3];
                    sum11_i += pA[2] * pB[2] + pA[3] * pB[3];
                    pA += 4;
                    pB += 4;
                    kk += 2;
                }
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
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk + 3 < max_kk0; kk += 4)
                {
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
                }
#endif // NCNN_GNU_INLINE_ASM
                int tmp[2];
                __mmi_pstw_s(tmp, _sum0);
                sum0_i += tmp[0] + tmp[1];
                __mmi_pstw_s(tmp, _sum1);
                sum1_i += tmp[0] + tmp[1];
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum1_i += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    pA += 8;
                    pB += 4;
                }
                if (kk + 1 < max_kk0)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1];
                    sum1_i += pA[2] * pB[0] + pA[3] * pB[1];
                    pA += 4;
                    pB += 2;
                    kk += 2;
                }
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
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            const signed char* pB0 = pB_panel + (size_t)4 * k;
            const signed char* pB1 = pB_panel + (size_t)4 * K + (size_t)4 * k;
            const float* pB_descales0 = pB_descales_panel + (size_t)4 * block_start;
            const float* pB_descales1 = pB_descales_panel + (size_t)4 * block_count + (size_t)4 * block_start;
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
                {
                    v4i32 _sum2 = __msa_fill_w(0);
                    v4i32 _sum3 = __msa_fill_w(0);
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __builtin_prefetch(pA + 32);
                        __builtin_prefetch(pB0 + 64);
                        __builtin_prefetch(pB1 + 64);
                        v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                        v16i8 _pB0 = __msa_ld_b(pB0, 0);
                        v16i8 _pB1 = __msa_ld_b(pB1, 0);
                        _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                        _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB1), _one);

                        _pA = (v16i8)__msa_fill_w(*(const int*)(pA + 4));
                        _pB0 = __msa_ld_b(pB0 + 16, 0);
                        _pB1 = __msa_ld_b(pB1 + 16, 0);
                        _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA, _pB0), _one);
                        _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA, _pB1), _one);
                        pA += 8;
                        pB0 += 32;
                        pB1 += 32;
                    }
                    _sum0 = __msa_addv_w(_sum0, _sum2);
                    _sum1 = __msa_addv_w(_sum1, _sum3);
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __builtin_prefetch(pA + 32);
                    __builtin_prefetch(pB0 + 64);
                    __builtin_prefetch(pB1 + 64);
                    v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                    v16i8 _pB0 = __msa_ld_b(pB0, 0);
                    v16i8 _pB1 = __msa_ld_b(pB1, 0);
                    _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                    _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB1), _one);
                    pA += 4;
                    pB0 += 16;
                    pB1 += 16;
                }
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = (v16i8)__msa_fill_h(*(const short*)pA);
                    v8i16 _s0 = __msa_dotp_s_h(_pA, (v16i8)__msa_fill_d_ptr(pB0));
                    v8i16 _s1 = __msa_dotp_s_h(_pA, (v16i8)__msa_fill_d_ptr(pB1));
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    pA += 2;
                    pB0 += 8;
                    pB1 += 8;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v8i16 _pA = __msa_fill_h(pA[0]);
                    v16i8 _pB08 = (v16i8)__msa_fill_w(*(const int*)pB0);
                    v16i8 _pB18 = (v16i8)__msa_fill_w(*(const int*)pB1);
                    v8i16 _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB08, 0), _pB08);
                    v8i16 _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB18, 0), _pB18);
                    v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                    v8i16 _s1 = __msa_mulv_h(_pA, _pB1);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                    _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                    pA++;
                    pB0 += 4;
                    pB1 += 4;
                }
                v4f32 _descaleB0 = (v4f32)__msa_ld_w(pB_descales0, 0);
                v4f32 _descaleB1 = (v4f32)__msa_ld_w(pB_descales1, 0);
                v4f32 _descaleA = __msa_fill_w_f32(pA_descales[0]);
                v4f32 _scale = __msa_fmul_w(_descaleA, _descaleB0);
                _fsum0 = __ncnn_msa_fmadd_w(_fsum0, (v4f32)__msa_ffint_s_w(_sum0), _scale);
                _scale = __msa_fmul_w(_descaleA, _descaleB1);
                _fsum1 = __ncnn_msa_fmadd_w(_fsum1, (v4f32)__msa_ffint_s_w(_sum1), _scale);
                pA_descales++;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }
            __msa_st_w((v4i32)_fsum0, outptr, 0);
            __msa_st_w((v4i32)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
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
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = (v16i8)__msa_fill_h(*(const short*)pA);
                    v8i16 _s = __msa_dotp_s_h(_pA, (v16i8)__msa_fill_d_ptr(pB));
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 2;
                    pB += 8;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v16i8 _pB8 = (v16i8)__msa_fill_w(*(const int*)pB);
                    v8i16 _pB = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    v8i16 _s = __msa_mulv_h(__msa_fill_h(pA[0]), _pB);
                    _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA++;
                    pB += 4;
                }
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
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
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
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = (v16i8)__msa_fill_h(*(const short*)pA);
                    v16i8 _pB = (v16i8)__msa_fill_w(*(const int*)pB);
                    v8i16 _s = __msa_dotp_s_h(_pA, _pB);
                    _sum = __msa_addv_w(_sum, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 2;
                    pB += 4;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v16i8 _pB8 = (v16i8)__msa_fill_h(*(const short*)pB);
                    v8i16 _pB = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_pB8, 0), _pB8);
                    v8i16 _s = __msa_mulv_h(__msa_fill_h(pA[0]), _pB);
                    _sum = __msa_addv_w(_sum, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA++;
                    pB += 2;
                }
                v4f32 _descaleB = (v4f32)__msa_fill_d_ptr(pB_descales);
                v4f32 _scale = __msa_fmul_w(_descaleB, __msa_fill_w_f32(pA_descales[0]));
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales++;
                pB_descales += 2;
            }
            __msa_storel_d((v4i32)_fsum, outptr);
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
                if (kk + 1 < max_kk0)
                {
                    v16i8 _pA = (v16i8)__msa_fill_h(*(const short*)pA);
                    v16i8 _pB = (v16i8)__msa_fill_h(*(const short*)pB);
                    v8i16 _s = __msa_dotp_s_h(_pA, _pB);
                    _sum = __msa_addv_w(_sum, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA += 2;
                    pB += 2;
                    kk += 2;
                }
                if (kk < max_kk0)
                {
                    v8i16 _s = __msa_fill_h(pA[0] * pB[0]);
                    _sum = __msa_addv_w(_sum, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s, 0), _s));
                    pA++;
                    pB++;
                }
                v4f32 _scale = __msa_fill_w_f32(pA_descales[0] * pB_descales[0]);
                _fsum = __ncnn_msa_fmadd_w(_fsum, (v4f32)__msa_ffint_s_w(_sum), _scale);
                pA_descales++;
                pB_descales++;
            }
            *outptr++ = _fsum[0];
            pB_panel += K;
            pB_descales_panel += block_count;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
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
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk + 3 < max_kk0; kk += 4)
                {
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
                }
#endif // NCNN_GNU_INLINE_ASM
                int tmp[2];
                __mmi_pstw_s(tmp, _sum0);
                sum0_i += tmp[0] + tmp[1];
                __mmi_pstw_s(tmp, _sum1);
                sum1_i += tmp[0] + tmp[1];
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum1_i += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    pA += 4;
                    pB += 8;
                }
                if (kk + 1 < max_kk0)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1];
                    sum1_i += pA[0] * pB[2] + pA[1] * pB[3];
                    pA += 2;
                    pB += 4;
                    kk += 2;
                }
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
#if NCNN_GNU_INLINE_ASM
                double _tmp0;
                double _tmp1;
                double _tmp2;
                double _tmp3;
                double _tmp4;
                double _tmp5;
                double _shift;
                const int shift_8 = 8;
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _pA = (int8x8_t)__mmi_pfillw_s(*(const int*)pA);
                    int8x8_t _pB = (int8x8_t)__mmi_pfillw_s(*(const int*)pB);
                    int16x4_t _pA0 = (int16x4_t)__mmi_punpcklbh_s(_pA, _zero);
                    int16x4_t _pB0 = (int16x4_t)__mmi_punpcklbh_s(_pB, _zero);
                    _pA0 = __mmi_psrah_s(__mmi_psllh_s(_pA0, 8), 8);
                    _pB0 = __mmi_psrah_s(__mmi_psllh_s(_pB0, 8), 8);
                    _sum0 = __mmi_paddw_s(_sum0, __mmi_pmaddhw(_pA0, _pB0));
                    pA += 4;
                    pB += 4;
                }
#endif // NCNN_GNU_INLINE_ASM
                int tmp[2];
                __mmi_pstw_s(tmp, _sum0);
                sum0_i += tmp[0] + tmp[1];
#endif // __mips_loongson_mmi
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    pA += 4;
                    pB += 4;
                }
                if (kk + 1 < max_kk0)
                {
                    sum0_i += pA[0] * pB[0] + pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                    kk += 2;
                }
                for (; kk < max_kk0; kk++)
                    sum0_i += *pA++ * *pB++;
                sum0 += sum0_i * pA_descales[0] * pB_descales[0];
                pA_descales++;
                pB_descales++;
            }

            *outptr++ = sum0;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* pp = topT;
    float* outptr = (float*)top_blob + (size_t)i * out_hstep + j;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = outptr;

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
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                    _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                    _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                    _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 4, 0), _beta));
                    _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 5, 0), _beta));
                    _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 6, 0), _beta));
                    _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 7, 0), _beta));
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

            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
            __msa_st_w((v4i32)_f4, p0 + out_hstep * 4, 0);
            __msa_st_w((v4i32)_f5, p0 + out_hstep * 5, 0);
            __msa_st_w((v4i32)_f6, p0 + out_hstep * 6, 0);
            __msa_st_w((v4i32)_f7, p0 + out_hstep * 7, 0);

            v4f32 _g0 = (v4f32)__msa_ld_w(pp + 32, 0);
            v4f32 _g4 = (v4f32)__msa_ld_w(pp + 36, 0);
            v4f32 _g1 = (v4f32)__msa_ld_w(pp + 40, 0);
            v4f32 _g5 = (v4f32)__msa_ld_w(pp + 44, 0);
            v4f32 _g2 = (v4f32)__msa_ld_w(pp + 48, 0);
            v4f32 _g6 = (v4f32)__msa_ld_w(pp + 52, 0);
            v4f32 _g3 = (v4f32)__msa_ld_w(pp + 56, 0);
            v4f32 _g7 = (v4f32)__msa_ld_w(pp + 60, 0);
            pp += 64;
            transpose4x4_ps(_g0, _g1, _g2, _g3);
            transpose4x4_ps(_g4, _g5, _g6, _g7);

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

            __msa_st_w((v4i32)_g0, p0 + 4, 0);
            __msa_st_w((v4i32)_g1, p0 + out_hstep + 4, 0);
            __msa_st_w((v4i32)_g2, p0 + out_hstep * 2 + 4, 0);
            __msa_st_w((v4i32)_g3, p0 + out_hstep * 3 + 4, 0);
            __msa_st_w((v4i32)_g4, p0 + out_hstep * 4 + 4, 0);
            __msa_st_w((v4i32)_g5, p0 + out_hstep * 5 + 4, 0);
            __msa_st_w((v4i32)_g6, p0 + out_hstep * 6 + 4, 0);
            __msa_st_w((v4i32)_g7, p0 + out_hstep * 7 + 4, 0);
            p0 += 8;
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
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4f32 _c = (v4f32)__msa_ld_w(pC, 0);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC + c_hstep, 0);
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC + c_hstep * 2, 0);
                    _f2 = __msa_fadd_w(_f2, __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC + c_hstep * 3, 0);
                    _f3 = __msa_fadd_w(_f3, __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC + c_hstep * 4, 0);
                    _f4 = __msa_fadd_w(_f4, __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC + c_hstep * 5, 0);
                    _f5 = __msa_fadd_w(_f5, __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC + c_hstep * 6, 0);
                    _f6 = __msa_fadd_w(_f6, __msa_fmul_w(_c, _beta));
                    _c = (v4f32)__msa_ld_w(pC + c_hstep * 7, 0);
                    _f7 = __msa_fadd_w(_f7, __msa_fmul_w(_c, _beta));
                    pC += 4;
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
            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, (p0 + out_hstep), 0);
            __msa_st_w((v4i32)_f2, (p0 + out_hstep * 2), 0);
            __msa_st_w((v4i32)_f3, (p0 + out_hstep * 3), 0);
            __msa_st_w((v4i32)_f4, (p0 + out_hstep * 4), 0);
            __msa_st_w((v4i32)_f5, (p0 + out_hstep * 5), 0);
            __msa_st_w((v4i32)_f6, (p0 + out_hstep * 6), 0);
            __msa_st_w((v4i32)_f7, (p0 + out_hstep * 7), 0);
            p0 += 4;
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
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                    v4f32 _c4 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                    v4f32 _c1 = (v4f32)__msa_set_w(__msa_load_w(pC + 1), __msa_load_w(pC + c_hstep + 1), __msa_load_w(pC + c_hstep * 2 + 1), __msa_load_w(pC + c_hstep * 3 + 1));
                    v4f32 _c5 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 1), __msa_load_w(pC + c_hstep * 5 + 1), __msa_load_w(pC + c_hstep * 6 + 1), __msa_load_w(pC + c_hstep * 7 + 1));
                    pC += 2;
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
            ((int*)p0)[0] = __msa_copy_s_w((v4i32)_f0, 0);
            ((int*)p0)[1] = __msa_copy_s_w((v4i32)_f1, 0);
            ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w((v4i32)_f0, 1);
            ((int*)(p0 + out_hstep))[1] = __msa_copy_s_w((v4i32)_f1, 1);
            ((int*)(p0 + out_hstep * 2))[0] = __msa_copy_s_w((v4i32)_f0, 2);
            ((int*)(p0 + out_hstep * 2))[1] = __msa_copy_s_w((v4i32)_f1, 2);
            ((int*)(p0 + out_hstep * 3))[0] = __msa_copy_s_w((v4i32)_f0, 3);
            ((int*)(p0 + out_hstep * 3))[1] = __msa_copy_s_w((v4i32)_f1, 3);
            ((int*)(p0 + out_hstep * 4))[0] = __msa_copy_s_w((v4i32)_f4, 0);
            ((int*)(p0 + out_hstep * 4))[1] = __msa_copy_s_w((v4i32)_f5, 0);
            ((int*)(p0 + out_hstep * 5))[0] = __msa_copy_s_w((v4i32)_f4, 1);
            ((int*)(p0 + out_hstep * 5))[1] = __msa_copy_s_w((v4i32)_f5, 1);
            ((int*)(p0 + out_hstep * 6))[0] = __msa_copy_s_w((v4i32)_f4, 2);
            ((int*)(p0 + out_hstep * 6))[1] = __msa_copy_s_w((v4i32)_f5, 2);
            ((int*)(p0 + out_hstep * 7))[0] = __msa_copy_s_w((v4i32)_f4, 3);
            ((int*)(p0 + out_hstep * 7))[1] = __msa_copy_s_w((v4i32)_f5, 3);
            p0 += 2;
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
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                    v4f32 _c4 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                    pC++;
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
            ((int*)p0)[0] = __msa_copy_s_w((v4i32)_f0, 0);
            ((int*)(p0 + out_hstep))[0] = __msa_copy_s_w((v4i32)_f0, 1);
            ((int*)(p0 + out_hstep * 2))[0] = __msa_copy_s_w((v4i32)_f0, 2);
            ((int*)(p0 + out_hstep * 3))[0] = __msa_copy_s_w((v4i32)_f0, 3);
            ((int*)(p0 + out_hstep * 4))[0] = __msa_copy_s_w((v4i32)_f4, 0);
            ((int*)(p0 + out_hstep * 5))[0] = __msa_copy_s_w((v4i32)_f4, 1);
            ((int*)(p0 + out_hstep * 6))[0] = __msa_copy_s_w((v4i32)_f4, 2);
            ((int*)(p0 + out_hstep * 7))[0] = __msa_copy_s_w((v4i32)_f4, 3);
            p0++;
        }
        outptr += out_hstep * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = outptr;

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
                    pC0 += 8;
                    v4f32 _c5 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                    pC1 += 8;
                    v4f32 _c6 = (v4f32)__msa_ld_w(pC2 + 4, 0);
                    pC2 += 8;
                    v4f32 _c7 = (v4f32)__msa_ld_w(pC3 + 4, 0);
                    pC3 += 8;
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

            transpose4x4_ps(_f0, _f1, _f2, _f3);
            transpose4x4_ps(_f4, _f5, _f6, _f7);
            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f4, p0 + 4, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f5, p0 + out_hstep + 4, 0);
            __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f6, p0 + out_hstep * 2 + 4, 0);
            __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
            __msa_st_w((v4i32)_f7, p0 + out_hstep * 3 + 4, 0);
            p0 += 8;
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
                    pC0 += 4;
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC1, 0);
                    pC1 += 4;
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC2, 0);
                    pC2 += 4;
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC3, 0);
                    pC3 += 4;
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

            transpose4x4_ps(_f0, _f1, _f2, _f3);
            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
            p0 += 4;
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
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC2), __msa_load_w(pC3));
                    v4f32 _c1 = (v4f32)__msa_set_w(__msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1), __msa_load_w(pC2 + 1), __msa_load_w(pC3 + 1));
                    pC0 += 2;
                    pC1 += 2;
                    pC2 += 2;
                    pC3 += 2;
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

            v4f32 _f2 = (v4f32)__msa_fill_w(0);
            v4f32 _f3 = (v4f32)__msa_fill_w(0);
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            *(int64_t*)p0 = __msa_copy_s_d((v2i64)_f0, 0);
            *(int64_t*)(p0 + out_hstep) = __msa_copy_s_d((v2i64)_f1, 0);
            *(int64_t*)(p0 + out_hstep * 2) = __msa_copy_s_d((v2i64)_f2, 0);
            *(int64_t*)(p0 + out_hstep * 3) = __msa_copy_s_d((v2i64)_f3, 0);
            p0 += 2;
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
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC2), __msa_load_w(pC3));
                    pC0++;
                    pC1++;
                    pC2++;
                    pC3++;
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
            p0[0] = _f0[0];
            p0[out_hstep] = _f0[1];
            p0[out_hstep * 2] = _f0[2];
            p0[out_hstep * 3] = _f0[3];
            p0++;
        }
        outptr += out_hstep * 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = outptr;
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

            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, p0 + 4, 0);
            __msa_st_w((v4i32)_f2, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f3, p0 + out_hstep + 4, 0);
            p0 += 8;
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

            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            p0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _f = (v4f32)__msa_ld_w(pp, 0);
            pp += 4;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __msa_fadd_w(_f, (v4f32)__msa_set_w(__msa_load_w(&c0), __msa_load_w(&c1), __msa_load_w(&c0), __msa_load_w(&c1)));
                if (broadcast_type_C == 3)
                {
                    v4f32 _c = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1));
                    pC0 += 2;
                    pC1 += 2;
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
                }
                if (broadcast_type_C == 4)
                {
                    float cc0 = pC[0];
                    float cc1 = pC[1];
                    pC += 2;
                    if (beta != 1.f)
                    {
                        cc0 *= beta;
                        cc1 *= beta;
                    }
                    _f = __msa_fadd_w(_f, (v4f32)__msa_set_w(__msa_load_w(&cc0), __msa_load_w(&cc0), __msa_load_w(&cc1), __msa_load_w(&cc1)));
                }
            }

            if (alpha != 1.f)
                _f = __msa_fmul_w(_f, __msa_fill_w_f32(alpha));

            v4i32 _f0 = __msa_pckev_w((v4i32)_f, (v4i32)_f);
            v4i32 _f1 = __msa_pckod_w((v4i32)_f, (v4i32)_f);
            __msa_storel_d(_f0, p0);
            __msa_storel_d(_f1, p0 + out_hstep);
            p0 += 2;
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
                    pC0 += 2;
                    float c11 = pC1[1];
                    pC1 += 2;
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
                    pC += 2;
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

            p0[0] = sum00;
            p0[out_hstep] = sum01;
            p0[1] = sum10;
            p0[out_hstep + 1] = sum11;
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
            p0[0] = sum0;
            p0[out_hstep] = sum1;
            p0++;
        }
        outptr += out_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        float* p0 = outptr;
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

            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, p0 + 4, 0);
            p0 += 8;
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

            __msa_st_w((v4i32)_f0, p0, 0);
            p0 += 4;
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
                    pC0 += 2;
                    v4f32 _c = (v4f32)_ci;
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
                }
            }

            if (alpha != 1.f)
                _f = __msa_fmul_w(_f, __msa_fill_w_f32(alpha));

            __msa_storel_d((v4i32)_f, p0);
            p0 += 2;
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
                    pC0 += 2;
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
                    pC0 += 2;
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
            p0[0] = sum0;
            p0[1] = sum1;
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
            p0[0] = sum0;
            p0++;
        }
        outptr += out_hstep;
    }
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* pp = topT;
    float* outptr = (float*)top_blob + (size_t)j * out_hstep + i;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = outptr;

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

            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f4, p0 + 4, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f5, p0 + out_hstep + 4, 0);
            __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f6, p0 + out_hstep * 2 + 4, 0);
            __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
            __msa_st_w((v4i32)_f7, p0 + out_hstep * 3 + 4, 0);
            __msa_st_w((v4i32)_g0, p0 + out_hstep * 4, 0);
            __msa_st_w((v4i32)_g4, p0 + out_hstep * 4 + 4, 0);
            __msa_st_w((v4i32)_g1, p0 + out_hstep * 5, 0);
            __msa_st_w((v4i32)_g5, p0 + out_hstep * 5 + 4, 0);
            __msa_st_w((v4i32)_g2, p0 + out_hstep * 6, 0);
            __msa_st_w((v4i32)_g6, p0 + out_hstep * 6 + 4, 0);
            __msa_st_w((v4i32)_g3, p0 + out_hstep * 7, 0);
            __msa_st_w((v4i32)_g7, p0 + out_hstep * 7 + 4, 0);
            p0 += out_hstep * 8;
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
                    v4f32 _cl0 = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                    v4f32 _ch0 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                    v4f32 _cl1 = (v4f32)__msa_set_w(__msa_load_w(pC + 1), __msa_load_w(pC + c_hstep + 1), __msa_load_w(pC + c_hstep * 2 + 1), __msa_load_w(pC + c_hstep * 3 + 1));
                    v4f32 _ch1 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 1), __msa_load_w(pC + c_hstep * 5 + 1), __msa_load_w(pC + c_hstep * 6 + 1), __msa_load_w(pC + c_hstep * 7 + 1));
                    v4f32 _cl2 = (v4f32)__msa_set_w(__msa_load_w(pC + 2), __msa_load_w(pC + c_hstep + 2), __msa_load_w(pC + c_hstep * 2 + 2), __msa_load_w(pC + c_hstep * 3 + 2));
                    v4f32 _ch2 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 2), __msa_load_w(pC + c_hstep * 5 + 2), __msa_load_w(pC + c_hstep * 6 + 2), __msa_load_w(pC + c_hstep * 7 + 2));
                    v4f32 _cl3 = (v4f32)__msa_set_w(__msa_load_w(pC + 3), __msa_load_w(pC + c_hstep + 3), __msa_load_w(pC + c_hstep * 2 + 3), __msa_load_w(pC + c_hstep * 3 + 3));
                    v4f32 _ch3 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 3), __msa_load_w(pC + c_hstep * 5 + 3), __msa_load_w(pC + c_hstep * 6 + 3), __msa_load_w(pC + c_hstep * 7 + 3));
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
                    pC += 4;
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
            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f4, p0 + 4, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f5, p0 + out_hstep + 4, 0);
            __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f6, p0 + out_hstep * 2 + 4, 0);
            __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
            __msa_st_w((v4i32)_f7, p0 + out_hstep * 3 + 4, 0);
            p0 += out_hstep * 4;
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
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                    v4f32 _c4 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                    v4f32 _c1 = (v4f32)__msa_set_w(__msa_load_w(pC + 1), __msa_load_w(pC + c_hstep + 1), __msa_load_w(pC + c_hstep * 2 + 1), __msa_load_w(pC + c_hstep * 3 + 1));
                    v4f32 _c5 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4 + 1), __msa_load_w(pC + c_hstep * 5 + 1), __msa_load_w(pC + c_hstep * 6 + 1), __msa_load_w(pC + c_hstep * 7 + 1));
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
                    pC += 2;
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
            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f4, p0 + 4, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f5, p0 + out_hstep + 4, 0);
            p0 += out_hstep * 2;
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
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC), __msa_load_w(pC + c_hstep), __msa_load_w(pC + c_hstep * 2), __msa_load_w(pC + c_hstep * 3));
                    v4f32 _c4 = (v4f32)__msa_set_w(__msa_load_w(pC + c_hstep * 4), __msa_load_w(pC + c_hstep * 5), __msa_load_w(pC + c_hstep * 6), __msa_load_w(pC + c_hstep * 7));
                    if (beta != 1.f)
                    {
                        v4f32 _beta = __msa_fill_w_f32(beta);
                        _c0 = __msa_fmul_w(_c0, _beta);
                        _c4 = __msa_fmul_w(_c4, _beta);
                    }
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c4);
                    pC++;
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
            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f4, p0 + 4, 0);
            p0 += out_hstep;
        }
        outptr += 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = outptr;

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
                    pC0 += 8;
                    v4f32 _c5 = (v4f32)__msa_ld_w(pC1 + 4, 0);
                    pC1 += 8;
                    v4f32 _c6 = (v4f32)__msa_ld_w(pC2 + 4, 0);
                    pC2 += 8;
                    v4f32 _c7 = (v4f32)__msa_ld_w(pC3 + 4, 0);
                    pC3 += 8;
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

            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
            __msa_st_w((v4i32)_f4, p0 + out_hstep * 4, 0);
            __msa_st_w((v4i32)_f5, p0 + out_hstep * 5, 0);
            __msa_st_w((v4i32)_f6, p0 + out_hstep * 6, 0);
            __msa_st_w((v4i32)_f7, p0 + out_hstep * 7, 0);
            p0 += out_hstep * 8;
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
                    pC0 += 4;
                    v4f32 _c1 = (v4f32)__msa_ld_w(pC1, 0);
                    pC1 += 4;
                    v4f32 _c2 = (v4f32)__msa_ld_w(pC2, 0);
                    pC2 += 4;
                    v4f32 _c3 = (v4f32)__msa_ld_w(pC3, 0);
                    pC3 += 4;
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

            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
            __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
            p0 += out_hstep * 4;
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
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC2), __msa_load_w(pC3));
                    v4f32 _c1 = (v4f32)__msa_set_w(__msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1), __msa_load_w(pC2 + 1), __msa_load_w(pC3 + 1));
                    pC0 += 2;
                    pC1 += 2;
                    pC2 += 2;
                    pC3 += 2;
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

            __msa_st_w((v4i32)_f0, p0, 0);
            __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
            p0 += out_hstep * 2;
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
                    v4f32 _c0 = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC2), __msa_load_w(pC3));
                    pC0++;
                    pC1++;
                    pC2++;
                    pC3++;
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
            __msa_st_w((v4i32)_f0, p0, 0);
            p0 += out_hstep;
        }
        outptr += 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = outptr;
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

            __msa_storel_d((v4i32)_f0, p0);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f0, (v16i8)_f0, 8), p0 + out_hstep);
            __msa_storel_d((v4i32)_f1, p0 + out_hstep * 2);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f1, (v16i8)_f1, 8), p0 + out_hstep * 3);
            __msa_storel_d((v4i32)_f2, p0 + out_hstep * 4);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f2, (v16i8)_f2, 8), p0 + out_hstep * 5);
            __msa_storel_d((v4i32)_f3, p0 + out_hstep * 6);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f3, (v16i8)_f3, 8), p0 + out_hstep * 7);
            p0 += out_hstep * 8;
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

            __msa_storel_d((v4i32)_f0, p0);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f0, (v16i8)_f0, 8), p0 + out_hstep);
            __msa_storel_d((v4i32)_f1, p0 + out_hstep * 2);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f1, (v16i8)_f1, 8), p0 + out_hstep * 3);
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _f = (v4f32)__msa_ld_w(pp, 0);
            pp += 4;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __msa_fadd_w(_f, (v4f32)__msa_set_w(__msa_load_w(&c0), __msa_load_w(&c1), __msa_load_w(&c0), __msa_load_w(&c1)));
                if (broadcast_type_C == 3)
                {
                    v4f32 _c = (v4f32)__msa_set_w(__msa_load_w(pC0), __msa_load_w(pC1), __msa_load_w(pC0 + 1), __msa_load_w(pC1 + 1));
                    pC0 += 2;
                    pC1 += 2;
                    if (beta != 1.f)
                        _c = __msa_fmul_w(_c, __msa_fill_w_f32(beta));
                    _f = __msa_fadd_w(_f, _c);
                }
                if (broadcast_type_C == 4)
                {
                    float cc0 = pC[0];
                    float cc1 = pC[1];
                    pC += 2;
                    if (beta != 1.f)
                    {
                        cc0 *= beta;
                        cc1 *= beta;
                    }
                    _f = __msa_fadd_w(_f, (v4f32)__msa_set_w(__msa_load_w(&cc0), __msa_load_w(&cc0), __msa_load_w(&cc1), __msa_load_w(&cc1)));
                }
            }

            if (alpha != 1.f)
                _f = __msa_fmul_w(_f, __msa_fill_w_f32(alpha));

            __msa_storel_d((v4i32)_f, p0);
            __msa_storel_d((v4i32)__msa_sldi_b((v16i8)_f, (v16i8)_f, 8), p0 + out_hstep);
            p0 += out_hstep * 2;
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
                    pC0 += 2;
                    float c11 = pC1[1];
                    pC1 += 2;
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
                    pC += 2;
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
            p0[0] = sum00;
            p0[1] = sum01;
            p0[out_hstep] = sum10;
            p0[out_hstep + 1] = sum11;
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
            p0[0] = sum0;
            p0[1] = sum1;
            p0 += out_hstep;
        }
        outptr += 2;
    }
    for (; ii < max_ii; ii++)
    {
        float* p0 = outptr;
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
            if (out_hstep == 1)
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + 4, 0);
            }
            else
            {
                *(int*)p0 = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(p0 + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
                *(int*)(p0 + out_hstep * 2) = __msa_copy_s_w((v4i32)_f0, 2);
                *(int*)(p0 + out_hstep * 3) = __msa_copy_s_w((v4i32)_f0, 3);
                *(int*)(p0 + out_hstep * 4) = __msa_copy_s_w((v4i32)_f1, 0);
                *(int*)(p0 + out_hstep * 5) = __msa_copy_s_w((v4i32)_f1, 1);
                *(int*)(p0 + out_hstep * 6) = __msa_copy_s_w((v4i32)_f1, 2);
                *(int*)(p0 + out_hstep * 7) = __msa_copy_s_w((v4i32)_f1, 3);
            }
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
            if (out_hstep == 1)
            {
                __msa_st_w((v4i32)_f0, p0, 0);
            }
            else
            {
                *(int*)p0 = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(p0 + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
                *(int*)(p0 + out_hstep * 2) = __msa_copy_s_w((v4i32)_f0, 2);
                *(int*)(p0 + out_hstep * 3) = __msa_copy_s_w((v4i32)_f0, 3);
            }
            p0 += out_hstep * 4;
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
                    pC0 += 2;
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
                *(int64_t*)p0 = __msa_copy_s_d((v2i64)_f0, 0);
            }
            else
            {
                *(int*)p0 = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(p0 + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
            }
            p0 += out_hstep * 2;
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
                    pC0 += 2;
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
            p0[0] = sum0;
            p0[out_hstep] = sum1;
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
            p0[0] = sum0;
            p0 += out_hstep;
        }
        outptr++;
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
        TILE_N = std::max(8, tile_size / 8 * 8);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(8, std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8));
#else
        TILE_N = std::max(2, tile_size / 2 * 2);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(2, std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2));
#endif
    }
    else
    {
#if __mips_msa
        TILE_N = 8;
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
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
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
