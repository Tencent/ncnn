// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_B_tile_int8s(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int j, int max_jj, int K)
{
    int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 16;

        for (int jj0 = 0; jj0 < 16; jj0 += 4)
        {
            const unsigned char* p0 = B.row<const unsigned char>(j + jj + jj0);
            const unsigned char* p1 = B.row<const unsigned char>(j + jj + jj0 + 1);
            const unsigned char* p2 = B.row<const unsigned char>(j + jj + jj0 + 2);
            const unsigned char* p3 = B.row<const unsigned char>(j + jj + jj0 + 3);
            unsigned char* pp = pp0 + jj0;

            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[bytes] = p0[1];
                pp[bytes + 1] = p1[1];
                pp[bytes + 2] = p2[1];
                pp[bytes + 3] = p3[1];
                pp[bytes * 2] = p0[2];
                pp[bytes * 2 + 1] = p1[2];
                pp[bytes * 2 + 2] = p2[2];
                pp[bytes * 2 + 3] = p3[2];
                pp[bytes * 3] = p0[3];
                pp[bytes * 3 + 1] = p1[3];
                pp[bytes * 3 + 2] = p2[3];
                pp[bytes * 3 + 3] = p3[3];

                pp += bytes * 4;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk < K; kk++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp += bytes;
            }
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 16; jj0++)
        {
            const float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 16 + jj0] = scales[g];
        }
    }
#endif // __loongarch_asx
    for (; jj + 7 < max_jj; jj += 8)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 8;

        for (int jj0 = 0; jj0 < 8; jj0 += 4)
        {
            const unsigned char* p0 = B.row<const unsigned char>(j + jj + jj0);
            const unsigned char* p1 = B.row<const unsigned char>(j + jj + jj0 + 1);
            const unsigned char* p2 = B.row<const unsigned char>(j + jj + jj0 + 2);
            const unsigned char* p3 = B.row<const unsigned char>(j + jj + jj0 + 3);
            unsigned char* pp = pp0 + jj0;

            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[bytes] = p0[1];
                pp[bytes + 1] = p1[1];
                pp[bytes + 2] = p2[1];
                pp[bytes + 3] = p3[1];
                pp[bytes * 2] = p0[2];
                pp[bytes * 2 + 1] = p1[2];
                pp[bytes * 2 + 2] = p2[2];
                pp[bytes * 2 + 3] = p3[2];
                pp[bytes * 3] = p0[3];
                pp[bytes * 3 + 1] = p1[3];
                pp[bytes * 3 + 2] = p2[3];
                pp[bytes * 3 + 3] = p3[3];

                pp += bytes * 4;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk < K; kk++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp += bytes;
            }
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 8; jj0++)
        {
            const float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 8 + jj0] = scales[g];
        }
    }
#endif // __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
        unsigned char* pp = BT.row<unsigned char>(j + jj);
        const int bytes = 4;

        int kk = 0;
        for (; kk + 3 < K; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[bytes] = p0[1];
            pp[bytes + 1] = p1[1];
            pp[bytes + 2] = p2[1];
            pp[bytes + 3] = p3[1];
            pp[bytes * 2] = p0[2];
            pp[bytes * 2 + 1] = p1[2];
            pp[bytes * 2 + 2] = p2[2];
            pp[bytes * 2 + 3] = p3[2];
            pp[bytes * 3] = p0[3];
            pp[bytes * 3 + 1] = p1[3];
            pp[bytes * 3 + 2] = p2[3];
            pp[bytes * 3 + 3] = p3[3];

            pp += bytes * 4;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < K; kk++)
        {
            pp[0] = *p0++;
            pp[1] = *p1++;
            pp[2] = *p2++;
            pp[3] = *p3++;
            pp += bytes;
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 4; jj0++)
        {
            const float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 4 + jj0] = scales[g];
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        unsigned char* pp = BT.row<unsigned char>(j + jj);

        int kk = 0;
        for (; kk + 1 < K; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p0[1];
            pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
        for (; kk < K; kk++)
        {
            pp[0] = *p0++;
            pp[1] = *p1++;
            pp += 2;
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 2; jj0++)
        {
            const float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 2 + jj0] = scales[g];
        }
    }
    for (; jj < max_jj; jj++)
    {
        if (BT.data != B.data)
            memcpy(BT.row<unsigned char>(j + jj), B.row<const unsigned char>(j + jj), B.w);

        float* descales = BT_descales.row(j + jj);
        const float* scales = B_scales.row(j + jj);
        for (int g = 0; g < BT_descales.w; g++)
            descales[g] = scales[g];
    }
}

static void dequantize_B_tile_int8s(const Mat& B, const Mat& B_scales, Mat& BT_tile, int j, int max_jj, int k, int max_kk, int block_size)
{
    const int k_end = k + max_kk;
    const int g0 = k / block_size;
    const int g_end = (k_end - 1) / block_size + 1;

    float* pp = BT_tile;
    int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 16;

            const unsigned char* p = pB + (size_t)k0 * 16;
            const __m128 _descale0 = (__m128)__lsx_vld(descales, 0);
            const __m128 _descale1 = (__m128)__lsx_vld(descales + 4, 0);
            const __m128 _descale2 = (__m128)__lsx_vld(descales + 8, 0);
            const __m128 _descale3 = (__m128)__lsx_vld(descales + 12, 0);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                __m128i _v0 = __lsx_vldrepl_w(p, 0);
                __m128i _v1 = __lsx_vldrepl_w(p + 4, 0);
                __m128i _v2 = __lsx_vldrepl_w(p + 8, 0);
                __m128i _v3 = __lsx_vldrepl_w(p + 12, 0);
                _v0 = __lsx_vsllwil_w_h(__lsx_vsllwil_h_b(_v0, 0), 0);
                _v1 = __lsx_vsllwil_w_h(__lsx_vsllwil_h_b(_v1, 0), 0);
                _v2 = __lsx_vsllwil_w_h(__lsx_vsllwil_h_b(_v2, 0), 0);
                _v3 = __lsx_vsllwil_w_h(__lsx_vsllwil_h_b(_v3, 0), 0);
                __lsx_vst(__lsx_vfmul_s(__lsx_vffint_s_w(_v0), _descale0), pp, 0);
                __lsx_vst(__lsx_vfmul_s(__lsx_vffint_s_w(_v1), _descale1), pp + 4, 0);
                __lsx_vst(__lsx_vfmul_s(__lsx_vffint_s_w(_v2), _descale2), pp + 8, 0);
                __lsx_vst(__lsx_vfmul_s(__lsx_vffint_s_w(_v3), _descale3), pp + 12, 0);
                p += 16;
                pp += 16;
            }
        }
    }
#endif // __loongarch_asx
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 8;

            const unsigned char* p = pB + (size_t)k0 * 8;
            const __m128 _descale0 = (__m128)__lsx_vld(descales, 0);
            const __m128 _descale1 = (__m128)__lsx_vld(descales + 4, 0);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                __m128i _v0 = __lsx_vldrepl_w(p, 0);
                __m128i _v1 = __lsx_vldrepl_w(p + 4, 0);
                _v0 = __lsx_vsllwil_w_h(__lsx_vsllwil_h_b(_v0, 0), 0);
                _v1 = __lsx_vsllwil_w_h(__lsx_vsllwil_h_b(_v1, 0), 0);
                __lsx_vst(__lsx_vfmul_s(__lsx_vffint_s_w(_v0), _descale0), pp, 0);
                __lsx_vst(__lsx_vfmul_s(__lsx_vffint_s_w(_v1), _descale1), pp + 4, 0);
                p += 8;
                pp += 8;
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 4;

#if __loongarch_sx
            const unsigned char* p = pB + (size_t)k0 * 4;
            const __m128 _descale = (__m128)__lsx_vld(descales, 0);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                __m128i _v = __lsx_vldrepl_w(p, 0);
                _v = __lsx_vsllwil_w_h(__lsx_vsllwil_h_b(_v, 0), 0);
                __lsx_vst(__lsx_vfmul_s(__lsx_vffint_s_w(_v), _descale), pp, 0);
                p += 4;
                pp += 4;
            }
#else
            const signed char* p = (const signed char*)pB + (size_t)k0 * 4;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                pp[0] = p[0] * descales[0];
                pp[1] = p[1] * descales[1];
                pp[2] = p[2] * descales[2];
                pp[3] = p[3] * descales[3];
                p += 4;
                pp += 4;
            }
#endif // __loongarch_sx
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 2;

            const signed char* p = (const signed char*)pB + (size_t)k0 * 2;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                pp[0] = p[0] * descales[0];
                pp[1] = p[1] * descales[1];
                p += 2;
                pp += 2;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float descale = scales[g];

            const signed char* p = (const signed char*)pB + k0;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                pp[0] = p[0] * descale;
                p += 1;
                pp += 1;
            }
        }
    }
}
