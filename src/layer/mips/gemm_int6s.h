// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static inline int unpack_int6(const unsigned char* ptr, int k)
{
    const size_t offset = (size_t)(k / 4) * 3 + k % 4 * 6 / 8;
    const int shift = k % 4 * 6 % 8;
    unsigned int v = ptr[offset];
    if (shift > 2)
        v |= (unsigned int)ptr[offset + 1] << 8;
    const int w = (v >> shift) & 63;
    return (w ^ 32) - 32;
}

static inline void pack_int6x4(unsigned char* pp, unsigned int v0, unsigned int v1, unsigned int v2, unsigned int v3)
{
    const unsigned int v = (v0 & 63) | ((v1 & 63) << 6) | ((v2 & 63) << 12) | ((v3 & 63) << 18);
    pp[0] = (unsigned char)v;
    pp[1] = (unsigned char)(v >> 8);
    pp[2] = (unsigned char)(v >> 16);
}

static inline uint32_t unpack_int6x4(const unsigned char* p)
{
    const uint32_t v = p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    const uint32_t q = (v & 0x3f) | ((v & 0xfc0) << 2) | ((v & 0x3f000) << 4) | ((v & 0xfc0000) << 6);
    return q | ((q & 0x20202020) * 6);
}

static void pack_B_tile_int6s(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int j, int max_jj, int K)
{
    // Keep the reference FP32 division with fast math enabled.
    int jj = 0;
#if __mips_msa
    for (; jj + 7 < max_jj; jj += 8)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 6;

        for (int jj0 = 0; jj0 < 8; jj0 += 4)
        {
            const unsigned char* p0 = B.row<const unsigned char>(j + jj + jj0);
            const unsigned char* p1 = B.row<const unsigned char>(j + jj + jj0 + 1);
            const unsigned char* p2 = B.row<const unsigned char>(j + jj + jj0 + 2);
            const unsigned char* p3 = B.row<const unsigned char>(j + jj + jj0 + 3);
            unsigned char* pp = pp0 + jj0 / 4 * 3;

            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                const unsigned int v0 = p0[0] | (p0[1] << 8) | (p0[2] << 16);
                const unsigned int v1 = p1[0] | (p1[1] << 8) | (p1[2] << 16);
                const unsigned int v2 = p2[0] | (p2[1] << 8) | (p2[2] << 16);
                const unsigned int v3 = p3[0] | (p3[1] << 8) | (p3[2] << 16);

                pack_int6x4(pp, v0, v1, v2, v3);
                pack_int6x4(pp + bytes, v0 >> 6, v1 >> 6, v2 >> 6, v3 >> 6);
                pack_int6x4(pp + bytes * 2, v0 >> 12, v1 >> 12, v2 >> 12, v3 >> 12);
                pack_int6x4(pp + bytes * 3, v0 >> 18, v1 >> 18, v2 >> 18, v3 >> 18);

                pp += bytes * 4;
                p0 += 3;
                p1 += 3;
                p2 += 3;
                p3 += 3;
            }
            for (int k = 0; kk < K; kk++, k++)
            {
                pack_int6x4(pp, unpack_int6(p0, k),
                            unpack_int6(p1, k), unpack_int6(p2, k),
                            unpack_int6(p3, k));
                pp += bytes;
            }
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 8; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 8 + jj0] = 1.f / scales[g];
        }
    }
#endif // __mips_msa
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
        unsigned char* pp = BT.row<unsigned char>(j + jj);
        const int bytes = 3;

        int kk = 0;
        for (; kk + 3 < K; kk += 4)
        {
            const unsigned int v0 = p0[0] | (p0[1] << 8) | (p0[2] << 16);
            const unsigned int v1 = p1[0] | (p1[1] << 8) | (p1[2] << 16);
            const unsigned int v2 = p2[0] | (p2[1] << 8) | (p2[2] << 16);
            const unsigned int v3 = p3[0] | (p3[1] << 8) | (p3[2] << 16);

            pack_int6x4(pp, v0, v1, v2, v3);
            pack_int6x4(pp + bytes, v0 >> 6, v1 >> 6, v2 >> 6, v3 >> 6);
            pack_int6x4(pp + bytes * 2, v0 >> 12, v1 >> 12, v2 >> 12, v3 >> 12);
            pack_int6x4(pp + bytes * 3, v0 >> 18, v1 >> 18, v2 >> 18, v3 >> 18);

            pp += bytes * 4;
            p0 += 3;
            p1 += 3;
            p2 += 3;
            p3 += 3;
        }
        for (int k = 0; kk < K; kk++, k++)
        {
            pack_int6x4(pp, unpack_int6(p0, k),
                        unpack_int6(p1, k), unpack_int6(p2, k),
                        unpack_int6(p3, k));
            pp += bytes;
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 4; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 4 + jj0] = 1.f / scales[g];
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        unsigned char* pp = BT.row<unsigned char>(j + jj);

        // Two columns of INT6 occupy twelve bits per K.
        int kk = 0;
        for (; kk + 1 < K; kk += 2)
        {
            pack_int6x4(pp, unpack_int6(p0, kk),
                        unpack_int6(p1, kk), unpack_int6(p0, kk + 1),
                        unpack_int6(p1, kk + 1));
            pp += 3;
        }
        if (kk < K)
        {
            const unsigned int v = (unpack_int6(p0, kk) & 63)
                                   | ((unpack_int6(p1, kk) & 63) << 6);
            pp[0] = (unsigned char)v;
            pp[1] = (unsigned char)(v >> 8);
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 2; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 2 + jj0] = 1.f / scales[g];
        }
    }
    for (; jj < max_jj; jj++)
    {
        if (BT.data != B.data)
            memcpy(BT.row<unsigned char>(j + jj), B.row<const unsigned char>(j + jj), B.w);

        float* descales = BT_descales.row(j + jj);
        const volatile float* scales = B_scales.row(j + jj);
        for (int g = 0; g < BT_descales.w; g++)
            descales[g] = 1.f / scales[g];
    }
}

static void dequantize_B_tile_int6s(const Mat& B, const Mat& B_scales, Mat& BT_tile, int j, int max_jj, int k, int max_kk, int block_size)
{
    const int k_end = k + max_kk;
    const int g0 = k / block_size;
    const int g_end = (k_end - 1) / block_size + 1;

    float* pp = BT_tile;
    int jj = 0;
#if __mips_msa
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

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            const unsigned char* p = pB + (size_t)k0 * 6;
            const v4f32 _descale0 = (v4f32)__msa_ld_w(descales, 0);
            const v4f32 _descale1 = (v4f32)__msa_ld_w(descales + 4, 0);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const uint32_t v0 = unpack_int6x4(p);
                const uint32_t v1 = unpack_int6x4(p + 3);
                const v16i8 _v0 = (v16i8)__msa_fill_w(v0);
                const v8i16 _v16_0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_v0, 0), _v0);
                const v4i32 _v32_0 = (v4i32)__msa_ilvr_h(__msa_clti_s_h(_v16_0, 0), _v16_0);
                const v16i8 _v1 = (v16i8)__msa_fill_w(v1);
                const v8i16 _v16_1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_v1, 0), _v1);
                const v4i32 _v32_1 = (v4i32)__msa_ilvr_h(__msa_clti_s_h(_v16_1, 0), _v16_1);
                __msa_st_w((v4i32)__msa_fmul_w(__msa_ffint_s_w(_v32_0), _descale0), pp, 0);
                __msa_st_w((v4i32)__msa_fmul_w(__msa_ffint_s_w(_v32_1), _descale1), pp + 4, 0);
                p += 6;
                pp += 8;
            }
#else
            const unsigned char* p = pB + (size_t)k0 * 6;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                for (int i = 0; i < 8; i += 4)
                {
                    const uint32_t v = unpack_int6x4(p + i / 4 * 3);
                    pp[i] = (signed char)v * descales[i];
                    pp[i + 1] = (signed char)(v >> 8) * descales[i + 1];
                    pp[i + 2] = (signed char)(v >> 16) * descales[i + 2];
                    pp[i + 3] = (signed char)(v >> 24) * descales[i + 3];
                }
                p += 6;
                pp += 8;
            }
#endif // __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        }
    }
#endif // __mips_msa
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

#if __mips_msa && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            const unsigned char* p = pB + (size_t)k0 * 3;
            const v4f32 _descale = (v4f32)__msa_ld_w(descales, 0);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const uint32_t v = unpack_int6x4(p);
                const v16i8 _v = (v16i8)__msa_fill_w(v);
                const v8i16 _v16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_v, 0), _v);
                const v4i32 _v32 = (v4i32)__msa_ilvr_h(__msa_clti_s_h(_v16, 0), _v16);
                __msa_st_w((v4i32)__msa_fmul_w(__msa_ffint_s_w(_v32), _descale), pp, 0);
                p += 3;
                pp += 4;
            }
#else
            const unsigned char* p = pB + (size_t)k0 * 3;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                for (int i = 0; i < 4; i += 4)
                {
                    const uint32_t v = unpack_int6x4(p + i / 4 * 3);
                    pp[i] = (signed char)v * descales[i];
                    pp[i + 1] = (signed char)(v >> 8) * descales[i + 1];
                    pp[i + 2] = (signed char)(v >> 16) * descales[i + 2];
                    pp[i + 3] = (signed char)(v >> 24) * descales[i + 3];
                }
                p += 3;
                pp += 4;
            }
#endif // __mips_msa && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
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

            for (int kk = 0; kk < max_kk0; kk++)
            {
                const int index = k0 + kk;
                const unsigned char* p = pB + (size_t)(index / 2) * 3;
                const int v = index % 2 ? (p[1] >> 4) | (p[2] << 4) : p[0] | (p[1] << 8);
                pp[0] = (((v & 63) ^ 32) - 32) * descales[0];
                pp[1] = ((((v >> 6) & 63) ^ 32) - 32) * descales[1];
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

            for (int kk = 0; kk < max_kk0; kk++)
            {
                *pp++ = unpack_int6(pB, k0 + kk) * descale;
            }
        }
    }
}
