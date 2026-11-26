// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static inline uint32_t unpack_int4x4(const unsigned char* p)
{
    const uint32_t v = p[0] | ((uint32_t)p[1] << 8);
    const uint32_t q = (v & 0xf) | ((v & 0xf0) << 4) | ((v & 0xf00) << 8) | ((v & 0xf000) << 12);
    return q | ((q & 0x08080808) * 30);
}

static void pack_B_tile_int4s(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int j, int max_jj, int K)
{
    // Keep the reference FP32 division with fast math enabled.
    int jj = 0;
#if __riscv_vector
    for (; jj + 15 < max_jj; jj += 16)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 8;

        for (int jj0 = 0; jj0 < 16; jj0 += 4)
        {
            const unsigned char* p0 = B.row<const unsigned char>(j + jj + jj0);
            const unsigned char* p1 = B.row<const unsigned char>(j + jj + jj0 + 1);
            const unsigned char* p2 = B.row<const unsigned char>(j + jj + jj0 + 2);
            const unsigned char* p3 = B.row<const unsigned char>(j + jj + jj0 + 3);
            unsigned char* pp = pp0 + jj0 / 2;

            int kk = 0;
            for (; kk + 1 < K; kk += 2)
            {
                const unsigned int v0 = *p0++;
                const unsigned int v1 = *p1++;
                const unsigned int v2 = *p2++;
                const unsigned int v3 = *p3++;
                pp[0] = (v0 & 15) | (v1 << 4);
                pp[1] = (v2 & 15) | (v3 << 4);
                pp[bytes] = (v0 >> 4) | (v1 & 0xf0);
                pp[bytes + 1] = (v2 >> 4) | (v3 & 0xf0);
                pp += bytes * 2;
            }
            if (kk < K)
            {
                pp[0] = (*p0 & 15) | (*p1 << 4);
                pp[1] = (*p2 & 15) | (*p3 << 4);
            }
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 16; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 16 + jj0] = 1.f / scales[g];
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 4;

        for (int jj0 = 0; jj0 < 8; jj0 += 4)
        {
            const unsigned char* p0 = B.row<const unsigned char>(j + jj + jj0);
            const unsigned char* p1 = B.row<const unsigned char>(j + jj + jj0 + 1);
            const unsigned char* p2 = B.row<const unsigned char>(j + jj + jj0 + 2);
            const unsigned char* p3 = B.row<const unsigned char>(j + jj + jj0 + 3);
            unsigned char* pp = pp0 + jj0 / 2;

            int kk = 0;
            for (; kk + 1 < K; kk += 2)
            {
                const unsigned int v0 = *p0++;
                const unsigned int v1 = *p1++;
                const unsigned int v2 = *p2++;
                const unsigned int v3 = *p3++;
                pp[0] = (v0 & 15) | (v1 << 4);
                pp[1] = (v2 & 15) | (v3 << 4);
                pp[bytes] = (v0 >> 4) | (v1 & 0xf0);
                pp[bytes + 1] = (v2 >> 4) | (v3 & 0xf0);
                pp += bytes * 2;
            }
            if (kk < K)
            {
                pp[0] = (*p0 & 15) | (*p1 << 4);
                pp[1] = (*p2 & 15) | (*p3 << 4);
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
        unsigned char* pp = BT.row<unsigned char>(j + jj);
        const int bytes = 2;

        int kk = 0;
        for (; kk + 1 < K; kk += 2)
        {
            const unsigned int v0 = *p0++;
            const unsigned int v1 = *p1++;
            const unsigned int v2 = *p2++;
            const unsigned int v3 = *p3++;
            pp[0] = (v0 & 15) | (v1 << 4);
            pp[1] = (v2 & 15) | (v3 << 4);
            pp[bytes] = (v0 >> 4) | (v1 & 0xf0);
            pp[bytes + 1] = (v2 >> 4) | (v3 & 0xf0);
            pp += bytes * 2;
        }
        if (kk < K)
        {
            pp[0] = (*p0 & 15) | (*p1 << 4);
            pp[1] = (*p2 & 15) | (*p3 << 4);
        }

        float* descales = BT_descales.row(j + jj);
        for (int jj0 = 0; jj0 < 4; jj0++)
        {
            const volatile float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 4 + jj0] = 1.f / scales[g];
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        unsigned char* pp = BT.row<unsigned char>(j + jj);

        int kk = 0;
        for (; kk + 1 < K; kk += 2)
        {
            const unsigned int v0 = *p0++;
            const unsigned int v1 = *p1++;
            pp[0] = (v0 & 15) | (v1 << 4);
            pp[1] = (v0 >> 4) | (v1 & 0xf0);
            pp += 2;
        }
        if (kk < K)
            pp[0] = (*p0 & 15) | (*p1 << 4);

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

static void dequantize_B_tile_int4s(const Mat& B, const Mat& B_scales, Mat& BT_tile, int j, int max_jj, int k, int max_kk, int block_size)
{
    const int k_end = k + max_kk;
    const int g0 = k / block_size;
    const int g_end = (k_end - 1) / block_size + 1;

    float* pp = BT_tile;
    int jj = 0;
#if __riscv_vector
    for (; jj + 15 < max_jj; jj += 16)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);
        const size_t vl16 = __riscv_vsetvl_e8m1(16);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 16;

            const unsigned char* p = pB + (size_t)k0 * 8;
            const vfloat32m4_t _descale = __riscv_vle32_v_f32m4(descales, vl16);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const uint32_t w[4] = {unpack_int4x4(p), unpack_int4x4(p + 2), unpack_int4x4(p + 4), unpack_int4x4(p + 6)};
                const vint8m1_t _v = __riscv_vle8_v_i8m1((const signed char*)w, vl16);
                const vfloat32m4_t _f = __riscv_vfmul_vv_f32m4(int8_to_float32(_v, vl16), _descale, vl16);
                __riscv_vse32_v_f32m4(pp, _f, vl16);
                p += 8;
                pp += 16;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);
        const size_t vl8 = __riscv_vsetvl_e8m1(8);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 8;

            const unsigned char* p = pB + (size_t)k0 * 4;
            const vfloat32m4_t _descale = __riscv_vle32_v_f32m4(descales, vl8);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const uint32_t w[2] = {unpack_int4x4(p), unpack_int4x4(p + 2)};
                const vint8m1_t _v = __riscv_vle8_v_i8m1((const signed char*)w, vl8);
                const vfloat32m4_t _f = __riscv_vfmul_vv_f32m4(int8_to_float32(_v, vl8), _descale, vl8);
                __riscv_vse32_v_f32m4(pp, _f, vl8);
                p += 4;
                pp += 8;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);
        const size_t vl4 = __riscv_vsetvl_e8m1(4);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 4;

            const unsigned char* p = pB + (size_t)k0 * 2;
            const vfloat32m4_t _descale = __riscv_vle32_v_f32m4(descales, vl4);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const uint32_t w = unpack_int4x4(p);
                const vint8m1_t _v = __riscv_vle8_v_i8m1((const signed char*)&w, vl4);
                const vfloat32m4_t _f = __riscv_vfmul_vv_f32m4(int8_to_float32(_v, vl4), _descale, vl4);
                __riscv_vse32_v_f32m4(pp, _f, vl4);
                p += 2;
                pp += 4;
            }
        }
    }
#endif // __riscv_vector
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

            const unsigned char* p = pB + k0;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                const int v = *p++;
                pp[0] = (((v & 15) ^ 8) - 8) * descales[0];
                pp[1] = (((v >> 4) ^ 8) - 8) * descales[1];
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
                const int index = k0 + kk;
                const int v = (pB[index / 2] >> (index % 2 * 4)) & 15;
                *pp++ = ((v ^ 8) - 8) * descale;
            }
        }
    }
}
