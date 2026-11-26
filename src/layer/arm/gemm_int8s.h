// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_B_tile_int8s(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int j, int max_jj, int K)
{
    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        unsigned char* pp0 = BT.row<unsigned char>(j + jj);
        const int bytes = 12;

        for (int jj0 = 0; jj0 < 12; jj0 += 4)
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
        for (int jj0 = 0; jj0 < 12; jj0++)
        {
            const float* scales = B_scales.row(j + jj + jj0);
            for (int g = 0; g < BT_descales.w; g++)
                descales[(size_t)g * 12 + jj0] = scales[g];
        }
    }
#endif // __aarch64__
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
#endif // __ARM_NEON
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
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        const unsigned char* pB = B.row<const unsigned char>(j + jj);
        const float* scales = B_scales.row(j + jj);

        for (int g = g0; g < g_end; g++)
        {
            const int k_block = g * block_size;
            const int k0 = std::max(k, k_block);
            const int max_kk0 = std::min(k_end - k0, block_size - (k0 - k_block));
            const float* descales = scales + (size_t)g * 12;

            const unsigned char* p = pB + (size_t)k0 * 12;
            const float32x4_t _descale0 = vld1q_f32(descales);
            const float32x4_t _descale1 = vld1q_f32(descales + 4);
            const float32x4_t _descale2 = vld1q_f32(descales + 8);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                uint32_t v0;
                memcpy(&v0, p, sizeof(v0));
                uint32_t v1;
                memcpy(&v1, p + 4, sizeof(v1));
                uint32_t v2;
                memcpy(&v2, p + 8, sizeof(v2));
                const int16x8_t _v0 = vmovl_s8(vreinterpret_s8_u64(vcreate_u64(v0)));
                const int16x8_t _v1 = vmovl_s8(vreinterpret_s8_u64(vcreate_u64(v1)));
                const int16x8_t _v2 = vmovl_s8(vreinterpret_s8_u64(vcreate_u64(v2)));
                const float32x4_t _f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_v0)));
                const float32x4_t _f1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_v1)));
                const float32x4_t _f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_v2)));
                vst1q_f32(pp, vmulq_f32(_f0, _descale0));
                vst1q_f32(pp + 4, vmulq_f32(_f1, _descale1));
                vst1q_f32(pp + 8, vmulq_f32(_f2, _descale2));
                p += 12;
                pp += 12;
            }
        }
    }
#endif // __aarch64__
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
            const float32x4_t _descale0 = vld1q_f32(descales);
            const float32x4_t _descale1 = vld1q_f32(descales + 4);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                uint32_t v0;
                memcpy(&v0, p, sizeof(v0));
                uint32_t v1;
                memcpy(&v1, p + 4, sizeof(v1));
                const int16x8_t _v0 = vmovl_s8(vreinterpret_s8_u64(vcreate_u64(v0)));
                const int16x8_t _v1 = vmovl_s8(vreinterpret_s8_u64(vcreate_u64(v1)));
                const float32x4_t _f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_v0)));
                const float32x4_t _f1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_v1)));
                vst1q_f32(pp, vmulq_f32(_f0, _descale0));
                vst1q_f32(pp + 4, vmulq_f32(_f1, _descale1));
                p += 8;
                pp += 8;
            }
        }
    }
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

            const unsigned char* p = pB + (size_t)k0 * 4;
            const float32x4_t _descale = vld1q_f32(descales);
            for (int kk = 0; kk < max_kk0; kk++)
            {
                uint32_t v;
                memcpy(&v, p, sizeof(v));
                const int16x8_t _v = vmovl_s8(vreinterpret_s8_u64(vcreate_u64(v)));
                const float32x4_t _f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_v)));
                vst1q_f32(pp, vmulq_f32(_f, _descale));
                p += 4;
                pp += 4;
            }
        }
    }
#endif // __ARM_NEON
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
