// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void pack_B_tile_wq_int4_i8mm(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void gemm_transB_packed_tile_wq_int4_i8mm(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
void pack_B_tile_wq_int4_asimddp(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void gemm_transB_packed_tile_wq_int4_asimddp(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

static inline signed char arm_wq_int4_unpack(const unsigned char* ptr, int index)
{
    return (signed char)(ptr[index >> 1] << ((index & 1) ? 0 : 4) & 0xf0);
}

static inline void arm_wq_int4_pack_pair(unsigned char* ptr)
{
    unsigned char tmp[8];
    for (int i = 0; i < 8; i++)
        tmp[i] = (ptr[i / 2] >> ((i & 1) * 4) & 15) | (ptr[(i + 8) / 2] >> (((i + 8) & 1) * 4) & 15) << 4;
    for (int i = 0; i < 8; i++)
        ptr[i] = tmp[i];
}

#if __ARM_NEON
static inline int8x8_t arm_wq_int4_load8(const unsigned char* ptr)
{
    int8x8_t _p4 = vreinterpret_s8_u32(vld1_dup_u32((const unsigned int*)ptr));
    int8x8_t _lo = vshl_n_s8(_p4, 4);
    int8x8_t _hi = vand_s8(_p4, vdup_n_s8((signed char)0xf0));
    return vzip_s8(_lo, _hi).val[0];
}

static inline int8x16_t arm_wq_int4_load16(const unsigned char* ptr)
{
    int8x8_t _p8 = vreinterpret_s8_u8(vld1_u8(ptr));
    int8x8_t _lo = vshl_n_s8(_p8, 4);
    int8x8_t _hi = vand_s8(_p8, vdup_n_s8((signed char)0xf0));
    int8x8x2_t _p = vzip_s8(_lo, _hi);
    return vcombine_s8(_p.val[0], _p.val[1]);
}

static inline int8x16_t arm_wq_int4_load16_pair(const unsigned char* ptr)
{
    int8x8_t _p = vreinterpret_s8_u8(vld1_u8(ptr));
    int8x8_t _lo = vshl_n_s8(_p, 4);
    int8x8_t _hi = vand_s8(_p, vdup_n_s8((signed char)0xf0));
    return vcombine_s8(_lo, _hi);
}

static inline int8x8_t arm_wq_int4_load4_dup(const unsigned char* ptr)
{
    int8x8_t _p2 = vreinterpret_s8_u16(vld1_dup_u16((const unsigned short*)ptr));
    int8x8_t _lo = vshl_n_s8(_p2, 4);
    int8x8_t _hi = vand_s8(_p2, vdup_n_s8((signed char)0xf0));
    return vzip_s8(_lo, _hi).val[0];
}

static inline int8x8_t arm_wq_int4_load2_dup(const unsigned char* ptr)
{
    int8x8_t _p1 = vld1_dup_s8((const signed char*)ptr);
    int8x8_t _lo = vshl_n_s8(_p1, 4);
    int8x8_t _hi = vand_s8(_p1, vdup_n_s8((signed char)0xf0));
    return vzip_s8(_lo, _hi).val[0];
}
#endif // __ARM_NEON

static void pack_B_tile_wq_int4(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        pack_B_tile_wq_int4_i8mm(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        pack_B_tile_wq_int4_asimddp(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

    const int block_count = (K + block_size - 1) / block_size;
    unsigned char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __ARM_NEON
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
            for (; kk + 31 < max_kk; kk += 32)
            {
                uint8x16_t _p0 = vld1q_u8(p0);
                uint8x16_t _p1 = vld1q_u8(p1);
                uint8x16_t _p2 = vld1q_u8(p2);
                uint8x16_t _p3 = vld1q_u8(p3);
#if __ARM_FEATURE_MATMUL_INT8
                uint32x4x4_t _r0123;
                _r0123.val[0] = vreinterpretq_u32_u8(_p0);
                _r0123.val[1] = vreinterpretq_u32_u8(_p1);
                _r0123.val[2] = vreinterpretq_u32_u8(_p2);
                _r0123.val[3] = vreinterpretq_u32_u8(_p3);
                vst4q_u32((unsigned int*)pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
                uint16x8x4_t _r0123;
                _r0123.val[0] = vreinterpretq_u16_u8(_p0);
                _r0123.val[1] = vreinterpretq_u16_u8(_p1);
                _r0123.val[2] = vreinterpretq_u16_u8(_p2);
                _r0123.val[3] = vreinterpretq_u16_u8(_p3);
                vst4q_u16((unsigned short*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                uint8x16x4_t _r0123;
                _r0123.val[0] = _p0;
                _r0123.val[1] = _p1;
                _r0123.val[2] = _p2;
                _r0123.val[3] = _p3;
                vst4q_u8(pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
                for (int q = 0; q < 64; q += 8)
                    arm_wq_int4_pack_pair(pp + q);
                pp += 64;
                p0 += 16;
                p1 += 16;
                p2 += 16;
                p3 += 16;
            }
            for (; kk + 15 < max_kk; kk += 16)
            {
                uint8x8_t _p0 = vld1_u8(p0);
                uint8x8_t _p1 = vld1_u8(p1);
                uint8x8_t _p2 = vld1_u8(p2);
                uint8x8_t _p3 = vld1_u8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                uint32x2x4_t _r0123;
                _r0123.val[0] = vreinterpret_u32_u8(_p0);
                _r0123.val[1] = vreinterpret_u32_u8(_p1);
                _r0123.val[2] = vreinterpret_u32_u8(_p2);
                _r0123.val[3] = vreinterpret_u32_u8(_p3);
                vst4_u32((unsigned int*)pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
                uint16x4x4_t _r0123;
                _r0123.val[0] = vreinterpret_u16_u8(_p0);
                _r0123.val[1] = vreinterpret_u16_u8(_p1);
                _r0123.val[2] = vreinterpret_u16_u8(_p2);
                _r0123.val[3] = vreinterpret_u16_u8(_p3);
                vst4_u16((unsigned short*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                uint8x8x4_t _r0123;
                _r0123.val[0] = _p0;
                _r0123.val[1] = _p1;
                _r0123.val[2] = _p2;
                _r0123.val[3] = _p3;
                vst4_u8(pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
                for (int q = 0; q < 32; q += 8)
                    arm_wq_int4_pack_pair(pp + q);
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
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
                arm_wq_int4_pack_pair(pp);
                arm_wq_int4_pack_pair(pp + 8);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp[4] = p2[0];
                pp[5] = p2[1];
                pp[6] = p3[0];
                pp[7] = p3[1];
                arm_wq_int4_pack_pair(pp);
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; kk + 1 < max_kk; kk += 2)
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
            for (; kk < max_kk; kk++)
            {
                pp[0] = (unsigned char)((p0[0] & 15) | ((p1[0] & 15) << 4));
                pp[1] = (unsigned char)((p2[0] & 15) | ((p3[0] & 15) << 4));
                pp += 2;
                p0++;
                p1++;
                p2++;
                p3++;
            }

            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
            *pd++ = (1.f / *ps2++) * 0.0625f;
            *pd++ = (1.f / *ps3++) * 0.0625f;
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
#if __ARM_NEON
            for (; kk + 31 < max_kk; kk += 32)
            {
                uint8x16_t _p0 = vld1q_u8(p0);
                uint8x16_t _p1 = vld1q_u8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                uint32x4x2_t _r01;
                _r01.val[0] = vreinterpretq_u32_u8(_p0);
                _r01.val[1] = vreinterpretq_u32_u8(_p1);
                vst2q_u32((unsigned int*)pp, _r01);
#else  // __ARM_FEATURE_MATMUL_INT8
                uint16x8x2_t _r01;
                _r01.val[0] = vreinterpretq_u16_u8(_p0);
                _r01.val[1] = vreinterpretq_u16_u8(_p1);
                vst2q_u16((unsigned short*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                uint8x16x2_t _r01;
                _r01.val[0] = _p0;
                _r01.val[1] = _p1;
                vst2q_u8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
                pp += 32;
                p0 += 16;
                p1 += 16;
            }
            for (; kk + 15 < max_kk; kk += 16)
            {
                uint8x8_t _p0 = vld1_u8(p0);
                uint8x8_t _p1 = vld1_u8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                uint32x2x2_t _r01;
                _r01.val[0] = vreinterpret_u32_u8(_p0);
                _r01.val[1] = vreinterpret_u32_u8(_p1);
                vst2_u32((unsigned int*)pp, _r01);
#else  // __ARM_FEATURE_MATMUL_INT8
                uint16x4x2_t _r01;
                _r01.val[0] = vreinterpret_u16_u8(_p0);
                _r01.val[1] = vreinterpret_u16_u8(_p1);
                vst2_u16((unsigned short*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                uint8x8x2_t _r01;
                _r01.val[0] = _p0;
                _r01.val[1] = _p1;
                vst2_u8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
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
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
#endif // __ARM_FEATURE_DOTPROD
#endif // __ARM_NEON
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
            for (; kk < max_kk; kk++)
            {
                *pp++ = (unsigned char)((p0[0] & 15) | ((p1[0] & 15) << 4));
                p0++;
                p1++;
            }

            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
#if __ARM_NEON
            for (; kk + 31 < max_kk; kk += 32)
            {
                vst1q_u8(pp, vld1q_u8(p0));
                pp += 16;
                p0 += 16;
            }
            for (; kk + 15 < max_kk; kk += 16)
            {
                vst1_u8(pp, vld1_u8(p0));
                pp += 8;
                p0 += 8;
            }
#endif // __ARM_NEON
            for (; kk + 1 < max_kk; kk += 2)
                *pp++ = *p0++;
            for (; kk < max_kk; kk++)
                *pp++ = *p0++ & 15;

            *pd++ = (1.f / *ps0++) * 0.0625f;
        }
    }
}

static void gemm_transB_packed_tile_wq_int4(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        gemm_transB_packed_tile_wq_int4_i8mm(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        gemm_transB_packed_tile_wq_int4_asimddp(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

    const signed char* pAT = AT_tile;
    const int A_hstep = AT_tile.w;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = AT_descales_tile.w;
    const unsigned char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    const int block_count = (K + block_size - 1) / block_size;
    const int block_start = k / block_size;

    float* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        int jj = 0;
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float32x4_t _fsum0;
            float32x4_t _fsum1;
            float32x4_t _fsum2;
            float32x4_t _fsum3;
            float32x4_t _fsum4;
            float32x4_t _fsum5;
            float32x4_t _fsum6;
            float32x4_t _fsum7;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
                _fsum1 = vdupq_n_f32(0.f);
                _fsum2 = vdupq_n_f32(0.f);
                _fsum3 = vdupq_n_f32(0.f);
                _fsum4 = vdupq_n_f32(0.f);
                _fsum5 = vdupq_n_f32(0.f);
                _fsum6 = vdupq_n_f32(0.f);
                _fsum7 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vld1q_f32(outptr);
                _fsum1 = vld1q_f32(outptr + 4);
                _fsum2 = vld1q_f32(outptr + 8);
                _fsum3 = vld1q_f32(outptr + 12);
                _fsum4 = vld1q_f32(outptr + 16);
                _fsum5 = vld1q_f32(outptr + 20);
                _fsum6 = vld1q_f32(outptr + 24);
                _fsum7 = vld1q_f32(outptr + 28);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
                int32x4_t _s4 = vdupq_n_s32(0);
                int32x4_t _s5 = vdupq_n_s32(0);
                int32x4_t _s6 = vdupq_n_s32(0);
                int32x4_t _s7 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x16_t _a2 = vld1q_s8(pA + 32);
                    int8x16_t _a3 = vld1q_s8(pA + 48);
                    int8x16_t _b0 = arm_wq_int4_load16_pair(pB);
                    int8x16_t _b1 = arm_wq_int4_load16_pair(pB + 8);
#if __ARM_FEATURE_MATMUL_INT8
                    _s0 = vmmlaq_s32(_s0, _a0, _b0);
                    _s1 = vmmlaq_s32(_s1, _a1, _b0);
                    _s2 = vmmlaq_s32(_s2, _a0, _b1);
                    _s3 = vmmlaq_s32(_s3, _a1, _b1);
                    _s4 = vmmlaq_s32(_s4, _a2, _b0);
                    _s5 = vmmlaq_s32(_s5, _a3, _b0);
                    _s6 = vmmlaq_s32(_s6, _a2, _b1);
                    _s7 = vmmlaq_s32(_s7, _a3, _b1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _a0, _b0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a0, _b0, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a0, _b0, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a0, _b0, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _a1, _b0, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _a1, _b0, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _a1, _b0, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _a1, _b0, 3);

                    _sum0 = vdotq_laneq_s32(_sum0, _a2, _b1, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a2, _b1, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a2, _b1, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a2, _b1, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _a3, _b1, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _a3, _b1, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _a3, _b1, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _a3, _b1, 3);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 64;
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss0 = vuzpq_s32(_s0, _s1);
                int32x4x2_t _ss1 = vuzpq_s32(_s2, _s3);
                int32x4x2_t _ss2 = vuzpq_s32(_s4, _s5);
                int32x4x2_t _ss3 = vuzpq_s32(_s6, _s7);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
                _sum4 = vaddq_s32(_sum4, _ss2.val[0]);
                _sum5 = vaddq_s32(_sum5, _ss2.val[1]);
                _sum6 = vaddq_s32(_sum6, _ss3.val[0]);
                _sum7 = vaddq_s32(_sum7, _ss3.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x16_t _b = arm_wq_int4_load16_pair(pB);
                    _sum0 = vdotq_laneq_s32(_sum0, _a0, _b, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a0, _b, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a0, _b, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a0, _b, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _a1, _b, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _a1, _b, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _a1, _b, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _a1, _b, 3);
                    pA += 32;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA2 = vld1q_s8(pA + 16);
                    int8x16_t _pB02 = arm_wq_int4_load16_pair(pB);
                    int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));
                    int8x16_t _pA3 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA2)));
                    int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB02));
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB02));
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB02));
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB02));
                    int16x8_t _s4 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB13));
                    int16x8_t _s5 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB13));
                    int16x8_t _s6 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB13));
                    int16x8_t _s7 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB13));
                    _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_high_s8(_pB02));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB02));
                    _s2 = vmlal_s8(_s2, vget_low_s8(_pA3), vget_high_s8(_pB02));
                    _s3 = vmlal_s8(_s3, vget_high_s8(_pA3), vget_high_s8(_pB02));
                    _s4 = vmlal_s8(_s4, vget_low_s8(_pA2), vget_high_s8(_pB13));
                    _s5 = vmlal_s8(_s5, vget_high_s8(_pA2), vget_high_s8(_pB13));
                    _s6 = vmlal_s8(_s6, vget_low_s8(_pA3), vget_high_s8(_pB13));
                    _s7 = vmlal_s8(_s7, vget_high_s8(_pA3), vget_high_s8(_pB13));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
                    _sum4 = vpadalq_s16(_sum4, _s4);
                    _sum5 = vpadalq_s16(_sum5, _s5);
                    _sum6 = vpadalq_s16(_sum6, _s6);
                    _sum7 = vpadalq_s16(_sum7, _s7);
                    pA += 32;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = arm_wq_int4_load8(pB);
                    int16x8_t _s0 = vmull_s8(vget_low_s8(_a), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 0)));
                    int16x8_t _s1 = vmull_s8(vget_low_s8(_a), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 1)));
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_a), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 2)));
                    int16x8_t _s3 = vmull_s8(vget_low_s8(_a), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 3)));
                    int16x8_t _s4 = vmull_s8(vget_high_s8(_a), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 0)));
                    int16x8_t _s5 = vmull_s8(vget_high_s8(_a), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 1)));
                    int16x8_t _s6 = vmull_s8(vget_high_s8(_a), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 2)));
                    int16x8_t _s7 = vmull_s8(vget_high_s8(_a), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 3)));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
                    _sum4 = vpadalq_s16(_sum4, _s4);
                    _sum5 = vpadalq_s16(_sum5, _s5);
                    _sum6 = vpadalq_s16(_sum6, _s6);
                    _sum7 = vpadalq_s16(_sum7, _s7);
#else  // __ARM_FEATURE_DOTPROD
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x8_t _pB0 = arm_wq_int4_load8(pB);
                    int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));
                    int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), _pB0);
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), _pB0);
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_pA1), _pB0);
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_pA1), _pB0);
                    int16x8_t _s4 = vmull_s8(vget_low_s8(_pA0), _pB1);
                    int16x8_t _s5 = vmull_s8(vget_high_s8(_pA0), _pB1);
                    int16x8_t _s6 = vmull_s8(vget_low_s8(_pA1), _pB1);
                    int16x8_t _s7 = vmull_s8(vget_high_s8(_pA1), _pB1);
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
                    _sum4 = vpadalq_s16(_sum4, _s4);
                    _sum5 = vpadalq_s16(_sum5, _s5);
                    _sum6 = vpadalq_s16(_sum6, _s6);
                    _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD
                    pA += 16;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = arm_wq_int4_load4_dup(pB);
                    int16x8_t _s0 = vmull_s8(_a, vdup_lane_s8(_b, 0));
                    int16x8_t _s1 = vmull_s8(_a, vdup_lane_s8(_b, 1));
                    int16x8_t _s2 = vmull_s8(_a, vdup_lane_s8(_b, 2));
                    int16x8_t _s3 = vmull_s8(_a, vdup_lane_s8(_b, 3));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_low_s16(_s1));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s2));
                    _sum3 = vaddw_s16(_sum3, vget_low_s16(_s3));
                    _sum4 = vaddw_s16(_sum4, vget_high_s16(_s0));
                    _sum5 = vaddw_s16(_sum5, vget_high_s16(_s1));
                    _sum6 = vaddw_s16(_sum6, vget_high_s16(_s2));
                    _sum7 = vaddw_s16(_sum7, vget_high_s16(_s3));
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA0 = vld1_s8(pA);
                    int8x8_t _pB0 = arm_wq_int4_load4_dup(pB);
                    int8x8_t _pA1 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));
                    int8x8_t _pB1 = vrev64_s8(_pB0);

                    int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                    int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                    int16x8_t _s45 = vmull_s8(_pA0, _pB1);
                    int16x8_t _s67 = vmull_s8(_pA1, _pB1);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                    _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                    _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                    _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                    _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
#endif // __ARM_FEATURE_DOTPROD
                    pA += 8;
                    pB += 2;
                }

                float32x4_t _descaleB0 = vld1q_f32(pB_descales);
                float32x4_t _descaleA0 = vld1q_f32(pA_descales);
                float32x4_t _descaleA1 = vld1q_f32(pA_descales + 4);
#if __ARM_FEATURE_DOTPROD
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_descaleA0, vgetq_lane_f32(_descaleB0, 0)));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_descaleA0, vgetq_lane_f32(_descaleB0, 1)));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_n_f32(_descaleA0, vgetq_lane_f32(_descaleB0, 2)));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_n_f32(_descaleA0, vgetq_lane_f32(_descaleB0, 3)));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_n_f32(_descaleA1, vgetq_lane_f32(_descaleB0, 0)));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_n_f32(_descaleA1, vgetq_lane_f32(_descaleB0, 1)));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_n_f32(_descaleA1, vgetq_lane_f32(_descaleB0, 2)));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_n_f32(_descaleA1, vgetq_lane_f32(_descaleB0, 3)));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _descaleA2 = vextq_f32(_descaleA0, _descaleA0, 2);
                float32x4_t _descaleA3 = vextq_f32(_descaleA1, _descaleA1, 2);
                float32x4_t _descaleB1 = vrev64q_f32(_descaleB0);
                _descaleB1 = vextq_f32(_descaleB1, _descaleB1, 2);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_descaleA0, _descaleB0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_descaleA1, _descaleB0));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_f32(_descaleA2, _descaleB0));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_f32(_descaleA3, _descaleB0));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_f32(_descaleA0, _descaleB1));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_f32(_descaleA1, _descaleB1));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_f32(_descaleA2, _descaleB1));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_f32(_descaleA3, _descaleB1));
#endif // __ARM_FEATURE_DOTPROD
                pA_descales += 8;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            vst1q_f32(outptr + 4, _fsum1);
            vst1q_f32(outptr + 8, _fsum2);
            vst1q_f32(outptr + 12, _fsum3);
            vst1q_f32(outptr + 16, _fsum4);
            vst1q_f32(outptr + 20, _fsum5);
            vst1q_f32(outptr + 24, _fsum6);
            vst1q_f32(outptr + 28, _fsum7);
            outptr += 32;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float32x4_t _fsum0;
            float32x4_t _fsum1;
            float32x4_t _fsum2;
            float32x4_t _fsum3;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
                _fsum1 = vdupq_n_f32(0.f);
                _fsum2 = vdupq_n_f32(0.f);
                _fsum3 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vld1q_f32(outptr);
                _fsum1 = vld1q_f32(outptr + 4);
                _fsum2 = vld1q_f32(outptr + 8);
                _fsum3 = vld1q_f32(outptr + 12);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x16_t _a45 = vld1q_s8(pA + 32);
                    int8x16_t _a67 = vld1q_s8(pA + 48);
                    int8x16_t _b = arm_wq_int4_load16(pB);
#if __ARM_FEATURE_MATMUL_INT8
                    _s0 = vmmlaq_s32(_s0, _a01, _b);
                    _s1 = vmmlaq_s32(_s1, _a23, _b);
                    _s2 = vmmlaq_s32(_s2, _a45, _b);
                    _s3 = vmmlaq_s32(_s3, _a67, _b);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _a01, _b, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a01, _b, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a23, _b, 0);
                    _sum3 = vdotq_laneq_s32(_sum3, _a23, _b, 1);
                    _sum0 = vdotq_laneq_s32(_sum0, _a45, _b, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _a45, _b, 3);
                    _sum2 = vdotq_laneq_s32(_sum2, _a67, _b, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a67, _b, 3);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 64;
                    pB += 8;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss0 = vuzpq_s32(_s0, _s1);
                int32x4x2_t _ss1 = vuzpq_s32(_s2, _s3);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x8_t _b = arm_wq_int4_load8(pB);
                    _sum0 = vdotq_lane_s32(_sum0, _a0, _b, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _a0, _b, 1);
                    _sum2 = vdotq_lane_s32(_sum2, _a1, _b, 0);
                    _sum3 = vdotq_lane_s32(_sum3, _a1, _b, 1);
                    pA += 32;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA2 = vld1q_s8(pA + 16);
                    int8x8_t _pB = arm_wq_int4_load8(pB);
                    int32x2x2_t _pBB = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));
                    int8x16_t _pB02 = vreinterpretq_s8_s32(vcombine_s32(_pBB.val[0], _pBB.val[1]));
                    int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB02));
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB02));
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB13));
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB13));
                    _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_high_s8(_pB02));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB02));
                    _s2 = vmlal_s8(_s2, vget_low_s8(_pA2), vget_high_s8(_pB13));
                    _s3 = vmlal_s8(_s3, vget_high_s8(_pA2), vget_high_s8(_pB13));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
                    pA += 32;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x16_t _a = vld1q_s8(pA);
                    int16x4_t _b = vreinterpret_s16_s8(arm_wq_int4_load4_dup(pB));
                    int16x4x2_t _b01 = vuzp_s16(_b, _b);
                    int8x8_t _b0 = vreinterpret_s8_s16(_b01.val[0]);
                    int8x8_t _b1 = vreinterpret_s8_s16(_b01.val[1]);
                    int16x8_t _s0 = vmull_s8(vget_low_s8(_a), _b0);
                    int16x8_t _s1 = vmull_s8(vget_low_s8(_a), _b1);
                    int16x8_t _s2 = vmull_s8(vget_high_s8(_a), _b0);
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_a), _b1);
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
#else  // __ARM_FEATURE_DOTPROD
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x8_t _pB0 = arm_wq_int4_load4_dup(pB);
                    int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_pA), _pB0);
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_pA), _pB1);
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_pA), _pB1);
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD
                    pA += 16;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = arm_wq_int4_load2_dup(pB);
                    int8x8x2_t _b01 = vuzp_s8(_b, _b);
                    int16x8_t _s0 = vmull_s8(_a, _b01.val[0]);
                    int16x8_t _s1 = vmull_s8(_a, _b01.val[1]);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_low_s16(_s1));
                    _sum2 = vaddw_s16(_sum2, vget_high_s16(_s0));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB0 = arm_wq_int4_load2_dup(pB);
                    int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);

                    int16x8_t _s0 = vmull_s8(_pA, _pB0);
                    int16x8_t _s1 = vmull_s8(_pA, _pB1);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
#endif // __ARM_FEATURE_DOTPROD
                    pA += 8;
                    pB += 1;
                }

                float32x2_t _descaleB = vld1_f32(pB_descales);
                float32x4_t _descaleA0 = vld1q_f32(pA_descales);
                float32x4_t _descaleA1 = vld1q_f32(pA_descales + 4);
#if __ARM_FEATURE_DOTPROD
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_descaleA0, _descaleB, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_descaleA0, _descaleB, 1));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_lane_f32(_descaleA1, _descaleB, 0));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_lane_f32(_descaleA1, _descaleB, 1));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _descaleB01 = vcombine_f32(_descaleB, _descaleB);
                float32x4_t _descaleB10 = vrev64q_f32(_descaleB01);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_descaleA0, _descaleB01));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_descaleA1, _descaleB01));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_f32(_descaleA0, _descaleB10));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_f32(_descaleA1, _descaleB10));
#endif // __ARM_FEATURE_DOTPROD
                pA_descales += 8;
                pB_descales += 2;
            }

            vst1q_f32(outptr, _fsum0);
            vst1q_f32(outptr + 4, _fsum1);
            vst1q_f32(outptr + 8, _fsum2);
            vst1q_f32(outptr + 12, _fsum3);
            outptr += 16;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float32x4_t _fsum0;
            float32x4_t _fsum1;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
                _fsum1 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vld1q_f32(outptr);
                _fsum1 = vld1q_f32(outptr + 4);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x16_t _a45 = vld1q_s8(pA + 32);
                    int8x16_t _a67 = vld1q_s8(pA + 48);
                    int8x8_t _b = arm_wq_int4_load8(pB);
#if __ARM_FEATURE_MATMUL_INT8
                    int8x16_t _bb = vcombine_s8(_b, _b);
                    int32x4_t _s0 = vdotq_s32(vdupq_n_s32(0), _a01, _bb);
                    int32x4_t _s1 = vdotq_s32(vdupq_n_s32(0), _a23, _bb);
                    int32x4_t _s2 = vdotq_s32(vdupq_n_s32(0), _a45, _bb);
                    int32x4_t _s3 = vdotq_s32(vdupq_n_s32(0), _a67, _bb);
                    _sum0 = vaddq_s32(_sum0, vpaddq_s32(_s0, _s1));
                    _sum1 = vaddq_s32(_sum1, vpaddq_s32(_s2, _s3));
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _a01, _b, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _a23, _b, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _a45, _b, 1);
                    _sum1 = vdotq_lane_s32(_sum1, _a67, _b, 1);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 64;
                    pB += 4;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x8_t _b0 = arm_wq_int4_load4_dup(pB);
                    int8x16_t _b = vcombine_s8(_b0, _b0);
                    _sum0 = vdotq_s32(_sum0, _a0, _b);
                    _sum1 = vdotq_s32(_sum1, _a1, _b);
                    pA += 32;
                    pB += 2;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int16x4_t _b = vreinterpret_s16_s8(arm_wq_int4_load4_dup(pB));
                    int8x8_t _b0 = vreinterpret_s8_s16(vdup_lane_s16(_b, 0));
                    int8x8_t _b1 = vreinterpret_s8_s16(vdup_lane_s16(_b, 1));
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(vget_low_s8(_a0), _b0));
                    _sum1 = vpadalq_s16(_sum1, vmull_s8(vget_high_s8(_a0), _b0));
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(vget_low_s8(_a1), _b1));
                    _sum1 = vpadalq_s16(_sum1, vmull_s8(vget_high_s8(_a1), _b1));
                    pA += 32;
                    pB += 2;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = arm_wq_int4_load2_dup(pB);
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(vget_low_s8(_a), _b));
                    _sum1 = vpadalq_s16(_sum1, vmull_s8(vget_high_s8(_a), _b));
                    pA += 16;
                    pB += 1;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int16x8_t _s = vmull_s8(_a, vdup_n_s8(arm_wq_int4_unpack(pB, 0)));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s));
                    pA += 8;
                    pB++;
                }

                float32x4_t _descaleB = vdupq_n_f32(pB_descales[0]);
                float32x4_t _descaleA0 = vld1q_f32(pA_descales);
                float32x4_t _descaleA1 = vld1q_f32(pA_descales + 4);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_descaleB, _descaleA0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_descaleB, _descaleA1));
                pA_descales += 8;
                pB_descales++;
            }

            vst1q_f32(outptr, _fsum0);
            vst1q_f32(outptr + 4, _fsum1);
            outptr += 8;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }
        pAT += (size_t)8 * A_hstep;
        pAT_descales += (size_t)8 * A_descales_hstep;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        int jj = 0;
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            float32x4_t _fsum0;
            float32x4_t _fsum1;
            float32x4_t _fsum2;
            float32x4_t _fsum3;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
                _fsum1 = vdupq_n_f32(0.f);
                _fsum2 = vdupq_n_f32(0.f);
                _fsum3 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vld1q_f32(outptr);
                _fsum1 = vld1q_f32(outptr + 4);
                _fsum2 = vld1q_f32(outptr + 8);
                _fsum3 = vld1q_f32(outptr + 12);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x16_t _b0 = arm_wq_int4_load16_pair(pB);
                    int8x16_t _b1 = arm_wq_int4_load16_pair(pB + 8);
#if __ARM_FEATURE_MATMUL_INT8
                    _s0 = vmmlaq_s32(_s0, _a0, _b0);
                    _s1 = vmmlaq_s32(_s1, _a1, _b0);
                    _s2 = vmmlaq_s32(_s2, _a0, _b1);
                    _s3 = vmmlaq_s32(_s3, _a1, _b1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _a0, _b0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a0, _b0, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a0, _b0, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a0, _b0, 3);
                    _sum0 = vdotq_laneq_s32(_sum0, _a1, _b1, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a1, _b1, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a1, _b1, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a1, _b1, 3);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 32;
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss0 = vuzpq_s32(_s0, _s1);
                int32x4x2_t _ss1 = vuzpq_s32(_s2, _s3);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x16_t _b = arm_wq_int4_load16_pair(pB);
                    _sum0 = vdotq_laneq_s32(_sum0, _a, _b, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a, _b, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a, _b, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a, _b, 3);
                    pA += 16;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _pA02 = vld1q_s8(pA);
                    int8x16_t _pB02 = arm_wq_int4_load16_pair(pB);
                    int8x16_t _pA13 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA02)));
                    int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB02));
                    int16x8_t _s1 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB02));
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB13));
                    int16x8_t _s3 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB13));
                    _s0 = vmlal_s8(_s0, vget_high_s8(_pA02), vget_high_s8(_pB02));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_pA13), vget_high_s8(_pB02));
                    _s2 = vmlal_s8(_s2, vget_high_s8(_pA02), vget_high_s8(_pB13));
                    _s3 = vmlal_s8(_s3, vget_high_s8(_pA13), vget_high_s8(_pB13));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
                    pA += 16;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = arm_wq_int4_load8(pB);
                    int16x8_t _s0 = vmull_s8(_a, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 0)));
                    int16x8_t _s1 = vmull_s8(_a, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 1)));
                    int16x8_t _s2 = vmull_s8(_a, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 2)));
                    int16x8_t _s3 = vmull_s8(_a, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 3)));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA0 = vld1_s8(pA);
                    int8x8_t _pB0 = arm_wq_int4_load8(pB);
                    int8x8_t _pA1 = vext_s8(_pA0, _pA0, 4);
                    int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                    int16x8_t _s0 = vmull_s8(_pA0, _pB0);
                    int16x8_t _s1 = vmull_s8(_pA1, _pB0);
                    int16x8_t _s2 = vmull_s8(_pA0, _pB1);
                    int16x8_t _s3 = vmull_s8(_pA1, _pB1);
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _b = arm_wq_int4_load4_dup(pB);
                    _b = vzip_s8(_b, _b).val[0];
                    int16x4x2_t _b0123 = vzip_s16(vreinterpret_s16_s8(_b), vreinterpret_s16_s8(_b));
                    int16x8_t _s01 = vmull_s8(_a, vreinterpret_s8_s16(_b0123.val[0]));
                    int16x8_t _s23 = vmull_s8(_a, vreinterpret_s8_s16(_b0123.val[1]));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA0 = vld1_s8(pA);
                    int8x8_t _pB0 = arm_wq_int4_load4_dup(pB);
                    int8x8_t _pA1 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));
                    int8x8_t _pA01 = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pA0), vreinterpret_s32_s8(_pA1)).val[0]);
                    int8x8_t _pB1 = vrev32_s8(_pB0);

                    int16x8_t _s01 = vmull_s8(_pA01, _pB0);
                    int16x8_t _s23 = vmull_s8(_pA01, _pB1);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
#endif // __ARM_FEATURE_DOTPROD
                    pA += 4;
                    pB += 2;
                }

                float32x4_t _descaleB0 = vld1q_f32(pB_descales);
                float32x4_t _descaleA = vld1q_f32(pA_descales);
#if __ARM_FEATURE_DOTPROD
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_descaleA, vgetq_lane_f32(_descaleB0, 0)));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_descaleA, vgetq_lane_f32(_descaleB0, 1)));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_n_f32(_descaleA, vgetq_lane_f32(_descaleB0, 2)));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_n_f32(_descaleA, vgetq_lane_f32(_descaleB0, 3)));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _descaleA1 = vextq_f32(_descaleA, _descaleA, 2);
                float32x4_t _descaleB1 = vrev64q_f32(_descaleB0);
                _descaleB1 = vextq_f32(_descaleB1, _descaleB1, 2);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_descaleA, _descaleB0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_descaleA1, _descaleB0));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_f32(_descaleA, _descaleB1));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_f32(_descaleA1, _descaleB1));
#endif // __ARM_FEATURE_DOTPROD

                pA_descales += 4;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            vst1q_f32(outptr, _fsum2);
            outptr += 4;
            vst1q_f32(outptr, _fsum3);
            outptr += 4;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float32x4_t _fsum0;
            float32x4_t _fsum1;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
                _fsum1 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vld1q_f32(outptr);
                _fsum1 = vld1q_f32(outptr + 4);
            }

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x16_t _b = arm_wq_int4_load16(pB);
#if __ARM_FEATURE_MATMUL_INT8
                    _s0 = vmmlaq_s32(_s0, _a0, _b);
                    _s1 = vmmlaq_s32(_s1, _a1, _b);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _a0, _b, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a0, _b, 1);
                    _sum0 = vdotq_laneq_s32(_sum0, _a1, _b, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _a1, _b, 3);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 32;
                    pB += 8;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss = vuzpq_s32(_s0, _s1);
                _sum0 = vaddq_s32(_sum0, _ss.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = arm_wq_int4_load8(pB);
                    _sum0 = vdotq_lane_s32(_sum0, _a, _b, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _a, _b, 1);
                    pA += 16;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x8_t _pB = arm_wq_int4_load8(pB);
                    int32x2x2_t _pBB = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));
                    int8x16_t _pB02 = vreinterpretq_s8_s32(vcombine_s32(_pBB.val[0], _pBB.val[1]));
                    int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB02));
                    int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB13));
                    _s0 = vmlal_s8(_s0, vget_high_s8(_pA), vget_high_s8(_pB02));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_pA), vget_high_s8(_pB13));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    pA += 16;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = arm_wq_int4_load4_dup(pB);
                    int16x8_t _s0 = vmull_s8(_a, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 0)));
                    int16x8_t _s1 = vmull_s8(_a, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 1)));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB0 = arm_wq_int4_load4_dup(pB);
                    int8x8_t _pB1 = vext_s8(_pB0, _pB0, 2);

                    int16x8_t _s0 = vmull_s8(_pA, _pB0);
                    int16x8_t _s1 = vmull_s8(_pA, _pB1);
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD
                    pA += 8;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _b = arm_wq_int4_load2_dup(pB);
                    _b = vuzp_s8(_b, vext_s8(_b, _b, 1)).val[0];
                    int16x8_t _s = vmull_s8(_a, _b);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s));
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _pB0 = arm_wq_int4_load2_dup(pB);
                    int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);
                    int8x8_t _pB = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB0), vreinterpret_s32_s8(_pB1)).val[0]);

                    int16x8_t _s0 = vmull_s8(_pA, _pB);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
#endif // __ARM_FEATURE_DOTPROD
                    pA += 4;
                    pB += 1;
                }

                float32x2_t _descaleB = vld1_f32(pB_descales);
                float32x4_t _descaleA = vld1q_f32(pA_descales);
#if __ARM_FEATURE_DOTPROD
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_descaleA, _descaleB, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_descaleA, _descaleB, 1));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _descaleB01 = vcombine_f32(_descaleB, _descaleB);
                float32x4_t _descaleB10 = vrev64q_f32(_descaleB01);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_descaleA, _descaleB01));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_descaleA, _descaleB10));
#endif // __ARM_FEATURE_DOTPROD
                pA_descales += 4;
                pB_descales += 2;
            }

            vst1q_f32(outptr, _fsum0);
            vst1q_f32(outptr + 4, _fsum1);
            outptr += 8;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float32x4_t _fsum;

            if (k == 0)
                _fsum = vdupq_n_f32(0.f);
            else
                _fsum = vld1q_f32(outptr);

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a01 = vld1q_s8(pA);
                    int8x16_t _a23 = vld1q_s8(pA + 16);
                    int8x8_t _b = arm_wq_int4_load8(pB);
#if __ARM_FEATURE_MATMUL_INT8
                    int8x16_t _bb = vcombine_s8(_b, _b);
                    int32x4_t _s0 = vdotq_s32(vdupq_n_s32(0), _a01, _bb);
                    int32x4_t _s1 = vdotq_s32(vdupq_n_s32(0), _a23, _bb);
                    _sum = vaddq_s32(_sum, vcombine_s32(vpadd_s32(vget_low_s32(_s0), vget_high_s32(_s0)), vpadd_s32(vget_low_s32(_s1), vget_high_s32(_s1))));
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum = vdotq_lane_s32(_sum, _a01, _b, 0);
                    _sum = vdotq_lane_s32(_sum, _a23, _b, 1);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 32;
                    pB += 4;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b0 = arm_wq_int4_load4_dup(pB);
                    int8x16_t _b = vcombine_s8(_b0, _b0);
                    _sum = vdotq_s32(_sum, _a, _b);
                    pA += 16;
                    pB += 2;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a0 = vld1_s8(pA);
                    int8x8_t _a1 = vld1_s8(pA + 8);
                    int16x4_t _b = vreinterpret_s16_s8(arm_wq_int4_load4_dup(pB));
                    int8x8_t _b0 = vreinterpret_s8_s16(vdup_lane_s16(_b, 0));
                    int8x8_t _b1 = vreinterpret_s8_s16(vdup_lane_s16(_b, 1));
                    _sum = vpadalq_s16(_sum, vmull_s8(_a0, _b0));
                    _sum = vpadalq_s16(_sum, vmull_s8(_a1, _b1));
                    pA += 16;
                    pB += 2;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = arm_wq_int4_load2_dup(pB);
                    _sum = vpadalq_s16(_sum, vmull_s8(_a, _b));
                    pA += 8;
                    pB += 1;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int16x8_t _s = vmull_s8(_a, vdup_n_s8(arm_wq_int4_unpack(pB, 0)));
                    _sum = vaddw_s16(_sum, vget_low_s16(_s));
                    pA += 4;
                    pB++;
                }

                float32x4_t _descaleB = vdupq_n_f32(pB_descales[0]);
                float32x4_t _descaleA = vld1q_f32(pA_descales);
                _fsum = vmlaq_f32(_fsum, vcvtq_f32_s32(_sum), vmulq_f32(_descaleB, _descaleA));
                pA_descales += 4;
                pB_descales++;
            }

            vst1q_f32(outptr, _fsum);
            outptr += 4;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }
        pAT += (size_t)4 * A_hstep;
        pAT_descales += (size_t)4 * A_descales_hstep;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        int jj = 0;
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
#if __ARM_NEON
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            float32x4_t _fsum0;
            float32x4_t _fsum1;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
                _fsum1 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vld1q_f32(outptr);
                _fsum1 = vld1q_f32(outptr + 4);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = arm_wq_int4_load16_pair(pB);
                    int8x16_t _b1 = arm_wq_int4_load16_pair(pB + 8);
                    int8x16_t _a0 = vld1q_s8(pA);
#if __ARM_FEATURE_MATMUL_INT8
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a0, 1);
                    _sum0 = vdotq_laneq_s32(_sum0, _b1, _a0, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _b1, _a0, 3);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 16;
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum1));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = arm_wq_int4_load16_pair(pB);
                    int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    pA += 8;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    int8x16_t _b = arm_wq_int4_load16_pair(pB);
                    int8x8_t _b01 = vget_low_s8(_b);
                    int8x8_t _b23 = vget_high_s8(_b);
                    int16x8_t _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)));
                    _sum1 = vpadalq_s16(_sum1, _s);
                    pA += 8;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _b0 = arm_wq_int4_load8(pB);
                    int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = arm_wq_int4_load4_dup(pB);
                    int8x8_t _a = vreinterpret_s8_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0));
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    pA += 2;
                    pB += 2;
                }

                float32x4_t _descaleB0 = vld1q_f32(pB_descales);
                float32x2_t _descaleA = vld1_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_descaleB0, _descaleA, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_descaleB0, _descaleA, 1));

                pA_descales += 2;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_NEON
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            float32x4_t _fsum;

            if (k == 0)
                _fsum = vdupq_n_f32(0.f);
            else
                _fsum = vld1q_f32(outptr);

            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x16_t _b = arm_wq_int4_load16(pB);
#if __ARM_FEATURE_MATMUL_INT8
                    _sum = vmmlaq_s32(_sum, _a, _b);
#else  // __ARM_FEATURE_MATMUL_INT8
                    int32x4x2_t _aa = vzipq_s32(vreinterpretq_s32_s8(_a), vreinterpretq_s32_s8(_a));
                    int8x16_t _a01 = vreinterpretq_s8_s32(_aa.val[0]);
                    int8x16_t _a23 = vreinterpretq_s8_s32(_aa.val[1]);
                    int8x16_t _b01 = vcombine_s8(vget_low_s8(_b), vget_low_s8(_b));
                    int8x16_t _b23 = vcombine_s8(vget_high_s8(_b), vget_high_s8(_b));
                    _sum = vdotq_s32(_sum, _a01, _b01);
                    _sum = vdotq_s32(_sum, _a23, _b23);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 16;
                    pB += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    int8x8_t _b = arm_wq_int4_load8(pB);
                    int32x4_t _s0 = vdotq_lane_s32(vdupq_n_s32(0), _a, _b, 0);
                    int32x4_t _s1 = vdotq_lane_s32(vdupq_n_s32(0), _a, _b, 1);
                    _sum = vaddq_s32(_sum, vzipq_s32(_s0, _s1).val[0]);
                    pA += 8;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB = arm_wq_int4_load8(pB);

                    int16x4x2_t _pA01 = vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA));
                    int32x2x2_t _pB01 = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));

                    int16x8_t _s0 = vmull_s8(vreinterpret_s8_s16(_pA01.val[0]), vreinterpret_s8_s32(_pB01.val[0]));
                    _s0 = vmlal_s8(_s0, vreinterpret_s8_s16(_pA01.val[1]), vreinterpret_s8_s32(_pB01.val[1]));
                    _sum = vpadalq_s16(_sum, _s0);
                    pA += 8;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int16x4_t _b = vreinterpret_s16_s8(arm_wq_int4_load4_dup(pB));
                    int16x4x2_t _b01 = vuzp_s16(_b, _b);
                    int32x4_t _s0 = vpaddlq_s16(vmull_s8(_a, vreinterpret_s8_s16(_b01.val[0])));
                    int32x4_t _s1 = vpaddlq_s16(vmull_s8(_a, vreinterpret_s8_s16(_b01.val[1])));
                    _sum = vaddq_s32(_sum, vzipq_s32(_s0, _s1).val[0]);
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                    int8x8_t _b = arm_wq_int4_load2_dup(pB);
                    int8x8_t _aa = vzip_s8(_a, _a).val[0];
                    _sum = vaddq_s32(_sum, vmovl_s16(vget_low_s16(vmull_s8(_aa, _b))));
                    pA += 2;
                    pB += 1;
                }

                float32x2_t _descaleA = vld1_f32(pA_descales);
                float32x2_t _descaleB = vld1_f32(pB_descales);
                float32x4_t _descaleA01 = vcombine_f32(_descaleA, _descaleA);
                float32x4_t _descaleB01 = vcombine_f32(_descaleB, _descaleB);
                _fsum = vmlaq_f32(_fsum, vcvtq_f32_s32(_sum), vmulq_f32(vzipq_f32(_descaleA01, _descaleA01).val[0], _descaleB01));
                pA_descales += 2;
                pB_descales += 2;
            }

            vst1q_f32(outptr, _fsum);
            outptr += 4;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
#else
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            float fsum00;
            float fsum01;
            float fsum10;
            float fsum11;

            if (k == 0)
            {
                fsum00 = 0.f;
                fsum01 = 0.f;
                fsum10 = 0.f;
                fsum11 = 0.f;
            }
            else
            {
                fsum00 = outptr[0];
                fsum10 = outptr[1];
                fsum01 = outptr[2];
                fsum11 = outptr[3];
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const int b00 = arm_wq_int4_unpack(pB, 0);
                    const int b01 = arm_wq_int4_unpack(pB, 1);
                    const int b10 = arm_wq_int4_unpack(pB, 2);
                    const int b11 = arm_wq_int4_unpack(pB, 3);
                    sum00 += pA[0] * b00 + pA[2] * b01;
                    sum01 += pA[0] * b10 + pA[2] * b11;
                    sum10 += pA[1] * b00 + pA[3] * b01;
                    sum11 += pA[1] * b10 + pA[3] * b11;
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const int b0 = arm_wq_int4_unpack(pB, 0);
                    const int b1 = arm_wq_int4_unpack(pB, 1);
                    sum00 += pA[0] * b0;
                    sum01 += pA[0] * b1;
                    sum10 += pA[1] * b0;
                    sum11 += pA[1] * b1;
                    pA += 2;
                    pB += 1;
                }

                const float bd0 = pB_descales[0];
                const float bd1 = pB_descales[1];
                const float ad0 = pA_descales[0];
                fsum00 += sum00 * ad0 * bd0;
                fsum01 += sum01 * ad0 * bd1;
                const float ad1 = pA_descales[1];
                fsum10 += sum10 * ad1 * bd0;
                fsum11 += sum11 * ad1 * bd1;

                pA_descales += 2;
                pB_descales += 2;
            }

            outptr[0] = fsum00;
            outptr++;
            outptr[0] = fsum10;
            outptr++;
            outptr[0] = fsum01;
            outptr++;
            outptr[0] = fsum11;
            outptr++;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
#endif // __ARM_NEON
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
#if __ARM_NEON
            float32x2_t _fsum;

            if (k == 0)
                _fsum = vdup_n_f32(0.f);
            else
                _fsum = vld1_f32(outptr);
#else
            float fsum00;
            float fsum10;

            if (k == 0)
            {
                fsum00 = 0.f;
                fsum10 = 0.f;
            }
            else
            {
                fsum00 = outptr[0];
                fsum10 = outptr[1];
            }
#endif // __ARM_NEON

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum0 = 0;
                int sum1 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_NEON
                int32x2_t _sum = vdup_n_s32(0);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                {
                    int32x4_t _sum0 = vdupq_n_s32(0);
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        int8x16_t _a = vld1q_s8(pA);
                        int8x8_t _b = arm_wq_int4_load8(pB);
                        int8x16_t _bb = vcombine_s8(_b, _b);
                        _sum0 = vdotq_s32(_sum0, _a, _bb);
                        pA += 16;
                        pB += 4;
                    }
                    _sum = vadd_s32(_sum, vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0)));
                }
#else  // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = arm_wq_int4_load8(pB);
                    _sum = vdot_lane_s32(_sum, vget_low_s8(_a), _b, 0);
                    _sum = vdot_lane_s32(_sum, vget_high_s8(_a), _b, 1);
                    pA += 16;
                    pB += 4;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = arm_wq_int4_load4_dup(pB);
                    _sum = vdot_s32(_sum, _a, _b);
                    pA += 8;
                    pB += 2;
                }
#else  // __ARM_FEATURE_DOTPROD
                {
                    int32x4_t _sum0 = vdupq_n_s32(0);
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        int8x8_t _a = vld1_s8(pA);
                        int8x8_t _b = arm_wq_int4_load4_dup(pB);
                        _b = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_b), vreinterpret_s16_s8(_b)).val[0]);
                        _sum0 = vpadalq_s16(_sum0, vmull_s8(_a, _b));
                        pA += 8;
                        pB += 2;
                    }
                    _sum = vadd_s32(_sum, vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0)));
                }
#endif // __ARM_FEATURE_DOTPROD
                sum0 = vget_lane_s32(_sum, 0);
                sum1 = vget_lane_s32(_sum, 1);
#endif // __ARM_NEON
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_NEON
                    sum0 += pA[0] * arm_wq_int4_unpack(pB, 0);
                    sum0 += pA[1] * arm_wq_int4_unpack(pB, 1);
                    sum1 += pA[2] * arm_wq_int4_unpack(pB, 0);
                    sum1 += pA[3] * arm_wq_int4_unpack(pB, 1);
#else
                    sum0 += pA[0] * arm_wq_int4_unpack(pB, 0);
                    sum0 += pA[2] * arm_wq_int4_unpack(pB, 1);
                    sum1 += pA[1] * arm_wq_int4_unpack(pB, 0);
                    sum1 += pA[3] * arm_wq_int4_unpack(pB, 1);
#endif // __ARM_NEON
                    pA += 4;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * arm_wq_int4_unpack(pB, 0);
                    sum1 += pA[1] * arm_wq_int4_unpack(pB, 0);
                    pA += 2;
                    pB++;
                }

#if __ARM_NEON
                float32x2_t _descaleA = vld1_f32(pA_descales);
                float32x2_t _scale = vmul_n_f32(_descaleA, pB_descales[0]);
                _sum = vset_lane_s32(sum0, _sum, 0);
                _sum = vset_lane_s32(sum1, _sum, 1);
                _fsum = vmla_f32(_fsum, vcvt_f32_s32(_sum), _scale);
#else
                const float bd0 = pB_descales[0];
                const float ad0 = pA_descales[0];
                fsum00 += sum0 * ad0 * bd0;
                const float ad1 = pA_descales[1];
                fsum10 += sum1 * ad1 * bd0;
#endif // __ARM_NEON
                pA_descales += 2;
                pB_descales++;
            }

#if __ARM_NEON
            vst1_f32(outptr, _fsum);
#else
            outptr[0] = fsum00;
            outptr[1] = fsum10;
#endif // __ARM_NEON
            outptr += 2;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }
        pAT += (size_t)2 * A_hstep;
        pAT_descales += (size_t)2 * A_descales_hstep;
    }
    for (; ii < max_ii; ii++)
    {
        int jj = 0;
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
#if __ARM_NEON
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            float32x4_t _fsum0;

            if (k == 0)
            {
                _fsum0 = vdupq_n_f32(0.f);
            }
            else
            {
                _fsum0 = vld1q_f32(outptr);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _b0 = arm_wq_int4_load16_pair(pB);
                    int8x16_t _b1 = arm_wq_int4_load16_pair(pB + 8);
                    int8x16_t _a0 = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
#if __ARM_FEATURE_MATMUL_INT8
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _b0, vget_low_s8(_a0), 0);
                    _sum0 = vdotq_lane_s32(_sum0, _b1, vget_low_s8(_a0), 1);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 8;
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = arm_wq_int4_load16_pair(pB);
                    int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    pA += 4;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    int8x16_t _b = arm_wq_int4_load16_pair(pB);
                    int16x8_t _s = vmull_s8(vget_low_s8(_b), vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, vget_high_s8(_b), vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    pA += 4;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _b0 = arm_wq_int4_load8(pB);
                    int8x8_t _a0 = vreinterpret_s8_s16(vdup_lane_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, _a0)));
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = arm_wq_int4_load4_dup(pB);
                    int8x8_t _a0 = vld1_lane_s8(pA, vdup_n_s8(0), 0);
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    pA++;
                    pB += 2;
                }

                float32x4_t _descaleB0 = vld1q_f32(pB_descales);
                const float _descaleA0 = pA_descales[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_descaleB0, _descaleA0));

                pA_descales++;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
#if __ARM_NEON
            float32x2_t _fsum0;

            if (k == 0)
            {
                _fsum0 = vdup_n_f32(0.f);
            }
            else
            {
                _fsum0 = vld1_f32(outptr);
            }
#else
            float fsum00;
            float fsum01;

            if (k == 0)
            {
                fsum00 = 0.f;
                fsum01 = 0.f;
            }
            else
            {
                fsum00 = outptr[0];
                fsum01 = outptr[1];
            }
#endif // __ARM_NEON

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum0 = 0;
                int sum1 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_NEON
                int32x2_t _sum0 = vdup_n_s32(0);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                {
                    int32x4_t _sum = vdupq_n_s32(0);
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        int8x8_t _a = vld1_s8(pA);
                        int8x16_t _b = arm_wq_int4_load16(pB);
                        int8x16_t _aa = vcombine_s8(_a, _a);
                        _sum = vdotq_s32(_sum, _aa, _b);
                        pA += 8;
                        pB += 8;
                    }
                    _sum0 = vadd_s32(_sum0, vpadd_s32(vget_low_s32(_sum), vget_high_s32(_sum)));
                }
#else  // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x16_t _b = arm_wq_int4_load16(pB);
                    _sum0 = vdot_lane_s32(_sum0, vget_low_s8(_b), _a, 0);
                    _sum0 = vdot_lane_s32(_sum0, vget_high_s8(_b), _a, 1);
                    pA += 8;
                    pB += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _b = arm_wq_int4_load8(pB);
                    _sum0 = vdot_s32(_sum0, _a, _b);
                    pA += 4;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                {
                    int32x4_t _sum = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);
                    for (; kk + 15 < max_kk0; kk += 16)
                    {
                        int8x16_t _a = vld1q_s8(pA);
                        int8x16_t _b0 = arm_wq_int4_load16(pB);
                        int8x16_t _b1 = arm_wq_int4_load16(pB + 8);
                        int16x8x2_t _aa = vzipq_s16(vreinterpretq_s16_s8(_a), vreinterpretq_s16_s8(_a));
                        int8x8_t _a0 = vreinterpret_s8_s16(vget_low_s16(_aa.val[0]));
                        int8x8_t _a1 = vreinterpret_s8_s16(vget_high_s16(_aa.val[0]));
                        int8x8_t _a2 = vreinterpret_s8_s16(vget_low_s16(_aa.val[1]));
                        int8x8_t _a3 = vreinterpret_s8_s16(vget_high_s16(_aa.val[1]));
                        int16x8_t _s0 = vmull_s8(_a0, vget_low_s8(_b0));
                        int16x8_t _s1 = vmull_s8(_a2, vget_low_s8(_b1));
                        _s0 = vmlal_s8(_s0, _a1, vget_high_s8(_b0));
                        _s1 = vmlal_s8(_s1, _a3, vget_high_s8(_b1));
                        _sum = vpadalq_s16(_sum, _s0);
                        _sum1 = vpadalq_s16(_sum1, _s1);
                        pA += 16;
                        pB += 16;
                    }
                    _sum = vaddq_s32(_sum, _sum1);
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        int8x8_t _a = vld1_s8(pA);
                        int8x16_t _b = arm_wq_int4_load16(pB);
                        int16x4x2_t _aa = vzip_s16(vreinterpret_s16_s8(_a), vreinterpret_s16_s8(_a));
                        int8x8_t _a0 = vreinterpret_s8_s16(_aa.val[0]);
                        int8x8_t _a1 = vreinterpret_s8_s16(_aa.val[1]);
                        int16x8_t _s0 = vmull_s8(_a0, vget_low_s8(_b));
                        _s0 = vmlal_s8(_s0, _a1, vget_high_s8(_b));
                        _sum = vpadalq_s16(_sum, _s0);
                        pA += 8;
                        pB += 8;
                    }
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        int8x8_t _a = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                        int8x8_t _b = arm_wq_int4_load8(pB);
                        _a = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_a), vreinterpret_s16_s8(_a)).val[0]);
                        _sum = vpadalq_s16(_sum, vmull_s8(_a, _b));
                        pA += 4;
                        pB += 4;
                    }
                    _sum0 = vadd_s32(_sum0, vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum)));
                }
#endif // __ARM_FEATURE_DOTPROD
                sum0 = vget_lane_s32(_sum0, 0);
                sum1 = vget_lane_s32(_sum0, 1);
#endif // __ARM_NEON
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    sum0 += pA[0] * arm_wq_int4_unpack(pB, 0);
                    sum0 += pA[1] * arm_wq_int4_unpack(pB, 1);
                    sum1 += pA[0] * arm_wq_int4_unpack(pB, 2);
                    sum1 += pA[1] * arm_wq_int4_unpack(pB, 3);
                    pA += 2;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * arm_wq_int4_unpack(pB, 0);
                    sum1 += pA[0] * arm_wq_int4_unpack(pB, 1);
                    pA++;
                    pB += 1;
                }

#if __ARM_NEON
                float32x2_t _descaleB0 = vld1_f32(pB_descales);
                float32x2_t _scale = vmul_n_f32(_descaleB0, pA_descales[0]);
                _sum0 = vset_lane_s32(sum0, _sum0, 0);
                _sum0 = vset_lane_s32(sum1, _sum0, 1);
                _fsum0 = vmla_f32(_fsum0, vcvt_f32_s32(_sum0), _scale);
#else
                const float bd0 = pB_descales[0];
                const float bd1 = pB_descales[1];
                const float ad0 = pA_descales[0];
                fsum00 += sum0 * ad0 * bd0;
                fsum01 += sum1 * ad0 * bd1;
#endif // __ARM_NEON
                pA_descales++;
                pB_descales += 2;
            }

#if __ARM_NEON
            vst1_f32(outptr, _fsum0);
#else
            outptr[0] = fsum00;
            outptr[1] = fsum01;
#endif // __ARM_NEON
            outptr += 2;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            float fsum00;

            if (k == 0)
            {
                fsum00 = 0.f;
            }
            else
            {
                fsum00 = outptr[0];
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum00 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
#if __ARM_NEON
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                for (; kk + 31 < max_kk0; kk += 32)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x16_t _b0 = arm_wq_int4_load16(pB);
                    int8x16_t _b1 = arm_wq_int4_load16(pB + 8);
#if __ARM_FEATURE_DOTPROD
                    _sum0 = vdotq_s32(_sum0, _a0, _b0);
                    _sum1 = vdotq_s32(_sum1, _a1, _b1);
#else  // __ARM_FEATURE_DOTPROD
                    int16x8_t _s0 = vmull_s8(vget_low_s8(_a0), vget_low_s8(_b0));
                    int16x8_t _s1 = vmull_s8(vget_low_s8(_a1), vget_low_s8(_b1));
                    _s0 = vmlal_s8(_s0, vget_high_s8(_a0), vget_high_s8(_b0));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_a1), vget_high_s8(_b1));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD
                    pA += 32;
                    pB += 16;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
#if __ARM_FEATURE_DOTPROD
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x16_t _b = arm_wq_int4_load16(pB);
                    _sum0 = vdotq_s32(_sum0, _a, _b);
                    pA += 16;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = arm_wq_int4_load8(pB);
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(_a, _b));
                    pA += 8;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
#if __aarch64__
                sum00 = vaddvq_s32(_sum0);
#else
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _ss = vpadd_s32(_ss, _ss);
                sum00 = vget_lane_s32(_ss, 0);
#endif
#endif // __ARM_NEON
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    sum00 += pA[0] * arm_wq_int4_unpack(pB, 0);
                    sum00 += pA[1] * arm_wq_int4_unpack(pB, 1);
                    pA += 2;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum00 += pA[0] * arm_wq_int4_unpack(pB, 0);
                    pA++;
                    pB++;
                }

                const float bd0 = pB_descales[0];
                const float ad0 = pA_descales[0];
                fsum00 += sum00 * ad0 * bd0;

                pA_descales++;
                pB_descales++;
            }

            outptr[0] = fsum00;
            outptr++;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void get_optimal_tile_mnk_wq_int4(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    const float bytes_per_k = sizeof(signed char) + 0.5f + 8.f / block_size;
    int tile_size = (int)sqrtf((float)l2_cache_size / (bytes_per_k + sizeof(float)));

#if __aarch64__
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(2, tile_size / 2 * 2);
#endif

    TILE_K = std::max(block_size, tile_size / block_size * block_size);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + block_size - 1) / block_size * block_size);
        TILE_K = std::min(TILE_K, K);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / bytes_per_k / TILE_K);

#if __aarch64__
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
            TILE_N = std::max(2, tile_size / 2 * 2);
#endif
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __aarch64__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __aarch64__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2);
#endif
    }

    if (nT > 1)
    {
#if __aarch64__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __aarch64__
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#elif __ARM_NEON
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_N > 0)
    {
#if __aarch64__
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#elif __ARM_NEON
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = (constant_TILE_N + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, (constant_TILE_K + block_size - 1) / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(TILE_K, K);
    }
}
