// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void quantize_A_tile_wq_int8_bf16s_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_bf16s_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
void quantize_A_tile_wq_int8_bf16s_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_bf16s_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
#endif

static void quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        quantize_A_tile_wq_int8_bf16s_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        quantize_A_tile_wq_int8_bf16s_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;

    if (input_scales.empty())
    {
        int ii = 0;
#if __ARM_NEON
#if __aarch64__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0a));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0a + A_hstep * 4));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        p0a += 4;
                    }

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    float absmax[8];
                    vst1q_f32(absmax, _absmax0);
                    vst1q_f32(absmax + 4, _absmax1);
                    const float scale0 = absmax[0] == 0.f ? 0.f : 127.f / absmax[0];
                    const float scale1 = absmax[1] == 0.f ? 0.f : 127.f / absmax[1];
                    const float scale2 = absmax[2] == 0.f ? 0.f : 127.f / absmax[2];
                    const float scale3 = absmax[3] == 0.f ? 0.f : 127.f / absmax[3];
                    const float scale4 = absmax[4] == 0.f ? 0.f : 127.f / absmax[4];
                    const float scale5 = absmax[5] == 0.f ? 0.f : 127.f / absmax[5];
                    const float scale6 = absmax[6] == 0.f ? 0.f : 127.f / absmax[6];
                    const float scale7 = absmax[7] == 0.f ? 0.f : 127.f / absmax[7];

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x8x4_t _p = vld4q_u16(p0);
                        uint16x8x4_t _q = vld4q_u16(p0 + A_hstep * 4);
                        int8x8_t _r0 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_p.val[0])), scale0), vmulq_n_f32(bfloat2float(vget_high_u16(_p.val[0])), scale0));
                        int8x8_t _r1 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_p.val[1])), scale1), vmulq_n_f32(bfloat2float(vget_high_u16(_p.val[1])), scale1));
                        int8x8_t _r2 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_p.val[2])), scale2), vmulq_n_f32(bfloat2float(vget_high_u16(_p.val[2])), scale2));
                        int8x8_t _r3 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_p.val[3])), scale3), vmulq_n_f32(bfloat2float(vget_high_u16(_p.val[3])), scale3));
                        int8x8_t _r4 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_q.val[0])), scale4), vmulq_n_f32(bfloat2float(vget_high_u16(_q.val[0])), scale4));
                        int8x8_t _r5 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_q.val[1])), scale5), vmulq_n_f32(bfloat2float(vget_high_u16(_q.val[1])), scale5));
                        int8x8_t _r6 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_q.val[2])), scale6), vmulq_n_f32(bfloat2float(vget_high_u16(_q.val[2])), scale6));
                        int8x8_t _r7 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_q.val[3])), scale7), vmulq_n_f32(bfloat2float(vget_high_u16(_q.val[3])), scale7));
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                        pp += 64;
                        p0 += 32;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        uint16x4x4_t _q = vld4_u16(p0 + A_hstep * 4);
                        int8x8_t _r01 = float2int8(vmulq_n_f32(bfloat2float(_p.val[0]), scale0), vmulq_n_f32(bfloat2float(_p.val[1]), scale1));
                        int8x8_t _r23 = float2int8(vmulq_n_f32(bfloat2float(_p.val[2]), scale2), vmulq_n_f32(bfloat2float(_p.val[3]), scale3));
                        int8x8_t _r45 = float2int8(vmulq_n_f32(bfloat2float(_q.val[0]), scale4), vmulq_n_f32(bfloat2float(_q.val[1]), scale5));
                        int8x8_t _r67 = float2int8(vmulq_n_f32(bfloat2float(_q.val[2]), scale6), vmulq_n_f32(bfloat2float(_q.val[3]), scale7));
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
                        vst1q_s8(pp + 16, vcombine_s8(_r45, _r67));
#else
                        int16x8x2_t _r04 = vuzpq_s16(vreinterpretq_s16_s8(vcombine_s8(_r01, _r23)), vreinterpretq_s16_s8(vcombine_s8(_r45, _r67)));
                        vst1q_s16((short*)pp, _r04.val[0]);
                        vst1q_s16((short*)pp + 8, _r04.val[1]);
#endif
                        pp += 32;
                        p0 += 16;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0 + 4));
                        float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                        float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4));
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 0) * scale0);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 0) * scale0);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 1) * scale1);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 1) * scale1);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 2) * scale2);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 2) * scale2);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 3) * scale3);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 3) * scale3);
                        *pp++ = float2int8(vgetq_lane_f32(_p2, 0) * scale4);
                        *pp++ = float2int8(vgetq_lane_f32(_p3, 0) * scale4);
                        *pp++ = float2int8(vgetq_lane_f32(_p2, 1) * scale5);
                        *pp++ = float2int8(vgetq_lane_f32(_p3, 1) * scale5);
                        *pp++ = float2int8(vgetq_lane_f32(_p2, 2) * scale6);
                        *pp++ = float2int8(vgetq_lane_f32(_p3, 2) * scale6);
                        *pp++ = float2int8(vgetq_lane_f32(_p2, 3) * scale7);
                        *pp++ = float2int8(vgetq_lane_f32(_p3, 3) * scale7);
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 0) * scale0);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 1) * scale1);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 2) * scale2);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 3) * scale3);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 0) * scale4);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 1) * scale5);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 2) * scale6);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 3) * scale7);
                        p0 += 4;
                    }
                    pd += 8;
                }
                if (elempack == 1)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    float32x4_t _absmax4 = vdupq_n_f32(0.f);
                    float32x4_t _absmax5 = vdupq_n_f32(0.f);
                    float32x4_t _absmax6 = vdupq_n_f32(0.f);
                    float32x4_t _absmax7 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0a));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0a + A_hstep));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        float32x4_t _p2 = bfloat2float(vld1_u16(p0a + A_hstep * 2));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                        float32x4_t _p3 = bfloat2float(vld1_u16(p0a + A_hstep * 3));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                        float32x4_t _p4 = bfloat2float(vld1_u16(p0a + A_hstep * 4));
                        _absmax4 = vmaxq_f32(_absmax4, vabsq_f32(_p4));
                        float32x4_t _p5 = bfloat2float(vld1_u16(p0a + A_hstep * 5));
                        _absmax5 = vmaxq_f32(_absmax5, vabsq_f32(_p5));
                        float32x4_t _p6 = bfloat2float(vld1_u16(p0a + A_hstep * 6));
                        _absmax6 = vmaxq_f32(_absmax6, vabsq_f32(_p6));
                        float32x4_t _p7 = bfloat2float(vld1_u16(p0a + A_hstep * 7));
                        _absmax7 = vmaxq_f32(_absmax7, vabsq_f32(_p7));
                        p0a += 4;
                    }
                    float absmax0 = vmaxvq_f32(_absmax0);
                    float absmax1 = vmaxvq_f32(_absmax1);
                    float absmax2 = vmaxvq_f32(_absmax2);
                    float absmax3 = vmaxvq_f32(_absmax3);
                    float absmax4 = vmaxvq_f32(_absmax4);
                    float absmax5 = vmaxvq_f32(_absmax5);
                    float absmax6 = vmaxvq_f32(_absmax6);
                    float absmax7 = vmaxvq_f32(_absmax7);
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0a[0]);
                        float v1 = bfloat16_to_float32(p0a[A_hstep]);
                        float v2 = bfloat16_to_float32(p0a[A_hstep * 2]);
                        float v3 = bfloat16_to_float32(p0a[A_hstep * 3]);
                        float v4 = bfloat16_to_float32(p0a[A_hstep * 4]);
                        float v5 = bfloat16_to_float32(p0a[A_hstep * 5]);
                        float v6 = bfloat16_to_float32(p0a[A_hstep * 6]);
                        float v7 = bfloat16_to_float32(p0a[A_hstep * 7]);
                        absmax0 = std::max(absmax0, fabsf(v0));
                        absmax1 = std::max(absmax1, fabsf(v1));
                        absmax2 = std::max(absmax2, fabsf(v2));
                        absmax3 = std::max(absmax3, fabsf(v3));
                        absmax4 = std::max(absmax4, fabsf(v4));
                        absmax5 = std::max(absmax5, fabsf(v5));
                        absmax6 = std::max(absmax6, fabsf(v6));
                        absmax7 = std::max(absmax7, fabsf(v7));
                        p0a++;
                    }

                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    pd[4] = absmax4 / 127.f;
                    pd[5] = absmax5 / 127.f;
                    pd[6] = absmax6 / 127.f;
                    pd[7] = absmax7 / 127.f;
                    const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                    const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
                    const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
                    const float scale4 = absmax4 == 0.f ? 0.f : 127.f / absmax4;
                    const float scale5 = absmax5 == 0.f ? 0.f : 127.f / absmax5;
                    const float scale6 = absmax6 == 0.f ? 0.f : 127.f / absmax6;
                    const float scale7 = absmax7 == 0.f ? 0.f : 127.f / absmax7;

                    kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p01 = bfloat2float(vld1_u16(p0 + 4));
                        float32x4_t _p10 = bfloat2float(vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = bfloat2float(vld1_u16(p0 + A_hstep + 4));
                        float32x4_t _p20 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p21 = bfloat2float(vld1_u16(p0 + A_hstep * 2 + 4));
                        float32x4_t _p30 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p31 = bfloat2float(vld1_u16(p0 + A_hstep * 3 + 4));
                        float32x4_t _p40 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                        float32x4_t _p41 = bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4));
                        float32x4_t _p50 = bfloat2float(vld1_u16(p0 + A_hstep * 5));
                        float32x4_t _p51 = bfloat2float(vld1_u16(p0 + A_hstep * 5 + 4));
                        float32x4_t _p60 = bfloat2float(vld1_u16(p0 + A_hstep * 6));
                        float32x4_t _p61 = bfloat2float(vld1_u16(p0 + A_hstep * 6 + 4));
                        float32x4_t _p70 = bfloat2float(vld1_u16(p0 + A_hstep * 7));
                        float32x4_t _p71 = bfloat2float(vld1_u16(p0 + A_hstep * 7 + 4));
                        int8x8_t _r0 = float2int8(vmulq_n_f32(_p00, scale0), vmulq_n_f32(_p01, scale0));
                        int8x8_t _r1 = float2int8(vmulq_n_f32(_p10, scale1), vmulq_n_f32(_p11, scale1));
                        int8x8_t _r2 = float2int8(vmulq_n_f32(_p20, scale2), vmulq_n_f32(_p21, scale2));
                        int8x8_t _r3 = float2int8(vmulq_n_f32(_p30, scale3), vmulq_n_f32(_p31, scale3));
                        int8x8_t _r4 = float2int8(vmulq_n_f32(_p40, scale4), vmulq_n_f32(_p41, scale4));
                        int8x8_t _r5 = float2int8(vmulq_n_f32(_p50, scale5), vmulq_n_f32(_p51, scale5));
                        int8x8_t _r6 = float2int8(vmulq_n_f32(_p60, scale6), vmulq_n_f32(_p61, scale6));
                        int8x8_t _r7 = float2int8(vmulq_n_f32(_p70, scale7), vmulq_n_f32(_p71, scale7));
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                        pp += 64;
                        p0 += 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                        float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p4 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                        float32x4_t _p5 = bfloat2float(vld1_u16(p0 + A_hstep * 5));
                        float32x4_t _p6 = bfloat2float(vld1_u16(p0 + A_hstep * 6));
                        float32x4_t _p7 = bfloat2float(vld1_u16(p0 + A_hstep * 7));
                        int8x8_t _r01 = float2int8(vmulq_n_f32(_p0, scale0), vmulq_n_f32(_p1, scale1));
                        int8x8_t _r23 = float2int8(vmulq_n_f32(_p2, scale2), vmulq_n_f32(_p3, scale3));
                        int8x8_t _r45 = float2int8(vmulq_n_f32(_p4, scale4), vmulq_n_f32(_p5, scale5));
                        int8x8_t _r67 = float2int8(vmulq_n_f32(_p6, scale6), vmulq_n_f32(_p7, scale7));
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
                        vst1q_s8(pp + 16, vcombine_s8(_r45, _r67));
#else
                        int16x8x2_t _r04 = vuzpq_s16(vreinterpretq_s16_s8(vcombine_s8(_r01, _r23)), vreinterpretq_s16_s8(vcombine_s8(_r45, _r67)));
                        vst1q_s16((short*)pp, _r04.val[0]);
                        vst1q_s16((short*)pp + 8, _r04.val[1]);
#endif
                        pp += 32;
                        p0 += 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float v00 = bfloat16_to_float32(p0[0]);
                        float v01 = bfloat16_to_float32(p0[1]);
                        float v10 = bfloat16_to_float32(p0[A_hstep]);
                        float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                        float v20 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v21 = bfloat16_to_float32(p0[A_hstep * 2 + 1]);
                        float v30 = bfloat16_to_float32(p0[A_hstep * 3]);
                        float v31 = bfloat16_to_float32(p0[A_hstep * 3 + 1]);
                        float v40 = bfloat16_to_float32(p0[A_hstep * 4]);
                        float v41 = bfloat16_to_float32(p0[A_hstep * 4 + 1]);
                        float v50 = bfloat16_to_float32(p0[A_hstep * 5]);
                        float v51 = bfloat16_to_float32(p0[A_hstep * 5 + 1]);
                        float v60 = bfloat16_to_float32(p0[A_hstep * 6]);
                        float v61 = bfloat16_to_float32(p0[A_hstep * 6 + 1]);
                        float v70 = bfloat16_to_float32(p0[A_hstep * 7]);
                        float v71 = bfloat16_to_float32(p0[A_hstep * 7 + 1]);
                        *pp++ = float2int8(v00 * scale0);
                        *pp++ = float2int8(v01 * scale0);
                        *pp++ = float2int8(v10 * scale1);
                        *pp++ = float2int8(v11 * scale1);
                        *pp++ = float2int8(v20 * scale2);
                        *pp++ = float2int8(v21 * scale2);
                        *pp++ = float2int8(v30 * scale3);
                        *pp++ = float2int8(v31 * scale3);
                        *pp++ = float2int8(v40 * scale4);
                        *pp++ = float2int8(v41 * scale4);
                        *pp++ = float2int8(v50 * scale5);
                        *pp++ = float2int8(v51 * scale5);
                        *pp++ = float2int8(v60 * scale6);
                        *pp++ = float2int8(v61 * scale6);
                        *pp++ = float2int8(v70 * scale7);
                        *pp++ = float2int8(v71 * scale7);
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[A_hstep]);
                        float v2 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v3 = bfloat16_to_float32(p0[A_hstep * 3]);
                        float v4 = bfloat16_to_float32(p0[A_hstep * 4]);
                        float v5 = bfloat16_to_float32(p0[A_hstep * 5]);
                        float v6 = bfloat16_to_float32(p0[A_hstep * 6]);
                        float v7 = bfloat16_to_float32(p0[A_hstep * 7]);
                        *pp++ = float2int8(v0 * scale0);
                        *pp++ = float2int8(v1 * scale1);
                        *pp++ = float2int8(v2 * scale2);
                        *pp++ = float2int8(v3 * scale3);
                        *pp++ = float2int8(v4 * scale4);
                        *pp++ = float2int8(v5 * scale5);
                        *pp++ = float2int8(v6 * scale6);
                        *pp++ = float2int8(v7 * scale7);
                        p0++;
                    }
                    pd += 8;
                }
            }
        }
#endif // __aarch64__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 4)
                {
                    float32x4_t _absmax = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = bfloat2float(vld1_u16(p0a));
                        _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
                        p0a += 4;
                    }

                    vst1q_f32(pd, vmulq_n_f32(_absmax, 1.f / 127.f));

                    float absmax[4];
                    vst1q_f32(absmax, _absmax);
                    const float scale0 = absmax[0] == 0.f ? 0.f : 127.f / absmax[0];
                    const float scale1 = absmax[1] == 0.f ? 0.f : 127.f / absmax[1];
                    const float scale2 = absmax[2] == 0.f ? 0.f : 127.f / absmax[2];
                    const float scale3 = absmax[3] == 0.f ? 0.f : 127.f / absmax[3];

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x8x4_t _p = vld4q_u16(p0);
                        int8x8_t _r0 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_p.val[0])), scale0), vmulq_n_f32(bfloat2float(vget_high_u16(_p.val[0])), scale0));
                        int8x8_t _r1 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_p.val[1])), scale1), vmulq_n_f32(bfloat2float(vget_high_u16(_p.val[1])), scale1));
                        int8x8_t _r2 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_p.val[2])), scale2), vmulq_n_f32(bfloat2float(vget_high_u16(_p.val[2])), scale2));
                        int8x8_t _r3 = float2int8(vmulq_n_f32(bfloat2float(vget_low_u16(_p.val[3])), scale3), vmulq_n_f32(bfloat2float(vget_high_u16(_p.val[3])), scale3));
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        pp += 32;
                        p0 += 32;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        int8x8_t _r01 = float2int8(vmulq_n_f32(bfloat2float(_p.val[0]), scale0), vmulq_n_f32(bfloat2float(_p.val[1]), scale1));
                        int8x8_t _r23 = float2int8(vmulq_n_f32(bfloat2float(_p.val[2]), scale2), vmulq_n_f32(bfloat2float(_p.val[3]), scale3));
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                        int16x8_t _r0123 = vreinterpretq_s16_s8(vcombine_s8(_r01, _r23));
                        int16x8x2_t _r02 = vuzpq_s16(_r0123, _r0123);
                        vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_r02.val[0]), vget_low_s16(_r02.val[1]))));
#endif
                        pp += 16;
                        p0 += 16;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0 + 4));
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 0) * scale0);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 0) * scale0);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 1) * scale1);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 1) * scale1);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 2) * scale2);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 2) * scale2);
                        *pp++ = float2int8(vgetq_lane_f32(_p0, 3) * scale3);
                        *pp++ = float2int8(vgetq_lane_f32(_p1, 3) * scale3);
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = bfloat2float(vld1_u16(p0));
                        *pp++ = float2int8(vgetq_lane_f32(_p, 0) * scale0);
                        *pp++ = float2int8(vgetq_lane_f32(_p, 1) * scale1);
                        *pp++ = float2int8(vgetq_lane_f32(_p, 2) * scale2);
                        *pp++ = float2int8(vgetq_lane_f32(_p, 3) * scale3);
                        p0 += 4;
                    }
                    pd += 4;
                }
                if (elempack == 1)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0a));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0a + A_hstep));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        float32x4_t _p2 = bfloat2float(vld1_u16(p0a + A_hstep * 2));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                        float32x4_t _p3 = bfloat2float(vld1_u16(p0a + A_hstep * 3));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                        p0a += 4;
                    }
#if __aarch64__
                    float absmax0 = vmaxvq_f32(_absmax0);
                    float absmax1 = vmaxvq_f32(_absmax1);
                    float absmax2 = vmaxvq_f32(_absmax2);
                    float absmax3 = vmaxvq_f32(_absmax3);
#else
                    float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    _max0 = vpmax_f32(_max0, _max0);
                    _max1 = vpmax_f32(_max1, _max1);
                    _max2 = vpmax_f32(_max2, _max2);
                    _max3 = vpmax_f32(_max3, _max3);
                    float absmax0 = vget_lane_f32(_max0, 0);
                    float absmax1 = vget_lane_f32(_max1, 0);
                    float absmax2 = vget_lane_f32(_max2, 0);
                    float absmax3 = vget_lane_f32(_max3, 0);
#endif
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0a[0]);
                        float v1 = bfloat16_to_float32(p0a[A_hstep]);
                        float v2 = bfloat16_to_float32(p0a[A_hstep * 2]);
                        float v3 = bfloat16_to_float32(p0a[A_hstep * 3]);
                        absmax0 = std::max(absmax0, fabsf(v0));
                        absmax1 = std::max(absmax1, fabsf(v1));
                        absmax2 = std::max(absmax2, fabsf(v2));
                        absmax3 = std::max(absmax3, fabsf(v3));
                        p0a++;
                    }
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                    const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
                    const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;

                    kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p01 = bfloat2float(vld1_u16(p0 + 4));
                        float32x4_t _p10 = bfloat2float(vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = bfloat2float(vld1_u16(p0 + A_hstep + 4));
                        float32x4_t _p20 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p21 = bfloat2float(vld1_u16(p0 + A_hstep * 2 + 4));
                        float32x4_t _p30 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p31 = bfloat2float(vld1_u16(p0 + A_hstep * 3 + 4));
                        int8x8_t _r0 = float2int8(vmulq_n_f32(_p00, scale0), vmulq_n_f32(_p01, scale0));
                        int8x8_t _r1 = float2int8(vmulq_n_f32(_p10, scale1), vmulq_n_f32(_p11, scale1));
                        int8x8_t _r2 = float2int8(vmulq_n_f32(_p20, scale2), vmulq_n_f32(_p21, scale2));
                        int8x8_t _r3 = float2int8(vmulq_n_f32(_p30, scale3), vmulq_n_f32(_p31, scale3));
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        pp += 32;
                        p0 += 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                        float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
                        int8x8_t _r01 = float2int8(vmulq_n_f32(_p0, scale0), vmulq_n_f32(_p1, scale1));
                        int8x8_t _r23 = float2int8(vmulq_n_f32(_p2, scale2), vmulq_n_f32(_p3, scale3));
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                        int16x8_t _r0123 = vreinterpretq_s16_s8(vcombine_s8(_r01, _r23));
                        int16x8x2_t _r02 = vuzpq_s16(_r0123, _r0123);
                        vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_r02.val[0]), vget_low_s16(_r02.val[1]))));
#endif
                        pp += 16;
                        p0 += 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float v00 = bfloat16_to_float32(p0[0]);
                        float v01 = bfloat16_to_float32(p0[1]);
                        float v10 = bfloat16_to_float32(p0[A_hstep]);
                        float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                        float v20 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v21 = bfloat16_to_float32(p0[A_hstep * 2 + 1]);
                        float v30 = bfloat16_to_float32(p0[A_hstep * 3]);
                        float v31 = bfloat16_to_float32(p0[A_hstep * 3 + 1]);
                        *pp++ = float2int8(v00 * scale0);
                        *pp++ = float2int8(v01 * scale0);
                        *pp++ = float2int8(v10 * scale1);
                        *pp++ = float2int8(v11 * scale1);
                        *pp++ = float2int8(v20 * scale2);
                        *pp++ = float2int8(v21 * scale2);
                        *pp++ = float2int8(v30 * scale3);
                        *pp++ = float2int8(v31 * scale3);
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[A_hstep]);
                        float v2 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v3 = bfloat16_to_float32(p0[A_hstep * 3]);
                        *pp++ = float2int8(v0 * scale0);
                        *pp++ = float2int8(v1 * scale1);
                        *pp++ = float2int8(v2 * scale2);
                        *pp++ = float2int8(v3 * scale3);
                        p0++;
                    }
                    pd += 4;
                }
            }
        }
#endif // __ARM_NEON
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                int kk = 0;
#if __ARM_NEON
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = bfloat2float(vld1_u16(p0a));
                    _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                    float32x4_t _p1 = bfloat2float(vld1_u16(p0a + A_hstep));
                    _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                    p0a += 4;
                }
#if __aarch64__
                absmax0 = vmaxvq_f32(_absmax0);
                absmax1 = vmaxvq_f32(_absmax1);
#else
                float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                _max0 = vpmax_f32(_max0, _max0);
                _max1 = vpmax_f32(_max1, _max1);
                absmax0 = vget_lane_f32(_max0, 0);
                absmax1 = vget_lane_f32(_max1, 0);
#endif
#endif // __ARM_NEON

                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0a[0]);
                    float v1 = bfloat16_to_float32(p0a[A_hstep]);
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                    p0a++;
                }

                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;

                kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p00 = bfloat2float(vld1_u16(p0));
                    float32x4_t _p01 = bfloat2float(vld1_u16(p0 + 4));
                    float32x4_t _p10 = bfloat2float(vld1_u16(p0 + A_hstep));
                    float32x4_t _p11 = bfloat2float(vld1_u16(p0 + A_hstep + 4));
                    int8x8_t _r0 = float2int8(vmulq_n_f32(_p00, scale0), vmulq_n_f32(_p01, scale0));
                    int8x8_t _r1 = float2int8(vmulq_n_f32(_p10, scale1), vmulq_n_f32(_p11, scale1));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    pp += 16;
                    p0 += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                    float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                    int8x8_t _r01 = float2int8(vmulq_n_f32(_p0, scale0), vmulq_n_f32(_p1, scale1));
#if __ARM_FEATURE_DOTPROD
                    vst1_s8(pp, _r01);
#else
                    int16x4_t _r01_s16 = vreinterpret_s16_s8(_r01);
                    int16x4_t _r10_s16 = vext_s16(_r01_s16, _r01_s16, 2);
                    vst1_s8(pp, vreinterpret_s8_s16(vzip_s16(_r01_s16, _r10_s16).val[0]));
#endif
                    pp += 8;
                    p0 += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float v00 = bfloat16_to_float32(p0[0]);
                    float v01 = bfloat16_to_float32(p0[1]);
                    float v10 = bfloat16_to_float32(p0[A_hstep]);
                    float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                    *pp++ = float2int8(v00 * scale0);
                    *pp++ = float2int8(v01 * scale0);
                    *pp++ = float2int8(v10 * scale1);
                    *pp++ = float2int8(v11 * scale1);
                    p0 += 2;
                }
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[A_hstep]);
                    *pp++ = float2int8(v0 * scale0);
                    *pp++ = float2int8(v1 * scale1);
                    p0++;
                }

                pd += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const unsigned short* p0a = p0;

                float absmax = 0.f;
                int kk = 0;
#if __ARM_NEON
                float32x4_t _absmax = vdupq_n_f32(0.f);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(p0a));
                    _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
                    p0a += 4;
                }
#if __aarch64__
                absmax = vmaxvq_f32(_absmax);
#else
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                _max2 = vpmax_f32(_max2, _max2);
                absmax = vget_lane_f32(_max2, 0);
#endif
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0a++);
                    absmax = std::max(absmax, fabsf(v));
                }

                if (absmax == 0.f)
                {
                    *pd++ = 0.f;
                    for (int kk0 = 0; kk0 < max_kk0; kk0++)
                        *pp++ = 0;
                    p0 += max_kk0;
                    continue;
                }

                const float scale = 127.f / absmax;
                *pd++ = absmax / 127.f;

                kk = 0;
#if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                    float32x4_t _p1 = bfloat2float(vld1_u16(p0 + 4));
                    vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale)));
                    pp += 8;
                    p0 += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(p0));
                    int8x8_t _r = float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale));
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                    pp += 4;
                    p0 += 4;
                }
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0++);
                    *pp++ = float2int8(v * scale);
                }
            }
        }

        return;
    }

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vmulq_n_f32(vabsq_f32(bfloat2float(vld1_u16(p0a))), psa[0]);
                    float32x4_t _p1 = vmulq_n_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + A_hstep * 4))), psa[0]);
                    _absmax0 = vmaxq_f32(_absmax0, _p0);
                    _absmax1 = vmaxq_f32(_absmax1, _p1);
                    p0a += 4;
                    psa++;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                float absmax[8];
                vst1q_f32(absmax, _absmax0);
                vst1q_f32(absmax + 4, _absmax1);
                const float scale0 = absmax[0] == 0.f ? 0.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 0.f : 127.f / absmax[1];
                const float scale2 = absmax[2] == 0.f ? 0.f : 127.f / absmax[2];
                const float scale3 = absmax[3] == 0.f ? 0.f : 127.f / absmax[3];
                const float scale4 = absmax[4] == 0.f ? 0.f : 127.f / absmax[4];
                const float scale5 = absmax[5] == 0.f ? 0.f : 127.f / absmax[5];
                const float scale6 = absmax[6] == 0.f ? 0.f : 127.f / absmax[6];
                const float scale7 = absmax[7] == 0.f ? 0.f : 127.f / absmax[7];

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    uint16x8x4_t _p = vld4q_u16(p0);
                    uint16x8x4_t _q = vld4q_u16(p0 + A_hstep * 4);
                    int8x8_t _r0 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_p.val[0])), _s0), scale0), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_p.val[0])), _s1), scale0));
                    int8x8_t _r1 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_p.val[1])), _s0), scale1), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_p.val[1])), _s1), scale1));
                    int8x8_t _r2 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_p.val[2])), _s0), scale2), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_p.val[2])), _s1), scale2));
                    int8x8_t _r3 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_p.val[3])), _s0), scale3), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_p.val[3])), _s1), scale3));
                    int8x8_t _r4 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_q.val[0])), _s0), scale4), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_q.val[0])), _s1), scale4));
                    int8x8_t _r5 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_q.val[1])), _s0), scale5), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_q.val[1])), _s1), scale5));
                    int8x8_t _r6 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_q.val[2])), _s0), scale6), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_q.val[2])), _s1), scale6));
                    int8x8_t _r7 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_q.val[3])), _s0), scale7), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_q.val[3])), _s1), scale7));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                    pp += 64;
                    p0 += 32;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    uint16x4x4_t _p = vld4_u16(p0);
                    uint16x4x4_t _q = vld4_u16(p0 + A_hstep * 4);
                    int8x8_t _r01 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[0]), _s), scale0), vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[1]), _s), scale1));
                    int8x8_t _r23 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[2]), _s), scale2), vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[3]), _s), scale3));
                    int8x8_t _r45 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[0]), _s), scale4), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[1]), _s), scale5));
                    int8x8_t _r67 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[2]), _s), scale6), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[3]), _s), scale7));
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
                    vst1q_s8(pp + 16, vcombine_s8(_r45, _r67));
#else
                    int16x8x2_t _r04 = vuzpq_s16(vreinterpretq_s16_s8(vcombine_s8(_r01, _r23)), vreinterpretq_s16_s8(vcombine_s8(_r45, _r67)));
                    vst1q_s16((short*)pp, _r04.val[0]);
                    vst1q_s16((short*)pp + 8, _r04.val[1]);
#endif
                    pp += 32;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                    float32x4_t _p1 = bfloat2float(vld1_u16(p0 + 4));
                    float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                    float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4));
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 0) * ps[0] * scale0);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 0) * ps[1] * scale0);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 1) * ps[0] * scale1);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 1) * ps[1] * scale1);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 2) * ps[0] * scale2);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 2) * ps[1] * scale2);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 3) * ps[0] * scale3);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 3) * ps[1] * scale3);
                    *pp++ = float2int8(vgetq_lane_f32(_p2, 0) * ps[0] * scale4);
                    *pp++ = float2int8(vgetq_lane_f32(_p3, 0) * ps[1] * scale4);
                    *pp++ = float2int8(vgetq_lane_f32(_p2, 1) * ps[0] * scale5);
                    *pp++ = float2int8(vgetq_lane_f32(_p3, 1) * ps[1] * scale5);
                    *pp++ = float2int8(vgetq_lane_f32(_p2, 2) * ps[0] * scale6);
                    *pp++ = float2int8(vgetq_lane_f32(_p3, 2) * ps[1] * scale6);
                    *pp++ = float2int8(vgetq_lane_f32(_p2, 3) * ps[0] * scale7);
                    *pp++ = float2int8(vgetq_lane_f32(_p3, 3) * ps[1] * scale7);
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vmulq_n_f32(bfloat2float(vld1_u16(p0)), ps[0]);
                    float32x4_t _p1 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), ps[0]);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 0) * scale0);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 1) * scale1);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 2) * scale2);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 3) * scale3);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 0) * scale4);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 1) * scale5);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 2) * scale6);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 3) * scale7);
                    p0 += 4;
                    ps++;
                }
                pd += 8;
            }
            if (elempack == 1)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                float32x4_t _absmax2 = vdupq_n_f32(0.f);
                float32x4_t _absmax3 = vdupq_n_f32(0.f);
                float32x4_t _absmax4 = vdupq_n_f32(0.f);
                float32x4_t _absmax5 = vdupq_n_f32(0.f);
                float32x4_t _absmax6 = vdupq_n_f32(0.f);
                float32x4_t _absmax7 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    float32x4_t _p0 = bfloat2float(vld1_u16(p0a));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    float32x4_t _p1 = bfloat2float(vld1_u16(p0a + A_hstep));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    float32x4_t _p2 = bfloat2float(vld1_u16(p0a + A_hstep * 2));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(_p2), _s));
                    float32x4_t _p3 = bfloat2float(vld1_u16(p0a + A_hstep * 3));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(_p3), _s));
                    float32x4_t _p4 = bfloat2float(vld1_u16(p0a + A_hstep * 4));
                    _absmax4 = vmaxq_f32(_absmax4, vmulq_f32(vabsq_f32(_p4), _s));
                    float32x4_t _p5 = bfloat2float(vld1_u16(p0a + A_hstep * 5));
                    _absmax5 = vmaxq_f32(_absmax5, vmulq_f32(vabsq_f32(_p5), _s));
                    float32x4_t _p6 = bfloat2float(vld1_u16(p0a + A_hstep * 6));
                    _absmax6 = vmaxq_f32(_absmax6, vmulq_f32(vabsq_f32(_p6), _s));
                    float32x4_t _p7 = bfloat2float(vld1_u16(p0a + A_hstep * 7));
                    _absmax7 = vmaxq_f32(_absmax7, vmulq_f32(vabsq_f32(_p7), _s));
                    p0a += 4;
                    psa += 4;
                }
                float absmax0 = vmaxvq_f32(_absmax0);
                float absmax1 = vmaxvq_f32(_absmax1);
                float absmax2 = vmaxvq_f32(_absmax2);
                float absmax3 = vmaxvq_f32(_absmax3);
                float absmax4 = vmaxvq_f32(_absmax4);
                float absmax5 = vmaxvq_f32(_absmax5);
                float absmax6 = vmaxvq_f32(_absmax6);
                float absmax7 = vmaxvq_f32(_absmax7);
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0a[0]);
                    float v1 = bfloat16_to_float32(p0a[A_hstep]);
                    float v2 = bfloat16_to_float32(p0a[A_hstep * 2]);
                    float v3 = bfloat16_to_float32(p0a[A_hstep * 3]);
                    float v4 = bfloat16_to_float32(p0a[A_hstep * 4]);
                    float v5 = bfloat16_to_float32(p0a[A_hstep * 5]);
                    float v6 = bfloat16_to_float32(p0a[A_hstep * 6]);
                    float v7 = bfloat16_to_float32(p0a[A_hstep * 7]);
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    absmax1 = std::max(absmax1, fabsf(v1) * s);
                    absmax2 = std::max(absmax2, fabsf(v2) * s);
                    absmax3 = std::max(absmax3, fabsf(v3) * s);
                    absmax4 = std::max(absmax4, fabsf(v4) * s);
                    absmax5 = std::max(absmax5, fabsf(v5) * s);
                    absmax6 = std::max(absmax6, fabsf(v6) * s);
                    absmax7 = std::max(absmax7, fabsf(v7) * s);
                    p0a++;
                }

                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
                const float scale4 = absmax4 == 0.f ? 0.f : 127.f / absmax4;
                const float scale5 = absmax5 == 0.f ? 0.f : 127.f / absmax5;
                const float scale6 = absmax6 == 0.f ? 0.f : 127.f / absmax6;
                const float scale7 = absmax7 == 0.f ? 0.f : 127.f / absmax7;

                kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    float32x4_t _p00 = vmulq_f32(bfloat2float(vld1_u16(p0)), _s0);
                    float32x4_t _p01 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s1);
                    float32x4_t _p10 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _s0);
                    float32x4_t _p11 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep + 4)), _s1);
                    float32x4_t _p20 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), _s0);
                    float32x4_t _p21 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2 + 4)), _s1);
                    float32x4_t _p30 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), _s0);
                    float32x4_t _p31 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3 + 4)), _s1);
                    float32x4_t _p40 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _s0);
                    float32x4_t _p41 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _s1);
                    float32x4_t _p50 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 5)), _s0);
                    float32x4_t _p51 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 5 + 4)), _s1);
                    float32x4_t _p60 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 6)), _s0);
                    float32x4_t _p61 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 6 + 4)), _s1);
                    float32x4_t _p70 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 7)), _s0);
                    float32x4_t _p71 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 7 + 4)), _s1);
                    int8x8_t _r0 = float2int8(vmulq_n_f32(_p00, scale0), vmulq_n_f32(_p01, scale0));
                    int8x8_t _r1 = float2int8(vmulq_n_f32(_p10, scale1), vmulq_n_f32(_p11, scale1));
                    int8x8_t _r2 = float2int8(vmulq_n_f32(_p20, scale2), vmulq_n_f32(_p21, scale2));
                    int8x8_t _r3 = float2int8(vmulq_n_f32(_p30, scale3), vmulq_n_f32(_p31, scale3));
                    int8x8_t _r4 = float2int8(vmulq_n_f32(_p40, scale4), vmulq_n_f32(_p41, scale4));
                    int8x8_t _r5 = float2int8(vmulq_n_f32(_p50, scale5), vmulq_n_f32(_p51, scale5));
                    int8x8_t _r6 = float2int8(vmulq_n_f32(_p60, scale6), vmulq_n_f32(_p61, scale6));
                    int8x8_t _r7 = float2int8(vmulq_n_f32(_p70, scale7), vmulq_n_f32(_p71, scale7));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                    pp += 64;
                    p0 += 8;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _s);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _s);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), _s);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), _s);
                    float32x4_t _p4 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _s);
                    float32x4_t _p5 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 5)), _s);
                    float32x4_t _p6 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 6)), _s);
                    float32x4_t _p7 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 7)), _s);
                    int8x8_t _r01 = float2int8(vmulq_n_f32(_p0, scale0), vmulq_n_f32(_p1, scale1));
                    int8x8_t _r23 = float2int8(vmulq_n_f32(_p2, scale2), vmulq_n_f32(_p3, scale3));
                    int8x8_t _r45 = float2int8(vmulq_n_f32(_p4, scale4), vmulq_n_f32(_p5, scale5));
                    int8x8_t _r67 = float2int8(vmulq_n_f32(_p6, scale6), vmulq_n_f32(_p7, scale7));
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
                    vst1q_s8(pp + 16, vcombine_s8(_r45, _r67));
#else
                    int16x8x2_t _r04 = vuzpq_s16(vreinterpretq_s16_s8(vcombine_s8(_r01, _r23)), vreinterpretq_s16_s8(vcombine_s8(_r45, _r67)));
                    vst1q_s16((short*)pp, _r04.val[0]);
                    vst1q_s16((short*)pp + 8, _r04.val[1]);
#endif
                    pp += 32;
                    p0 += 4;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const float s0 = ps[0];
                    const float s1 = ps[1];
                    float v00 = bfloat16_to_float32(p0[0]);
                    float v01 = bfloat16_to_float32(p0[1]);
                    float v10 = bfloat16_to_float32(p0[A_hstep]);
                    float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                    float v20 = bfloat16_to_float32(p0[A_hstep * 2]);
                    float v21 = bfloat16_to_float32(p0[A_hstep * 2 + 1]);
                    float v30 = bfloat16_to_float32(p0[A_hstep * 3]);
                    float v31 = bfloat16_to_float32(p0[A_hstep * 3 + 1]);
                    float v40 = bfloat16_to_float32(p0[A_hstep * 4]);
                    float v41 = bfloat16_to_float32(p0[A_hstep * 4 + 1]);
                    float v50 = bfloat16_to_float32(p0[A_hstep * 5]);
                    float v51 = bfloat16_to_float32(p0[A_hstep * 5 + 1]);
                    float v60 = bfloat16_to_float32(p0[A_hstep * 6]);
                    float v61 = bfloat16_to_float32(p0[A_hstep * 6 + 1]);
                    float v70 = bfloat16_to_float32(p0[A_hstep * 7]);
                    float v71 = bfloat16_to_float32(p0[A_hstep * 7 + 1]);
                    v00 *= s0;
                    v01 *= s1;
                    v10 *= s0;
                    v11 *= s1;
                    v20 *= s0;
                    v21 *= s1;
                    v30 *= s0;
                    v31 *= s1;
                    v40 *= s0;
                    v41 *= s1;
                    v50 *= s0;
                    v51 *= s1;
                    v60 *= s0;
                    v61 *= s1;
                    v70 *= s0;
                    v71 *= s1;
                    *pp++ = float2int8(v00 * scale0);
                    *pp++ = float2int8(v01 * scale0);
                    *pp++ = float2int8(v10 * scale1);
                    *pp++ = float2int8(v11 * scale1);
                    *pp++ = float2int8(v20 * scale2);
                    *pp++ = float2int8(v21 * scale2);
                    *pp++ = float2int8(v30 * scale3);
                    *pp++ = float2int8(v31 * scale3);
                    *pp++ = float2int8(v40 * scale4);
                    *pp++ = float2int8(v41 * scale4);
                    *pp++ = float2int8(v50 * scale5);
                    *pp++ = float2int8(v51 * scale5);
                    *pp++ = float2int8(v60 * scale6);
                    *pp++ = float2int8(v61 * scale6);
                    *pp++ = float2int8(v70 * scale7);
                    *pp++ = float2int8(v71 * scale7);
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = ps[0];
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[A_hstep]);
                    float v2 = bfloat16_to_float32(p0[A_hstep * 2]);
                    float v3 = bfloat16_to_float32(p0[A_hstep * 3]);
                    float v4 = bfloat16_to_float32(p0[A_hstep * 4]);
                    float v5 = bfloat16_to_float32(p0[A_hstep * 5]);
                    float v6 = bfloat16_to_float32(p0[A_hstep * 6]);
                    float v7 = bfloat16_to_float32(p0[A_hstep * 7]);
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    v4 *= s;
                    v5 *= s;
                    v6 *= s;
                    v7 *= s;
                    *pp++ = float2int8(v0 * scale0);
                    *pp++ = float2int8(v1 * scale1);
                    *pp++ = float2int8(v2 * scale2);
                    *pp++ = float2int8(v3 * scale3);
                    *pp++ = float2int8(v4 * scale4);
                    *pp++ = float2int8(v5 * scale5);
                    *pp++ = float2int8(v6 * scale6);
                    *pp++ = float2int8(v7 * scale7);
                    p0++;
                    ps++;
                }
                pd += 8;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            if (elempack == 4)
            {
                float32x4_t _absmax = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vmulq_n_f32(vabsq_f32(bfloat2float(vld1_u16(p0a))), psa[0]);
                    _absmax = vmaxq_f32(_absmax, _p);
                    p0a += 4;
                    psa++;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax, 1.f / 127.f));

                float absmax[4];
                vst1q_f32(absmax, _absmax);
                const float scale0 = absmax[0] == 0.f ? 0.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 0.f : 127.f / absmax[1];
                const float scale2 = absmax[2] == 0.f ? 0.f : 127.f / absmax[2];
                const float scale3 = absmax[3] == 0.f ? 0.f : 127.f / absmax[3];

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    uint16x8x4_t _p = vld4q_u16(p0);
                    int8x8_t _r0 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_p.val[0])), _s0), scale0), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_p.val[0])), _s1), scale0));
                    int8x8_t _r1 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_p.val[1])), _s0), scale1), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_p.val[1])), _s1), scale1));
                    int8x8_t _r2 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_p.val[2])), _s0), scale2), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_p.val[2])), _s1), scale2));
                    int8x8_t _r3 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(vget_low_u16(_p.val[3])), _s0), scale3), vmulq_n_f32(vmulq_f32(bfloat2float(vget_high_u16(_p.val[3])), _s1), scale3));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    pp += 32;
                    p0 += 32;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    uint16x4x4_t _p = vld4_u16(p0);
                    int8x8_t _r01 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[0]), _s), scale0), vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[1]), _s), scale1));
                    int8x8_t _r23 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[2]), _s), scale2), vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[3]), _s), scale3));
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                    int16x8_t _r0123 = vreinterpretq_s16_s8(vcombine_s8(_r01, _r23));
                    int16x8x2_t _r02 = vuzpq_s16(_r0123, _r0123);
                    vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_r02.val[0]), vget_low_s16(_r02.val[1]))));
#endif
                    pp += 16;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                    float32x4_t _p1 = bfloat2float(vld1_u16(p0 + 4));
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 0) * ps[0] * scale0);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 0) * ps[1] * scale0);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 1) * ps[0] * scale1);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 1) * ps[1] * scale1);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 2) * ps[0] * scale2);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 2) * ps[1] * scale2);
                    *pp++ = float2int8(vgetq_lane_f32(_p0, 3) * ps[0] * scale3);
                    *pp++ = float2int8(vgetq_lane_f32(_p1, 3) * ps[1] * scale3);
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vmulq_n_f32(bfloat2float(vld1_u16(p0)), ps[0]);
                    *pp++ = float2int8(vgetq_lane_f32(_p, 0) * scale0);
                    *pp++ = float2int8(vgetq_lane_f32(_p, 1) * scale1);
                    *pp++ = float2int8(vgetq_lane_f32(_p, 2) * scale2);
                    *pp++ = float2int8(vgetq_lane_f32(_p, 3) * scale3);
                    p0 += 4;
                    ps++;
                }
                pd += 4;
            }
            if (elempack == 1)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                float32x4_t _absmax2 = vdupq_n_f32(0.f);
                float32x4_t _absmax3 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    float32x4_t _p0 = bfloat2float(vld1_u16(p0a));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    float32x4_t _p1 = bfloat2float(vld1_u16(p0a + A_hstep));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    float32x4_t _p2 = bfloat2float(vld1_u16(p0a + A_hstep * 2));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(_p2), _s));
                    float32x4_t _p3 = bfloat2float(vld1_u16(p0a + A_hstep * 3));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(_p3), _s));
                    p0a += 4;
                    psa += 4;
                }
#if __aarch64__
                float absmax0 = vmaxvq_f32(_absmax0);
                float absmax1 = vmaxvq_f32(_absmax1);
                float absmax2 = vmaxvq_f32(_absmax2);
                float absmax3 = vmaxvq_f32(_absmax3);
#else
                float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                _max0 = vpmax_f32(_max0, _max0);
                _max1 = vpmax_f32(_max1, _max1);
                _max2 = vpmax_f32(_max2, _max2);
                _max3 = vpmax_f32(_max3, _max3);
                float absmax0 = vget_lane_f32(_max0, 0);
                float absmax1 = vget_lane_f32(_max1, 0);
                float absmax2 = vget_lane_f32(_max2, 0);
                float absmax3 = vget_lane_f32(_max3, 0);
#endif
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0a[0]);
                    float v1 = bfloat16_to_float32(p0a[A_hstep]);
                    float v2 = bfloat16_to_float32(p0a[A_hstep * 2]);
                    float v3 = bfloat16_to_float32(p0a[A_hstep * 3]);
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    absmax1 = std::max(absmax1, fabsf(v1) * s);
                    absmax2 = std::max(absmax2, fabsf(v2) * s);
                    absmax3 = std::max(absmax3, fabsf(v3) * s);
                    p0a++;
                }
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;

                kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    float32x4_t _p00 = vmulq_f32(bfloat2float(vld1_u16(p0)), _s0);
                    float32x4_t _p01 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s1);
                    float32x4_t _p10 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _s0);
                    float32x4_t _p11 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep + 4)), _s1);
                    float32x4_t _p20 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), _s0);
                    float32x4_t _p21 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2 + 4)), _s1);
                    float32x4_t _p30 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), _s0);
                    float32x4_t _p31 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3 + 4)), _s1);
                    int8x8_t _r0 = float2int8(vmulq_n_f32(_p00, scale0), vmulq_n_f32(_p01, scale0));
                    int8x8_t _r1 = float2int8(vmulq_n_f32(_p10, scale1), vmulq_n_f32(_p11, scale1));
                    int8x8_t _r2 = float2int8(vmulq_n_f32(_p20, scale2), vmulq_n_f32(_p21, scale2));
                    int8x8_t _r3 = float2int8(vmulq_n_f32(_p30, scale3), vmulq_n_f32(_p31, scale3));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    pp += 32;
                    p0 += 8;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _s);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _s);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), _s);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), _s);
                    int8x8_t _r01 = float2int8(vmulq_n_f32(_p0, scale0), vmulq_n_f32(_p1, scale1));
                    int8x8_t _r23 = float2int8(vmulq_n_f32(_p2, scale2), vmulq_n_f32(_p3, scale3));
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                    int16x8_t _r0123 = vreinterpretq_s16_s8(vcombine_s8(_r01, _r23));
                    int16x8x2_t _r02 = vuzpq_s16(_r0123, _r0123);
                    vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_r02.val[0]), vget_low_s16(_r02.val[1]))));
#endif
                    pp += 16;
                    p0 += 4;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const float s0 = ps[0];
                    const float s1 = ps[1];
                    float v00 = bfloat16_to_float32(p0[0]);
                    float v01 = bfloat16_to_float32(p0[1]);
                    float v10 = bfloat16_to_float32(p0[A_hstep]);
                    float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                    float v20 = bfloat16_to_float32(p0[A_hstep * 2]);
                    float v21 = bfloat16_to_float32(p0[A_hstep * 2 + 1]);
                    float v30 = bfloat16_to_float32(p0[A_hstep * 3]);
                    float v31 = bfloat16_to_float32(p0[A_hstep * 3 + 1]);
                    v00 *= s0;
                    v01 *= s1;
                    v10 *= s0;
                    v11 *= s1;
                    v20 *= s0;
                    v21 *= s1;
                    v30 *= s0;
                    v31 *= s1;
                    *pp++ = float2int8(v00 * scale0);
                    *pp++ = float2int8(v01 * scale0);
                    *pp++ = float2int8(v10 * scale1);
                    *pp++ = float2int8(v11 * scale1);
                    *pp++ = float2int8(v20 * scale2);
                    *pp++ = float2int8(v21 * scale2);
                    *pp++ = float2int8(v30 * scale3);
                    *pp++ = float2int8(v31 * scale3);
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = ps[0];
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[A_hstep]);
                    float v2 = bfloat16_to_float32(p0[A_hstep * 2]);
                    float v3 = bfloat16_to_float32(p0[A_hstep * 3]);
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    *pp++ = float2int8(v0 * scale0);
                    *pp++ = float2int8(v1 * scale1);
                    *pp++ = float2int8(v2 * scale2);
                    *pp++ = float2int8(v3 * scale3);
                    p0++;
                    ps++;
                }
                pd += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const unsigned short* p0a = p0;
            const float* psa = ps;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _s = vld1q_f32(psa);
                float32x4_t _p0 = bfloat2float(vld1_u16(p0a));
                _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0a + A_hstep));
                _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                p0a += 4;
                psa += 4;
            }
#if __aarch64__
            absmax0 = vmaxvq_f32(_absmax0);
            absmax1 = vmaxvq_f32(_absmax1);
#else
            float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
            float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
            _max0 = vpmax_f32(_max0, _max0);
            _max1 = vpmax_f32(_max1, _max1);
            absmax0 = vget_lane_f32(_max0, 0);
            absmax1 = vget_lane_f32(_max1, 0);
#endif
#endif // __ARM_NEON

            for (; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(p0a[0]);
                float v1 = bfloat16_to_float32(p0a[A_hstep]);
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(v0) * s);
                absmax1 = std::max(absmax1, fabsf(v1) * s);
                p0a++;
            }

            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;

            kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk0; kk += 8)
            {
                float32x4_t _s0 = vld1q_f32(ps);
                float32x4_t _s1 = vld1q_f32(ps + 4);
                float32x4_t _p00 = vmulq_f32(bfloat2float(vld1_u16(p0)), _s0);
                float32x4_t _p01 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s1);
                float32x4_t _p10 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _s0);
                float32x4_t _p11 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep + 4)), _s1);
                int8x8_t _r0 = float2int8(vmulq_n_f32(_p00, scale0), vmulq_n_f32(_p01, scale0));
                int8x8_t _r1 = float2int8(vmulq_n_f32(_p10, scale1), vmulq_n_f32(_p11, scale1));
                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                pp += 16;
                p0 += 8;
                ps += 8;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _s = vld1q_f32(ps);
                float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _s);
                float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _s);
                int8x8_t _r01 = float2int8(vmulq_n_f32(_p0, scale0), vmulq_n_f32(_p1, scale1));
#if __ARM_FEATURE_DOTPROD
                vst1_s8(pp, _r01);
#else
                int16x4_t _r01_s16 = vreinterpret_s16_s8(_r01);
                int16x4_t _r10_s16 = vext_s16(_r01_s16, _r01_s16, 2);
                vst1_s8(pp, vreinterpret_s8_s16(vzip_s16(_r01_s16, _r10_s16).val[0]));
#endif
                pp += 8;
                p0 += 4;
                ps += 4;
            }
            for (; kk + 1 < max_kk0; kk += 2)
            {
                float v00 = bfloat16_to_float32(p0[0]);
                float v01 = bfloat16_to_float32(p0[1]);
                float v10 = bfloat16_to_float32(p0[A_hstep]);
                float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                v00 *= ps[0];
                v01 *= ps[1];
                v10 *= ps[0];
                v11 *= ps[1];
                *pp++ = float2int8(v00 * scale0);
                *pp++ = float2int8(v01 * scale0);
                *pp++ = float2int8(v10 * scale1);
                *pp++ = float2int8(v11 * scale1);
                p0 += 2;
                ps += 2;
            }
#endif // __ARM_NEON
            for (; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(p0[0]);
                float v1 = bfloat16_to_float32(p0[A_hstep]);
                v0 *= ps[0];
                v1 *= ps[0];
                *pp++ = float2int8(v0 * scale0);
                *pp++ = float2int8(v1 * scale1);
                p0++;
                ps++;
            }

            pd += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            const unsigned short* p0a = p0;
            const float* psa = ps;

            float absmax = 0.f;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(p0a));
                float32x4_t _s = vld1q_f32(psa);
                _absmax = vmaxq_f32(_absmax, vmulq_f32(vabsq_f32(_p), _s));
                p0a += 4;
                psa += 4;
            }
#if __aarch64__
            absmax = vmaxvq_f32(_absmax);
#else
            float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
            _max2 = vpmax_f32(_max2, _max2);
            absmax = vget_lane_f32(_max2, 0);
#endif
#endif // __ARM_NEON
            for (; kk < max_kk0; kk++)
            {
                float v = bfloat16_to_float32(*p0a++);
                absmax = std::max(absmax, fabsf(v) * *psa++);
            }

            if (absmax == 0.f)
            {
                *pd++ = 0.f;
                for (int kk0 = 0; kk0 < max_kk0; kk0++)
                    *pp++ = 0;
                p0 += max_kk0;
                ps += max_kk0;
                continue;
            }

            const float scale = 127.f / absmax;
            *pd++ = absmax / 127.f;

            kk = 0;
#if __ARM_NEON
            float32x4_t _scale = vdupq_n_f32(scale);
            for (; kk + 7 < max_kk0; kk += 8)
            {
                float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), vld1q_f32(ps));
                float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), vld1q_f32(ps + 4));
                vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale)));
                pp += 8;
                p0 += 8;
                ps += 8;
            }
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _p = vmulq_f32(bfloat2float(vld1_u16(p0)), vld1q_f32(ps));
                int8x8_t _r = float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale));
                vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                pp += 4;
                p0 += 4;
                ps += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk0; kk++)
            {
                float v = bfloat16_to_float32(*p0++);
                v *= *ps++;
                *pp++ = float2int8(v * scale);
            }
        }
    }
}

static void transpose_quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_quantize_A_tile_wq_int8_bf16s_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_quantize_A_tile_wq_int8_bf16s_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (!input_scales.empty())
    {
        int ii = 0;
#if __ARM_NEON
#if __aarch64__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
            const float* ps = (const float*)input_scales + k;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    float32x4_t _absmax4 = vdupq_n_f32(0.f);
                    float32x4_t _absmax5 = vdupq_n_f32(0.f);
                    float32x4_t _absmax6 = vdupq_n_f32(0.f);
                    float32x4_t _absmax7 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _s = vld1q_f32(psa);
                        _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a))), _s));
                        _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 4))), _s));
                        _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 8))), _s));
                        _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 12))), _s));
                        _absmax4 = vmaxq_f32(_absmax4, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 16))), _s));
                        _absmax5 = vmaxq_f32(_absmax5, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 20))), _s));
                        _absmax6 = vmaxq_f32(_absmax6, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 24))), _s));
                        _absmax7 = vmaxq_f32(_absmax7, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 28))), _s));
                        p0a += A_hstep * 4;
                        psa += 4;
                    }

                    float absmax0 = vmaxvq_f32(_absmax0);
                    float absmax1 = vmaxvq_f32(_absmax1);
                    float absmax2 = vmaxvq_f32(_absmax2);
                    float absmax3 = vmaxvq_f32(_absmax3);
                    float absmax4 = vmaxvq_f32(_absmax4);
                    float absmax5 = vmaxvq_f32(_absmax5);
                    float absmax6 = vmaxvq_f32(_absmax6);
                    float absmax7 = vmaxvq_f32(_absmax7);
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    pd[4] = absmax4 / 127.f;
                    pd[5] = absmax5 / 127.f;
                    pd[6] = absmax6 / 127.f;
                    pd[7] = absmax7 / 127.f;

                    float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    _scale0 = vsetq_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale0, 1);
                    _scale0 = vsetq_lane_f32(absmax2 == 0.f ? 0.f : 127.f / absmax2, _scale0, 2);
                    _scale0 = vsetq_lane_f32(absmax3 == 0.f ? 0.f : 127.f / absmax3, _scale0, 3);
                    float32x4_t _scale1 = vdupq_n_f32(absmax4 == 0.f ? 0.f : 127.f / absmax4);
                    _scale1 = vsetq_lane_f32(absmax5 == 0.f ? 0.f : 127.f / absmax5, _scale1, 1);
                    _scale1 = vsetq_lane_f32(absmax6 == 0.f ? 0.f : 127.f / absmax6, _scale1, 2);
                    _scale1 = vsetq_lane_f32(absmax7 == 0.f ? 0.f : 127.f / absmax7, _scale1, 3);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        uint16x4x4_t _q = vld4_u16(p0 + 16);
                        uint16x4x4_t _r = vld4_u16(p0 + A_hstep * 4);
                        uint16x4x4_t _s = vld4_u16(p0 + A_hstep * 4 + 16);
                        int8x8_t _r0 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[0]), _scale0), ps[0]), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[0]), _scale1), ps[0]));
                        int8x8_t _r1 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[1]), _scale0), ps[1]), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[1]), _scale1), ps[1]));
                        int8x8_t _r2 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[2]), _scale0), ps[2]), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[2]), _scale1), ps[2]));
                        int8x8_t _r3 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[3]), _scale0), ps[3]), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[3]), _scale1), ps[3]));
                        int8x8_t _r4 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_r.val[0]), _scale0), ps[4]), vmulq_n_f32(vmulq_f32(bfloat2float(_s.val[0]), _scale1), ps[4]));
                        int8x8_t _r5 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_r.val[1]), _scale0), ps[5]), vmulq_n_f32(vmulq_f32(bfloat2float(_s.val[1]), _scale1), ps[5]));
                        int8x8_t _r6 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_r.val[2]), _scale0), ps[6]), vmulq_n_f32(vmulq_f32(bfloat2float(_s.val[2]), _scale1), ps[6]));
                        int8x8_t _r7 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_r.val[3]), _scale0), ps[7]), vmulq_n_f32(vmulq_f32(bfloat2float(_s.val[3]), _scale1), ps[7]));
                        int8x8x2_t _r04 = vzip_s8(_r0, _r4);
                        int8x8x2_t _r15 = vzip_s8(_r1, _r5);
                        int8x8x2_t _r26 = vzip_s8(_r2, _r6);
                        int8x8x2_t _r37 = vzip_s8(_r3, _r7);
                        int8x8x4_t _r0123;
                        _r0123.val[0] = _r04.val[0];
                        _r0123.val[1] = _r15.val[0];
                        _r0123.val[2] = _r26.val[0];
                        _r0123.val[3] = _r37.val[0];
                        int8x8x4_t _r4567;
                        _r4567.val[0] = _r04.val[1];
                        _r4567.val[1] = _r15.val[1];
                        _r4567.val[2] = _r26.val[1];
                        _r4567.val[3] = _r37.val[1];
                        vst4_s8(pp, _r0123);
                        vst4_s8(pp + 32, _r4567);
                        pp += 64;
                        p0 += A_hstep * 8;
                        ps += 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        uint16x4x4_t _q = vld4_u16(p0 + 16);
                        int8x8_t _r0 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[0]), _scale0), ps[0]), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[0]), _scale1), ps[0]));
                        int8x8_t _r1 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[1]), _scale0), ps[1]), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[1]), _scale1), ps[1]));
                        int8x8_t _r2 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[2]), _scale0), ps[2]), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[2]), _scale1), ps[2]));
                        int8x8_t _r3 = float2int8(vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[3]), _scale0), ps[3]), vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[3]), _scale1), ps[3]));
#if __ARM_FEATURE_DOTPROD
                        int8x8x4_t _r0123;
                        _r0123.val[0] = _r0;
                        _r0123.val[1] = _r1;
                        _r0123.val[2] = _r2;
                        _r0123.val[3] = _r3;
                        vst4_s8(pp, _r0123);
#else
                        int8x8x2_t _r01;
                        _r01.val[0] = _r0;
                        _r01.val[1] = _r1;
                        int8x8x2_t _r23;
                        _r23.val[0] = _r2;
                        _r23.val[1] = _r3;
                        vst2_s8(pp, _r01);
                        vst2_s8(pp + 16, _r23);
#endif
                        pp += 32;
                        p0 += A_hstep * 4;
                        ps += 4;
                    }
                    pd += 8;
                }
                if (elempack == 1)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        const float s = *psa++;
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0a));
                        _absmax0 = vmaxq_f32(_absmax0, vmulq_n_f32(vabsq_f32(_p0), s));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0a + 4));
                        _absmax1 = vmaxq_f32(_absmax1, vmulq_n_f32(vabsq_f32(_p1), s));
                        p0a += A_hstep;
                    }

                    float absmax0 = vgetq_lane_f32(_absmax0, 0);
                    float absmax1 = vgetq_lane_f32(_absmax0, 1);
                    float absmax2 = vgetq_lane_f32(_absmax0, 2);
                    float absmax3 = vgetq_lane_f32(_absmax0, 3);
                    float absmax4 = vgetq_lane_f32(_absmax1, 0);
                    float absmax5 = vgetq_lane_f32(_absmax1, 1);
                    float absmax6 = vgetq_lane_f32(_absmax1, 2);
                    float absmax7 = vgetq_lane_f32(_absmax1, 3);
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    pd[4] = absmax4 / 127.f;
                    pd[5] = absmax5 / 127.f;
                    pd[6] = absmax6 / 127.f;
                    pd[7] = absmax7 / 127.f;

                    float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    _scale0 = vsetq_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale0, 1);
                    _scale0 = vsetq_lane_f32(absmax2 == 0.f ? 0.f : 127.f / absmax2, _scale0, 2);
                    _scale0 = vsetq_lane_f32(absmax3 == 0.f ? 0.f : 127.f / absmax3, _scale0, 3);
                    float32x4_t _scale1 = vdupq_n_f32(absmax4 == 0.f ? 0.f : 127.f / absmax4);
                    _scale1 = vsetq_lane_f32(absmax5 == 0.f ? 0.f : 127.f / absmax5, _scale1, 1);
                    _scale1 = vsetq_lane_f32(absmax6 == 0.f ? 0.f : 127.f / absmax6, _scale1, 2);
                    _scale1 = vsetq_lane_f32(absmax7 == 0.f ? 0.f : 127.f / absmax7, _scale1, 3);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p01 = bfloat2float(vld1_u16(p0 + 4));
                        const float s0 = *ps++;
                        _p00 = vmulq_n_f32(_p00, s0);
                        _p01 = vmulq_n_f32(_p01, s0);
                        float32x4_t _p10 = bfloat2float(vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = bfloat2float(vld1_u16(p0 + A_hstep + 4));
                        const float s1 = *ps++;
                        _p10 = vmulq_n_f32(_p10, s1);
                        _p11 = vmulq_n_f32(_p11, s1);
                        float32x4_t _p20 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p21 = bfloat2float(vld1_u16(p0 + A_hstep * 2 + 4));
                        const float s2 = *ps++;
                        _p20 = vmulq_n_f32(_p20, s2);
                        _p21 = vmulq_n_f32(_p21, s2);
                        float32x4_t _p30 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p31 = bfloat2float(vld1_u16(p0 + A_hstep * 3 + 4));
                        const float s3 = *ps++;
                        _p30 = vmulq_n_f32(_p30, s3);
                        _p31 = vmulq_n_f32(_p31, s3);
                        float32x4_t _p40 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                        float32x4_t _p41 = bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4));
                        const float s4 = *ps++;
                        _p40 = vmulq_n_f32(_p40, s4);
                        _p41 = vmulq_n_f32(_p41, s4);
                        float32x4_t _p50 = bfloat2float(vld1_u16(p0 + A_hstep * 5));
                        float32x4_t _p51 = bfloat2float(vld1_u16(p0 + A_hstep * 5 + 4));
                        const float s5 = *ps++;
                        _p50 = vmulq_n_f32(_p50, s5);
                        _p51 = vmulq_n_f32(_p51, s5);
                        float32x4_t _p60 = bfloat2float(vld1_u16(p0 + A_hstep * 6));
                        float32x4_t _p61 = bfloat2float(vld1_u16(p0 + A_hstep * 6 + 4));
                        const float s6 = *ps++;
                        _p60 = vmulq_n_f32(_p60, s6);
                        _p61 = vmulq_n_f32(_p61, s6);
                        float32x4_t _p70 = bfloat2float(vld1_u16(p0 + A_hstep * 7));
                        float32x4_t _p71 = bfloat2float(vld1_u16(p0 + A_hstep * 7 + 4));
                        const float s7 = *ps++;
                        _p70 = vmulq_n_f32(_p70, s7);
                        _p71 = vmulq_n_f32(_p71, s7);
                        int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                        int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                        int8x8_t _r2 = float2int8(vmulq_f32(_p20, _scale0), vmulq_f32(_p21, _scale1));
                        int8x8_t _r3 = float2int8(vmulq_f32(_p30, _scale0), vmulq_f32(_p31, _scale1));
                        int8x8_t _r4 = float2int8(vmulq_f32(_p40, _scale0), vmulq_f32(_p41, _scale1));
                        int8x8_t _r5 = float2int8(vmulq_f32(_p50, _scale0), vmulq_f32(_p51, _scale1));
                        int8x8_t _r6 = float2int8(vmulq_f32(_p60, _scale0), vmulq_f32(_p61, _scale1));
                        int8x8_t _r7 = float2int8(vmulq_f32(_p70, _scale0), vmulq_f32(_p71, _scale1));
                        int8x8x2_t _r04 = vzip_s8(_r0, _r4);
                        int8x8x2_t _r15 = vzip_s8(_r1, _r5);
                        int8x8x2_t _r26 = vzip_s8(_r2, _r6);
                        int8x8x2_t _r37 = vzip_s8(_r3, _r7);
                        int8x8x4_t _r0123;
                        _r0123.val[0] = _r04.val[0];
                        _r0123.val[1] = _r15.val[0];
                        _r0123.val[2] = _r26.val[0];
                        _r0123.val[3] = _r37.val[0];
                        int8x8x4_t _r4567;
                        _r4567.val[0] = _r04.val[1];
                        _r4567.val[1] = _r15.val[1];
                        _r4567.val[2] = _r26.val[1];
                        _r4567.val[3] = _r37.val[1];
                        vst4_s8(pp, _r0123);
                        vst4_s8(pp + 32, _r4567);
                        pp += 64;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p00 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p01 = bfloat2float(vld1_u16(p0 + 4));
                        const float s0 = *ps++;
                        _p00 = vmulq_n_f32(_p00, s0);
                        _p01 = vmulq_n_f32(_p01, s0);
                        float32x4_t _p10 = bfloat2float(vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = bfloat2float(vld1_u16(p0 + A_hstep + 4));
                        const float s1 = *ps++;
                        _p10 = vmulq_n_f32(_p10, s1);
                        _p11 = vmulq_n_f32(_p11, s1);
                        float32x4_t _p20 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p21 = bfloat2float(vld1_u16(p0 + A_hstep * 2 + 4));
                        const float s2 = *ps++;
                        _p20 = vmulq_n_f32(_p20, s2);
                        _p21 = vmulq_n_f32(_p21, s2);
                        float32x4_t _p30 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p31 = bfloat2float(vld1_u16(p0 + A_hstep * 3 + 4));
                        const float s3 = *ps++;
                        _p30 = vmulq_n_f32(_p30, s3);
                        _p31 = vmulq_n_f32(_p31, s3);
                        int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                        int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                        int8x8_t _r2 = float2int8(vmulq_f32(_p20, _scale0), vmulq_f32(_p21, _scale1));
                        int8x8_t _r3 = float2int8(vmulq_f32(_p30, _scale0), vmulq_f32(_p31, _scale1));
#if __ARM_FEATURE_DOTPROD
                        int8x8x4_t _r0123;
                        _r0123.val[0] = _r0;
                        _r0123.val[1] = _r1;
                        _r0123.val[2] = _r2;
                        _r0123.val[3] = _r3;
                        vst4_s8(pp, _r0123);
#else
                        int8x8x2_t _r01;
                        _r01.val[0] = _r0;
                        _r01.val[1] = _r1;
                        int8x8x2_t _r23;
                        _r23.val[0] = _r2;
                        _r23.val[1] = _r3;
                        vst2_s8(pp, _r01);
                        vst2_s8(pp + 16, _r23);
#endif
                        pp += 32;
                        p0 += A_hstep * 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float32x4_t _p00 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p01 = bfloat2float(vld1_u16(p0 + 4));
                        const float s0 = *ps++;
                        _p00 = vmulq_n_f32(_p00, s0);
                        _p01 = vmulq_n_f32(_p01, s0);
                        float32x4_t _p10 = bfloat2float(vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = bfloat2float(vld1_u16(p0 + A_hstep + 4));
                        const float s1 = *ps++;
                        _p10 = vmulq_n_f32(_p10, s1);
                        _p11 = vmulq_n_f32(_p11, s1);
                        int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                        int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                        int8x8x2_t _r01;
                        _r01.val[0] = _r0;
                        _r01.val[1] = _r1;
                        vst2_s8(pp, _r01);
                        pp += 16;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                        float32x4_t _p1 = bfloat2float(vld1_u16(p0 + 4));
                        const float s = *ps++;
                        _p0 = vmulq_n_f32(_p0, s);
                        _p1 = vmulq_n_f32(_p1, s);
                        vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale0), vmulq_f32(_p1, _scale1)));
                        pp += 8;
                        p0 += A_hstep;
                    }
                    pd += 8;
                }
            }
        }
#endif // __aarch64__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
            const float* ps = (const float*)input_scales + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _s = vld1q_f32(psa);
                        _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a))), _s));
                        _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 4))), _s));
                        _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 8))), _s));
                        _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 12))), _s));
                        p0a += A_hstep * 4;
                        psa += 4;
                    }

#if __aarch64__
                    float absmax0 = vmaxvq_f32(_absmax0);
                    float absmax1 = vmaxvq_f32(_absmax1);
                    float absmax2 = vmaxvq_f32(_absmax2);
                    float absmax3 = vmaxvq_f32(_absmax3);
#else
                    float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    _max0 = vpmax_f32(_max0, _max0);
                    _max1 = vpmax_f32(_max1, _max1);
                    _max2 = vpmax_f32(_max2, _max2);
                    _max3 = vpmax_f32(_max3, _max3);
                    float absmax0 = vget_lane_f32(_max0, 0);
                    float absmax1 = vget_lane_f32(_max1, 0);
                    float absmax2 = vget_lane_f32(_max2, 0);
                    float absmax3 = vget_lane_f32(_max3, 0);
#endif
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    float32x4_t _scale = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    _scale = vsetq_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale, 1);
                    _scale = vsetq_lane_f32(absmax2 == 0.f ? 0.f : 127.f / absmax2, _scale, 2);
                    _scale = vsetq_lane_f32(absmax3 == 0.f ? 0.f : 127.f / absmax3, _scale, 3);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        uint16x4x4_t _q = vld4_u16(p0 + A_hstep * 4);
                        float32x4_t _p0 = vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[0]), _scale), ps[0]);
                        float32x4_t _p1 = vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[1]), _scale), ps[1]);
                        float32x4_t _p2 = vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[2]), _scale), ps[2]);
                        float32x4_t _p3 = vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[3]), _scale), ps[3]);
                        float32x4_t _p4 = vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[0]), _scale), ps[4]);
                        float32x4_t _p5 = vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[1]), _scale), ps[5]);
                        float32x4_t _p6 = vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[2]), _scale), ps[6]);
                        float32x4_t _p7 = vmulq_n_f32(vmulq_f32(bfloat2float(_q.val[3]), _scale), ps[7]);
                        float32x4x2_t _p04 = vzipq_f32(_p0, _p4);
                        float32x4x2_t _p15 = vzipq_f32(_p1, _p5);
                        float32x4x2_t _p26 = vzipq_f32(_p2, _p6);
                        float32x4x2_t _p37 = vzipq_f32(_p3, _p7);
                        int8x8x4_t _r0123;
                        _r0123.val[0] = float2int8(_p04.val[0], _p04.val[1]);
                        _r0123.val[1] = float2int8(_p15.val[0], _p15.val[1]);
                        _r0123.val[2] = float2int8(_p26.val[0], _p26.val[1]);
                        _r0123.val[3] = float2int8(_p37.val[0], _p37.val[1]);
                        vst4_s8(pp, _r0123);
                        pp += 32;
                        p0 += A_hstep * 8;
                        ps += 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        float32x4_t _p0 = vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[0]), _scale), ps[0]);
                        float32x4_t _p1 = vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[1]), _scale), ps[1]);
                        float32x4_t _p2 = vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[2]), _scale), ps[2]);
                        float32x4_t _p3 = vmulq_n_f32(vmulq_f32(bfloat2float(_p.val[3]), _scale), ps[3]);
#if __ARM_FEATURE_DOTPROD
                        transpose4x4_ps(_p0, _p1, _p2, _p3);
                        int8x8_t _r01 = float2int8(_p0, _p1);
                        int8x8_t _r23 = float2int8(_p2, _p3);
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                        int8x8_t _r01 = float2int8(_p0, _p1);
                        int8x8_t _r23 = float2int8(_p2, _p3);
                        int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                        int8x8_t _r32 = vext_s8(_r23, _r23, 4);
                        vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                        vst1_s8(pp + 8, vzip_s8(_r23, _r32).val[0]);
#endif
                        pp += 16;
                        p0 += A_hstep * 4;
                        ps += 4;
                    }
                    pd += 4;
                }
                if (elempack == 1)
                {
                    float32x4_t _absmax = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = bfloat2float(vld1_u16(p0a));
                        _absmax = vmaxq_f32(_absmax, vmulq_n_f32(vabsq_f32(_p), *psa++));
                        p0a += A_hstep;
                    }

                    vst1q_f32(pd, vmulq_n_f32(_absmax, 1.f / 127.f));
                    float absmax0 = vgetq_lane_f32(_absmax, 0);
                    float absmax1 = vgetq_lane_f32(_absmax, 1);
                    float absmax2 = vgetq_lane_f32(_absmax, 2);
                    float absmax3 = vgetq_lane_f32(_absmax, 3);
                    float32x4_t _scale = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    _scale = vsetq_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale, 1);
                    _scale = vsetq_lane_f32(absmax2 == 0.f ? 0.f : 127.f / absmax2, _scale, 2);
                    _scale = vsetq_lane_f32(absmax3 == 0.f ? 0.f : 127.f / absmax3, _scale, 3);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p0 = vmulq_n_f32(bfloat2float(vld1_u16(p0)), *ps++);
                        float32x4_t _p1 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep)), *ps++);
                        float32x4_t _p2 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), *ps++);
                        float32x4_t _p3 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), *ps++);
                        float32x4_t _p4 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), *ps++);
                        float32x4_t _p5 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 5)), *ps++);
                        float32x4_t _p6 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 6)), *ps++);
                        float32x4_t _p7 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 7)), *ps++);
                        _p0 = vmulq_f32(_p0, _scale);
                        _p1 = vmulq_f32(_p1, _scale);
                        _p2 = vmulq_f32(_p2, _scale);
                        _p3 = vmulq_f32(_p3, _scale);
                        _p4 = vmulq_f32(_p4, _scale);
                        _p5 = vmulq_f32(_p5, _scale);
                        _p6 = vmulq_f32(_p6, _scale);
                        _p7 = vmulq_f32(_p7, _scale);
                        float32x4x2_t _p04 = vzipq_f32(_p0, _p4);
                        float32x4x2_t _p15 = vzipq_f32(_p1, _p5);
                        float32x4x2_t _p26 = vzipq_f32(_p2, _p6);
                        float32x4x2_t _p37 = vzipq_f32(_p3, _p7);
                        int8x8x4_t _r0123;
                        _r0123.val[0] = float2int8(_p04.val[0], _p04.val[1]);
                        _r0123.val[1] = float2int8(_p15.val[0], _p15.val[1]);
                        _r0123.val[2] = float2int8(_p26.val[0], _p26.val[1]);
                        _r0123.val[3] = float2int8(_p37.val[0], _p37.val[1]);
                        vst4_s8(pp, _r0123);
                        pp += 32;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vmulq_n_f32(bfloat2float(vld1_u16(p0)), *ps++);
                        float32x4_t _p1 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep)), *ps++);
                        float32x4_t _p2 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), *ps++);
                        float32x4_t _p3 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), *ps++);
                        _p0 = vmulq_f32(_p0, _scale);
                        _p1 = vmulq_f32(_p1, _scale);
                        _p2 = vmulq_f32(_p2, _scale);
                        _p3 = vmulq_f32(_p3, _scale);
#if __ARM_FEATURE_DOTPROD
                        transpose4x4_ps(_p0, _p1, _p2, _p3);
                        int8x8_t _r01 = float2int8(_p0, _p1);
                        int8x8_t _r23 = float2int8(_p2, _p3);
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                        int8x8_t _r01 = float2int8(_p0, _p1);
                        int8x8_t _r23 = float2int8(_p2, _p3);
                        int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                        int8x8_t _r32 = vext_s8(_r23, _r23, 4);
                        vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                        vst1_s8(pp + 8, vzip_s8(_r23, _r32).val[0]);
#endif
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float32x4_t _p0 = vmulq_n_f32(bfloat2float(vld1_u16(p0)), *ps++);
                        float32x4_t _p1 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep)), *ps++);
                        int8x8_t _r01 = float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale));
                        int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                        vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                        pp += 8;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vmulq_n_f32(bfloat2float(vld1_u16(p0)), *ps++);
                        vst1_lane_s32((int*)pp, vreinterpret_s32_s8(float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale))), 0);
                        pp += 4;
                        p0 += A_hstep;
                    }
                    pd += 4;
                }
            }
        }
#endif // __ARM_NEON
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
            const float* ps = (const float*)input_scales + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
#if __ARM_NEON
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _s = vld1q_f32(psa);
                        _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a))), _s));
                        _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a + 4))), _s));
                        p0a += A_hstep * 4;
                        psa += 4;
                    }

#if __aarch64__
                    float absmax0 = vmaxvq_f32(_absmax0);
                    float absmax1 = vmaxvq_f32(_absmax1);
#else
                    float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    _max0 = vpmax_f32(_max0, _max0);
                    _max1 = vpmax_f32(_max1, _max1);
                    float absmax0 = vget_lane_f32(_max0, 0);
                    float absmax1 = vget_lane_f32(_max1, 0);
#endif
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    float32x4_t _scale1 = vdupq_n_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1);

                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _s0 = vld1q_f32(ps);
                        float32x4_t _s1 = vld1q_f32(ps + 4);
                        float32x4_t _p0 = vmulq_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _s0), _scale0);
                        float32x4_t _p1 = vmulq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s0), _scale1);
                        float32x4_t _p2 = vmulq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _s1), _scale0);
                        float32x4_t _p3 = vmulq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _s1), _scale1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p2);
                        int8x8_t _r1 = float2int8(_p1, _p3);
#else
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
#endif
#else
                        int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                        int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                        int16x4x2_t _t01 = vzip_s16(_t0, _t1);
                        int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                        int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));

                        pp += 16;
                        p0 += A_hstep * 8;
                        ps += 8;
                    }
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _s = vld1q_f32(ps);
                        float32x4_t _p0 = vmulq_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _s), _scale0);
                        float32x4_t _p1 = vmulq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s), _scale1);

#if __ARM_FEATURE_DOTPROD
                        int8x8_t _r01 = float2int8(_p0, _p1);
#else
                        float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                        float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));
                        int8x8_t _r01 = float2int8(_t0, _t1);
#endif

                        vst1_s8(pp, _r01);

                        pp += 8;
                        p0 += A_hstep * 4;
                        ps += 4;
                    }
                    pd += 2;
                }
#endif // __ARM_NEON
                if (elempack == 1)
                {
#if __ARM_NEON
                    float32x2_t _absmax = vdup_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x2_t _p = vget_low_f32(bfloat2float(vld1_u16(p0a)));
                        _absmax = vmax_f32(_absmax, vmul_n_f32(vabs_f32(_p), *psa++));
                        p0a += A_hstep;
                    }

                    vst1_f32(pd, vmul_n_f32(_absmax, 1.f / 127.f));
                    float absmax0 = vget_lane_f32(_absmax, 0);
                    float absmax1 = vget_lane_f32(_absmax, 1);
                    float32x2_t _scale = vdup_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    _scale = vset_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale, 1);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x2_t _p0 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0))), *ps++);
                        float32x2_t _p1 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep))), *ps++);
                        float32x2_t _p2 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2))), *ps++);
                        float32x2_t _p3 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3))), *ps++);
                        float32x2_t _p4 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4))), *ps++);
                        float32x2_t _p5 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep * 5))), *ps++);
                        float32x2_t _p6 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep * 6))), *ps++);
                        float32x2_t _p7 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep * 7))), *ps++);
                        float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                        float32x4_t _p23 = vmulq_f32(vcombine_f32(_p2, _p3), _scale_scale);
                        float32x4_t _p45 = vmulq_f32(vcombine_f32(_p4, _p5), _scale_scale);
                        float32x4_t _p67 = vmulq_f32(vcombine_f32(_p6, _p7), _scale_scale);
                        int8x8_t _r0 = float2int8(_p01, _p23);
                        int8x8_t _r1 = float2int8(_p45, _p67);
                        int8x8x2_t _r01 = vuzp_s8(_r0, _r1);
                        vst1q_s8(pp, vcombine_s8(_r01.val[0], _r01.val[1]));
                        pp += 16;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x2_t _p0 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0))), *ps++);
                        float32x2_t _p1 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep))), *ps++);
                        float32x2_t _p2 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2))), *ps++);
                        float32x2_t _p3 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3))), *ps++);
                        float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                        float32x4_t _p23 = vmulq_f32(vcombine_f32(_p2, _p3), _scale_scale);
                        int8x8_t _r0 = float2int8(_p01, _p23);
#if __ARM_FEATURE_DOTPROD
                        int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                        vst1_s8(pp, vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_r01.val[0]), vreinterpret_s32_s8(_r01.val[1])).val[0]));
#else
                        int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                        vst1_s8(pp, vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_r01.val[0]), vreinterpret_s16_s8(_r01.val[1])).val[0]));
#endif
                        pp += 8;
                        p0 += A_hstep * 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float32x2_t _p0 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0))), *ps++);
                        float32x2_t _p1 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0 + A_hstep))), *ps++);
                        float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                        int8x8_t _r0 = float2int8(_p01, _p01);
                        int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                        vst1_lane_s16((short*)pp, vreinterpret_s16_s8(_r01.val[0]), 0);
                        vst1_lane_s16((short*)pp + 1, vreinterpret_s16_s8(_r01.val[1]), 0);
                        pp += 4;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x2_t _p0 = vmul_n_f32(vget_low_f32(bfloat2float(vld1_u16(p0))), *ps++);
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p0), vcombine_f32(_scale, _scale));
                        vst1_lane_s16((short*)pp, vreinterpret_s16_s8(float2int8(_p01, _p01)), 0);
                        pp += 2;
                        p0 += A_hstep;
                    }
#else
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    const unsigned short* p0a = p0;
                    const float* psa = ps;

                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        const float s = *psa++;
                        float v0 = bfloat16_to_float32(p0a[0]);
                        absmax0 = std::max(absmax0, fabsf(v0) * s);
                        float v1 = bfloat16_to_float32(p0a[1]);
                        absmax1 = std::max(absmax1, fabsf(v1) * s);
                        p0a += A_hstep;
                    }

                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;

                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        const float s = *ps++;
                        float v0 = bfloat16_to_float32(p0[0]) * s;
                        float v1 = bfloat16_to_float32(p0[1]) * s;
                        *pp++ = float2int8(v0 * scale0);
                        *pp++ = float2int8(v1 * scale1);
                        p0 += A_hstep;
                    }
#endif // __ARM_NEON
                    pd += 2;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
            const float* ps = (const float*)input_scales + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __ARM_NEON
                if (elempack == 4)
                {
                    float32x4_t _absmax = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _p = vmulq_f32(vabsq_f32(bfloat2float(vld1_u16(p0a))), vld1q_f32(psa));
                        _absmax = vmaxq_f32(_absmax, _p);
                        p0a += A_hstep * 4;
                        psa += 4;
                    }
#if __aarch64__
                    float absmax = vmaxvq_f32(_absmax);
#else
                    float32x2_t _max = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                    _max = vpmax_f32(_max, _max);
                    float absmax = vget_lane_f32(_max, 0);
#endif
                    const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                    *pd++ = absmax / 127.f;

                    float32x4_t _scale = vdupq_n_f32(scale);
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _p = vmulq_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), vld1q_f32(ps)), _scale);
                        int8x8_t _r = float2int8(_p, _p);
                        vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                        pp += 4;
                        p0 += A_hstep * 4;
                        ps += 4;
                    }
                }
#endif // __ARM_NEON
                if (elempack == 1)
                {
                    float absmax = 0.f;
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v = bfloat16_to_float32(*p0a);
                        absmax = std::max(absmax, fabsf(v) * *psa++);
                        p0a += A_hstep;
                    }

                    if (absmax == 0.f)
                    {
                        *pd++ = 0.f;
                        for (int kk0 = 0; kk0 < max_kk0; kk0++)
                            *pp++ = 0;
                        p0 += (size_t)max_kk0 * A_hstep;
                        ps += max_kk0;
                        continue;
                    }

                    const float scale = 127.f / absmax;
                    *pd++ = absmax / 127.f;

                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v = bfloat16_to_float32(*p0) * *ps++;
                        *pp++ = float2int8(v * scale);
                        p0 += A_hstep;
                    }
                }
            }
        }

        return;
    }

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                float32x4_t _absmax2 = vdupq_n_f32(0.f);
                float32x4_t _absmax3 = vdupq_n_f32(0.f);
                float32x4_t _absmax4 = vdupq_n_f32(0.f);
                float32x4_t _absmax5 = vdupq_n_f32(0.f);
                float32x4_t _absmax6 = vdupq_n_f32(0.f);
                float32x4_t _absmax7 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(bfloat2float(vld1_u16(p0a))));
                    _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(bfloat2float(vld1_u16(p0a + 4))));
                    _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(bfloat2float(vld1_u16(p0a + 8))));
                    _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(bfloat2float(vld1_u16(p0a + 12))));
                    _absmax4 = vmaxq_f32(_absmax4, vabsq_f32(bfloat2float(vld1_u16(p0a + 16))));
                    _absmax5 = vmaxq_f32(_absmax5, vabsq_f32(bfloat2float(vld1_u16(p0a + 20))));
                    _absmax6 = vmaxq_f32(_absmax6, vabsq_f32(bfloat2float(vld1_u16(p0a + 24))));
                    _absmax7 = vmaxq_f32(_absmax7, vabsq_f32(bfloat2float(vld1_u16(p0a + 28))));
                    p0a += A_hstep * 4;
                }

                float absmax0 = vmaxvq_f32(_absmax0);
                float absmax1 = vmaxvq_f32(_absmax1);
                float absmax2 = vmaxvq_f32(_absmax2);
                float absmax3 = vmaxvq_f32(_absmax3);
                float absmax4 = vmaxvq_f32(_absmax4);
                float absmax5 = vmaxvq_f32(_absmax5);
                float absmax6 = vmaxvq_f32(_absmax6);
                float absmax7 = vmaxvq_f32(_absmax7);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                _scale0 = vsetq_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale0, 1);
                _scale0 = vsetq_lane_f32(absmax2 == 0.f ? 0.f : 127.f / absmax2, _scale0, 2);
                _scale0 = vsetq_lane_f32(absmax3 == 0.f ? 0.f : 127.f / absmax3, _scale0, 3);
                float32x4_t _scale1 = vdupq_n_f32(absmax4 == 0.f ? 0.f : 127.f / absmax4);
                _scale1 = vsetq_lane_f32(absmax5 == 0.f ? 0.f : 127.f / absmax5, _scale1, 1);
                _scale1 = vsetq_lane_f32(absmax6 == 0.f ? 0.f : 127.f / absmax6, _scale1, 2);
                _scale1 = vsetq_lane_f32(absmax7 == 0.f ? 0.f : 127.f / absmax7, _scale1, 3);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    uint16x4x4_t _p = vld4_u16(p0);
                    uint16x4x4_t _q = vld4_u16(p0 + 16);
                    uint16x4x4_t _r = vld4_u16(p0 + A_hstep * 4);
                    uint16x4x4_t _s = vld4_u16(p0 + A_hstep * 4 + 16);
                    int8x8_t _r0 = float2int8(vmulq_f32(bfloat2float(_p.val[0]), _scale0), vmulq_f32(bfloat2float(_q.val[0]), _scale1));
                    int8x8_t _r1 = float2int8(vmulq_f32(bfloat2float(_p.val[1]), _scale0), vmulq_f32(bfloat2float(_q.val[1]), _scale1));
                    int8x8_t _r2 = float2int8(vmulq_f32(bfloat2float(_p.val[2]), _scale0), vmulq_f32(bfloat2float(_q.val[2]), _scale1));
                    int8x8_t _r3 = float2int8(vmulq_f32(bfloat2float(_p.val[3]), _scale0), vmulq_f32(bfloat2float(_q.val[3]), _scale1));
                    int8x8_t _r4 = float2int8(vmulq_f32(bfloat2float(_r.val[0]), _scale0), vmulq_f32(bfloat2float(_s.val[0]), _scale1));
                    int8x8_t _r5 = float2int8(vmulq_f32(bfloat2float(_r.val[1]), _scale0), vmulq_f32(bfloat2float(_s.val[1]), _scale1));
                    int8x8_t _r6 = float2int8(vmulq_f32(bfloat2float(_r.val[2]), _scale0), vmulq_f32(bfloat2float(_s.val[2]), _scale1));
                    int8x8_t _r7 = float2int8(vmulq_f32(bfloat2float(_r.val[3]), _scale0), vmulq_f32(bfloat2float(_s.val[3]), _scale1));
                    int8x8x2_t _r04 = vzip_s8(_r0, _r4);
                    int8x8x2_t _r15 = vzip_s8(_r1, _r5);
                    int8x8x2_t _r26 = vzip_s8(_r2, _r6);
                    int8x8x2_t _r37 = vzip_s8(_r3, _r7);
                    int8x8x4_t _r0123;
                    _r0123.val[0] = _r04.val[0];
                    _r0123.val[1] = _r15.val[0];
                    _r0123.val[2] = _r26.val[0];
                    _r0123.val[3] = _r37.val[0];
                    int8x8x4_t _r4567;
                    _r4567.val[0] = _r04.val[1];
                    _r4567.val[1] = _r15.val[1];
                    _r4567.val[2] = _r26.val[1];
                    _r4567.val[3] = _r37.val[1];
                    vst4_s8(pp, _r0123);
                    vst4_s8(pp + 32, _r4567);
                    pp += 64;
                    p0 += A_hstep * 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
#if !__ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + 16);
                    uint16x8_t _s = vld1q_u16(p0 + 24);
                    uint16x8_t _t = vld1q_u16(p0 + A_hstep * 4);
                    uint16x8_t _u = vld1q_u16(p0 + A_hstep * 4 + 8);
                    uint16x8_t _v = vld1q_u16(p0 + A_hstep * 4 + 16);
                    uint16x8_t _w = vld1q_u16(p0 + A_hstep * 4 + 24);
                    float32x4_t _p0 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p)), _scale0, 0);
                    float32x4_t _p1 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p)), _scale0, 1);
                    float32x4_t _p2 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_q)), _scale0, 2);
                    float32x4_t _p3 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_q)), _scale0, 3);
                    float32x4_t _p4 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_r)), _scale1, 0);
                    float32x4_t _p5 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_r)), _scale1, 1);
                    float32x4_t _p6 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_s)), _scale1, 2);
                    float32x4_t _p7 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_s)), _scale1, 3);
                    float32x4_t _p8 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_t)), _scale0, 0);
                    float32x4_t _p9 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_t)), _scale0, 1);
                    float32x4_t _pa = vmulq_laneq_f32(bfloat2float(vget_low_u16(_u)), _scale0, 2);
                    float32x4_t _pb = vmulq_laneq_f32(bfloat2float(vget_high_u16(_u)), _scale0, 3);
                    float32x4_t _pc = vmulq_laneq_f32(bfloat2float(vget_low_u16(_v)), _scale1, 0);
                    float32x4_t _pd = vmulq_laneq_f32(bfloat2float(vget_high_u16(_v)), _scale1, 1);
                    float32x4_t _pe = vmulq_laneq_f32(bfloat2float(vget_low_u16(_w)), _scale1, 2);
                    float32x4_t _pf = vmulq_laneq_f32(bfloat2float(vget_high_u16(_w)), _scale1, 3);

#if __ARM_FEATURE_DOTPROD
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    int8x8_t _r2 = float2int8(_p4, _p5);
                    int8x8_t _r3 = float2int8(_p6, _p7);
                    int8x8_t _r4 = float2int8(_p8, _p9);
                    int8x8_t _r5 = float2int8(_pa, _pb);
                    int8x8_t _r6 = float2int8(_pc, _pd);
                    int8x8_t _r7 = float2int8(_pe, _pf);
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else
                    int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(float2int8(_p0, _p1), float2int8(_p2, _p3)));
                    int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(float2int8(_p4, _p5), float2int8(_p6, _p7)));
                    int16x8_t _r45 = vreinterpretq_s16_s8(vcombine_s8(float2int8(_p8, _p9), float2int8(_pa, _pb)));
                    int16x8_t _r67 = vreinterpretq_s16_s8(vcombine_s8(float2int8(_pc, _pd), float2int8(_pe, _pf)));
                    int16x8x2_t _rr0 = vuzpq_s16(_r01, _r23);
                    int16x8x2_t _rr1 = vuzpq_s16(_r45, _r67);
                    vst1q_s8(pp, vreinterpretq_s8_s16(_rr0.val[0]));
                    vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr0.val[1]));
                    vst1q_s8(pp + 32, vreinterpretq_s8_s16(_rr1.val[0]));
                    vst1q_s8(pp + 48, vreinterpretq_s8_s16(_rr1.val[1]));
#endif
                    pp += 64;
                    p0 += A_hstep * 8;
                }
#endif // !__ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + 16);
                    uint16x8_t _s = vld1q_u16(p0 + 24);
                    float32x4_t _p0 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p)), _scale0, 0);
                    float32x4_t _p1 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p)), _scale0, 1);
                    float32x4_t _p2 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_q)), _scale0, 2);
                    float32x4_t _p3 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_q)), _scale0, 3);
                    float32x4_t _p4 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_r)), _scale1, 0);
                    float32x4_t _p5 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_r)), _scale1, 1);
                    float32x4_t _p6 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_s)), _scale1, 2);
                    float32x4_t _p7 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_s)), _scale1, 3);
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    int8x8_t _r2 = float2int8(_p4, _p5);
                    int8x8_t _r3 = float2int8(_p6, _p7);
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else
                    int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                    int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                    int16x8x2_t _rr = vuzpq_s16(_r01, _r23);
                    vst1q_s8(pp, vreinterpretq_s8_s16(_rr.val[0]));
                    vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr.val[1]));
#endif
                    pp += 32;
                    p0 += A_hstep * 4;
                }
                pd += 8;
            }
            if (elempack == 1)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(bfloat2float(vld1_u16(p0a))));
                    _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(bfloat2float(vld1_u16(p0a + 4))));
                    p0a += A_hstep;
                }

                float absmax0 = vgetq_lane_f32(_absmax0, 0);
                float absmax1 = vgetq_lane_f32(_absmax0, 1);
                float absmax2 = vgetq_lane_f32(_absmax0, 2);
                float absmax3 = vgetq_lane_f32(_absmax0, 3);
                float absmax4 = vgetq_lane_f32(_absmax1, 0);
                float absmax5 = vgetq_lane_f32(_absmax1, 1);
                float absmax6 = vgetq_lane_f32(_absmax1, 2);
                float absmax7 = vgetq_lane_f32(_absmax1, 3);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                _scale0 = vsetq_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale0, 1);
                _scale0 = vsetq_lane_f32(absmax2 == 0.f ? 0.f : 127.f / absmax2, _scale0, 2);
                _scale0 = vsetq_lane_f32(absmax3 == 0.f ? 0.f : 127.f / absmax3, _scale0, 3);
                float32x4_t _scale1 = vdupq_n_f32(absmax4 == 0.f ? 0.f : 127.f / absmax4);
                _scale1 = vsetq_lane_f32(absmax5 == 0.f ? 0.f : 127.f / absmax5, _scale1, 1);
                _scale1 = vsetq_lane_f32(absmax6 == 0.f ? 0.f : 127.f / absmax6, _scale1, 2);
                _scale1 = vsetq_lane_f32(absmax7 == 0.f ? 0.f : 127.f / absmax7, _scale1, 3);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p00 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0);
                    float32x4_t _p01 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale1);
                    float32x4_t _p10 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _scale0);
                    float32x4_t _p11 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep + 4)), _scale1);
                    float32x4_t _p20 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), _scale0);
                    float32x4_t _p21 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2 + 4)), _scale1);
                    float32x4_t _p30 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), _scale0);
                    float32x4_t _p31 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3 + 4)), _scale1);
                    float32x4_t _p40 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _scale0);
                    float32x4_t _p41 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _scale1);
                    float32x4_t _p50 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 5)), _scale0);
                    float32x4_t _p51 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 5 + 4)), _scale1);
                    float32x4_t _p60 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 6)), _scale0);
                    float32x4_t _p61 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 6 + 4)), _scale1);
                    float32x4_t _p70 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 7)), _scale0);
                    float32x4_t _p71 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 7 + 4)), _scale1);
                    int8x8_t _r0 = float2int8(_p00, _p01);
                    int8x8_t _r1 = float2int8(_p10, _p11);
                    int8x8_t _r2 = float2int8(_p20, _p21);
                    int8x8_t _r3 = float2int8(_p30, _p31);
                    int8x8_t _r4 = float2int8(_p40, _p41);
                    int8x8_t _r5 = float2int8(_p50, _p51);
                    int8x8_t _r6 = float2int8(_p60, _p61);
                    int8x8_t _r7 = float2int8(_p70, _p71);
                    int8x8x2_t _r04 = vzip_s8(_r0, _r4);
                    int8x8x2_t _r15 = vzip_s8(_r1, _r5);
                    int8x8x2_t _r26 = vzip_s8(_r2, _r6);
                    int8x8x2_t _r37 = vzip_s8(_r3, _r7);
                    int8x8x4_t _r0123;
                    _r0123.val[0] = _r04.val[0];
                    _r0123.val[1] = _r15.val[0];
                    _r0123.val[2] = _r26.val[0];
                    _r0123.val[3] = _r37.val[0];
                    int8x8x4_t _r4567;
                    _r4567.val[0] = _r04.val[1];
                    _r4567.val[1] = _r15.val[1];
                    _r4567.val[2] = _r26.val[1];
                    _r4567.val[3] = _r37.val[1];
                    vst4_s8(pp, _r0123);
                    vst4_s8(pp + 32, _r4567);
                    pp += 64;
                    p0 += A_hstep * 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
#if !__ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + A_hstep);
                    uint16x8_t _r = vld1q_u16(p0 + A_hstep * 2);
                    uint16x8_t _s = vld1q_u16(p0 + A_hstep * 3);
                    uint16x8_t _t = vld1q_u16(p0 + A_hstep * 4);
                    uint16x8_t _u = vld1q_u16(p0 + A_hstep * 5);
                    uint16x8_t _v = vld1q_u16(p0 + A_hstep * 6);
                    uint16x8_t _w = vld1q_u16(p0 + A_hstep * 7);
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vget_low_u16(_p)), _scale0);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vget_high_u16(_p)), _scale1);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(vget_low_u16(_q)), _scale0);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(vget_high_u16(_q)), _scale1);
                    float32x4_t _p4 = vmulq_f32(bfloat2float(vget_low_u16(_r)), _scale0);
                    float32x4_t _p5 = vmulq_f32(bfloat2float(vget_high_u16(_r)), _scale1);
                    float32x4_t _p6 = vmulq_f32(bfloat2float(vget_low_u16(_s)), _scale0);
                    float32x4_t _p7 = vmulq_f32(bfloat2float(vget_high_u16(_s)), _scale1);
                    float32x4_t _p8 = vmulq_f32(bfloat2float(vget_low_u16(_t)), _scale0);
                    float32x4_t _p9 = vmulq_f32(bfloat2float(vget_high_u16(_t)), _scale1);
                    float32x4_t _pa = vmulq_f32(bfloat2float(vget_low_u16(_u)), _scale0);
                    float32x4_t _pb = vmulq_f32(bfloat2float(vget_high_u16(_u)), _scale1);
                    float32x4_t _pc = vmulq_f32(bfloat2float(vget_low_u16(_v)), _scale0);
                    float32x4_t _pd = vmulq_f32(bfloat2float(vget_high_u16(_v)), _scale1);
                    float32x4_t _pe = vmulq_f32(bfloat2float(vget_low_u16(_w)), _scale0);
                    float32x4_t _pf = vmulq_f32(bfloat2float(vget_high_u16(_w)), _scale1);

#if __ARM_FEATURE_DOTPROD
                    int8x8x4_t _r0123;
                    _r0123.val[0] = float2int8(_p0, _p1);
                    _r0123.val[1] = float2int8(_p2, _p3);
                    _r0123.val[2] = float2int8(_p4, _p5);
                    _r0123.val[3] = float2int8(_p6, _p7);
                    int8x8x4_t _r4567;
                    _r4567.val[0] = float2int8(_p8, _p9);
                    _r4567.val[1] = float2int8(_pa, _pb);
                    _r4567.val[2] = float2int8(_pc, _pd);
                    _r4567.val[3] = float2int8(_pe, _pf);
                    vst4_s8(pp, _r0123);
                    vst4_s8(pp + 32, _r4567);
#else
                    int8x16x2_t _r01;
                    _r01.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p4, _p5));
                    _r01.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_p6, _p7));
                    int8x16x2_t _r23;
                    _r23.val[0] = vcombine_s8(float2int8(_p8, _p9), float2int8(_pc, _pd));
                    _r23.val[1] = vcombine_s8(float2int8(_pa, _pb), float2int8(_pe, _pf));
                    vst2q_s8(pp, _r01);
                    vst2q_s8(pp + 32, _r23);
#endif
                    pp += 64;
                    p0 += A_hstep * 8;
                }
#endif // !__ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale1);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _scale0);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep + 4)), _scale1);
                    float32x4_t _p4 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), _scale0);
                    float32x4_t _p5 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2 + 4)), _scale1);
                    float32x4_t _p6 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), _scale0);
                    float32x4_t _p7 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3 + 4)), _scale1);
#if __ARM_FEATURE_DOTPROD
                    int8x8x4_t _r0123;
                    _r0123.val[0] = float2int8(_p0, _p1);
                    _r0123.val[1] = float2int8(_p2, _p3);
                    _r0123.val[2] = float2int8(_p4, _p5);
                    _r0123.val[3] = float2int8(_p6, _p7);
                    vst4_s8(pp, _r0123);
#else
                    int8x16x2_t _r01;
                    _r01.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p4, _p5));
                    _r01.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_p6, _p7));
                    vst2q_s8(pp, _r01);
#endif
                    pp += 32;
                    p0 += A_hstep * 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale1);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _scale0);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep + 4)), _scale1);
                    int8x8x2_t _r01;
                    _r01.val[0] = float2int8(_p0, _p1);
                    _r01.val[1] = float2int8(_p2, _p3);
                    vst2_s8(pp, _r01);
                    pp += 16;
                    p0 += A_hstep * 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale1);
                    vst1_s8(pp, float2int8(_p0, _p1));
                    pp += 8;
                    p0 += A_hstep;
                }
                pd += 8;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                float32x4_t _absmax2 = vdupq_n_f32(0.f);
                float32x4_t _absmax3 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(bfloat2float(vld1_u16(p0a))));
                    _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(bfloat2float(vld1_u16(p0a + 4))));
                    _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(bfloat2float(vld1_u16(p0a + 8))));
                    _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(bfloat2float(vld1_u16(p0a + 12))));
                    p0a += A_hstep * 4;
                }

#if __aarch64__
                float absmax0 = vmaxvq_f32(_absmax0);
                float absmax1 = vmaxvq_f32(_absmax1);
                float absmax2 = vmaxvq_f32(_absmax2);
                float absmax3 = vmaxvq_f32(_absmax3);
#else
                float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                _max0 = vpmax_f32(_max0, _max0);
                _max1 = vpmax_f32(_max1, _max1);
                _max2 = vpmax_f32(_max2, _max2);
                _max3 = vpmax_f32(_max3, _max3);
                float absmax0 = vget_lane_f32(_max0, 0);
                float absmax1 = vget_lane_f32(_max1, 0);
                float absmax2 = vget_lane_f32(_max2, 0);
                float absmax3 = vget_lane_f32(_max3, 0);
#endif
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                float32x4_t _scale = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                _scale = vsetq_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale, 1);
                _scale = vsetq_lane_f32(absmax2 == 0.f ? 0.f : 127.f / absmax2, _scale, 2);
                _scale = vsetq_lane_f32(absmax3 == 0.f ? 0.f : 127.f / absmax3, _scale, 3);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    uint16x4x4_t _p = vld4_u16(p0);
                    uint16x4x4_t _q = vld4_u16(p0 + A_hstep * 4);
                    float32x4_t _p0 = vmulq_f32(bfloat2float(_p.val[0]), _scale);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(_p.val[1]), _scale);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(_p.val[2]), _scale);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(_p.val[3]), _scale);
                    float32x4_t _p4 = vmulq_f32(bfloat2float(_q.val[0]), _scale);
                    float32x4_t _p5 = vmulq_f32(bfloat2float(_q.val[1]), _scale);
                    float32x4_t _p6 = vmulq_f32(bfloat2float(_q.val[2]), _scale);
                    float32x4_t _p7 = vmulq_f32(bfloat2float(_q.val[3]), _scale);
                    float32x4x2_t _p04 = vzipq_f32(_p0, _p4);
                    float32x4x2_t _p15 = vzipq_f32(_p1, _p5);
                    float32x4x2_t _p26 = vzipq_f32(_p2, _p6);
                    float32x4x2_t _p37 = vzipq_f32(_p3, _p7);
                    int8x8x4_t _r0123;
                    _r0123.val[0] = float2int8(_p04.val[0], _p04.val[1]);
                    _r0123.val[1] = float2int8(_p15.val[0], _p15.val[1]);
                    _r0123.val[2] = float2int8(_p26.val[0], _p26.val[1]);
                    _r0123.val[3] = float2int8(_p37.val[0], _p37.val[1]);
                    vst4_s8(pp, _r0123);
                    pp += 32;
                    p0 += A_hstep * 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    uint16x4x4_t _p = vld4_u16(p0);
                    float32x4_t _p0 = vmulq_f32(bfloat2float(_p.val[0]), _scale);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(_p.val[1]), _scale);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(_p.val[2]), _scale);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(_p.val[3]), _scale);
#if __ARM_FEATURE_DOTPROD
                    transpose4x4_ps(_p0, _p1, _p2, _p3);
                    int8x8_t _r01 = float2int8(_p0, _p1);
                    int8x8_t _r23 = float2int8(_p2, _p3);
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                    int8x8_t _r01 = float2int8(_p0, _p1);
                    int8x8_t _r23 = float2int8(_p2, _p3);
                    int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                    int8x8_t _r32 = vext_s8(_r23, _r23, 4);
                    vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                    vst1_s8(pp + 8, vzip_s8(_r23, _r32).val[0]);
#endif
                    pp += 16;
                    p0 += A_hstep * 4;
                }
                pd += 4;
            }
            if (elempack == 1)
            {
                float32x4_t _absmax = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    _absmax = vmaxq_f32(_absmax, vabsq_f32(bfloat2float(vld1_u16(p0a))));
                    p0a += A_hstep;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax, 1.f / 127.f));
                float absmax0 = vgetq_lane_f32(_absmax, 0);
                float absmax1 = vgetq_lane_f32(_absmax, 1);
                float absmax2 = vgetq_lane_f32(_absmax, 2);
                float absmax3 = vgetq_lane_f32(_absmax, 3);
                float32x4_t _scale = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                _scale = vsetq_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale, 1);
                _scale = vsetq_lane_f32(absmax2 == 0.f ? 0.f : 127.f / absmax2, _scale, 2);
                _scale = vsetq_lane_f32(absmax3 == 0.f ? 0.f : 127.f / absmax3, _scale, 3);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _scale);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), _scale);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), _scale);
                    float32x4_t _p4 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _scale);
                    float32x4_t _p5 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 5)), _scale);
                    float32x4_t _p6 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 6)), _scale);
                    float32x4_t _p7 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 7)), _scale);
                    float32x4x2_t _p04 = vzipq_f32(_p0, _p4);
                    float32x4x2_t _p15 = vzipq_f32(_p1, _p5);
                    float32x4x2_t _p26 = vzipq_f32(_p2, _p6);
                    float32x4x2_t _p37 = vzipq_f32(_p3, _p7);
                    int8x8x4_t _r0123;
                    _r0123.val[0] = float2int8(_p04.val[0], _p04.val[1]);
                    _r0123.val[1] = float2int8(_p15.val[0], _p15.val[1]);
                    _r0123.val[2] = float2int8(_p26.val[0], _p26.val[1]);
                    _r0123.val[3] = float2int8(_p37.val[0], _p37.val[1]);
                    vst4_s8(pp, _r0123);
                    pp += 32;
                    p0 += A_hstep * 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep)), _scale);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 2)), _scale);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 3)), _scale);
#if __ARM_FEATURE_DOTPROD
                    transpose4x4_ps(_p0, _p1, _p2, _p3);
                    int8x8_t _r01 = float2int8(_p0, _p1);
                    int8x8_t _r23 = float2int8(_p2, _p3);
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                    int8x8_t _r01 = float2int8(_p0, _p1);
                    int8x8_t _r23 = float2int8(_p2, _p3);
                    int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                    int8x8_t _r32 = vext_s8(_r23, _r23, 4);
                    vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                    vst1_s8(pp + 8, vzip_s8(_r23, _r32).val[0]);
#endif
                    pp += 16;
                    p0 += A_hstep * 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                    float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                    int8x8_t _r01 = float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale));
                    int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                    vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                    pp += 8;
                    p0 += A_hstep * 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(p0));
                    _p = vmulq_f32(_p, _scale);
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(float2int8(_p, _p)), 0);
                    pp += 4;
                    p0 += A_hstep;
                }
                pd += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __ARM_NEON
            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(bfloat2float(vld1_u16(p0a))));
                    _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(bfloat2float(vld1_u16(p0a + 4))));
                    p0a += A_hstep * 4;
                }

#if __aarch64__
                float absmax0 = vmaxvq_f32(_absmax0);
                float absmax1 = vmaxvq_f32(_absmax1);
#else
                float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                _max0 = vpmax_f32(_max0, _max0);
                _max1 = vpmax_f32(_max1, _max1);
                float absmax0 = vget_lane_f32(_max0, 0);
                float absmax1 = vget_lane_f32(_max1, 0);
#endif
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                float32x4_t _scale1 = vdupq_n_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1);

                int kk = 0;
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale1);
                    float32x4_t _p2 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _scale0);
                    float32x4_t _p3 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _scale1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p2);
                    int8x8_t _r1 = float2int8(_p1, _p3);
#else
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
#endif
#else
                    int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                    int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                    int16x4x2_t _t01 = vzip_s16(_t0, _t1);
                    int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                    int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));

                    pp += 16;
                    p0 += A_hstep * 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0);
                    float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale1);

#if __ARM_FEATURE_DOTPROD
                    int8x8_t _r01 = float2int8(_p0, _p1);
#else
                    float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                    float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));
                    int8x8_t _r01 = float2int8(_t0, _t1);
#endif

                    vst1_s8(pp, _r01);

                    pp += 8;
                    p0 += A_hstep * 4;
                }
                pd += 2;
            }
            if (elempack == 1)
            {
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(p0a[0]);
                    float v1 = bfloat16_to_float32(p0a[1]);
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                    p0a += A_hstep;
                }

                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                float32x4_t _scale0 = vdupq_n_f32(scale0);
                float32x4_t _scale1 = vdupq_n_f32(scale1);
                float32x4_t _scale = vzipq_f32(_scale0, _scale1).val[0];

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    uint16x8_t _p = uint16x8_t();
                    _p = vsetq_lane_u16(p0[0], _p, 0);
                    _p = vsetq_lane_u16(p0[1], _p, 1);
                    _p = vsetq_lane_u16(p0[A_hstep], _p, 2);
                    _p = vsetq_lane_u16(p0[A_hstep + 1], _p, 3);
                    _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 4);
                    _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 5);
                    _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 6);
                    _p = vsetq_lane_u16(p0[A_hstep * 3 + 1], _p, 7);
                    uint16x8_t _q = uint16x8_t();
                    _q = vsetq_lane_u16(p0[A_hstep * 4], _q, 0);
                    _q = vsetq_lane_u16(p0[A_hstep * 4 + 1], _q, 1);
                    _q = vsetq_lane_u16(p0[A_hstep * 5], _q, 2);
                    _q = vsetq_lane_u16(p0[A_hstep * 5 + 1], _q, 3);
                    _q = vsetq_lane_u16(p0[A_hstep * 6], _q, 4);
                    _q = vsetq_lane_u16(p0[A_hstep * 6 + 1], _q, 5);
                    _q = vsetq_lane_u16(p0[A_hstep * 7], _q, 6);
                    _q = vsetq_lane_u16(p0[A_hstep * 7 + 1], _q, 7);
                    float32x4_t _p01 = vmulq_f32(bfloat2float(vget_low_u16(_p)), _scale);
                    float32x4_t _p23 = vmulq_f32(bfloat2float(vget_high_u16(_p)), _scale);
                    float32x4_t _p45 = vmulq_f32(bfloat2float(vget_low_u16(_q)), _scale);
                    float32x4_t _p67 = vmulq_f32(bfloat2float(vget_high_u16(_q)), _scale);
                    int8x8_t _r0 = float2int8(_p01, _p23);
                    int8x8_t _r1 = float2int8(_p45, _p67);
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8x2_t _r01 = vuzp_s8(_r0, _r1);
                    vst1q_s8(pp, vcombine_s8(_r01.val[0], _r01.val[1]));
#else
                    int8x8x2_t _r01 = vtrn_s8(_r0, _r1);
                    int8x8x2_t _rr01 = vuzp_s8(_r01.val[0], _r01.val[1]);
                    vst1q_s8(pp, vcombine_s8(_rr01.val[0], _rr01.val[1]));
#endif
                    pp += 16;
                    p0 += A_hstep * 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    uint16x8_t _p = uint16x8_t();
                    _p = vsetq_lane_u16(p0[0], _p, 0);
                    _p = vsetq_lane_u16(p0[1], _p, 1);
                    _p = vsetq_lane_u16(p0[A_hstep], _p, 2);
                    _p = vsetq_lane_u16(p0[A_hstep + 1], _p, 3);
                    _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 4);
                    _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 5);
                    _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 6);
                    _p = vsetq_lane_u16(p0[A_hstep * 3 + 1], _p, 7);
                    float32x4_t _p01 = vmulq_f32(bfloat2float(vget_low_u16(_p)), _scale);
                    float32x4_t _p23 = vmulq_f32(bfloat2float(vget_high_u16(_p)), _scale);
                    float32x4x2_t _p0123 = vuzpq_f32(_p01, _p23);
                    int8x8_t _r01 = float2int8(_p0123.val[0], _p0123.val[1]);
                    vst1_s8(pp, _r01);
                    pp += 8;
                    p0 += A_hstep * 4;
                }
#else
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    uint16x8_t _p = uint16x8_t();
                    _p = vsetq_lane_u16(p0[0], _p, 0);
                    _p = vsetq_lane_u16(p0[1], _p, 1);
                    _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 2);
                    _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 3);
                    _p = vsetq_lane_u16(p0[A_hstep * 4], _p, 4);
                    _p = vsetq_lane_u16(p0[A_hstep * 4 + 1], _p, 5);
                    _p = vsetq_lane_u16(p0[A_hstep * 6], _p, 6);
                    _p = vsetq_lane_u16(p0[A_hstep * 6 + 1], _p, 7);
                    uint16x8_t _q = uint16x8_t();
                    _q = vsetq_lane_u16(p0[A_hstep], _q, 0);
                    _q = vsetq_lane_u16(p0[A_hstep + 1], _q, 1);
                    _q = vsetq_lane_u16(p0[A_hstep * 3], _q, 2);
                    _q = vsetq_lane_u16(p0[A_hstep * 3 + 1], _q, 3);
                    _q = vsetq_lane_u16(p0[A_hstep * 5], _q, 4);
                    _q = vsetq_lane_u16(p0[A_hstep * 5 + 1], _q, 5);
                    _q = vsetq_lane_u16(p0[A_hstep * 7], _q, 6);
                    _q = vsetq_lane_u16(p0[A_hstep * 7 + 1], _q, 7);
                    float32x4_t _p02 = vmulq_f32(bfloat2float(vget_low_u16(_p)), _scale);
                    float32x4_t _p46 = vmulq_f32(bfloat2float(vget_high_u16(_p)), _scale);
                    float32x4_t _p13 = vmulq_f32(bfloat2float(vget_low_u16(_q)), _scale);
                    float32x4_t _p57 = vmulq_f32(bfloat2float(vget_high_u16(_q)), _scale);
                    int8x8x2_t _r01;
                    _r01.val[0] = float2int8(_p02, _p46);
                    _r01.val[1] = float2int8(_p13, _p57);
                    vst2_s8(pp, _r01);
                    pp += 16;
                    p0 += A_hstep * 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    uint16x8_t _p = uint16x8_t();
                    _p = vsetq_lane_u16(p0[0], _p, 0);
                    _p = vsetq_lane_u16(p0[1], _p, 1);
                    _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 2);
                    _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 3);
                    _p = vsetq_lane_u16(p0[A_hstep], _p, 4);
                    _p = vsetq_lane_u16(p0[A_hstep + 1], _p, 5);
                    _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 6);
                    _p = vsetq_lane_u16(p0[A_hstep * 3 + 1], _p, 7);
                    float32x4_t _p02 = vmulq_f32(bfloat2float(vget_low_u16(_p)), _scale);
                    float32x4_t _p13 = vmulq_f32(bfloat2float(vget_high_u16(_p)), _scale);
                    float32x4x2_t _p0123 = vzipq_f32(_p02, _p13);
                    int8x8_t _r01 = float2int8(_p0123.val[0], _p0123.val[1]);
                    vst1_s8(pp, _r01);
                    pp += 8;
                    p0 += A_hstep * 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(p0[A_hstep]) * scale0);
                    pp[2] = float2int8(bfloat16_to_float32(p0[1]) * scale1);
                    pp[3] = float2int8(bfloat16_to_float32(p0[A_hstep + 1]) * scale1);
                    pp += 4;
                    p0 += A_hstep * 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                    pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale1);
                    pp += 2;
                    p0 += A_hstep;
                }
                pd += 2;
            }
#else
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const unsigned short* p0a = p0;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(p0a[0]);
                float v1 = bfloat16_to_float32(p0a[1]);
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
                p0a += A_hstep;
            }

            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;

            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(p0[0]);
                float v1 = bfloat16_to_float32(p0[1]);
                *pp++ = float2int8(v0 * scale0);
                *pp++ = float2int8(v1 * scale1);
                p0 += A_hstep;
            }
            pd += 2;
#endif // __ARM_NEON
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __ARM_NEON
            if (elempack == 4)
            {
                float32x4_t _absmax = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    _absmax = vmaxq_f32(_absmax, vabsq_f32(bfloat2float(vld1_u16(p0a))));
                    p0a += A_hstep * 4;
                }

#if __aarch64__
                float absmax = vmaxvq_f32(_absmax);
#else
                float32x2_t _max = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                _max = vpmax_f32(_max, _max);
                float absmax = vget_lane_f32(_max, 0);
#endif
                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;

                float32x4_t _scale = vdupq_n_f32(scale);
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _p = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale);
                    int8x8_t _r = float2int8(_p, _p);
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                    pp += 4;
                    p0 += A_hstep * 4;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                float absmax = 0.f;
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0a);
                    absmax = std::max(absmax, fabsf(v));
                    p0a += A_hstep;
                }

                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = bfloat16_to_float32(*p0);
                    *pp++ = float2int8(v * scale);
                    p0 += A_hstep;
                }
            }
        }
    }
}

static void unpack_output_tile_wq_int8_bf16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    const float* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        float32x4_t _c0;
        float32x4_t _c1;
        float32x4_t _alpha = vdupq_n_f32(alpha);
        float32x4_t _beta = vdupq_n_f32(beta);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(pC[0] * beta);
                _c1 = _c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = vld1q_f32(pC);
                _c1 = vld1q_f32(pC + 4);
                if (beta != 1.f)
                {
                    _c0 = vmulq_f32(_c0, _beta);
                    _c1 = vmulq_f32(_c1, _beta);
                }
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out00 = vld1q_f32(pp);
            float32x4_t _out01 = vld1q_f32(pp + 32);
            float32x4_t _out10 = vld1q_f32(pp + 4);
            float32x4_t _out11 = vld1q_f32(pp + 36);
            float32x4_t _out20 = vld1q_f32(pp + 8);
            float32x4_t _out21 = vld1q_f32(pp + 40);
            float32x4_t _out30 = vld1q_f32(pp + 12);
            float32x4_t _out31 = vld1q_f32(pp + 44);

            float32x4_t _out40 = vld1q_f32(pp + 16);
            float32x4_t _out41 = vld1q_f32(pp + 48);
            float32x4_t _out50 = vld1q_f32(pp + 20);
            float32x4_t _out51 = vld1q_f32(pp + 52);
            float32x4_t _out60 = vld1q_f32(pp + 24);
            float32x4_t _out61 = vld1q_f32(pp + 56);
            float32x4_t _out70 = vld1q_f32(pp + 28);
            float32x4_t _out71 = vld1q_f32(pp + 60);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out00 = vaddq_f32(_out00, vdupq_laneq_f32(_c0, 0));
                    _out01 = vaddq_f32(_out01, vdupq_laneq_f32(_c0, 0));
                    _out10 = vaddq_f32(_out10, vdupq_laneq_f32(_c0, 1));
                    _out11 = vaddq_f32(_out11, vdupq_laneq_f32(_c0, 1));
                    _out20 = vaddq_f32(_out20, vdupq_laneq_f32(_c0, 2));
                    _out21 = vaddq_f32(_out21, vdupq_laneq_f32(_c0, 2));
                    _out30 = vaddq_f32(_out30, vdupq_laneq_f32(_c0, 3));
                    _out31 = vaddq_f32(_out31, vdupq_laneq_f32(_c0, 3));
                    _out40 = vaddq_f32(_out40, vdupq_laneq_f32(_c1, 0));
                    _out41 = vaddq_f32(_out41, vdupq_laneq_f32(_c1, 0));
                    _out50 = vaddq_f32(_out50, vdupq_laneq_f32(_c1, 1));
                    _out51 = vaddq_f32(_out51, vdupq_laneq_f32(_c1, 1));
                    _out60 = vaddq_f32(_out60, vdupq_laneq_f32(_c1, 2));
                    _out61 = vaddq_f32(_out61, vdupq_laneq_f32(_c1, 2));
                    _out70 = vaddq_f32(_out70, vdupq_laneq_f32(_c1, 3));
                    _out71 = vaddq_f32(_out71, vdupq_laneq_f32(_c1, 3));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out00 = vaddq_f32(_out00, _c0);
                            _out10 = vaddq_f32(_out10, _c1);
                            _out20 = vaddq_f32(_out20, _c2);
                            _out30 = vaddq_f32(_out30, _c3);
                        }
                        else
                        {
                            _out00 = vmlaq_f32(_out00, _c0, _beta);
                            _out10 = vmlaq_f32(_out10, _c1, _beta);
                            _out20 = vmlaq_f32(_out20, _c2, _beta);
                            _out30 = vmlaq_f32(_out30, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + 16);
                        _c1 = vld1q_f32(pC + 20);
                        _c2 = vld1q_f32(pC + 24);
                        _c3 = vld1q_f32(pC + 28);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out01 = vaddq_f32(_out01, _c0);
                            _out11 = vaddq_f32(_out11, _c1);
                            _out21 = vaddq_f32(_out21, _c2);
                            _out31 = vaddq_f32(_out31, _c3);
                        }
                        else
                        {
                            _out01 = vmlaq_f32(_out01, _c0, _beta);
                            _out11 = vmlaq_f32(_out11, _c1, _beta);
                            _out21 = vmlaq_f32(_out21, _c2, _beta);
                            _out31 = vmlaq_f32(_out31, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 8);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out40 = vaddq_f32(_out40, _c0);
                            _out50 = vaddq_f32(_out50, _c1);
                            _out60 = vaddq_f32(_out60, _c2);
                            _out70 = vaddq_f32(_out70, _c3);
                        }
                        else
                        {
                            _out40 = vmlaq_f32(_out40, _c0, _beta);
                            _out50 = vmlaq_f32(_out50, _c1, _beta);
                            _out60 = vmlaq_f32(_out60, _c2, _beta);
                            _out70 = vmlaq_f32(_out70, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4 + 16);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 20);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 24);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 28);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out41 = vaddq_f32(_out41, _c0);
                            _out51 = vaddq_f32(_out51, _c1);
                            _out61 = vaddq_f32(_out61, _c2);
                            _out71 = vaddq_f32(_out71, _c3);
                        }
                        else
                        {
                            _out41 = vmlaq_f32(_out41, _c0, _beta);
                            _out51 = vmlaq_f32(_out51, _c1, _beta);
                            _out61 = vmlaq_f32(_out61, _c2, _beta);
                            _out71 = vmlaq_f32(_out71, _c3, _beta);
                        }
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        if (beta == 1.f)
                        {
                            _out00 = vaddq_f32(_out00, _c0);
                            _out01 = vaddq_f32(_out01, _c1);
                            _out10 = vaddq_f32(_out10, _c2);
                            _out11 = vaddq_f32(_out11, _c3);
                            _out20 = vaddq_f32(_out20, _c4);
                            _out21 = vaddq_f32(_out21, _c5);
                            _out30 = vaddq_f32(_out30, _c6);
                            _out31 = vaddq_f32(_out31, _c7);
                        }
                        else
                        {
                            _out00 = vmlaq_f32(_out00, _c0, _beta);
                            _out01 = vmlaq_f32(_out01, _c1, _beta);
                            _out10 = vmlaq_f32(_out10, _c2, _beta);
                            _out11 = vmlaq_f32(_out11, _c3, _beta);
                            _out20 = vmlaq_f32(_out20, _c4, _beta);
                            _out21 = vmlaq_f32(_out21, _c5, _beta);
                            _out30 = vmlaq_f32(_out30, _c6, _beta);
                            _out31 = vmlaq_f32(_out31, _c7, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 5);
                        _c3 = vld1q_f32(pC + c_hstep * 5 + 4);
                        _c4 = vld1q_f32(pC + c_hstep * 6);
                        _c5 = vld1q_f32(pC + c_hstep * 6 + 4);
                        _c6 = vld1q_f32(pC + c_hstep * 7);
                        _c7 = vld1q_f32(pC + c_hstep * 7 + 4);
                        if (beta == 1.f)
                        {
                            _out40 = vaddq_f32(_out40, _c0);
                            _out41 = vaddq_f32(_out41, _c1);
                            _out50 = vaddq_f32(_out50, _c2);
                            _out51 = vaddq_f32(_out51, _c3);
                            _out60 = vaddq_f32(_out60, _c4);
                            _out61 = vaddq_f32(_out61, _c5);
                            _out70 = vaddq_f32(_out70, _c6);
                            _out71 = vaddq_f32(_out71, _c7);
                        }
                        else
                        {
                            _out40 = vmlaq_f32(_out40, _c0, _beta);
                            _out41 = vmlaq_f32(_out41, _c1, _beta);
                            _out50 = vmlaq_f32(_out50, _c2, _beta);
                            _out51 = vmlaq_f32(_out51, _c3, _beta);
                            _out60 = vmlaq_f32(_out60, _c4, _beta);
                            _out61 = vmlaq_f32(_out61, _c5, _beta);
                            _out70 = vmlaq_f32(_out70, _c6, _beta);
                            _out71 = vmlaq_f32(_out71, _c7, _beta);
                        }
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_f32(_c, _beta);
                    _out00 = vaddq_f32(_out00, _c);
                    _out10 = vaddq_f32(_out10, _c);
                    _out20 = vaddq_f32(_out20, _c);
                    _out30 = vaddq_f32(_out30, _c);
                    _out40 = vaddq_f32(_out40, _c);
                    _out50 = vaddq_f32(_out50, _c);
                    _out60 = vaddq_f32(_out60, _c);
                    _out70 = vaddq_f32(_out70, _c);
                    _c = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                        _c = vmulq_f32(_c, _beta);
                    _out01 = vaddq_f32(_out01, _c);
                    _out11 = vaddq_f32(_out11, _c);
                    _out21 = vaddq_f32(_out21, _c);
                    _out31 = vaddq_f32(_out31, _c);
                    _out41 = vaddq_f32(_out41, _c);
                    _out51 = vaddq_f32(_out51, _c);
                    _out61 = vaddq_f32(_out61, _c);
                    _out71 = vaddq_f32(_out71, _c);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                _out00 = vmulq_f32(_out00, _alpha);
                _out01 = vmulq_f32(_out01, _alpha);
                _out10 = vmulq_f32(_out10, _alpha);
                _out11 = vmulq_f32(_out11, _alpha);
                _out20 = vmulq_f32(_out20, _alpha);
                _out21 = vmulq_f32(_out21, _alpha);
                _out30 = vmulq_f32(_out30, _alpha);
                _out31 = vmulq_f32(_out31, _alpha);
                _out40 = vmulq_f32(_out40, _alpha);
                _out41 = vmulq_f32(_out41, _alpha);
                _out50 = vmulq_f32(_out50, _alpha);
                _out51 = vmulq_f32(_out51, _alpha);
                _out60 = vmulq_f32(_out60, _alpha);
                _out61 = vmulq_f32(_out61, _alpha);
                _out70 = vmulq_f32(_out70, _alpha);
                _out71 = vmulq_f32(_out71, _alpha);
            }

            if (out_elempack == 4)
            {
                transpose4x4_ps(_out00, _out10, _out20, _out30);
                transpose4x4_ps(_out01, _out11, _out21, _out31);
                transpose4x4_ps(_out40, _out50, _out60, _out70);
                transpose4x4_ps(_out41, _out51, _out61, _out71);
                vst1_u16(p0, float2bfloat(_out00));
                vst1_u16(p0 + 4, float2bfloat(_out10));
                vst1_u16(p0 + 8, float2bfloat(_out20));
                vst1_u16(p0 + 12, float2bfloat(_out30));
                vst1_u16(p0 + 16, float2bfloat(_out01));
                vst1_u16(p0 + 20, float2bfloat(_out11));
                vst1_u16(p0 + 24, float2bfloat(_out21));
                vst1_u16(p0 + 28, float2bfloat(_out31));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out40));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_out50));
                vst1_u16(p0 + out_hstep * 4 + 8, float2bfloat(_out60));
                vst1_u16(p0 + out_hstep * 4 + 12, float2bfloat(_out70));
                vst1_u16(p0 + out_hstep * 4 + 16, float2bfloat(_out41));
                vst1_u16(p0 + out_hstep * 4 + 20, float2bfloat(_out51));
                vst1_u16(p0 + out_hstep * 4 + 24, float2bfloat(_out61));
                vst1_u16(p0 + out_hstep * 4 + 28, float2bfloat(_out71));
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, float2bfloat(_out00));
                vst1_u16(p0 + 4, float2bfloat(_out01));
                vst1_u16(p0 + out_hstep, float2bfloat(_out10));
                vst1_u16(p0 + out_hstep + 4, float2bfloat(_out11));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_out20));
                vst1_u16(p0 + out_hstep * 2 + 4, float2bfloat(_out21));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_out30));
                vst1_u16(p0 + out_hstep * 3 + 4, float2bfloat(_out31));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out40));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_out41));
                vst1_u16(p0 + out_hstep * 5, float2bfloat(_out50));
                vst1_u16(p0 + out_hstep * 5 + 4, float2bfloat(_out51));
                vst1_u16(p0 + out_hstep * 6, float2bfloat(_out60));
                vst1_u16(p0 + out_hstep * 6 + 4, float2bfloat(_out61));
                vst1_u16(p0 + out_hstep * 7, float2bfloat(_out70));
                vst1_u16(p0 + out_hstep * 7 + 4, float2bfloat(_out71));
                p0 += 8;
            }

            pp += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);
            float32x4_t _out4 = vld1q_f32(pp + 16);
            float32x4_t _out5 = vld1q_f32(pp + 20);
            float32x4_t _out6 = vld1q_f32(pp + 24);
            float32x4_t _out7 = vld1q_f32(pp + 28);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, vdupq_laneq_f32(_c0, 0));
                    _out1 = vaddq_f32(_out1, vdupq_laneq_f32(_c0, 1));
                    _out2 = vaddq_f32(_out2, vdupq_laneq_f32(_c0, 2));
                    _out3 = vaddq_f32(_out3, vdupq_laneq_f32(_c0, 3));
                    _out4 = vaddq_f32(_out4, vdupq_laneq_f32(_c1, 0));
                    _out5 = vaddq_f32(_out5, vdupq_laneq_f32(_c1, 1));
                    _out6 = vaddq_f32(_out6, vdupq_laneq_f32(_c1, 2));
                    _out7 = vaddq_f32(_out7, vdupq_laneq_f32(_c1, 3));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c0);
                            _out1 = vaddq_f32(_out1, _c1);
                            _out2 = vaddq_f32(_out2, _c2);
                            _out3 = vaddq_f32(_out3, _c3);
                        }
                        else
                        {
                            _out0 = vmlaq_f32(_out0, _c0, _beta);
                            _out1 = vmlaq_f32(_out1, _c1, _beta);
                            _out2 = vmlaq_f32(_out2, _c2, _beta);
                            _out3 = vmlaq_f32(_out3, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 8);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out4 = vaddq_f32(_out4, _c0);
                            _out5 = vaddq_f32(_out5, _c1);
                            _out6 = vaddq_f32(_out6, _c2);
                            _out7 = vaddq_f32(_out7, _c3);
                        }
                        else
                        {
                            _out4 = vmlaq_f32(_out4, _c0, _beta);
                            _out5 = vmlaq_f32(_out5, _c1, _beta);
                            _out6 = vmlaq_f32(_out6, _c2, _beta);
                            _out7 = vmlaq_f32(_out7, _c3, _beta);
                        }
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 5);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 6);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 7);
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c0);
                            _out1 = vaddq_f32(_out1, _c1);
                            _out2 = vaddq_f32(_out2, _c2);
                            _out3 = vaddq_f32(_out3, _c3);
                            _out4 = vaddq_f32(_out4, _c4);
                            _out5 = vaddq_f32(_out5, _c5);
                            _out6 = vaddq_f32(_out6, _c6);
                            _out7 = vaddq_f32(_out7, _c7);
                        }
                        else
                        {
                            _out0 = vmlaq_f32(_out0, _c0, _beta);
                            _out1 = vmlaq_f32(_out1, _c1, _beta);
                            _out2 = vmlaq_f32(_out2, _c2, _beta);
                            _out3 = vmlaq_f32(_out3, _c3, _beta);
                            _out4 = vmlaq_f32(_out4, _c4, _beta);
                            _out5 = vmlaq_f32(_out5, _c5, _beta);
                            _out6 = vmlaq_f32(_out6, _c6, _beta);
                            _out7 = vmlaq_f32(_out7, _c7, _beta);
                        }
                        pC += 4;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_f32(_c, _beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    _out2 = vaddq_f32(_out2, _c);
                    _out3 = vaddq_f32(_out3, _c);
                    _out4 = vaddq_f32(_out4, _c);
                    _out5 = vaddq_f32(_out5, _c);
                    _out6 = vaddq_f32(_out6, _c);
                    _out7 = vaddq_f32(_out7, _c);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
                _out4 = vmulq_f32(_out4, _alpha);
                _out5 = vmulq_f32(_out5, _alpha);
                _out6 = vmulq_f32(_out6, _alpha);
                _out7 = vmulq_f32(_out7, _alpha);
            }
            if (out_elempack == 4)
            {
                transpose4x4_ps(_out0, _out1, _out2, _out3);
                transpose4x4_ps(_out4, _out5, _out6, _out7);
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out1));
                vst1_u16(p0 + 8, float2bfloat(_out2));
                vst1_u16(p0 + 12, float2bfloat(_out3));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out4));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_out5));
                vst1_u16(p0 + out_hstep * 4 + 8, float2bfloat(_out6));
                vst1_u16(p0 + out_hstep * 4 + 12, float2bfloat(_out7));
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + out_hstep, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_out3));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out4));
                vst1_u16(p0 + out_hstep * 5, float2bfloat(_out5));
                vst1_u16(p0 + out_hstep * 6, float2bfloat(_out6));
                vst1_u16(p0 + out_hstep * 7, float2bfloat(_out7));
                p0 += 4;
            }
            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _p0 = vld1q_f32(pp);
            float32x4_t _p1 = vld1q_f32(pp + 4);
            float32x4_t _p2 = vld1q_f32(pp + 8);
            float32x4_t _p3 = vld1q_f32(pp + 12);
            float32x4x2_t _r01 = vzipq_f32(_p0, _p1);
            float32x4x2_t _r23 = vzipq_f32(_p2, _p3);
            float32x2_t _out0 = vget_low_f32(_r01.val[0]);
            float32x2_t _out1 = vget_high_f32(_r01.val[0]);
            float32x2_t _out2 = vget_low_f32(_r01.val[1]);
            float32x2_t _out3 = vget_high_f32(_r01.val[1]);
            float32x2_t _out4 = vget_low_f32(_r23.val[0]);
            float32x2_t _out5 = vget_high_f32(_r23.val[0]);
            float32x2_t _out6 = vget_low_f32(_r23.val[1]);
            float32x2_t _out7 = vget_high_f32(_r23.val[1]);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vdup_lane_f32(vget_low_f32(_c0), 0));
                    _out1 = vadd_f32(_out1, vdup_lane_f32(vget_low_f32(_c0), 1));
                    _out2 = vadd_f32(_out2, vdup_lane_f32(vget_high_f32(_c0), 0));
                    _out3 = vadd_f32(_out3, vdup_lane_f32(vget_high_f32(_c0), 1));
                    _out4 = vadd_f32(_out4, vdup_lane_f32(vget_low_f32(_c1), 0));
                    _out5 = vadd_f32(_out5, vdup_lane_f32(vget_low_f32(_c1), 1));
                    _out6 = vadd_f32(_out6, vdup_lane_f32(vget_high_f32(_c1), 0));
                    _out7 = vadd_f32(_out7, vdup_lane_f32(vget_high_f32(_c1), 1));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4x2_t _c01 = vzipq_f32(vld1q_f32(pC), vld1q_f32(pC + 4));
                        float32x4x2_t _c23 = vzipq_f32(vld1q_f32(pC + c_hstep * 4), vld1q_f32(pC + c_hstep * 4 + 4));
                        if (beta == 1.f)
                        {
                            _out0 = vadd_f32(_out0, vget_low_f32(_c01.val[0]));
                            _out1 = vadd_f32(_out1, vget_high_f32(_c01.val[0]));
                            _out2 = vadd_f32(_out2, vget_low_f32(_c01.val[1]));
                            _out3 = vadd_f32(_out3, vget_high_f32(_c01.val[1]));
                            _out4 = vadd_f32(_out4, vget_low_f32(_c23.val[0]));
                            _out5 = vadd_f32(_out5, vget_high_f32(_c23.val[0]));
                            _out6 = vadd_f32(_out6, vget_low_f32(_c23.val[1]));
                            _out7 = vadd_f32(_out7, vget_high_f32(_c23.val[1]));
                        }
                        else
                        {
                            _out0 = vmla_n_f32(_out0, vget_low_f32(_c01.val[0]), beta);
                            _out1 = vmla_n_f32(_out1, vget_high_f32(_c01.val[0]), beta);
                            _out2 = vmla_n_f32(_out2, vget_low_f32(_c01.val[1]), beta);
                            _out3 = vmla_n_f32(_out3, vget_high_f32(_c01.val[1]), beta);
                            _out4 = vmla_n_f32(_out4, vget_low_f32(_c23.val[0]), beta);
                            _out5 = vmla_n_f32(_out5, vget_high_f32(_c23.val[0]), beta);
                            _out6 = vmla_n_f32(_out6, vget_low_f32(_c23.val[1]), beta);
                            _out7 = vmla_n_f32(_out7, vget_high_f32(_c23.val[1]), beta);
                        }
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        float32x2_t _c0 = vld1_f32(pC);
                        float32x2_t _c1 = vld1_f32(pC + c_hstep);
                        float32x2_t _c2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _c3 = vld1_f32(pC + c_hstep * 3);
                        float32x2_t _c4 = vld1_f32(pC + c_hstep * 4);
                        float32x2_t _c5 = vld1_f32(pC + c_hstep * 5);
                        float32x2_t _c6 = vld1_f32(pC + c_hstep * 6);
                        float32x2_t _c7 = vld1_f32(pC + c_hstep * 7);
                        if (beta == 1.f)
                        {
                            _out0 = vadd_f32(_out0, _c0);
                            _out1 = vadd_f32(_out1, _c1);
                            _out2 = vadd_f32(_out2, _c2);
                            _out3 = vadd_f32(_out3, _c3);
                            _out4 = vadd_f32(_out4, _c4);
                            _out5 = vadd_f32(_out5, _c5);
                            _out6 = vadd_f32(_out6, _c6);
                            _out7 = vadd_f32(_out7, _c7);
                        }
                        else
                        {
                            _out0 = vmla_n_f32(_out0, _c0, beta);
                            _out1 = vmla_n_f32(_out1, _c1, beta);
                            _out2 = vmla_n_f32(_out2, _c2, beta);
                            _out3 = vmla_n_f32(_out3, _c3, beta);
                            _out4 = vmla_n_f32(_out4, _c4, beta);
                            _out5 = vmla_n_f32(_out5, _c5, beta);
                            _out6 = vmla_n_f32(_out6, _c6, beta);
                            _out7 = vmla_n_f32(_out7, _c7, beta);
                        }
                        pC += 2;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    _out0 = vadd_f32(_out0, _c);
                    _out1 = vadd_f32(_out1, _c);
                    _out2 = vadd_f32(_out2, _c);
                    _out3 = vadd_f32(_out3, _c);
                    _out4 = vadd_f32(_out4, _c);
                    _out5 = vadd_f32(_out5, _c);
                    _out6 = vadd_f32(_out6, _c);
                    _out7 = vadd_f32(_out7, _c);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
                _out1 = vmul_n_f32(_out1, alpha);
                _out2 = vmul_n_f32(_out2, alpha);
                _out3 = vmul_n_f32(_out3, alpha);
                _out4 = vmul_n_f32(_out4, alpha);
                _out5 = vmul_n_f32(_out5, alpha);
                _out6 = vmul_n_f32(_out6, alpha);
                _out7 = vmul_n_f32(_out7, alpha);
            }
            if (out_elempack == 4)
            {
                float32x4x2_t _out01 = vtrnq_f32(vcombine_f32(_out0, _out2), vcombine_f32(_out1, _out3));
                float32x4x2_t _out23 = vtrnq_f32(vcombine_f32(_out4, _out6), vcombine_f32(_out5, _out7));
                vst1_u16(p0, float2bfloat(_out01.val[0]));
                vst1_u16(p0 + 4, float2bfloat(_out01.val[1]));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out23.val[0]));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_out23.val[1]));
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out0, _out0))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out1, _out1))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 2), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out2, _out2))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 3), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out3, _out3))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 4), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out4, _out4))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 5), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out5, _out5))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 6), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out6, _out6))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 7), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out7, _out7))), 0);
                p0 += 2;
            }
            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            float f2 = pp[2];
            float f3 = pp[3];
            float f4 = pp[4];
            float f5 = pp[5];
            float f6 = pp[6];
            float f7 = pp[7];
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += vgetq_lane_f32(_c0, 0);
                    f1 += vgetq_lane_f32(_c0, 1);
                    f2 += vgetq_lane_f32(_c0, 2);
                    f3 += vgetq_lane_f32(_c0, 3);
                    f4 += vgetq_lane_f32(_c1, 0);
                    f5 += vgetq_lane_f32(_c1, 1);
                    f6 += vgetq_lane_f32(_c1, 2);
                    f7 += vgetq_lane_f32(_c1, 3);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep * 4);
                        f0 += vgetq_lane_f32(_c0, 0) * beta;
                        f1 += vgetq_lane_f32(_c0, 1) * beta;
                        f2 += vgetq_lane_f32(_c0, 2) * beta;
                        f3 += vgetq_lane_f32(_c0, 3) * beta;
                        f4 += vgetq_lane_f32(_c1, 0) * beta;
                        f5 += vgetq_lane_f32(_c1, 1) * beta;
                        f6 += vgetq_lane_f32(_c1, 2) * beta;
                        f7 += vgetq_lane_f32(_c1, 3) * beta;
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        f0 += pC[0] * beta;
                        f1 += pC[c_hstep] * beta;
                        f2 += pC[c_hstep * 2] * beta;
                        f3 += pC[c_hstep * 3] * beta;
                        f4 += pC[c_hstep * 4] * beta;
                        f5 += pC[c_hstep * 5] * beta;
                        f6 += pC[c_hstep * 6] * beta;
                        f7 += pC[c_hstep * 7] * beta;
                        pC++;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    const float c = pC[0] * beta;
                    f0 += c;
                    f1 += c;
                    f2 += c;
                    f3 += c;
                    f4 += c;
                    f5 += c;
                    f6 += c;
                    f7 += c;
                    pC++;
                }
            }
            if (out_elempack == 4)
            {
                float32x4_t _out0 = vdupq_n_f32(f0 * alpha);
                _out0 = vsetq_lane_f32(f1 * alpha, _out0, 1);
                _out0 = vsetq_lane_f32(f2 * alpha, _out0, 2);
                _out0 = vsetq_lane_f32(f3 * alpha, _out0, 3);
                float32x4_t _out1 = vdupq_n_f32(f4 * alpha);
                _out1 = vsetq_lane_f32(f5 * alpha, _out1, 1);
                _out1 = vsetq_lane_f32(f6 * alpha, _out1, 2);
                _out1 = vsetq_lane_f32(f7 * alpha, _out1, 3);
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out1));
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                p0[0] = float32_to_bfloat16(f0 * alpha);
                p0[out_hstep] = float32_to_bfloat16(f1 * alpha);
                p0[out_hstep * 2] = float32_to_bfloat16(f2 * alpha);
                p0[out_hstep * 3] = float32_to_bfloat16(f3 * alpha);
                p0[out_hstep * 4] = float32_to_bfloat16(f4 * alpha);
                p0[out_hstep * 5] = float32_to_bfloat16(f5 * alpha);
                p0[out_hstep * 6] = float32_to_bfloat16(f6 * alpha);
                p0[out_hstep * 7] = float32_to_bfloat16(f7 * alpha);
                p0++;
            }
            pp += 8;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        float32x4_t _c0123;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0123 = vdupq_n_f32(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0123 = vld1q_f32(pC);
                if (beta != 1.f)
                    _c0123 = vmulq_n_f32(_c0123, beta);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 16);
            float32x4_t _out2 = vld1q_f32(pp + 4);
            float32x4_t _out3 = vld1q_f32(pp + 20);
            float32x4_t _out4 = vld1q_f32(pp + 8);
            float32x4_t _out5 = vld1q_f32(pp + 24);
            float32x4_t _out6 = vld1q_f32(pp + 12);
            float32x4_t _out7 = vld1q_f32(pp + 28);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c = vdupq_lane_f32(vget_low_f32(_c0123), 0);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    _c = vdupq_lane_f32(vget_low_f32(_c0123), 1);
                    _out2 = vaddq_f32(_out2, _c);
                    _out3 = vaddq_f32(_out3, _c);
                    _c = vdupq_lane_f32(vget_high_f32(_c0123), 0);
                    _out4 = vaddq_f32(_out4, _c);
                    _out5 = vaddq_f32(_out5, _c);
                    _c = vdupq_lane_f32(vget_high_f32(_c0123), 1);
                    _out6 = vaddq_f32(_out6, _c);
                    _out7 = vaddq_f32(_out7, _c);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta != 1.f)
                        {
                            _c0 = vmulq_n_f32(_c0, beta);
                            _c1 = vmulq_n_f32(_c1, beta);
                            _c2 = vmulq_n_f32(_c2, beta);
                            _c3 = vmulq_n_f32(_c3, beta);
                        }
                        _out0 = vaddq_f32(_out0, _c0);
                        _out2 = vaddq_f32(_out2, _c1);
                        _out4 = vaddq_f32(_out4, _c2);
                        _out6 = vaddq_f32(_out6, _c3);
                        _c0 = vld1q_f32(pC + 16);
                        _c1 = vld1q_f32(pC + 20);
                        _c2 = vld1q_f32(pC + 24);
                        _c3 = vld1q_f32(pC + 28);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta != 1.f)
                        {
                            _c0 = vmulq_n_f32(_c0, beta);
                            _c1 = vmulq_n_f32(_c1, beta);
                            _c2 = vmulq_n_f32(_c2, beta);
                            _c3 = vmulq_n_f32(_c3, beta);
                        }
                        _out1 = vaddq_f32(_out1, _c0);
                        _out3 = vaddq_f32(_out3, _c1);
                        _out5 = vaddq_f32(_out5, _c2);
                        _out7 = vaddq_f32(_out7, _c3);
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        if (beta != 1.f)
                        {
                            _c0 = vmulq_n_f32(_c0, beta);
                            _c1 = vmulq_n_f32(_c1, beta);
                            _c2 = vmulq_n_f32(_c2, beta);
                            _c3 = vmulq_n_f32(_c3, beta);
                            _c4 = vmulq_n_f32(_c4, beta);
                            _c5 = vmulq_n_f32(_c5, beta);
                            _c6 = vmulq_n_f32(_c6, beta);
                            _c7 = vmulq_n_f32(_c7, beta);
                        }
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c1);
                        _out2 = vaddq_f32(_out2, _c2);
                        _out3 = vaddq_f32(_out3, _c3);
                        _out4 = vaddq_f32(_out4, _c4);
                        _out5 = vaddq_f32(_out5, _c5);
                        _out6 = vaddq_f32(_out6, _c6);
                        _out7 = vaddq_f32(_out7, _c7);
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vld1q_f32(pC);
                    float32x4_t _cc1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _cc0 = vmulq_n_f32(_cc0, beta);
                        _cc1 = vmulq_n_f32(_cc1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out2 = vaddq_f32(_out2, _cc0);
                    _out4 = vaddq_f32(_out4, _cc0);
                    _out6 = vaddq_f32(_out6, _cc0);
                    _out1 = vaddq_f32(_out1, _cc1);
                    _out3 = vaddq_f32(_out3, _cc1);
                    _out5 = vaddq_f32(_out5, _cc1);
                    _out7 = vaddq_f32(_out7, _cc1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
                _out4 = vmulq_f32(_out4, _alpha);
                _out5 = vmulq_f32(_out5, _alpha);
                _out6 = vmulq_f32(_out6, _alpha);
                _out7 = vmulq_f32(_out7, _alpha);
            }

            if (out_elempack == 4)
            {
                transpose4x4_ps(_out0, _out2, _out4, _out6);
                transpose4x4_ps(_out1, _out3, _out5, _out7);
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out2));
                vst1_u16(p0 + 8, float2bfloat(_out4));
                vst1_u16(p0 + 12, float2bfloat(_out6));
                vst1_u16(p0 + 16, float2bfloat(_out1));
                vst1_u16(p0 + 20, float2bfloat(_out3));
                vst1_u16(p0 + 24, float2bfloat(_out5));
                vst1_u16(p0 + 28, float2bfloat(_out7));
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep + 4, float2bfloat(_out3));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_out4));
                vst1_u16(p0 + out_hstep * 2 + 4, float2bfloat(_out5));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_out6));
                vst1_u16(p0 + out_hstep * 3 + 4, float2bfloat(_out7));
                p0 += 8;
            }

            pp += 32;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c = vdupq_lane_f32(vget_low_f32(_c0123), 0);
                    _out0 = vaddq_f32(_out0, _c);
                    _c = vdupq_lane_f32(vget_low_f32(_c0123), 1);
                    _out1 = vaddq_f32(_out1, _c);
                    _c = vdupq_lane_f32(vget_high_f32(_c0123), 0);
                    _out2 = vaddq_f32(_out2, _c);
                    _c = vdupq_lane_f32(vget_high_f32(_c0123), 1);
                    _out3 = vaddq_f32(_out3, _c);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta != 1.f)
                        {
                            _c0 = vmulq_n_f32(_c0, beta);
                            _c1 = vmulq_n_f32(_c1, beta);
                            _c2 = vmulq_n_f32(_c2, beta);
                            _c3 = vmulq_n_f32(_c3, beta);
                        }
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c1);
                        _out2 = vaddq_f32(_out2, _c2);
                        _out3 = vaddq_f32(_out3, _c3);
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 3);
                        if (beta != 1.f)
                        {
                            _c0 = vmulq_n_f32(_c0, beta);
                            _c1 = vmulq_n_f32(_c1, beta);
                            _c2 = vmulq_n_f32(_c2, beta);
                            _c3 = vmulq_n_f32(_c3, beta);
                        }
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c1);
                        _out2 = vaddq_f32(_out2, _c2);
                        _out3 = vaddq_f32(_out3, _c3);
                        pC += 4;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vld1q_f32(pC);
                    if (beta != 1.f)
                        _cc0 = vmulq_n_f32(_cc0, beta);
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out1 = vaddq_f32(_out1, _cc0);
                    _out2 = vaddq_f32(_out2, _cc0);
                    _out3 = vaddq_f32(_out3, _cc0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
            }

            if (out_elempack == 4)
            {
                transpose4x4_ps(_out0, _out1, _out2, _out3);
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out1));
                vst1_u16(p0 + 8, float2bfloat(_out2));
                vst1_u16(p0 + 12, float2bfloat(_out3));
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + out_hstep, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_out3));
                p0 += 4;
            }

            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c01 = vcombine_f32(vdup_lane_f32(vget_low_f32(_c0123), 0), vdup_lane_f32(vget_low_f32(_c0123), 1));
                    float32x4_t _c23 = vcombine_f32(vdup_lane_f32(vget_high_f32(_c0123), 0), vdup_lane_f32(vget_high_f32(_c0123), 1));
                    _out0 = vaddq_f32(_out0, _c01);
                    _out1 = vaddq_f32(_out1, _c23);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4x2_t _c01 = vzipq_f32(vld1q_f32(pC), vld1q_f32(pC + 4));
                        if (beta != 1.f)
                        {
                            _c01.val[0] = vmulq_n_f32(_c01.val[0], beta);
                            _c01.val[1] = vmulq_n_f32(_c01.val[1], beta);
                        }
                        _out0 = vaddq_f32(_out0, _c01.val[0]);
                        _out1 = vaddq_f32(_out1, _c01.val[1]);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c01 = vcombine_f32(vld1_f32(pC), vld1_f32(pC + c_hstep));
                        float32x4_t _c23 = vcombine_f32(vld1_f32(pC + c_hstep * 2), vld1_f32(pC + c_hstep * 3));
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c01);
                            _out1 = vaddq_f32(_out1, _c23);
                        }
                        else
                        {
                            _out0 = vmlaq_n_f32(_out0, _c01, beta);
                            _out1 = vmlaq_n_f32(_out1, _c23, beta);
                        }
                        pC += 2;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    float32x4_t _cc0 = vcombine_f32(_c, _c);
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out1 = vaddq_f32(_out1, _cc0);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }
            if (out_elempack == 4)
            {
                float32x4x2_t _out01 = vuzpq_f32(_out0, _out1);
                vst1_u16(p0, float2bfloat(_out01.val[0]));
                vst1_u16(p0 + 4, float2bfloat(_out01.val[1]));
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_low_f32(_out0), vget_low_f32(_out0)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_high_f32(_out0), vget_high_f32(_out0)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 2), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_low_f32(_out1), vget_low_f32(_out1)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 3), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_high_f32(_out1), vget_high_f32(_out1)))), 0);
                p0 += 2;
            }

            pp += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _out0 = vld1q_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c = vld1q_f32(pC);
                        if (beta == 1.f)
                            _out0 = vaddq_f32(_out0, _c);
                        else
                            _out0 = vmlaq_n_f32(_out0, _c, beta);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c = vdupq_n_f32(pC[0]);
                        _c = vsetq_lane_f32(pC[c_hstep], _c, 1);
                        _c = vsetq_lane_f32(pC[c_hstep * 2], _c, 2);
                        _c = vsetq_lane_f32(pC[c_hstep * 3], _c, 3);
                        if (beta == 1.f)
                            _out0 = vaddq_f32(_out0, _c);
                        else
                            _out0 = vmlaq_n_f32(_out0, _c, beta);
                        pC++;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vdupq_n_f32(pC[0] * beta);
                    _out0 = vaddq_f32(_out0, _cc0);
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                p0[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 0));
                (p0 + out_hstep)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 1));
                (p0 + out_hstep * 2)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 2));
                (p0 + out_hstep * 3)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 3));
                p0++;
            }

            pp += 4;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
#if __ARM_NEON
        float32x4_t _c0 = vdupq_n_f32(0.f);
        float32x4_t _c1 = vdupq_n_f32(0.f);
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
#if __ARM_NEON
                _c0 = vdupq_n_f32(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
#if __ARM_NEON
                _c0 = vdupq_n_f32(pC[0] * beta);
                _c1 = vdupq_n_f32(pC[1] * beta);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

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
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 8);
            float32x4_t _out2 = vld1q_f32(pp + 4);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                    _out2 = vaddq_f32(_out2, _c0);
                    _out3 = vaddq_f32(_out3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                    _out2 = vaddq_f32(_out2, _c1);
                    _out3 = vaddq_f32(_out3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                    float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                        _c2 = vmulq_n_f32(_c2, beta);
                        _c3 = vmulq_n_f32(_c3, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    _out2 = vaddq_f32(_out2, _c2);
                    _out3 = vaddq_f32(_out3, _c3);
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out2 = vaddq_f32(_out2, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    _out3 = vaddq_f32(_out3, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
            }

            vst1_u16(p0, float2bfloat(_out0));
            vst1_u16(p0 + 4, float2bfloat(_out1));
            vst1_u16(p0 + out_hstep, float2bfloat(_out2));
            vst1_u16(p0 + out_hstep + 4, float2bfloat(_out3));

            pp += 16;
            p0 += 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_n_f32(_c, beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            vst1_u16(p0, float2bfloat(_out0));
            vst1_u16(p0 + out_hstep, float2bfloat(_out1));

            pp += 8;
            p0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);
            float32x2_t _out1 = vld1_f32(pp + 2);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c0));
                    _out1 = vadd_f32(_out1, vget_low_f32(_c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c0));
                    _out1 = vadd_f32(_out1, vget_low_f32(_c1));
                }
                if (broadcast_type_C == 3)
                {
                    float32x2_t _c0 = vld1_f32(pC);
                    float32x2_t _c1 = vld1_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _out0 = vadd_f32(_out0, _c0);
                        _out1 = vadd_f32(_out1, _c1);
                    }
                    else
                    {
                        _out0 = vmla_n_f32(_out0, _c0, beta);
                        _out1 = vmla_n_f32(_out1, _c1, beta);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    _out0 = vadd_f32(_out0, _c);
                    _out1 = vadd_f32(_out1, _c);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
                _out1 = vmul_n_f32(_out1, alpha);
            }

            vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out0, _out0))), 0);
            vst1_lane_u32((unsigned int*)(p0 + out_hstep), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out1, _out1))), 0);

            pp += 4;
            p0 += 2;
        }
#endif // __ARM_NEON

        for (; jj < max_jj; jj += 1)
        {
            float out00 = pp[0];
            float out10 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out10 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out10 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += pC[0] * beta;
                    out10 += pC[c_hstep] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0] * beta;
                    out00 += c;
                    out10 += c;
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out10 *= alpha;
            }

            p0[0] = float32_to_bfloat16(out00);
            p0[out_hstep] = float32_to_bfloat16(out10);

            pp += 2;
            p0++;
        }
    }
    for (; ii < max_ii; ii++)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
#if __ARM_NEON
        float32x4_t _c0 = vdupq_n_f32(0.f);
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
#if __ARM_NEON
                _c0 = vdupq_n_f32(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
#if __ARM_NEON
                _c0 = vdupq_n_f32(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[0] * beta;
        }

        int jj = 0;
#if __ARM_NEON
        for (; jj + 15 < max_jj; jj += 16)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                    _out2 = vaddq_f32(_out2, _c0);
                    _out3 = vaddq_f32(_out3, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + 8);
                    float32x4_t _c3 = vld1q_f32(pC + 12);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                        _c2 = vmulq_n_f32(_c2, beta);
                        _c3 = vmulq_n_f32(_c3, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    _out2 = vaddq_f32(_out2, _c2);
                    _out3 = vaddq_f32(_out3, _c3);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
            }

            vst1_u16(p0, float2bfloat(_out0));
            vst1_u16(p0 + 4, float2bfloat(_out1));
            vst1_u16(p0 + 8, float2bfloat(_out2));
            vst1_u16(p0 + 12, float2bfloat(_out3));

            pp += 16;
            p0 += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            vst1_u16(p0, float2bfloat(_out0));
            vst1_u16(p0 + 4, float2bfloat(_out1));

            pp += 8;
            p0 += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_n_f32(_c, beta);
                    _out0 = vaddq_f32(_out0, _c);
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_n_f32(_c, beta);
                    _out0 = vaddq_f32(_out0, _c);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
            }

            vst1_u16(p0, float2bfloat(_out0));

            pp += 4;
            p0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c0));
                }
                if (broadcast_type_C == 3)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta == 1.f)
                        _out0 = vadd_f32(_out0, _c);
                    else
                        _out0 = vmla_n_f32(_out0, _c, beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    _out0 = vadd_f32(_out0, _c);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
            }

            vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out0, _out0))), 0);

            pp += 2;
            p0 += 2;
        }
#endif // __ARM_NEON

        for (; jj < max_jj; jj += 1)
        {
            float out00 = pp[0];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += pC[0] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += pC[0] * beta;
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
            }

            p0[0] = float32_to_bfloat16(out00);

            pp++;
            p0++;
        }
    }
}

static void transpose_unpack_output_tile_wq_int8_bf16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    const float* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack;

        float32x4_t _c0;
        float32x4_t _c1;
        float32x4_t _alpha = vdupq_n_f32(alpha);
        float32x4_t _beta = vdupq_n_f32(beta);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(pC[0] * beta);
                _c1 = _c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = vld1q_f32(pC);
                _c1 = vld1q_f32(pC + 4);
                if (beta != 1.f)
                {
                    _c0 = vmulq_f32(_c0, _beta);
                    _c1 = vmulq_f32(_c1, _beta);
                }
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);
            float32x4_t _out4 = vld1q_f32(pp + 16);
            float32x4_t _out5 = vld1q_f32(pp + 20);
            float32x4_t _out6 = vld1q_f32(pp + 24);
            float32x4_t _out7 = vld1q_f32(pp + 28);

            float32x4_t _out8 = vld1q_f32(pp + 32);
            float32x4_t _out9 = vld1q_f32(pp + 36);
            float32x4_t _outa = vld1q_f32(pp + 40);
            float32x4_t _outb = vld1q_f32(pp + 44);
            float32x4_t _outc = vld1q_f32(pp + 48);
            float32x4_t _outd = vld1q_f32(pp + 52);
            float32x4_t _oute = vld1q_f32(pp + 56);
            float32x4_t _outf = vld1q_f32(pp + 60);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, vdupq_laneq_f32(_c0, 0));
                    _out8 = vaddq_f32(_out8, vdupq_laneq_f32(_c0, 0));
                    _out1 = vaddq_f32(_out1, vdupq_laneq_f32(_c0, 1));
                    _out9 = vaddq_f32(_out9, vdupq_laneq_f32(_c0, 1));
                    _out2 = vaddq_f32(_out2, vdupq_laneq_f32(_c0, 2));
                    _outa = vaddq_f32(_outa, vdupq_laneq_f32(_c0, 2));
                    _out3 = vaddq_f32(_out3, vdupq_laneq_f32(_c0, 3));
                    _outb = vaddq_f32(_outb, vdupq_laneq_f32(_c0, 3));
                    _out4 = vaddq_f32(_out4, vdupq_laneq_f32(_c1, 0));
                    _outc = vaddq_f32(_outc, vdupq_laneq_f32(_c1, 0));
                    _out5 = vaddq_f32(_out5, vdupq_laneq_f32(_c1, 1));
                    _outd = vaddq_f32(_outd, vdupq_laneq_f32(_c1, 1));
                    _out6 = vaddq_f32(_out6, vdupq_laneq_f32(_c1, 2));
                    _oute = vaddq_f32(_oute, vdupq_laneq_f32(_c1, 2));
                    _out7 = vaddq_f32(_out7, vdupq_laneq_f32(_c1, 3));
                    _outf = vaddq_f32(_outf, vdupq_laneq_f32(_c1, 3));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c0);
                            _out1 = vaddq_f32(_out1, _c1);
                            _out2 = vaddq_f32(_out2, _c2);
                            _out3 = vaddq_f32(_out3, _c3);
                        }
                        else
                        {
                            _out0 = vmlaq_f32(_out0, _c0, _beta);
                            _out1 = vmlaq_f32(_out1, _c1, _beta);
                            _out2 = vmlaq_f32(_out2, _c2, _beta);
                            _out3 = vmlaq_f32(_out3, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + 16);
                        _c1 = vld1q_f32(pC + 20);
                        _c2 = vld1q_f32(pC + 24);
                        _c3 = vld1q_f32(pC + 28);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out8 = vaddq_f32(_out8, _c0);
                            _out9 = vaddq_f32(_out9, _c1);
                            _outa = vaddq_f32(_outa, _c2);
                            _outb = vaddq_f32(_outb, _c3);
                        }
                        else
                        {
                            _out8 = vmlaq_f32(_out8, _c0, _beta);
                            _out9 = vmlaq_f32(_out9, _c1, _beta);
                            _outa = vmlaq_f32(_outa, _c2, _beta);
                            _outb = vmlaq_f32(_outb, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 8);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out4 = vaddq_f32(_out4, _c0);
                            _out5 = vaddq_f32(_out5, _c1);
                            _out6 = vaddq_f32(_out6, _c2);
                            _out7 = vaddq_f32(_out7, _c3);
                        }
                        else
                        {
                            _out4 = vmlaq_f32(_out4, _c0, _beta);
                            _out5 = vmlaq_f32(_out5, _c1, _beta);
                            _out6 = vmlaq_f32(_out6, _c2, _beta);
                            _out7 = vmlaq_f32(_out7, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4 + 16);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 20);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 24);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 28);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _outc = vaddq_f32(_outc, _c0);
                            _outd = vaddq_f32(_outd, _c1);
                            _oute = vaddq_f32(_oute, _c2);
                            _outf = vaddq_f32(_outf, _c3);
                        }
                        else
                        {
                            _outc = vmlaq_f32(_outc, _c0, _beta);
                            _outd = vmlaq_f32(_outd, _c1, _beta);
                            _oute = vmlaq_f32(_oute, _c2, _beta);
                            _outf = vmlaq_f32(_outf, _c3, _beta);
                        }
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 5);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 6);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 7);
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c0);
                            _out1 = vaddq_f32(_out1, _c1);
                            _out2 = vaddq_f32(_out2, _c2);
                            _out3 = vaddq_f32(_out3, _c3);
                            _out4 = vaddq_f32(_out4, _c4);
                            _out5 = vaddq_f32(_out5, _c5);
                            _out6 = vaddq_f32(_out6, _c6);
                            _out7 = vaddq_f32(_out7, _c7);
                        }
                        else
                        {
                            _out0 = vmlaq_f32(_out0, _c0, _beta);
                            _out1 = vmlaq_f32(_out1, _c1, _beta);
                            _out2 = vmlaq_f32(_out2, _c2, _beta);
                            _out3 = vmlaq_f32(_out3, _c3, _beta);
                            _out4 = vmlaq_f32(_out4, _c4, _beta);
                            _out5 = vmlaq_f32(_out5, _c5, _beta);
                            _out6 = vmlaq_f32(_out6, _c6, _beta);
                            _out7 = vmlaq_f32(_out7, _c7, _beta);
                        }
                        _c0 = vld1q_f32(pC + 4);
                        _c1 = vld1q_f32(pC + c_hstep + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 2 + 4);
                        _c3 = vld1q_f32(pC + c_hstep * 3 + 4);
                        _c4 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c5 = vld1q_f32(pC + c_hstep * 5 + 4);
                        _c6 = vld1q_f32(pC + c_hstep * 6 + 4);
                        _c7 = vld1q_f32(pC + c_hstep * 7 + 4);
                        if (beta == 1.f)
                        {
                            _out8 = vaddq_f32(_out8, _c0);
                            _out9 = vaddq_f32(_out9, _c1);
                            _outa = vaddq_f32(_outa, _c2);
                            _outb = vaddq_f32(_outb, _c3);
                            _outc = vaddq_f32(_outc, _c4);
                            _outd = vaddq_f32(_outd, _c5);
                            _oute = vaddq_f32(_oute, _c6);
                            _outf = vaddq_f32(_outf, _c7);
                        }
                        else
                        {
                            _out8 = vmlaq_f32(_out8, _c0, _beta);
                            _out9 = vmlaq_f32(_out9, _c1, _beta);
                            _outa = vmlaq_f32(_outa, _c2, _beta);
                            _outb = vmlaq_f32(_outb, _c3, _beta);
                            _outc = vmlaq_f32(_outc, _c4, _beta);
                            _outd = vmlaq_f32(_outd, _c5, _beta);
                            _oute = vmlaq_f32(_oute, _c6, _beta);
                            _outf = vmlaq_f32(_outf, _c7, _beta);
                        }
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_f32(_c, _beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    _out2 = vaddq_f32(_out2, _c);
                    _out3 = vaddq_f32(_out3, _c);
                    _out4 = vaddq_f32(_out4, _c);
                    _out5 = vaddq_f32(_out5, _c);
                    _out6 = vaddq_f32(_out6, _c);
                    _out7 = vaddq_f32(_out7, _c);
                    _c = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                        _c = vmulq_f32(_c, _beta);
                    _out8 = vaddq_f32(_out8, _c);
                    _out9 = vaddq_f32(_out9, _c);
                    _outa = vaddq_f32(_outa, _c);
                    _outb = vaddq_f32(_outb, _c);
                    _outc = vaddq_f32(_outc, _c);
                    _outd = vaddq_f32(_outd, _c);
                    _oute = vaddq_f32(_oute, _c);
                    _outf = vaddq_f32(_outf, _c);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
                _out4 = vmulq_f32(_out4, _alpha);
                _out5 = vmulq_f32(_out5, _alpha);
                _out6 = vmulq_f32(_out6, _alpha);
                _out7 = vmulq_f32(_out7, _alpha);
                _out8 = vmulq_f32(_out8, _alpha);
                _out9 = vmulq_f32(_out9, _alpha);
                _outa = vmulq_f32(_outa, _alpha);
                _outb = vmulq_f32(_outb, _alpha);
                _outc = vmulq_f32(_outc, _alpha);
                _outd = vmulq_f32(_outd, _alpha);
                _oute = vmulq_f32(_oute, _alpha);
                _outf = vmulq_f32(_outf, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out1));
                vst1_u16(p0 + 8, float2bfloat(_out2));
                vst1_u16(p0 + 12, float2bfloat(_out3));
                vst1_u16(p0 + 16, float2bfloat(_out4));
                vst1_u16(p0 + 20, float2bfloat(_out5));
                vst1_u16(p0 + 24, float2bfloat(_out6));
                vst1_u16(p0 + 28, float2bfloat(_out7));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out8));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_out9));
                vst1_u16(p0 + out_hstep * 4 + 8, float2bfloat(_outa));
                vst1_u16(p0 + out_hstep * 4 + 12, float2bfloat(_outb));
                vst1_u16(p0 + out_hstep * 4 + 16, float2bfloat(_outc));
                vst1_u16(p0 + out_hstep * 4 + 20, float2bfloat(_outd));
                vst1_u16(p0 + out_hstep * 4 + 24, float2bfloat(_oute));
                vst1_u16(p0 + out_hstep * 4 + 28, float2bfloat(_outf));
            }
            if (out_elempack == 1)
            {
                transpose4x4_ps(_out0, _out1, _out2, _out3);
                transpose4x4_ps(_out4, _out5, _out6, _out7);
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out4));
                vst1_u16(p0 + out_hstep, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep + 4, float2bfloat(_out5));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep * 2 + 4, float2bfloat(_out6));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_out3));
                vst1_u16(p0 + out_hstep * 3 + 4, float2bfloat(_out7));
                transpose4x4_ps(_out8, _out9, _outa, _outb);
                transpose4x4_ps(_outc, _outd, _oute, _outf);
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out8));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_outc));
                vst1_u16(p0 + out_hstep * 5, float2bfloat(_out9));
                vst1_u16(p0 + out_hstep * 5 + 4, float2bfloat(_outd));
                vst1_u16(p0 + out_hstep * 6, float2bfloat(_outa));
                vst1_u16(p0 + out_hstep * 6 + 4, float2bfloat(_oute));
                vst1_u16(p0 + out_hstep * 7, float2bfloat(_outb));
                vst1_u16(p0 + out_hstep * 7 + 4, float2bfloat(_outf));
            }
            pp += 64;
            p0 += out_hstep * 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);
            float32x4_t _out4 = vld1q_f32(pp + 16);
            float32x4_t _out5 = vld1q_f32(pp + 20);
            float32x4_t _out6 = vld1q_f32(pp + 24);
            float32x4_t _out7 = vld1q_f32(pp + 28);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, vdupq_laneq_f32(_c0, 0));
                    _out1 = vaddq_f32(_out1, vdupq_laneq_f32(_c0, 1));
                    _out2 = vaddq_f32(_out2, vdupq_laneq_f32(_c0, 2));
                    _out3 = vaddq_f32(_out3, vdupq_laneq_f32(_c0, 3));
                    _out4 = vaddq_f32(_out4, vdupq_laneq_f32(_c1, 0));
                    _out5 = vaddq_f32(_out5, vdupq_laneq_f32(_c1, 1));
                    _out6 = vaddq_f32(_out6, vdupq_laneq_f32(_c1, 2));
                    _out7 = vaddq_f32(_out7, vdupq_laneq_f32(_c1, 3));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c0);
                            _out1 = vaddq_f32(_out1, _c1);
                            _out2 = vaddq_f32(_out2, _c2);
                            _out3 = vaddq_f32(_out3, _c3);
                        }
                        else
                        {
                            _out0 = vmlaq_f32(_out0, _c0, _beta);
                            _out1 = vmlaq_f32(_out1, _c1, _beta);
                            _out2 = vmlaq_f32(_out2, _c2, _beta);
                            _out3 = vmlaq_f32(_out3, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 8);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _out4 = vaddq_f32(_out4, _c0);
                            _out5 = vaddq_f32(_out5, _c1);
                            _out6 = vaddq_f32(_out6, _c2);
                            _out7 = vaddq_f32(_out7, _c3);
                        }
                        else
                        {
                            _out4 = vmlaq_f32(_out4, _c0, _beta);
                            _out5 = vmlaq_f32(_out5, _c1, _beta);
                            _out6 = vmlaq_f32(_out6, _c2, _beta);
                            _out7 = vmlaq_f32(_out7, _c3, _beta);
                        }
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        float32x4_t _c0 = vld1q_f32(pC);
                        float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 5);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 6);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 7);
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c0);
                            _out1 = vaddq_f32(_out1, _c1);
                            _out2 = vaddq_f32(_out2, _c2);
                            _out3 = vaddq_f32(_out3, _c3);
                            _out4 = vaddq_f32(_out4, _c4);
                            _out5 = vaddq_f32(_out5, _c5);
                            _out6 = vaddq_f32(_out6, _c6);
                            _out7 = vaddq_f32(_out7, _c7);
                        }
                        else
                        {
                            _out0 = vmlaq_f32(_out0, _c0, _beta);
                            _out1 = vmlaq_f32(_out1, _c1, _beta);
                            _out2 = vmlaq_f32(_out2, _c2, _beta);
                            _out3 = vmlaq_f32(_out3, _c3, _beta);
                            _out4 = vmlaq_f32(_out4, _c4, _beta);
                            _out5 = vmlaq_f32(_out5, _c5, _beta);
                            _out6 = vmlaq_f32(_out6, _c6, _beta);
                            _out7 = vmlaq_f32(_out7, _c7, _beta);
                        }
                        pC += 4;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_f32(_c, _beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    _out2 = vaddq_f32(_out2, _c);
                    _out3 = vaddq_f32(_out3, _c);
                    _out4 = vaddq_f32(_out4, _c);
                    _out5 = vaddq_f32(_out5, _c);
                    _out6 = vaddq_f32(_out6, _c);
                    _out7 = vaddq_f32(_out7, _c);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
                _out4 = vmulq_f32(_out4, _alpha);
                _out5 = vmulq_f32(_out5, _alpha);
                _out6 = vmulq_f32(_out6, _alpha);
                _out7 = vmulq_f32(_out7, _alpha);
            }
            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out1));
                vst1_u16(p0 + 8, float2bfloat(_out2));
                vst1_u16(p0 + 12, float2bfloat(_out3));
                vst1_u16(p0 + 16, float2bfloat(_out4));
                vst1_u16(p0 + 20, float2bfloat(_out5));
                vst1_u16(p0 + 24, float2bfloat(_out6));
                vst1_u16(p0 + 28, float2bfloat(_out7));
            }
            if (out_elempack == 1)
            {
                transpose4x4_ps(_out0, _out1, _out2, _out3);
                transpose4x4_ps(_out4, _out5, _out6, _out7);
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out4));
                vst1_u16(p0 + out_hstep, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep + 4, float2bfloat(_out5));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep * 2 + 4, float2bfloat(_out6));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_out3));
                vst1_u16(p0 + out_hstep * 3 + 4, float2bfloat(_out7));
            }
            pp += 32;
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _p0 = vld1q_f32(pp);
            float32x4_t _p1 = vld1q_f32(pp + 4);
            float32x4_t _p2 = vld1q_f32(pp + 8);
            float32x4_t _p3 = vld1q_f32(pp + 12);
            float32x4x2_t _r01 = vzipq_f32(_p0, _p1);
            float32x4x2_t _r23 = vzipq_f32(_p2, _p3);
            float32x2_t _out0 = vget_low_f32(_r01.val[0]);
            float32x2_t _out1 = vget_high_f32(_r01.val[0]);
            float32x2_t _out2 = vget_low_f32(_r01.val[1]);
            float32x2_t _out3 = vget_high_f32(_r01.val[1]);
            float32x2_t _out4 = vget_low_f32(_r23.val[0]);
            float32x2_t _out5 = vget_high_f32(_r23.val[0]);
            float32x2_t _out6 = vget_low_f32(_r23.val[1]);
            float32x2_t _out7 = vget_high_f32(_r23.val[1]);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vdup_lane_f32(vget_low_f32(_c0), 0));
                    _out1 = vadd_f32(_out1, vdup_lane_f32(vget_low_f32(_c0), 1));
                    _out2 = vadd_f32(_out2, vdup_lane_f32(vget_high_f32(_c0), 0));
                    _out3 = vadd_f32(_out3, vdup_lane_f32(vget_high_f32(_c0), 1));
                    _out4 = vadd_f32(_out4, vdup_lane_f32(vget_low_f32(_c1), 0));
                    _out5 = vadd_f32(_out5, vdup_lane_f32(vget_low_f32(_c1), 1));
                    _out6 = vadd_f32(_out6, vdup_lane_f32(vget_high_f32(_c1), 0));
                    _out7 = vadd_f32(_out7, vdup_lane_f32(vget_high_f32(_c1), 1));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        float32x4x2_t _c01 = vzipq_f32(vld1q_f32(pC), vld1q_f32(pC + 4));
                        float32x4x2_t _c23 = vzipq_f32(vld1q_f32(pC + c_hstep * 4), vld1q_f32(pC + c_hstep * 4 + 4));
                        if (beta == 1.f)
                        {
                            _out0 = vadd_f32(_out0, vget_low_f32(_c01.val[0]));
                            _out1 = vadd_f32(_out1, vget_high_f32(_c01.val[0]));
                            _out2 = vadd_f32(_out2, vget_low_f32(_c01.val[1]));
                            _out3 = vadd_f32(_out3, vget_high_f32(_c01.val[1]));
                            _out4 = vadd_f32(_out4, vget_low_f32(_c23.val[0]));
                            _out5 = vadd_f32(_out5, vget_high_f32(_c23.val[0]));
                            _out6 = vadd_f32(_out6, vget_low_f32(_c23.val[1]));
                            _out7 = vadd_f32(_out7, vget_high_f32(_c23.val[1]));
                        }
                        else
                        {
                            _out0 = vmla_n_f32(_out0, vget_low_f32(_c01.val[0]), beta);
                            _out1 = vmla_n_f32(_out1, vget_high_f32(_c01.val[0]), beta);
                            _out2 = vmla_n_f32(_out2, vget_low_f32(_c01.val[1]), beta);
                            _out3 = vmla_n_f32(_out3, vget_high_f32(_c01.val[1]), beta);
                            _out4 = vmla_n_f32(_out4, vget_low_f32(_c23.val[0]), beta);
                            _out5 = vmla_n_f32(_out5, vget_high_f32(_c23.val[0]), beta);
                            _out6 = vmla_n_f32(_out6, vget_low_f32(_c23.val[1]), beta);
                            _out7 = vmla_n_f32(_out7, vget_high_f32(_c23.val[1]), beta);
                        }
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        float32x2_t _c0 = vld1_f32(pC);
                        float32x2_t _c1 = vld1_f32(pC + c_hstep);
                        float32x2_t _c2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _c3 = vld1_f32(pC + c_hstep * 3);
                        float32x2_t _c4 = vld1_f32(pC + c_hstep * 4);
                        float32x2_t _c5 = vld1_f32(pC + c_hstep * 5);
                        float32x2_t _c6 = vld1_f32(pC + c_hstep * 6);
                        float32x2_t _c7 = vld1_f32(pC + c_hstep * 7);
                        if (beta == 1.f)
                        {
                            _out0 = vadd_f32(_out0, _c0);
                            _out1 = vadd_f32(_out1, _c1);
                            _out2 = vadd_f32(_out2, _c2);
                            _out3 = vadd_f32(_out3, _c3);
                            _out4 = vadd_f32(_out4, _c4);
                            _out5 = vadd_f32(_out5, _c5);
                            _out6 = vadd_f32(_out6, _c6);
                            _out7 = vadd_f32(_out7, _c7);
                        }
                        else
                        {
                            _out0 = vmla_n_f32(_out0, _c0, beta);
                            _out1 = vmla_n_f32(_out1, _c1, beta);
                            _out2 = vmla_n_f32(_out2, _c2, beta);
                            _out3 = vmla_n_f32(_out3, _c3, beta);
                            _out4 = vmla_n_f32(_out4, _c4, beta);
                            _out5 = vmla_n_f32(_out5, _c5, beta);
                            _out6 = vmla_n_f32(_out6, _c6, beta);
                            _out7 = vmla_n_f32(_out7, _c7, beta);
                        }
                        pC += 2;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    _out0 = vadd_f32(_out0, _c);
                    _out1 = vadd_f32(_out1, _c);
                    _out2 = vadd_f32(_out2, _c);
                    _out3 = vadd_f32(_out3, _c);
                    _out4 = vadd_f32(_out4, _c);
                    _out5 = vadd_f32(_out5, _c);
                    _out6 = vadd_f32(_out6, _c);
                    _out7 = vadd_f32(_out7, _c);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
                _out1 = vmul_n_f32(_out1, alpha);
                _out2 = vmul_n_f32(_out2, alpha);
                _out3 = vmul_n_f32(_out3, alpha);
                _out4 = vmul_n_f32(_out4, alpha);
                _out5 = vmul_n_f32(_out5, alpha);
                _out6 = vmul_n_f32(_out6, alpha);
                _out7 = vmul_n_f32(_out7, alpha);
            }
            float32x4x2_t _t0 = vuzpq_f32(vcombine_f32(_out0, _out1), vcombine_f32(_out2, _out3));
            float32x4x2_t _t1 = vuzpq_f32(vcombine_f32(_out4, _out5), vcombine_f32(_out6, _out7));
            vst1_u16(p0, float2bfloat(_t0.val[0]));
            vst1_u16(p0 + 4, float2bfloat(_t1.val[0]));
            vst1_u16(p0 + out_hstep, float2bfloat(_t0.val[1]));
            vst1_u16(p0 + out_hstep + 4, float2bfloat(_t1.val[1]));
            pp += 16;
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj++)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep * 4);
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c0);
                            _out1 = vaddq_f32(_out1, _c1);
                        }
                        else
                        {
                            _out0 = vmlaq_n_f32(_out0, _c0, beta);
                            _out1 = vmlaq_n_f32(_out1, _c1, beta);
                        }
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vdupq_n_f32(pC[0]);
                        _c0 = vsetq_lane_f32(pC[c_hstep], _c0, 1);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 2], _c0, 2);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 3], _c0, 3);
                        _c1 = vdupq_n_f32(pC[c_hstep * 4]);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 5], _c1, 1);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 6], _c1, 2);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 7], _c1, 3);
                        if (beta == 1.f)
                        {
                            _out0 = vaddq_f32(_out0, _c0);
                            _out1 = vaddq_f32(_out1, _c1);
                        }
                        else
                        {
                            _out0 = vmlaq_n_f32(_out0, _c0, beta);
                            _out1 = vmlaq_n_f32(_out1, _c1, beta);
                        }
                        pC++;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0] * beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }
            vst1_u16(p0, float2bfloat(_out0));
            vst1_u16(p0 + 4, float2bfloat(_out1));
            pp += 8;
            p0 += out_hstep;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack;
        float32x4_t _c0123 = vdupq_n_f32(0.f);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                _c0123 = vdupq_n_f32(c);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0123 = vld1q_f32(pC);
                if (beta != 1.f)
                    _c0123 = vmulq_n_f32(_c0123, beta);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 16);
            float32x4_t _out2 = vld1q_f32(pp + 4);
            float32x4_t _out3 = vld1q_f32(pp + 20);
            float32x4_t _out4 = vld1q_f32(pp + 8);
            float32x4_t _out5 = vld1q_f32(pp + 24);
            float32x4_t _out6 = vld1q_f32(pp + 12);
            float32x4_t _out7 = vld1q_f32(pp + 28);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0;
                    float32x4_t _c1;
                    float32x4_t _c2;
                    float32x4_t _c3;
                    float32x4_t _c4;
                    float32x4_t _c5;
                    float32x4_t _c6;
                    float32x4_t _c7;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c2 = vld1q_f32(pC + 4);
                        _c4 = vld1q_f32(pC + 8);
                        _c6 = vld1q_f32(pC + 12);
                        _c1 = vld1q_f32(pC + 16);
                        _c3 = vld1q_f32(pC + 20);
                        _c5 = vld1q_f32(pC + 24);
                        _c7 = vld1q_f32(pC + 28);
                        transpose4x4_ps(_c0, _c2, _c4, _c6);
                        transpose4x4_ps(_c1, _c3, _c5, _c7);
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + c_hstep);
                        _c3 = vld1q_f32(pC + c_hstep + 4);
                        _c4 = vld1q_f32(pC + c_hstep * 2);
                        _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        _c6 = vld1q_f32(pC + c_hstep * 3);
                        _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c1);
                        _out2 = vaddq_f32(_out2, _c2);
                        _out3 = vaddq_f32(_out3, _c3);
                        _out4 = vaddq_f32(_out4, _c4);
                        _out5 = vaddq_f32(_out5, _c5);
                        _out6 = vaddq_f32(_out6, _c6);
                        _out7 = vaddq_f32(_out7, _c7);
                    }
                    else
                    {
                        _out0 = vmlaq_n_f32(_out0, _c0, beta);
                        _out1 = vmlaq_n_f32(_out1, _c1, beta);
                        _out2 = vmlaq_n_f32(_out2, _c2, beta);
                        _out3 = vmlaq_n_f32(_out3, _c3, beta);
                        _out4 = vmlaq_n_f32(_out4, _c4, beta);
                        _out5 = vmlaq_n_f32(_out5, _c5, beta);
                        _out6 = vmlaq_n_f32(_out6, _c6, beta);
                        _out7 = vmlaq_n_f32(_out7, _c7, beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vld1q_f32(pC);
                    float32x4_t _cc1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _cc0 = vmulq_n_f32(_cc0, beta);
                        _cc1 = vmulq_n_f32(_cc1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out2 = vaddq_f32(_out2, _cc0);
                    _out4 = vaddq_f32(_out4, _cc0);
                    _out6 = vaddq_f32(_out6, _cc0);
                    _out1 = vaddq_f32(_out1, _cc1);
                    _out3 = vaddq_f32(_out3, _cc1);
                    _out5 = vaddq_f32(_out5, _cc1);
                    _out7 = vaddq_f32(_out7, _cc1);
                    pC += 8;
                }
            }

            if (out_elempack == 1)
            {
                transpose4x4_ps(_out0, _out2, _out4, _out6);
                transpose4x4_ps(_out1, _out3, _out5, _out7);
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_elempack == 1)
                    {
                        _out0 = vaddq_f32(_out0, _c0123);
                        _out1 = vaddq_f32(_out1, _c0123);
                        _out2 = vaddq_f32(_out2, _c0123);
                        _out3 = vaddq_f32(_out3, _c0123);
                        _out4 = vaddq_f32(_out4, _c0123);
                        _out5 = vaddq_f32(_out5, _c0123);
                        _out6 = vaddq_f32(_out6, _c0123);
                        _out7 = vaddq_f32(_out7, _c0123);
                    }
                    else
                    {
                        float32x4_t _c = vdupq_lane_f32(vget_low_f32(_c0123), 0);
                        _out0 = vaddq_f32(_out0, _c);
                        _out1 = vaddq_f32(_out1, _c);
                        _c = vdupq_lane_f32(vget_low_f32(_c0123), 1);
                        _out2 = vaddq_f32(_out2, _c);
                        _out3 = vaddq_f32(_out3, _c);
                        _c = vdupq_lane_f32(vget_high_f32(_c0123), 0);
                        _out4 = vaddq_f32(_out4, _c);
                        _out5 = vaddq_f32(_out5, _c);
                        _c = vdupq_lane_f32(vget_high_f32(_c0123), 1);
                        _out6 = vaddq_f32(_out6, _c);
                        _out7 = vaddq_f32(_out7, _c);
                    }
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
                _out4 = vmulq_f32(_out4, _alpha);
                _out5 = vmulq_f32(_out5, _alpha);
                _out6 = vmulq_f32(_out6, _alpha);
                _out7 = vmulq_f32(_out7, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out2));
                vst1_u16(p0 + 8, float2bfloat(_out4));
                vst1_u16(p0 + 12, float2bfloat(_out6));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_out3));
                vst1_u16(p0 + out_hstep * 4 + 8, float2bfloat(_out5));
                vst1_u16(p0 + out_hstep * 4 + 12, float2bfloat(_out7));
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + out_hstep, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_out4));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_out6));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep * 5, float2bfloat(_out3));
                vst1_u16(p0 + out_hstep * 6, float2bfloat(_out5));
                vst1_u16(p0 + out_hstep * 7, float2bfloat(_out7));
            }

            pp += 32;
            p0 += out_hstep * 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0;
                    float32x4_t _c1;
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + 8);
                        _c3 = vld1q_f32(pC + 12);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep);
                        _c2 = vld1q_f32(pC + c_hstep * 2);
                        _c3 = vld1q_f32(pC + c_hstep * 3);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c1);
                        _out2 = vaddq_f32(_out2, _c2);
                        _out3 = vaddq_f32(_out3, _c3);
                    }
                    else
                    {
                        _out0 = vmlaq_n_f32(_out0, _c0, beta);
                        _out1 = vmlaq_n_f32(_out1, _c1, beta);
                        _out2 = vmlaq_n_f32(_out2, _c2, beta);
                        _out3 = vmlaq_n_f32(_out3, _c3, beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vld1q_f32(pC);
                    if (beta != 1.f)
                        _cc0 = vmulq_n_f32(_cc0, beta);
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out1 = vaddq_f32(_out1, _cc0);
                    _out2 = vaddq_f32(_out2, _cc0);
                    _out3 = vaddq_f32(_out3, _cc0);
                    pC += 4;
                }
            }

            if (out_elempack == 1)
            {
                transpose4x4_ps(_out0, _out1, _out2, _out3);
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_elempack == 1)
                    {
                        _out0 = vaddq_f32(_out0, _c0123);
                        _out1 = vaddq_f32(_out1, _c0123);
                        _out2 = vaddq_f32(_out2, _c0123);
                        _out3 = vaddq_f32(_out3, _c0123);
                    }
                    else
                    {
                        _out0 = vaddq_f32(_out0, vdupq_lane_f32(vget_low_f32(_c0123), 0));
                        _out1 = vaddq_f32(_out1, vdupq_lane_f32(vget_low_f32(_c0123), 1));
                        _out2 = vaddq_f32(_out2, vdupq_lane_f32(vget_high_f32(_c0123), 0));
                        _out3 = vaddq_f32(_out3, vdupq_lane_f32(vget_high_f32(_c0123), 1));
                    }
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out1));
                vst1_u16(p0 + 8, float2bfloat(_out2));
                vst1_u16(p0 + 12, float2bfloat(_out3));
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + out_hstep, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep * 2, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep * 3, float2bfloat(_out3));
            }

            pp += 16;
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c01;
                    float32x4_t _c23;
                    if (c_elempack == 4)
                    {
                        float32x4x2_t _c = vzipq_f32(vld1q_f32(pC), vld1q_f32(pC + 4));
                        _c01 = _c.val[0];
                        _c23 = _c.val[1];
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        _c01 = vcombine_f32(vld1_f32(pC), vld1_f32(pC + c_hstep));
                        _c23 = vcombine_f32(vld1_f32(pC + c_hstep * 2), vld1_f32(pC + c_hstep * 3));
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _out0 = vaddq_f32(_out0, _c01);
                        _out1 = vaddq_f32(_out1, _c23);
                    }
                    else
                    {
                        _out0 = vmlaq_n_f32(_out0, _c01, beta);
                        _out1 = vmlaq_n_f32(_out1, _c23, beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    float32x4_t _cc0 = vcombine_f32(_c, _c);
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out1 = vaddq_f32(_out1, _cc0);
                    pC += 2;
                }
            }

            if (out_elempack == 1)
            {
                float32x4x2_t _t = vuzpq_f32(_out0, _out1);
                _out0 = _t.val[0];
                _out1 = _t.val[1];
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_elempack == 1)
                    {
                        _out0 = vaddq_f32(_out0, _c0123);
                        _out1 = vaddq_f32(_out1, _c0123);
                    }
                    else
                    {
                        float32x4x2_t _c = vzipq_f32(_c0123, _c0123);
                        _out0 = vaddq_f32(_out0, _c.val[0]);
                        _out1 = vaddq_f32(_out1, _c.val[1]);
                    }
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            if (out_elempack == 4)
            {
                uint16x4_t _r0 = float2bfloat(_out0);
                uint16x4_t _r1 = float2bfloat(_out1);
                vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(_r0), 0);
                vst1_lane_u32((unsigned int*)(p0 + 4), vreinterpret_u32_u16(_r0), 1);
                vst1_lane_u32((unsigned int*)(p0 + 8), vreinterpret_u32_u16(_r1), 0);
                vst1_lane_u32((unsigned int*)(p0 + 12), vreinterpret_u32_u16(_r1), 1);
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + out_hstep, float2bfloat(_out1));
            }

            pp += 8;
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _out0 = vld1q_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0123);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c;
                    if (c_elempack == 4)
                    {
                        _c = vld1q_f32(pC);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c = vdupq_n_f32(pC[0]);
                        _c = vsetq_lane_f32(pC[c_hstep], _c, 1);
                        _c = vsetq_lane_f32(pC[c_hstep * 2], _c, 2);
                        _c = vsetq_lane_f32(pC[c_hstep * 3], _c, 3);
                        pC++;
                    }
                    if (beta == 1.f)
                        _out0 = vaddq_f32(_out0, _c);
                    else
                        _out0 = vmlaq_n_f32(_out0, _c, beta);
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vdupq_n_f32(pC[0] * beta);
                    _out0 = vaddq_f32(_out0, _cc0);
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
            }

            vst1_u16(p0, float2bfloat(_out0));

            pp += 4;
            p0 += out_hstep;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack;
#if __ARM_NEON
        float32x4_t _c01 = vdupq_n_f32(0.f);
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
#if __ARM_NEON
                _c01 = vdupq_n_f32(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
#if __ARM_NEON
                float32x2_t _c = vld1_f32(pC);
                if (beta != 1.f)
                    _c = vmul_n_f32(_c, beta);
                _c01 = vcombine_f32(_c, _c);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

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
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 8);
            float32x4_t _out2 = vld1q_f32(pp + 4);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                    float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                    if (beta == 1.f)
                    {
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c1);
                        _out2 = vaddq_f32(_out2, _c2);
                        _out3 = vaddq_f32(_out3, _c3);
                    }
                    else
                    {
                        _out0 = vmlaq_n_f32(_out0, _c0, beta);
                        _out1 = vmlaq_n_f32(_out1, _c1, beta);
                        _out2 = vmlaq_n_f32(_out2, _c2, beta);
                        _out3 = vmlaq_n_f32(_out3, _c3, beta);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out2 = vaddq_f32(_out2, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    _out3 = vaddq_f32(_out3, _c1);
                    pC += 8;
                }
            }

            if (out_elempack == 1)
            {
                float32x4x2_t _t0 = vzipq_f32(_out0, _out2);
                float32x4x2_t _t1 = vzipq_f32(_out1, _out3);
                _out0 = _t0.val[0];
                _out1 = _t0.val[1];
                _out2 = _t1.val[0];
                _out3 = _t1.val[1];
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_elempack == 1)
                    {
                        _out0 = vaddq_f32(_out0, _c01);
                        _out1 = vaddq_f32(_out1, _c01);
                        _out2 = vaddq_f32(_out2, _c01);
                        _out3 = vaddq_f32(_out3, _c01);
                    }
                    else
                    {
                        float32x4_t _c = vdupq_lane_f32(vget_low_f32(_c01), 0);
                        _out0 = vaddq_f32(_out0, _c);
                        _out1 = vaddq_f32(_out1, _c);
                        _c = vdupq_lane_f32(vget_low_f32(_c01), 1);
                        _out2 = vaddq_f32(_out2, _c);
                        _out3 = vaddq_f32(_out3, _c);
                    }
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep * 4 + 4, float2bfloat(_out3));
            }
            if (out_elempack == 1)
            {
                vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_low_f32(_out0), vget_low_f32(_out0)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_high_f32(_out0), vget_high_f32(_out0)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 2), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_low_f32(_out1), vget_low_f32(_out1)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 3), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_high_f32(_out1), vget_high_f32(_out1)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 4), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_low_f32(_out2), vget_low_f32(_out2)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 5), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_high_f32(_out2), vget_high_f32(_out2)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 6), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_low_f32(_out3), vget_low_f32(_out3)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 7), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_high_f32(_out3), vget_high_f32(_out3)))), 0);
            }

            pp += 16;
            p0 += out_hstep * 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c1);
                    }
                    else
                    {
                        _out0 = vmlaq_n_f32(_out0, _c0, beta);
                        _out1 = vmlaq_n_f32(_out1, _c1, beta);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_n_f32(_c, beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    pC += 4;
                }
            }

            if (out_elempack == 1)
            {
                float32x4x2_t _t = vzipq_f32(_out0, _out1);
                _out0 = _t.val[0];
                _out1 = _t.val[1];
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_elempack == 1)
                    {
                        _out0 = vaddq_f32(_out0, _c01);
                        _out1 = vaddq_f32(_out1, _c01);
                    }
                    else
                    {
                        _out0 = vaddq_f32(_out0, vdupq_lane_f32(vget_low_f32(_c01), 0));
                        _out1 = vaddq_f32(_out1, vdupq_lane_f32(vget_low_f32(_c01), 1));
                    }
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + 4, float2bfloat(_out1));
            }
            if (out_elempack == 1)
            {
                vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_low_f32(_out0), vget_low_f32(_out0)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_high_f32(_out0), vget_high_f32(_out0)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 2), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_low_f32(_out1), vget_low_f32(_out1)))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep * 3), vreinterpret_u32_u16(float2bfloat(vcombine_f32(vget_high_f32(_out1), vget_high_f32(_out1)))), 0);
            }

            pp += 8;
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);
            float32x2_t _out1 = vld1_f32(pp + 2);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    float32x2_t _c0 = vld1_f32(pC);
                    float32x2_t _c1 = vld1_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _out0 = vadd_f32(_out0, _c0);
                        _out1 = vadd_f32(_out1, _c1);
                    }
                    else
                    {
                        _out0 = vmla_n_f32(_out0, _c0, beta);
                        _out1 = vmla_n_f32(_out1, _c1, beta);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    _out0 = vadd_f32(_out0, _c);
                    _out1 = vadd_f32(_out1, _c);
                    pC += 2;
                }
            }

            if (out_elempack == 1)
            {
                float32x2x2_t _t = vzip_f32(_out0, _out1);
                _out0 = _t.val[0];
                _out1 = _t.val[1];
            }

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    if (broadcast_type_C == 0 || out_elempack == 1)
                    {
                        _out0 = vadd_f32(_out0, vget_low_f32(_c01));
                        _out1 = vadd_f32(_out1, vget_low_f32(_c01));
                    }
                    else
                    {
                        _out0 = vadd_f32(_out0, vdup_lane_f32(vget_low_f32(_c01), 0));
                        _out1 = vadd_f32(_out1, vdup_lane_f32(vget_low_f32(_c01), 1));
                    }
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
                _out1 = vmul_n_f32(_out1, alpha);
            }

            if (out_elempack == 4)
            {
                uint16x4_t _r0 = float2bfloat(vcombine_f32(_out0, _out0));
                uint16x4_t _r1 = float2bfloat(vcombine_f32(_out1, _out1));
                vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(_r0), 0);
                vst1_lane_u32((unsigned int*)(p0 + 4), vreinterpret_u32_u16(_r1), 0);
            }
            if (out_elempack == 1)
            {
                vst1_lane_u32((unsigned int*)p0, vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out0, _out0))), 0);
                vst1_lane_u32((unsigned int*)(p0 + out_hstep), vreinterpret_u32_u16(float2bfloat(vcombine_f32(_out1, _out1))), 0);
            }

            pp += 4;
            p0 += out_hstep * 2;
        }
#endif // __ARM_NEON

        for (; jj < max_jj; jj += 1)
        {
            float out00 = pp[0];
            float out10 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out10 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out10 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += pC[0] * beta;
                    out10 += pC[c_hstep] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    float c = pC[0] * beta;
                    out00 += c;
                    out10 += c;
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out10 *= alpha;
            }

            p0[0] = float32_to_bfloat16(out00);
            p0[out_elempack] = float32_to_bfloat16(out10);

            pp += 2;
            p0 += out_hstep;
        }
    }
    for (; ii < max_ii; ii++)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack;
#if __ARM_NEON
        float32x4_t _c0 = vdupq_n_f32(0.f);
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
#if __ARM_NEON
                _c0 = vdupq_n_f32(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
#if __ARM_NEON
                _c0 = vdupq_n_f32(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[0] * beta;
        }

        int jj = 0;
#if __ARM_NEON
        for (; jj + 15 < max_jj; jj += 16)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                    _out2 = vaddq_f32(_out2, _c0);
                    _out3 = vaddq_f32(_out3, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + 8);
                    float32x4_t _c3 = vld1q_f32(pC + 12);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                        _c2 = vmulq_n_f32(_c2, beta);
                        _c3 = vmulq_n_f32(_c3, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    _out2 = vaddq_f32(_out2, _c2);
                    _out3 = vaddq_f32(_out3, _c3);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _out0 = vmulq_f32(_out0, _alpha);
                _out1 = vmulq_f32(_out1, _alpha);
                _out2 = vmulq_f32(_out2, _alpha);
                _out3 = vmulq_f32(_out3, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out1));
                vst1_u16(p0 + out_hstep * 8, float2bfloat(_out2));
                vst1_u16(p0 + out_hstep * 12, float2bfloat(_out3));
            }
            if (out_elempack == 1)
            {
                p0[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 0));
                (p0 + out_hstep)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 1));
                (p0 + out_hstep * 2)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 2));
                (p0 + out_hstep * 3)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 3));
                (p0 + out_hstep * 4)[0] = float32_to_bfloat16(vgetq_lane_f32(_out1, 0));
                (p0 + out_hstep * 5)[0] = float32_to_bfloat16(vgetq_lane_f32(_out1, 1));
                (p0 + out_hstep * 6)[0] = float32_to_bfloat16(vgetq_lane_f32(_out1, 2));
                (p0 + out_hstep * 7)[0] = float32_to_bfloat16(vgetq_lane_f32(_out1, 3));
                (p0 + out_hstep * 8)[0] = float32_to_bfloat16(vgetq_lane_f32(_out2, 0));
                (p0 + out_hstep * 9)[0] = float32_to_bfloat16(vgetq_lane_f32(_out2, 1));
                (p0 + out_hstep * 10)[0] = float32_to_bfloat16(vgetq_lane_f32(_out2, 2));
                (p0 + out_hstep * 11)[0] = float32_to_bfloat16(vgetq_lane_f32(_out2, 3));
                (p0 + out_hstep * 12)[0] = float32_to_bfloat16(vgetq_lane_f32(_out3, 0));
                (p0 + out_hstep * 13)[0] = float32_to_bfloat16(vgetq_lane_f32(_out3, 1));
                (p0 + out_hstep * 14)[0] = float32_to_bfloat16(vgetq_lane_f32(_out3, 2));
                (p0 + out_hstep * 15)[0] = float32_to_bfloat16(vgetq_lane_f32(_out3, 3));
            }

            pp += 16;
            p0 += out_hstep * 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        _c0 = vmulq_n_f32(_c0, beta);
                        _c1 = vmulq_n_f32(_c1, beta);
                    }
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
                vst1_u16(p0 + out_hstep * 4, float2bfloat(_out1));
            }
            if (out_elempack == 1)
            {
                p0[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 0));
                (p0 + out_hstep)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 1));
                (p0 + out_hstep * 2)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 2));
                (p0 + out_hstep * 3)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 3));
                (p0 + out_hstep * 4)[0] = float32_to_bfloat16(vgetq_lane_f32(_out1, 0));
                (p0 + out_hstep * 5)[0] = float32_to_bfloat16(vgetq_lane_f32(_out1, 1));
                (p0 + out_hstep * 6)[0] = float32_to_bfloat16(vgetq_lane_f32(_out1, 2));
                (p0 + out_hstep * 7)[0] = float32_to_bfloat16(vgetq_lane_f32(_out1, 3));
            }

            pp += 8;
            p0 += out_hstep * 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_n_f32(_c, beta);
                    _out0 = vaddq_f32(_out0, _c);
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    if (beta != 1.f)
                        _c = vmulq_n_f32(_c, beta);
                    _out0 = vaddq_f32(_out0, _c);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
            }

            if (out_elempack == 4)
            {
                vst1_u16(p0, float2bfloat(_out0));
            }
            if (out_elempack == 1)
            {
                p0[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 0));
                (p0 + out_hstep)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 1));
                (p0 + out_hstep * 2)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 2));
                (p0 + out_hstep * 3)[0] = float32_to_bfloat16(vgetq_lane_f32(_out0, 3));
            }

            pp += 4;
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c0));
                }
                if (broadcast_type_C == 3)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta == 1.f)
                        _out0 = vadd_f32(_out0, _c);
                    else
                        _out0 = vmla_n_f32(_out0, _c, beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    _out0 = vadd_f32(_out0, _c);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
            }

            if (out_elempack == 1)
            {
                p0[0] = float32_to_bfloat16(vget_lane_f32(_out0, 0));
                p0[out_hstep] = float32_to_bfloat16(vget_lane_f32(_out0, 1));
            }

            pp += 2;
            p0 += out_hstep * 2;
        }
#endif // __ARM_NEON

        for (; jj < max_jj; jj += 1)
        {
            float out00 = pp[0];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += pC[0] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += pC[0] * beta;
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
            }

            p0[0] = float32_to_bfloat16(out00);

            pp++;
            p0 += out_hstep;
        }
    }
}
