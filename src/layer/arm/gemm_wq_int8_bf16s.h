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

                    float32x4_t _v127 = vdupq_n_f32(127.f);
                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                    float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        int8x8_t _r01 = float2int8(vmulq_laneq_f32(bfloat2float(_p.val[0]), _scale0, 0), vmulq_laneq_f32(bfloat2float(_p.val[1]), _scale0, 1));
                        int8x8_t _r23 = float2int8(vmulq_laneq_f32(bfloat2float(_p.val[2]), _scale0, 2), vmulq_laneq_f32(bfloat2float(_p.val[3]), _scale0, 3));
                        uint16x4x4_t _q = vld4_u16(p0 + 16);
                        int8x8_t _s01 = float2int8(vmulq_laneq_f32(bfloat2float(_q.val[0]), _scale0, 0), vmulq_laneq_f32(bfloat2float(_q.val[1]), _scale0, 1));
                        int8x8_t _s23 = float2int8(vmulq_laneq_f32(bfloat2float(_q.val[2]), _scale0, 2), vmulq_laneq_f32(bfloat2float(_q.val[3]), _scale0, 3));
                        int32x2x2_t _r0 = vzip_s32(vreinterpret_s32_s8(_r01), vreinterpret_s32_s8(_s01));
                        int32x2x2_t _r2 = vzip_s32(vreinterpret_s32_s8(_r23), vreinterpret_s32_s8(_s23));
                        vst1q_s8(pp, vcombine_s8(vreinterpret_s8_s32(_r0.val[0]), vreinterpret_s8_s32(_r0.val[1])));
                        vst1q_s8(pp + 16, vcombine_s8(vreinterpret_s8_s32(_r2.val[0]), vreinterpret_s8_s32(_r2.val[1])));

                        _p = vld4_u16(p0 + A_hstep * 4);
                        _r01 = float2int8(vmulq_laneq_f32(bfloat2float(_p.val[0]), _scale1, 0), vmulq_laneq_f32(bfloat2float(_p.val[1]), _scale1, 1));
                        _r23 = float2int8(vmulq_laneq_f32(bfloat2float(_p.val[2]), _scale1, 2), vmulq_laneq_f32(bfloat2float(_p.val[3]), _scale1, 3));
                        _q = vld4_u16(p0 + A_hstep * 4 + 16);
                        _s01 = float2int8(vmulq_laneq_f32(bfloat2float(_q.val[0]), _scale1, 0), vmulq_laneq_f32(bfloat2float(_q.val[1]), _scale1, 1));
                        _s23 = float2int8(vmulq_laneq_f32(bfloat2float(_q.val[2]), _scale1, 2), vmulq_laneq_f32(bfloat2float(_q.val[3]), _scale1, 3));
                        _r0 = vzip_s32(vreinterpret_s32_s8(_r01), vreinterpret_s32_s8(_s01));
                        _r2 = vzip_s32(vreinterpret_s32_s8(_r23), vreinterpret_s32_s8(_s23));
                        vst1q_s8(pp + 32, vcombine_s8(vreinterpret_s8_s32(_r0.val[0]), vreinterpret_s8_s32(_r0.val[1])));
                        vst1q_s8(pp + 48, vcombine_s8(vreinterpret_s8_s32(_r2.val[0]), vreinterpret_s8_s32(_r2.val[1])));
                        pp += 64;
                        p0 += 32;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        uint16x4x4_t _q = vld4_u16(p0 + A_hstep * 4);
                        int8x8_t _r01 = float2int8(vmulq_laneq_f32(bfloat2float(_p.val[0]), _scale0, 0), vmulq_laneq_f32(bfloat2float(_p.val[1]), _scale0, 1));
                        int8x8_t _r23 = float2int8(vmulq_laneq_f32(bfloat2float(_p.val[2]), _scale0, 2), vmulq_laneq_f32(bfloat2float(_p.val[3]), _scale0, 3));
                        int8x8_t _r45 = float2int8(vmulq_laneq_f32(bfloat2float(_q.val[0]), _scale1, 0), vmulq_laneq_f32(bfloat2float(_q.val[1]), _scale1, 1));
                        int8x8_t _r67 = float2int8(vmulq_laneq_f32(bfloat2float(_q.val[2]), _scale1, 2), vmulq_laneq_f32(bfloat2float(_q.val[3]), _scale1, 3));
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
                        float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0);
                        float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale0);
                        float32x4_t _p2 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _scale1);
                        float32x4_t _p3 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _scale1);
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
                        _r0 = vzip_s8(_r0, vext_s8(_r0, _r0, 4)).val[0];
                        _r1 = vzip_s8(_r1, vext_s8(_r1, _r1, 4)).val[0];
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        pp += 16;
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0);
                        float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _scale1);
                        vst1_s8(pp, float2int8(_p0, _p1));
                        pp += 8;
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
                    float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                    float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                    float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                    float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                    _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vdupq_n_f32(bfloat16_to_float32(p0a[0]));
                        _p0 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep]), _p0, 1);
                        _p0 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 2]), _p0, 2);
                        _p0 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 3]), _p0, 3);
                        float32x4_t _p1 = vdupq_n_f32(bfloat16_to_float32(p0a[A_hstep * 4]));
                        _p1 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 5]), _p1, 1);
                        _p1 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 6]), _p1, 2);
                        _p1 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 7]), _p1, 3);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        p0a++;
                    }

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    float32x4_t _v127 = vdupq_n_f32(127.f);
                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                    float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

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
                        int8x8_t _r0 = float2int8(vmulq_laneq_f32(_p00, _scale0, 0), vmulq_laneq_f32(_p01, _scale0, 0));
                        int8x8_t _r1 = float2int8(vmulq_laneq_f32(_p10, _scale0, 1), vmulq_laneq_f32(_p11, _scale0, 1));
                        int8x8_t _r2 = float2int8(vmulq_laneq_f32(_p20, _scale0, 2), vmulq_laneq_f32(_p21, _scale0, 2));
                        int8x8_t _r3 = float2int8(vmulq_laneq_f32(_p30, _scale0, 3), vmulq_laneq_f32(_p31, _scale0, 3));
                        int8x8_t _r4 = float2int8(vmulq_laneq_f32(_p40, _scale1, 0), vmulq_laneq_f32(_p41, _scale1, 0));
                        int8x8_t _r5 = float2int8(vmulq_laneq_f32(_p50, _scale1, 1), vmulq_laneq_f32(_p51, _scale1, 1));
                        int8x8_t _r6 = float2int8(vmulq_laneq_f32(_p60, _scale1, 2), vmulq_laneq_f32(_p61, _scale1, 2));
                        int8x8_t _r7 = float2int8(vmulq_laneq_f32(_p70, _scale1, 3), vmulq_laneq_f32(_p71, _scale1, 3));
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
                        int8x8_t _r01 = float2int8(vmulq_laneq_f32(_p0, _scale0, 0), vmulq_laneq_f32(_p1, _scale0, 1));
                        int8x8_t _r23 = float2int8(vmulq_laneq_f32(_p2, _scale0, 2), vmulq_laneq_f32(_p3, _scale0, 3));
                        int8x8_t _r45 = float2int8(vmulq_laneq_f32(_p4, _scale1, 0), vmulq_laneq_f32(_p5, _scale1, 1));
                        int8x8_t _r67 = float2int8(vmulq_laneq_f32(_p6, _scale1, 2), vmulq_laneq_f32(_p7, _scale1, 3));
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
                    float32x4x2_t _scale01 = vzipq_f32(_scale0, _scale0);
                    float32x4x2_t _scale45 = vzipq_f32(_scale1, _scale1);
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        uint32x2_t _r0 = vdup_n_u32(0);
                        uint32x2_t _r1 = vdup_n_u32(0);
                        uint32x2_t _r2 = vdup_n_u32(0);
                        uint32x2_t _r3 = vdup_n_u32(0);
                        _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                        _r0 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep), _r0, 1);
                        _r1 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 2), _r1, 0);
                        _r1 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 3), _r1, 1);
                        _r2 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 4), _r2, 0);
                        _r2 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 5), _r2, 1);
                        _r3 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 6), _r3, 0);
                        _r3 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 7), _r3, 1);
                        float32x4_t _p01 = bfloat2float(vreinterpret_u16_u32(_r0));
                        float32x4_t _p23 = bfloat2float(vreinterpret_u16_u32(_r1));
                        float32x4_t _p45 = bfloat2float(vreinterpret_u16_u32(_r2));
                        float32x4_t _p67 = bfloat2float(vreinterpret_u16_u32(_r3));
                        int8x8_t _q0 = float2int8(vmulq_f32(_p01, _scale01.val[0]), vmulq_f32(_p23, _scale01.val[1]));
                        int8x8_t _q1 = float2int8(vmulq_f32(_p45, _scale45.val[0]), vmulq_f32(_p67, _scale45.val[1]));
                        vst1q_s8(pp, vcombine_s8(_q0, _q1));
                        pp += 16;
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vdupq_n_f32(bfloat16_to_float32(p0[0]));
                        _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep]), _p0, 1);
                        _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 2]), _p0, 2);
                        _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 3]), _p0, 3);
                        float32x4_t _p1 = vdupq_n_f32(bfloat16_to_float32(p0[A_hstep * 4]));
                        _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 5]), _p1, 1);
                        _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 6]), _p1, 2);
                        _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 7]), _p1, 3);
                        vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale0), vmulq_f32(_p1, _scale1)));
                        pp += 8;
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

                    float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                    float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax);
#else
                    float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax);
#endif
                    _scale = vbslq_f32(vceqq_f32(_absmax, _zero), _zero, _scale);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        int8x8_t _r01 = float2int8(vmulq_lane_f32(bfloat2float(_p.val[0]), vget_low_f32(_scale), 0), vmulq_lane_f32(bfloat2float(_p.val[1]), vget_low_f32(_scale), 1));
                        int8x8_t _r23 = float2int8(vmulq_lane_f32(bfloat2float(_p.val[2]), vget_high_f32(_scale), 0), vmulq_lane_f32(bfloat2float(_p.val[3]), vget_high_f32(_scale), 1));
                        uint16x4x4_t _q = vld4_u16(p0 + 16);
                        int8x8_t _s01 = float2int8(vmulq_lane_f32(bfloat2float(_q.val[0]), vget_low_f32(_scale), 0), vmulq_lane_f32(bfloat2float(_q.val[1]), vget_low_f32(_scale), 1));
                        int8x8_t _s23 = float2int8(vmulq_lane_f32(bfloat2float(_q.val[2]), vget_high_f32(_scale), 0), vmulq_lane_f32(bfloat2float(_q.val[3]), vget_high_f32(_scale), 1));
                        int32x2x2_t _r0 = vzip_s32(vreinterpret_s32_s8(_r01), vreinterpret_s32_s8(_s01));
                        int32x2x2_t _r2 = vzip_s32(vreinterpret_s32_s8(_r23), vreinterpret_s32_s8(_s23));
                        vst1q_s8(pp, vcombine_s8(vreinterpret_s8_s32(_r0.val[0]), vreinterpret_s8_s32(_r0.val[1])));
                        vst1q_s8(pp + 16, vcombine_s8(vreinterpret_s8_s32(_r2.val[0]), vreinterpret_s8_s32(_r2.val[1])));
                        pp += 32;
                        p0 += 32;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        uint16x4x4_t _p = vld4_u16(p0);
                        int8x8_t _r01 = float2int8(vmulq_lane_f32(bfloat2float(_p.val[0]), vget_low_f32(_scale), 0), vmulq_lane_f32(bfloat2float(_p.val[1]), vget_low_f32(_scale), 1));
                        int8x8_t _r23 = float2int8(vmulq_lane_f32(bfloat2float(_p.val[2]), vget_high_f32(_scale), 0), vmulq_lane_f32(bfloat2float(_p.val[3]), vget_high_f32(_scale), 1));
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
                        float32x4_t _p0 = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale);
                        float32x4_t _p1 = vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale);
                        int8x8_t _r = float2int8(_p0, _p1);
                        _r = vzip_s8(_r, vext_s8(_r, _r, 4)).val[0];
                        vst1_s8(pp, _r);
                        pp += 8;
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vmulq_f32(bfloat2float(vld1_u16(p0)), _scale);
                        int8x8_t _r = float2int8(_p, _p);
                        vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                        pp += 4;
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
                    float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vdupq_n_f32(bfloat16_to_float32(p0a[0]));
                        _p = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep]), _p, 1);
                        _p = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 2]), _p, 2);
                        _p = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 3]), _p, 3);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p));
                        p0a++;
                    }
                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                    float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                    float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
#else
                    float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax0);
#endif
                    _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

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
                        int8x8_t _r0 = float2int8(vmulq_lane_f32(_p00, vget_low_f32(_scale), 0), vmulq_lane_f32(_p01, vget_low_f32(_scale), 0));
                        int8x8_t _r1 = float2int8(vmulq_lane_f32(_p10, vget_low_f32(_scale), 1), vmulq_lane_f32(_p11, vget_low_f32(_scale), 1));
                        int8x8_t _r2 = float2int8(vmulq_lane_f32(_p20, vget_high_f32(_scale), 0), vmulq_lane_f32(_p21, vget_high_f32(_scale), 0));
                        int8x8_t _r3 = float2int8(vmulq_lane_f32(_p30, vget_high_f32(_scale), 1), vmulq_lane_f32(_p31, vget_high_f32(_scale), 1));
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
                        int8x8_t _r01 = float2int8(vmulq_lane_f32(_p0, vget_low_f32(_scale), 0), vmulq_lane_f32(_p1, vget_low_f32(_scale), 1));
                        int8x8_t _r23 = float2int8(vmulq_lane_f32(_p2, vget_high_f32(_scale), 0), vmulq_lane_f32(_p3, vget_high_f32(_scale), 1));
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
                    float32x4x2_t _scale01 = vzipq_f32(_scale, _scale);
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        uint32x2_t _r0 = vdup_n_u32(0);
                        uint32x2_t _r1 = vdup_n_u32(0);
                        _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                        _r0 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep), _r0, 1);
                        _r1 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 2), _r1, 0);
                        _r1 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 3), _r1, 1);
                        float32x4_t _p01 = bfloat2float(vreinterpret_u16_u32(_r0));
                        float32x4_t _p23 = bfloat2float(vreinterpret_u16_u32(_r1));
                        int8x8_t _r = float2int8(vmulq_f32(_p01, _scale01.val[0]), vmulq_f32(_p23, _scale01.val[1]));
                        vst1_s8(pp, _r);
                        pp += 8;
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vdupq_n_f32(bfloat16_to_float32(p0[0]));
                        _p = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep]), _p, 1);
                        _p = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 2]), _p, 2);
                        _p = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 3]), _p, 3);
                        int8x8_t _r = float2int8(vmulq_f32(_p, _scale), _p);
                        vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                        pp += 4;
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

                float32x4_t _v127 = vdupq_n_f32(127.f);
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    uint16x4x4_t _p = vld4_u16(p0);
                    int8x8_t _r01 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[0]), _s0), _scale0, 0), vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[1]), _s0), _scale0, 1));
                    int8x8_t _r23 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[2]), _s0), _scale0, 2), vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[3]), _s0), _scale0, 3));
                    uint16x4x4_t _q = vld4_u16(p0 + 16);
                    int8x8_t _t01 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[0]), _s1), _scale0, 0), vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[1]), _s1), _scale0, 1));
                    int8x8_t _t23 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[2]), _s1), _scale0, 2), vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[3]), _s1), _scale0, 3));
                    int32x2x2_t _r0 = vzip_s32(vreinterpret_s32_s8(_r01), vreinterpret_s32_s8(_t01));
                    int32x2x2_t _r2 = vzip_s32(vreinterpret_s32_s8(_r23), vreinterpret_s32_s8(_t23));
                    vst1q_s8(pp, vcombine_s8(vreinterpret_s8_s32(_r0.val[0]), vreinterpret_s8_s32(_r0.val[1])));
                    vst1q_s8(pp + 16, vcombine_s8(vreinterpret_s8_s32(_r2.val[0]), vreinterpret_s8_s32(_r2.val[1])));

                    _p = vld4_u16(p0 + A_hstep * 4);
                    _r01 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[0]), _s0), _scale1, 0), vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[1]), _s0), _scale1, 1));
                    _r23 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[2]), _s0), _scale1, 2), vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[3]), _s0), _scale1, 3));
                    _q = vld4_u16(p0 + A_hstep * 4 + 16);
                    _t01 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[0]), _s1), _scale1, 0), vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[1]), _s1), _scale1, 1));
                    _t23 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[2]), _s1), _scale1, 2), vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[3]), _s1), _scale1, 3));
                    _r0 = vzip_s32(vreinterpret_s32_s8(_r01), vreinterpret_s32_s8(_t01));
                    _r2 = vzip_s32(vreinterpret_s32_s8(_r23), vreinterpret_s32_s8(_t23));
                    vst1q_s8(pp + 32, vcombine_s8(vreinterpret_s8_s32(_r0.val[0]), vreinterpret_s8_s32(_r0.val[1])));
                    vst1q_s8(pp + 48, vcombine_s8(vreinterpret_s8_s32(_r2.val[0]), vreinterpret_s8_s32(_r2.val[1])));
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
                    int8x8_t _r01 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[0]), _s), _scale0, 0), vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[1]), _s), _scale0, 1));
                    int8x8_t _r23 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[2]), _s), _scale0, 2), vmulq_laneq_f32(vmulq_f32(bfloat2float(_p.val[3]), _s), _scale0, 3));
                    int8x8_t _r45 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[0]), _s), _scale1, 0), vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[1]), _s), _scale1, 1));
                    int8x8_t _r67 = float2int8(vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[2]), _s), _scale1, 2), vmulq_laneq_f32(vmulq_f32(bfloat2float(_q.val[3]), _s), _scale1, 3));
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
                    float32x4_t _p0 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0), ps[0]);
                    float32x4_t _p1 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale0), ps[1]);
                    float32x4_t _p2 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _scale1), ps[0]);
                    float32x4_t _p3 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _scale1), ps[1]);
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    _r0 = vzip_s8(_r0, vext_s8(_r0, _r0, 4)).val[0];
                    _r1 = vzip_s8(_r1, vext_s8(_r1, _r1, 4)).val[0];
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    pp += 16;
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _scale0), ps[0]);
                    float32x4_t _p1 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _scale1), ps[0]);
                    vst1_s8(pp, float2int8(_p0, _p1));
                    pp += 8;
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
                float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));
                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    float32x4_t _p0 = vdupq_n_f32(bfloat16_to_float32(p0a[0]));
                    _p0 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep]), _p0, 1);
                    _p0 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 2]), _p0, 2);
                    _p0 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 3]), _p0, 3);
                    float32x4_t _p1 = vdupq_n_f32(bfloat16_to_float32(p0a[A_hstep * 4]));
                    _p1 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 5]), _p1, 1);
                    _p1 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 6]), _p1, 2);
                    _p1 = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 7]), _p1, 3);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_n_f32(vabsq_f32(_p0), s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_n_f32(vabsq_f32(_p1), s));
                    p0a++;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                float32x4_t _v127 = vdupq_n_f32(127.f);
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

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
                    int8x8_t _r0 = float2int8(vmulq_laneq_f32(_p00, _scale0, 0), vmulq_laneq_f32(_p01, _scale0, 0));
                    int8x8_t _r1 = float2int8(vmulq_laneq_f32(_p10, _scale0, 1), vmulq_laneq_f32(_p11, _scale0, 1));
                    int8x8_t _r2 = float2int8(vmulq_laneq_f32(_p20, _scale0, 2), vmulq_laneq_f32(_p21, _scale0, 2));
                    int8x8_t _r3 = float2int8(vmulq_laneq_f32(_p30, _scale0, 3), vmulq_laneq_f32(_p31, _scale0, 3));
                    int8x8_t _r4 = float2int8(vmulq_laneq_f32(_p40, _scale1, 0), vmulq_laneq_f32(_p41, _scale1, 0));
                    int8x8_t _r5 = float2int8(vmulq_laneq_f32(_p50, _scale1, 1), vmulq_laneq_f32(_p51, _scale1, 1));
                    int8x8_t _r6 = float2int8(vmulq_laneq_f32(_p60, _scale1, 2), vmulq_laneq_f32(_p61, _scale1, 2));
                    int8x8_t _r7 = float2int8(vmulq_laneq_f32(_p70, _scale1, 3), vmulq_laneq_f32(_p71, _scale1, 3));
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
                    int8x8_t _r01 = float2int8(vmulq_laneq_f32(_p0, _scale0, 0), vmulq_laneq_f32(_p1, _scale0, 1));
                    int8x8_t _r23 = float2int8(vmulq_laneq_f32(_p2, _scale0, 2), vmulq_laneq_f32(_p3, _scale0, 3));
                    int8x8_t _r45 = float2int8(vmulq_laneq_f32(_p4, _scale1, 0), vmulq_laneq_f32(_p5, _scale1, 1));
                    int8x8_t _r67 = float2int8(vmulq_laneq_f32(_p6, _scale1, 2), vmulq_laneq_f32(_p7, _scale1, 3));
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
                float32x4x2_t _scale01 = vzipq_f32(_scale0, _scale0);
                float32x4x2_t _scale45 = vzipq_f32(_scale1, _scale1);
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _s = vcombine_f32(vld1_f32(ps), vld1_f32(ps));
                    float32x4_t _s01 = vmulq_f32(_s, _scale01.val[0]);
                    float32x4_t _s23 = vmulq_f32(_s, _scale01.val[1]);
                    float32x4_t _s45 = vmulq_f32(_s, _scale45.val[0]);
                    float32x4_t _s67 = vmulq_f32(_s, _scale45.val[1]);
                    uint32x2_t _r0 = vdup_n_u32(0);
                    uint32x2_t _r1 = vdup_n_u32(0);
                    uint32x2_t _r2 = vdup_n_u32(0);
                    uint32x2_t _r3 = vdup_n_u32(0);
                    _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                    _r0 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep), _r0, 1);
                    _r1 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 2), _r1, 0);
                    _r1 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 3), _r1, 1);
                    _r2 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 4), _r2, 0);
                    _r2 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 5), _r2, 1);
                    _r3 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 6), _r3, 0);
                    _r3 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 7), _r3, 1);
                    float32x4_t _p01 = bfloat2float(vreinterpret_u16_u32(_r0));
                    float32x4_t _p23 = bfloat2float(vreinterpret_u16_u32(_r1));
                    float32x4_t _p45 = bfloat2float(vreinterpret_u16_u32(_r2));
                    float32x4_t _p67 = bfloat2float(vreinterpret_u16_u32(_r3));
                    int8x8_t _q0 = float2int8(vmulq_f32(_p01, _s01), vmulq_f32(_p23, _s23));
                    int8x8_t _q1 = float2int8(vmulq_f32(_p45, _s45), vmulq_f32(_p67, _s67));
                    vst1q_s8(pp, vcombine_s8(_q0, _q1));
                    pp += 16;
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = ps[0];
                    float32x4_t _p0 = vdupq_n_f32(bfloat16_to_float32(p0[0]));
                    _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep]), _p0, 1);
                    _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 2]), _p0, 2);
                    _p0 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 3]), _p0, 3);
                    float32x4_t _p1 = vdupq_n_f32(bfloat16_to_float32(p0[A_hstep * 4]));
                    _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 5]), _p1, 1);
                    _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 6]), _p1, 2);
                    _p1 = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 7]), _p1, 3);
                    _p0 = vmulq_n_f32(_p0, s);
                    _p1 = vmulq_n_f32(_p1, s);
                    vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale0), vmulq_f32(_p1, _scale1)));
                    pp += 8;
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

                float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax);
#else
                float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax);
#endif
                _scale = vbslq_f32(vceqq_f32(_absmax, _zero), _zero, _scale);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    uint16x4x4_t _p = vld4_u16(p0);
                    int8x8_t _r01 = float2int8(vmulq_lane_f32(vmulq_f32(bfloat2float(_p.val[0]), _s0), vget_low_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(bfloat2float(_p.val[1]), _s0), vget_low_f32(_scale), 1));
                    int8x8_t _r23 = float2int8(vmulq_lane_f32(vmulq_f32(bfloat2float(_p.val[2]), _s0), vget_high_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(bfloat2float(_p.val[3]), _s0), vget_high_f32(_scale), 1));
                    uint16x4x4_t _q = vld4_u16(p0 + 16);
                    int8x8_t _s01 = float2int8(vmulq_lane_f32(vmulq_f32(bfloat2float(_q.val[0]), _s1), vget_low_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(bfloat2float(_q.val[1]), _s1), vget_low_f32(_scale), 1));
                    int8x8_t _s23 = float2int8(vmulq_lane_f32(vmulq_f32(bfloat2float(_q.val[2]), _s1), vget_high_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(bfloat2float(_q.val[3]), _s1), vget_high_f32(_scale), 1));
                    int32x2x2_t _r0 = vzip_s32(vreinterpret_s32_s8(_r01), vreinterpret_s32_s8(_s01));
                    int32x2x2_t _r2 = vzip_s32(vreinterpret_s32_s8(_r23), vreinterpret_s32_s8(_s23));
                    vst1q_s8(pp, vcombine_s8(vreinterpret_s8_s32(_r0.val[0]), vreinterpret_s8_s32(_r0.val[1])));
                    vst1q_s8(pp + 16, vcombine_s8(vreinterpret_s8_s32(_r2.val[0]), vreinterpret_s8_s32(_r2.val[1])));
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
                    int8x8_t _r01 = float2int8(vmulq_lane_f32(vmulq_f32(bfloat2float(_p.val[0]), _s), vget_low_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(bfloat2float(_p.val[1]), _s), vget_low_f32(_scale), 1));
                    int8x8_t _r23 = float2int8(vmulq_lane_f32(vmulq_f32(bfloat2float(_p.val[2]), _s), vget_high_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(bfloat2float(_p.val[3]), _s), vget_high_f32(_scale), 1));
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
                    float32x4_t _p0 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _scale), ps[0]);
                    float32x4_t _p1 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale), ps[1]);
                    int8x8_t _r = float2int8(_p0, _p1);
                    _r = vzip_s8(_r, vext_s8(_r, _r, 4)).val[0];
                    vst1_s8(pp, _r);
                    pp += 8;
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _scale), ps[0]);
                    int8x8_t _r = float2int8(_p, _p);
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                    pp += 4;
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
                float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    float32x4_t _p = vdupq_n_f32(bfloat16_to_float32(p0a[0]));
                    _p = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep]), _p, 1);
                    _p = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 2]), _p, 2);
                    _p = vsetq_lane_f32(bfloat16_to_float32(p0a[A_hstep * 3]), _p, 3);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_n_f32(vabsq_f32(_p), s));
                    p0a++;
                }
                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
#else
                float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax0);
#endif
                _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

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
#if __aarch64__
                    int8x8_t _r0 = float2int8(vmulq_laneq_f32(_p00, _scale, 0), vmulq_laneq_f32(_p01, _scale, 0));
                    int8x8_t _r1 = float2int8(vmulq_laneq_f32(_p10, _scale, 1), vmulq_laneq_f32(_p11, _scale, 1));
                    int8x8_t _r2 = float2int8(vmulq_laneq_f32(_p20, _scale, 2), vmulq_laneq_f32(_p21, _scale, 2));
                    int8x8_t _r3 = float2int8(vmulq_laneq_f32(_p30, _scale, 3), vmulq_laneq_f32(_p31, _scale, 3));
#else
                    int8x8_t _r0 = float2int8(vmulq_lane_f32(_p00, vget_low_f32(_scale), 0), vmulq_lane_f32(_p01, vget_low_f32(_scale), 0));
                    int8x8_t _r1 = float2int8(vmulq_lane_f32(_p10, vget_low_f32(_scale), 1), vmulq_lane_f32(_p11, vget_low_f32(_scale), 1));
                    int8x8_t _r2 = float2int8(vmulq_lane_f32(_p20, vget_high_f32(_scale), 0), vmulq_lane_f32(_p21, vget_high_f32(_scale), 0));
                    int8x8_t _r3 = float2int8(vmulq_lane_f32(_p30, vget_high_f32(_scale), 1), vmulq_lane_f32(_p31, vget_high_f32(_scale), 1));
#endif
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
#if __aarch64__
                    int8x8_t _r01 = float2int8(vmulq_laneq_f32(_p0, _scale, 0), vmulq_laneq_f32(_p1, _scale, 1));
                    int8x8_t _r23 = float2int8(vmulq_laneq_f32(_p2, _scale, 2), vmulq_laneq_f32(_p3, _scale, 3));
#else
                    int8x8_t _r01 = float2int8(vmulq_lane_f32(_p0, vget_low_f32(_scale), 0), vmulq_lane_f32(_p1, vget_low_f32(_scale), 1));
                    int8x8_t _r23 = float2int8(vmulq_lane_f32(_p2, vget_high_f32(_scale), 0), vmulq_lane_f32(_p3, vget_high_f32(_scale), 1));
#endif
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
                float32x4x2_t _scale01 = vzipq_f32(_scale, _scale);
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _s = vcombine_f32(vld1_f32(ps), vld1_f32(ps));
                    float32x4_t _s01 = vmulq_f32(_s, _scale01.val[0]);
                    float32x4_t _s23 = vmulq_f32(_s, _scale01.val[1]);
                    uint32x2_t _r0 = vdup_n_u32(0);
                    uint32x2_t _r1 = vdup_n_u32(0);
                    _r0 = vld1_lane_u32((const uint32_t*)p0, _r0, 0);
                    _r0 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep), _r0, 1);
                    _r1 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 2), _r1, 0);
                    _r1 = vld1_lane_u32((const uint32_t*)(p0 + A_hstep * 3), _r1, 1);
                    float32x4_t _p01 = bfloat2float(vreinterpret_u16_u32(_r0));
                    float32x4_t _p23 = bfloat2float(vreinterpret_u16_u32(_r1));
                    int8x8_t _r = float2int8(vmulq_f32(_p01, _s01), vmulq_f32(_p23, _s23));
                    vst1_s8(pp, _r);
                    pp += 8;
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = ps[0];
                    float32x4_t _p = vdupq_n_f32(bfloat16_to_float32(p0[0]));
                    _p = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep]), _p, 1);
                    _p = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 2]), _p, 2);
                    _p = vsetq_lane_f32(bfloat16_to_float32(p0[A_hstep * 3]), _p, 3);
                    _p = vmulq_n_f32(_p, s);
                    int8x8_t _r = float2int8(vmulq_f32(_p, _scale), _p);
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                    pp += 4;
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

                    float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                    float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                    float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                    float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                    _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    float32x4_t _v127 = vdupq_n_f32(127.f);
                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                    float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0)), _scale0, 0);
                        float32x4_t _p01 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _scale0, 0);
                        float32x4_t _p10 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + 4)), _scale0, 1);
                        float32x4_t _p11 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _scale0, 1);
                        int8x8_t _r0 = float2int8(_p00, _p01);
                        int8x8_t _r1 = float2int8(_p10, _p11);
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));

                        float32x4_t _p20 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + 8)), _scale0, 2);
                        float32x4_t _p21 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 8)), _scale0, 2);
                        float32x4_t _p30 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + 12)), _scale0, 3);
                        float32x4_t _p31 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 12)), _scale0, 3);
                        int8x8_t _r2 = float2int8(_p20, _p21);
                        int8x8_t _r3 = float2int8(_p30, _p31);
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                        float32x4_t _p40 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + 16)), _scale1, 0);
                        float32x4_t _p41 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 16)), _scale1, 0);
                        float32x4_t _p50 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + 20)), _scale1, 1);
                        float32x4_t _p51 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 20)), _scale1, 1);
                        int8x8_t _r4 = float2int8(_p40, _p41);
                        int8x8_t _r5 = float2int8(_p50, _p51);
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));

                        float32x4_t _p60 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + 24)), _scale1, 2);
                        float32x4_t _p61 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 24)), _scale1, 2);
                        float32x4_t _p70 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + 28)), _scale1, 3);
                        float32x4_t _p71 = vmulq_laneq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 28)), _scale1, 3);
                        int8x8_t _r6 = float2int8(_p60, _p61);
                        int8x8_t _r7 = float2int8(_p70, _p71);
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
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

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    float32x4_t _v127 = vdupq_n_f32(127.f);
                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                    float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

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

                    float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                    float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                    float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
#else
                    float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax0);
#endif
                    _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = vmulq_n_f32(bfloat2float(vld1_u16(p0)), vgetq_lane_f32(_scale, 0));
                        float32x4_t _p01 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), vgetq_lane_f32(_scale, 0));
                        float32x4_t _p10 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + 4)), vgetq_lane_f32(_scale, 1));
                        float32x4_t _p11 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), vgetq_lane_f32(_scale, 1));
                        int8x8_t _r0 = float2int8(_p00, _p01);
                        int8x8_t _r1 = float2int8(_p10, _p11);
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        float32x4_t _p20 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + 8)), vgetq_lane_f32(_scale, 2));
                        float32x4_t _p21 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 8)), vgetq_lane_f32(_scale, 2));
                        float32x4_t _p30 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + 12)), vgetq_lane_f32(_scale, 3));
                        float32x4_t _p31 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 12)), vgetq_lane_f32(_scale, 3));
                        int8x8_t _r2 = float2int8(_p20, _p21);
                        int8x8_t _r3 = float2int8(_p30, _p31);
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        pp += 32;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vmulq_n_f32(bfloat2float(vld1_u16(p0)), vgetq_lane_f32(_scale, 0));
                        float32x4_t _p1 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + 4)), vgetq_lane_f32(_scale, 1));
                        float32x4_t _p2 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + 8)), vgetq_lane_f32(_scale, 2));
                        float32x4_t _p3 = vmulq_n_f32(bfloat2float(vld1_u16(p0 + 12)), vgetq_lane_f32(_scale, 3));
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else
                        int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                        int16x8x2_t _rr = vuzpq_s16(_r01, _r01);
                        vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_rr.val[0]), vget_low_s16(_rr.val[1]))));
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
                    float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                    float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax);
#else
                    float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax);
#endif
                    _scale = vbslq_f32(vceqq_f32(_absmax, _zero), _zero, _scale);

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

        return;
    }

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

                float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                float32x4_t _v127 = vdupq_n_f32(127.f);
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);

                    float32x4_t _p00 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _s0), _scale0, 0);
                    float32x4_t _p01 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _s1), _scale0, 0);
                    float32x4_t _p10 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s0), _scale0, 1);
                    float32x4_t _p11 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _s1), _scale0, 1);
                    int8x8_t _r0 = float2int8(_p00, _p01);
                    int8x8_t _r1 = float2int8(_p10, _p11);
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));

                    float32x4_t _p20 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 8)), _s0), _scale0, 2);
                    float32x4_t _p21 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 8)), _s1), _scale0, 2);
                    float32x4_t _p30 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 12)), _s0), _scale0, 3);
                    float32x4_t _p31 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 12)), _s1), _scale0, 3);
                    int8x8_t _r2 = float2int8(_p20, _p21);
                    int8x8_t _r3 = float2int8(_p30, _p31);
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                    float32x4_t _p40 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 16)), _s0), _scale1, 0);
                    float32x4_t _p41 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 16)), _s1), _scale1, 0);
                    float32x4_t _p50 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 20)), _s0), _scale1, 1);
                    float32x4_t _p51 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 20)), _s1), _scale1, 1);
                    int8x8_t _r4 = float2int8(_p40, _p41);
                    int8x8_t _r5 = float2int8(_p50, _p51);
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));

                    float32x4_t _p60 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 24)), _s0), _scale1, 2);
                    float32x4_t _p61 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 24)), _s1), _scale1, 2);
                    float32x4_t _p70 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 28)), _s0), _scale1, 3);
                    float32x4_t _p71 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 28)), _s1), _scale1, 3);
                    int8x8_t _r6 = float2int8(_p60, _p61);
                    int8x8_t _r7 = float2int8(_p70, _p71);
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                    pp += 64;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _s), _scale0, 0);
                    float32x4_t _p1 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s), _scale0, 1);
                    float32x4_t _p2 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 8)), _s), _scale0, 2);
                    float32x4_t _p3 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 12)), _s), _scale0, 3);
                    float32x4_t _p4 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 16)), _s), _scale1, 0);
                    float32x4_t _p5 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 20)), _s), _scale1, 1);
                    float32x4_t _p6 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 24)), _s), _scale1, 2);
                    float32x4_t _p7 = vmulq_laneq_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 28)), _s), _scale1, 3);
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

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                float32x4_t _v127 = vdupq_n_f32(127.f);
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

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

                float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
#else
                float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax0);
#endif
                _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    float32x4_t _p00 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _s0), vgetq_lane_f32(_scale, 0));
                    float32x4_t _p01 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4)), _s1), vgetq_lane_f32(_scale, 0));
                    float32x4_t _p10 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s0), vgetq_lane_f32(_scale, 1));
                    float32x4_t _p11 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 4)), _s1), vgetq_lane_f32(_scale, 1));
                    int8x8_t _r0 = float2int8(_p00, _p01);
                    int8x8_t _r1 = float2int8(_p10, _p11);
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    float32x4_t _p20 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 8)), _s0), vgetq_lane_f32(_scale, 2));
                    float32x4_t _p21 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 8)), _s1), vgetq_lane_f32(_scale, 2));
                    float32x4_t _p30 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 12)), _s0), vgetq_lane_f32(_scale, 3));
                    float32x4_t _p31 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + A_hstep * 4 + 12)), _s1), vgetq_lane_f32(_scale, 3));
                    int8x8_t _r2 = float2int8(_p20, _p21);
                    int8x8_t _r3 = float2int8(_p30, _p31);
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    pp += 32;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0)), _s), vgetq_lane_f32(_scale, 0));
                    float32x4_t _p1 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 4)), _s), vgetq_lane_f32(_scale, 1));
                    float32x4_t _p2 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 8)), _s), vgetq_lane_f32(_scale, 2));
                    float32x4_t _p3 = vmulq_n_f32(vmulq_f32(bfloat2float(vld1_u16(p0 + 12)), _s), vgetq_lane_f32(_scale, 3));
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else
                    int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                    int16x8x2_t _rr = vuzpq_s16(_r01, _r01);
                    vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_rr.val[0]), vget_low_s16(_rr.val[1]))));
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
                float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax);
#else
                float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax);
#endif
                _scale = vbslq_f32(vceqq_f32(_absmax, _zero), _zero, _scale);

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
