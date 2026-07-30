// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void pack_B_tile_wq_int8_i8mm(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_fp32_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_fp32_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void gemm_transB_packed_tile_wq_int8_i8mm(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
void pack_B_tile_wq_int8_asimddp(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size);
void quantize_A_tile_wq_int8_fp32_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_fp32_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void unpack_output_tile_wq_int8_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_elemtype, int output_transpose);
void gemm_transB_packed_tile_wq_int8_asimddp(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size);
#endif

static void quantize_A_tile_wq_int8_fp32(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        quantize_A_tile_wq_int8_fp32_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        quantize_A_tile_wq_int8_fp32_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vld1q_f32(p0a);
                        float32x4_t _p1 = vld1q_f32(p0a + A_hstep * 4);
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
                        float32x4x4_t _p = vld4q_f32(p0);
                        float32x4x4_t _q = vld4q_f32(p0 + 16);
                        float32x4x4_t _r = vld4q_f32(p0 + A_hstep * 4);
                        float32x4x4_t _s = vld4q_f32(p0 + A_hstep * 4 + 16);
                        int8x8_t _r0 = float2int8(vmulq_laneq_f32(_p.val[0], _scale0, 0), vmulq_laneq_f32(_q.val[0], _scale0, 0));
                        int8x8_t _r1 = float2int8(vmulq_laneq_f32(_p.val[1], _scale0, 1), vmulq_laneq_f32(_q.val[1], _scale0, 1));
                        int8x8_t _r2 = float2int8(vmulq_laneq_f32(_p.val[2], _scale0, 2), vmulq_laneq_f32(_q.val[2], _scale0, 2));
                        int8x8_t _r3 = float2int8(vmulq_laneq_f32(_p.val[3], _scale0, 3), vmulq_laneq_f32(_q.val[3], _scale0, 3));
                        int8x8_t _r4 = float2int8(vmulq_laneq_f32(_r.val[0], _scale1, 0), vmulq_laneq_f32(_s.val[0], _scale1, 0));
                        int8x8_t _r5 = float2int8(vmulq_laneq_f32(_r.val[1], _scale1, 1), vmulq_laneq_f32(_s.val[1], _scale1, 1));
                        int8x8_t _r6 = float2int8(vmulq_laneq_f32(_r.val[2], _scale1, 2), vmulq_laneq_f32(_s.val[2], _scale1, 2));
                        int8x8_t _r7 = float2int8(vmulq_laneq_f32(_r.val[3], _scale1, 3), vmulq_laneq_f32(_s.val[3], _scale1, 3));
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
                        float32x4x4_t _p = vld4q_f32(p0);
                        float32x4x4_t _q = vld4q_f32(p0 + A_hstep * 4);
                        int8x8_t _r01 = float2int8(vmulq_laneq_f32(_p.val[0], _scale0, 0), vmulq_laneq_f32(_p.val[1], _scale0, 1));
                        int8x8_t _r23 = float2int8(vmulq_laneq_f32(_p.val[2], _scale0, 2), vmulq_laneq_f32(_p.val[3], _scale0, 3));
                        int8x8_t _r45 = float2int8(vmulq_laneq_f32(_q.val[0], _scale1, 0), vmulq_laneq_f32(_q.val[1], _scale1, 1));
                        int8x8_t _r67 = float2int8(vmulq_laneq_f32(_q.val[2], _scale1, 2), vmulq_laneq_f32(_q.val[3], _scale1, 3));
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
                        float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _scale0);
                        float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + 4), _scale0);
                        float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _scale1);
                        float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 4), _scale1);
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
                        float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _scale0);
                        float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _scale1);
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
                    const float* p0a = p0;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vld1q_f32(p0a);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        float32x4_t _p1 = vld1q_f32(p0a + A_hstep);
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        float32x4_t _p2 = vld1q_f32(p0a + A_hstep * 2);
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                        float32x4_t _p3 = vld1q_f32(p0a + A_hstep * 3);
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                        float32x4_t _p4 = vld1q_f32(p0a + A_hstep * 4);
                        _absmax4 = vmaxq_f32(_absmax4, vabsq_f32(_p4));
                        float32x4_t _p5 = vld1q_f32(p0a + A_hstep * 5);
                        _absmax5 = vmaxq_f32(_absmax5, vabsq_f32(_p5));
                        float32x4_t _p6 = vld1q_f32(p0a + A_hstep * 6);
                        _absmax6 = vmaxq_f32(_absmax6, vabsq_f32(_p6));
                        float32x4_t _p7 = vld1q_f32(p0a + A_hstep * 7);
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
                        float32x4_t _p0 = vdupq_n_f32(p0a[0]);
                        _p0 = vsetq_lane_f32(p0a[A_hstep], _p0, 1);
                        _p0 = vsetq_lane_f32(p0a[A_hstep * 2], _p0, 2);
                        _p0 = vsetq_lane_f32(p0a[A_hstep * 3], _p0, 3);
                        float32x4_t _p1 = vdupq_n_f32(p0a[A_hstep * 4]);
                        _p1 = vsetq_lane_f32(p0a[A_hstep * 5], _p1, 1);
                        _p1 = vsetq_lane_f32(p0a[A_hstep * 6], _p1, 2);
                        _p1 = vsetq_lane_f32(p0a[A_hstep * 7], _p1, 3);
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
                        float32x4_t _p00 = vld1q_f32(p0);
                        float32x4_t _p01 = vld1q_f32(p0 + 4);
                        float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                        float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
                        float32x4_t _p20 = vld1q_f32(p0 + A_hstep * 2);
                        float32x4_t _p21 = vld1q_f32(p0 + A_hstep * 2 + 4);
                        float32x4_t _p30 = vld1q_f32(p0 + A_hstep * 3);
                        float32x4_t _p31 = vld1q_f32(p0 + A_hstep * 3 + 4);
                        float32x4_t _p40 = vld1q_f32(p0 + A_hstep * 4);
                        float32x4_t _p41 = vld1q_f32(p0 + A_hstep * 4 + 4);
                        float32x4_t _p50 = vld1q_f32(p0 + A_hstep * 5);
                        float32x4_t _p51 = vld1q_f32(p0 + A_hstep * 5 + 4);
                        float32x4_t _p60 = vld1q_f32(p0 + A_hstep * 6);
                        float32x4_t _p61 = vld1q_f32(p0 + A_hstep * 6 + 4);
                        float32x4_t _p70 = vld1q_f32(p0 + A_hstep * 7);
                        float32x4_t _p71 = vld1q_f32(p0 + A_hstep * 7 + 4);
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
                        float32x4_t _p0 = vld1q_f32(p0);
                        float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                        float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                        float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);
                        float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 4);
                        float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 5);
                        float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 6);
                        float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 7);
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
                        float32x4_t _p01 = vcombine_f32(vld1_f32(p0), vld1_f32(p0 + A_hstep));
                        float32x4_t _p23 = vcombine_f32(vld1_f32(p0 + A_hstep * 2), vld1_f32(p0 + A_hstep * 3));
                        float32x4_t _p45 = vcombine_f32(vld1_f32(p0 + A_hstep * 4), vld1_f32(p0 + A_hstep * 5));
                        float32x4_t _p67 = vcombine_f32(vld1_f32(p0 + A_hstep * 6), vld1_f32(p0 + A_hstep * 7));
                        int8x8_t _r0 = float2int8(vmulq_f32(_p01, _scale01.val[0]), vmulq_f32(_p23, _scale01.val[1]));
                        int8x8_t _r1 = float2int8(vmulq_f32(_p45, _scale45.val[0]), vmulq_f32(_p67, _scale45.val[1]));
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        pp += 16;
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = float32x4_t();
                        _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                        _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                        _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                        _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);
                        float32x4_t _p1 = float32x4_t();
                        _p1 = vsetq_lane_f32(p0[A_hstep * 4], _p1, 0);
                        _p1 = vsetq_lane_f32(p0[A_hstep * 5], _p1, 1);
                        _p1 = vsetq_lane_f32(p0[A_hstep * 6], _p1, 2);
                        _p1 = vsetq_lane_f32(p0[A_hstep * 7], _p1, 3);
                        int8x8_t _r0 = float2int8(vmulq_f32(_p0, _scale0), vmulq_f32(_p1, _scale1));
                        vst1_s8(pp, _r0);
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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 4)
                {
                    float32x4_t _absmax = vdupq_n_f32(0.f);
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vld1q_f32(p0a);
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
                        float32x4x4_t _p = vld4q_f32(p0);
                        float32x4x4_t _q = vld4q_f32(p0 + 16);
                        int8x8_t _r0 = float2int8(vmulq_lane_f32(_p.val[0], vget_low_f32(_scale), 0), vmulq_lane_f32(_q.val[0], vget_low_f32(_scale), 0));
                        int8x8_t _r1 = float2int8(vmulq_lane_f32(_p.val[1], vget_low_f32(_scale), 1), vmulq_lane_f32(_q.val[1], vget_low_f32(_scale), 1));
                        int8x8_t _r2 = float2int8(vmulq_lane_f32(_p.val[2], vget_high_f32(_scale), 0), vmulq_lane_f32(_q.val[2], vget_high_f32(_scale), 0));
                        int8x8_t _r3 = float2int8(vmulq_lane_f32(_p.val[3], vget_high_f32(_scale), 1), vmulq_lane_f32(_q.val[3], vget_high_f32(_scale), 1));
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        pp += 32;
                        p0 += 32;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4x4_t _p = vld4q_f32(p0);
                        int8x8_t _r01 = float2int8(vmulq_lane_f32(_p.val[0], vget_low_f32(_scale), 0), vmulq_lane_f32(_p.val[1], vget_low_f32(_scale), 1));
                        int8x8_t _r23 = float2int8(vmulq_lane_f32(_p.val[2], vget_high_f32(_scale), 0), vmulq_lane_f32(_p.val[3], vget_high_f32(_scale), 1));
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
                        float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _scale);
                        float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + 4), _scale);
                        int8x8_t _r = float2int8(_p0, _p1);
                        _r = vzip_s8(_r, vext_s8(_r, _r, 4)).val[0];
                        vst1_s8(pp, _r);
                        pp += 8;
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vmulq_f32(vld1q_f32(p0), _scale);
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
                    const float* p0a = p0;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vld1q_f32(p0a);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        float32x4_t _p1 = vld1q_f32(p0a + A_hstep);
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        float32x4_t _p2 = vld1q_f32(p0a + A_hstep * 2);
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                        float32x4_t _p3 = vld1q_f32(p0a + A_hstep * 3);
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
                        float32x4_t _p = vdupq_n_f32(p0a[0]);
                        _p = vsetq_lane_f32(p0a[A_hstep], _p, 1);
                        _p = vsetq_lane_f32(p0a[A_hstep * 2], _p, 2);
                        _p = vsetq_lane_f32(p0a[A_hstep * 3], _p, 3);
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
                        float32x4_t _p00 = vld1q_f32(p0);
                        float32x4_t _p01 = vld1q_f32(p0 + 4);
                        float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                        float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
                        float32x4_t _p20 = vld1q_f32(p0 + A_hstep * 2);
                        float32x4_t _p21 = vld1q_f32(p0 + A_hstep * 2 + 4);
                        float32x4_t _p30 = vld1q_f32(p0 + A_hstep * 3);
                        float32x4_t _p31 = vld1q_f32(p0 + A_hstep * 3 + 4);
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
                        float32x4_t _p0 = vld1q_f32(p0);
                        float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                        float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                        float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);
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
                        float32x4_t _p01 = vcombine_f32(vld1_f32(p0), vld1_f32(p0 + A_hstep));
                        float32x4_t _p23 = vcombine_f32(vld1_f32(p0 + A_hstep * 2), vld1_f32(p0 + A_hstep * 3));
                        int8x8_t _r0 = float2int8(vmulq_f32(_p01, _scale01.val[0]), vmulq_f32(_p23, _scale01.val[1]));
                        vst1_s8(pp, _r0);
                        pp += 8;
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = float32x4_t();
                        _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                        _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                        _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                        _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);
                        int8x8_t _r0 = float2int8(vmulq_f32(_p0, _scale), _p0);
                        vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r0), 0);
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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0;
                int kk = 0;
#if __ARM_NEON
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = vld1q_f32(p0a);
                    _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                    float32x4_t _p1 = vld1q_f32(p0a + A_hstep);
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
                    float v0 = p0a[0];
                    float v1 = p0a[A_hstep];
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
                    float32x4_t _p00 = vld1q_f32(p0);
                    float32x4_t _p01 = vld1q_f32(p0 + 4);
                    float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                    float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
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
                    float32x4_t _p0 = vld1q_f32(p0);
                    float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
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
                    float v00 = p0[0];
                    float v01 = p0[1];
                    float v10 = p0[A_hstep];
                    float v11 = p0[A_hstep + 1];
                    *pp++ = float2int8(v00 * scale0);
                    *pp++ = float2int8(v01 * scale0);
                    *pp++ = float2int8(v10 * scale1);
                    *pp++ = float2int8(v11 * scale1);
                    p0 += 2;
                }
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    float v0 = p0[0];
                    float v1 = p0[A_hstep];
                    *pp++ = float2int8(v0 * scale0);
                    *pp++ = float2int8(v1 * scale1);
                    p0++;
                }

                pd += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const float* p0a = p0;

                float absmax = 0.f;
                int kk = 0;
#if __ARM_NEON
                float32x4_t _absmax = vdupq_n_f32(0.f);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p = vld1q_f32(p0a);
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
                    float v = *p0a++;
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
                    float32x4_t _p0 = vld1q_f32(p0);
                    float32x4_t _p1 = vld1q_f32(p0 + 4);
                    vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale)));
                    pp += 8;
                    p0 += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p = vld1q_f32(p0);
                    int8x8_t _r = float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale));
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                    pp += 4;
                    p0 += 4;
                }
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    float v = *p0++;
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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vmulq_n_f32(vabsq_f32(vld1q_f32(p0a)), psa[0]);
                    float32x4_t _p1 = vmulq_n_f32(vabsq_f32(vld1q_f32(p0a + A_hstep * 4)), psa[0]);
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
                    float32x4x4_t _p = vld4q_f32(p0);
                    float32x4x4_t _q = vld4q_f32(p0 + 16);
                    float32x4x4_t _r = vld4q_f32(p0 + A_hstep * 4);
                    float32x4x4_t _s = vld4q_f32(p0 + A_hstep * 4 + 16);
                    int8x8_t _r0 = float2int8(vmulq_laneq_f32(vmulq_f32(_p.val[0], _s0), _scale0, 0), vmulq_laneq_f32(vmulq_f32(_q.val[0], _s1), _scale0, 0));
                    int8x8_t _r1 = float2int8(vmulq_laneq_f32(vmulq_f32(_p.val[1], _s0), _scale0, 1), vmulq_laneq_f32(vmulq_f32(_q.val[1], _s1), _scale0, 1));
                    int8x8_t _r2 = float2int8(vmulq_laneq_f32(vmulq_f32(_p.val[2], _s0), _scale0, 2), vmulq_laneq_f32(vmulq_f32(_q.val[2], _s1), _scale0, 2));
                    int8x8_t _r3 = float2int8(vmulq_laneq_f32(vmulq_f32(_p.val[3], _s0), _scale0, 3), vmulq_laneq_f32(vmulq_f32(_q.val[3], _s1), _scale0, 3));
                    int8x8_t _r4 = float2int8(vmulq_laneq_f32(vmulq_f32(_r.val[0], _s0), _scale1, 0), vmulq_laneq_f32(vmulq_f32(_s.val[0], _s1), _scale1, 0));
                    int8x8_t _r5 = float2int8(vmulq_laneq_f32(vmulq_f32(_r.val[1], _s0), _scale1, 1), vmulq_laneq_f32(vmulq_f32(_s.val[1], _s1), _scale1, 1));
                    int8x8_t _r6 = float2int8(vmulq_laneq_f32(vmulq_f32(_r.val[2], _s0), _scale1, 2), vmulq_laneq_f32(vmulq_f32(_s.val[2], _s1), _scale1, 2));
                    int8x8_t _r7 = float2int8(vmulq_laneq_f32(vmulq_f32(_r.val[3], _s0), _scale1, 3), vmulq_laneq_f32(vmulq_f32(_s.val[3], _s1), _scale1, 3));
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
                    float32x4x4_t _p = vld4q_f32(p0);
                    float32x4x4_t _q = vld4q_f32(p0 + A_hstep * 4);
                    int8x8_t _r01 = float2int8(vmulq_laneq_f32(vmulq_f32(_p.val[0], _s), _scale0, 0), vmulq_laneq_f32(vmulq_f32(_p.val[1], _s), _scale0, 1));
                    int8x8_t _r23 = float2int8(vmulq_laneq_f32(vmulq_f32(_p.val[2], _s), _scale0, 2), vmulq_laneq_f32(vmulq_f32(_p.val[3], _s), _scale0, 3));
                    int8x8_t _r45 = float2int8(vmulq_laneq_f32(vmulq_f32(_q.val[0], _s), _scale1, 0), vmulq_laneq_f32(vmulq_f32(_q.val[1], _s), _scale1, 1));
                    int8x8_t _r67 = float2int8(vmulq_laneq_f32(vmulq_f32(_q.val[2], _s), _scale1, 2), vmulq_laneq_f32(vmulq_f32(_q.val[3], _s), _scale1, 3));
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
                    float32x4_t _p0 = vmulq_n_f32(vmulq_f32(vld1q_f32(p0), _scale0), ps[0]);
                    float32x4_t _p1 = vmulq_n_f32(vmulq_f32(vld1q_f32(p0 + 4), _scale0), ps[1]);
                    float32x4_t _p2 = vmulq_n_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _scale1), ps[0]);
                    float32x4_t _p3 = vmulq_n_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 4), _scale1), ps[1]);
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
                    float32x4_t _p0 = vmulq_n_f32(vmulq_f32(vld1q_f32(p0), _scale0), ps[0]);
                    float32x4_t _p1 = vmulq_n_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _scale1), ps[0]);
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
                const float* p0a = p0;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    float32x4_t _p0 = vld1q_f32(p0a);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    float32x4_t _p1 = vld1q_f32(p0a + A_hstep);
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    float32x4_t _p2 = vld1q_f32(p0a + A_hstep * 2);
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(_p2), _s));
                    float32x4_t _p3 = vld1q_f32(p0a + A_hstep * 3);
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(_p3), _s));
                    float32x4_t _p4 = vld1q_f32(p0a + A_hstep * 4);
                    _absmax4 = vmaxq_f32(_absmax4, vmulq_f32(vabsq_f32(_p4), _s));
                    float32x4_t _p5 = vld1q_f32(p0a + A_hstep * 5);
                    _absmax5 = vmaxq_f32(_absmax5, vmulq_f32(vabsq_f32(_p5), _s));
                    float32x4_t _p6 = vld1q_f32(p0a + A_hstep * 6);
                    _absmax6 = vmaxq_f32(_absmax6, vmulq_f32(vabsq_f32(_p6), _s));
                    float32x4_t _p7 = vld1q_f32(p0a + A_hstep * 7);
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
                    float32x4_t _p0 = vdupq_n_f32(p0a[0]);
                    _p0 = vsetq_lane_f32(p0a[A_hstep], _p0, 1);
                    _p0 = vsetq_lane_f32(p0a[A_hstep * 2], _p0, 2);
                    _p0 = vsetq_lane_f32(p0a[A_hstep * 3], _p0, 3);
                    float32x4_t _p1 = vdupq_n_f32(p0a[A_hstep * 4]);
                    _p1 = vsetq_lane_f32(p0a[A_hstep * 5], _p1, 1);
                    _p1 = vsetq_lane_f32(p0a[A_hstep * 6], _p1, 2);
                    _p1 = vsetq_lane_f32(p0a[A_hstep * 7], _p1, 3);
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
                    float32x4_t _p00 = vmulq_f32(vld1q_f32(p0), _s0);
                    float32x4_t _p01 = vmulq_f32(vld1q_f32(p0 + 4), _s1);
                    float32x4_t _p10 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s0);
                    float32x4_t _p11 = vmulq_f32(vld1q_f32(p0 + A_hstep + 4), _s1);
                    float32x4_t _p20 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _s0);
                    float32x4_t _p21 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2 + 4), _s1);
                    float32x4_t _p30 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _s0);
                    float32x4_t _p31 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3 + 4), _s1);
                    float32x4_t _p40 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _s0);
                    float32x4_t _p41 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 4), _s1);
                    float32x4_t _p50 = vmulq_f32(vld1q_f32(p0 + A_hstep * 5), _s0);
                    float32x4_t _p51 = vmulq_f32(vld1q_f32(p0 + A_hstep * 5 + 4), _s1);
                    float32x4_t _p60 = vmulq_f32(vld1q_f32(p0 + A_hstep * 6), _s0);
                    float32x4_t _p61 = vmulq_f32(vld1q_f32(p0 + A_hstep * 6 + 4), _s1);
                    float32x4_t _p70 = vmulq_f32(vld1q_f32(p0 + A_hstep * 7), _s0);
                    float32x4_t _p71 = vmulq_f32(vld1q_f32(p0 + A_hstep * 7 + 4), _s1);
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
                    float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _s);
                    float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s);
                    float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _s);
                    float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _s);
                    float32x4_t _p4 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _s);
                    float32x4_t _p5 = vmulq_f32(vld1q_f32(p0 + A_hstep * 5), _s);
                    float32x4_t _p6 = vmulq_f32(vld1q_f32(p0 + A_hstep * 6), _s);
                    float32x4_t _p7 = vmulq_f32(vld1q_f32(p0 + A_hstep * 7), _s);
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
                    float32x4_t _p01 = vcombine_f32(vld1_f32(p0), vld1_f32(p0 + A_hstep));
                    float32x4_t _p23 = vcombine_f32(vld1_f32(p0 + A_hstep * 2), vld1_f32(p0 + A_hstep * 3));
                    float32x4_t _p45 = vcombine_f32(vld1_f32(p0 + A_hstep * 4), vld1_f32(p0 + A_hstep * 5));
                    float32x4_t _p67 = vcombine_f32(vld1_f32(p0 + A_hstep * 6), vld1_f32(p0 + A_hstep * 7));
                    int8x8_t _r0 = float2int8(vmulq_f32(_p01, _s01), vmulq_f32(_p23, _s23));
                    int8x8_t _r1 = float2int8(vmulq_f32(_p45, _s45), vmulq_f32(_p67, _s67));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    pp += 16;
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = ps[0];
                    float32x4_t _p0 = float32x4_t();
                    _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                    _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                    _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                    _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);
                    float32x4_t _p1 = float32x4_t();
                    _p1 = vsetq_lane_f32(p0[A_hstep * 4], _p1, 0);
                    _p1 = vsetq_lane_f32(p0[A_hstep * 5], _p1, 1);
                    _p1 = vsetq_lane_f32(p0[A_hstep * 6], _p1, 2);
                    _p1 = vsetq_lane_f32(p0[A_hstep * 7], _p1, 3);
                    _p0 = vmulq_n_f32(_p0, s);
                    _p1 = vmulq_n_f32(_p1, s);
                    int8x8_t _r0 = float2int8(vmulq_f32(_p0, _scale0), vmulq_f32(_p1, _scale1));
                    vst1_s8(pp, _r0);
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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            if (elempack == 4)
            {
                float32x4_t _absmax = vdupq_n_f32(0.f);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vmulq_n_f32(vabsq_f32(vld1q_f32(p0a)), psa[0]);
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
                    float32x4x4_t _p = vld4q_f32(p0);
                    float32x4x4_t _q = vld4q_f32(p0 + 16);
                    int8x8_t _r0 = float2int8(vmulq_lane_f32(vmulq_f32(_p.val[0], _s0), vget_low_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(_q.val[0], _s1), vget_low_f32(_scale), 0));
                    int8x8_t _r1 = float2int8(vmulq_lane_f32(vmulq_f32(_p.val[1], _s0), vget_low_f32(_scale), 1), vmulq_lane_f32(vmulq_f32(_q.val[1], _s1), vget_low_f32(_scale), 1));
                    int8x8_t _r2 = float2int8(vmulq_lane_f32(vmulq_f32(_p.val[2], _s0), vget_high_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(_q.val[2], _s1), vget_high_f32(_scale), 0));
                    int8x8_t _r3 = float2int8(vmulq_lane_f32(vmulq_f32(_p.val[3], _s0), vget_high_f32(_scale), 1), vmulq_lane_f32(vmulq_f32(_q.val[3], _s1), vget_high_f32(_scale), 1));
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
#if __ARM_FEATURE_DOTPROD
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4x4_t _p = vld4q_f32(p0);
                    int8x8_t _r01 = float2int8(vmulq_lane_f32(vmulq_f32(_p.val[0], _s), vget_low_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(_p.val[1], _s), vget_low_f32(_scale), 1));
                    int8x8_t _r23 = float2int8(vmulq_lane_f32(vmulq_f32(_p.val[2], _s), vget_high_f32(_scale), 0), vmulq_lane_f32(vmulq_f32(_p.val[3], _s), vget_high_f32(_scale), 1));
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else // __ARM_FEATURE_DOTPROD
                    float32x4_t _p0 = vmulq_f32(vmulq_n_f32(vld1q_f32(p0), ps[0]), _scale);
                    float32x4_t _p1 = vmulq_f32(vmulq_n_f32(vld1q_f32(p0 + 4), ps[1]), _scale);
                    float32x4_t _p2 = vmulq_f32(vmulq_n_f32(vld1q_f32(p0 + 8), ps[2]), _scale);
                    float32x4_t _p3 = vmulq_f32(vmulq_n_f32(vld1q_f32(p0 + 12), ps[3]), _scale);
                    int8x8x2_t _r01;
                    _r01.val[0] = float2int8(_p0, _p2);
                    _r01.val[1] = float2int8(_p1, _p3);
                    vst2_s8(pp, _r01);
#endif
                    pp += 16;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _p0 = vmulq_n_f32(vmulq_f32(vld1q_f32(p0), _scale), ps[0]);
                    float32x4_t _p1 = vmulq_n_f32(vmulq_f32(vld1q_f32(p0 + 4), _scale), ps[1]);
                    int8x8_t _r = float2int8(_p0, _p1);
                    _r = vzip_s8(_r, vext_s8(_r, _r, 4)).val[0];
                    vst1_s8(pp, _r);
                    pp += 8;
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vmulq_n_f32(vmulq_f32(vld1q_f32(p0), _scale), ps[0]);
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
                const float* p0a = p0;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    float32x4_t _p0 = vld1q_f32(p0a);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    float32x4_t _p1 = vld1q_f32(p0a + A_hstep);
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    float32x4_t _p2 = vld1q_f32(p0a + A_hstep * 2);
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(_p2), _s));
                    float32x4_t _p3 = vld1q_f32(p0a + A_hstep * 3);
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
                    float32x4_t _p = vdupq_n_f32(p0a[0]);
                    _p = vsetq_lane_f32(p0a[A_hstep], _p, 1);
                    _p = vsetq_lane_f32(p0a[A_hstep * 2], _p, 2);
                    _p = vsetq_lane_f32(p0a[A_hstep * 3], _p, 3);
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
                    float32x4_t _p00 = vmulq_f32(vld1q_f32(p0), _s0);
                    float32x4_t _p01 = vmulq_f32(vld1q_f32(p0 + 4), _s1);
                    float32x4_t _p10 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s0);
                    float32x4_t _p11 = vmulq_f32(vld1q_f32(p0 + A_hstep + 4), _s1);
                    float32x4_t _p20 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _s0);
                    float32x4_t _p21 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2 + 4), _s1);
                    float32x4_t _p30 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _s0);
                    float32x4_t _p31 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3 + 4), _s1);
                    int8x8_t _r0 = float2int8(vmulq_lane_f32(_p00, vget_low_f32(_scale), 0), vmulq_lane_f32(_p01, vget_low_f32(_scale), 0));
                    int8x8_t _r1 = float2int8(vmulq_lane_f32(_p10, vget_low_f32(_scale), 1), vmulq_lane_f32(_p11, vget_low_f32(_scale), 1));
                    int8x8_t _r2 = float2int8(vmulq_lane_f32(_p20, vget_high_f32(_scale), 0), vmulq_lane_f32(_p21, vget_high_f32(_scale), 0));
                    int8x8_t _r3 = float2int8(vmulq_lane_f32(_p30, vget_high_f32(_scale), 1), vmulq_lane_f32(_p31, vget_high_f32(_scale), 1));
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
                    float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _s);
                    float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s);
                    float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _s);
                    float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _s);
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
                    ps += 4;
                }
                float32x4x2_t _scale01 = vzipq_f32(_scale, _scale);
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _s = vcombine_f32(vld1_f32(ps), vld1_f32(ps));
                    float32x4_t _s01 = vmulq_f32(_s, _scale01.val[0]);
                    float32x4_t _s23 = vmulq_f32(_s, _scale01.val[1]);
                    float32x4_t _p01 = vcombine_f32(vld1_f32(p0), vld1_f32(p0 + A_hstep));
                    float32x4_t _p23 = vcombine_f32(vld1_f32(p0 + A_hstep * 2), vld1_f32(p0 + A_hstep * 3));
                    int8x8_t _r0 = float2int8(vmulq_f32(_p01, _s01), vmulq_f32(_p23, _s23));
                    vst1_s8(pp, _r0);
                    pp += 8;
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = ps[0];
                    float32x4_t _p0 = float32x4_t();
                    _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                    _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                    _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                    _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);
                    _p0 = vmulq_n_f32(_p0, s);
                    _p0 = vmulq_f32(_p0, _scale);
                    int8x8_t _r0 = float2int8(_p0, _p0);
                    pp[0] = vget_lane_s8(_r0, 0);
                    pp[1] = vget_lane_s8(_r0, 1);
                    pp[2] = vget_lane_s8(_r0, 2);
                    pp[3] = vget_lane_s8(_r0, 3);
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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0;
            const float* psa = ps;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _s = vld1q_f32(psa);
                float32x4_t _p0 = vld1q_f32(p0a);
                _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                float32x4_t _p1 = vld1q_f32(p0a + A_hstep);
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
                float v0 = p0a[0];
                float v1 = p0a[A_hstep];
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
                float32x4_t _p00 = vmulq_f32(vld1q_f32(p0), _s0);
                float32x4_t _p01 = vmulq_f32(vld1q_f32(p0 + 4), _s1);
                float32x4_t _p10 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s0);
                float32x4_t _p11 = vmulq_f32(vld1q_f32(p0 + A_hstep + 4), _s1);
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
                float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _s);
                float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + A_hstep), _s);
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
                float v00 = p0[0];
                float v01 = p0[1];
                float v10 = p0[A_hstep];
                float v11 = p0[A_hstep + 1];
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
                float v0 = p0[0];
                float v1 = p0[A_hstep];
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
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            const float* p0a = p0;
            const float* psa = ps;

            float absmax = 0.f;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _p = vld1q_f32(p0a);
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
                float v = *p0a++;
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
                float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), vld1q_f32(ps));
                float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + 4), vld1q_f32(ps + 4));
                vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale)));
                pp += 8;
                p0 += 8;
                ps += 8;
            }
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _p = vmulq_f32(vld1q_f32(p0), vld1q_f32(ps));
                int8x8_t _r = float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale));
                vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                pp += 4;
                p0 += 4;
                ps += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk0; kk++)
            {
                float v = *p0++;
                v *= *ps++;
                *pp++ = float2int8(v * scale);
            }
        }
    }
}

static void transpose_quantize_A_tile_wq_int8_fp32(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_quantize_A_tile_wq_int8_fp32_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_quantize_A_tile_wq_int8_fp32_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
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
            const float* p0 = (const float*)A + (size_t)k * A_hstep + (i + ii) * elempack;
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
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vld1q_f32(p0a)));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vld1q_f32(p0a + 4)));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(vld1q_f32(p0a + 8)));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(vld1q_f32(p0a + 12)));
                        _absmax4 = vmaxq_f32(_absmax4, vabsq_f32(vld1q_f32(p0a + 16)));
                        _absmax5 = vmaxq_f32(_absmax5, vabsq_f32(vld1q_f32(p0a + 20)));
                        _absmax6 = vmaxq_f32(_absmax6, vabsq_f32(vld1q_f32(p0a + 24)));
                        _absmax7 = vmaxq_f32(_absmax7, vabsq_f32(vld1q_f32(p0a + 28)));
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
                        float32x4_t _p00 = vmulq_laneq_f32(vld1q_f32(p0), _scale0, 0);
                        float32x4_t _p01 = vmulq_laneq_f32(vld1q_f32(p0 + A_hstep * 4), _scale0, 0);
                        float32x4_t _p10 = vmulq_laneq_f32(vld1q_f32(p0 + 4), _scale0, 1);
                        float32x4_t _p11 = vmulq_laneq_f32(vld1q_f32(p0 + A_hstep * 4 + 4), _scale0, 1);
                        int8x8_t _r0 = float2int8(_p00, _p01);
                        int8x8_t _r1 = float2int8(_p10, _p11);
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));

                        float32x4_t _p20 = vmulq_laneq_f32(vld1q_f32(p0 + 8), _scale0, 2);
                        float32x4_t _p21 = vmulq_laneq_f32(vld1q_f32(p0 + A_hstep * 4 + 8), _scale0, 2);
                        float32x4_t _p30 = vmulq_laneq_f32(vld1q_f32(p0 + 12), _scale0, 3);
                        float32x4_t _p31 = vmulq_laneq_f32(vld1q_f32(p0 + A_hstep * 4 + 12), _scale0, 3);
                        int8x8_t _r2 = float2int8(_p20, _p21);
                        int8x8_t _r3 = float2int8(_p30, _p31);
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                        float32x4_t _p40 = vmulq_laneq_f32(vld1q_f32(p0 + 16), _scale1, 0);
                        float32x4_t _p41 = vmulq_laneq_f32(vld1q_f32(p0 + A_hstep * 4 + 16), _scale1, 0);
                        float32x4_t _p50 = vmulq_laneq_f32(vld1q_f32(p0 + 20), _scale1, 1);
                        float32x4_t _p51 = vmulq_laneq_f32(vld1q_f32(p0 + A_hstep * 4 + 20), _scale1, 1);
                        int8x8_t _r4 = float2int8(_p40, _p41);
                        int8x8_t _r5 = float2int8(_p50, _p51);
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));

                        float32x4_t _p60 = vmulq_laneq_f32(vld1q_f32(p0 + 24), _scale1, 2);
                        float32x4_t _p61 = vmulq_laneq_f32(vld1q_f32(p0 + A_hstep * 4 + 24), _scale1, 2);
                        float32x4_t _p70 = vmulq_laneq_f32(vld1q_f32(p0 + 28), _scale1, 3);
                        float32x4_t _p71 = vmulq_laneq_f32(vld1q_f32(p0 + A_hstep * 4 + 28), _scale1, 3);
                        int8x8_t _r6 = float2int8(_p60, _p61);
                        int8x8_t _r7 = float2int8(_p70, _p71);
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                        pp += 64;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vmulq_laneq_f32(vld1q_f32(p0), _scale0, 0);
                        float32x4_t _p1 = vmulq_laneq_f32(vld1q_f32(p0 + 4), _scale0, 1);
                        float32x4_t _p2 = vmulq_laneq_f32(vld1q_f32(p0 + 8), _scale0, 2);
                        float32x4_t _p3 = vmulq_laneq_f32(vld1q_f32(p0 + 12), _scale0, 3);
                        float32x4_t _p4 = vmulq_laneq_f32(vld1q_f32(p0 + 16), _scale1, 0);
                        float32x4_t _p5 = vmulq_laneq_f32(vld1q_f32(p0 + 20), _scale1, 1);
                        float32x4_t _p6 = vmulq_laneq_f32(vld1q_f32(p0 + 24), _scale1, 2);
                        float32x4_t _p7 = vmulq_laneq_f32(vld1q_f32(p0 + 28), _scale1, 3);
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
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vld1q_f32(p0a);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        float32x4_t _p1 = vld1q_f32(p0a + 4);
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
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
                        float32x4_t _p00 = vld1q_f32(p0);
                        float32x4_t _p01 = vld1q_f32(p0 + 4);
                        float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                        float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
                        float32x4_t _p20 = vld1q_f32(p0 + A_hstep * 2);
                        float32x4_t _p21 = vld1q_f32(p0 + A_hstep * 2 + 4);
                        float32x4_t _p30 = vld1q_f32(p0 + A_hstep * 3);
                        float32x4_t _p31 = vld1q_f32(p0 + A_hstep * 3 + 4);
                        float32x4_t _p40 = vld1q_f32(p0 + A_hstep * 4);
                        float32x4_t _p41 = vld1q_f32(p0 + A_hstep * 4 + 4);
                        float32x4_t _p50 = vld1q_f32(p0 + A_hstep * 5);
                        float32x4_t _p51 = vld1q_f32(p0 + A_hstep * 5 + 4);
                        float32x4_t _p60 = vld1q_f32(p0 + A_hstep * 6);
                        float32x4_t _p61 = vld1q_f32(p0 + A_hstep * 6 + 4);
                        float32x4_t _p70 = vld1q_f32(p0 + A_hstep * 7);
                        float32x4_t _p71 = vld1q_f32(p0 + A_hstep * 7 + 4);
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
                        float32x4_t _p00 = vld1q_f32(p0);
                        float32x4_t _p01 = vld1q_f32(p0 + 4);
                        float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                        float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
                        float32x4_t _p20 = vld1q_f32(p0 + A_hstep * 2);
                        float32x4_t _p21 = vld1q_f32(p0 + A_hstep * 2 + 4);
                        float32x4_t _p30 = vld1q_f32(p0 + A_hstep * 3);
                        float32x4_t _p31 = vld1q_f32(p0 + A_hstep * 3 + 4);
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
                        float32x4_t _p00 = vld1q_f32(p0);
                        float32x4_t _p01 = vld1q_f32(p0 + 4);
                        float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                        float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
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
                        float32x4_t _p0 = vld1q_f32(p0);
                        float32x4_t _p1 = vld1q_f32(p0 + 4);
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
            const float* p0 = (const float*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vld1q_f32(p0a)));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vld1q_f32(p0a + 4)));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(vld1q_f32(p0a + 8)));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(vld1q_f32(p0a + 12)));
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
                        float32x4_t _p00 = vld1q_f32(p0);
                        float32x4_t _p01 = vld1q_f32(p0 + A_hstep * 4);
                        float32x4_t _p10 = vld1q_f32(p0 + 4);
                        float32x4_t _p11 = vld1q_f32(p0 + A_hstep * 4 + 4);
#if __aarch64__
                        _p00 = vmulq_laneq_f32(_p00, _scale, 0);
                        _p01 = vmulq_laneq_f32(_p01, _scale, 0);
                        _p10 = vmulq_laneq_f32(_p10, _scale, 1);
                        _p11 = vmulq_laneq_f32(_p11, _scale, 1);
#else
                        _p00 = vmulq_lane_f32(_p00, vget_low_f32(_scale), 0);
                        _p01 = vmulq_lane_f32(_p01, vget_low_f32(_scale), 0);
                        _p10 = vmulq_lane_f32(_p10, vget_low_f32(_scale), 1);
                        _p11 = vmulq_lane_f32(_p11, vget_low_f32(_scale), 1);
#endif
                        int8x8_t _r0 = float2int8(_p00, _p01);
                        int8x8_t _r1 = float2int8(_p10, _p11);
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));

                        float32x4_t _p20 = vld1q_f32(p0 + 8);
                        float32x4_t _p21 = vld1q_f32(p0 + A_hstep * 4 + 8);
                        float32x4_t _p30 = vld1q_f32(p0 + 12);
                        float32x4_t _p31 = vld1q_f32(p0 + A_hstep * 4 + 12);
#if __aarch64__
                        _p20 = vmulq_laneq_f32(_p20, _scale, 2);
                        _p21 = vmulq_laneq_f32(_p21, _scale, 2);
                        _p30 = vmulq_laneq_f32(_p30, _scale, 3);
                        _p31 = vmulq_laneq_f32(_p31, _scale, 3);
#else
                        _p20 = vmulq_lane_f32(_p20, vget_high_f32(_scale), 0);
                        _p21 = vmulq_lane_f32(_p21, vget_high_f32(_scale), 0);
                        _p30 = vmulq_lane_f32(_p30, vget_high_f32(_scale), 1);
                        _p31 = vmulq_lane_f32(_p31, vget_high_f32(_scale), 1);
#endif
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
                        float32x4_t _p0 = vld1q_f32(p0);
                        float32x4_t _p1 = vld1q_f32(p0 + 4);
                        float32x4_t _p2 = vld1q_f32(p0 + 8);
                        float32x4_t _p3 = vld1q_f32(p0 + 12);
#if __aarch64__
                        _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                        _p1 = vmulq_laneq_f32(_p1, _scale, 1);
                        _p2 = vmulq_laneq_f32(_p2, _scale, 2);
                        _p3 = vmulq_laneq_f32(_p3, _scale, 3);
#else
                        _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale), 0);
                        _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale), 1);
                        _p2 = vmulq_lane_f32(_p2, vget_high_f32(_scale), 0);
                        _p3 = vmulq_lane_f32(_p3, vget_high_f32(_scale), 1);
#endif
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
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vld1q_f32(p0a);
                        _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
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
                        float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _scale);
                        float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + A_hstep), _scale);
                        float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _scale);
                        float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _scale);
                        float32x4_t _p4 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _scale);
                        float32x4_t _p5 = vmulq_f32(vld1q_f32(p0 + A_hstep * 5), _scale);
                        float32x4_t _p6 = vmulq_f32(vld1q_f32(p0 + A_hstep * 6), _scale);
                        float32x4_t _p7 = vmulq_f32(vld1q_f32(p0 + A_hstep * 7), _scale);
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
                        float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _scale);
                        float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + A_hstep), _scale);
                        float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + A_hstep * 2), _scale);
                        float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + A_hstep * 3), _scale);
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
                        float32x4_t _p0 = vld1q_f32(p0);
                        float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                        int8x8_t _r01 = float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale));
                        int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                        vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                        pp += 8;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vld1q_f32(p0);
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
            const float* p0 = (const float*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
#if __ARM_NEON
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vld1q_f32(p0a)));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vld1q_f32(p0a + 4)));
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
                        float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _scale0);
                        float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + 4), _scale1);
                        float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _scale0);
                        float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 4), _scale1);
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
                        float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _scale0);
                        float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + 4), _scale1);

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
#endif // __ARM_NEON
                if (elempack == 1)
                {
#if __ARM_NEON
                    float32x2_t _absmax = vdup_n_f32(0.f);
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x2_t _p = vld1_f32(p0a);
                        _absmax = vmax_f32(_absmax, vabs_f32(_p));
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
                        float32x2_t _p0 = vld1_f32(p0);
                        float32x2_t _p1 = vld1_f32(p0 + A_hstep);
                        float32x2_t _p2 = vld1_f32(p0 + A_hstep * 2);
                        float32x2_t _p3 = vld1_f32(p0 + A_hstep * 3);
                        float32x2_t _p4 = vld1_f32(p0 + A_hstep * 4);
                        float32x2_t _p5 = vld1_f32(p0 + A_hstep * 5);
                        float32x2_t _p6 = vld1_f32(p0 + A_hstep * 6);
                        float32x2_t _p7 = vld1_f32(p0 + A_hstep * 7);
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
                        float32x2_t _p0 = vld1_f32(p0);
                        float32x2_t _p1 = vld1_f32(p0 + A_hstep);
                        float32x2_t _p2 = vld1_f32(p0 + A_hstep * 2);
                        float32x2_t _p3 = vld1_f32(p0 + A_hstep * 3);
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
                        float32x2_t _p0 = vld1_f32(p0);
                        float32x2_t _p1 = vld1_f32(p0 + A_hstep);
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
                        float32x2_t _p0 = vld1_f32(p0);
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p0), vcombine_f32(_scale, _scale));
                        vst1_lane_s16((short*)pp, vreinterpret_s16_s8(float2int8(_p01, _p01)), 0);
                        pp += 2;
                        p0 += A_hstep;
                    }
#else
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    const float* p0a = p0;

                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = p0a[0];
                        float v1 = p0a[1];
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
                        float v0 = p0[0];
                        float v1 = p0[1];
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
            const float* p0 = (const float*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
#if __ARM_NEON
                if (elempack == 4)
                {
                    float32x4_t _absmax = vdupq_n_f32(0.f);
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        _absmax = vmaxq_f32(_absmax, vabsq_f32(vld1q_f32(p0a)));
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
                        float32x4_t _p = vmulq_f32(vld1q_f32(p0), _scale);
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
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v = *p0a;
                        absmax = std::max(absmax, fabsf(v));
                        p0a += A_hstep;
                    }

                    if (absmax == 0.f)
                    {
                        *pd++ = 0.f;
                        for (int kk0 = 0; kk0 < max_kk0; kk0++)
                            *pp++ = 0;
                        p0 += (size_t)max_kk0 * A_hstep;
                        continue;
                    }

                    const float scale = 127.f / absmax;
                    *pd++ = absmax / 127.f;

                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v = *p0;
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
        const float* p0 = (const float*)A + (size_t)k * A_hstep + (i + ii) * elempack;
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
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(vld1q_f32(p0a)), _s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 4)), _s));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 8)), _s));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 12)), _s));
                    _absmax4 = vmaxq_f32(_absmax4, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 16)), _s));
                    _absmax5 = vmaxq_f32(_absmax5, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 20)), _s));
                    _absmax6 = vmaxq_f32(_absmax6, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 24)), _s));
                    _absmax7 = vmaxq_f32(_absmax7, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 28)), _s));
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
                    float32x4_t _p00 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0), _s0), _scale0, 0);
                    float32x4_t _p01 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _s1), _scale0, 0);
                    float32x4_t _p10 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 4), _s0), _scale0, 1);
                    float32x4_t _p11 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 4), _s1), _scale0, 1);
                    int8x8_t _r0 = float2int8(_p00, _p01);
                    int8x8_t _r1 = float2int8(_p10, _p11);
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));

                    float32x4_t _p20 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 8), _s0), _scale0, 2);
                    float32x4_t _p21 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 8), _s1), _scale0, 2);
                    float32x4_t _p30 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 12), _s0), _scale0, 3);
                    float32x4_t _p31 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 12), _s1), _scale0, 3);
                    int8x8_t _r2 = float2int8(_p20, _p21);
                    int8x8_t _r3 = float2int8(_p30, _p31);
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                    float32x4_t _p40 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 16), _s0), _scale1, 0);
                    float32x4_t _p41 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 16), _s1), _scale1, 0);
                    float32x4_t _p50 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 20), _s0), _scale1, 1);
                    float32x4_t _p51 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 20), _s1), _scale1, 1);
                    int8x8_t _r4 = float2int8(_p40, _p41);
                    int8x8_t _r5 = float2int8(_p50, _p51);
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));

                    float32x4_t _p60 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 24), _s0), _scale1, 2);
                    float32x4_t _p61 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 24), _s1), _scale1, 2);
                    float32x4_t _p70 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 28), _s0), _scale1, 3);
                    float32x4_t _p71 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 28), _s1), _scale1, 3);
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
                    float32x4_t _p0 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0), _s), _scale0, 0);
                    float32x4_t _p1 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 4), _s), _scale0, 1);
                    float32x4_t _p2 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 8), _s), _scale0, 2);
                    float32x4_t _p3 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 12), _s), _scale0, 3);
                    float32x4_t _p4 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 16), _s), _scale1, 0);
                    float32x4_t _p5 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 20), _s), _scale1, 1);
                    float32x4_t _p6 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 24), _s), _scale1, 2);
                    float32x4_t _p7 = vmulq_laneq_f32(vmulq_f32(vld1q_f32(p0 + 28), _s), _scale1, 3);
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
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    float32x4_t _p0 = vld1q_f32(p0a);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_n_f32(vabsq_f32(_p0), s));
                    float32x4_t _p1 = vld1q_f32(p0a + 4);
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
                    float32x4_t _p00 = vld1q_f32(p0);
                    float32x4_t _p01 = vld1q_f32(p0 + 4);
                    const float s0 = *ps++;
                    _p00 = vmulq_n_f32(_p00, s0);
                    _p01 = vmulq_n_f32(_p01, s0);
                    float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                    float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
                    const float s1 = *ps++;
                    _p10 = vmulq_n_f32(_p10, s1);
                    _p11 = vmulq_n_f32(_p11, s1);
                    float32x4_t _p20 = vld1q_f32(p0 + A_hstep * 2);
                    float32x4_t _p21 = vld1q_f32(p0 + A_hstep * 2 + 4);
                    const float s2 = *ps++;
                    _p20 = vmulq_n_f32(_p20, s2);
                    _p21 = vmulq_n_f32(_p21, s2);
                    float32x4_t _p30 = vld1q_f32(p0 + A_hstep * 3);
                    float32x4_t _p31 = vld1q_f32(p0 + A_hstep * 3 + 4);
                    const float s3 = *ps++;
                    _p30 = vmulq_n_f32(_p30, s3);
                    _p31 = vmulq_n_f32(_p31, s3);
                    float32x4_t _p40 = vld1q_f32(p0 + A_hstep * 4);
                    float32x4_t _p41 = vld1q_f32(p0 + A_hstep * 4 + 4);
                    const float s4 = *ps++;
                    _p40 = vmulq_n_f32(_p40, s4);
                    _p41 = vmulq_n_f32(_p41, s4);
                    float32x4_t _p50 = vld1q_f32(p0 + A_hstep * 5);
                    float32x4_t _p51 = vld1q_f32(p0 + A_hstep * 5 + 4);
                    const float s5 = *ps++;
                    _p50 = vmulq_n_f32(_p50, s5);
                    _p51 = vmulq_n_f32(_p51, s5);
                    float32x4_t _p60 = vld1q_f32(p0 + A_hstep * 6);
                    float32x4_t _p61 = vld1q_f32(p0 + A_hstep * 6 + 4);
                    const float s6 = *ps++;
                    _p60 = vmulq_n_f32(_p60, s6);
                    _p61 = vmulq_n_f32(_p61, s6);
                    float32x4_t _p70 = vld1q_f32(p0 + A_hstep * 7);
                    float32x4_t _p71 = vld1q_f32(p0 + A_hstep * 7 + 4);
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
                    float32x4_t _p00 = vld1q_f32(p0);
                    float32x4_t _p01 = vld1q_f32(p0 + 4);
                    const float s0 = *ps++;
                    _p00 = vmulq_n_f32(_p00, s0);
                    _p01 = vmulq_n_f32(_p01, s0);
                    float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                    float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
                    const float s1 = *ps++;
                    _p10 = vmulq_n_f32(_p10, s1);
                    _p11 = vmulq_n_f32(_p11, s1);
                    float32x4_t _p20 = vld1q_f32(p0 + A_hstep * 2);
                    float32x4_t _p21 = vld1q_f32(p0 + A_hstep * 2 + 4);
                    const float s2 = *ps++;
                    _p20 = vmulq_n_f32(_p20, s2);
                    _p21 = vmulq_n_f32(_p21, s2);
                    float32x4_t _p30 = vld1q_f32(p0 + A_hstep * 3);
                    float32x4_t _p31 = vld1q_f32(p0 + A_hstep * 3 + 4);
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
                    float32x4_t _p00 = vld1q_f32(p0);
                    float32x4_t _p01 = vld1q_f32(p0 + 4);
                    const float s0 = *ps++;
                    _p00 = vmulq_n_f32(_p00, s0);
                    _p01 = vmulq_n_f32(_p01, s0);
                    float32x4_t _p10 = vld1q_f32(p0 + A_hstep);
                    float32x4_t _p11 = vld1q_f32(p0 + A_hstep + 4);
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
                    float32x4_t _p0 = vld1q_f32(p0);
                    float32x4_t _p1 = vld1q_f32(p0 + 4);
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
        const float* p0 = (const float*)A + (size_t)k * A_hstep + (i + ii) * elempack;
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
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(vld1q_f32(p0a)), _s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 4)), _s));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 8)), _s));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 12)), _s));
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
                    float32x4_t _p00 = vmulq_f32(vld1q_f32(p0), _s0);
                    float32x4_t _p01 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _s1);
                    float32x4_t _p10 = vmulq_f32(vld1q_f32(p0 + 4), _s0);
                    float32x4_t _p11 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 4), _s1);
#if __aarch64__
                    _p00 = vmulq_laneq_f32(_p00, _scale, 0);
                    _p01 = vmulq_laneq_f32(_p01, _scale, 0);
                    _p10 = vmulq_laneq_f32(_p10, _scale, 1);
                    _p11 = vmulq_laneq_f32(_p11, _scale, 1);
#else
                    _p00 = vmulq_lane_f32(_p00, vget_low_f32(_scale), 0);
                    _p01 = vmulq_lane_f32(_p01, vget_low_f32(_scale), 0);
                    _p10 = vmulq_lane_f32(_p10, vget_low_f32(_scale), 1);
                    _p11 = vmulq_lane_f32(_p11, vget_low_f32(_scale), 1);
#endif
                    int8x8_t _r0 = float2int8(_p00, _p01);
                    int8x8_t _r1 = float2int8(_p10, _p11);
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));

                    float32x4_t _p20 = vmulq_f32(vld1q_f32(p0 + 8), _s0);
                    float32x4_t _p21 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 8), _s1);
                    float32x4_t _p30 = vmulq_f32(vld1q_f32(p0 + 12), _s0);
                    float32x4_t _p31 = vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 12), _s1);
#if __aarch64__
                    _p20 = vmulq_laneq_f32(_p20, _scale, 2);
                    _p21 = vmulq_laneq_f32(_p21, _scale, 2);
                    _p30 = vmulq_laneq_f32(_p30, _scale, 3);
                    _p31 = vmulq_laneq_f32(_p31, _scale, 3);
#else
                    _p20 = vmulq_lane_f32(_p20, vget_high_f32(_scale), 0);
                    _p21 = vmulq_lane_f32(_p21, vget_high_f32(_scale), 0);
                    _p30 = vmulq_lane_f32(_p30, vget_high_f32(_scale), 1);
                    _p31 = vmulq_lane_f32(_p31, vget_high_f32(_scale), 1);
#endif
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
                    float32x4_t _p0 = vmulq_f32(vld1q_f32(p0), _s);
                    float32x4_t _p1 = vmulq_f32(vld1q_f32(p0 + 4), _s);
                    float32x4_t _p2 = vmulq_f32(vld1q_f32(p0 + 8), _s);
                    float32x4_t _p3 = vmulq_f32(vld1q_f32(p0 + 12), _s);
#if __aarch64__
                    _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                    _p1 = vmulq_laneq_f32(_p1, _scale, 1);
                    _p2 = vmulq_laneq_f32(_p2, _scale, 2);
                    _p3 = vmulq_laneq_f32(_p3, _scale, 3);
#else
                    _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale), 0);
                    _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale), 1);
                    _p2 = vmulq_lane_f32(_p2, vget_high_f32(_scale), 0);
                    _p3 = vmulq_lane_f32(_p3, vget_high_f32(_scale), 1);
#endif
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
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vld1q_f32(p0a);
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
                    float32x4_t _p0 = vmulq_n_f32(vld1q_f32(p0), *ps++);
                    float32x4_t _p1 = vmulq_n_f32(vld1q_f32(p0 + A_hstep), *ps++);
                    float32x4_t _p2 = vmulq_n_f32(vld1q_f32(p0 + A_hstep * 2), *ps++);
                    float32x4_t _p3 = vmulq_n_f32(vld1q_f32(p0 + A_hstep * 3), *ps++);
                    float32x4_t _p4 = vmulq_n_f32(vld1q_f32(p0 + A_hstep * 4), *ps++);
                    float32x4_t _p5 = vmulq_n_f32(vld1q_f32(p0 + A_hstep * 5), *ps++);
                    float32x4_t _p6 = vmulq_n_f32(vld1q_f32(p0 + A_hstep * 6), *ps++);
                    float32x4_t _p7 = vmulq_n_f32(vld1q_f32(p0 + A_hstep * 7), *ps++);
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
                    float32x4_t _p0 = vmulq_n_f32(vld1q_f32(p0), *ps++);
                    float32x4_t _p1 = vmulq_n_f32(vld1q_f32(p0 + A_hstep), *ps++);
                    float32x4_t _p2 = vmulq_n_f32(vld1q_f32(p0 + A_hstep * 2), *ps++);
                    float32x4_t _p3 = vmulq_n_f32(vld1q_f32(p0 + A_hstep * 3), *ps++);
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
                    float32x4_t _p0 = vmulq_n_f32(vld1q_f32(p0), *ps++);
                    float32x4_t _p1 = vmulq_n_f32(vld1q_f32(p0 + A_hstep), *ps++);
                    int8x8_t _r01 = float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale));
                    int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                    vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                    pp += 8;
                    p0 += A_hstep * 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vmulq_n_f32(vld1q_f32(p0), *ps++);
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
        const float* p0 = (const float*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
#if __ARM_NEON
            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(vld1q_f32(p0a)), _s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(vld1q_f32(p0a + 4)), _s));
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
                    float32x4_t _p0 = vmulq_f32(vmulq_f32(vld1q_f32(p0), _s0), _scale0);
                    float32x4_t _p1 = vmulq_f32(vmulq_f32(vld1q_f32(p0 + 4), _s0), _scale1);
                    float32x4_t _p2 = vmulq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4), _s1), _scale0);
                    float32x4_t _p3 = vmulq_f32(vmulq_f32(vld1q_f32(p0 + A_hstep * 4 + 4), _s1), _scale1);
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
                    float32x4_t _p0 = vmulq_f32(vmulq_f32(vld1q_f32(p0), _s), _scale0);
                    float32x4_t _p1 = vmulq_f32(vmulq_f32(vld1q_f32(p0 + 4), _s), _scale1);

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
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x2_t _p = vld1_f32(p0a);
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
                    float32x2_t _p0 = vmul_n_f32(vld1_f32(p0), *ps++);
                    float32x2_t _p1 = vmul_n_f32(vld1_f32(p0 + A_hstep), *ps++);
                    float32x2_t _p2 = vmul_n_f32(vld1_f32(p0 + A_hstep * 2), *ps++);
                    float32x2_t _p3 = vmul_n_f32(vld1_f32(p0 + A_hstep * 3), *ps++);
                    float32x2_t _p4 = vmul_n_f32(vld1_f32(p0 + A_hstep * 4), *ps++);
                    float32x2_t _p5 = vmul_n_f32(vld1_f32(p0 + A_hstep * 5), *ps++);
                    float32x2_t _p6 = vmul_n_f32(vld1_f32(p0 + A_hstep * 6), *ps++);
                    float32x2_t _p7 = vmul_n_f32(vld1_f32(p0 + A_hstep * 7), *ps++);
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
                    float32x2_t _p0 = vmul_n_f32(vld1_f32(p0), *ps++);
                    float32x2_t _p1 = vmul_n_f32(vld1_f32(p0 + A_hstep), *ps++);
                    float32x2_t _p2 = vmul_n_f32(vld1_f32(p0 + A_hstep * 2), *ps++);
                    float32x2_t _p3 = vmul_n_f32(vld1_f32(p0 + A_hstep * 3), *ps++);
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
                    float32x2_t _p0 = vmul_n_f32(vld1_f32(p0), *ps++);
                    float32x2_t _p1 = vmul_n_f32(vld1_f32(p0 + A_hstep), *ps++);
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
                    float32x2_t _p0 = vmul_n_f32(vld1_f32(p0), *ps++);
                    float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p0), vcombine_f32(_scale, _scale));
                    vst1_lane_s16((short*)pp, vreinterpret_s16_s8(float2int8(_p01, _p01)), 0);
                    pp += 2;
                    p0 += A_hstep;
                }
#else
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0;
                const float* psa = ps;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    float v0 = p0a[0];
                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    float v1 = p0a[1];
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
                    float v0 = p0[0] * s;
                    float v1 = p0[1] * s;
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
        const float* p0 = (const float*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __ARM_NEON
            if (elempack == 4)
            {
                float32x4_t _absmax = vdupq_n_f32(0.f);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _p = vmulq_f32(vabsq_f32(vld1q_f32(p0a)), vld1q_f32(psa));
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
                    float32x4_t _p = vmulq_f32(vmulq_f32(vld1q_f32(p0), vld1q_f32(ps)), _scale);
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
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = *p0a;
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
                    float v = *p0 * *ps++;
                    *pp++ = float2int8(v * scale);
                    p0 += A_hstep;
                }
            }
        }
    }
}

static void pack_B_tile_wq_int8(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        pack_B_tile_wq_int8_i8mm(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        pack_B_tile_wq_int8_asimddp(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
        return;
    }
#endif

    const int block_count = (K + block_size - 1) / block_size;
    signed char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __ARM_NEON
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
            for (; kk + 15 < max_kk; kk += 16)
            {
                int8x16_t _p0 = vld1q_s8(p0);
                int8x16_t _p1 = vld1q_s8(p1);
                int8x16_t _p2 = vld1q_s8(p2);
                int8x16_t _p3 = vld1q_s8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int64x2x4_t _r0123;
                _r0123.val[0] = vreinterpretq_s64_s8(_p0);
                _r0123.val[1] = vreinterpretq_s64_s8(_p1);
                _r0123.val[2] = vreinterpretq_s64_s8(_p2);
                _r0123.val[3] = vreinterpretq_s64_s8(_p3);
                vst4q_s64((int64_t*)pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x4x4_t _r0123;
                _r0123.val[0] = vreinterpretq_s32_s8(_p0);
                _r0123.val[1] = vreinterpretq_s32_s8(_p1);
                _r0123.val[2] = vreinterpretq_s32_s8(_p2);
                _r0123.val[3] = vreinterpretq_s32_s8(_p3);
                vst4q_s32((int*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x8x4_t _r0123;
                _r0123.val[0] = vreinterpretq_s16_s8(_p0);
                _r0123.val[1] = vreinterpretq_s16_s8(_p1);
                _r0123.val[2] = vreinterpretq_s16_s8(_p2);
                _r0123.val[3] = vreinterpretq_s16_s8(_p3);
                vst4q_s16((short*)pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
                pp += 64;
                p0 += 16;
                p1 += 16;
                p2 += 16;
                p3 += 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _p0 = vld1_s8(p0);
                int8x8_t _p1 = vld1_s8(p1);
                int8x8_t _p2 = vld1_s8(p2);
                int8x8_t _p3 = vld1_s8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                vst1q_s8(pp, vcombine_s8(_p0, _p1));
                vst1q_s8(pp + 16, vcombine_s8(_p2, _p3));
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2x4_t _r0123;
                _r0123.val[0] = vreinterpret_s32_s8(_p0);
                _r0123.val[1] = vreinterpret_s32_s8(_p1);
                _r0123.val[2] = vreinterpret_s32_s8(_p2);
                _r0123.val[3] = vreinterpret_s32_s8(_p3);
                vst4_s32((int*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4x4_t _r0123;
                _r0123.val[0] = vreinterpret_s16_s8(_p0);
                _r0123.val[1] = vreinterpret_s16_s8(_p1);
                _r0123.val[2] = vreinterpret_s16_s8(_p2);
                _r0123.val[3] = vreinterpret_s16_s8(_p3);
                vst4_s16((short*)pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
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
#else  // __ARM_FEATURE_DOTPROD
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp[4] = p2[0];
                pp[5] = p2[1];
                pp[6] = p3[0];
                pp[7] = p3[1];
                pp[8] = p0[2];
                pp[9] = p0[3];
                pp[10] = p1[2];
                pp[11] = p1[3];
                pp[12] = p2[2];
                pp[13] = p2[3];
                pp[14] = p3[2];
                pp[15] = p3[3];
#endif // __ARM_FEATURE_DOTPROD
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
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

            *pd++ = 1.f / *ps0++;
            *pd++ = 1.f / *ps1++;
            *pd++ = 1.f / *ps2++;
            *pd++ = 1.f / *ps3++;
        }
    }
#endif // __ARM_NEON
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
#if __ARM_NEON
            for (; kk + 15 < max_kk; kk += 16)
            {
                int8x16_t _p0 = vld1q_s8(p0);
                int8x16_t _p1 = vld1q_s8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int64x2x2_t _r01;
                _r01.val[0] = vreinterpretq_s64_s8(_p0);
                _r01.val[1] = vreinterpretq_s64_s8(_p1);
                vst2q_s64((int64_t*)pp, _r01);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _r01;
                _r01.val[0] = vreinterpretq_s32_s8(_p0);
                _r01.val[1] = vreinterpretq_s32_s8(_p1);
                vst2q_s32((int*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x8x2_t _r01;
                _r01.val[0] = vreinterpretq_s16_s8(_p0);
                _r01.val[1] = vreinterpretq_s16_s8(_p1);
                vst2q_s16((short*)pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
                pp += 32;
                p0 += 16;
                p1 += 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _p0 = vld1_s8(p0);
                int8x8_t _p1 = vld1_s8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                vst1q_s8(pp, vcombine_s8(_p0, _p1));
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2x2_t _r01;
                _r01.val[0] = vreinterpret_s32_s8(_p0);
                _r01.val[1] = vreinterpret_s32_s8(_p1);
                vst2_s32((int*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4x2_t _r01;
                _r01.val[0] = vreinterpret_s16_s8(_p0);
                _r01.val[1] = vreinterpret_s16_s8(_p1);
                vst2_s16((short*)pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p1[0];
                pp[5] = p1[1];
                pp[6] = p1[2];
                pp[7] = p1[3];
#else  // __ARM_FEATURE_DOTPROD
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp[4] = p0[2];
                pp[5] = p0[3];
                pp[6] = p1[2];
                pp[7] = p1[3];
#endif // __ARM_FEATURE_DOTPROD
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __ARM_NEON
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp += 4;
                p0 += 2;
                p1 += 2;
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
#if __ARM_NEON
            for (; kk + 15 < max_kk; kk += 16)
            {
                vst1q_s8(pp, vld1q_s8(p0));
                pp += 16;
                p0 += 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                vst1_s8(pp, vld1_s8(p0));
                pp += 8;
                p0 += 8;
            }
#if __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += 4;
            }
#endif // __ARM_FEATURE_DOTPROD
#endif // __ARM_NEON
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
                *pp++ = *p0++;

            *pd++ = 1.f / *ps0++;
        }
    }
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_elemtype, int output_transpose)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        unpack_output_tile_wq_int8_asimddp(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_elemtype, output_transpose);
        return;
    }
#endif

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
        float* p0 = 0;
        unsigned short* p0s = 0;
        if (output_elemtype == 1)
        {
            if (output_transpose)
                p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            else
                p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }
        else
        {
            if (output_transpose)
                p0s = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            else
                p0s = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        float32x4_t _c0;
        float32x4_t _c1;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = vld1q_f32(pC);
                _c1 = vld1q_f32(pC + 4);
                _c0 = vmulq_n_f32(_c0, beta);
                _c1 = vmulq_n_f32(_c1, beta);
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
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0 = vld1q_f32(pp);
            float32x4_t _sum1 = vld1q_f32(pp + 4);
            float32x4_t _sum2 = vld1q_f32(pp + 8);
            float32x4_t _sum3 = vld1q_f32(pp + 12);
            float32x4_t _sum4 = vld1q_f32(pp + 16);
            float32x4_t _sum5 = vld1q_f32(pp + 20);
            float32x4_t _sum6 = vld1q_f32(pp + 24);
            float32x4_t _sum7 = vld1q_f32(pp + 28);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
            //      e0 f0 g0 h0
            //      e1 f1 g1 h1
            //      e2 f2 g2 h2
            //      e3 f3 g3 h3
#else
            // from
            //      a0 b1 c2 d3
            //      e0 f1 g2 h3
            //      c0 d1 a2 b3
            //      g0 h1 e2 f3
            //      a3 b2 c1 d0
            //      e3 f2 g1 h0
            //      c3 d2 a1 b0
            //      g3 h2 e1 f0

            // to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
            //      e0 f0 g0 h0
            //      e1 f1 g1 h1
            //      e2 f2 g2 h2
            //      e3 f3 g3 h3
            {
                _sum4 = vrev64q_f32(_sum4);
                _sum5 = vrev64q_f32(_sum5);
                _sum6 = vrev64q_f32(_sum6);
                _sum7 = vrev64q_f32(_sum7);
                _sum4 = vextq_f32(_sum4, _sum4, 2);
                _sum5 = vextq_f32(_sum5, _sum5, 2);
                _sum6 = vextq_f32(_sum6, _sum6, 2);
                _sum7 = vextq_f32(_sum7, _sum7, 2);
                float32x4x2_t _t0 = vzipq_f32(_sum0, _sum6);
                float32x4x2_t _t1 = vzipq_f32(_sum2, _sum4);
                float32x4x2_t _t2 = vzipq_f32(_sum1, _sum7);
                float32x4x2_t _t3 = vzipq_f32(_sum3, _sum5);
                _sum0 = vcombine_f32(vget_low_f32(_t0.val[0]), vget_low_f32(_t1.val[0]));
                _sum1 = vcombine_f32(vget_high_f32(_t0.val[0]), vget_high_f32(_t1.val[0]));
                _sum2 = vcombine_f32(vget_low_f32(_t1.val[1]), vget_low_f32(_t0.val[1]));
                _sum3 = vcombine_f32(vget_high_f32(_t1.val[1]), vget_high_f32(_t0.val[1]));
                _sum4 = vcombine_f32(vget_low_f32(_t2.val[0]), vget_low_f32(_t3.val[0]));
                _sum5 = vcombine_f32(vget_high_f32(_t2.val[0]), vget_high_f32(_t3.val[0]));
                _sum6 = vcombine_f32(vget_low_f32(_t3.val[1]), vget_low_f32(_t2.val[1]));
                _sum7 = vcombine_f32(vget_high_f32(_t3.val[1]), vget_high_f32(_t2.val[1]));
                _sum1 = vrev64q_f32(_sum1);
                _sum3 = vrev64q_f32(_sum3);
                _sum5 = vrev64q_f32(_sum5);
                _sum7 = vrev64q_f32(_sum7);
            }
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = _sum0;
            float32x4_t _f1 = _sum1;
            float32x4_t _f2 = _sum2;
            float32x4_t _f3 = _sum3;
            float32x4_t _f4 = _sum4;
            float32x4_t _f5 = _sum5;
            float32x4_t _f6 = _sum6;
            float32x4_t _f7 = _sum7;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c1);
                    _f5 = vaddq_f32(_f5, _c1);
                    _f6 = vaddq_f32(_f6, _c1);
                    _f7 = vaddq_f32(_f7, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + 8);
                        _c3 = vld1q_f32(pC + 12);
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep);
                        _c2 = vld1q_f32(pC + c_hstep * 2);
                        _c3 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 8);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 12);
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 5);
                        _c2 = vld1q_f32(pC + c_hstep * 6);
                        _c3 = vld1q_f32(pC + c_hstep * 7);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c2);
                        _f7 = vaddq_f32(_f7, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f4 = vmlaq_f32(_f4, _c0, _beta);
                        _f5 = vmlaq_f32(_f5, _c1, _beta);
                        _f6 = vmlaq_f32(_f6, _c2, _beta);
                        _f7 = vmlaq_f32(_f7, _c3, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    _c = vmulq_n_f32(_c, beta);
                    _c0 = vdupq_laneq_f32(_c, 0);
                    _c1 = vdupq_laneq_f32(_c, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_c, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_c, 3);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c1);
                    _f6 = vaddq_f32(_f6, _c2);
                    _f7 = vaddq_f32(_f7, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float32x4x4_t _r0;
                        float32x4x4_t _r1;
                        _r0.val[0] = _f0;
                        _r0.val[1] = _f1;
                        _r0.val[2] = _f2;
                        _r0.val[3] = _f3;
                        _r1.val[0] = _f4;
                        _r1.val[1] = _f5;
                        _r1.val[2] = _f6;
                        _r1.val[3] = _f7;
                        vst4q_f32(p0, _r0);
                        vst4q_f32(p0 + 16, _r1);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + 4, _f4);
                        vst1q_f32(p0 + out_hstep, _f1);
                        vst1q_f32(p0 + out_hstep + 4, _f5);
                        vst1q_f32(p0 + out_hstep * 2, _f2);
                        vst1q_f32(p0 + out_hstep * 2 + 4, _f6);
                        vst1q_f32(p0 + out_hstep * 3, _f3);
                        vst1q_f32(p0 + out_hstep * 3 + 4, _f7);
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + 4, _f1);
                        vst1q_f32(p0 + 8, _f2);
                        vst1q_f32(p0 + 12, _f3);
                        vst1q_f32(p0 + out_hstep * 4, _f4);
                        vst1q_f32(p0 + out_hstep * 4 + 4, _f5);
                        vst1q_f32(p0 + out_hstep * 4 + 8, _f6);
                        vst1q_f32(p0 + out_hstep * 4 + 12, _f7);
                        p0 += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_f0, _f1, _f2, _f3);
                        transpose4x4_ps(_f4, _f5, _f6, _f7);
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + out_hstep, _f1);
                        vst1q_f32(p0 + out_hstep * 2, _f2);
                        vst1q_f32(p0 + out_hstep * 3, _f3);
                        vst1q_f32(p0 + out_hstep * 4, _f4);
                        vst1q_f32(p0 + out_hstep * 5, _f5);
                        vst1q_f32(p0 + out_hstep * 6, _f6);
                        vst1q_f32(p0 + out_hstep * 7, _f7);
                        p0 += 4;
                    }
                }
            }
            else
            {
                uint16x4_t _out0;
                uint16x4_t _out1;
                uint16x4_t _out2;
                uint16x4_t _out3;
                uint16x4_t _out4;
                uint16x4_t _out5;
                uint16x4_t _out6;
                uint16x4_t _out7;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                {
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                    _out1 = vreinterpret_u16_f16(vcvt_f16_f32(_f1));
                    _out2 = vreinterpret_u16_f16(vcvt_f16_f32(_f2));
                    _out3 = vreinterpret_u16_f16(vcvt_f16_f32(_f3));
                    _out4 = vreinterpret_u16_f16(vcvt_f16_f32(_f4));
                    _out5 = vreinterpret_u16_f16(vcvt_f16_f32(_f5));
                    _out6 = vreinterpret_u16_f16(vcvt_f16_f32(_f6));
                    _out7 = vreinterpret_u16_f16(vcvt_f16_f32(_f7));
                }
                else
#endif
                {
                    _out0 = float2bfloat(_f0);
                    _out1 = float2bfloat(_f1);
                    _out2 = float2bfloat(_f2);
                    _out3 = float2bfloat(_f3);
                    _out4 = float2bfloat(_f4);
                    _out5 = float2bfloat(_f5);
                    _out6 = float2bfloat(_f6);
                    _out7 = float2bfloat(_f7);
                }

                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        transpose4x4_u16(_out0, _out1, _out2, _out3);
                        transpose4x4_u16(_out4, _out5, _out6, _out7);
                        const int jj_m8 = jj % 8;
                        unsigned short* p1 = p0s - out_hstep * jj_m8 + jj_m8;
                        vst1_u16(p1, _out0);
                        vst1_u16(p1 + 8, _out1);
                        vst1_u16(p1 + 16, _out2);
                        vst1_u16(p1 + 24, _out3);
                        vst1_u16(p1 + 32, _out4);
                        vst1_u16(p1 + 40, _out5);
                        vst1_u16(p1 + 48, _out6);
                        vst1_u16(p1 + 56, _out7);
                    }
                    if (out_elempack == 4)
                    {
                        uint16x8x4_t _r;
                        _r.val[0] = vcombine_u16(_out0, _out4);
                        _r.val[1] = vcombine_u16(_out1, _out5);
                        _r.val[2] = vcombine_u16(_out2, _out6);
                        _r.val[3] = vcombine_u16(_out3, _out7);
                        vst4q_u16(p0s, _r);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_u16(p0s, vcombine_u16(_out0, _out4));
                        vst1q_u16(p0s + out_hstep, vcombine_u16(_out1, _out5));
                        vst1q_u16(p0s + out_hstep * 2, vcombine_u16(_out2, _out6));
                        vst1q_u16(p0s + out_hstep * 3, vcombine_u16(_out3, _out7));
                    }
                    p0s += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        vst1q_u16(p0s, vcombine_u16(_out0, _out4));
                        vst1q_u16(p0s + 8, vcombine_u16(_out1, _out5));
                        vst1q_u16(p0s + 16, vcombine_u16(_out2, _out6));
                        vst1q_u16(p0s + 24, vcombine_u16(_out3, _out7));
                        p0s += 32;
                    }
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0s, _out0);
                        vst1_u16(p0s + 4, _out1);
                        vst1_u16(p0s + 8, _out2);
                        vst1_u16(p0s + 12, _out3);
                        vst1_u16(p0s + out_hstep * 4, _out4);
                        vst1_u16(p0s + out_hstep * 4 + 4, _out5);
                        vst1_u16(p0s + out_hstep * 4 + 8, _out6);
                        vst1_u16(p0s + out_hstep * 4 + 12, _out7);
                        p0s += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_u16(_out0, _out1, _out2, _out3);
                        transpose4x4_u16(_out4, _out5, _out6, _out7);
                        vst1_u16(p0s, _out0);
                        vst1_u16(p0s + out_hstep, _out1);
                        vst1_u16(p0s + out_hstep * 2, _out2);
                        vst1_u16(p0s + out_hstep * 3, _out3);
                        vst1_u16(p0s + out_hstep * 4, _out4);
                        vst1_u16(p0s + out_hstep * 5, _out5);
                        vst1_u16(p0s + out_hstep * 6, _out6);
                        vst1_u16(p0s + out_hstep * 7, _out7);
                        p0s += 4;
                    }
                }
            }

            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum0 = vld1q_f32(pp);
            float32x4_t _sum1 = vld1q_f32(pp + 4);
            float32x4_t _sum2 = vld1q_f32(pp + 8);
            float32x4_t _sum3 = vld1q_f32(pp + 12);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      e0 f0 g0 h0
            //      e1 f1 g1 h1
#else
            // from
            //      a0 b1 c0 d1
            //      e0 f1 g0 h1
            //      a1 b0 c1 d0
            //      e1 f0 g1 h0

            // to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      e0 f0 g0 h0
            //      e1 f1 g1 h1
            {
                _sum2 = vrev64q_f32(_sum2);
                _sum3 = vrev64q_f32(_sum3);
                float32x4x2_t _t0 = vzipq_f32(_sum0, _sum2);
                float32x4x2_t _t1 = vzipq_f32(_sum1, _sum3);
                _sum0 = vcombine_f32(vget_low_f32(_t0.val[0]), vget_low_f32(_t0.val[1]));
                _sum1 = vcombine_f32(vget_high_f32(_t0.val[0]), vget_high_f32(_t0.val[1]));
                _sum2 = vcombine_f32(vget_low_f32(_t1.val[0]), vget_low_f32(_t1.val[1]));
                _sum3 = vcombine_f32(vget_high_f32(_t1.val[0]), vget_high_f32(_t1.val[1]));
                _sum1 = vrev64q_f32(_sum1);
                _sum3 = vrev64q_f32(_sum3);
            }
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = _sum0;
            float32x4_t _f1 = _sum1;
            float32x4_t _f2 = _sum2;
            float32x4_t _f3 = _sum3;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c1);
                    _f3 = vaddq_f32(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 4);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        float32x2_t _cc0 = vld1_f32(pC);
                        float32x2_t _cc1 = vld1_f32(pC + c_hstep);
                        float32x2_t _cc2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _cc3 = vld1_f32(pC + c_hstep * 3);
                        float32x4_t _c01 = vcombine_f32(_cc0, _cc1);
                        float32x4_t _c23 = vcombine_f32(_cc2, _cc3);
                        float32x4x2_t _ccc0 = vuzpq_f32(_c01, _c23);
                        _c0 = _ccc0.val[0];
                        _c1 = _ccc0.val[1];
                        float32x2_t _cc4 = vld1_f32(pC + c_hstep * 4);
                        float32x2_t _cc5 = vld1_f32(pC + c_hstep * 5);
                        float32x2_t _cc6 = vld1_f32(pC + c_hstep * 6);
                        float32x2_t _cc7 = vld1_f32(pC + c_hstep * 7);
                        float32x4_t _c45 = vcombine_f32(_cc4, _cc5);
                        float32x4_t _c67 = vcombine_f32(_cc6, _cc7);
                        float32x4x2_t _ccc1 = vuzpq_f32(_c45, _c67);
                        _c2 = _ccc1.val[0];
                        _c3 = _ccc1.val[1];
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    _c = vmul_n_f32(_c, beta);
                    _c0 = vdupq_lane_f32(_c, 0);
                    _c1 = vdupq_lane_f32(_c, 1);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    vst1q_f32(p0, _f0);
                    vst1q_f32(p0 + 4, _f2);
                    vst1q_f32(p0 + out_hstep, _f1);
                    vst1q_f32(p0 + out_hstep + 4, _f3);
                    p0 += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + 4, _f1);
                        vst1q_f32(p0 + out_hstep * 4, _f2);
                        vst1q_f32(p0 + out_hstep * 4 + 4, _f3);
                        p0 += 8;
                    }
                    if (out_elempack == 1)
                    {
                        float32x4x2_t _f01 = vzipq_f32(_f0, _f1);
                        float32x4x2_t _f23 = vzipq_f32(_f2, _f3);
                        vst1_f32(p0, vget_low_f32(_f01.val[0]));
                        vst1_f32(p0 + out_hstep, vget_high_f32(_f01.val[0]));
                        vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f01.val[1]));
                        vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f01.val[1]));
                        vst1_f32(p0 + out_hstep * 4, vget_low_f32(_f23.val[0]));
                        vst1_f32(p0 + out_hstep * 5, vget_high_f32(_f23.val[0]));
                        vst1_f32(p0 + out_hstep * 6, vget_low_f32(_f23.val[1]));
                        vst1_f32(p0 + out_hstep * 7, vget_high_f32(_f23.val[1]));
                        p0 += 2;
                    }
                }
            }
            else
            {
                uint16x4_t _out0;
                uint16x4_t _out1;
                uint16x4_t _out2;
                uint16x4_t _out3;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                {
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                    _out1 = vreinterpret_u16_f16(vcvt_f16_f32(_f1));
                    _out2 = vreinterpret_u16_f16(vcvt_f16_f32(_f2));
                    _out3 = vreinterpret_u16_f16(vcvt_f16_f32(_f3));
                }
                else
#endif
                {
                    _out0 = float2bfloat(_f0);
                    _out1 = float2bfloat(_f1);
                    _out2 = float2bfloat(_f2);
                    _out3 = float2bfloat(_f3);
                }

                if (output_transpose)
                {
                    vst1q_u16(p0s, vcombine_u16(_out0, _out2));
                    vst1q_u16(p0s + out_hstep, vcombine_u16(_out1, _out3));
                    p0s += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        vst1q_u16(p0s, vcombine_u16(_out0, _out2));
                        vst1q_u16(p0s + 8, vcombine_u16(_out1, _out3));
                        p0s += 16;
                    }
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0s, vcombine_u16(_out0, _out1));
                        vst1q_u16(p0s + out_hstep * 4, vcombine_u16(_out2, _out3));
                        p0s += 8;
                    }
                    if (out_elempack == 1)
                    {
                        p0s[0] = vget_lane_u16(_out0, 0);
                        p0s[1] = vget_lane_u16(_out1, 0);
                        p0s[out_hstep] = vget_lane_u16(_out0, 1);
                        p0s[out_hstep + 1] = vget_lane_u16(_out1, 1);
                        p0s[out_hstep * 2] = vget_lane_u16(_out0, 2);
                        p0s[out_hstep * 2 + 1] = vget_lane_u16(_out1, 2);
                        p0s[out_hstep * 3] = vget_lane_u16(_out0, 3);
                        p0s[out_hstep * 3 + 1] = vget_lane_u16(_out1, 3);
                        p0s[out_hstep * 4] = vget_lane_u16(_out2, 0);
                        p0s[out_hstep * 4 + 1] = vget_lane_u16(_out3, 0);
                        p0s[out_hstep * 5] = vget_lane_u16(_out2, 1);
                        p0s[out_hstep * 5 + 1] = vget_lane_u16(_out3, 1);
                        p0s[out_hstep * 6] = vget_lane_u16(_out2, 2);
                        p0s[out_hstep * 6 + 1] = vget_lane_u16(_out3, 2);
                        p0s[out_hstep * 7] = vget_lane_u16(_out2, 3);
                        p0s[out_hstep * 7 + 1] = vget_lane_u16(_out3, 3);
                        p0s += 2;
                    }
                }
            }

            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            float32x4_t _sum0 = vld1q_f32(pp);
            float32x4_t _sum1 = vld1q_f32(pp + 4);

            float32x4_t _f0 = _sum0;
            float32x4_t _f1 = _sum1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep * 4);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vsetq_lane_f32(pC[0], _c0, 0);
                        _c0 = vsetq_lane_f32(pC[c_hstep], _c0, 1);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 2], _c0, 2);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 3], _c0, 3);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 4], _c1, 0);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 5], _c1, 1);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 6], _c1, 2);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 7], _c1, 3);
                        pC += 1;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    vst1q_f32(p0, _f0);
                    vst1q_f32(p0 + 4, _f1);
                    p0 += out_hstep;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + out_hstep * 4, _f1);
                        p0 += 4;
                    }
                    if (out_elempack == 1)
                    {
                        p0[0] = vgetq_lane_f32(_f0, 0);
                        p0[out_hstep] = vgetq_lane_f32(_f0, 1);
                        p0[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                        p0[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                        p0[out_hstep * 4] = vgetq_lane_f32(_f1, 0);
                        p0[out_hstep * 5] = vgetq_lane_f32(_f1, 1);
                        p0[out_hstep * 6] = vgetq_lane_f32(_f1, 2);
                        p0[out_hstep * 7] = vgetq_lane_f32(_f1, 3);
                        p0++;
                    }
                }
            }
            else
            {
                uint16x4_t _out0;
                uint16x4_t _out1;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                {
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                    _out1 = vreinterpret_u16_f16(vcvt_f16_f32(_f1));
                }
                else
#endif
                {
                    _out0 = float2bfloat(_f0);
                    _out1 = float2bfloat(_f1);
                }

                if (output_transpose)
                {
                    vst1q_u16(p0s, vcombine_u16(_out0, _out1));
                    p0s += out_hstep;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        vst1q_u16(p0s, vcombine_u16(_out0, _out1));
                        p0s += 8;
                    }
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0s, _out0);
                        vst1_u16(p0s + out_hstep * 4, _out1);
                        p0s += 4;
                    }
                    if (out_elempack == 1)
                    {
                        p0s[0] = vget_lane_u16(_out0, 0);
                        p0s[out_hstep] = vget_lane_u16(_out0, 1);
                        p0s[out_hstep * 2] = vget_lane_u16(_out0, 2);
                        p0s[out_hstep * 3] = vget_lane_u16(_out0, 3);
                        p0s[out_hstep * 4] = vget_lane_u16(_out1, 0);
                        p0s[out_hstep * 5] = vget_lane_u16(_out1, 1);
                        p0s[out_hstep * 6] = vget_lane_u16(_out1, 2);
                        p0s[out_hstep * 7] = vget_lane_u16(_out1, 3);
                        p0s++;
                    }
                }
            }

            pp += 8;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = 0;
        unsigned short* p0s = 0;
        if (output_elemtype == 1)
        {
            if (output_transpose)
                p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            else
                p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }
        else
        {
            if (output_transpose)
                p0s = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            else
                p0s = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        float32x4_t _c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = vld1q_f32(pC);
                _c0 = vmulq_n_f32(_c0, beta);
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
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0 = vld1q_f32(pp);
            float32x4_t _sum1 = vld1q_f32(pp + 4);
            float32x4_t _sum2 = vld1q_f32(pp + 8);
            float32x4_t _sum3 = vld1q_f32(pp + 12);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
#else
            // from
            //      a0 b1 c2 d3
            //      c0 d1 a2 b3
            //      a3 b2 c1 d0
            //      c3 d2 a1 b0

            // to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
            {
                _sum2 = vrev64q_f32(_sum2);
                _sum3 = vrev64q_f32(_sum3);
                _sum2 = vextq_f32(_sum2, _sum2, 2);
                _sum3 = vextq_f32(_sum3, _sum3, 2);
                float32x4x2_t _t0 = vzipq_f32(_sum0, _sum3);
                float32x4x2_t _t1 = vzipq_f32(_sum1, _sum2);
                _sum0 = vcombine_f32(vget_low_f32(_t0.val[0]), vget_low_f32(_t1.val[0]));
                _sum1 = vcombine_f32(vget_high_f32(_t0.val[0]), vget_high_f32(_t1.val[0]));
                _sum2 = vcombine_f32(vget_low_f32(_t1.val[1]), vget_low_f32(_t0.val[1]));
                _sum3 = vcombine_f32(vget_high_f32(_t1.val[1]), vget_high_f32(_t0.val[1]));
                _sum1 = vrev64q_f32(_sum1);
                _sum3 = vrev64q_f32(_sum3);
            }
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = _sum0;
            float32x4_t _f1 = _sum1;
            float32x4_t _f2 = _sum2;
            float32x4_t _f3 = _sum3;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c1;
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + 8);
                        _c3 = vld1q_f32(pC + 12);
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep * 1);
                        _c2 = vld1q_f32(pC + c_hstep * 2);
                        _c3 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    _c = vmulq_n_f32(_c, beta);
#if __aarch64__
                    _c0 = vdupq_laneq_f32(_c, 0);
                    float32x4_t _c1 = vdupq_laneq_f32(_c, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_c, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_c, 3);
#else
                    _c0 = vdupq_lane_f32(vget_low_f32(_c), 0);
                    float32x4_t _c1 = vdupq_lane_f32(vget_low_f32(_c), 1);
                    float32x4_t _c2 = vdupq_lane_f32(vget_high_f32(_c), 0);
                    float32x4_t _c3 = vdupq_lane_f32(vget_high_f32(_c), 1);
#endif
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float32x4x4_t _r;
                        _r.val[0] = _f0;
                        _r.val[1] = _f1;
                        _r.val[2] = _f2;
                        _r.val[3] = _f3;
                        vst4q_f32(p0, _r);
                    }
                    if (out_elempack == 1)
                    {
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + out_hstep, _f1);
                        vst1q_f32(p0 + out_hstep * 2, _f2);
                        vst1q_f32(p0 + out_hstep * 3, _f3);
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + 4, _f1);
                        vst1q_f32(p0 + 8, _f2);
                        vst1q_f32(p0 + 12, _f3);
                        p0 += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_f0, _f1, _f2, _f3);
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + out_hstep, _f1);
                        vst1q_f32(p0 + out_hstep * 2, _f2);
                        vst1q_f32(p0 + out_hstep * 3, _f3);
                        p0 += 4;
                    }
                }
            }
            else
            {
                uint16x4_t _out0;
                uint16x4_t _out1;
                uint16x4_t _out2;
                uint16x4_t _out3;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                {
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                    _out1 = vreinterpret_u16_f16(vcvt_f16_f32(_f1));
                    _out2 = vreinterpret_u16_f16(vcvt_f16_f32(_f2));
                    _out3 = vreinterpret_u16_f16(vcvt_f16_f32(_f3));
                }
                else
#endif
                {
                    _out0 = float2bfloat(_f0);
                    _out1 = float2bfloat(_f1);
                    _out2 = float2bfloat(_f2);
                    _out3 = float2bfloat(_f3);
                }

                if (output_transpose)
                {
#if __aarch64__
                    if (out_elempack == 8)
                    {
                        transpose4x4_u16(_out0, _out1, _out2, _out3);
                        const int jj_m8 = jj % 8;
                        unsigned short* p1 = p0s - out_hstep * jj_m8 + jj_m8;
                        vst1_u16(p1, _out0);
                        vst1_u16(p1 + 8, _out1);
                        vst1_u16(p1 + 16, _out2);
                        vst1_u16(p1 + 24, _out3);
                    }
#endif // __aarch64__
                    if (out_elempack == 4)
                    {
                        uint16x4x4_t _r;
                        _r.val[0] = _out0;
                        _r.val[1] = _out1;
                        _r.val[2] = _out2;
                        _r.val[3] = _out3;
                        vst4_u16(p0s, _r);
                    }
                    if (out_elempack == 1)
                    {
                        vst1_u16(p0s, _out0);
                        vst1_u16(p0s + out_hstep, _out1);
                        vst1_u16(p0s + out_hstep * 2, _out2);
                        vst1_u16(p0s + out_hstep * 3, _out3);
                    }
                    p0s += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0s, _out0);
                        vst1_u16(p0s + 4, _out1);
                        vst1_u16(p0s + 8, _out2);
                        vst1_u16(p0s + 12, _out3);
                        p0s += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_u16(_out0, _out1, _out2, _out3);
                        vst1_u16(p0s, _out0);
                        vst1_u16(p0s + out_hstep, _out1);
                        vst1_u16(p0s + out_hstep * 2, _out2);
                        vst1_u16(p0s + out_hstep * 3, _out3);
                        p0s += 4;
                    }
                }
            }

            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum0 = vld1q_f32(pp);
            float32x4_t _sum1 = vld1q_f32(pp + 4);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
#else
            // from
            //      a0 b1 c0 d1
            //      a1 b0 c1 d0

            // to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            {
                _sum1 = vrev64q_f32(_sum1);
                float32x4x2_t _t0 = vzipq_f32(_sum0, _sum1);
                _sum0 = vcombine_f32(vget_low_f32(_t0.val[0]), vget_low_f32(_t0.val[1]));
                _sum1 = vcombine_f32(vget_high_f32(_t0.val[0]), vget_high_f32(_t0.val[1]));
                _sum1 = vrev64q_f32(_sum1);
            }
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = _sum0;
            float32x4_t _f1 = _sum1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c1;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        float32x2_t _cc0 = vld1_f32(pC);
                        float32x2_t _cc1 = vld1_f32(pC + c_hstep);
                        float32x2_t _cc2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _cc3 = vld1_f32(pC + c_hstep * 3);
                        float32x4_t _c01 = vcombine_f32(_cc0, _cc1);
                        float32x4_t _c23 = vcombine_f32(_cc2, _cc3);
                        float32x4x2_t _cc = vuzpq_f32(_c01, _c23);
                        _c0 = _cc.val[0];
                        _c1 = _cc.val[1];
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    _c = vmul_n_f32(_c, beta);
                    _c0 = vdupq_lane_f32(_c, 0);
                    float32x4_t _c1 = vdupq_lane_f32(_c, 1);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    vst1q_f32(p0, _f0);
                    vst1q_f32(p0 + out_hstep, _f1);
                    p0 += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + 4, _f1);
                        p0 += 8;
                    }
                    if (out_elempack == 1)
                    {
                        float32x4x2_t _f01 = vzipq_f32(_f0, _f1);
                        vst1_f32(p0, vget_low_f32(_f01.val[0]));
                        vst1_f32(p0 + out_hstep, vget_high_f32(_f01.val[0]));
                        vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f01.val[1]));
                        vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f01.val[1]));
                        p0 += 2;
                    }
                }
            }
            else
            {
                uint16x4_t _out0;
                uint16x4_t _out1;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                {
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                    _out1 = vreinterpret_u16_f16(vcvt_f16_f32(_f1));
                }
                else
#endif
                {
                    _out0 = float2bfloat(_f0);
                    _out1 = float2bfloat(_f1);
                }

                if (output_transpose)
                {
                    vst1_u16(p0s, _out0);
                    vst1_u16(p0s + out_hstep, _out1);
                    p0s += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_u16(p0s, vcombine_u16(_out0, _out1));
                        p0s += 8;
                    }
                    if (out_elempack == 1)
                    {
                        p0s[0] = vget_lane_u16(_out0, 0);
                        p0s[1] = vget_lane_u16(_out1, 0);
                        p0s[out_hstep] = vget_lane_u16(_out0, 1);
                        p0s[out_hstep + 1] = vget_lane_u16(_out1, 1);
                        p0s[out_hstep * 2] = vget_lane_u16(_out0, 2);
                        p0s[out_hstep * 2 + 1] = vget_lane_u16(_out1, 2);
                        p0s[out_hstep * 3] = vget_lane_u16(_out0, 3);
                        p0s[out_hstep * 3 + 1] = vget_lane_u16(_out1, 3);
                        p0s += 2;
                    }
                }
            }

            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            float32x4_t _f0 = vld1q_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vsetq_lane_f32(pC[0], _c0, 0);
                        _c0 = vsetq_lane_f32(pC[c_hstep], _c0, 1);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 2], _c0, 2);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 3], _c0, 3);
                        pC += 1;
                    }
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    vst1q_f32(p0, _f0);
                    p0 += out_hstep;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0, _f0);
                        p0 += 4;
                    }
                    if (out_elempack == 1)
                    {
                        p0[0] = vgetq_lane_f32(_f0, 0);
                        p0[out_hstep] = vgetq_lane_f32(_f0, 1);
                        p0[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                        p0[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                        p0++;
                    }
                }
            }
            else
            {
                uint16x4_t _out0;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                else
#endif
                    _out0 = float2bfloat(_f0);

                if (output_transpose)
                {
                    vst1_u16(p0s, _out0);
                    p0s += out_hstep;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0s, _out0);
                        p0s += 4;
                    }
                    if (out_elempack == 1)
                    {
                        p0s[0] = vget_lane_u16(_out0, 0);
                        p0s[out_hstep] = vget_lane_u16(_out0, 1);
                        p0s[out_hstep * 2] = vget_lane_u16(_out0, 2);
                        p0s[out_hstep * 3] = vget_lane_u16(_out0, 3);
                        p0s++;
                    }
                }
            }

            pp += 4;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        // out_elempack == 1
        float* p0 = 0;
        unsigned short* p0s = 0;
        if (output_elemtype == 1)
        {
            if (output_transpose)
                p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            else
                p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }
        else
        {
            if (output_transpose)
                p0s = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            else
                p0s = (unsigned short*)top_blob + (i + ii) * out_hstep + j;
        }

        float c0;
        float c1;
#if __ARM_NEON
        float32x4_t _c0;
        float32x4_t _c1;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
                _c1 = vdupq_n_f32(c1);
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __ARM_NEON
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0 = vld1q_f32(pp);
            float32x4_t _sum1 = vld1q_f32(pp + 4);

            float32x4_t _f0 = _sum0;
            float32x4_t _f1 = _sum1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = vld1q_f32(pC);
                    _c1 = vld1q_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vld1q_f32(pC);
                    _c0 = vmulq_n_f32(_c0, beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        vst1q_f32(p0, _f0);
                        vst1q_f32(p0 + 4, _f1);
                    }
                    if (out_elempack == 1)
                    {
                        float32x4x2_t _f01 = vzipq_f32(_f0, _f1);
                        vst1_f32(p0, vget_low_f32(_f01.val[0]));
                        vst1_f32(p0 + out_hstep, vget_high_f32(_f01.val[0]));
                        vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f01.val[1]));
                        vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f01.val[1]));
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    vst1q_f32(p0, _f0);
                    vst1q_f32(p0 + out_hstep, _f1);
                    p0 += 4;
                }
            }
            else
            {
                uint16x4_t _out0;
                uint16x4_t _out1;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                {
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                    _out1 = vreinterpret_u16_f16(vcvt_f16_f32(_f1));
                }
                else
#endif
                {
                    _out0 = float2bfloat(_f0);
                    _out1 = float2bfloat(_f1);
                }
                if (output_transpose)
                {
#if __aarch64__
                    if (out_elempack == 8)
                    {
                        const int jj_m8 = jj % 8;
                        unsigned short* p1 = p0s - out_hstep * jj_m8 + jj_m8;
                        vst1_u16(p1, _out0);
                        vst1_u16(p1 + 8, _out1);
                    }
#endif // __aarch64__
                    if (out_elempack == 4)
                    {
                        vst1_u16(p0s, _out0);
                        vst1_u16(p0s + 4, _out1);
                    }
                    if (out_elempack == 1)
                    {
                        uint16x4x2_t _out01 = vzip_u16(_out0, _out1);
                        vst1_lane_u32((unsigned int*)p0s, vreinterpret_u32_u16(_out01.val[0]), 0);
                        vst1_lane_u32((unsigned int*)(p0s + out_hstep), vreinterpret_u32_u16(_out01.val[0]), 1);
                        vst1_lane_u32((unsigned int*)(p0s + out_hstep * 2), vreinterpret_u32_u16(_out01.val[1]), 0);
                        vst1_lane_u32((unsigned int*)(p0s + out_hstep * 3), vreinterpret_u32_u16(_out01.val[1]), 1);
                    }
                    p0s += out_hstep * 4;
                }
                else
                {
                    vst1_u16(p0s, _out0);
                    vst1_u16(p0s + out_hstep, _out1);
                    p0s += 4;
                }
            }

            pp += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _f0;
            if (output_transpose)
            {
                float32x2x2_t _sum0 = vld2_f32(pp);
                _f0 = vcombine_f32(_sum0.val[0], _sum0.val[1]);
            }
            else
            {
                _f0 = vld1q_f32(pp);
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _cc;
                    if (output_transpose)
                        _cc = vzipq_f32(_c0, _c1).val[0];
                    else
                        _cc = vcombine_f32(vget_low_f32(_c0), vget_high_f32(_c1));
                    _f0 = vaddq_f32(_f0, _cc);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x2_t _cc0 = vld1_f32(pC);
                    float32x2_t _cc1 = vld1_f32(pC + c_hstep);
                    if (output_transpose)
                    {
                        float32x2x2_t _c01 = vzip_f32(_cc0, _cc1);
                        _c0 = vcombine_f32(_c01.val[0], _c01.val[1]);
                    }
                    else
                    {
                        _c0 = vcombine_f32(_cc0, _cc1);
                    }
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _cc = vld1_f32(pC);
                    if (output_transpose)
                    {
                        float32x2x2_t _c01 = vzip_f32(_cc, _cc);
                        _c0 = vcombine_f32(_c01.val[0], _c01.val[1]);
                    }
                    else
                    {
                        _c0 = vcombine_f32(_cc, _cc);
                    }
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            if (output_elemtype == 1)
            {
                vst1_f32(p0, vget_low_f32(_f0));
                vst1_f32(p0 + out_hstep, vget_high_f32(_f0));
                if (output_transpose)
                    p0 += out_hstep * 2;
                else
                    p0 += 2;
            }
            else
            {
                uint16x4_t _out0;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                else
#endif
                    _out0 = float2bfloat(_f0);
                vst1_lane_u32((unsigned int*)p0s, vreinterpret_u32_u16(_out0), 0);
                vst1_lane_u32((unsigned int*)(p0s + out_hstep), vreinterpret_u32_u16(_out0), 1);
                if (output_transpose)
                    p0s += out_hstep * 2;
                else
                    p0s += 2;
            }

            pp += 4;
        }
#endif // __ARM_NEON
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            float f1 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    f0 += pC[0] * beta;
                    f1 += pC[c_hstep] * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    p0[0] = f0;
                    p0[1] = f1;
                    p0 += out_hstep;
                }
                else
                {
                    p0[0] = f0;
                    p0[out_hstep] = f1;
                    p0++;
                }
            }
            else
            {
                unsigned short out0;
                unsigned short out1;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                {
                    out0 = float32_to_float16(f0);
                    out1 = float32_to_float16(f1);
                }
                else
#endif
                {
                    out0 = float32_to_bfloat16(f0);
                    out1 = float32_to_bfloat16(f1);
                }

                if (output_transpose)
                {
                    p0s[0] = out0;
                    p0s[1] = out1;
                    p0s += out_hstep;
                }
                else
                {
                    p0s[0] = out0;
                    p0s[out_hstep] = out1;
                    p0s++;
                }
            }

            pp += 2;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // out_elempack == 1
        float* p0 = 0;
        unsigned short* p0s = 0;
        if (output_elemtype == 1)
        {
            if (output_transpose)
                p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            else
                p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }
        else
        {
            if (output_transpose)
                p0s = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            else
                p0s = (unsigned short*)top_blob + (i + ii) * out_hstep + j;
        }

        float c0;
#if __ARM_NEON
        float32x4_t _c0;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __ARM_NEON
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _f0 = vld1q_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = vld1q_f32(pC);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 4;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_hstep == 1)
                    {
                        vst1q_f32(p0, _f0);
                    }
                    else
                    {
                        if (out_elempack == 4)
                        {
                            vst1q_f32(p0, _f0);
                        }
                        if (out_elempack == 1)
                        {
                            p0[0] = vgetq_lane_f32(_f0, 0);
                            p0[out_hstep] = vgetq_lane_f32(_f0, 1);
                            p0[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                            p0[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                        }
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    vst1q_f32(p0, _f0);
                    p0 += 4;
                }
            }
            else
            {
                uint16x4_t _out0;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(_f0));
                else
#endif
                    _out0 = float2bfloat(_f0);
                if (output_transpose)
                {
                    if (out_hstep == 1)
                    {
                        vst1_u16(p0s, _out0);
                    }
                    else
                    {
#if __aarch64__
                        if (out_elempack == 8)
                        {
                            const int jj_m8 = jj % 8;
                            unsigned short* p1 = p0s - out_hstep * jj_m8 + jj_m8;
                            vst1_u16(p1, _out0);
                        }
#endif // __aarch64__
                        if (out_elempack == 4)
                        {
                            vst1_u16(p0s, _out0);
                        }
                        if (out_elempack == 1)
                        {
                            p0s[0] = vget_lane_u16(_out0, 0);
                            p0s[out_hstep] = vget_lane_u16(_out0, 1);
                            p0s[out_hstep * 2] = vget_lane_u16(_out0, 2);
                            p0s[out_hstep * 3] = vget_lane_u16(_out0, 3);
                        }
                    }
                    p0s += out_hstep * 4;
                }
                else
                {
                    vst1_u16(p0s, _out0);
                    p0s += 4;
                }
            }

            pp += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _f0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vadd_f32(_f0, vget_low_f32(_c0));
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    float32x2_t _c = vld1_f32(pC);
                    _f0 = vmla_n_f32(_f0, _c, beta);
                    pC += 2;
                }
            }

            _f0 = vmul_n_f32(_f0, alpha);

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_hstep == 1)
                    {
                        vst1_f32(p0, _f0);
                    }
                    else
                    {
                        p0[0] = vget_lane_f32(_f0, 0);
                        p0[out_hstep] = vget_lane_f32(_f0, 1);
                    }
                    p0 += out_hstep * 2;
                }
                else
                {
                    vst1_f32(p0, _f0);
                    p0 += 2;
                }
            }
            else
            {
                uint16x4_t _out0;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                    _out0 = vreinterpret_u16_f16(vcvt_f16_f32(vcombine_f32(_f0, _f0)));
                else
#endif
                    _out0 = float2bfloat(vcombine_f32(_f0, _f0));
                if (output_transpose)
                {
                    p0s[0] = vget_lane_u16(_out0, 0);
                    p0s[out_hstep] = vget_lane_u16(_out0, 1);
                    p0s += out_hstep * 2;
                }
                else
                {
                    vst1_lane_u32((unsigned int*)p0s, vreinterpret_u32_u16(_out0), 0);
                    p0s += 2;
                }
            }

            pp += 2;
        }
#endif // __ARM_NEON
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    f0 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            if (output_elemtype == 1)
            {
                p0[0] = f0;
                if (output_transpose)
                    p0 += out_hstep;
                else
                    p0++;
            }
            else
            {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || (__ARM_FP & 2)
                if (output_elemtype == 2)
                    p0s[0] = float32_to_float16(f0);
                else
#endif
                    p0s[0] = float32_to_bfloat16(f0);
                if (output_transpose)
                    p0s += out_hstep;
                else
                    p0s++;
            }

            pp += 1;
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        gemm_transB_packed_tile_wq_int8_i8mm(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        gemm_transB_packed_tile_wq_int8_asimddp(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
        return;
    }
#endif

    const signed char* pAT = AT_tile;
    const int A_hstep = AT_tile.w;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = AT_descales_tile.w;
    const signed char* pBT = BT_tile;
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
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
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
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
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
                    pB += 32;
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
                    int8x16_t _b = vld1q_s8(pB);
                    _sum0 = vdotq_laneq_s32(_sum0, _a0, _b, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a0, _b, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a0, _b, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a0, _b, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _a1, _b, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _a1, _b, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _a1, _b, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _a1, _b, 3);
                    pA += 32;
                    pB += 16;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA2 = vld1q_s8(pA + 16);
                    int8x16_t _pB02 = vld1q_s8(pB);
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
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = vld1_s8(pB);
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
                    int8x8_t _pB0 = vld1_s8(pB);
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
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vld1_s8(pA);
                    int16x8_t _s0 = vmull_s8(_a, vdup_n_s8(pB[0]));
                    int16x8_t _s1 = vmull_s8(_a, vdup_n_s8(pB[1]));
                    int16x8_t _s2 = vmull_s8(_a, vdup_n_s8(pB[2]));
                    int16x8_t _s3 = vmull_s8(_a, vdup_n_s8(pB[3]));
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
                    int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
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
                    pB += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales);
                float32x4_t _ad0 = vld1q_f32(pA_descales);
                float32x4_t _ad1 = vld1q_f32(pA_descales + 4);
#if __ARM_FEATURE_DOTPROD
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_ad0, vgetq_lane_f32(_bd0, 0)));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_ad0, vgetq_lane_f32(_bd0, 1)));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_n_f32(_ad0, vgetq_lane_f32(_bd0, 2)));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_n_f32(_ad0, vgetq_lane_f32(_bd0, 3)));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_n_f32(_ad1, vgetq_lane_f32(_bd0, 0)));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_n_f32(_ad1, vgetq_lane_f32(_bd0, 1)));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_n_f32(_ad1, vgetq_lane_f32(_bd0, 2)));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_n_f32(_ad1, vgetq_lane_f32(_bd0, 3)));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _ad2 = vextq_f32(_ad0, _ad0, 2);
                float32x4_t _ad3 = vextq_f32(_ad1, _ad1, 2);
                float32x4_t _bd1 = vrev64q_f32(_bd0);
                _bd1 = vextq_f32(_bd1, _bd1, 2);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_ad0, _bd0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_ad1, _bd0));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_f32(_ad2, _bd0));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_f32(_ad3, _bd0));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_f32(_ad0, _bd1));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_f32(_ad1, _bd1));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_f32(_ad2, _bd1));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_f32(_ad3, _bd1));
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
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
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
                    int8x16_t _b = vld1q_s8(pB);
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
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x8_t _b = vld1_s8(pB);
                    _sum0 = vdotq_lane_s32(_sum0, _a0, _b, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _a0, _b, 1);
                    _sum2 = vdotq_lane_s32(_sum2, _a1, _b, 0);
                    _sum3 = vdotq_lane_s32(_sum3, _a1, _b, 1);
                    pA += 32;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA2 = vld1q_s8(pA + 16);
                    int8x8_t _pB = vld1_s8(pB);
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
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x16_t _a = vld1q_s8(pA);
                    int16x4_t _b = vreinterpret_s16_s32(vld1_dup_s32((const int*)pB));
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
                    int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
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
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    int8x8x2_t _b01 = vuzp_s8(_b, _b);
                    int16x8_t _s0 = vmull_s8(_a, _b01.val[0]);
                    int16x8_t _s1 = vmull_s8(_a, _b01.val[1]);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_low_s16(_s1));
                    _sum2 = vaddw_s16(_sum2, vget_high_s16(_s0));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);

                    int16x8_t _s0 = vmull_s8(_pA, _pB0);
                    int16x8_t _s1 = vmull_s8(_pA, _pB1);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
#endif // __ARM_FEATURE_DOTPROD
                    pA += 8;
                    pB += 2;
                }

                float32x2_t _bd = vld1_f32(pB_descales);
                float32x4_t _ad0 = vld1q_f32(pA_descales);
                float32x4_t _ad1 = vld1q_f32(pA_descales + 4);
#if __ARM_FEATURE_DOTPROD
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_ad0, _bd, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_ad0, _bd, 1));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_lane_f32(_ad1, _bd, 0));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_lane_f32(_ad1, _bd, 1));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _bd01 = vcombine_f32(_bd, _bd);
                float32x4_t _bd10 = vrev64q_f32(_bd01);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_ad0, _bd01));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_ad1, _bd01));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_f32(_ad0, _bd10));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_f32(_ad1, _bd10));
#endif // __ARM_FEATURE_DOTPROD
                pA_descales += 8;
                pB_descales += 2;
            }

            vst1q_f32(outptr, _fsum0);
            vst1q_f32(outptr + 4, _fsum1);
            vst1q_f32(outptr + 8, _fsum2);
            vst1q_f32(outptr + 12, _fsum3);
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
                    int8x8_t _b = vld1_s8(pB);
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
                    pB += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int8x16_t _b = vreinterpretq_s8_s32(vld1q_dup_s32((const int*)pB));
                    _sum0 = vdotq_s32(_sum0, _a0, _b);
                    _sum1 = vdotq_s32(_sum1, _a1, _b);
                    pA += 32;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a0 = vld1q_s8(pA);
                    int8x16_t _a1 = vld1q_s8(pA + 16);
                    int16x4_t _b = vreinterpret_s16_s32(vld1_dup_s32((const int*)pB));
                    int8x8_t _b0 = vreinterpret_s8_s16(vdup_lane_s16(_b, 0));
                    int8x8_t _b1 = vreinterpret_s8_s16(vdup_lane_s16(_b, 1));
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(vget_low_s8(_a0), _b0));
                    _sum1 = vpadalq_s16(_sum1, vmull_s8(vget_high_s8(_a0), _b0));
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(vget_low_s8(_a1), _b1));
                    _sum1 = vpadalq_s16(_sum1, vmull_s8(vget_high_s8(_a1), _b1));
                    pA += 32;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(vget_low_s8(_a), _b));
                    _sum1 = vpadalq_s16(_sum1, vmull_s8(vget_high_s8(_a), _b));
                    pA += 16;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int16x8_t _s = vmull_s8(_a, vld1_dup_s8(pB));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s));
                    pA += 8;
                    pB++;
                }

                float32x4_t _bd = vdupq_n_f32(pB_descales[0]);
                float32x4_t _ad0 = vld1q_f32(pA_descales);
                float32x4_t _ad1 = vld1q_f32(pA_descales + 4);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_bd, _ad0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_bd, _ad1));
                pA_descales += 8;
                pB_descales++;
            }

            vst1q_f32(outptr, _fsum0);
            vst1q_f32(outptr + 4, _fsum1);
            outptr += 8;
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += (size_t)8 * A_hstep;
        pAT_descales += (size_t)8 * A_descales_hstep;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        int jj = 0;
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
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
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
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
                    pB += 32;
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
                    int8x16_t _b = vld1q_s8(pB);
                    _sum0 = vdotq_laneq_s32(_sum0, _a, _b, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _a, _b, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _a, _b, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _a, _b, 3);
                    pA += 16;
                    pB += 16;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _pA02 = vld1q_s8(pA);
                    int8x16_t _pB02 = vld1q_s8(pB);
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
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = vld1_s8(pB);
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
                    int8x8_t _pB0 = vld1_s8(pB);
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
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _b = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
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
                    int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
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
                    pB += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales);
                float32x4_t _ad = vld1q_f32(pA_descales);
#if __ARM_FEATURE_DOTPROD
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_ad, vgetq_lane_f32(_bd0, 0)));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_ad, vgetq_lane_f32(_bd0, 1)));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_n_f32(_ad, vgetq_lane_f32(_bd0, 2)));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_n_f32(_ad, vgetq_lane_f32(_bd0, 3)));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _ad1 = vextq_f32(_ad, _ad, 2);
                float32x4_t _bd1 = vrev64q_f32(_bd0);
                _bd1 = vextq_f32(_bd1, _bd1, 2);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_ad, _bd0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_ad1, _bd0));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_f32(_ad, _bd1));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_f32(_ad1, _bd1));
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
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
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
                    int8x16_t _b = vld1q_s8(pB);
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
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss = vuzpq_s32(_s0, _s1);
                _sum0 = vaddq_s32(_sum0, _ss.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = vld1_s8(pB);
                    _sum0 = vdotq_lane_s32(_sum0, _a, _b, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _a, _b, 1);
                    pA += 16;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x8_t _pB = vld1_s8(pB);
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
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = vld1_s8(pB);
                    int16x8_t _s0 = vmull_s8(_a, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 0)));
                    int16x8_t _s1 = vmull_s8(_a, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_b), 1)));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                    int8x8_t _pB1 = vext_s8(_pB0, _pB0, 2);

                    int16x8_t _s0 = vmull_s8(_pA, _pB0);
                    int16x8_t _s1 = vmull_s8(_pA, _pB1);
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
#if __ARM_FEATURE_DOTPROD
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    _b = vuzp_s8(_b, vext_s8(_b, _b, 1)).val[0];
                    int16x8_t _s = vmull_s8(_a, _b);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s));
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);
                    int8x8_t _pB = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB0), vreinterpret_s32_s8(_pB1)).val[0]);

                    int16x8_t _s0 = vmull_s8(_pA, _pB);
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
#endif // __ARM_FEATURE_DOTPROD
                    pA += 4;
                    pB += 2;
                }

                float32x2_t _bd = vld1_f32(pB_descales);
                float32x4_t _ad = vld1q_f32(pA_descales);
#if __ARM_FEATURE_DOTPROD
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_ad, _bd, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_ad, _bd, 1));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _bd01 = vcombine_f32(_bd, _bd);
                float32x4_t _bd10 = vrev64q_f32(_bd01);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_f32(_ad, _bd01));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_f32(_ad, _bd10));
#endif // __ARM_FEATURE_DOTPROD
                pA_descales += 4;
                pB_descales += 2;
            }

            vst1q_f32(outptr, _fsum0);
            vst1q_f32(outptr + 4, _fsum1);
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
                    int8x8_t _b = vld1_s8(pB);
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
                    pB += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x16_t _b = vreinterpretq_s8_s32(vld1q_dup_s32((const int*)pB));
                    _sum = vdotq_s32(_sum, _a, _b);
                    pA += 16;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a0 = vld1_s8(pA);
                    int8x8_t _a1 = vld1_s8(pA + 8);
                    int16x4_t _b = vreinterpret_s16_s32(vld1_dup_s32((const int*)pB));
                    int8x8_t _b0 = vreinterpret_s8_s16(vdup_lane_s16(_b, 0));
                    int8x8_t _b1 = vreinterpret_s8_s16(vdup_lane_s16(_b, 1));
                    _sum = vpadalq_s16(_sum, vmull_s8(_a0, _b0));
                    _sum = vpadalq_s16(_sum, vmull_s8(_a1, _b1));
                    pA += 16;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    _sum = vpadalq_s16(_sum, vmull_s8(_a, _b));
                    pA += 8;
                    pB += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int16x8_t _s = vmull_s8(_a, vld1_dup_s8(pB));
                    _sum = vaddw_s16(_sum, vget_low_s16(_s));
                    pA += 4;
                    pB++;
                }

                float32x4_t _bd = vdupq_n_f32(pB_descales[0]);
                float32x4_t _ad = vld1q_f32(pA_descales);
                _fsum = vmlaq_f32(_fsum, vcvtq_f32_s32(_sum), vmulq_f32(_bd, _ad));
                pA_descales += 4;
                pB_descales++;
            }

            vst1q_f32(outptr, _fsum);
            outptr += 4;
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += (size_t)4 * A_hstep;
        pAT_descales += (size_t)4 * A_descales_hstep;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        int jj = 0;
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
#if __ARM_NEON
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
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
                    int8x16_t _b0 = vld1q_s8(pB + 0);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
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
                    pB += 32;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum1));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    pA += 8;
                    pB += 16;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    int8x16_t _b = vld1q_s8(pB);
                    int8x8_t _b01 = vget_low_s8(_b);
                    int8x8_t _b23 = vget_high_s8(_b);
                    int16x8_t _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    _s = vmull_s8(_b01, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _s = vmlal_s8(_s, _b23, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)));
                    _sum1 = vpadalq_s16(_sum1, _s);
                    pA += 8;
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _b0 = vld1_s8(pB);
                    int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    int8x8_t _a = vreinterpret_s8_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0));
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    pA += 2;
                    pB += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales);
                float32x2_t _ad = vld1_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_bd0, _ad, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_bd0, _ad, 1));

                pA_descales += 2;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_NEON
            const signed char* pB = pB_panel + (size_t)2 * k;
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
                    int8x16_t _b = vld1q_s8(pB);
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
                    pB += 16;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    int8x8_t _b = vld1_s8(pB);
                    int32x4_t _s0 = vdotq_lane_s32(vdupq_n_s32(0), _a, _b, 0);
                    int32x4_t _s1 = vdotq_lane_s32(vdupq_n_s32(0), _a, _b, 1);
                    _sum = vaddq_s32(_sum, vzipq_s32(_s0, _s1).val[0]);
                    pA += 8;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB = vld1_s8(pB);

                    int16x4x2_t _pA01 = vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA));
                    int32x2x2_t _pB01 = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));

                    int16x8_t _s0 = vmull_s8(vreinterpret_s8_s16(_pA01.val[0]), vreinterpret_s8_s32(_pB01.val[0]));
                    _s0 = vmlal_s8(_s0, vreinterpret_s8_s16(_pA01.val[1]), vreinterpret_s8_s32(_pB01.val[1]));
                    _sum = vpadalq_s16(_sum, _s0);
                    pA += 8;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int16x4_t _b = vreinterpret_s16_s32(vld1_dup_s32((const int*)pB));
                    int16x4x2_t _b01 = vuzp_s16(_b, _b);
                    int32x4_t _s0 = vpaddlq_s16(vmull_s8(_a, vreinterpret_s8_s16(_b01.val[0])));
                    int32x4_t _s1 = vpaddlq_s16(vmull_s8(_a, vreinterpret_s8_s16(_b01.val[1])));
                    _sum = vaddq_s32(_sum, vzipq_s32(_s0, _s1).val[0]);
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _a = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                    int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    int8x8_t _aa = vzip_s8(_a, _a).val[0];
                    _sum = vaddq_s32(_sum, vmovl_s16(vget_low_s16(vmull_s8(_aa, _b))));
                    pA += 2;
                    pB += 2;
                }

                float32x2_t _ad = vld1_f32(pA_descales);
                float32x2_t _bd = vld1_f32(pB_descales);
                float32x4_t _adad = vcombine_f32(_ad, _ad);
                float32x4_t _bdbd = vcombine_f32(_bd, _bd);
                _fsum = vmlaq_f32(_fsum, vcvtq_f32_s32(_sum), vmulq_f32(vzipq_f32(_adad, _adad).val[0], _bdbd));
                pA_descales += 2;
                pB_descales += 2;
            }

            vst1q_f32(outptr, _fsum);
            outptr += 4;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
#else
            const signed char* pB = pB_panel + (size_t)2 * k;
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
                    const int b00 = pB[0];
                    const int b01 = pB[1];
                    const int b10 = pB[2];
                    const int b11 = pB[3];
                    sum00 += pA[0] * b00 + pA[2] * b01;
                    sum01 += pA[0] * b10 + pA[2] * b11;
                    sum10 += pA[1] * b00 + pA[3] * b01;
                    sum11 += pA[1] * b10 + pA[3] * b11;
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const int b0 = pB[0];
                    const int b1 = pB[1];
                    sum00 += pA[0] * b0;
                    sum01 += pA[0] * b1;
                    sum10 += pA[1] * b0;
                    sum11 += pA[1] * b1;
                    pA += 2;
                    pB += 2;
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
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
#endif // __ARM_NEON
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
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
                        int8x8_t _b = vld1_s8(pB);
                        int8x16_t _bb = vcombine_s8(_b, _b);
                        _sum0 = vdotq_s32(_sum0, _a, _bb);
                        pA += 16;
                        pB += 8;
                    }
                    _sum = vadd_s32(_sum, vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0)));
                }
#else  // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x8_t _b = vld1_s8(pB);
                    _sum = vdot_lane_s32(_sum, vget_low_s8(_a), _b, 0);
                    _sum = vdot_lane_s32(_sum, vget_high_s8(_a), _b, 1);
                    pA += 16;
                    pB += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                    _sum = vdot_s32(_sum, _a, _b);
                    pA += 8;
                    pB += 4;
                }
#else  // __ARM_FEATURE_DOTPROD
                {
                    int32x4_t _sum0 = vdupq_n_s32(0);
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        int8x8_t _a = vld1_s8(pA);
                        int8x8_t _b = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));
                        _b = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_b), vreinterpret_s16_s8(_b)).val[0]);
                        _sum0 = vpadalq_s16(_sum0, vmull_s8(_a, _b));
                        pA += 8;
                        pB += 4;
                    }
                    _sum = vadd_s32(_sum, vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0)));
                }
#endif // __ARM_FEATURE_DOTPROD
                sum0 = vget_lane_s32(_sum, 0);
                sum1 = vget_lane_s32(_sum, 1);
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[1];
                    sum1 += pA[2] * pB[0];
                    sum1 += pA[3] * pB[1];
                    pA += 4;
                    pB += 2;
                }
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[1] * pB[0];
                    pA += 2;
                    pB++;
                }

#if __ARM_NEON
                float32x2_t _ad = vld1_f32(pA_descales);
                float32x2_t _scale = vmul_n_f32(_ad, pB_descales[0]);
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
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += (size_t)2 * A_hstep;
        pAT_descales += (size_t)2 * A_descales_hstep;
    }
    for (; ii < max_ii; ii++)
    {
        int jj = 0;
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
#if __ARM_NEON
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
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
                    int8x16_t _b0 = vld1q_s8(pB + 0);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
                    int8x16_t _a0 = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
#if __ARM_FEATURE_MATMUL_INT8
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _b0, vget_low_s8(_a0), 0);
                    _sum0 = vdotq_lane_s32(_sum0, _b1, vget_low_s8(_a0), 1);
#endif // __ARM_FEATURE_MATMUL_INT8
                    pA += 8;
                    pB += 32;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    pA += 4;
                    pB += 16;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    int8x16_t _b = vld1q_s8(pB);
                    int16x8_t _s = vmull_s8(vget_low_s8(_b), vreinterpret_s8_s16(vdup_lane_s16(_a, 0)));
                    _s = vmlal_s8(_s, vget_high_s8(_b), vreinterpret_s8_s16(vdup_lane_s16(_a, 1)));
                    _sum0 = vpadalq_s16(_sum0, _s);
                    pA += 4;
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    int8x8_t _b0 = vld1_s8(pB);
                    int8x8_t _a0 = vreinterpret_s8_s16(vdup_lane_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, _a0)));
                    pA += 2;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    int8x8_t _b = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    int8x8_t _a0 = vld1_lane_s8(pA, vdup_n_s8(0), 0);
                    int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    pA++;
                    pB += 4;
                }

                float32x4_t _bd0 = vld1q_f32(pB_descales);
                const float _ad0 = pA_descales[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, _ad0));

                pA_descales++;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
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
                        int8x16_t _b = vld1q_s8(pB);
                        int8x16_t _aa = vcombine_s8(_a, _a);
                        _sum = vdotq_s32(_sum, _aa, _b);
                        pA += 8;
                        pB += 16;
                    }
                    _sum0 = vadd_s32(_sum0, vpadd_s32(vget_low_s32(_sum), vget_high_s32(_sum)));
                }
#else  // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x16_t _b = vld1q_s8(pB);
                    _sum0 = vdot_lane_s32(_sum0, vget_low_s8(_b), _a, 0);
                    _sum0 = vdot_lane_s32(_sum0, vget_high_s8(_b), _a, 1);
                    pA += 8;
                    pB += 16;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    int8x8_t _a = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                    int8x8_t _b = vld1_s8(pB);
                    _sum0 = vdot_s32(_sum0, _a, _b);
                    pA += 4;
                    pB += 8;
                }
#else  // __ARM_FEATURE_DOTPROD
                {
                    int32x4_t _sum = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);
                    for (; kk + 15 < max_kk0; kk += 16)
                    {
                        int8x16_t _a = vld1q_s8(pA);
                        int8x16_t _b0 = vld1q_s8(pB);
                        int8x16_t _b1 = vld1q_s8(pB + 16);
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
                        pB += 32;
                    }
                    _sum = vaddq_s32(_sum, _sum1);
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        int8x8_t _a = vld1_s8(pA);
                        int8x16_t _b = vld1q_s8(pB);
                        int16x4x2_t _aa = vzip_s16(vreinterpret_s16_s8(_a), vreinterpret_s16_s8(_a));
                        int8x8_t _a0 = vreinterpret_s8_s16(_aa.val[0]);
                        int8x8_t _a1 = vreinterpret_s8_s16(_aa.val[1]);
                        int16x8_t _s0 = vmull_s8(_a0, vget_low_s8(_b));
                        _s0 = vmlal_s8(_s0, _a1, vget_high_s8(_b));
                        _sum = vpadalq_s16(_sum, _s0);
                        pA += 8;
                        pB += 16;
                    }
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        int8x8_t _a = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                        int8x8_t _b = vld1_s8(pB);
                        _a = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_a), vreinterpret_s16_s8(_a)).val[0]);
                        _sum = vpadalq_s16(_sum, vmull_s8(_a, _b));
                        pA += 4;
                        pB += 8;
                    }
                    _sum0 = vadd_s32(_sum0, vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum)));
                }
#endif // __ARM_FEATURE_DOTPROD
                sum0 = vget_lane_s32(_sum0, 0);
                sum1 = vget_lane_s32(_sum0, 1);
#endif // __ARM_NEON
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[1];
                    sum1 += pA[0] * pB[2];
                    sum1 += pA[1] * pB[3];
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[0] * pB[1];
                    pA++;
                    pB += 2;
                }

#if __ARM_NEON
                float32x2_t _bd0 = vld1_f32(pB_descales);
                float32x2_t _scale = vmul_n_f32(_bd0, pA_descales[0]);
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
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
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
                    int8x16_t _b0 = vld1q_s8(pB);
                    int8x16_t _b1 = vld1q_s8(pB + 16);
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
                    pB += 32;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
#if __ARM_FEATURE_DOTPROD
                for (; kk + 15 < max_kk0; kk += 16)
                {
                    int8x16_t _a = vld1q_s8(pA);
                    int8x16_t _b = vld1q_s8(pB);
                    _sum0 = vdotq_s32(_sum0, _a, _b);
                    pA += 16;
                    pB += 16;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    int8x8_t _a = vld1_s8(pA);
                    int8x8_t _b = vld1_s8(pB);
                    _sum0 = vpadalq_s16(_sum0, vmull_s8(_a, _b));
                    pA += 8;
                    pB += 8;
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
                for (; kk < max_kk0; kk++)
                {
                    sum00 += pA[0] * pB[0];
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
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(float)));

#if __aarch64__
    TILE_M = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
    TILE_M = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
#endif
#if __aarch64__
    TILE_N = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
    TILE_N = std::max(4, tile_size / 4 * 4);
#else
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
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

#if __aarch64__
            TILE_M = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
            TILE_M = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
#endif
#if __aarch64__
            TILE_N = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
            TILE_N = std::max(4, tile_size / 4 * 4);
#else
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

    // always take constant TILE_M/N value when provided
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
